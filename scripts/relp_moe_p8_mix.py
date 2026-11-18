#!/usr/bin/env python3
"""Phase 8 follow-up: the add-back curves of S2.1 (noise / denoise, expert-position units, held-out SVA pairs) under five
patch semantics at the patched MoE sites (moetrace/relp_moe_p8.py, MIX_MODES):
  full          the engine's complete replacement (routing weights of the two runs mixed; the S2.1 curves)
  content_both  live routing kept; only experts routed in both runs get the other run's content (guide section 8)
  content_all   live routing kept; every selected live-routed expert gets the content it computes on the other run's input
  renorm        full, with the weights at the site rescaled to the live run's total (1 for Qwen3 / Mixtral)
  route         the other run's routing decision at the site is moved too; unselected experts keep the live content
The rankings and random sets are rebuilt exactly as in relp_moe_p8_run.py addback (rankings.json of p8_addback; random sets
from S.random_matched with the same seeds and eligibility), so the 'full' curves repeat S2.1 in another batch composition.
Same verification gate as the Phase-7 / Phase-8 runners; the verified engine is not modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

EP_BUDGETS = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--verification", type=Path, required=True)
    p.add_argument("--phase7-dir", type=Path, required=True, help="results/relp_sva_moe/<model>/paper/<task>")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rankings", nargs="+", default=["relp", "exact", "lu"])
    p.add_argument("--random", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--budgets", type=int, nargs="+", default=EP_BUDGETS)
    p.add_argument("--eval-chunk", type=int, default=1040)
    p.add_argument("--pass-gb", type=float, default=3.0, help="residual-state budget per pass (labels are split into passes)")
    p.add_argument("--max-rows", type=int, default=0, help="first N rows per split (smoke only)")
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from moetrace import relp_sva as S
    from moetrace.relp_moe_p8 import MIX_MODES, evaluate_mix
    from moetrace.relp_moe_stream import StreamRelP
    import relp_sva_moe_run as R
    meta, _ = S.read_split(a.manifest_dir, a.task, "calibration")
    S.check_tokenizer_files(a.model_dir, meta)
    m = StreamRelP(str(a.model_dir.resolve()), device=a.device, dtype=torch.bfloat16)
    ident = R.identity(m, a.model_dir, S)
    v = json.loads(a.verification.read_text())
    if not v.get("passed") or v["identity"] != ident or v["task"] != a.task or v["manifest_sha256"] != S.sha(a.manifest_dir / "manifest.json"):
        p.error("Verification is failed, stale, or for another model/task/manifest/runtime")
    out = a.output_dir
    if (out / "run_meta.json").exists() and json.loads((out / "run_meta.json").read_text()).get("status") == "complete":
        print("complete; nothing to do")
        return
    out.mkdir(parents=True, exist_ok=True)
    common = {"utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "task": a.task,
              "manifest_sha256": S.sha(a.manifest_dir / "manifest.json"), "identity": ident,
              "verification_sha256": S.sha(a.verification), "script_sha256": S.sha(__file__),
              "p8_module_sha256": S.sha(ROOT / "moetrace/relp_moe_p8.py"), "modes": list(MIX_MODES), "status": "running"}
    S.write_json(out / "run_meta.json", common)
    t0 = time.time()
    p7 = a.phase7_dir

    rows, split_of = [], []
    for sp in ("validation", "test"):
        _, rs = S.read_split(a.manifest_dir, a.task, sp)
        rs = rs[:a.max_rows] if a.max_rows else rs
        rows += rs
        split_of += [sp] * len(rs)
    A, Bb = R.ids_tensor(rows), R.ids_tensor(rows, True)
    tg, fo = [r["target_id"] for r in rows], [r["foil_id"] for r in rows]
    n, T = A.shape
    # eligibility, rankings and random sets as in relp_moe_p8_run.py addback (expert-position units)
    elig = torch.load(p7 / "score_zero/aggregate.pt", map_location="cpu", weights_only=True, mmap=True)["support"].any(-1).clone()
    L, T_, E = elig.shape
    assert T_ == T and E == m.E
    n_el = int(elig.sum())
    rk = json.loads((p7 / "p8_addback/rankings.json").read_text())["expert_position"]
    seeds = json.loads((p7 / "score_mean/selection_plan.json").read_text())["random_seeds"]
    Label = namedtuple("Label", "ranking seed budget")
    masks = {}
    for name in a.rankings:
        o = torch.tensor(rk[name], dtype=torch.long)
        for b in a.budgets:
            if b > len(o) or b >= n_el:
                continue
            c_ = torch.zeros_like(elig)
            c_.reshape(-1)[o[:b]] = True
            masks[Label(name, 0, b)] = c_
            if name == "relp" and a.random:
                for seed in seeds:
                    masks[Label("random", seed, b)] = S.random_matched(c_, elig, torch.Generator().manual_seed(seed + b))
    masks[Label("all", 0, n_el)] = elig.clone()
    pool = R.pool_mask(rows, "non_special")[0]
    rolepos = torch.arange(T)[None].expand(n, T).contiguous()
    per_label = 2 * len(MIX_MODES) * n * T * m.H * 2
    lpp = max(1, int(a.pass_gb * 1e9 // per_label))
    labs = list(masks)
    print(f"{a.task}: {n} rows, T = {T}, {len(labs)} labels, {lpp} labels per pass", flush=True)
    recs = []
    for c0 in range(0, len(labs), lpp):
        sub = {k_: masks[k_] for k_ in labs[c0:c0 + lpp]}
        res = evaluate_mix(m, A, Bb, tg, fo, pool, rolepos, sub, "role", MIX_MODES, a.eval_chunk)
        mA, mB = res["margin_A"].numpy(), res["margin_B"].numpy()
        for lab, per_mode in res["margin"].items():
            for mode, per_kind in per_mode.items():
                for kind, mg in per_kind.items():
                    recs.append(pd.DataFrame({"row_id": [r["row_id"] for r in rows], "split": split_of, "ranking": lab.ranking,
                                              "seed": lab.seed, "budget": lab.budget, "mode": mode, "intervention": kind,
                                              "margin": mg.numpy(), "margin_A": mA, "margin_B": mB}))
        print(f"labels {min(len(labs), c0 + lpp)}/{len(labs)} ({time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    df = pd.concat(recs, ignore_index=True)
    df.to_parquet(out / "margins.parquet", index=False)
    S.write_json(out / "run_meta.json", {**common, "status": "complete", "seconds": time.time() - t0, "rows": n, "labels": len(labs),
                                         "n_eligible": n_el, "budgets": a.budgets, "rankings": a.rankings + (["random"] if a.random else []),
                                         "random_seeds": seeds})
    print(f"mix done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
