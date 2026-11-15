#!/usr/bin/env python3
"""Phase 7 S3 / S4 runner (docs/relp_moe_plan.md): RelP at the B endpoint, and Lu-style STR activation patching grids on
the SVA pairs, with the verified streamed engine (moetrace/relp_moe_stream.py, unmodified; same verification gate as
scripts/relp_sva_moe_run.py: the task's passing verify/verification.json must match the current identity).

  endpointB  discovery split: (q_A - q_B) . beta_B summed over rows -> aggregate.pt {mean, support, n, endpoint},
             independent neuron / expert_position / expert_all_positions rankings (same functions as the protocol).
  patch      splits (default discovery validation test): every unit of enumerate_units on the chosen positions
             (--positions changed_to_final = first changed position .. final position (earlier positions are identical in
             A and B, so their patches are exactly 0); final = Lu's original scope), both directions, position units and
             expert units (union of the experts routed at A or B). Row-level parquet per pass (resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DIRECTIONS = ("denoise", "noise")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("endpointB", "patch", "diagnose"))
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--task", choices=("simple", "rc", "within_rc", "nounpp"), required=True)
    p.add_argument("--verification", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--pool", choices=("non_special", "all", "final"), default="non_special")
    p.add_argument("--splits", nargs="+", default=["discovery", "validation", "test"])
    p.add_argument("--positions", choices=("changed_to_final", "final"), default="changed_to_final")
    p.add_argument("--no-experts", action="store_true", help="position units only (layer x position grid)")
    p.add_argument("--buf-gb", type=float, default=6.0, help="GPU memory for patched rows per pass")
    p.add_argument("--sub-rows", type=int, default=6000)
    p.add_argument("--max-rows", type=int, default=0, help="first N rows per split (timing / smoke only)")
    p.add_argument("--units-json", type=Path, help="diagnose: [{kind: position|expert, layer, pos, expert}, ...]")
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from moetrace import relp_sva as S
    from moetrace.relp_moe_stream import Group, StreamRelP, forward_pass
    from moetrace.relp_moe_patch import enumerate_units, patch_pass, relp_backward_endpoint, routes_pass
    import relp_sva_moe_run as R
    meta, _ = S.read_split(a.manifest_dir, a.task, "calibration")
    S.check_tokenizer_files(a.model_dir, meta)
    m = StreamRelP(str(a.model_dir.resolve()), device=a.device, dtype=getattr(torch, a.dtype))
    ident = R.identity(m, a.model_dir, S)
    v = json.loads(a.verification.read_text())
    if not v.get("passed") or v["identity"] != ident or v["task"] != a.task or v["manifest_sha256"] != S.sha(a.manifest_dir / "manifest.json"):
        p.error("Verification is failed, stale, or for another model/task/manifest/runtime")
    common = {"utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "task": a.task,
              "manifest_sha256": S.sha(a.manifest_dir / "manifest.json"), "identity": ident,
              "verification_sha256": S.sha(a.verification), "script_sha256": S.sha(__file__),
              "patch_module_sha256": S.sha(ROOT / "moetrace/relp_moe_patch.py"), "status": "running"}
    out = a.output_dir
    if (out / "run_meta.json").exists() and json.loads((out / "run_meta.json").read_text()).get("status") == "complete":
        print("complete; nothing to do")
        return
    out.mkdir(parents=True, exist_ok=True)
    S.write_json(out / "run_meta.json", common)
    t0 = time.time()
    if a.command == "endpointB":
        _, rows = S.read_split(a.manifest_dir, a.task, "discovery")
        n = len(rows)
        A, Bb = R.ids_tensor(rows), R.ids_tensor(rows, True)
        t = [r["target_id"] for r in rows]
        f = [r["foil_id"] for r in rows]
        g = Group("score", torch.cat([A, Bb]), t + t, f + f, record_h_in=True)
        forward_pass(m, [g])
        pm = R.pool_mask(rows, a.pool)
        total, support, rows_b = relp_backward_endpoint(m, g, n, pm, "B", per_row=True)
        total_a, support_a, rows_a = relp_backward_endpoint(m, g, n, pm, "A", per_row=True)
        mean = total.cpu() / n
        torch.save({"mean": mean, "support": support.cpu(), "n": n, "endpoint": "B", "pool": a.pool, "mode": "paired"},
                   out / "aggregate.pt")
        # per-row expert-position scores of both endpoints (bootstrap / per-row diagnostics); endpoint A is the protocol score
        torch.save({"A": rows_a, "B": rows_b, "row_ids": [r["row_id"] for r in rows], "group_ids": [r["group_id"] for r in rows]},
                   out / "per_row_expert_scores.pt")
        torch.save({"mean": total_a.cpu() / n, "support": support_a.cpu(), "n": n, "endpoint": "A"}, out / "aggregate_A_recomputed.pt")
        for unit, values in S.rankings(mean, support.cpu(), 1000).items():
            S.write_json(out / f"{unit}_ranking.json", values)
        d = g.delta.cpu()
        pd.DataFrame([{"row_id": r["row_id"], "group_id": r["group_id"], "margin_A": float(d[i]), "margin_B": float(d[n + i])}
                      for i, r in enumerate(rows)]).to_parquet(out / "cases.parquet", index=False)
        S.write_json(out / "run_meta.json", {**common, "status": "complete", "seconds": time.time() - t0,
                     "score": "(q_A - q_B) . beta_B (coefficients at the B endpoint), mean over discovery rows",
                     "aggregate_sha256": S.sha(out / "aggregate.pt")})
        print(f"endpointB done in {time.time() - t0:.0f}s", flush=True)
        return
    if a.command == "diagnose":
        from moetrace.relp_moe_patch import VARIANTS, diagnose_pass
        spec = json.loads(a.units_json.read_text())
        _, rows = S.read_split(a.manifest_dir, a.task, "discovery")
        if a.max_rows:
            rows = rows[:a.max_rows]
        n = len(rows)
        A, Bb = R.ids_tensor(rows), R.ids_tensor(rows, True)
        ta, tb = routes_pass(m, A, Bb)
        units = []
        for i in range(n):
            for di in range(2):
                for sp in spec:
                    l, t, e = int(sp["layer"]), int(sp["pos"]), int(sp.get("expert", -1))
                    if sp["kind"] == "position":
                        units.append((i, di, 0, l, t, -1, 0, 0))
                    else:
                        units.append((i, di, 1, l, t, e, int(e in ta[l, i, t].tolist()), int(e in tb[l, i, t].tolist())))
        units = np.asarray(units, dtype=np.int64)
        res = diagnose_pass(m, A, Bb, [r["target_id"] for r in rows], [r["foil_id"] for r in rows], units, sub_rows=a.sub_rows)
        mA, mB = res["margin_A"][units[:, 0]], res["margin_B"][units[:, 0]]
        cols = {"row": units[:, 0], "row_id": [rows[i]["row_id"] for i in units[:, 0]],
                "group_id": [rows[i]["group_id"] for i in units[:, 0]], "direction": np.asarray(DIRECTIONS)[units[:, 1]],
                "kind": np.where(units[:, 2] == 0, "position", "expert"), "layer": units[:, 3], "pos": units[:, 4],
                "expert": units[:, 5], "active_A": units[:, 6].astype(bool), "active_B": units[:, 7].astype(bool),
                "margin_A": mA, "margin_B": mB,
                # direct part in the unit's direction convention (denoise: towards the clean answer; noise: the drop)
                "direct": np.where(units[:, 1] == 0, res["direct"], -res["direct"])}
        for v in VARIANTS:  # effect in the unit's direction: denoise F(B<-A) - F_B, noise F_A - F(A<-B)
            cols[f"effect_{v}"] = np.where(units[:, 1] == 0, res[v] - mB, mA - res[v])
        pd.DataFrame(cols).to_parquet(out / "diagnose.parquet", index=False)
        S.write_json(out / "run_meta.json", {**common, "status": "complete", "seconds": time.time() - t0, "rows": n,
                     "units_spec": spec, "variants": list(VARIANTS),
                     "direct": "final-position units: change of the MoE output read out through the final norm with the input run's denominator"})
        print(f"diagnose done in {time.time() - t0:.0f}s", flush=True)
        return
    # ---- patch
    rows, split_of = [], []
    for sp in a.splits:
        _, rs = S.read_split(a.manifest_dir, a.task, sp)
        if a.max_rows:
            rs = rs[:a.max_rows]
        rows += rs
        split_of += [sp] * len(rs)
    n = len(rows)
    A, Bb = R.ids_tensor(rows), R.ids_tensor(rows, True)
    T = A.shape[1]
    pool = R.pool_mask(rows, a.pool)[0]
    changed = sorted({p_ for r in rows for p_ in r["changed_positions"]})
    if a.positions == "final":
        positions = [T - 1]
    else:
        positions = [t for t in range(min(changed), T) if bool(pool[t])]
    routes_file = out / "routes.pt"
    if routes_file.exists():
        ta, tb = torch.load(routes_file, weights_only=True)
    else:
        ta, tb = routes_pass(m, A, Bb)
        torch.save((ta, tb), routes_file)
    units = enumerate_units(ta.numpy(), tb.numpy(), positions, DIRECTIONS, experts=not a.no_experts)
    per_row = np.bincount(units[:, 0], minlength=n)
    cap = int(a.buf_gb * 1e9 / (T * m.H * (2 if m.dtype == torch.bfloat16 else 4)))
    passes, cur, cnt = [], [], 0
    for i in range(n):
        if cur and cnt + per_row[i] > cap:
            passes.append(cur)
            cur, cnt = [], 0
        cur.append(i)
        cnt += per_row[i]
    if cur:
        passes.append(cur)
    print(f"{n} rows, {len(units)} units ({len(positions)} positions), {len(passes)} passes (cap {cap} units)", flush=True)
    S.write_json(out / "plan.json", {"rows": n, "units": int(len(units)), "positions": positions, "passes": len(passes),
                                     "units_per_pass_cap": cap, "directions": list(DIRECTIONS),
                                     "experts": not a.no_experts, "splits": a.splits})
    for pi, rows_i in enumerate(passes):
        part = out / f"part_{pi:03d}.parquet"
        if part.exists():
            continue
        tp = time.time()
        idx = np.asarray(rows_i)
        sel = np.isin(units[:, 0], idx)
        uu = units[sel].copy()
        remap = {int(r): j for j, r in enumerate(idx)}
        uu[:, 0] = [remap[int(r)] for r in uu[:, 0]]
        sub = [rows[i] for i in idx]
        res = patch_pass(m, A[idx], Bb[idx], [r["target_id"] for r in sub], [r["foil_id"] for r in sub], uu,
                         sub_rows=a.sub_rows)
        gi = idx[uu[:, 0]]
        mA, mB = res.margin_A[uu[:, 0]], res.margin_B[uu[:, 0]]
        effect = np.where(uu[:, 1] == 0, res.margin - mB, mA - res.margin)
        df = pd.DataFrame({
            "split": [split_of[i] for i in gi], "row": gi, "row_id": [rows[i]["row_id"] for i in gi],
            "group_id": [rows[i]["group_id"] for i in gi], "case": [rows[i].get("case") for i in gi],
            "direction": np.asarray(DIRECTIONS)[uu[:, 1]], "kind": np.where(uu[:, 2] == 0, "position", "expert"),
            "layer": uu[:, 3], "pos": uu[:, 4], "expert": uu[:, 5], "active_A": uu[:, 6].astype(bool),
            "active_B": uu[:, 7].astype(bool), "margin_A": mA, "margin_B": mB, "margin": res.margin, "effect": effect,
            "route_changes": res.route_changes})
        df.to_parquet(part, index=False)
        print(f"pass {pi + 1}/{len(passes)}: {len(uu)} units in {time.time() - tp:.0f}s", flush=True)
        del res, df
        torch.cuda.empty_cache()
    S.write_json(out / "run_meta.json", {**common, "status": "complete", "seconds": time.time() - t0, "rows": n,
                 "units": int(len(units)), "positions": positions, "passes": len(passes),
                 "effect": "denoise: F(B <- A) - F_B (Lu rescue); noise: F_A - F(A <- B) (drop)",
                 "unit_semantics": "complete replacement at one layer (position: MoE output; expert: w_e E_e(x)); later layers natural"})
    print(f"patch done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
