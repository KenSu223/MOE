"""ext7 W4: joint sublayer decomposition at the final position on STR pairs (needs the ext8 `multi` step kinds
attn_layer / block; see moetrace/engine.py docstring, "ext8 (Phase 3) additions to multi").

Per directed case, one wavefront row per (direction, kind), each a `multi` spawn with one step at EVERY layer:
    all_moe    steps (l, 'layer')       every MoE output at the final position := the other run's
    all_attn   steps (l, 'attn_layer')  every attention output at the final position := the other run's
    all_block  steps (l, 'block')       both (= the other run's final residual; sanity: normalised effect 1)
Directions: denoise (parent = corrupted row, clean = clean row: sufficiency, "how much is restored") and noise
(parent = clean row, clean = corrupted row: necessity, "how much is destroyed"). Optional --tail-layers K adds the
same three kinds restricted to the last K layers (l >= L - K) and to the first L - K layers (prefix / suffix split).

Per directed case and direction: A = all_attn effect / drop, M = all_moe effect / drop (effect = |change of Delta| in
the direction of the patch: denoise rescue = Delta_patched - Delta_corrupt; noise damage = Delta_clean - Delta_patched),
two-player Shapley split phi_attn = 1/2 [A + (1 - M)], phi_moe = 1 - phi_attn; redundancy A + M - 1 (analysis script).

Usage: python scripts/ext7_wino_joint.py --model qwen3 --pairs <parquet> --case-sets <json> --out wino_qwen3_str
           [--families main,rep] [--tail-layers 0] [--max-rows 30000] [--dry-run]
Outputs results/<out>/joint_rows.parquet, run_meta.json ("joint")
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import MODELS
from moetrace import ext7_pairs as P

STEP = {"all_moe": "layer", "all_attn": "attn_layer", "all_block": "block"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    P.add_common_args(ap)
    ap.add_argument("--families", default="main,rep")
    ap.add_argument("--tail-layers", type=int, default=0, help="also patch only the last K layers / only the first L-K layers")
    ap.add_argument("--max-rows", type=int, default=30000)
    ap.add_argument("--wf-chunk", type=int, default=2048, help="wavefront attention row chunk (memory)")
    args = ap.parse_args()
    m = MODELS[args.model]
    from moetrace.arch import load_spec
    L = load_spec(m["repo"])[0].n_layers
    cs, pairs, _, _, _ = P.setup_run(args, "joint")
    fam_all = P.load_families(args.out)
    fam = {k: v for k, v in fam_all.items() if k in args.families.split(",")}
    ids = P.all_ids(fam)
    dc = P.directed_cases(pairs, ids)
    n = len(ids)
    spans = {"all": list(range(L))}
    if args.tail_layers:
        K = args.tail_layers
        spans[f"last{K}"] = list(range(L - K, L))
        spans[f"first{L - K}"] = list(range(L - K))
    variants = [(direction, kind, span) for direction in ("denoise", "noise") for kind in STEP for span in spans]
    per_case = len(variants)
    chunks = [ids[i:i + max(1, args.max_rows // per_case)] for i in range(0, n, max(1, args.max_rows // per_case))]
    log(f"{args.model} -> {args.out}: {n} directed cases, families {list(fam)}, {per_case} multi rows per case "
        f"({len(spans)} layer spans), {len(chunks)} passes")
    if args.dry_run:
        return
    import torch
    from moetrace import engine as E
    assert "attn_layer" in E.MULTI_STEP_KINDS and "block" in E.MULTI_STEP_KINDS, "engine without ext8 multi step kinds (wait for ENGINE READY)"
    meta = {"model": args.model, "repo": m["repo"], "families": list(fam), "n_cases": n, "spans": {k: [v[0], v[-1]] for k, v in spans.items()},
            "variants": variants, "agent": args.agent, "command": "python " + " ".join(sys.argv),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    P.write_meta(args.out, "joint", meta)
    eng = E.Engine(m["repo"])
    rows, times = [], []
    for ci, ch in enumerate(chunks):
        k = len(ch)
        pre = [E.PrefillSpec(dc[c].clean_ids, dc[c].true_id, dc[c].foil_id) for c in ch]
        pre += [E.PrefillSpec(dc[c].corrupt_ids, dc[c].true_id, dc[c].foil_id, clean_ref=i) for i, c in enumerate(ch)]
        spawns, tags = [], []
        for i, c in enumerate(ch):
            for direction, kind, span in variants:
                ls = spans[span]
                par, cl = (k + i, i) if direction == "denoise" else (i, k + i)
                steps = tuple((l, STEP[kind], ()) for l in ls)
                spawns.append(E.SpawnSpec(ls[0], par, cl, "multi", steps=steps)); tags.append((i, direction, kind, span))
        log(f"pass {ci + 1}/{len(chunks)}: {len(pre)} prefill rows, {len(spawns)} multi rows")
        res = eng.run(pre, spawns, record_routing=False, log=log if ci == 0 else None, wf_chunk=args.wf_chunk)
        times.append(res.extra["total_s"])
        d, sd = res.delta, res.sp_delta
        for jj, (i, direction, kind, span) in enumerate(tags):
            c = ch[i]
            dcl, dco = float(d[i]), float(d[k + i])
            base = dco if direction == "denoise" else dcl
            rows.append(dict(case_id=c, pair_idx=c // 2, d=c % 2, direction=direction, kind=kind, span=span, delta=float(sd[jj]),
                             delta_clean=dcl, delta_corrupt=dco, base=base, change=float(sd[jj] - base),
                             effect=float(sd[jj] - dco) if direction == "denoise" else float(dcl - sd[jj]),
                             drop=dcl - dco, vnorm_first=float(res.sp_vnorm[jj])))
        pd.DataFrame(rows).to_parquet(os.path.join(P.run_dir(args.out), "joint_rows.parquet"), index=False)
        del res
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    for (direction, kind, span), g in df.groupby(["direction", "kind", "span"]):
        log(f"{direction:8s} {kind:9s} {span:8s}: effect/drop {g.effect.mean() / g['drop'].mean():+.3f} (mean effect {g.effect.mean():+.2f}, drop {g['drop'].mean():.2f})")
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "pass_times_s": times, "n_rows": len(df)})
    P.write_meta(args.out, "joint", meta)


if __name__ == "__main__":
    main()
