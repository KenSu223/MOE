"""ext7 W3: position x layer grid on STR pairs (Zhang & Nanda 2024 Section 4.1 / Figure 4; executor
moetrace/ext5_subject.py, as scripts/ext6_str_grid.py).

A unit is (directed case, position p) for every p from the first STR position (min str_pos) to the final token;
positions before it are identical in the clean and corrupted prompts, so every patch there is exactly zero (skipped).
Per unit: a clean and a corrupted prefill row recording at p; per layer l and kind one suffix wavefront row
(parent = corrupted row, clean = clean row):
    layer       MoE output at p := clean       (--window W > 1: at layers max(0, l-h)..min(L-1, l+h), h = (W-1)//2)
    attn_layer  attention output at p := clean (window 1 only)
    resid       residual after layer l at p := clean (classic hidden-state restoration; window 1 only)
    block       attention + MoE output at p := clean (window 1 only)
Delta and the full-softmax p(true) / rank at the final position. Units are packed into passes by suffix length
(padded suffix token-rows <= --token-budget, rows <= --max-rows).

Usage: python scripts/ext7_wino_grid.py --model qwen3 --pairs <parquet> --case-sets <json> --out wino_qwen3_str
           [--families main] [--kinds layer,attn_layer,resid] [--window 1] [--token-budget 150000] [--max-rows 40000]
Outputs results/<out>/str_grid_w<W>_rows.parquet, str_grid_w<W>_prefill.parquet, run_meta.json ("grid_w<W>")
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import MODELS
from moetrace import ext7_pairs as P

MET = ("logp_true", "logp_foil", "p_true", "p_foil", "rank_true")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def plan(units, rows_per_unit, budget, max_rows):
    units = sorted(units, key=lambda u: (u["S"], u["case_id"], u["pos"]))
    passes, cur, rows = [], [], 0
    for u in units:
        if cur and ((rows + rows_per_unit) * u["S"] > budget or rows + rows_per_unit > max_rows):
            passes.append(cur); cur, rows = [], 0
        cur.append(u); rows += rows_per_unit
    if cur:
        passes.append(cur)
    return passes


def main():
    ap = argparse.ArgumentParser()
    P.add_common_args(ap)
    ap.add_argument("--families", default="main")
    ap.add_argument("--kinds", default="layer,attn_layer,resid")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--token-budget", type=int, default=150_000)
    ap.add_argument("--max-rows", type=int, default=40_000)
    args = ap.parse_args()
    assert args.window >= 1 and args.window % 2 == 1, "odd window (centred)"
    kinds = args.kinds.split(",")
    if args.window > 1:
        assert kinds == ["layer"], "windows > 1 only for kind layer"
    m = MODELS[args.model]
    from moetrace.arch import load_spec
    L = load_spec(m["repo"])[0].n_layers
    cs, pairs, fam_new, _, _ = P.setup_run(args, "grid")
    fam_all = P.load_families(args.out)
    fam = {k: v for k, v in fam_all.items() if k in args.families.split(",")}
    ids = P.all_ids(fam)
    dc = P.directed_cases(pairs, ids)
    so = P.split_of(fam)
    h = (args.window - 1) // 2
    units = []
    for c in ids:
        x = dc[c]
        for p in range(x.first_str, x.T):
            units.append({"case_id": c, "pos": p, "S": x.T - p, "cat": P.position_category(p, x.str_pos, x.T)})
    per_unit = L * len(kinds)
    passes = plan(units, per_unit, args.token_budget, args.max_rows)
    log(f"{args.model} -> {args.out}: window {args.window}, kinds {kinds}, {len(ids)} directed cases, {len(units)} units, "
        f"{len(units) * per_unit} suffix rows, {len(passes)} passes; units per category {pd.Series([u['cat'] for u in units]).value_counts().to_dict()}")
    if args.dry_run:
        for i, ps in enumerate(passes):
            log(f"[dry-run] pass {i + 1}: {len(ps)} units, S {ps[0]['S']}..{ps[-1]['S']}, {len(ps) * per_unit} rows")
        return
    import torch
    from moetrace.ext5_subject import SubjectEngine, SubjectPrefill, SubjectSpawn
    key = f"grid_w{args.window}"
    meta = {"model": args.model, "repo": m["repo"], "families": list(fam), "kinds": kinds, "window": args.window, "window_half": h,
            "n_cases": len(ids), "n_units": len(units), "n_suffix_rows": len(units) * per_unit, "n_passes": len(passes),
            "token_budget": args.token_budget, "max_rows": args.max_rows, "agent": args.agent, "command": "python " + " ".join(sys.argv),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    P.write_meta(args.out, key, meta)
    eng = SubjectEngine(m["repo"])
    assert eng.spec.n_layers == L
    od = P.run_dir(args.out)
    rows_path = os.path.join(od, f"str_grid_w{args.window}_rows.parquet")
    pre_path = os.path.join(od, f"str_grid_w{args.window}_prefill.parquet")
    all_rows, all_pre, times = [], [], []
    for pi, ps in enumerate(passes):
        pre, spawns, tags, ptags = [], [], [], []
        for u in ps:
            x = dc[u["case_id"]]
            p = u["pos"]
            ci = len(pre); pre.append(SubjectPrefill(x.clean_ids, x.true_id, x.foil_id, rec_pos=p)); ptags.append((u, -1))
            di = len(pre); pre.append(SubjectPrefill(x.corrupt_ids, x.true_id, x.foil_id, rec_pos=p)); ptags.append((u, 0))
            for l in range(L):
                for k in kinds:
                    a, b = (max(0, l - h), min(L - 1, l + h)) if k == "layer" else (l, l)
                    spawns.append(SubjectSpawn(a, di, ci, k, p, window=b - a + 1)); tags.append((u, k, l, a, b, di, ci))
        log(f"pass {pi + 1}/{len(passes)}: {len(ps)} units (S {ps[0]['S']}..{ps[-1]['S']}), {len(pre)} prefill rows, {len(spawns)} suffix rows")
        res = eng.run_subject(pre, spawns, log=log if pi == 0 else None, metrics=True)
        times.append(res.extra["total_s"])
        d, sd = res.delta, res.sp_delta
        mp, ms = res.extra["metrics_prefill"], res.extra["metrics_spawn"]
        for j, (u, slot) in enumerate(ptags):
            all_pre.append(dict(case_id=u["case_id"], pos=u["pos"], cat=u["cat"], slot=slot, delta=float(d[j]),
                                **{k: (int(mp[k][j]) if k == "rank_true" else float(mp[k][j])) for k in MET}, pass_idx=pi))
        for jj, (u, k, l, a, b, di, ci) in enumerate(tags):
            c = u["case_id"]
            x = dc[c]
            all_rows.append(dict(case_id=c, pair_idx=x.pair_idx, d=x.d, split=so.get(c, ""), pos=u["pos"], rel_pos=u["pos"] - (x.T - 1),
                                 off_str=u["pos"] - x.first_str, S=u["S"], cat=u["cat"], kind=k, layer=l, start=a, end=b, delta=float(sd[jj]),
                                 delta_corrupt=float(d[di]), delta_clean=float(d[ci]), rescue=float(sd[jj] - d[di]),
                                 p_true=float(ms["p_true"][jj]), p_true_corrupt=float(mp["p_true"][di]), p_true_clean=float(mp["p_true"][ci]),
                                 dp=float(ms["p_true"][jj] - mp["p_true"][di]), logp_true=float(ms["logp_true"][jj]),
                                 rank_true=int(ms["rank_true"][jj]), rank_true_corrupt=int(mp["rank_true"][di]), vnorm=float(res.sp_vnorm[jj]),
                                 pass_idx=pi))
        pd.DataFrame(all_rows).to_parquet(rows_path, index=False)
        pd.DataFrame(all_pre).to_parquet(pre_path, index=False)
        log(f"pass {pi + 1}: {res.extra['total_s']:.1f}s; {len(all_rows)} rows saved")
        del res
        torch.cuda.empty_cache()
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "pass_times_s": times,
                 "n_rows": len(all_rows)})
    P.write_meta(args.out, key, meta)
    log("done")


if __name__ == "__main__":
    main()
