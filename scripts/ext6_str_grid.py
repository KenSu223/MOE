"""ext6 STR layer x position grid: the paper's MoE-output patch at EVERY position from the first subject token to the
final token and at every layer, with symmetric token replacement (Zhang & Nanda 2024, Section 4.1 / Figure 4).

Uses the retained cases and selected donors of an ext6 STR run (results/<run>/, from ext6_str_filter.py). A unit is
(case, position p); positions before the first subject token are skipped because the clean and donor prompts share that
prefix, so the patch is exactly zero there. Per unit: a clean prefill row and one row per donor, all recording at p; per
donor row and layer l one suffix wavefront row (moetrace/ext5_subject.py) with the MoE output at p set to the clean one at
layer l (--window 1) or at layers max(0, l-h)..min(L-1, l+h) with h = (window-1)//2 (sliding window, centred as in
Meng et al.). Metrics on: Delta and the full-softmax p(true) / p(foil) / rank at the final position. Units are packed
into passes by suffix length (padded suffix token-rows <= --token-budget, rows <= --max-rows); every rescue is measured
against the donor row of the same pass.

Usage: python scripts/ext6_str_grid.py <model> --out <run> [--no-special-tokens] [--window 1] [--token-budget 150000]
       [--max-rows 40000] [--dry-run]
Outputs results/<run>/str_grid_w<W>_rows.parquet (donor level), str_grid_w<W>_prefill.parquet, run_meta.json ("grid_w<W>")
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS
from moetrace.protocol import cases_by_id
from moetrace import ext6_str as S

MET = ("logp_true", "logp_foil", "p_true", "p_foil", "rank_true")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def plan(units, n_layers, budget, max_rows):
    """Greedy packing of units (sorted by suffix length) into passes."""
    units = sorted(units, key=lambda u: (u["S"], u["case_id"], u["pos"]))
    passes, cur, rows = [], [], 0
    for u in units:
        r = len(u["donors"]) * n_layers
        if cur and ((rows + r) * u["S"] > budget or rows + r > max_rows):
            passes.append(cur); cur, rows = [], 0
        cur.append(u); rows += r
    if cur:
        passes.append(cur)
    return passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--token-budget", type=int, default=150_000)
    ap.add_argument("--max-rows", type=int, default=40_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    assert args.window >= 1 and args.window % 2 == 1, "odd window (centred)"
    m = MODELS[args.model]
    st = not args.no_special_tokens
    od = S.run_dir(args.out)
    sets = json.load(open(os.path.join(od, "case_sets.json")))
    ids = sets["paper"]["discovery"] + sets["paper"]["validation"]
    split = {c: "discovery" for c in sets["paper"]["discovery"]} | {c: "validation" for c in sets["paper"]["validation"]}
    cases, rej = cases_by_id(args.model, ids, special_tokens=st)
    assert not rej, rej
    dn = S.load_donors(args.out)
    dn = dn[dn.case_id.isin(ids)]
    don = {c: [(int(r.slot), r.ids_list) for r in g.itertuples()] for c, g in dn.groupby("case_id")}
    L = len(json.load(open(os.path.join(od, "str_sweep_summary.json")))["val_curve"])  # layers (the STR sweep must exist)
    h = (args.window - 1) // 2
    units = []
    for c in ids:
        cs = cases[c]
        T = len(cs.ids)
        for p in range(cs.subject_pos[0], T):
            units.append({"case_id": c, "pos": p, "S": T - p, "cat": S.position_category(p, cs.subject_pos, T), "donors": don[c]})
    passes = plan(units, L, args.token_budget, args.max_rows)
    n_rows = sum(len(u["donors"]) for u in units) * L
    n_tok = sum(len(u["donors"]) * u["S"] for u in units) * L
    log(f"{args.model} -> {args.out}: window {args.window}, {len(ids)} cases, {len(units)} (case, position) units, {n_rows} suffix rows, "
        f"{n_tok} suffix tokens, {len(passes)} passes; units per category {pd.Series([u['cat'] for u in units]).value_counts().to_dict()}")
    if args.dry_run:
        for i, ps in enumerate(passes):
            r = sum(len(u["donors"]) for u in ps) * L
            log(f"[dry-run] pass {i + 1}: {len(ps)} units, S {ps[0]['S']}..{ps[-1]['S']}, {r} rows, padded tokens {r * ps[-1]['S']}")
        return
    from moetrace.ext5_subject import SubjectEngine, SubjectPrefill, SubjectSpawn
    key = f"grid_w{args.window}"
    meta = {"model": args.model, "repo": m["repo"], "special_tokens": st, "window": args.window, "window_half": h, "n_cases": len(ids),
            "n_units": len(units), "n_suffix_rows": n_rows, "n_suffix_tokens": n_tok, "n_passes": len(passes),
            "token_budget": args.token_budget, "max_rows": args.max_rows, "command": "python " + " ".join(sys.argv),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    S.write_meta(args.out, key, meta)
    eng = SubjectEngine(m["repo"])
    rows_path = os.path.join(od, f"str_grid_w{args.window}_rows.parquet")
    pre_path = os.path.join(od, f"str_grid_w{args.window}_prefill.parquet")
    all_rows, all_pre, times = [], [], []
    for pi, ps in enumerate(passes):
        pre, spawns, tags, ptags = [], [], [], []
        for u in ps:
            cs = cases[u["case_id"]]
            p = u["pos"]
            ci = len(pre)
            pre.append(SubjectPrefill(cs.ids, cs.true_id, cs.foil_id, rec_pos=p)); ptags.append((u, -1, ci))
            for slot, dids in u["donors"]:
                di = len(pre)
                pre.append(SubjectPrefill(dids, cs.true_id, cs.foil_id, rec_pos=p)); ptags.append((u, slot, ci))
                for l in range(L):
                    a, b = max(0, l - h), min(L - 1, l + h)
                    spawns.append(SubjectSpawn(a, di, ci, "layer", p, window=b - a + 1))
                    tags.append((u, slot, l, a, b, di, ci))
        log(f"pass {pi + 1}/{len(passes)}: {len(ps)} units (S {ps[0]['S']}..{ps[-1]['S']}), {len(pre)} prefill rows, {len(spawns)} suffix rows")
        res = eng.run_subject(pre, spawns, log=log if pi == 0 else None, metrics=True)
        times.append(res.extra["total_s"])
        d, sd = res.delta, res.sp_delta
        mp, ms = res.extra["metrics_prefill"], res.extra["metrics_spawn"]
        for j, (u, slot, ci) in enumerate(ptags):
            all_pre.append(dict(case_id=u["case_id"], pos=u["pos"], cat=u["cat"], slot=slot, delta=float(d[j]),
                                **{k: (int(mp[k][j]) if k == "rank_true" else float(mp[k][j])) for k in MET}, pass_idx=pi))
        for jj, (u, slot, l, a, b, di, ci) in enumerate(tags):
            all_rows.append(dict(case_id=u["case_id"], split=split[u["case_id"]], pos=u["pos"], S=u["S"], cat=u["cat"], slot=slot,
                                 layer=l, start=a, end=b, delta=float(sd[jj]), delta_corrupt=float(d[di]), delta_clean=float(d[ci]),
                                 rescue=float(sd[jj] - d[di]), p_true=float(ms["p_true"][jj]), p_true_corrupt=float(mp["p_true"][di]),
                                 p_true_clean=float(mp["p_true"][ci]), dp=float(ms["p_true"][jj] - mp["p_true"][di]),
                                 logp_true=float(ms["logp_true"][jj]), rank_true=int(ms["rank_true"][jj]),
                                 rank_true_corrupt=int(mp["rank_true"][di]), vnorm=float(res.sp_vnorm[jj]), pass_idx=pi))
        pd.DataFrame(all_rows).to_parquet(rows_path, index=False)
        pd.DataFrame(all_pre).to_parquet(pre_path, index=False)
        log(f"pass {pi + 1}: {res.extra['total_s']:.1f}s (metrics {res.extra.get('metrics_s', 0):.1f}s); {len(all_rows)} rows saved")
        del res
        torch.cuda.empty_cache()
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "pass_times_s": times,
                 "n_rows": len(all_rows)})
    S.write_meta(args.out, key, meta)
    log("done")


if __name__ == "__main__":
    main()
