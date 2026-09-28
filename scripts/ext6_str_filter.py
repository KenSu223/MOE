"""ext6 STR pass 1: donor filter.

For every case of the paper set: build all symmetric donors (moetrace.ext6_str.candidates), run the clean prompts and
every donor prompt as plain prefill rows, and record Delta_donor = logit(true) - logit(foil) on the donor prompt (true
and foil = the CASE's objects). A donor qualifies when Delta_donor <= -1.0 (the model knows the donor's fact, whose
answer is the case's foil); per case the first K = 5 qualifying donors in the fixed random order are selected.
Case set = the paper's discovery/validation IDs restricted to cases with >= 1 selected donor.

Usage: python scripts/ext6_str_filter.py <model> --out <run> [--no-special-tokens] [--max-rows 3072] [--dry-run]
Outputs results/<run>/str_candidates.parquet, case_sets.json, str_filter_summary.json, run_meta.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from transformers import AutoTokenizer
from moetrace.models import MODELS
from moetrace.arch import snapshot_dir
from moetrace.data import load_records, load_paper_ids
from moetrace.protocol import cases_by_id
from moetrace import ext6_str as S


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--max-rows", type=int, default=3072, help="prefill rows per engine pass")
    ap.add_argument("--k", type=int, default=S.K_DONORS)
    ap.add_argument("--margin", type=float, default=S.MARGIN)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    m = MODELS[args.model]
    st = not args.no_special_tokens
    tok = AutoTokenizer.from_pretrained(snapshot_dir(m["repo"]))
    recs = load_records()
    pids = load_paper_ids(m["paper_key"])
    all_ids = pids["discovery"] + pids["validation"]
    cases, rej = cases_by_id(args.model, all_ids, tok=tok, special_tokens=st)
    if rej:
        log(f"WARNING: {len(rej)} paper ids not tokenizable: {rej}")
    index = S.donor_index(recs)
    rows = []
    for cid in all_ids:
        if cid not in cases:
            continue
        for d in S.candidates(cases[cid], recs[cid], tok, index, st):
            rows.append(dict(case_id=cid, order=d.order, donor_record=d.donor_record, donor_subject=d.subject,
                             prompt=d.prompt, ids=json.dumps(d.ids), n_tokens=len(d.ids)))
    cand = pd.DataFrame(rows)
    per = cand.groupby("case_id").size().reindex([c for c in all_ids if c in cases]).fillna(0).astype(int)
    log(f"{args.model} (special_tokens={st}): {len(cases)} cases, {len(cand)} symmetric candidates; cases with >= 1: {(per > 0).sum()}; "
        f"per case quartiles {np.percentile(per, [25, 50, 75]).tolist()} max {per.max()}")
    if args.dry_run:
        return
    from moetrace.engine import Engine, PrefillSpec
    od = S.run_dir(args.out)
    meta = {"model": args.model, "repo": m["repo"], "special_tokens": st, "k": args.k, "margin": args.margin, "seed_base": S.SEED_BASE,
            "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    S.write_meta(args.out, "filter", meta)
    eng = Engine(m["repo"])
    ids = [c for c in all_ids if c in cases]
    clean_pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
    cand_pre = [PrefillSpec(json.loads(r.ids), cases[r.case_id].true_id, cases[r.case_id].foil_id) for r in cand.itertuples()]
    # chunk: the clean rows ride in the first pass
    chunks, start = [], 0
    first = max(1, args.max_rows - len(clean_pre))
    while start < len(cand_pre):
        size = first if not chunks else args.max_rows
        chunks.append((start, min(len(cand_pre), start + size)))
        start += size
    d_cand = np.full(len(cand_pre), np.nan)
    lt, lf, t1 = (np.full(len(cand_pre), np.nan) for _ in range(3))
    pass_times = []
    for ci, (a, b) in enumerate(chunks):
        pre = (clean_pre if ci == 0 else []) + cand_pre[a:b]
        off = len(clean_pre) if ci == 0 else 0
        log(f"pass {ci + 1}/{len(chunks)}: {len(pre)} prefill rows")
        res = eng.run(pre, [], record_routing=False, log=log if ci == 0 else None)
        pass_times.append(res.extra["total_s"])
        if ci == 0:
            d_clean = res.delta[: len(clean_pre)].copy()
            top1_clean = res.top1[: len(clean_pre)].copy()
        d_cand[a:b] = res.delta[off:]
        lt[a:b] = res.logit_true[off:]
        lf[a:b] = res.logit_foil[off:]
        t1[a:b] = res.top1[off:]
    cand["delta_donor"] = d_cand
    cand["logit_true"] = lt
    cand["logit_foil"] = lf
    cand["top1"] = t1.astype(np.int64)
    cand["top1_is_foil"] = [int(t) == cases[c].foil_id for t, c in zip(cand.top1, cand.case_id)]
    cand = S.select(cand, args.k, args.margin)
    cand.to_parquet(os.path.join(od, "str_candidates.parquet"), index=False)
    dc = dict(zip(ids, d_clean.tolist()))
    n_sel = cand[cand.selected].groupby("case_id").size()
    keep = set(n_sel.index)
    sets = {"paper": {sp: [c for c in pids[sp] if c in keep] for sp in ("discovery", "validation")},
            "paper_full": {sp: list(pids[sp]) for sp in ("discovery", "validation")}}
    with open(os.path.join(od, "case_sets.json"), "w") as f:
        json.dump(sets, f, indent=1)
    q = cand.groupby("case_id").qualifies.sum().reindex(ids).fillna(0)
    summ = {"n_cases": len(ids), "n_candidates": int(len(cand)), "cases_with_candidate": int((per > 0).sum()),
            "cases_with_qualifying": int((q > 0).sum()), "cases_kept": len(keep),
            "kept_discovery": len(sets["paper"]["discovery"]), "kept_validation": len(sets["paper"]["validation"]),
            "qualify_rate": float(cand.qualifies.mean()), "donor_top1_is_foil_rate": float(cand.top1_is_foil.mean()),
            "donor_top1_is_foil_rate_selected": float(cand[cand.selected].top1_is_foil.mean()),
            "n_selected": int(cand.selected.sum()), "selected_per_case": n_sel.value_counts().sort_index().to_dict(),
            "delta_donor_selected_mean": float(cand[cand.selected].delta_donor.mean()),
            "delta_donor_all_median": float(cand.delta_donor.median()),
            "delta_clean_mean_kept": float(np.mean([dc[c] for c in keep])),
            "delta_clean_lt1_kept": int(sum(dc[c] < 1.0 for c in keep)),
            "top1_clean_is_true_kept": int(sum(int(top1_clean[ids.index(c)]) == cases[c].true_id for c in keep)),
            "dropped_no_candidate": [c for c in ids if per.get(c, 0) == 0],
            "dropped_no_qualifying": [c for c in ids if per.get(c, 0) > 0 and q[c] == 0],
            "pass_times_s": pass_times, "rejected_tokenizer": rej}
    with open(os.path.join(od, "str_filter_summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=str)
    with open(os.path.join(od, "str_clean_delta.json"), "w") as f:
        json.dump({str(k): v for k, v in dc.items()}, f)
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "pass_times_s": pass_times})
    S.write_meta(args.out, "filter", meta)
    log("summary:", json.dumps({k: v for k, v in summ.items() if not k.startswith("dropped")}, default=str))


if __name__ == "__main__":
    main()
