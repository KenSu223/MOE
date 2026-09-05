"""Pass 1: filter scan. Scans CounterFact in seed-0 shuffled order in chunks of tokenizable records, running clean and
noised (sigma = 3.0 * embed std) prefill rows, until >= 256 strict and >= 512 relaxed cases exist.

Usage: python scripts/run_filter.py <model_key> [--chunk 1024] [--min-strict 256] [--min-relaxed 512] [--max-records N]
Outputs results/<model>/filter_scan.parquet, filter_rejects.parquet, case_sets.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer
from moetrace.models import MODELS, RESULTS
from moetrace.arch import snapshot_dir
from moetrace.data import load_records, shuffled_order, prepare_case, passes, split_cases, load_paper_ids, STRICT, RELAXED
from moetrace.engine import Engine, PrefillSpec
from moetrace.noise import noise_draw


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--min-strict", type=int, default=256)
    ap.add_argument("--min-relaxed", type=int, default=512)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    args = ap.parse_args()
    m = MODELS[args.model]
    out_dir = os.path.join(RESULTS, args.model)
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(snapshot_dir(m["repo"]))
    recs = load_records()
    order = shuffled_order(len(recs), 0)
    if args.max_records:
        order = order[: args.max_records]
    eng = Engine(m["repo"])
    sigma = args.sigma_mult * eng.embed_std
    Hd = eng.hidden
    log(f"{args.model}: embed_std={eng.embed_std:.6f} sigma={sigma:.6f} hidden={Hd} layers={eng.spec.n_layers}")
    rows, rejects = [], []
    n_strict = n_relaxed = 0
    scan_rank = 0
    chunk_cases = []
    pass_times = []
    done = False

    def flush(chunk_cases):
        nonlocal n_strict, n_relaxed
        n = len(chunk_cases)
        pre = [PrefillSpec(c.ids, c.true_id, c.foil_id) for c in chunk_cases]
        pre += [PrefillSpec(c.ids, c.true_id, c.foil_id, c.subject_pos, noise_draw(c.case_id, len(c.subject_pos), Hd, sigma)) for c in chunk_cases]
        t0 = time.time()
        res = eng.run(pre, [], record_routing=False, log=log if not pass_times else None)
        pass_times.append(time.time() - t0)
        d = res.delta
        dfull = res.logit_true_full - res.logit_foil_full
        for i, c in enumerate(chunk_cases):
            dc, dn = float(d[i]), float(d[n + i])
            st, rl = passes(dc, dn, STRICT), passes(dc, dn, RELAXED)
            n_strict += int(st)
            n_relaxed += int(rl)
            r = c.to_row()
            r.update(dict(scan_rank=c._rank, sigma_mult=args.sigma_mult, delta_clean=dc, delta_noised=dn, drop=dc - dn,
                          strict=bool(st), relaxed=bool(rl),
                          logit_true_clean=float(res.logit_true[i]), logit_foil_clean=float(res.logit_foil[i]),
                          logit_true_noised=float(res.logit_true[n + i]), logit_foil_noised=float(res.logit_foil[n + i]),
                          delta_clean_full=float(dfull[i]), delta_noised_full=float(dfull[n + i]),
                          top1_clean=int(res.top1[i]), top1_noised=int(res.top1[n + i]),
                          top1_clean_is_true=bool(res.top1[i] == c.true_id)))
            rows.append(r)
        log(f"chunk done ({n} records, {pass_times[-1]:.1f}s): scanned={scan_rank} tokenizable={len(rows)} strict={n_strict} relaxed={n_relaxed}")

    for rank, ri in enumerate(order):
        scan_rank = rank + 1
        c, why = prepare_case(recs[ri], tok)
        if c is None:
            rejects.append(dict(scan_rank=rank, case_id=int(recs[ri]["case_id"]), reason=why,
                                relation=recs[ri]["requested_rewrite"]["relation_id"]))
            continue
        c._rank = rank
        chunk_cases.append(c)
        if len(chunk_cases) >= args.chunk:
            flush(chunk_cases)
            chunk_cases = []
            if n_strict >= args.min_strict and n_relaxed >= args.min_relaxed:
                done = True
                break
    if chunk_cases and not done:
        flush(chunk_cases)
    df = pd.DataFrame(rows).sort_values("scan_rank").reset_index(drop=True)
    df.to_parquet(os.path.join(out_dir, "filter_scan.parquet"), index=False)
    pd.DataFrame(rejects).to_parquet(os.path.join(out_dir, "filter_rejects.parquet"), index=False)
    strict_ids = df[df.strict].case_id.tolist()[: args.min_strict]
    relaxed_ids = df[df.relaxed].case_id.tolist()[: args.min_relaxed]
    sets = {"scan": {"records_scanned": int(scan_rank), "tokenizable": int(len(df)), "rejected": int(len(rejects)),
                     "n_strict_total": int(n_strict), "n_relaxed_total": int(n_relaxed),
                     "strict_pass_rate_tokenizable": float(df.strict.mean()), "relaxed_pass_rate_tokenizable": float(df.relaxed.mean()),
                     "reject_reasons": pd.Series([r["reason"] for r in rejects]).value_counts().to_dict(),
                     "embed_std": eng.embed_std, "sigma": sigma, "pass_times_s": pass_times}}
    if len(strict_ids) >= args.min_strict:
        d, v = split_cases(strict_ids, args.min_strict // 2, 0)
        sets["strict"] = {"discovery": d, "validation": v, "scan_rank_last": int(df[df.case_id == strict_ids[-1]].scan_rank.iloc[0])}
    if len(relaxed_ids) >= args.min_relaxed:
        d, v = split_cases(relaxed_ids, args.min_relaxed // 2, 0)
        sets["relaxed"] = {"discovery": d, "validation": v, "scan_rank_last": int(df[df.case_id == relaxed_ids[-1]].scan_rank.iloc[0])}
    if m["paper_key"]:
        pids = load_paper_ids(m["paper_key"])
        sets["paper"] = {"discovery": pids["discovery"], "validation": pids["validation"]}
        allp = pids["discovery"] + pids["validation"]
        rank_of = {int(recs[ri]["case_id"]): r for r, ri in enumerate(shuffled_order(len(recs), 0))}
        pr = sorted(rank_of[c] for c in allp)
        in_df = df[df.case_id.isin(allp)]
        sets["paper_check"] = {
            "paper_ids_scan_rank_min": pr[0], "paper_ids_scan_rank_max": pr[-1], "paper_ids_scan_rank_median": pr[len(pr) // 2],
            "our_scan_last_rank": int(scan_rank),
            "paper_ids_within_scanned_prefix": int(sum(r < scan_rank for r in pr)),
            "paper_ids_tokenizable_in_scan": int(len(in_df)),
            "paper_ids_strict_in_scan": int(in_df.strict.sum()), "paper_ids_relaxed_in_scan": int(in_df.relaxed.sum()),
            "overlap_paper_vs_our_strict256": len(set(allp) & set(strict_ids)),
            "overlap_paper_vs_our_relaxed512": len(set(allp) & set(relaxed_ids)),
            "paper_ids_rejected_by_tokenizer": [r["case_id"] for r in rejects if r["case_id"] in set(allp)],
        }
    with open(os.path.join(out_dir, "case_sets.json"), "w") as f:
        json.dump(sets, f, indent=1)
    log("scan summary:", json.dumps(sets["scan"]))
    if "paper_check" in sets:
        log("paper check:", json.dumps(sets["paper_check"]))
    log(f"strict={len(strict_ids)} relaxed={len(relaxed_ids)} written to {out_dir}")


if __name__ == "__main__":
    main()
