"""ext13 step 2 (GPU): competence scan + association probe for the attribute -> entity WinoGrande pairs (F2, F1).

Per pair, prefill rows only (one engine pass per chunk), all scored as Delta = logit(entity of A) - logit(entity of B):
  clean_a / clean_b        the two twins' prompts (STR directions)
  probe_a1, probe_a2       association probe with A's trigger word, both candidate orders
  probe_b1, probe_b2       the same with B's trigger word
Pair flags (thresholds of ext6 / Phase 3; association rule = RESEARCH_PLAN.md Phase 5, literature-based):
  margin        Delta(clean_a) >= +1 and Delta(clean_b) <= -1
  probe_right_a mean(probe_a1, probe_a2) >= +1;  probe_right_b  mean(probe_b1, probe_b2) <= -1
  assoc_cat     associative (both right) | half (one right) | hard (neither right)
  balanced      margin & main & assoc_cat != associative        (main set pool)
F1 is scanned on a seed-0 sample of its main pairs (--f1-sample) plus all its connective pairs; F2 in full.

Usage: python scripts/ext13_scan.py <proto: qwen3|mixtral_bos> [--formats F2,F1] [--f1-sample 3000] [--max-tokens 140000]
       (--formats F1 --f1-sample 0 = the full F1 pool; summary / run_meta entries are merged, not overwritten)
Outputs results/wino_ref_<proto>/scan_pairs_<fmt>.parquet, scan_rows_<fmt>.parquet, scan_summary.json, run_meta.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.models import MODELS, RESULTS

PROTOS = {"qwen3": ("qwen3", True), "mixtral_bos": ("mixtral", True)}
D = "/home/ubuntu/MOE/data/wino_ref"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chunks_by_tokens(rows, max_tokens):
    order = sorted(range(len(rows)), key=lambda j: len(rows[j][2].ids))
    out, cur, cur_max = [], [], 0
    for j in order:
        T = len(rows[j][2].ids)
        if cur and max(cur_max, T) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, cur_max = [], 0
        cur.append(j)
        cur_max = max(cur_max, T)
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto", choices=list(PROTOS))
    ap.add_argument("--f1-sample", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=140000)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--formats", default="F2,F1", help="comma list of formats to scan (pair files pairs_<fmt>_train_xl_<proto>)")
    ap.add_argument("--verbs", action="store_true", help="scan the P5-3 verb-swap pairs (verbswap_<dir>_<proto>.parquet; "
                    "clean rows only, no probes) into scan_pairs_verbs_<dir>.parquet")
    ap.add_argument("--verb-dirs", default="ent2attr,attr2ent")
    args = ap.parse_args()
    key, st = PROTOS[args.proto]
    m = MODELS[key]
    od = os.path.join(RESULTS, f"wino_ref_{args.proto}")
    os.makedirs(od, exist_ok=True)
    if args.verbs:
        return scan_verbs(args, key, m, od)
    sets = {}
    for fmt in args.formats.split(","):
        p = pd.read_parquet(f"{D}/pairs_{fmt}_train_xl_{args.proto}.parquet")
        if fmt == "F1" and args.f1_sample:
            mains = p[p.main].sample(n=min(args.f1_sample, int(p.main.sum())), random_state=0)
            p = pd.concat([mains, p[~p.main]]).sort_index()
        if args.limit:
            p = p.iloc[: args.limit]
        sets[fmt] = p.reset_index(drop=True)
    meta = {"model": key, "repo": m["repo"], "proto": args.proto, "special_tokens": st, "n_pairs": {k: len(v) for k, v in sets.items()},
            "f1_sample": args.f1_sample, "margin": args.margin, "max_tokens": args.max_tokens,
            "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "complete": False}
    mkey = "scan" if args.formats == "F2,F1" else "scan_" + args.formats.replace(",", "_")
    rm = os.path.join(od, "run_meta.json")
    allm = json.load(open(rm)) if os.path.exists(rm) else {}
    allm[mkey] = meta
    json.dump(allm, open(rm, "w"), indent=1)

    from moetrace.engine import Engine, PrefillSpec
    eng = Engine(m["repo"])
    sp = os.path.join(od, "scan_summary.json")
    summ = json.load(open(sp)) if os.path.exists(sp) else {}
    for fmt, pairs in sets.items():
        rows = []
        for i, r in enumerate(pairs.itertuples()):
            rows.append((i, "clean_a", PrefillSpec(json.loads(r.ids_a), int(r.trig_a), int(r.trig_b))))
            rows.append((i, "clean_b", PrefillSpec(json.loads(r.ids_b), int(r.trig_a), int(r.trig_b))))
            pr = json.loads(r.probe_json)
            for k in ("a1", "a2", "b1", "b2"):
                q, qi, ea, eb = pr[k]
                rows.append((i, "probe_" + k, PrefillSpec(qi, int(ea), int(eb))))
        ch = chunks_by_tokens(rows, args.max_tokens)
        delta = np.full(len(rows), np.nan)
        top1 = np.full(len(rows), -1, dtype=np.int64)
        log(f"{fmt}: {len(pairs)} pairs, {len(rows)} rows, {len(ch)} passes")
        for ci, c in enumerate(ch):
            res = eng.run([rows[j][2] for j in c], [], record_routing=False, log=log if ci == 0 else None)
            delta[c], top1[c] = res.delta, res.top1
            log(f"  pass {ci + 1}/{len(ch)} done ({res.extra['total_s']:.0f}s)")
        long = pd.DataFrame({"pair": [r[0] for r in rows], "kind": [r[1] for r in rows], "delta": delta, "top1": top1})
        long.to_parquet(os.path.join(od, f"scan_rows_{fmt}.parquet"), index=False)
        w = long.pivot(index="pair", columns="kind", values="delta")
        t1 = long.pivot(index="pair", columns="kind", values="top1")
        p = pairs.drop(columns=["ids_a", "ids_b", "probe_json"]).copy()
        p["d_clean_a"], p["d_clean_b"] = w["clean_a"].values, w["clean_b"].values
        p["probe_a"] = (w["probe_a1"].values + w["probe_a2"].values) / 2
        p["probe_b"] = (w["probe_b1"].values + w["probe_b2"].values) / 2
        p["drop_str"] = p.d_clean_a - p.d_clean_b
        p["probe_drop"] = p.probe_a - p.probe_b
        p["margin"] = (p.d_clean_a >= args.margin) & (p.d_clean_b <= -args.margin)
        p["top1_both"] = (t1["clean_a"].values == p.trig_a) & (t1["clean_b"].values == p.trig_b)
        p["correct_both"] = (p.d_clean_a > 0) & (p.d_clean_b < 0)
        p["probe_right_a"] = p.probe_a >= args.margin
        p["probe_right_b"] = p.probe_b <= -args.margin
        nr = p.probe_right_a.astype(int) + p.probe_right_b.astype(int)
        p["assoc_cat"] = np.where(nr == 2, "associative", np.where(nr == 1, "half", "hard"))
        p["balanced"] = p.margin & p.main & (p.assoc_cat != "associative")
        p.to_parquet(os.path.join(od, f"scan_pairs_{fmt}.parquet"), index=False)
        M = p.margin
        mm = M & p.main

        def c(x):
            return int(x.sum())
        summ[fmt] = {"pairs": len(p), "main": c(p.main), "correct_both": c(p.correct_both), "margin": c(M),
                     "margin_main": c(mm), "margin_main_names": c(mm & p.names), "margin_main_objects": c(mm & ~p.names),
                     "margin_connective": c(M & ~p.main),
                     "margin_main_associative": c(mm & (p.assoc_cat == "associative")),
                     "margin_main_half": c(mm & (p.assoc_cat == "half")), "margin_main_hard": c(mm & (p.assoc_cat == "hard")),
                     "balanced": c(p.balanced), "balanced_names": c(p.balanced & p.names),
                     "balanced_objects": c(p.balanced & ~p.names), "hard": c(mm & (p.assoc_cat == "hard")),
                     "margin_main_top1_both": c(mm & p.top1_both),
                     "mean_drop_margin_main": float(p.drop_str[mm].mean()) if c(mm) else None,
                     "probe_drop_over_drop_margin_main": float(p.probe_drop[mm].sum() / p.drop_str[mm].sum()) if c(mm) else None,
                     "n_passes": len(ch)}
        log(fmt, json.dumps(summ[fmt]))
    json.dump(summ, open(sp, "w"), indent=1)
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True})
    allm = json.load(open(rm)) if os.path.exists(rm) else {}
    allm[mkey] = meta
    json.dump(allm, open(rm, "w"), indent=1)


def scan_verbs(args, key, m, od):
    """P5-3: Delta on both twins of every verb-swap pair (orig / neutral / unrelated x name pairs), both directions."""
    meta = {"model": key, "repo": m["repo"], "proto": args.proto, "mode": "verbs", "margin": args.margin,
            "max_tokens": args.max_tokens, "command": "python " + " ".join(sys.argv),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    from moetrace.engine import Engine, PrefillSpec
    eng = Engine(m["repo"])
    sp = os.path.join(od, "scan_summary_verbs.json")
    summ = json.load(open(sp)) if os.path.exists(sp) else {}
    for d in args.verb_dirs.split(","):
        pairs = pd.read_parquet(f"{D}/verbswap_{d}_{args.proto}.parquet")
        if args.limit:
            pairs = pairs.iloc[: args.limit]
        rows = []
        for i, r in enumerate(pairs.itertuples()):
            for kind, ids in (("clean_a", r.ids_a), ("clean_b", r.ids_b)):
                rows.append((i, kind, PrefillSpec(json.loads(ids), int(r.trig_a), int(r.trig_b))))
        ch = chunks_by_tokens(rows, args.max_tokens)
        delta = np.full(len(rows), np.nan)
        log(f"verbs {d}: {len(pairs)} pairs, {len(rows)} rows, {len(ch)} passes")
        for ci, c in enumerate(ch):
            res = eng.run([rows[j][2] for j in c], [], record_routing=False, log=log if ci == 0 else None)
            delta[c] = res.delta
            log(f"  pass {ci + 1}/{len(ch)} done ({res.extra['total_s']:.0f}s)")
        p = pairs.drop(columns=["ids_a", "ids_b"]).copy()
        p["d_clean_a"], p["d_clean_b"] = delta[0::2], delta[1::2]
        p["drop_str"] = p.d_clean_a - p.d_clean_b
        p["margin"] = (p.d_clean_a >= args.margin) & (p.d_clean_b <= -args.margin)
        p["correct_both"] = (p.d_clean_a > 0) & (p.d_clean_b < 0)
        p.to_parquet(os.path.join(od, f"scan_pairs_verbs_{d}.parquet"), index=False)
        g = p.groupby("variant")
        summ[d] = {v: {"pairs": int(len(x)), "correct_both": int(x.correct_both.sum()), "margin": int(x.margin.sum()),
                       "mean_drop": float(x.drop_str.mean()), "median_drop": float(x.drop_str.median()),
                       "seeds_with_margin": int(x[x.margin].seed.nunique())} for v, x in g}
        log(d, json.dumps(summ[d]))
    json.dump(summ, open(sp, "w"), indent=1)
    meta["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rm = os.path.join(od, "run_meta.json")
    allm = json.load(open(rm)) if os.path.exists(rm) else {}
    allm["scan_verbs_" + args.verb_dirs.replace(",", "_")] = meta
    json.dump(allm, open(rm, "w"), indent=1)


if __name__ == "__main__":
    main()
