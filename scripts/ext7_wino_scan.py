"""ext7 step 2 (GPU): competence scan of the WinoGrande STR pairs (one prefill-only engine pass per chunk).

STR only (user decision 2026-10-04: no Gaussian-noise rows, filter or control). Per pair (built by
scripts/ext7_wino_build.py) four prefill rows, all scored with true = trigger of A, foil = trigger of B, i.e.
Delta = logit(r_A) - logit(r_B) at the final position:
  clean_a / clean_b   prompt A / prompt B (the two STR directions; B is A with the filled option swapped)
  local_a / local_b   context ablation: only "<determiner> <option> ..." from the blank on (association baseline)
Filters written to the pair table (thresholds of ext6):
  margin       Delta_A >= +1 and Delta_B <= -1 (clean margin both ways = paper clean margin + ext6 donor margin;
               the STR drop is then >= 2) -- the primary pair set
  top1_both    the trigger is the top-1 token after both prompts (Zhang & Nanda's arithmetic filter; sensitivity)
  assoc        the context-free prompts already pass the margin both ways (shortcut flag, recorded, not filtered)

Usage: python scripts/ext7_wino_scan.py <proto: qwen3|mixtral_nobos|mixtral_bos|olmoe> [--split train_xl]
       [--max-tokens 140000] [--limit N]
Outputs results/wino_<proto>/scan_pairs.parquet, scan_summary.json, run_meta.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from transformers import AutoTokenizer

from moetrace.arch import snapshot_dir
from moetrace.models import MODELS, RESULTS

PROTOS = {"qwen3": ("qwen3", True), "mixtral_nobos": ("mixtral", False), "mixtral_bos": ("mixtral", True), "olmoe": ("olmoe", True)}
DETS = {"the", "a", "an", "his", "her", "their", "my", "our", "your", "its", "this", "that", "these", "those"}
PAIRS = "/home/ubuntu/MOE/data/wino_str"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def local_prompt(context: str, ans: str) -> str:
    """'... because the _ was too' -> 'The bag was too' (determiner kept when it directly precedes the blank)."""
    i = context.index("_")
    before = context[:i].rstrip().split(" ")
    start = before[-1] + " " if before and before[-1].lower() in DETS else ""
    s = start + ans + context[i + 1:]
    return s[0].upper() + s[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto", choices=list(PROTOS))
    ap.add_argument("--split", default="train_xl")
    ap.add_argument("--max-tokens", type=int, default=140000, help="padded prefill tokens per engine pass")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only (smoke test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    key, st = PROTOS[args.proto]
    m = MODELS[key]
    out = args.out or f"wino_{args.proto}" + ("" if args.split == "train_xl" else f"_{args.split}")
    od = os.path.join(RESULTS, out)
    os.makedirs(od, exist_ok=True)
    pairs = pd.read_parquet(os.path.join(PAIRS, f"pairs_{args.split}_{args.proto}.parquet"))
    if args.limit:
        pairs = pairs.iloc[: args.limit].copy()
    pairs = pairs.reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(snapshot_dir(m["repo"]))
    meta = {"model": key, "repo": m["repo"], "proto": args.proto, "special_tokens": st, "split": args.split,
            "n_pairs": len(pairs), "margin": args.margin,
            "max_tokens": args.max_tokens, "command": "python " + " ".join(sys.argv),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    json.dump({"scan": meta}, open(os.path.join(od, "run_meta.json"), "w"), indent=1)

    from moetrace.engine import Engine, PrefillSpec
    eng = Engine(m["repo"])
    log(f"{args.proto}: {len(pairs)} pairs")
    rows = []  # (pair index, kind, PrefillSpec)
    for i, r in enumerate(pairs.itertuples()):
        ia, ib = json.loads(r.ids_a), json.loads(r.ids_b)
        rows.append((i, "clean_a", PrefillSpec(ia, r.trig_a, r.trig_b)))
        rows.append((i, "clean_b", PrefillSpec(ib, r.trig_a, r.trig_b)))
        for k, ans in (("local_a", r.ans_a), ("local_b", r.ans_b)):
            ids = tok(local_prompt(r.context, ans), add_special_tokens=st)["input_ids"]
            rows.append((i, k, PrefillSpec(ids, r.trig_a, r.trig_b)))
    order = sorted(range(len(rows)), key=lambda j: len(rows[j][2].ids))
    chunks, cur, cur_max = [], [], 0
    for j in order:
        T = len(rows[j][2].ids)
        if cur and max(cur_max, T) * (len(cur) + 1) > args.max_tokens:
            chunks.append(cur)
            cur, cur_max = [], 0
        cur.append(j)
        cur_max = max(cur_max, T)
    if cur:
        chunks.append(cur)
    delta = np.full(len(rows), np.nan)
    top1 = np.full(len(rows), -1, dtype=np.int64)
    lt, lf = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    times = []
    for ci, ch in enumerate(chunks):
        log(f"pass {ci + 1}/{len(chunks)}: {len(ch)} rows, T <= {max(len(rows[j][2].ids) for j in ch)}")
        res = eng.run([rows[j][2] for j in ch], [], record_routing=False, log=log if ci == 0 else None)
        times.append(res.extra["total_s"])
        delta[ch], top1[ch] = res.delta, res.top1
        lt[ch], lf[ch] = res.logit_true, res.logit_foil
    long = pd.DataFrame({"pair": [r[0] for r in rows], "kind": [r[1] for r in rows], "delta": delta, "top1": top1,
                         "logit_true": lt, "logit_foil": lf})
    wide = long.pivot(index="pair", columns="kind", values="delta")
    t1 = long.pivot(index="pair", columns="kind", values="top1")
    p = pairs.drop(columns=["ids_a", "ids_b"]).copy()
    for k in ("clean_a", "clean_b", "local_a", "local_b"):
        p["d_" + k] = wide[k].values
    p["top1_a"], p["top1_b"] = t1["clean_a"].values, t1["clean_b"].values
    p["drop_str"] = p.d_clean_a - p.d_clean_b  # identical for both directions (sign-flipped metric)
    p["margin"] = (p.d_clean_a >= args.margin) & (p.d_clean_b <= -args.margin)
    p["top1_both"] = (p.top1_a == p.trig_a) & (p.top1_b == p.trig_b)
    p["correct_both"] = (p.d_clean_a > 0) & (p.d_clean_b < 0)  # Elazar group scoring at the trigger token
    p["assoc"] = (p.d_local_a >= args.margin) & (p.d_local_b <= -args.margin)
    p.to_parquet(os.path.join(od, "scan_pairs.parquet"), index=False)
    long.to_parquet(os.path.join(od, "scan_rows.parquet"), index=False)

    def cnt(mask):
        return int(mask.sum())
    M = p.margin
    summ = {"n_pairs": len(p), "correct_both": cnt(p.correct_both), "margin": cnt(M),
            "margin_and_top1_both": cnt(M & p.top1_both),
            "margin_assoc": cnt(M & p.assoc), "margin_not_assoc": cnt(M & ~p.assoc), "margin_names": cnt(M & p.names),
            "margin_one_token_option": cnt(M & (p.n_opt_tokens == 1)), "margin_debiased": cnt(M & p.debiased),
            "margin_trigger_in_context": cnt(M & p.trigger_in_context),
            "mean_d_clean_a_margin": float(p.d_clean_a[M].mean()), "mean_d_clean_b_margin": float(p.d_clean_b[M].mean()),
            "mean_drop_str_margin": float(p.drop_str[M].mean()),
            "n_passes": len(chunks), "pass_times_s": times}
    json.dump(summ, open(os.path.join(od, "scan_summary.json"), "w"), indent=1)
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True})
    json.dump({"scan": meta}, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    log("summary:", json.dumps({k: v for k, v in summ.items() if k != "pass_times_s"}))


if __name__ == "__main__":
    main()
