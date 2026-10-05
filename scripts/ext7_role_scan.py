"""ext7-controls W7 step 2 (GPU, prefill only): competence scan of the WinoGrande role-swap STR pairs.

Per role pair two prefill rows, both scored with true = trig_a, foil = trig_b (Delta = LD(trig_a, trig_b)):
  a  the clean prompt (WinoGrande prompt A or B, filled with its own answer)
  b  the role-swapped prompt (the two first mentions exchanged; its answer is trig_b)
margin = Delta_a >= +1 and Delta_b <= -1 (margin both ways, as the WinoGrande option swap and ext6); top1_both = the
answer is the top-1 token after both prompts (sensitivity). No Gaussian-noise rows (STR only).

Usage: python scripts/ext7_role_scan.py <proto: qwen3|mixtral_bos> [--max-tokens 140000]
Outputs results/wino_role_<proto>/scan_pairs.parquet, scan_rows.parquet, scan_summary.json, run_meta.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import pandas as pd
from moetrace import ext7_controls as C
from moetrace.models import MODELS, RESULTS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto", choices=list(C.PROTOS))
    ap.add_argument("--max-tokens", type=int, default=140000)
    args = ap.parse_args()
    key, st = C.PROTOS[args.proto]
    od = os.path.join(RESULTS, f"wino_role_{args.proto}")
    os.makedirs(od, exist_ok=True)
    pairs = pd.read_parquet(os.path.join(C.ROLE_DIR, f"pairs_{args.proto}.parquet"))
    meta = {"model": key, "repo": MODELS[key]["repo"], "proto": args.proto, "special_tokens": st, "n_pairs": len(pairs),
            "margin": C.MARGIN, "pairs": f"data/wino_role/pairs_{args.proto}.parquet", "command": "python " + " ".join(sys.argv),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    json.dump({"scan": meta}, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    from moetrace.engine import Engine
    eng = Engine(MODELS[key]["repo"])
    long = C.run_scan(eng, C.scan_rows_to_engine(pairs), args.max_tokens, log)
    wide = long.pivot(index="pair", columns="kind", values="delta")
    t1 = long.pivot(index="pair", columns="kind", values="top1")
    p = pairs.drop(columns=["ids_a", "ids_b"]).copy()
    p["d_a"], p["d_b"] = wide["a"].values, wide["b"].values
    p["top1_a"], p["top1_b"] = t1["a"].values, t1["b"].values
    p["drop"] = p.d_a - p.d_b
    p["margin"] = (p.d_a >= C.MARGIN) & (p.d_b <= -C.MARGIN)
    p["correct_both"] = (p.d_a > 0) & (p.d_b < 0)
    p["top1_both"] = (p.top1_a == p.trig_a) & (p.top1_b == p.trig_b)
    p.to_parquet(os.path.join(od, "scan_pairs.parquet"), index=False)
    long.to_parquet(os.path.join(od, "scan_rows.parquet"), index=False)
    M = p.margin
    tw = p[M].groupby("twin_id").size()
    summ = {"n_pairs": len(p), "n_twins": int(p.twin_id.nunique()), "correct_both": int(p.correct_both.sum()), "margin": int(M.sum()),
            "margin_twins": int(len(tw)), "margin_twins_both_roles": int((tw == 2).sum()), "margin_top1_both": int((M & p.top1_both).sum()),
            "mean_d_a_margin": float(p.d_a[M].mean()), "mean_d_b_margin": float(p.d_b[M].mean()), "mean_drop_margin": float(p["drop"][M].mean()),
            "mean_d_a_all": float(p.d_a.mean()), "mean_d_b_all": float(p.d_b.mean()), "pass_times_s": long.attrs.get("pass_times_s")}
    json.dump(summ, open(os.path.join(od, "scan_summary.json"), "w"), indent=1)
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True})
    json.dump({"scan": meta}, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    log("summary:", json.dumps({k: v for k, v in summ.items() if k != "pass_times_s"}))


if __name__ == "__main__":
    main()
