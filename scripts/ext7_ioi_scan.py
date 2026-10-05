"""ext7-controls W8 step 2 (GPU, prefill only): competence scan of the IOI STR pairs (data/ioi/, identical items for
both models).

Per item three prefill rows, all scored with true = IO, foil = S (Delta = LD(IO, S)):
  clean  "Then, John and Mary went to the store. John gave a drink to"            (answer IO)
  s2io   corruption (i)  S2 -> IO: "... Mary gave a drink to"                       (answer S)
  s1io   corruption (ii) S1, IO -> two other names: "Then, Alice and Carol ... John gave a drink to"
Competence: (i) margin both ways (Delta_clean >= 1, Delta_s2io <= -1); (ii) clean margin (Delta_clean >= 1); the drop of
(ii) is reported. No Gaussian-noise rows (STR only).

Usage: python scripts/ext7_ioi_scan.py <proto: qwen3|mixtral_bos> [--max-tokens 140000]
Outputs results/ioi_<proto>/scan_pairs.parquet, scan_rows.parquet, scan_summary.json, run_meta.json
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
    od = os.path.join(RESULTS, f"ioi_{args.proto}")
    os.makedirs(od, exist_ok=True)
    p1 = pd.read_parquet(os.path.join(C.IOI_DIR, f"pairs_s2io_{args.proto}.parquet"))
    p2 = pd.read_parquet(os.path.join(C.IOI_DIR, f"pairs_s1io_{args.proto}.parquet"))
    assert (p1.pair_id == p2.pair_id).all() and (p1.ids_a == p2.ids_a).all()
    pairs = p1.copy()
    pairs["ids_c"] = p2.ids_b
    meta = {"model": key, "repo": MODELS[key]["repo"], "proto": args.proto, "special_tokens": st, "n_items": len(pairs),
            "margin": C.MARGIN, "pairs": [f"data/ioi/pairs_s2io_{args.proto}.parquet", f"data/ioi/pairs_s1io_{args.proto}.parquet"],
            "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    json.dump({"scan": meta}, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    from moetrace.engine import Engine
    eng = Engine(MODELS[key]["repo"])
    long = C.run_scan(eng, C.scan_rows_to_engine(pairs, {"c": "ids_c"}), args.max_tokens, log)
    long["kind"] = long.kind.map({"a": "clean", "b": "s2io", "c": "s1io"})
    wide = long.pivot(index="pair", columns="kind", values="delta")
    t1 = long.pivot(index="pair", columns="kind", values="top1")
    p = pairs.drop(columns=["ids_a", "ids_b", "ids_c", "prompt_b", "corruption", "str_pos"]).copy()
    for k in ("clean", "s2io", "s1io"):
        p["d_" + k] = wide[k].values
        p["top1_" + k] = t1[k].values
    p["drop_s2io"] = p.d_clean - p.d_s2io
    p["drop_s1io"] = p.d_clean - p.d_s1io
    p["margin_clean"] = p.d_clean >= C.MARGIN
    p["margin_s2io"] = p.margin_clean & (p.d_s2io <= -C.MARGIN)
    p["top1_clean_io"] = p.top1_clean == p.trig_a
    p["top1_both_s2io"] = p.top1_clean_io & (p.top1_s2io == p.trig_b)
    p.to_parquet(os.path.join(od, "scan_pairs.parquet"), index=False)
    long.to_parquet(os.path.join(od, "scan_rows.parquet"), index=False)
    summ = {"n_items": len(p), "clean_correct": int((p.d_clean > 0).sum()), "margin_clean": int(p.margin_clean.sum()),
            "margin_s2io": int(p.margin_s2io.sum()), "top1_clean_io": int(p.top1_clean_io.sum()), "top1_both_s2io": int(p.top1_both_s2io.sum()),
            "mean_d_clean": float(p.d_clean.mean()), "mean_d_s2io": float(p.d_s2io.mean()), "mean_d_s1io": float(p.d_s1io.mean()),
            "by_template_type_margin_s2io": p.groupby("template_type").margin_s2io.mean().round(3).to_dict(),
            "pass_times_s": long.attrs.get("pass_times_s")}
    json.dump(summ, open(os.path.join(od, "scan_summary.json"), "w"), indent=1)
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True})
    json.dump({"scan": meta}, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    log("summary:", json.dumps({k: v for k, v in summ.items() if k != "pass_times_s"}))


if __name__ == "__main__":
    main()
