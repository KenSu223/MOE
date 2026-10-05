"""ext7 step 1 (CPU): WinoGrande twins -> token-symmetric STR pairs per tokenizer / protocol (rules W1-W6 in
moetrace/ext7_wino.py).

Usage: python scripts/ext7_wino_build.py [--split train_xl] [--protos qwen3,mixtral_nobos,mixtral_bos,olmoe]
Outputs data/wino_str/pairs_<split>_<proto>.parquet and data/wino_str/funnel_<split>.json
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from transformers import AutoTokenizer

from moetrace.arch import snapshot_dir
from moetrace.ext7_wino import build_pairs, debiased_qids
from moetrace.models import MODELS

OUT = "/home/ubuntu/MOE/data/wino_str"
PROTOS = {"qwen3": ("qwen3", True), "mixtral_nobos": ("mixtral", False), "mixtral_bos": ("mixtral", True), "olmoe": ("olmoe", True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train_xl")
    ap.add_argument("--protos", default=",".join(PROTOS))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    deb = debiased_qids()
    funnels = {}
    for proto in args.protos.split(","):
        key, st = PROTOS[proto]
        tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[key]["repo"]))
        pairs, f = build_pairs(tok, st, args.split)
        df = pd.DataFrame([p.to_row() for p in pairs])
        df["debiased"] = df.q_a.isin(deb) | df.q_b.isin(deb)
        df.to_parquet(os.path.join(OUT, f"pairs_{args.split}_{proto}.parquet"), index=False)
        f.update({"special_tokens": st, "repo": MODELS[key]["repo"],
                  "kept_one_token_option": int((df.n_opt_tokens == 1).sum()),
                  "kept_names": int(df.names.sum()), "kept_trigger_in_context": int(df.trigger_in_context.sum()),
                  "kept_debiased": int(df.debiased.sum()), "median_tokens": float(df.n_tokens.median()),
                  "max_tokens": int(df.n_tokens.max())})
        funnels[proto] = f
        print(time.strftime("%H:%M:%S"), proto, json.dumps({k: v for k, v in f.items() if k != "steps"}), flush=True)
        for s in f["steps"]:
            print(f"   {s['rule']:<36} -{s['dropped']:>6}  -> {s['remaining']}")
    with open(os.path.join(OUT, f"funnel_{args.split}.json"), "w") as fh:
        json.dump(funnels, fh, indent=1)


if __name__ == "__main__":
    main()
