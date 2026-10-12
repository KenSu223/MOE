"""ext13 step 1 (CPU): build the attribute -> entity WinoGrande pair tables (F2 primary, F1 comparison) with association
probes, per model protocol.

Usage: python scripts/ext13_build.py [--protos qwen3,mixtral_bos] [--split train_xl] [--formats F2,F1,F2R]
Outputs data/wino_ref/pairs_<fmt>_<split>_<proto>.parquet, data/wino_ref/funnel_<split>.json
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from transformers import AutoTokenizer

from moetrace.arch import snapshot_dir
from moetrace.ext13_wino_ref import build_ref_pairs
from moetrace.models import MODELS

PROTOS = {"qwen3": ("qwen3", True), "mixtral_bos": ("mixtral", True), "mixtral_nobos": ("mixtral", False), "olmoe": ("olmoe", True)}
OUT = "/home/ubuntu/MOE/data/wino_ref"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protos", default="qwen3,mixtral_bos")
    ap.add_argument("--split", default="train_xl")
    ap.add_argument("--formats", default="F2,F1")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fpath = os.path.join(OUT, f"funnel_{args.split}.json")
    funnel = json.load(open(fpath)) if os.path.exists(fpath) else {}
    for proto in args.protos.split(","):
        key, st = PROTOS[proto]
        tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[key]["repo"]))
        for fmt in args.formats.split(","):
            pairs, f = build_ref_pairs(tok, st, fmt, args.split)
            df = pd.DataFrame([p.to_row() for p in pairs])
            df.to_parquet(os.path.join(OUT, f"pairs_{fmt}_{args.split}_{proto}.parquet"), index=False)
            funnel.setdefault(proto, {})[fmt] = f
            print(proto, fmt, json.dumps(f))
    json.dump(funnel, open(fpath, "w"), indent=1)


if __name__ == "__main__":
    main()
