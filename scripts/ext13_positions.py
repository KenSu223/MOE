"""ext13 (CPU): token positions for the head / grid analyses of the attribute -> entity pair files (F1, F2R).

Adds to data/wino_ref/pairs_<fmt>_train_xl_<proto>.parquet (in place):
  mention_a_pos, mention_b_pos  first occurrence of ans_a / ans_b in the WinoGrande sentence part of prompt A (JSON token
                                positions; read by scripts/ext7_wino_heads.py instead of its Phase-3 'context' lookup)
  blank_pos                     F1: the '_' token(s); F2R: the 'one of them' tokens (first occurrence)
Prompt A and B share these positions (token-aligned pairs; only str_pos differ).
Usage: python scripts/ext13_positions.py [--formats F1,F2R] [--protos qwen3,mixtral_bos]
"""
import argparse
import json
import re
import sys

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
D = "/home/ubuntu/MOE/data/wino_ref"
PROTOS = {"qwen3": ("qwen3", True), "mixtral_bos": ("mixtral", True)}


def span(enc, start, end):
    return [i for i, ((a, b), sp) in enumerate(zip(enc["offset_mapping"], enc["special_tokens_mask"]))
            if not sp and b > a and a < end and b > start]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default="F1,F2R")
    ap.add_argument("--protos", default="qwen3,mixtral_bos")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    from moetrace.models import MODELS
    for proto in a.protos.split(","):
        key, st = PROTOS[proto]
        tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[key]["repo"]))
        for fmt in a.formats.split(","):
            path = f"{D}/pairs_{fmt}_train_xl_{proto}.parquet"
            p = pd.read_parquet(path)
            ma, mb, bl, nf = [], [], [], 0
            for r in p.itertuples():
                enc = tok(r.prompt_a, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=st)
                assert list(enc["input_ids"]) == json.loads(r.ids_a), r.pair_id
                if fmt == "F1":
                    lo = len("Fill in the blank: ")
                    hi = r.prompt_a.rindex("\nAnswer:")
                    bj = r.prompt_a.index("_", lo)
                    blank = span(enc, bj, bj + 1)
                else:  # F2R: sentence body = everything before " The one "
                    lo, hi = 0, r.prompt_a.rindex(" The one ")
                    bj = r.prompt_a.lower().index("one of them")
                    blank = span(enc, bj, bj + len("one of them"))
                res = []
                for ans in (r.ans_a, r.ans_b):
                    m = None
                    for cand in (ans, ans.lower(), ans[:1].upper() + ans[1:]):
                        m = re.compile(r"\b" + re.escape(cand) + r"\b").search(r.prompt_a, lo, hi)
                        if m:
                            break
                    res.append(span(enc, m.start(), m.end()) if m else [])
                nf += bool(res[0]) + bool(res[1])
                ma.append(json.dumps(res[0]))
                mb.append(json.dumps(res[1]))
                bl.append(json.dumps(blank))
            p["mention_a_pos"], p["mention_b_pos"], p["blank_pos"] = ma, mb, bl
            p.to_parquet(path, index=False)
            print(proto, fmt, f"mentions found {nf} of {2 * len(p)}; blank found {sum(x != '[]' for x in bl)} of {len(p)}")


if __name__ == "__main__":
    main()
