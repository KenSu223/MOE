"""ext14 Phase 6 tokenizer identity check for the Stage-2 models (CPU, agent p6-data).

For each Stage-2 checkpoint: the tokenizer of its family (Mistral-7B -> the Mixtral-8x7B build family 'mistral', Qwen2 / Qwen3 dense ->
the Qwen3-30B-A3B build family 'qwen') is compared on (1) the vocabulary, (2) the special tokens added by default (BOS), (3) the token
ids of EVERY prompt of every build parquet (prompt_a / prompt_b vs ids_a / ids_b) and of the probe-spec prompts. If all are identical,
the family's pair files are reused for that model; otherwise the mismatching cells are listed for a rebuild.
Outputs data/dial/qa/tokenizer_identity.json and .md.
Usage: python scripts/ext14_build_tokcheck.py [--repos ...]
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
from moetrace import ext14_data as X

STAGE2 = {"Mistral-7B-v0.1 (mistral7b_bos)": ("mistralai/Mistral-7B-v0.1", "mistral"),
          "Qwen2-7B (qwen2_7b)": ("Qwen/Qwen2-7B", "qwen"), "Qwen2-57B-A14B (qwen2_57b)": ("Qwen/Qwen2-57B-A14B", "qwen"),
          "Qwen3-1.7B-Base (qwen3_1p7b)": ("Qwen/Qwen3-1.7B-Base", "qwen"), "Qwen3-4B-Base (qwen3_4b)": ("Qwen/Qwen3-4B-Base", "qwen"),
          "Qwen3-8B-Base (qwen3_8b)": ("Qwen/Qwen3-8B-Base", "qwen"), "Qwen3-14B-Base (qwen3_14b)": ("Qwen/Qwen3-14B-Base", "qwen")}


def load(repo):
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    return AutoTokenizer.from_pretrained(snapshot_dir(repo))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    out = {}
    ref = {fam: X.family_tokenizer(fam)[0] for fam in ("qwen", "mistral")}
    builds = {fam: sorted(glob.glob(f"{X.BUILD}/*_{fam}.parquet")) for fam in ref}
    for name, (repo, fam) in STAGE2.items():
        try:
            tok = load(repo)
        except Exception as e:
            out[name] = {"repo": repo, "error": repr(e)[:300]}
            continue
        r = ref[fam]
        va, vb = tok.get_vocab(), r.get_vocab()
        shared = set(va) & set(vb)
        e = {"repo": repo, "family": fam, "vocab_identical": va == vb,
             "vocab_same_ids_on_shared": all(va[k] == vb[k] for k in shared), "only_in_model": sorted(set(va) - set(vb))[:20],
             "only_in_family": sorted(set(vb) - set(va))[:25],
             "bos_default": tok("x")["input_ids"][:1] == r("x")["input_ids"][:1] and len(tok("x")["input_ids"]) == len(r("x")["input_ids"]),
             "cells": {}}
        n_all = mism_all = 0
        for p in builds[fam]:
            cell = os.path.basename(p)[: -len(f"_{fam}.parquet")]
            df = pd.read_parquet(p)
            if "prompt" in df.columns and "ids" in df.columns:  # probe specs
                texts, ids = list(df.prompt), list(df.ids)
            elif "prompt_a" in df.columns:
                texts = list(df.prompt_a) + list(df.prompt_b)
                ids = list(df.ids_a) + list(df.ids_b)
            else:
                continue
            mism = sum(list(tok(t, add_special_tokens=True)["input_ids"]) != json.loads(i) for t, i in zip(texts, ids))
            e["cells"][cell] = {"prompts": len(texts), "mismatch": int(mism)}
            n_all += len(texts)
            mism_all += mism
        e["prompts_checked"], e["prompts_mismatch"] = n_all, mism_all
        # reuse when every prompt tokenises identically and the shared vocabulary has the same ids (answer tokens are ordinary
        # vocabulary items; Qwen2 lacks only the 23 special tokens Qwen3 added, e.g. <think>)
        e["reuse_family_pairs"] = bool(e["vocab_same_ids_on_shared"] and not e["only_in_model"] and e["bos_default"] and mism_all == 0)
        out[name] = e
        print(name, {k: v for k, v in e.items() if k != "cells"}, flush=True)
    json.dump(out, open(f"{X.DIAL}/qa/tokenizer_identity.json", "w"), indent=1)
    lines = ["# Stage-2 tokenizer identity (p6-data)", "", "| Model | Family | Vocab identical | Same ids on the shared vocabulary (tokens only in the family tokenizer) | Same default special tokens | Prompts checked | Mismatches | Reuse family pair files |",
             "|---|---|---|---|---|---|---|---|"]
    for name, e in out.items():
        if "error" in e:
            lines.append(f"| {name} | | error: {e['error']} | | | | |")
            continue
        lines.append(f"| {name} | {e['family']} | {e['vocab_identical']} | {e['vocab_same_ids_on_shared']} ({len(e['only_in_family'])}: {', '.join(e['only_in_family'][:6])}) | {e['bos_default']} | {e['prompts_checked']} | {e['prompts_mismatch']} | {e['reuse_family_pairs']} |")
    open(f"{X.DIAL}/qa/tokenizer_identity.md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
