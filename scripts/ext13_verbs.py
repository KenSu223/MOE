"""ext13 P5-3 (CPU): verb-swap controls on WinoGrande name items = curated seed templates x IOI name pairs
(user decision 2026-10-11, option 2; module moetrace/ext13_templates.py).

  python scripts/ext13_verbs.py seeds   -> data/wino_ref/verb_seeds.parquet + verb_seeds.tsv (eligible seeds for curation:
                                           filled with test names, the direction's pair builder passes under Qwen3 AND
                                           Mixtral BOS)
  python scripts/ext13_verbs.py expand [--k 48]
                                        -> data/wino_ref/verbswap_<dir>_<proto>.parquet (curated seeds of
                                           data/wino_ref/verb_templates.json x k name pairs x {orig, neutral, unrelated}),
                                           data/wino_ref/verbswap_review.tsv (one filled example per seed x variant),
                                           data/wino_ref/verbswap_funnel.json
Rows keep the builder's columns plus seed, variant, name_x, name_y, gender, dir. Same name pairs in every variant.
"""
import argparse
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.ext13_templates import build, fill, name_pairs, seed_pool

D = "/home/ubuntu/MOE/data/wino_ref"
PROTOS = {"qwen3": ("qwen3", True), "mixtral_bos": ("mixtral", True)}
TEST = {"m": ("James", "Robert"), "f": ("Sarah", "Laura"), "n": ("James", "Robert"), "x": ("James", "Sarah")}


def tokenizers():
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    from moetrace.models import MODELS
    return {p: (AutoTokenizer.from_pretrained(snapshot_dir(MODELS[k]["repo"])), st) for p, (k, st) in PROTOS.items()}


def seeds():
    toks = tokenizers()
    pool = seed_pool()
    rows, why = [], {}
    for s in pool:
        if s["gender"] == "x":
            why["mixed_gender_pronouns"] = why.get("mixed_gender_pronouns", 0) + 1
            continue
        t = fill(s, *TEST[s["gender"]])
        ok = True
        for p, (tok, st) in toks.items():
            pr, r = build(t, s["dir"], tok, st)
            if pr is None:
                why[f"{p}:{r}"] = why.get(f"{p}:{r}", 0) + 1
                ok = False
                break
        if ok:
            rows.append(s)
    df = pd.DataFrame(rows)
    df.to_parquet(f"{D}/verb_seeds.parquet", index=False)
    df.sort_values(["dir", "verb", "seed"])[["seed", "dir", "verb", "gender", "ans_a", "ans_b", "tmpl_a", "tmpl_b"]].to_csv(
        f"{D}/verb_seeds.tsv", sep="\t", index=False)
    print(f"pool {len(pool)}  eligible {len(df)}  by dir {df.dir.value_counts().to_dict()}  dropped {why}")


def expand(k: int):
    toks = tokenizers()
    vt = json.load(open(f"{D}/verb_templates.json"))
    sd = pd.read_parquet(f"{D}/verb_seeds.parquet").set_index("seed")
    out = {(d, p): [] for d in ("ent2attr", "attr2ent") for p in PROTOS}
    review, funnel = [], {}
    for sid, spec in vt["seeds"].items():
        s = dict(sd.loc[sid]) | {"seed": sid}
        pairs = name_pairs(s, k, random.Random(f"ext13-verbs-{sid}"))
        for variant in ("orig", "neutral", "unrelated"):
            edits = spec.get(variant, []) if variant != "orig" else []
            for j, (x, y) in enumerate(pairs):
                t = fill(s, x, y, edits, variant)
                if t is None:
                    funnel["edit_not_found"] = funnel.get("edit_not_found", 0) + 1
                    break
                if j == 0:
                    review.append(dict(seed=sid, dir=s["dir"], variant=variant, sent_a=t.sent_a, sent_b=t.sent_b,
                                       ans_a=t.ans_a, ans_b=t.ans_b))
                for p, (tok, st) in toks.items():
                    pr, r = build(t, s["dir"], tok, st)
                    key = f"{p}:{s['dir']}:{variant}:{r}"
                    funnel[key] = funnel.get(key, 0) + 1
                    if pr is not None:
                        row = pr.to_row()
                        if "probe_json" in row:
                            row.pop("probe_json")
                        out[(s["dir"], p)].append(row | dict(seed=sid, variant=variant, name_x=x, name_y=y,
                                                              gender=s["gender"], dir=s["dir"]))
    for (d, p), rows in out.items():
        df = pd.DataFrame(rows)
        df.to_parquet(f"{D}/verbswap_{d}_{p}.parquet", index=False)
        if len(df):
            print(p, d, len(df), df.groupby("variant").size().to_dict(), "seeds", df.seed.nunique())
    pd.DataFrame(review).to_csv(f"{D}/verbswap_review.tsv", sep="\t", index=False)
    json.dump(dict(sorted(funnel.items())), open(f"{D}/verbswap_funnel.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seeds", "expand"])
    ap.add_argument("--k", type=int, default=48)
    a = ap.parse_args()
    seeds() if a.cmd == "seeds" else expand(a.k)
