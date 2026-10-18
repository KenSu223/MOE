"""ext14 Phase 6 ladder v2 builds (CPU, agent p6-data): l_ant, l_prop, l_bind, l_ind (moetrace/ext14_data.py ladder section).

LRE files (data/dial/lre/, downloaded from github.com/evandez/relations data/) are read as data only; run this script with
`python -I` from the repo root (it never imports from the data folder).
Synthetic tasks (l_bind, l_ind) draw their words from lists filtered to words that are single tokens in BOTH tokenizer families, so
the same texts are built for Qwen and Mistral. l_ant / l_prop keep each family's own valid pairs; both_fams marks the shared ones
(preferred when capping).
Split: l_prop by subject halves per relation (builder); others by item, random.Random(14) over the sorted item ids, half / half.
Outputs data/dial/build/<cell>_<fam>.parquet, data/dial/build/funnel_ladder_<fam>.json, data/dial/qa/<cell>_samples.md (qwen texts).
Usage: python -I scripts/ext14_build_ladder.py [--cells l_ant,l_prop,l_bind,l_ind]
"""
import argparse
import collections
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
from moetrace import ext14_data as X

QA = f"{X.DIAL}/qa"


def item_split(ids, seed=14):
    ks = sorted(set(ids))
    random.Random(seed).shuffle(ks)
    h = len(ks) // 2
    return {k: ("discovery" if i < h else "validation") for i, k in enumerate(ks)}, {k: i for i, k in enumerate(ks)}


def samples_md(cell, df, n=50):
    ids = list(df.drop_duplicates("src_pair_id").sample(n=min(n, df.src_pair_id.nunique()), random_state=0).src_pair_id)
    lines = [f"# {cell}: {len(ids)} seeded samples (qwen-family texts; random_state 0)", ""]
    for k, sid in enumerate(ids, 1):
        g = df[df.src_pair_id == sid]
        lines.append(f"## {k}. {sid} ({g.source.iloc[0]}{', ' + g.relation.iloc[0] if 'relation' in g and isinstance(g.relation.iloc[0], str) else ''})")
        for r in g.itertuples():
            tag = f"template {r.template}" if hasattr(r, "template") and r.template == r.template else f"order {r.order}"
            lines.append(f"- {tag}: `{r.prompt_a!r}` -> **{r.answer_a}** | B: `{r.prompt_b!r}` -> **{r.answer_b}**")
        lines.append("")
    open(f"{QA}/{cell}_samples.md", "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="l_ant,l_prop,l_bind,l_ind")
    ap.add_argument("--ind-frame", default="I0", help="l_ind format (IND PILOT decision): I0 | I1 | I2")
    a = ap.parse_args()
    os.makedirs(X.BUILD, exist_ok=True)
    os.makedirs(QA, exist_ok=True)
    toks = {fam: X.family_tokenizer(fam) for fam in ("qwen", "mistral")}
    objs = X.single_token_words(X.BIND_OBJECTS, toks.values())
    cols = X.single_token_words(X.BIND_COLOURS, toks.values())
    words = X.single_token_words(X.IND_WORDS, toks.values())
    print("shared single-token words: objects", len(objs), "colours", len(cols), "induction words", len(words), flush=True)
    built = {}
    for fam, (tok, st) in toks.items():
        fp = f"{X.BUILD}/funnel_ladder_{fam}.json"
        funnel = json.load(open(fp)) if os.path.exists(fp) else {}
        funnel["word_lists"] = {"bind_objects": objs, "bind_colours": cols, "ind_words": words}
        for cell in a.cells.split(","):
            if cell == "l_ant":
                rows, why = X.build_l_ant(tok, st)
            elif cell == "l_prop":
                rows, why = X.build_l_prop(tok, st)
            elif cell == "l_bind":
                rows, why = X.build_l_bind(tok, st, objects=objs, colours=cols)
            elif cell == "l_ind":
                keys = X.line_start_words(words, toks.values())
                rows, why = X.build_l_ind(tok, st, words=words, frame=a.ind_frame, keys=keys)
                for r in rows:
                    r["frame"] = a.ind_frame
            df = pd.DataFrame(rows)
            if cell != "l_prop":
                sp, rank = item_split(df.src_pair_id)
                df["split"] = df.src_pair_id.map(sp)
                df["rank"] = df.src_pair_id.map(rank)
            else:
                df["rank"] = df.pair_idx
            df["level"], df["direction"] = cell, ""
            built[(fam, cell)] = df
            funnel[cell] = {"reasons": dict(sorted(why.items())), "pairs": int(len(df)), "items": int(df.src_pair_id.nunique()),
                            "by_source": df.drop_duplicates("src_pair_id").source.value_counts().to_dict()}
            print(fam, cell, json.dumps({k: v for k, v in funnel[cell].items() if k != "reasons"}), flush=True)
        json.dump(funnel, open(f"{X.BUILD}/funnel_ladder_{fam}.json", "w"), indent=1)
    for cell in a.cells.split(","):
        shared = set(built[("qwen", cell)].pair_id) & set(built[("mistral", cell)].pair_id)
        for fam in ("qwen", "mistral"):
            df = built[(fam, cell)]
            df["both_fams"] = df.pair_id.isin(shared)
            df.to_parquet(f"{X.BUILD}/{cell}_{fam}.parquet", index=False)
        print(cell, "pairs shared by both families", len(shared))
        samples_md(cell, built[("qwen", cell)])


if __name__ == "__main__":
    main()
