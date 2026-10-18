"""ext14 Phase 6 builds (CPU, agent p6-data): Dial E (e1 / e1p / e2 / e3 x a2e / e2a), Set C (sc_agree_a2e, sc_conflict_a2e,
statement orientation), R-site (rsite_e1_a2e) on the WinoGrande Dial-E item pool (moetrace/ext14_data.py docstring).

Per tokenizer family (qwen = Qwen3-30B-A3B tokenizer, mistral = Mixtral-8x7B tokenizer with BOS):
  data/dial/build/<cell>_<fam>.parquet   every candidate pair (pair contract: pair_id, ids_a, ids_b, str_pos, trig_a, trig_b,
                                         + level, direction, order, src_pair_id, twin_idx, pair_idx, n_tokens, strata, positions,
                                         completeness flags)
  data/dial/build/funnel_<fam>.json      item / pair funnel per cell (reasons)
  data/dial/twin_split.json              ONE discovery / validation assignment per twin (src_pair_id), used by every cell
  data/dial/qa/completeness.md           completeness audit table (Phase-5 F2-cut guard), all cells x families
  data/dial/qa/<cell>_samples.md         50 seeded samples per cell (qwen family texts), written before any scan
Usage: python scripts/ext14_build_dial.py [--fams qwen,mistral] [--cells ...]
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

CELLS = ["e1_a2e", "e1_e2a", "e3_a2e", "e3_e2a", "e3p3_a2e", "e1p_a2e", "e1p_e2a", "e2_a2e", "e2_e2a", "sc_agree_a2e", "sc_conflict_a2e",
         "rsite_e1_a2e"]
QA = f"{X.DIAL}/qa"


def twin_split(items):
    """One split per twin: core 776 and extension twins shuffled separately (random.Random(14) / (15)), first half discovery.
    rank = priority order used to cap the case sets (core before extension)."""
    p = f"{X.DIAL}/twin_split.json"
    if os.path.exists(p):
        return json.load(open(p))
    core = sorted(X.core_776())
    ext = sorted({it.src_pair_id for it in items if not it.core776})
    out = {}
    r = 0
    for ids, seed in ((core, 14), (ext, 15)):
        ids = list(ids)
        random.Random(seed).shuffle(ids)
        h = len(ids) // 2
        for i, k in enumerate(ids):
            out[k] = {"split": "discovery" if i < h else "validation", "rank": r, "core776": seed == 14}
            r += 1
    json.dump({"note": "ext14 Dial-E twin split (p6-data): core 776 = Phase-3 shared twins, random.Random(14) shuffle of the sorted HIT "
                       "ids, first half discovery; extension twins (train_xl Phase-3 geometry with a copular blank clause) the same with "
                       "random.Random(15); rank = priority for capping (core first).", "twins": out}, open(p, "w"), indent=0)
    return json.load(open(p))


A2E_E1_FAMILY = ("e1_a2e", "e1p_a2e", "e2_a2e", "rsite_e1_a2e")


def row_for(cell, it, order, built, tok, st, para=None, para_src=None, frame="P0"):
    level = cell.split("_")[0]
    direction = cell.split("_")[-1]
    extra = {}
    if cell == "e3p3_a2e":
        r = X.build_e3p3(it)
        if isinstance(r, str):
            return None, r
        ta, tb, aa, ab, n, parts = r
        p, why = X.make_pair(tok, st, ta, tb, aa, ab, n)
        if p is None:
            return None, why
        p["answer_a"], p["answer_b"] = aa, ab
        order = 1 if _first_mention(it, parts["event"]) == "a" else 2
        extra = {"dropped_connective": parts["dropped"], "event_clause": parts["event"]}
    elif level == "e3":
        p, why = X.build_e3(it, direction, tok, st)
        if p is None:
            return None, why
        order = 1 if _first_mention(it, p["prompt_a"]) == "a" else 2
    else:
        if level == "sc":
            r = X.build_setc(it, cell.split("_")[1], order)
        else:
            r = X.build_level(it, level, direction, order, para=para, frame=frame if cell in A2E_E1_FAMILY else "P0")
        if isinstance(r, str):
            return None, r
        ta, tb, aa, ab, n = r
        p, why = X.make_pair(tok, st, ta, tb, aa, ab, n)
        if p is None:
            return None, why
        p["answer_a"], p["answer_b"] = aa, ab
    if level == "e3" and cell != "e3p3_a2e":
        p["answer_a"], p["answer_b"] = (it.ent_a, it.ent_b) if direction == "a2e" else (it.trig_a, it.trig_b)
    pos = p.pop("positions")
    row = {"pair_id": f"{it.src_pair_id}|{cell}|o{order}", "cell": cell, "level": level, "direction": direction, "order": order,
           "src_pair_id": it.src_pair_id, "twin_idx": it.twin_idx, "pair_idx": 2 * it.twin_idx + order - 1, "source": it.source,
           "core776": it.core776, "names": it.names, "det": it.det, "cop": it.cop, "adv": it.adv,
           "ans_a": it.ent_a, "ans_b": it.ent_b, "word_a": it.trig_a, "word_b": it.trig_b, **p, **X.pos_columns(pos),
           "para_a": (para or {}).get(it.trig_a), "para_b": (para or {}).get(it.trig_b), "para_src": para_src,
           "frame": frame if cell in A2E_E1_FAMILY else ("P3" if cell in ("e3p3_a2e", "sc_agree_a2e", "sc_conflict_a2e") else ("F1" if direction == "a2e" else "opt")),
           **extra,
           "norm_key": it.norm_ents(p["prompt_a"]) if level in ("e3", "e3p3", "sc") else it.norm_text(p["prompt_a"]),
           "trig_key": "|".join(sorted((it.trig_a.lower(), it.trig_b.lower()))), "antonym_pair": X.antonym_pair(it.trig_a, it.trig_b)}
    if cell == "sc_conflict_a2e":
        row["orientation"] = "statement"
    chk = X.completeness(cell, it, row, para)
    row["completeness"] = json.dumps(chk)
    row["complete"] = all(chk.values()) if chk else True
    return row, "ok"


def _first_mention(it, prompt):
    import re
    lo = len(X.F1_HEAD) if prompt.startswith(X.F1_HEAD) else 0
    pa = [m.start() for m in re.finditer(r"\b%s\b" % re.escape(it.ent_a), prompt[lo:], re.I)]
    pb = [m.start() for m in re.finditer(r"\b%s\b" % re.escape(it.ent_b), prompt[lo:], re.I)]
    if not pa:
        return "b"
    if not pb:
        return "a"
    return "a" if pa[0] < pb[0] else "b"


def build_cell(cell, items, tok, st, frame="P0"):
    level = cell.split("_")[0]
    rows, why = [], collections.Counter()
    for it in items:
        para, psrc = (None, None)
        if level == "e1p":
            para, psrc = X.paraphrase_for(it)
            if para is None:
                why[f"item:{psrc}"] += 1
                continue
        orders = [None] if level in ("e3", "e3p3") else [1, 2]
        got = []
        for o in orders:
            r, w = row_for(cell, it, o, None, tok, st, para, psrc, frame)
            why[w] += 1
            if r is not None:
                got.append(r)
        if level not in ("e3", "e3p3") and len(got) != 2:
            why["item:not_both_orders_built"] += 1
            continue
        rows.extend(got)
    df = pd.DataFrame(rows)
    # R4 dedup by normalised prompt (entities -> placeholders), first by twin_idx; an item drops with all its orders
    if len(df):
        df = df.sort_values(["twin_idx", "order"])
        dup_items = set()
        seen = {}
        for r in df.itertuples():
            k = (r.norm_key, r.order)
            if k in seen and seen[k] != r.src_pair_id:
                dup_items.add(r.src_pair_id)
            else:
                seen.setdefault(k, r.src_pair_id)
        why["item:R4_duplicate_prompt"] += len(dup_items)
        df = df[~df.src_pair_id.isin(dup_items)].reset_index(drop=True)
        for r in df.itertuples():
            for k, v in json.loads(r.completeness).items():
                why[f"check_n:{k}"] += 1
                if not v:
                    why[f"check_fail:{k}"] += 1
        inc = set(df[~df.complete].src_pair_id)
        why["item:incomplete_dropped"] += len(inc)
        df = df[~df.src_pair_id.isin(inc)].reset_index(drop=True)
    return df, why


def samples_md(cell, df, n=50):
    s = df.drop_duplicates("src_pair_id").sample(n=min(n, df.src_pair_id.nunique()), random_state=0)
    ids = list(s.src_pair_id)
    lines = [f"# {cell}: {len(ids)} seeded samples (qwen-family texts; random_state 0)", "",
             "Each item: prompt A -> answer A | prompt B -> answer B (STR at token positions str_pos). Orders shown where built.", ""]
    for k, sid in enumerate(ids, 1):
        g = df[df.src_pair_id == sid].sort_values("order")
        lines.append(f"## {k}. {sid} ({'names' if g.names.iloc[0] else 'objects'}, {'core' if g.core776.iloc[0] else 'ext'})")
        for r in g.itertuples():
            lines.append(f"- order {r.order}: `{r.prompt_a!r}` -> **{r.answer_a}**")
            lines.append(f"  - B: `{r.prompt_b!r}` -> **{r.answer_b}**" + (f"  (paraphrases {r.para_a} / {r.para_b})" if r.para_a else ""))
        lines.append("")
    with open(f"{QA}/{cell}_samples.md", "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fams", default="qwen,mistral")
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--frame-qwen", default="P0", help="attribute -> entity frame of the E1-family cells (FRAME PILOT decision)")
    ap.add_argument("--frame-mistral", default="P0")
    a = ap.parse_args()
    os.makedirs(X.BUILD, exist_ok=True)
    os.makedirs(QA, exist_ok=True)
    items, item_funnel = X.dial_items()
    split = twin_split(items)
    comp_rows = []
    for fam in a.fams.split(","):
        tok, st = X.family_tokenizer(fam)
        fp = f"{X.BUILD}/funnel_{fam}.json"
        funnel = json.load(open(fp)) if os.path.exists(fp) else {}
        funnel["items"] = item_funnel
        funnel["items_kept"] = {"core776": sum(i.core776 for i in items), "ext": sum(not i.core776 for i in items)}
        for cell in a.cells.split(","):
            frame = getattr(a, f"frame_{fam}")
            df, why = build_cell(cell, items, tok, st, frame)
            df["split"] = df.src_pair_id.map(lambda k: split["twins"][k]["split"])
            df["rank"] = df.src_pair_id.map(lambda k: split["twins"][k]["rank"])
            tmp = f"{X.BUILD}/.{cell}_{fam}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, f"{X.BUILD}/{cell}_{fam}.parquet")  # atomic: scan jobs may read the build at any time
            funnel[cell] = {"frame": frame if cell in A2E_E1_FAMILY else None, "reasons": dict(sorted(why.items())), "pairs": int(len(df)), "items": int(df.src_pair_id.nunique()),
                            "items_core": int(df[df.core776].src_pair_id.nunique()), "items_names": int(df[df.names].src_pair_id.nunique())}
            print(fam, cell, json.dumps(funnel[cell]), flush=True)
            if fam == "qwen":
                samples_md(cell, df)
        json.dump(funnel, open(fp, "w"), indent=1)


if __name__ == "__main__":
    main()
