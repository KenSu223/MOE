"""Protocol helpers shared by the pass scripts: case loading by id, set membership, job builders, rescue tables."""
from __future__ import annotations

import json
import os
import random
from typing import Optional

import numpy as np
import pandas as pd

from .data import Case, load_records, prepare_case, shuffled_order
from .models import MODELS, RESULTS


def out_dir(model: str) -> str:
    d = os.path.join(RESULTS, model)
    os.makedirs(d, exist_ok=True)
    return d


def load_case_sets(model: str) -> dict:
    with open(os.path.join(out_dir(model), "case_sets.json")) as f:
        return json.load(f)


def set_names(sets: dict) -> list[str]:
    return [k for k in ("paper", "strict", "relaxed") if k in sets]


def cases_by_id(model: str, case_ids: list[int], tok=None, token_rule: str = "space", special_tokens: bool = True) -> tuple[dict[int, Case], list[dict]]:
    """Tokenise the given case ids with the model tokenizer. Returns (id -> Case, list of rejects)."""
    if tok is None:
        from transformers import AutoTokenizer
        from .arch import snapshot_dir
        tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[model]["repo"]))
    recs = load_records()
    out, rej = {}, []
    for cid in case_ids:
        c, why = prepare_case(recs[cid], tok, token_rule, special_tokens)
        if c is None:
            rej.append({"case_id": cid, "reason": why})
        else:
            out[cid] = c
    return out, rej


def membership_table(sets: dict, case_ids: list[int]) -> pd.DataFrame:
    """One row per case with in_<set> and split_<set> columns."""
    rows = []
    rank_of = {}
    recs_n = 21919
    order = shuffled_order(recs_n, 0)
    for r, ri in enumerate(order):
        rank_of[ri] = r  # case_id == record index
    for cid in case_ids:
        row = {"case_id": cid, "scan_rank": rank_of.get(cid, -1)}
        for s in set_names(sets):
            d, v = set(sets[s]["discovery"]), set(sets[s]["validation"])
            row[f"in_{s}"] = cid in d or cid in v
            row[f"split_{s}"] = "discovery" if cid in d else ("validation" if cid in v else "")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# post-processing helpers
# ---------------------------------------------------------------------------------------------------------------
def layer_rescue_matrix(sweep_rows: pd.DataFrame) -> pd.DataFrame:
    """Pivot: index case_id, columns layer -> rescue."""
    lr = sweep_rows[sweep_rows.kind == "layer"]
    return lr.pivot(index="case_id", columns="layer", values="rescue")


def select_layer(R: pd.DataFrame, discovery_ids: list[int]) -> tuple[int, pd.Series]:
    m = R.loc[[c for c in discovery_ids if c in R.index]].mean(0)
    return int(m.idxmax()), m


def active_controls(clean_experts: list[int], e_star: int, n: int, case_id: int, seed_base: int = 1000) -> list[int]:
    """Active-random controls: other clean-active experts of the same case (Qwen3: 3 sampled; Mixtral: the other)."""
    others = [e for e in clean_experts if e != e_star]
    if len(others) <= n:
        return others
    rng = random.Random(seed_base + int(case_id))
    return rng.sample(others, n)
