"""ext6: symmetric token replacement (STR; Zhang & Nanda 2024, arXiv:2309.16042) for the paper's CounterFact protocol.

Corruption. For a case (template t, subject s, true object o, foil o_f = target_new) a *donor* is another CounterFact
subject s' of the same relation whose TRUE object is o_f. The corrupted prompt is t.format(s'): same template, other
subject, and its correct answer is the case's foil. The paper's metric Delta = logit(o) - logit(o_f) is kept unchanged
and becomes Zhang & Nanda's logit difference LD(r, r') with r' = the corrupted prompt's answer. Donors must be
*symmetric*: equal token length, the subject at the same token positions, identical template tokens, and the same
single-token continuation ids for o and o_f.

Selection (user decisions 2026-09-28). Candidates are evaluated in one forward pass; a donor qualifies when the model
knows its fact, Delta_corrupt = logit(o) - logit(o_f) <= -margin (margin 1.0, mirror of the clean-margin filter).
Per case up to K = 5 qualifying donors are taken in a fixed random order (random.Random(2000 + case_id)), i.e. a
uniformly random K-subset of the qualifying donors. Case set = the paper's discovery/validation IDs restricted to cases
with >= 1 selected donor; recurrence threshold = half the retained discovery cases.

Patching is unchanged: the corrupted run is a plain prefill row (no noise) and every spawn uses parent = donor row,
clean = the case's clean row. Per-case quantities are means over the case's donors (primary) or the first donor
(single-donor sensitivity). This module builds candidates, selects donors, and assembles donor-aggregated tables in
the schema that moetrace.analysis expects (kind 'noised' there = the corrupted run).
"""
from __future__ import annotations

import collections
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .data import Case, _object_token, load_records
from .models import RESULTS

MARGIN = 1.0
K_DONORS = 5
SEED_BASE = 2000


@dataclass
class Donor:
    case_id: int  # the case being corrupted
    donor_record: int  # CounterFact record whose subject is used
    subject: str
    prompt: str
    ids: list[int]
    subject_pos: list[int]
    order: int  # position in the case's fixed random candidate order


def donor_index(recs: list[dict]) -> dict[tuple[str, str], list[tuple[str, int]]]:
    """(relation, true object string) -> sorted unique (subject, first record id)."""
    idx: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(dict)
    for r in recs:
        rw = r["requested_rewrite"]
        idx[(rw["relation_id"], rw["target_true"]["str"])].setdefault(rw["subject"], int(r["case_id"]))
    return {k: sorted(v.items()) for k, v in idx.items()}


def tokenize_span(tok, tpl: str, subj: str, special_tokens: bool) -> tuple[list[int], list[int], str]:
    """Same tokenisation and subject-span rule as data.prepare_case."""
    start = tpl.index("{}")
    end = start + len(subj)
    prompt = tpl.format(subj)
    enc = tok(prompt, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=special_tokens)
    ids = list(enc["input_ids"])
    sp = [i for i, ((a, b), m) in enumerate(zip(enc["offset_mapping"], enc["special_tokens_mask"]))
          if not m and b > a and a < end and b > start]
    return ids, sp, prompt


def candidates(case: Case, rec: dict, tok, index: dict, special_tokens: bool, token_rule: str = "space") -> list[Donor]:
    """All symmetric donors of a case, in the case's fixed random order."""
    rw = rec["requested_rewrite"]
    tpl = rw["prompt"]
    pool = [(s, rid) for s, rid in index.get((rw["relation_id"], rw["target_new"]["str"]), []) if s != case.subject]
    random.Random(SEED_BASE + case.case_id).shuffle(pool)
    rest_c = [t for i, t in enumerate(case.ids) if i not in set(case.subject_pos)]
    out = []
    for s, rid in pool:
        ids, sp, prompt = tokenize_span(tok, tpl, s, special_tokens)
        if len(ids) != len(case.ids) or sp != list(case.subject_pos):
            continue
        if [t for i, t in enumerate(ids) if i not in set(sp)] != rest_c:
            continue
        if sp and sp[-1] == len(ids) - 1:
            continue
        # the same continuation tokens for the true object and the foil after the donor prompt
        if _object_token(tok, ids, prompt, case.true_str, token_rule, special_tokens) != case.true_id:
            continue
        if _object_token(tok, ids, prompt, case.foil_str, token_rule, special_tokens) != case.foil_id:
            continue
        out.append(Donor(case.case_id, rid, s, prompt, ids, sp, len(out)))
    return out


def select(cand: pd.DataFrame, k: int = K_DONORS, margin: float = MARGIN) -> pd.DataFrame:
    """Mark qualifying donors (delta_donor <= -margin) and the first k qualifying in each case's fixed order."""
    cand = cand.sort_values(["case_id", "order"]).copy()
    cand["qualifies"] = cand.delta_donor <= -margin
    cand["slot"] = -1
    for cid, g in cand[cand.qualifies].groupby("case_id"):
        take = g.index[:k]
        cand.loc[take, "slot"] = np.arange(len(take))
    cand["selected"] = cand.slot >= 0
    return cand


# ---------------------------------------------------------------------------------------------------------------
# run-directory helpers
# ---------------------------------------------------------------------------------------------------------------
def run_dir(run: str) -> str:
    d = os.path.join(RESULTS, run)
    os.makedirs(d, exist_ok=True)
    return d


def load_donors(run: str) -> pd.DataFrame:
    """Selected donors (one row per case x slot) with their token ids."""
    c = pd.read_parquet(os.path.join(run_dir(run), "str_candidates.parquet"))
    c = c[c.selected].sort_values(["case_id", "slot"]).reset_index(drop=True)
    c["ids_list"] = c.ids.map(json.loads)
    return c


def write_meta(run: str, key: str, upd: dict):
    p = os.path.join(run_dir(run), "run_meta.json")
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.setdefault(key, {}).update(upd)
    with open(p, "w") as f:
        json.dump(cur, f, indent=1, default=str)


def aggregate(rows: pd.DataFrame, keys: list[str], donors: str = "mean") -> pd.DataFrame:
    """Per-case values from donor-level rows: mean over the case's donors, or the first donor only (slot 0)."""
    if donors == "first":
        return rows[rows.slot == 0].drop(columns=["slot"]).reset_index(drop=True)
    num = [c for c in rows.columns if c not in keys + ["slot"] and pd.api.types.is_numeric_dtype(rows[c]) and rows[c].dtype != bool]
    other = [c for c in rows.columns if c not in keys + ["slot"] + num]
    g = rows.groupby(keys, sort=False)
    out = g[num].mean()
    if other:
        out = out.join(g[other].first())
    out["n_donors"] = g.size()
    return out.reset_index()


def model_data(run: str, donors: str = "mean"):
    """analysis.ModelData for an STR run: donor-aggregated sweep and expert rows in the base schema."""
    from .analysis import ModelData
    od = run_dir(run)
    sets = json.load(open(os.path.join(od, "case_sets.json")))
    cases = pd.read_parquet(os.path.join(od, "sweep_cases.parquet"))
    sw = pd.read_parquet(os.path.join(od, "str_sweep_rows.parquet"))
    lay = aggregate(sw[sw.kind == "layer"][["case_id", "slot", "layer", "rescue", "delta", "delta_corrupt"]], ["case_id", "layer"], donors)
    lay["kind"] = "layer"
    if donors == "first":  # per-case drop of the first donor
        first = sw[(sw.kind == "corrupt") & (sw.slot == 0)].set_index("case_id").delta
        cases = cases.copy()
        cases["delta_corrupt"] = cases.case_id.map(first)
        cases["drop"] = cases.delta_clean - cases.delta_corrupt
    er_path = os.path.join(od, "str_expert_rows.parquet")
    er = None
    if os.path.exists(er_path):
        e = pd.read_parquet(er_path)
        keys = ["case_id", "layer", "kind", "expert", "partner"]
        ca = e[e.kind != "expert_corrupt_only"]
        er = aggregate(ca[keys + ["slot", "rescue", "delta", "delta_corrupt", "vnorm", "clean_weight", "clean_active"]], keys, donors)
        er["clean_active"] = er.clean_active.astype(bool)
    return ModelData(run, sets, lay, None, cases, er, None)


def normalised(md, layer: int, ids: list[int]) -> dict:
    """Zhang & Nanda's normalised logit difference for the block patch at a layer: population mean rescue / mean drop
    (paired bootstrap), and the per-case ratio."""
    from .stats import ratio_ci, summarize
    R = md.R
    ct = md.cases.set_index("case_id")
    r = R.loc[ids, layer].values
    d = ct.loc[ids, "drop"].values
    return {"layer": int(layer), "mean_rescue": float(r.mean()), "mean_drop": float(d.mean()), "ratio": ratio_ci(r, d),
            "per_case": summarize(r / d, with_p=False)}
