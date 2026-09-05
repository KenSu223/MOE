"""CounterFact loading, seed-0 record order, prompt/subject-span/object tokenisation, filters and splits."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from typing import Optional

DATA_PATH = "/home/ubuntu/MOE/data/counterfact.json"
PAPER_IDS_PATH = "/home/ubuntu/MOE/data/paper_case_ids.json"

STRICT = (1.0, 0.5)  # (clean margin, subject-noise drop)
RELAXED = (0.5, 0.25)


def load_records(path: str = DATA_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def shuffled_order(n: int, seed: int = 0) -> list[int]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


def load_paper_ids(model_key: str, path: str = PAPER_IDS_PATH) -> dict[str, list[int]]:
    with open(path) as f:
        return json.load(f)[model_key]


@dataclass
class Case:
    case_id: int
    relation: str
    prompt: str
    subject: str
    true_str: str
    foil_str: str
    ids: list[int]
    subject_pos: list[int]
    true_id: int
    foil_id: int

    def to_row(self) -> dict:
        d = asdict(self)
        d["n_tokens"] = len(self.ids)
        d["n_subject_tokens"] = len(self.subject_pos)
        d["ids"] = json.dumps(self.ids)
        d["subject_pos"] = json.dumps(self.subject_pos)
        return d


def _single_token_continuation(tok, prompt_ids: list[int], prompt: str, obj: str, special_tokens: bool = True) -> Optional[int]:
    full = tok(prompt + " " + obj, add_special_tokens=special_tokens)["input_ids"]
    if len(full) != len(prompt_ids) + 1 or full[: len(prompt_ids)] != prompt_ids:
        return None
    return int(full[-1])


def _object_token(tok, prompt_ids: list[int], prompt: str, obj: str, token_rule: str, special_tokens: bool = True) -> Optional[int]:
    """token_rule 'space': continuation token with leading space (default). 'paper_like': if tok(obj) without a
    leading space is a single token use it, otherwise fall back to the 'space' rule (best match to the paper's funnel)."""
    if token_rule == "paper_like":
        ids = tok(obj, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return int(ids[0])
    elif token_rule != "space":
        raise ValueError(token_rule)
    return _single_token_continuation(tok, prompt_ids, prompt, obj, special_tokens)


def prepare_case(rec: dict, tok, token_rule: str = "space", special_tokens: bool = True) -> tuple[Optional[Case], str]:
    """Tokenise one CounterFact record. Returns (Case or None, reason)."""
    rw = rec["requested_rewrite"]
    tpl, subj = rw["prompt"], rw["subject"]
    if tpl.count("{}") != 1:
        return None, "template"
    start = tpl.index("{}")
    end = start + len(subj)
    prompt = tpl.format(subj)
    enc = tok(prompt, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=special_tokens)
    ids = list(enc["input_ids"])
    subject_pos = []
    for i, ((a, b), sp) in enumerate(zip(enc["offset_mapping"], enc["special_tokens_mask"])):
        if sp or b <= a:
            continue
        if a < end and b > start:
            subject_pos.append(i)
    if not subject_pos:
        return None, "subject_span"
    if subject_pos[-1] == len(ids) - 1:
        return None, "subject_is_final"  # the final position must be a non-subject token
    true_id = _object_token(tok, ids, prompt, rw["target_true"]["str"], token_rule, special_tokens)
    foil_id = _object_token(tok, ids, prompt, rw["target_new"]["str"], token_rule, special_tokens)
    if true_id is None or foil_id is None:
        return None, "multi_token"
    if true_id == foil_id:
        return None, "same_token"
    return Case(case_id=int(rec["case_id"]), relation=rw["relation_id"], prompt=prompt, subject=subj,
                true_str=rw["target_true"]["str"], foil_str=rw["target_new"]["str"], ids=ids,
                subject_pos=subject_pos, true_id=true_id, foil_id=foil_id), "ok"


def passes(delta_clean: float, delta_noised: float, thresholds=STRICT) -> bool:
    margin, drop = thresholds
    return (delta_clean >= margin) and ((delta_clean - delta_noised) >= drop)


def split_cases(case_ids: list[int], n_discovery: int, seed: int = 0) -> tuple[list[int], list[int]]:
    ids = list(case_ids)
    random.Random(seed).shuffle(ids)
    return ids[:n_discovery], ids[n_discovery:]
