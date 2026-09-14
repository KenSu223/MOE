"""Direction 4 (ext4-codefact): CodeFact items -> Case objects compatible with moetrace.data.Case.

An *item* (one JSON line of data/codefact/items.jsonl) is tokenizer-independent:
    item_id, category (S1,S2,S3,R1,R2,R3), subcategory, source, licence, unit_id,
    prefix_text  the code up to (not including) the answer,
    true_str     the true continuation (e.g. ')', 'else', ' in', 'total', 'append', 'key'),
    foil_str     the foil continuation, appended to the SAME prefix (e.g. ']', 'elif', ',', 'count', 'extend', 'name'),
    subject_start, subject_end   character span (in prefix_text) of the subject = the tokens that determine the answer,
    subject_text, plus bookkeeping fields (n_prior_occurrences, foil_kind, ...).

`resolve_item` turns an item into a CodeCase for a given tokenizer: true and foil must each be exactly ONE token as a
continuation of the prefix (tok(prefix + s) == tok(prefix) + [id]); if the split fails at the natural boundary the
boundary is moved back over up to MAX_BACKOFF non-alphanumeric characters (e.g. Qwen merges '.append' and ' else' into
one token, so prefix 'result.' + 'append' becomes 'result' + '.append'). Subject positions are the prefix tokens
overlapping the subject span (as data.prepare_case does); the subject must not be the final prefix token. Items where
true == foil or where the true token occurs in the last 3 prefix tokens (trivial copying) are rejected.

Protocols: special_tokens=True/False (Mixtral paper protocol = False), and an optional chat-template wrapper
(prefix_ids prepended after resolution, subject positions shifted; see chat_prefix_ids)."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Optional

from .data import Case

ITEMS_PATH = "/home/ubuntu/MOE/data/codefact/items.jsonl"
CATEGORIES = ("S1", "S2", "S3", "R1", "R2", "R3")
CATEGORY_LABEL = {"S1": "closing bracket", "S2": "block keyword", "S3": "keyword completion",
                  "R1": "variable recall", "R2": "attribute / API recall", "R3": "constant recall"}
MAX_BACKOFF = 4
COPY_WINDOW = 3


@dataclass
class CodeCase(Case):
    category: str = ""
    subcategory: str = ""
    source: str = ""
    item_id: str = ""
    backoff: int = 0
    true_tok: str = ""
    foil_tok: str = ""
    subject_tok: str = ""

    def to_row(self) -> dict:
        d = super().to_row()
        return d


def load_items(path: str = ITEMS_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _cont(tok, prefix_ids: list[int], prefix: str, s: str, special_tokens: bool) -> Optional[int]:
    full = tok(prefix + s, add_special_tokens=special_tokens)["input_ids"]
    if len(full) != len(prefix_ids) + 1 or list(full[: len(prefix_ids)]) != list(prefix_ids):
        return None
    return int(full[-1])


def resolve_item(item: dict, tok, special_tokens: bool = True, case_id: Optional[int] = None,
                 prefix_ids: Optional[list[int]] = None, max_backoff: int = MAX_BACKOFF,
                 max_tokens: Optional[int] = None) -> tuple[Optional[CodeCase], str]:
    """Tokenise one item. Returns (CodeCase or None, reason)."""
    prefix0, true0, foil0 = item["prefix_text"], item["true_str"], item["foil_str"]
    s0, s1 = int(item["subject_start"]), int(item["subject_end"])
    if true0 == foil0:
        return None, "same_string"
    last = None
    for k in range(0, max_backoff + 1):
        if k > len(prefix0):
            break
        tail = prefix0[len(prefix0) - k:] if k else ""
        if k and any(ch.isalnum() or ch == "_" for ch in tail):
            break
        prefix, true, foil = prefix0[: len(prefix0) - k], tail + true0, tail + foil0
        if len(prefix) - k < s1 and k:
            break  # backoff would cut into the subject span
        enc = tok(prefix, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=special_tokens)
        ids = [int(x) for x in enc["input_ids"]]
        t = _cont(tok, ids, prefix, true, special_tokens)
        f = _cont(tok, ids, prefix, foil, special_tokens)
        if t is None or f is None:
            last = "multi_token"
            continue
        if t == f:
            return None, "same_token"
        subject_pos = []
        for i, ((a, b), sp) in enumerate(zip(enc["offset_mapping"], enc["special_tokens_mask"])):
            if sp or b <= a:
                continue
            if a < s1 and b > s0:
                subject_pos.append(i)
        if not subject_pos:
            return None, "subject_span"
        if subject_pos[-1] == len(ids) - 1:
            return None, "subject_is_final"
        if t in ids[-COPY_WINDOW:]:
            return None, "copy"
        if max_tokens and len(ids) > max_tokens:
            return None, "too_long"
        if len(ids) < 4:
            return None, "too_short"
        toks = tok.convert_ids_to_tokens
        cid = int(case_id if case_id is not None else item.get("case_id", 0))
        if prefix_ids:
            kk = len(prefix_ids)
            ids = [int(x) for x in prefix_ids] + ids
            subject_pos = [p + kk for p in subject_pos]
        return CodeCase(case_id=cid, relation=item["category"], prompt=prefix, subject=item.get("subject_text", ""),
                        true_str=true, foil_str=foil, ids=ids, subject_pos=subject_pos, true_id=t, foil_id=f,
                        category=item["category"], subcategory=item.get("subcategory", ""), source=item.get("source", ""),
                        item_id=item.get("item_id", ""), backoff=k, true_tok=str(toks(t)), foil_tok=str(toks(f)),
                        subject_tok=" ".join(str(x) for x in toks([ids[p] for p in subject_pos]))), "ok"
    return None, last or "multi_token"


def resolve_items(items: list[dict], tok, special_tokens: bool = True, prefix_ids: Optional[list[int]] = None,
                  max_tokens: Optional[int] = None) -> tuple[dict[int, CodeCase], list[dict]]:
    """case_id = the item's `case_id` field (its index in items.jsonl). Returns (id -> CodeCase, rejects)."""
    out, rej = {}, []
    for it in items:
        c, why = resolve_item(it, tok, special_tokens, case_id=it["case_id"], prefix_ids=prefix_ids, max_tokens=max_tokens)
        if c is None:
            rej.append({"case_id": it["case_id"], "item_id": it.get("item_id", ""), "category": it["category"], "reason": why})
        else:
            out[it["case_id"]] = c
    return out, rej


# ---------------------------------------------------------------------------------------------------------------
# protocols
# ---------------------------------------------------------------------------------------------------------------
CHAT_USER_MSG = "Complete the following Python code."


def chat_prefix_ids(tok, user_msg: str = CHAT_USER_MSG, fence: bool = True) -> list[int]:
    """Token ids of a minimal chat template whose assistant turn is open: the code prefix is the assistant's
    continuation (inside a ```python fence when fence=True). Uses the tokenizer's own chat template."""
    msgs = [{"role": "user", "content": user_msg}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if fence:
        txt = txt + "```python\n"
    return [int(x) for x in tok(txt, add_special_tokens=False)["input_ids"]]


def load_cases(model_key: str, case_ids: list[int], protocol: str = "raw", tok=None, items: Optional[list[dict]] = None,
               items_path: str = ITEMS_PATH) -> tuple[dict[int, CodeCase], list[dict], dict]:
    """protocol: 'raw' (tokenizer defaults), 'nobos' (add_special_tokens=False), 'chat' (chat template + fence)."""
    from .models import MODELS
    if tok is None:
        from transformers import AutoTokenizer
        from .arch import snapshot_dir
        tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[model_key]["repo"]))
    items = items or load_items(items_path)
    by_id = {it["case_id"]: it for it in items}
    sel = [by_id[c] for c in case_ids if c in by_id]
    st = protocol != "nobos"
    pre = chat_prefix_ids(tok) if protocol == "chat" else None
    cases, rej = resolve_items(sel, tok, special_tokens=st, prefix_ids=pre)
    meta = {"protocol": protocol, "special_tokens": st, "chat_prefix_ids": pre, "chat_prefix_len": len(pre) if pre else 0,
            "chat_prefix_text": tok.decode(pre) if pre else None}
    return cases, rej, meta


def split_ids(ids: list[int], seed: int = 0) -> tuple[list[int], list[int]]:
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    return ids[: len(ids) // 2], ids[len(ids) // 2:]
