"""ext7: WinoGrande twins as symmetric-token-replacement (STR) pairs for activation patching.

A WinoGrande twin is two crowd-written sentences with the same blank "_", the same two options and answers that flip
because one or two trigger words differ. In 86 % of the train_xl twins the trigger comes AFTER the blank, so a
next-token prompt truncated at the blank is identical for both twins. We therefore reverse the direction (the same
quantity lm-eval's partial scoring compares): fill the blank with each twin's own answer and predict the trigger.

    sentence A:  The coroner tried to put the body in the bag but the _ was too small.   answer bag
    sentence B:  The coroner tried to put the body in the bag but the _ was too large.   answer body
    prompt A:    The coroner tried to put the body in the bag but the bag was too        -> " small" (r)
    prompt B:    The coroner tried to put the body in the bag but the body was too       -> " large" (r')

Prompt B is prompt A with the filled option replaced by the other option: an STR corruption in the sense of Zhang &
Nanda (2024), in-distribution by construction (both are WinoGrande sentences filled with their correct answer), with the
corrupted prompt's own answer r'. Metric Delta = logit(r) - logit(r') = LD(r, r'). The option span plays the role of the
CounterFact subject, the two trigger tokens the role of true object / foil.

Deterministic pair filters (this module; per tokenizer). A pair is kept only if
  W1  twin integrity: exactly two items per HIT id, one blank each, the same option pair, answers differ;
  W2  sentence-final trigger (user rule 1): the two sentences are identical except for their last word (trailing
      punctuation ignored), so the trigger is one word at the very end of the item and everything that determines the
      answer precedes the prediction position;
  W3  single-token trigger under the model's tokenizer (user rule 2): " " + word is exactly one continuation token after
      BOTH prompts, with the same id after both (CounterFact rule `_single_token_continuation`; ext6 rule "the same
      continuation ids after the donor prompt"), and the two trigger ids differ;
  W4  token symmetry (ext6 donor rule): the two prompts have equal length, the option at the same token positions and
      identical tokens everywhere else;
  W5  the option is not the final prompt token (CounterFact rule "subject_is_final": the STR site and the prediction
      position must be distinct patch sites);
  W6  de-duplication / split hygiene: one pair per normalised context (options replaced by placeholders), so that the
      same sentence template cannot sit in both discovery and validation.
Model-dependent filters (scripts/ext7_wino_scan.py): Delta(prompt A) >= +1 and Delta(prompt B) <= -1 (the clean margin
of the paper filter and the ext6 donor margin, both directions: Zhang & Nanda use PairedFacts both ways).
Recorded, not filtered (stratification / sensitivity): option token count, trigger word present in the prompt (copy
shortcut), both options are person names (social vs physical), membership in train_debiased (AfLite survivor),
local-context association baseline, top-1 status, GN drop.
"""
from __future__ import annotations

import collections
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from .data import _single_token_continuation

WG_DIR = "/home/ubuntu/MOE/data/winogrande_1.1"
TRAIL = '.!?"\' '
NAME_RE = re.compile(r"^[A-Z][a-z]+$")


@dataclass
class Twin:
    pair_id: str  # HIT id shared by the two items
    q_a: str
    q_b: str
    sent_a: str
    sent_b: str
    opt1: str
    opt2: str
    ans_a: str  # option that fills sentence A
    ans_b: str


def load_items(split: str) -> list[dict]:
    return [json.loads(l) for l in open(os.path.join(WG_DIR, f"{split}.jsonl"))]


def load_twins(split: str = "train_xl") -> tuple[list[Twin], collections.Counter]:
    """W1. Items grouped by HIT id (qID without its -1/-2 suffix); A = the lexicographically first qID."""
    funnel = collections.Counter()
    groups = collections.defaultdict(list)
    for r in load_items(split):
        groups[r["qID"].rsplit("-", 1)[0]].append(r)
    funnel["hit_groups"] = len(groups)
    out = []
    for hid in sorted(groups):
        v = sorted(groups[hid], key=lambda r: r["qID"])
        if len(v) != 2:
            funnel["W1_not_two_items"] += 1
            continue
        a, b = v
        if (a["option1"], a["option2"]) != (b["option1"], b["option2"]) and {a["option1"], a["option2"]} != {b["option1"], b["option2"]}:
            funnel["W1_option_mismatch"] += 1
            continue
        ans_a, ans_b = a["option" + a["answer"]], b["option" + b["answer"]]
        if ans_a == ans_b:
            funnel["W1_same_answer"] += 1
            continue
        if a["sentence"].count("_") != 1 or b["sentence"].count("_") != 1:
            funnel["W1_blank_count"] += 1
            continue
        out.append(Twin(hid, a["qID"], b["qID"], a["sentence"], b["sentence"], a["option1"], a["option2"], ans_a, ans_b))
    funnel["W1_twins"] = len(out)
    return out, funnel


def split_final_word(sentence: str) -> tuple[str, str]:
    """(text before the last word without the separating space, last word without trailing punctuation)."""
    body = sentence.rstrip(TRAIL)
    if " " not in body:
        return "", body
    head, last = body.rsplit(" ", 1)
    return head.rstrip(" "), last


@dataclass
class Pair:
    pair_id: str
    q_a: str
    q_b: str
    context: str  # shared text before the trigger, with the blank
    ans_a: str
    ans_b: str
    word_a: str  # sentence-final trigger word of A (= r for prompt A)
    word_b: str
    prompt_a: str
    prompt_b: str
    ids_a: list
    ids_b: list
    opt_pos: list  # token positions of the filled option (identical in both prompts)
    trig_a: int  # token id of " " + word_a
    trig_b: int
    n_opt_tokens: int
    n_opt_diff_tokens: int
    trigger_in_context: bool
    names: bool
    norm_key: str

    def to_row(self) -> dict:
        d = dict(self.__dict__)
        d["ids_a"] = json.dumps(self.ids_a)
        d["ids_b"] = json.dumps(self.ids_b)
        d["opt_pos"] = json.dumps(self.opt_pos)
        d["n_tokens"] = len(self.ids_a)
        return d


def _span_positions(enc, start: int, end: int) -> list[int]:
    return [i for i, ((a, b), sp) in enumerate(zip(enc["offset_mapping"], enc["special_tokens_mask"]))
            if not sp and b > a and a < end and b > start]


def _words(s: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z']+", s)}


def build_pair(t: Twin, tok, special_tokens: bool) -> tuple[Optional[Pair], str]:
    """W2-W5 for one twin under one tokenizer."""
    head_a, word_a = split_final_word(t.sent_a)
    head_b, word_b = split_final_word(t.sent_b)
    if head_a != head_b or word_a == word_b:
        return None, "W2_trigger_not_single_final_word"
    if "_" not in head_a or "_" in word_a or "_" in word_b or not word_a or not word_b:
        return None, "W2_blank_position"
    i = head_a.index("_")
    prompt_a = head_a[:i] + t.ans_a + head_a[i + 1:]
    prompt_b = head_a[:i] + t.ans_b + head_a[i + 1:]
    enc_a = tok(prompt_a, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=special_tokens)
    enc_b = tok(prompt_b, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=special_tokens)
    ids_a, ids_b = list(enc_a["input_ids"]), list(enc_b["input_ids"])
    # W3 single-token sentence-final trigger, same id after both prompts
    ta_a = _single_token_continuation(tok, ids_a, prompt_a, word_a, special_tokens)
    tb_b = _single_token_continuation(tok, ids_b, prompt_b, word_b, special_tokens)
    if ta_a is None or tb_b is None:
        return None, "W3_trigger_multi_token"
    ta_b = _single_token_continuation(tok, ids_b, prompt_b, word_a, special_tokens)
    tb_a = _single_token_continuation(tok, ids_a, prompt_a, word_b, special_tokens)
    if ta_b != ta_a or tb_a != tb_b:
        return None, "W3_trigger_id_depends_on_prompt"
    if ta_a == tb_b:
        return None, "W3_same_trigger_token"
    # W4 token symmetry
    if len(ids_a) != len(ids_b):
        return None, "W4_length_mismatch"
    pos_a = _span_positions(enc_a, i, i + len(t.ans_a))
    pos_b = _span_positions(enc_b, i, i + len(t.ans_b))
    if not pos_a or pos_a != pos_b:
        return None, "W4_option_span_mismatch"
    ps = set(pos_a)
    if any(x != y for k, (x, y) in enumerate(zip(ids_a, ids_b)) if k not in ps):
        return None, "W4_context_tokens_differ"
    n_diff = sum(ids_a[k] != ids_b[k] for k in pos_a)
    if n_diff == 0:
        return None, "W4_identical_prompts"
    # W5 option not the final token
    if pos_a[-1] == len(ids_a) - 1:
        return None, "W5_option_is_final"
    ctx_words = _words(prompt_a) | _words(prompt_b)
    norm = head_a.replace(t.opt1, "<O1>").replace(t.opt2, "<O2>")
    norm = re.sub(r"\s+", " ", norm).strip().lower()
    return Pair(t.pair_id, t.q_a, t.q_b, head_a, t.ans_a, t.ans_b, word_a, word_b, prompt_a, prompt_b, ids_a, ids_b,
                pos_a, int(ta_a), int(tb_b), len(pos_a), int(n_diff),
                word_a.lower() in ctx_words or word_b.lower() in ctx_words,
                bool(NAME_RE.match(t.opt1) and NAME_RE.match(t.opt2)), norm), "ok"


def debiased_qids() -> set[str]:
    return {r["qID"] for r in load_items("train_debiased")}


def build_pairs(tok, special_tokens: bool, split: str = "train_xl") -> tuple[list[Pair], dict]:
    """All W1-W6 pairs of a split under one tokenizer, plus the funnel (counts after each rule)."""
    twins, funnel = load_twins(split)
    kept, reasons = [], collections.Counter()
    for t in twins:
        p, why = build_pair(t, tok, special_tokens)
        if p is None:
            reasons[why] += 1
        else:
            kept.append(p)
    # W6 one pair per normalised context (first by HIT id)
    seen, dedup = set(), []
    for p in kept:
        if p.norm_key in seen:
            reasons["W6_duplicate_context"] += 1
            continue
        seen.add(p.norm_key)
        dedup.append(p)
    order = ["W2_trigger_not_single_final_word", "W2_blank_position", "W3_trigger_multi_token",
             "W3_trigger_id_depends_on_prompt", "W3_same_trigger_token", "W4_length_mismatch", "W4_option_span_mismatch",
             "W4_context_tokens_differ", "W4_identical_prompts", "W5_option_is_final", "W6_duplicate_context"]
    f = dict(funnel)
    left = funnel["W1_twins"]
    steps = []
    for k in order:
        left -= reasons.get(k, 0)
        steps.append({"rule": k, "dropped": int(reasons.get(k, 0)), "remaining": int(left)})
    f["steps"] = steps
    f["kept"] = len(dedup)
    return dedup, f
