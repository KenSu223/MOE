"""ext13 Phase 5 (agent p5-data): probe v2, association strata, copy-prefix pairs (P5-8), the fill-in-the-blank format on
the verb templates, original-name memorisation pairs, token positions, and the condition registry
(RESEARCH_PLAN.md "Phase 5 execution plan").

Association probes (context-free partial-input baselines; Delta = logit(ans_a) - logit(ans_b) on the probe prompt, entity
ids resolved on the probe prompt, both candidate orders averaged per trigger):
    v1  "Word: dirty. Which fits better, the mirror or the spray? Answer: the"      (ext13_wino_ref.probe_prompt)
    v2  'Fill in the blank: Between the mirror and the spray, the _ goes with "dirty".\nAnswer: the'
        names: 'Fill in the blank: Between Brett and Kevin, _ goes with "generous".\nAnswer:'
Pre-registered rule (PROGRESS 2026-10-11T06:10Z): a probe RESPONDS in a model if >= 10 % of the trigger probes of the
margin pairs (valid attribute -> entity formats F1 / F2E / F2R pooled, unique twins) reach |probe| >= 1; R = the responding
probes (none -> R = {v1, v2}, flagged). Strata per pair (assoc_stratum): neutral (|probe| < 1 for both triggers under every
probe in R), reverse (a trigger points to the other candidate by >= 1 under some probe in R), associative (not reverse, both
triggers point to their own answers by >= 1), one-sided (the rest).

Copy prefix, CURRENT (coordinator decision 2026-10-11, options_prefix): "Options: {t1}, {t2}." + newline before the Phase-3
prompt, both attribute orders as separate pairs; checks: the answer token is in the prefix, each prefix attribute is ONE token
with the continuation id, token alignment by the Phase-3 builder. No grammar rules. The prefix below ("respectively" style) is
VOID for tracing (it pairs X with t1 by position; kept for the behaviour record).

Copy prefix, VOID (P5-8 as first specified; RESEARCH_PLAN.md "P5-8 Copy vs generate control"), Phase-3 geometry (twins differ only in the
sentence-final trigger; fill the blank, predict the trigger):
    "Of {X} and {Y}, one {frame} {t1} and the other {frame} {t2}. " + Phase-3 sentence
    objects: "Of the {X} and the {Y}, ..." with the determiner before the blank ("a" / "an" chosen per noun)
{frame} = the sentence's text between the blank and the trigger; X, Y = option1, option2; both attribute orders
(t1, t2) = (word_a, word_b) [order 1] and (word_b, word_a) [order 2] are separate pairs. Grammar rules (copy_prefix; fixed
after a hand review of natural-item prefixes, see scripts/ext13_p5_build.py review):
    C1 no possessive blank ("_'s");  C2 non-empty frame;  C3 subject blank: the first frame word (after adverbs) is a verb
    or auxiliary, and not a noun followed by a verb ("the _ emu is quite", "the _ art because it was");
    C3b the first verb is not directly followed by a finite copula ("the glue _ had was empty": the blank is the subject
       of a relative clause; also catches the typo "the _ was was too");
    C4 singular verb (no "were / are / have / do": "one were beautiful");
    C6 no personal pronoun in the frame other than a possessive that "one" can bind (his / their / its / my / our / your;
       "her" counts as a pronoun): an object pronoun in the prefix has no antecedent yet ("one made her feel happy", "one
       does them right", "one makes it possible").
    Long frames were reviewed and are grammatical ("one was born and raised speaking English and the other ..."), so no
    length rule (C5 dropped after the review; frame_words is kept as a column).
"""
from __future__ import annotations

import json
import math
import re
from typing import Optional

from .data import _single_token_continuation
from .ext7_wino import NAME_RE, Twin, build_pair, split_final_word
from .ext13_templates import IRREG
from .ext13_wino_ref import ADVS, DETS, F1_HEAD, VERBS, probe_prompt as probe_v1_prompt, word_diff

PROBE_VERSIONS = ("v1", "v2")
THR = 1.0
RESPOND_FRAC = 0.10
STRATA = ("associative", "one-sided", "reverse", "neutral")


# ---------------------------------------------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------------------------------------------
def probe_prompt(version: str, word: str, c1: str, c2: str, names: bool) -> str:
    if version == "v1":
        return probe_v1_prompt(word, c1, c2, names)
    if names:
        return f'Fill in the blank: Between {c1} and {c2}, _ goes with "{word}".\nAnswer:'
    return f'Fill in the blank: Between the {c1} and the {c2}, the _ goes with "{word}".\nAnswer: the'


def probe_key(names: bool, word_a: str, word_b: str, ans_a: str, ans_b: str) -> str:
    return json.dumps([bool(names), word_a, word_b, ans_a, ans_b])


def probe_specs(tok, st: bool, key: str, version: str) -> Optional[dict]:
    """{'a1','a2','b1','b2': (prompt, ids, id(ans_a), id(ans_b))} for one probe version, None if an entity is not a single
    distinct continuation token on some probe prompt. a = trigger of A, b = trigger of B; 1 = order (ans_a, ans_b)."""
    names, wa, wb, aa, ab = json.loads(key)
    out = {}
    for k, word in (("a", wa), ("b", wb)):
        for o, (c1, c2) in (("1", (aa, ab)), ("2", (ab, aa))):
            q = probe_prompt(version, word, c1, c2, names)
            qi = list(tok(q, add_special_tokens=st)["input_ids"])
            ia = _single_token_continuation(tok, qi, q, aa, st)
            ib = _single_token_continuation(tok, qi, q, ab, st)
            if ia is None or ib is None or ia == ib:
                return None
            out[k + o] = (q, qi, int(ia), int(ib))
    return out


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def assoc_stratum(vals: dict, responding: list, thr: float = THR) -> str:
    """vals = {version: (probe_a, probe_b)}; probe_a = Delta of A's trigger (own answer = ans_a, so own if >= thr), probe_b
    = Delta of B's trigger (own answer = ans_b, so own if <= -thr)."""
    ps = [vals[v] for v in responding if v in vals and _ok(vals[v][0]) and _ok(vals[v][1])]
    if not ps:
        return "no_probe"
    if all(abs(a) < thr and abs(b) < thr for a, b in ps):
        return "neutral"
    if any(a <= -thr or b >= thr for a, b in ps):
        return "reverse"
    own_a = any(a >= thr for a, _ in ps)
    own_b = any(b <= -thr for _, b in ps)
    return "associative" if own_a and own_b else "one-sided"


def stratum_conflict(vals: dict, responding: list, thr: float = THR) -> bool:
    """A reverse pair whose trigger also points to its own answer under another responding probe (reported)."""
    ps = [vals[v] for v in responding if v in vals and _ok(vals[v][0]) and _ok(vals[v][1])]
    rev = any(a <= -thr or b >= thr for a, b in ps)
    own = any(a >= thr or b <= -thr for a, b in ps)
    return rev and own


# ---------------------------------------------------------------------------------------------------------------
# copy prefix (P5-8)
# ---------------------------------------------------------------------------------------------------------------
PRON = {"he", "she", "her", "him", "they", "them", "it", "hers", "theirs", "himself", "herself", "itself", "themselves", "i",
        "me", "we", "us", "you"}
AUX = {"would", "will", "may", "might", "can", "could", "should", "must", "seemed", "looked", "looks", "became", "learned",
       "learns", "smelled", "smells", "improved", "ended", "pursued", "held", "loves", "got", "felt", "made", "makes", "had"}
PLURAL = {"were", "are", "have", "do", "don't", "weren't", "aren't", "haven't"}


def _is_verb(w: str, nxt: Optional[str]) -> bool:
    if w in VERBS or w in AUX or w in PLURAL or w in IRREG:
        return True
    if w.endswith("ed") and len(w) > 3:
        return True
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        # a noun followed by a verb ("the _ pants were") is not a verb
        return not (nxt is not None and (nxt in VERBS or nxt in AUX or nxt in PLURAL))
    return False


def frame_of(t: Twin) -> tuple[Optional[dict], str]:
    """Phase-3 geometry: {'frame', 'det', 'word_a', 'word_b'} or (None, rule)."""
    ha, wa = split_final_word(t.sent_a)
    hb, wb = split_final_word(t.sent_b)
    if ha != hb or wa == wb or "_" not in ha:
        return None, "C0_not_phase3_geometry"
    i = ha.index("_")
    if re.match(r"\s*'", ha[i + 1:]):
        return None, "C1_possessive_blank"
    frame = ha[i + 1:].strip()
    words = frame.split()
    if not words:
        return None, "C2_empty_frame"
    lw = [re.sub(r"[^a-z']", "", w.lower()) for w in words]
    k = 0
    while k < len(lw) and lw[k] in ADVS:
        k += 1
    if k >= len(lw) or not _is_verb(lw[k], lw[k + 1] if k + 1 < len(lw) else None):
        return None, "C3_not_subject_blank"
    if k + 1 < len(lw) and lw[k + 1] in ("was", "is", "were", "are"):
        return None, "C3_relative_clause_blank"  # "the glue _ had was empty", typo "the _ was was too"
    if lw[k] in PLURAL:
        return None, "C4_plural_verb"
    if set(lw) & PRON:
        return None, "C6_pronoun_in_frame"
    pre = ha[:i].rstrip().split(" ")
    det = pre[-1].lower() if pre and pre[-1].lower() in DETS else ""
    return {"frame": frame, "det": det, "word_a": wa, "word_b": wb, "frame_words": len(words)}, "ok"


def _np(det: str, noun: str) -> str:
    if not det:
        return noun
    if det in ("a", "an"):
        det = "an" if noun[:1].lower() in "aeiou" else "a"
    return f"{det} {noun}"


def copy_prefix(t: Twin, order: int) -> tuple[Optional[str], str, Optional[dict]]:
    f, why = frame_of(t)
    if f is None:
        return None, why, None
    names = bool(NAME_RE.match(t.opt1) and NAME_RE.match(t.opt2))
    t1, t2 = (f["word_a"], f["word_b"]) if order == 1 else (f["word_b"], f["word_a"])
    x, y = (t.opt1, t.opt2) if names else (_np(f["det"], t.opt1), _np(f["det"], t.opt2))
    pre = f"Of {x} and {y}, one {f['frame']} {t1} and the other {f['frame']} {t2}. "
    return pre, "ok", dict(f, t1=t1, t2=t2, names=names)


def options_prefix(t: Twin, order: int) -> tuple[Optional[str], str, Optional[dict]]:
    """Copy prefix of the coordinator decision (2026-10-11): 'Options: t1, t2.' + newline, both attribute orders; pairs nothing
    with the names (no grammar rules needed). An order preference is identical in both twins and cancels in the STR drop."""
    ha, wa = split_final_word(t.sent_a)
    hb, wb = split_final_word(t.sent_b)
    if ha != hb or wa == wb or "_" not in ha:
        return None, "C0_not_phase3_geometry", None
    names = bool(NAME_RE.match(t.opt1) and NAME_RE.match(t.opt2))
    t1, t2 = (wa, wb) if order == 1 else (wb, wa)
    return f"Options: {t1}, {t2}.\n", "ok", {"frame": "", "det": "", "word_a": wa, "word_b": wb, "t1": t1, "t2": t2, "names": names, "frame_words": 0}


PREFIX_STYLES = {"respectively": copy_prefix, "options": options_prefix}


def copy_twin(t: Twin, order: int, tag: str = "", style: str = "options") -> tuple[Optional[Twin], str, Optional[dict]]:
    pre, why, info = PREFIX_STYLES[style](t, order)
    if pre is None:
        return None, why, None
    pid = f"{t.pair_id}|o{order}" if not tag else f"{t.pair_id}|{tag}|o{order}"
    return Twin(pid, t.q_a, t.q_b, pre + t.sent_a, pre + t.sent_b, t.opt1, t.opt2, t.ans_a, t.ans_b), "ok", dict(info, prefix=pre)


# ---------------------------------------------------------------------------------------------------------------
# token positions
# ---------------------------------------------------------------------------------------------------------------
def span(enc, start: int, end: int) -> list[int]:
    return [i for i, ((a, b), sp) in enumerate(zip(enc["offset_mapping"], enc["special_tokens_mask"]))
            if not sp and b > a and a < end and b > start]


def find_word(text: str, word: str, lo: int = 0, hi: Optional[int] = None) -> Optional[tuple[int, int]]:
    hi = len(text) if hi is None else hi
    for cand in (word, word.lower(), word[:1].upper() + word[1:]):
        m = re.compile(r"\b" + re.escape(cand) + r"\b").search(text, lo, hi)
        if m:
            return m.start(), m.end()
    return None


def encode(tok, text: str, st: bool):
    return tok(text, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=st)


def positions_phase3(tok, st: bool, row: dict, prefix: str = "") -> dict:
    """Phase-3-format pair (prompt = sentence filled up to the trigger): mention_a/b_pos = first occurrence of ans_a / ans_b
    in the sentence before the blank (after the copy prefix if any); with a prefix also prefix_ment_a/b_pos (X / Y in the
    prefix) and attr_a/b_pos (word_a / word_b in the prefix = the in-context answers)."""
    enc = encode(tok, row["prompt_a"], st)
    ids = json.loads(row["ids_a"]) if isinstance(row["ids_a"], str) else [int(x) for x in row["ids_a"]]
    assert list(enc["input_ids"]) == ids, row["pair_id"]
    lo = len(prefix)
    blank = row["context"].index("_", lo)
    out = {}
    for k, ans in (("a", row["ans_a"]), ("b", row["ans_b"])):
        m = find_word(row["prompt_a"], ans, lo, blank)
        out[f"mention_{k}_pos"] = json.dumps(span(enc, *m) if m else [])
    if prefix:
        for k, ans in (("a", row["ans_a"]), ("b", row["ans_b"])):
            m = find_word(prefix, ans, 0, len(prefix))
            out[f"prefix_ment_{k}_pos"] = json.dumps(span(enc, *m) if m else [])
        # attributes: t1 is the first ", one {frame} " slot, t2 the "the other {frame} " slot
        for k, w in (("a", row["word_a"]), ("b", row["word_b"])):
            ms = [mm for mm in re.finditer(r"\b" + re.escape(w) + r"\b", prefix)]
            out[f"attr_{k}_pos"] = json.dumps(span(enc, ms[-1].start(), ms[-1].end()) if ms else [])
        pp = span(enc, 0, len(prefix))
        out["prefix_len_tokens"] = (max(pp) + 1) if pp else 0
    return out


def positions_f1(tok, st: bool, row: dict) -> dict:
    """Fill-in-the-blank pair: mentions in the sentence part, blank_pos = the '_' token(s) (as scripts/ext13_positions.py)."""
    enc = encode(tok, row["prompt_a"], st)
    ids = json.loads(row["ids_a"]) if isinstance(row["ids_a"], str) else [int(x) for x in row["ids_a"]]
    assert list(enc["input_ids"]) == ids, row["pair_id"]
    lo = len(F1_HEAD)
    hi = row["prompt_a"].rindex("\nAnswer:")
    bj = row["prompt_a"].index("_", lo)
    out = {"blank_pos": json.dumps(span(enc, bj, bj + 1))}
    for k, ans in (("a", row["ans_a"]), ("b", row["ans_b"])):
        m = find_word(row["prompt_a"], ans, lo, hi)
        out[f"mention_{k}_pos"] = json.dumps(span(enc, *m) if m else [])
    return out


def positions_f2(tok, st: bool, row: dict) -> dict:
    """Natural truncation (prompt = sentence up to the blank): first mentions in the prompt."""
    enc = encode(tok, row["prompt_a"], st)
    out = {}
    for k, ans in (("a", row["ans_a"]), ("b", row["ans_b"])):
        m = find_word(row["prompt_a"], ans, 0, len(row["prompt_a"]))
        out[f"mention_{k}_pos"] = json.dumps(span(enc, *m) if m else [])
    return out


def answer_in_prefix(row: dict) -> bool:
    ids = json.loads(row["ids_a"]) if isinstance(row["ids_a"], str) else list(row["ids_a"])
    n = int(row["prefix_len_tokens"])
    return int(row["trig_a"]) in ids[:n] and int(row["trig_b"]) in ids[:n]


def ensure_pair_row(p) -> dict:
    """Pair (ext7_wino) -> row with str_pos = opt_pos (pair contract)."""
    r = p.to_row()
    r["str_pos"] = r["opt_pos"]
    return r


def build_copy(t: Twin, order: int, tok, st: bool, tag: str = "", style: str = "options") -> tuple[Optional[dict], str]:
    t2, why, info = copy_twin(t, order, tag, style)
    if t2 is None:
        return None, why
    p, why = build_pair(t2, tok, st)
    if p is None:
        return None, why
    r = ensure_pair_row(p)
    r.update(order=order, prefix=info["prefix"], frame=info["frame"], t1=info["t1"], t2=info["t2"], base_pair_id=t.pair_id, prefix_style=style)
    r.update(positions_phase3(tok, st, r, info["prefix"]))
    r["answer_in_prefix"] = answer_in_prefix(r)
    if not r["answer_in_prefix"]:
        return None, "C7_answer_token_not_in_prefix"
    if style == "options":  # each attribute is exactly one prefix token with the continuation id (leading-space form after ': ' / ', ')
        ids = json.loads(r["ids_a"])
        aa, ab = json.loads(r["attr_a_pos"]), json.loads(r["attr_b_pos"])
        if len(aa) != 1 or len(ab) != 1 or ids[aa[0]] != int(r["trig_a"]) or ids[ab[0]] != int(r["trig_b"]):
            return None, "C8_prefix_attribute_id_differs"
    return r, "ok"
