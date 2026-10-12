"""ext13 (Phase 5): WinoGrande in its original direction, attribute -> one of the two entities (reference resolution).

Phase 3 filled the blank and predicted the sentence-final trigger (entity -> attribute). Here the STR corruption is the
twin's own difference (the trigger, e.g. dirty <-> weak) and the readout is the entity the blank refers to:

    Delta = logit(entity_r) - logit(entity_r')  at the final prompt token.

Formats (user decisions, RESEARCH_PLAN.md "Phase 5 decisions"; revised 2026-10-11 after the truncation correction):
  The model must see the same information as in WinoGrande (the whole sentence). Cutting at the blank drops the words
  after it, which decide the answer in 69 % of the F2 pairs ("... because _ had been running on the treadmill all day").
  F2R restate (PRIMARY): F2 twins (only difference before the blank) with a subject blank; the whole sentence with the
        blank -> "one of them", the words after the blank restated, the entity slot last (restate_prompts):
        "Natalie was more tired than Betty because one of them had been running on the treadmill all day. The one who
         had been running on the treadmill all day was" -> " Natalie"   (B: more energetic -> " Betty")
  F2E natural truncation, blank at the sentence end only (comparison; = the F2 pairs with no word after the blank, where
        the cut loses nothing): "The cake was sweeter than the pie because there was more sugar used in the" -> " cake"
  F2  the original cut at the blank: built and scanned 2026-10-11, NOT a valid task outside F2E (kept for the record).
  F1 fill-in-the-blank (comparison, covers the twins whose trigger follows the blank):
        "Fill in the blank: The janitor tried to clean the mirror with the spray but the _ was too dirty.\nAnswer: the"
        -> " mirror"   (B: weak -> " spray")
Pair rules (per tokenizer):
  R1  twin integrity (ext7_wino.load_twins: two items per HIT, one blank, same options, answers differ);
  R2  F2 only: the sentences are identical from the blank on and differ before it;
  R3  both entities are single continuation tokens after both prompts, with the same ids, and differ;
  R4  token alignment: equal length, 1-3 differing token positions (= str_pos), identical elsewhere;
  R5  the differing tokens are not the final prompt token (STR site != readout position);
  R6  one pair per normalised prompt (options replaced by placeholders);
  R7  F2R only: subject blank (a verb after the blank, after optional adverbs), not inside a prepositional phrase
      ("the speakers on the _ were bigger"), not plural ("the _ were fresher"), no possessive blank, predicate >= 2
      words; copula 'is' for present-tense
      predicates, else 'was'.
Strata recorded, not filtered here: swap_type (content / connective: because <-> although, but, ...), names (both
options capitalised names), debiased (both qIDs in train_debiased), n_diff_tokens.
main = content swaps (connective swaps are a separate stratum; user decision 2026-10-11).

Association probe (context-free partial-input baseline; Trichelair et al. 2019, Elazar et al. 2021): the twin's trigger
word(s) and the two candidates in a neutral frame, both candidate orders,
    "Word: dirty. Which fits better, the mirror or the spray? Answer: the" -> " mirror" / " spray"
    names: "Word: generous. Which fits better, Brett or Kevin? Answer:" -> " Brett" / " Kevin"
associative pair = the probe is right in BOTH directions with margin >= 1 logit (mean over the two orders); main
(balanced) set = non-associative pairs; strict hard subset = right in neither direction.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .data import _single_token_continuation
from .ext7_wino import NAME_RE, Twin, load_items, load_twins

CONN = {"because", "but", "although", "though", "so", "since", "while", "as", "yet", "and", "therefore", "however",
        "whereas", "unless", "if", "when", "until", "before", "after", "or", "thus", "hence", "despite"}
DETS = {"the", "a", "an", "his", "her", "their", "my", "our", "your", "its", "this", "that", "these", "those"}
F1_HEAD = "Fill in the blank: "
TRAIL_P = ".!?\"' "


def _words(s: str) -> list[str]:
    return s.split()


def _clean(w: str) -> str:
    return w.strip(".,;:!?\"'()").lower()


def word_diff(a: str, b: str) -> tuple[str, str, bool]:
    """Differing words of two equal-word-length strings: (words of a, words of b, ok). ok = same word count."""
    wa, wb = _words(a), _words(b)
    if len(wa) != len(wb):
        return "", "", False
    d = [i for i, (x, y) in enumerate(zip(wa, wb)) if x != y]
    return " ".join(_clean(wa[i]) for i in d), " ".join(_clean(wb[i]) for i in d), True


def f2_prompts(t: Twin) -> Optional[tuple[str, str]]:
    a, b = t.sent_a, t.sent_b
    ia, ib = a.index("_"), b.index("_")
    ha, hb = a[:ia].rstrip(), b[:ib].rstrip()
    if ha == hb or a[ia:] != b[ib:]:
        return None
    return ha, hb


def f1_prompt(sent: str) -> str:
    i = sent.index("_")
    before = sent[:i].rstrip().split(" ")
    det = before[-1].lower() if before and before[-1].lower() in DETS else ""
    return F1_HEAD + sent.strip() + "\nAnswer:" + (" " + det if det else "")


def f1_prompts(t: Twin) -> Optional[tuple[str, str]]:
    pa, pb = f1_prompt(t.sent_a), f1_prompt(t.sent_b)
    return None if pa == pb else (pa, pb)


ADVS = {"always", "never", "also", "usually", "often", "really", "just", "still", "even", "only", "already", "clearly",
        "simply", "finally", "actually", "rarely", "sometimes", "generally", "probably", "certainly", "definitely", "obviously",
        "constantly", "frequently", "easily", "quickly", "slowly", "truly", "barely", "hardly", "mostly", "seldom", "then"}
VERBS = {"was", "is", "had", "has", "were", "are", "did", "does", "could", "would", "should", "will", "can", "might", "must",
         "may", "shall", "got", "gets", "felt", "feels", "knew", "knows", "made", "makes", "took", "takes", "went", "goes",
         "thought", "thinks", "seemed", "seems", "kept", "keeps", "ate", "eats", "lost", "loses", "won", "wins", "gave", "gives",
         "found", "finds", "became", "becomes", "came", "comes", "ran", "runs", "saw", "sees", "said", "says", "told", "tells",
         "wore", "wears", "spent", "spends", "left", "leaves", "grew", "grows", "drank", "drinks", "slept", "sleeps", "bought",
         "buys", "paid", "pays", "brought", "brings", "taught", "teaches", "held", "holds", "sold", "sells", "led", "leads",
         "fell", "falls", "drove", "drives", "wrote", "writes", "read", "reads", "lives", "likes", "loves", "hates", "needs",
         "wants", "prefers", "enjoys", "tends", "uses", "works", "plays", "owns", "looks", "tries", "cares", "gets", "doesn't",
         "didn't", "wasn't", "isn't", "hasn't", "hadn't", "couldn't", "wouldn't", "won't", "can't", "weren't", "aren't",
         "shouldn't", "dislikes", "suffers", "struggles", "worries", "smokes", "exercises", "eats", "has", "lacks"}
PRESENT = {"is", "are", "has", "does", "can", "will", "may", "shall", "must", "doesn't", "isn't", "hasn't", "can't", "won't",
           "aren't"}


PREPS = {"at", "on", "of", "in", "from", "for", "with", "to", "by", "about", "under", "over", "inside", "near", "behind",
         "into", "onto", "within", "without", "between", "through", "around", "outside", "across", "beside"}


def _present(w: str) -> bool:
    return w in PRESENT or (w.endswith("s") and not w.endswith("ss") and w not in ("was", "has") and not w.endswith("'s"))


def _verb_word(w: str) -> bool:
    return w in VERBS or (w.endswith("ed") and len(w) > 3)


def restate_prompts(t: Twin, names: bool) -> tuple[Optional[tuple[str, str]], str]:
    """F2R: the whole sentence with the blank -> 'one of them' (a determiner before the blank is dropped), then the
    words after the blank restated as a question frame ending at the entity slot:
        names    "... because one of them had been running on the treadmill all day. The one who had been running on the
                  treadmill all day was" -> " Natalie"
        objects  "... because one of them was porous. The one that was porous was the" -> " grass"
    Needs the F2 geometry (twins differ before the blank, identical from it on) and a subject blank: the next word after
    the blank (after optional adverbs) is a verb, and the predicate has >= 2 words (no elliptical '_ did.')."""
    a, b = t.sent_a.strip(), t.sent_b.strip()
    ia, ib = a.index("_"), b.index("_")
    if a[:ia] == b[:ib] or a[ia:] != b[ib:]:
        return None, "R2_not_f2_geometry"
    post = a[ia + 1:]
    if re.match(r"\s*'", post):
        return None, "R7_possessive_blank"
    pred = post.strip().rstrip(TRAIL_P).strip()
    words = re.findall(r"[A-Za-z']+", pred.lower())
    if not words:
        return None, "R7_blank_at_end"
    k = 0
    while k < len(words) and words[k] in ADVS:
        k += 1
    if k >= len(words) or not _verb_word(words[k]):
        return None, "R7_not_subject_blank"
    if len(words) < 2:
        return None, "R7_elliptical_predicate"
    if words[k] in ("were", "are", "have", "do", "don't", "weren't", "aren't", "haven't"):
        return None, "R7_plural_blank"  # "one of them were ..." is not grammatical
    cop = "is" if _present(words[k]) else "was"
    out = []
    for s, i in ((a, ia), (b, ib)):
        pre = s[:i].rstrip()
        pw = pre.split(" ")
        if pw and pw[-1].lower() in DETS:
            pre = " ".join(pw[:-1])
            pw = pw[:-1]
            # "the speakers on the _ were bigger": the predicate is about the PP's head noun ("for _ was" is a conjunction)
            if pw and pw[-1].lower() in PREPS:
                return None, "R7_blank_in_pp"
        start = not pre or pre.endswith((".", "!", "?"))
        one = "One of them" if start else "one of them"
        body = (pre + " " if pre else "") + one + post.rstrip()
        if not body.endswith((".", "!", "?")):
            body += "."
        tail = f" The one who {pred} {cop}" if names else f" The one that {pred} {cop} the"
        out.append(body + tail)
    return (out[0], out[1]), "ok"


def probe_prompt(word: str, c1: str, c2: str, names: bool) -> str:
    if names:
        return f"Word: {word}. Which fits better, {c1} or {c2}? Answer:"
    return f"Word: {word}. Which fits better, the {c1} or the {c2}? Answer: the"


@dataclass
class RefPair:
    pair_id: str
    q_a: str
    q_b: str
    fmt: str
    sent_a: str
    sent_b: str
    ans_a: str
    ans_b: str
    word_a: str  # differing word(s) of A (the trigger)
    word_b: str
    prompt_a: str
    prompt_b: str
    ids_a: list
    ids_b: list
    str_pos: list
    trig_a: int  # entity token of A's answer (true for direction d = 0)
    trig_b: int
    n_diff_tokens: int
    swap_type: str
    names: bool
    debiased: bool
    norm_key: str
    probe: dict = field(default_factory=dict)  # {"a1","a2","b1","b2": (prompt, ids, ent_a_id, ent_b_id)}

    @property
    def main(self) -> bool:
        return self.swap_type == "content"

    def to_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "probe"}
        d["ids_a"], d["ids_b"], d["str_pos"] = json.dumps(self.ids_a), json.dumps(self.ids_b), json.dumps(self.str_pos)
        d["main"] = self.main
        d["n_tokens"] = len(self.ids_a)
        d["probe_json"] = json.dumps(self.probe)
        return d


def _ent_ids(tok, ids: list, prompt: str, ans_true: str, ans_other: str, st: bool):
    return (_single_token_continuation(tok, ids, prompt, ans_true, st),
            _single_token_continuation(tok, ids, prompt, ans_other, st))


def build_ref_pair(t: Twin, fmt: str, tok, st: bool, debiased: set) -> tuple[Optional[RefPair], str]:
    names = bool(NAME_RE.match(t.opt1) and NAME_RE.match(t.opt2))
    if fmt == "F2R":
        pr, why = restate_prompts(t, names)
        if pr is None:
            return None, why
    else:
        pr = f2_prompts(t) if fmt == "F2" else f1_prompts(t)
        if pr is None:
            return None, "R2_no_difference_before_blank" if fmt == "F2" else "R2_identical_prompts"
    pa, pb = pr
    ida, idb = list(tok(pa, add_special_tokens=st)["input_ids"]), list(tok(pb, add_special_tokens=st)["input_ids"])
    ea_a, eb_a = _ent_ids(tok, ida, pa, t.ans_a, t.ans_b, st)
    eb_b, ea_b = _ent_ids(tok, idb, pb, t.ans_b, t.ans_a, st)
    if None in (ea_a, eb_a, eb_b, ea_b):
        return None, "R3_entity_multi_token"
    if ea_a != ea_b or eb_a != eb_b:
        return None, "R3_entity_id_depends_on_prompt"
    if ea_a == eb_b:
        return None, "R3_same_entity_token"
    if len(ida) != len(idb):
        return None, "R4_length_mismatch"
    diff = [k for k, (x, y) in enumerate(zip(ida, idb)) if x != y]
    if not diff:
        return None, "R4_identical_tokens"
    if len(diff) > 3:
        return None, "R4_more_than_3_tokens"
    if diff[-1] == len(ida) - 1:
        return None, "R5_difference_at_final_token"
    src_a = pa if fmt == "F2" else t.sent_a
    src_b = pb if fmt == "F2" else t.sent_b
    wa, wb, ok = word_diff(src_a, src_b)
    if not ok or not wa or not wb:
        return None, "R4_word_count_mismatch"
    swap = "connective" if any(w in CONN for w in (wa.split() + wb.split())) else "content"
    norm = pa.replace(t.opt1, "<O1>").replace(t.opt2, "<O2>")
    norm = re.sub(r"\s+", " ", norm).strip().lower()
    p = RefPair(t.pair_id, t.q_a, t.q_b, fmt, t.sent_a, t.sent_b, t.ans_a, t.ans_b, wa, wb, pa, pb, ida, idb, diff,
                int(ea_a), int(eb_b), len(diff), swap, names, t.q_a in debiased and t.q_b in debiased, norm)
    # association probe: both directions x both candidate orders; entity ids resolved on the probe prompt
    for key, word in (("a", wa), ("b", wb)):
        for o, (c1, c2) in (("1", (t.ans_a, t.ans_b)), ("2", (t.ans_b, t.ans_a))):
            q = probe_prompt(word, c1, c2, names)
            qi = list(tok(q, add_special_tokens=st)["input_ids"])
            ia, ib = _ent_ids(tok, qi, q, t.ans_a, t.ans_b, st)
            if ia is None or ib is None or ia == ib:
                return None, "R3_entity_multi_token_in_probe"
            p.probe[key + o] = [q, qi, int(ia), int(ib)]
    return p, "ok"


def build_ref_pairs(tok, st: bool, fmt: str, split: str = "train_xl") -> tuple[list[RefPair], dict]:
    twins, funnel = load_twins(split)
    deb = {r["qID"] for r in load_items("train_debiased")} if split == "train_xl" else set()
    kept, reasons = [], {}
    for t in twins:
        p, why = build_ref_pair(t, fmt, tok, st, deb)
        if p is None:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            kept.append(p)
    seen, out = set(), []
    for p in kept:
        if p.norm_key in seen:
            reasons["R6_duplicate_prompt"] = reasons.get("R6_duplicate_prompt", 0) + 1
            continue
        seen.add(p.norm_key)
        out.append(p)
    f = {"twins": funnel["W1_twins"], "dropped": reasons, "kept": len(out),
         "kept_main": sum(p.main for p in out), "kept_connective": sum(not p.main for p in out),
         "kept_names": sum(p.names for p in out), "kept_main_names": sum(p.main and p.names for p in out)}
    return out, f
