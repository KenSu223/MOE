"""ext14 Phase 6 (agent p6-data): Phase-6 cells as STR pairs (RESEARCH_PLAN.md "Phase 6"; briefs logs/agent_brief_p6_*.md).

Cells (run dir results/p6_<cell>_<proto>):
  Dial E (same WinoGrande twins, only the evidence changes; a2e = attribute -> entity, e2a = entity -> attribute)
    e3_a2e   F1 frame on the original sentence      "Fill in the blank: <sentence with _>\nAnswer: the" -> " jug"   (STR = trigger)
    e3_e2a   Phase-3 option swap                     "<sentence, blank filled, cut before the trigger>" -> " empty" (STR = entity)
    e1_*     evidence clause -> both filled blank clauses ("The jug was empty and the pot was full, so the _ was empty.")
    e1p_*    as E1 with paraphrased triggers in the statements ("The jug was drained and the pot was filled, so ...")
    e2_*     negation ("The jug was not full and the pot was not empty, so ...")
    e0_*     prior only ("Of the elephant and the mouse, the _ was big."), strong-prior items
  Dial A   aagr_* / acon_*  E1 frame, statements agree / conflict with a strong lexical prior (controlled noun pairs + WinoGrande
           strong-prior twins)
  Set C    sc_agree_a2e / sc_conflict_a2e  "Fill in the blank: <statement>. <original sentence>\nAnswer: the"
  R-site   rsite_e1_a2e  E1-a2e with the corruption in the evidence (the two entity mentions swapped; two tokens)
  Ladder   l_ant, l_prop, l_bind, l_ind (built here), l_cf / l_ioi (anchors: existing runs)

Item rules (RESEARCH_PLAN "Common rules for every Phase-6 cell"):
  R1  both answers are single leading-space continuation tokens with the same ids after both prompts, and differ;
  R2  equal length, exactly one differing token (R-site: exactly two), not the final token;
  R3  margin logit(answer) - logit(contrast) >= 1 in both directions (scan, scripts/ext14_scan.py);
  R4  dedup by normalised prompt (entities -> placeholders); two-entity frames in both orders, an item counts only if both
      orders pass (scripts/ext14_casesets.py);
  R5  discovery / validation 50 / 50 by item (seeded), cap 128 / 128 pairs; >= 64 validation pairs or "insufficient".
Dial-E blank clause: Phase-3 geometry twins (the two sentences differ only in their final word = the trigger) whose blank clause is
"(det) _ COP [ADV ...] TRIGGER" with COP in was / is / were / are (BC_RE); det in the, his, her, their, my, our, its, your or none
(names). E2 needs exactly this form (brief), and the same item set is used at every level (pairing by src_pair_id).

pair_idx (unique per cell): WinoGrande items 2 * twin_idx + (order - 1), twin_idx = index of the HIT id in the sorted train_xl twins
(ext7_wino.load_twins, as data/wino_str/case_sets.json); E3 uses the natural order of the sentence (entity of A mentioned first = 1).
Controlled Dial-A items 100000 + 2 * item_idx + (order - 1). Ladder tasks: see the builders.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from .data import _single_token_continuation
from .ext7_wino import NAME_RE, Twin, build_pair as p3_build_pair, load_twins, split_final_word

ROOT = "/home/ubuntu/MOE"
DIAL = f"{ROOT}/data/dial"
BUILD = f"{DIAL}/build"
F1_HEAD = "Fill in the blank: "

# proto -> (model key, add special tokens, tokenizer family). Stage-2 protos are added when their engine support is ready.
PROTOS = {"qwen3": ("qwen3", True, "qwen"), "mixtral_bos": ("mixtral", True, "mistral"),
          # Stage 2 (tokenizer identity checked: data/dial/qa/tokenizer_identity.md -> the family's pair files are reused)
          "mistral7b_bos": ("mistral7b", True, "mistral"), "qwen2_7b": ("qwen2_7b", True, "qwen"), "qwen2_57b": ("qwen2_57b", True, "qwen"),
          "qwen3_1p7b": ("qwen3_1p7b", True, "qwen"), "qwen3_4b": ("qwen3_4b", True, "qwen"), "qwen3_8b": ("qwen3_8b", True, "qwen"),
          "qwen3_14b": ("qwen3_14b", True, "qwen")}
# Stage-2 model -> its Stage-1 counterpart, whose selected items are preferred when capping (paired twin / series intersections)
PREFER = {"mistral7b_bos": "mixtral_bos", "qwen2_7b": "qwen2_57b", "qwen2_57b": "qwen3", "qwen3_1p7b": "qwen3", "qwen3_4b": "qwen3",
          "qwen3_8b": "qwen3", "qwen3_14b": "qwen3"}
FAMILY_TOKENIZER = {"qwen": "qwen3", "mistral": "mixtral"}  # family -> model key whose tokenizer builds the family's items


def tokenizer_of(model_key: str):
    from transformers import AutoTokenizer
    from .arch import snapshot_dir
    from .models import MODELS
    return AutoTokenizer.from_pretrained(snapshot_dir(MODELS[model_key]["repo"]))


def family_tokenizer(fam: str):
    return tokenizer_of(FAMILY_TOKENIZER[fam]), True


# ---------------------------------------------------------------------------------------------------------------
# generic STR pair from two texts, with labelled character spans
# ---------------------------------------------------------------------------------------------------------------
class Text:
    """Prompt assembled from pieces; labelled pieces keep their character spans."""

    def __init__(self):
        self.s = ""
        self.spans: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)

    def add(self, piece: str, label: Optional[str] = None) -> "Text":
        if label and piece:
            self.spans[label].append((len(self.s), len(self.s) + len(piece)))
        self.s += piece
        return self


def encode(tok, text: str, st: bool):
    return tok(text, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=st)


def tok_span(enc, a: int, b: int) -> list[int]:
    return [i for i, ((x, y), sp) in enumerate(zip(enc["offset_mapping"], enc["special_tokens_mask"])) if not sp and y > x and x < b and y > a]


def make_pair(tok, st: bool, ta: Text, tb: Text, ans_a: str, ans_b: str, n_swap: int = 1, allow_final: bool = False) -> tuple[Optional[dict], str]:
    """R1 / R2 for one pair of prompts. ans_x = answer word after prompt x (leading-space continuation)."""
    pa, pb = ta.s, tb.s
    ea, eb = encode(tok, pa, st), encode(tok, pb, st)
    ia, ib = list(ea["input_ids"]), list(eb["input_ids"])
    x_aa = _single_token_continuation(tok, ia, pa, ans_a, st)
    x_ab = _single_token_continuation(tok, ia, pa, ans_b, st)
    x_ba = _single_token_continuation(tok, ib, pb, ans_a, st)
    x_bb = _single_token_continuation(tok, ib, pb, ans_b, st)
    if None in (x_aa, x_ab, x_ba, x_bb):
        return None, "R1_answer_multi_token"
    if x_aa != x_ba or x_ab != x_bb:
        return None, "R1_answer_id_depends_on_prompt"
    if x_aa == x_bb:
        return None, "R1_same_answer_token"
    if len(ia) != len(ib):
        return None, "R2_length_mismatch"
    diff = [k for k, (x, y) in enumerate(zip(ia, ib)) if x != y]
    if len(diff) != n_swap:
        return None, f"R2_{len(diff)}_differing_tokens"
    if diff[-1] == len(ia) - 1 and not allow_final:
        return None, "R2_swap_at_final_token"
    pos = {}
    for lab, sp in ta.spans.items():
        pos[lab] = sorted({i for a, b in sp for i in tok_span(ea, a, b)})
    return {"prompt_a": pa, "prompt_b": pb, "ids_a": json.dumps(ia), "ids_b": json.dumps(ib), "str_pos": json.dumps(diff),
            "trig_a": int(x_aa), "trig_b": int(x_bb), "n_tokens": len(ia), "positions": pos}, "ok"


POS_COLS = ("mention_a_pos", "mention_b_pos", "attr_a_pos", "attr_b_pos", "blank_pos", "q_pos")


def pos_columns(pos: dict) -> dict:
    """Labelled spans -> JSON position columns (labels: ment_a, ment_b, attr_a, attr_b, blank, q)."""
    m = {"mention_a_pos": "ment_a", "mention_b_pos": "ment_b", "attr_a_pos": "attr_a", "attr_b_pos": "attr_b", "blank_pos": "blank",
         "q_pos": "q"}
    return {c: json.dumps(pos.get(lab, [])) for c, lab in m.items()}


# ---------------------------------------------------------------------------------------------------------------
# Dial-E items from WinoGrande twins
# ---------------------------------------------------------------------------------------------------------------
DET_OK = ("the", "his", "her", "their", "my", "our", "its", "your")
PP_PREPS = {"of", "on", "in", "with", "at", "to", "from", "by", "about", "under", "over", "into", "onto", "near", "behind", "inside",
            "outside", "between", "through", "around", "across", "beside", "within", "without", "like", "than", "toward", "towards",
            "upon", "against", "among", "beneath", "below", "above"}
COMPLEMENT_VERBS = {"thought", "think", "thinks", "realized", "realised", "believed", "believe", "believes", "said", "say", "says", "guess",
                    "seems", "seemed", "felt", "feel", "feels", "knew", "know", "knows", "noticed", "decided", "found", "hoped", "hope",
                    "assumed", "assume", "figured", "learned", "saw", "see", "sees", "heard", "suspected", "concluded", "claimed",
                    "argued", "agreed", "explained", "complained", "worried", "admitted", "insisted", "suggested", "determined",
                    "discovered", "remembered", "understood", "guessed", "reckon", "reckoned", "doubt", "doubted", "sure", "clear", "obvious"}
ADV_ALT = ["a bit", "a lot", "a little", "much more", "so much", "far more", "way too", "too", "very", "more", "so", "really", "quite",
           "rather", "less", "extremely", "much", "completely", "pretty", "fairly", "totally", "entirely", "overly", "incredibly",
           "super", "increasingly", "even", "still", "always", "simply", "just", "too much", "somewhat", "slightly", "highly", "seriously",
           "absolutely", "particularly", "especially", "noticeably", "considerably", "significantly", "ever", "all"]
ADV_ALT = sorted(set(ADV_ALT), key=lambda s: -len(s))
BC_RE = re.compile(r"(?:^|(?<=[\s,;:\"(]))((?:(%s)\s+)?_\s+(was|is|were|are)((?:\s+(?:%s))*))$"
                   % ("|".join(DET_OK), "|".join(re.escape(a) for a in ADV_ALT)), re.I)


def twin_index() -> dict[str, int]:
    twins, _ = load_twins("train_xl")
    return {t.pair_id: i for i, t in enumerate(twins)}


@dataclass
class Item:
    """One WinoGrande twin (or a controlled Dial-A item) in the Dial-E geometry."""
    src_pair_id: str
    twin_idx: int
    source: str  # wino | ctrl
    names: bool
    det: str  # lower-case determiner of the blank NP ('' for names)
    cop: str
    adv: str  # '' or the degree adverb(s) between the copula and the trigger
    ent_a: str  # twin binding: ent_a <-> trig_a (the answer of sentence A)
    ent_b: str
    trig_a: str
    trig_b: str
    sent_a: str = ""
    sent_b: str = ""
    head: str = ""  # sentence without the final word (Phase-3 geometry), blank included
    bc_start: int = -1  # char index of the blank clause in head
    core776: bool = False
    extra: dict = field(default_factory=dict)

    # -- text pieces -------------------------------------------------------------------------------
    def bc_text(self) -> str:
        """Blank clause up to the trigger, verbatim from the sentence (sentence-initial determiner lower-cased)."""
        if self.source != "wino":
            return (f"{self.det} " if self.det else "") + "_ " + self.cop + (f" {self.adv}" if self.adv else "")
        s = re.sub(r"\s+", " ", self.head[self.bc_start:])
        if self.det and s[:1].isupper():
            s = s[0].lower() + s[1:]
        return s

    def stmt(self, t: Text, ent: str, word: str, ent_lab: str, attr_lab: str, neg: bool = False, cap: bool = False) -> Text:
        det = self.det
        if det:
            t.add((det[0].upper() + det[1:]) if cap else det).add(" ")
            t.add(ent, ent_lab)
        else:
            t.add(ent, ent_lab)
        t.add(" " + self.cop + (" not" if neg else "") + (f" {self.adv}" if self.adv else "") + " ")
        t.add(word, attr_lab)
        return t

    def norm_ents(self, s: str) -> str:
        """Phase-3 W6 style key (entities -> placeholders): used for E3 / Set C (the sentence is the item)."""
        s = re.sub(r"\b%s\b" % re.escape(self.ent_a), "<E1>", s)
        s = re.sub(r"\b%s\b" % re.escape(self.ent_b), "<E2>", s)
        return re.sub(r"\s+", " ", s).strip().lower()

    @staticmethod
    def norm_text(s: str) -> str:
        """Exact normalised text (case, whitespace): used for E0-E2 / Dial A / R-site, whose prompts no longer contain the event,
        so the same trigger pair with different entities is a different item (the trigger pair is kept as trig_key for clustering)."""
        return re.sub(r"\s+", " ", s).strip().lower()


def parse_twin(t: Twin, tidx: int, core: bool) -> tuple[Optional[Item], str]:
    ha, wa = split_final_word(t.sent_a)
    hb, wb = split_final_word(t.sent_b)
    if ha != hb or wa == wb or "_" not in ha:
        return None, "D0_not_phase3_geometry"
    if not re.fullmatch(r"[A-Za-z][A-Za-z\-]*", wa) or not re.fullmatch(r"[A-Za-z][A-Za-z\-]*", wb):
        return None, "D1_trigger_not_a_word"
    m = BC_RE.search(ha)
    if not m:
        return None, "D2_blank_clause_not_copular"
    det = (m.group(2) or "").lower()
    names = bool(NAME_RE.match(t.opt1) and NAME_RE.match(t.opt2))
    if not det and not names:
        return None, "D3_object_blank_without_determiner"
    if det and names:
        return None, "D3_name_blank_with_determiner"
    adv = re.sub(r"\s+", " ", (m.group(4) or "").strip()).lower()
    pre = re.sub(r"\s+", " ", ha[:m.start(1)]).rstrip()
    last = pre.split()[-1].lower() if pre.split() else ""
    if last in PP_PREPS or (last == "for" and not pre[:-3].rstrip().endswith(",")):
        return None, "D5_blank_in_pp"  # 'the sauce on the _ was', 'the job of _ was': the blank is not the clause subject
    if " " in t.ans_a.strip() or " " in t.ans_b.strip():
        return None, "D4_multiword_option"
    it = Item(t.pair_id, tidx, "wino", names, det, m.group(3).lower(), adv, t.ans_a, t.ans_b, wa, wb, t.sent_a, t.sent_b, ha,
              m.start(1), core)
    return it, "ok"


def core_776() -> dict[str, str]:
    """HIT id -> Phase-3 split of the 776 shared twins (data/wino_str/case_sets.json)."""
    cs = json.load(open(f"{ROOT}/data/wino_str/case_sets.json"))
    inv = {v: k for k, v in cs["pair_idx_of"].items()}
    return {inv[i]: s for s, v in cs["pairs"].items() for i in v}


def dial_items() -> tuple[list[Item], dict]:
    """All Phase-3-geometry twins of train_xl with a copular blank clause (776 core twins flagged)."""
    twins, _ = load_twins("train_xl")
    core = core_776()
    out, why = [], collections.Counter()
    for i, t in enumerate(twins):
        it, w = parse_twin(t, i, t.pair_id in core)
        k = "core" if t.pair_id in core else "ext"
        why[f"{k}:{w}"] += 1
        if it is not None:
            out.append(it)
    return out, dict(why)


# ---------------------------------------------------------------------------------------------------------------
# level builders (one item, one order, both twins) -> (Text A, Text B, answer A, answer B, n_swap) or reason
# ---------------------------------------------------------------------------------------------------------------
def _ents_in_order(it: Item, order: int, binding: dict):
    """binding {ent: word}: statements in order (order 1 = ent_a first)."""
    e1, e2 = (it.ent_a, it.ent_b) if order == 1 else (it.ent_b, it.ent_a)
    return [(e1, binding[e1]), (e2, binding[e2])]


def _lab(it: Item, ent: str) -> tuple[str, str]:
    return ("ment_a", "attr_a") if ent == it.ent_a else ("ment_b", "attr_b")


def evidence(t: Text, it: Item, order: int, binding: dict, neg: bool = False, sep: str = ", so ", joiner: str = " and ") -> Text:
    (e1, w1), (e2, w2) = _ents_in_order(it, order, binding)
    it.stmt(t, e1, w1, *_lab(it, e1), neg=neg, cap=True)
    t.add(joiner)
    it.stmt(t, e2, w2, *_lab(it, e2), neg=neg, cap=joiner.endswith(". "))
    t.add(sep)
    return t


# attribute -> entity frames of the E1 family (FRAME PILOT, coordinator 2026-10-16; P0 = the original Phase-6 frame)
#   P0 'Fill in the blank: {A} and {B}, so the _ was X.\nAnswer: the'   P1 'Fill in the blank: {A}. {B}. The _ was X.\nAnswer: the'
#   P2 'Fill in the blank: {A} and {B}. The _ was X.\nAnswer: the'       P3 '{A} and {B}. Fill in the blank: The _ was X.\nAnswer: the'
A2E_FRAMES = {"P0": (F1_HEAD, " and ", ", so ", "", False), "P1": (F1_HEAD, ". ", ". ", "", True), "P2": (F1_HEAD, " and ", ". ", "", True),
              "P3": ("", " and ", ". ", F1_HEAD, True)}  # head, joiner, sep, mid, capitalised blank clause


def a2e_tail(t: Text, it: Item, trig: str, cap: bool = False) -> Text:
    bc = it.bc_text()
    if cap and bc[:1].isalpha():
        bc = bc[0].upper() + bc[1:]
    i = bc.index("_")
    t.add(bc[:i]).add("_", "blank").add(bc[i + 1:] + " ").add(trig, "q").add(".")
    t.add("\nAnswer:" + (f" {it.det}" if it.det else ""))
    return t


def e2a_tail(t: Text, it: Item, ent: str) -> Text:
    bc = it.bc_text()
    i = bc.index("_")
    t.add(bc[:i]).add(ent, "q").add(bc[i + 1:])
    return t


def build_level(it: Item, level: str, direction: str, order: int, para: Optional[dict] = None, binding: Optional[dict] = None,
                conflict: bool = False, frame: str = "P0"):
    """E1 / E1p / E2 / E0 / Dial A / R-site in the a2e or e2a frame -> (Text A, Text B, answer A, answer B, n_swap) or a reason.
    binding = {ent: word}, the binding the prompt states (default: the twin's ent_a <-> trig_a; Dial A / E0: the prior binding);
    conflict=True reverses the stated binding (Dial-A conflict). Answers always follow the stated binding: a2e asks for the
    entity of trig_a in A and of trig_b in B; e2a asks for the attribute of ent_a in A and of ent_b in B.
      e1     '<E1 stated> and <E2 stated>, so <blank clause>'
      e1p    the same with paraphrased attribute words (para = {trigger: paraphrase})
      e2     each entity stated as NOT the other trigger
      e0     'Of the <E1> and the <E2>, <blank clause>' (no evidence)
      rsite  a2e only; corrupted prompt = the statements with the two entity mentions swapped (same question, trig_a)"""
    stated = dict(binding or {it.ent_a: it.trig_a, it.ent_b: it.trig_b})
    if conflict:
        stated = {it.ent_a: stated[it.ent_b], it.ent_b: stated[it.ent_a]}
    inv = {w: e for e, w in stated.items()}
    other = {it.ent_a: stated[it.ent_b], it.ent_b: stated[it.ent_a]}
    if level == "e1p" and (para is None or it.trig_a not in para or it.trig_b not in para):
        return "P0_no_paraphrase"

    fh, joiner, sep, mid, capq = A2E_FRAMES[frame] if direction == "a2e" else ("", " and ", ", so ", "", False)

    def ev(t: Text) -> Text:
        if level == "e0":  # no statements: the frame's separator after the 'Of ...' phrase (P0 ', ')
            e1, e2 = (it.ent_a, it.ent_b) if order == 1 else (it.ent_b, it.ent_a)
            d = f"{it.det} " if it.det else ""
            t.add("Of " + d).add(e1, _lab(it, e1)[0]).add(" and " + d).add(e2, _lab(it, e2)[0]).add(", " if sep == ", so " else sep)
        elif level == "e1p":
            evidence(t, it, order, {e: para[w] for e, w in stated.items()}, sep=sep, joiner=joiner)
        elif level == "e2":
            evidence(t, it, order, other, neg=True, sep=sep, joiner=joiner)
        else:  # e1, aagr, acon, rsite
            evidence(t, it, order, stated, sep=sep, joiner=joiner)
        return t.add(mid) if direction == "a2e" else t

    head = fh if direction == "a2e" else ""
    if level == "rsite":
        assert direction == "a2e"
        ta = a2e_tail(ev(Text().add(head)), it, it.trig_a, capq)
        tb = a2e_tail(_rsite_evidence(Text().add(head), it, order, stated, sep=sep, joiner=joiner).add(mid), it, it.trig_a, capq)
        a = inv[it.trig_a]
        return ta, tb, a, (it.ent_b if a == it.ent_a else it.ent_a), 2
    if direction == "a2e":
        return (a2e_tail(ev(Text().add(head)), it, it.trig_a, capq), a2e_tail(ev(Text().add(head)), it, it.trig_b, capq),
                inv[it.trig_a], inv[it.trig_b], 1)
    return e2a_tail(ev(Text()), it, it.ent_a), e2a_tail(ev(Text()), it, it.ent_b), stated[it.ent_a], stated[it.ent_b], 1


def _rsite_evidence(t: Text, it: Item, order: int, stated: dict, sep: str = ", so ", joiner: str = " and ") -> Text:
    """R-site corrupted evidence: the statements of `order` with the two entity MENTIONS swapped (attributes stay in place)."""
    (e1, w1), (e2, w2) = _ents_in_order(it, order, stated)
    it.stmt(t, e2, w1, *_lab(it, e2), cap=True)
    t.add(joiner)
    it.stmt(t, e1, w2, *_lab(it, e1), cap=joiner.endswith(". "))
    t.add(sep)
    return t


def build_e3(it: Item, direction: str, tok, st: bool):
    """E3 = the original sentence: a2e = Phase-5 F1 frame (ext13_wino_ref.f1_prompt), e2a = Phase-3 option swap
    (ext7_wino.build_pair). Returns (pair dict, reason)."""
    if direction == "e2a":
        t = Twin(it.src_pair_id, "", "", it.sent_a, it.sent_b, it.ent_a, it.ent_b, it.ent_a, it.ent_b)
        p, why = p3_build_pair(t, tok, st)
        if p is None:
            return None, why
        if p.n_opt_tokens != 1:
            return None, "R2_option_not_one_token"
        ta, tb = Text().add(p.prompt_a), Text().add(p.prompt_b)
        _label_mentions(ta, it, p.prompt_a, upto=p.prompt_a.rfind(" " + it.ent_a))
        return make_pair(tok, st, ta, tb, it.trig_a, it.trig_b)
    ta, tb = Text(), Text()
    for t, sent, trig in ((ta, it.sent_a, it.trig_a), (tb, it.sent_b, it.trig_b)):
        s = sent.strip()
        t.add(F1_HEAD)
        j = s.index("_")
        k = s.rindex(trig)
        t.add(s[:j]).add("_", "blank").add(s[j + 1:k]).add(trig, "q").add(s[k + len(trig):])
        t.add("\nAnswer:" + (f" {it.det}" if it.det else ""))
    _label_mentions(ta, it, ta.s, lo=len(F1_HEAD), upto=ta.s.index("_"))
    return make_pair(tok, st, ta, tb, it.ent_a, it.ent_b)


def _label_mentions(t: Text, it: Item, s: str, lo: int = 0, upto: Optional[int] = None):
    for ent, lab in ((it.ent_a, "ment_a"), (it.ent_b, "ment_b")):
        for cand in (ent, ent.lower(), ent[:1].upper() + ent[1:]):
            m = re.compile(r"\b" + re.escape(cand) + r"\b").search(s, lo, upto if upto is not None and upto > lo else len(s))
            if m:
                t.spans[lab].append((m.start(), m.end()))
                break


def build_setc(it: Item, kind: str, order: int):
    """Set C (a2e), P3 layout (coordinator 2026-10-16): '<statement>. Fill in the blank: <original sentence>\nAnswer: the'. kind
    agree | conflict. Answers in the STATEMENT orientation (conflict: A -> ent_b). Returns (TA, TB, ans_a, ans_b, 1)."""
    twin_bind = {it.ent_a: it.trig_a, it.ent_b: it.trig_b}
    stated = twin_bind if kind == "agree" else {it.ent_a: it.trig_b, it.ent_b: it.trig_a}
    inv = {w: e for e, w in stated.items()}
    out = []
    for sent, trig in ((it.sent_a, it.trig_a), (it.sent_b, it.trig_b)):
        t = Text()
        evidence(t, it, order, stated, sep=". ")
        t.add(F1_HEAD)
        s = sent.strip()
        j = s.index("_")
        k = s.rindex(trig)
        t.add(s[:j]).add("_", "blank").add(s[j + 1:k]).add(trig, "q").add(s[k + len(trig):])
        t.add("\nAnswer:" + (f" {it.det}" if it.det else ""))
        out.append(t)
    return out[0], out[1], inv[it.trig_a], inv[it.trig_b], 1


E3P3_CONN = {"because", "but", "although", "though", "so", "since", "while", "as", "yet", "and", "therefore", "however", "whereas",
             "unless", "if", "when", "until", "before", "after", "or", "thus", "hence", "then", "once", "whenever", "that", "even",
             "still", "now", "consequently", "thereafter", "afterwards", "meanwhile", "instead", "obviously", "clearly", "apparently",
             "unfortunately", "luckily", "fortunately", "sadly", "surprisingly", "ultimately", "eventually", "finally", "naturally",
             "where", "why", "which", "who", "whose", "lest", "despite", "for"}


def e3p3_parts(it: Item) -> tuple[Optional[dict], str]:
    """E3 content in the P3 frame: event clause = the sentence before the blank clause with the connective word(s) that introduce
    the blank clause dropped (at most two words from E3P3_CONN, plus punctuation), final punctuation normalised to '.'."""
    ev = re.sub(r"\s+", " ", it.head[:it.bc_start]).rstrip()
    w0 = re.sub(r"[^a-z]", "", ev.split()[-1].lower()) if ev.split() else ""
    if w0 in COMPLEMENT_VERBS:
        return None, "E3P3_complement_clause"  # 'I thought the _ was': the blank clause is the object of the verb
    dropped = []
    core = ev
    for _ in range(2):
        core = core.rstrip(" ,;:-")
        m = re.search(r"(?:^|\s)([A-Za-z]+)$", core)
        if m and m.group(1).lower() in E3P3_CONN:
            dropped.insert(0, m.group(1))
            core = core[:m.start(1)].rstrip()
        else:
            break
    core = core.rstrip(" ,;:-")
    if dropped and dropped[-1].lower() == "that" and not (len(dropped) == 2 and dropped[0].lower() in ("so", "now", "given")):
        return None, "E3P3_complement_clause"  # 'realized that the _ was'
    w1 = re.sub(r"[^a-z]", "", core.split()[-1].lower()) if core.split() else ""
    if w1 in COMPLEMENT_VERBS:
        return None, "E3P3_complement_clause"
    if not core or len(core.split()) < 3:
        return None, "E3P3_no_event_clause"
    if "_" in core:
        return None, "E3P3_blank_before_connective"
    if not dropped and not ev.endswith((".", "!", "?", ",", ";", ":")):
        return None, "E3P3_text_before_blank_np"  # e.g. "the dog's _ was": something other than a connective / punctuation precedes the NP
    if not core.endswith((".", "!", "?")):
        core += "."
    return {"event": core, "dropped": " ".join(dropped).lower() if dropped else ("(sentence break)" if ev.rstrip().endswith((".", "!", "?")) else "(comma only)"),
            "ev_raw": ev}, "ok"


def build_e3p3(it: Item):
    """'{event clause}. Fill in the blank: The _ was X.\nAnswer: the' (both twins share the event clause; STR = the trigger)."""
    parts, why = e3p3_parts(it)
    if parts is None:
        return why
    out = []
    for trig in (it.trig_a, it.trig_b):
        t = Text()
        _label_mentions_text(t, it, parts["event"])
        t.add(parts["event"] + " ")
        t.add(F1_HEAD)
        a2e_tail(t, it, trig, cap=True)
        out.append(t)
    return out[0], out[1], it.ent_a, it.ent_b, 1, parts


def _label_mentions_text(t: Text, it: Item, s: str):
    base = len(t.s)
    for ent, lab in ((it.ent_a, "ment_a"), (it.ent_b, "ment_b")):
        for cand in (ent, ent.lower(), ent[:1].upper() + ent[1:]):
            m = re.compile(r"\b" + re.escape(cand) + r"\b").search(s)
            if m:
                t.spans[lab].append((base + m.start(), base + m.end()))
                break


# ---------------------------------------------------------------------------------------------------------------
# completeness (Phase-5 F2-cut guard)
# ---------------------------------------------------------------------------------------------------------------
def completeness(cell: str, it: Item, row: dict, para: Optional[dict] = None) -> dict[str, bool]:
    """String-level checks that every level contains what its definition says (RESEARCH_PLAN guard table)."""
    ws = lambda x: re.sub(r"\s+", " ", x)
    pa, pb = ws(row["prompt_a"]), ws(row["prompt_b"])
    level, direction = cell.split("_")[0], cell.split("_")[-1]
    chk = {}
    if it.source == "wino":
        bc = it.bc_text()
        filled_a = bc.replace("_", it.ent_a, 1) + " " + it.trig_a
        filled_b = bc.replace("_", it.ent_b, 1) + " " + it.trig_b
    else:
        bc, filled_a, filled_b = it.bc_text(), None, None
    if cell == "e3p3_a2e":
        parts, _ = e3p3_parts(it)
        ev = pa[:pa.index(F1_HEAD)] if F1_HEAD in pa else ""
        evl = ev.lower()
        chk["both_entities_in_event_clause"] = bool(re.search(r"\b%s\b" % re.escape(it.ent_a.lower()), evl)) and \
            bool(re.search(r"\b%s\b" % re.escape(it.ent_b.lower()), evl))
        chk["whole_blank_clause_a"] = (it.bc_text() + " " + it.trig_a + ".").lower() in pa.lower()
        chk["whole_blank_clause_b"] = (it.bc_text() + " " + it.trig_b + ".").lower() in pb.lower()
        # only the connective word(s) and punctuation are dropped from the sentence before the blank clause
        norm = lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).split()
        raw = norm(parts["ev_raw"])
        drop = norm(parts["dropped"]) if not parts["dropped"].startswith("(") else []
        chk["only_connective_dropped"] = raw == norm(parts["event"]) + drop
        # string-wise reconstruction: event + connective + '<det> _ <rest>' + trigger == the original sentence (words, case-insensitive)
        wnorm = lambda x: re.sub(r"[^a-z0-9_']+", " ", x.lower()).split()
        for x, sent, trig in (("a", it.sent_a, it.trig_a), ("b", it.sent_b, it.trig_b)):
            chk[f"reconstructs_sentence_{x}"] = wnorm(sent) == wnorm(parts["event"]) + drop + wnorm(it.bc_text()) + wnorm(trig)
        np_ = it.bc_text()[:it.bc_text().index("_")].strip().lower()
        chk["blank_np_is_det_only"] = np_ in ("",) + DET_OK + ("a", "an", "this", "that")
    if cell == "e3_a2e" or cell.startswith("sc_"):
        chk["whole_sentence_a"] = ws(it.sent_a.strip()) in pa
        chk["whole_sentence_b"] = ws(it.sent_b.strip()) in pb
    if cell.startswith("sc_"):
        # string-wise reconstruction: prompt == '<statement 1> and <statement 2>. Fill in the blank: <sentence>\nAnswer: <det>'
        kind = cell.split("_")[1]
        stated = {it.ent_a: it.trig_a, it.ent_b: it.trig_b} if kind == "agree" else {it.ent_a: it.trig_b, it.ent_b: it.trig_a}
        o = int(row.get("order", 1))
        (e1, w1), (e2, w2) = _ents_in_order(it, o, stated)
        st_ = lambda e, w: (f"{it.det} " if it.det else "") + f"{e} {it.cop}" + (f" {it.adv}" if it.adv else "") + f" {w}"
        cap = lambda x: x[0].upper() + x[1:]
        exp = cap(st_(e1, w1)) + " and " + st_(e2, w2) + ". " + F1_HEAD
        ans = "\nAnswer:" + (f" {it.det}" if it.det else "")
        chk["reconstructs_prompt_a"] = pa == ws(exp + it.sent_a.strip() + ans)
        chk["reconstructs_prompt_b"] = pb == ws(exp + it.sent_b.strip() + ans)
    if level == "rsite":
        # B == A with exactly the two entity mentions of the evidence exchanged (string-wise), the question unchanged
        q = pa.rfind(F1_HEAD) if F1_HEAD in pa and not pa.startswith(F1_HEAD) else pa.lower().rfind(it.bc_text().lower())
        eva, evb = pa[:q], pb[:q]
        sw = re.sub(r"\b(%s|%s)\b" % (re.escape(it.ent_a), re.escape(it.ent_b)), lambda m: it.ent_b if m.group(0) == it.ent_a else it.ent_a, eva)
        chk["evidence_mentions_swapped"] = sw == evb and eva != evb
        chk["question_identical"] = pa[q:] == pb[q:]
    elif cell == "e3_e2a":
        for x, p, ent, trig, sent in (("a", pa, it.ent_a, it.trig_a, it.sent_a), ("b", pb, it.ent_b, it.trig_b, it.sent_b)):
            filled = ws(sent.replace("_", ent, 1).strip())
            chk[f"prompt_plus_trigger_is_sentence_{x}"] = filled.startswith(p + " " + trig) and \
                not re.search(r"[A-Za-z]", filled[len(p + " " + trig):])
    if level in ("e1", "e1p", "e2", "rsite", "aagr", "acon", "e0") and it.source == "wino":
        lo = lambda x: x.lower()  # P1-P3 frames capitalise the blank clause's determiner
        if direction == "a2e":
            chk["whole_blank_clause_a"] = lo(bc + " " + it.trig_a + ".") in lo(pa)
            chk["whole_blank_clause_b"] = lo(bc + " " + (it.trig_a if level == "rsite" else it.trig_b) + ".") in lo(pb)
        else:
            chk["whole_blank_clause_a"] = pa.endswith(bc.replace("_", it.ent_a, 1))
            chk["whole_blank_clause_b"] = pb.endswith(bc.replace("_", it.ent_b, 1))
    ev = ""
    if it.source == "wino" and level in ("e1", "e1p", "e2", "rsite"):
        j = pa.lower().rfind(it.bc_text().lower()) if direction == "a2e" else pa.lower().rfind(it.bc_text().lower().replace("_", it.ent_a.lower(), 1))
        ev = pa[:j] if j > 0 else ""
    if level == "e1":
        chk["both_filled_clauses_in_evidence"] = filled_a.lower()[len(it.det):] in ev.lower() and filled_b.lower()[len(it.det):] in ev.lower()
    if level == "e1p":
        evw = set(re.findall(r"[a-z]+", ev.lower()))
        chk["no_verbatim_trigger_in_evidence"] = it.trig_a.lower() not in evw and it.trig_b.lower() not in evw
        chk["paraphrases_in_evidence"] = bool(para) and para[it.trig_a] in ev and para[it.trig_b] in ev
    if level == "e2":
        chk["negated_other_trigger"] = (f"not{(' ' + it.adv) if it.adv else ''} {it.trig_b}" in ev and
                                        f"not{(' ' + it.adv) if it.adv else ''} {it.trig_a}" in ev)
    if level == "e0":
        d = f"{it.det} " if it.det else ""
        body = pa[len(F1_HEAD):] if pa.startswith(F1_HEAD) else pa
        chk["no_evidence"] = any(body.startswith(f"Of {d}{x} and {d}{y}") for x, y in ((it.ent_a, it.ent_b), (it.ent_b, it.ent_a)))
    return chk


# ---------------------------------------------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------------------------------------------
def probe_key(names: bool, word_a: str, word_b: str, ans_a: str, ans_b: str) -> str:
    """Phase-5 probe key (moetrace/ext13_p5.probe_key)."""
    return json.dumps([bool(names), word_a, word_b, ans_a, ans_b])


def seeded_order(keys, seed) -> dict:
    ks = sorted(keys)
    random.Random(seed).shuffle(ks)
    return {k: i for i, k in enumerate(ks)}


# ---------------------------------------------------------------------------------------------------------------
# E1p paraphrases: hand-checked list (data/dial/paraphrase_hand.json) + WordNet fallback (flagged)
# ---------------------------------------------------------------------------------------------------------------
_HAND = None


def hand_paraphrases() -> dict:
    global _HAND
    if _HAND is None:
        d = json.load(open(f"{DIAL}/paraphrase_hand.json"))
        _HAND = {k: v for k, v in d.items() if not k.startswith("_")}
    return _HAND


def share_stem(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    wa, wb = set(a.split()), set(b.split())
    for x in wa:
        for y in wb:
            if x == y or x[:4] == y[:4] or (len(x) >= 4 and x in y) or (len(y) >= 4 and y in x):
                return True
    return False


def wordnet_paraphrase(word: str) -> Optional[str]:
    """Most frequent single-word adjective synonym / similar-to lemma of the word's first adjective synsets (WordNet)."""
    from nltk.corpus import wordnet as wn
    cands = collections.Counter()
    syns = [s for s in wn.synsets(word) if s.pos() in ("a", "s")][:2]
    for s in syns:
        for rel in [s] + s.similar_tos():
            for l in rel.lemmas():
                n = l.name()
                if "_" in n or "-" in n or not n.isalpha() or not n.islower():
                    continue
                if share_stem(n, word):
                    continue
                cands[n] += l.count() + (2 if rel is s else 0)
    if not cands:
        return None
    return cands.most_common(1)[0][0]


def paraphrase_for(it: "Item", allow_wordnet: bool = False) -> tuple[Optional[dict], str]:
    """{trig_a: para_a, trig_b: para_b} and the source ('hand' if both from the hand list, else 'wordnet'), or (None, reason).
    Default: hand list only (WordNet fallback candidates were often wrong in a look at 40 items, e.g. rusty -> chromatic)."""
    hand = hand_paraphrases()
    out, src = {}, "hand"
    for w in (it.trig_a, it.trig_b):
        k = w.lower()
        if k in hand:
            p = hand[k]
            if p is None:
                return None, "P1_hand_list_no_paraphrase"
        else:
            if not allow_wordnet:
                return None, "P2_not_in_hand_list"
            p = wordnet_paraphrase(k)
            src = "wordnet"
            if p is None:
                return None, "P2_no_wordnet_paraphrase"
        out[w] = p
    for w, p in out.items():
        if share_stem(p, it.trig_a) or share_stem(p, it.trig_b):
            return None, "P3_paraphrase_shares_stem_with_a_trigger"
    if share_stem(out[it.trig_a], out[it.trig_b]):
        return None, "P4_paraphrases_share_stem"
    return out, src


_ANT = None


def antonym_pair(w1: str, w2: str) -> bool:
    """True if WordNet lists the two words as direct antonyms (any adjective / adverb / verb lemma) or the LRE adjective_antonym
    data has the pair (either direction). Recorded as a stratum (E2 negation is a clean inference for antonym pairs)."""
    global _ANT
    if _ANT is None:
        _ANT = set()
        p = f"{DIAL}/lre/adj_antonym.json"
        if os.path.exists(p):
            for s in json.load(open(p))["samples"]:
                _ANT.add((s["subject"].lower(), s["object"].lower()))
                _ANT.add((s["object"].lower(), s["subject"].lower()))
    a, b = w1.lower(), w2.lower()
    if (a, b) in _ANT:
        return True
    from nltk.corpus import wordnet as wn
    for s in wn.synsets(a):
        for l in s.lemmas():
            if l.name().lower() == a and any(x.name().lower() == b for x in l.antonyms()):
                return True
    return False


# ---------------------------------------------------------------------------------------------------------------
# Ladder v2 (l_ant, l_prop, l_bind, l_ind): single-frame STR pairs
# ---------------------------------------------------------------------------------------------------------------
LRE_DIR = f"{DIAL}/lre"
ANT_TEMPLATES = ["The opposite of {} is", "The antonym of {} is", "Q: What is the opposite of {}?\nA:"]
# LRE commonsense relations whose subjects are single words; templates keep the subject inside the prompt and never list the
# candidate values ({A} = 'A' / 'An' chosen per subject; a pair whose articles differ fails R2 and is dropped)
PROP_TEMPLATES = {
    "fruit_outside_color": ["On the outside, {} are", "The color of the skin of {} is"],
    "fruit_inside_color": ["On the inside, {} are", "What color are {} on the inside? They are"],
    "substance_phase": ["At room temperature, {} is a", "What is the phase of matter of {} at room temperature? It is a"],
    "object_superclass": ["{A} {} is in the category of a", "{A} {} is a kind of"],
    "work_location": ["{A} {} typically works at a", "You can usually find a {} working in a"],
}
BIND_OBJECTS = ["apple", "key", "coin", "ball", "book", "pen", "ring", "shoe", "cup", "toy", "sock", "watch", "phone", "card", "hat",
                "bell", "doll", "egg", "map", "lamp", "spoon", "brush", "comb", "candle", "button", "stone", "shell", "flower", "letter",
                "ticket", "rope", "knife", "fork", "plate", "bottle", "glove", "scarf", "wallet", "clock", "mirror", "pencil", "towel",
                "cookie", "banana", "orange", "lemon", "carrot", "feather", "marble", "battery"]
BIND_COLOURS = ["red", "blue", "green", "yellow", "white", "black", "brown", "pink", "purple", "orange", "gray", "silver", "gold"]
IND_WORDS = ["dog", "table", "cup", "river", "chair", "cloud", "horse", "window", "garden", "bread", "mountain", "pencil", "doctor",
             "village", "camera", "forest", "bottle", "piano", "rabbit", "ocean", "candle", "jacket", "bridge", "kitchen", "letter",
             "island", "button", "wagon", "basket", "ladder", "planet", "pillow", "rocket", "tiger", "violin", "wallet", "hammer",
             "lemon", "castle", "carpet", "mirror", "engine", "farmer", "glass", "monkey", "needle", "orange", "paper", "queen",
             "school", "sugar", "teacher", "umbrella", "valley", "winter", "yellow", "anchor", "blanket", "circle", "desert",
             "feather", "guitar", "helmet", "jungle", "kettle", "lantern", "meadow", "napkin", "oven", "parrot", "quilt", "saddle",
             "tunnel", "vase", "whistle", "zebra", "bucket", "cabin", "dolphin", "falcon"]


def lre(name: str) -> dict:
    """LRE relation JSON (data/dial/lre, downloaded from github.com/evandez/relations; read as data only)."""
    return json.load(open(f"{LRE_DIR}/{name}.json"))


def _fill(tpl: str, w: str, lab: str = "q") -> Text:
    art = "An" if w[:1].lower() in "aeiou" else "A"
    tpl = tpl.replace("{A}", art)
    pre, post = tpl.split("{}")
    return Text().add(pre).add(w, lab).add(post)


def build_l_ant(tok, st) -> tuple[list[dict], collections.Counter]:
    rows, why = [], collections.Counter()
    seen, items = set(), []
    cur = json.load(open(f"{DIAL}/curated_antonyms.json"))["pairs"]
    for src, pairs in (("lre", [(s["subject"], s["object"]) for s in lre("adj_antonym")["samples"]]), ("curated", cur)):
        for a, b in pairs:
            a, b = a.strip().lower(), b.strip().lower()
            if not (a.isalpha() and b.isalpha()) or a == b:
                why[f"item:{src}:not_single_words"] += 1
                continue
            k = tuple(sorted((a, b)))
            if k in seen:
                why[f"item:{src}:R4_duplicate_pair"] += 1
                continue
            seen.add(k)
            items.append((k, src))
    for i, ((a, b), src) in enumerate(items):
        for t, tpl in enumerate(ANT_TEMPLATES):
            p, w = make_pair(tok, st, _fill(tpl, a), _fill(tpl, b), b, a)
            why[w] += 1
            if p is None:
                continue
            pos = p.pop("positions")
            rows.append({"pair_id": f"ant|{a}|{b}|t{t}", "cell": "l_ant", "src_pair_id": f"ant|{a}|{b}", "item_idx": i, "template": t,
                         "pair_idx": 4 * i + t, "order": 1, "answer_a": b, "answer_b": a, "word_a": a, "word_b": b, **p,
                         **pos_columns(pos), "relation": "antonym", "source": src})
    return rows, why


def build_l_prop(tok, st, per_subject: int = 4, seed: int = 6) -> tuple[list[dict], collections.Counter]:
    """Pairs of subjects of one LRE relation with different single-token values; STR = the subject. Subjects are split into
    discovery / validation halves per relation (random.Random(seed)), pairs only within a half; each subject in <= per_subject
    pairs per template."""
    rows, why = [], collections.Counter()
    idx = 0
    cur = json.load(open(f"{DIAL}/curated_props.json"))["relations"]
    rels = [(rel, tpls, [(s["subject"].strip(), s["object"].strip()) for s in lre(rel)["samples"]], "lre") for rel, tpls in PROP_TEMPLATES.items()]
    rels += [(rel, d["templates"], list(d["samples"].items()), "curated") for rel, d in cur.items()]
    for rel, tpls, sam, src in rels:
        sam = [(s, o) for s, o in sam if " " not in s and " " not in o]
        why[f"{rel}:subjects_single_word"] += len(sam)
        subs = sorted({s for s, _ in sam})
        val = dict(sam)
        rng = random.Random(f"{seed}-{rel}")
        sh = list(subs)
        rng.shuffle(sh)
        half = {s: ("discovery" if i < len(sh) // 2 else "validation") for i, s in enumerate(sh)}
        for t, tpl in enumerate(tpls):
            use = collections.Counter()
            cand = [(a, b) for i, a in enumerate(sh) for b in sh[i + 1:] if val[a] != val[b] and half[a] == half[b]]
            rng.shuffle(cand)
            for a, b in cand:
                if use[a] >= per_subject or use[b] >= per_subject:
                    continue
                p, w = make_pair(tok, st, _fill(tpl, a), _fill(tpl, b), val[a], val[b])
                why[f"{rel}:{w}"] += 1
                if p is None:
                    continue
                use[a] += 1
                use[b] += 1
                pos = p.pop("positions")
                rows.append({"pair_id": f"prop|{rel}|{a}|{b}|t{t}", "cell": "l_prop", "src_pair_id": f"prop|{rel}|{a}|{b}", "item_idx": idx,
                             "template": t, "pair_idx": idx, "order": 1, "answer_a": val[a], "answer_b": val[b], "word_a": a, "word_b": b,
                             **p, **pos_columns(pos), "relation": rel, "split": half[a], "source": src})
                idx += 1
    return rows, why


def _two_item_rows(cell, n_items, sample, build, tok, st, seed):
    rows, why, seen = [], collections.Counter(), set()
    rng = random.Random(seed)
    i = 0
    tries = 0
    while i < n_items and tries < 50 * n_items:
        tries += 1
        it = sample(rng)
        key = tuple(it)
        if key in seen:
            continue
        seen.add(key)
        got = []
        for o in (1, 2):
            ta, tb, aa, ab = build(it, o)
            p, w = make_pair(tok, st, ta, tb, aa, ab)
            why[w] += 1
            if p is None:
                break
            pos = p.pop("positions")
            got.append({"pair_id": f"{cell}|{'|'.join(it)}|o{o}", "cell": cell, "src_pair_id": f"{cell}|{'|'.join(it)}", "item_idx": i,
                        "pair_idx": 2 * i + o - 1, "order": o, "answer_a": aa, "answer_b": ab, "word_a": it[0], "word_b": it[1], **p,
                        **pos_columns(pos), "source": "synthetic"})
        if len(got) != 2:
            why["item:not_both_orders_built"] += 1
            continue
        rows.extend(got)
        i += 1
    return rows, why


def line_start_words(words, toks) -> list[str]:
    """Words that are one token at a line start (no leading space) in EVERY given tokenizer (l_ind keys in the I1 / I2 formats)."""
    out = []
    for w in words:
        ok = True
        for tok, st in toks:
            a = tok("x:\n" + w + ":", add_special_tokens=False)["input_ids"]
            b = tok("x:\n", add_special_tokens=False)["input_ids"]
            ok &= len(a) == len(b) + 2
        if ok:
            out.append(w)
    return out


def single_token_words(words, toks) -> list[str]:
    """Words that are one leading-space continuation token after a neutral prompt in EVERY given (tokenizer, special tokens)."""
    out = []
    for w in words:
        ok = True
        for tok, st in toks:
            for pr in ("The word is", "Pairs: x ="):
                ids = list(tok(pr, add_special_tokens=st)["input_ids"])
                if _single_token_continuation(tok, ids, pr, w, st) is None:
                    ok = False
        if ok:
            out.append(w)
    return out


def build_l_bind(tok, st, n_items: int = 1600, seed: int = 7, objects=None, colours=None):
    """'The apple is in the red box and the key is in the blue box. The red box contains the' -> ' apple' (STR = the colour in the
    question). Item (o1, o2, c1, c2); A asks c1 -> o1, B asks c2 -> o2; order 1 = (o1, c1) stated first. 1,600 items (Qwen3 passes
    ~10 % of the items in both orders: a strong first-mention bias), objects exclude colour words."""
    objects, colours = objects or BIND_OBJECTS, colours or BIND_COLOURS
    objects = [o for o in objects if o not in colours]  # 'orange' is a colour and a fruit

    def sample(rng):
        o1, o2 = rng.sample(objects, 2)
        c1, c2 = rng.sample(colours, 2)
        return (o1, o2, c1, c2)

    def build(it, order):
        o1, o2, c1, c2 = it
        st_ = [(o1, c1, "ment_a", "attr_a"), (o2, c2, "ment_b", "attr_b")]
        if order == 2:
            st_ = st_[::-1]
        out = []
        for cq in (c1, c2):
            t = Text()
            for j, (o, c, lo, lc) in enumerate(st_):
                t.add("The " if j == 0 else " and the ").add(o, lo).add(" is in the ").add(c, lc).add(" box")
            t.add(". The ").add(cq, "q").add(" box contains the")
            out.append(t)
        return out[0], out[1], o1, o2
    return _two_item_rows("l_bind", n_items, sample, build, tok, st, seed)


def build_l_ind(tok, st, n_items: int = 800, seed: int = 8, words=None, frame: str = "I0", keys=None):
    """Induction on unrelated word pairs, STR = the key in the query. frame (IND PILOT): I0 'Pairs: dog = table, cup = river. dog =',
    I1 '\\ndog: table\\ncup: river\\ndog:', I2 = I1 + two distractor pairs after the first two (moetrace/ext14_data.ind_text).
    Item (k1, k2, v1, v2[, k3, v3, k4, v4]); A queries k1 -> v1, B queries k2 -> v2; order 1 = (k1, v1) first. I1 / I2 keys come from
    `keys` (single tokens at a line start in both families)."""
    words = words or IND_WORDS
    keys = keys or words

    def sample(rng):
        ks = rng.sample(keys, 4 if frame == "I2" else 2)
        vs = rng.sample([w for w in words if w not in ks], len(ks))
        if frame == "I2":
            return (ks[0], ks[1], vs[0], vs[1], ks[2], vs[2], ks[3], vs[3])
        return (ks[0], ks[1], vs[0], vs[1])

    def build(it, order):
        pairs = [(it[0], it[2]), (it[1], it[3])] + ([(it[4], it[5]), (it[6], it[7])] if frame == "I2" else [])
        if frame == "I0":
            pairs += [("x", "y"), ("z", "w")]
        return ind_text(frame, pairs, order, it[0]), ind_text(frame, pairs, order, it[1]), it[2], it[3]
    return _two_item_rows("l_ind", n_items, sample, build, tok, st, seed)


# complementary (non-gradable) antonym pairs: negating one entails the other (coordinator QA 2026-10-16); everything else counts as
# gradable / contrary (full / empty included), where 'not too big' does not entail 'too small'
COMPLEMENTARY = {frozenset(p) for p in [
    ("open", "closed"), ("open", "shut"), ("alive", "dead"), ("present", "absent"), ("awake", "asleep"), ("on", "off"), ("true", "false"),
    ("male", "female"), ("married", "single"), ("married", "unmarried"), ("guilty", "innocent"), ("visible", "invisible"),
    ("possible", "impossible"), ("legal", "illegal"), ("correct", "incorrect"), ("right", "wrong"), ("lost", "found"),
    ("inside", "outside"), ("in", "out"), ("broken", "working"), ("broken", "fixed"), ("pass", "fail"), ("win", "lose"),
    ("available", "unavailable"), ("finite", "infinite"), ("real", "fake"), ("real", "imaginary"), ("hidden", "visible"),
    ("even", "odd"), ("armed", "unarmed"), ("employed", "unemployed"), ("hired", "fired"), ("pregnant", "barren"),
    ("covered", "exposed"), ("covered", "uncovered"), ("cooked", "raw"), ("raw", "baked"), ("locked", "unlocked"),
    ("required", "optional"), ("needed", "unneeded")]}


def antonym_type(w1: str, w2: str) -> str:
    return "complementary" if frozenset((w1.lower(), w2.lower())) in COMPLEMENTARY else "gradable"


# l_ind formats (IND PILOT, coordinator 2026-10-16). pairs = [(k1, v1), (k2, v2), (k3, v3), (k4, v4)]; the STR swaps the query
# between k1 and k2; order 2 lists (k2, v2) before (k1, v1); distractor pairs (k3, k4) follow.
#   I0 'Pairs: a = b, c = d. a ='             I1 '\na: b\nc: d\na:'        I2 '\na: b\nc: d\ne: f\ng: h\na:'
#      (I1 / I2 start with a newline so that the first key has the same line-start token as the query; keys must be single tokens in the
#      line-start form, values in the leading-space form)
#   I3 'List: a b c d e f a' (flat sequence, 3 pairs; 'List:' gives every word its leading-space token so the repeated key is the same token;
#      the query key is the FINAL token, so I3 violates R2 (STR at the readout position): behaviour reference only)
#   I4 'The code for a is b. The code for c is d. The code for a is'  (reference)
def ind_text(frame: str, pairs: list, order: int, query: str) -> "Text":
    p = list(pairs)
    if order == 2:
        p[0], p[1] = p[1], p[0]
    lab = {pairs[0][0]: ("ment_a", "attr_a"), pairs[1][0]: ("ment_b", "attr_b")}
    t = Text()
    if frame == "I0":
        t.add("Pairs:")
        for j, (k, v) in enumerate(p[:2]):
            lk, lv = lab[k]
            t.add(" " if j == 0 else ", ").add(k, lk).add(" = ").add(v, lv)
        t.add(". ").add(query, "q").add(" =")
    elif frame in ("I1", "I2"):
        t.add("\n")  # every key at a line start (same token for the listed and the queried key, also for the first line)
        for k, v in p[:2] if frame == "I1" else p[:4]:
            lk, lv = lab.get(k, (None, None))
            t.add(k, lk).add(": ").add(v, lv).add("\n")
        t.add(query, "q").add(":")
    elif frame == "I3":
        t.add("List:")
        for k, v in p[:3]:
            lk, lv = lab.get(k, (None, None))
            t.add(" ").add(k, lk).add(" ").add(v, lv)
        t.add(" ").add(query, "q")
    elif frame == "I4":
        for j, (k, v) in enumerate(p[:2]):
            lk, lv = lab[k]
            t.add(("" if j == 0 else " ") + "The code for ").add(k, lk).add(" is ").add(v, lv).add(".")
        t.add(" The code for ").add(query, "q").add(" is")
    else:
        raise ValueError(frame)
    return t
