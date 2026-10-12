"""ext13 P5-3: verb-swap controls on WinoGrande NAME items as seed templates x IOI name pairs (user decision 2026-10-11,
option 2: curated seed templates, many name pairs; semi-synthetic, addresses memorization of the original items).

Seed = a WinoGrande name twin whose first-mentioned name is directly followed by a past-tense verb V and whose second
name follows within three words (interpersonal "X V Y ..."), in one of the two directions:
  ent2attr  Phase-3 option swap (the twins differ only in the sentence-final trigger; fill the blank with each twin's
            answer, predict the trigger): ext7_wino.build_pair
  attr2ent  Phase-5 F2R (the twins differ before the blank and are identical from it on; whole sentence with the blank
            -> "one of them", the words after the blank restated, predict the entity): ext13_wino_ref.build_ref_pair "F2R"
            (2026-10-11: was the F2 cut at the blank, which drops the deciding words after it; all 13 seeds were affected)
The two options become slots {X} (option1) and {Y} (option2). A filled twin replaces them with two IOI names (Wang et al.
2023) of the template's gender (from its pronouns) that are single tokens in both Qwen3 and Mixtral, and is passed to the
unchanged pair builder of its direction, so every Phase-3 / Phase-5 rule (W2-W5, R2-R5) applies per tokenizer.
Verb variants (curated per seed in data/wino_ref/verb_templates.json, each a list of (old, new) substring edits applied
to both template sentences):
  orig       the original verb phrase (new names only)
  neutral    a more general verb, answer preserved (bought Y dinner -> got Y dinner)
  unrelated  a verb from another domain under which the original answer no longer follows (memorization / shortcut
             check: if the original answer still wins, the model does not read the verb)
"""
from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import Optional

from .ext7_wino import NAME_RE, Twin, build_pair, load_twins, split_final_word
from .ext13_wino_ref import build_ref_pair

# IOI_NAMES (ext7_controls) that are one token in Qwen3 and Mixtral, after a space and at sentence start; Jamie and
# Kelly dropped (gender-ambiguous). Checked 2026-10-11.
NAMES_M = ["Michael", "Christopher", "Matthew", "Daniel", "David", "James", "Robert", "John", "Joseph", "Andrew", "Ryan",
           "Jason", "Justin", "William", "Jonathan", "Brian", "Anthony", "Eric", "Adam", "Kevin", "Steven", "Thomas",
           "Kyle", "Richard", "Jeremy", "Mark", "Aaron", "Charles", "Jacob", "Stephen", "Patrick", "Sean", "Paul",
           "Tyler", "Scott", "Jose", "Alexander"]
NAMES_F = ["Jessica", "Jennifer", "Sarah", "Elizabeth", "Rachel", "Laura", "Amy", "Michelle", "Emily", "Mary", "Lisa"]
PRON_M = {"he", "him", "his", "himself"}
PRON_F = {"she", "her", "hers", "herself"}
IRREG = set(("gave took got made bought told showed ate won went saw met sent brought taught held kept left lost paid sold "
             "spent stole threw wore wrote drove beat broke caught chose fought found forgave hid hit hurt knew led lent let "
             "lit meant read rode ran said sought shook shot sang sat slept spoke stood struck swore swam thought understood "
             "woke bit drew drank flew forgot froze grew heard hung laid lay bent bet built dealt dug fed felt fled forbade "
             "overtook").split())


def gender(*sents: str) -> str:
    """m / f from the template's pronouns, n if none, x if both."""
    w = {x.lower() for s in sents for x in re.findall(r"[A-Za-z]+", s)}
    m, f = bool(w & PRON_M), bool(w & PRON_F)
    return "x" if m and f else "m" if m else "f" if f else "n"


def first_verb(text: str, a: str, b: str) -> Optional[tuple[str, int]]:
    """(past-tense verb right after the first-mentioned of a / b, its char offset)."""
    i, j = text.find(a), text.find(b)
    if i < 0 or j < 0:
        return None
    n1, i1 = (a, i) if i < j else (b, j)
    m = re.match(r"\s+([a-z]+)\b", text[i1 + len(n1):])
    if not m:
        return None
    w = m.group(1)
    return (w, i1 + len(n1) + m.start(1)) if (w.endswith("ed") or w in IRREG) else None


def interpersonal_verb(text: str, a: str, b: str) -> Optional[str]:
    """V if text has "<first name> V (up to 3 words) <second name>" with no , . ; in between."""
    v = first_verb(text, a, b)
    if not v:
        return None
    other = b if text.find(a) < text.find(b) else a
    m = re.match(r"((?:\s+[A-Za-z']+){0,3}?)\s+" + re.escape(other) + r"\b", text[v[1] + len(v[0]):])
    if not m or re.search(r"[,.;]", m.group(0)):
        return None
    return v[0]


def direction(t: Twin) -> Optional[str]:
    ha, wa = split_final_word(t.sent_a)
    hb, wb = split_final_word(t.sent_b)
    if ha == hb and wa != wb and "_" in ha:
        return "ent2attr"
    ia, ib = t.sent_a.index("_"), t.sent_b.index("_")
    if t.sent_a[:ia] != t.sent_b[:ib] and t.sent_a[ia:] == t.sent_b[ib:]:
        return "attr2ent"
    return None


def _slot(s: str, o1: str, o2: str) -> str:
    s = re.sub(r"\b" + re.escape(o1) + r"\b", "{X}", s)
    return re.sub(r"\b" + re.escape(o2) + r"\b", "{Y}", s)


def seed_pool(split: str = "train_xl") -> list[dict]:
    """Interpersonal name twins in either direction, as slot templates."""
    twins, _ = load_twins(split)
    out = []
    for t in twins:
        if not (NAME_RE.match(t.opt1) and NAME_RE.match(t.opt2)) or t.opt1 in t.opt2 or t.opt2 in t.opt1:
            continue
        d = direction(t)
        if d is None:
            continue
        v = interpersonal_verb(t.sent_a, t.opt1, t.opt2)
        if not v or interpersonal_verb(t.sent_b, t.opt1, t.opt2) != v:
            continue
        ta, tb = _slot(t.sent_a, t.opt1, t.opt2), _slot(t.sent_b, t.opt1, t.opt2)
        if "{X}" not in ta or "{Y}" not in ta:
            continue
        out.append(dict(seed=t.pair_id, dir=d, verb=v, gender=gender(t.sent_a, t.sent_b), tmpl_a=ta, tmpl_b=tb,
                        ans_a="X" if t.ans_a == t.opt1 else "Y", ans_b="X" if t.ans_b == t.opt1 else "Y",
                        q_a=t.q_a, q_b=t.q_b, sent_a=t.sent_a, sent_b=t.sent_b))
    return out


def apply_edits(s: str, edits: list) -> Optional[str]:
    """Apply the (old, new) edits present in s (comparative seeds list both twins' strings); None if none applies."""
    if not edits:
        return s
    n = 0
    for old, new in edits:
        if old in s:
            s, n = s.replace(old, new, 1), n + 1
    return s if n else None


def fill(seed: dict, x: str, y: str, edits: list = (), tag: str = "") -> Optional[Twin]:
    a, b = apply_edits(seed["tmpl_a"], edits), apply_edits(seed["tmpl_b"], edits)
    if a is None or b is None:
        return None
    a, b = a.replace("{X}", x).replace("{Y}", y), b.replace("{X}", x).replace("{Y}", y)
    nm = {"X": x, "Y": y}
    return Twin(f"{seed['seed']}|{tag}|{x}|{y}", seed["q_a"], seed["q_b"], a, b, x, y, nm[seed["ans_a"]], nm[seed["ans_b"]])


def build(t: Twin, d: str, tok, st: bool):
    """The direction's own pair builder (Phase 3 for ent2attr, Phase 5 F2 for attr2ent)."""
    return build_pair(t, tok, st) if d == "ent2attr" else build_ref_pair(t, "F2R", tok, st, set())


def name_pool(g: str) -> list[str]:
    return NAMES_F if g == "f" else NAMES_M


def name_pairs(seed: dict, k: int, rng: random.Random) -> list[tuple[str, str]]:
    """k ordered same-gender pairs of new names (template gender; n -> alternate m / f), not in the template text."""
    txt = seed["tmpl_a"] + " " + seed["tmpl_b"]
    out = []
    pools = [name_pool(seed["gender"])] if seed["gender"] in "mf" else [NAMES_M, NAMES_F]
    allp = [[(x, y) for x in p for y in p if x != y and x not in txt and y not in txt] for p in pools]
    for p in allp:
        rng.shuffle(p)
    i = 0
    while len(out) < k and any(allp):
        p = allp[i % len(allp)]
        if p:
            out.append(p.pop())
        i += 1
    return out
