"""ext7-controls (Phase 3): calibration tasks for the WinoGrande STR study and the CounterFact STR attention sweep.

Three parts, all STR (symmetric token replacement, Zhang & Nanda 2024) with Delta = LD(r, r') at the final position:

1. CounterFact STR attention / MoE / block sweep (task 1). Donor-level rows of scripts/ext7_cf_attnsweep.py are
   aggregated per case (donor mean primary, first donor sensitivity) into the interface of moetrace.ext2_attn.Run, so
   that the Direction-2b statistics (peaks, attention share of the positive rescue, additivity) apply unchanged.

2. W7 WinoGrande role swap (second corruption site, Z7). From a model's WinoGrande margin pool (results/wino_<proto>/
   scan_pairs.parquet) keep twins whose two options are person names each mentioned exactly once before the blank (and
   not after it). The role-swapped prompt exchanges those two first mentions and keeps the filled option:
       A:        "... Dennis helped Adam get more muscular since Dennis was the"  -> " trainer"  (trig_a)
       swap(A):  "... Adam helped Dennis get more muscular since Dennis was the"  -> " student"  (trig_b)
   Two STR pairs per twin: (A, swap(A)) with r = trig_a, r' = trig_b and (B, swap(B)) with r = trig_b, r' = trig_a. In
   the pair-file contract each role pair is stored with its own "A" = the prompt whose answer is the pair's trig_a.
   Token symmetry (W4 analogue): same length, the two mention spans at identical token positions, identical tokens
   elsewhere, and the same single-token continuation ids for both triggers after both prompts.

3. W8 IOI (Wang et al. 2023 templates, copied from Easy-Transformer `easy_transformer/ioi_dataset.py`: 15 BABA templates
   and their ABBA versions; names, places and objects of that file filtered to single leading-space tokens under the
   Qwen3 and Mixtral tokenizers). Clean prompt "Then, John and Mary went to the store. John gave a drink to" -> " Mary"
   (IO), metric LD(IO, S). Two STR corruptions (Zhang & Nanda Section 3 and Appendix F):
     (i)  S2 -> IO:  "... Mary gave a drink to" -> " John" (the corrupted prompt is an IOI sentence whose answer is S;
          symmetric, both directions), primary;
     (ii) S1 and IO -> two other random names (S2 kept): "Then, Alice and Carol went ... John gave a drink to"; the
          corrupted prompt has no answer among {IO, S}, so only d = 0 exists and the drop is reported, no corrupted
          margin.

Pair-file contract (consumed by agent ext7-wino's generic runner, moetrace/ext7_pairs.py): parquet with pair_id, ids_a,
ids_b (JSON lists), str_pos (JSON list of the positions where A and B differ), trig_a, trig_b (+ strata); case-set JSON
with pair_idx_of, pairs {discovery, validation, rep_discovery, rep_validation}, directed {same keys}, directed id =
2 * pair_idx + d (d = 0: clean A, corrupted B, true trig_a; d = 1 reversed).
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Optional

import numpy as np
import pandas as pd

from .data import _single_token_continuation
from .models import RESULTS

ROOT = "/home/ubuntu/MOE"
ROLE_DIR = os.path.join(ROOT, "data", "wino_role")
IOI_DIR = os.path.join(ROOT, "data", "ioi")
MARGIN = 1.0
PROTOS = {"qwen3": ("qwen3", True), "mixtral_bos": ("mixtral", True)}  # Phase-3 models (decision (e))


def load_tok(key: str):
    from transformers import AutoTokenizer
    from .arch import snapshot_dir
    from .models import MODELS
    return AutoTokenizer.from_pretrained(snapshot_dir(MODELS[key]["repo"]))


def _enc(tok, s: str, st: bool):
    return tok(s, return_offsets_mapping=True, return_special_tokens_mask=True, add_special_tokens=st)


def _span_positions(enc, start: int, end: int) -> list[int]:
    return [i for i, ((a, b), sp) in enumerate(zip(enc["offset_mapping"], enc["special_tokens_mask"]))
            if not sp and b > a and a < end and b > start]


# =================================================================================================================
# W7 role swap
# =================================================================================================================
def _mentions(text: str, name: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\b" + re.escape(name) + r"\b", text)]


def role_swap_text(context: str, opt1: str, opt2: str) -> tuple[Optional[str], str, Optional[dict]]:
    """Exchange the single pre-blank mentions of opt1 and opt2 in a WinoGrande context (with '_').
    Returns (swapped context, reason, info)."""
    i = context.index("_")
    before, after = context[:i], context[i + 1:]
    m1, m2 = _mentions(before, opt1), _mentions(before, opt2)
    if len(m1) != 1 or len(m2) != 1:
        return None, "R1_not_one_mention_each_before_blank", None
    if _mentions(after, opt1) or _mentions(after, opt2):
        return None, "R2_mention_after_blank", None
    (s1, e1), (s2, e2) = m1[0], m2[0]
    if max(s1, s2) < min(e1, e2):
        return None, "R1_overlapping_mentions", None
    (fs, fe, fname), (ls, le, lname) = sorted([(s1, e1, opt1), (s2, e2, opt2)])
    sw = before[:fs] + lname + before[fe:ls] + fname + before[le:] + "_" + after
    return sw, "ok", {"first": fname, "second": lname}


def build_role_pair(row, tok, st: bool, role: str) -> tuple[Optional[dict], str]:
    """One STR role pair from a WinoGrande margin pair row (scan_pairs / pairs parquet; needs ids not required).
    role 'A': clean = prompt A (filled ans_a), r = trig_a;  role 'B': clean = prompt B (filled ans_b), r = trig_b."""
    ctx = row.context
    opt_names = {row.ans_a, row.ans_b}
    if len(opt_names) != 2:
        return None, "R0_same_option"
    o1, o2 = row.ans_a, row.ans_b
    sw_ctx, why, info = role_swap_text(ctx, o1, o2)
    if sw_ctx is None:
        return None, why
    filled = row.ans_a if role == "A" else row.ans_b
    i = ctx.index("_")
    p_clean = ctx[:i] + filled + ctx[i + 1:]
    j = sw_ctx.index("_")
    p_swap = sw_ctx[:j] + filled + sw_ctx[j + 1:]
    ea, eb = _enc(tok, p_clean, st), _enc(tok, p_swap, st)
    ia, ib = list(ea["input_ids"]), list(eb["input_ids"])
    if len(ia) != len(ib):
        return None, "R3_length_mismatch"
    # mention spans in both prompts (the prompt text before the blank is ctx[:i] / sw_ctx[:j], same length only if
    # the names have equal length; compute spans separately)
    ma = sorted(_mentions(ctx[:i], o1) + _mentions(ctx[:i], o2))
    mb = sorted(_mentions(sw_ctx[:j], o1) + _mentions(sw_ctx[:j], o2))
    pa = [_span_positions(ea, s, e) for s, e in ma]
    pb = [_span_positions(eb, s, e) for s, e in mb]
    if pa != pb or any(not p for p in pa):
        return None, "R3_mention_span_mismatch"
    mpos = sorted(set(pa[0]) | set(pa[1]))
    if any(x != y for k, (x, y) in enumerate(zip(ia, ib)) if k not in set(mpos)):
        return None, "R3_context_tokens_differ"
    diff = [k for k in range(len(ia)) if ia[k] != ib[k]]
    if not diff:
        return None, "R3_identical_prompts"
    # filled option positions unchanged (it is outside the mention spans)
    fpos = _span_positions(ea, i, i + len(filled))
    # triggers: same single-token continuation ids after both prompts
    w_r, w_rp = (row.word_a, row.word_b) if role == "A" else (row.word_b, row.word_a)
    t_r = _single_token_continuation(tok, ia, p_clean, w_r, st)
    t_rp = _single_token_continuation(tok, ia, p_clean, w_rp, st)
    if t_r is None or t_rp is None:
        return None, "R4_trigger_multi_token"
    if _single_token_continuation(tok, ib, p_swap, w_r, st) != t_r or _single_token_continuation(tok, ib, p_swap, w_rp, st) != t_rp:
        return None, "R4_trigger_id_depends_on_prompt"
    exp_r = row.trig_a if role == "A" else row.trig_b
    exp_rp = row.trig_b if role == "A" else row.trig_a
    if (t_r, t_rp) != (int(exp_r), int(exp_rp)):
        return None, "R4_trigger_id_differs_from_wino_pair"
    if fpos and max(fpos) == len(ia) - 1:
        return None, "R5_option_is_final"
    return {"pair_id": f"{row.pair_id}|{role}", "twin_id": row.pair_id, "role": role, "context": ctx, "swapped_context": sw_ctx,
            "filled": filled, "first_mention": info["first"], "second_mention": info["second"],
            "prompt_a": p_clean, "prompt_b": p_swap, "word_a": w_r, "word_b": w_rp,
            "ids_a": json.dumps(ia), "ids_b": json.dumps(ib), "str_pos": json.dumps(diff), "mention_pos": json.dumps(pa),
            "opt_pos": json.dumps(fpos), "trig_a": int(t_r), "trig_b": int(t_rp), "n_tokens": len(ia), "n_str_pos": len(diff),
            "debiased": bool(getattr(row, "debiased", False)), "trigger_in_context": bool(getattr(row, "trigger_in_context", False)),
            "wino_assoc": bool(getattr(row, "assoc", False)), "wino_top1_both": bool(getattr(row, "top1_both", False)),
            "wino_d_clean_a": float(row.d_clean_a), "wino_d_clean_b": float(row.d_clean_b)}, "ok"


# =================================================================================================================
# W8 IOI
# =================================================================================================================
# Verbatim from redwoodresearch/Easy-Transformer easy_transformer/ioi_dataset.py (Wang et al. 2023), fetched 2026-10-04.
IOI_NAMES = ["Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer", "Joshua", "Amanda", "Daniel", "David",
             "James", "Robert", "John", "Joseph", "Andrew", "Ryan", "Brandon", "Jason", "Justin", "Sarah", "William",
             "Jonathan", "Stephanie", "Brian", "Nicole", "Nicholas", "Anthony", "Heather", "Eric", "Elizabeth", "Adam",
             "Megan", "Melissa", "Kevin", "Steven", "Thomas", "Timothy", "Christina", "Kyle", "Rachel", "Laura", "Lauren",
             "Amber", "Brittany", "Danielle", "Richard", "Kimberly", "Jeffrey", "Amy", "Crystal", "Michelle", "Tiffany",
             "Jeremy", "Benjamin", "Mark", "Emily", "Aaron", "Charles", "Rebecca", "Jacob", "Stephen", "Patrick", "Sean",
             "Erin", "Jamie", "Kelly", "Samantha", "Nathan", "Sara", "Dustin", "Paul", "Angela", "Tyler", "Scott",
             "Katherine", "Andrea", "Gregory", "Erica", "Mary", "Travis", "Lisa", "Kenneth", "Bryan", "Lindsey", "Kristen",
             "Jose", "Alexander", "Jesse", "Katie", "Lindsay", "Shannon", "Vanessa", "Courtney", "Christine", "Alicia",
             "Cody", "Allison", "Bradley", "Samuel"]
BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]
IOI_PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "house", "station"]
IOI_OBJECTS = ["ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack"]


def _abba(templates: list[str]) -> list[str]:
    """ABBA versions exactly as ioi_dataset.py builds them: in the first clause the first [B] becomes [A] and the first
    [A] becomes [B]."""
    out = []
    for t in templates:
        first_clause = True
        for j in range(1, len(t) - 1):
            if t[j - 1: j + 2] == "[B]" and first_clause:
                t = t[:j] + "A" + t[j + 1:]
            elif t[j - 1: j + 2] == "[A]" and first_clause:
                first_clause = False
                t = t[:j] + "B" + t[j + 1:]
        out.append(t)
    return out


ABBA_TEMPLATES = _abba(BABA_TEMPLATES)
IOI_TEMPLATES = [("BABA", i, t) for i, t in enumerate(BABA_TEMPLATES)] + [("ABBA", i, t) for i, t in enumerate(ABBA_TEMPLATES)]


def single_token_words(toks: dict, words: list[str], frame: str) -> list[str]:
    """Words that are ONE leading-space token inside `frame` (with '{}' for the word, not at the start) and as the
    continuation of the frame's prefix, under every tokenizer in toks {key: (tok, st)}."""
    keep = []
    for w in words:
        ok = True
        for key, (tok, st) in toks.items():
            s = frame.format(w)
            e = _enc(tok, s, st)
            a = s.index(w)
            if len(_span_positions(e, a, a + len(w))) != 1:
                ok = False
                break
            pre = frame[: frame.index("{}")].rstrip(" ")
            pid = tok(pre, add_special_tokens=st)["input_ids"]
            if _single_token_continuation(tok, pid, pre, w, st) is None:
                ok = False
                break
        if ok:
            keep.append(w)
    return keep


def fill(template: str, io: str, s: str, place: str, obj: str) -> tuple[str, str]:
    """(prompt without the final ' [A]', answer IO). Templates end with ' [A]'."""
    assert template.endswith(" [A]")
    body = template[: -len(" [A]")]
    return body.replace("[A]", io).replace("[B]", s).replace("[PLACE]", place).replace("[OBJECT]", obj), io


def name_slots(template: str) -> dict:
    """Character-independent description of the name slots of the prompt part: order of [A]/[B] occurrences."""
    body = template[: -len(" [A]")]
    return [m.group(0) for m in re.finditer(r"\[A\]|\[B\]", body)]


def ioi_prompts(template: str, io: str, s: str, place: str, obj: str, c: str, d: str) -> dict:
    """Clean prompt, corruption (i) S2 -> IO, corruption (ii) S1 -> c and IO -> d (S2 kept). Returns texts and the
    character spans of S1, IO and S2 in each prompt."""
    body = template[: -len(" [A]")]
    slots = name_slots(template)
    assert slots.count("[B]") == 2 and slots.count("[A]") == 1, template
    out = {}
    for name, (a_name, b1, b2) in {"clean": (io, s, s), "s2io": (io, s, io), "s1io": (d, c, s)}.items():
        txt, spans, k_b, cur = "", {}, 0, 0
        for m in re.finditer(r"\[A\]|\[B\]|\[PLACE\]|\[OBJECT\]", body):
            txt += body[cur: m.start()]
            tag = m.group(0)
            if tag == "[A]":
                rep, lab = a_name, "IO"
            elif tag == "[B]":
                rep, lab = (b1, "S1") if k_b == 0 else (b2, "S2")
                k_b += 1
            elif tag == "[PLACE]":
                rep, lab = place, None
            else:
                rep, lab = obj, None
            if lab:
                spans[lab] = (len(txt), len(txt) + len(rep))
            txt += rep
            cur = m.end()
        txt += body[cur:]
        out[name] = (txt, spans)
    return out


# =================================================================================================================
# generic STR-pair helpers (scan, case sets)
# =================================================================================================================
def directed(pairs: list[int], both: bool = True) -> list[int]:
    return [2 * i + d for i in pairs for d in ((0, 1) if both else (0,))]


def case_sets(pair_ids: list[str], pool_idx: list[int], seed: int = 0, n: int = 128, groups: Optional[dict] = None,
              both: bool = True, note: str = "") -> dict:
    """Contract case-set JSON. pair_idx = index of pair_id in the sorted list `pair_ids` (all pairs of the parquet).
    pool_idx = indices eligible for the sets. random.Random(seed) shuffle of the pool (of the groups when `groups` maps
    pair_idx -> group key, e.g. the WinoGrande twin, so that both role pairs of a twin never fall into different splits;
    a group that does not fit into the remaining room of a split contributes its first pairs and the rest go to
    'unused'). Sets discovery / validation / rep_discovery / rep_validation of n pairs each when >= 2n pairs exist (the
    replication sets as far as the pool allows); otherwise the whole pool is a fixed-hypothesis validation set."""
    pool = sorted(pool_idx)
    rng = random.Random(seed)
    if groups:
        keys = sorted({groups[i] for i in pool})
        rng.shuffle(keys)
        by = {}
        for i in pool:
            by.setdefault(groups[i], []).append(i)
        units = [sorted(by[k]) for k in keys]
    else:
        order = list(pool)
        rng.shuffle(order)
        units = [[i] for i in order]
    names = ["discovery", "validation", "rep_discovery", "rep_validation"]
    if len(pool) >= 2 * n:
        sets = {k: [] for k in names + ["unused"]}
        si = 0
        for u in units:
            if si >= len(names):
                sets["unused"] += u
                continue
            room = n - len(sets[names[si]])
            sets[names[si]] += u[:room]
            sets["unused"] += u[room:]
            if len(sets[names[si]]) >= n:
                si += 1
        if len(sets["rep_validation"]) < n:  # no partial replication: incomplete replication sets go to 'unused'
            sets["unused"] = sets["rep_discovery"] + sets["rep_validation"] + sets["unused"]
            sets["rep_discovery"], sets["rep_validation"] = [], []
        mode = f"seed-{seed} split {n}/{n} discovery/validation; replication {len(sets['rep_discovery'])}/{len(sets['rep_validation'])}"
    else:
        sets = {"discovery": [], "validation": [i for u in units for i in u], "rep_discovery": [], "rep_validation": [], "unused": []}
        mode = f"fixed-hypothesis validation set of all {len(pool)} pairs (fewer than {2 * n})"
    return {"note": note, "mode": mode, "seed": seed, "pair_idx_of": {p: i for i, p in enumerate(pair_ids)},
            "pool": pool, "pairs": sets, "directed": {k: directed(v, both) for k, v in sets.items()},
            "both_directions": both}


def scan_rows_to_engine(pairs: pd.DataFrame, extra: Optional[dict] = None):
    """PrefillSpecs for the competence scan: per pair rows 'a' and 'b' (+ extra {name: column of JSON ids}), all
    scored with true = trig_a, foil = trig_b."""
    from .engine import PrefillSpec
    rows = []
    for i, r in enumerate(pairs.itertuples()):
        rows.append((i, "a", PrefillSpec(json.loads(r.ids_a), int(r.trig_a), int(r.trig_b))))
        rows.append((i, "b", PrefillSpec(json.loads(r.ids_b), int(r.trig_a), int(r.trig_b))))
        for name, col in (extra or {}).items():
            rows.append((i, name, PrefillSpec(json.loads(getattr(r, col)), int(r.trig_a), int(r.trig_b))))
    return rows


def run_scan(eng, rows, max_tokens: int = 140000, log=None) -> pd.DataFrame:
    """Prefill-only passes over (pair, kind, PrefillSpec) rows, packed by length. Returns long table."""
    order = sorted(range(len(rows)), key=lambda j: len(rows[j][2].ids))
    chunks, cur, cur_max = [], [], 0
    for j in order:
        T = len(rows[j][2].ids)
        if cur and max(cur_max, T) * (len(cur) + 1) > max_tokens:
            chunks.append(cur)
            cur, cur_max = [], 0
        cur.append(j)
        cur_max = max(cur_max, T)
    if cur:
        chunks.append(cur)
    n = len(rows)
    delta, top1 = np.full(n, np.nan), np.full(n, -1, dtype=np.int64)
    lt, lf = np.full(n, np.nan), np.full(n, np.nan)
    times = []
    for ci, ch in enumerate(chunks):
        res = eng.run([rows[j][2] for j in ch], [], record_routing=False, log=log if ci == 0 else None)
        times.append(res.extra["total_s"])
        delta[ch], top1[ch], lt[ch], lf[ch] = res.delta, res.top1, res.logit_true, res.logit_foil
    out = pd.DataFrame({"pair": [r[0] for r in rows], "kind": [r[1] for r in rows], "delta": delta, "top1": top1,
                        "logit_true": lt, "logit_foil": lf})
    out.attrs["pass_times_s"] = times
    return out


# =================================================================================================================
# task 1: CounterFact STR attention / MoE / block sweep (analysis)
# =================================================================================================================
CF_RUNS = {"qwen3": {"str": "qwen3_str_attnsweep", "gn": "qwen3_bos_attnsweep", "ext6": "qwen3_str", "label": "Qwen3-30B-A3B-Base",
                     "paper_layer": 44},
           "mixtral_bos": {"str": "mixtral_bos_str_attnsweep", "gn": "mixtral_bos_attnsweep", "ext6": "mixtral_bos_str",
                           "label": "Mixtral-8x7B, BOS", "paper_layer": 19}}
SUBLAYER_KINDS = ("attn_layer", "layer", "block")


class StrAttnRun:
    """moetrace.ext2_attn.Run interface for a donor-level STR sweep (str_attn_rows.parquet): per-case rescue = mean
    over the case's donors (donors='mean', primary) or the first donor (donors='first'). drop = same-pass clean delta
    minus the (donor-mean / first-donor) corrupted delta."""

    def __init__(self, run: str, donors: str = "mean", rows: Optional[pd.DataFrame] = None):
        d = os.path.join(RESULTS, run)
        self.run, self.dir, self.donors = run, d, donors
        rows = rows if rows is not None else pd.read_parquet(os.path.join(d, "str_attn_rows.parquet"))
        self.sets = json.load(open(os.path.join(d, "case_sets.json")))
        self.case_set = "paper"
        if donors == "first":
            rows = rows[(rows.slot == 0) | (rows.kind == "clean")]
        cor = rows[(rows.kind == "corrupt") & (rows.chunk == rows.chunk.min())]
        self.delta_clean = rows[rows.kind == "clean"].set_index("case_id").delta
        self.delta_noised = cor.groupby("case_id").delta.mean()
        self.drop = (self.delta_clean - self.delta_noised).dropna()
        self.kinds = [k for k in SUBLAYER_KINDS if k in set(rows.kind)]
        self.R, self.V = {}, {}
        for k in self.kinds:
            g = rows[rows.kind == k].groupby(["case_id", "layer"])
            self.R[k] = g.rescue.mean().unstack()
            self.V[k] = g.vnorm.mean().unstack()
        self.layers = list(self.R[self.kinds[0]].columns)
        self.L = len(self.layers)
        present = set(self.R[self.kinds[0]].index)
        self.disc = [c for c in self.sets["paper"]["discovery"] if c in present]
        self.val = [c for c in self.sets["paper"]["validation"] if c in present]
        self.cfg = {"short": run, "label": run}

    def mat(self, kind: str, ids: list[int]) -> np.ndarray:
        return self.R[kind].loc[ids].to_numpy(dtype=np.float64)


def gn_run_on(run_gn: str, ids_disc: list[int], ids_val: list[int]):
    """Direction-2b GN attention sweep restricted to the given discovery / validation cases (same split as STR)."""
    from .ext2_attn import Run
    r = Run(run_gn)
    present = set(r.R["layer"].index)
    r.disc = [c for c in ids_disc if c in present]
    r.val = [c for c in ids_val if c in present]
    r.drop = (r.delta_clean - r.delta_noised)
    return r


def share_ci(run, ids: list[int], n_boot: int = 5000, seed: int = 0) -> dict:
    """Direction-2b attention share of the positive rescue: AUC+(attn) / (AUC+(attn) + AUC+(moe)) of the mean curves of
    the given cases, case-bootstrap CI (ext2_attn.share_overall with an arbitrary case list)."""
    from .ext2_attn import share_overall
    return share_overall(run, ids)


def peak_rows(run, ids_drop: Optional[pd.Series] = None) -> pd.DataFrame:
    """ext2_attn.peaks plus drop-normalised values (validation mean rescue / validation mean drop)."""
    from .ext2_attn import peaks
    pk = peaks(run, [k for k in SUBLAYER_KINDS if k in run.kinds])
    drop = float(run.drop.loc[run.val].mean())
    pk["val_mean_drop"] = drop
    for c in ("val_at_L_disc", "val_at_L_disc_lo", "val_at_L_disc_hi", "val_max", "auc_pos"):
        pk[c + "_norm"] = pk[c] / drop
    return pk


def additivity_at(run, layers: list[int]) -> pd.DataFrame:
    from .ext2_attn import additivity
    a = additivity(run, run.val)
    return a[a.layer.isin(layers)]


def curve_table(run, label: str, corruption: str) -> pd.DataFrame:
    from .ext2_attn import all_curves
    c = all_curves(run, run.val, [k for k in SUBLAYER_KINDS if k in run.kinds])
    drop = float(run.drop.loc[run.val].mean())
    c["mean_norm"], c["ci_lo_norm"], c["ci_hi_norm"] = c["mean"] / drop, c["ci_lo"] / drop, c["ci_hi"] / drop
    c.insert(0, "model", label)
    c.insert(1, "corruption", corruption)
    return c


def layer_consistency(run_str: str, run_ext6: str) -> dict:
    """Same-pass MoE `layer` rows of the attention sweep vs the Direction-6 sweep rows (same donors, other pass)."""
    a = pd.read_parquet(os.path.join(RESULTS, run_str, "str_attn_rows.parquet"))
    a = a[a.kind == "layer"].set_index(["case_id", "slot", "layer"]).rescue
    b = pd.read_parquet(os.path.join(RESULTS, run_ext6, "str_sweep_rows.parquet"))
    b = b[b.kind == "layer"].set_index(["case_id", "slot", "layer"]).rescue
    j = pd.concat([a.rename("new"), b.rename("ext6")], axis=1).dropna()
    d = (j.new - j.ext6).abs()
    return {"n_rows": int(len(j)), "r": float(np.corrcoef(j.new, j.ext6)[0, 1]), "max_absdiff": float(d.max()),
            "mean_absdiff": float(d.mean()), "frac_identical": float((d == 0).mean())}


# =================================================================================================================
# W7 / W8 analysis on runs of the ext7 generic pair runner (moetrace/ext7_pairs.py)
# =================================================================================================================
def _split_ids(run, family: str) -> tuple[list[int], list[int]]:
    f = run.fam.get(family, {})
    return list(f.get("discovery", [])), list(f.get("validation", []))


def pair_run_summary(run_name: str, family: str = "main", fixed_layers: Optional[dict] = None) -> dict:
    """Final-position W2 summary of a pair run: drop, Delta means, per kind discovery argmax (or, without a discovery
    split, the validation argmax as a descriptive peak) with pair-level validation rescue and drop-normalised value,
    attention share of the positive rescue (Direction-2b AUC+ definition) with pair-bootstrap CI, shares and additivity
    at the peak layers, and validation values at fixed hypothesis layers {name: layer}."""
    from . import ext7_pairs as P
    run = P.PairRun(run_name)
    disc, val = _split_ids(run, family)
    cases = run.cases
    out = {"run": run_name, "family": family, "n_disc_pairs": len({c // 2 for c in disc}), "n_val_pairs": len({c // 2 for c in val}),
           "n_val_directed": len(val), "kinds": run.kinds, "L": run.L}
    cv = cases.loc[val]
    out["delta_clean"] = P.summ(cv.delta_clean, with_p=False)
    out["delta_corrupt"] = P.summ(cv.delta_corrupt, with_p=False)
    out["drop"] = P.summ(cv["drop"], with_p=False)
    peaks = {}
    for k in run.kinds:
        Mv = run.mat(k, val)
        mv = P.matrix_pairs(Mv).mean(0)
        if disc:
            md = P.matrix_pairs(run.mat(k, disc)).mean(0)
            ld = int(md.idxmax())
            top = md.sort_values(ascending=False)
            gap = float(top.iloc[0] - top.iloc[1])
        else:
            ld, gap = int(mv.idxmax()), float("nan")
        lv = int(mv.idxmax())
        peaks[k] = {"L_sel": ld, "selected_on": "discovery" if disc else "validation (descriptive, no discovery split)",
                    "sel_gap_to_2nd": gap, "val_at_L_sel": P.summ(Mv[ld]), "norm_at_L_sel": P.ratio(Mv[ld], run.drop.loc[val]),
                    "L_val": lv, "val_max": P.summ(Mv[lv], with_p=False), "norm_at_L_val": P.ratio(Mv[lv], run.drop.loc[val]),
                    "auc_pos": float(np.clip(mv, 0, None).sum()),
                    "val_top5": [(int(l), round(float(v), 3)) for l, v in mv.sort_values(ascending=False).head(5).items()]}
    out["peaks"] = peaks
    dval = float(P.pair_means(run.drop.loc[val]).mean())
    out["layer_sum_norm"] = {k: float(P.matrix_pairs(run.mat(k, val)).mean(0).sum() / dval) for k in run.kinds}
    if {"attn_layer", "layer"} <= set(run.kinds):
        out["share"] = P.attention_share(run, val)
        out["share_at_moe_peak"] = _share_at(run, peaks["layer"]["L_sel"], val)
        out["share_at_attn_peak"] = _share_at(run, peaks["attn_layer"]["L_sel"], val)
        if "block" in run.kinds:
            ad = P.additivity(run, val)
            ls = sorted({peaks[k]["L_sel"] for k in ("attn_layer", "layer", "block")})
            out["additivity_at_peaks"] = ad[ad.layer.isin(ls)].to_dict("records")
            out["additivity_mean_gap_all_layers"] = float(ad.gap.mean())
    if fixed_layers:
        fx = {}
        for name, (kind, l) in fixed_layers.items():
            if kind in run.kinds and l is not None:
                M = run.mat(kind, val)[int(l)]
                fx[name] = {"kind": kind, "layer": int(l), "val": P.summ(M), "norm": P.ratio(M, run.drop.loc[val])}
        out["fixed"] = fx
    return out


def _share_at(run, layer: int, ids: list[int]) -> dict:
    from . import ext7_pairs as P
    a = run.mat("attn_layer", ids)[layer]
    m = run.mat("layer", ids)[layer]
    r = P.ratio(a, a + m)
    return {"layer": int(layer), "attn": float(P.pair_means(a).mean()), "moe": float(P.pair_means(m).mean()), "share": r[0],
            "share_lo": r[1], "share_hi": r[2]}


def pair_run_curves(run_name: str, family: str = "main", split: str = "validation") -> pd.DataFrame:
    from . import ext7_pairs as P
    run = P.PairRun(run_name)
    ids = list(run.fam[family][split])
    return P.sweep_curves(run, ids, [k for k in SUBLAYER_KINDS if k in run.kinds])


SUMMARY = os.path.join(RESULTS, "ext7_controls_summary.json")


def joint_summary(run_name: str, family: str = "main", spans=("all",), split: Optional[str] = "validation") -> dict:
    """W4 two-player Shapley split from results/<run>/joint_rows.parquet (scripts/ext7_wino_joint.py), computed as in
    agent ext7-wino's scripts/ext7_wino_analyze.py: A = mean all-attention effect / mean drop, M = mean all-MoE effect /
    mean drop (population ratios, pair bootstrap), phi_attn = 1/2 [A + (1 - M)], phi_moe = 1 - phi_attn, redundancy
    A + M - 1; block sanity = all_block effect / drop (should be 1). Directed cases of the family's `split` (validation by
    default, as the W2 peaks and ext8's CounterFact W4; None = all splits); both directions of the patch (denoise =
    sufficiency, noise = necessity)."""
    from . import ext7_pairs as P
    df = pd.read_parquet(os.path.join(RESULTS, run_name, "joint_rows.parquet"))
    fam = P.load_families(run_name)
    ids = [c for k, v in fam[family].items() if split is None or k == split for c in v]
    sub = df[df.case_id.isin(ids)]
    out = {}
    for (direction, span), g in sub.groupby(["direction", "span"]):
        if span not in spans:
            continue
        piv = g.pivot(index="case_id", columns="kind", values="effect")
        drop = g.groupby("case_id")["drop"].first()
        r = {k: P.ratio(piv[k], drop) for k in piv.columns}
        a0, m0 = r["all_attn"][0], r["all_moe"][0]
        pa, pm, dd = P.pair_means(piv["all_attn"]), P.pair_means(piv["all_moe"]), P.pair_means(drop)
        idx = P._boot_idx(len(dd))
        ab = pa.to_numpy()[idx].mean(1) / dd.to_numpy()[idx].mean(1)
        mb = pm.to_numpy()[idx].mean(1) / dd.to_numpy()[idx].mean(1)
        phib, redb = 0.5 * (ab + 1 - mb), ab + mb - 1
        out[f"{direction}_{span}"] = {"A": r["all_attn"], "M": r["all_moe"], "block": r.get("all_block"),
                                      "phi_attn": [0.5 * (a0 + 1 - m0), float(np.percentile(phib, 2.5)), float(np.percentile(phib, 97.5))],
                                      "redundancy": [a0 + m0 - 1, float(np.percentile(redb, 2.5)), float(np.percentile(redb, 97.5))],
                                      "n_pairs": int(len(dd)), "n_directed": int(len(drop))}
    return out


# =================================================================================================================
# W3 grid restricted to named positions (same executor and row schema as scripts/ext7_wino_grid.py; only the unit
# selection differs: instead of every position from min(str_pos) to T-1, the positions returned by `positions_of`)
# =================================================================================================================
GRID_MET = ("logp_true", "logp_foil", "p_true", "p_foil", "rank_true")


def run_grid_positions(args, positions_of, log=print):
    """args: argparse namespace with the ext7_pairs common args + kinds (list), token_budget, max_rows.
    positions_of(pair_row, directed_case) -> list of (position, class label). Window 1 only (no window>1 rows, so the
    attn_layer rows are never mixed with windowed rows in one pass). Writes results/<out>/str_grid_w1_rows.parquet with the
    columns of scripts/ext7_wino_grid.py plus 'cls'."""
    import sys
    import time
    import torch
    from . import ext7_pairs as P
    from .models import MODELS
    from .arch import load_spec
    from .ext5_subject import SubjectEngine, SubjectPrefill, SubjectSpawn
    m = MODELS[args.model]
    L = load_spec(m["repo"])[0].n_layers
    mp = os.path.join(P.run_dir(args.out), "run_meta.json")
    if not getattr(args, "force", False) and os.path.exists(mp) and json.load(open(mp)).get("grid_w1", {}).get("complete") \
            and os.path.exists(os.path.join(P.run_dir(args.out), "str_grid_w1_rows.parquet")):
        log(f"{args.out}: grid_w1 already complete (run_meta.json), nothing to do (use --force to recompute)")
        return
    cs, pairs, _, _, _ = P.setup_run(args, "grid")
    fam = P.load_families(args.out)
    ids = P.all_ids(fam)
    dc = P.directed_cases(pairs, ids)
    so = P.split_of(fam)
    units = []
    for c in ids:
        x = dc[c]
        for p, lab in positions_of(pairs.loc[x.pair_idx], x):
            units.append({"case_id": c, "pos": int(p), "S": x.T - int(p), "cls": lab, "cat": P.position_category(int(p), x.str_pos, x.T)})
    per_unit = L * len(args.kinds)
    units = sorted(units, key=lambda u: (u["S"], u["case_id"], u["pos"]))
    passes, cur, rows = [], [], 0
    for u in units:
        if cur and ((rows + per_unit) * u["S"] > args.token_budget or rows + per_unit > args.max_rows):
            passes.append(cur); cur, rows = [], 0
        cur.append(u); rows += per_unit
    if cur:
        passes.append(cur)
    log(f"{args.model} -> {args.out}: kinds {args.kinds}, {len(ids)} directed cases, {len(units)} units, {len(units) * per_unit} suffix rows, "
        f"{len(passes)} passes; units per class {pd.Series([u['cls'] for u in units]).value_counts().to_dict()}")
    if args.dry_run:
        return
    meta = {"model": args.model, "repo": m["repo"], "families": list(fam), "kinds": args.kinds, "window": 1, "n_cases": len(ids),
            "n_units": len(units), "n_suffix_rows": len(units) * per_unit, "n_passes": len(passes), "token_budget": args.token_budget,
            "max_rows": args.max_rows, "positions": args.positions_doc, "agent": "ext7-controls", "command": "python " + " ".join(sys.argv),
            "executor": "moetrace.ext5_subject.SubjectEngine.run_subject (as scripts/ext7_wino_grid.py, verified by ext7-wino)",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    P.write_meta(args.out, "grid_w1", meta)
    eng = SubjectEngine(m["repo"])
    od = P.run_dir(args.out)
    rows_path, pre_path = os.path.join(od, "str_grid_w1_rows.parquet"), os.path.join(od, "str_grid_w1_prefill.parquet")
    all_rows, all_pre, times = [], [], []
    for pi, ps in enumerate(passes):
        pre, spawns, tags, ptags = [], [], [], []
        for u in ps:
            x = dc[u["case_id"]]
            p = u["pos"]
            ci = len(pre); pre.append(SubjectPrefill(x.clean_ids, x.true_id, x.foil_id, rec_pos=p)); ptags.append((u, -1))
            di = len(pre); pre.append(SubjectPrefill(x.corrupt_ids, x.true_id, x.foil_id, rec_pos=p)); ptags.append((u, 0))
            for l in range(L):
                for k in args.kinds:
                    spawns.append(SubjectSpawn(l, di, ci, k, p, window=1)); tags.append((u, k, l, di, ci))
        log(f"pass {pi + 1}/{len(passes)}: {len(ps)} units (S {ps[0]['S']}..{ps[-1]['S']}), {len(pre)} prefill rows, {len(spawns)} suffix rows")
        res = eng.run_subject(pre, spawns, log=log if pi == 0 else None, metrics=True)
        times.append(res.extra["total_s"])
        d, sd = res.delta, res.sp_delta
        mp, ms = res.extra["metrics_prefill"], res.extra["metrics_spawn"]
        for j, (u, slot) in enumerate(ptags):
            all_pre.append(dict(case_id=u["case_id"], pos=u["pos"], cls=u["cls"], cat=u["cat"], slot=slot, delta=float(d[j]),
                                **{k: (int(mp[k][j]) if k == "rank_true" else float(mp[k][j])) for k in GRID_MET}, pass_idx=pi))
        for jj, (u, k, l, di, ci) in enumerate(tags):
            c = u["case_id"]
            x = dc[c]
            all_rows.append(dict(case_id=c, pair_idx=x.pair_idx, d=x.d, split=so.get(c, ""), pos=u["pos"], rel_pos=u["pos"] - (x.T - 1),
                                 off_str=u["pos"] - x.first_str, S=u["S"], cls=u["cls"], cat=u["cat"], kind=k, layer=l, start=l, end=l,
                                 delta=float(sd[jj]), delta_corrupt=float(d[di]), delta_clean=float(d[ci]), rescue=float(sd[jj] - d[di]),
                                 p_true=float(ms["p_true"][jj]), p_true_corrupt=float(mp["p_true"][di]), p_true_clean=float(mp["p_true"][ci]),
                                 dp=float(ms["p_true"][jj] - mp["p_true"][di]), rank_true=int(ms["rank_true"][jj]),
                                 vnorm=float(res.sp_vnorm[jj]), pass_idx=pi))
        pd.DataFrame(all_rows).to_parquet(rows_path, index=False)
        pd.DataFrame(all_pre).to_parquet(pre_path, index=False)
        log(f"pass {pi + 1}: {res.extra['total_s']:.1f}s; {len(all_rows)} rows saved")
        del res
        torch.cuda.empty_cache()
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "pass_times_s": times,
                 "n_rows": len(all_rows)})
    P.write_meta(args.out, "grid_w1", meta)


def grid_args(ap):
    from . import ext7_pairs as P
    P.add_common_args(ap)
    ap.add_argument("--kinds", default="layer,attn_layer")
    ap.add_argument("--token-budget", type=int, default=200_000)
    ap.add_argument("--max-rows", type=int, default=60_000)
    ap.add_argument("--force", action="store_true", help="recompute even if the run's grid_w1 is complete")


# =================================================================================================================
# W4 complement: direct-path split at the final position (as agent ext8-addback's scripts/ext8_addback_run.py
# direct_split; prefill only). At the final position the all-attention patch restores the clean residual exactly
# (the final token is shared and the MoE is a per-token function), so the W4 Shapley game is degenerate (A = 1,
# phi_MoE = M/2); the direct-path split is the non-degenerate view of "which sublayer writes carry the drop".
# =================================================================================================================
def direct_split_pairs(model: str, pairs_path: str, case_sets_path: str, splits: Optional[list] = None, log=print) -> pd.DataFrame:
    """One prefill pass over prompts A and B of every pair used by the case-set families (DiagSpec attn_out_final,
    resid_final). Per directed case (clean, corrupt, u = gamma * (W_U[true] - W_U[foil])), offline in fp32:
      d_clean_fp32 / d_corrupt_fp32   Delta from the final residuals with the exact final RMSNorm
      d_attn_direct                   Delta of h_corrupt + sum_l (Attn_clean - Attn_corrupt)   (exact norm)
      d_moe_direct                    Delta of h_corrupt + sum_l (MoE_clean - MoE_corrupt) = h_clean - sum_l dAttn
      dla_attn, dla_moe, dla_total    linear DLA of the summed writes with the norm frozen at the corrupted run's scale."""
    import torch
    from . import ext7_pairs as P
    from .engine import DiagSpec, Engine, PrefillSpec
    from .models import MODELS
    cs = json.load(open(case_sets_path))
    fam = P.resolve_sets(cs, splits)
    ids = P.all_ids(fam)
    pairs = P.load_pairs(pairs_path, cs)
    pidx = sorted({c // 2 for c in ids})
    eng = Engine(MODELS[model]["repo"])
    L = eng.spec.n_layers
    pre = []
    for pi in pidx:
        r = pairs.loc[pi]
        pre += [PrefillSpec(list(r.ids_a_l), int(r.trig_a), int(r.trig_b)), PrefillSpec(list(r.ids_b_l), int(r.trig_a), int(r.trig_b))]
    res = eng.run(pre, [], record_routing=False, diag=DiagSpec(attn_out_final=True, resid_final=True), log=log)
    att, rf = res.extra["diag"]["attn_out_final"], res.extra["diag"]["resid_final"]
    gamma = eng.g["norm"].float().cpu()
    head = eng.g["head"]
    eps = eng.spec.rms_eps
    have = set(ids)
    out = []
    for j, pi in enumerate(pidx):
        r = pairs.loc[pi]
        u0 = (head[int(r.trig_a)].float() - head[int(r.trig_b)].float()).cpu() * gamma
        for d in (0, 1):
            c = 2 * pi + d
            if c not in have:
                continue
            ci, cr = (2 * j, 2 * j + 1) if d == 0 else (2 * j + 1, 2 * j)
            u = u0 if d == 0 else -u0
            hc, hj = rf[L - 1, ci].float(), rf[L - 1, cr].float()
            dA = (att[:, ci].float() - att[:, cr].float()).sum(0)

            def D(h):
                return float((h * torch.rsqrt(h.pow(2).mean() + eps)) @ u)
            rms_j = float(torch.sqrt(hj.pow(2).mean() + eps))
            out.append(dict(case_id=c, pair_idx=pi, d=d, delta_clean_engine=float(res.delta[ci] if d == 0 else -res.delta[ci]),
                            delta_corrupt_engine=float(res.delta[cr] if d == 0 else -res.delta[cr]), d_clean_fp32=D(hc), d_corrupt_fp32=D(hj),
                            d_attn_direct=D(hj + dA), d_moe_direct=D(hc - dA), dla_attn=float(dA @ u) / rms_j,
                            dla_moe=float((hc - hj - dA) @ u) / rms_j, dla_total=float((hc - hj) @ u) / rms_j))
    return pd.DataFrame(out)


def direct_summary(run_name: str, family: str = "main", split: Optional[str] = "validation") -> dict:
    """A_direct = mean(d_attn_direct - d_corrupt) / mean drop, M_direct likewise (fp32 path, exact norm), DLA shares
    dla_attn / dla_total and dla_moe / dla_total (population ratios, pair bootstrap); all directed cases of the family."""
    from . import ext7_pairs as P
    df = pd.read_parquet(os.path.join(RESULTS, run_name, "direct_split.parquet")).set_index("case_id")
    fam = P.load_families(run_name)
    ids = [c for k, v in fam[family].items() if split is None or k == split for c in v if c in df.index]
    x = df.loc[ids]
    drop = x.d_clean_fp32 - x.d_corrupt_fp32
    return {"A_direct": P.ratio(x.d_attn_direct - x.d_corrupt_fp32, drop), "M_direct": P.ratio(x.d_moe_direct - x.d_corrupt_fp32, drop),
            "dla_attn_share": P.ratio(x.dla_attn, x.dla_total), "dla_moe_share": P.ratio(x.dla_moe, x.dla_total),
            "drop_fp32": float(drop.mean()), "drop_engine": float((x.delta_clean_engine - x.delta_corrupt_engine).mean()),
            "n_directed": int(len(x)), "n_pairs": int(len({c // 2 for c in ids}))}


def _cli():
    import argparse
    import sys
    import time
    ap = argparse.ArgumentParser(description="ext7-controls CLI: 'direct' (W4 direct-path split, GPU prefill) | 'section' (CPU)")
    ap.add_argument("cmd", choices=["direct", "section"])
    ap.add_argument("--model")
    ap.add_argument("--pairs")
    ap.add_argument("--case-sets")
    ap.add_argument("--out")
    ap.add_argument("--splits", default=None)
    a = ap.parse_args()
    log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)
    if a.cmd == "direct":
        from . import ext7_pairs as P
        t0 = time.time()
        df = direct_split_pairs(a.model, a.pairs, a.case_sets, a.splits.split(",") if a.splits else None, log)
        od = P.run_dir(a.out)
        df.to_parquet(os.path.join(od, "direct_split.parquet"), index=False)
        P.write_meta(a.out, "direct_split", {"model": a.model, "pairs": a.pairs, "case_sets": a.case_sets, "splits": a.splits,
                                             "n_rows": len(df), "seconds": time.time() - t0, "command": "python -m moetrace.ext7_controls " + " ".join(sys.argv[1:]),
                                             "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        dr = df.d_clean_fp32 - df.d_corrupt_fp32
        log(f"{a.out}: {len(df)} directed cases; A_direct {((df.d_attn_direct - df.d_corrupt_fp32).mean() / dr.mean()):.3f}, "
            f"M_direct {((df.d_moe_direct - df.d_corrupt_fp32).mean() / dr.mean()):.3f}, DLA attn share {df.dla_attn.mean() / df.dla_total.mean():.3f}; "
            f"fp32 vs engine Delta clean max |diff| {(df.d_clean_fp32 - df.delta_clean_engine).abs().max():.3f}")
    else:
        write_section()



# =================================================================================================================
# task 4: three-task table and the section
# =================================================================================================================
MODEL_LAB = {"qwen3": "Qwen3-30B-A3B-Base", "mixtral_bos": "Mixtral-8x7B, BOS"}


def _f(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _rci(r, d=2):
    """(ratio, lo, hi) or {"ratio", "lo", "hi"} -> 'x [lo, hi]'."""
    if r is None:
        return "pending"
    if isinstance(r, dict):
        r = (r.get("ratio"), r.get("lo"), r.get("hi"))
    try:
        return f"{float(r[0]):.{d}f} [{float(r[1]):.{d}f}, {float(r[2]):.{d}f}]"
    except (TypeError, ValueError):
        return "pending"


def _ext8_w4(task_key: str, m: str) -> Optional[dict]:
    p = os.path.join(RESULTS, "ext8_addback_summary.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    return (s.get(task_key) or {}).get(m)


def three_task_rows(S: dict) -> list[dict]:
    """One row per (model, task): drop, MoE / attention / block peaks (layer, validation rescue / drop), attention share of
    the positive rescue (AUC+), W4 (all-attention A, all-MoE M, phi_attn, redundancy; denoise and noise) and the
    direct-path split (A_direct, M_direct; DLA attention share)."""
    rows = []
    for m in PROTOS:
        # CounterFact STR (task 1 + ext8 W4)
        cf = (S.get("cf") or {}).get(m, {}).get("STR (donor mean)")
        if cf:
            pk = cf["peaks"]
            w4 = _ext8_w4("w4_counterfact", m)
            row = {"model": MODEL_LAB[m], "task": "CounterFact STR (subject swap, ext6 donors)", "n_val": f"{(S['cf'][m]['n_val'])} cases",
                   "drop": cf["val_mean_drop"], "moe_peak": (pk["layer"]["L_disc"], pk["layer"]["val_norm"]),
                   "attn_peak": (pk["attn_layer"]["L_disc"], pk["attn_layer"]["val_norm"]), "block_peak": (pk["block"]["L_disc"], pk["block"]["val_norm"]),
                   "share": (cf["share"]["share"], cf["share"]["share_lo"], cf["share"]["share_hi"]), "src_w4": "ext8-addback (validation)",
                   "sum_attn": cf.get("layer_sum_norm", {}).get("attn_layer"), "sum_moe": cf.get("layer_sum_norm", {}).get("layer")}
            if w4:
                row.update({"A": w4.get("A_all_attention"), "M": w4.get("M_all_moe"), "phi_attn": w4.get("phi_attn"),
                            "red": w4.get("redundancy_A_plus_M_minus_1"), "M_noise": (w4.get("noising") or {}).get("M"),
                            "direct": w4.get("direct_path_exact_norm"), "dla": w4.get("dla_split")})
            rows.append(row)
        # WinoGrande option swap (ext7-wino run), role swap, IOI
        role = (S.get("role") or {}).get(m)
        items = []
        if role and role.get("option_swap"):
            items.append(("WinoGrande option swap (ext7-wino)", role["option_swap"], f"wino_{m}_str", "w4_winogrande"))
        if role:
            items.append(("WinoGrande role swap (W7)", role, f"wino_role_{m}_str", None))
        for corr, lab in (("s2io", "IOI (i) S2 -> IO"), ("s1io", "IOI (ii) S1, IO -> other names")):
            x = (S.get("ioi") or {}).get(f"{m}:{corr}")
            if x:
                items.append((lab, x, f"ioi_{m}_{corr}", None))
        for lab, x, run, ext8key in items:
            pk = x["peaks"]
            row = {"model": MODEL_LAB[m], "task": lab, "n_val": f"{x['n_val_pairs']} pairs" + (" (d = 0 only)" if x["n_val_directed"] == x["n_val_pairs"] else ""),
                   "drop": x["drop"]["mean"], "moe_peak": (pk["layer"]["L_sel"], pk["layer"]["norm_at_L_sel"][0]),
                   "attn_peak": (pk["attn_layer"]["L_sel"], pk["attn_layer"]["norm_at_L_sel"][0]),
                   "block_peak": (pk["block"]["L_sel"], pk["block"]["norm_at_L_sel"][0]),
                   "share": (x["share"]["share"], x["share"]["share_lo"], x["share"]["share_hi"]),
                   "sum_attn": (x.get("layer_sum_norm") or {}).get("attn_layer"), "sum_moe": (x.get("layer_sum_norm") or {}).get("layer")}
            jp = os.path.join(RESULTS, run, "joint_rows.parquet")
            if os.path.exists(jp):
                j = joint_summary(run, "main")
                dn, no = j.get("denoise_all"), j.get("noise_all")
                if dn:
                    row.update({"A": dn["A"], "M": dn["M"], "phi_attn": dn["phi_attn"], "red": dn["redundancy"], "src_w4": f"{run}/joint_rows (main)"})
                if no:
                    row["M_noise"] = no["M"]
            dp = os.path.join(RESULTS, run, "direct_split.parquet")
            if os.path.exists(dp):
                ds = direct_summary(run, "main")
                row["direct"] = {"A_direct": ds["A_direct"], "M_direct": ds["M_direct"]}
                row["dla"] = {"attn": ds["dla_attn_share"], "moe": ds["dla_moe_share"]}
            elif ext8key:
                w4 = _ext8_w4(ext8key, m)
                if w4:
                    row["direct"], row["dla"] = w4.get("direct_path_exact_norm"), w4.get("dla_split")
                    row.setdefault("A", w4.get("A_all_attention")); row.setdefault("M", w4.get("M_all_moe"))
                    row.setdefault("phi_attn", w4.get("phi_attn")); row.setdefault("src_w4", "ext8-addback (validation)")
            rows.append(row)
    return rows


def _direct_fmt(d) -> str:
    """Direct-path split -> 'A_dir x / M_dir y' from either this module's or ext8's structure."""
    if not d:
        return "pending"
    if isinstance(d, dict):
        a = d.get("A_direct", d.get("a_dir", d.get("attn")))
        mo = d.get("M_direct", d.get("m_dir", d.get("moe")))
        if a is not None and mo is not None:
            return f"{_rci(a)} / {_rci(mo)}"
    return str(d)[:60]


def three_task_table(S: dict) -> tuple[pd.DataFrame, str]:
    rows = three_task_rows(S)
    md = ["| Model | Task (STR site) | Val. set | Mean drop | MoE peak: L, rescue/drop | Attention peak: L, rescue/drop | Block peak: L, rescue/drop | "
          "Attention share of the positive rescue (AUC+) | Σ_l attention / Σ_l MoE (signed, / drop) | W4 all-attention A / all-MoE M (denoise) | "
          "W4 φ_attn = ½[A + 1 − M] | W4 M (noising) | Direct path: A_dir / M_dir |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    flat = []
    for r in rows:
        mp, ap, bp = r["moe_peak"], r["attn_peak"], r["block_peak"]
        w4a = f"{_rci(r.get('A'))} / {_rci(r.get('M'))}" if r.get("M") is not None else "pending"
        md.append(f"| {r['model']} | {r['task']} | {r['n_val']} | {r['drop']:+.2f} | L{mp[0]}: {mp[1]:.3f} | L{ap[0]}: {ap[1]:.3f} | L{bp[0]}: {bp[1]:.3f} | "
                  f"{_rci(r['share'])} | {_f(r.get('sum_attn'), 2)} / {_f(r.get('sum_moe'), 2)} | {w4a} | {_rci(r.get('phi_attn'))} | "
                  f"{_rci(r.get('M_noise'))} | {_direct_fmt(r.get('direct'))} |")
        flat.append({"model": r["model"], "task": r["task"], "n_val": r["n_val"], "drop": r["drop"], "moe_peak_layer": mp[0], "moe_peak_norm": mp[1],
                     "attn_peak_layer": ap[0], "attn_peak_norm": ap[1], "block_peak_layer": bp[0], "block_peak_norm": bp[1],
                     "attn_share": r["share"][0], "attn_share_lo": r["share"][1], "attn_share_hi": r["share"][2],
                     "sum_attn_norm": r.get("sum_attn"), "sum_moe_norm": r.get("sum_moe"),
                     "w4_A": json.dumps(r.get("A"), default=float), "w4_M": json.dumps(r.get("M"), default=float),
                     "w4_phi_attn": json.dumps(r.get("phi_attn"), default=float), "w4_M_noise": json.dumps(r.get("M_noise"), default=float),
                     "direct": json.dumps(r.get("direct"), default=float), "dla": json.dumps(r.get("dla"), default=float)})
    return pd.DataFrame(flat), "\n".join(md) + "\n"


def _tab(name: str) -> str:
    p = os.path.join(RESULTS, "tables", name)
    return open(p).read().strip() if os.path.exists(p) else "(pending)"


def write_section(path: Optional[str] = None) -> str:
    """results/sections/ext7_controls.md from results/ext7_controls_summary.json, the tables written by
    scripts/ext7_{cf,role,ioi}_analyze.py and the three-task table (built here); prose in render_section."""
    S = json.load(open(SUMMARY))
    tt, tt_md = three_task_table(S)
    tt.to_csv(os.path.join(RESULTS, "tables", "ext7_controls_three_task.csv"), index=False)
    open(os.path.join(RESULTS, "tables", "ext7_controls_three_task.md"), "w").write(tt_md)
    S["three_task"] = tt.to_dict("records")
    json.dump(S, open(SUMMARY, "w"), indent=1, default=float)
    three_task_figure(tt)
    md = render_section(S, tt, tt_md)
    path = path or os.path.join(RESULTS, "sections", "ext7_controls.md")
    open(path, "w").write(md)
    print(f"wrote {path} ({len(md)} chars)")
    return md


def _p(s: Optional[dict], d: int = 3) -> str:
    """pair/case summary dict -> '+x [lo, hi]'."""
    if not s or s.get("n", 0) == 0:
        return "n/a"
    return f"{s['mean']:+.{d}f} [{s['ci_lo']:+.{d}f}, {s['ci_hi']:+.{d}f}]"


def _gpu_minutes() -> dict:
    """GPU seconds of this agent's jobs from logs/gpu_queue.log (job names ext7c-*), by group."""
    p = os.path.join(ROOT, "logs", "gpu_queue.log")
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        mt = re.search(r"END\s+(ext7c-\S+) rc=(\d+) \((\d+)s\)", line)
        if mt:
            name, sec = mt.group(1), int(mt.group(3))
            grp = name.split("-")[1]
            out[grp] = out.get(grp, 0) + sec
    return out


def render_section(S: dict, tt: pd.DataFrame, tt_md: str) -> str:
    S = {**S, **prose(S, tt), **TEXT}
    cf, role, ioi = S.get("cf", {}), S.get("role", {}), S.get("ioi", {})
    rd, idata = S.get("role_data", {}), S.get("ioi_data") or {}
    gpu = _gpu_minutes()
    gpu_tot = sum(gpu.values()) / 60
    L = []
    A = L.append

    def row(m, task_prefix):
        x = tt[(tt.model == MODEL_LAB[m]) & tt.task.str.startswith(task_prefix)]
        return None if x.empty else x.iloc[0]

    # ---------------------------------------------------------------- summary
    shares = {}
    for m in PROTOS:
        for key, pre in (("cf", "CounterFact"), ("opt", "WinoGrande option"), ("role", "WinoGrande role"), ("ioi1", "IOI (i)"), ("ioi2", "IOI (ii)")):
            r = row(m, pre)
            if r is not None:
                shares[(m, key)] = r
    def sh(m, k):
        r = shares.get((m, k))
        return "pending" if r is None else f"{r.attn_share:.2f} [{r.attn_share_lo:.2f}, {r.attn_share_hi:.2f}]"
    A("**Summary.** Calibration tasks for the WinoGrande STR study (agent ext7-controls): the same final-position patches, with "
      "symmetric token replacement only (no Gaussian noise) and Δ = LD(r, r′), on CounterFact (subject swap), WinoGrande (option "
      "swap and role swap) and IOI (S2 → IO and S1, IO → other names), in Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS. The "
      "question is whether WinoGrande's repair runs through attention, as in IOI, or through the MoE sublayers, as the paper reads "
      "factual recall. " + S.get("summary_reading", ""))
    A("")
    # ---------------------------------------------------------------- what was run
    A("### What was run")
    A("")
    cfq, cfm = cf.get("qwen3", {}), cf.get("mixtral_bos", {})
    A("- **CounterFact STR attention sweep (task 1).** The Direction-6 STR cases and donors (`results/qwen3_str`, "
      "`results/mixtral_bos_str`: paper discovery/validation IDs with ≥ 1 known same-relation donor, up to five donors per case, "
      f"{cfq.get('n_donor_rows', '?')} / {cfm.get('n_donor_rows', '?')} donor rows) re-run with the kinds `attn_layer` (attention-sublayer "
      "output at the final position := clean; the MoE of that layer recomputes), `layer` (MoE output, repeated in the same pass) and "
      "`block` (both) at every layer, parent = donor row (`scripts/ext7_cf_attnsweep.py`; runs `results/{qwen3,mixtral_bos}_str_attnsweep`, "
      "donor-level rows). Per-case values are donor means (primary) or the first donor (sensitivity); the GN comparison uses the "
      "Direction-2b sweeps `results/{qwen3,mixtral_bos}_bos_attnsweep` restricted to the same cases and split. Statistics are those "
      "of `moetrace/ext2_attn.py` (peaks, AUC+ attention share, additivity), applied through an adapter (`ext7_controls.StrAttnRun`).")
    f7 = {m: (rd.get(m) or {}).get("funnel", {}) for m in PROTOS}
    s7 = {m: (rd.get(m) or {}).get("scan") or {} for m in PROTOS}
    c7 = {m: (rd.get(m) or {}).get("case_sets") or {} for m in PROTOS}
    A("- **W7 WinoGrande role swap.** From each model's WinoGrande margin pool (`results/wino_<proto>/scan_pairs.parquet`, margin both "
      "ways, both options capitalised names) the twins whose two names are each mentioned exactly once before the blank and not after "
      "it; the role-swapped prompt exchanges those two first mentions and keeps the filled option (\"Dennis helped Adam … since Dennis "
      "was the\" → \" trainer\"; \"Adam helped Dennis … since Dennis was the\" → \" student\"). Two STR pairs per twin, (A, swap(A)) with "
      "r = trig_a and (B, swap(B)) with r = trig_b, kept when token-symmetric (same length, the two mention spans at the same positions, "
      "identical tokens elsewhere, the same single-token trigger ids after both prompts). "
      f"Qwen3: {f7['qwen3'].get('margin_names', '?')} name twins → {f7['qwen3'].get('role_pairs', '?')} role pairs "
      f"({f7['qwen3'].get('A:R3_length_mismatch', 0)} twins fail symmetry: a sentence-initial name tokenises differently from a "
      f"mid-sentence one in Qwen3's BPE) → {s7['qwen3'].get('margin', '?')} pass the margin both ways (Δ ≥ 1 on the clean prompt, ≤ −1 on "
      f"the swapped one; {s7['qwen3'].get('margin_twins', '?')} twins); Mixtral BOS: {f7['mixtral_bos'].get('margin_names', '?')} → "
      f"{f7['mixtral_bos'].get('role_pairs', '?')} → {s7['mixtral_bos'].get('margin', '?')} ({s7['mixtral_bos'].get('margin_twins', '?')} twins). "
      "Case sets (`data/wino_role/case_sets_<proto>.json`, contract format): random.Random(0) shuffle over twins (both role pairs of a "
      f"twin stay in one split) → 128/128 discovery/validation pairs in both models (Mixtral just reaches 256), replication 128/128 for "
      f"Qwen3 only. The option swap of the same model (agent ext7-wino, `results/wino_<proto>_str`, its shared 776-pair set) provides "
      "the fixed hypotheses (its discovery peak layers).")
    A(f"- **W8 IOI.** The 15 BABA templates of Wang et al. (2023) and their ABBA versions, names, places and objects copied from "
      "Easy-Transformer `easy_transformer/ioi_dataset.py` (fetched 2026-10-04), restricted to words that are one leading-space token "
      f"under both tokenizers ({len(idata.get('names', [])) if isinstance(idata.get('names'), list) else '65'} of 99 names, 8 places, "
      "6 of 8 objects: \" necklace\" and \" snack\" split); 1,600 items (seed 0; `data/ioi/`). Two STR corruptions as in Zhang & Nanda: "
      "(i) S2 → IO (\"… Mary gave a drink to\" → \" John\"; the corrupted prompt is an IOI sentence whose answer is S; both directions, "
      "primary) and (ii) S1 and IO → two other random names, S2 kept (Appendix F; the corrupted prompt has no answer among IO and S, so "
      "only d = 0, and its Δ is the residual preference for IO over S, mean −1.6 to −1.7). Competence: {idata.get('pool_s2io_both_models', '?')} of 1,600 items pass (i)'s "
      "margin both ways in both models; one seed-0 split of these items (128/128 + replication 128/128) serves both corruptions and "
      "both models.")
    A("- **Runs.** W2 final-position sweeps (`layer`, `attn_layer`, `block` at every layer) with agent ext7-wino's generic STR-pair "
      "runner (`scripts/ext7_wino_sweep.py`), W5 per-head patches on IOI (`scripts/ext7_wino_heads.py`, top-4 discovery attention "
      "layers + one null layer), W3 position × layer grids (window 1, kinds `layer` and `attn_layer`, first 64 validation pairs) at named "
      "positions only (`scripts/ext7_{role,ioi}_grid.py`: the grid executor of `scripts/ext7_wino_grid.py` restricted to the two "
      "exchanged mentions, the filled option and the final position, or to S1, IO, S2, the tokens after S2 and the final position), "
      "W4 joint decomposition (`scripts/ext7_wino_joint.py`, ext8 `multi` steps) and its direct-path complement (`python -m "
      "moetrace.ext7_controls direct`, prefill only). Final-position attention of every head on IO / S1 / S2 for the IOI items: "
      "`scripts/ext7_ioi_attn.py`.")
    v = S.get("verify_cf") or {}
    A(f"- **Verification.** `results/verify_ext7_controls_cf_olmoe.json`: on OLMoE, 12 CounterFact STR (case, donor) pairs × 16 "
      "layers against transformers hooks (self_attn / MoE outputs replaced at the final position on the donor run): rescue r "
      f"{_f(v.get('attn_layer_rescue_corr_vs_hf'), 3, False)} / {_f(v.get('block_rescue_corr_vs_hf'), 3, False)} / "
      f"{_f(v.get('layer_rescue_corr_vs_hf'), 3, False)} (attn_layer / block / layer), max |ΔΔ| {_f(v.get('attn_layer_maxdiff'), 2, False)} / "
      f"{_f(v.get('block_maxdiff'), 2, False)} / {_f(v.get('layer_maxdiff'), 2, False)}, identity (clean parent) ≤ "
      f"{_f(v.get('attn_layer_identity_maxdiff'), 2, False)}; the GN calibration of Direction 2b had r 0.990 / 0.994 / 0.983. The pair "
      "runner, the grid executor and the W4 multi steps were verified by agents ext7-wino (`results/verify_ext7_wino_olmoe.json`) and "
      "ext8-addback (`results/verify_ext8_engine_olmoe.json`); the restricted grids use the same executor with a different unit list; "
      "the direct-path split was smoke-tested on OLMoE WinoGrande pairs (fp32 Δ vs engine Δ within 0.06).")
    A(f"- **GPU.** ≈ {gpu_tot:.0f} min in {len(gpu)} job groups ({', '.join(f'{k} {v / 60:.0f}' for k, v in sorted(gpu.items()))} min), "
      "shared with two other agents through `scripts/gpu_queue.sh`.")
    A("")
    # ---------------------------------------------------------------- three-task table
    A("### Three tasks on one attention–MoE axis")
    A("")
    A("**Final-position sublayer attribution per task (validation; rescue / drop = mean rescue over mean drop; W4 = all layers at once)**")
    A("")
    A(tt_md.strip())
    A("")
    A("![Attention share of the positive final-position rescue per task (left) and drop-normalised attention / MoE peaks (right)]"
      "(../figures/ext7_controls_three_task.png)")
    A("")
    A(S.get("three_task_notes", ""))
    A("")
    # ---------------------------------------------------------------- CF
    A("### CounterFact under STR: attention and MoE at the final position")
    A("")
    A("**Peaks of the validation rescue curves (L* on discovery; / drop = rescue over the mean validation drop)**")
    A("")
    A(_tab("ext7_controls_cf_peaks.md"))
    A("")
    A("**Attention share of the positive rescue (AUC+) and at the peak layers (ratio of validation means, paired bootstrap)**")
    A("")
    A(_tab("ext7_controls_cf_share.md"))
    A("")
    A("**Additivity at the peak layers: block vs attention + MoE (validation)**")
    A("")
    A(_tab("ext7_controls_cf_additivity.md"))
    A("")
    A("![CounterFact STR vs GN, attention / MoE / block, normalised by the mean drop](../figures/ext7_controls_cf_curves.png)")
    A("")
    A(S.get("cf_notes", ""))
    A("")
    # ---------------------------------------------------------------- role
    A("### WinoGrande role swap (W7)")
    A("")
    A("**Final-position peaks, role swap vs option swap (validation pairs; pair bootstrap)**")
    A("")
    A(_tab("ext7_controls_role_w2.md"))
    A("")
    A("**Fixed hypotheses: the option swap's discovery peak layers evaluated on the role-swap validation pairs**")
    A("")
    A(_tab("ext7_controls_role_fixed.md"))
    A("")
    A("**Position × layer grid at the exchanged mentions, the filled option and the final position (first 64 validation pairs; "
      "summed over a class's tokens, / mean drop)**")
    A("")
    A(_tab("ext7_controls_role_grid.md"))
    A("")
    A("![Role swap vs option swap, normalised final-position curves](../figures/ext7_controls_role_curves.png)")
    A("")
    A("![Role swap grid](../figures/ext7_controls_role_grid.png)")
    A("")
    A(S.get("role_notes", ""))
    A("")
    # ---------------------------------------------------------------- IOI
    A("### IOI (W8)")
    A("")
    A("**Final-position peaks (validation; main = discovery/validation, rep = replication split)**")
    A("")
    A(_tab("ext7_controls_ioi_w2.md"))
    A("")
    A("**Attention heads at the top attention layers (validation; z over all scanned heads, detection at |z| ≥ 2 on validation AND "
      "discovery; attention = final-position attention probability on the position, d = 0 view: clean prompt / corrupted prompt)**")
    A("")
    A(_tab("ext7_controls_ioi_heads.md"))
    A("")
    A("**Position × layer grid (first 64 validation items; classes summed over their tokens, / mean drop)**")
    A("")
    A(_tab("ext7_controls_ioi_grid.md"))
    A("")
    A("![IOI final-position curves](../figures/ext7_controls_ioi_curves.png)")
    A("")
    A("![IOI grid](../figures/ext7_controls_ioi_grid.png)")
    A("")
    A(S.get("ioi_notes", ""))
    A("")
    # ---------------------------------------------------------------- reading / caveats / files
    A("### Reading")
    A("")
    A(S.get("reading", "(pending)"))
    A("")
    A("### Caveats")
    A("")
    A(S.get("caveats", "(pending)"))
    A("")
    A("### Files")
    A("")
    A("Code: `moetrace/ext7_controls.py` (role-swap and IOI builders, case sets, CounterFact adapter, pair-run summaries, restricted "
      "grid runner, direct-path split, three-task table, this section: `python -m moetrace.ext7_controls section`); scripts "
      "`scripts/ext7_cf_{verify,attnsweep,analyze}.py`, `scripts/ext7_cf_chain.sh`, `scripts/ext7_role_{build,scan,casesets,grid,analyze}.py`, "
      "`scripts/ext7_ioi_{build,scan,casesets,attn,grid,analyze}.py`, chains `scripts/ext7_role_ioi_scan_chain.sh`, "
      "`scripts/ext7_ioi_attn_chain.sh`, `scripts/ext7_role_ioi_chain.sh` (W2, W5, W3), `scripts/ext7_role_ioi_grids_all.sh` (W3 in "
      "one GPU job), `scripts/ext7_role_ioi_w4_chain.sh`, `scripts/ext7_role_ioi_direct_chain.sh` (the runners themselves are agent ext7-wino's `scripts/ext7_wino_{sweep,heads,joint}.py`). "
      "Data: `data/wino_role/` (pairs, funnels, case sets incl. `_grid` subsets), `data/ioi/` (items, pairs per corruption and model, "
      "case sets, pool). Runs: `results/{qwen3,mixtral_bos}_str_attnsweep`, `results/wino_role_<proto>` (scan), "
      "`results/wino_role_<proto>_str` (W2, W4, direct split), `results/wino_role_<proto>_grid`, `results/ioi_<proto>` (scan, "
      "attn_names.npz), `results/ioi_<proto>_<s2io|s1io>` (W2, W5, W4, direct split), `results/ioi_<proto>_<corr>_grid`; "
      "verification `results/verify_ext7_controls_cf_olmoe.json`; tables `results/tables/ext7_controls_*`; figures "
      "`results/figures/ext7_controls_*`; numbers `results/ext7_controls_summary.json`.")
    return "\n".join(L) + "\n"


def three_task_figure(tt: pd.DataFrame):
    """Attention share of the positive final-position rescue (AUC+) per task and model, with 95% CIs, and the
    drop-normalised attention / MoE peaks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["IOI (i)", "IOI (ii)", "CounterFact", "WinoGrande role", "WinoGrande option"]
    lab = {"IOI (i)": "IOI S2→IO", "IOI (ii)": "IOI S1,IO→names", "CounterFact": "CounterFact STR", "WinoGrande role": "WinoGrande role swap",
           "WinoGrande option": "WinoGrande option swap"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    cols = {"Qwen3-30B-A3B-Base": "#4269d0", "Mixtral-8x7B, BOS": "#efb118"}
    labelled = set()
    for k, (m, c) in enumerate(cols.items()):
        x = tt[tt.model == m]
        for i, o in enumerate(order):
            r = x[x.task.str.startswith(o)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = i + (k - 0.5) * 0.25
            axes[0].errorbar(r.attn_share, y, xerr=[[r.attn_share - r.attn_share_lo], [r.attn_share_hi - r.attn_share]], fmt="o", color=c,
                             label=None if m in labelled else m)
            labelled.add(m)
            axes[1].scatter(r.attn_peak_norm, y, marker="o", color=c)
            axes[1].scatter(r.moe_peak_norm, y, marker="s", facecolors="none", edgecolors=c)
    for ax in axes:
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([lab[o] for o in order], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="grey", lw=0.5)
    axes[0].axvline(0.5, color="grey", lw=0.5, ls=":")
    axes[0].set_xlabel("attention share of the positive rescue (AUC+), 95% CI")
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].set_xlabel("peak rescue / drop: attention (filled) and MoE (open)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "ext7_controls_three_task.png"), dpi=130)
    plt.close(fig)


def _get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _pk(x: dict, kind: str) -> str:
    """'L45 +4.16 [3.97, 4.35] (0.329 of the drop)' from a pair_run_summary peak."""
    p = _get(x, "peaks", kind)
    if not p:
        return "pending"
    v, n = p["val_at_L_sel"], p["norm_at_L_sel"]
    return f"L{p['L_sel']} {v['mean']:+.2f} [{v['ci_lo']:+.2f}, {v['ci_hi']:+.2f}] ({n[0]:.3f} of the drop)"


def _share(x: dict) -> str:
    s = _get(x, "share")
    return "pending" if not s else f"{s['share']:.2f} [{s['share_lo']:.2f}, {s['share_hi']:.2f}]"


def _w4(x: dict, direction: str = "denoise_all") -> str:
    j = _get(x, "joint", direction)
    if not j:
        return "pending"
    return f"A {j['A'][0]:.2f}, M {j['M'][0]:+.2f} [{j['M'][1]:+.2f}, {j['M'][2]:+.2f}]"


def prose(S: dict, tt: pd.DataFrame) -> dict:
    cf, role, ioi = S.get("cf", {}), S.get("role", {}), S.get("ioi", {})
    out = {}
    q, mx = "qwen3", "mixtral_bos"

    def cfp(m, corr, kind):
        p = _get(cf, m, corr, "peaks", kind)
        return None if not p else p

    # ---- three-task notes (W4 degeneracy, direct path)
    out["three_task_notes"] = (
        "Columns: MoE / attention / block peak = discovery argmax layer (CounterFact: paper discovery split, donor mean) and its "
        "validation rescue over the mean validation drop. Attention share = AUC+(attention) / (AUC+(attention) + AUC+(MoE)) of the "
        "validation mean curves (Direction 2b). W4 = all layers at once at the final position (`multi` rows; validation directed "
        "cases; denoise = clean values into the corrupted run, noise = corrupted values into the clean run). **The W4 two-player "
        "split is degenerate at the final position:** the final token is the same in both prompts and the MoE is a per-token "
        "function, so patching every attention output at the final position reproduces the source run's final residual exactly "
        "(A = 1 up to bf16, both directions; also noted by agent ext8-addback in the engine docstring). Hence φ_attn = ½[A + 1 − M] = "
        "1 − M/2 and redundancy = M; the informative number is M, the fraction of the drop that the MoE writes at the final "
        "position restore when the attention writes stay corrupted (M < 0: the clean MoE writes push further toward the corrupted "
        "answer). The direct-path split is the non-degenerate complement: A_dir (M_dir) = change of Δ when only the summed "
        "attention (MoE) writes of all layers at the final position are swapped into the corrupted final residual, exact final "
        "RMSNorm, fp32, over the drop (prefill only; A_dir + M_dir ≈ 1 up to the norm's non-linearity). CounterFact W4 and "
        "direct-path numbers are agent ext8-addback's (`results/ext8_addback_summary.json`, validation cases); the "
        "WinoGrande option-swap W4 and direct-path numbers come from agent ext7-wino's run `results/wino_<proto>_str` "
        "(`joint_rows.parquet`, `direct_split.parquet`, computed with the same code), summarised here on its validation pairs.")
    # ---- CF notes
    def cfl(m):
        a, mo, b = cfp(m, "STR (donor mean)", "attn_layer"), cfp(m, "STR (donor mean)", "layer"), cfp(m, "STR (donor mean)", "block")
        ga, gm = cfp(m, "GN (same cases)", "attn_layer"), cfp(m, "GN (same cases)", "layer")
        sh, gsh = _get(cf, m, "STR (donor mean)", "share"), _get(cf, m, "GN (same cases)", "share")
        con, cr = _get(cf, m, "layer_consistency_vs_ext6"), _get(cf, m, "curve_r_str_vs_gn")
        if not (a and mo and ga and sh):
            return "pending"
        return (f"attention L{a['L_disc']} {a['val']:+.2f} ({a['val_norm']:.3f} of the drop; GN on the same cases L{ga['L_disc']} "
                f"{ga['val_norm']:.3f}), MoE L{mo['L_disc']} {mo['val']:+.2f} ({mo['val_norm']:.3f}; GN L{gm['L_disc']} {gm['val_norm']:.3f}), "
                f"block L{b['L_disc']} ({b['val_norm']:.3f}); attention share {sh['share']:.2f} [{sh['share_lo']:.2f}, {sh['share_hi']:.2f}] "
                f"(GN {gsh['share']:.2f}); STR–GN curve r {cr['attn_layer']:.2f} / {cr['layer']:.2f} / {cr['block']:.2f} (attention / MoE / "
                f"block); the same-pass MoE rows reproduce the Direction-6 sweep at per-row r {con['r']:.2f} (mean |diff| {con['mean_absdiff']:.2f}, "
                f"max {con['max_absdiff']:.1f}: bf16 batch-composition noise)")
    out["cf_notes"] = (f"- Qwen3: {cfl(q)}.\n- Mixtral BOS: {cfl(mx)}.\n"
                       "- Under STR the Direction-2b picture holds: attention and MoE carry comparable parts of the positive "
                       "rescue, attention peaking a few layers before the MoE (Qwen3 L40 vs L44; Mixtral L18 vs L19–L21), and the "
                       "block equals attention + MoE to within bf16 noise except at Mixtral's shared L18/L19 peak, which is mildly "
                       "super-additive under STR (sub-additive under GN). Normalised by the drop, STR does not shrink the attention "
                       "effect (Qwen3 0.34 vs GN 0.28; Mixtral 0.18 vs 0.19), whereas Mixtral's MoE peak is 25 % lower under STR "
                       "(0.085 vs 0.113), the same direction as Direction 6. Mixtral's attention curve has three near-equal discrete "
                       "peaks, L18 / L19 / L24 at 0.180 / 0.160 / 0.188 of the drop (GN 0.192 / 0.182 / 0.164; plus L15 0.06), so its "
                       "validation argmax (L24) and discovery argmax (L18) differ by a tie, as Direction 2b's mover heads at L18 and "
                       "L24 suggested.")

    # ---- role notes
    def rl(m):
        r = role.get(m)
        if not r:
            return "pending"
        o = r.get("option_swap") or {}
        rep = r.get("replication")
        txt = (f"role swap MoE {_pk(r, 'layer')}, attention {_pk(r, 'attn_layer')}, block {_pk(r, 'block')}; attention share "
               f"{_share(r)}" + (f" (replication split {_share(rep)}, peaks L{rep['peaks']['layer']['L_sel']} / L{rep['peaks']['attn_layer']['L_sel']})" if rep else "")
               + f". Option swap (same model, ext7-wino's run): MoE {_pk(o, 'layer')}, attention {_pk(o, 'attn_layer')}, share {_share(o)}")
        fx = r.get("fixed") or {}
        att = [v for k, v in fx.items() if v["kind"] == "attn_layer" and "attention peak" in k]
        if att:
            v = att[0]
            txt += (f". At the option swap's attention-peak layer L{v['layer']} the role swap's attention patch gives {v['val']['mean']:+.2f} "
                    f"[{v['val']['ci_lo']:+.2f}, {v['val']['ci_hi']:+.2f}] ({v['norm'][0]:.3f} of the drop)")
        j = r.get("joint")
        if j and j.get("denoise_all"):
            txt += f". W4: {_w4(r)} (noising M {j['noise_all']['M'][0]:+.2f})" if j.get("noise_all") else f". W4: {_w4(r)}"
        d = r.get("direct")
        if d:
            txt += f"; direct path A_dir {d['A_direct'][0]:.2f} / M_dir {d['M_direct'][0]:.2f}"
        return txt
    out["role_notes"] = f"- Qwen3: {rl(q)}.\n- Mixtral BOS: {rl(mx)}."

    def grid_txt(gp, gs, order):
        if not gp:
            return None
        pk = {(d["cls"], d["kind"]): d for d in gp}
        sm = {(d["cls"], d["kind"]): d["norm"] for d in (gs or [])}
        parts = []
        for c in order:
            a, mo = pk.get((c, "attn_layer")), pk.get((c, "layer"))
            if not a and not mo:
                continue
            seg = f"{c}: MoE peak L{mo['layer']} {mo['norm']:+.3f}" if mo else f"{c}:"
            if a:
                seg += f", attention peak L{a['layer']} {a['norm']:+.3f}"
            if (c, "layer") in sm:
                seg += f" (layer sums {sm.get((c, 'layer'), float('nan')):+.2f} / {sm.get((c, 'attn_layer'), float('nan')):+.2f})"
            parts.append(seg)
        return "; ".join(parts)
    gl = []
    for m in PROTOS:
        t = grid_txt(_get(role, m, "grid_peaks"), _get(role, m, "grid_layer_sums"), ["mention1", "mention2", "filled", "final"])
        if t:
            gl.append(f"- Grid, {MODEL_LAB[m]} (rescue / drop, MoE and attention at the position; layer sums MoE / attention): {t}.")
    out["role_notes"] += ("\n" + "\n".join(gl)) if gl else "\n- Grid: pending."
    if gl:
        out["role_notes"] += (
            "\n- Reading of the grid: at the exchanged mentions only the first layers matter (MoE L0–L2: the identity of the "
            "swapped name token, Z9; Mixtral's L0 MoE at a mention alone restores 0.35–0.38 of the drop). At the filled option, "
            "attention in early-middle layers carries part of the difference — Mixtral L9 restores 0.18 of the drop (layer sum "
            "0.59), Qwen3 L6 0.05 with a MoE contribution at L24 (0.05) — i.e. the option token reads which role its name had in "
            "the first clause before the final position's attention (L13 / L38) and MoE (L19 / L42) complete the repair. The role "
            "swap's attention therefore acts at the option position and at the final position, while its final-position repair "
            "is still MoE-heavy.")

    # ---- IOI notes
    def il(m, corr):
        x = ioi.get(f"{m}:{corr}")
        if not x:
            return "pending"
        txt = (f"drop {x['drop']['mean']:+.2f} (Δ clean {x['delta_clean']['mean']:+.2f}, corrupted {x['delta_corrupt']['mean']:+.2f}); attention "
               f"{_pk(x, 'attn_layer')}, MoE {_pk(x, 'layer')}, block {_pk(x, 'block')}; attention share {_share(x)}")
        sa = x.get("share_at_attn_peak")
        if sa:
            txt += f"; at the attention peak the MoE patch of the same layer gives {sa['moe']:+.2f}"
        rep = x.get("replication")
        if rep:
            txt += f"; replication: attention L{rep['peaks']['attn_layer']['L_sel']} ({rep['peaks']['attn_layer']['norm_at_L_sel'][0]:.3f}), share {_share(rep)}"
        j = x.get("joint")
        if j and j.get("denoise_all"):
            txt += f"; W4 {_w4(x)}" + (f", noising M {j['noise_all']['M'][0]:+.2f}" if j.get("noise_all") else "")
        d = x.get("direct")
        if d:
            txt += f"; direct path A_dir {d['A_direct'][0]:.2f} / M_dir {d['M_direct'][0]:.2f}"
        h = x.get("heads")
        if h:
            det = h.get("detected_2sd") or []
            top = h.get("top5") or []
            if top:
                t0 = top[0]
                txt += (f"; top head L{t0['layer']}H{t0['head']} {t0['val_mean']:+.2f} (z {t0['z_val']:+.1f}, "
                        f"{t0['share_attn_layer']:.0%} of its layer's attention patch")
                if "clean_IO" in t0:
                    txt += (f"; clean attention IO {t0['clean_IO']:.2f} / S1 {t0['clean_S1']:.2f} / S2 {t0.get('clean_S2', float('nan')):.2f}, "
                            f"corrupted IO {t0['corr_IO']:.2f} / S1 {t0['corr_S1']:.2f}")
                txt += ")"
            if det:
                txt += "; ≥ 2 SD on validation and discovery: " + ", ".join(f"L{d_['layer']}H{d_['head']} ({d_['val_mean']:+.2f})" for d_ in det[:8])
        return txt
    out["ioi_notes"] = "\n".join(f"- {MODEL_LAB[m]}, {lab}: {il(m, c)}." for m in PROTOS for c, lab in (("s2io", "(i) S2 → IO"), ("s1io", "(ii) S1, IO → other names")))
    gl = []
    for m in PROTOS:
        for c, lab in (("s2io", "(i)"), ("s1io", "(ii)")):
            x = ioi.get(f"{m}:{c}") or {}
            t = grid_txt(x.get("grid_peaks"), x.get("grid_layer_sums"), ["S1", "IO", "S2", "after_S2", "final"])
            if t:
                gl.append(f"- Grid, {MODEL_LAB[m]} {lab}: {t}.")
    out["ioi_notes"] += ("\n" + "\n".join(gl)) if gl else "\n- Grid: pending."
    if gl:
        out["ioi_notes"] += (
            "\n- Reading of the grid: at the corrupted name tokens the earliest layers carry the token identity (Z9, not "
            "interpreted as computation): Mixtral's L0 MoE output at S2 restores 0.99 of the drop under (i) and at IO 0.64 under "
            "(ii); in Qwen3 the same early effect sits in attention L3 (S2 0.26, IO 0.25). Between S2 and the final token little "
            "is patchable (≤ 0.09 of the drop at any layer). At the final position the single-layer attention patches add up to "
            "the drop (layer sums 0.99–1.12) and the MoE patches to ≤ 0.")
        out["ioi_notes"] += (
            "\n- Heads (W5, final-position `attn_head` patches at the top-4 discovery attention layers + one null layer; detection "
            "= |z| ≥ 2 over all scanned heads on validation and discovery): under (i) Qwen3's largest head L42H11 (+2.25, 67 % of the "
            "L42 attention patch) attends S2 (0.75) in both runs — it reads which name is duplicated, the S-inhibition signal of "
            "Wang et al.; the classic name mover L42H10 (+0.87) attends IO (0.87) and follows the IO name to S1 (0.86) when S2 "
            "becomes IO; L43H29 (+0.80) attends the final token itself; L42H14 (−0.52) is a negative name mover (IO 0.21 → S1 "
            "0.21). Mixtral repeats the pattern under (i): its largest head L21H6 (+1.15, 67 % of the L21 attention patch) attends S2 "
            "(0.61), as do L19H10, L31H5, L16H15 and L16H3 (0.26–0.35 on S2); L19H8 is the name mover (IO 0.69 → S1 0.65 after "
            "S2 → IO) and L22H29 a negative name mover (−0.32; IO 0.31 → S1 0.32). Under (ii) the name movers take over (L42H10 +1.05 = 68 % of L42; L45H9 / H29 / H20 / H22 +0.51 to +0.69, each "
            "with 0.20–0.35 of its attention on IO and the rest on position 0), and so they do in Mixtral (L30H2 +1.02 = 67 % of L30, "
            "L26H6 +0.57, L31H6, L31H18, L30H26; 0.16–0.39 of their attention on IO, the rest mostly on position 0). The S2 reader "
            "matters less under (ii) (Qwen3 L42H11 not detected; Mixtral L21H6 +0.39 vs +1.15): the S2 token is unchanged but no "
            "longer duplicates S1. This is the corruption-site dependence Zhang & Nanda show for GPT-2 small: corrupting S2 puts the "
            "heads that read S2 first, corrupting S1 and IO puts the name movers first (Qwen3 L42H10 is detected under both, the "
            "L45 movers only under (ii); Mixtral L19H8 under (i), the L26/L30/L31 movers under (ii), which is why Mixtral's "
            "attention peak moves from L19 to L30).")

    # ---- summary reading, reading, caveats (numbers from the three-task table)
    def T(m, pre):
        x = tt[(tt.model == MODEL_LAB[m]) & tt.task.str.startswith(pre)]
        return None if x.empty else x.iloc[0]

    def shv(m, pre):
        r = T(m, pre)
        return float("nan") if r is None else float(r.attn_share)

    def M_of(m, pre):
        r = T(m, pre)
        if r is None:
            return None
        try:
            v = json.loads(r.w4_M)
            return v["ratio"] if isinstance(v, dict) else v[0]
        except Exception:
            return None

    def mfmt(m, pre):
        v = M_of(m, pre)
        return "pending" if v is None else f"{v:+.2f}"
    def sums(m, pre):
        r = T(m, pre)
        return (float("nan"), float("nan")) if r is None or r.sum_attn_norm is None else (float(r.sum_attn_norm), float(r.sum_moe_norm))
    def dfmt(m, pre):
        r = T(m, pre)
        try:
            d = json.loads(r.direct)
            a = d.get("A_direct", d.get("a_dir"))
            return f"{(a['ratio'] if isinstance(a, dict) else a[0]):.2f}"
        except Exception:
            return "pending"
    out["summary_reading"] = (
        "**WinoGrande is not IOI-like at the final position; it is the most MoE-heavy of the three tasks, and the two WinoGrande "
        "corruption sites agree.** Four views of the final-position repair put the tasks in the same order IOI < CounterFact < "
        "WinoGrande on the MoE side in both models (Qwen3 / Mixtral; within WinoGrande three of the four make the option swap the "
        "more MoE-heavy site): the attention share of the positive single-layer rescue (Direction-2b AUC+) is IOI S2 → IO "
        f"{shv(q, 'IOI (i)'):.2f} / {shv(mx, 'IOI (i)'):.2f} > CounterFact {shv(q, 'CounterFact'):.2f} / {shv(mx, 'CounterFact'):.2f} > "
        f"WinoGrande role swap {shv(q, 'WinoGrande role'):.2f} / {shv(mx, 'WinoGrande role'):.2f} > option swap "
        f"{shv(q, 'WinoGrande option'):.2f} / {shv(mx, 'WinoGrande option'):.2f}; the summed single-layer MoE patches are "
        f"{sums(q, 'IOI (i)')[1]:+.2f} / {sums(mx, 'IOI (i)')[1]:+.2f} of the drop in IOI (net negative, from the last two or three "
        f"layers), {sums(q, 'CounterFact')[1]:+.2f} / {sums(mx, 'CounterFact')[1]:+.2f} in CounterFact and {sums(q, 'WinoGrande role')[1]:+.2f} / "
        f"{sums(mx, 'WinoGrande role')[1]:+.2f} (role) and {sums(q, 'WinoGrande option')[1]:+.2f} / {sums(mx, 'WinoGrande option')[1]:+.2f} "
        "(option) in WinoGrande; patching every MoE output at the final position at once (W4) restores "
        f"{mfmt(q, 'IOI (i)')} / {mfmt(mx, 'IOI (i)')}, {mfmt(q, 'CounterFact')} / {mfmt(mx, 'CounterFact')}, "
        f"{mfmt(q, 'WinoGrande role')} / {mfmt(mx, 'WinoGrande role')} and {mfmt(q, 'WinoGrande option')} / {mfmt(mx, 'WinoGrande option')} "
        "of the drop; and the summed attention writes carry a direct-path share of "
        f"{dfmt(q, 'IOI (i)')} / {dfmt(mx, 'IOI (i)')}, {dfmt(q, 'CounterFact')} / {dfmt(mx, 'CounterFact')}, {dfmt(q, 'WinoGrande role')} / "
        f"{dfmt(mx, 'WinoGrande role')} and {dfmt(q, 'WinoGrande option')} / {dfmt(mx, 'WinoGrande option')}. IOI, measured with the same "
        "patches, is attention-driven (its single-layer attention patches add up to the whole drop), so the method does see an "
        "attention task as one; its heads behave as in GPT-2 small: with S2 → IO the largest Qwen3 head reads S2 (L42H11, 0.75 of "
        "its attention on S2), with S1, IO → other names the name movers dominate (L42H10, 0.87 on IO, whose attention moves to S1 "
        "when S2 becomes IO; four L45 heads), the corruption-site dependence of Zhang & Nanda's App. F. The attention side of "
        "WinoGrande differs by model: small in Qwen3 (Σ attention "
        f"{sums(q, 'WinoGrande option')[0]:+.2f} option, {sums(q, 'WinoGrande role')[0]:+.2f} role vs CounterFact {sums(q, 'CounterFact')[0]:+.2f}), "
        f"as large as in CounterFact in Mixtral ({sums(mx, 'WinoGrande option')[0]:+.2f} / {sums(mx, 'WinoGrande role')[0]:+.2f} vs "
        f"{sums(mx, 'CounterFact')[0]:+.2f}) but outweighed by the MoE. The role swap (exchange the candidates' first mentions, keep "
        "the filled option) shifts weight toward attention relative to the option swap (Qwen3: an attention peak at L38 that the "
        "option swap lacks; Mixtral: same attention L13 and MoE L19–L20 peaks, slightly larger attention). CounterFact under STR "
        "keeps Direction 2b's split (attention ≈ MoE; attention peaks L40 / L18 before the MoE peaks L44 / L21) with no GN "
        "inflation of the attention effect.")
    out["reading"] = (
        "1. **The hypothesis \"WinoGrande is like IOI\" fails at the final position, in both models and at both corruption sites.** "
        "On every attention–MoE view the order is IOI > CounterFact > WinoGrande. In WinoGrande the MoE patches add up to more than "
        "the drop and the MoE writes alone restore 70–84 % of it; in IOI the attention patches add up to the drop and the MoE writes "
        "restore nothing (or act against the answer). Whatever attention does for WinoGrande (at the filled option in early-middle "
        "layers, e.g. Mixtral L9 in the role swap, and at the final position, Mixtral L13, Qwen3 L38) is followed by MoE sublayers (Qwen3 L41–L42, Mixtral "
        "L19–L20) that carry the larger part of the repair — WinoGrande looks like factual recall with a heavier MoE share, not "
        "like IOI.\n"
        "2. **IOI calibrates the measurement.** The same final-position patches, statistics and models classify IOI as attention-"
        "driven (AUC+ share 0.80–0.92, single-layer MoE peaks ≤ 0.10 of the drop) and recover the GPT-2-small mechanism at the head "
        "level (S2-reading and name-mover heads, negative name movers; Qwen3 L42H14 attends IO 0.21 and has a negative effect), so the "
        "low WinoGrande attention share is a property of the task, not a blind spot. The MoE writes at the final position are net "
        "negative in IOI, from the last two or three layers (Qwen3 L45–L47, Mixtral L30–L31); a smaller negative last-layer MoE "
        "effect is present in every task (e.g. Mixtral L31 −0.15 of the drop in WinoGrande, −0.07 in CounterFact), i.e. a late "
        "MoE that counteracts the answer, as negative name movers do in Wang et al.; in IOI nothing else is on the MoE side.\n"
        "3. **The corruption site changes which components are found (Z7), not the side of the axis.** IOI (ii) (S1, IO → other "
        "names) has a smaller drop than (i) (the corrupted prompt prefers neither name) and, as in Zhang & Nanda's App. F, its top "
        "heads are name movers, whereas (i) puts the S2-reading head first in Qwen3; Mixtral's attention peak moves from L19 under "
        "(i) to L30–L31 under (ii). The WinoGrande role swap has a larger attention share than the option swap as run here (Qwen3 "
        "0.32 vs 0.16, direct-path attention 0.19 vs 0.05), but the two were run on different items (role swap: name twins only). "
        "**Revised in Phase 4 (Direction 12, 4b):** on identical name items the role swap and the option swap give the same "
        "attention share (Qwen3 0.304 vs 0.306, difference −0.002 [−0.015, +0.009]; Mixtral 0.406 vs 0.414) and the same "
        "direct-path split, because the role-swapped prompt is exactly the option-swapped prompt with the two names exchanged. "
        "The difference is an item effect (names lean toward attention, objects toward the MoE), not a binding-vs-reference effect "
        "of the corruption site; both stay on the MoE side of CounterFact.\n"
        "4. **The W4 Shapley split is not the right summary at the final position.** Restoring every attention output restores the "
        "whole final residual (A = 1 by construction; the MoE is a per-token function), so φ_attn = 1 − M/2 ≥ ½ for any task with "
        "M ≤ 1, and the redundancy A + M − 1 is just M. The per-layer attention share (W2), the all-MoE fraction M and the "
        "direct-path split are the informative quantities; they agree in their ordering of the tasks.\n"
        "5. **CounterFact under STR (task 1).** STR keeps Direction 2b's conclusion (attention and MoE comparable at the final "
        "position, attention first), normalises the attention peak to the same or a larger share of the drop than GN, and lowers "
        "Mixtral's MoE peak (0.085 vs 0.113); Zhang & Nanda's GN inflation does not show at this patch site.")
    out["caveats"] = (
        "All attributions are at the final position: attention patches there measure what the final position reads in a layer; "
        "processing at earlier positions (the option, the mentions, S2) enters only through what it writes into keys and values, "
        "and the position grids (W3, first 64 validation pairs) give the layer-level view at those positions. Head patches are "
        "final-position patches (Zhang & Nanda patch heads at all positions), so duplicate-token and induction heads, which act at "
        "S2, are not visible as heads (S-inhibition-like heads act at the final position and are: Qwen3 L42H11 reads S2). The WinoGrande option swap numbers are agent ext7-wino's run on its shared 776-pair "
        "set, while the role swap uses each model's own pool (name twins only; 213 Qwen3 twins drop out because a sentence-initial "
        "name tokenises differently from a mid-sentence one), so the two corruptions are compared on different items here (Direction 12, 4b, compares them on identical items); the role "
        "swap has no replication split in Mixtral (270 pairs). IOI is easy for both models (≥ 99.6 % of the items pass the margin both "
        "ways), so its drops are large and its CIs narrow; it uses 65 of Wang et al.'s 99 names (single tokens under both tokenizers). "
        "CounterFact here is the Direction-6 STR set (215 / 213 paper cases, donor mean). bf16 batch-composition noise between passes is "
        "≈ 0.1 logits per row on average (same-pass vs Direction-6 MoE rows: r 0.97–0.98, max 2.5), negligible for means over 100+ "
        "pairs.")
    return out


# hand-written prose blocks (filled in after the runs; numbers quoted here were read from results/ext7_controls_summary.json)
TEXT: dict = {}


if __name__ == "__main__":
    _cli()
