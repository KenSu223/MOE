"""ext14 circuit C3 (Phase 6, P6-5, agent p6-circuit): concept circuits of heads + experts at the final token.

Concepts and their tasks (each task = an STR pair set in the ext7 pair-file contract, restricted to the concept's items):
  SIZE_STATE   ss_opt     Phase-3 WinoGrande option swap (entity -> attribute; data/wino_str), answers in the SIZE / STATE
                          lexicon (large / small, empty / full, ...)
               ss_e1e2a   Phase-6 Dial-E E1 entity -> attribute (binding stated verbatim), same lexicon
               ss_e3e2a   Phase-6 Dial-E E3 entity -> attribute (implied by the event), same lexicon (Phase-6 item build)
  COUNTRY      co_P17 / co_P27 / co_P495 / co_P36   CounterFact STR pairs of one relation (scripts/ext14ci_c3_cf_build.py)
  PERSON_ROLE  pr_ioi     IOI, S2 -> IO corruption (data/ioi, Phase 3)
               pr_f1n     Phase-5 F1 attribute -> entity cloze, name items (the answer is a person chosen by role)
               pr_e1a2e   Phase-6 E1 attribute -> entity, name items (if >= 16 validation pairs)
  controls     ct_P103 / ct_P178   CounterFact native language / developer (no concept; specificity columns only)
Per task <= 64 + 64 pairs (discovery / validation, the source split kept) = 128 + 128 directed cases.

Components: attention heads at the final position (kind attn_head) at the model's head layers L_h and experts (kind expert,
clean- or corrupt-active) at the expert layers L_e, both common to every task of a model (chosen from the tasks' sweeps:
the union of each task's top attention / MoE layers by discovery mean single-layer rescue).
  effect e_c(case) = single-component rescue / drop of the case (drop = Delta_clean - Delta_corrupt)
  task top set     = the top-K components by discovery mean effect (K_TOP heads + K_TOP experts)
  concept circuit  = components in the top set of >= 2 tasks of the concept and in no top set of another concept's task
  AUROC(c, concept) on validation cases of all tasks: P(e_c(case of the concept) > e_c(case of another concept)) (+ ties / 2)
Validation (scripts/ext14ci_c3_validate.py): concept x task knockout matrix (heads: PrefillSpec.head_mask, zero; experts:
route_mask reroute, all positions) on the clean rows of every task's validation cases, F = 1 - Delta_ko / Delta_clean;
matched random sets (the same number of heads at the same layers, the same number of experts at the same layers, 5 draws);
add-back of the concept set (multi spawn, heads_experts steps per layer) in every task; edges A -> B among the circuit's
components (layer A < layer B) as the mediated effect rescue(A) + rescue(B) - rescue(A u B) (patching B to its clean value
cuts A's path through B).
"""
from __future__ import annotations

import json
import os
import random
from typing import Optional

import numpy as np
import pandas as pd

from .models import RESULTS

ROOT = "/home/ubuntu/MOE"
DATA = os.path.join(RESULTS, "p6ci_c3_data")
CAP = 64
K_TOP = 10
SIZE = {"big", "small", "large", "tiny", "huge", "little", "wide", "narrow", "tall", "short", "long", "thin", "thick", "heavy",
        "light", "bigger", "smaller", "larger", "massive", "longer", "shorter", "deep", "shallow", "high", "low"}
STATE = {"full", "empty", "wet", "dry", "hot", "cold", "warm", "cool", "open", "closed", "clean", "dirty", "loose", "tight",
         "raw", "cooked", "frozen", "melted", "broken", "fixed", "soft", "hard"}
SS = SIZE | STATE
CONCEPTS = ("SIZE_STATE", "COUNTRY", "PERSON_ROLE")
MIN_VAL = 16  # a task enters C3 with >= 16 validation pairs
# ss_opt (Phase-3 option swap) shares its twins with the Phase-6 E3-e2a cell, so it is not a separate task (kept as a fallback)
TASK_ORDER = ("ss_e3e2a", "ss_e1e2a", "ss_e1pe2a", "co_P17", "co_P27", "co_P495", "co_P36", "pr_ioi", "pr_f1n", "pr_e1a2e",
              "ct_P103")
PROTO_MODEL = {"qwen3": "qwen3", "mixtral_bos": "mixtral"}


def tasks(proto: str) -> dict:
    """task -> dict(concept, pairs, case_sets (source), select (callable on the pair frame -> bool mask), needs)."""
    m = "qwen3" if proto == "qwen3" else "mixtral_bos"
    T = {
        "ss_opt": dict(concept="SIZE_STATE", pairs=f"data/wino_str/pairs_train_xl_{m}.parquet", case_sets="data/wino_str/case_sets.json",
                       sel="ss_words"),
        "ss_e1e2a": dict(concept="SIZE_STATE", pairs=f"data/dial/pairs_e1_e2a_{proto}.parquet", case_sets=f"data/dial/case_sets_e1_e2a_{proto}.json",
                         sel="ss_answers", needs=f"e1_e2a/{proto}"),
        "ss_e1pe2a": dict(concept="SIZE_STATE", pairs=f"data/dial/pairs_e1p_e2a_{proto}.parquet",
                          case_sets=f"data/dial/case_sets_e1p_e2a_{proto}.json", sel="ss_answers", needs=f"e1p_e2a/{proto}"),
        "ss_e3e2a": dict(concept="SIZE_STATE", pairs=f"data/dial/pairs_e3_e2a_{proto}.parquet", case_sets=f"data/dial/case_sets_e3_e2a_{proto}.json",
                         sel="ss_answers", needs=f"e3_e2a/{proto}"),
        "pr_ioi": dict(concept="PERSON_ROLE", pairs=f"data/ioi/pairs_s2io_{m}.parquet", case_sets="data/ioi/case_sets_s2io.json", sel="all"),
        "pr_f1n": dict(concept="PERSON_ROLE", pairs=f"data/wino_ref/pairs_p5_f1_{m}.parquet", case_sets=f"data/wino_ref/case_sets_p5_f1_{m}.json",
                       sel="names"),
        "pr_e1a2e": dict(concept="PERSON_ROLE", pairs=f"data/dial/pairs_e1_a2e_{proto}.parquet", case_sets=f"data/dial/case_sets_e1_a2e_{proto}.json",
                         sel="names", needs=f"e1_a2e/{proto}"),
    }
    for rel, conc in (("P17", "COUNTRY"), ("P27", "COUNTRY"), ("P495", "COUNTRY"), ("P36", "COUNTRY"), ("P103", "CONTROL"), ("P178", "CONTROL")):
        T[("co_" if conc == "COUNTRY" else "ct_") + rel] = dict(concept=conc, pairs=os.path.join(DATA, f"cf_{rel}_{proto}.parquet"),
                                                               case_sets=os.path.join(DATA, f"case_sets_cf_{rel}_{proto}.json"), sel="all")
    return T


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _split_pairs(cs: dict, with_rep: bool = False) -> dict:
    """split -> list of pair_idx from a case-set JSON (pairs {split: [...]} or directed ids); with_rep adds the replication
    family (rep_discovery / rep_validation) to the same split."""
    if "pairs" in cs and isinstance(cs["pairs"], dict) and "discovery" in cs["pairs"]:
        src = cs["pairs"]
        out = {s: [int(i) for i in src[s]] for s in ("discovery", "validation")}
    else:
        src = cs["directed"]
        out = {s: sorted({int(c) // 2 for c in src[s]}) for s in ("discovery", "validation")}
    if with_rep:
        for s in ("discovery", "validation"):
            k = f"rep_{s}"
            if k in src:
                ids = [int(i) for i in src[k]] if src is cs.get("pairs") else sorted({int(c) // 2 for c in src[k]})
                out[s] = sorted(set(out[s]) | set(ids))
    return out


def build_subset(task: str, proto: str, cap: int = CAP) -> Optional[dict]:
    """Writes results/p6ci_c3_data/{pairs,case_sets}_<task>_<proto> restricted to the task's items; returns a summary."""
    spec = tasks(proto)[task]
    pp, cp = _abs(spec["pairs"]), _abs(spec["case_sets"])
    if not (os.path.exists(pp) and os.path.exists(cp)):
        return None
    pairs = pd.read_parquet(pp)
    cs = json.load(open(cp))
    inv = {int(v): k for k, v in cs["pair_idx_of"].items()}
    sel = spec["sel"]
    sp = _split_pairs(cs, with_rep=(sel == "names"))
    pid_ok = set(pairs.pair_id)
    if sel == "ss_words":
        ok = {p for p, a, b in zip(pairs.pair_id, pairs.word_a, pairs.word_b) if str(a).strip().lower() in SS and str(b).strip().lower() in SS}
    elif sel == "ss_answers":
        ca, cb = ("answer_a", "answer_b") if "answer_a" in pairs.columns else ("word_a", "word_b")
        ok = {p for p, a, b in zip(pairs.pair_id, pairs[ca], pairs[cb]) if str(a).strip().lower() in SS and str(b).strip().lower() in SS}
    elif sel == "names":
        ok = set(pairs.pair_id[pairs.names.astype(bool)])
    else:
        ok = pid_ok
    out_idx, splits = {}, {}
    rng = random.Random(29)
    for s in ("discovery", "validation"):
        ids = [i for i in sp[s] if inv.get(i) in ok]
        rng.shuffle(ids)
        ids = sorted(ids[:cap])
        splits[s] = ids
    keep = [inv[i] for s in ("discovery", "validation") for i in splits[s]]
    sub = pairs[pairs.pair_id.isin(keep)].copy()
    new_idx = {p: j for j, p in enumerate(keep)}
    nd = len(splits["discovery"])
    js = {"note": f"p6-circuit C3 task {task} ({spec['concept']}) for {proto}: subset '{sel}' of {spec['pairs']} / {spec['case_sets']}; "
                  f"source split kept, <= {cap} pairs per split (random.Random(29))",
          "pair_idx_of": new_idx,
          "pairs": {"discovery": list(range(nd)), "validation": list(range(nd, len(keep)))},
          "directed": {"discovery": [2 * i + d for i in range(nd) for d in (0, 1)],
                       "validation": [2 * i + d for i in range(nd, len(keep)) for d in (0, 1)]}}
    os.makedirs(DATA, exist_ok=True)
    sub = sub.set_index("pair_id").loc[keep].reset_index()
    sub["concept"] = spec["concept"]
    sub["c3_task"] = task
    sub.to_parquet(os.path.join(DATA, f"pairs_{task}_{proto}.parquet"), index=False)
    json.dump(js, open(os.path.join(DATA, f"case_sets_{task}_{proto}.json"), "w"), indent=1)
    return {"task": task, "proto": proto, "concept": spec["concept"], "n_eligible": len(ok), "discovery": nd,
            "validation": len(keep) - nd}


def pair_paths(task: str, proto: str) -> tuple[str, str]:
    return os.path.join(DATA, f"pairs_{task}_{proto}.parquet"), os.path.join(DATA, f"case_sets_{task}_{proto}.json")


def run_name(task: str, proto: str) -> str:
    return f"p6ci_c3_{task}_{proto}"


# ------------------------------------------------------------------------------------------------------------------
# analysis helpers (CPU)
# ------------------------------------------------------------------------------------------------------------------
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(x_pos > x_neg) + 0.5 P(tie), via ranks."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def comp_name(kind: str, layer: int, idx: int) -> str:
    return f"L{int(layer)}H{int(idx)}" if kind == "head" else f"L{int(layer)}E{int(idx):03d}"
