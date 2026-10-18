"""ext14 (Phase 6, agent p6-analysis): cross-cell analysis of the Phase-6 cells (RESEARCH_PLAN.md "Phase 6").

A *cell* is one task condition (Dial E level x direction, Dial A class, Set C, ladder task, R-site) on one model
(protocol). Cells are registered by p6-data in data/dial/p6_conditions.json ({"conditions": {cell: {proto: entry}}};
entry: pair_file, case_sets, run_dir, status {cases, base}, ...). Every tier-1 run dir has the Phase-3 pair-run schema
(moetrace.ext7_pairs.PairRun: str_sweep_rows / sweep_cases / case_sets.json, direct_split, joint_rows, dla_*, head_rows,
str_grid_w1_rows). The CounterFact anchor of the two Stage-1 models is the donor-level Phase-3 attention sweep
(results/<proto>_str_attnsweep, ext7_cf_attnsweep schema) and is read by a separate loader.

Measures (exactly as Phase 5; scripts/ext13_p5_analyze.cluster_stats, moetrace/ext13syn.stats):
  attention share  AUC+(attention) / (AUC+(attention) + AUC+(MoE or MLP)), AUC+ = sum over layers of the positive part of
                   the mean single-layer rescue curve at the final token (validation pairs, pair means of the two directed
                   cases);
  A_direct, M_direct  direct-path split with the exact final norm / drop (direct_split.parquet);
  M (all-MoE)      every final-position MoE output patched at once, denoising, / drop (joint_rows.parquet);
  hand-off         from the position x layer grid: last layer at which the swapped-token residual still restores >= 0.5
                   of the drop, first layer at which the final-token residual does (ext7_wino_analyze.grid convention);
  heads            single-head rescue at the final token (head_rows.parquet), top heads by discovery mean.
R-metric: the share recomputed under the per-case normalised LD (rescue / drop of the directed case) and under
dp = p(answer | patched) - p(answer | corrupted) (sweep metric columns).

Statistics: cluster bootstrap (5,000 resamples, seed 0) over the cell's independent unit: the WinoGrande twin
(src_pair_id) for the Dial-E / Dial-A / Set-C / R-site cells, the template seed where the pair parquet carries one,
otherwise the pair. Paired comparisons (across levels, twins, a capacity series) restrict both cells to their common
items and apply the SAME unit resamples to both (ext13syn.paired), so the CI of a difference is a paired cluster bootstrap.
Item identity: across Dial-E levels / directions = the WinoGrande twin (src_pair_id); across models of one tokenizer
family = the pair id (identical prompts).
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from . import ext7_pairs as P
from .models import RESULTS

ROOT = os.path.dirname(RESULTS)
REG_JSON = os.environ.get("EXT14_REG", os.path.join(ROOT, "data", "dial", "p6_conditions.json"))
PRED_JSON = os.path.join(ROOT, "data", "dial", "predictions.json")
PROGRESS = os.path.join(ROOT, "logs", "PROGRESS.md")
SUMMARY = os.path.join(RESULTS, "ext14_summary.json")
TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
NB = 5000

# ------------------------------------------------------------------------------------------------------------------
# models / protocols
# ------------------------------------------------------------------------------------------------------------------
STAGE1 = ("qwen3", "mixtral_bos")
STAGE2 = ("mistral7b_bos", "qwen2_7b", "qwen2_57b", "qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3_1p7b")
ALL_PROTOS = STAGE1 + STAGE2
MODEL_OF = {"qwen3": "qwen3", "mixtral_bos": "mixtral", "mistral7b_bos": "mistral7b", "qwen2_7b": "qwen2_7b", "qwen2_57b": "qwen2_57b",
            "qwen3_4b": "qwen3_4b", "qwen3_8b": "qwen3_8b", "qwen3_14b": "qwen3_14b", "qwen3_1p7b": "qwen3_1p7b", "mixtral22_bos": "mixtral22"}
LABEL = {"qwen3": "Qwen3-30B-A3B-Base", "mixtral_bos": "Mixtral-8x7B (BOS)", "mistral7b_bos": "Mistral-7B (BOS)", "qwen2_7b": "Qwen2-7B",
         "qwen2_57b": "Qwen2-57B-A14B", "qwen3_4b": "Qwen3-4B-Base", "qwen3_8b": "Qwen3-8B-Base", "qwen3_14b": "Qwen3-14B-Base",
         "qwen3_1p7b": "Qwen3-1.7B-Base", "mixtral22_bos": "Mixtral-8x22B (BOS)"}
SHORT = {"qwen3": "Qwen3-30B-A3B", "mixtral_bos": "Mixtral-8x7B", "mistral7b_bos": "Mistral-7B", "qwen2_7b": "Qwen2-7B",
         "qwen2_57b": "Qwen2-57B-A14B", "qwen3_4b": "Qwen3-4B", "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "qwen3_1p7b": "Qwen3-1.7B",
         "mixtral22_bos": "Mixtral-8x22B"}
MOE = {"qwen3", "mixtral_bos", "qwen2_57b", "mixtral22_bos"}
N_LAYERS = {"qwen3": 48, "mixtral_bos": 32, "mistral7b_bos": 32, "qwen2_7b": 28, "qwen2_57b": 28, "qwen3_4b": 36, "qwen3_8b": 36,
            "qwen3_14b": 40, "qwen3_1p7b": 28, "mixtral22_bos": 56}
TWINS = (("mistral7b_bos", "mixtral_bos"), ("qwen2_7b", "qwen2_57b"))  # (dense parent, MoE child)
SERIES = {"qwen3": ("qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3"), "mistral": ("mistral7b_bos", "mixtral_bos"), "qwen2": ("qwen2_7b", "qwen2_57b")}
PARAMS_B = {"qwen3_1p7b": 1.7, "qwen3_4b": 4.0, "qwen3_8b": 8.2, "qwen3_14b": 14.8, "qwen3": 30.5, "mistral7b_bos": 7.2, "mixtral_bos": 46.7,
            "qwen2_7b": 7.6, "qwen2_57b": 57.4, "mixtral22_bos": 141.0}
ACTIVE_B = {"qwen3": 3.3, "mixtral_bos": 12.9, "qwen2_57b": 14.2, "mixtral22_bos": 39.0}

# ------------------------------------------------------------------------------------------------------------------
# cells
# ------------------------------------------------------------------------------------------------------------------
LADDER = {  # ladder v2 grades fixed in the pre-registration (prediction L9)
    "l_ant": (1.0, "antonym"), "l_prop": (1.5, "object property"), "e3_e2a": (2.0, "WG entity → attribute (E3)"),
    "l_cf": (2.0, "CounterFact"), "e3_a2e": (3.0, "WG attribute → entity (E3)"), "l_bind": (4.0, "binding look-up"),
    "l_ioi": (5.0, "IOI"), "l_ind": (5.0, "induction (word pairs)"),
}
DIAL_E_LEVELS = ("e1", "e1p", "e2", "e3")
LEVEL_LABEL = {"e0": "E0 prior only", "e1": "E1 stated verbatim", "e1p": "E1p paraphrase", "e2": "E2 negation", "e3": "E3 event (original)",
               "e3p3": "E3 event, P3 frame (frame-matched)"}
# a2e E1-family cells use frame P3 ('{A} and {B}. Fill in the blank: The _ was X.\nAnswer: the', frame pilot 2026-10-16);
# e3p3_a2e = E3 content in the same frame (frame-matched E3 of the paired a2e dial); e3_a2e (F1) stays the Phase-5 anchor.
DIAL_LEVELS = {"a2e": ("e1", "e1p", "e2", "e3p3"), "e2a": ("e1", "e1p", "e2", "e3")}
DIRS = ("a2e", "e2a")
DIR_LABEL = {"a2e": "attribute → entity", "e2a": "entity → attribute"}
SET_A = ("l_ant", "l_prop", "l_cf", "l_ioi", "l_bind", "l_ind", "e1_a2e", "e1_e2a", "e1p_a2e", "e1p_e2a", "e2_a2e", "e2_e2a", "e3_a2e", "e3_e2a")
SET_B = ("aagr_a2e", "aagr_e2a", "acon_a2e", "acon_e2a", "e0_a2e", "e0_e2a")
COMPARE_CELLS = SET_A + ("e3p3_a2e",)  # cross-model comparisons also carry the frame-matched E3 (P3 frame)
MIN_COMMON = 32  # below this many common pairs a cross-model contrast falls back to the models' own validation sets
KNOWLEDGE_CELLS = ("l_ant", "l_prop", "l_cf", "e3_e2a")  # D2 (H-capacity)
COPY_CELLS = ("l_ind", "l_ioi", "l_bind", "e1_a2e", "e1_e2a")
TIER2_CELLS = ("e1_a2e", "e1_e2a")
CAUSAL_CELLS = ("e1_a2e", "e3_e2a")
# Phase-5 references (tier 2 of E3 = the Phase-5 dir_f1 / dir_opt runs)
P5_REF = {"e3_a2e": "dir_f1", "e3_e2a": "dir_opt"}
CELL_LABEL = {
    "l_ant": "L-ant antonym", "l_prop": "L-prop object property", "l_cf": "L-cf CounterFact", "l_ioi": "L-ioi IOI", "l_bind": "L-bind binding",
    "l_ind": "L-ind induction", "rsite_e1_a2e": "R-site E1 a→e (evidence swap)", "sc_agree_a2e": "Set C agree (a→e)",
    "sc_conflict_a2e": "Set C conflict (a→e)", "aagr_a2e": "A-agree a→e", "aagr_e2a": "A-agree e→a", "acon_a2e": "A-conflict a→e",
    "acon_e2a": "A-conflict e→a", "e0_a2e": "E0 prior only a→e", "e0_e2a": "E0 prior only e→a",
}
for _lv in ("e1", "e1p", "e2", "e3"):
    for _d in DIRS:
        CELL_LABEL[f"{_lv}_{_d}"] = f"{_lv.upper().replace('P', 'p')} {'a→e' if _d == 'a2e' else 'e→a'}"
CELL_LABEL["e3_a2e"] = "E3 a→e (F1 frame, Phase-5 anchor)"
CELL_LABEL["e3p3_a2e"] = "E3 a→e (P3 frame)"


def cell_label(c: str) -> str:
    if c.endswith("_all") and c[:-4] in CELL_LABEL:
        return CELL_LABEL[c[:-4]] + " (all strata)"
    if c in CELL_LABEL:
        return CELL_LABEL[c]
    if c.startswith("sc_conflict_a2e"):
        return "Set C conflict, " + ("statement-following" if is_stmt_subset(c) else "event-following" if is_event_subset(c) else c[len("sc_conflict_a2e"):].strip("_"))
    return c


def is_stmt_subset(c: str) -> bool:
    return c.startswith("sc_conflict") and bool(re.search(r"stmt|statement|stat", c))


def is_event_subset(c: str) -> bool:
    return c.startswith("sc_conflict") and "event" in c


# ------------------------------------------------------------------------------------------------------------------
# registry, handshakes
# ------------------------------------------------------------------------------------------------------------------
def registry(path: Optional[str] = None) -> dict:
    """{cell: {proto: entry}} from p6_conditions.json (tolerant: missing file -> {})."""
    path = path or REG_JSON
    if not os.path.exists(path):
        return {}
    try:
        js = json.load(open(path))
    except Exception:  # file being rewritten by p6-data
        import time
        time.sleep(2)
        js = json.load(open(path))
    body = js.get("conditions", js) if isinstance(js, dict) else {}
    out = {}
    for c, v in body.items():
        if c.startswith("_") or not isinstance(v, dict):
            continue
        for p, e in v.items():
            if isinstance(e, dict):
                out.setdefault(c, {})[p] = e
    return out


def entry_of(cell: str, proto: str, reg: Optional[dict] = None) -> Optional[dict]:
    reg = registry() if reg is None else reg
    return reg.get(cell, {}).get(proto)


def ready_lines(tag: str = "p6-data", what: str = "BASE READY") -> set:
    """{(cell, proto)} of the exact PROGRESS handshake lines '<UTC> [p6-data] BASE READY <cell>/<proto>' (only lines whose
    first bracket tag is the given agent)."""
    if not os.path.exists(PROGRESS):
        return set()
    out = set()
    pat = re.compile(r"^\s*(-\s*)?\S+\s+\[" + re.escape(tag) + r"\]\s+" + what + r"\s+([\w.]+)/(\w+)")
    for ln in open(PROGRESS, errors="replace"):
        m = pat.match(ln)
        if m:
            out.add((m.group(2), m.group(3)))
    return out


def voided() -> set:
    if not os.path.exists(PROGRESS):
        return set()
    out = set()
    for ln in open(PROGRESS, errors="replace"):
        m = re.search(r"\[coordinator\]\s+VOID\s+([\w.]+)/(\w+)", ln)
        if m:
            out.add((m.group(1), m.group(2)))
    return out


def _abs(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def run_name(e: dict) -> Optional[str]:
    r = None
    for k in ("run_dir", "run", "out", "base_run"):
        if e.get(k):
            r = e[k]
            break
    if isinstance(r, (list, tuple)):
        r = r[0]
    if not r:
        return None
    r = str(r).rstrip("/")
    if os.path.isabs(r):
        r = os.path.relpath(r, RESULTS)
    if r.startswith("results/"):
        r = r[len("results/"):]
    return r


def is_ready(cell: str, proto: str, reg: Optional[dict] = None, lines: Optional[set] = None) -> bool:
    """Base runs usable: registry status.base true (anchors) and a BASE READY line, or the run dir has the sweep rows."""
    e = entry_of(cell, proto, reg)
    if not e or e.get("behaviour_only") or (cell, proto) in voided():
        return False
    st = e.get("status") or {}
    lines = ready_lines() if lines is None else lines
    if not (st.get("base") or (cell, proto) in lines):
        return False
    r = run_name(e)
    if not r:
        return False
    d = os.path.join(RESULTS, r)
    return os.path.exists(os.path.join(d, "str_sweep_rows.parquet")) or os.path.exists(os.path.join(d, "str_attn_rows.parquet"))


def ready_cells(protos=ALL_PROTOS, reg: Optional[dict] = None) -> list[tuple]:
    reg = registry() if reg is None else reg
    lines = ready_lines()
    return [(c, p) for c in reg for p in reg[c] if p in protos and is_ready(c, p, reg, lines)]


# ------------------------------------------------------------------------------------------------------------------
# bootstrap helpers (ext13syn conventions)
# ------------------------------------------------------------------------------------------------------------------
def unit_weights(units, nb: int = NB, seed: int = 0):
    u = np.unique(np.asarray(units).astype(str))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(nb, len(u)))
    cnt = np.zeros((nb, len(u)))
    np.add.at(cnt, (np.repeat(np.arange(nb), len(u)), idx.ravel()), 1.0)
    return u, np.vstack([np.ones(len(u)), cnt])


def ci(x) -> list:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return [float(x), float("nan"), float("nan")]
    b = x[1:][np.isfinite(x[1:])]
    if not np.isfinite(x[0]) or len(b) < 10:
        return [float(x[0]), float("nan"), float("nan")]
    return [float(x[0]), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def fnum(x, spec: str = ".2f", na: str = "n/a") -> str:
    """None / NaN / inf / non-numeric safe number formatting for tables and section text."""
    if isinstance(x, (list, tuple)):
        x = x[0] if x else None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    if not np.isfinite(v):
        return na
    return format(v, spec)


def fr(x, d: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, (list, tuple)) and (len(x) == 0 or x[0] is None or not np.isfinite(x[0]))):
        return "n/a"
    if not isinstance(x, (list, tuple)):
        return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    f = f"{{:+.{d}f}}" if signed else f"{{:.{d}f}}"
    if len(x) < 3 or not np.isfinite(x[1]):
        return f.format(x[0])
    return f"{f.format(x[0])} [{f.format(x[1])}, {f.format(x[2])}]"


# ------------------------------------------------------------------------------------------------------------------
# cell data
# ------------------------------------------------------------------------------------------------------------------
UNIT_COLS = ("unit", "src_pair_id", "seed_id", "seed", "template_id", "phase3_pair_idx", "base_pair_id")
ITEM_SRC_COLS = ("src_pair_id", "base_pair_id", "phase3_pair_idx")
PRIOR_COLS = ("prior_support", "prior_logodds", "probe_logodds", "prior", "probe_support", "probe_own", "probe_score")


def _pair_table(pair_file: Optional[str]) -> Optional[pd.DataFrame]:
    if not pair_file or not os.path.exists(pair_file):
        return None
    pf = pd.read_parquet(pair_file)
    return pf.drop_duplicates("pair_id").set_index("pair_id", drop=False)


class CellData:
    """Pair-level arrays of one cell restricted to (family, splits).

    A, M     [pairs, L] single-layer attention / MoE(MLP) rescue at the final token (LD), pair means;
    An, Mn   the same per directed case divided by its drop (normalised LD), pair means;
    Ap, Mp   dp = p(answer | patched) - p(answer | corrupted), pair means;
    drop     pair-mean drop (LD); dnum / dden direct attention; mnum direct MoE; jnum / jden all-MoE joint;
    dla_attn / dla_moe / dla_tot  clean - corrupted DLA of the final-position sublayer outputs (direct_split.parquet);
    unit / item  bootstrap unit and item identity per pair.
    """

    def __init__(self, cell: str, proto: str, family: str = "main", splits=("validation",), item_mode: str = "src",
                 entry: Optional[dict] = None, run: Optional[str] = None, pair_file: Optional[str] = None):
        self.cell, self.proto = cell, proto
        e = entry if entry is not None else (entry_of(cell, proto) or {})
        self.entry = e
        self.run = run or run_name(e)
        # CounterFact anchors of Stage 1 (p6-data registry: run_dir = the donor-level Direction-6 run, anchor_runs.sweep_attention =
        # the Phase-3 attention sweep with attention, MoE and block rows) -> the donor-level attention-sweep loader
        aw = ((e.get("anchor_runs") or {}).get("sweep_attention") or "") if run is None else ""
        if aw:
            self.run = aw[len("results/"):] if aw.startswith("results/") else aw
        self.pair_file = pair_file or _abs(e.get("pair_file") or e.get("pairs"))
        od = os.path.join(RESULTS, self.run)
        self.dir = od
        self.kind = "cf" if os.path.exists(os.path.join(od, "str_attn_rows.parquet")) and not os.path.exists(os.path.join(od, "str_sweep_rows.parquet")) else "pair"
        # pooled cells (< 64 validation but >= 64 pairs overall; coordinator rule 2026-10-16): tier 1 on all pairs, no
        # held-out split -> the "validation" summary of such a cell is the pooled set (flagged, never used for prediction tests)
        self.pooled = bool(e.get("pooled"))
        if self.pooled and tuple(splits) == ("validation",):
            splits = ("discovery", "validation")
        self.family, self.splits, self.item_mode = family, tuple(splits), item_mode
        if self.kind == "cf":
            self._load_cf()
        else:
            self._load_pair()

    # ---- pair runs ------------------------------------------------------------------------------------------------
    def _load_pair(self):
        pr = P.PairRun(self.run)
        self.pr = pr
        self.L = pr.L
        fam = self.family
        if fam not in pr.fam:  # e.g. anchors with other family names
            fam = "main" if "main" in pr.fam else next(iter(pr.fam))
        ids = []
        for s in self.splits:
            ids += [int(c) for c in pr.fam[fam].get(s, pr.fam[fam].get("all", []) if s == "validation" and "all" in pr.fam[fam] else [])]
        ids = sorted(set(ids))
        self.ids = ids
        c = pr.cases.loc[ids]
        pidx = np.array(sorted({i // 2 for i in ids}))
        self.pidx = pidx
        g = c.groupby(c.case_id // 2)
        self.pair_id = g.pair_id.first().reindex(pidx).astype(str).to_numpy()
        self.T = g["T"].first().reindex(pidx).to_numpy(np.float64)
        self.n_dir = g.size().reindex(pidx).to_numpy()
        # pair-level columns from the pair parquet
        pt = _pair_table(self.pair_file)
        self.cols = {}
        if pt is not None:
            sub = pt.reindex(self.pair_id)
            self.cols = {k: sub[k].to_numpy() for k in sub.columns if k not in ("ids_a", "ids_b")}
        self.unit = self._units()
        self.item = self._items()
        self.names = np.array([str(x).strip().lower() in ("true", "1", "1.0") for x in self.cols["names"]]) if "names" in self.cols else np.zeros(len(pidx), bool)
        # curves
        R = pr.R
        self.A = P.matrix_pairs(R["attn_layer"].loc[ids]).reindex(pidx).to_numpy(np.float64)
        self.M = P.matrix_pairs(R["layer"].loc[ids]).reindex(pidx).to_numpy(np.float64)
        self.B = P.matrix_pairs(R["block"].loc[ids]).reindex(pidx).to_numpy(np.float64) if "block" in R else None
        drop_c = pr.drop.loc[ids]
        self.drop = P.pair_means(drop_c).reindex(pidx).to_numpy(np.float64)
        self.An = P.matrix_pairs(R["attn_layer"].loc[ids].div(drop_c, axis=0)).reindex(pidx).to_numpy(np.float64)
        self.Mn = P.matrix_pairs(R["layer"].loc[ids].div(drop_c, axis=0)).reindex(pidx).to_numpy(np.float64)
        self.Ap = self.Mp = None
        sw = pr.sw
        if "p_true" in sw.columns and sw.p_true.notna().any():
            pc = sw[sw.kind == "corrupt"].drop_duplicates("case_id").set_index("case_id").p_true
            for k, attr in (("attn_layer", "Ap"), ("layer", "Mp")):
                x = sw[(sw.kind == k) & sw.case_id.isin(ids)].pivot(index="case_id", columns="layer", values="p_true")
                x = x.sub(pc.reindex(x.index), axis=0)
                setattr(self, attr, P.matrix_pairs(x.loc[ids]).reindex(pidx).to_numpy(np.float64))
            pcl = sw[sw.kind == "clean"].drop_duplicates("case_id").set_index("case_id").p_true
            self.dp_drop = P.pair_means((pcl - pc).reindex(ids)).reindex(pidx).to_numpy(np.float64)
        else:
            self.dp_drop = None
        # direct split / joint / dla
        self.dnum = self.dden = self.mnum = self.jnum = self.jden = None
        self.dla_attn = self.dla_moe = self.dla_tot = None
        dp = os.path.join(self.dir, "direct_split.parquet")
        if os.path.exists(dp):
            x = pd.read_parquet(dp).drop_duplicates("case_id").set_index("case_id")
            if set(ids) <= set(x.index):
                x = x.loc[ids]
                self.dnum = P.pair_means(x.d_attn_direct - x.d_corrupt_fp32).reindex(pidx).to_numpy(np.float64)
                self.mnum = P.pair_means(x.d_moe_direct - x.d_corrupt_fp32).reindex(pidx).to_numpy(np.float64)
                self.dden = P.pair_means(x.d_clean_fp32 - x.d_corrupt_fp32).reindex(pidx).to_numpy(np.float64)
                if "dla_attn" in x:
                    self.dla_attn = P.pair_means(x.dla_attn).reindex(pidx).to_numpy(np.float64)
                    self.dla_moe = P.pair_means(x.dla_moe).reindex(pidx).to_numpy(np.float64)
                    self.dla_tot = P.pair_means(x.dla_total).reindex(pidx).to_numpy(np.float64)
        jp = os.path.join(self.dir, "joint_rows.parquet")
        if os.path.exists(jp):
            j = pd.read_parquet(jp)
            j = j[(j.span == "all") & (j.direction == "denoise") & (j.kind == "all_moe")].drop_duplicates("case_id").set_index("case_id")
            if set(ids) <= set(j.index):
                self.jnum = P.pair_means(j.effect.loc[ids]).reindex(pidx).to_numpy(np.float64)
                self.jden = P.pair_means(j["drop"].loc[ids]).reindex(pidx).to_numpy(np.float64)
        self.dclean = P.pair_means(c.delta_clean).reindex(pidx).to_numpy(np.float64) if "delta_clean" in c else None
        self.clean_top1 = None
        if "top1_clean" in c and "true_id" in c:
            self.clean_top1 = P.pair_means((c.top1_clean == c.true_id).astype(float)).reindex(pidx).to_numpy(np.float64)

    def _units(self) -> np.ndarray:
        e = self.entry
        col = e.get("unit_col")
        if col and col in self.cols:
            return np.asarray([str(x) for x in self.cols[col]])
        for k in UNIT_COLS:
            if k in self.cols and pd.notna(pd.Series(self.cols[k])).all():
                return np.asarray([str(x) for x in self.cols[k]])
        return np.asarray([str(p).split("|")[0] for p in self.pair_id])

    def _items(self) -> np.ndarray:
        if self.item_mode == "pair":
            return np.asarray([str(p) for p in self.pair_id])
        for k in ITEM_SRC_COLS:
            if k in self.cols and pd.notna(pd.Series(self.cols[k])).all():
                return np.asarray([str(x) for x in self.cols[k]])
        from .ext13syn import item_key
        return np.asarray([item_key(p) for p in self.pair_id])

    # ---- CounterFact donor-level anchor (ext7_cf_attnsweep schema) ------------------------------------------------
    def _load_cf(self):
        od = self.dir
        cs = json.load(open(os.path.join(od, "case_sets.json")))
        fam = cs.get("paper") or cs.get("main") or next(v for k, v in cs.items() if not k.startswith("_"))
        ids = []
        for s in self.splits:
            ids += [int(c) for c in fam.get(s, [])]
        ids = sorted(set(ids))
        rows = pd.read_parquet(os.path.join(od, "str_attn_rows.parquet"))
        cases = pd.read_parquet(os.path.join(od, "sweep_cases.parquet")).set_index("case_id")
        ids = [c for c in ids if c in cases.index]
        self.ids = ids
        self.pidx = np.array(ids)
        self.pair_id = np.array([str(c) for c in ids])
        self.unit = self.pair_id.copy()
        self.item = self.pair_id.copy()
        self.T = cases.loc[ids].n_tokens.to_numpy(np.float64) if "n_tokens" in cases else np.full(len(ids), np.nan)
        self.n_dir = np.ones(len(ids), int)
        self.cols = {}
        self.names = np.zeros(len(ids), bool)
        L = int(rows.layer.max()) + 1
        self.L = L

        def mat(kind):
            x = rows[(rows.kind == kind) & rows.case_id.isin(ids)].groupby(["case_id", "layer"]).rescue.mean().unstack()
            return x.reindex(index=ids, columns=range(L)).to_numpy(np.float64)
        self.A, self.M = mat("attn_layer"), mat("layer")
        self.B = mat("block") if (rows.kind == "block").any() else None
        self.drop = cases.loc[ids, "drop"].to_numpy(np.float64)
        self.An, self.Mn = self.A / self.drop[:, None], self.M / self.drop[:, None]
        self.Ap = self.Mp = self.dp_drop = None
        self.dnum = self.dden = self.mnum = self.jnum = self.jden = None
        self.dla_attn = self.dla_moe = self.dla_tot = None
        self.clean_top1 = None
        self.dclean = cases.loc[ids, "delta_clean"].to_numpy(np.float64) if "delta_clean" in cases else None

    @property
    def n(self) -> int:
        return len(self.pidx)


def subset(d: CellData, mask) -> CellData:
    """Shallow copy of a CellData restricted to the pairs in mask (bool [n_pairs]); per-pair arrays are filtered."""
    import copy
    mask = np.asarray(mask, bool)
    o = copy.copy(d)
    for k, v in list(vars(d).items()):
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == d.n and k != "ids":
            setattr(o, k, v[mask])
    o.cols = {k: np.asarray(v)[mask] for k, v in d.cols.items()}
    keep = set(int(p) for p in np.asarray(d.pidx)[mask])
    o.ids = [c for c in d.ids if c // 2 in keep] if d.kind == "pair" else [c for c in d.ids if c in keep]
    return o


def flag(d: CellData, *names) -> Optional[np.ndarray]:
    """Boolean per-pair column (first of names present; 'True' / 'False' strings accepted)."""
    for n in names:
        if n in d.cols:
            return np.array([str(x).strip().lower() in ("true", "1", "1.0", "yes") for x in d.cols[n]])
    return None


# ------------------------------------------------------------------------------------------------------------------
# weighted statistics
# ------------------------------------------------------------------------------------------------------------------
METRICS = ("ld", "nld", "dp")


def _share(Am, Mm, W, ws):
    a = np.clip((W @ Am) / ws[:, None], 0, None).sum(1)
    m = np.clip((W @ Mm) / ws[:, None], 0, None).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / (a + m), a, m


def tail_cut(L: int) -> int:
    """Depth-normalised interior rule for cross-model comparisons (coordinator 2026-10-16): exclude the last ceil(L / 12)
    layers (48 -> 4, 32 -> 3, 28 -> 3, 36 -> 3, 40 -> 4). Within Qwen3 / Mixtral the Phase-5 rule (<= L - 5) is kept."""
    return int(np.ceil(L / 12))


def stats(d: CellData, W: np.ndarray, metrics=METRICS) -> dict:
    """Weighted pair statistics, one value per bootstrap row (row 0 = point estimate)."""
    ws = W.sum(1)
    ws = np.where(ws > 0, ws, np.nan)
    out = {}
    sh, a, m = _share(d.A, d.M, W, ws)
    Lk = d.A.shape[1] - tail_cut(d.A.shape[1])
    out["share_int"] = _share(d.A[:, :Lk], d.M[:, :Lk], W, ws)[0]  # depth-normalised interior variant
    dr = (W @ d.drop) / ws
    out.update(share=sh, auc_attn=a / dr, auc_moe=m / dr, auc_attn_raw=a, auc_moe_raw=m, drop=dr)
    if d.B is not None:
        out["auc_block"] = np.clip((W @ d.B) / ws[:, None], 0, None).sum(1) / dr
    if "nld" in metrics:
        out["share_nld"] = _share(d.An, d.Mn, W, ws)[0]
    if "dp" in metrics and d.Ap is not None:
        s_, a_, m_ = _share(d.Ap, d.Mp, W, ws)
        out["share_dp"] = s_
        if d.dp_drop is not None:
            dpd = (W @ d.dp_drop) / ws
            out["auc_attn_dp"], out["auc_moe_dp"] = a_ / dpd, m_ / dpd
    if d.dnum is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            den = W @ d.dden
            out["A_direct"] = (W @ d.dnum) / den
            out["M_direct"] = (W @ d.mnum) / den
            if d.dla_attn is not None:
                out["dla_attn"] = (W @ d.dla_attn) / den
                out["dla_moe"] = (W @ d.dla_moe) / den
                out["dla_attn_share"] = (W @ d.dla_attn) / (W @ d.dla_tot)
    if d.jnum is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["M_all"] = (W @ d.jnum) / (W @ d.jden)
    if d.clean_top1 is not None:
        out["clean_top1"] = (W @ d.clean_top1) / ws
    if getattr(d, "dclean", None) is not None:
        out["margin_clean"] = (W @ d.dclean) / ws
    return out


def pair_weights(d: CellData, u: np.ndarray, Wu: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    pos = {x: i for i, x in enumerate(u)}
    cols = np.array([pos.get(str(x), -1) for x in d.unit])
    W = np.where(cols[None, :] >= 0, Wu[:, np.clip(cols, 0, None)], 0.0)
    if mask is not None:
        W = W * np.asarray(mask)[None, :].astype(float)
    return W


def summarise(d: CellData, mask: Optional[np.ndarray] = None, nb: int = NB, seed: int = 0) -> dict:
    sel = np.ones(d.n, bool) if mask is None else np.asarray(mask, bool)
    u, Wu = unit_weights(d.unit[sel], nb, seed)
    W = pair_weights(d, u, Wu, sel)
    s = stats(d, W)
    out = {k: ci(v) for k, v in s.items()}
    out.update(n_pairs=int(sel.sum()), n_units=int(len(u)), T_median=float(np.nanmedian(d.T[sel])) if sel.any() else float("nan"),
               names_frac=float(d.names[sel].mean()) if sel.any() else float("nan"))
    # mean curves (normalised by the mean drop) and peaks
    if sel.any():
        dr = d.drop[sel].mean()
        ca, cm = d.A[sel].mean(0) / dr, d.M[sel].mean(0) / dr
        out["curve_attn"] = [round(float(x), 5) for x in ca]
        out["curve_moe"] = [round(float(x), 5) for x in cm]
        if d.B is not None:
            out["curve_block"] = [round(float(x), 5) for x in d.B[sel].mean(0) / dr]
        out["peak_attn"] = [int(np.argmax(ca)), float(np.max(ca))]
        out["peak_moe"] = [int(np.argmax(cm)), float(np.max(cm))]
        out["L"] = int(d.L)
    return out


def joint(datas: list, keys=("share",), common: bool = True, nb: int = NB, seed: int = 0) -> Optional[dict]:
    """Statistics of several cells on the SAME unit resamples; with common=True every cell is restricted to the items
    present in all cells. Returns {'n_items', 'n_units', 'stats': [per-cell {key: array}], 'masks'}."""
    if common:
        items = set(datas[0].item)
        for d in datas[1:]:
            items &= set(d.item)
        if not items:
            return None
        masks = [np.array([x in items for x in d.item]) for d in datas]
    else:
        masks = [np.ones(d.n, bool) for d in datas]
    units = sorted(set().union(*[set(d.unit[m]) for d, m in zip(datas, masks)]))
    u, Wu = unit_weights(np.array(units), nb, seed)
    st = [stats(d, pair_weights(d, u, Wu, m)) for d, m in zip(datas, masks)]
    return {"n_items": int(len(items)) if common else None, "n_units": int(len(u)), "stats": st, "masks": masks,
            "n_pairs": [int(m.sum()) for m in masks]}


def paired(da: CellData, db: CellData, keys=("share", "share_int", "share_nld", "share_dp", "auc_attn", "auc_moe", "A_direct", "M_direct", "M_all", "drop"),
           nb: int = NB, seed: int = 0) -> Optional[dict]:
    """Paired contrast a - b on the common items (ext13syn.paired with the Phase-6 item / unit arrays)."""
    j = joint([da, db], nb=nb, seed=seed)
    if j is None:
        return None
    sa, sb = j["stats"]
    out = {"n_items": j["n_items"], "n_units": j["n_units"], "n_pairs_a": j["n_pairs"][0], "n_pairs_b": j["n_pairs"][1],
           "pooled": bool(getattr(da, "pooled", False) or getattr(db, "pooled", False))}
    for k in keys:
        if k in sa and k in sb:
            out[k] = {"a": ci(sa[k]), "b": ci(sb[k]), "diff": ci(sa[k] - sb[k])}
    return out


# ------------------------------------------------------------------------------------------------------------------
# grid: hand-off and sites
# ------------------------------------------------------------------------------------------------------------------
def grid_summary(d: CellData, nb: int = 1000, seed: int = 0) -> Optional[dict]:
    """Position x layer grid (str_grid_w1_rows) on the cell's pairs: per site (swapped token = last STR token, tokens in
    between, final token) and kind the drop-normalised curve and peak; residual hand-off layers (point estimate and a
    pair-cluster bootstrap CI of the layer index)."""
    p = os.path.join(d.dir, "str_grid_w1_rows.parquet")
    if d.kind != "pair" or not os.path.exists(p):
        return None
    g = pd.read_parquet(p, columns=["case_id", "cat", "kind", "layer", "rescue", "delta_clean", "delta_corrupt"])
    ids = set(d.ids)
    g = g[g.case_id.isin(ids)]
    if g.empty:
        return None
    L = int(g.layer.max()) + 1
    drop_c = g.groupby("case_id").apply(lambda x: (x.delta_clean - x.delta_corrupt).iloc[0], include_groups=False)
    site_of = {"last STR token": "swap", "first STR token": "swap_first", "last token": "final", "first subsequent token": "between",
               "further tokens": "between", "between STR tokens": "between_str", "middle STR tokens": "swap_mid"}
    g = g.assign(site=g.cat.map(site_of).fillna("other"))
    cc = g.groupby(["case_id", "site", "kind", "layer"], as_index=False).rescue.mean()
    out = {"L": L, "sites": {}}
    pair_of = {c: c // 2 for c in ids}
    pidx = list(d.pidx)
    upos = {pi: i for i, pi in enumerate(pidx)}
    u, Wu = unit_weights(d.unit, nb, seed)
    W = pair_weights(d, u, Wu)  # [nb+1, pairs]
    ws = W.sum(1)
    dvec = P.pair_means(drop_c).reindex(pidx).to_numpy(np.float64)
    dr = (W @ dvec) / ws
    curves_b = {}
    for (site, kind), x in cc.groupby(["site", "kind"]):
        Mx = x.pivot(index="case_id", columns="layer", values="rescue").reindex(columns=range(L))
        Pm = P.matrix_pairs(Mx).reindex(pidx)
        have = Pm.notna().all(1).to_numpy()
        A_ = np.nan_to_num(Pm.to_numpy(np.float64))
        Wh = W * have[None, :]
        wsh = np.where(Wh.sum(1) > 0, Wh.sum(1), np.nan)
        cb = (Wh @ A_) / wsh[:, None] / ((Wh @ np.nan_to_num(dvec)) / wsh)[:, None]  # [nb+1, L]
        curves_b[(site, kind)] = cb
        c0 = cb[0]
        l = int(np.nanargmax(c0)) if np.isfinite(c0).any() else 0
        out["sites"].setdefault(site, {})[kind] = {"curve": [round(float(v), 5) for v in c0], "peak_layer": l, "peak": float(c0[l]),
                                                   "n_pairs": int(have.sum())}
    ho = {}
    s = curves_b.get(("swap", "resid"))
    f = curves_b.get(("final", "resid"))

    def last_ge(c, t):
        h = np.nonzero(c >= t)[0]
        return int(h.max()) if len(h) else np.nan

    def first_ge(c, t):
        h = np.nonzero(c >= t)[0]
        return int(h.min()) if len(h) else np.nan
    if s is not None:
        v = np.array([last_ge(c, 0.5) for c in s])
        ho["swap_last_ge_0.5"] = ci(v)
    if f is not None:
        v = np.array([first_ge(c, 0.5) for c in f])
        ho["final_first_ge_0.5"] = ci(v)
        ho["final_first_ge_0.1"] = ci(np.array([first_ge(c, 0.1) for c in f]))
        ho["final_first_ge_0.9"] = ci(np.array([first_ge(c, 0.9) for c in f]))
    if s is not None and f is not None:
        # crossing: first layer at which the final-token residual restores more than the swapped-token residual
        cr = []
        for a_, b_ in zip(s, f):
            h = np.nonzero(b_ > a_)[0]
            cr.append(int(h.min()) if len(h) else np.nan)
        ho["crossing"] = ci(np.array(cr, float))
    out["handoff"] = ho
    out["handoff_depth"] = {k: [x / max(L - 1, 1) if np.isfinite(x) else np.nan for x in v] for k, v in ho.items()}
    return out


# ------------------------------------------------------------------------------------------------------------------
# heads
# ------------------------------------------------------------------------------------------------------------------
def head_summary(d: CellData, top: int = 8) -> Optional[dict]:
    """Single-head rescue at the final token (head_rows.parquet): top heads by DISCOVERY mean (selection), their
    validation means with pair-bootstrap CIs, share of the layer's attention rescue; top-3 sum / attention AUC+."""
    p = os.path.join(d.dir, "head_rows.parquet")
    if d.kind != "pair" or not os.path.exists(p):
        return None
    hr = pd.read_parquet(p)
    fam = d.pr.fam.get(d.family) or d.pr.fam.get("main")
    disc = set(int(c) for c in fam.get("discovery", []))
    val = set(int(c) for c in fam.get("validation", []))
    h = hr[hr.kind == "attn_head"]
    if h.empty:
        return None
    pm = h.assign(pair=h.case_id // 2, split=np.where(h.case_id.isin(disc), "d", np.where(h.case_id.isin(val), "v", "x")))
    agg = pm.groupby(["split", "layer", "head", "pair"]).rescue.mean().reset_index()
    dm = agg[agg.split == "d"].groupby(["layer", "head"]).rescue.mean()
    vm = agg[agg.split == "v"].groupby(["layer", "head"]).rescue.mean()
    order = dm.sort_values(ascending=False).index.tolist() if len(dm) else vm.sort_values(ascending=False).index.tolist()
    attn_ref = hr[(hr.kind == "attn_layer") & hr.case_id.isin(val)].assign(pair=lambda x: x.case_id // 2).groupby(["layer", "pair"]).rescue.mean().groupby("layer").mean()
    drop_v = float(np.mean(d.drop)) if d.n else np.nan
    rows = []
    for (l, hh) in order[:top]:
        x = agg[(agg.split == "v") & (agg.layer == l) & (agg["head"] == hh)].rescue.to_numpy()
        b = np.random.default_rng(0).integers(0, len(x), size=(2000, len(x))) if len(x) else None
        bm = x[b].mean(1) if b is not None else np.array([np.nan])
        rows.append({"layer": int(l), "head": int(hh), "name": f"L{int(l)}H{int(hh)}", "disc_mean": float(dm.get((l, hh), np.nan)),
                     "val_mean": float(x.mean()) if len(x) else np.nan, "val_lo": float(np.percentile(bm, 2.5)), "val_hi": float(np.percentile(bm, 97.5)),
                     "val_norm": float(x.mean() / drop_v) if len(x) else np.nan,
                     "share_of_layer_attn": float(x.mean() / attn_ref.get(l, np.nan)) if len(x) and attn_ref.get(l, 0) else np.nan})
    top3 = rows[:3]
    auc_attn = float(np.clip(d.A.mean(0), 0, None).sum()) if d.n else np.nan
    out = {"layers": sorted(int(l) for l in h.layer.unique()), "n_heads_scanned": int(h.groupby(["layer", "head"]).ngroups), "top": rows,
           "top3": [(r["layer"], r["head"]) for r in top3], "top3_val_sum_norm": float(sum(r["val_mean"] for r in top3) / drop_v) if top3 and drop_v else np.nan,
           "top3_over_auc_attn": float(sum(r["val_mean"] for r in top3) / auc_attn) if top3 and auc_attn else np.nan}
    return out


# ------------------------------------------------------------------------------------------------------------------
# per-cell summary
# ------------------------------------------------------------------------------------------------------------------
def behaviour(e: dict) -> dict:
    """Registry descriptors (pool, margin pass, order pass rates, top-1 rate, split sizes, strata, pooled / behaviour-only
    flags) plus the case-set JSON's pool statistics (margin rate, order gap, which answer wins where recorded)."""
    keep = ("n_pool", "n_margin", "order_pass_rates", "top1_rate", "split_sizes", "n_val", "insufficient", "pooled", "behaviour_only",
            "statement_rate", "follow_statement", "behaviour", "yield", "probe_R", "family_tokenizer", "n_both_orders", "review")
    out = {k: e[k] for k in keep if k in e}
    cs = _abs(e.get("case_sets"))
    if cs and os.path.exists(cs):
        try:
            pools = json.load(open(cs)).get("pools") or {}
        except Exception:  # noqa: BLE001
            pools = {}
        for k, v in pools.items():
            if k in ("margin_rate_pool", "order_gap", "top1_rate_pool", "n_pool_items", "n_items_pass", "n_pairs_of_passing_items") or \
                    any(w in k for w in ("statement", "prior", "event", "behaviour", "follow", "wins")):
                out[f"pool_{k}" if not k.startswith(("behaviour", "pool")) else k] = v
    return out


def cell_summary(cell: str, proto: str, reg: Optional[dict] = None, nb: int = NB) -> Optional[dict]:
    e = entry_of(cell, proto, reg)
    if e is None:
        return None
    d = CellData(cell, proto, entry=e)
    s = {"cell": cell, "proto": proto, "label": cell_label(cell) + (" †" if d.pooled else ""), "model": LABEL.get(proto, proto), "run": d.run,
         "kind": d.kind, "L": int(d.L), "registry": behaviour(e), "pooled": d.pooled}
    s["validation"] = summarise(d, nb=nb)
    try:
        dall = CellData(cell, proto, entry=e, splits=("discovery", "validation"))
        s["all"] = summarise(dall, nb=nb)
    except Exception as ex:  # noqa: BLE001
        s["all"] = {"error": repr(ex)}
    if d.kind == "pair":
        # names / objects strata, order strata
        for col in ("names", "order", "degree_adv", "antonym_type", "core776", "source"):
            if col in d.cols:
                v = np.asarray(d.cols[col]).astype(str)
                st = {}
                for lab in sorted(set(v)):
                    m = v == lab
                    if m.sum() >= 16:
                        st[lab] = {k: x for k, x in summarise(d, m, nb=1000).items() if not k.startswith("curve")}
                if st:
                    s.setdefault("strata", {})[col] = st
        try:
            s["grid"] = grid_summary(d)
        except Exception as ex:  # noqa: BLE001
            s["grid"] = {"error": repr(ex)}
        try:
            s["heads"] = head_summary(d)
        except Exception as ex:  # noqa: BLE001
            s["heads"] = {"error": repr(ex)}
    return s


# ------------------------------------------------------------------------------------------------------------------
# small helpers for cross-cell analyses
# ------------------------------------------------------------------------------------------------------------------
def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).correlation)


def load_predictions() -> dict:
    return {p["id"]: p for p in json.load(open(PRED_JSON))["predictions"]}


def band(pred: str):
    """'<= 0.30' -> (None, 0.30); '>= 0.75' -> (0.75, None); '0.30-0.55 ...' -> (0.30, 0.55)."""
    s = pred.strip()
    m = re.match(r"^(<=|>=)\s*([0-9.]+)", s)
    if m:
        v = float(m.group(2))
        return (None, v) if m.group(1) == "<=" else (v, None)
    m = re.match(r"^([0-9.]+)\s*-\s*([0-9.]+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def in_band(v: float, b) -> Optional[bool]:
    if b is None or v is None or not np.isfinite(v):
        return None
    lo, hi = b
    return (lo is None or v >= lo) and (hi is None or v <= hi)


# ------------------------------------------------------------------------------------------------------------------
# cross-cell analyses (compute; tables / figures / sections in scripts/ext14_all.py)
# ------------------------------------------------------------------------------------------------------------------
class Cache:
    """CellData per (cell, proto, item_mode, splits) and per-cell bootstrap share vectors (independent seeds)."""

    def __init__(self, reg: Optional[dict] = None):
        self.reg = registry() if reg is None else reg
        self.lines = ready_lines()
        self._d = {}
        self._boot = {}

    def ready(self, cell: str, proto: str) -> bool:
        return is_ready(cell, proto, self.reg, self.lines)

    def data(self, cell: str, proto: str, item_mode: str = "src", splits=("validation",)) -> Optional[CellData]:
        k = (cell, proto, item_mode, tuple(splits))
        if k not in self._d:
            if not self.ready(cell, proto):
                self._d[k] = None
            else:
                try:
                    self._d[k] = CellData(cell, proto, entry=entry_of(cell, proto, self.reg), item_mode=item_mode, splits=splits)
                except Exception as ex:  # noqa: BLE001
                    print(f"[ext14] load {cell}/{proto} failed: {ex!r}")
                    self._d[k] = None
        return self._d[k]

    def boot(self, cell: str, proto: str, key: str = "share", nb: int = 2000) -> Optional[np.ndarray]:
        """Bootstrap vector [nb + 1] of one statistic on the cell's own validation pairs (seed from the cell name, so
        different cells get independent resamples)."""
        k = (cell, proto, key, nb)
        if k not in self._boot:
            d = self.data(cell, proto)
            if d is None:
                self._boot[k] = None
            else:
                seed = abs(hash((cell, proto))) % (2 ** 31)
                u, Wu = unit_weights(d.unit, nb, seed)
                s = stats(d, pair_weights(d, u, Wu))
                self._boot[k] = s.get(key)
        return self._boot[k]


def cue_distance(d: CellData) -> float:
    """Median number of tokens between the (last) swapped token and the final token."""
    if d.kind != "pair":
        c = pd.read_parquet(os.path.join(d.dir, "sweep_cases.parquet")).set_index("case_id").loc[d.ids]
        if "subject_pos" in c:
            last = c.subject_pos.map(lambda x: max(json.loads(x) if isinstance(x, str) else list(x)))
            return float(np.median(c.n_tokens - 1 - last))
        return float("nan")
    c = d.pr.cases.loc[d.ids]
    return float(np.median(c["T"] - 1 - c["last_str"]))


def partial_spearman(x, y, z) -> float:
    """Spearman partial correlation of x and y given z (Pearson on ranks, residualised on z's ranks)."""
    from scipy.stats import rankdata
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 4:
        return float("nan")
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    Z = np.c_[np.ones(len(rz)), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def ladder(cache: Cache, proto: str) -> Optional[dict]:
    rows = []
    for cell, (grade, lab) in LADDER.items():
        d = cache.data(cell, proto)
        if d is None:
            continue
        s = summarise(d)
        rows.append({"cell": cell, "label": lab + (" †" if d.pooled else ""), "grade": grade, "pooled": d.pooled, "share": s["share"], "share_nld": s.get("share_nld"), "share_dp": s.get("share_dp"),
                     "A_direct": s.get("A_direct"), "M_all": s.get("M_all"), "n_pairs": s["n_pairs"], "cue_distance": cue_distance(d),
                     "T_median": s["T_median"], "peak_attn": s.get("peak_attn"), "peak_moe": s.get("peak_moe"), "L": s.get("L")})
    if not rows:
        return None
    out = {"rows": rows, "n_cells_incl_pooled": len(rows)}
    if any(r["pooled"] for r in rows) and len(rows) >= 3:
        out["spearman_incl_pooled"] = spearman([r["grade"] for r in rows], [r["share"][0] for r in rows])
    rows = [r for r in rows if not r["pooled"]]  # L9 uses cells with a held-out validation split only
    out["n_cells"] = len(rows)
    if len(rows) >= 3:
        g = [r["grade"] for r in rows]
        sh = [r["share"][0] for r in rows]
        out["spearman"] = spearman(g, sh)
        out["spearman_cue"] = spearman([r["cue_distance"] for r in rows], sh)
        out["partial_spearman_given_cue"] = partial_spearman(g, sh, [r["cue_distance"] for r in rows])
        B = [cache.boot(r["cell"], proto) for r in rows]
        if all(b is not None for b in B):
            M = np.stack(B, 1)  # [nb+1, cells]
            from scipy.stats import spearmanr
            rb = np.array([spearmanr(g, M[i]).correlation for i in range(M.shape[0])])
            out["spearman_ci"] = ci(rb)
        for m in ("share_nld", "share_dp"):
            v = [r[m][0] if r.get(m) else np.nan for r in rows]
            out[f"spearman_{m}"] = spearman(g, v)
        # diagnostic: the order without the antonym task (the grade-1 task whose share was far above its band)
        rr = [r for r in rows if r["cell"] != "l_ant"]
        if len(rr) >= 3:
            out["spearman_without_l_ant"] = spearman([r["grade"] for r in rr], [r["share"][0] for r in rr])
    return out


def dial_e(cache: Cache, proto: str) -> Optional[dict]:
    out = {"levels": {}, "paired": {}, "gap": {}}
    have = False
    for dr in DIRS:
        for lv in DIAL_E_LEVELS + ("e3p3", "e0"):
            d = cache.data(f"{lv}_{dr}", proto)
            if d is None:
                continue
            have = True
            s = summarise(d)
            out["levels"].setdefault(dr, {})[lv] = {k: s.get(k) for k in ("share", "share_nld", "share_dp", "auc_attn", "auc_moe", "A_direct", "M_direct",
                                                                          "M_all", "drop", "n_pairs", "n_units", "peak_attn", "peak_moe", "clean_top1", "T_median")}
            out["levels"][dr][lv]["pooled"] = d.pooled
        # pairwise level contrasts on common twins (a2e: E3 = the frame-matched e3p3 and the F1 anchor e3)
        for a, b in (("e1p", "e1"), ("e1", "e2"), ("e1p", "e2"), ("e1", "e3"), ("e2", "e3"), ("e1p", "e3"), ("e1", "e3p3"), ("e2", "e3p3"),
                     ("e1p", "e3p3"), ("e3p3", "e3")):
            da, db = cache.data(f"{a}_{dr}", proto), cache.data(f"{b}_{dr}", proto)
            if da is not None and db is not None:
                p = paired(da, db)
                if p:
                    out["paired"][f"{a}-{b}_{dr}"] = p
        # all four levels on the items they share (ordering test E14); a2e: frame-matched E3 (e3p3), also the F1 E3
        for tag, lvls in (("joint4", DIAL_LEVELS[dr]), ("joint4_f1", DIAL_E_LEVELS)):
            if tag == "joint4_f1" and dr == "e2a":
                continue
            ds = [cache.data(f"{lv}_{dr}", proto) for lv in lvls]
            if all(x is not None for x in ds):
                j = joint(ds)
                if j:
                    out.setdefault(tag, {})[dr] = {"n_items": j["n_items"], "n_units": j["n_units"], "levels": list(lvls),
                                                   "share": {lv: ci(st["share"]) for lv, st in zip(lvls, j["stats"])}}
    if not have:
        return None
    # E2 caveat (coordinator QA 2026-10-16): negation of gradable adjectives with degree adverbs does not entail the antonym;
    # E2 and its contrasts are also reported without degree-adverb twins and by antonym type
    out["e2_strata"] = {}
    for dr in DIRS:
        d2 = cache.data(f"e2_{dr}", proto)
        if d2 is None:
            continue
        deg = flag(d2, "degree_adv", "stratum_degree_adv")
        ent = {}
        if deg is not None and (~deg).sum() >= 16:
            s2 = subset(d2, ~deg)
            ent["no_degree_adv"] = {k: v for k, v in summarise(s2).items() if not k.startswith("curve")}
            for a in ("e1", "e1p", "e3"):
                da = cache.data(f"{a}_{dr}", proto)
                if da is not None:
                    p = paired(da, s2)
                    if p:
                        ent[f"{a}-e2_no_degree_adv"] = p
        at = None
        for n in ("antonym_type", "stratum_antonym_type"):
            if n in d2.cols:
                at = np.asarray([str(x) for x in d2.cols[n]])
        if at is not None:
            for lab in sorted(set(at)):
                m = at == lab
                if m.sum() >= 16:
                    ent[f"antonym_type={lab}"] = {k: v for k, v in summarise(subset(d2, m)).items() if not k.startswith("curve")}
        if ent:
            out["e2_strata"][dr] = ent
    # direction gap a2e - e2a per level (paired twins)
    for lv, (ca, cb) in {"e1": ("e1_a2e", "e1_e2a"), "e1p": ("e1p_a2e", "e1p_e2a"), "e2": ("e2_a2e", "e2_e2a"), "e3": ("e3_a2e", "e3_e2a"),
                         "e3p3": ("e3p3_a2e", "e3_e2a"), "e0": ("e0_a2e", "e0_e2a")}.items():
        da, db = cache.data(ca, proto), cache.data(cb, proto)
        if da is not None and db is not None:
            p = paired(da, db)
            if p:
                out["gap"][lv] = p
    # E13: gap(E1) and gap(E3) on the twins common to the four cells; primary E3 for a2e = the frame-matched e3p3_a2e,
    # secondary = the F1 anchor e3_a2e (the pre-registered E3 gap)
    for tag, e3a in (("h_test", "e3p3_a2e"), ("h_test_f1", "e3_a2e")):
        ds = [cache.data(c, proto) for c in ("e1_a2e", "e1_e2a", e3a, "e3_e2a")]
        if all(x is not None for x in ds):
            j = joint(ds)
            if j:
                s = [st["share"] for st in j["stats"]]
                g1, g3 = s[0] - s[1], s[2] - s[3]
                out[tag] = {"e3_a2e_cell": e3a, "n_items": j["n_items"], "n_units": j["n_units"], "gap_e1": ci(g1), "gap_e3": ci(g3),
                            "gap_e1_minus_half_e3": ci(g1 - 0.5 * g3), "gap_ratio_e1_e3": ci(g1 / g3), "method": "four cells on their common twins",
                            "pooled": any(x.pooled for x in ds)}
    # pairwise variant: gap(E1) and gap(E3) each on their own common twins (independent resamples), used as the primary
    # H-test when the four-cell intersection is small (< 20 twins)
    for tag, e3a in (("h_test_pw", "e3p3_a2e"), ("h_test_pw_f1", "e3_a2e")):
        da1, de1, da3, de3 = (cache.data(c, proto) for c in ("e1_a2e", "e1_e2a", e3a, "e3_e2a"))
        if all(x is not None for x in (da1, de1, da3, de3)):
            j1, j3 = joint([da1, de1], seed=0), joint([da3, de3], seed=1)
            if j1 and j3:
                g1 = j1["stats"][0]["share"] - j1["stats"][1]["share"]
                g3 = j3["stats"][0]["share"] - j3["stats"][1]["share"]
                out[tag] = {"e3_a2e_cell": e3a, "n_items": f"{j1['n_items']} / {j3['n_items']}", "n_items_e1": j1["n_items"], "n_items_e3": j3["n_items"],
                            "gap_e1": ci(g1), "gap_e3": ci(g3), "gap_e1_minus_half_e3": ci(g1 - 0.5 * g3), "gap_ratio_e1_e3": ci(g1 / g3),
                            "method": "pairwise (independent resamples of the two gaps)",
                            "pooled": any(x.pooled for x in (da1, de1, da3, de3))}
    h4 = out.get("h_test")
    if "h_test_pw" in out and (h4 is None or (isinstance(h4.get("n_items"), int) and h4["n_items"] < 20)):
        if h4 is not None:
            out["h_test_4cell"] = h4
        out["h_test"] = out["h_test_pw"]
    if "h_test" not in out and "h_test_f1" in out:
        out["h_test"] = out["h_test_f1"]
    return out


def _prior_score(d: CellData, proto: str) -> np.ndarray:
    """Prior support of the STATED (clean) answer per pair: an explicit column of the pair parquet if p6-data wrote one,
    else the Phase-5 probe convention (mean over the responding probes of (probe_a - probe_b) / 2)."""
    for k in PRIOR_COLS:
        if k in d.cols:
            return np.asarray(d.cols[k], dtype=float)
    R = tuple(d.entry.get("probe_R") or {"qwen3": ("v1", "v2")}.get(proto, ("v2",)))
    cols = [(f"probe_{v}_a", f"probe_{v}_b") for v in R if f"probe_{v}_a" in d.cols and f"probe_{v}_b" in d.cols]
    if not cols:
        return np.full(d.n, np.nan)
    return np.nanmean(np.stack([(np.asarray(d.cols[a], float) - np.asarray(d.cols[b], float)) / 2.0 for a, b in cols]), axis=0)


def dial_a(cache: Cache, proto: str) -> Optional[dict]:
    out = {"classes": {}, "dose": {}}
    have = False
    for dr in DIRS:
        for cls, cell in (("E0", f"e0_{dr}"), ("agree", f"aagr_{dr}"), ("neutral", f"e1_{dr}"), ("conflict", f"acon_{dr}")):
            d = cache.data(cell, proto)
            if d is None:
                continue
            have = True
            s = summarise(d)
            out["classes"].setdefault(dr, {})[cls] = {k: s.get(k) for k in ("share", "share_nld", "share_dp", "A_direct", "M_direct", "M_all", "dla_moe",
                                                                           "drop", "n_pairs", "n_units", "clean_top1")}
            out["classes"][dr][cls]["cell"] = cell
            out["classes"][dr][cls]["pooled"] = d.pooled
            pr = _prior_score(d, proto)
            out["classes"][dr][cls]["prior_mean"] = float(np.nanmean(pr)) if np.isfinite(pr).any() else None
        # dose-response: agree + conflict pooled, 5 quantile bins of the prior support of the stated answer
        parts = [(cache.data(f"aagr_{dr}", proto), "agree"), (cache.data(f"acon_{dr}", proto), "conflict")]
        parts = [(d, c) for d, c in parts if d is not None]
        if parts:
            sc = np.concatenate([_prior_score(d, proto) for d, _ in parts])
            if np.isfinite(sc).sum() >= 25:
                qs = np.nanquantile(sc, [0.2, 0.4, 0.6, 0.8])
                bins = []
                for b in range(5):
                    lo = -np.inf if b == 0 else qs[b - 1]
                    hi = np.inf if b == 4 else qs[b]
                    rows_b = []
                    for d, _ in parts:
                        s_ = _prior_score(d, proto)
                        m = np.isfinite(s_) & (s_ > lo) & (s_ <= hi) if b > 0 else np.isfinite(s_) & (s_ <= hi)
                        if m.sum():
                            rows_b.append((d, m))
                    if not rows_b:
                        continue
                    # one bootstrap over the bin's pairs of both cells (independent seeds per bin)
                    j_units = np.concatenate([d.unit[m] for d, m in rows_b])
                    u, Wu = unit_weights(j_units, 2000, 100 + b)
                    # pool: AUC+ of the pooled mean curves (weights = pair counts)
                    ws = [pair_weights(d, u, Wu, m) for d, m in rows_b]
                    tot = sum(w.sum(1) for w in ws)
                    Am = sum((w @ d.A) for w, (d, _) in zip(ws, rows_b)) / tot[:, None]
                    Mm = sum((w @ d.M) for w, (d, _) in zip(ws, rows_b)) / tot[:, None]
                    a = np.clip(Am, 0, None).sum(1)
                    mm = np.clip(Mm, 0, None).sum(1)
                    allsc = np.concatenate([_prior_score(d, proto)[m] for d, m in rows_b])
                    bins.append({"bin": b + 1, "lo": None if b == 0 else float(qs[b - 1]), "hi": None if b == 4 else float(qs[b]),
                                 "n_pairs": int(sum(m.sum() for _, m in rows_b)), "prior_mean": float(np.mean(allsc)), "share": ci(a / (a + mm))})
                if len(bins) >= 3:
                    out["dose"][dr] = {"bins": bins, "spearman": spearman([x["prior_mean"] for x in bins], [x["share"][0] for x in bins]),
                                       "n_scored": int(np.isfinite(sc).sum())}
    if not have:
        return None
    # paired contrasts where items overlap (conflict vs agree share the noun pairs: same twin / item key)
    out["paired"] = {}
    for dr in DIRS:
        for a, b in (("acon", "aagr"), ("acon", "e1"), ("e1", "aagr"), ("aagr", "e0"), ("acon", "e0")):
            da, db = cache.data(f"{a}_{dr}", proto), cache.data(f"{b}_{dr}", proto)
            if da is not None and db is not None:
                p = paired(da, db)
                if p:
                    out["paired"][f"{a}-{b}_{dr}"] = p
    return out


def sc_behaviour(e: dict) -> dict:
    """Set C behaviour from the case-set JSON pools (p6-data, scripts/ext14_casesets.py): items following the statement /
    the event (margin >= 1 both ways); statement_rate = statement / (statement + event) items (C1: 'margin items')."""
    b = behaviour(e)
    cs = _abs(e.get("case_sets"))
    if cs and os.path.exists(cs):
        pools = json.load(open(cs)).get("pools") or {}
        bi = pools.get("behaviour_items")
        if bi:
            st, ev, n = bi.get("statement_following", 0), bi.get("event_following", 0), bi.get("items", 0)
            b.update({"statement_items": st, "event_items": ev, "scanned_items": n,
                      "statement_rate": st / (st + ev) if st + ev else None, "statement_rate_all_items": st / n if n else None})
        if pools.get("behaviour_pairs"):
            b["behaviour_pairs"] = pools["behaviour_pairs"]
    return b


def set_c(cache: Cache, proto: str) -> Optional[dict]:
    reg = cache.reg
    out = {"behaviour": {}, "subsets": {}}
    for cell in reg:
        if not cell.startswith("sc_"):
            continue
        e = reg[cell].get(proto)
        if not e:
            continue
        b = sc_behaviour(e) if cell.startswith("sc_conflict") else behaviour(e)
        out["behaviour"][cell] = b
        d = cache.data(cell, proto)
        if d is None:
            continue
        s = summarise(d)
        out["subsets"][cell] = {k: s.get(k) for k in ("share", "share_nld", "share_dp", "A_direct", "M_direct", "M_all", "n_pairs", "n_units", "drop")}
        ref = "e1_a2e" if (is_stmt_subset(cell) or cell == "sc_agree_a2e") else ("e3_a2e" if is_event_subset(cell) else None)
        if ref:
            dr = cache.data(ref, proto)
            if dr is not None:
                p = paired(d, dr)
                if p:
                    out["subsets"][cell]["vs_" + ref] = p
    return out if (out["behaviour"] or out["subsets"]) else None


def r_metric(cache: Cache, proto: str, cells: Optional[list] = None) -> Optional[dict]:
    cells = cells or [c for c in cache.reg if proto in cache.reg[c]]
    rows = {}
    for c in sorted(cells):
        d = cache.data(c, proto)
        if d is None:
            continue
        s = summarise(d, nb=1000)
        rows[c] = {"ld": s["share"], "nld": s.get("share_nld"), "dp": s.get("share_dp"), "n_pairs": s["n_pairs"]}
    if len(rows) < 2:
        return {"cells": rows} if rows else None
    ks = list(rows)
    ld = [rows[k]["ld"][0] for k in ks]
    out = {"cells": rows}
    for m in ("nld", "dp"):
        v = [rows[k][m][0] if rows[k].get(m) else np.nan for k in ks]
        out[f"spearman_ld_{m}"] = spearman(ld, v)
        out[f"n_{m}"] = int(np.isfinite(v).sum())
        out[f"max_abs_diff_{m}"] = float(np.nanmax(np.abs(np.asarray(v) - np.asarray(ld)))) if np.isfinite(v).any() else None
    return out


def r_site(cache: Cache, proto: str) -> Optional[dict]:
    da, db = cache.data("rsite_e1_a2e", proto), cache.data("e1_a2e", proto)
    if da is None:
        return None
    out = {"rsite": {k: v for k, v in summarise(da).items() if not k.startswith("curve")}}
    try:
        out["rsite_grid"] = grid_summary(da)
    except Exception as ex:  # noqa: BLE001
        out["rsite_grid"] = {"error": repr(ex)}
    if db is not None:
        out["paired"] = paired(da, db)
        try:
            out["e1_grid"] = grid_summary(db)
        except Exception as ex:  # noqa: BLE001
            out["e1_grid"] = {"error": repr(ex)}
    return out


def handoff_table(summ_cells: dict, proto: str) -> list:
    rows = []
    for k, s in summ_cells.items():
        c, p = k.split("/")
        if p != proto or not isinstance(s.get("grid"), dict) or "handoff" not in s["grid"]:
            continue
        rows.append({"cell": c, **s["grid"]["handoff"], "L": s["grid"]["L"]})
    return rows


# ---- Stage 2 --------------------------------------------------------------------------------------------------------
def variant(cache: "Cache", c: str, protos) -> Optional[str]:
    """Cell variant for cross-model comparisons (coordinator 2026-10-17 03:19Z): the all-association-strata cell <c>_all when it
    is ready for every model compared (comparable items), else the plain cell when it is ready for every model."""
    for v in (f"{c}_all", c):
        if all(cache.ready(v, p) for p in protos):
            return v
    return None


def own_diff(cache: "Cache", ca: str, pa: str, cb: str, pb: str, key: str = "share") -> Optional[list]:
    """a - b of two cells' OWN validation-set statistics with independent cluster bootstraps (different items)."""
    ba, bb = cache.boot(ca, pa, key), cache.boot(cb, pb, key)
    if ba is None or bb is None:
        return None
    n = min(len(ba), len(bb))
    return ci(np.asarray(ba[:n]) - np.asarray(bb[:n]))


def twin_compare(cache: Cache, dense: str, moe: str, cells=COMPARE_CELLS) -> Optional[dict]:
    out = {"dense": dense, "moe": moe, "cells": {}}
    for c0 in cells:
        c = variant(cache, c0, (dense, moe))
        if c is None:
            continue
        dd, dm = cache.data(c, dense, item_mode="pair"), cache.data(c, moe, item_mode="pair")
        if dd is None or dm is None:
            continue
        p = paired(dm, dd)  # MoE - dense on the common pairs (identical prompts)
        if p is None:
            continue
        sd, sm = summarise(dd), summarise(dm)
        out["cells"][c0] = {"variant": c, "pooled": bool(dd.pooled or dm.pooled), "paired": p, "dense_own": sd["share"], "moe_own": sm["share"], "curve_dense_attn": sd.get("curve_attn"),
                           "curve_dense_mlp": sd.get("curve_moe"), "curve_moe_attn": sm.get("curve_attn"), "curve_moe_moe": sm.get("curve_moe"),
                           "L_dense": sd.get("L"), "L_moe": sm.get("L")}
    if not out["cells"]:
        return None
    for c0, v in out["cells"].items():  # contrast used for D1-D4: paired on >= MIN_COMMON common pairs, else own sets
        c = v["variant"]
        if v["paired"]["n_items"] >= MIN_COMMON:
            v["diff_used"], v["basis"] = v["paired"]["share"]["diff"], f"paired ({v['paired']['n_items']} common pairs)"
        else:
            od = own_diff(cache, c, moe, c, dense)
            v["diff_used"], v["basis"] = (od, f"own sets (only {v['paired']['n_items']} common pairs)") if od else (v["paired"]["share"]["diff"], "paired (few pairs)")
        v["own_diff"] = own_diff(cache, c, moe, c, dense)
    diffs = {c: v["diff_used"] for c, v in out["cells"].items()}
    dvals = np.array([x[0] for x in diffs.values()])
    out["n_cells"] = len(diffs)
    out["frac_within_0.10"] = float(np.mean(np.abs(dvals) <= 0.10))
    out["frac_moe_higher_0.10"] = float(np.mean(dvals >= 0.10))
    out["spearman_cells"] = spearman([v["dense_own"][0] for v in out["cells"].values()], [v["moe_own"][0] for v in out["cells"].values()])  # own-set cell order
    kn = [diffs[c][0] for c in KNOWLEDGE_CELLS if c in diffs]
    cp = [diffs[c][0] for c in COPY_CELLS if c in diffs]
    out["knowledge_diffs"] = {c: diffs[c] for c in KNOWLEDGE_CELLS if c in diffs}
    out["copy_diffs"] = {c: diffs[c] for c in COPY_CELLS if c in diffs}
    out["D1"] = out["frac_within_0.10"] >= 0.8
    out["D2"] = (len(kn) > 0 and all(x <= -0.10 for x in kn)) and (len(cp) > 0 and all(abs(x) <= 0.10 for x in cp))
    out["D3"] = out["frac_moe_higher_0.10"] > 0.5
    out["D4"] = (out["spearman_cells"] >= 0.8) if np.isfinite(out["spearman_cells"]) else None
    return out


def series_compare(cache: Cache, protos: tuple, cells=COMPARE_CELLS) -> Optional[dict]:
    """Shares per cell along a capacity series on the pairs common to every member (item = pair id)."""
    out = {"protos": list(protos), "cells": {}}
    for c0 in cells:
        # variant with the most members (ties -> the all-strata cell, comparable items across models)
        opts = [(sum(cache.ready(v, p) for p in protos), v) for v in (f"{c0}_all", c0)]
        n_best, c = max(opts, key=lambda x: x[0])
        if n_best < 2:
            continue
        ds = [cache.data(c, p, item_mode="pair") for p in protos]
        avail = [(p, d) for p, d in zip(protos, ds) if d is not None]
        if len(avail) < 2:
            continue
        j = joint([d for _, d in avail])
        if not j:
            continue
        out["cells"][c0] = {"variant": c, "pooled": any(d.pooled for _, d in avail), "members": [p for p, _ in avail], "n_items": j["n_items"],
                            "share": {p: ci(st["share"]) for (p, _), st in zip(avail, j["stats"])},
                           "share_int": {p: ci(st["share_int"]) for (p, _), st in zip(avail, j["stats"])},
                           "own": {p: summarise(d)["share"] for p, d in avail}}
        if len(avail) >= 2:
            first, last = j["stats"][0]["share"], j["stats"][-1]["share"]
            out["cells"][c0]["last_minus_first"] = ci(last - first)
    return out if out["cells"] else None


def depth_curves(cache: Cache, proto: str, cell: str) -> Optional[dict]:
    d = cache.data(cell, proto)
    if d is None:
        return None
    dr = d.drop.mean()
    L = d.L
    x = (np.arange(L) / L).round(4).tolist()  # layer axis l / L (coordinator 2026-10-16)
    return {"x": x, "attn": (d.A.mean(0) / dr).round(5).tolist(), "mlp": (d.M.mean(0) / dr).round(5).tolist(), "L": L, "n_pairs": d.n}


# ------------------------------------------------------------------------------------------------------------------
# pre-registered predictions: observed values and verdicts (results/ext14_summary.json "predictions")
# ------------------------------------------------------------------------------------------------------------------
MIN_VAL = 64  # R5: a cell is reported only with >= 64 validation pairs
CELL_OF_PRED = {"L-ant": "l_ant", "L-prop": "l_prop", "L-bind": "l_bind", "L-ind": "l_ind", "E1-a2e": "e1_a2e", "E1-e2a": "e1_e2a",
                "E3-a2e": "e3_a2e", "E3-e2a": "e3_e2a", "E0": ("e0_a2e", "e0_e2a")}
HELD, FAILED, INSUFF, PENDING = "held", "failed", "insufficient data", "pending"
INSUFF_POOLED = "insufficient (pooled only)"


def _q(S: dict, cell: str, proto: str, key: str = "share"):
    c = S.get("cells", {}).get(f"{cell}/{proto}")
    if not c:
        return None, None
    v = c.get("validation", {})
    if c.get("pooled"):
        return v.get(key), "pooled"
    return v.get(key), v.get("n_pairs")


def _protos(models: str, S: dict) -> list:
    if models == "QM":
        return list(STAGE1)
    if models in ALL_PROTOS:
        return [models]
    if models == "all":
        have = sorted({k.split("/")[1] for k in S.get("cells", {})})
        return [p for p in ALL_PROTOS if p in have] or list(STAGE1)
    return list(STAGE1)


def _combine(per: dict) -> str:
    vs = [v.get("verdict") for v in per.values()]
    if not vs:
        return PENDING
    if any(v == FAILED for v in vs):
        return FAILED
    if all(v == HELD for v in vs):
        return HELD
    if any(v == PENDING for v in vs):
        return PENDING
    if all(v == INSUFF_POOLED for v in vs):
        return INSUFF_POOLED
    return INSUFF


def _band_eval(x, n, b):
    if x is None:
        return {"verdict": PENDING}
    if n == "pooled":
        return {"verdict": INSUFF_POOLED, "observed": x, "note": "pooled cell (no held-out validation split)"}
    if n is not None and n < MIN_VAL:
        return {"verdict": INSUFF, "observed": x, "n_pairs": n}
    ok = in_band(x[0], b)
    lo, hi = b
    ci_over = (x[2] >= (lo if lo is not None else -9)) and (x[1] <= (hi if hi is not None else 9)) if len(x) >= 3 and np.isfinite(x[1]) else None
    return {"verdict": HELD if ok else FAILED, "observed": x, "n_pairs": n, "ci_overlaps_band": ci_over}


def _diff_eval(pp, cond, need_ci_excl0=False, band=None):
    """pp: a paired() result (keys 'share' = {'a','b','diff'}, 'pooled', 'n_items'); cond(point of the share difference) -> bool."""
    if not pp or not pp.get("share"):
        return {"verdict": PENDING}
    p = pp["share"]
    if pp.get("pooled"):
        return {"verdict": INSUFF_POOLED, "observed": p["diff"], "note": "involves a pooled cell"}
    d = p["diff"]
    ok = bool(cond(d[0]))
    if need_ci_excl0 and np.isfinite(d[1]):
        ok = ok and (d[1] > 0 or d[2] < 0)
    out = {"verdict": HELD if ok else FAILED, "observed": d, "a": p["a"], "b": p["b"], "n_items": pp.get("n_items")}
    if band is not None and np.isfinite(d[1]):
        out["ci_overlaps_band"] = bool(d[2] >= band[0] and d[1] <= band[1])
    return out


def evaluate_predictions(S: dict) -> dict:
    P = load_predictions()
    out = {}
    cells = S.get("cells", {})
    de = S.get("dial_e", {})
    da = S.get("dial_a", {})
    for pid, p in P.items():
        r = {"id": pid, "models": p["models"], "cell": p["cell"], "metric": p["metric"], "prediction": p["prediction"], "test": p["test"]}
        per = {}
        try:
            if pid in ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "E1", "E2", "E5", "E6", "E7", "E8", "E11", "E12", "A3", "A4"):
                b = band(p["prediction"])
                cs = CELL_OF_PRED[p["cell"]]
                cs = cs if isinstance(cs, tuple) else (cs,)
                for proto in _protos(p["models"], S):
                    for c in cs:
                        x, n = _q(S, c, proto)
                        per[f"{proto}" if len(cs) == 1 else f"{proto}:{c}"] = _band_eval(x, n, b)
            elif pid == "L9":
                for proto, L in (S.get("ladder") or {}).items():
                    if not L or L.get("n_cells", 0) < 3:
                        per[proto] = {"verdict": PENDING}
                        continue
                    rho = L.get("spearman")
                    per[proto] = {"verdict": (HELD if rho >= 0.8 else FAILED) if L["n_cells"] >= 6 else PENDING, "observed": rho,
                                  "ci": L.get("spearman_ci"), "n_cells": L["n_cells"], "partial_given_cue": L.get("partial_spearman_given_cue")}
            elif pid in ("E3", "E4", "E9", "E10"):
                spec = {"E3": ("e1p-e1_a2e", lambda d: abs(d) <= 0.10, False, (-0.10, 0.10), 1),
                        "E4": ("e1-e2_a2e", lambda d: d >= 0.10, True, None, 1),
                        "E9": ("e1p-e1_e2a", lambda d: -d >= 0.10, True, None, -1),  # E1 - E1p >= 0.10 (stored as e1p - e1)
                        "E10": ("e1p-e2_e2a", lambda d: d > 0, False, None, 1)}[pid]
                key, cond, excl, bnd, sign = spec
                for proto in STAGE1:
                    pp = (de.get(proto) or {}).get("paired", {}).get(key)
                    e = _diff_eval(pp, cond, excl, bnd)
                    if sign == -1 and "observed" in e:
                        o = e["observed"]
                        e["observed_e1_minus_e1p"] = [-o[0], -o[2], -o[1]]
                    per[proto] = e
            elif pid == "E13":
                for proto in STAGE1:
                    h = (de.get(proto) or {}).get("h_test")
                    if not h:
                        per[proto] = {"verdict": PENDING}
                        continue
                    if h.get("pooled"):
                        per[proto] = {"verdict": INSUFF_POOLED, "gap_e1": h["gap_e1"], "gap_e3": h["gap_e3"], "note": "involves a pooled cell"}
                        continue
                    g1, g3, gd = h["gap_e1"], h["gap_e3"], h["gap_e1_minus_half_e3"]
                    hi_ = bool(abs(g1[0]) <= 0.10)
                    hi_ci = bool(np.isfinite(g1[1]) and g1[1] >= -0.10 and g1[2] <= 0.10)
                    ht = bool(g1[0] >= 0.5 * g3[0])
                    ht_ci = bool(np.isfinite(gd[1]) and gd[1] > 0)
                    decision = ("H-inference" if hi_ci else "H-type" if ht_ci else "H-inference (point only)" if hi_ else
                                "H-type (point only)" if ht else "neither (gap between half the E3 gap and 0.10)")
                    per[proto] = {"verdict": HELD if hi_ else FAILED, "decision": decision, "gap_e1": g1, "gap_e3": g3, "gap_e1_minus_half_e3": gd,
                                  "n_items": h["n_items"], "e3_a2e_cell": h.get("e3_a2e_cell"),
                                  "note": "verdict = the best-guess hypothesis H-inference; E3 gap with the frame-matched e3p3_a2e (P3 frame)"}
                    hf = (de.get(proto) or {}).get("h_test_f1")
                    if hf and hf is not h:
                        per[proto]["f1_variant"] = {"gap_e3": hf["gap_e3"], "gap_e1_minus_half_e3": hf["gap_e1_minus_half_e3"], "n_items": hf["n_items"]}
            elif pid == "E14":
                for proto in STAGE1:
                    lv = (de.get(proto) or {}).get("levels", {})
                    j = (de.get(proto) or {}).get("joint4", {})
                    for dr in DIRS:
                        lvls = DIAL_LEVELS[dr] if (dr == "e2a" or (lv.get(dr) or {}).get("e3p3")) else DIAL_E_LEVELS
                        sh = {k: v["share"][0] for k, v in (j.get(dr, {}).get("share") or {}).items()} if j.get(dr) and j[dr].get("levels") == list(lvls) else \
                             {k: (lv.get(dr, {}).get(k) or {}).get("share", [np.nan])[0] for k in lvls}
                        if any(not np.isfinite(sh.get(k, np.nan)) for k in lvls):
                            per[f"{proto}:{dr}"] = {"verdict": PENDING, "observed": sh}
                            continue
                        e1, e1p, e2, e3 = (sh[k] for k in lvls)
                        ok = (e1 >= e1p > e2 and e1 > e3) if dr == "a2e" else (e1 > e1p > e2 >= e3)
                        per[f"{proto}:{dr}"] = {"verdict": HELD if ok else FAILED, "observed": sh, "E3_cell": f"{lvls[3]}_{dr}",
                                                "source": "common items" if j.get(dr) and j[dr].get("levels") == list(lvls) else "own sets"}
            elif pid == "A1":
                for proto in STAGE1:
                    cl = (da.get(proto) or {}).get("classes", {})
                    for dr in DIRS:
                        c = cl.get(dr, {})
                        sh = {k: (c.get(k) or {}).get("share", [np.nan])[0] for k in ("conflict", "neutral", "agree", "E0")}
                        if any(not np.isfinite(v) for v in sh.values()):
                            per[f"{proto}:{dr}"] = {"verdict": PENDING, "observed": sh}
                            continue
                        ok = sh["conflict"] >= sh["neutral"] > sh["agree"] > sh["E0"]
                        per[f"{proto}:{dr}"] = {"verdict": HELD if ok else FAILED, "observed": sh}
            elif pid == "A2":
                for proto in STAGE1:
                    cl = (da.get(proto) or {}).get("classes", {})
                    for dr in DIRS:
                        c = cl.get(dr, {}).get("conflict")
                        if not c or not c.get("M_direct"):
                            per[f"{proto}:{dr}"] = {"verdict": PENDING}
                            continue
                        m = c["M_direct"]
                        n = c.get("n_pairs")
                        if n is not None and n < MIN_VAL:
                            per[f"{proto}:{dr}"] = {"verdict": INSUFF, "observed": m, "n_pairs": n}
                            continue
                        per[f"{proto}:{dr}"] = {"verdict": HELD if (m[0] < 0 and np.isfinite(m[2]) and m[2] < 0) else FAILED, "observed": m,
                                                "dla_moe": c.get("dla_moe"), "n_pairs": n}
            elif pid == "A5":
                for proto in STAGE1:
                    ds = (da.get(proto) or {}).get("dose", {})
                    for dr in DIRS:
                        x = ds.get(dr)
                        per[f"{proto}:{dr}"] = {"verdict": PENDING} if not x else {"verdict": HELD if x["spearman"] < 0 else FAILED,
                                                                                    "observed": x["spearman"], "bins": [(b["prior_mean"], b["share"][0]) for b in x["bins"]]}
            elif pid == "C1":
                for proto, sc in (S.get("set_c") or {}).items():
                    rate = None
                    for cell, b in (sc or {}).get("behaviour", {}).items():
                        if "conflict" in cell and b.get("statement_rate") is not None:
                            rate = float(b["statement_rate"])
                            continue
                        if "conflict" in cell and rate is None:
                            for k in ("statement_rate", "follow_statement"):
                                if isinstance(b.get(k), (int, float)):
                                    rate = float(b[k])
                            if rate is None and isinstance(b.get("behaviour"), dict):
                                for k in ("statement_rate", "follow_statement", "frac_statement"):
                                    if isinstance(b["behaviour"].get(k), (int, float)):
                                        rate = float(b["behaviour"][k])
                    per[proto] = {"verdict": PENDING} if rate is None else {"verdict": HELD if rate >= 0.6 else FAILED, "observed": rate}
            elif pid in ("C2", "C3"):
                for proto in STAGE1:
                    sc = (S.get("set_c") or {}).get(proto) or {}
                    hit = None
                    for cell, v in sc.get("subsets", {}).items():
                        if (pid == "C2" and is_stmt_subset(cell)) or (pid == "C3" and is_event_subset(cell)):
                            hit = (cell, v)
                    if not hit:
                        per[proto] = {"verdict": PENDING}
                        continue
                    cell, v = hit
                    ref = "vs_e1_a2e" if pid == "C2" else "vs_e3_a2e"
                    n = v.get("n_pairs")
                    if n is not None and n < MIN_VAL:
                        per[proto] = {"verdict": INSUFF, "n_pairs": n}
                        continue
                    pp = v.get(ref)
                    per[proto] = _diff_eval(pp, lambda d: abs(d) <= 0.10, False, (-0.10, 0.10))
                    per[proto]["subset"] = cell
            elif pid == "R1":
                for proto in STAGE1:
                    rm = (S.get("r_metric") or {}).get(proto)
                    if not rm or "spearman_ld_nld" not in rm:
                        per[proto] = {"verdict": PENDING}
                        continue
                    a, b_ = rm["spearman_ld_nld"], rm.get("spearman_ld_dp")
                    ncell = len(rm.get("cells", {}))
                    okv = lambda v: v is not None and np.isfinite(v)  # noqa: E731
                    ok = okv(a) and a >= 0.8 and (not okv(b_) or b_ >= 0.8)
                    per[proto] = {"verdict": (HELD if ok else FAILED) if (ncell >= 6 and okv(a)) else PENDING, "observed": {"nld": a, "dp": b_},
                                  "n_cells": ncell, "note": "verdict once >= 6 Stage-1 cells are in"}
            elif pid == "R2":
                for proto in STAGE1:
                    rs = (S.get("r_site") or {}).get(proto)
                    pp = (rs or {}).get("paired")
                    per[proto] = _diff_eval(pp, lambda d: abs(d) <= 0.15, False, (-0.15, 0.15))
            elif pid in ("R3", "R4"):
                rc = S.get("r_causal") or {}
                for k, v in rc.get("tests", {}).items():
                    if v.get("pred") == pid:
                        per[k] = v
                if not per:
                    per["all"] = {"verdict": PENDING}
            elif pid in ("D0", "D1", "D2", "D3", "D4"):
                for tw, t in (S.get("twins") or {}).items():
                    if not t:
                        per[tw] = {"verdict": PENDING}
                        continue
                    key = {"D0": "D1", "D1": "D1", "D2": "D2", "D3": "D3", "D4": "D4"}[pid]
                    if pid == "D0" and not tw.startswith("mistral7b"):
                        continue
                    v = t.get(key)
                    obs = {"D1": t.get("frac_within_0.10"), "D2": {"knowledge": t.get("knowledge_diffs"), "copy": t.get("copy_diffs")},
                           "D3": t.get("frac_moe_higher_0.10"), "D4": t.get("spearman_cells")}[key]
                    per[tw] = {"verdict": PENDING if v is None else (HELD if v else FAILED), "observed": obs, "n_cells": t.get("n_cells")}
            elif pid == "D5":
                se = (S.get("series") or {}).get("qwen3")
                if not se:
                    per["qwen3 series"] = {"verdict": PENDING}
                else:
                    tr = {c: v.get("dense_trend") for c, v in se.get("cells", {}).items() if v.get("dense_trend") is not None}
                    rho = se.get("spearman_dense_vs_30b") or {}
                    if not tr:
                        per["qwen3 series"] = {"verdict": PENDING}
                    else:
                        ok = all(abs(x[0]) <= 0.10 for x in tr.values()) and all(v >= 0.8 for v in rho.values() if v is not None and np.isfinite(v))
                        per["qwen3 series"] = {"verdict": HELD if ok else FAILED, "trend_14b_minus_4b": tr, "spearman_vs_30b": rho,
                                               "frac_cells_trend_le_0.10": float(np.mean([abs(x[0]) <= 0.10 for x in tr.values()]))}
            elif pid == "D6":
                sh = (S.get("qwen2_shared") or {}).get("knowledge_shared_share")
                per["qwen2_57b"] = {"verdict": PENDING} if sh is None else {"verdict": HELD if all(v[0] >= 0.5 for v in sh.values()) else FAILED, "observed": sh}
            elif pid in ("N1", "N2", "N3"):
                per["p6-circuit"] = {"verdict": PENDING, "note": "evaluated from p6-circuit's outputs (ext14_circuit)"}
                cs = S.get("circuit_predictions") or {}
                if pid in cs:
                    per = {"p6-circuit": cs[pid]}
        except Exception as ex:  # noqa: BLE001
            per["error"] = {"verdict": PENDING, "error": repr(ex)}
        _flag_cells(pid, per, S)
        r["per_model"] = per
        r["verdict"] = _combine(per)
        tv = [v.get("verdict") for v in per.values() if v.get("verdict") in (HELD, FAILED)]
        r["verdict_testable"] = (FAILED if FAILED in tv else HELD) if tv else None  # over the model / direction entries that could be tested
        r["n_testable"] = len(tv)
        out[pid] = r
    return out


# cells each prediction reads (per direction where the prediction is per direction); used to turn 'pending' into
# 'insufficient data' for behaviour-only cells and any verdict into 'insufficient (pooled only)' when a pooled cell is involved
PRED_CELLS = {
    "L1": ["l_ant"], "L2": ["l_ant"], "L3": ["l_prop"], "L4": ["l_prop"], "L5": ["l_bind"], "L6": ["l_bind"], "L7": ["l_ind"], "L8": ["l_ind"],
    "E1": ["e1_a2e"], "E2": ["e1_a2e"], "E5": ["e3_a2e"], "E6": ["e3_a2e"], "E7": ["e1_e2a"], "E8": ["e1_e2a"], "E11": ["e3_e2a"], "E12": ["e3_e2a"],
    "E3": ["e1p_a2e", "e1_a2e"], "E4": ["e1_a2e", "e2_a2e"], "E9": ["e1p_e2a", "e1_e2a"], "E10": ["e1p_e2a", "e2_e2a"],
    "E13": ["e1_a2e", "e1_e2a", "e3p3_a2e", "e3_e2a"], "E14": ["e1_{d}", "e1p_{d}", "e2_{d}", "e3p3_a2e|e3_e2a"],
    "A1": ["e0_{d}", "aagr_{d}", "e1_{d}", "acon_{d}"], "A2": ["acon_{d}"], "A3": ["e0_{d}"], "A4": ["e0_{d}"], "A5": ["aagr_{d}", "acon_{d}"],
    "C2": ["sc_conflict_stmt_a2e", "e1_a2e"], "C3": ["sc_conflict_event_a2e", "e3p3_a2e"], "R2": ["rsite_e1_a2e", "e1_a2e"],
}


def _flag_cells(pid, per, S):
    if pid not in PRED_CELLS:
        return
    cells, bo = S.get("cells", {}), S.get("behaviour_only", {})
    for key, v in per.items():
        if ":" in key:
            proto, rest = key.split(":", 1)
        else:
            proto, rest = key, ""
        if proto not in ALL_PROTOS:
            continue
        dr = rest if rest in DIRS else (rest.split("_")[-1] if rest.endswith(("a2e", "e2a")) else None)
        names = []
        for c in PRED_CELLS[pid]:
            if "{d}" in c:
                names += [c.format(d=dr)] if dr else [c.format(d=x) for x in DIRS]
            elif "|" in c:
                names += [x for x in c.split("|") if dr is None or x.endswith(dr)]
            else:
                names.append(c)
        if rest and rest.startswith(("e0_", "e1", "aagr", "acon")) and "_" in rest:  # per-cell keys (A3 / A4 'proto:cell')
            names = [rest]
        b_only = [c for c in names if f"{c}/{proto}" in bo]
        pooled = [c for c in names if (cells.get(f"{c}/{proto}") or {}).get("pooled")]
        if v.get("verdict") == PENDING and b_only:
            v["verdict"] = INSUFF
            v["note"] = f"behaviour only (< 64 pairs): {', '.join(b_only)}"
        elif pooled and v.get("verdict") in (HELD, FAILED):
            v["verdict_if_validation"] = v["verdict"]
            v["verdict"] = INSUFF_POOLED
            v["note"] = f"pooled cell(s) without a held-out split: {', '.join(pooled)}"


# ------------------------------------------------------------------------------------------------------------------
# behaviour of every scanned pair (also for behaviour-only cells): target vs the other answer
# ------------------------------------------------------------------------------------------------------------------
BEH_STRATA = ("degree_adv", "antonym_type", "names", "order")


def behaviour_rates(cell: str, proto: str, e: Optional[dict] = None, nb: int = 1000) -> Optional[dict]:
    """From the cell's pair parquet (every scanned pair; d_clean_a / d_clean_b = logit(trig_a) - logit(trig_b) on prompt A / B):
      target_margin  margin >= 1 in both directions toward the target answers (the reporting rule);
      other_margin   the OTHER answer wins by >= 1 in both directions (E2: the copy trap = the word stated with the other
                     entity; Dial A conflict: the prior; Set C conflict: the event);
      target_sign / other_sign   the same at sign level (> 0);
      items_target_all_orders / items_other_all_orders   items (twins) whose every built order is target-margin / other-margin.
    Overall and per stratum (degree adverb, antonym type, names, statement order); CIs by twin-cluster bootstrap."""
    e = e if e is not None else entry_of(cell, proto)
    pf = _abs((e or {}).get("pair_file"))
    if not pf or not os.path.exists(pf):
        return None
    d = pd.read_parquet(pf)
    if "d_clean_a" not in d.columns:
        return None
    ok = d.d_clean_a.notna() & d.d_clean_b.notna()
    d = d[ok].copy()
    if d.empty:
        return None
    d["t_sign"] = (d.d_clean_a > 0) & (d.d_clean_b < 0)
    d["o_sign"] = (d.d_clean_a < 0) & (d.d_clean_b > 0)
    d["t_marg"] = (d.d_clean_a >= 1) & (d.d_clean_b <= -1)
    d["o_marg"] = (d.d_clean_a <= -1) & (d.d_clean_b >= 1)
    unit = d.src_pair_id.astype(str) if "src_pair_id" in d.columns else d.pair_id.astype(str)
    d["unit"] = unit.values

    def rates(x):
        u, inv = np.unique(x.unit.values, return_inverse=True)
        idx = np.random.default_rng(0).integers(0, len(u), size=(nb, len(u)))
        cnt = np.zeros((nb, len(u)))
        np.add.at(cnt, (np.repeat(np.arange(nb), len(u)), idx.ravel()), 1.0)
        W = np.vstack([np.ones(len(u)), cnt])[:, inv]
        ws = W.sum(1)
        out = {"n_pairs": int(len(x)), "n_items": int(len(u))}
        for k in ("t_marg", "o_marg", "t_sign", "o_sign"):
            out[k] = ci((W @ x[k].to_numpy(float)) / ws)
        g = x.groupby("unit")
        out["items_target_all_orders"] = int(g.t_marg.all().sum())
        out["items_other_all_orders"] = int(g.o_marg.all().sum())
        out["mean_delta_toward_target"] = float(((x.d_clean_a - x.d_clean_b) / 2).mean())
        return out
    res = {"cell": cell, "proto": proto, "all": rates(d), "strata": {}}
    for col in BEH_STRATA:
        if col in d.columns and d[col].nunique() > 1:
            for v, x in d.groupby(d[col].astype(str)):
                if len(x) >= 10:
                    res["strata"][f"{col}={v}"] = rates(x)
    return res


# ------------------------------------------------------------------------------------------------------------------
# Qwen2-57B-A14B: shared vs routed experts (prediction D6) from p6-data's joint patches
# ------------------------------------------------------------------------------------------------------------------
def shared_routed(cache: "Cache", cell: str, proto: str = "qwen2_57b", nb: int = NB) -> Optional[dict]:
    """joint_shared_rows.parquet (scripts/ext14_scan_joint_shared.py, p6-data): per directed case and direction the effect of
    patching every layer's whole MoE output (all_moe), only the routed experts (routed_only) or only the shared expert
    (shared_only) at the final token. Validation pairs, pair means, cluster bootstrap (same resamples for every ratio):
    shared share of the MoE rescue = shared_only / (shared_only + routed_only); also each / all_moe and / drop.
    Routed-only numbers are never reported as 'all experts'."""
    d = cache.data(cell, proto)
    if d is None or d.kind != "pair":
        return None
    p = os.path.join(d.dir, "joint_shared_rows.parquet")
    if not os.path.exists(p):
        return None
    j = pd.read_parquet(p)
    j = j[(j.span == "all") & j.case_id.isin(d.ids)] if "span" in j.columns else j[j.case_id.isin(d.ids)]
    u, Wu = unit_weights(d.unit, nb, 0)
    W = pair_weights(d, u, Wu)
    out = {"cell": cell, "proto": proto, "n_pairs": int(d.n), "n_units": int(len(u)), "pooled": d.pooled}
    for direction in ("denoise", "noise"):
        g = j[j.direction == direction]
        if g.empty:
            continue
        piv = g.pivot_table(index="case_id", columns="kind", values="effect", aggfunc="first")
        dr = g.groupby("case_id")["drop"].first()
        if not {"all_moe", "routed_only", "shared_only"} <= set(piv.columns):
            continue
        pm = {k: P.pair_means(piv[k].reindex(d.ids)).reindex(d.pidx).to_numpy(np.float64) for k in ("all_moe", "routed_only", "shared_only")}
        drp = P.pair_means(dr.reindex(d.ids)).reindex(d.pidx).to_numpy(np.float64)
        s = {k: W @ np.nan_to_num(v) for k, v in pm.items()}
        sd = W @ np.nan_to_num(drp)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[direction] = {"shared_share": ci(s["shared_only"] / (s["shared_only"] + s["routed_only"])),
                              "shared_over_all": ci(s["shared_only"] / s["all_moe"]), "routed_over_all": ci(s["routed_only"] / s["all_moe"]),
                              "all_moe_norm": ci(s["all_moe"] / sd), "shared_norm": ci(s["shared_only"] / sd), "routed_norm": ci(s["routed_only"] / sd),
                              "additivity_gap_norm": ci((s["all_moe"] - s["shared_only"] - s["routed_only"]) / sd)}
    # DLA split shared vs routed when p6-data's DLA rows carry it
    dp = os.path.join(d.dir, "dla_rows.parquet")
    if os.path.exists(dp):
        dl = pd.read_parquet(dp)
        cols = [c for c in dl.columns if "shared" in c]
        if cols:
            out["dla_shared_columns"] = cols
            x = dl[dl.case_id.isin(d.ids)]
            if {"shared_c", "shared_k"} <= set(x.columns):
                sh = (x.shared_c - x.shared_k).groupby(x.case_id).sum()
                mo = (x.moe_c - x.moe_k).groupby(x.case_id).sum()
                a = P.pair_means(sh.reindex(d.ids)).reindex(d.pidx).to_numpy(np.float64)
                b = P.pair_means(mo.reindex(d.ids)).reindex(d.pidx).to_numpy(np.float64)
                out["dla_shared_over_moe"] = ci((W @ np.nan_to_num(a)) / (W @ np.nan_to_num(b)))
    return out if ("denoise" in out or "noise" in out) else None
