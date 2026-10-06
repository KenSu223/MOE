"""ext12 (Phase 4, Direction 12): completeness checks of the Phase-3 claims. No engine change.

4a  Add-back replication. The ext8 machinery (moetrace/ext8_addback.py, read-only here) is reused with two new task
    variants:
      * WinoGrande replication split: ext8's WinoGrande task on case-set family "rep" (population ranking from
        rep_discovery, curves on rep_validation) of the ext7 run, whose all-layer expert rows already cover "rep";
      * CounterFact fold swap: ext8's CounterFact task with the two paper folds exchanged, i.e. the population ranking and
        the layer-wise baseline are computed on the paper's VALIDATION cases and every curve is evaluated on the paper's
        DISCOVERY cases. Inside the Task the swapped folds carry the labels the ext8 code expects ("discovery" = ranking
        source, "validation" = evaluation set); `orig_split` keeps the paper labels.
    4c reuses the WinoGrande loader on the Mixtral no-BOS run (family "main").
4b  role swap vs option swap on identical items: pairing helpers (twin-level and clean-prompt-matched).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import ext8_addback as X
from .models import RESULTS

ROOT = "/home/ubuntu/MOE"

# ------------------------------------------------------------------------------------------------------------------
# 4a / 4c task registry (one entry per add-back run of ext12)
# ------------------------------------------------------------------------------------------------------------------
TASKS = {
    "wino_qwen3_rep": dict(model="qwen3", kind="wino", src="wino_qwen3_str", family="rep", out="wino_qwen3_str_addback_rep",
                           pairs="data/wino_str/pairs_train_xl_qwen3.parquet", case_sets="data/wino_str/case_sets.json",
                           label="Qwen3-30B-A3B-Base", tlabel="WinoGrande (replication split)", main_key="wino_qwen3", pool=32),
    "cf_qwen3_fold": dict(model="qwen3", kind="cf_swap", src="qwen3_str", out="qwen3_str_addback_fold_rep", label="Qwen3-30B-A3B-Base",
                          tlabel="CounterFact (fold swap)", main_key="cf_qwen3", pool=32),
    "wino_mixtral_rep": dict(model="mixtral", kind="wino", src="wino_mixtral_bos_str", family="rep", out="wino_mixtral_bos_str_addback_rep",
                             pairs="data/wino_str/pairs_train_xl_mixtral_bos.parquet", case_sets="data/wino_str/case_sets.json",
                             label="Mixtral-8x7B (BOS)", tlabel="WinoGrande (replication split)", main_key="wino_mixtral", pool=64),
    "cf_mixtral_fold": dict(model="mixtral", kind="cf_swap", src="mixtral_bos_str", out="mixtral_bos_str_addback_fold_rep",
                            label="Mixtral-8x7B (BOS)", tlabel="CounterFact (fold swap)", main_key="cf_mixtral", pool=64),
    "wino_mixtral_nobos": dict(model="mixtral", kind="wino", src="wino_mixtral_nobos_str", family="main", out="wino_mixtral_nobos_str_addback",
                               pairs="data/wino_str/pairs_train_xl_mixtral_nobos.parquet", case_sets="data/wino_str/case_sets.json",
                               label="Mixtral-8x7B (no BOS)", tlabel="WinoGrande (no BOS)", main_key="wino_mixtral", pool=64),
}
GROUPS = {"qwen3": ("wino_qwen3_rep", "cf_qwen3_fold"), "mixtral": ("wino_mixtral_rep", "cf_mixtral_fold", "wino_mixtral_nobos")}


def load_cf_swapped(model: str, str_run: str) -> X.Task:
    """ext8 CounterFact task with the paper folds exchanged (ranking from the paper's validation cases, curves on its
    discovery cases). pop / layer_disc are recomputed with exactly ext8's estimator on the new ranking fold."""
    t = X.load_cf(model, str_run)
    orig = t.cases.split.copy()
    t.cases = t.cases.copy()
    t.cases["orig_split"] = orig
    t.cases["split"] = np.where(orig == "validation", "discovery", "validation")
    new_disc = [int(c) for c in t.cases.case_id[orig == "validation"]]
    od = os.path.join(RESULTS, str_run)
    ids = [int(c) for c in t.cases.case_id]
    er = pd.read_parquet(os.path.join(od, "str_expert_rows.parquet"))
    er = er[(er.kind == "expert") & er.clean_active & er.case_id.isin(ids)][["case_id", "slot", "layer", "expert", "rescue", "vnorm"]]
    sw = pd.read_parquet(os.path.join(od, "str_sweep_rows.parquet"))
    sw = sw[(sw.kind == "layer") & sw.case_id.isin(ids)][["case_id", "slot", "layer", "rescue"]]
    t.pop, t.layer_disc = X._pop_and_layers(er, sw, new_disc)
    t.case_index = {int(c): i for i, c in enumerate(t.cases.case_id)}
    return t


def load_task(key: str) -> X.Task:
    c = TASKS[key]
    if c["kind"] == "cf_swap":
        return load_cf_swapped(c["model"], c["src"])
    if c["kind"] == "wino":
        return X.load_wino(c["model"], c["src"], os.path.join(ROOT, c["pairs"]), os.path.join(ROOT, c["case_sets"]), family=c["family"])
    raise ValueError(c["kind"])


def static_queue(t: X.Task) -> list:
    """ext12 static work queue (subset of ext8's base queue): A0 ceilings for every row (both directions), A1 add-back curves
    of every static ordering except dla on the evaluation rows, the all-clean-active endpoint. No deletion curves, noising
    singles, exact subsets, Shapley sweeps or beam (4a brief)."""
    rows = t.rows
    split = rows.case_id.map(dict(zip(t.cases.case_id, t.cases.split)))
    val = rows.row_id[split == "validation"].tolist()
    allr = rows.row_id.tolist()
    q = []
    q += X.a0_groups(t, allr)
    q += X.curve_groups(t, val, [o for o in X.A1_ORDERS if o != "dla"], "d", "a1")
    q += X.all_active_group(t, val, "d", "a1")
    return q


def eval_rows(t: X.Task) -> list[int]:
    split = t.rows.case_id.map(dict(zip(t.cases.case_id, t.cases.split)))
    return t.rows.row_id[split == "validation"].tolist()


def dla_queue(t: X.Task, od: str) -> list:
    dla_df = pd.read_parquet(os.path.join(od, "addback_dla.parquet"))
    dla = {int(r_): dict(zip(zip(g.layer.astype(int), g.expert.astype(int)), g.dla)) for r_, g in dla_df.groupby("row_id")}
    return X.curve_groups(t, eval_rows(t), ["dla"], "d", "a1", dla=dla)


# ------------------------------------------------------------------------------------------------------------------
# small statistics helpers shared by the analysis
# ------------------------------------------------------------------------------------------------------------------
def boot_idx(n: int, n_boot: int = 5000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def ci(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
