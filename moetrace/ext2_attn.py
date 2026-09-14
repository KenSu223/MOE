"""ext2 (RESEARCH_PLAN Direction 2, Question B): attention-vs-MoE attribution of the factual-recall rescue.

Post-processing of `scripts/ext2_attn_sweep.py` runs: per-layer validation curves with bootstrap CIs for the kinds
attn_layer (attention-sublayer output patch), layer (MoE output patch), block (both), resid (clean residual), peaks,
areas, attention share, additivity (block vs attn + moe) and protocol comparisons. Statistics via moetrace.stats.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .models import RESULTS
from .stats import N_BOOT, SEED, bootstrap_ci, ratio_ci, summarize

RUNS = {
    "qwen3_bos_attnsweep": {"model": "qwen3", "short": "qwen3_bos", "label": "Qwen3-30B-A3B-Base (tokenizer defaults)",
                            "protocol": "bos"},
    "mixtral_bos_attnsweep": {"model": "mixtral", "short": "mixtral_bos", "label": "Mixtral-8x7B-v0.1 (BOS, tokenizer default)",
                              "protocol": "bos"},
    "mixtral_nobos_attnsweep": {"model": "mixtral", "short": "mixtral_nobos", "label": "Mixtral-8x7B-v0.1 (no BOS, paper protocol)",
                                "protocol": "nobos"},
    "olmoe_attnsweep": {"model": "olmoe", "short": "olmoe", "label": "OLMoE-1B-7B-0125 (pilot)", "protocol": "bos"},
}
MAIN_KINDS = ("attn_layer", "layer", "block")
ALL_KINDS = ("attn_layer", "layer", "block", "resid")
KIND_LABEL = {"attn_layer": "attention output", "layer": "MoE output", "block": "attention + MoE (block)",
              "resid": "residual after layer (hidden state)", "block_diff": "block (difference form)"}


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------
class Run:
    def __init__(self, run: str, case_set: str = "paper"):
        self.run = run
        self.cfg = RUNS.get(run, {"model": run.split("_")[0], "short": run, "label": run, "protocol": "?"})
        d = os.path.join(RESULTS, run)
        self.dir = d
        self.rows = pd.read_parquet(os.path.join(d, "sweep_rows.parquet"))
        with open(os.path.join(d, "case_sets.json")) as f:
            self.sets = json.load(f)
        self.meta = json.load(open(os.path.join(d, "run_meta.json"))) if os.path.exists(os.path.join(d, "run_meta.json")) else {}
        self.case_set = case_set if case_set in self.sets else [k for k in ("paper", "strict", "relaxed") if k in self.sets][0]
        self.kinds = [k for k in self.rows.kind.unique() if k not in ("clean", "noised")]
        self.R = {k: self.rows[self.rows.kind == k].pivot(index="case_id", columns="layer", values="rescue") for k in self.kinds}
        self.V = {k: self.rows[self.rows.kind == k].pivot(index="case_id", columns="layer", values="vnorm") for k in self.kinds}
        self.layers = list(self.R[self.kinds[0]].columns)
        self.L = len(self.layers)
        present = set(self.R[self.kinds[0]].index)
        self.disc = [c for c in self.sets[self.case_set]["discovery"] if c in present]
        self.val = [c for c in self.sets[self.case_set]["validation"] if c in present]
        self.delta_clean = self.rows[self.rows.kind == "clean"].set_index("case_id").delta
        self.delta_noised = self.rows[self.rows.kind == "noised"].set_index("case_id").delta

    def mat(self, kind: str, ids: list[int]) -> np.ndarray:
        """[n_cases, L] rescue matrix of the given cases (row order = ids)."""
        return self.R[kind].loc[ids].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------------------------------------------
# curves and peaks
# ---------------------------------------------------------------------------------------------------------------
def layer_curve(run: Run, kind: str, ids: list[int]) -> pd.DataFrame:
    M = run.mat(kind, ids)
    rows = []
    for j, l in enumerate(run.layers):
        s = summarize(M[:, j], with_p=False)
        rows.append({"kind": kind, "layer": int(l), "mean": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                     "pos_frac": s["pos_frac"], "n": s["n"], "vnorm_mean": float(run.V[kind].loc[ids].iloc[:, j].mean())})
    return pd.DataFrame(rows)


def all_curves(run: Run, ids: list[int], kinds=None) -> pd.DataFrame:
    kinds = kinds or [k for k in ALL_KINDS if k in run.kinds]
    return pd.concat([layer_curve(run, k, ids) for k in kinds], ignore_index=True)


def auc_positive(mean_curve: np.ndarray) -> float:
    """Area under the positive part of a per-layer mean curve (sum over layers of max(mean, 0))."""
    return float(np.clip(mean_curve, 0, None).sum())


def centre_of_mass(mean_curve: np.ndarray, layers) -> float:
    w = np.clip(mean_curve, 0, None)
    return float((w * np.asarray(layers)).sum() / w.sum()) if w.sum() > 0 else float("nan")


def _boot_idx(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def auc_ci(M: np.ndarray) -> tuple[float, float, float]:
    """AUC+ of the mean curve of M [n, L] with a case-bootstrap CI."""
    idx = _boot_idx(M.shape[0])
    a = np.clip(M[idx].mean(1), 0, None).sum(1)
    return auc_positive(M.mean(0)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def peaks(run: Run, kinds=None) -> pd.DataFrame:
    """Per kind: discovery argmax layer, its validation value, validation argmax and max, AUC+ and centre of mass
    (validation), all with CIs where meaningful."""
    kinds = kinds or [k for k in ALL_KINDS if k in run.kinds]
    rows = []
    for k in kinds:
        Md, Mv = run.mat(k, run.disc), run.mat(k, run.val)
        md, mv = Md.mean(0), Mv.mean(0)
        jd, jv = int(np.argmax(md)), int(np.argmax(mv))
        sd = summarize(Mv[:, jd], with_p=True)
        sv = summarize(Mv[:, jv], with_p=False)
        auc, alo, ahi = auc_ci(Mv)
        rows.append({"kind": k, "label": KIND_LABEL.get(k, k), "L_disc": run.layers[jd], "disc_mean": float(md[jd]),
                     "val_at_L_disc": sd["mean"], "val_at_L_disc_lo": sd["ci_lo"], "val_at_L_disc_hi": sd["ci_hi"],
                     "val_pos_frac_at_L_disc": sd["pos_frac"], "val_p_at_L_disc": sd["p"],
                     "L_val": run.layers[jv], "val_max": sv["mean"], "val_max_lo": sv["ci_lo"], "val_max_hi": sv["ci_hi"],
                     "auc_pos": auc, "auc_lo": alo, "auc_hi": ahi, "com": centre_of_mass(mv, run.layers),
                     "n_disc": len(run.disc), "n_val": len(run.val)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# attention share and additivity
# ---------------------------------------------------------------------------------------------------------------
def share_at_layer(run: Run, layer: int, ids: list[int]) -> dict:
    """attn / (attn + moe) mean rescue at one layer, paired case bootstrap CI (ratio of means)."""
    j = run.layers.index(layer)
    a = run.mat("attn_layer", ids)[:, j]
    m = run.mat("layer", ids)[:, j]
    r, lo, hi = ratio_ci(a, a + m)
    return {"layer": int(layer), "attn_mean": float(a.mean()), "moe_mean": float(m.mean()), "share": r, "share_lo": lo, "share_hi": hi}


def share_overall(run: Run, ids: list[int]) -> dict:
    """AUC+(attn) / (AUC+(attn) + AUC+(moe)) over the validation mean curves, case-bootstrap CI."""
    A, M = run.mat("attn_layer", ids), run.mat("layer", ids)
    idx = _boot_idx(A.shape[0])
    aa = np.clip(A[idx].mean(1), 0, None).sum(1)
    mm = np.clip(M[idx].mean(1), 0, None).sum(1)
    r = aa / (aa + mm)
    a0, m0 = auc_positive(A.mean(0)), auc_positive(M.mean(0))
    return {"auc_attn": a0, "auc_moe": m0, "share": a0 / (a0 + m0), "share_lo": float(np.percentile(r, 2.5)),
            "share_hi": float(np.percentile(r, 97.5))}


def additivity(run: Run, ids: list[int]) -> pd.DataFrame:
    """Per layer: mean block rescue vs mean(attn) + mean(moe); mean of the per-case gap block - attn - moe with CI;
    per-case Pearson r between block and attn + moe; fraction of cases where block exceeds the sum."""
    A, M, B = (run.mat(k, ids) for k in ("attn_layer", "layer", "block"))
    S = A + M
    rows = []
    for j, l in enumerate(run.layers):
        gap = B[:, j] - S[:, j]
        lo, hi = bootstrap_ci(gap)
        with np.errstate(invalid="ignore"):
            r = float(np.corrcoef(B[:, j], S[:, j])[0, 1]) if S[:, j].std() > 0 and B[:, j].std() > 0 else float("nan")
        rows.append({"layer": int(l), "attn": float(A[:, j].mean()), "moe": float(M[:, j].mean()), "sum": float(S[:, j].mean()),
                     "block": float(B[:, j].mean()), "gap": float(gap.mean()), "gap_lo": lo, "gap_hi": hi, "r_case": r,
                     "frac_block_gt_sum": float((gap > 0).mean()), "n": len(ids)})
    return pd.DataFrame(rows)


def resid_increments(curves: pd.DataFrame) -> pd.DataFrame:
    """Layer-to-layer increments of the resid (hidden-state) validation curve: where the final position acquires
    the answer. Returns layer, resid, increment."""
    c = curves[curves.kind == "resid"].sort_values("layer")
    inc = np.diff(np.concatenate([[0.0], c["mean"].to_numpy()]))
    return pd.DataFrame({"layer": c.layer.to_numpy(), "resid": c["mean"].to_numpy(), "increment": inc})


# ---------------------------------------------------------------------------------------------------------------
# protocol comparison (Mixtral BOS vs no BOS)
# ---------------------------------------------------------------------------------------------------------------
def compare_runs(ra: Run, rb: Run, kinds=None) -> pd.DataFrame:
    """Curve-level comparison on the validation split of each run (the case sets are the same ids, tokenised under two
    protocols): per kind, argmax layers, max values, curve correlation across layers, max |difference|, and the paired
    per-case difference at each run's argmax on the cases present in both."""
    kinds = kinds or [k for k in ALL_KINDS if k in ra.kinds and k in rb.kinds]
    common = [c for c in ra.val if c in set(rb.val)]
    rows = []
    for k in kinds:
        ma, mb = ra.mat(k, ra.val).mean(0), rb.mat(k, rb.val).mean(0)
        ja, jb = int(np.argmax(ma)), int(np.argmax(mb))
        d_at_a = ra.mat(k, common)[:, ja] - rb.mat(k, common)[:, ja]
        lo, hi = bootstrap_ci(d_at_a)
        rows.append({"kind": k, "L_a": ra.layers[ja], "max_a": float(ma[ja]), "L_b": rb.layers[jb], "max_b": float(mb[jb]),
                     "curve_r": float(np.corrcoef(ma, mb)[0, 1]), "max_absdiff": float(np.abs(ma - mb).max()),
                     "L_max_absdiff": int(ra.layers[int(np.argmax(np.abs(ma - mb)))]),
                     "paired_diff_at_L_a": float(d_at_a.mean()), "paired_lo": lo, "paired_hi": hi, "n_common": len(common)})
    return pd.DataFrame(rows)
