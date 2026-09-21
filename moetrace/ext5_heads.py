"""ext5 F2: analysis of per-head attention patching runs (scripts/ext5_heads_sweep.py).

Per layer: head ranking by rescue (validation means with bootstrap CIs, discovery rank for stability), head specificity
(rescue_h minus the mean rescue of the other heads of the layer, per case), additivity of the head patches against the
whole attention-output patch (sum over heads vs attn_layer, per case), the minimal head set reaching a fraction of the
attention rescue under the additive approximation (population greedy and per case), and the attention distribution
of the top heads over position classes (final, subject-last, other subject, position 0, other) in the clean vs the
noised run from DiagSpec.attn_final. Statistics via moetrace.stats (5,000-resample percentile bootstrap, seed 0).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .models import RESULTS
from .stats import N_BOOT, SEED, bootstrap_ci, summarize

RUNS = {
    "qwen3_heads": {"model": "qwen3", "short": "qwen3", "label": "Qwen3-30B-A3B-Base (tokenizer defaults)", "protocol": "bos",
                    "moe_peak": 44, "attn_peaks": (40, 43)},
    "mixtral_nobos_heads": {"model": "mixtral", "short": "mixtral_nobos", "label": "Mixtral-8x7B-v0.1 (no BOS, paper protocol)",
                            "protocol": "nobos", "moe_peak": 19, "attn_peaks": (15, 18)},
    "mixtral_bos_heads": {"model": "mixtral", "short": "mixtral_bos", "label": "Mixtral-8x7B-v0.1 (BOS, tokenizer default)",
                          "protocol": "bos", "moe_peak": 19, "attn_peaks": (15, 18)},
    "olmoe_heads": {"model": "olmoe", "short": "olmoe", "label": "OLMoE-1B-7B-0125 (pilot)", "protocol": "bos",
                    "moe_peak": None, "attn_peaks": ()},
}
REF_KINDS = ("attn_layer", "layer", "block")
POS_CLASSES = ("final", "subj_last", "subj_other", "pos0", "other")
POS_LABEL = {"final": "final token", "subj_last": "last subject token", "subj_other": "other subject tokens",
             "pos0": "position 0 (non-subject)", "other": "other (relation) tokens"}


class HeadRun:
    def __init__(self, run: str, case_set: str | None = None):
        self.run = run
        self.cfg = RUNS.get(run, {"model": run.split("_")[0], "short": run, "label": run, "protocol": "?", "moe_peak": None, "attn_peaks": ()})
        d = os.path.join(RESULTS, run)
        self.dir = d
        self.rows = pd.read_parquet(os.path.join(d, "head_rows.parquet"))
        self.cases = pd.read_parquet(os.path.join(d, "sweep_cases.parquet")).set_index("case_id")
        with open(os.path.join(d, "case_sets.json")) as f:
            self.sets = json.load(f)
        mp = os.path.join(d, "run_meta.json")
        self.meta = json.load(open(mp)) if os.path.exists(mp) else {}
        avail = [k for k in ("paper", "strict", "relaxed") if k in self.sets]
        self.case_set = case_set if (case_set in self.sets) else avail[0]
        self.layers = sorted(int(l) for l in self.rows.layer.unique())
        hr = self.rows[self.rows.kind == "attn_head"]
        self.nH = int(hr["head"].max()) + 1
        self.R = {l: hr[hr.layer == l].pivot(index="case_id", columns="head", values="rescue") for l in self.layers}
        self.V = {l: hr[hr.layer == l].pivot(index="case_id", columns="head", values="vnorm") for l in self.layers}
        self.ref = {l: {k: self.rows[(self.rows.kind == k) & (self.rows.layer == l)].set_index("case_id").rescue for k in REF_KINDS}
                    for l in self.layers}
        present = set(self.R[self.layers[0]].index)
        self.disc = [c for c in self.sets[self.case_set]["discovery"] if c in present]
        self.val = [c for c in self.sets[self.case_set]["validation"] if c in present]
        self.all_ids = self.disc + self.val
        self.delta_clean = self.cases.delta_clean
        self.delta_noised = self.cases.delta_noised
        ap = os.path.join(d, "head_attn_final.npz")
        self.attn = None
        if os.path.exists(ap):
            z = np.load(ap)
            self.attn = {int(l): z["attn"][i] for i, l in enumerate(z["layers"])}  # [2n, nH, T] fp16
            self.attn_ids = [int(c) for c in z["case_ids"]]
            self.attn_lens = z["lens"]

    def mat(self, l: int, ids: list[int]) -> np.ndarray:
        return self.R[l].loc[ids].to_numpy(dtype=np.float64)  # [n, nH]

    def refv(self, l: int, kind: str, ids: list[int]) -> np.ndarray:
        return self.ref[l][kind].loc[ids].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------------------------------------------
# ranking and specificity
# ---------------------------------------------------------------------------------------------------------------
def head_ranking(run: HeadRun, l: int) -> pd.DataFrame:
    """One row per head: validation rescue (mean, CI, positive fraction, sign-flip p), discovery mean and rank,
    validation rank, Spec (rescue_h - mean of the other heads, per case, validation), share of the attn_layer rescue,
    mean |v_h| (validation)."""
    Mv, Md = run.mat(l, run.val), run.mat(l, run.disc)
    attn_v = run.refv(l, "attn_layer", run.val).mean()
    Vn = run.V[l].loc[run.val].to_numpy(dtype=np.float64)
    rows = []
    nH = Mv.shape[1]
    for h in range(nH):
        s = summarize(Mv[:, h])
        others = (Mv.sum(1) - Mv[:, h]) / (nH - 1)
        sp = summarize(Mv[:, h] - others, with_p=False)
        rows.append(dict(layer=l, head=h, val_mean=s["mean"], val_ci_lo=s["ci_lo"], val_ci_hi=s["ci_hi"], val_pos_frac=s["pos_frac"], val_p=s["p"],
                         disc_mean=float(Md[:, h].mean()), spec=sp["mean"], spec_ci_lo=sp["ci_lo"], spec_ci_hi=sp["ci_hi"],
                         share_of_attn_layer=float(Mv[:, h].mean() / attn_v) if attn_v != 0 else np.nan, vnorm_mean=float(Vn[:, h].mean())))
    df = pd.DataFrame(rows)
    df["val_rank"] = df.val_mean.rank(ascending=False).astype(int)
    df["disc_rank"] = df.disc_mean.rank(ascending=False).astype(int)
    return df.sort_values("val_mean", ascending=False).reset_index(drop=True)


def rank_stability(rk: pd.DataFrame, top: int = 3) -> dict:
    """Agreement between the discovery and validation head rankings of one layer (Spearman rho, top-k overlap)."""
    from scipy.stats import spearmanr
    rho = spearmanr(rk.disc_mean, rk.val_mean).correlation
    tv = set(rk.sort_values("val_mean", ascending=False).head(top)["head"].tolist())
    td = set(rk.sort_values("disc_mean", ascending=False).head(top)["head"].tolist())
    return {"spearman_disc_val": float(rho), f"top{top}_overlap": len(tv & td), f"top{top}_val": sorted(tv), f"top{top}_disc": sorted(td)}


# ---------------------------------------------------------------------------------------------------------------
# additivity
# ---------------------------------------------------------------------------------------------------------------
def additivity(run: HeadRun, l: int, ids: list[int] | None = None) -> dict:
    """Sum over heads of the single-head rescues vs the attn_layer rescue (same pass), per case."""
    ids = ids or run.val
    M = run.mat(l, ids)
    ssum = M.sum(1)
    attn = run.refv(l, "attn_layer", ids)
    moe = run.refv(l, "layer", ids)
    blk = run.refv(l, "block", ids)
    gap = summarize(ssum - attn, with_p=False)
    s_sum, s_attn = summarize(ssum, with_p=False), summarize(attn, with_p=False)
    r = float(np.corrcoef(ssum, attn)[0, 1]) if ssum.std() > 0 and attn.std() > 0 else np.nan
    # noise reference: the sum of nH per-row bf16 noises; approximate the per-row noise by the identity-row spread is
    # not available here, so report the spread of the gap directly
    return {"layer": l, "n": len(ids), "sum_heads_mean": s_sum["mean"], "sum_heads_ci": (s_sum["ci_lo"], s_sum["ci_hi"]),
            "attn_layer_mean": s_attn["mean"], "attn_layer_ci": (s_attn["ci_lo"], s_attn["ci_hi"]),
            "gap_mean": gap["mean"], "gap_ci": (gap["ci_lo"], gap["ci_hi"]), "gap_sd": float((ssum - attn).std(ddof=1)),
            "per_case_r": r, "frac_sum_gt_attn": float((ssum > attn).mean()),
            "moe_layer_mean": float(moe.mean()), "block_mean": float(blk.mean()),
            "max_single_head_mean": float(M.mean(0).max()), "n_heads_positive_mean": int((M.mean(0) > 0).sum())}


# ---------------------------------------------------------------------------------------------------------------
# minimal head sets (additive approximation)
# ---------------------------------------------------------------------------------------------------------------
def minimal_set_population(run: HeadRun, l: int, frac: float = 0.8, select_ids=None, eval_ids=None) -> dict:
    """Greedy on the additive approximation: heads ordered by their mean rescue on select_ids (default: discovery),
    the cumulative sum of the validation means is compared with frac x the attn_layer validation mean (and with
    frac x the summed-heads mean). Exact only if head rescues add up; the additivity table quantifies the error."""
    select_ids = select_ids or run.disc
    eval_ids = eval_ids or run.val
    md = run.mat(l, select_ids).mean(0)
    mv = run.mat(l, eval_ids).mean(0)
    order = np.argsort(-md)
    attn = run.refv(l, "attn_layer", eval_ids).mean()
    total = mv.sum()
    out = {"layer": l, "frac": frac, "attn_layer_val_mean": float(attn), "sum_heads_val_mean": float(total), "order_by_disc": [int(h) for h in order]}
    cum = np.cumsum(mv[order])
    out["cum_val_by_rank"] = [round(float(x), 3) for x in cum]
    for name, ref in (("attn_layer", attn), ("sum_heads", total)):
        if ref <= 0:
            out[f"k_{name}"] = None
            continue
        hit = np.nonzero(cum >= frac * ref)[0]
        out[f"k_{name}"] = int(hit[0]) + 1 if len(hit) else None
        out[f"set_{name}"] = [int(h) for h in order[: out[f"k_{name}"]]] if out[f"k_{name}"] else None
        out[f"cum_at_k_{name}"] = float(cum[out[f"k_{name}"] - 1]) if out[f"k_{name}"] else None
    # also the top-3 set's share
    out["top3_disc"] = [int(h) for h in order[:3]]
    out["top3_val_sum"] = float(mv[order[:3]].sum())
    out["top3_share_of_attn"] = float(mv[order[:3]].sum() / attn) if attn > 0 else np.nan
    return out


def minimal_set_per_case(run: HeadRun, l: int, frac: float = 0.8, ids=None) -> dict:
    """Per case: the number of heads (sorted by that case's own head rescues) whose additive sum reaches frac x the
    case's attn_layer rescue. Only cases with attn_layer rescue > 0.25 (a quarter logit) are counted; cases where even
    all positive heads do not reach the target are reported as unreachable."""
    ids = ids or run.val
    M = run.mat(l, ids)
    attn = run.refv(l, "attn_layer", ids)
    ks, unreach, n_used = [], 0, 0
    for i in range(len(ids)):
        if attn[i] <= 0.25:
            continue
        n_used += 1
        r = np.sort(M[i])[::-1]
        cum = np.cumsum(np.maximum(r, 0))
        hit = np.nonzero(cum >= frac * attn[i])[0]
        if len(hit):
            ks.append(int(hit[0]) + 1)
        else:
            unreach += 1
    ks = np.array(ks)
    return {"layer": l, "frac": frac, "n_cases_used": n_used, "n_unreachable": unreach,
            "k_median": float(np.median(ks)) if len(ks) else np.nan, "k_mean": float(ks.mean()) if len(ks) else np.nan,
            "k_q25": float(np.percentile(ks, 25)) if len(ks) else np.nan, "k_q75": float(np.percentile(ks, 75)) if len(ks) else np.nan,
            "frac_k_le_3": float((ks <= 3).mean()) if len(ks) else np.nan, "frac_k_eq_1": float((ks == 1).mean()) if len(ks) else np.nan}


# ---------------------------------------------------------------------------------------------------------------
# attention distributions
# ---------------------------------------------------------------------------------------------------------------
def position_classes(run: HeadRun, cid: int) -> np.ndarray:
    """Class index per position of a case (POS_CLASSES order), priority final > subj_last > subj_other > pos0 > other."""
    row = run.cases.loc[cid]
    T = int(row.n_tokens)
    sp = json.loads(row.subject_pos) if isinstance(row.subject_pos, str) else list(row.subject_pos)
    cls = np.full(T, POS_CLASSES.index("other"), dtype=np.int64)
    cls[0] = POS_CLASSES.index("pos0")
    for p in sp[:-1]:
        cls[p] = POS_CLASSES.index("subj_other")
    cls[sp[-1]] = POS_CLASSES.index("subj_last")
    cls[T - 1] = POS_CLASSES.index("final")
    return cls


def attention_mass(run: HeadRun, l: int, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Mass per position class, [n, nH, n_classes] for the clean and the noised run (fp32)."""
    assert run.attn is not None, "no head_attn_final.npz in the run dir"
    A = run.attn[l].astype(np.float32)  # [2n, nH, T]
    n = len(run.attn_ids)
    pos = {c: i for i, c in enumerate(run.attn_ids)}
    out_c = np.zeros((len(ids), run.nH, len(POS_CLASSES)), dtype=np.float32)
    out_n = np.zeros_like(out_c)
    for j, cid in enumerate(ids):
        i = pos[cid]
        cls = position_classes(run, cid)
        T = len(cls)
        for k in range(len(POS_CLASSES)):
            m = cls == k
            if m.any():
                out_c[j, :, k] = A[i, :, :T][:, m].sum(-1)
                out_n[j, :, k] = A[n + i, :, :T][:, m].sum(-1)
    return out_c, out_n


def attention_table(run: HeadRun, l: int, heads: list[int], ids=None) -> pd.DataFrame:
    """Mean mass per class (clean and noised) for the given heads, with bootstrap CIs of the clean-noised shift on the
    subject classes."""
    ids = ids or run.val
    mc, mn = attention_mass(run, l, ids)
    rows = []
    for h in heads:
        r = {"layer": l, "head": h}
        for k, name in enumerate(POS_CLASSES):
            r[f"clean_{name}"] = float(mc[:, h, k].mean())
            r[f"noised_{name}"] = float(mn[:, h, k].mean())
        subj_c = mc[:, h, 1] + mc[:, h, 2]
        subj_n = mn[:, h, 1] + mn[:, h, 2]
        lo, hi = bootstrap_ci(subj_n - subj_c)
        r.update(clean_subject=float(subj_c.mean()), noised_subject=float(subj_n.mean()), subject_shift=float((subj_n - subj_c).mean()),
                 subject_shift_ci_lo=lo, subject_shift_ci_hi=hi)
        rows.append(r)
    return pd.DataFrame(rows)


def rescue_vs_subject_attention(run: HeadRun, l: int, ids=None) -> dict:
    """Across the heads of a layer: Spearman correlation between the mean rescue and the mean clean attention mass on
    the subject span (subj_last + subj_other), and on the last subject token alone."""
    from scipy.stats import spearmanr
    ids = ids or run.val
    mc, mn = attention_mass(run, l, ids)
    mv = run.mat(l, ids).mean(0)
    subj = (mc[:, :, 1] + mc[:, :, 2]).mean(0)
    last = mc[:, :, 1].mean(0)
    drop = ((mc[:, :, 1] + mc[:, :, 2]) - (mn[:, :, 1] + mn[:, :, 2])).mean(0)
    return {"layer": l, "spearman_rescue_vs_clean_subject_mass": float(spearmanr(mv, subj).correlation),
            "spearman_rescue_vs_clean_subj_last_mass": float(spearmanr(mv, last).correlation),
            "spearman_rescue_vs_subject_mass_drop": float(spearmanr(mv, drop).correlation),
            "layer_mean_clean_subject_mass": float(subj.mean()), "layer_mean_noised_subject_mass": float((mn[:, :, 1] + mn[:, :, 2]).mean())}


# ---------------------------------------------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------------------------------------------
def fmt_ci(m, lo, hi, d=3):
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def md_table(df: pd.DataFrame, cols: list[tuple[str, str]], floatfmt="{:+.3f}") -> str:
    """cols: list of (column, header). Floats formatted, ints as is."""
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in df.to_dict("records"):  # keeps per-column dtypes (iterrows would upcast ints to floats)
        cells = []
        for c, _ in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v) if not np.isnan(v) else "n/a")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
