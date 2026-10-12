"""ext13atp analysis: join engine truth with the HF estimates; agreement metrics; routing-switch split; add-back curves.

Joined table (load_joined), one row per (directed case, component):
  truth            engine activation patch rescue (bf16 logits, the official ground truth)
  truth_hf         HF activation patch rescue, Delta read out in fp32 from the final-norm output (same numerics as the
                   gradients; no logit rounding); truth_hf_bf16 the same patch read out with bf16 logits as the engine
  frozen           HF patch with the downstream top-k frozen to the corrupted run's (only for nodes whose free patch
                   switched routing; = truth_hf otherwise)
  atp, atp_clean, ig8 (EAP-IG), ig8mid, ig16, dla   gradient / direct estimates (moetrace/ext13atp.py)
  fix1             one sublayer after the node exact (attention nodes: the next router = "router fix"; MoE nodes: the next
                   attention = "QK fix"), linearised after it
  atpstar          one attention AND one router after the node exact (QK + router fix), linearised after it
  cutL             everything exact up to the last residual h_post(L-1); linear only through the final RMSNorm + unembed
Expert rows exist for every expert routed in the clean or corrupted run in EITHER implementation; an expert routed in
neither run of one implementation has delta_e = 0 there, so its truth / estimates are exactly 0 (filled).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats as SS

from . import ext13atp as A

KEYS = ["case_id", "ctype", "layer", "idx"]
EST = ("atp", "atpstar", "fix1", "ig8", "ig8mid", "ig16", "atp_clean", "dla", "cutL")
EST_LABEL = {"atp": "AtP", "atpstar": "AtP* (QK + router fix)", "fix1": "AtP + 1-sublayer fix", "ig8": "EAP-IG (m=8)",
             "ig8mid": "IG midpoint (m=8)", "ig16": "EAP-IG (m=16)", "atp_clean": "AtP at the clean run", "dla": "DLA",
             "cutL": "exact to h_post(L-1)", "truth_hf": "HF activation patch", "truth": "engine activation patch"}
L = 16


def load_joined(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    od = A.run_dir(task)
    tr = pd.read_parquet(os.path.join(od, "truth_rows.parquet"))
    tc = pd.read_parquet(os.path.join(od, "truth_cases.parquet"))
    hn = pd.read_parquet(os.path.join(od, "hf_nodes.parquet"))
    tr = tr.rename(columns={"rescue": "truth", "vnorm": "vnorm_eng"})[KEYS + ["truth", "vnorm_eng"]]
    J = tr.merge(hn, on=KEYS, how="outer", indicator=True)
    J["in_eng"] = J["_merge"] != "right_only"
    J["in_hf"] = J["_merge"] != "left_only"
    J = J.drop(columns="_merge")
    fill0 = ["truth", "vnorm_eng", "atp", "atp_clean", "ig8", "ig8mid", "ig16", "dla", "vnorm", "rescue_hf", "rescue_hf_bf16"]
    for c in fill0:
        J[c] = J[c].fillna(0.0)
    cutc = [f"c{c:02d}" for c in range(2 * L)]
    cut = J.ctype.map(lambda t: 0 if t in A.ATTN_SIDE else 1).to_numpy() + 2 * J.layer.to_numpy()
    J["cut"] = cut
    C = J[cutc].to_numpy(dtype=np.float64)
    ar = np.arange(len(J))
    J["fix1"] = np.nan_to_num(C[ar, np.minimum(cut + 1, 2 * L - 1)])
    J["atpstar"] = np.nan_to_num(C[ar, np.minimum(cut + 2, 2 * L - 1)])
    J["cutL"] = np.nan_to_num(C[ar, 2 * L - 1])
    J["truth_hf"] = J.rescue_hf
    J["truth_hf_bf16"] = J.rescue_hf_bf16
    J["frozen"] = np.where(J.rescue_frozen.notna(), J.rescue_frozen, J.truth_hf)
    for c in ("n_switch_layers", "n_switch_slots", "switch_mass"):
        J[c] = J[c].fillna(0)
    J["first_switch"] = J.first_switch.fillna(-1)
    J["switch"] = J.n_switch_layers > 0
    # engine routing -> clean / corrupt activity of expert rows
    rc = {int(c): json.loads(r) for c, r in zip(tc.case_id, tc.route_clean)}
    ro = {int(c): json.loads(r) for c, r in zip(tc.case_id, tc.route_corrupt)}
    isx = (J.ctype == "expert").to_numpy()
    ca = np.zeros(len(J), dtype=np.int8)
    oa = np.zeros(len(J), dtype=np.int8)
    for j in np.nonzero(isx)[0]:
        c, l, e = int(J.case_id.iat[j]), int(J.layer.iat[j]), int(J.idx.iat[j])
        ca[j] = e in rc[c][l]
        oa[j] = e in ro[c][l]
    J["clean_active_eng"] = ca
    J["corrupt_active_eng"] = oa
    cases = tc.copy()
    hc = os.path.join(od, "hf_cases.parquet")
    if os.path.exists(hc):
        cases = cases.merge(pd.read_parquet(hc).drop(columns=["split"], errors="ignore"), on="case_id", how="left")
    J = J.merge(cases[["case_id", "split", "drop"]], on="case_id", how="left")
    return J.reset_index(drop=True), cases


# ===================================================================================================================
# agreement metrics
# ===================================================================================================================
def _rank_metrics(t: np.ndarray, e: np.ndarray, ks=(5, 10)) -> dict:
    """Spearman, Kendall tau-b, signed top-k overlap and effect recovery@k (sum of truth over the estimate's top k /
    sum of truth over the truth's top k) for one row."""
    out = {}
    if len(t) < 3 or np.all(t == t[0]) or np.all(e == e[0]):
        out.update(spearman=np.nan, kendall=np.nan)
    else:
        out["spearman"] = SS.spearmanr(t, e).statistic
        out["kendall"] = SS.kendalltau(t, e).statistic
    ot = np.argsort(-t, kind="stable")
    oe = np.argsort(-e, kind="stable")
    for k in ks:
        kk = min(k, len(t))
        out[f"top{k}"] = len(set(ot[:kk]) & set(oe[:kk])) / kk
        best = t[ot[:kk]].sum()
        out[f"rec{k}"] = t[oe[:kk]].sum() / best if best > 0 else np.nan
    return out


CT_GROUPS = {"expert": ("expert",), "head": ("head",), "attn": ("attn",), "moe": ("moe",), "head+expert": ("head", "expert"),
             "layer (attn+moe)": ("attn", "moe")}


def per_row_agreement(J: pd.DataFrame, truth_col: str, ests=EST, groups=CT_GROUPS) -> pd.DataFrame:
    rows = []
    for g, cts in groups.items():
        sub = J[J.ctype.isin(cts)]
        ks = (3, 5) if g in ("attn", "moe", "layer (attn+moe)") else (5, 10)
        for c, d in sub.groupby("case_id"):
            t = d[truth_col].to_numpy(dtype=np.float64)
            for est in ests:
                m = _rank_metrics(t, d[est].to_numpy(dtype=np.float64), ks)
                m.update(group=g, case_id=int(c), est=est, n=len(d))
                rows.append(m)
    return pd.DataFrame(rows)


def pooled_agreement(J: pd.DataFrame, truth_col: str, ests=EST, groups=CT_GROUPS) -> pd.DataFrame:
    """Pooled over (row, component): Pearson r, OLS slope of estimate on truth, R^2 of estimate as a predictor
    (1 - SSE / SST), all in logits; also on drop-normalised values."""
    rows = []
    for g, cts in groups.items():
        sub = J[J.ctype.isin(cts)]
        t = sub[truth_col].to_numpy(dtype=np.float64)
        for est in ests:
            e = sub[est].to_numpy(dtype=np.float64)
            r = np.corrcoef(t, e)[0, 1]
            slope = np.polyfit(t, e, 1)[0] if t.std() > 0 else np.nan
            r2 = 1 - ((t - e) ** 2).sum() / ((t - t.mean()) ** 2).sum()
            rows.append(dict(group=g, est=est, n=len(sub), pearson=r, slope=slope, r2_pred=r2,
                             mae=np.abs(t - e).mean(), mean_abs_truth=np.abs(t).mean()))
    return pd.DataFrame(rows)


def population_agreement(J: pd.DataFrame, truth_col: str, ests=EST, groups=CT_GROUPS, split=None) -> pd.DataFrame:
    """Population ranking: per component the mean over rows (absent = 0) of truth and estimate; Spearman, Kendall,
    top-10 overlap and recovery over components."""
    rows = []
    sub0 = J if split is None else J[J.split == split]
    n_rows = sub0.case_id.nunique()
    for g, cts in groups.items():
        sub = sub0[sub0.ctype.isin(cts)]
        agg = sub.groupby(["ctype", "layer", "idx"])[[truth_col] + list(ests)].sum() / n_rows
        t = agg[truth_col].to_numpy()
        ks = (3, 5) if g in ("attn", "moe", "layer (attn+moe)") else (5, 10)
        for est in ests:
            m = _rank_metrics(t, agg[est].to_numpy(), ks)
            m.update(group=g, est=est, n_comp=len(agg))
            rows.append(m)
    return pd.DataFrame(rows)


def summarise_rows(R: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (group, est): median, mean and pair-bootstrap CI of a per-row metric (pairs = directed ids // 2)."""
    from .ext7_pairs import summ
    out = []
    for (g, est), d in R.groupby(["group", "est"], sort=False):
        s = summ(pd.Series(d[metric].to_numpy(), index=d.case_id.to_numpy()), with_p=False)
        out.append(dict(group=g, est=est, metric=metric, median=float(np.nanmedian(d[metric])), mean=s["mean"],
                        ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], n_rows=int(d[metric].notna().sum())))
    return pd.DataFrame(out)


# ===================================================================================================================
# routing switches
# ===================================================================================================================
def routing_table(J: pd.DataFrame) -> pd.DataFrame:
    """Per component type: share of nodes whose HF patch switches downstream top-k, error of AtP vs the HF truth split
    by switch, and the decomposition exact - AtP = (exact - frozen) + (frozen - AtP)."""
    rows = []
    for g, cts in CT_GROUPS.items():
        if g in ("head+expert", "layer (attn+moe)"):
            continue
        sub = J[J.ctype.isin(cts) & J.in_hf]
        for sw_lab, d in (("all", sub), ("switch", sub[sub.switch]), ("no switch", sub[~sub.switch])):
            if len(d) < 3:
                continue
            t, a, fz = d.truth_hf.to_numpy(), d.atp.to_numpy(), d.frozen.to_numpy()
            sw_eff, smooth = t - fz, fz - a
            rows.append(dict(group=g, subset=sw_lab, n=len(d), share=len(d) / len(sub),
                             r_atp_truth=np.corrcoef(t, a)[0, 1] if t.std() > 0 else np.nan,
                             r_atp_frozen=np.corrcoef(fz, a)[0, 1] if fz.std() > 0 else np.nan,
                             r_ig8_truth=np.corrcoef(t, d.ig8)[0, 1] if t.std() > 0 else np.nan,
                             r_atpstar_truth=np.corrcoef(t, d.atpstar)[0, 1] if t.std() > 0 else np.nan,
                             mean_abs_truth=np.abs(t).mean(), mae_atp=np.abs(t - a).mean(),
                             mean_abs_switch_effect=np.abs(sw_eff).mean(), mean_abs_smooth_err=np.abs(smooth).mean(),
                             sse_total=((t - a) ** 2).sum(), sse_switch=(sw_eff ** 2).sum(), sse_smooth=(smooth ** 2).sum(),
                             mean_switch_layers=d.n_switch_layers.mean(), mean_switch_slots=d.n_switch_slots.mean(),
                             frac_big_switch=float((np.abs(sw_eff) > np.maximum(0.1, 0.25 * np.abs(t))).mean())))
    return pd.DataFrame(rows)


def cut_curve(J: pd.DataFrame, truth_col: str = "truth_hf", smax: int = 8) -> pd.DataFrame:
    """Pooled Pearson r and per-row median Spearman of the cut estimate at offset s (s exact sublayers after the node,
    capped at the last residual) against the HF truth; s = 0 is AtP (the patched-run version), s = 'exact' the truth."""
    C = J[[f"c{c:02d}" for c in range(2 * L)]].to_numpy(dtype=np.float64)
    cut = J.cut.to_numpy()
    ar = np.arange(len(J))
    rows = []
    for g, cts in (("expert", ("expert",)), ("head", ("head",)), ("attn", ("attn",)), ("moe", ("moe",))):
        msk = J.ctype.isin(cts).to_numpy() & J.in_hf.to_numpy()
        for s in list(range(smax + 1)):
            v = np.nan_to_num(C[ar, np.minimum(cut + s, 2 * L - 1)])
            t = J[truth_col].to_numpy()
            rr = np.corrcoef(t[msk], v[msk])[0, 1]
            sp = J[msk].assign(_v=v[msk]).groupby("case_id").apply(
                lambda d: SS.spearmanr(d[truth_col], d._v).statistic if d[truth_col].std() > 0 and d._v.std() > 0 else np.nan, include_groups=False)
            rows.append(dict(group=g, s=s, pearson=rr, spearman_median=float(np.nanmedian(sp))))
    return pd.DataFrame(rows)


# ===================================================================================================================
# add-back curves
# ===================================================================================================================
def addback_summary(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean normalised add-back r(k) per (pool, order, k) over validation rows with pair-bootstrap CI; AUC over
    k = 1..16 (linear interpolation between grid points, mean of r) per row, summarised."""
    from .ext7_pairs import summ
    ab = pd.read_parquet(os.path.join(A.run_dir(task), "addback_rows.parquet"))
    cur = []
    for (pool, o, k), d in ab.groupby(["pool", "order", "k"]):
        s = summ(pd.Series(d.r.to_numpy(), index=d.case_id.to_numpy()), with_p=False)
        cur.append(dict(pool=pool, order=o, k=int(k), mean=s["mean"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], n=s["n"]))
    cur = pd.DataFrame(cur)
    auc = []
    kk = np.arange(1, 17)
    for (pool, o), d in ab[ab.pool != "ceiling"].groupby(["pool", "order"]):
        vals = {}
        for c, dd in d.groupby("case_id"):
            dd = dd.sort_values("k")
            vals[c] = float(np.interp(kk, dd.k.to_numpy(), dd.r.to_numpy()).mean())
        s = summ(pd.Series(vals), with_p=False)
        auc.append(dict(pool=pool, order=o, auc16=s["mean"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"]))
    return cur, pd.DataFrame(auc)
