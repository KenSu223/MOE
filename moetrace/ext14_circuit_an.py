"""ext14 circuit (p6-circuit): C1 analysis (CPU) of the neuron runs written by scripts/ext14ci_c1_run.py.

Aggregation: CounterFact rows are (case, donor) rows, averaged over donors per case; WinoGrande rows are directed cases
(unit = pair). Every population value is a ratio of case means (sum over cases of the patched rescue / sum of the
expert's own single-expert rescue), with a cluster bootstrap over units (cases / pairs). Validation rows only; rows where
the expert is routed in neither run contribute exactly 0 to numerator and denominator (dense models: rows that fail
the model's own margin are excluded).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from . import ext14_circuit as C

N_BOOT = 2000
T_MIN = 0.25  # per-case statistics only on cases whose expert rescue is >= T_MIN logits


def boot_counts(units: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    u, inv = np.unique(units, return_inverse=True)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(u), size=(n_boot, len(u)))
    cnt = np.zeros((n_boot, len(u)), dtype=np.float64)
    for b in range(n_boot):
        np.add.at(cnt[b], draw[b], 1.0)
    return cnt[:, inv], u  # [n_boot, n_cases] weights per case (cases of one unit share the unit's count)


def interp_k(ks: np.ndarray, r: np.ndarray, thr: float) -> float:
    """First crossing of r >= thr, interpolated linearly in log k between grid points; nan = never."""
    ks = np.asarray(ks, dtype=float)
    r = np.asarray(r, dtype=float)
    hit = np.nonzero(r >= thr)[0]
    if not len(hit):
        return float("nan")
    j = int(hit[0])
    if j == 0:
        return float(ks[0])
    a, b = r[j - 1], r[j]
    if b == a:
        return float(ks[j])
    f = (thr - a) / (b - a)
    return float(np.exp(np.log(ks[j - 1]) + f * (np.log(ks[j]) - np.log(ks[j - 1]))))


class ExpertRun:
    """Case-level matrices of one (task, expert) patch run."""

    def __init__(self, task: str, nm: str):
        self.task, self.nm = task, nm
        od = C.task_dir(task)
        self.rows = pd.read_parquet(os.path.join(od, "c1_rows.parquet"))
        self.p = pd.read_parquet(os.path.join(od, f"c1_patch_{nm}.parquet"))
        self.cfg = C.C1_TASKS[task]
        v = self.rows[self.rows.split == "validation"].copy()
        if self.cfg.get("margin") is not None:
            m = self.cfg["margin"]
            ok = (v.delta_clean >= m) & (v.delta_corrupt <= -m)
            if self.cfg["items"][0] == "wino":
                ok = ok & ok.groupby(v.unit).transform("all")
            v = v[ok]
        self.val = v
        self.n_rows_val = len(v)
        self.case_unit = v.groupby("case_id").unit.first()
        self.cases = self.case_unit.index.to_numpy()
        self.units = self.case_unit.to_numpy()
        self.drop = (v.delta_clean - v.delta_corrupt).groupby(v.case_id).mean().reindex(self.cases).to_numpy()
        self.inter = int(self.p.loc[self.p.ranking == "all", "k"].iloc[0])
        # per (ranking, k): case-mean rescue (rows without a patch line = inactive = 0)
        rid_case = dict(zip(v.row_id, v.case_id))
        nrow_case = v.groupby("case_id").size().reindex(self.cases).to_numpy()
        self.nrow_case = nrow_case
        p = self.p[self.p.row_id.isin(rid_case)]
        self.p_val = p
        ci = {c: i for i, c in enumerate(self.cases)}

        def case_mean(df: pd.DataFrame, col: str = "rescue") -> np.ndarray:
            s = np.zeros(len(self.cases))
            np.add.at(s, df.case_id.map(ci).to_numpy(), df[col].to_numpy())
            return s / nrow_case

        self.T = case_mean(p[p.ranking == "ref"])
        self.T_pred = case_mean(p[p.ranking == "ref"], "pred_dla")
        self.ALL = case_mean(p[p.ranking == "all"])
        self.curves, self.preds = {}, {}
        for rk in C.RANKINGS:
            q = p[p.ranking == rk]
            ks = sorted(q.k.unique())
            self.curves[rk] = (np.array(ks + [self.inter]),
                               np.stack([case_mean(q[q.k == k]) for k in ks] + [self.ALL], 1))
            self.preds[rk] = np.stack([case_mean(q[q.k == k], "pred_dla") for k in ks] + [self.T_pred], 1)
        self.n_active_rows = int(p.row_id.nunique())

    # ---- population curves --------------------------------------------------------------------------------------
    def ratio(self, rk: str, w: Optional[np.ndarray] = None, den: Optional[np.ndarray] = None) -> np.ndarray:
        ks, M = self.curves[rk]
        den = self.T if den is None else den
        if w is None:
            return M.sum(0) / den.sum()
        return (w @ M) / (w @ den)[:, None] if w.ndim == 2 else (w @ M) / (w @ den)

    def rand_ratio(self, w=None):
        return 0.5 * (self.ratio("rand0", w) + self.ratio("rand1", w))

    def summary(self, n_boot: int = N_BOOT) -> dict:
        W, _ = boot_counts(self.units, n_boot)
        out = {"task": self.task, "expert": self.nm, "inter": self.inter, "n_cases": int(len(self.cases)),
               "n_units": int(len(np.unique(self.units))), "n_rows_val": int(self.n_rows_val),
               "n_active_rows": self.n_active_rows,
               "T_mean": float(self.T.mean()), "drop_mean": float(self.drop.mean()),
               "T_over_drop": float(self.T.sum() / self.drop.sum()), "all_over_T": float(self.ALL.sum() / self.T.sum()),
               "direct_share": float(self.T_pred.sum() / self.T.sum())}
        bT = W @ self.T
        out["T_ci"] = [float(np.percentile(bT / W.sum(1), 2.5)), float(np.percentile(bT / W.sum(1), 97.5))]
        curves = {}
        for rk in C.RANKINGS + ("rand",):
            ks = self.curves["dla_case"][0]
            r = self.rand_ratio() if rk == "rand" else self.ratio(rk)
            rb = (0.5 * ((W @ self.curves["rand0"][1]) + (W @ self.curves["rand1"][1])) if rk == "rand"
                  else W @ self.curves[rk][1]) / bT[:, None]
            k50 = interp_k(ks, r, 0.5)
            k80 = interp_k(ks, r, 0.8)
            k80b = np.array([interp_k(ks, rb[b], 0.8) for b in range(len(rb))])
            k50b = np.array([interp_k(ks, rb[b], 0.5) for b in range(len(rb))])
            curves[rk] = {"k": ks.tolist(), "r": r.tolist(), "r_lo": np.percentile(rb, 2.5, 0).tolist(),
                          "r_hi": np.percentile(rb, 97.5, 0).tolist(), "k50": k50, "k80": k80,
                          "k50_ci": _ci_nan(k50b), "k80_ci": _ci_nan(k80b),
                          "never80_frac": float(np.isnan(k80b).mean())}
        out["curves"] = curves
        # per-case k80 (dla_case) on cases with a substantial expert rescue
        ks, M = self.curves["dla_case"]
        big = self.T >= T_MIN
        kc = np.array([interp_k(ks, M[i] / self.T[i], 0.8) for i in np.nonzero(big)[0]])
        out["per_case_k80"] = {"n": int(big.sum()), "median": _nanmed(kc), "q25": _nanq(kc, 25), "q75": _nanq(kc, 75),
                               "never": int(np.isnan(kc).sum()), "le10": float(np.nanmean(kc <= 10)) if len(kc) else None,
                               "le50": float(np.mean(np.nan_to_num(kc, nan=1e9) <= 50)) if len(kc) else None}
        out["agreement"] = self.agreement()
        out["overlap"] = self.overlap()
        return out

    # ---- DLA vs patch ---------------------------------------------------------------------------------------------
    def agreement(self) -> dict:
        p = self.p_val
        a = {}
        ref = p[p.ranking == "ref"]
        a["expert_r"] = _r(ref.pred_dla, ref.rescue)
        sub = p[p.ranking.isin(["dla_case", "mag_case", "dla_pop"]) & (p.k < self.inter)]
        a["subset_r"] = _r(sub.pred_dla, sub.rescue)
        a["subset_slope"] = _slope(sub.pred_dla, sub.rescue)
        sg = p[p.ranking.str.startswith("single")]
        a["single_r"] = _r(sg.pred_dla, sg.rescue)
        a["single_slope"] = _slope(sg.pred_dla, sg.rescue)
        big = sg[sg.pred_dla.abs() >= 0.05]
        a["single_sign_agree"] = float((np.sign(big.pred_dla) == np.sign(big.rescue)).mean()) if len(big) else None
        a["single_n"] = int(len(sg))
        # rescue / pred per k for the dla_case ranking (direct share of the top-k neurons)
        dc = p[p.ranking == "dla_case"]
        g = dc.groupby("k")
        a["dla_case_share_by_k"] = {int(k): float(x.rescue.sum() / x.pred_dla.sum()) if x.pred_dla.sum() != 0 else None
                                    for k, x in g}
        return a

    # ---- overlap of top neurons across cases ------------------------------------------------------------------------
    def overlap(self, ks=(10, 50)) -> dict:
        try:
            full = C.load_full(self.task, self.nm)
        except FileNotFoundError:
            return {}
        pos = {int(r): j for j, r in enumerate(full["row_id"])}
        v = self.val[self.val.row_id.isin(self.p_val.row_id.unique())]
        dla = full["dla"][[pos[int(r)] for r in v.row_id]].astype(np.float32)
        cm = pd.DataFrame(dla).groupby(v.case_id.to_numpy()).mean()
        A = cm.to_numpy()  # case-mean dla [n_cases_active, I]
        out = {"n_cases": int(A.shape[0])}
        if A.shape[0] < 2:
            return out
        for k in ks:
            if k >= A.shape[1]:
                continue
            top = np.argsort(-A, axis=1)[:, :k]
            M = np.zeros_like(A, dtype=bool)
            np.put_along_axis(M, top, True, 1)
            inter = M.astype(np.float32) @ M.T.astype(np.float32)
            jac = inter / (2 * k - inter)
            iu = np.triu_indices(len(A), 1)
            freq = M.mean(0)
            # population top-k (validation case mean) vs each case's top-k
            pop = np.argsort(-A.mean(0))[:k]
            cov = M[:, pop].sum(1) / k
            out[f"k{k}"] = {"jaccard_mean": float(jac[iu].mean()), "jaccard_median": float(np.median(jac[iu])),
                            "n_neurons_any": int(M.any(0).sum()), "n_in_half": int((freq >= 0.5).sum()),
                            "max_freq": float(freq.max()), "pop_cover_mean": float(cov.mean()),
                            "rand_jaccard": float(k / (2 * A.shape[1] - k))}
        pos_sum = np.clip(A, 0, None).sum(0)
        srt = np.sort(pos_sum)[::-1]
        tot = srt.sum()
        out["pos_mass_top"] = {str(n): float(srt[:n].sum() / tot) for n in (10, 50, 200) if n < len(srt)} if tot > 0 else {}
        return out


def _ci_nan(x: np.ndarray):
    y = x[~np.isnan(x)]
    if len(y) < 0.5 * len(x) or not len(y):
        return [None, None]
    return [float(np.percentile(y, 2.5)), float(np.percentile(y, 97.5))]


def _nanmed(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if len(x) and not np.isnan(x).all() else None


def _nanq(x, q):
    x = np.asarray(x, dtype=float)
    return float(np.nanpercentile(x, q)) if len(x) and not np.isnan(x).all() else None


def _r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def dense_sweep(task: str) -> Optional[dict]:
    """MLP-layer rescue per layer (normalised by the drop) on the margin-passing validation rows (dense tasks)."""
    od = C.task_dir(task)
    p = os.path.join(od, "c1_sweep_rows.parquet")
    if not os.path.exists(p):
        return None
    sw = pd.read_parquet(p)
    R = pd.read_parquet(os.path.join(od, "c1_rows.parquet"))
    cfg = C.C1_TASKS[task]
    v = R[R.split == "validation"]
    m = cfg.get("margin", 1.0)
    ok = (v.delta_clean >= m) & (v.delta_corrupt <= -m)
    if cfg["items"][0] == "wino":
        ok = ok & ok.groupby(v.unit).transform("all")
    v = v[ok]
    s = sw[sw.row_id.isin(v.row_id)].merge(v[["row_id", "case_id"]], on="row_id")
    cm = s.groupby(["case_id", "layer"]).rescue.mean().unstack()
    drop = (v.delta_clean - v.delta_corrupt).groupby(v.case_id).mean().reindex(cm.index)
    allv = R[R.split == "validation"]
    return {"n_cases": int(len(cm)), "n_val_cases_total": int(allv.case_id.nunique()),
            "pass_rate_rows": float(ok.mean()), "drop_mean": float(drop.mean()),
            "mlp_rescue_by_layer": cm.mean(0).round(4).tolist(),
            "mlp_norm_by_layer": (cm.mean(0) / drop.mean()).round(4).tolist(),
            "argmax_layer": int(cm.mean(0).idxmax())}


def oracle_summary(task: str, nm: str) -> Optional[dict]:
    """Exact single-neuron oracle (scripts/ext14ci_c1_run.py <task> oracle): curves of the oracle ranking vs the DLA /
    magnitude / fixed-set / random rankings on the same rows, neuron-level DLA vs exact effect, additivity of singles."""
    od = C.task_dir(task)
    ps, po = os.path.join(od, f"c1_oracle_singles_{nm}.parquet"), os.path.join(od, f"c1_oracle_{nm}.parquet")
    if not (os.path.exists(ps) and os.path.exists(po)):
        return None
    S = pd.read_parquet(ps)
    O = pd.read_parquet(po)
    P = pd.read_parquet(os.path.join(od, f"c1_patch_{nm}.parquet"))
    rows = sorted(S.row_id.unique())
    P = P[P.row_id.isin(rows)]
    ref = S[S.ranking == "ref"].set_index("row_id").rescue
    case = S[S.ranking == "ref"].set_index("row_id").case_id
    out = {"n_rows": int(len(rows)), "n_cases": int(case.nunique()), "T_mean": float(ref.mean())}

    def curve(df, rk):
        q = df[df.ranking == rk]
        piv = q.pivot_table(index="row_id", columns="k", values="rescue").reindex(rows)
        cm = piv.groupby(case.reindex(rows).to_numpy()).mean()
        den = ref.groupby(case).mean().reindex(cm.index)
        ks = np.array(list(cm.columns), float)
        r = cm.sum(0).to_numpy() / den.sum()
        return ks, r

    cur = {}
    for rk, df in (("oracle", O), ("dla_case", P), ("mag_case", P), ("dla_pop", P), ("rand0", P)):
        ks, r = curve(df, rk)
        I = int(P.loc[P.ranking == "all", "k"].iloc[0]) if (P.ranking == "all").any() else None
        cur[rk] = {"k": ks.tolist(), "r": np.round(r, 4).tolist(), "k50": interp_k(ks, r, 0.5), "k80": interp_k(ks, r, 0.8)}
    out["curves"] = cur
    sg = S[S.ranking == "oracle_single"]
    out["single_r_dla_vs_exact"] = _r(sg.pred_dla, sg.rescue)
    out["single_slope"] = _slope(sg.pred_dla, sg.rescue)
    tot = sg.groupby("row_id").rescue.sum()
    out["additivity_sum_singles_over_ref"] = float(tot.sum() / ref.reindex(tot.index).sum())
    # sparsity of exact single effects: neurons (sorted by |effect|) holding 80 % of the summed |effect| per row
    k80 = []
    for r_, g in sg.groupby("row_id"):
        a = np.sort(np.abs(g.rescue.to_numpy()))[::-1]
        if a.sum() > 0:
            k80.append(int(np.searchsorted(np.cumsum(a) / a.sum(), 0.8) + 1))
    out["abs_mass_k80_median"] = float(np.median(k80)) if k80 else None
    # does a neuron with zero DLA have exact effect? share of exact |effect| carried by neurons outside the DLA top-50
    return out
