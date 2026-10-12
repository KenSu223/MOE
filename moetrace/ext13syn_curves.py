"""ext13syn_curves (Phase 5, agent p5-synth), Task B: why the add-back curves differ (RESEARCH_PLAN.md "P5-4").

Phase 3 found two curve shapes: WinoGrande option swap rises to the all-MoE ceiling M without overshoot, CounterFact
overshoots M and falls at the end; Phase 5's attribute -> entity conditions overshoot strongly. This module explains the
shape per condition and model from existing rows (CPU) plus one optional GPU check (scripts/ext13syn_gpucheck.py).

Quantities per condition (validation directed cases of the add-back run; ratios = sum over cases / sum of drops,
unit-cluster bootstrap 5,000 resamples, seed 0, units = pairs / cases as in ext8 / ext13ab):
  * M (all MoE outputs patched), the oracle add-back curve r(k) (each case's own single-expert ranking), its maximum and
    argmax k*, overshoot = max r - M;
  * late MoE = M(<= L-5) - M (A0 'int_moe' minus 'all_moe'): what the last four MoE layers do jointly (> 0: they work
    against the answer);
  * single-expert decomposition with CROSS-FITTED signs. The per-row single rescues carry bf16 noise (SD ~0.17 logits
    per single in Qwen3, 384 singles per row), so 'sum of negative singles' taken on the same rows is dominated by noise
    (about -2.6 logits per row from L0-L4 alone). The sign of each (layer, expert) is therefore fixed on the DISCOVERY
    cases (all-case population effect, 0 when not routed) and the masses are summed on the VALIDATION cases: N = sum of
    the singles of the discovery-negative experts, P = discovery-positive, U = experts unseen in discovery;
  * the same sets' direct logit attribution (addback_dla.parquet: DLA of delta_e = c_e(clean) - c_e(corrupted), final
    norm frozen at the corrupted run) -> direct share D/T of the negative set, by depth band;
  * compensation regression (base run dla_rows.parquet, exact per-layer DLA of attention / MoE outputs, clean minus
    corrupted): per case y = DLA of the last-4 MoE layers, x_A = DLA of all attention outputs, x_M = DLA of the MoE
    outputs below L-4 (+ embedding); OLS y ~ x_A + x_M. A negative coefficient = the late MoE writes against the answer
    in proportion to the answer evidence that the attention (or the earlier MoE) has written.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

from . import ext8_addback as X
from . import ext13ab as AB
from .models import RESULTS

sys.path.insert(0, "/home/ubuntu/MOE/scripts")
NB = 5000
BANDS = ("first half", "L/2..L-9", "L-8..L-5", "last 4")
P3_KEYS = {"cf_qwen3": ("qwen3", "CounterFact (Phase 3)", "qwen3_str"), "cf_mixtral": ("mixtral", "CounterFact (Phase 3)", "mixtral_bos_str"),
           "wino_qwen3": ("qwen3", "WinoGrande option swap (Phase 3)", "wino_qwen3_str"),
           "wino_mixtral": ("mixtral", "WinoGrande option swap (Phase 3)", "wino_mixtral_bos_str")}
PROTO = {"qwen3": "qwen3", "mixtral": "mixtral_bos"}


def band(l: int, L: int) -> str:
    if l < L // 2:
        return BANDS[0]
    if l <= L - 9:
        return BANDS[1]
    if l <= L - 5:
        return BANDS[2]
    return BANDS[3]


# ------------------------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------------------------
def condition_keys(include_rep: bool = True) -> list[str]:
    """Add-back tasks with a complete run: Phase-3 references + Phase-5 registry tasks."""
    out = []
    for k, (model, lab, src) in P3_KEYS.items():
        out.append(k)
    reg = AB.registry()
    for k, s in reg.items():
        if s.family == "rep" and not include_rep:
            continue
        if AB.run_complete(s.out):
            out.append(k)
    return out


def load(key: str) -> dict:
    """Task, add-back rows and per-case matrices of one add-back run."""
    import ext12_analyze as E12
    if key in P3_KEYS:
        model, lab, src = P3_KEYS[key]
        cfg = E12.main_cfg(key)
        t = E12.load_main_task(key)
        out, cond, proto, family = cfg["out"], ("P3_cf" if key.startswith("cf") else "P3_opt"), PROTO[model], "main"
        tlabel = lab
    else:
        spec = AB.registry()[key]
        t = AB.load_task(spec)
        out, cond, proto, model, family, src = spec.out, spec.cond, spec.proto, spec.model, spec.family, spec.src
        tlabel = spec.tlabel
    R = X.load_out(out)
    m = E12.addback_mats(t, R)
    rows = R["rows"]
    val = m["val"]
    a0 = rows[(rows.fam == "a0") & (rows.dir == "d") & rows.case_id.isin(val)]
    m["int"] = a0[a0.order == "int_moe"].groupby("case_id").rescue.mean().reindex(val)
    m["int_attn"] = a0[a0.order == "int_attn"].groupby("case_id").rescue.mean().reindex(val)
    return dict(key=key, t=t, R=R, m=m, out=out, src=src, cond=cond, proto=proto, model=model, family=family, tlabel=tlabel,
                L=t.L, K=t.K)


def signs_from_discovery(t: X.Task) -> dict:
    """(layer, expert) -> discovery all-row population effect (sum of single rescues over discovery rows / n rows)."""
    split = t.rows.case_id.map(dict(zip(t.cases.case_id, t.cases.split)))
    disc = t.rows.row_id[split == "discovery"].tolist()
    acc = {}
    for r in disc:
        for k, v in t.single[r].items():
            acc[k] = acc.get(k, 0.0) + v
    n = max(len(disc), 1)
    return {k: v / n for k, v in acc.items()}


def neg_sets(t: X.Task) -> tuple[set, set]:
    pop = signs_from_discovery(t)
    return {k for k, v in pop.items() if v < 0}, {k for k, v in pop.items() if v > 0}


# ------------------------------------------------------------------------------------------------------------------
# per-case decomposition
# ------------------------------------------------------------------------------------------------------------------
def case_table(c: dict) -> pd.DataFrame:
    """Per validation case: drop, M, M(<= L-5), oracle curve value at the population argmax k*, cross-fitted negative /
    positive / unseen single masses (total and by band), their DLA, total singles and DLA by band."""
    t, m, L = c["t"], c["m"], c["L"]
    neg, pos = neg_sets(t)
    val = m["val"]
    vset = set(val)
    dla = c["R"].get("dla")
    dl = {}
    if dla is not None:
        dsub = dla[dla.row_id.isin(set(t.rows.row_id[t.rows.case_id.isin(vset)]))]
        for r_, g in dsub.groupby("row_id"):
            dl[int(r_)] = dict(zip(zip(g.layer.astype(int), g.expert.astype(int)), g.dla.astype(float)))
    recs = []
    for r in t.rows.itertuples():
        if int(r.case_id) not in vset:
            continue
        sv = t.single[int(r.row_id)]
        dv = dl.get(int(r.row_id), {})
        e = {"case_id": int(r.case_id), "row_id": int(r.row_id)}
        for nm in ("N", "P", "U", "Nrow", "Prow"):
            e[nm] = 0.0
            e["D" + nm] = 0.0
            for b in BANDS:
                e[f"{nm}|{b}"] = 0.0
                e[f"D{nm}|{b}"] = 0.0
        for b in BANDS:
            e[f"T|{b}"] = 0.0
            e[f"DT|{b}"] = 0.0
        for l in range(L):
            e[f"N@{l}"] = e[f"P@{l}"] = e[f"DN@{l}"] = 0.0
        for k, s in sv.items():
            b = band(k[0], L)
            d = dv.get(k, np.nan)
            grp = "N" if k in neg else ("P" if k in pos else "U")
            grp2 = "Nrow" if s < 0 else "Prow"
            for g_ in (grp, grp2):
                e[g_] += s
                e[f"{g_}|{b}"] += s
                if np.isfinite(d):
                    e["D" + g_] += d
                    e[f"D{g_}|{b}"] += d
            if grp in ("N", "P"):
                e[f"{grp}@{k[0]}"] += s
                if grp == "N" and np.isfinite(d):
                    e[f"DN@{k[0]}"] += d
            e[f"T|{b}"] += s
            if np.isfinite(d):
                e[f"DT|{b}"] += d
        e["n_neg_routed"] = sum(1 for k in sv if k in neg)
        recs.append(e)
    df = pd.DataFrame(recs).groupby("case_id").mean(numeric_only=True).reindex(val)
    df["drop"] = m["drop"]
    df["M"] = m["ceil"]
    df["Mint"] = m["int"]
    df["unit"] = m["unit"]
    O = m["mats"]["oracle"]
    rk = O.sum(0) / m["drop"].sum()
    kstar = int(rk.idxmax())
    df["O_kstar"] = O[kstar]
    df.attrs["kstar"] = kstar
    df.attrs["oracle_curve"] = {int(k): float(v) for k, v in rk.items()}
    df.attrs["n_neg"], df.attrs["n_pos"] = len(neg), len(pos)
    return df


def ratios(df: pd.DataFrame, cols: list, den: str = "drop") -> dict:
    rc = X.ratio_cols(df[cols], df[den], df["unit"])
    return {k: [float(v.ratio), float(v.lo), float(v.hi)] for k, v in rc.iterrows()}


def overshoot_boot(c: dict, nb: int = NB, seed: int = 0) -> dict:
    """Population oracle curve: max r - M, k*, with the unit bootstrap (k* re-estimated per resample)."""
    m = c["m"]
    O = m["mats"]["oracle"]
    units = m["unit"].to_numpy()
    u, W = X.boot_weights(units, nb, seed)
    pos = {x: i for i, x in enumerate(u)}
    Mu = np.zeros((len(u), len(units)))
    Mu[[pos[x] for x in units], np.arange(len(units))] = 1.0
    WU = np.vstack([np.ones(len(units)), W @ Mu])
    sd = WU @ m["drop"].to_numpy()
    r = (WU @ O.to_numpy()) / sd[:, None]
    Mc = (WU @ m["ceil"].to_numpy()) / sd
    Mi = (WU @ m["int"].to_numpy()) / sd
    mx = r.max(1)
    f = lambda x: [float(x[0]), float(np.percentile(x[1:], 2.5)), float(np.percentile(x[1:], 97.5))]
    ks = np.array(O.columns)
    return {"M": f(Mc), "M_int": f(Mi), "late": f(Mi - Mc), "max_r": f(mx), "overshoot": f(mx - Mc), "kstar": int(ks[int(np.argmax(r[0]))]),
            "r_end": f(r[:, -1]), "K": int(ks[-1]), "curve": {int(k): float(v) for k, v in zip(ks, r[0])}}


def corr_boot(x: np.ndarray, y: np.ndarray, units: np.ndarray, nb: int = 2000, seed: int = 0) -> list:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, units = x[ok], y[ok], units[ok]
    if len(x) < 10:
        return [float("nan")] * 3
    u, W = X.boot_weights(units, nb, seed)
    pos = {k: i for i, k in enumerate(u)}
    Wc = np.vstack([np.ones(len(x)), W[:, [pos[k] for k in units]]])
    ws = Wc.sum(1)
    mx, my = Wc @ x / ws, Wc @ y / ws
    sxy = Wc @ (x * y) / ws - mx * my
    sxx = Wc @ (x * x) / ws - mx * mx
    syy = Wc @ (y * y) / ws - my * my
    r = sxy / np.sqrt(sxx * syy)
    return [float(r[0]), float(np.nanpercentile(r[1:], 2.5)), float(np.nanpercentile(r[1:], 97.5))]


def ols_boot(Xm: np.ndarray, y: np.ndarray, units: np.ndarray, nb: int = 2000, seed: int = 0) -> list:
    """Weighted OLS y ~ [1, X] on unit resamples; returns per coefficient [point, lo, hi]."""
    ok = np.isfinite(y) & np.isfinite(Xm).all(1)
    Xm, y, units = Xm[ok], y[ok], units[ok]
    Z = np.column_stack([np.ones(len(y)), Xm])
    u, W = X.boot_weights(units, nb, seed)
    pos = {k: i for i, k in enumerate(u)}
    Wc = np.vstack([np.ones(len(y)), W[:, [pos[k] for k in units]]])
    ZZ = np.einsum("ci,cj->cij", Z, Z)
    A = np.einsum("bc,cij->bij", Wc, ZZ)
    bvec = Wc @ (Z * y[:, None])
    beta = np.stack([np.linalg.lstsq(A[i], bvec[i], rcond=None)[0] for i in range(len(A))])
    return [[float(beta[0, j]), float(np.percentile(beta[1:, j], 2.5)), float(np.percentile(beta[1:, j], 97.5))] for j in range(Z.shape[1])]


# ------------------------------------------------------------------------------------------------------------------
# compensation regression from the base run's exact per-layer DLA
# ------------------------------------------------------------------------------------------------------------------
def dla_layers(src: str, cases: list[int]) -> Optional[pd.DataFrame]:
    p = os.path.join(RESULTS, src, "dla_rows.parquet")
    pc = os.path.join(RESULTS, src, "dla_cases.parquet")
    if not os.path.exists(p) or not os.path.exists(pc):
        return None
    d = pd.read_parquet(p)
    d = d[d.case_id.isin(set(cases))]
    if d.case_id.nunique() < len(set(cases)):
        return None
    dc = pd.read_parquet(pc).drop_duplicates("case_id").set_index("case_id").reindex(cases)
    L = int(d.layer.max()) + 1
    d["a"] = d.attn_c - d.attn_k
    d["mo"] = d.moe_c - d.moe_k
    late = d[d.layer >= L - 4].groupby("case_id").mo.sum()
    early = d[d.layer < L - 4].groupby("case_id").mo.sum()
    att = d.groupby("case_id").a.sum()
    att_late = d[d.layer >= L - 4].groupby("case_id").a.sum()
    out = pd.DataFrame({"y_late_moe": late, "x_attn": att, "x_attn_late": att_late, "x_moe_early": early}).reindex(cases)
    out["x_emb"] = (dc.emb_c - dc.emb_k).to_numpy()
    out["x_moe_early"] = out.x_moe_early + out.x_emb
    out["dla_total"] = (dc.dla_total_c - dc.dla_total_k).to_numpy()
    # clean-run-only version (the corrupted twin's own evidence removed)
    dd = d.copy()
    out["y_late_moe_clean"] = dd[dd.layer >= L - 4].groupby("case_id").moe_c.sum().reindex(cases)
    out["x_attn_clean"] = dd.groupby("case_id").attn_c.sum().reindex(cases)
    out["x_moe_early_clean"] = dd[dd.layer < L - 4].groupby("case_id").moe_c.sum().reindex(cases) + dc.emb_c.to_numpy()
    return out


def dla_from_addback(c: dict, cases: list[int]) -> Optional[pd.DataFrame]:
    """CounterFact fallback (ext6 runs have no per-layer DLA): attention total from addback_direct.dla_attn, late MoE from
    the clean-active experts' DLA in addback_dla (approximation: experts routed only in the corrupted run are missing),
    early MoE = dla_moe - late."""
    R, t, L = c["R"], c["t"], c["L"]
    if R.get("direct") is None or R.get("dla") is None:
        return None
    dr = R["direct"]
    dla = R["dla"]
    late = dla[dla.layer >= L - 4].groupby("row_id").dla.sum()
    rr = dr.set_index("row_id")
    df = pd.DataFrame({"case_id": t.rows.set_index("row_id").case_id.reindex(rr.index).values, "x_attn": rr.dla_attn.values,
                       "dla_moe": rr.dla_moe.values, "y_late_moe": late.reindex(rr.index).values}, index=rr.index)
    df["x_moe_early"] = df.dla_moe - df.y_late_moe
    df["dla_total"] = rr.dla_total.values
    g = df.groupby("case_id")[["y_late_moe", "x_attn", "x_moe_early", "dla_total"]].mean().reindex(cases)
    return g
