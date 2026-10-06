"""ext11 (Phase 4, Direction 11): are the localised experts "writers" or "computers"?

A writer writes the answer directly into the logit difference at the final position; a computer acts through later
layers. For a single-expert STR patch at layer l (final position, parent = corrupted run, c_e := c_e(clean)):

    total     T = rescue = Delta_patched - Delta_corrupt            (source runs: ext6 CounterFact, ext7 WinoGrande)
    direct    D = DLA of delta_e = c_e(clean) - c_e(corrupt)       (ext8 pass 0: (delta_e * gamma) . (W_U[r] - W_U[r'])
                                                                     / rms(h_final, corrupted run), final norm FROZEN)
    indirect  I = T - D  (everything that happens downstream of layer l at the final position: later attention reads the
                          changed residual through the final position's own query / key / value, later MoE routing and
                          outputs change; plus the final-norm nonlinearity, which D ignores)

At the last layer there is no downstream computation, so T - D there is the frozen-norm error plus bf16 / pass noise
(a built-in check). All values are in logits and as fractions of the row's STR drop (Delta_clean - Delta_corrupt).

Units and aggregation follow ext8: CounterFact rows are (case, donor) pairs averaged over donors per case, bootstrap over
cases; WinoGrande rows are directed cases, bootstrap over pairs (both directions of a pair together). Population values
are ratios of means; 'share' = sum D / sum T over the same units.

This module holds the Part-A loaders and statistics (CPU only); Part C (routing diagnostics) and Part B (vocabulary
projections) helpers are added below them.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from .models import RESULTS

ROOT = "/home/ubuntu/MOE"
N_BOOT = 2000
SUMMARY = os.path.join(RESULTS, "ext11_writer_summary.json")

RUNS = {
    "cf_qwen3": dict(model="qwen3", task="cf", src="qwen3_str", out="qwen3_str_addback", label="Qwen3-30B-A3B-Base",
                     tlabel="CounterFact"),
    "cf_mixtral": dict(model="mixtral", task="cf", src="mixtral_bos_str", out="mixtral_bos_str_addback",
                       label="Mixtral-8x7B (BOS)", tlabel="CounterFact"),
    "wino_qwen3": dict(model="qwen3", task="wino", src="wino_qwen3_str", out="wino_qwen3_str_addback", label="Qwen3-30B-A3B-Base",
                       tlabel="WinoGrande", pairs="data/wino_str/pairs_train_xl_qwen3.parquet", case_sets="data/wino_str/case_sets.json"),
    "wino_mixtral": dict(model="mixtral", task="wino", src="wino_mixtral_bos_str", out="wino_mixtral_bos_str_addback",
                         label="Mixtral-8x7B (BOS)", tlabel="WinoGrande", pairs="data/wino_str/pairs_train_xl_mixtral_bos.parquet",
                         case_sets="data/wino_str/case_sets.json"),
}
# experts selected in earlier directions (two-stage / joint search under STR)
TARGETS = {
    "qwen3": {"wino": [(41, 117)], "cf": [(44, 69), (42, 115)]},
    "mixtral": {"wino": [(20, 0)], "cf": [(19, 2), (21, 1), (18, 1)]},
}


def ename(le) -> str:
    return f"L{int(le[0])}E{int(le[1]):03d}"


def parse_ename(s: str) -> tuple[int, int]:
    a, b = s[1:].split("E")
    return int(a), int(b)


def model_targets(model: str) -> list[tuple[int, int]]:
    t = TARGETS[model]
    return t["wino"] + t["cf"]


def update_summary(key: str, value) -> None:
    s = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {}
    s[key] = value
    tmp = SUMMARY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1, default=_json_default)
    os.replace(tmp, SUMMARY)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------------------------------
# Part A: data
# ------------------------------------------------------------------------------------------------------------------
def load_task(key: str):
    from . import ext8_addback as X
    c = RUNS[key]
    if c["task"] == "cf":
        return X.load_cf(c["model"], c["src"])
    return X.load_wino(c["model"], c["src"], os.path.join(ROOT, c["pairs"]), os.path.join(ROOT, c["case_sets"]))


def pair_table(key: str, t=None) -> pd.DataFrame:
    """One row per (STR row, clean-active (layer, expert)): T (single-expert rescue, source run), D (ext8 DLA),
    drop (source run, same pass as T), with case / unit / split / first-donor flags."""
    c = RUNS[key]
    t = t if t is not None else load_task(key)
    er = pd.read_parquet(os.path.join(RESULTS, c["src"], "str_expert_rows.parquet"),
                         columns=["case_id", "slot", "layer", "kind", "expert", "rescue", "vnorm", "delta_clean", "delta_corrupt",
                                  "clean_active", "corrupt_active", "clean_weight"])
    er = er[(er.kind == "expert") & er.clean_active]
    rid = {(int(a), int(b)): int(r) for r, a, b in zip(t.rows.row_id, t.rows.case_id, t.rows.slot)}
    er = er.assign(row_id=[rid.get((int(a), int(b)), -1) for a, b in zip(er.case_id, er.slot)])
    er = er[er.row_id >= 0]
    drop = (er.delta_clean - er.delta_corrupt).groupby(er.row_id).mean()
    dla = pd.read_parquet(os.path.join(RESULTS, c["out"], "addback_dla.parquet"))
    m = er.drop(columns=["kind", "clean_active"]).merge(dla, on=["row_id", "layer", "expert"], how="inner", validate="one_to_one")
    assert len(m) == len(dla) == len(er), (len(m), len(dla), len(er))
    m = m.rename(columns={"rescue": "T", "dla": "D"})
    m["I"] = m["T"] - m["D"]
    m["drop"] = m.row_id.map(drop).values
    cases = t.cases.set_index("case_id")
    m["split"] = m.case_id.map(cases.split).values
    m["unit"] = m.case_id.map(cases.unit).values
    m["first"] = m.row_id.map(t.rows.set_index("row_id")["first"]).values
    return m.reset_index(drop=True)


def row_drops(m: pd.DataFrame) -> pd.DataFrame:
    """Per STR row: case, unit, split, drop."""
    return m.groupby("row_id").agg(case_id=("case_id", "first"), unit=("unit", "first"), split=("split", "first"),
                                    drop=("drop", "first")).reset_index()


# ------------------------------------------------------------------------------------------------------------------
# Part A: statistics (cluster bootstrap over units; values first averaged over donors per case)
# ------------------------------------------------------------------------------------------------------------------
def boot_matrix(units: np.ndarray, n_boot: int = N_BOOT, seed: int = 0):
    """Unit-cluster bootstrap: (unique units, count matrix [n_boot, n_units])."""
    u = np.unique(units)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    W = np.zeros((n_boot, len(u)), dtype=np.float64)
    np.add.at(W, (np.repeat(np.arange(n_boot), len(u)), idx.ravel()), 1.0)
    return u, W


def unit_sums(df: pd.DataFrame, cols: list[str], u: np.ndarray) -> np.ndarray:
    """[n_units, len(cols)] sums of df[cols] per unit (rows of df carry a 'unit' column), aligned to u."""
    g = df.groupby("unit")[cols].sum()
    return g.reindex(u).fillna(0.0).values


def ratio_ci(num: np.ndarray, den: np.ndarray, W: np.ndarray) -> dict:
    """Ratio of sums (num / den per unit arrays) with bootstrap percentile CI from the count matrix W."""
    full = float(num.sum() / den.sum()) if den.sum() != 0 else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        R = (W @ num) / (W @ den)
    lo, hi = np.nanpercentile(R, [2.5, 97.5]) if np.isfinite(R).any() else (np.nan, np.nan)
    return {"v": full, "lo": float(lo), "hi": float(hi)}


def corr_ci(x: np.ndarray, y: np.ndarray, units: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Pearson r, OLS slope of y on x and of x on y, with a unit-cluster bootstrap (sufficient statistics per unit)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return {"n": int(len(x)), "r": float("nan"), "r_lo": float("nan"), "r_hi": float("nan"), "slope_yx": float("nan"),
                "slope_yx_lo": float("nan"), "slope_yx_hi": float("nan")}
    df = pd.DataFrame(dict(unit=units, n=1.0, x=x, y=y, xx=x * x, yy=y * y, xy=x * y))
    u, W = boot_matrix(units, n_boot)
    S = unit_sums(df, ["n", "x", "y", "xx", "yy", "xy"], u)

    def stats(s):
        n, sx, sy, sxx, syy, sxy = [s[..., i] for i in range(6)]
        cxx = sxx - sx * sx / n
        cyy = syy - sy * sy / n
        cxy = sxy - sx * sy / n
        with np.errstate(invalid="ignore", divide="ignore"):
            return cxy / np.sqrt(cxx * cyy), cxy / cxx, cxy / cyy
    r, b, b2 = stats(S.sum(0))
    B = W @ S
    rb, bb, bb2 = stats(B)
    q = lambda a: np.nanpercentile(a, [2.5, 97.5])
    return {"n": int(len(x)), "r": float(r), "r_lo": float(q(rb)[0]), "r_hi": float(q(rb)[1]),
            "slope_yx": float(b), "slope_yx_lo": float(q(bb)[0]), "slope_yx_hi": float(q(bb)[1]),
            "slope_xy": float(b2), "slope_xy_lo": float(q(bb2)[0]), "slope_xy_hi": float(q(bb2)[1])}


def case_level(m: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Donor-mean values per (case, by...) for T, D, I, drop (CounterFact; identity for WinoGrande)."""
    g = m.groupby(["case_id"] + by).agg(T=("T", "mean"), D=("D", "mean"), I=("I", "mean"), unit=("unit", "first"),
                                          split=("split", "first")).reset_index()
    return g


def case_drops(m: pd.DataFrame) -> pd.Series:
    rd = row_drops(m)
    return rd.groupby("case_id").drop.mean()


def layer_profile(m: pd.DataFrame, L: int, splits=("discovery", "validation")) -> pd.DataFrame:
    """Per layer: mean over cases of sum_e T, D, I (logits and / drop) with CIs; direct share sum D / sum T; |T| mass;
    r and slope of D on T over (case, expert) pairs of the layer."""
    m = m[m.split.isin(splits)]
    cd = case_drops(m)
    cl = case_level(m, ["layer", "expert"])
    cases = cd.index.values
    units = m.groupby("case_id").unit.first().reindex(cases).values
    u, W = boot_matrix(units)
    upos = {x: i for i, x in enumerate(u)}
    drop_u = np.zeros(len(u))
    np.add.at(drop_u, [upos[x] for x in units], cd.values)
    ncase_u = np.zeros(len(u))
    np.add.at(ncase_u, [upos[x] for x in units], 1.0)
    out = []
    lay = cl.groupby(["case_id", "layer"])[["T", "D", "I"]].sum().reset_index()
    lay["absT"] = cl.assign(a=cl["T"].abs()).groupby(["case_id", "layer"]).a.sum().values
    lay["absD"] = cl.assign(a=cl["D"].abs()).groupby(["case_id", "layer"]).a.sum().values
    lay["unit"] = lay.case_id.map(m.groupby("case_id").unit.first())
    for l in range(L):
        s = lay[lay.layer == l]
        S = unit_sums(s, ["T", "D", "I", "absT", "absD"], u)
        row = {"layer": l}
        for i, k in enumerate(["T", "D", "I", "absT", "absD"]):
            row[k] = ratio_ci(S[:, i], ncase_u, W)  # mean logits per case
            row[k + "_frac"] = ratio_ci(S[:, i], drop_u, W)  # fraction of the drop
        row["share"] = ratio_ci(S[:, 1], S[:, 0], W)
        cc = cl[cl.layer == l]
        row["corr"] = corr_ci(cc["T"].values, cc["D"].values, cc.unit.values, n_boot=500)
        out.append(row)
    return out


def band_agreement(m: pd.DataFrame, L: int, splits=("discovery", "validation")) -> dict:
    """D vs T over (case, expert) pairs: all layers and by band (early < L/2, middle, last 8 .. 5, last 4)."""
    m = m[m.split.isin(splits)]
    cl = case_level(m, ["layer", "expert"])
    cd = case_drops(m)
    bands = {"all": (0, L), "early": (0, L // 2), "middle": (L // 2, L - 8), "late": (L - 8, L - 4), "last4": (L - 4, L),
             "last": (L - 1, L)}
    out = {}
    units = m.groupby("case_id").unit.first()
    u, W = boot_matrix(units.reindex(cd.index).values)
    upos = {x: i for i, x in enumerate(u)}
    drop_u = np.zeros(len(u))
    np.add.at(drop_u, [upos[x] for x in units.reindex(cd.index).values], cd.values)
    for b, (lo, hi) in bands.items():
        s = cl[(cl.layer >= lo) & (cl.layer < hi)]
        S = unit_sums(s, ["T", "D", "I"], u)
        out[b] = {"layers": [lo, hi - 1], "corr": corr_ci(s["T"].values, s["D"].values, s.unit.values, n_boot=500),
                  "T_frac": ratio_ci(S[:, 0], drop_u, W), "D_frac": ratio_ci(S[:, 1], drop_u, W),
                  "I_frac": ratio_ci(S[:, 2], drop_u, W), "share": ratio_ci(S[:, 1], S[:, 0], W),
                  "mean_T_minus_D_logits": float((s["T"] - s["D"]).mean()), "n_pairs": int(len(s))}
    return out


def expert_stats(m: pd.DataFrame, experts: list[tuple[int, int]], splits=("validation",)) -> list[dict]:
    """Per expert (cases where it is clean-active): mean T / D / I (logits) and as fractions of the drop (ratio of means
    over the active cases), direct share sum D / sum T, r and slope of D on T across active cases; all-case means
    (0 when not active, the ext8 'pop' convention) as fractions of the mean drop of all cases."""
    m = m[m.split.isin(splits)]
    cd = case_drops(m)
    units_all = m.groupby("case_id").unit.first()
    n_cases = len(cd)
    out = []
    for le in experts:
        s = m[(m.layer == le[0]) & (m.expert == le[1])]
        cl = case_level(s, ["layer", "expert"]) if len(s) else pd.DataFrame(columns=["case_id", "T", "D", "I", "unit"])
        row = {"expert": ename(le), "layer": int(le[0]), "n_active": int(len(cl)), "n_cases": int(n_cases),
               "active_frac": float(len(cl) / n_cases) if n_cases else float("nan")}
        if len(cl) >= 3:
            cl = cl.assign(drop=cl.case_id.map(cd).values)
            u, W = boot_matrix(cl.unit.values)
            S = unit_sums(cl.assign(one=1.0), ["T", "D", "I", "drop", "one"], u)
            for i, k in enumerate(["T", "D", "I"]):
                row[k] = ratio_ci(S[:, i], S[:, 4], W)
                row[k + "_frac"] = ratio_ci(S[:, i], S[:, 3], W)
            row["share"] = ratio_ci(S[:, 1], S[:, 0], W)
            row["corr"] = corr_ci(cl["T"].values, cl["D"].values, cl.unit.values, n_boot=1000)
            # all-case (pop) convention
            ua, Wa = boot_matrix(units_all.reindex(cd.index).values)
            allc = pd.DataFrame(dict(case_id=cd.index, unit=units_all.reindex(cd.index).values, drop=cd.values))
            allc = allc.merge(cl[["case_id", "T", "D"]], on="case_id", how="left").fillna({"T": 0.0, "D": 0.0})
            Sa = unit_sums(allc, ["T", "D", "drop"], ua)
            row["T_allcase_frac"] = ratio_ci(Sa[:, 0], Sa[:, 2], Wa)
            row["D_allcase_frac"] = ratio_ci(Sa[:, 1], Sa[:, 2], Wa)
            row["frac_cases_D_gt_half_T"] = float(((cl["D"] > 0.5 * cl["T"]) & (cl["T"] > 0)).sum() / max((cl["T"] > 0).sum(), 1))
        out.append(row)
    return out


def set_stats(m: pd.DataFrame, experts: list[tuple[int, int]], splits=("validation",)) -> dict:
    """Sum over a set of experts per case (active members only): T, D, I as fractions of the drop, share."""
    m = m[m.split.isin(splits)]
    cd = case_drops(m)
    units = m.groupby("case_id").unit.first().reindex(cd.index).values
    key = set(map(tuple, experts))
    s = m[[(int(a), int(b)) in key for a, b in zip(m.layer, m.expert)]]
    cl = case_level(s, ["layer", "expert"]).groupby("case_id")[["T", "D", "I"]].sum().reindex(cd.index).fillna(0.0)
    df = pd.DataFrame(dict(unit=units, T=cl["T"].values, D=cl["D"].values, I=cl["I"].values, drop=cd.values))
    u, W = boot_matrix(units)
    S = unit_sums(df, ["T", "D", "I", "drop"], u)
    return {"n_cases": int(len(cd)), "T_frac": ratio_ci(S[:, 0], S[:, 3], W), "D_frac": ratio_ci(S[:, 1], S[:, 3], W),
            "I_frac": ratio_ci(S[:, 2], S[:, 3], W), "share": ratio_ci(S[:, 1], S[:, 0], W)}


def pop_top(t, n: int = 10) -> list[tuple[int, int]]:
    """The task's population ranking from ext8 (discovery all-case single-expert rescue), top n."""
    return [le for le, _ in sorted(t.pop.items(), key=lambda kv: -kv[1])[:n]]


def frozen_norm_aggregate(key: str) -> dict:
    """Frozen-norm (DLA) vs exact final RMSNorm for the summed sublayer writes (ext8 addback_direct.parquet, fp32,
    same path): MoE direct, attention direct, all writes (= the drop)."""
    c = RUNS[key]
    d = pd.read_parquet(os.path.join(RESULTS, c["out"], "addback_direct.parquet"))
    drop = d.d_clean_fp32 - d.d_corrupt_fp32
    em = d.d_moe_direct - d.d_corrupt_fp32
    ea = d.d_attn_direct - d.d_corrupt_fp32
    out = {"n_rows": int(len(d)), "mean_drop": float(drop.mean())}
    for name, ex, fr in (("moe", em, d.dla_moe), ("attn", ea, d.dla_attn), ("total", drop, d.dla_total)):
        err = fr - ex
        out[name] = {"exact_frac": float(ex.mean() / drop.mean()), "frozen_frac": float(fr.mean() / drop.mean()),
                     "mean_err_logits": float(err.mean()), "mean_abs_err_logits": float(err.abs().mean()),
                     "median_abs_err_frac_of_drop": float((err.abs() / drop).median()),
                     "p95_abs_err_frac_of_drop": float((err.abs() / drop).quantile(0.95)),
                     "r": float(np.corrcoef(ex, fr)[0, 1]),
                     "rel_err_of_mean": float(err.mean() / ex.mean()) if abs(ex.mean()) > 1e-9 else float("nan")}
    return out


def inpass_noise(key: str, m: pd.DataFrame) -> dict:
    """Pass-to-pass noise of single-expert rescues: ext8 'exact' family k = 1 rows (in-pass singles of each first-donor
    row's top-10) vs the source run's T for the same (row, expert)."""
    from . import ext8_addback as X
    c = RUNS[key]
    od = os.path.join(RESULTS, c["out"])
    parts = []
    for f in sorted(os.listdir(od)):
        if f.startswith("addback_rows_p"):
            x = pd.read_parquet(os.path.join(od, f), columns=["fam", "k", "row_id", "delta", "delta_corrupt", "set"])
            parts.append(x[(x.fam == "exact") & (x.k == 1)])
    x = pd.concat(parts, ignore_index=True)
    le = [X.parse_set(s)[0] for s in x.set]
    x = x.assign(layer=[a for a, _ in le], expert=[b for _, b in le], T8=x.delta - x.delta_corrupt)
    j = x.merge(m[["row_id", "layer", "expert", "T", "D", "unit"]], on=["row_id", "layer", "expert"], how="inner")
    diff = j["T8"] - j["T"]
    return {"n": int(len(j)), "sd_diff": float(diff.std()), "sd_single": float(diff.std() / np.sqrt(2)),
            "mean_diff": float(diff.mean()), "r_T_T8": float(np.corrcoef(j["T"], j["T8"])[0, 1]),
            "r_D_T": float(np.corrcoef(j["D"], j["T"])[0, 1]), "r_D_T8": float(np.corrcoef(j["D"], j["T8"])[0, 1]),
            "r_D_meanT": float(np.corrcoef(j["D"], 0.5 * (j["T"] + j["T8"]))[0, 1])}


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional tabulate dependency."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------
# Part C helpers: token / word classes for the context analysis (heuristic word lists; no POS tagger is installed)
# ------------------------------------------------------------------------------------------------------------------
COPULA = {"was", "is", "were", "are", "be", "been", "being", "am", "seemed", "seems", "looked", "looks", "felt", "feels",
          "became", "becomes", "remained", "got", "get", "appeared"}
DEGREE = {"too", "very", "so", "quite", "rather", "more", "less", "most", "least", "extremely", "really", "pretty", "fairly",
          "incredibly", "highly", "much"}
DET = {"the", "a", "an", "this", "that", "these", "those", "his", "her", "their", "its", "my", "our", "your", "some", "any"}
PREP = {"in", "of", "on", "at", "by", "for", "with", "from", "to", "into", "about", "after", "before", "during", "under",
        "over", "between", "through", "against", "near", "within", "across", "toward", "towards", "as"}
CONJ = {"and", "or", "but", "because", "while", "although", "though", "since", "if", "when", "than", "nor", "yet"}
PRON = {"he", "she", "it", "they", "we", "i", "you", "him", "them", "us", "me", "who", "which", "what"}
PLACE_RELATIONS = {"P17", "P19", "P20", "P27", "P30", "P36", "P131", "P159", "P190", "P276", "P495", "P740", "P937"}
REL_GROUPS = {
    "place": PLACE_RELATIONS,
    "language": {"P103", "P1412", "P37", "P364", "P407"},
    "organisation": {"P176", "P178", "P449", "P127", "P108", "P264", "P463"},
    "occupation / field": {"P106", "P39", "P101", "P413", "P136", "P1303", "P641"},
    "other": {"P140", "P138"},
}


def rel_group(rel: str) -> str:
    for g, s in REL_GROUPS.items():
        if rel in s:
            return g
    return "other"


def cur_class(piece: str) -> str:
    """Class of the current token from its decoded string (leading space = word start)."""
    s = piece
    core = s.strip()
    low = core.lower()
    if core == "":
        return "space/newline"
    starts = s.startswith(" ") or s.startswith("\n")
    if all(not ch.isalnum() for ch in core):
        return "punct"
    if core.isdigit():
        return "number"
    if not starts:
        return "word continuation"
    for name, st in (("copula", COPULA), ("degree adverb", DEGREE), ("determiner", DET), ("preposition", PREP),
                     ("conjunction", CONJ), ("pronoun", PRON)):
        if low in st:
            return name
    if core[0].isupper():
        return "capitalised word"
    return "other word"


def load_cf_vocab() -> dict:
    """CounterFact object strings (target_true and target_new) per relation group, and all subjects."""
    d = json.load(open(os.path.join(ROOT, "data", "counterfact.json")))
    out: dict = {g: set() for g in REL_GROUPS}
    for r in d:
        rr = r["requested_rewrite"]
        g = rel_group(rr["relation_id"])
        out[g].add(rr["target_true"]["str"].strip())
        out[g].add(rr["target_new"]["str"].strip())
    return out


class PhraseMatcher:
    """Does the text starting at a position begin with one of the phrases (whole-word match)?"""

    def __init__(self, phrases):
        self.by_first: dict = {}
        for p in phrases:
            w = p.split()
            if not w or not w[0][:1].isalpha():
                continue
            self.by_first.setdefault(w[0], []).append(p)
        for v in self.by_first.values():
            v.sort(key=len, reverse=True)

    def match(self, text: str) -> Optional[str]:
        t = text.lstrip()
        m = re.match(r"[A-Za-zÀ-ɏ][\w\-\.'À-ɏ]*", t)
        if not m:
            return None
        first = m.group(0).rstrip(".'")
        for p in self.by_first.get(first, ()):
            if t.startswith(p) and (len(t) == len(p) or not (t[len(p)].isalnum())):
                return p
        return None


def ranking_agreement(m: pd.DataFrame, splits=("validation",), top: int = 10) -> dict:
    """Population rankings of (layer, expert) by all-case mean T vs all-case mean D (sum over active cases / n cases):
    Spearman rho over experts active in >= 5 % of the cases, overlap of the top-n lists, rank of the T top-3 under D."""
    from scipy.stats import spearmanr
    m = m[m.split.isin(splits)]
    cl = case_level(m, ["layer", "expert"])
    n = cl.case_id.nunique()
    g = cl.groupby(["layer", "expert"]).agg(T=("T", "sum"), D=("D", "sum"), na=("T", "size")).reset_index()
    g = g[g.na >= 0.05 * n]
    rho = spearmanr(g["T"], g["D"]).statistic
    tT = g.sort_values("T", ascending=False)
    tD = g.sort_values("D", ascending=False)
    topT = [ename((a, b)) for a, b in zip(tT.layer.head(top), tT.expert.head(top))]
    topD = [ename((a, b)) for a, b in zip(tD.layer.head(top), tD.expert.head(top))]
    rankD = {ename((a, b)): i + 1 for i, (a, b) in enumerate(zip(tD.layer, tD.expert))}
    return {"n_experts": int(len(g)), "spearman": float(rho), "top_overlap": len(set(topT) & set(topD)), "top": top,
            "top_by_T": topT, "top_by_D": topD, "rank_under_D_of_T_top3": {e: rankD[e] for e in topT[:3]}}


# ------------------------------------------------------------------------------------------------------------------
# Part B helpers: token classes of the vocabulary
# ------------------------------------------------------------------------------------------------------------------
def first_cont_token(tok, word: str) -> Optional[int]:
    """First token of ' ' + word as a continuation (after 'the'), for BPE and sentencepiece tokenizers alike."""
    base = tok("the", add_special_tokens=False)["input_ids"]
    ids = tok("the " + word.strip(), add_special_tokens=False)["input_ids"]
    if ids[: len(base)] != base or len(ids) <= len(base):
        return None
    return int(ids[len(base)])


def vocab_classes(model: str, tok) -> dict:
    """Token-id sets: WinoGrande trigger vocabulary (every trig_a / trig_b of the model's W1-W6 pairs), CounterFact
    object first tokens per relation group and overall, IOI names."""
    proto = "qwen3" if model == "qwen3" else "mixtral_bos"
    pp = pd.read_parquet(os.path.join(ROOT, f"data/wino_str/pairs_train_xl_{proto}.parquet"), columns=["trig_a", "trig_b"])
    out = {"wg_trigger": {int(x) for x in pp.trig_a} | {int(x) for x in pp.trig_b}}
    d = json.load(open(os.path.join(ROOT, "data", "counterfact.json")))
    words: dict = {g: set() for g in REL_GROUPS}
    for r in d:
        rr = r["requested_rewrite"]
        g = rel_group(rr["relation_id"])
        words[g].add(rr["target_true"]["str"])
        words[g].add(rr["target_new"]["str"])
    allcf = set()
    for g, ws in words.items():
        s = {first_cont_token(tok, w) for w in ws}
        s.discard(None)
        out[f"cf_{g}"] = s
        allcf |= s
    out["cf_any"] = allcf
    names = json.load(open(os.path.join(ROOT, "data/ioi/build.json")))["names"]
    out["ioi_names"] = {x for x in (first_cont_token(tok, n) for n in names) if x is not None}
    return out
