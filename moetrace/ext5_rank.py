"""Phase 2 / F1 (ext5-analysis): expert rankings, population-level minimal sets (additive approximation) and per-case
minimal sets (exact subset patches).

Consumes the all-layer expert passes (results/<run>/expert_rows.parquet, kinds layer / coalition_clean / coalition_union /
expert for every layer) through the public API of analysis.py and ext1_analysis.py (nothing there is modified) and, for
F1.3, the exhaustive subset passes results/<run>_subsets/subset_rows.parquet (schema agreed in RESEARCH_PLAN.md:
case_id, layer, experts = sorted comma-joined expert ids, n_experts, rescue).

Conventions (documented defaults, see results/sections/ext5_f1_rankings.md):
- "block" = the MoE-block output patch of the layer (kind `layer` in the same expert pass), i.e. the paper's layer patch.
- all-case rescue of an expert = its rescue where clean-active, 0 elsewhere (analysis.select_expert / evaluate_expert).
- selections (orderings) are made on the paper discovery split and evaluated on the validation split.
- bootstrap CIs use the same 5,000-resample seed-0 index matrix as stats.summarize, vectorised over experts.
"""
from __future__ import annotations

import itertools
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from . import analysis as A
from . import ext1_analysis as X
from .protocol import active_controls
from .stats import N_BOOT, SEED, bootstrap_ci, ratio_ci, summarize

RUNS = {
    "qwen3_bos_alllayers": dict(model="qwen3", short="qwen3_bos", label="Qwen3-30B-A3B-Base (tokenizer defaults)", n_controls=3,
                                two_stage=(44, 69), second=(42, 115), minimal_layers=(44, 42, 43, 40), ref_layer=44, base_run="qwen3",
                                singletons=((44, 69), (42, 115))),
    "mixtral_bos_alllayers": dict(model="mixtral", short="mixtral_bos", label="Mixtral-8x7B-v0.1 (BOS, tokenizer default)", n_controls=1,
                                  two_stage=(19, 2), second=(18, 1), minimal_layers=(17, 18, 19, 20, 21, 22), ref_layer=19, base_run="mixtral",
                                  singletons=((19, 2), (18, 1))),
    "mixtral_nobos_alllayers": dict(model="mixtral", short="mixtral_nobos", label="Mixtral-8x7B-v0.1 (no BOS, paper protocol)", n_controls=1,
                                    two_stage=(19, 6), second=(18, 1), minimal_layers=(17, 18, 19, 20, 21, 22), ref_layer=19, base_run="mixtral_nobos",
                                    singletons=((19, 6), (18, 1), (19, 2))),
}
METRICS = ["val_rescue", "val_active_only", "val_spec", "block_share", "mean_percentile", "disc_allcase"]
METRIC_LABEL = {"val_rescue": "all-case rescue (val)", "val_active_only": "active-only rescue (val)", "val_spec": "Spec (val)",
                "block_share": "share of block rescue (val)", "mean_percentile": "mean per-case percentile among active (val)",
                "disc_allcase": "all-case rescue (disc; selection statistic)"}
TARGETS = (0.5, 0.8, 0.9)


def pair(layer: int, expert: int) -> str:
    return X.pair(layer, expert)


def boot_idx(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """The exact resampling index matrix stats.bootstrap_ci draws for n cases (so vectorised CIs equal summarize's)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def boot_means(M: np.ndarray, idx: np.ndarray, chunk: int = 16) -> np.ndarray:
    """(n_boot, E) bootstrap means of the columns of M (n, E), chunked over columns."""
    out = np.empty((idx.shape[0], M.shape[1]))
    for j in range(0, M.shape[1], chunk):
        out[:, j:j + chunk] = M[:, j:j + chunk][idx].mean(axis=1)
    return out


def pct(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


# ---------------------------------------------------------------------------------------------------------------
# F1.1 full ranking
# ---------------------------------------------------------------------------------------------------------------
def layer_matrices(cache: X.ExpertCache, layer: int, ids: list[int]) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Clean-active pivot (NaN where inactive), all-case matrix (0 where inactive) and same-pass block rescue on ids."""
    P = cache.piv[layer].reindex(ids)
    P = P[sorted(P.columns)]
    M = P.fillna(0.0).values
    B = cache.layer_rescue.loc[ids, layer]
    return P, M, B


def spec_matrix(cache: X.ExpertCache, layer: int, P: pd.DataFrame, ids: list[int], n_controls: int) -> np.ndarray:
    """All-case Spec[c, e] = rescue(e; 0 if inactive) - mean rescue of active-random controls (protocol.active_controls)."""
    cs = cache.clean_sets[layer]
    experts = [int(e) for e in P.columns]
    col = {e: j for j, e in enumerate(experts)}
    vals = P.values
    S = np.full(vals.shape, np.nan)
    for i, cid in enumerate(ids):
        act = cs.get(cid, [])
        row = vals[i]
        for j, e in enumerate(experts):
            ctrls = active_controls(act, e, n_controls, cid)
            if not ctrls:
                continue
            r = row[j] if e in act else 0.0
            S[i, j] = r - float(np.mean([row[col[c]] for c in ctrls]))
    return S


def rank_layer(cache: X.ExpertCache, layer: int, disc: list[int], val: list[int], n_controls: int, threshold: int) -> pd.DataFrame:
    """Every expert with any clean-active case at this layer: activity, all-case / active-only rescue, Spec, block share,
    per-case rank among the case's active experts, recurrence flag. CIs: vectorised percentile bootstrap (seed 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # experts with no validation activity give empty-slice means (NaN, intended)
        return _rank_layer(cache, layer, disc, val, n_controls, threshold)


def _rank_layer(cache: X.ExpertCache, layer: int, disc: list[int], val: list[int], n_controls: int, threshold: int) -> pd.DataFrame:
    Pd, Md, _ = layer_matrices(cache, layer, disc)
    Pv, Mv, Bv = layer_matrices(cache, layer, val)
    experts = [int(e) for e in Pv.columns]
    n_val = len(val)
    idx = boot_idx(n_val)
    bm = boot_means(Mv, idx)
    lo, hi = pct(bm)
    S = spec_matrix(cache, layer, Pv, val, n_controls)
    Sm = np.nanmean(S, axis=0)
    Sb = boot_means(np.nan_to_num(S, nan=0.0), idx)  # controls exist for every case here (all cases have >= 2 active experts)
    slo, shi = pct(Sb)
    bvals = Bv.values
    block_mean = float(bvals.mean())
    block_b = bvals[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = Mv.mean(0) / block_mean
        share_b = bm / block_b[:, None]
    sh_lo, sh_hi = pct(share_b)
    ranks = Pv.rank(axis=1, ascending=False, method="min")
    n_act = Pv.notna().sum(1).values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        percentile = (n_act[:, None] - ranks.values) / (n_act[:, None] - 1)
    df = pd.DataFrame({
        "layer": layer, "expert": experts,
        "disc_active": Pd.notna().sum(0).values.astype(int), "val_active": Pv.notna().sum(0).values.astype(int),
        "disc_allcase": Md.mean(0), "disc_active_only": np.nanmean(np.where(Pd.notna().values, Pd.values, np.nan), axis=0),
        "val_rescue": Mv.mean(0), "val_rescue_lo": lo, "val_rescue_hi": hi, "val_pos_frac": (Mv > 0).mean(0),
        "val_active_only": np.nanmean(np.where(Pv.notna().values, Pv.values, np.nan), axis=0),
        "val_spec": Sm, "val_spec_lo": slo, "val_spec_hi": shi, "val_control": Mv.mean(0) - Sm,
        "block_val": block_mean, "block_val_lo": float(np.percentile(block_b, 2.5)), "block_val_hi": float(np.percentile(block_b, 97.5)),
        "block_share": share, "block_share_lo": sh_lo, "block_share_hi": sh_hi,
        "mean_rank": ranks.mean(0).values, "mean_percentile": np.nanmean(percentile, axis=0), "top1_frac": (ranks == 1).sum(0).values / np.maximum(Pv.notna().sum(0).values, 1),
        "n_disc": len(disc), "n_val": n_val, "threshold": threshold,
    })
    df["recurrent"] = df.disc_active >= threshold
    df["pair"] = [pair(layer, e) for e in experts]
    return df


def full_ranking(md: A.ModelData, cache: X.ExpertCache, set_name: str, n_controls: int, threshold: Optional[int] = None,
                 layers: Optional[list[int]] = None) -> pd.DataFrame:
    disc, val = md.ids(set_name, "discovery"), md.ids(set_name, "validation")
    th = threshold or len(disc) // 2
    frames = [rank_layer(cache, l, disc, val, n_controls, th) for l in (layers or cache.layers)]
    df = pd.concat(frames, ignore_index=True)
    df["block_clearly_positive"] = df.block_val_lo > 0
    df["block_share_raw"] = df.block_share
    df.loc[~df.block_clearly_positive, ["block_share", "block_share_lo", "block_share_hi"]] = np.nan  # undefined where the block does nothing
    df["rescue_clearly_positive"] = df.val_rescue_lo > 0
    for m in METRICS:
        # rank among recurrent pairs (1 = best); NaN for non-recurrent
        r = df.loc[df.recurrent, m].rank(ascending=False, method="min")
        df[f"rank_{m}"] = r
    return df


def kendall_matrix(df: pd.DataFrame, metrics: list[str] = METRICS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kendall tau-b and p between metric orderings over the rows of df."""
    from scipy.stats import kendalltau
    T = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
    Pm = T.copy()
    for a in metrics:
        for b in metrics:
            x, y = df[a].values, df[b].values
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 3:
                T.loc[a, b] = np.nan; Pm.loc[a, b] = np.nan
                continue
            t, p = kendalltau(x[ok], y[ok])
            T.loc[a, b] = t; Pm.loc[a, b] = p
    return T, Pm


def disagreements(df: pd.DataFrame, metrics: list[str] = METRICS, top: int = 10) -> pd.DataFrame:
    """Union of the top-`top` recurrent pairs under each metric, with the rank under every metric and the max rank spread."""
    rec = df[df.recurrent].copy()
    keep = set()
    for m in metrics:
        keep |= set(rec.sort_values(m, ascending=False).head(top).index)
    sub = rec.loc[sorted(keep)].copy()
    rk = sub[[f"rank_{m}" for m in metrics]]
    sub["rank_spread"] = rk.max(1) - rk.min(1)
    sub["best_metric"] = rk.idxmin(1).str.replace("rank_", "")
    sub["worst_metric"] = rk.idxmax(1).str.replace("rank_", "")
    return sub.sort_values("rank_val_rescue")


# ---------------------------------------------------------------------------------------------------------------
# F1.2 population-level minimal sets under the additive approximation
# ---------------------------------------------------------------------------------------------------------------
def additivity_check(md: A.ModelData, cache: X.ExpertCache, layer: int, ids: list[int]) -> dict:
    """Sum of single-expert rescues vs exact clean-top-k coalition vs block patch on ids (same pass)."""
    e = md.expert_rows
    P = cache.piv[layer].reindex(ids)
    s = P.fillna(0.0).sum(1).values
    co = e[(e.layer == layer) & (e.kind == "coalition_clean")].set_index("case_id").rescue.reindex(ids).values
    un = e[(e.layer == layer) & (e.kind == "coalition_union")].set_index("case_id").rescue.reindex(ids).values
    b = cache.layer_rescue.loc[ids, layer].values
    d = s - co
    return dict(layer=int(layer), n=len(ids), sum_singles=float(s.mean()), coalition_clean=float(co.mean()), coalition_union=float(un.mean()),
                block=float(b.mean()), r_sum_vs_coalition=float(np.corrcoef(s, co)[0, 1]), r_sum_vs_block=float(np.corrcoef(s, b)[0, 1]),
                mean_abs_diff=float(np.abs(d).mean()), median_abs_diff=float(np.median(np.abs(d))), frac_within_025=float((np.abs(d) <= 0.25).mean()),
                frac_within_05=float((np.abs(d) <= 0.5).mean()), diff_summary=summarize(d, with_p=True), n_active_per_case=float(P.notna().sum(1).mean()))


def additive_curve(cache: X.ExpertCache, layer: int, disc: list[int], val: list[int]) -> pd.DataFrame:
    """Greedy forward selection of a fixed set S maximising mean_c sum_{e in S ∩ active(c)} rescue_e on discovery.
    The objective is linear in S, so the greedy order is the descending order of all-case mean discovery rescue and the
    curve is its cumulative sum; evaluated on validation as the fraction of the validation block rescue. Also returns the
    in-sample (validation-ordered) curve as an optimistic bound and the coverage-greedy variant (see coverage_curve)."""
    Pd, Md, Bd = layer_matrices(cache, layer, disc)
    Pv, Mv, Bv = layer_matrices(cache, layer, val)
    experts = [int(e) for e in Pv.columns]
    dm, vm = Md.mean(0), Mv.mean(0)
    order = np.argsort(-dm, kind="stable")
    order_v = np.argsort(-vm, kind="stable")
    bv, bd = float(Bv.values.mean()), float(Bd.values.mean())
    rows = []
    cum_d = cum_v = cum_in = 0.0
    for k, (j, jv) in enumerate(zip(order, order_v), start=1):
        cum_d += dm[j]; cum_v += vm[j]; cum_in += vm[jv]
        S = [experts[i] for i in order[:k]]
        covered = Pv.notna().values[:, order[:k]].sum(1) == Pv.notna().values.sum(1)  # cases whose whole clean set is inside S
        rows.append(dict(layer=layer, k=k, expert=experts[j], pair=pair(layer, experts[j]), marginal_disc=float(dm[j]), marginal_val=float(vm[j]),
                         cum_disc=cum_d, cum_val=cum_v, frac_disc=cum_d / bd if bd else np.nan, frac_val=cum_v / bv if bv else np.nan,
                         cum_val_insample=cum_in, frac_val_insample=cum_in / bv if bv else np.nan, n_cases_fully_covered=int(covered.sum())))
    return pd.DataFrame(rows)


def coverage_curve(cache: X.ExpertCache, layer: int, disc: list[int], val: list[int], target: float = 0.8, min_block: float = 0.0) -> pd.DataFrame:
    """Non-linear variant: greedy set maximising the number of discovery cases whose additive sum over S ∩ active(c)
    reaches `target` x that case's block rescue (cases with block > min_block); ties broken by the additive sum.
    Evaluated on validation as the fraction of eligible validation cases covered."""
    Pd, Md, Bd = layer_matrices(cache, layer, disc)
    Pv, Mv, Bv = layer_matrices(cache, layer, val)
    experts = [int(e) for e in Pv.columns]
    eligible_d, eligible_v = Bd.values > min_block, Bv.values > min_block
    goal_d, goal_v = target * Bd.values, target * Bv.values
    chosen: list[int] = []
    cur_d, cur_v = np.zeros(len(disc)), np.zeros(len(val))
    rows = []
    remaining = list(range(len(experts)))
    while remaining:
        best, best_key = None, None
        for j in remaining:
            nd = cur_d + Md[:, j]
            key = (int(((nd >= goal_d) & eligible_d).sum()), float(nd.mean()))
            if best_key is None or key > best_key:
                best, best_key = j, key
        chosen.append(best); remaining.remove(best)
        cur_d += Md[:, best]; cur_v += Mv[:, best]
        rows.append(dict(layer=layer, k=len(chosen), expert=experts[best], pair=pair(layer, experts[best]),
                         cov_disc=float(((cur_d >= goal_d) & eligible_d).sum() / max(eligible_d.sum(), 1)),
                         cov_val=float(((cur_v >= goal_v) & eligible_v).sum() / max(eligible_v.sum(), 1)),
                         n_eligible_disc=int(eligible_d.sum()), n_eligible_val=int(eligible_v.sum())))
    return pd.DataFrame(rows)


def size_for_targets(curve: pd.DataFrame, col: str, targets=TARGETS) -> dict:
    out = {}
    for t in targets:
        hit = curve[curve[col] >= t]
        out[t] = int(hit.k.iloc[0]) if len(hit) else None
    return out


def cross_layer_curve(cache: X.ExpertCache, disc: list[int], val: list[int], ref_layer: int, max_k: int = 60) -> pd.DataFrame:
    """Greedy over all (layer, expert) pairs, still additive across layers (unvalidated: exact multi-layer patches are
    F1.4). Fraction relative to the best single-layer block rescue (ref_layer) on validation."""
    frames = []
    for l in cache.layers:
        st = cache.disc_stats(l, disc)
        Pv = cache.piv[l].reindex(val)
        vm = Pv.fillna(0.0).mean(0)
        st["val_allcase"] = [float(vm.get(e, 0.0)) for e in st.expert]
        st["val_active"] = [int(Pv[e].notna().sum()) if e in Pv.columns else 0 for e in st.expert]
        frames.append(st)
    df = pd.concat(frames, ignore_index=True).sort_values(["disc_allcase_mean", "layer", "expert"], ascending=[False, True, True]).head(max_k)
    bv = float(cache.layer_rescue.loc[val, ref_layer].mean())
    df["k"] = np.arange(1, len(df) + 1)
    df["cum_val"] = df.val_allcase.cumsum()
    df["frac_val_of_ref_block"] = df.cum_val / bv
    # per-case maximum over the pairs in S (the 'union' reading of ext1: what a joint patch gives if the pairs restore
    # the same information; the sum is the other extreme, full independence)
    running = np.full(len(val), -np.inf)
    cum_max = []
    for l, e in zip(df.layer, df.expert):
        Pv = cache.piv[l].reindex(val)
        r = Pv[e].fillna(0.0).values if e in Pv.columns else np.zeros(len(val))
        running = np.maximum(running, r)
        cum_max.append(float(running.mean()))
    df["cum_max_val"] = cum_max
    df["frac_max_of_ref_block"] = df.cum_max_val / bv
    df["pair"] = [pair(l, e) for l, e in zip(df.layer, df.expert)]
    df["ref_block_val"] = bv
    return df.reset_index(drop=True)


def two_locus_comparison(md: A.ModelData, cache: X.ExpertCache, a: tuple[int, int], b: tuple[int, int], val: list[int], n_controls: int) -> dict:
    """Additive sum of two pairs in different layers vs each alone vs the per-case max (union), with paired CIs."""
    pa = cache.per_case(a[0], a[1], val, n_controls).set_index("case_id").loc[val].rescue.values
    pb = cache.per_case(b[0], b[1], val, n_controls).set_index("case_id").loc[val].rescue.values
    ref = cache.layer_rescue.loc[val, a[0]].values
    return dict(a=pair(*a), b=pair(*b), n=len(val), a_alone=summarize(pa, with_p=False), b_alone=summarize(pb, with_p=False),
                additive_sum=summarize(pa + pb, with_p=False), per_case_max=summarize(np.maximum(pa, pb), with_p=False),
                block_a_layer=summarize(ref, with_p=False), sum_over_block=ratio_ci(pa + pb, ref), max_over_block=ratio_ci(np.maximum(pa, pb), ref),
                both_positive=float(((pa > 0) & (pb > 0)).mean()), either_positive=float(((pa > 0) | (pb > 0)).mean()))


# ---------------------------------------------------------------------------------------------------------------
# F1.3 per-case minimal sets from exact subset patches
# ---------------------------------------------------------------------------------------------------------------
def load_subsets(path: str) -> pd.DataFrame:
    """Load subset_rows.parquet. Delivered schema (ext5-engine): case_id, layer, kind (coalition_set | layer | coalition_clean |
    block reference rows of the same pass), experts (sorted comma-joined; '' for reference kinds), n_experts (0 for reference
    kinds), rescue, ... . Returns the coalition_set rows with a frozenset column `eset`; the reference rows are kept in
    df.attrs['reference'] (kind, case_id, layer, rescue)."""
    df = pd.read_parquet(path)
    need = {"case_id", "layer", "experts", "n_experts", "rescue"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"subset_rows missing columns {sorted(missing)}")
    df = df.copy()
    df["experts"] = df.experts.fillna("").astype(str)
    ref = None
    if "kind" in df.columns:
        ref = df[df.kind != "coalition_set"][["kind", "case_id", "layer", "rescue"]].copy()
        df = df[df.kind == "coalition_set"].copy()
    df = df[df.n_experts > 0]
    df["eset"] = df.experts.map(lambda s: frozenset(int(x) for x in s.split(",") if x != ""))
    df.attrs["reference"] = ref
    return df


def block_reference(sub_dir: str, fallback_md: Optional[A.ModelData], layer: int, ids: list[int], sub: Optional[pd.DataFrame] = None) -> pd.Series:
    """Per-case MoE-block rescue at `layer` (kind `layer`): from the subset pass's own reference rows (sub.attrs['reference'])
    if present, else from the subset run's expert_rows.parquet, else from the fallback all-layer pass (bf16 noise only)."""
    if sub is not None and sub.attrs.get("reference") is not None:
        ref = sub.attrs["reference"]
        lay = ref[(ref.kind == "layer") & (ref.layer == layer)]
        if len(lay):
            return lay.set_index("case_id").rescue.reindex(ids)
    p = os.path.join(sub_dir, "expert_rows.parquet")
    if os.path.exists(p):
        e = pd.read_parquet(p)
        lay = e[(e.layer == layer) & (e.kind == "layer")]
        if len(lay):
            return lay.set_index("case_id").rescue.reindex(ids)
    if fallback_md is None:
        raise FileNotFoundError("no block reference available")
    return fallback_md.expert_rows.query("layer == @layer and kind == 'layer'").set_index("case_id").rescue.reindex(ids)


def per_case_minimal_sets(sub: pd.DataFrame, block: pd.Series, layer: int, targets=TARGETS, min_block: float = 0.0) -> pd.DataFrame:
    """For each case with block > min_block: the smallest subset whose exact rescue reaches t x block (ties -> highest
    rescue); the additive prediction (singles sorted descending, cumulative until t x block); agreement of the two."""
    s = sub[sub.layer == layer]
    rows = []
    for cid, g in s.groupby("case_id"):
        if cid not in block.index or not (block[cid] > min_block):
            continue
        b = float(block[cid])
        singles = {next(iter(r.eset)): float(r.rescue) for r in g.itertuples() if r.n_experts == 1}
        k = max(g.n_experts)
        full = g[g.n_experts == k]
        full_r = float(full.rescue.iloc[0]) if len(full) else np.nan
        row = dict(case_id=int(cid), layer=layer, block=b, k=int(k), full_rescue=full_r, full_over_block=full_r / b,
                   sum_singles=float(sum(singles.values())), best_single=max(singles, key=singles.get), best_single_rescue=max(singles.values()))
        order = sorted(singles, key=lambda e: -singles[e])
        for t in targets:
            goal = t * b
            ok = g[g.rescue >= goal].sort_values(["n_experts", "rescue"], ascending=[True, False])
            tag = f"{int(round(100 * t))}"
            if len(ok):
                best = ok.iloc[0]
                row[f"min_size_{tag}"] = int(best.n_experts)
                row[f"min_set_{tag}"] = best.experts
                row[f"min_rescue_{tag}"] = float(best.rescue)
            else:
                row[f"min_size_{tag}"] = None  # even the full clean set does not reach the target
                row[f"min_set_{tag}"] = ""
                row[f"min_rescue_{tag}"] = np.nan
            # additive prediction
            cum, add_set = 0.0, []
            for e in order:
                add_set.append(e); cum += singles[e]
                if cum >= goal:
                    break
            reached = cum >= goal
            row[f"add_size_{tag}"] = len(add_set) if reached else None
            row[f"add_set_{tag}"] = ",".join(str(e) for e in sorted(add_set)) if reached else ""
            exact_add = g[g.eset == frozenset(add_set)]
            row[f"add_set_exact_rescue_{tag}"] = float(exact_add.rescue.iloc[0]) if len(exact_add) else np.nan
            row[f"add_set_reaches_{tag}"] = bool(len(exact_add) and exact_add.rescue.iloc[0] >= goal)
        rows.append(row)
    return pd.DataFrame(rows)


def singleton_sufficiency(mins: pd.DataFrame, sub: pd.DataFrame, layer: int, expert: int, targets=TARGETS) -> dict:
    """How often {expert} alone reaches t x block among cases where it is clean-active (and among all eligible cases)."""
    s = sub[(sub.layer == layer) & (sub.n_experts == 1)]
    single = {int(c): float(r) for c, r, es in zip(s.case_id, s.rescue, s.eset) if expert in es}
    out = dict(layer=layer, expert=expert, pair=pair(layer, expert), n_eligible=int(len(mins)), n_active=int(mins.case_id.isin(single).sum()))
    for t in targets:
        tag = f"{int(round(100 * t))}"
        hit = [c for c in mins.case_id if c in single and single[c] >= t * float(mins.set_index("case_id").block[c])]
        out[f"suffices_{tag}"] = len(hit)
        out[f"is_min_set_{tag}"] = int((mins[f"min_set_{tag}"] == str(expert)).sum())
    return out


def interaction_matrix(sub: pd.DataFrame, layer: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pairwise interaction at S = {}: I(a,b) = rescue({a,b}) - rescue({a}) - rescue({b}) per case; returns the mean matrix
    (experts x experts, NaN where the pair never co-occurs), the co-occurrence count matrix and the long per-case table."""
    s = sub[sub.layer == layer]
    rows = []
    for cid, g in s.groupby("case_id"):
        r = {es: float(v) for es, v in zip(g.eset, g.rescue)}
        singles = sorted(e for es in r if len(es) == 1 for e in es)
        for a, b in itertools.combinations(singles, 2):
            ab = frozenset((a, b))
            if ab in r:
                rows.append((int(cid), a, b, r[ab] - r[frozenset((a,))] - r[frozenset((b,))], r[frozenset((a,))], r[frozenset((b,))], r[ab]))
    long = pd.DataFrame(rows, columns=["case_id", "a", "b", "interaction", "r_a", "r_b", "r_ab"])
    if long.empty:
        return pd.DataFrame(), pd.DataFrame(), long
    experts = sorted(set(long.a) | set(long.b))
    Mi = pd.DataFrame(np.nan, index=experts, columns=experts)
    Cn = pd.DataFrame(0, index=experts, columns=experts, dtype=int)
    for (a, b), g in long.groupby(["a", "b"]):
        Mi.loc[a, b] = Mi.loc[b, a] = g.interaction.mean()
        Cn.loc[a, b] = Cn.loc[b, a] = len(g)
    return Mi, Cn, long


def nonadditivity_decomposition(sub: pd.DataFrame, layer: int) -> pd.DataFrame:
    """Per case: rescue(full) - sum(singles) (total non-additivity) vs the sum of all pairwise interactions (second order)."""
    s = sub[sub.layer == layer]
    rows = []
    for cid, g in s.groupby("case_id"):
        r = {es: float(v) for es, v in zip(g.eset, g.rescue)}
        singles = {next(iter(es)): v for es, v in r.items() if len(es) == 1}
        k = max(len(es) for es in r)
        full = [v for es, v in r.items() if len(es) == k]
        if not full:
            continue
        pairs = sum(r[frozenset((a, b))] - singles[a] - singles[b] for a, b in itertools.combinations(sorted(singles), 2) if frozenset((a, b)) in r)
        rows.append(dict(case_id=int(cid), k=k, full=full[0], sum_singles=sum(singles.values()), total_nonadd=full[0] - sum(singles.values()), pairwise_sum=pairs))
    df = pd.DataFrame(rows)
    if len(df):
        df["higher_order"] = df.total_nonadd - df.pairwise_sum
    return df
