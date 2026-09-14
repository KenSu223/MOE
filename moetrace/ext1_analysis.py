"""Direction 1 (ext1-joint-search): joint layer x expert search versus the paper's two-stage selection.

Consumes the all-layer expert passes (results/<run>/expert_rows.parquet with kinds layer, coalition_*, expert,
expert_noised_only for EVERY layer) through the public API of analysis.py (select_expert / evaluate_expert semantics are
reused; nothing in analysis.py is modified). Every function is a pure post-processing step over the row-level Parquet.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd

from . import analysis as A
from .protocol import active_controls
from .stats import ratio_ci, summarize

RUNS = {
    "qwen3_bos_alllayers": dict(model="qwen3", short="qwen3_bos", label="Qwen3-30B-A3B-Base (tokenizer defaults)",
                                sets=("paper", "strict", "relaxed"), two_stage=(44, 69), neighbours=(42, 43, 45), base_run="qwen3"),
    "mixtral_bos_alllayers": dict(model="mixtral", short="mixtral_bos", label="Mixtral-8x7B-v0.1 (BOS, tokenizer default)",
                                  sets=("paper", "strict", "relaxed"), two_stage=(19, 2), neighbours=(20, 21), base_run="mixtral"),
    "mixtral_nobos_alllayers": dict(model="mixtral", short="mixtral_nobos", label="Mixtral-8x7B-v0.1 (no BOS, paper protocol)",
                                    sets=("paper",), two_stage=(19, 6), neighbours=(20, 21), base_run="mixtral_nobos"),
}
SEEDS = (0, 1, 2, 3, 4)
THRESHOLDS = (32, 48, 64, 80, 96)


def pair(layer: int, expert: int) -> str:
    return f"L{int(layer)}E{int(expert):03d}"


# ---------------------------------------------------------------------------------------------------------------
class ExpertCache:
    """Per-layer case x expert rescue pivots over clean-active rows, plus the same-pass layer-patch rescue."""

    def __init__(self, md: A.ModelData):
        e = md.expert_rows
        et = e[(e.kind == "expert") & e.clean_active.astype(bool)]
        self.layers = sorted(int(l) for l in et.layer.unique())
        self.piv = {int(l): g.pivot(index="case_id", columns="expert", values="rescue") for l, g in et.groupby("layer")}
        self.clean_sets = {int(l): A.clean_sets(g) for l, g in et.groupby("layer")}
        lay = e[e.kind == "layer"]
        self.layer_rescue = lay.pivot(index="case_id", columns="layer", values="rescue")

    def disc_stats(self, layer: int, disc_ids: list[int]) -> pd.DataFrame:
        """Activity count and all-case mean rescue (zero where not clean-active) of every expert on the discovery ids."""
        P = self.piv[layer].reindex(disc_ids)
        act = P.notna().sum(0)
        allcase = P.fillna(0.0).sum(0) / len(disc_ids)
        return pd.DataFrame({"layer": layer, "expert": act.index.astype(int), "disc_active": act.values.astype(int),
                             "disc_allcase_mean": allcase.values.astype(float)})

    def per_case(self, layer: int, e: int, ids: list[int], n_controls: int) -> pd.DataFrame:
        """Same quantities as analysis.evaluate_expert's per-case table (rescue 0 where not active; active-random controls)."""
        P = self.piv[layer]
        cs = self.clean_sets[layer]
        rows = []
        for cid in ids:
            act = cs.get(cid, [])
            active = e in act
            r = float(P.at[cid, e]) if active else 0.0
            ctrls = active_controls(act, e, n_controls, cid)
            cr = [float(P.at[cid, c]) for c in ctrls]
            cm = float(np.mean(cr)) if cr else np.nan
            rows.append((cid, active, r, cm, (r - cm) if cr else np.nan))
        return pd.DataFrame(rows, columns=["case_id", "active", "rescue", "control_mean", "spec"])


def select_at_layer(cache: ExpertCache, layer: int, disc_ids: list[int], threshold: int) -> tuple[Optional[int], pd.DataFrame]:
    """Recurrence-first selection at one layer (identical rule to analysis.select_expert)."""
    st = cache.disc_stats(layer, disc_ids)
    cands = st[st.disc_active >= threshold]
    if cands.empty:
        return None, st
    best = cands.sort_values(["disc_allcase_mean", "expert"], ascending=[False, True]).iloc[0]
    return int(best.expert), st


def joint_candidates(cache: ExpertCache, disc_ids: list[int], threshold: int) -> pd.DataFrame:
    """All recurrent (layer, expert) pairs ranked by discovery all-case mean rescue."""
    frames = [cache.disc_stats(l, disc_ids) for l in cache.layers]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.disc_active >= threshold].sort_values(["disc_allcase_mean", "layer", "expert"], ascending=[False, True, True]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------------------------------------------
def per_layer_curve(md: A.ModelData, cache: ExpertCache, set_name: str, n_controls: int, threshold: Optional[int] = None) -> pd.DataFrame:
    """Per layer: MoE-block rescue (sweep pass and same-pass), recurrence-first best expert, its validation rescue / Spec
    with bootstrap CIs, activity counts and the concentration ratio Rescue(e*)/Rescue(layer) on validation."""
    disc, val = md.ids(set_name, "discovery"), md.ids(set_name, "validation")
    th = threshold or len(disc) // 2
    R = md.R
    rows = []
    for l in cache.layers:
        lv = summarize(R.loc[val, l].values, with_p=False)
        same = cache.layer_rescue.loc[val, l].values
        ls = summarize(same, with_p=False)
        e, st = select_at_layer(cache, l, disc, th)
        row = dict(layer=l, layer_disc_mean=float(R.loc[disc, l].mean()), layer_val_mean=lv["mean"], layer_val_lo=lv["ci_lo"], layer_val_hi=lv["ci_hi"],
                   layer_samepass_val_mean=ls["mean"], layer_samepass_val_lo=ls["ci_lo"], layer_samepass_val_hi=ls["ci_hi"],
                   n_candidates=int((st.disc_active >= th).sum()), max_disc_activity=int(st.disc_active.max()) if len(st) else 0, threshold=th)
        if e is None:
            row.update(e_star=None, disc_active=np.nan, disc_allcase_mean=np.nan, val_active=np.nan, val_rescue=np.nan, val_rescue_lo=np.nan,
                       val_rescue_hi=np.nan, val_spec=np.nan, val_spec_lo=np.nan, val_spec_hi=np.nan, val_control=np.nan, val_rescue_p=np.nan,
                       concentration=np.nan, concentration_lo=np.nan, concentration_hi=np.nan)
        else:
            ev = A.evaluate_expert(md, l, e, val, n_controls)
            pc = ev["per_case"].set_index("case_id").loc[val]
            with np.errstate(divide="ignore", invalid="ignore"):
                c, clo, chi = ratio_ci(pc.rescue.values, same)
            srow = st[st.expert == e].iloc[0]
            row.update(e_star=e, disc_active=int(srow.disc_active), disc_allcase_mean=float(srow.disc_allcase_mean), val_active=ev["val_active"],
                       val_rescue=ev["rescue_all"]["mean"], val_rescue_lo=ev["rescue_all"]["ci_lo"], val_rescue_hi=ev["rescue_all"]["ci_hi"],
                       val_rescue_p=ev["rescue_all"]["p"], val_spec=ev["spec_all"]["mean"], val_spec_lo=ev["spec_all"]["ci_lo"],
                       val_spec_hi=ev["spec_all"]["ci_hi"], val_control=ev["control_all"]["mean"], concentration=c, concentration_lo=clo, concentration_hi=chi)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_pair(md: A.ModelData, cache: ExpertCache, layer: int, expert: int, disc: list[int], val: list[int], n_controls: int) -> dict:
    """Discovery statistics (regardless of recurrence) and validation summaries of one (layer, expert) pair."""
    st = cache.disc_stats(layer, disc)
    s = st[st.expert == expert]
    ev = A.evaluate_expert(md, layer, expert, val, n_controls)
    return dict(layer=int(layer), expert=int(expert), pair=pair(layer, expert),
                disc_active=int(s.disc_active.iloc[0]) if len(s) else 0, disc_allcase_mean=float(s.disc_allcase_mean.iloc[0]) if len(s) else 0.0,
                val_active=int(ev["val_active"]), val_rescue=ev["rescue_all"]["mean"], val_rescue_lo=ev["rescue_all"]["ci_lo"],
                val_rescue_hi=ev["rescue_all"]["ci_hi"], val_rescue_p=ev["rescue_all"]["p"], val_spec=ev["spec_all"]["mean"],
                val_spec_lo=ev["spec_all"]["ci_lo"], val_spec_hi=ev["spec_all"]["ci_hi"], val_spec_p=ev["spec_all"]["p"],
                val_control=ev["control_all"]["mean"], val_rescue_active=ev["rescue_active"]["mean"])


def joint_search(md: A.ModelData, cache: ExpertCache, set_name: str, n_controls: int, two_stage: tuple[int, int],
                 threshold: Optional[int] = None, top_k: int = 10) -> dict:
    """Argmax over all recurrent (layer, expert) pairs of the discovery all-case mean rescue; validation of the top-k;
    comparison with the fixed two-stage winner; discovery-to-validation shrinkage of the maximum."""
    disc, val = md.ids(set_name, "discovery"), md.ids(set_name, "validation")
    th = threshold or len(disc) // 2
    cands = joint_candidates(cache, disc, th)
    rows = []
    for _, c in cands.head(top_k).iterrows():
        r = evaluate_pair(md, cache, int(c.layer), int(c.expert), disc, val, n_controls)
        r["rank"] = int(c["rank"])
        r["is_two_stage"] = (int(c.layer), int(c.expert)) == tuple(two_stage)
        rows.append(r)
    top = pd.DataFrame(rows)
    ts = evaluate_pair(md, cache, two_stage[0], two_stage[1], disc, val, n_controls)
    all_rows = rows + [dict(evaluate_pair(md, cache, int(c.layer), int(c.expert), disc, val, n_controls), rank=int(c["rank"]))
                       for _, c in cands.iloc[top_k:].iterrows()]
    allc = pd.DataFrame(all_rows)
    vs_ref = dict(n_pairs=int(len(allc)), n_beat_ref_rescue=int((allc.val_rescue > ts["val_rescue"]).sum()),
                  n_rescue_ci_above_ref_ci=int((allc.val_rescue_lo > ts["val_rescue_hi"]).sum()), n_beat_ref_spec=int((allc.val_spec > ts["val_spec"]).sum()),
                  n_spec_ci_above_ref_ci=int((allc.val_spec_lo > ts["val_spec_hi"]).sum()), n_spec_ci_above_zero=int((allc.val_spec_lo > 0).sum()),
                  n_rescue_ci_above_zero=int((allc.val_rescue_lo > 0).sum()),
                  pairs_spec_ci_above_zero=allc[allc.val_spec_lo > 0].pair.tolist(), pairs_beat_ref_rescue=allc[allc.val_rescue > ts["val_rescue"]].pair.tolist())
    m = cands[(cands.layer == two_stage[0]) & (cands.expert == two_stage[1])]
    ts["rank"] = int(m["rank"].iloc[0]) if len(m) else None
    ts["recurrent"] = bool(len(m))
    w = top.iloc[0].to_dict() if len(top) else None
    shrink = None
    if w is not None:
        shrink = dict(disc_max=w["disc_allcase_mean"], val_of_winner=w["val_rescue"], abs=w["disc_allcase_mean"] - w["val_rescue"],
                      rel=(w["disc_allcase_mean"] - w["val_rescue"]) / w["disc_allcase_mean"] if w["disc_allcase_mean"] else np.nan,
                      two_stage_disc=ts["disc_allcase_mean"], two_stage_val=ts["val_rescue"], two_stage_abs=ts["disc_allcase_mean"] - ts["val_rescue"])
    # second-best pair outside the two-stage layer (the strongest challenger from another layer)
    other = cands[cands.layer != two_stage[0]]
    challenger = None
    if len(other):
        o = other.iloc[0]
        challenger = evaluate_pair(md, cache, int(o.layer), int(o.expert), disc, val, n_controls)
        challenger["rank"] = int(o["rank"])
    return dict(set=set_name, n_disc=len(disc), n_val=len(val), threshold=th, n_candidates=int(len(cands)),
                n_layers_with_candidates=int(cands.layer.nunique()), top=top, winner=w, two_stage=ts, challenger=challenger,
                same_as_two_stage=bool(w and (w["layer"], w["expert"]) == tuple(two_stage)),
                winner_in_two_stage_layer=bool(w and w["layer"] == two_stage[0]), shrinkage=shrink, candidates=cands, all_evaluated=allc, vs_reference=vs_ref)


def joint_stability(md: A.ModelData, cache: ExpertCache, set_name: str, n_controls: int, two_stage: tuple[int, int],
                    seeds=SEEDS, thresholds=THRESHOLDS) -> dict:
    """Appendix D grid for the joint search: split seeds x recurrence thresholds over the set's discovery+validation ids
    (thresholds are scaled by n/256 so that 64 stays 'half the discovery split' for 512-case sets). Reports how often the
    joint winner equals the fixed two-stage winner, lies outside its layer, and equals the per-split two-stage selection."""
    all_ids = md.ids(set_name, "discovery") + md.ids(set_name, "validation")
    scale = len(all_ids) / 256.0
    R = md.R
    rows = []
    for seed in seeds:
        ids = list(all_ids)
        random.Random(seed).shuffle(ids)
        disc, val = ids[: len(ids) // 2], ids[len(ids) // 2:]
        l_split = int(R.loc[disc].mean(0).idxmax())
        for th in thresholds:
            th_eff = int(round(th * scale))
            cands = joint_candidates(cache, disc, th_eff)
            e_ts, _ = select_at_layer(cache, l_split, disc, th_eff)
            if cands.empty:
                rows.append(dict(seed=seed, threshold=th, threshold_eff=th_eff, joint_layer=None, joint_expert=None, n_candidates=0,
                                 split_two_stage_layer=l_split, split_two_stage_expert=e_ts))
                continue
            w = cands.iloc[0]
            jl, je = int(w.layer), int(w.expert)
            pc = cache.per_case(jl, je, val, n_controls)
            rows.append(dict(seed=seed, threshold=th, threshold_eff=th_eff, joint_layer=jl, joint_expert=je, joint_pair=pair(jl, je),
                             joint_disc_mean=float(w.disc_allcase_mean), joint_disc_active=int(w.disc_active),
                             joint_val_rescue=float(pc.rescue.mean()), joint_val_spec=float(pc.spec.dropna().mean()) if pc.spec.notna().any() else np.nan,
                             n_candidates=int(len(cands)), split_two_stage_layer=l_split, split_two_stage_expert=e_ts,
                             equals_reference=(jl, je) == tuple(two_stage), same_layer_as_reference=jl == two_stage[0],
                             equals_split_two_stage=(e_ts is not None and (jl, je) == (l_split, e_ts))))
    df = pd.DataFrame(rows)
    ok = df[df.joint_layer.notna()]
    ts_none = ok.split_two_stage_expert.isna() if len(ok) else pd.Series(dtype=bool)
    outside = ~ok.same_layer_as_reference.astype(bool) if len(ok) else pd.Series(dtype=bool)
    return dict(set=set_name, grid=df, n_settings=int(len(df)), n_with_winner=int(len(ok)),
                n_split_two_stage_none=int(ts_none.sum()), n_outside_with_two_stage_candidate=int((outside & ~ts_none).sum()),
                n_outside_without_two_stage_candidate=int((outside & ts_none).sum()),
                n_equals_reference=int(ok.equals_reference.sum()) if len(ok) else 0,
                n_outside_reference_layer=int((~ok.same_layer_as_reference.astype(bool)).sum()) if len(ok) else 0,
                n_equals_split_two_stage=int(ok.equals_split_two_stage.sum()) if len(ok) else 0,
                winner_counts=ok.joint_pair.value_counts().to_dict() if len(ok) else {},
                split_layer_counts=df.split_two_stage_layer.value_counts().to_dict(),
                mean_val_rescue=float(ok.joint_val_rescue.mean()) if len(ok) else np.nan,
                mean_val_spec=float(ok.joint_val_spec.mean()) if len(ok) else np.nan)


def neighbour_check(md: A.ModelData, cache: ExpertCache, set_name: str, n_controls: int, layers: tuple[int, ...],
                    two_stage: tuple[int, int], threshold: Optional[int] = None) -> dict:
    """Every expert with any clean-active validation case in the given layers, evaluated on validation exactly as the
    selected expert is (rescue 0 where inactive; active-random Spec), against the two-stage winner's validation numbers."""
    disc, val = md.ids(set_name, "discovery"), md.ids(set_name, "validation")
    th = threshold or len(disc) // 2
    ref = cache.per_case(two_stage[0], two_stage[1], val, n_controls)
    ref_r, ref_s = float(ref.rescue.mean()), float(ref.spec.dropna().mean())
    rows = []
    for l in layers:
        st = cache.disc_stats(l, disc).set_index("expert")
        P = cache.piv[l]
        for e in P.columns:
            e = int(e)
            pc = cache.per_case(l, e, val, n_controls)
            va = int(pc.active.sum())
            if va == 0:
                continue
            r, s = float(pc.rescue.mean()), float(pc.spec.dropna().mean()) if pc.spec.notna().any() else np.nan
            rows.append(dict(layer=l, expert=e, pair=pair(l, e), disc_active=int(st.disc_active.get(e, 0)), disc_allcase_mean=float(st.disc_allcase_mean.get(e, 0.0)),
                             recurrent=bool(st.disc_active.get(e, 0) >= th), val_active=va, val_rescue=r, val_spec=s,
                             beats_rescue=r > ref_r, beats_spec=(s > ref_s) if not np.isnan(s) else False))
    df = pd.DataFrame(rows).sort_values("val_rescue", ascending=False).reset_index(drop=True)
    out = dict(set=set_name, layers=list(layers), threshold=th, reference=dict(pair=pair(*two_stage), val_rescue=ref_r, val_spec=ref_s),
               n_experts=int(len(df)), n_beat_rescue=int(df.beats_rescue.sum()), n_beat_spec=int(df.beats_spec.sum()),
               n_recurrent=int(df.recurrent.sum()), n_recurrent_beat_rescue=int((df.recurrent & df.beats_rescue).sum()),
               n_recurrent_beat_spec=int((df.recurrent & df.beats_spec).sum()), table=df)
    for key, col in (("best_by_rescue", "val_rescue"), ("best_by_spec", "val_spec")):
        if len(df) and df[col].notna().any():
            b = df.loc[df[col].idxmax()]
            out[key] = evaluate_pair(md, cache, int(b.layer), int(b.expert), disc, val, n_controls)
            out[key]["recurrent"] = bool(b.recurrent)
        else:
            out[key] = None
    rec = df[df.recurrent]
    out["best_recurrent_by_rescue"] = None
    if len(rec):
        b = rec.loc[rec.val_rescue.idxmax()]
        out["best_recurrent_by_rescue"] = evaluate_pair(md, cache, int(b.layer), int(b.expert), disc, val, n_controls)
    return out


def consistency_with_base(md_all: A.ModelData, base_run: str, set_name: str, two_stage: tuple[int, int], n_controls: int) -> dict:
    """bf16 fingerprint: the two-stage winner's validation numbers in the all-layer pass vs the original single-layer pass."""
    md_base = A.load_model(base_run)
    val = md_all.ids(set_name, "validation")
    a = A.evaluate_expert(md_all, two_stage[0], two_stage[1], val, n_controls)
    b = A.evaluate_expert(md_base, two_stage[0], two_stage[1], val, n_controls)
    return dict(pair=pair(*two_stage), all_layer_pass=dict(val_active=a["val_active"], val_rescue=a["rescue_all"]["mean"], val_spec=a["spec_all"]["mean"]),
                base_pass=dict(val_active=b["val_active"], val_rescue=b["rescue_all"]["mean"], val_spec=b["spec_all"]["mean"]),
                diff_rescue=a["rescue_all"]["mean"] - b["rescue_all"]["mean"], diff_spec=a["spec_all"]["mean"] - b["spec_all"]["mean"])


def pair_overlap(md: A.ModelData, cache: ExpertCache, a: tuple[int, int], b: tuple[int, int], ids: list[int], n_controls: int) -> dict:
    """Do two (layer, expert) pairs rescue the same cases? Per-case validation rescues of both, their correlation, the
    fraction of cases where both / either / neither is positive, and the mean of the per-case maximum (union effect)."""
    from scipy.stats import pearsonr, spearmanr
    pa = cache.per_case(a[0], a[1], ids, n_controls).set_index("case_id").loc[ids]
    pb = cache.per_case(b[0], b[1], ids, n_controls).set_index("case_id").loc[ids]
    ra, rb = pa.rescue.values, pb.rescue.values
    both_active = (pa.active & pb.active).values
    d_rescue = summarize(rb - ra)  # paired per-case difference b - a (all-case rescue, zero where inactive)
    sa, sb = pa.spec.values, pb.spec.values
    ok = ~(np.isnan(sa) | np.isnan(sb))
    d_spec = summarize(sb[ok] - sa[ok])
    return dict(a=pair(*a), b=pair(*b), n=len(ids), pearson_r=float(pearsonr(ra, rb)[0]), spearman_r=float(spearmanr(ra, rb)[0]),
                diff_rescue_b_minus_a=d_rescue, diff_spec_b_minus_a=d_spec, n_spec_pairs=int(ok.sum()),
                both_positive=float(((ra > 0) & (rb > 0)).mean()), either_positive=float(((ra > 0) | (rb > 0)).mean()),
                a_only_positive=float(((ra > 0) & ~(rb > 0)).mean()), b_only_positive=float((~(ra > 0) & (rb > 0)).mean()),
                both_active=int(both_active.sum()), mean_a=float(ra.mean()), mean_b=float(rb.mean()), mean_max=float(np.maximum(ra, rb).mean()),
                mean_sum=float((ra + rb).mean()), corr_active_only=float(pearsonr(ra[both_active], rb[both_active])[0]) if both_active.sum() > 2 else np.nan)


def concentration_table(curve: pd.DataFrame, min_layer_rescue: float = 0.1) -> pd.DataFrame:
    """Layers whose same-pass MoE-block validation rescue is clearly positive (CI above 0 and mean >= min), sorted by
    Rescue(best expert)/Rescue(layer)."""
    c = curve[(curve.layer_samepass_val_lo > 0) & (curve.layer_samepass_val_mean >= min_layer_rescue) & curve.e_star.notna()].copy()
    return c.sort_values("concentration", ascending=False).reset_index(drop=True)
