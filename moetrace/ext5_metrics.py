"""Phase 2 / F5 (ext5-analysis): probability-scale metrics alongside the paper's logit difference.

Consumes the *_metrics runs produced by the ext5-engine agent (results/<run>_metrics/{sweep_rows,expert_rows}.parquet with the
agreed extra columns logp_true, logp_foil, p_true, p_foil, rank_true, kl_to_clean on EVERY row) and re-derives the paper's
selections under each metric by swapping the `rescue` column and re-using analysis.py / ext1_analysis.py unchanged.

Identity stated first: Δ = logit(true) - logit(foil) = log p(true) - log p(foil) (the softmax normaliser cancels), so the
paper's "logit difference" is already the log-odds of the two-way contrast. The alternative metrics below add what Δ lacks:
the absolute probability of the true token, its rank in the full vocabulary and the distance of the whole distribution
from the clean one.

Metric definitions (rescue-type quantities: positive = the patch moves the noised run towards the clean run):
- delta : Δ_patched - Δ_noised (paper)
- dp    : p_true(patched) - p_true(noised)
- logp  : log p_true(patched) - log p_true(noised)  (one-sided version of Δ; drops the foil)
- rank  : log2 rank_true(noised) - log2 rank_true(patched)  (positive = the true token climbs)
- kl    : KL(noised || clean) - KL(patched || clean)  (positive = the patched distribution is closer to the clean one)
- ratio : Δ rescue / Δ drop per case, evaluated on the cases with drop >= 1 (the population-level version, mean rescue /
          mean drop with a paired bootstrap, is a rescaling that cannot change any selection and is reported separately)
"""
from __future__ import annotations

import copy
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from . import analysis as A
from . import ext1_analysis as X
from .models import RESULTS
from .stats import ratio_ci, summarize

NEW_COLS = ["logp_true", "logp_foil", "p_true", "p_foil", "rank_true", "kl_to_clean"]
RUNS = {
    "qwen3_metrics": dict(base="qwen3", model="qwen3", short="qwen3", label="Qwen3-30B-A3B-Base (tokenizer defaults)", two_stage=(44, 69), second=(42, 115), n_controls=3),
    "mixtral_nobos_metrics": dict(base="mixtral_nobos", model="mixtral", short="mixtral_nobos", label="Mixtral-8x7B-v0.1 (no BOS, paper protocol)", two_stage=(19, 6), second=(18, 1), n_controls=1),
    "mixtral_bos_metrics": dict(base="mixtral", model="mixtral", short="mixtral_bos", label="Mixtral-8x7B-v0.1 (BOS, tokenizer default)", two_stage=(19, 2), second=(18, 1), n_controls=1),
}
METRICS = {
    "delta": dict(col="rescue", label="Δ rescue (paper)", short="Δ"),
    "dp": dict(col="rescue_dp", label="Δp = p_true(patched) − p_true(noised)", short="Δp"),
    "logp": dict(col="rescue_logp", label="log p_true(patched) − log p_true(noised)", short="Δlog p"),
    "rank": dict(col="rescue_rank", label="log2 rank(noised) − log2 rank(patched)", short="rank"),
    "kl": dict(col="rescue_kl", label="KL(noised‖clean) − KL(patched‖clean)", short="KL"),
    "ratio": dict(col="rescue_ratio", label="Δ rescue / Δ drop, cases with drop ≥ 1", short="Δ/drop"),
}
PATCH_KINDS = ("layer", "expert", "coalition_clean", "coalition_union", "expert_scaled", "expert_noised_only", "attn_layer", "block", "resid", "zero")


# ---------------------------------------------------------------------------------------------------------------
def load_run(path_or_name: str) -> A.ModelData:
    """analysis.load_model for a run name under results/ or for an arbitrary directory (synthetic tests)."""
    d = path_or_name if os.path.isdir(path_or_name) else os.path.join(RESULTS, path_or_name)
    rd = lambda n: pd.read_parquet(os.path.join(d, n)) if os.path.exists(os.path.join(d, n)) else None
    with open(os.path.join(d, "case_sets.json")) as f:
        sets = json.load(f)
    return A.ModelData(os.path.basename(d.rstrip("/")), sets, rd("sweep_rows.parquet"), rd("sweep_routing.parquet"), rd("sweep_cases.parquet"),
                       rd("expert_rows.parquet"), rd("noise_rows.parquet"))


def check_schema(md: A.ModelData) -> dict:
    out = {}
    for name, df in (("sweep_rows", md.sweep_rows), ("expert_rows", md.expert_rows)):
        if df is None:
            out[name] = dict(present=False)
            continue
        miss = [c for c in NEW_COLS if c not in df.columns]
        out[name] = dict(present=True, n_rows=int(len(df)), missing=miss, complete=not miss,
                         n_nan={c: int(df[c].isna().sum()) for c in NEW_COLS if c in df.columns})
    return out


def identity_check(md: A.ModelData) -> dict:
    """Δ = logp_true - logp_foil (and = logit_true - logit_foil) to bf16 / fp32 rounding on every row."""
    out = {}
    for name, df in (("sweep_rows", md.sweep_rows), ("expert_rows", md.expert_rows)):
        if df is None or "logp_true" not in df.columns:
            continue
        d1 = (df.delta - (df.logp_true - df.logp_foil)).abs()
        d2 = (df.delta - (df.logit_true - df.logit_foil)).abs()
        out[name] = dict(n=int(len(df)), max_abs_delta_minus_logodds=float(d1.max()), mean_abs_delta_minus_logodds=float(d1.mean()),
                         max_abs_delta_minus_logitdiff=float(d2.max()), p_sum_max=float((df.p_true + df.p_foil).max()),
                         p_range=(float(df.p_true.min()), float(df.p_true.max())), rank_min=int(df.rank_true.min()),
                         kl_min=float(df.kl_to_clean.min()))
    return out


def reference_table(md: A.ModelData) -> pd.DataFrame:
    """Per case: clean and noised values of every quantity (from sweep_rows kinds clean / noised) plus derived drops."""
    sr = md.sweep_rows
    cl = sr[sr.kind == "clean"].set_index("case_id")
    no = sr[sr.kind == "noised"].set_index("case_id")
    ids = cl.index.intersection(no.index)
    ref = pd.DataFrame(index=ids)
    for c in ("delta", "logp_true", "logp_foil", "p_true", "p_foil", "rank_true", "top1"):
        if c in cl.columns:
            ref[f"{c}_clean"] = cl.loc[ids, c]
            ref[f"{c}_noised"] = no.loc[ids, c]
    ref["kl_noised"] = no.loc[ids, "kl_to_clean"] if "kl_to_clean" in no.columns else np.nan
    ref["drop"] = ref.delta_clean - ref.delta_noised
    ref["sweep_drop"] = ref["drop"]
    ref["p_drop"] = ref.p_true_clean - ref.p_true_noised
    ref["logp_drop"] = ref.logp_true_clean - ref.logp_true_noised
    ref["rank_drop"] = np.log2(ref.rank_true_noised) - np.log2(ref.rank_true_clean)
    ref["top1_clean"] = ref.rank_true_clean == 1
    ref["top1_flipped"] = (ref.rank_true_clean == 1) & (ref.rank_true_noised > 1)
    ref["p_clean_saturated"] = ref.p_true_clean >= 0.9
    ref.index.name = "case_id"
    return ref


def add_metric_columns(rows: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Add rescue_dp / rescue_logp / rescue_rank / rescue_kl / rescue_ratio to patched rows. The noised reference of a row is
    taken from the SAME pass when available (expert_rows carry p_true_noised / logp_true_noised; rank and KL of the noised
    run come from the pass's expert_prefill file passed in as ref columns rank_true_noised / kl_noised), else from the
    sweep pass (ref built from sweep_rows kinds clean / noised). Cross-pass bf16 noise moves Δ_noised by up to ~1 logit,
    so mixing passes would add noise to every per-case rescue."""
    df = rows.copy()
    p_no = df["p_true_noised"] if "p_true_noised" in df.columns else df.case_id.map(ref.p_true_noised)
    lp_no = df["logp_true_noised"] if "logp_true_noised" in df.columns else df.case_id.map(ref.logp_true_noised)
    rk_no = df.case_id.map(ref.rank_true_noised).astype(float)
    kl_no = df.case_id.map(ref.kl_noised)
    df["rescue_dp"] = df.p_true - p_no
    df["rescue_logp"] = df.logp_true - lp_no
    df["rescue_rank"] = np.log2(rk_no) - np.log2(df.rank_true.astype(float))
    df["rescue_kl"] = kl_no - df.kl_to_clean
    # per-case ratio: divide by the SWEEP-pass drop (a fixed per-case constant, so every row of a case uses the same divisor
    # and the case set 'drop >= 1' is the same for sweep and expert rows); ref carries the sweep drop as sweep_drop
    drop = df.case_id.map(ref["sweep_drop"] if "sweep_drop" in ref.columns else ref["drop"])
    df["rescue_ratio"] = np.where(drop >= 1.0, df.rescue / drop, np.nan)
    df["top1_recovered"] = (df.rank_true == 1) & (rk_no > 1)
    return df


def expert_reference(run_dir: str, ref: pd.DataFrame) -> pd.DataFrame:
    """Same-pass reference for expert rows from expert_prefill_L*.parquet (<metric>_clean / _noised per case); falls back
    to the sweep-pass reference for any column that is missing."""
    import glob
    files = sorted(glob.glob(os.path.join(run_dir, "expert_prefill_*.parquet")))
    if not files:
        return ref
    pf = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("case_id").set_index("case_id")
    r2 = ref.copy()
    for src, dst in (("p_true_noised", "p_true_noised"), ("logp_true_noised", "logp_true_noised"), ("rank_true_noised", "rank_true_noised"),
                     ("kl_to_clean_noised", "kl_noised"), ("p_true_clean", "p_true_clean"), ("logp_true_clean", "logp_true_clean"), ("rank_true_clean", "rank_true_clean"),
                     ("delta_clean", "delta_clean"), ("delta_noised", "delta_noised")):
        if src in pf.columns:
            r2.loc[r2.index.intersection(pf.index), dst] = pf.loc[r2.index.intersection(pf.index), src]
    r2["drop"] = r2.delta_clean - r2.delta_noised
    r2["p_drop"] = r2.p_true_clean - r2.p_true_noised
    r2["top1_flipped"] = (r2.rank_true_clean == 1) & (r2.rank_true_noised > 1)
    return r2


def enrich(md: A.ModelData, run_dir: Optional[str] = None) -> tuple[A.ModelData, pd.DataFrame]:
    """Sweep rows get the sweep-pass reference; expert rows get the same-pass reference (expert_prefill files)."""
    ref = reference_table(md)
    md2 = copy.copy(md)
    md2.sweep_rows = add_metric_columns(md.sweep_rows, ref)
    if md.expert_rows is not None:
        ref_e = expert_reference(run_dir, ref) if run_dir else ref
        md2.expert_rows = add_metric_columns(md.expert_rows, ref_e)
    return md2, ref


def with_metric(md: A.ModelData, metric: str, ref: Optional[pd.DataFrame] = None) -> A.ModelData:
    """A ModelData whose `rescue` column IS the metric, so analysis.py / ext1_analysis.py functions compute the paper's
    quantities under that metric. For 'ratio' the case sets are restricted to cases with drop >= 1."""
    col = METRICS[metric]["col"]
    md2 = copy.copy(md)
    sr = md.sweep_rows.copy()
    sr["rescue_delta"] = sr.rescue
    sr["rescue"] = sr[col] if col in sr.columns else sr.rescue
    md2.sweep_rows = sr
    if md.expert_rows is not None:
        er = md.expert_rows.copy()
        er["rescue_delta"] = er.rescue
        er["rescue"] = er[col] if col in er.columns else er.rescue
        md2.expert_rows = er
    if metric == "ratio":
        assert ref is not None
        keep = set(ref.index[ref["drop"] >= 1.0])
        sets = {}
        for s, v in md.sets.items():
            if isinstance(v, dict) and "discovery" in v:
                sets[s] = dict(v, discovery=[c for c in v["discovery"] if c in keep], validation=[c for c in v["validation"] if c in keep])
            else:
                sets[s] = v
        md2.sets = sets
    return md2


# ---------------------------------------------------------------------------------------------------------------
def layer_curves(mds: dict[str, A.ModelData], set_name: str) -> tuple[pd.DataFrame, dict]:
    """Per metric: the layer curve (discovery mean, validation mean + CI) and the discovery argmax."""
    frames, arg = [], {}
    for m, md in mds.items():
        la = A.layer_analysis(md, set_name)
        cur = pd.DataFrame(la["curve"]); cur["metric"] = m
        peak = cur.loc[cur.val_mean.idxmax()]
        arg[m] = dict(L_star=la["L_star"], disc_mean=la["disc_mean_at_Lstar"], val=la["val_at_Lstar"], n_disc=la["n_disc"], n_val=la["n_val"],
                      val_argmax_layer=int(peak.layer), val_argmax=float(peak.val_mean), disc_top5=la["disc_curve_top5"],
                      sharpness=la["sharpness"])
        frames.append(cur)
    return pd.concat(frames, ignore_index=True), arg


def normalised_rescue(md: A.ModelData, ref: pd.DataFrame, layer: int, ids: list[int], col: str = "rescue") -> dict:
    """Population-level normalised rescue (mean rescue / mean drop, paired bootstrap) and the per-case ratio on drop >= 1."""
    R = md.sweep_rows[md.sweep_rows.kind == "layer"].pivot(index="case_id", columns="layer", values=col)
    r = R.loc[ids, layer].values
    d = ref.loc[ids, "drop"].values
    m, lo, hi = ratio_ci(r, d)
    big = d >= 1.0
    pc = summarize(r[big] / d[big]) if big.any() else summarize([])
    pd_ = ref.loc[ids, "p_drop"].values
    out = dict(layer=int(layer), n=len(ids), mean_rescue=float(r.mean()), mean_drop=float(d.mean()), ratio=m, ratio_lo=lo, ratio_hi=hi,
               n_drop_ge_1=int(big.sum()), per_case_ratio=pc)
    if "rescue_dp" in md.sweep_rows.columns:
        Rp = md.sweep_rows[md.sweep_rows.kind == "layer"].pivot(index="case_id", columns="layer", values="rescue_dp")
        rp = Rp.loc[ids, layer].values
        mp, lop, hip = ratio_ci(rp, pd_)
        out.update(mean_rescue_dp=float(rp.mean()), mean_p_drop=float(pd_.mean()), ratio_dp=mp, ratio_dp_lo=lop, ratio_dp_hi=hip)
    return out


def expert_selection(md: A.ModelData, layer: int, disc: list[int], val: list[int], n_controls: int, threshold: Optional[int] = None) -> dict:
    """Paper's recurrence-first selection at `layer` under md's rescue column, with validation rescue / Spec CIs."""
    et = A.expert_table(md, layer)
    if et.empty:
        return dict(layer=layer, available=False)
    th = threshold or len(disc) // 2
    sel = A.select_expert(et, disc, th)
    out = dict(layer=layer, available=True, threshold=th, selection=sel)
    if sel["e_star"] is not None:
        ev = A.evaluate_expert(md, layer, sel["e_star"], val, n_controls)
        out.update(e_star=sel["e_star"], val_active=ev["val_active"], rescue=ev["rescue_all"], spec=ev["spec_all"], control=ev["control_all"],
                   rescue_active=ev["rescue_active"], spec_sign=("positive" if ev["spec_all"]["ci_lo"] > 0 else "negative" if ev["spec_all"]["ci_hi"] < 0 else "indeterminate"),
                   per_case=ev["per_case"])
    return out


def evaluate_fixed(md: A.ModelData, layer: int, expert: int, val: list[int], n_controls: int) -> dict:
    et = A.expert_table(md, layer)
    if et.empty or expert not in set(et.expert.astype(int)):
        return dict(layer=layer, expert=expert, available=False)
    ev = A.evaluate_expert(md, layer, expert, val, n_controls)
    return dict(layer=layer, expert=expert, available=True, val_active=ev["val_active"], rescue=ev["rescue_all"], spec=ev["spec_all"], control=ev["control_all"],
                spec_sign=("positive" if ev["spec_all"]["ci_lo"] > 0 else "negative" if ev["spec_all"]["ci_hi"] < 0 else "indeterminate"), per_case=ev["per_case"])


def joint_top(md: A.ModelData, disc: list[int], val: list[int], n_controls: int, two_stage: tuple[int, int], top_k: int = 10) -> dict:
    """ext1 joint search over the layers present in expert_rows (the metrics runs carry 2-3 layers, so this is the joint
    search restricted to those layers; stated in the section)."""
    cache = X.ExpertCache(md)
    js = X.joint_search(md, cache, "paper", n_controls, two_stage, top_k=top_k)
    js["layers_present"] = cache.layers
    return js


def funnel_alternative(md: A.ModelData, ref: pd.DataFrame, p_thr: float = 0.5) -> dict:
    """p_clean(true) >= p_thr as an alternative funnel vs the paper's Δ funnels, over every case in the run and per set."""
    ct = md.cases.set_index("case_id") if md.cases is not None else None
    ids = ref.index
    strict = (ref.delta_clean >= 1.0) & (ref["drop"] >= 0.5)
    relaxed = (ref.delta_clean >= 0.5) & (ref["drop"] >= 0.25)
    palt = ref.p_true_clean >= p_thr
    palt_drop = palt & (ref.p_drop >= 0.25)
    def ov(a, b):
        return dict(n_a=int(a.sum()), n_b=int(b.sum()), both=int((a & b).sum()), a_only=int((a & ~b).sum()), b_only=int((~a & b).sum()), neither=int((~a & ~b).sum()),
                    jaccard=float((a & b).sum() / max((a | b).sum(), 1)))
    out = dict(n=int(len(ids)), p_thr=p_thr, strict_vs_p=ov(strict, palt), relaxed_vs_p=ov(relaxed, palt), strict_vs_p_drop=ov(strict, palt_drop),
               p_clean=dict(median=float(ref.p_true_clean.median()), mean=float(ref.p_true_clean.mean()), frac_ge_05=float(palt.mean()), frac_ge_09=float((ref.p_true_clean >= 0.9).mean()),
                            frac_top1=float(ref.top1_clean.mean())),
               p_noised=dict(median=float(ref.p_true_noised.median()), frac_top1=float((ref.rank_true_noised == 1).mean()), frac_flipped=float(ref.top1_flipped.mean())))
    per_set = {}
    for s in ("paper", "strict", "relaxed"):
        if s in md.sets and isinstance(md.sets[s], dict):
            sid = [c for c in md.sets[s]["discovery"] + md.sets[s]["validation"] if c in ids]
            r = ref.loc[sid]
            per_set[s] = dict(n=len(sid), pass_p=int((r.p_true_clean >= p_thr).sum()), pass_strict=int(((r.delta_clean >= 1.0) & (r["drop"] >= 0.5)).sum()),
                              top1_clean=int(r.top1_clean.sum()), top1_flipped=int(r.top1_flipped.sum()), p_clean_median=float(r.p_true_clean.median()),
                              p_clean_ge_09=int((r.p_true_clean >= 0.9).sum()), fail_ids=[int(c) for c in r.index[r.p_true_clean < p_thr]][:40])
    out["per_set"] = per_set
    return out


def per_case_frame(md: A.ModelData, ref: pd.DataFrame, layer: int, ids: list[int], expert: Optional[int] = None) -> pd.DataFrame:
    """Per case: Δ rescue and the alternative rescues of the block patch at `layer` (and of `expert` if given) with the
    case's clean/noised state, for the explanatory scatter plots."""
    sr = md.sweep_rows
    lay = sr[(sr.kind == "layer") & (sr.layer == layer)].set_index("case_id").reindex(ids)
    df = pd.DataFrame(index=ids)
    for m, spec in METRICS.items():
        if spec["col"] in lay.columns:
            df[f"block_{m}"] = lay[spec["col"]]
    df["block_top1_recovered"] = lay.top1_recovered if "top1_recovered" in lay.columns else np.nan
    for c in ("p_true_clean", "p_true_noised", "delta_clean", "delta_noised", "drop", "p_drop", "rank_true_clean", "rank_true_noised", "top1_flipped", "p_clean_saturated", "kl_noised"):
        df[c] = ref.loc[ids, c]
    if expert is not None and md.expert_rows is not None:
        er = md.expert_rows
        ex = er[(er.layer == layer) & (er.kind == "expert") & (er.expert == expert) & er.clean_active.astype(bool)].set_index("case_id").reindex(ids)
        for m, spec in METRICS.items():
            if spec["col"] in ex.columns:
                df[f"expert_{m}"] = ex[spec["col"]].fillna(0.0)
        df["expert_active"] = ex.rescue.notna()
    df.index.name = "case_id"
    return df.reset_index()


# ---------------------------------------------------------------------------------------------------------------
FUNCTION_WORDS = {"the", "a", "an", "of", "in", "to", "and", "which", "that", "for", "with", "his", "her", "its", "their", "on", "at", "by",
                  "from", "as", "is", "was", "called", "former", "not", "about", ",", ".", ":", ";", "(", ")", "\"", "'", "-", "__", ""}


def clean_top1_table(md: A.ModelData, ref: pd.DataFrame, model_key: str) -> dict:
    """What the clean run's top-1 token is when it is not the true object: decoded with the model tokenizer."""
    from transformers import AutoTokenizer
    from .arch import snapshot_dir
    from .models import MODELS
    tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[model_key]["repo"]))
    sr = md.sweep_rows
    cl = sr[sr.kind == "clean"].set_index("case_id")
    ids = [c for c in ref.index if c in cl.index]
    cl = cl.loc[ids]
    top1 = pd.Series([tok.decode([int(t)]) for t in cl.top1], index=ids)
    not_true = cl.rank_true > 1
    nt = top1[not_true]
    fw = nt.str.strip().str.lower().isin(FUNCTION_WORDS)
    counts = nt.value_counts()
    return dict(n=int(len(ids)), n_true_top1=int((~not_true).sum()), n_not_top1=int(not_true.sum()), n_function_word=int(fw.sum()),
                frac_function_word_of_not_top1=float(fw.mean()) if len(nt) else np.nan,
                top_tokens=[(repr(k), int(v)) for k, v in counts.head(10).items()],
                median_p_true_when_top1=float(cl.p_true[~not_true].median()) if (~not_true).any() else np.nan,
                median_p_true_when_not=float(cl.p_true[not_true].median()) if not_true.any() else np.nan,
                median_rank_when_not=float(cl.rank_true[not_true].median()) if not_true.any() else np.nan)


def tail_concentration(md: A.ModelData, ref: pd.DataFrame, layer: int, ids: list[int], col: str, by: str = "kl_noised", q: float = 0.9) -> dict:
    """How much of the summed rescue (column `col`) of the block patch at `layer` comes from the cases in the top (1-q)
    quantile of `by` (default: KL(noised||clean), i.e. the cases whose noised distribution is most degenerate)."""
    sr = md.sweep_rows
    lay = sr[(sr.kind == "layer") & (sr.layer == layer)].set_index("case_id").reindex(ids)
    v = lay[col].values.astype(float)
    b = ref.loc[ids, by].values.astype(float)
    ok = ~np.isnan(v)  # the per-case ratio is undefined where drop < 1
    v, b = v[ok], b[ok]
    top = b >= np.quantile(b, q)
    tot = v.sum()
    return dict(layer=int(layer), col=col, by=by, q=q, n=int(ok.sum()), n_top=int(top.sum()), mean=float(v.mean()), mean_top=float(v[top].mean()),
                mean_rest=float(v[~top].mean()), share_top=float(v[top].sum() / tot) if tot else np.nan,
                corr=float(np.corrcoef(v, b)[0, 1]) if ok.sum() > 2 else np.nan, by_threshold=float(np.quantile(b, q)))
