"""Post-processing of the row-level parquet files into the paper's quantities (Tables 1-16, Figure 1 inputs)."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .models import MODELS, RESULTS
from .protocol import active_controls, load_case_sets, out_dir
from .stats import bootstrap_ci, ratio_ci, signflip_p, summarize


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class ModelData:
    model: str
    sets: dict
    sweep_rows: pd.DataFrame
    routing: pd.DataFrame
    cases: pd.DataFrame
    expert_rows: Optional[pd.DataFrame]
    noise_rows: Optional[pd.DataFrame]

    @property
    def R(self) -> pd.DataFrame:  # case x layer rescue matrix
        lr = self.sweep_rows[self.sweep_rows.kind == "layer"]
        return lr.pivot(index="case_id", columns="layer", values="rescue")

    def ids(self, set_name: str, split: str) -> list[int]:
        return [c for c in self.sets[set_name][split] if c in set(self.cases.case_id)]

    def relation_of(self) -> dict[int, str]:
        return dict(zip(self.cases.case_id, self.cases.relation))


def load_model(model: str) -> ModelData:
    od = out_dir(model)
    rd = lambda n: pd.read_parquet(os.path.join(od, n)) if os.path.exists(os.path.join(od, n)) else None
    return ModelData(model, load_case_sets(model), rd("sweep_rows.parquet"), rd("sweep_routing.parquet"),
                     rd("sweep_cases.parquet"), rd("expert_rows.parquet"), rd("noise_rows.parquet"))


# ---------------------------------------------------------------------------------------------------------------
# layer level
# ---------------------------------------------------------------------------------------------------------------
def layer_analysis(md: ModelData, set_name: str) -> dict:
    R = md.R
    disc, val = md.ids(set_name, "discovery"), md.ids(set_name, "validation")
    md_ = R.loc[disc].mean(axis=0)
    lstar = int(md_.idxmax())
    val_mean = R.loc[val].mean(axis=0)
    curve = []
    for l in R.columns:
        s = summarize(R.loc[val, l].values, with_p=False)
        curve.append({"layer": int(l), "disc_mean": float(md_[l]), "val_mean": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                      "pos_frac": s["pos_frac"]})
    order = val_mean.sort_values(ascending=False)
    top_l, next_l = int(order.index[0]), int(order.index[1])
    sel = summarize(R.loc[val, lstar].values)
    return {"set": set_name, "n_disc": len(disc), "n_val": len(val), "L_star": lstar, "disc_mean_at_Lstar": float(md_[lstar]),
            "val_at_Lstar": sel, "curve": curve,
            "sharpness": {"top_layer": top_l, "top_rescue": float(order.iloc[0]), "next_layer": next_l, "next_rescue": float(order.iloc[1]),
                          "gap": float(order.iloc[0] - order.iloc[1])},
            "disc_curve_top5": [(int(l), round(float(v), 3)) for l, v in md_.sort_values(ascending=False).head(5).items()]}


# ---------------------------------------------------------------------------------------------------------------
# expert level
# ---------------------------------------------------------------------------------------------------------------
def expert_table(md: ModelData, layer: int) -> pd.DataFrame:
    """Rows of kind 'expert' with clean_active at the given layer."""
    e = md.expert_rows
    return e[(e.layer == layer) & (e.kind == "expert") & (e.clean_active)].copy()


def clean_sets(et: pd.DataFrame) -> dict[int, list[int]]:
    return {cid: sorted(g.expert.astype(int).tolist()) for cid, g in et.groupby("case_id")}


def rescue_lookup(et: pd.DataFrame) -> dict[tuple[int, int], float]:
    return {(int(c), int(e)): float(r) for c, e, r in zip(et.case_id, et.expert, et.rescue)}


def select_expert(et: pd.DataFrame, disc_ids: list[int], threshold: int) -> dict:
    sub = et[et.case_id.isin(disc_ids)]
    act = sub.groupby("expert").size()
    n = len(disc_ids)
    allcase = sub.groupby("expert").rescue.sum() / n
    active_only = sub.groupby("expert").rescue.mean()
    cands = act[act >= threshold].index
    if len(cands) == 0:
        return {"e_star": None, "n_candidates": 0, "threshold": threshold, "max_activity": int(act.max()) if len(act) else 0}
    ranking = allcase.loc[cands].sort_values(ascending=False)
    e_star = int(ranking.index[0])
    return {"e_star": e_star, "n_candidates": int(len(cands)), "threshold": threshold, "n_disc": n,
            "disc_active": int(act[e_star]), "disc_allcase_mean": float(allcase[e_star]), "disc_active_only_mean": float(active_only[e_star]),
            "top_candidates": [(int(e), round(float(v), 3), int(act[e])) for e, v in ranking.head(8).items()],
            "active_only_best": (int(active_only.loc[cands].idxmax()), round(float(active_only.loc[cands].max()), 3))}


def evaluate_expert(md: ModelData, layer: int, e_star: int, val_ids: list[int], n_controls: int) -> dict:
    """Validation summaries for a fixed expert: all-case and anchor-active rescue, specificity, zero rows, per-case table."""
    et = expert_table(md, layer)
    cs = clean_sets(et)
    rl = rescue_lookup(et)
    rows = []
    for cid in val_ids:
        active_set = cs.get(cid, [])
        active = e_star in active_set
        r = rl[(cid, e_star)] if active else 0.0
        ctrls = active_controls(active_set, e_star, n_controls, cid)
        cr = [rl[(cid, c)] for c in ctrls]
        rows.append({"case_id": cid, "active": active, "rescue": r, "controls": ctrls, "control_rescues": cr,
                     "control_mean": float(np.mean(cr)) if cr else np.nan, "spec": r - float(np.mean(cr)) if cr else np.nan,
                     "n_active": len(active_set)})
    df = pd.DataFrame(rows)
    zero_rows = int((df.rescue == 0).sum() + sum(int(v == 0) for cr in df.control_rescues for v in cr))
    anc = df[df.active]
    out = {"e_star": e_star, "layer": layer, "n_val": len(df), "val_active": int(df.active.sum()),
           "rescue_all": summarize(df.rescue.values), "spec_all": summarize(df.spec.dropna().values),
           "control_all": summarize(df.control_mean.dropna().values),
           "rescue_active": summarize(anc.rescue.values), "spec_active": summarize(anc.spec.dropna().values),
           "control_active": summarize(anc.control_mean.dropna().values),
           "zero_rows_selected_plus_controls": zero_rows, "zero_rows_selected_only": int((df.rescue == 0).sum()),
           "not_active_rows": int((~df.active).sum()),
           "per_case": df}
    return out


def relation_breakdown(md: ModelData, ev: dict, min_n: int = 5) -> pd.DataFrame:
    rel = md.relation_of()
    df = ev["per_case"].copy()
    df["relation"] = df.case_id.map(rel)
    g = df.groupby("relation")
    tab = pd.DataFrame({"n": g.size(), "expert_rescue": g.rescue.mean(), "specificity": g.spec.mean(),
                        "pos_frac": g.rescue.apply(lambda x: float((x > 0).mean()))})
    tab = tab[tab.n >= min_n].sort_values("n", ascending=False)
    return tab.reset_index()


def ratios(md: ModelData, layer: int, ev: dict, val_ids: list[int]) -> dict:
    R = md.R
    lr = R.loc[val_ids, layer].values
    pc = ev["per_case"].set_index("case_id").loc[val_ids]
    er = pc.rescue.values
    sp = pc.spec.values
    r1 = ratio_ci(er, lr)
    r2 = ratio_ci(sp, lr)
    return {"layer_rescue": float(lr.mean()), "expert_over_layer": r1, "spec_over_layer": r2}


def gate_matched_control(md: ModelData, layer: int, e_star: int, val_ids: list[int]) -> dict:
    """Qwen3 Table 9: gate-weight-matched control and equal-norm variant on validation cases where e* is clean-active."""
    e = md.expert_rows
    e = e[e.layer == layer]
    et = e[(e.kind == "expert") & e.clean_active]
    sc = e[e.kind == "expert_scaled"]
    rl = rescue_lookup(et)
    w = {(int(c), int(x)): float(v) for c, x, v in zip(et.case_id, et.expert, et.clean_weight)}
    cs = clean_sets(et)
    scl = {(int(c), int(a), int(b)): float(r) for c, a, b, r in zip(sc.case_id, sc.expert, sc.partner, sc.rescue)}
    rows = []
    for cid in val_ids:
        act = cs.get(cid, [])
        if e_star not in act or len(act) < 2:
            continue
        others = [x for x in act if x != e_star]
        r_star = min(others, key=lambda x: abs(w[(cid, x)] - w[(cid, e_star)]))
        rows.append({"case_id": cid, "matched": r_star, "sel_raw": rl[(cid, e_star)], "ctrl_raw": rl[(cid, r_star)],
                     "sel_eq": scl[(cid, e_star, r_star)], "ctrl_eq": scl[(cid, r_star, e_star)],
                     "w_sel": w[(cid, e_star)], "w_ctrl": w[(cid, r_star)]})
    df = pd.DataFrame(rows)
    return {"n": len(df), "selected_raw": summarize(df.sel_raw.values), "control_raw": summarize(df.ctrl_raw.values),
            "spec_raw": summarize((df.sel_raw - df.ctrl_raw).values), "selected_eq": summarize(df.sel_eq.values),
            "control_eq": summarize(df.ctrl_eq.values), "spec_eq": summarize((df.sel_eq - df.ctrl_eq).values), "per_case": df}


def all_active_rank(md: ModelData, layer: int, e_star: int, val_ids: list[int]) -> dict:
    """Table 10: rank of rescue(e*) among all clean-active experts of the case (rank 1 = best)."""
    et = expert_table(md, layer)
    cs = clean_sets(et)
    rl = rescue_lookup(et)
    rows = []
    for cid in val_ids:
        act = cs.get(cid, [])
        if e_star not in act or len(act) < 2:
            continue
        rs = {x: rl[(cid, x)] for x in act}
        r_star = rs[e_star]
        others = [rs[x] for x in act if x != e_star]
        rank = 1 + sum(1 for v in others if v > r_star)
        pct = (len(act) - rank) / (len(act) - 1)
        rows.append({"case_id": cid, "rank": rank, "n_active": len(act), "percentile": pct, "sel_minus_others": r_star - float(np.mean(others))})
    df = pd.DataFrame(rows)
    return {"n": len(df), "top1": int((df["rank"] == 1).sum()), "top2": int((df["rank"] <= 2).sum()), "mean_rank": float(df["rank"].mean()),
            "mean_percentile": float(df.percentile.mean()), "sel_minus_all_other": summarize(df.sel_minus_others.values), "per_case": df}


def active_pair_equal_norm(md: ModelData, layer: int, e_star: int, val_ids: list[int]) -> dict:
    """Mixtral Table 11: anchor-active validation cases; the other clean-active expert is the control."""
    e = md.expert_rows
    e = e[e.layer == layer]
    et = e[(e.kind == "expert") & e.clean_active]
    sc = e[e.kind == "expert_scaled"]
    rl = rescue_lookup(et)
    cs = clean_sets(et)
    scl = {(int(c), int(a), int(b)): float(r) for c, a, b, r in zip(sc.case_id, sc.expert, sc.partner, sc.rescue)}
    rows = []
    for cid in val_ids:
        act = cs.get(cid, [])
        if e_star not in act or len(act) != 2:
            continue
        other = [x for x in act if x != e_star][0]
        rows.append({"case_id": cid, "other": other, "sel_raw": rl[(cid, e_star)], "other_raw": rl[(cid, other)],
                     "sel_eq": scl[(cid, e_star, other)], "other_eq": scl[(cid, other, e_star)]})
    df = pd.DataFrame(rows)
    return {"n": len(df), "spec_raw": summarize((df.sel_raw - df.other_raw).values), "selected_eq": summarize(df.sel_eq.values),
            "other_eq": summarize(df.other_eq.values), "spec_eq": summarize((df.sel_eq - df.other_eq).values), "per_case": df}


def coalitions(md: ModelData, layer: int, val_ids: list[int]) -> dict:
    e = md.expert_rows
    e = e[(e.layer == layer) & e.case_id.isin(val_ids)]
    out = {}
    for kind in ("coalition_clean", "coalition_union", "layer"):
        sub = e[e.kind == kind].set_index("case_id").loc[val_ids]
        out[kind] = summarize(sub.rescue.values)
    out["layer_from_sweep"] = summarize(md.R.loc[val_ids, layer].values)
    return out


def stability_grid(md: ModelData, layer: int, set_name: str, n_controls: int, seeds=(0, 1, 2, 3, 4),
                   thresholds=(32, 48, 64, 80, 96), reference_expert: Optional[int] = None) -> dict:
    """Appendix D: re-run selection for split seeds x recurrence thresholds over the set's 256 cases."""
    et = expert_table(md, layer)
    all_ids = md.ids(set_name, "discovery") + md.ids(set_name, "validation")
    rows = []
    for seed in seeds:
        ids = list(all_ids)
        random.Random(seed).shuffle(ids)
        disc, val = ids[: len(ids) // 2], ids[len(ids) // 2 :]
        for th in thresholds:
            sel = select_expert(et, disc, th)
            e_star = sel["e_star"]
            if e_star is None:
                rows.append({"seed": seed, "threshold": th, "e_star": None})
                continue
            ev = evaluate_expert(md, layer, e_star, val, n_controls)
            rows.append({"seed": seed, "threshold": th, "e_star": e_star, "val_rescue": ev["rescue_all"]["mean"],
                         "val_spec": ev["spec_all"]["mean"], "val_active": ev["val_active"], "n_candidates": sel["n_candidates"]})
    df = pd.DataFrame(rows)
    ref = reference_expert
    return {"grid": df, "n_settings": len(df), "n_reference_selected": int((df.e_star == ref).sum()) if ref is not None else None,
            "selected_counts": df.e_star.value_counts().to_dict(), "mean_val_rescue": float(df.val_rescue.mean()),
            "mean_val_spec": float(df.val_spec.mean())}


def relation_heldout(md: ModelData, layer: int, set_name: str, n_controls: int, n_folds: int = 5, seed: int = 0,
                     reference_expert: Optional[int] = None) -> dict:
    """Appendix D: select on non-held-out relations (threshold = half the selection cases), evaluate on held-out."""
    et = expert_table(md, layer)
    all_ids = md.ids(set_name, "discovery") + md.ids(set_name, "validation")
    rel = md.relation_of()
    rels = sorted(set(rel[c] for c in all_ids))
    random.Random(seed).shuffle(rels)
    folds = [rels[i::n_folds] for i in range(n_folds)]
    per_case = []
    fold_info = []
    for fi, held in enumerate(folds):
        held = set(held)
        train = [c for c in all_ids if rel[c] not in held]
        test = [c for c in all_ids if rel[c] in held]
        sel = select_expert(et, train, len(train) // 2)
        e_star = sel["e_star"]
        ev = evaluate_expert(md, layer, e_star, test, n_controls)
        pc = ev["per_case"].copy()
        pc["fold"] = fi
        pc["e_star"] = e_star
        per_case.append(pc)
        fold_info.append({"fold": fi, "held_relations": sorted(held), "n_train": len(train), "n_test": len(test), "e_star": e_star,
                          "test_rescue": ev["rescue_all"]["mean"], "test_spec": ev["spec_all"]["mean"], "test_active": ev["val_active"]})
    pc = pd.concat(per_case)
    return {"folds": fold_info, "n_selected_reference": int(sum(f["e_star"] == reference_expert for f in fold_info)),
            "n_heldout": len(pc), "active": int(pc.active.sum()), "rescue": summarize(pc.rescue.values), "spec": summarize(pc.spec.dropna().values),
            "per_case": pc}


def noise_table(md: ModelData, layer: int, e_star: int, val_ids: list[int]) -> pd.DataFrame:
    """Table 13: sigma rows; sigma=3 from the expert pass, others from noise_rows."""
    rows = []
    e = md.expert_rows
    e3 = e[(e.layer == layer) & (e.kind == "expert") & (e.expert == e_star) & e.case_id.isin(val_ids)]
    # all-case: zero where e* not clean-active
    et = expert_table(md, layer)
    cs = clean_sets(et)
    r3 = []
    dr3 = []
    ct = md.cases.set_index("case_id")
    for cid in val_ids:
        active = e_star in cs.get(cid, [])
        rr = e3[e3.case_id == cid]
        r3.append(float(rr.rescue.iloc[0]) if (active and len(rr)) else 0.0)
        dr3.append(float(ct.loc[cid, "drop"]))
    s = summarize(np.array(r3))
    rows.append({"sigma": 3.0, "active": int(sum(e_star in cs.get(c, []) for c in val_ids)), "n": len(val_ids), "drop": float(np.mean(dr3)),
                 "rescue": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "pos_frac": s["pos_frac"], "source": "expert pass"})
    if md.noise_rows is not None:
        nr = md.noise_rows[md.noise_rows.case_id.isin(val_ids)]
        for sm, g in nr[nr.kind == "expert"].groupby("sigma_mult"):
            g = g.set_index("case_id").loc[[c for c in val_ids if c in set(g.case_id)]]
            resc = np.where(g.clean_active.astype(bool).values, g.rescue.values, 0.0)
            s = summarize(resc)
            nz = nr[(nr.kind == "noised") & (nr.sigma_mult == sm)]
            lay = nr[(nr.kind == "layer") & (nr.sigma_mult == sm)]
            rows.append({"sigma": float(sm), "active": int(g.clean_active.astype(bool).sum()), "n": len(g), "drop": float((nz.delta_clean - nz.delta_noised).mean()),
                         "rescue": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "pos_frac": s["pos_frac"], "source": "noise pass",
                         "layer_rescue": float(lay.rescue.mean())})
    return pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)


def funnel_check(md: ModelData) -> dict:
    ct = md.cases
    p = ct[ct.in_paper] if "in_paper" in ct else None
    out = {}
    if p is not None:
        out["paper_n"] = int(len(p))
        out["paper_strict_pass"] = int(p.strict.sum())
        out["paper_relaxed_pass"] = int(p.relaxed.sum())
        out["paper_fail_strict_ids"] = p[~p.strict].case_id.tolist()
        out["paper_delta_clean_mean"] = float(p.delta_clean.mean())
        out["paper_drop_mean"] = float(p["drop"].mean())
    return out
