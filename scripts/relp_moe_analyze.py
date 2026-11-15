#!/usr/bin/env python3
"""Phase 7 S5 analysis (docs/relp_moe_plan.md). CPU only; re-runnable at any time on whatever results exist.

A  RelP protocol results per model / profile / task (results/relp_sva_moe/...): verification, discovery margins, top
   neurons / experts (positions labelled by template role), layer and position profiles of the expert scores, held-out
   interventions (RelP vs the mean of the matched random sets), macro averages over tasks.
B  Lu-style STR patching (patch_full / patch_layerpos / patch_final): layer x position maps (denoise rescue, noise drop),
   the paper's two-stage selection through moetrace.analysis unchanged (layer_analysis, select_expert with recurrence
   gate = half of discovery, evaluate_expert with Spec; n_controls 3 Qwen3 / 1 Mixtral), confirmed on validation and test.
C  Comparison (agreed design, 2026-11-07): (a) RelP endpoint-B scores through Lu's rule, (b) each method's own rule,
   (c) all (layer, position, expert) units: Spearman / Kendall, top-1 / top-10 agreement, sign agreement, slope;
   denoise <-> RelP endpoint B, noise <-> RelP endpoint A; group-bootstrap stability of each method's top-1.
Writes results/relp_sva_moe/summary.json and results/tables/relp_moe_*.md.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results/relp_sva_moe"
TAB = ROOT / "results/tables"
MODELS = ("qwen3", "mixtral")
PROFILES = ("smoke", "paper", "grouped")
TASKS = ("simple", "nounpp", "rc", "within_rc")
ROLES = {"simple": ["the", "SUBJ"], "nounpp": ["the", "SUBJ", "prep", "the", "noun2"],
         "rc": ["the", "SUBJ", "that", "the", "noun2", "verb2"], "within_rc": ["the", "noun1", "that", "the", "SUBJ"]}
N_CONTROLS = {"qwen3": 3, "mixtral": 1}
KINDS = ("remove", "keep", "noise", "denoise")


def jload(p):
    return json.loads(Path(p).read_text())


def complete(d):
    p = Path(d) / "run_meta.json"
    return p.exists() and jload(p).get("status") == "complete"


def role_of(task, model):
    """position -> role label (Mixtral has <s> at position 0)."""
    r = ROLES[task]
    return (["<s>"] + r) if model == "mixtral" else list(r)


def manifest_rows(model, profile, task, split):
    p = ROOT / f"data/relp_sva/{model}_{profile}_v1/{task}/{split}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# ----------------------------------------------------------------------------------------------------------------------
# A. RelP protocol results
# ----------------------------------------------------------------------------------------------------------------------
def load_aggregate(d):
    import torch
    return torch.load(Path(d) / "aggregate.pt", map_location="cpu", weights_only=True)


def relp_task(model, profile, task):
    b = RES / model / profile / task
    out = {"model": model, "profile": profile, "task": task}
    if not complete(b / "verify"):
        out["status"] = "no passing verify" if (b / "verify").exists() else "not run"
        return out
    v = jload(b / "verify/verification.json")
    out["verify"] = {"passed": v["passed"], "n_checks": len(v["checks"]), **v["diagnostics"]}
    sz = b / "score_zero"
    if not complete(sz):
        out["status"] = "verify only"
        return out
    cases = pd.read_parquet(sz / "cases.parquet")
    out["discovery"] = {"n": len(cases), "margin_A": float(cases.margin_A.mean()), "margin_B": float(cases.margin_B.mean()),
                        "acc_A": float((cases.margin_A > 0).mean()), "acc_B": float((cases.margin_B < 0).mean()),
                        "paired_diff": float((cases.margin_A - cases.margin_B).mean())}
    roles = role_of(task, model)
    agg = load_aggregate(sz)
    mean = agg["mean"]                                   # [L, T, E, I] float64
    ep = mean.sum(-1)                                    # [L, T, E]
    out["top_expert_position"] = [{"layer": c["coordinate"][0], "pos": c["coordinate"][1], "role": roles[c["coordinate"][1]],
                                   "expert": c["coordinate"][2], "score": c["score"]}
                                  for c in jload(sz / "expert_position_ranking.json")[:10]]
    out["top_expert_all_positions"] = [{"layer": c["coordinate"][0], "expert": c["coordinate"][1], "score": c["score"]}
                                       for c in jload(sz / "expert_all_positions_ranking.json")[:10]]
    out["top_neuron"] = [{"layer": c["coordinate"][0], "pos": c["coordinate"][1], "role": roles[c["coordinate"][1]],
                          "expert": c["coordinate"][2], "neuron": c["coordinate"][3], "score": c["score"]}
                         for c in jload(sz / "neuron_ranking.json")[:10]]
    absep = ep.abs()
    tot = float(absep.sum())
    flat = absep.flatten().sort(descending=True).values
    out["concentration"] = {"top1_share_abs": float(flat[0] / tot), "top10_share_abs": float(flat[:10].sum() / tot),
                            "n_expert_positions_for_50pct": int((flat.cumsum(0) < 0.5 * tot).sum()) + 1}
    out["position_profile"] = {roles[t]: float(ep[:, t].sum()) for t in range(ep.shape[1])}
    out["position_profile_abs"] = {roles[t]: float(absep[:, t].sum()) for t in range(ep.shape[1])}
    out["layer_profile"] = [float(x) for x in ep.sum((1, 2))]
    out["total_signed"] = float(ep.sum())
    # held-out interventions
    ev = {}
    for split in ("validation", "test"):
        root = b / split
        if not complete(root):
            continue
        for sub in sorted(p for p in root.iterdir() if p.is_dir() and complete(p)):
            summ = jload(sub / "summary.json")
            tab = defaultdict(dict)
            for r in summ:
                key = (r["budget"], r["intervention"])
                if r["ranking"] == "relp":
                    tab[key]["relp"] = r["normalized"]
                    tab[key]["relp_acc"] = r["accuracy_after"]
                    tab[key]["acc_A"] = r["accuracy_A"]
                else:
                    tab[key].setdefault("random", []).append(r["normalized"])
            rows = []
            for (bud, kind), d in sorted(tab.items()):
                rnd = [x for x in d.get("random", []) if x is not None]
                rows.append({"budget": bud, "intervention": kind, "relp": d.get("relp"), "relp_acc_after": d.get("relp_acc"),
                             "random_mean": float(np.mean(rnd)) if rnd else None, "acc_A": d.get("acc_A")})
            ev[sub.name] = rows
    out["evaluation"] = ev
    out["status"] = "complete" if complete(b / "test") or profile == "smoke" else "partial"
    return out


def stability(model, profile, task):
    """S2: the same paired RelP score with another batch composition (3 chunks of 100 pairs instead of 300): overlap of
    the protocol's ranked lists and correlation of the expert-level means."""
    import torch
    b = RES / model / profile / task
    if not (complete(b / "score_zero") and complete(b / "score_zero_chunk100")):
        return None
    out = {}
    for unit, k in (("neuron", 1000), ("expert_position", 50), ("expert_all_positions", 20)):
        r1 = [tuple(c["coordinate"]) for c in jload(b / f"score_zero/{unit}_ranking.json")[:k]]
        r2 = [tuple(c["coordinate"]) for c in jload(b / f"score_zero_chunk100/{unit}_ranking.json")[:k]]
        out[unit] = {"k": k, "overlap": len(set(r1) & set(r2)) / k, "top1_same": r1[0] == r2[0],
                     "top10_same_set": set(r1[:10]) == set(r2[:10])}
    a1 = load_aggregate(b / "score_zero")["mean"].sum(-1).flatten().numpy()
    a2 = load_aggregate(b / "score_zero_chunk100")["mean"].sum(-1).flatten().numpy()
    out["expert_position_corr"] = float(np.corrcoef(a1, a2)[0, 1])
    out["expert_position_rel_l2"] = float(np.linalg.norm(a1 - a2) / np.linalg.norm(a1))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# B. patching
# ----------------------------------------------------------------------------------------------------------------------
def load_patch(b):
    parts = []
    for name in ("patch_full", "patch_layerpos", "patch_final"):
        d = b / name
        if complete(d):
            df = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("part_*.parquet"))], ignore_index=True)
            df["source_run"] = name
            parts.append(df)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    # one row per (row, direction, kind, layer, pos, expert): prefer patch_full, then the others (identical units recur)
    return df.drop_duplicates(["split", "row", "direction", "kind", "layer", "pos", "expert"], keep="first")


def lu_modeldata(df, rows_by_split, final_pos):
    """moetrace.analysis.ModelData from the denoise rows at the final position (the paper's protocol)."""
    from moetrace.analysis import ModelData
    d = df[(df.direction == "denoise") & (df.pos == final_pos)]
    cid = {}
    sets = {"sva": {}}
    cases = []
    for sp in ("discovery", "validation", "test"):
        ids = []
        for i, r in enumerate(rows_by_split.get(sp, [])):
            k = len(cid)
            cid[(sp, r["row_id"])] = k
            ids.append(k)
        sets["sva"][sp] = ids
    key = list(zip(d.split, d.row_id))
    d = d.assign(case_id=[cid[k] for k in key])
    lay = d[d.kind == "position"][["case_id", "layer", "effect", "margin_A", "margin_B"]].rename(columns={"effect": "rescue"})
    lay["kind"] = "layer"
    base = lay.groupby("case_id")[["margin_A", "margin_B"]].first().reset_index()
    rel = {cid[(sp, r["row_id"])]: r.get("case") for sp, rs in rows_by_split.items() for r in rs if (sp, r["row_id"]) in cid}
    cases = pd.DataFrame({"case_id": base.case_id, "relation": base.case_id.map(rel), "delta_clean": base.margin_A,
                          "delta_corrupt": base.margin_B, "drop": base.margin_A - base.margin_B})
    ex = d[d.kind == "expert"][["case_id", "layer", "expert", "effect", "active_A"]].rename(
        columns={"effect": "rescue", "active_A": "clean_active"})
    ex["kind"] = "expert"
    ex["partner"] = -1
    return ModelData("sva", sets, lay, None, cases, ex, None)


def lu_selection(md, model):
    from moetrace.analysis import evaluate_expert, expert_table, layer_analysis, select_expert
    la = layer_analysis(md, "sva")
    disc, val, test = md.sets["sva"]["discovery"], md.sets["sva"]["validation"], md.sets["sva"]["test"]
    L = la["L_star"]
    sel = select_expert(expert_table(md, L), disc, len(disc) // 2)
    out = {"L_star": L, "layer_disc_top5": la["disc_curve_top5"], "layer_val_at_Lstar": la["val_at_Lstar"],
           "selection": {k: v for k, v in sel.items()}}
    if sel["e_star"] is not None:
        for name, ids in (("validation", val), ("test", test)):
            if ids:
                ev = evaluate_expert(md, L, sel["e_star"], ids, N_CONTROLS[model])
                out[name] = {k: v for k, v in ev.items() if k != "per_case"}
    # joint search over all layers (final position, recurrence gate), Direction 1 style
    best = None
    for l in sorted(md.expert_rows.layer.unique()):
        s = select_expert(expert_table(md, int(l)), disc, len(disc) // 2)
        if s["e_star"] is not None and (best is None or s["disc_allcase_mean"] > best[2]):
            best = (int(l), s["e_star"], s["disc_allcase_mean"])
    out["joint_best"] = best
    return out


def unit_means(df, split="discovery"):
    """Mean effect per unit over all rows of the split (rows where the unit was not enumerated count as 0, i.e. the
    expert is routed at neither endpoint and the patch is exactly a no-op), per direction and kind."""
    d = df[df.split == split]
    n = d.row.nunique()
    g = d.groupby(["direction", "kind", "layer", "pos", "expert"]).effect.sum() / n
    return g.rename("mean_effect").reset_index(), n


# ----------------------------------------------------------------------------------------------------------------------
# C. comparison
# ----------------------------------------------------------------------------------------------------------------------
def spearman(x, y):
    from scipy.stats import kendalltau, spearmanr
    if len(x) < 3:
        return None, None
    return float(spearmanr(x, y).statistic), float(kendalltau(x, y).statistic)


def compare(model, profile, task, df, lu):
    import torch
    b = RES / model / profile / task
    if not complete(b / "score_endpointB") or not complete(b / "score_zero"):
        return None
    epB = load_aggregate(b / "score_endpointB")["mean"].sum(-1).numpy()      # [L, T, E]
    epA = load_aggregate(b / "score_zero")["mean"].sum(-1).numpy()
    um, n = unit_means(df)
    out = {"n_discovery": n}
    T = epB.shape[1]
    for direction, ep, label in (("denoise", epB, "RelP endpoint B vs Lu rescue (denoise)"),
                                 ("noise", epA, "RelP endpoint A (protocol) vs noise drop")):
        for kind in ("expert", "position"):
            u = um[(um.direction == direction) & (um.kind == kind)]
            if u.empty:
                continue
            if kind == "expert":
                pred = ep[u.layer.values, u.pos.values, u.expert.values]
            else:
                pred = ep[u.layer.values, u.pos.values].sum(-1)
            exact = u.mean_effect.values
            rho, tau = spearman(pred, exact)
            k = min(10, len(u))
            top_e = set(np.argsort(-exact)[:k])
            top_p = set(np.argsort(-pred)[:k])
            big = np.abs(exact) >= 0.05
            sig = np.maximum(np.abs(exact), np.abs(pred)) >= 0.05
            rho_sig, _ = spearman(pred[sig], exact[sig]) if sig.sum() > 2 else (None, None)
            res = {"label": label, "units": int(len(u)), "spearman": rho, "kendall": tau,
                   "pearson": float(np.corrcoef(pred, exact)[0, 1]) if len(u) > 2 else None,
                   "units_effect>=0.05": int(sig.sum()), "spearman_effect>=0.05": rho_sig,
                   "top1_same": bool(np.argmax(exact) == np.argmax(pred)), "top10_overlap": len(top_e & top_p) / k,
                   "sign_agree_|exact|>=0.05": float((np.sign(pred[big]) == np.sign(exact[big])).mean()) if big.any() else None,
                   "slope_exact_on_pred": float(np.polyfit(pred, exact, 1)[0]) if len(u) > 2 else None,
                   "top1_exact": u.iloc[int(np.argmax(exact))][["layer", "pos", "expert"]].astype(int).tolist() + [float(exact.max())],
                   "top1_relp": u.iloc[int(np.argmax(pred))][["layer", "pos", "expert"]].astype(int).tolist() + [float(pred.max())]}
            for scope, mask in (("final", u.pos.values == T - 1),):
                if mask.sum() > 2:
                    r2, _ = spearman(pred[mask], exact[mask])
                    res[f"spearman_{scope}"] = r2
            out[f"{direction}/{kind}"] = res
    # early layers: whole-MoE-output patches at layers 0-2 (exact denoise vs RelP endpoint B summed over experts)
    u = um[(um.direction == "denoise") & (um.kind == "position") & (um.layer <= 2)]
    out["early_position_units"] = [{"layer": int(r.layer), "pos": int(r.pos), "exact": float(r.mean_effect),
                                    "relp_B": float(epB[int(r.layer), int(r.pos)].sum())} for r in u.itertuples()]
    # S4f unit list: the units that decide agreement or disagreement (written next to the task's results)
    out["diagnose_units"] = diagnose_units(um, epB, lu, T, out)
    # (a) same rule: RelP endpoint B through Lu's two-stage rule at the final position
    if lu and "L_star" in lu:
        fin = T - 1
        lay_score = epB[:, fin].sum(-1)
        L_r = int(np.argmax(lay_score))
        d = df[(df.split == "discovery") & (df.direction == "denoise") & (df.kind == "expert") & (df.pos == fin) & df.active_A]
        act = d[d.layer == L_r].groupby("expert").row.nunique()
        cands = act[act >= n // 2].index.tolist()
        e_r = int(max(cands, key=lambda e: epB[L_r, fin, e])) if cands else None
        out["same_rule"] = {"lu": [lu["L_star"], lu["selection"]["e_star"]], "relp": [L_r, e_r],
                            "layer_same": L_r == lu["L_star"], "expert_same": (L_r, e_r) == (lu["L_star"], lu["selection"]["e_star"]),
                            "relp_layer_top5": [(int(l), round(float(lay_score[l]), 3)) for l in np.argsort(-lay_score)[:5]]}
    # (b) own rules: RelP protocol top expert_position (|score|) vs Lu
    rk = jload(b / "score_zero/expert_position_ranking.json")[:5]
    out["own_rule"] = {"relp_top5_abs": [c["coordinate"] + [round(c["score"], 3)] for c in rk],
                       "lu": [lu.get("L_star"), T - 1, lu.get("selection", {}).get("e_star")] if lu else None}
    out["bootstrap_top1"] = bootstrap_top1(b, df, T)
    return out


def diagnose_units(um, epB, lu, T, cmp, k_gap=5, k_pos=3):
    """Units for the S4f diagnostics: the position units with the largest |exact - RelP| gap (denoise), the selected
    experts of both methods (Lu e*, joint best, RelP top-1 at the final position), and the expert units with the largest
    gap among those with an effect >= 0.05. Deduplicated, at most ~15."""
    units = []
    add = lambda kind, l, t, e=-1: units.append({"kind": kind, "layer": int(l), "pos": int(t), "expert": int(e)})
    pos = um[(um.direction == "denoise") & (um.kind == "position")].copy()
    pos["pred"] = epB[pos.layer.values, pos.pos.values].sum(-1)
    pos["gap"] = (pos.mean_effect - pos.pred).abs()
    for r in pos.nlargest(k_pos, "gap").itertuples():
        add("position", r.layer, r.pos)
    if lu and lu.get("selection", {}).get("e_star") is not None:
        add("expert", lu["L_star"], T - 1, lu["selection"]["e_star"])
    if lu and lu.get("joint_best"):
        add("expert", lu["joint_best"][0], T - 1, lu["joint_best"][1])
    ex = um[(um.direction == "denoise") & (um.kind == "expert")].copy()
    ex["pred"] = epB[ex.layer.values, ex.pos.values, ex.expert.values]
    fin = ex[ex.pos == T - 1]
    if len(fin):
        r = fin.loc[fin.pred.idxmax()]
        add("expert", r.layer, r.pos, r.expert)
    sig = ex[np.maximum(ex.mean_effect.abs(), ex.pred.abs()) >= 0.05].copy()
    sig["gap"] = (sig.mean_effect - sig.pred).abs()
    for r in sig.nlargest(k_gap, "gap").itertuples():
        add("expert", r.layer, r.pos, r.expert)
    seen, uniq = set(), []
    for x in units:
        key = (x["kind"], x["layer"], x["pos"], x["expert"])
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    return uniq


def bootstrap_top1(b, df, T, n_boot=1000, seed=0):
    """Group bootstrap (resample prefix-pair groups of the discovery split) of each method's top-1 final-position expert
    (no recurrence gate): exact denoise rescue vs RelP endpoint B (per-row scores)."""
    import torch
    p = b / "score_endpointB/per_row_expert_scores.pt"
    if not p.exists():
        return None
    pr = torch.load(p, weights_only=False)
    rb = pr["B"][:, :, T - 1].numpy()                        # [n, L, E]
    groups = np.asarray(pr["group_ids"])
    rid = {r: i for i, r in enumerate(pr["row_ids"])}
    d = df[(df.split == "discovery") & (df.direction == "denoise") & (df.kind == "expert") & (df.pos == T - 1)]
    ex = np.zeros_like(rb)
    ex[[rid[r] for r in d.row_id], d.layer.values, d.expert.values] = d.effect.values
    ug, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    cnt_e, cnt_r = defaultdict(int), defaultdict(int)
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        w = np.bincount(pick, minlength=len(ug))[inv]
        me = (ex * w[:, None, None]).sum(0)
        mr = (rb * w[:, None, None]).sum(0)
        cnt_e[np.unravel_index(np.argmax(me), me.shape)] += 1
        cnt_r[np.unravel_index(np.argmax(mr), mr.shape)] += 1
    top = lambda c: [(int(k[0]), int(k[1]), v / n_boot) for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:3]]
    return {"exact_denoise": top(cnt_e), "relp_endpointB": top(cnt_r), "n_groups": int(len(ug))}


# ----------------------------------------------------------------------------------------------------------------------
# D. diagnostics of disagreements (S4f)
# ----------------------------------------------------------------------------------------------------------------------
def diagnose_summary(b, epA, epB, gap_abs=0.1, gap_rel=0.25):
    """Per diagnosed unit and direction: mean effects (natural, frozen routing, frozen attention, both frozen, direct),
    the RelP prediction of the same direction (denoise <- endpoint B, noise <- endpoint A), shares of the natural effect
    carried by later routing switches / attention-pattern changes / the direct path, and, for disagreements
    (|RelP - natural| > max(gap_abs, gap_rel |natural|)), the dominant cause:
      routing-mediated      the natural effect collapses once later routing is frozen (RelP freezes the router)
      attention-mediated    it collapses once later attention patterns are frozen (RelP freezes them too)
      nonlinear (frozen)    RelP also misses the both-frozen effect: the remaining nonlinearity of the half rule /
                            frozen gates and norms (true linearisation error)."""
    p = b / "diagnose/diagnose.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    cols = [c for c in d.columns if c.startswith("effect_")] + ["direct"]
    g = d.groupby(["direction", "kind", "layer", "pos", "expert"])[cols].mean().reset_index()
    rows = []
    for r in g.itertuples():
        ep = epB if r.direction == "denoise" else epA
        pred = float(ep[r.layer, r.pos].sum()) if r.kind == "position" else float(ep[r.layer, r.pos, r.expert])
        nat = r.effect_natural
        gap = pred - nat
        disagree = abs(gap) > max(gap_abs, gap_rel * abs(nat))
        via_route = nat - r.effect_frozen_routing
        via_attn = nat - r.effect_frozen_attention
        cause = None
        if disagree:
            if abs(via_route) >= 0.5 * abs(gap) and abs(r.effect_frozen_routing - pred) < abs(gap):
                cause = "routing-mediated"
            elif abs(via_attn) >= 0.5 * abs(gap) and abs(r.effect_frozen_attention - pred) < abs(gap):
                cause = "attention-mediated"
            else:
                cause = "nonlinear (frozen)"
        rows.append({"direction": r.direction, "kind": r.kind, "layer": int(r.layer), "pos": int(r.pos), "expert": int(r.expert),
                     "natural": float(nat), "frozen_routing": float(r.effect_frozen_routing),
                     "frozen_attention": float(r.effect_frozen_attention), "frozen_both": float(r.effect_frozen_both),
                     "direct": None if np.isnan(r.direct) else float(r.direct), "relp": pred, "gap_relp_minus_exact": float(gap),
                     "disagree": bool(disagree), "cause": cause})
    return rows


# ----------------------------------------------------------------------------------------------------------------------
def main():
    summary = {"relp": {}, "patch": {}, "compare": {}}
    for model in MODELS:
        for profile in PROFILES:
            for task in TASKS:
                if profile == "smoke" and task not in ("simple", "nounpp"):
                    continue
                key = f"{model}/{profile}/{task}"
                summary["relp"][key] = relp_task(model, profile, task)
                st = stability(model, profile, task)
                if st:
                    summary.setdefault("stability", {})[key] = st
                b = RES / model / profile / task
                df = load_patch(b)
                if df is None:
                    continue
                rows = {sp: manifest_rows(model, profile, task, sp) for sp in ("discovery", "validation", "test")}
                T = len(rows["discovery"][0]["input_ids"])
                lu = None
                try:
                    lu = lu_selection(lu_modeldata(df, rows, T - 1), model)
                except Exception as e:  # recorded, not hidden
                    lu = {"error": repr(e)}
                um, n = unit_means(df)
                pos = um[(um.kind == "position")]
                summary["patch"][key] = {"lu": lu, "n_discovery": n,
                                         "layer_pos_top5": {dr: pos[pos.direction == dr].nlargest(5, "mean_effect")[
                                             ["layer", "pos", "mean_effect"]].values.tolist() for dr in ("denoise", "noise")},
                                         "route_changes_mean": float(df.route_changes.mean())}
                c = compare(model, profile, task, df, lu if lu and "error" not in lu else None)
                if c and complete(b / "diagnose"):
                    c["diagnostics"] = diagnose_summary(b, load_aggregate(b / "score_zero")["mean"].sum(-1).numpy(),
                                                        load_aggregate(b / "score_endpointB")["mean"].sum(-1).numpy())
                if c:
                    summary["compare"][key] = c
                    (b / "diagnose_units.json").write_text(json.dumps(c["diagnose_units"], indent=1) + "\n")
    RES.mkdir(parents=True, exist_ok=True)
    # comparison: selections and agreement per task
    lines = ["| model/profile/task | Lu: L*, e* (val rescue, Spec, test rescue) | joint best (final pos) | RelP same rule | RelP top-1 (own rule) | "
             "exact top-1 unit | agree top-1 | Pearson | rank corr (effect>=0.05) | slope exact~RelP | bootstrap top-1 exact / RelP |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in summary["compare"].items():
        lu = summary["patch"][key]["lu"]
        x = c.get("denoise/expert", {})
        v, te = lu.get("validation", {}), lu.get("test", {})
        sel = lu.get("selection", {})
        lus = f"L{lu.get('L_star')} E{sel.get('e_star')} ({v.get('rescue_all', {}).get('mean', float('nan')):.2f}, " \
              f"{v.get('spec_all', {}).get('mean', float('nan')):.2f}, {te.get('rescue_all', {}).get('mean', float('nan')):.2f})"
        jb = lu.get("joint_best")
        sr = c.get("same_rule", {}).get("relp")
        own = c["own_rule"]["relp_top5_abs"][0]
        bt = c.get("bootstrap_top1") or {}
        fb = lambda z: f"L{z[0]}E{z[1]} {z[2]:.2f}" if z else ""
        lines.append(f"| {key} | {lus} | {'L%dE%d' % tuple(jb[:2]) if jb else ''} | {'L%sE%s' % tuple(sr) if sr else ''} | "
                     f"L{own[0]} p{own[1]} E{own[2]} | L{x['top1_exact'][0]} p{x['top1_exact'][1]} E{x['top1_exact'][2]} | {x['top1_same']} | "
                     f"{x['pearson']:.3f} | {(x['spearman_effect>=0.05'] or float('nan')):.3f} | {x['slope_exact_on_pred']:.2f} | "
                     f"{fb((bt.get('exact_denoise') or [None])[0])} / {fb((bt.get('relp_endpointB') or [None])[0])} |")
    (TAB / "relp_moe_compare.md").write_text("\n".join(lines) + "\n")
    # diagnostics
    lines = ["| model/profile/task | unit | natural (exact) | frozen routing | frozen attention | both frozen | direct | RelP | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, c in summary["compare"].items():
        for r in c.get("diagnostics") or []:
            if r["direction"] != "denoise":
                continue
            unit = f"L{r['layer']} p{r['pos']}" + ("" if r["kind"] == "position" else f" E{r['expert']}")
            g = lambda z: "" if z is None else f"{z:.2f}"
            lines.append(f"| {key} | {r['kind']} {unit} | {g(r['natural'])} | {g(r['frozen_routing'])} | {g(r['frozen_attention'])} | "
                         f"{g(r['frozen_both'])} | {g(r['direct'])} | {g(r['relp'])} | {r['cause'] or 'agree'} |")
    (TAB / "relp_moe_diagnostics.md").write_text("\n".join(lines) + "\n")
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_tables(summary)
    print("wrote", RES / "summary.json")


def write_tables(summary):
    TAB.mkdir(parents=True, exist_ok=True)
    lines = ["| model | profile | task | verify | n disc | acc A | margin A | margin B | top expert-position (L, role, E, score) | top expert (L, E, score) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in summary["relp"].items():
        if "discovery" not in r:
            lines.append(f"| {r['model']} | {r['profile']} | {r['task']} | {r.get('status')} | | | | | | |")
            continue
        d = r["discovery"]
        te = r["top_expert_position"][0]
        ta = r["top_expert_all_positions"][0]
        lines.append(f"| {r['model']} | {r['profile']} | {r['task']} | {r['verify']['n_checks']} ok | {d['n']} | {d['acc_A']:.2f} | "
                     f"{d['margin_A']:.2f} | {d['margin_B']:.2f} | L{te['layer']} {te['role']} E{te['expert']} {te['score']:+.3f} | "
                     f"L{ta['layer']}E{ta['expert']} {ta['score']:+.3f} |")
    (TAB / "relp_moe_overview.md").write_text("\n".join(lines) + "\n")
    lines = ["| model/profile/task | eval | budget | remove | keep | noise | denoise | (random mean: remove / keep / noise / denoise) |",
             "|---|---|---|---|---|---|---|---|"]
    for key, r in summary["relp"].items():
        for name, rows in r.get("evaluation", {}).items():
            by = defaultdict(dict)
            for x in rows:
                by[x["budget"]][x["intervention"]] = x
            f = lambda v: "" if v is None else f"{v:.2f}"
            for bud in sorted(by):
                rr = by[bud]
                lines.append(f"| {key} | {name} | {bud} | " + " | ".join(f(rr.get(k, {}).get("relp")) for k in KINDS) + " | " +
                             " / ".join(f(rr.get(k, {}).get("random_mean")) for k in KINDS) + " |")
    (TAB / "relp_moe_interventions.md").write_text("\n".join(lines) + "\n")
    # macro averages over the four tasks (test split): RelP list vs matched random, per model / profile / unit / baseline
    macro = {}
    for key, r in summary["relp"].items():
        model, profile, task = key.split("/")
        if profile == "smoke":
            continue
        for name, rows in r.get("evaluation", {}).items():
            if not name.startswith("test_"):
                continue
            unit_base = name[len("test_"):]
            for x in rows:
                k = (model, profile, unit_base, x["budget"], x["intervention"])
                macro.setdefault(k, {"relp": [], "random": []})
                if x["relp"] is not None:
                    macro[k]["relp"].append(x["relp"])
                if x["random_mean"] is not None:
                    macro[k]["random"].append(x["random_mean"])
    summary["macro_test"] = {"/".join(map(str, k)): {"relp": float(np.mean(v["relp"])) if v["relp"] else None,
                                                   "random": float(np.mean(v["random"])) if v["random"] else None,
                                                   "n_tasks": len(v["relp"])} for k, v in macro.items()}
    lines = ["| model | profile | unit_baseline | budget | denoise RelP / random | remove RelP / random | keep RelP / random |",
             "|---|---|---|---|---|---|---|"]
    keys = sorted({k[:4] for k in macro})
    f = lambda v: "" if v is None else f"{v:.2f}"
    for k in keys:
        cells = []
        for kind in ("denoise", "remove", "keep"):
            v = summary["macro_test"].get("/".join(map(str, k + (kind,))))
            cells.append(f"{f(v and v['relp'])} / {f(v and v['random'])}" if v else "")
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(cells) + " |")
    (TAB / "relp_moe_macro_test.md").write_text("\n".join(lines) + "\n")
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
