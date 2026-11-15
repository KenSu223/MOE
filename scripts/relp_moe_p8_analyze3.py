#!/usr/bin/env python3
"""Phase 8 stage 3 analysis (docs/relp_moe_phase8_plan.md section 8d): CounterFact and two-hop capitals, outputs of
scripts/relp_moe_p8_run3.py in results/relp_p8/<model>/<task>/. Units = (layer, role, expert), role 0 = subject (last token
of the changed word), role 1 = final token; discovery means count a pair in which the expert is not routed as 0.
CPU only (.venv-analysis). Writes results/tables/relp_moe_p8s3_*.md, results/relp_p8/summary.json, figures
results/figures/relp_moe_p8s3_*.png. Usage: .venv-analysis/bin/python scripts/relp_moe_p8_analyze3.py [--base results/relp_p8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import relp_moe_p8_data as D  # noqa: E402
from relp_moe_p8_analyze import md_table, pearson, slope0, spearman  # noqa: E402

TAB = ROOT / "results/tables"
FIG = ROOT / "results/figures"
ROLES = ("subject", "final")
CLS = ("subject_other", "subject", "between", "final")
NL = {"qwen3": 48, "mixtral": 32}
MODEL_NAME = {"qwen3": "Qwen3-30B-A3B", "mixtral": "Mixtral-8x7B"}
TASK_NAME = {"counterfact": "CounterFact", "twohop": "two-hop capitals"}


def complete(p):
    f = Path(p) / "run_meta.json"
    return f.exists() and json.loads(f.read_text()).get("status") == "complete"


def ename(l, r, e):
    return f"L{l}E{e:03d}@{ROLES[r]}"


class T3:
    def __init__(self, base, model, task):
        self.b, self.model, self.task = base / model / task, model, task
        self.L = NL[model]
        self.rows = pd.read_parquet(self.b / "patch/rows.parquet")
        self.nd = int((self.rows.split == "discovery").sum())
        self.sB, self.sA = np.load(self.b / "rules/scores_B.npz"), np.load(self.b / "rules/scores_A.npz")
        self.cases = pd.read_parquet(self.b / "rules/cases.parquet")
        p = pd.read_parquet(self.b / "patch/patch.parquet")
        self.p = p[p.row < self.nd]
        self.held = p[p.row >= self.nd]
        self.E = int(self.sB["v_relp_role"].shape[-1])

    def pair_units(self, direction="denoise"):
        e = self.p[(self.p.kind == "expert") & (self.p.role >= 0) & (self.p.direction == direction)].copy()
        s = self.sB if direction == "denoise" else self.sA
        for v in s["variants"]:
            e[str(v)] = s[f"v_{v}_role"][e.row.values, e.layer.values, e.role.values, e.expert.values]
        e["cls"] = np.where(e.active_A & e.active_B, "both", np.where(e.active_A, "clean_only", "corrupt_only"))
        e["band"] = D.band_of(e.layer.values, self.L)
        return e

    def unit_table(self):
        n = self.nd
        e = self.pair_units("denoise")
        vs = [str(v) for v in self.sB["variants"]]
        e["e2"] = e.effect.astype(float) ** 2
        g = e.groupby(["layer", "role", "expert"])
        u = pd.DataFrame({"exact": g.effect.sum() / n, "n_present": g.size(), "route_changes": g.route_changes.mean()})
        u["exact_se"] = np.sqrt(((g.e2.sum() / n - u.exact ** 2).clip(lower=0) * n / (n - 1)) / n)
        u = u.reset_index()
        for v in vs:        # RelP means over ALL discovery pairs (non-zero only where routed at A or B)
            u[v] = self.sB[f"v_{v}_role"].mean(0)[u.layer.values, u.role.values, u.expert.values]
        rsd = self.sB["v_relp_role"].std(0, ddof=1)[u.layer.values, u.role.values, u.expert.values] / np.sqrt(n)
        u["relp_se"] = rsd
        u["relp_A"] = self.sA["v_relp_role"].mean(0)[u.layer.values, u.role.values, u.expert.values]
        en = self.pair_units("noise") if (self.p.direction == "noise").any() else None
        if en is not None and len(en):
            u = u.merge((en.groupby(["layer", "role", "expert"]).effect.sum() / n).rename("exact_noise").reset_index(), how="left",
                        on=["layer", "role", "expert"]).fillna({"exact_noise": 0.0})
        pz = self.b / "path/scores_path.npz"
        if pz.exists():
            z = np.load(pz)
            for v in z["variants"]:
                u[f"path_{v}"] = z[f"v_{v}_role"].mean(0)[u.layer.values, u.role.values, u.expert.values]
        u["band"] = D.band_of(u.layer.values, self.L)
        u["sig_exact"] = u.exact.abs() > 2 * u.exact_se
        u["sig_relp"] = u.relp.abs() > 2 * u.relp_se
        for c in ("exact", "relp"):
            u[f"rank_{c}"] = u[c].abs().rank(ascending=False, method="first").astype(int)
        return u, e


def analyse(base, summary):
    R = {k: [] for k in ("overview", "sites", "top", "agree", "depth", "rules", "posunits", "classes", "switches", "freeze", "noise",
                         "addback", "probe", "splithalf", "layerdist", "crosscheck")}
    for model in ("qwen3", "mixtral"):
        for task in ("counterfact", "twohop"):
            b = base / model / task
            if not (complete(b / "patch") and complete(b / "rules")):
                continue
            t = T3(base, model, task)
            key = dict(model=model, task=task)
            L, n = t.L, t.nd
            u, e = t.unit_table()
            u.to_parquet(b / "units.parquet", index=False)
            c = t.cases
            R["overview"].append({**key, "discovery_pairs": n, "heldout_pairs": int((t.rows.split != "discovery").sum()),
                                  "lengths": f"{t.rows['T'].min()}-{t.rows['T'].max()}", "accuracy_A": float((c.margin_A > 0).mean()),
                                  "accuracy_B_for_its_own_answer": float((c.margin_B < 0).mean()), "mean_F_A": float(c.margin_A.mean()),
                                  "mean_F_B": float(c.margin_B.mean()), "paired_difference": float((c.margin_A - c.margin_B).mean()),
                                  "role_units": len(u), "units_exact_nonzero": int(u.sig_exact.sum())})
            # ---- sites: whole-MoE-output patches by layer x position class (sum over the class's positions; absent = 0)
            pu = t.p[(t.p.kind == "position") & (t.p.direction == "denoise")]
            site = pu.groupby(["layer", "cls"]).effect.sum().unstack(fill_value=0.0) / n
            rcls = t.sB["v_relp_cls"].mean(0)                                           # [L, 4]
            diff = float((c.margin_A - c.margin_B).mean())
            for k_, cn in enumerate(CLS):
                if k_ not in site.columns:
                    continue
                x = site[k_].reindex(range(L), fill_value=0.0).values
                best = int(np.argmax(x))
                for (lo, hi, nm) in ((0, 0, "L0"), (1, 2, "L1-2"), (3, L - 9, "middle"), (L - 8, L - 1, "last8")):
                    R["sites"].append({**key, "position_class": cn, "layers": nm, "exact_sum": float(x[lo:hi + 1].sum()),
                                       "relp_sum": float(rcls[lo:hi + 1, k_].sum()), "exact_max_layer": best, "exact_max": float(x[best]),
                                       "exact_max_share_of_difference": float(x[best] / diff)})
            # ---- top units
            for meth, col in (("exact", "rank_exact"), ("RelP (B)", "rank_relp")):
                top = u.nsmallest(8, col)
                R["top"].append({**key, "method": meth, "top_units (value)": "; ".join(
                    f"{ename(r.layer, r.role, r.expert)} {(r.exact if meth == 'exact' else r.relp):+.2f}" for r in top.itertuples())})
            for K in (10, 20, 50):
                for meth, col in (("exact", "rank_exact"), ("RelP (B)", "rank_relp")):
                    top = u.nsmallest(K, col)
                    R["layerdist"].append({**key, "K": K, "method": meth, "last8": float((top.layer >= L - 8).mean()),
                                           "L0-2": float((top.layer <= 2).mean()), "final_role": float((top.role == 1).mean()),
                                           "median_layer": float(top.layer.median())})
            # ---- agreement
            sig = u[u.sig_exact | u.sig_relp]
            for scope, uu in (("all role units", u), ("final role", u[u.role == 1]), ("subject role", u[u.role == 0]),
                              ("final role, last 8 layers", u[(u.role == 1) & (u.layer >= L - 8)])):
                if len(uu) < 5:
                    continue
                s2 = uu[uu.sig_exact | uu.sig_relp]
                R["agree"].append({**key, "scope": scope, "units": len(uu), "pearson": pearson(uu.relp, uu.exact),
                                   "spearman_nonzero": spearman(s2.relp, s2.exact) if len(s2) > 4 else float("nan"),
                                   "slope_exact_on_relp": slope0(uu.relp, uu.exact),
                                   "top1_same": bool(uu.relp.abs().idxmax() == uu.exact.abs().idxmax()),
                                   "sum_exact": float(uu.exact.sum()), "sum_relp": float(uu.relp.sum())})
            ru, xu = u.sort_values("rank_relp"), u.sort_values("rank_exact")
            for K in (1, 5, 10, 20, 50, 100, 200):
                if K <= len(u):
                    a_, b_ = set(ru.head(K).index), set(xu.head(K).index)
                    R["depth"].append({**key, "K": K, "overlap": len(a_ & b_) / K})
            # split-half reliability of both rankings
            idx = {k_: i for i, k_ in enumerate(map(tuple, u[["layer", "role", "expert"]].values))}
            col = np.array([idx[k_] for k_ in map(tuple, e[["layer", "role", "expert"]].values)])
            MX = np.zeros((n, len(u)), np.float32)
            MX[e.row.values, col] = e.effect.values
            MR = t.sB["v_relp_role"][:, u.layer.values, u.role.values, u.expert.values]
            grp = t.rows.group_id.values[:n]
            groups = np.unique(grp)
            rng = np.random.default_rng(0)
            sh = {(m_, K): [] for m_ in ("RelP", "exact") for K in (10, 20, 50, 100)}
            for _ in range(100):
                gs = set(rng.permutation(groups)[: len(groups) // 2])
                h = np.array([g in gs for g in grp])
                for m_, M in (("RelP", MR), ("exact", MX)):
                    o1, o2 = np.argsort(-np.abs(M[h].mean(0))), np.argsort(-np.abs(M[~h].mean(0)))
                    for K in (10, 20, 50, 100):
                        sh[(m_, K)].append(len(set(o1[:K]) & set(o2[:K])) / K)
            for (m_, K), v in sh.items():
                R["splithalf"].append({**key, "method": m_, "K": K, "overlap": float(np.mean(v))})
            # ---- rule variants and path variants vs exact (role units; whole-MoE-output units at the two sites)
            vs = [str(v) for v in t.sB["variants"]] + [c_ for c_ in u.columns if c_.startswith("path_")]
            late = u[(u.role == 1) & (u.layer >= L - 8)]
            xs = site.reindex(range(L), fill_value=0.0)
            for v in vs:
                rk = set(u.reindex(u[v].abs().sort_values(ascending=False).index).head(20).index)
                R["rules"].append({**key, "variant": v, "pearson": pearson(u[v], u.exact), "slope_exact_on_score": slope0(u[v], u.exact),
                                   "top20_overlap_exact": len(rk & set(xu.head(20).index)) / 20,
                                   "pearson_final_last8": pearson(late[v], late.exact), "slope_final_last8": slope0(late[v], late.exact)})
                z = np.load(b / "path/scores_path.npz") if v.startswith("path_") else t.sB
                arr = z[f"v_{v[5:] if v.startswith('path_') else v}_cls"].mean(0)        # [L, 4]
                for (lo, hi, nm) in ((0, 0, "L0"), (0, 2, "L0-2"), (3, L - 9, "middle"), (L - 8, L - 1, "last8")):
                    for k_, cn in ((1, "subject"), (3, "final")):
                        ex = float(xs[k_].values[lo:hi + 1].sum()) if k_ in xs.columns else float("nan")
                        R["posunits"].append({**key, "variant": v, "layers": nm, "position_class": cn, "exact": ex,
                                              "score": float(arr[lo:hi + 1, k_].sum()),
                                              "ratio": float(arr[lo:hi + 1, k_].sum() / ex) if abs(ex) > 1e-6 else float("nan")})
            # ---- routing classes and route switches (single pair-units)
            for band in ("all", "last8"):
                eb = e if band == "all" else e[e.band == "last8"]
                for cl, g in eb.groupby("cls"):
                    if len(g) > 10:
                        R["classes"].append({**key, "band": band, "class": cl, "pair_units": len(g), "share_units": len(g) / len(eb),
                                             "pearson": pearson(g.relp, g.effect), "slope": slope0(g.relp, g.effect),
                                             "mean_abs_exact": float(g.effect.abs().mean())})
            e["q"] = pd.qcut(e.route_changes, 4, duplicates="drop")
            for qv, g in e.groupby("q", observed=True):
                R["switches"].append({**key, "route_changes": str(qv), "pair_units": len(g), "mean_abs_exact": float(g.effect.abs().mean()),
                                      "pearson": pearson(g.relp, g.effect)})
            # ---- cross-check against the Phase-1..6 engine (Direction 6 STR run): final-position expert rescues, same case and donor
            old = ROOT / "results" / {"qwen3": "qwen3_str", "mixtral": "mixtral_bos_str"}[model] / "str_expert_rows.parquet"
            if task == "counterfact" and old.exists():
                o = pd.read_parquet(old, columns=["case_id", "slot", "layer", "kind", "expert", "rescue"])
                o = o[o.kind.isin(["expert", "expert_corrupt_only", "layer"]) & (o.slot < 2)]
                o["row_id"] = "cf:" + o.case_id.astype(str) + ":" + o.slot.astype(str)
                o["kind"] = np.where(o.kind == "layer", "position", "expert")
                o["expert"] = np.where(o.kind == "position", -1, o.expert)
                allp = pd.read_parquet(t.b / "patch/patch.parquet", columns=["row", "direction", "kind", "layer", "role", "expert", "effect"])
                allp = allp[(allp.direction == "denoise") & (allp.role == 1)].merge(t.rows[["row", "row_id"]], on="row")
                j = allp.merge(o, on=["row_id", "kind", "layer", "expert"])
                for kind, g in j.groupby("kind"):
                    um = g.groupby(["layer", "expert"]).agg(n=("effect", "size"), new=("effect", "mean"), old=("rescue", "mean"))
                    big = um[um.n >= 30]
                    late = g[g.layer >= L - 8]
                    early = g[g.layer < L // 2]
                    R["crosscheck"].append({**key, "kind": kind, "pair_units": len(g),
                                            "pearson_pairs_first_half_layers": pearson(early.effect, early.rescue),
                                            "pearson_pairs_last8": pearson(late.effect, late.rescue),
                                            "units_30plus_pairs": len(big), "pearson_unit_means": pearson(big.new, big.old),
                                            "max_abs_unit_mean_diff": float((big.new - big.old).abs().max()),
                                            "top_unit_new_vs_old": "; ".join(f"L{l}E{e:03d} {r.new:+.3f} / {r.old:+.3f}" for (l, e), r in
                                                                             big.reindex(big.new.abs().sort_values(ascending=False).index).head(3).iterrows())})
            # ---- noise floor: the same role-position patches of the first rows in other batches
            if complete(b / "noise"):
                nz = pd.read_parquet(b / "noise/noise.parquet")
                ref = t.p[(t.p.direction == "denoise") & (t.p.role >= 0)][["row", "kind", "layer", "pos", "expert", "effect", "route_changes"]]
                j = nz.merge(ref, on=["row", "kind", "layer", "pos", "expert"], suffixes=("_rep", ""))
                j["band"] = D.band_of(j.layer.values, L)
                j["diff"] = j.effect_natural - j.effect
                for (kind, band), g in j.groupby(["kind", "band"]):
                    R["noise"].append({**key, "kind": kind, "band": band, "pair_units": len(g), "sd_diff": float(g["diff"].std()),
                                       "mean_abs_effect": float(g.effect.abs().mean()), "corr": pearson(g.effect_natural, g.effect)})
            # ---- freeze decomposition (role-level units, first rows of discovery)
            if complete(b / "freeze"):
                fz = pd.read_parquet(b / "freeze/freeze.parquet")
                spec = D.load_json(b / "freeze/units.json")
                nf = int(fz.row.nunique())
                effs = [c_ for c_ in fz.columns if c_.startswith("effect_")]
                g = fz.groupby("unit")[effs].sum() / nf
                sub = t.sB
                rows_f = np.sort(fz.row.unique())
                rr = {v: sub[f"v_{v}_role"][rows_f].mean(0) for v in ("relp", "relp_nohalf")}
                rec = []
                for ui, sp in enumerate(spec):
                    if ui not in g.index:
                        continue
                    l, r_, ex = sp["layer"], sp["role"], sp["expert"]
                    val = {v: float(rr[v][l, r_].sum() if sp["kind"] == "position" else rr[v][l, r_, ex]) for v in rr}
                    rec.append({"kind": sp["kind"], "layer": l, "role": r_, "expert": ex, **g.loc[ui].to_dict(), **val})
                U = pd.DataFrame(rec)
                U["gap"] = U.effect_natural - U.relp
                U["t_set"] = U.effect_natural - U.effect_Rset
                U["t_weights"] = U.effect_Rset - U.effect_R
                U["t_attention"] = U.effect_R - U.effect_RA
                U["t_normnl"] = U.effect_RA - U.relp_nohalf
                U["t_half"] = U.relp_nohalf - U.relp
                U["band"] = D.band_of(U.layer.values, L)
                U["model"], U["task"] = model, task
                U.to_parquet(b / "freeze_units.parquet", index=False)
                terms = ["t_set", "t_weights", "t_attention", "t_normnl", "t_half"]
                pos = U[U.kind == "position"]
                for (lo, hi, nm) in ((0, 0, "L0"), (1, 2, "L1-2"), (3, L - 9, "middle"), (L - 8, L - 1, "last8")):
                    for r_ in (0, 1):
                        s_ = pos[(pos.layer >= lo) & (pos.layer <= hi) & (pos.role == r_)]
                        R["freeze"].append({**key, "unit": f"MoE output, {nm} @ {ROLES[r_]}", "exact": float(s_.effect_natural.sum()),
                                            "relp": float(s_.relp.sum()), "gap": float(s_.gap.sum()), **{k_[2:]: float(s_[k_].sum()) for k_ in terms}})
                ex_ = U[U.kind == "expert"]
                for r_ in (0, 1):
                    s_ = ex_[ex_.role == r_]
                    for row in s_.reindex(s_.effect_natural.abs().sort_values(ascending=False).index).head(2).itertuples():
                        R["freeze"].append({**key, "unit": ename(row.layer, row.role, row.expert), "exact": row.effect_natural, "relp": row.relp,
                                            "gap": row.gap, **{k_[2:]: getattr(row, k_) for k_ in terms}})
            # ---- add-back
            if complete(b / "addback"):
                iv = pd.read_parquet(b / "addback/interventions.parquet")
                den = (iv.groupby("row_id").margin_A.first() - iv.groupby("row_id").margin_B.first()).mean()
                g = iv.groupby(["unit", "ranking", "seed", "budget", "intervention"])
                s_ = pd.DataFrame({"m": g.margin.mean(), "a": g.margin_A.mean(), "b": g.margin_B.mean()}).reset_index()
                s_["value"] = np.where(s_.intervention == "denoise", (s_.m - s_.b) / (s_.a - s_.b), (s_.m - s_.b) / (s_.a - s_.b))
                s_["model"], s_["task"] = model, task
                R["addback"].append(s_)
            # ---- probe
            if complete(b / "probe"):
                z = np.load(b / "probe.npz") if (b / "probe.npz").exists() else np.load(b / "probe/probe.npz")
                dsc = z["split"] == "discovery"
                for (lo, hi, nm) in ((0, 0, "L0"), (1, 2, "L1-2"), (3, L // 2 - 1, "early-mid"), (L // 2, L - 9, "mid-late"), (L - 8, L - 1, "last8")):
                    row = {**key, "layers": nm}
                    for q in ("jaccard", "tv_w", "dsig_mean", "lognorm2", "attn_tv", "resid_rel"):
                        for ri, rn in enumerate(ROLES):
                            row[f"{q}@{rn}"] = float(np.nanmean(z[q + "_role"][dsc][:, lo:hi + 1, ri]))
                    R["probe"].append(row)
    out = {k: (pd.concat(v) if k == "addback" and v else pd.DataFrame(v)) for k, v in R.items()}
    return out


def write(out):
    def W(name, title, blocks):
        text = f"# {title}\n\nGenerated by scripts/relp_moe_p8_analyze3.py (Phase 8 stage 3; docs/relp_moe_phase8_plan.md section 8d).\n\n"
        for head, df, nd, note in blocks:
            if df is None or len(df) == 0:
                continue
            text += f"## {head}\n\n" + (note + "\n\n" if note else "") + md_table(df, nd) + "\n"
        (TAB / f"relp_moe_p8s3_{name}.md").write_text(text)
    st = out["sites"]
    W("overview", "Stage 3 overview: tasks, sites and top units", [
        ("Tasks", out["overview"], 2, "accuracy_A = share of clean prompts with F > 0; F = logit(clean answer) - logit(corrupted run's answer)."),
        ("Exact whole-MoE-output patches by position class and layers (denoise, discovery mean; sum over the layers and the class's positions)", st, 2,
         "exact_max_layer / exact_max = the single best layer for that class; RelP = endpoint B."),
        ("Top role units", out["top"], 2, "subject = last token of the changed word; final = last token."),
        ("Layers of the top-K role units", out["layerdist"], 2, None)])
    W("agreement", "Stage 3: RelP vs exact patching on role units", [
        ("Agreement (unit means over discovery, endpoint B vs denoise)", out["agree"], 3, "nonzero = |mean| > 2 SE under at least one method."),
        ("Top-K overlap", out["depth"].pivot_table(index=["model", "task"], columns="K", values="overlap").reset_index() if len(out["depth"]) else None, 2, None),
        ("Split-half reliability (top-K overlap between halves of the discovery groups)",
         out["splithalf"].pivot_table(index=["model", "task", "method"], columns="K", values="overlap").reset_index() if len(out["splithalf"]) else None, 2, None),
        ("Routing classes (single pair-units)", out["classes"], 3, None),
        ("By later top-k sets switched by the patch (single pair-units)", out["switches"], 3, None),
        ("Noise floor: the same role-position patches in other batches", out["noise"], 3, None),
        ("Cross-check with the Phase-1..6 engine (Direction 6 STR run): final-position rescues of the same case and donor", out["crosscheck"], 3,
         "position = whole MoE output; expert = one expert's contribution (old engine: clean-active and corrupt-only experts). Unit means over the pairs both engines patched, units with >= 30 such pairs; single pairs in the first half of the layers are noise in both engines.")])
    pu = out["posunits"]
    W("rules", "Stage 3: backward-rule and multi-step variants vs exact patching", [
        ("Role units", out["rules"], 3, "path_* = 8-step path average of that rule."),
        ("Whole-MoE-output units: score / exact at the subject token (early layers) and at the final token (last 8 layers)",
         pu[((pu.position_class == "subject") & pu.layers.isin(["L0", "L0-2"])) | ((pu.position_class == "final") & (pu.layers == "last8"))]
         .pivot_table(index=["model", "task", "variant"], columns=["position_class", "layers"], values="ratio").reset_index() if len(pu) else None, 3, None),
        ("Whole-MoE-output units, all", pu, 3, None)])
    W("freeze", "Stage 3: which frozen quantity causes RelP's error (denoise, first 100 discovery pairs)", [
        ("Role-level units", out["freeze"], 3, "gap = exact - RelP = set + weights + attention + normnl + half (set / weights: later top-k set switches / routing weights; attention: later attention patterns; normnl: later norm denominators + expert nonlinearity + higher order; half: the half rule).")])
    ab = out["addback"]
    blocks = []
    if len(ab):
        ab = ab.assign(ranking=np.where(ab.ranking == "random", "random (3 seeds)", ab.ranking),
                       budget=np.where(ab.ranking == "all", "all", ab.budget.astype(str)))
        for unit, ul in (("role", "role units (layer, role, expert)"), ("all", "whole experts (all positions)")):
            for iv_, lab in (("denoise", "faithfulness: share of the paired difference restored when only these units carry the clean run"),
                             ("noise", "completeness: share left when these units carry the corrupted run (ideal 0)")):
                x = ab[(ab.unit == unit) & (ab.intervention == iv_)].groupby(["model", "task", "ranking", "budget"]).value.mean().reset_index()
                pv = x.pivot_table(index=["model", "task", "ranking"], columns="budget", values="value")
                order = sorted([c_ for c_ in pv.columns if c_ != "all"], key=int) + (["all"] if "all" in pv.columns else [])
                blocks.append((f"{ul}: {lab} (validation + test)", pv[order].reset_index(), 2, None))
    W("addback", "Stage 3: experts added back vs faithfulness and completeness (held-out pairs)", blocks)
    W("probe", "Stage 3: clean vs corrupted run at the two roles (discovery means)", [
        ("Gate probe", out["probe"], 3, "jaccard = top-k set overlap; tv_w = total variation of routing weights; dsig_mean = mean |change| of the SiLU gate; lognorm2 = |log| ratio of the MoE-input RMS; attn_tv = total variation of the attention pattern; resid_rel = ||h_A - h_B|| / ||h_B||.")])




def cross_task(base):
    """One row per (model, task) over SVA (stage 2 outputs) and the stage-3 tasks: late-unit agreement, the early subject site and
    its causes, held-out faithfulness at K = 20 under each ranking."""
    P8 = ROOT / "results/relp_sva_moe/p8"
    rows = []
    # ---- SVA (four tasks, mean)
    fz = pd.read_parquet(P8 / "s23_freeze_units.parquet")
    ab = pd.read_parquet(P8 / "s21_addback.parquet")
    ab = ab[(ab.split == "all") & (ab.unit == "expert_position") & (ab.intervention == "denoise") & (ab.budget == 20)]
    for model in ("qwen3", "mixtral"):
        L = NL[model]
        r_, s_, t1 = [], [], []
        for task in D.TASKS:
            u = pd.read_parquet(P8 / f"units_{model}_{task}.parquet")
            late = u[(u.layer >= L - 8) & u.pos_class.isin(["final", "subject=final"])]
            r_.append(pearson(late.relp_B, late.exact_denoise))
            s_.append(slope0(late.relp_B, late.exact_denoise))
            t1.append(bool(u.relp_B.abs().idxmax() == u.exact_denoise.abs().idxmax()))
        f0 = fz[(fz.model == model) & (fz.kind == "position") & (fz.layer == 0) & fz.pos_class.str.startswith("subject")]
        g = f0.groupby("task")[["effect_natural", "relp", "gap", "t_set", "t_weights", "t_attention", "t_normnl", "t_half"]].sum().mean()
        a = ab[ab.model == model]
        k20 = {rk: float(a[a.ranking == rk].groupby("task").normalized.mean().mean()) for rk in ("relp", "exact", "lu")}
        pg = a[a.ranking == "path_grad_rw"]
        rows.append({"model": model, "task": "SVA (4 tasks)", "paired_difference": float("nan"),
                     "late_final_pearson": float(np.mean(r_)), "late_final_slope": float(np.mean(s_)), "top1_same": f"{sum(t1)}/4",
                     "L0_subject_exact": float(g.effect_natural), "L0_subject_relp_share": float(g.relp / g.effect_natural),
                     "gap_routing_share": float((g.t_set + g.t_weights) / g.gap), "gap_attention_share": float(g.t_attention / g.gap),
                     "gap_nonlin_share": float(g.t_normnl / g.gap), "gap_half_share": float(g.t_half / g.gap),
                     "faith20_relp": k20["relp"], "faith20_exact": k20["exact"], "faith20_lu": k20["lu"],
                     "faith20_path_grad": float(pg.groupby("task").normalized.mean().mean()) if len(pg) else float("nan")})
    # ---- stage 3
    A = base / "analysis"
    if (A / "agree.parquet").exists():
        ag, ov = pd.read_parquet(A / "agree.parquet"), pd.read_parquet(A / "overview.parquet")
        fr = pd.read_parquet(A / "freeze.parquet") if (A / "freeze.parquet").exists() else pd.DataFrame()
        ad = pd.read_parquet(A / "addback.parquet") if (A / "addback.parquet").exists() else pd.DataFrame()
        for r in ov.itertuples():
            a_ = ag[(ag.model == r.model) & (ag.task == r.task)]
            late = a_[a_.scope == "final role, last 8 layers"].iloc[0]
            allu = a_[a_.scope == "all role units"].iloc[0]
            row = {"model": r.model, "task": TASK_NAME[r.task], "paired_difference": r.paired_difference,
                   "late_final_pearson": late.pearson, "late_final_slope": late.slope_exact_on_relp, "top1_same": "1/1" if allu.top1_same else "0/1"}
            if len(fr):
                f_ = fr[(fr.model == r.model) & (fr.task == r.task) & fr.unit.isin(["MoE output, L0 @ subject", "MoE output, L1-2 @ subject"])]
                if len(f_):
                    g = f_[["exact", "relp", "gap", "set", "weights", "attention", "normnl", "half"]].sum()
                    f0 = f_[f_.unit == "MoE output, L0 @ subject"].iloc[0]
                    row.update(L0_subject_exact=float(f0.exact), L0_subject_relp_share=float(f0.relp / f0.exact),
                               **{"L0-2_subject_exact": float(g.exact), "L0-2_subject_relp_share": float(g.relp / g.exact)},
                               gap_routing_share=float((g.set + g.weights) / g.gap), gap_attention_share=float(g.attention / g.gap),
                               gap_nonlin_share=float(g.normnl / g.gap), gap_half_share=float(g.half / g.gap))
            if len(ad):
                x = ad[(ad.model == r.model) & (ad.task == r.task) & (ad.unit == "role") & (ad.intervention == "denoise") & (ad.budget == 20)]
                for rk, nm in (("relp", "faith20_relp"), ("exact", "faith20_exact"), ("lu", "faith20_lu"), ("path_grad_rw", "faith20_path_grad")):
                    v = x[x.ranking == rk].value
                    row[nm] = float(v.mean()) if len(v) else float("nan")
            rows.append(row)
    T = pd.DataFrame(rows)
    T["gap_linearisation_share"] = T.gap_nonlin_share + T.gap_half_share
    T.to_parquet(base / "analysis" / "tasks.parquet", index=False)
    cols = [c for c in ["model", "task", "paired_difference", "late_final_pearson", "late_final_slope", "top1_same", "L0_subject_exact",
                        "L0_subject_relp_share", "L0-2_subject_exact", "L0-2_subject_relp_share", "gap_routing_share", "gap_attention_share",
                        "gap_linearisation_share", "gap_nonlin_share", "gap_half_share", "faith20_relp", "faith20_exact", "faith20_path_grad", "faith20_lu"] if c in T.columns]
    text = ("# Phase 8: SVA, CounterFact and two-hop capitals side by side\n\nGenerated by scripts/relp_moe_p8_analyze3.py.\n\n"
            "late_final_* = RelP (endpoint B) vs exact denoise over final-token expert units of the last 8 layers; L0_subject_* = whole-MoE-output "
            "patch at the (last) subject token in layer 0; gap_*_share = share of exact - RelP at the early subject site (SVA: layer 0; stage 3: "
            "layers 0-2) from later routing (set switches + weights), later attention patterns, and the linearisation of the remaining model (= later norms + expert "
            "nonlinearity + higher order, plus the half rule; the two parts are also listed and can have opposite signs); routing + attention + "
            "linearisation = 1; faith20_* = held-out share of the paired difference restored "
            "by the top 20 units of each ranking (SVA: expert-positions; stage 3: role units).\n\n" + md_table(T[cols], 2))
    (TAB / "relp_moe_p8s3_tasks.md").write_text(text)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=ROOT / "results/relp_p8")
    a = ap.parse_args()
    out = analyse(a.base, {})
    write(out)
    (a.base / "analysis").mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        if len(v):
            v.to_parquet(a.base / "analysis" / f"{k}.parquet", index=False)
    (a.base / "summary.json").write_text(json.dumps({k: v.to_dict(orient="records") for k, v in out.items() if k != "addback" and len(v)},
                                                     indent=1, default=float) + "\n")
    cross_task(a.base)
    print("wrote", a.base / "summary.json")


if __name__ == "__main__":
    main()
