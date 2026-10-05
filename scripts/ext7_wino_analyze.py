"""ext7 WinoGrande STR analysis (CPU only): W2 final-position sweep, W6 experts, W3 position x layer grid, W5 heads,
W4 joint sublayer decomposition, strata and robustness. Every statistic on pair means (mean over the two directions),
bootstrap CIs resample pairs (5,000, seed 0); expert recurrence / Spec on directed cases (moetrace.ext7_pairs).

Runs: results/wino_qwen3_str, results/wino_mixtral_bos_str. Missing stages are skipped.
Usage: python scripts/ext7_wino_analyze.py
Outputs results/ext7_wino_summary.json, results/tables/ext7_wino_*.{md,csv}, results/figures/ext7_wino_*.{png,pdf}
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext7_pairs as P
from moetrace import analysis as A
from moetrace import ext1_analysis as X1
from moetrace.models import MODELS

ROOT = "/home/ubuntu/MOE/results"
TAB, FIG = os.path.join(ROOT, "tables"), os.path.join(ROOT, "figures")
RUNS = [
    {"run": "wino_qwen3_str", "model": "qwen3", "label": "Qwen3-30B-A3B-Base", "scan": "wino_qwen3", "cf": "qwen3_str",
     "cf_experts": [(44, 69), (42, 115)]},
    {"run": "wino_mixtral_bos_str", "model": "mixtral", "label": "Mixtral-8x7B, BOS", "scan": "wino_mixtral_bos", "cf": "mixtral_bos_str",
     "cf_experts": [(19, 2), (21, 1), (18, 1), (19, 6)]},
]
KIND_LABEL = {"layer": "MoE output", "attn_layer": "attention output", "block": "attention + MoE (block)", "resid": "residual (hidden state)"}
STRATA = [("assoc", "context-free association passes (assoc)"), ("names", "person-name options"), ("top1_both", "top-1 in both directions"),
          ("debiased", "AfLite survivor (train_debiased)"), ("one_token_option", "one-token option"), ("trigger_in_context", "trigger word in context")]
F = P.fmt


def md_table(header, rows, name, caption=None):
    lines = ([f"**{caption}**", ""] if caption else []) + ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(TAB, name + ".md"), "w") as f:
        f.write(txt)
    pd.DataFrame(rows, columns=header).to_csv(os.path.join(TAB, name + ".csv"), index=False)
    return txt


def clean(o):
    """JSON-safe copy (drops DataFrames)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------------------------------------------------------
def load_run(cfg):
    run = P.PairRun(cfg["run"])
    sc = pd.read_parquet(os.path.join(ROOT, cfg["scan"], "scan_pairs.parquet"))
    sc = sc.set_index("pair_id")
    c = run.cases
    for k in ("assoc", "top1_both", "d_local_a", "d_local_b"):
        c[k] = c.pair_id.map(sc[k])
    c["one_token_option"] = c.n_opt_tokens == 1 if "n_opt_tokens" in c else np.nan
    return run


def descriptors(run, cfg):
    out = {}
    for f, sp in run.fam.items():
        ids = [x for v in sp.values() for x in v]
        c = run.cases.loc[ids]
        out[f] = {"n_pairs": len(ids) // 2, "n_directed": len(ids), "delta_clean": float(c.delta_clean.mean()),
                  "delta_corrupt": float(c.delta_corrupt.mean()), "drop": P.summ(c["drop"], with_p=False),
                  "clean_top1_rate": float((c.top1_clean == c.true_id).mean()), "corrupt_top1_foil_rate": float((c.top1_corrupt == c.foil_id).mean()),
                  "p_true_clean_median": float(c.p_true_clean.median()) if "p_true_clean" in c else None,
                  "names": float(c.names.mean()), "assoc": float(c.assoc.mean()), "top1_both": float(c.top1_both.mean()),
                  "debiased": float(c.debiased.mean()), "one_token_option": float(c.one_token_option.mean()),
                  "trigger_in_context": float(c.trigger_in_context.mean()), "T_median": float(c["T"].median())}
    return out


# ---------------------------------------------------------------------------------------------------------------
# W2
# ---------------------------------------------------------------------------------------------------------------
def w2(run, cfg):
    s = {"peaks": {}, "share": {}, "curves": {}, "share_at_peaks": {}, "additivity": {}}
    kinds = [k for k in ("layer", "attn_layer", "block") if k in run.kinds]
    for f in run.fam:
        if "discovery" not in run.fam[f]:
            continue
        s["peaks"][f] = P.sweep_peaks(run, f, kinds)
        val = run.ids(f, "validation")
        s["share"][f] = P.attention_share(run, val)
        s["share"][f + "_all"] = P.attention_share(run, run.ids(f, "discovery") + val)
    val = run.ids("main", "validation")
    cur = P.sweep_curves(run, val, kinds)
    s["curves"]["main_val"] = cur
    # interior-layer rule (<= L-5)
    s["peaks"]["main_interior"] = P.sweep_peaks(run, "main", kinds, max_layer=run.L - 5)
    # direction A->B only (d = 0) and B->A only
    for dd in (0, 1):
        v = [c for c in val if c % 2 == dd]
        s["share"][f"main_d{dd}"] = P.attention_share(run, v)
        s[f"peaks_d{dd}"] = {k: {"L": int(run.mat(k, [c for c in run.ids('main', 'discovery') if c % 2 == dd]).mean(0).idxmax()),
                                "val_at_main_L": P.summ(run.mat(k, v)[[p for p in s['peaks']['main'] if p['kind'] == k][0]['L_disc']], with_p=False)}
                            for k in kinds}
    for p in s["peaks"]["main"]:
        if p["kind"] in ("layer", "attn_layer"):
            s["share_at_peaks"][p["kind"]] = P.share_at_layer(run, p["L_disc"], val)
    add = P.additivity(run, val)
    s["additivity"]["table"] = add
    s["additivity"]["mean_abs_gap"] = float(add.gap.abs().mean())
    top = add.reindex(add.block.abs().sort_values(ascending=False).index).head(5)
    s["additivity"]["top_block_layers"] = top.to_dict("records")
    s["additivity"]["r_pair_median"] = float(add.r_pair.median())
    # strata (all 256 main pairs; layer fixed by the main discovery argmax)
    ids_all = run.ids("main", "discovery") + val
    pk = {p["kind"]: p["L_disc"] for p in s["peaks"]["main"]}
    st = []
    for col, lab in STRATA:
        for v in (True, False):
            ids = [c for c in ids_all if bool(run.cases.at[c, col]) == v]
            if len(ids) < 20:
                st.append({"stratum": col, "label": lab, "value": v, "n_pairs": len(ids) // 2})
                continue
            sh = P.attention_share(run, ids)
            row = {"stratum": col, "label": lab, "value": v, "n_pairs": len(ids) // 2, "drop": float(run.cases.loc[ids, "drop"].mean()), "share": sh}
            for k in kinds:
                row[f"norm_{k}"] = P.ratio(run.mat(k, ids)[pk[k]], run.drop.loc[ids])
                row[f"norm_peak_{k}"] = float((run.mat(k, ids).mean(0) / run.drop.loc[ids].mean()).max())
            st.append(row)
    s["strata"] = st
    return s


def cf_curve(cfg):
    """CounterFact STR (ext6) MoE-layer curve on its validation cases, normalised by the mean drop (donor mean)."""
    p = os.path.join(ROOT, cfg["cf"], "str_sweep_rows.parquet")
    if not os.path.exists(p):
        return None
    sw = pd.read_parquet(p)
    sets = json.load(open(os.path.join(ROOT, cfg["cf"], "case_sets.json")))
    R = sw[sw.kind == "layer"].groupby(["case_id", "layer"]).rescue.mean().unstack()
    cases = pd.read_parquet(os.path.join(ROOT, cfg["cf"], "sweep_cases.parquet")).set_index("case_id")
    val = [c for c in sets["paper"]["validation"] if c in R.index]
    return (R.loc[val].mean(0) / cases.loc[val, "drop"].mean()).values


def cf_attn_curves(cfg):
    """CounterFact STR attention / MoE / block sweep of ext7-controls (results/<cf>_attnsweep/str_attn_rows.parquet, donor
    level) on its validation cases: donor-mean curves, normalised by the mean drop; AUC+ attention share (same definition)."""
    d = os.path.join(ROOT, cfg["cf"] + "_attnsweep")
    p = os.path.join(d, "str_attn_rows.parquet")
    if not os.path.exists(p):
        return None
    try:
        sw = pd.read_parquet(p)
        cs = json.load(open(os.path.join(d, "case_sets.json")))
        cases = pd.read_parquet(os.path.join(d, "sweep_cases.parquet")).set_index("case_id")
        out = {"n_val": 0}
        for k in ("layer", "attn_layer", "block"):
            g = sw[sw.kind == k].groupby(["case_id", "layer"]).rescue.mean().unstack()
            val = [c for c in cs["paper"]["validation"] if c in g.index]
            out["n_val"] = len(val)
            out[k] = (g.loc[val].mean(0) / cases.loc[val, "drop"].mean()).to_numpy()
        A_, M_ = np.clip(out["attn_layer"], 0, None).sum(), np.clip(out["layer"], 0, None).sum()
        out["share"] = float(A_ / (A_ + M_))
        for k in ("layer", "attn_layer", "block"):
            out[f"peak_{k}"] = [int(np.argmax(out[k])), float(np.max(out[k]))]
        out["auc_attn_norm"], out["auc_moe_norm"] = float(A_), float(M_)
        return out
    except Exception as e:  # format owned by another agent
        return {"error": repr(e)}


# ---------------------------------------------------------------------------------------------------------------
# W6
# ---------------------------------------------------------------------------------------------------------------
def w6(run, cfg, s2):
    p = os.path.join(run.dir, "str_expert_rows.parquet")
    if not os.path.exists(p):
        return None
    md = P.model_data(cfg["run"])
    cache = X1.ExpertCache(md)
    nc = MODELS[cfg["model"]]["n_controls"]
    out = {}
    for f in ("main", "rep"):
        if f not in md.sets:
            continue
        disc, val = md.sets[f]["discovery"], md.sets[f]["validation"]
        if not set(disc) <= set(md.expert_rows.case_id):
            continue
        th = len(disc) // 2
        lstar = int(md.R.loc[disc].mean(0).idxmax())
        ent = {"L_star": lstar, "threshold": th}
        for tag, L_ in (("two_stage", lstar), ("interior", int(md.R.loc[disc].mean(0)[lambda x: x.index <= run.L - 5].idxmax()))):
            et = A.expert_table(md, L_)
            sel = A.select_expert(et, disc, th)
            e = sel["e_star"]
            r = {"layer": L_, "selection": {k: v for k, v in sel.items()}, "layer_val": P.summ(md.R.loc[val, L_]),
                 "layer_norm": P.ratio(md.R.loc[val, L_], md.cases.set_index("case_id")["drop"].loc[val]),
                 "coalitions": P.coalitions_pairs(md, L_, val)}
            if e is not None:
                ev = P.evaluate_expert_pairs(md, L_, e, val, nc)
                r["eval"] = ev
                r["pattern"] = P.pattern(r["layer_val"], ev["spec"], ev["rescue"])
                if L_ == lstar and f == "main":
                    try:
                        if nc == 3:
                            g = A.gate_matched_control(md, L_, e, val)
                        else:
                            g = A.active_pair_equal_norm(md, L_, e, val)
                        pc = g["per_case"].set_index("case_id")
                        eq = {"n": g["n"], "kind": "gate_matched" if nc == 3 else "active_pair"}
                        if nc == 3:
                            eq.update(spec_raw=P.summ(pc.sel_raw - pc.ctrl_raw), sel_eq=P.summ(pc.sel_eq), ctrl_eq=P.summ(pc.ctrl_eq),
                                      spec_eq=P.summ(pc.sel_eq - pc.ctrl_eq))
                        else:
                            eq.update(spec_raw=P.summ(pc.sel_raw - pc.other_raw), sel_eq=P.summ(pc.sel_eq), ctrl_eq=P.summ(pc.other_eq),
                                      spec_eq=P.summ(pc.sel_eq - pc.other_eq))
                        r["equal_norm"] = eq
                    except Exception as ex:
                        r["equal_norm"] = {"error": repr(ex)}
                    rk = A.all_active_rank(md, L_, e, val)
                    r["all_active_rank"] = {k: v for k, v in rk.items() if k != "per_case"}
            else:
                r["pattern"] = P.pattern(r["layer_val"], None, None)
            ent[tag] = r
        # joint search (all layers, recurrence gate half of discovery)
        cands = X1.joint_candidates(cache, disc, th)
        top = []
        for _, c in cands.head(8).iterrows():
            ev = P.evaluate_expert_pairs(md, int(c.layer), int(c.expert), val, nc)
            top.append({"rank": int(c["rank"]), "layer": int(c.layer), "expert": int(c.expert), "disc_active": int(c.disc_active),
                        "disc_allcase_mean": float(c.disc_allcase_mean), "val_rescue": ev["rescue"], "val_spec": ev["spec"], "val_active": ev["val_active"]})
        ent["joint_top"] = top
        ent["n_joint_candidates"] = int(len(cands))
        # per-layer best expert concentration (validation): best recurrent expert / layer rescue
        if f == "main":
            fixed = []
            for (l, e) in cfg["cf_experts"]:
                if l in cache.layers:
                    ev = P.evaluate_expert_pairs(md, l, e, val, nc)
                    st = cache.disc_stats(l, disc)
                    da = int(st[st.expert == e].disc_active.iloc[0]) if (st.expert == e).any() else 0
                    fixed.append({"layer": l, "expert": e, "disc_active": da, "val_active": ev["val_active"], "rescue": ev["rescue"], "spec": ev["spec"],
                                  "rescue_norm": ev["rescue_norm"]})
            ent["cf_experts"] = fixed
            # direction A->B only: selection on d=0 discovery, evaluation on d=0 validation
            d0d, d0v = [c for c in disc if c % 2 == 0], [c for c in val if c % 2 == 0]
            l0 = int(md.R.loc[d0d].mean(0).idxmax())
            sel0 = A.select_expert(A.expert_table(md, l0), d0d, len(d0d) // 2)
            ent["d0"] = {"L_star": l0, "e_star": sel0["e_star"]}
            if sel0["e_star"] is not None:
                ev0 = A.evaluate_expert(md, l0, sel0["e_star"], d0v, nc)
                pc0 = ev0["per_case"]
                ent["d0"].update(rescue=P.summ(pc0.set_index("case_id").rescue), spec=P.summ(pc0.set_index("case_id").spec.dropna()))
        out[f] = ent
    # strata at the main two-stage expert (all 256 main pairs; descriptive)
    m = out.get("main", {}).get("two_stage")
    if m and m.get("eval"):
        L_, e = m["layer"], m["eval"]["expert"]
        ids_all = md.sets["main"]["discovery"] + md.sets["main"]["validation"]
        st = []
        for col, lab in STRATA:
            for v in (True, False):
                ids = [c for c in ids_all if bool(run.cases.at[c, col]) == v]
                if len(ids) < 20:
                    continue
                ev = P.evaluate_expert_pairs(md, L_, e, ids, nc)
                st.append({"stratum": col, "value": v, "n_pairs": len(ids) // 2, "rescue": ev["rescue"], "spec": ev["spec"],
                           "layer": P.summ(md.R.loc[ids, L_], with_p=False)})
        out["strata"] = st
    return out


# ---------------------------------------------------------------------------------------------------------------
# W3 grid
# ---------------------------------------------------------------------------------------------------------------
GROUPS = ("first STR token", "middle STR tokens", "last STR token", "first subsequent token", "further tokens", "last token")
GSHORT = {"first STR token": "first option tok.", "middle STR tokens": "middle option tok.", "last STR token": "last option tok.",
          "first subsequent token": "first subseq.", "further tokens": "further", "last token": "final token"}


def grid(run, cfg):
    out = {}
    for w in (1, 5):
        p = os.path.join(run.dir, f"str_grid_w{w}_rows.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        df["drop"] = df.delta_clean - df.delta_corrupt
        L = int(df.layer.max()) + 1
        ent = {"n_rows": len(df), "n_cases": int(df.case_id.nunique()), "kinds": {}}
        drop = df.groupby("case_id")["drop"].first()
        for k, g in df.groupby("kind"):
            cc = g.groupby(["case_id", "cat", "layer"], as_index=False)[["rescue", "dp"]].mean()
            kk = {}
            for cat in GROUPS:
                sub = cc[cc.cat == cat]
                if sub.empty:
                    continue
                M = sub.pivot(index="case_id", columns="layer", values="rescue").reindex(columns=range(L))
                Dp = sub.pivot(index="case_id", columns="layer", values="dp").reindex(columns=range(L))
                nc = P.norm_curve(M, drop)
                l = int(nc.norm.idxmax())
                ids = list(M.index)
                kk[cat] = {"n_dir": len(ids), "n_pairs": len(set(c // 2 for c in ids)), "curve_norm": nc.norm.round(5).tolist(),
                           "curve_dp": Dp.mean(0).round(5).tolist(), "peak_layer": l, "peak": [float(nc.norm[l]), float(nc.norm_lo[l]), float(nc.norm_hi[l])],
                           "sum_over_layers_norm": float(nc.norm.sum()), "peak_dp": float(Dp.mean(0).max()), "peak_dp_layer": int(Dp.mean(0).idxmax())}
            ent["kinds"][k] = kk
        # consistency: final-token column vs final-position sweep (window 1, kind layer / attn_layer)
        if w == 1:
            cons = {}
            for k in ("layer", "attn_layer"):
                if k not in run.kinds:
                    continue
                g = df[(df.kind == k) & (df.cat == "last token")].pivot(index="case_id", columns="layer", values="rescue")
                sw = run.R[k].loc[g.index, g.columns]
                cons[k] = {"curve_r": float(np.corrcoef(g.mean(0), sw.mean(0))[0, 1]), "case_r": float(np.corrcoef(g.values.ravel(), sw.values.ravel())[0, 1]),
                           "max_abs_mean_diff": float((g.mean(0) - sw.mean(0)).abs().max())}
            ent["consistency_vs_sweep"] = cons
        if w == 1 and "resid" in ent["kinds"]:  # hand-off of the option information from the STR site to the final position
            rk = ent["kinds"]["resid"]
            ho = {}
            if "last STR token" in rk:
                c = np.array(rk["last STR token"]["curve_norm"])
                ho["str_last_layer_ge_0.5"] = int(np.nonzero(c >= 0.5)[0].max()) if (c >= 0.5).any() else None
                ho["str_first_layer_le_0.1"] = int(np.nonzero(c <= 0.1)[0].min()) if (c <= 0.1).any() else None
            if "last token" in rk:
                c = np.array(rk["last token"]["curve_norm"])
                ho["final_first_layer_ge_0.5"] = int(np.nonzero(c >= 0.5)[0].min()) if (c >= 0.5).any() else None
                ho["final_first_layer_ge_0.1"] = int(np.nonzero(c >= 0.1)[0].min()) if (c >= 0.1).any() else None
                ho["final_first_layer_ge_0.9"] = int(np.nonzero(c >= 0.9)[0].min()) if (c >= 0.9).any() else None
            for cat in ("first subsequent token", "further tokens"):
                if cat in rk:
                    ho[f"max_{cat.replace(' ', '_')}"] = [rk[cat]["peak_layer"], rk[cat]["peak"][0]]
            ent["handoff"] = ho
        out[f"w{w}"] = ent
    if "w1" in out and "w5" in out and "layer" in out["w5"]["kinds"]:
        sv = []
        for cat, x5 in out["w5"]["kinds"]["layer"].items():
            x1 = out["w1"]["kinds"]["layer"].get(cat)
            if not x1:
                continue
            c1 = np.array(x1["curve_norm"])
            L = len(c1)
            add = np.array([c1[max(0, l - 2): min(L, l + 3)].sum() for l in range(L)])
            c5 = np.array(x5["curve_norm"])
            sv.append({"cat": cat, "sliding_peak": float(c5.max()), "sliding_layer": int(c5.argmax()), "adding_peak": float(add.max()),
                       "adding_layer": int(add.argmax()), "ratio": float(c5.max() / add.max()) if add.max() > 0 else float("nan")})
        out["sliding_vs_adding"] = sv
    return out


# ---------------------------------------------------------------------------------------------------------------
# W5 heads
# ---------------------------------------------------------------------------------------------------------------
def heads(run, cfg):
    if not os.path.exists(os.path.join(run.dir, "head_rows.parquet")):
        return None
    hr = P.HeadRun(cfg["run"])
    disc, val = run.ids("main", "discovery"), run.ids("main", "validation")
    tab = P.head_table(hr, disc, val)
    meta = run.meta.get("heads", {})
    out = {"layers": hr.layers, "null_layer": meta.get("null_layer"), "table": tab, "layer_summary": [P.head_layer_summary(hr, l, disc, val) for l in hr.layers]}
    det = tab[tab.detected_2sd]
    out["detected_2sd"] = det[["layer", "head", "val_mean", "val_lo", "val_hi", "z_val", "z_disc", "spec", "share_attn_layer"]].to_dict("records")
    out["detected_2sd_val_only"] = int((tab.z_val.abs() >= 2).sum())
    out["n_heads_scanned"] = int(len(tab))
    # attention mass of the top heads (by validation mean; up to 6 with positive mean) clean vs corrupted
    topk = tab[tab.val_mean > 0].head(6)
    am = []
    for _, r in topk.iterrows():
        l, h = int(r.layer), int(r["head"])
        oc, on = P.attention_mass(hr, l, val)
        row = {"layer": l, "head": h, "val_mean": float(r.val_mean)}
        for k, nm in enumerate(P.HEAD_CLASSES):
            row[f"clean_{nm}"] = float(oc[:, h, k].mean())
            row[f"corrupt_{nm}"] = float(on[:, h, k].mean())
        am.append(row)
    # layer-average mass and the head-level correlation of rescue with clean mass on the filled option / mentions
    corr = []
    from scipy.stats import spearmanr
    for l in hr.layers:
        oc, on = P.attention_mass(hr, l, val)
        mv = P.matrix_pairs(hr.R[l].loc[val]).mean(0).to_numpy()
        ent = {"layer": l}
        for k, nm in enumerate(P.HEAD_CLASSES):
            ent[f"rho_rescue_vs_clean_{nm}"] = float(spearmanr(mv, oc[:, :, k].mean(0)).correlation)
            ent[f"layer_mean_clean_{nm}"] = float(oc[:, :, k].mean())
        corr.append(ent)
    out["attention_mass_top"] = am
    out["rescue_vs_mass"] = corr
    return out


# ---------------------------------------------------------------------------------------------------------------
# W4 joint
# ---------------------------------------------------------------------------------------------------------------
def _w4_block(run, df, ds, ids):
    """Revised W4 (coordinator, 2026-10-04): A = all-attention patch only as a sanity check of the multi path; M = all-MoE
    patch, denoising (corrupted run) and noising (clean run, MoE outputs set to the corrupted values = necessity), both /
    drop; direct-path split from ext7-controls' shared direct_split_pairs (= ext8 direct_split): A_direct, M_direct (exact
    final norm) and the linear DLA split. Population ratios with pair bootstrap."""
    out = {"n_pairs": len({c // 2 for c in ids}), "n_directed": len(ids)}
    if df is not None:
        sub = df[df.case_id.isin(ids) & (df.span == "all")]
        for direction in ("denoise", "noise"):
            g = sub[sub.direction == direction]
            piv = g.pivot(index="case_id", columns="kind", values="effect")
            drop = g.groupby("case_id")["drop"].first()
            out[f"M_{direction}"] = P.ratio(piv["all_moe"], drop)
            out[f"A_sanity_{direction}"] = P.ratio(piv["all_attn"], drop)
            out[f"block_sanity_{direction}"] = P.ratio(piv["all_block"], drop)
        # mirror identity: noising of direction d == denoising of direction 1 - d (same intervention, metric sign-flipped)
        den = sub[(sub.direction == "denoise") & (sub.kind == "all_moe")].set_index("case_id").effect
        noi = sub[(sub.direction == "noise") & (sub.kind == "all_moe")].set_index("case_id").effect
        mirror = [abs(den[c] - noi[c ^ 1]) for c in den.index if (c ^ 1) in noi.index]
        out["mirror_max_abs"] = float(max(mirror)) if mirror else None
    if ds is not None:
        x = ds.loc[[c for c in ids if c in ds.index]]
        drop = x.d_clean_fp32 - x.d_corrupt_fp32
        out["direct"] = {"A_direct": P.ratio(x.d_attn_direct - x.d_corrupt_fp32, drop), "M_direct": P.ratio(x.d_moe_direct - x.d_corrupt_fp32, drop),
                         "phi_attn_direct": P.ratio(0.5 * ((x.d_attn_direct - x.d_corrupt_fp32) + (x.d_clean_fp32 - x.d_moe_direct)), drop)}
        out["dla"] = {"attn": P.ratio(x.dla_attn, x.dla_total), "moe": P.ratio(x.dla_moe, x.dla_total)}
        out["drop_fp32"] = float(drop.mean())
        out["fp32_vs_engine_max_abs"] = float((x.d_clean_fp32 - x.delta_clean_engine).abs().max())
    return out


def joint(run, cfg):
    jp, dp = os.path.join(run.dir, "joint_rows.parquet"), os.path.join(run.dir, "direct_split.parquet")
    df = pd.read_parquet(jp) if os.path.exists(jp) else None
    ds = pd.read_parquet(dp).set_index("case_id", drop=False) if os.path.exists(dp) else None
    if df is None and ds is None:
        return None
    out = {"definition": "A = all final-position attention outputs patched (sanity: = 1 by construction, MoE is per token and the final token "
                         "is shared); M_denoise = all final-position MoE outputs := clean in the corrupted run; M_noise = := corrupted in the "
                         "clean run; both / drop. direct = ext7_controls.direct_split_pairs (= ext8 direct_split): A_direct = "
                         "Delta(h_corrupt + sum dAttn) - Delta_corrupt, M_direct = Delta(h_clean - sum dAttn) - Delta_corrupt, exact final "
                         "norm, / drop; dla = linear DLA shares (norm frozen at the corrupted run). Noising of direction d = denoising of "
                         "direction 1 - d in exact arithmetic (symmetric pairs used both ways); mirror_max_abs = numerical per-case deviation."}
    for f in ("main", "rep"):
        if f not in run.fam:
            continue
        out[f + "_validation"] = _w4_block(run, df, ds, run.ids(f, "validation"))
        out[f + "_all"] = _w4_block(run, df, ds, run.ids(f, "discovery") + run.ids(f, "validation"))
    if "main" in run.fam:  # strata: all 256 main pairs
        ids_all = run.ids("main", "discovery") + run.ids("main", "validation")
        st = []
        for col, v in [(col, v) for col, _ in STRATA for v in (True, False)] + [("d", 0), ("d", 1)]:
            ids = [c for c in ids_all if (c % 2 == v if col == "d" else bool(run.cases.at[c, col]) == v)]
            if len(ids) < 20:
                continue
            b = _w4_block(run, df, ds, ids)
            b.update(stratum=col, value=v)
            st.append(b)
        out["strata"] = st
    return out


def dla(run, cfg):
    """Direct logit attribution of the final-position attention / MoE outputs (scripts/ext7_wino_dla.py): per layer the
    clean - corrupted DLA difference / mean drop (pair bootstrap), totals and shares of the direct effect."""
    p = os.path.join(run.dir, "dla_rows.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    cs = pd.read_parquet(os.path.join(run.dir, "dla_cases.parquet")).set_index("case_id")
    df["attn"] = df.attn_c - df.attn_k
    df["moe"] = df.moe_c - df.moe_k
    out = {"max_abs_total_minus_drop": float(cs.dla_total_vs_drop.abs().max())}
    for f in ("main", "rep"):
        if f not in run.fam:
            continue
        ids = [c for v in run.fam[f].values() for c in v]
        sub = df[df.case_id.isin(ids)]
        drop = cs.loc[ids, "drop"]
        A_ = sub.pivot(index="case_id", columns="layer", values="attn").loc[ids]
        M_ = sub.pivot(index="case_id", columns="layer", values="moe").loc[ids]
        emb = (cs.emb_c - cs.emb_k).loc[ids]
        ta, tm = A_.sum(1), M_.sum(1)
        ent = {"attn_total": P.ratio(ta, drop), "moe_total": P.ratio(tm, drop), "emb_total": P.ratio(emb, drop),
               "attn_share_of_direct": P.ratio(ta, ta + tm),
               "attn_curve": P.norm_curve(A_, drop).norm.round(5).tolist(), "moe_curve": P.norm_curve(M_, drop).norm.round(5).tolist()}
        ca, cm = np.array(ent["attn_curve"]), np.array(ent["moe_curve"])
        ent["attn_peak"] = [int(ca.argmax()), float(ca.max())]
        ent["moe_peak"] = [int(cm.argmax()), float(cm.max())]
        ent["moe_top5"] = [(int(l), round(float(cm[l]), 3)) for l in np.argsort(-cm)[:5]]
        ent["attn_top5"] = [(int(l), round(float(ca[l]), 3)) for l in np.argsort(-ca)[:5]]
        if "den_attn" in cs.columns:  # direct-path split with the exact final norm
            c = cs.loc[ids]
            for k in ("den_attn", "den_moe", "noi_attn", "noi_moe"):
                ent[k] = P.ratio(c[k], drop)
            ent["phi_attn_direct"] = P.ratio(0.5 * (c.den_attn + c.noi_attn), drop)
            ent["recon_max_abs"] = float(c.recon.abs().max())
        out[f] = ent
    return out


# ---------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------
def fig_w2(summ, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"layer": "#d95f02", "attn_layer": "#1b9e77", "block": "#7570b3"}
    fig, axes = plt.subplots(2, len(runs), figsize=(6 * len(runs), 7), squeeze=False)
    for j, (cfg, run) in enumerate(runs):
        cur = summ[cfg["run"]]["w2"]["curves"]["main_val"]
        for i, (y, lo, hi, yl) in enumerate((("mean", "ci_lo", "ci_hi", "rescue (logits)"), ("norm", "norm_lo", "norm_hi", "rescue / mean drop"))):
            ax = axes[i][j]
            for k, g in cur.groupby("kind"):
                ax.plot(g.layer, g[y], color=cols.get(k), label=KIND_LABEL[k], lw=1.6)
                ax.fill_between(g.layer, g[lo], g[hi], color=cols.get(k), alpha=0.18)
            if i == 1:
                cf = summ[cfg["run"]].get("cf_moe_norm_curve")
                if cf is not None:
                    ax.plot(range(len(cf)), cf, color=cols["layer"], ls="--", lw=1.2, label="MoE output, CounterFact STR (ext6)")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("layer"); ax.set_ylabel(yl)
            ax.set_title(f"{cfg['label']}: WinoGrande STR, final position (validation, {cur.n.iloc[0]} pairs)", fontsize=9)
            ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext7_wino_w2_curves.{ext}"), dpi=150)
    plt.close(fig)


def fig_grid(summ, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for cfg, run in runs:
        g = summ[cfg["run"]].get("w3", {}).get("w1")
        if not g:
            continue
        kinds = [k for k in ("layer", "attn_layer", "resid") if k in g["kinds"]]
        w5 = summ[cfg["run"]]["w3"].get("w5", {}).get("kinds", {}).get("layer")
        panels = [(k, g["kinds"][k], f"{KIND_LABEL[k]}, window 1") for k in kinds] + ([("layer5", w5, "MoE output, window 5")] if w5 else [])
        fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 5.2), squeeze=False)
        for ax, (k, kk, title) in zip(axes[0], panels):
            cats = [c for c in GROUPS if c in kk]
            H = np.array([kk[c]["curve_norm"] for c in cats]).T
            vmax = np.nanmax(np.abs(H))
            im = ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest", origin="lower")
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels([f"{GSHORT[c]}\n(n={kk[c]['n_pairs']})" for c in cats], rotation=40, ha="right", fontsize=7)
            ax.set_ylabel("layer"); ax.set_title(title, fontsize=8.5)
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        fig.suptitle(f"{cfg['label']}: WinoGrande STR, rescue / mean drop by layer and token group", fontsize=10)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"ext7_wino_w3_grid_{os.path.basename(cfg['run'])}.{ext}"), dpi=150)
        plt.close(fig)


def fig_heads(summ, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have = [(cfg, run) for cfg, run in runs if summ[cfg["run"]].get("w5")]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(7 * len(have), 3.4), squeeze=False)
    for ax, (cfg, run) in zip(axes[0], have):
        t = summ[cfg["run"]]["w5"]["table"]
        H = t.pivot(index="layer", columns="head", values="val_mean")
        vmax = np.nanmax(np.abs(H.values))
        im = ax.imshow(H.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(len(H.index))); ax.set_yticklabels([f"L{l}" + (" (null)" if l == summ[cfg['run']]['w5']['null_layer'] else "") for l in H.index], fontsize=7)
        ax.set_xlabel("head"); ax.set_title(f"{cfg['label']}: per-head attention patch at the final position (validation rescue, logits)", fontsize=8.5)
        for _, r in t[t.detected_2sd].iterrows():
            ax.text(int(r["head"]), list(H.index).index(int(r.layer)), "*", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext7_wino_w5_heads.{ext}"), dpi=150)
    plt.close(fig)


def fig_dla(summ, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have = [(cfg, run) for cfg, run in runs if (summ[cfg["run"]].get("w4_dla") or {}).get("main")]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(6 * len(have), 3.4), squeeze=False)
    for ax, (cfg, run) in zip(axes[0], have):
        x = summ[cfg["run"]]["w4_dla"]["main"]
        ax.bar(np.arange(len(x["moe_curve"])) - 0.2, x["moe_curve"], width=0.4, color="#d95f02", label="MoE output")
        ax.bar(np.arange(len(x["attn_curve"])) + 0.2, x["attn_curve"], width=0.4, color="#1b9e77", label="attention output")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("layer"); ax.set_ylabel("DLA difference / mean drop")
        ax.set_title(f"{cfg['label']}: direct logit attribution of the final-position sublayer outputs (clean − corrupted)", fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext7_wino_w4_dla.{ext}"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def tables(summ, runs):
    out = {}
    rows = []
    for cfg, run in runs:
        for f, d in summ[cfg["run"]]["descriptors"].items():
            rows.append([cfg["label"], f, d["n_pairs"], f"{d['delta_clean']:+.2f}", f"{d['delta_corrupt']:+.2f}", F(d["drop"], 2),
                         f"{d['clean_top1_rate']:.2f}", f"{d['corrupt_top1_foil_rate']:.2f}", f"{d['names']:.2f}", f"{d['assoc']:.2f}", f"{d['top1_both']:.2f}",
                         f"{d['debiased']:.2f}", f"{d['one_token_option']:.2f}", f"{d['T_median']:.0f}"])
    out["desc"] = md_table(["Model", "Pair set", "Pairs", "Mean Δ clean", "Mean Δ corrupt", "Drop [95% CI]", "Clean top-1 = r", "Corrupt top-1 = r′",
                            "Names", "assoc", "top-1 both", "debiased", "One-token option", "Median T"], rows, "ext7_wino_descriptors",
                           "WinoGrande STR pair sets (directed cases pooled; drop = Δ_clean − Δ_corrupt, pair means)")
    rows = []
    for cfg, run in runs:
        s = summ[cfg["run"]]["w2"]
        for f in ("main", "rep", "own", "main_interior"):
            for p in s["peaks"].get(f, []):
                rows.append([cfg["label"], {"main": "main", "rep": "replication", "own": "own pool", "main_interior": "main, L ≤ L−5"}[f], KIND_LABEL[p["kind"]],
                             f"L{p['L_disc']}", f"{p['disc_mean']:+.2f} (gap {p['disc_gap_to_2nd']:+.2f} to L{p['disc_2nd']})", F(p["val_at_L_disc"]),
                             P.fmt_r(p["norm_at_L_disc"]), f"L{p['L_val']} {p['val_max']['mean']:+.3f}", ", ".join(f"L{l} {v:+.2f}" for l, v in p["disc_top5"][:4])])
    out["peaks"] = md_table(["Model", "Pair set", "Patched output", "L* (discovery)", "Discovery mean at L*", "Validation rescue at L* [95% CI]",
                             "Normalised (rescue / drop)", "Validation argmax", "Discovery top 4"], rows, "ext7_wino_w2_peaks",
                            "W2: final-position single-layer patches under STR (pairs; discovery argmax, validation value with pair-bootstrap CI)")
    rows = []
    for cfg, run in runs:
        s = summ[cfg["run"]]["w2"]["share"]
        for f, lab in (("main", "main, validation"), ("main_all", "main, all 256 pairs"), ("rep", "replication, validation"),
                       ("own", "own pool, validation"), ("main_d0", "main val., A→B only"), ("main_d1", "main val., B→A only")):
            if f not in s:
                continue
            x = s[f]
            rows.append([cfg["label"], lab, x["n_pairs"], f"{x['auc_attn']:.2f}", f"{x['auc_moe']:.2f}", f"{x.get('auc_block', float('nan')):.2f}",
                         f"{x['auc_attn_norm']:.3f}", f"{x['auc_moe_norm']:.3f}", f"{x['share']:.2f} [{x['share_lo']:.2f}, {x['share_hi']:.2f}]"])
        cfa = summ[cfg["run"]].get("cf_attn")
        if cfa and "share" in cfa:
            rows.append([cfg["label"], f"CounterFact STR, validation (ext7-controls' {cfg['cf']}_attnsweep, donor mean)", cfa["n_val"], "", "", "",
                         f"{cfa['auc_attn_norm']:.3f}", f"{cfa['auc_moe_norm']:.3f}", f"{cfa['share']:.2f}"])
    out["share"] = md_table(["Model", "Set", "Pairs", "AUC+ attention", "AUC+ MoE", "AUC+ block", "AUC+ attention / drop", "AUC+ MoE / drop",
                             "Attention share [95% CI]"], rows, "ext7_wino_w2_share",
                            "W2: attention share of the positive final-position rescue (Direction-2b definition: AUC+(attn) / (AUC+(attn) + AUC+(MoE)), sums over layers of the positive part of the mean curve)")
    rows = []
    for cfg, run in runs:
        s = summ[cfg["run"]]["w2"]
        for k, x in s["share_at_peaks"].items():
            rows.append([cfg["label"], f"L{x['layer']} ({KIND_LABEL[k]} peak)", F(x["attn"]), F(x["moe"]), P.fmt_r(x["share"], 2)])
        for x in s["additivity"]["top_block_layers"][:3]:
            rows.append([cfg["label"], f"L{x['layer']} (block {x['block']:+.2f})", f"{x['attn']:+.3f}", f"{x['moe']:+.3f}",
                         f"block − (attn + MoE) {x['gap']:+.3f} [{x['gap_lo']:+.3f}, {x['gap_hi']:+.3f}], r = {x['r_pair']:.2f}"])
    out["atpeak"] = md_table(["Model", "Layer", "Attention output rescue", "MoE output rescue", "Attention share at the layer / additivity"], rows,
                             "ext7_wino_w2_layers", "W2: attention vs MoE at the peak layers (validation pairs) and block additivity at the largest block layers")
    rows = []
    for cfg, run in runs:
        for x in summ[cfg["run"]]["w2"]["strata"]:
            if "share" not in x:
                rows.append([cfg["label"], x["label"], x["value"], x["n_pairs"], "", "", "", "", "", ""])
                continue
            rows.append([cfg["label"], x["label"], x["value"], x["n_pairs"], f"{x['drop']:.2f}", P.fmt_r(x["norm_layer"]), P.fmt_r(x["norm_attn_layer"]),
                         P.fmt_r(x["norm_block"]) if "norm_block" in x else "", f"{x['share']['share']:.2f} [{x['share']['share_lo']:.2f}, {x['share']['share_hi']:.2f}]",
                         f"{x['norm_peak_attn_layer']:.3f} / {x['norm_peak_layer']:.3f}"])
    out["strata"] = md_table(["Model", "Stratum", "Value", "Pairs", "Mean drop", "MoE at L*_MoE / drop", "Attention at L*_attn / drop", "Block at L*_block / drop",
                              "Attention share", "Peak attention / peak MoE (normalised)"], rows, "ext7_wino_w2_strata",
                             "W2 strata (all 256 main pairs, layers fixed by the main discovery argmax; descriptive)")
    # W6
    rows, rows_f, rows_j, rows_eq = [], [], [], []
    for cfg, run in runs:
        s = summ[cfg["run"]].get("w6")
        if not s:
            continue
        for f in ("main", "rep"):
            if f not in s:
                continue
            for tag in ("two_stage", "interior"):
                r = s[f][tag]
                sel = r["selection"]
                ev = r.get("eval")
                rows.append([cfg["label"], "main" if f == "main" else "replication", "two-stage" if tag == "two_stage" else "interior (≤ L−5)", f"L{r['layer']}",
                             F(r["layer_val"]), P.fmt_r(r["layer_norm"]),
                             f"L{r['layer']}E{sel['e_star']:03d}" if sel["e_star"] is not None else "none",
                             f"{sel.get('disc_active', sel.get('max_activity'))}/{len(run.ids(f, 'discovery'))} (≥ {sel['threshold']}), {sel['n_candidates']} cand.",
                             F(ev["rescue"]) if ev else "", F(ev["spec"]) if ev else "", P.fmt_r(ev["spec_norm"]) if ev else "",
                             F(r["coalitions"]["coalition_clean"]), r["pattern"]])
            for x in s[f]["joint_top"][:5]:
                rows_j.append([cfg["label"], "main" if f == "main" else "replication", x["rank"], f"L{x['layer']}E{x['expert']:03d}", x["disc_active"],
                               f"{x['disc_allcase_mean']:+.3f}", F(x["val_rescue"]), F(x["val_spec"])])
        for x in s["main"].get("cf_experts", []):
            rows_f.append([cfg["label"], f"L{x['layer']}E{x['expert']:03d}", f"{x['disc_active']}/{len(run.ids('main', 'discovery'))}", x["val_active"], F(x["rescue"]), F(x["spec"]), P.fmt_r(x["rescue_norm"])])
        eq = s["main"]["two_stage"].get("equal_norm")
        if eq and "spec_eq" in eq:
            rows_eq.append([cfg["label"], f"L{s['main']['two_stage']['layer']}E{s['main']['two_stage']['eval']['expert']:03d}", eq["kind"], eq["n"],
                            F(eq["spec_raw"]), F(eq["sel_eq"]), F(eq["ctrl_eq"]), F(eq["spec_eq"])])
    out["w6"] = md_table(["Model", "Pair set", "Rule", "Layer", "Layer rescue (val) [95% CI]", "Layer / drop", "Selected expert", "Disc. active", "Val rescue",
                          "Spec", "Spec / drop", "Clean top-k coalition", "Pattern"], rows, "ext7_wino_w6_experts",
                         "W6: paper two-stage selection on WinoGrande STR (MoE-layer argmax on discovery → recurrence-first expert; validation pairs)")
    out["w6j"] = md_table(["Model", "Pair set", "Rank", "(layer, expert)", "Disc. active (directed)", "Disc. all-case", "Val rescue", "Spec"], rows_j,
                          "ext7_wino_w6_joint", "W6: joint (layer, expert) search over all layers (recurrence gate = half of the discovery directed cases)")
    out["w6f"] = md_table(["Model", "CounterFact expert", "Disc. active", "Val active", "Val rescue", "Spec", "Rescue / drop"], rows_f, "ext7_wino_w6_cf_experts",
                          "W6: the CounterFact STR experts (Direction 6) as fixed hypotheses on WinoGrande validation pairs")
    rows = []
    lab = dict(STRATA)
    for cfg, run in runs:
        s = summ[cfg["run"]].get("w6")
        if not s or "strata" not in s:
            continue
        e = s["main"]["two_stage"]
        for x in s["strata"]:
            rows.append([cfg["label"], f"L{e['layer']}E{e['eval']['expert']:03d}", lab[x["stratum"]], x["value"], x["n_pairs"], F(x["layer"]), F(x["rescue"]), F(x["spec"])])
    out["w6s"] = md_table(["Model", "Expert", "Stratum", "Value", "Pairs", "Layer rescue", "Expert rescue", "Spec"], rows, "ext7_wino_w6_strata",
                          "W6 strata: the main two-stage expert on all 256 main pairs by stratum (descriptive; layer and expert fixed)")
    out["w6eq"] = md_table(["Model", "Expert", "Control", "n (anchor-active val, directed)", "Raw Spec vs control", "Selected, equal norm", "Control, equal norm",
                            "Equal-norm Spec"], rows_eq, "ext7_wino_w6_equalnorm", "W6: equal-norm check at the selected layer (Qwen3: gate-matched control, Table 9; Mixtral: other active expert, Table 11)")
    # W3
    rows = []
    for cfg, run in runs:
        g = summ[cfg["run"]].get("w3", {})
        for w in ("w1", "w5"):
            if w not in g:
                continue
            for k, kk in g[w]["kinds"].items():
                for cat in GROUPS:
                    if cat not in kk:
                        continue
                    x = kk[cat]
                    rows.append([cfg["label"], w[1:], KIND_LABEL[k], GSHORT[cat], x["n_pairs"], f"L{x['peak_layer']}",
                                 f"{x['peak'][0]:+.3f} [{x['peak'][1]:+.3f}, {x['peak'][2]:+.3f}]", f"{x['sum_over_layers_norm']:+.3f}",
                                 f"{x['peak_dp']:+.4f} (L{x['peak_dp_layer']})"])
    out["w3"] = md_table(["Model", "Window", "Patched quantity at p", "Token group", "Pairs", "Peak layer", "Peak rescue / drop [95% CI]", "Sum over layers",
                          "Peak Δp (descriptive)"], rows, "ext7_wino_w3_grid_peaks", "W3: position × layer grid under STR (main set, 256 pairs; positions before the option are exactly zero)")
    # W5
    rows, rows_l, rows_a = [], [], []
    for cfg, run in runs:
        s = summ[cfg["run"]].get("w5")
        if not s:
            continue
        t = s["table"]
        for _, r in pd.concat([t.head(6), t.tail(3)]).iterrows():
            rows.append([cfg["label"], f"L{int(r.layer)}H{int(r['head'])}", f"{r.val_mean:+.3f} [{r.val_lo:+.3f}, {r.val_hi:+.3f}]", f"{r.disc_mean:+.3f}",
                         f"{r.z_val:+.1f} / {r.z_disc:+.1f}", "yes" if r.detected_2sd else "no", f"{r.spec:+.3f} [{r.spec_lo:+.3f}, {r.spec_hi:+.3f}]",
                         f"{r.share_attn_layer:.2f}"])
        for x in s["layer_summary"]:
            rows_l.append([cfg["label"], f"L{x['layer']}" + (" (null)" if x["layer"] == s["null_layer"] else ""), F(x["attn_layer"]), F(x["sum_heads"]),
                           f"{x['r_sum_vs_attn_pair']:.2f}", F(x["moe_layer"]), F(x["block"]), f"H{x['top1_head_disc']}", x["k_50"] if x["k_50"] is not None else "not reached", x["k_80"] if x["k_80"] is not None else "not reached",
                           f"{x['top3_share_of_attn']:.2f}"])
        for x in s["attention_mass_top"]:
            rows_a.append([cfg["label"], f"L{x['layer']}H{x['head']}", f"{x['val_mean']:+.3f}"] +
                          [f"{x['clean_' + nm]:.2f} → {x['corrupt_' + nm]:.2f}" for nm in P.HEAD_CLASSES])
    out["w5"] = md_table(["Model", "Head", "Val rescue [95% CI]", "Disc. mean", "z (val / disc)", "≥ 2 SD both splits", "Spec vs other heads", "Share of attn_layer"],
                         rows, "ext7_wino_w5_heads", "W5: top six and bottom three heads by validation rescue (z over all scanned heads of the model; detection = |z| ≥ 2 on discovery AND validation, Zhang & Nanda §3)")
    out["w5l"] = md_table(["Model", "Layer", "Attention output (val)", "Sum of heads", "r(sum, attn) pairs", "MoE output", "Block", "Top head (disc.)",
                           "Heads for 50 %", "Heads for 80 %", "Top-3 share"], rows_l, "ext7_wino_w5_layers",
                          "W5: per layer, additivity of the head patches and greedy additive minimal head sets (order by discovery, validation sums)")
    out["w5a"] = md_table(["Model", "Head", "Val rescue"] + [f"{nm} (clean → corrupt)" for nm in P.HEAD_CLASSES], rows_a, "ext7_wino_w5_attention",
                          "W5: final-position attention mass of the top heads by position class (str = the filled option; ment_filled / ment_other = first mention of the candidate filled in the clean prompt / of the other one)")
    # W4 (revised form)
    def g(x, k, sub=None, d=3):
        v = x.get(k) if sub is None else (x.get(k) or {}).get(sub)
        return P.fmt_r(v, d) if v is not None else "pending"
    rows = []
    for cfg, run in runs:
        s = summ[cfg["run"]].get("w4")
        if not s:
            continue
        for key, lab in (("main_validation", "main, validation"), ("rep_validation", "replication, validation"), ("main_all", "main, all 256"),
                         ("rep_all", "replication, all 256")):
            if key not in s:
                continue
            x = s[key]
            rows.append([cfg["label"], lab, x["n_pairs"], g(x, "M_denoise"), g(x, "M_noise"), g(x, "direct", "A_direct"), g(x, "direct", "M_direct"),
                         g(x, "dla", "attn", 2), g(x, "dla", "moe", 2), g(x, "A_sanity_denoise"), g(x, "block_sanity_denoise"),
                         f"{x['mirror_max_abs']:.3g}" if x.get("mirror_max_abs") is not None else ""])
    out["w4"] = md_table(["Model", "Pair set", "Pairs", "M: all MoE outputs, denoise (sufficiency)", "M: all MoE outputs, noise (necessity)",
                          "A_direct (direct path of attention outputs)", "M_direct (direct path of MoE outputs)", "DLA share attention", "DLA share MoE",
                          "A: all attention (sanity, = 1 by construction)", "Block all layers (sanity)", "max |noise(d) − denoise(1−d)| (directed, logits)"], rows, "ext7_wino_w4_joint",
                         "W4 (revised form shared by ext7-wino / ext7-controls / ext8-addback): final-position joint patches and direct paths, all / drop (population ratios, pair bootstrap); direct split = moetrace/ext7_controls.direct_split_pairs")
    rows = []
    lab = dict(STRATA) | {"d": "direction"}
    for cfg, run in runs:
        s = summ[cfg["run"]].get("w4")
        if not s or "strata" not in s:
            continue
        for x in s["strata"]:
            v = x["value"] if x["stratum"] != "d" else ("A→B" if x["value"] == 0 else "B→A")
            rows.append([cfg["label"], lab[x["stratum"]], v, x["n_pairs"], g(x, "M_denoise", d=2), g(x, "M_noise", d=2), g(x, "direct", "A_direct", 2),
                         g(x, "direct", "M_direct", 2), g(x, "dla", "attn", 2)])
    out["w4s"] = md_table(["Model", "Stratum", "Value", "Pairs", "M denoise", "M noise", "A_direct", "M_direct", "DLA share attention"], rows,
                          "ext7_wino_w4_strata", "W4 strata (main set, all 256 pairs)")
    rows = []
    for cfg, run in runs:
        x = summ[cfg["run"]].get("w4_dla")
        if not x:
            continue
        for f in ("main", "rep"):
            if f not in x:
                continue
            e = x[f]
            rows.append([cfg["label"], f, P.fmt_r(e["attn_total"]), P.fmt_r(e["moe_total"]), P.fmt_r(e["emb_total"]), P.fmt_r(e["attn_share_of_direct"], 2),
                         P.fmt_r(e["den_attn"]) if "den_attn" in e else "", P.fmt_r(e["den_moe"]) if "den_moe" in e else "",
                         P.fmt_r(e["noi_attn"]) if "noi_attn" in e else "", P.fmt_r(e["noi_moe"]) if "noi_moe" in e else "",
                         P.fmt_r(e["phi_attn_direct"], 2) if "phi_attn_direct" in e else "",
                         ", ".join(f"L{l} {v:+.3f}" for l, v in e["attn_top5"][:3]), ", ".join(f"L{l} {v:+.3f}" for l, v in e["moe_top5"][:3])])
    out["dla"] = md_table(["Model", "Pair set", "DLA attention / drop", "DLA MoE / drop", "DLA embedding (norm scale) / drop", "Attention share of DLA",
                           "Direct: corrupt + ΣdAttn", "Direct: corrupt + ΣdMoE", "Direct: clean − ΣdAttn (damage)", "Direct: clean − ΣdMoE (damage)",
                           "Direct Shapley φ_attn", "Top attention layers (DLA / drop)", "Top MoE layers (DLA / drop)"], rows, "ext7_wino_w4_dla",
                          "W4 cross-check with ext7-wino's own implementation (scripts/ext7_wino_dla.py; DLA linearised at each run's own final RMS, direct = exact final norm; all / drop, main and replication families, all pairs) and the per-layer DLA peaks")
    return out


def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    runs, summ = [], {}
    for cfg in RUNS:
        if not os.path.exists(os.path.join(ROOT, cfg["run"], "str_sweep_rows.parquet")):
            continue
        run = load_run(cfg)
        runs.append((cfg, run))
        s = {"label": cfg["label"], "L": run.L, "descriptors": descriptors(run, cfg)}
        s["w2"] = w2(run, cfg)
        cf = cf_curve(cfg)
        s["cf_moe_norm_curve"] = cf
        if cf is not None:
            s["cf_moe_norm_peak"] = [int(np.argmax(cf)), float(np.max(cf))]
        s["cf_attn"] = cf_attn_curves(cfg)
        s["w6"] = w6(run, cfg, s["w2"])
        s["w3"] = grid(run, cfg)
        s["w5"] = heads(run, cfg)
        s["w4"] = joint(run, cfg)
        s["w4_dla"] = dla(run, cfg)
        summ[cfg["run"]] = s
        print(cfg["run"], "analysed")
    fig_w2(summ, runs)
    fig_grid(summ, runs)
    fig_heads(summ, runs)
    fig_dla(summ, runs)
    t = tables(summ, runs)
    for k, v in t.items():
        print(v)
    js = {}
    for run, s in summ.items():
        s2 = dict(s)
        s2["w2"] = dict(s["w2"])
        s2["w2"]["curves"] = {k: v.to_dict("records") for k, v in s["w2"]["curves"].items()}
        s2["w2"]["additivity"] = {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in s["w2"]["additivity"].items()}
        if s.get("w5"):
            s2["w5"] = dict(s["w5"])
            s2["w5"]["table"] = s["w5"]["table"].head(40).to_dict("records")
        if s.get("cf_moe_norm_curve") is not None:
            s2["cf_moe_norm_curve"] = [round(float(x), 5) for x in s["cf_moe_norm_curve"]]
        if isinstance(s.get("cf_attn"), dict):
            s2["cf_attn"] = {k: (list(map(float, v)) if isinstance(v, np.ndarray) else v) for k, v in s["cf_attn"].items()}
        js[run] = clean(s2)
    with open(os.path.join(ROOT, "ext7_wino_summary.json"), "w") as f:
        json.dump(js, f, indent=1, default=str)


if __name__ == "__main__":
    main()
