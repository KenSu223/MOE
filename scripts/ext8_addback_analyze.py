"""ext8 A0-A3 analysis: ceilings / W4 decomposition, add-back and deletion curves, adaptive strategies, Shapley values.

Reads results/<out>/addback_* of every available run (see RUNS), writes results/tables/ext8_*.md|csv,
results/figures/ext8_*.png|pdf and results/ext8_addback_summary.json.

Normalisation: rescue_k = Delta_k - Delta_corrupt (same pass), damage_k = Delta_clean - Delta_k (same pass); both are
divided by the reference drop of the row (pass-0 prefill, Delta_clean - Delta_corrupt). Population values are ratios of
means over validation cases (CounterFact: donor means per case; WinoGrande: directed cases) with a percentile bootstrap
over units (CounterFact cases; WinoGrande pairs, both directions together). Per-case values are case ratios.

Usage: python scripts/ext8_addback_analyze.py [--runs key,key] [--partial]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import MODELS, RESULTS
from moetrace import ext8_addback as X

TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
RUNS = {
    "cf_qwen3": dict(model="qwen3", task="cf", src="qwen3_str", out="qwen3_str_addback", label="Qwen3-30B-A3B-Base", tlabel="CounterFact"),
    "cf_mixtral": dict(model="mixtral", task="cf", src="mixtral_bos_str", out="mixtral_bos_str_addback", label="Mixtral-8x7B (BOS)",
                       tlabel="CounterFact"),
    "wino_qwen3": dict(model="qwen3", task="wino", src="wino_qwen3_str", out="wino_qwen3_str_addback", label="Qwen3-30B-A3B-Base",
                       tlabel="WinoGrande", pairs="data/wino_str/pairs_train_xl_qwen3.parquet", case_sets="data/wino_str/case_sets.json"),
    "wino_mixtral": dict(model="mixtral", task="wino", src="wino_mixtral_bos_str", out="wino_mixtral_bos_str_addback",
                         label="Mixtral-8x7B (BOS)", tlabel="WinoGrande", pairs="data/wino_str/pairs_train_xl_mixtral_bos.parquet",
                         case_sets="data/wino_str/case_sets.json"),
    "smoke_olmoe": dict(model="olmoe", task="cf", src="olmoe_addback_src", out="olmoe_addback_smoke", label="OLMoE (smoke)", tlabel="CounterFact"),
}
A1_SHOW = ("oracle", "pop", "layerwise", "dla", "vnorm", "weight", "rand")
A3_SHOW = ("noise_oracle", "oracle", "pop", "layerwise", "dla", "rand")
Q = (0.5, 0.8, 0.9)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_task(cfg):
    if cfg["task"] == "cf":
        return X.load_cf(cfg["model"], cfg["src"])
    return X.load_wino(cfg["model"], cfg["src"], os.path.join("/home/ubuntu/MOE", cfg["pairs"]), os.path.join("/home/ubuntu/MOE", cfg["case_sets"]))


def ci_s(r, d=3):
    return f"{r['ratio']:.{d}f} [{r['lo']:.{d}f}, {r['hi']:.{d}f}]"


def fk(x):
    return "never" if not np.isfinite(x) else f"{int(x)}"


# ------------------------------------------------------------------------------------------------------------------
def analyse(key, cfg):
    t = load_task(cfg)
    R = X.load_out(cfg["out"])
    rows = R["rows"]
    st = R["state"]
    K = t.K
    cases = t.cases.set_index("case_id")
    val = cases.index[cases.split == "validation"].tolist()
    allc = cases.index.tolist()
    unit = cases.unit
    row_first = set(t.rows.row_id[t.rows["first"]])
    rows = rows.merge(t.rows[["row_id", "first"]], on="row_id", how="left")
    # reference drop per case (donor mean) and per first-donor row
    dref = rows.drop_duplicates("row_id").set_index("row_id").drop_ref
    rcase = t.rows.set_index("row_id").case_id
    drop_case = dref.groupby(rcase.loc[dref.index].values).mean()
    out = {"model": cfg["model"], "task": cfg["task"], "label": cfg["label"], "K": K, "L": t.L, "topk": t.topk,
           "n_cases_val": len(val), "n_rows_val": int(t.rows.case_id.isin(val).sum()), "prefill_range": R["prefill_range"],
           "mean_drop_val": float(drop_case.loc[val].mean()), "passes": int(rows.pass_.nunique())}

    def mat(df, value, cols, sub_cases, first_only=False):
        d = df[df.case_id.isin(sub_cases)]
        if first_only:
            d = d[d["first"]]
        return d.groupby(["case_id"] + cols)[value].mean().unstack(cols) if cols else d.groupby("case_id")[value].mean()

    def drop_of(sub_cases, first_only=False):
        if not first_only:
            return drop_case.loc[sub_cases]
        fr = t.rows[t.rows["first"] & t.rows.case_id.isin(sub_cases)].set_index("case_id").row_id
        return pd.Series(dref.loc[fr.values].values, index=fr.index)

    # ---------------- A0
    a0 = rows[rows.fam == "a0"]
    a0r = {}
    for sub_name, sub in (("validation", val), ("all", allc)):
        dmat = mat(a0[a0.dir == "d"], "rescue", ["order"], sub)
        nmat = mat(a0[a0.dir == "n"], "damage", ["order"], sub)
        dr = drop_of(dmat.index.tolist())
        res = {}
        for name, M in (("d", dmat), ("n", nmat)):
            rc = X.ratio_cols(M, dr, unit)
            res[name] = {k: {"ratio": float(v.ratio), "lo": float(v.lo), "hi": float(v.hi), "n": int(v.n)} for k, v in rc.iterrows()}
            # Shapley split of the two-player game (attention, MoE) and redundancy, bootstrapped as derived columns
            A, Mo = M["all_attn"], M["all_moe"]
            der = pd.DataFrame({"phi_attn": 0.5 * (A + dr - Mo), "phi_moe": 0.5 * (Mo + dr - A), "redundancy": A + Mo - dr})
            rd = X.ratio_cols(der[["phi_attn", "phi_moe", "redundancy"]], dr, unit)
            for k, v in rd.iterrows():
                res[name][k] = {"ratio": float(v.ratio), "lo": float(v.lo), "hi": float(v.hi), "n": int(v.n)}
            pc = (M.div(dr, axis=0))
            res[name]["per_case_median"] = {k: float(pc[k].median()) for k in M.columns}
            res[name]["frac_cases_all_moe_ge_0.8"] = float((pc["all_moe"] >= 0.8).mean())
        # direct-path split (exact final norm, offline fp32) and linear DLA split
        if R["direct"] is not None:
            dd = R["direct"][R["direct"].case_id.isin(sub)].copy()
            dd["dr32"] = dd.d_clean_fp32 - dd.d_corrupt_fp32
            dd["a_dir"] = dd.d_attn_direct - dd.d_corrupt_fp32
            dd["m_dir"] = dd.d_moe_direct - dd.d_corrupt_fp32
            g = dd.groupby("case_id")[["dr32", "a_dir", "m_dir", "dla_attn", "dla_moe", "dla_total"]].mean()
            rc = X.ratio_cols(g[["a_dir", "m_dir"]], g.dr32, unit)
            res["direct"] = {k: {"ratio": float(v.ratio), "lo": float(v.lo), "hi": float(v.hi)} for k, v in rc.iterrows()}
            rc = X.ratio_cols(g[["dla_attn", "dla_moe"]], g.dla_total, unit)
            res["dla_split"] = {k: {"ratio": float(v.ratio), "lo": float(v.lo), "hi": float(v.hi)} for k, v in rc.iterrows()}
            res["dla_total_over_drop"] = float(g.dla_total.sum() / g.dr32.sum())
        a0r[sub_name] = res
    out["a0"] = a0r
    ceil_case = mat(a0[(a0.dir == "d") & (a0.order == "all_moe")], "rescue", [], allc)
    ceil_case_n = mat(a0[(a0.dir == "n") & (a0.order == "all_moe")], "damage", [], allc)

    # ---------------- A1 / A3 curves
    def curves(fam, dr_, value, ceil, show, first_only=False, sub=None):
        sub = sub or val
        d = rows[(rows.fam == fam) & (rows.dir == dr_)].copy()
        allact = d[d.order == "all_active"]
        d = d[d.order != "all_active"]
        d["order2"] = np.where(d.order.str.startswith("rand"), "rand", d.order)
        # random orders: mean over permutations per row first
        d = d.groupby(["case_id", "row_id", "first", "order2", "k"], as_index=False)[[value, "delta", "rank_true"]].agg(
            {value: "mean", "delta": "mean", "rank_true": lambda x: float((x == 1).mean())})
        aa = allact.groupby(["case_id", "row_id", "first", "k"], as_index=False)[[value, "delta", "rank_true"]].agg(
            {value: "mean", "delta": "mean", "rank_true": lambda x: float((x == 1).mean())})
        res = {}
        ks_all = [k for k in X.k_grid(K)]
        drp = drop_of(sub, first_only)
        cl = ceil.loc[drp.index]
        for o in show:
            dd = d[d.order2 == o]
            if dd.empty:
                continue
            dd = pd.concat([dd, aa.assign(order2=o)], ignore_index=True)
            dd = dd[dd.case_id.isin(sub)]
            if first_only:
                dd = dd[dd["first"]]
            Mv = dd.groupby(["case_id", "k"])[value].mean().unstack("k").reindex(index=drp.index)
            Md = dd.groupby(["case_id", "k"])["delta"].mean().unstack("k").reindex(index=drp.index)
            Mt = dd.groupby(["k"])["rank_true"].mean()
            ks = np.array([k for k in ks_all if k in Mv.columns])
            Mv, Md = Mv[ks], Md[ks]
            rc = X.ratio_cols(Mv, drp, unit)
            rcc = X.ratio_cols(Mv, cl, unit)  # relative to the all-MoE ceiling (ratio of means)
            ceil_r = float(cl.sum() / drp.sum())
            pc = Mv.div(drp, axis=0).values  # per-case r(k)
            pcc = Mv.div(cl.where(cl > 0), axis=0).values
            R_ = rc.ratio.values
            entry = {"k": ks.tolist(), "r": [float(x) for x in R_], "lo": [float(x) for x in rc.lo], "hi": [float(x) for x in rc.hi],
                     "r_ceiling": [float(x) for x in rcc.ratio], "top1_frac": [float(Mt.get(k, np.nan)) for k in ks],
                     "per_case_median": [float(x) for x in np.nanmedian(pc, 0)],
                     "per_case_q25": [float(x) for x in np.nanpercentile(pc, 25, 0)], "per_case_q75": [float(x) for x in np.nanpercentile(pc, 75, 0)],
                     "ceiling": ceil_r, "auc_log": X.auc_log(ks, R_), "auc_lin": X.auc_lin(ks, R_, K),
                     "auc_log_ceiling": X.auc_log(ks, rcc.ratio.values),
                     "max_r": float(np.max(R_)), "k_max_r": int(ks[int(np.argmax(R_))])}
            for q in Q:
                entry[f"k{int(q * 100)}_ceiling"] = X.k_at(ks, R_, q * ceil_r) if ceil_r > 0 else float("inf")
                entry[f"k{int(q * 100)}_drop"] = X.k_at(ks, R_, q)
                kc = X.first_k(ks, np.nan_to_num(pcc, nan=-np.inf), q)
                kc = kc[(cl > 0).values]
                entry[f"case_k{int(q * 100)}_ceiling"] = X.med_iqr(kc)
                entry[f"case_k{int(q * 100)}_drop"] = X.med_iqr(X.first_k(ks, np.nan_to_num(pc, nan=-np.inf), q))
            sign = (Md.values > 0) if value == "rescue" else (Md.values < 0)
            kr = np.where(sign.any(1), ks[np.argmax(sign, axis=1)], np.inf)
            entry["case_k_restored" if value == "rescue" else "case_k_flipped"] = X.med_iqr(kr)
            entry["auc_log_per_case"] = X.med_iqr([X.auc_log(ks, row) for row in pc])
            res[o] = entry
        # endpoint and ceiling
        return res

    out["a1"] = curves("a1", "d", "rescue", ceil_case, A1_SHOW)
    out["a1_first"] = curves("a1", "d", "rescue", ceil_case, A1_SHOW, first_only=True)
    out["a3"] = curves("a3", "n", "damage", ceil_case_n, A3_SHOW)
    out["a3_first"] = curves("a3", "n", "damage", ceil_case_n, A3_SHOW, first_only=True)
    # restoration / flip at the ceilings
    a0d = a0[(a0.order == "all_moe") & a0.case_id.isin(val)]
    out["ceiling_restores_answer_frac_cases"] = float((a0d[a0d.dir == "d"].groupby("case_id").delta.mean() > 0).mean())
    out["ceiling_flips_answer_frac_cases"] = float((a0d[a0d.dir == "n"].groupby("case_id").delta.mean() < 0).mean())
    aa = rows[(rows.order == "all_active") & rows.case_id.isin(val)]
    out["all_active_restores_answer_frac_cases"] = float((aa[aa.dir == "d"].groupby("case_id").delta.mean() > 0).mean())
    pf0 = R["prefill"][R["prefill"].pass_ == R["prefill"].pass_.min()]
    clean_top1 = pf0[(pf0.kind == "clean") & pf0.case_id.isin(val)]
    out["clean_top1_frac_val"] = float((clean_top1.rank_true == 1).mean())

    # ---------------- A2 adaptive: greedy (all validation rows), beam and exact (first donor)
    G = st.greedy
    gk = np.arange(1, X.GREEDY_STEPS + 1)
    grows = []
    for rid, g in G.items():
        rr = list(g.resc) + [g.resc[-1]] * (X.GREEDY_STEPS - len(g.resc)) if g.resc else [np.nan] * X.GREEDY_STEPS
        grows.append(dict(row_id=rid, case_id=int(t.rows.case_id.iloc[rid]), first=bool(t.rows["first"].iloc[rid]), steps=len(g.S),
                          stop=g.stop_reason, done=g.done, **{f"k{k}": rr[k - 1] for k in gk}))
    gdf = pd.DataFrame(grows)
    if len(gdf):
        Mg = gdf.groupby("case_id")[[f"k{k}" for k in gk]].mean()
        drp = drop_of(Mg.index.tolist())
        rc = X.ratio_cols(Mg, drp, unit)
        cl = ceil_case.loc[Mg.index]
        rcc = X.ratio_cols(Mg, cl, unit)
        out["greedy"] = {"k": gk.tolist(), "r": rc.ratio.tolist(), "lo": rc.lo.tolist(), "hi": rc.hi.tolist(), "r_ceiling": rcc.ratio.tolist(),
                         "n_rows": int(len(gdf)), "frac_done": float(gdf.done.mean()), "frac_stop_ceiling": float((gdf.stop == "ceiling").mean()),
                         "median_steps": float(gdf.steps.median()), "pool": int(len(next(iter(G.values())).pool)),
                         "ceiling": float(cl.sum() / drp.sum())}
        for q in Q:
            out["greedy"][f"k{int(q * 100)}_ceiling"] = X.k_at(gk, rc.ratio.values, q * out["greedy"]["ceiling"])
            out["greedy"][f"k{int(q * 100)}_drop"] = X.k_at(gk, rc.ratio.values, q)
            pcc = Mg.div(cl.where(cl > 0), axis=0).values
            kc = X.first_k(gk, np.nan_to_num(pcc, nan=-np.inf), q)[(cl > 0).values]
            out["greedy"][f"case_k{int(q * 100)}_ceiling"] = X.med_iqr(kc)
        out["greedy"]["auc_log_k15"] = X.auc_log(gk, rc.ratio.values)
        import collections
        c1, c5, lay = collections.Counter(), collections.Counter(), collections.Counter()
        for g_ in G.values():
            c1[g_.S[0]] += 1
            c5.update(g_.S[:5])
            lay.update(p_[0] for p_ in g_.S)
        nG = len(G)
        out["greedy"]["first_pick_top"] = [(f"L{l}E{e:03d}", round(v / nG, 3)) for (l, e), v in c1.most_common(5)]
        out["greedy"]["in_first5_top"] = [(f"L{l}E{e:03d}", round(v / nG, 3)) for (l, e), v in c5.most_common(6)]
        out["greedy"]["distinct_in_first5"] = len(c5)
        tl = sum(lay.values())
        out["greedy"]["layer_share_top"] = [(int(l), round(v / tl, 3)) for l, v in lay.most_common(6)]
        # static curves on the same k (<= 15) for comparison: oracle / pop at grid points
        for o in ("oracle", "pop"):
            if o in out["a1"]:
                e = out["a1"][o]
                out["greedy"][f"{o}_at_grid"] = {int(k): float(r) for k, r in zip(e["k"], e["r"]) if k <= 16}
        out["greedy"]["auc_log_k15_oracle_interp"] = X.auc_log(gk, np.interp(np.log(gk), np.log(out["a1"]["oracle"]["k"]), out["a1"]["oracle"]["r"])) \
            if "oracle" in out["a1"] else None
    B = st.beam
    if B:
        brow = []
        for rid, b in B.items():
            for size, (s_, v) in b.best.items():
                brow.append(dict(row_id=rid, size=size, delta=v))
        bdf = pd.DataFrame(brow)
        pf = R["prefill"]
        # rescue relative to the corrupted run's pass-0 value (beam values come from several passes)
        dcor0 = pf[(pf.pass_ == pf.pass_.min()) & (pf.kind == "corrupt")].set_index("row_id").delta
        bdf["rescue"] = bdf.delta - bdf.row_id.map(dcor0)
        bdf["case_id"] = bdf.row_id.map(rcase)
        Mb = bdf.groupby(["case_id", "size"]).rescue.mean().unstack("size")
        drp = drop_of(Mb.index.tolist(), first_only=True)
        rb = X.ratio_cols(Mb, drp, unit)
        gsub = gdf[gdf.row_id.isin(list(B))].groupby("case_id")[[f"k{k}" for k in range(1, X.BEAM_MAX + 1)]].mean().reindex(Mb.index)
        gsub.columns = list(range(1, X.BEAM_MAX + 1))
        rg = X.ratio_cols(gsub, drp, unit)
        diff = (Mb[list(range(1, X.BEAM_MAX + 1))] - gsub).div(drp, axis=0)
        out["beam"] = {"n_rows": len(B), "size": list(range(1, X.BEAM_MAX + 1)), "r_beam": rb.ratio.tolist(), "lo": rb.lo.tolist(), "hi": rb.hi.tolist(),
                       "r_greedy_same_rows": rg.ratio.tolist(), "beam_minus_greedy_mean": diff.mean().tolist(),
                       "frac_beam_better_0.01": [(diff[s] > 0.01).mean() for s in range(1, X.BEAM_MAX + 1)]}
    ex = rows[rows.fam == "exact"]
    if len(ex):
        out["exact"] = exact_analysis(t, ex, gdf, drop_of, unit, dref)
    sh = rows[rows.fam == "shapley"]
    if len(sh):
        out["shapley"] = shapley_analysis(t, sh, rows, dref)
    return out, rows, t


def exact_analysis(t, ex, gdf, drop_of, unit, dref):
    """Per first-donor row: best subset of the top-10 per size vs greedy within the top-10 (simulated on the same table)
    vs the static oracle prefix (top-k singles)."""
    res = []
    for rid, g in ex.groupby("row_id"):
        val = {X.parse_set(s): float(r) for s, r in zip(g.set, g.rescue)}
        top = sorted({p for s in val for p in s}, key=lambda p: (-t.single[rid][p], p))  # oracle order of the top-10
        n = len(top)
        best = {}
        for s, v in val.items():
            if len(s) not in best or v > best[len(s)][1]:
                best[len(s)] = (s, v)
        S, gv = (), {}
        for k in range(1, n + 1):  # greedy on the exact table
            cand = [(val[tuple(sorted(S + (p,)))], p) for p in top if p not in S]
            v, p = max(cand, key=lambda x: x[0])
            S = tuple(sorted(S + (p,)))
            gv[k] = v
        for k in range(1, n + 1):
            pre = tuple(sorted(top[:k]))
            res.append(dict(row_id=rid, case_id=int(t.rows.case_id.iloc[rid]), k=k, best=best[k][1], greedy10=gv[k], oracle_prefix=val[pre],
                            worst=min(v for s, v in val.items() if len(s) == k), drp=float(dref.loc[rid])))
    df = pd.DataFrame(res)
    out = {"n_rows": int(df.row_id.nunique()), "k": list(range(1, 11))}
    for col in ("best", "greedy10", "oracle_prefix", "worst"):
        M = df.groupby(["case_id", "k"])[col].mean().unstack("k")
        drp = df.groupby("case_id").drp.mean()
        rc = X.ratio_cols(M, drp, unit)
        out[col] = {"r": rc.ratio.tolist(), "lo": rc.lo.tolist(), "hi": rc.hi.tolist()}
    df["gap"] = (df.best - df.greedy10) / df.drp
    df["gap_oracle"] = (df.best - df.oracle_prefix) / df.drp
    out["greedy_gap_mean"] = df.groupby("k").gap.mean().tolist()
    out["greedy_gap_max"] = df.groupby("k").gap.max().tolist()
    out["greedy_optimal_frac"] = df.groupby("k").apply(lambda x: float((x.best - x.greedy10 <= 1e-6).mean())).tolist()
    out["oracle_prefix_gap_mean"] = df.groupby("k").gap_oracle.mean().tolist()
    out["oracle_prefix_optimal_frac"] = df.groupby("k").apply(lambda x: float((x.best - x.oracle_prefix <= 1e-6).mean())).tolist()
    # full greedy (pool 32 / 64) at size k vs the exact top-10 optimum on the same rows
    if len(gdf):
        gg = gdf.set_index("row_id")
        cmp = []
        for k in range(1, 11):
            sub = df[df.k == k]
            gv = gg.loc[sub.row_id, f"k{k}"].values
            cmp.append(float(((gv - sub.best.values) / sub.drp.values).mean()))
        out["full_greedy_minus_exact10_mean"] = cmp
    return out


def shapley_analysis(t, sh, rows, dref):
    """Shapley values from the full prefix sweeps of the random permutations (first donor): marginal gains of each
    expert, averaged over permutations. Delta(empty) = the corrupted run of the same pass; Delta(all) = the all_active row."""
    K = t.K
    aa = rows[(rows.order == "all_active") & (rows.dir == "d")].set_index("row_id")
    res, summ = [], []
    for (rid, o), g in sh.groupby(["row_id", "order"]):
        g = g.sort_values("k")
        if len(g) != K - 1:
            continue
        od = X.ordering(t, rid, o)
        d0 = float(g.delta_corrupt.iloc[0])
        seq = np.concatenate([[d0], g.delta.values, [float(aa.loc[rid].delta) if rid in aa.index else np.nan]])
        if not np.isfinite(seq[-1]):
            continue
        marg = np.diff(seq)
        for p, m in zip(od, marg):
            res.append((rid, o, p[0], p[1], float(m)))
    df = pd.DataFrame(res, columns=["row_id", "order", "layer", "expert", "marg"])
    phi = df.groupby(["row_id", "layer", "expert"]).marg.agg(["mean", "std", "count"]).reset_index()
    phi["single"] = [t.single[r][(l, e)] for r, l, e in zip(phi.row_id, phi.layer, phi.expert)]
    out = {"n_rows": int(phi.row_id.nunique()), "n_perm": int(df.order.nunique())}
    cors, top_agree, n80, add_ratio, pos_mass_top1, sd_rel = [], [], [], [], [], []
    for rid, g in phi.groupby("row_id"):
        cors.append(np.corrcoef(g["mean"], g.single)[0, 1])
        top_agree.append(int(g["mean"].idxmax() == g.single.idxmax()))
        tot = g["mean"].sum()
        srt = np.sort(g["mean"].values)[::-1]
        cs = np.cumsum(srt)
        n80.append(int(np.argmax(cs >= 0.8 * tot) + 1) if tot > 0 and (cs >= 0.8 * tot).any() else np.inf)
        add_ratio.append(g.single.sum() / tot if tot != 0 else np.nan)
        pos = srt[srt > 0].sum()
        pos_mass_top1.append(srt[0] / pos if pos > 0 else np.nan)
        sd_rel.append(float((g["std"] / np.sqrt(g["count"])).mean()))
    out.update({"corr_phi_single": X.med_iqr(cors), "top1_agree_frac": float(np.mean(top_agree)), "n_experts_80pct_phi": X.med_iqr(n80),
                "sum_single_over_sum_phi": X.med_iqr(add_ratio), "top1_phi_share_of_positive_mass": X.med_iqr(pos_mass_top1),
                "mean_se_of_phi": float(np.mean(sd_rel)),
                "phi_mean_abs_minus_single": float((phi["mean"] - phi.single).abs().mean())})
    return out


# ------------------------------------------------------------------------------------------------------------------
def write_tables(res):
    os.makedirs(TAB, exist_ok=True)
    # A0
    rows_ = []
    for key, r in res.items():
        for sub in ("validation", "all"):
            a = r["a0"][sub]
            for dr in ("d", "n"):
                x = a[dr]
                rows_.append({"run": key, "model": r["label"], "task": r["task"], "cases": sub, "direction": "add-back" if dr == "d" else "deletion",
                              "all MoE (M)": ci_s(x["all_moe"]), "all attention (A)": ci_s(x["all_attn"]), "both": ci_s(x["all_block"]),
                              "MoE <= L-5": ci_s(x["int_moe"]), "attention <= L-5": ci_s(x["int_attn"]),
                              "phi_attn": ci_s(x["phi_attn"]), "phi_MoE": ci_s(x["phi_moe"]), "A+M-1": ci_s(x["redundancy"]),
                              "per-case median M": f"{x['per_case_median']['all_moe']:.3f}", "cases M >= 0.8": f"{x['frac_cases_all_moe_ge_0.8']:.2f}",
                              "direct A / M (exact norm)": (f"{a['direct']['a_dir']['ratio']:.3f} / {a['direct']['m_dir']['ratio']:.3f}" if dr == "d" and "direct" in a else ""),
                              "DLA share attn / MoE": (f"{a['dla_split']['dla_attn']['ratio']:.3f} / {a['dla_split']['dla_moe']['ratio']:.3f}" if dr == "d" and "dla_split" in a else "")})
    save(pd.DataFrame(rows_), "ext8_a0_ceilings")
    # A1 / A3 summary
    for fam, title in (("a1", "add-back"), ("a3", "deletion")):
        rows_ = []
        for key, r in res.items():
            for donors, src in (("donor mean", fam), ("first donor", fam + "_first")):
                for o, e in r[src].items():
                    rows_.append({"run": key, "model": r["label"], "task": r["task"], "donors": donors, "ordering": o,
                                  "ceiling": f"{e['ceiling']:.3f}", "r(1)": f"{e['r'][0]:.3f}", "r(10)": f"{e['r'][e['k'].index(10)]:.3f}" if 10 in e["k"] else "",
                                  "r(all clean-active)": f"{e['r'][-1]:.3f}", "max r (k)": f"{e['max_r']:.3f} ({e['k_max_r']})",
                                  "AUC log k": f"{e['auc_log']:.3f}", "AUC k/K": f"{e['auc_lin']:.3f}",
                                  "AUC log k (of ceiling)": f"{e['auc_log_ceiling']:.3f}",
                                  "k50/80/90 ceiling": "/".join(fk(e[f"k{q}_ceiling"]) for q in (50, 80, 90)),
                                  "k50/80/90 drop": "/".join(fk(e[f"k{q}_drop"]) for q in (50, 80, 90)),
                                  "case k80 ceiling (median [IQR], reached)": f"{fk(e['case_k80_ceiling']['median'])} [{fk(e['case_k80_ceiling']['q25'])}, {fk(e['case_k80_ceiling']['q75'])}], {e['case_k80_ceiling']['frac_finite']:.2f}",
                                  ("case k answer restored" if fam == "a1" else "case k answer flipped"): _kr(e, fam),
                                  "top-1 at k=10 / all": (f"{e['top1_frac'][e['k'].index(10)]:.2f} / {e['top1_frac'][-1]:.2f}" if 10 in e["k"] else "")})
        save(pd.DataFrame(rows_), f"ext8_{fam}_summary")
        # long curves
        cr = []
        for key, r in res.items():
            for donors, src in (("mean", fam), ("first", fam + "_first")):
                for o, e in r[src].items():
                    for i, k in enumerate(e["k"]):
                        cr.append(dict(run=key, donors=donors, ordering=o, k=k, r=e["r"][i], lo=e["lo"][i], hi=e["hi"][i], r_ceiling=e["r_ceiling"][i],
                                       per_case_median=e["per_case_median"][i], top1_frac=e["top1_frac"][i]))
        pd.DataFrame(cr).to_csv(os.path.join(TAB, f"ext8_{fam}_curves.csv"), index=False)
    # A2
    rows_ = []
    for key, r in res.items():
        g = r.get("greedy")
        if not g:
            continue
        o = r["a1"].get("oracle", {})
        rows_.append({"run": key, "model": r["label"], "task": r["task"], "rows": g["n_rows"], "pool": g["pool"],
                      "greedy r(1)/r(2)/r(5)/r(10)/r(15)": "/".join(f"{g['r'][k - 1]:.3f}" for k in (1, 2, 5, 10, 15)),
                      "oracle static r(1)/r(2)/r(5)/r(10)/r(16)": "/".join(f"{o['r'][o['k'].index(k)]:.3f}" for k in (1, 2, 5, 10, 16) if k in o.get("k", [])),
                      "greedy k50/80/90 ceiling": "/".join(fk(g[f"k{q}_ceiling"]) for q in (50, 80, 90)),
                      "ceiling": f"{g['ceiling']:.3f}",
                      "most frequent in first 5 picks (share of rows)": ", ".join(f"{a} {b:.2f}" for a, b in g["in_first5_top"][:4]),
                      "beam - greedy (k=2..6, of drop)": "/".join(f"{x:+.3f}" for x in r["beam"]["beam_minus_greedy_mean"][1:]) if "beam" in r else "",
                      "exact10 - greedy10 (k=2..10)": "/".join(f"{x:+.3f}" for x in r["exact"]["greedy_gap_mean"][1:]) if "exact" in r else "",
                      "greedy10 optimal frac (k=2..10)": "/".join(f"{x:.2f}" for x in r["exact"]["greedy_optimal_frac"][1:]) if "exact" in r else "",
                      "exact10 - oracle prefix (k=2..10)": "/".join(f"{x:+.3f}" for x in r["exact"]["oracle_prefix_gap_mean"][1:]) if "exact" in r else ""})
    save(pd.DataFrame(rows_), "ext8_a2_adaptive")
    # CounterFact vs WinoGrande saturation (one row per model x task)
    rows_ = []
    for key, r in res.items():
        a = r["a0"]["validation"]
        o, p, rnd = r["a1"].get("oracle"), r["a1"].get("pop"), r["a1"].get("rand")
        g = r.get("greedy", {})
        sh = r.get("shapley", {})
        rows_.append({"model": r["label"], "task": "CounterFact" if r["task"] == "cf" else "WinoGrande", "K": r["K"],
                      "mean drop": f"{r['mean_drop_val']:.2f}", "all-MoE ceiling M": ci_s(a["d"]["all_moe"]),
                      "direct split attn / MoE": f"{a['direct']['a_dir']['ratio']:.2f} / {a['direct']['m_dir']['ratio']:.2f}" if "direct" in a else "",
                      "MoE <= L-5": f"{a['d']['int_moe']['ratio']:.3f}",
                      "AUC log k oracle / pop / rand": f"{o['auc_log']:.3f} / {p['auc_log']:.3f} / {rnd['auc_log']:.3f}" if o and p and rnd else "",
                      "AUC (of ceiling) oracle / pop": f"{o['auc_log_ceiling']:.2f} / {p['auc_log_ceiling']:.2f}" if o and p else "",
                      "k80 ceiling oracle / pop / greedy / rand": f"{fk(o['k80_ceiling'])} / {fk(p['k80_ceiling'])} / {fk(g.get('k80_ceiling', float('inf')))} / {fk(rnd['k80_ceiling'])}" if o and p and rnd else "",
                      "max r oracle (k)": f"{o['max_r']:.3f} ({o['k_max_r']})" if o else "",
                      "case k restored (oracle, median; frac)": f"{fk(o['case_k_restored']['median'])}; {o['case_k_restored']['frac_finite']:.2f}" if o else "",
                      "cases restored by all MoE": f"{r['ceiling_restores_answer_frac_cases']:.2f}",
                      "Shapley: experts for 80 %": fk(sh["n_experts_80pct_phi"]["median"]) if sh else ""})
    save(pd.DataFrame(rows_), "ext8_saturation_cf_vs_wino")
    rows_ = []
    for key, r in res.items():
        s = r.get("shapley")
        if not s:
            continue
        rows_.append({"run": key, "model": r["label"], "task": r["task"], "rows": s["n_rows"], "permutations": s["n_perm"],
                      "r(phi, single) median": f"{s['corr_phi_single']['median']:.2f}", "top-1 agree": f"{s['top1_agree_frac']:.2f}",
                      "experts for 80% of sum phi (median [IQR])": f"{fk(s['n_experts_80pct_phi']['median'])} [{fk(s['n_experts_80pct_phi']['q25'])}, {fk(s['n_experts_80pct_phi']['q75'])}]",
                      "sum single / sum phi (median)": f"{s['sum_single_over_sum_phi']['median']:.2f}",
                      "top-1 phi / positive mass": f"{s['top1_phi_share_of_positive_mass']['median']:.2f}", "mean SE of phi": f"{s['mean_se_of_phi']:.3f}"})
    if rows_:
        save(pd.DataFrame(rows_), "ext8_shapley")


def _kr(e, fam):
    k = e.get("case_k_restored" if fam == "a1" else "case_k_flipped")
    return f"{fk(k['median'])} [{fk(k['q25'])}, {fk(k['q75'])}], {k['frac_finite']:.2f}"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def save(df, name):
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    with open(os.path.join(TAB, f"{name}.md"), "w") as f:
        f.write(md_table(df))
    log(f"wrote {name} ({len(df)} rows)")


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    keys = [k for k in res if not k.startswith("smoke")] or list(res)
    colors = {"oracle": "#1f77b4", "pop": "#ff7f0e", "layerwise": "#2ca02c", "dla": "#9467bd", "vnorm": "#8c564b", "weight": "#e377c2",
              "rand": "#7f7f7f", "noise_oracle": "#17becf", "greedy": "#d62728"}
    for fam, ttl, ylab in (("a1", "add-back (denoising)", "r(k) = rescue / drop"), ("a3", "deletion (noising)", "damage / drop")):
        fig, axes = plt.subplots(1, len(keys), figsize=(5.0 * len(keys), 4.2), squeeze=False)
        for ax, key in zip(axes[0], keys):
            r = res[key]
            for o, e in r[fam].items():
                ax.plot(e["k"], e["r"], marker=".", ms=3, lw=1.4 if o in ("oracle", "pop", "noise_oracle") else 1.0, color=colors.get(o), label=o)
                if o in ("oracle", "pop", "noise_oracle"):
                    ax.fill_between(e["k"], e["lo"], e["hi"], color=colors.get(o), alpha=0.15)
            if fam == "a1" and "greedy" in r:
                g = r["greedy"]
                ax.plot(g["k"], g["r"], color=colors["greedy"], lw=1.8, label="greedy (adaptive)")
            ceil = r["a0"]["validation"]["d" if fam == "a1" else "n"]["all_moe"]["ratio"]
            ax.axhline(ceil, color="k", ls="--", lw=0.8, label=f"all MoE outputs ({ceil:.2f})")
            ax.axhline(1.0, color="k", ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel(f"k (experts patched; K = {r['K']})")
            ax.set_ylabel(ylab)
            ax.set_title(f"{r['label']}, {r['task']}: {ttl}", fontsize=9)
            ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"ext8_{fam}_curves.{ext}"), dpi=150)
        plt.close(fig)
    # A0: ceilings and direct-path split per run
    fig, ax = plt.subplots(figsize=(1.6 + 1.9 * len(keys), 3.6))
    xs = np.arange(len(keys))
    bars = (("M add-back", lambda a: a["d"]["all_moe"], "#1f77b4"), ("M deletion", lambda a: a["n"]["all_moe"], "#aec7e8"),
            ("MoE <= L-5", lambda a: a["d"]["int_moe"], "#9edae5"), ("direct MoE share", lambda a: a.get("direct", {}).get("m_dir"), "#d62728"),
            ("direct attention share", lambda a: a.get("direct", {}).get("a_dir"), "#ff9896"))
    wdt = 0.16
    for j, (lab, f, c) in enumerate(bars):
        vals, err = [], []
        for key in keys:
            v = f(res[key]["a0"]["validation"])
            vals.append(v["ratio"] if v else np.nan)
            err.append([v["ratio"] - v["lo"], v["hi"] - v["ratio"]] if v else [0, 0])
        ax.bar(xs + (j - 2) * wdt, vals, wdt, color=c, label=lab, yerr=np.array(err).T, capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{res[k]['label']}\n{'CounterFact' if res[k]['task'] == 'cf' else 'WinoGrande'}" for k in keys], fontsize=8)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_ylabel("fraction of the drop (validation)")
    ax.set_title("A0: all final-position MoE outputs (M), direct-path split\n(all attention outputs = 1 by construction)", fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext8_a0_split.{ext}"), dpi=150)
    plt.close(fig)
    # adaptive strategies, k <= 10
    ks = [k for k in keys if "exact" in res[k]]
    if ks:
        fig, axes = plt.subplots(1, len(ks), figsize=(5.0 * len(ks), 4.0), squeeze=False)
        for ax, key in zip(axes[0], ks):
            r = res[key]
            e = r["exact"]
            for col, lab, c in (("best", "exact optimum (top-10)", "k"), ("greedy10", "greedy within top-10", "#d62728"),
                                ("oracle_prefix", "static oracle prefix", "#1f77b4"), ("worst", "worst subset (top-10)", "#bbbbbb")):
                ax.plot(e["k"], e[col]["r"], marker="o", ms=3, color=c, label=lab)
            if "beam" in r:
                ax.plot(r["beam"]["size"], r["beam"]["r_beam"], marker="s", ms=3, color="#2ca02c", label="beam (width 4, pool)")
            ax.axhline(r["a0"]["validation"]["d"]["all_moe"]["ratio"], color="k", ls="--", lw=0.8)
            ax.set_xlabel("set size k")
            ax.set_ylabel("r(k) = rescue / drop (first donor)")
            ax.set_title(f"{r['label']}, {r['task']}", fontsize=9)
            ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"ext8_a2_adaptive.{ext}"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--partial", action="store_true", help="also analyse runs whose passes are not complete")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--outdir", default=None, help="write tables / figures / summary JSON here instead of results/ (testing)")
    args = ap.parse_args()
    global TAB, FIG
    summ_p = os.path.join(RESULTS, "ext8_addback_summary.json")
    if args.outdir:
        TAB = FIG = args.outdir
        os.makedirs(args.outdir, exist_ok=True)
        summ_p = os.path.join(args.outdir, "ext8_addback_summary.json")
    keys = args.runs.split(",") if args.runs else [k for k in RUNS if not k.startswith("smoke")]
    res = {}
    for key in keys:
        cfg = RUNS[key]
        od = os.path.join(RESULTS, cfg["out"])
        if not os.path.exists(os.path.join(od, "run_meta.json")):
            log(f"{key}: no run yet")
            continue
        meta = json.load(open(os.path.join(od, "run_meta.json")))
        if not meta.get("complete") and not args.partial:
            log(f"{key}: run not complete (use --partial)")
            continue
        log(f"analysing {key}")
        res[key], _, _ = analyse(key, cfg)
    if not res or args.no_write:
        print(json.dumps(res, indent=1, default=str)[:4000])
        return
    write_tables(res)
    figures(res)
    summ = json.load(open(summ_p)) if os.path.exists(summ_p) else {}
    summ.setdefault("runs", {}).update(res)
    for task, wk in (("cf", "w4_counterfact"), ("wino", "w4_winogrande")):
        w = {}
        for key, r in res.items():
            if r["task"] != task:
                continue
            a = r["a0"]["validation"]
            w[r["model"] if r["model"] == "qwen3" else "mixtral_bos"] = {
                "cases": "validation", "n_cases": r["n_cases_val"],
                "A_all_attention": a["d"]["all_attn"], "M_all_moe": a["d"]["all_moe"], "both": a["d"]["all_block"],
                "phi_attn": a["d"]["phi_attn"], "phi_moe": a["d"]["phi_moe"], "redundancy_A_plus_M_minus_1": a["d"]["redundancy"],
                "noising": {"A": a["n"]["all_attn"], "M": a["n"]["all_moe"], "both": a["n"]["all_block"], "phi_attn": a["n"]["phi_attn"],
                            "phi_moe": a["n"]["phi_moe"]},
                "direct_path_exact_norm": a.get("direct"), "dla_split": a.get("dla_split"),
                "interior_le_L-5": {"A": a["d"]["int_attn"], "M": a["d"]["int_moe"]},
                "note": "final-position patches; A = 1 by construction when the final token is shared (MoE is per-token), so "
                        "phi_MoE = M/2; the direct-path split (sum of sublayer writes, exact final norm) is the non-degenerate view"}
        if w:
            summ[wk] = w
    summ["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(summ_p, "w") as f:
        json.dump(summ, f, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x))
    log(f"wrote {summ_p}")


if __name__ == "__main__":
    main()
