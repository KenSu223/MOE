"""ext10 step 2 analysis: joint add-back curves over heads + experts (greedy, head-only greedy, static mixed orderings)
against ext8's expert-only curves, with ceiling 1 (= all heads / all attention outputs) and the all-MoE ceiling line.

r(k) = (Delta_k - Delta_corrupt(same pass)) / drop_ref, drop_ref = Delta_clean - Delta_corrupt of the row in pass 0 of the
run; population values = ratio of means over validation cases (CounterFact: donor means; WinoGrande: directed cases) with
a percentile bootstrap over units (CounterFact cases, WinoGrande pairs). AUC = ext8_addback.auc_log over k = 1..15 (the
range of ext8's greedy) and over k = 1..20; static orderings are interpolated in log k at k = 15 / 20 (as ext8).

Writes results/tables/ext10_circuit_*.md|csv, results/figures/ext10_circuit_*.png, key "step2" of
results/ext10_circuit_summary.json.

Usage: python scripts/ext10_circuit_analyze.py [--models qwen3,mixtral] [--suffix _smoke]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import RESULTS
from moetrace import ext8_addback as X
from moetrace import ext10_circuit as C

TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
SUMMARY = os.path.join(RESULTS, "ext10_circuit_summary.json")
Q = (0.5, 0.8, 0.9)
KG = 20


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


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


def cname(c):
    return f"L{c[1]}{'H' if c[0] == 'h' else 'E'}{c[2]:03d}" if c[0] == "e" else f"L{c[1]}H{c[2]}"


def fk(x):
    return "never" if x is None or not np.isfinite(x) else f"{int(x)}"


def pop_curve(M, drop_c, unit):
    """M [cases x k] (donor-mean rescue), drop_c [cases] -> DataFrame k -> ratio, lo, hi."""
    return X.ratio_cols(M, drop_c, unit)


def interp_at(ks, r, k):
    ks = np.asarray(ks, dtype=float)
    return float(np.interp(np.log(k), np.log(ks), np.asarray(r, dtype=float)))


def auc_upto(ks, r, kmax):
    gk = np.arange(1, kmax + 1)
    return X.auc_log(gk, np.interp(np.log(gk), np.log(np.asarray(ks, dtype=float)), np.asarray(r, dtype=float)))


def k_first(ks, r, thr):
    for k, v in zip(ks, r):
        if v >= thr:
            return float(k)
    return float("inf")


def ext8_curves(c, key, vrows, case_of, case_ser, ucases, unit, to_case, curves):
    """ext8's expert-only greedy on the same rows (normalised by ext8's own pass-0 drop) and its static expert curves."""
    e8s = X.load_state(os.path.join(RESULTS, c["ext8"], "addback_state.pkl"))
    p8 = pd.read_parquet(os.path.join(RESULTS, c["ext8"], "addback_prefill_p00.parquet"))
    d8c = p8[p8.kind == "clean"].set_index("case_id").delta
    d8j = p8[p8.kind == "corrupt"].set_index("row_id").delta
    drop8 = pd.Series({r_: float(d8c.loc[case_of[r_]] - d8j.loc[r_]) for r_ in vrows})
    e8m = pd.DataFrame({r_: e8s.greedy[r_].resc for r_ in vrows}).T
    e8m.columns = np.arange(1, e8m.shape[1] + 1)
    drop8_c = drop8.groupby(case_ser.loc[drop8.index].values).mean().loc[ucases]
    curves["greedy_expert_ext8"] = pop_curve(to_case(e8m), drop8_c, unit)
    e8c = pd.read_csv(os.path.join(TAB, "ext8_a1_curves.csv"))
    for o in ("oracle", "dla"):
        cc = e8c[(e8c.run == key) & (e8c.donors == "mean") & (e8c.ordering == o)].sort_values("k")
        curves[f"static_expert_{o}_ext8"] = pd.DataFrame(dict(ratio=cc.r.values, lo=cc.lo.values, hi=cc.hi.values), index=cc.k.values)


def analyse_task(key, model_dir, st, rows_all, pf0):
    c = C.cfg(key)
    t = C.load_task(key)
    cases = t.cases.set_index("case_id")
    unit = cases.unit
    R = rows_all[rows_all.task == key]
    P = pf0[pf0.task == key]
    dclean = P[P.kind == "clean"].set_index("case_id").delta
    dcor = P[P.kind == "corrupt"].set_index("row_id").delta
    vrows = sorted(int(x) for x in P[P.kind == "corrupt"].row_id)
    case_of = {r_: int(t.rows.case_id.iloc[r_]) for r_ in vrows}
    drop_ref = pd.Series({r_: float(dclean.loc[case_of[r_]] - dcor.loc[r_]) for r_ in vrows})
    first = t.rows.set_index("row_id")["first"]
    ucases = list(dict.fromkeys(case_of[r_] for r_ in vrows))
    drop_c = drop_ref.groupby(pd.Series(case_of)).mean().loc[ucases]
    out = {"key": key, "label": c["label"], "task": c["task_label"], "n_rows": len(vrows), "n_cases": len(ucases),
           "mean_drop": float(drop_c.mean())}
    # ---------------- ceilings and sanity
    ce = R[R.fam == "ceil"].copy()
    ce["rescue"] = ce.delta - ce.delta_corrupt
    cm = ce.groupby(["case_id", "order"]).rescue.mean().unstack("order").loc[ucases]
    rc = X.ratio_cols(cm, drop_c, unit)
    out["ceilings"] = {k: {"ratio": float(v.ratio), "lo": float(v.lo), "hi": float(v.hi)} for k, v in rc.iterrows()}
    ah = ce[ce.order == "all_heads"]
    out["sanity_all_heads"] = {"max_abs_dev": float((ah.delta - ah.delta_clean).abs().max()),
                               "median_abs_dev": float((ah.delta - ah.delta_clean).abs().median()),
                               "frac_exact": float(((ah.delta - ah.delta_clean).abs() < 1e-6).mean())}
    sa = R[R.fam == "sanity"].copy()
    if len(sa):
        sa["layer"] = sa.set.str.split(":").str[1].astype(int)
        pv = sa.pivot_table(index=["row_id", "layer"], columns="order", values="delta", aggfunc="first").dropna()
        dv = (pv["layer_heads"] - pv["attn_layer"]).abs()
        out["sanity_layer_heads"] = {"n": int(len(pv)), "max_abs_dev": float(dv.max()), "median_abs_dev": float(dv.median()),
                                     "frac_within_0.25": float((dv <= 0.25).mean()), "frac_exact": float((dv < 1e-6).mean()),
                                     "r": float(np.corrcoef(pv["layer_heads"], pv["attn_layer"])[0, 1])}
    case_ser = pd.Series(case_of)

    def to_case(Mrow):  # rows x k DataFrame -> cases x k (donor mean)
        return Mrow.groupby(case_ser.loc[Mrow.index].values).mean().loc[ucases]

    curves = {}
    # ---------------- greedy (mixed, head-only) from the state
    G = {}
    for kind in C.GREEDY_KINDS:
        res_m = pd.DataFrame({r_: st.greedy[(key, kind, r_)].resc[:KG] for r_ in vrows}).T
        res_m.columns = np.arange(1, res_m.shape[1] + 1)
        val_m = pd.DataFrame({r_: st.greedy[(key, kind, r_)].vals[:KG] for r_ in vrows}).T
        val_m.columns = res_m.columns
        G[kind] = (res_m, val_m)
        pc = pop_curve(to_case(res_m), drop_c, unit)
        curves[f"greedy_{kind}"] = pc
    # ext8 expert-only greedy (same rows; its own pass-0 drop)
    if c.get("ext8"):
        ext8_curves(c, key, vrows, case_of, case_ser, ucases, unit, to_case, curves)
    out["expert_single_is_dla"] = not bool(t.single)
    # ---------------- static orderings
    sr = R[R.fam == "static"].copy()
    sr["rescue"] = sr.delta - sr.delta_corrupt
    for o in C.STATIC_ORDERS:
        M = sr[sr.order == o].pivot_table(index="row_id", columns="k", values="rescue", aggfunc="first").loc[vrows]
        curves[f"static_{o}"] = pop_curve(to_case(M), drop_c, unit)
    # ---------------- curve summaries
    cs = {}
    for name, pc in curves.items():
        ks = np.array(pc.index, dtype=float)
        r = pc.ratio.values
        e = {"k": [int(k) for k in ks], "r": [float(v) for v in r], "lo": [float(v) for v in pc.lo.values], "hi": [float(v) for v in pc.hi.values],
             "auc_log_k15": auc_upto(ks, r, 15) if ks.max() >= 15 else None, "auc_log_k20": auc_upto(ks, r, 20) if ks.max() >= 20 else None,
             "k_q": {str(q): k_first(ks, r, q) for q in Q}, "max_r": float(r.max()), "k_max_r": int(ks[int(np.argmax(r))])}
        for kk in (1, 2, 5, 10, 15, 20, 32, 64, 128):
            if kk in pc.index:
                e[f"r{kk}"] = {"ratio": float(pc.ratio.loc[kk]), "lo": float(pc.lo.loc[kk]), "hi": float(pc.hi.loc[kk])}
        cs[name] = e
    out["curves"] = cs
    # ---------------- per-case / per-row metrics for the greedy curves
    for kind in C.GREEDY_KINDS:
        res_m, val_m = G[kind]
        rr = res_m.div(drop_ref.loc[res_m.index], axis=0)
        rc_ = to_case(res_m).div(drop_c, axis=0)
        vc = to_case(val_m)
        ks = np.arange(1, res_m.shape[1] + 1)
        e = cs[f"greedy_{kind}"]
        e["case_k_q"] = {str(q): X.med_iqr(X.first_k(ks, rc_.values, q)) for q in Q}
        e["case_k_restored"] = X.med_iqr(X.first_k(ks, vc.values, 1e-9))
        e["frac_cases_restored"] = {str(k): float((vc[k] > 0).mean()) for k in (1, 5, 10, 20) if k in vc.columns}
        e["frac_rows_r_ge_0.9"] = {str(k): float((rr[k] >= 0.9).mean()) for k in (5, 10, 20) if k in rr.columns}
        e["frac_cases_r_ge_0.9"] = {str(k): float((rc_[k] >= 0.9).mean()) for k in (5, 10, 20) if k in rc_.columns}
        e["frac_rows_restored"] = {str(k): float((val_m[k] > 0).mean()) for k in (1, 5, 10, 20) if k in val_m.columns}
        # first-donor sensitivity (CounterFact)
        if c["task"] == "cf":
            fr = [r_ for r_ in vrows if bool(first.loc[r_])]
            fm = res_m.loc[fr]
            fc = pd.DataFrame(fm.values, index=[case_of[r_] for r_ in fr], columns=fm.columns)
            fd = pd.Series([drop_ref.loc[r_] for r_ in fr], index=fc.index)
            pcf = X.ratio_cols(fc, fd, unit)
            e["first_donor"] = {"r20": float(pcf.ratio.iloc[-1]), "r10": float(pcf.ratio.loc[10]) if 10 in pcf.index else None, "k_q": {str(q): k_first(pcf.index, pcf.ratio.values, q) for q in Q}}
        # composition and recurrence
        comp = {}
        picks = {r_: st.greedy[(key, kind, r_)].S[:KG] for r_ in vrows}
        for kk in (1, 3, 5, 10, 20):
            nh = [sum(1 for c_ in p[:kk] if c_[0] == "h") for p in picks.values()]
            comp[str(kk)] = {"mean_heads": float(np.mean(nh)), "mean_experts": float(kk - np.mean(nh)),
                             "frac_rows_first_pick_head": float(np.mean([p[0][0] == "h" for p in picks.values()])) if kk == 1 else None}
        lay_h = [c_[1] for p in picks.values() for c_ in p[:10] if c_[0] == "h"]
        lay_e = [c_[1] for p in picks.values() for c_ in p[:10] if c_[0] == "e"]
        comp["layers_first10"] = {"heads": X.med_iqr(lay_h) if lay_h else None, "experts": X.med_iqr(lay_e) if lay_e else None}
        e["composition"] = comp
        cnt = {}
        for p in picks.values():
            for c_ in set(p[:10]):
                cnt[c_] = cnt.get(c_, 0) + 1
        top = sorted(cnt.items(), key=lambda kv: -kv[1])[:12]
        e["recurring_first10"] = [{"comp": cname(c_), "kind": c_[0], "share": v / len(picks)} for c_, v in top]
        cnt5 = {}
        for p in picks.values():
            for c_ in set(p[:5]):
                cnt5[c_] = cnt5.get(c_, 0) + 1
        e["recurring_first5"] = [{"comp": cname(c_), "kind": c_[0], "share": v / len(picks)} for c_, v in
                                 sorted(cnt5.items(), key=lambda kv: -kv[1])[:8]]
        # mean marginal gain per step by component kind (mixed greedy)
        if kind == "mix":
            gains_h, gains_e = [], []
            for r_ in vrows:
                g = st.greedy[(key, kind, r_)]
                rs = np.array(g.resc) / drop_ref.loc[r_]
                for j in range(1, min(KG, len(g.S))):
                    (gains_h if g.S[j][0] == "h" else gains_e).append(rs[j] - rs[j - 1])
            e["marginal_gain"] = {"heads": X.med_iqr(gains_h) if gains_h else None, "experts": X.med_iqr(gains_e) if gains_e else None}
    # top-1 at the last greedy step (metrics of the chosen expansion rows)
    gr = R[R.fam.str.startswith("greedy")]
    for kind in C.GREEDY_KINDS:
        g20 = []
        sub = gr[gr.fam == f"greedy_{kind}"]
        idx = sub.set_index(["row_id", "set"])
        for r_ in vrows:
            g = st.greedy[(key, kind, r_)]
            s = C.set_str(tuple(g.S[:KG]))
            try:
                rk = idx.loc[(r_, s)].rank_true
                rk = float(rk.iloc[-1]) if hasattr(rk, "iloc") else float(rk)
            except KeyError:
                rk = np.nan
            g20.append(rk)
        g20 = np.array(g20)
        cs[f"greedy_{kind}"]["frac_rows_top1_k20"] = float(np.nanmean(g20 == 1))
    cl = pf0[(pf0.task == key) & (pf0.kind == "clean")]
    out["clean_top1_rate"] = float((cl.rank_true == 1).mean())
    return out


def analyse_model(model, suffix=""):
    od = C.model_dir(model) + suffix
    st = C.load_cstate(os.path.join(od, "circuit_state.pkl"))
    rows = C._cat(od, "circuit_rows_p")
    pf = C._cat(od, "circuit_prefill_p")
    pf0 = pf[pf.pass_ == pf.pass_.min()]
    meta = json.load(open(os.path.join(od, "run_meta.json")))
    # pass-to-pass numerics of the prefill rows
    sdc = pf[pf.kind == "clean"].groupby(["task", "case_id"]).delta.std()
    sdj = pf[pf.kind == "corrupt"].groupby(["task", "row_id"]).delta.std()
    res = {}
    for key in meta["tasks"]:
        log(f"analysing {key}")
        res[key] = analyse_task(key, od, st, rows, pf0)
        res[key]["passes"] = int(pf.pass_.nunique())
        res[key]["prefill_sd"] = {"clean_median": float(sdc.loc[key].median()), "corrupt_median": float(sdj.loc[key].median()),
                                  "clean_p99": float(sdc.loc[key].quantile(0.99)), "corrupt_p99": float(sdj.loc[key].quantile(0.99))}
    notes = [n for n in st.notes if "pass" in n]
    gpu_s = float(sum(n.get("pass_s", 0) for n in notes))
    return res, {"passes": len(notes), "pass_s_sum": gpu_s, "peak_GB_max": float(max(n.get("peak_GB", 0) for n in notes)) if notes else None}


def tables(res, suffix=""):
    tb = []
    for key, r in res.items():
        cs = r["curves"]
        ce = r["ceilings"]
        for name in ("greedy_mix", "greedy_head", "greedy_expert_ext8", "static_mix_oracle", "static_mix_dla", "static_head_oracle",
                     "static_head_dla", "static_expert_oracle_ext8", "static_expert_dla_ext8"):
            e = cs.get(name)
            if e is None:
                continue
            rk = lambda kk: f"{e[f'r{kk}']['ratio']:.3f}" if f"r{kk}" in e else ""
            tb.append({"Run": f"{r['label']}, {r['task']}", "Curve": name, "r(1)": rk(1), "r(5)": rk(5), "r(10)": rk(10),
                       "r(20) [95% CI]": f"{e['r20']['ratio']:.3f} [{e['r20']['lo']:.3f}, {e['r20']['hi']:.3f}]" if "r20" in e else
                       (f"{e['r15']['ratio']:.3f} (k=15)" if "r15" in e else ""),
                       "r(128)": rk(128), "max r (k)": f"{e['max_r']:.3f} ({e['k_max_r']})",
                       "k50/80/90 of drop": "/".join(fk(e["k_q"][str(q)]) for q in Q),
                       "AUC log k (1..15)": "" if e["auc_log_k15"] is None else f"{e['auc_log_k15']:.3f}",
                       "AUC log k (1..20)": "" if e["auc_log_k20"] is None else f"{e['auc_log_k20']:.3f}",
                       "all-MoE ceiling": f"{ce['all_moe']['ratio']:.3f}"})
    save(pd.DataFrame(tb), f"ext10_circuit_curves{suffix}")
    gt = []
    for key, r in res.items():
        for kind in C.GREEDY_KINDS:
            e = r["curves"][f"greedy_{kind}"]
            cq = e["case_k_q"]
            comp = e["composition"]
            gt.append({"Run": f"{r['label']}, {r['task']}", "Greedy": kind,
                       "case k50/80/90 (median; reached)": "/".join(f"{fk(cq[str(q)]['median'])}" for q in Q) + "; " +
                       "/".join(f"{cq[str(q)]['frac_finite']:.2f}" for q in Q),
                       "case k answer restored (median; reached)": f"{fk(e['case_k_restored']['median'])}; {e['case_k_restored']['frac_finite']:.2f}",
                       "cases restored k=1/5/10/20": "/".join(f"{e['frac_cases_restored'].get(str(k), float('nan')):.2f}" for k in (1, 5, 10, 20)),
                       "rows r ≥ 0.9 k=5/10/20": "/".join(f"{e['frac_rows_r_ge_0.9'].get(str(k), float('nan')):.2f}" for k in (5, 10, 20)),
                       "rows top-1 at k=20": f"{e['frac_rows_top1_k20']:.2f}",
                       "heads among first 1/5/10/20": "/".join(f"{comp[str(k)]['mean_heads']:.1f}" for k in (1, 5, 10, 20)),
                       "most frequent in first 5 (share of rows)": ", ".join(f"{x['comp']} {x['share']:.2f}" for x in e["recurring_first5"][:5])})
    save(pd.DataFrame(gt), f"ext10_circuit_greedy{suffix}")
    sn = []
    for key, r in res.items():
        ce = r["ceilings"]
        sl = r.get("sanity_layer_heads", {})
        sn.append({"Run": f"{r['label']}, {r['task']}", "rows / cases": f"{r['n_rows']} / {r['n_cases']}",
                   "all heads (= 1)": f"{ce['all_heads']['ratio']:.3f}", "all attention": f"{ce['all_attn']['ratio']:.3f}",
                   "all MoE": f"{ce['all_moe']['ratio']:.3f} [{ce['all_moe']['lo']:.3f}, {ce['all_moe']['hi']:.3f}]",
                   "all heads: max |Δ − Δ_clean|": f"{r['sanity_all_heads']['max_abs_dev']:.3f}",
                   "layer heads vs attn_layer: median / max |ΔΔ|, r": f"{sl.get('median_abs_dev', float('nan')):.3f} / {sl.get('max_abs_dev', float('nan')):.3f}, {sl.get('r', float('nan')):.4f}",
                   "prefill SD across passes (median clean / corrupt)": f"{r['prefill_sd']['clean_median']:.3f} / {r['prefill_sd']['corrupt_median']:.3f}"})
    save(pd.DataFrame(sn), f"ext10_circuit_sanity{suffix}")


def figures(res, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(res)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2), squeeze=False)
    sty = {"greedy_mix": ("C3", "-", "greedy heads+experts"), "greedy_head": ("C0", "-", "greedy heads"),
           "greedy_expert_ext8": ("C2", "-", "greedy experts (ext8)"), "static_mix_oracle": ("C3", "--", "oracle heads+experts"),
           "static_mix_dla": ("C3", ":", "DLA heads+experts"), "static_head_oracle": ("C0", "--", "oracle heads"),
           "static_head_dla": ("C0", ":", "DLA heads"), "static_expert_oracle_ext8": ("C2", "--", "oracle experts (ext8)"),
           "static_expert_dla_ext8": ("C2", ":", "DLA experts (ext8)")}
    for ax, k in zip(axes[0], keys):
        r = res[k]
        for name, (col, ls, lab) in sty.items():
            e = r["curves"].get(name)
            if e is None:
                continue
            ax.plot(e["k"], e["r"], color=col, ls=ls, lw=1.4 if name.startswith("greedy") else 1.0, label=lab)
            if name.startswith("greedy"):
                ax.fill_between(e["k"], e["lo"], e["hi"], color=col, alpha=0.12)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.axhline(r["ceilings"]["all_moe"]["ratio"], color="C2", lw=0.8, ls="-.", label="all-MoE ceiling")
        ax.set_xscale("log")
        ax.set_xlabel("components patched jointly (k)")
        ax.set_ylabel("r(k) = rescue / drop")
        ax.set_title(f"{r['label']}\n{r['task']}", fontsize=9)
        ax.set_ylim(min(-0.05, r["ceilings"]["all_moe"]["ratio"] - 0.05), 1.15)
    axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"ext10_circuit_curves{suffix}.png"), dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        e = res[k]["curves"]["greedy_mix"]["composition"]
        ks = [1, 3, 5, 10, 20]
        h = [e[str(x)]["mean_heads"] for x in ks]
        x_ = [e[str(x)]["mean_experts"] for x in ks]
        ax.bar(range(len(ks)), h, color="C0", label="heads")
        ax.bar(range(len(ks)), x_, bottom=h, color="C2", label="experts")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(x) for x in ks])
        ax.set_xlabel("first k greedy picks")
        ax.set_ylabel("mean count")
        ax.set_title(f"{res[k]['label']}, {res[k]['task']}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"ext10_circuit_composition{suffix}.png"), dpi=140)
    plt.close(fig)
    log("figures written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3,mixtral")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    res, runs = {}, {}
    for m in args.models.split(","):
        od = C.model_dir(m) + args.suffix
        if not os.path.exists(os.path.join(od, "circuit_state.pkl")):
            log(f"{m}: no step-2 run, skipped")
            continue
        r, info = analyse_model(m, args.suffix)
        res.update(r)
        runs[m] = info
    order = [k for k in C.RUN_ORDER if k in res]
    res = {k: res[k] for k in order}
    tables(res, args.suffix)
    figures(res, args.suffix)
    path = SUMMARY if not args.suffix else os.path.join(RESULTS, f"ext10_circuit_summary{args.suffix}.json")
    cur = json.load(open(path)) if os.path.exists(path) else {}
    cur.setdefault("step2", {}).update(res)
    cur.setdefault("step2_runs", {}).update(runs)
    cur["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "w") as f:
        json.dump(cur, f, indent=1, default=float)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
