"""p5-atp analysis (CPU): agreement of AtP / AtP* / EAP-IG / DLA with activation patching, routing-switch split,
cut curves, add-back curves, bf16 vs fp32 numerics.

Usage: python scripts/ext13atp_analyze.py [--tasks wino,ioi]
Outputs results/tables/ext13atp_*.{md,csv}, results/figures/ext13atp_*.png, results/ext13atp_summary.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace import ext13atp as A
from moetrace import ext13atp_analysis as N

TAB = os.path.join(A.RESULTS, "tables")
FIG = os.path.join(A.RESULTS, "figures")
MAIN_EST = ("atp", "atpstar", "ig8", "dla")


def to_md(df: pd.DataFrame, floatfmt=".3f") -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def md(df: pd.DataFrame, path: str, floatfmt=".3f"):
    df.to_csv(path + ".csv", index=False)
    with open(path + ".md", "w") as f:
        f.write(to_md(df, floatfmt))


def fmt_ci(m, lo, hi, d=2):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def sanity(task, J, cases):
    out = {}
    hc = cases.dropna(subset=["delta_corrupt_hf"])
    out["n_cases"] = int(len(cases))
    out["drop_mean"] = float(cases["drop"].mean())
    out["delta_clean_mean"] = float(cases.delta_clean.mean())
    out["eng_vs_hf_delta_corrupt_mae"] = float((hc.delta_corrupt - hc.delta_corrupt_hf).abs().mean())
    out["eng_vs_hf_delta_clean_mae"] = float((hc.delta_clean - hc.delta_clean_hf).abs().mean())
    out["null_route_eq_grad_route"] = float(hc.null_route_eq_grad_route.mean()) if "null_route_eq_grad_route" in hc else None
    out["null_route_consistent"] = float(hc.null_route_consistent.mean()) if "null_route_consistent" in hc else None
    # truth noise floor: engine (bf16 logits) vs HF (bf16 logits read-out and fp32 read-out)
    nf = {}
    for g, cts in N.CT_GROUPS.items():
        s = J[J.ctype.isin(cts)]
        nf[g] = {"r_eng_hfbf16": float(np.corrcoef(s.truth, s.truth_hf_bf16)[0, 1]),
                 "r_eng_hf32": float(np.corrcoef(s.truth, s.truth_hf)[0, 1]),
                 "r_hfbf16_hf32": float(np.corrcoef(s.truth_hf_bf16, s.truth_hf)[0, 1]),
                 "mae_eng_hf32": float((s.truth - s.truth_hf).abs().mean()), "mean_abs_truth": float(s.truth.abs().mean()),
                 "frac_eng_zero": float((s.truth == 0).mean())}
    out["truth_noise"] = nf
    # AtP linearity checks: sum over experts of a layer = MoE-layer AtP; sum over heads = attention-layer AtP
    lin = {}
    for part, whole in (("expert", "moe"), ("head", "attn")):
        a = J[J.ctype == part].groupby(["case_id", "layer"]).atp.sum()
        b = J[J.ctype == whole].set_index(["case_id", "layer"]).atp
        b = b.reindex(a.index)
        lin[f"{part}_sum_vs_{whole}"] = {"r": float(np.corrcoef(a, b)[0, 1]), "mae": float((a - b).abs().mean()),
                                          "mean_abs": float(b.abs().mean())}
    out["linearity"] = lin
    # cut 0 of the patched run vs the gradient AtP (same quantity up to bf16 rounding of the write)
    C = J[[f"c{c:02d}" for c in range(2 * N.L)]].to_numpy(dtype=np.float64)
    c0 = C[np.arange(len(J)), J.cut.to_numpy()]
    ok = J.in_hf.to_numpy() & ~np.isnan(c0)
    out["cut0_vs_atp_r"] = float(np.corrcoef(c0[ok], J.atp.to_numpy()[ok])[0, 1])
    return out


def fp32_check(task):
    od = A.run_dir(task)
    p32 = os.path.join(od, "hf_fp32_nodes.parquet")
    if not os.path.exists(p32):
        return None
    f = pd.read_parquet(p32)
    b = pd.read_parquet(os.path.join(od, "hf_nodes.parquet"))
    m = f.merge(b, on=N.KEYS, suffixes=("_32", "_16"))
    out = {"n_cases": int(f.case_id.nunique()), "n_nodes": int(len(m))}
    for est in ("atp", "ig8", "ig16", "dla"):
        x, y = m[f"{est}_32"].to_numpy(), m[f"{est}_16"].to_numpy()
        out[est] = {"r": float(np.corrcoef(x, y)[0, 1]), "rel_l2": float(np.linalg.norm(x - y) / np.linalg.norm(x)),
                    "spearman_per_case_median": float(np.median([N.SS.spearmanr(d[f"{est}_32"], d[f"{est}_16"]).statistic for _, d in m.groupby("case_id")]))}
    s32 = torch.load(os.path.join(od, "hf_grad_sample_fp32.pt"), weights_only=False)
    s16 = torch.load(os.path.join(od, "hf_grad_sample_bf16.pt"), weights_only=False)
    cos = []
    for c in s32:
        if c not in s16:
            continue
        for k in ("g_mid", "g_post"):
            a, bb = s32[c][k][0].float(), s16[c][k][0].float()  # alpha = 0, [L, H]
            cos.append((torch.nn.functional.cosine_similarity(a, bb, dim=-1)).numpy())
    if cos:
        cos = np.concatenate(cos)
        out["grad_cosine_alpha0"] = {"median": float(np.median(cos)), "min": float(cos.min())}
    out["delta_fp32_vs_bf16"] = {int(c): [float(s32[c]["delta"][0]), float(s16[c]["delta"][0])] for c in s32 if c in s16}
    pp = os.path.join(od, "hf_fp32_patch_nodes.parquet")
    if os.path.exists(pp):
        f = pd.read_parquet(pp)
        tr = pd.read_parquet(os.path.join(od, "truth_rows.parquet")).rename(columns={"rescue": "truth_eng"})[N.KEYS + ["truth_eng"]]
        m = f.merge(b, on=N.KEYS, how="outer", suffixes=("_32", "_16")).merge(tr, on=N.KEYS, how="left")
        m = m[m.case_id.isin(f.case_id.unique())]
        for c in ("atp_32", "atp_16", "rescue_hf_32", "rescue_hf_16", "truth_eng", "ig8_32", "ig8_16"):
            m[c] = m[c].fillna(0.0)
        res = {"n_cases": int(f.case_id.nunique())}
        for ct, d in m.groupby("ctype"):
            cc = lambda a, b_: float(np.corrcoef(d[a], d[b_])[0, 1])
            res[ct] = {"n": int(len(d)), "r_truth32_eng": cc("rescue_hf_32", "truth_eng"), "r_truth32_hf16": cc("rescue_hf_32", "rescue_hf_16"),
                       "r_atp32_truth32": cc("atp_32", "rescue_hf_32"), "r_atp16_truth16": cc("atp_16", "rescue_hf_16"),
                       "r_atp16_truth32": cc("atp_16", "rescue_hf_32"), "r_atp16_eng": cc("atp_16", "truth_eng"),
                       "r_ig8_32_truth32": cc("ig8_32", "rescue_hf_32"),
                       "switch_frac_32": float((d.n_switch_layers_32 > 0).mean()), "switch_frac_16": float((d.n_switch_layers_16 > 0).mean()),
                       "mean_abs_truth32": float(d.rescue_hf_32.abs().mean())}
        out["fp32_patch"] = res
    return out


def population_top(task, J, n=8):
    """Population (all 256 rows) top components by mean engine truth, with the estimators' means and ranks; per-row
    top-1 hit rate of AtP / AtP* / IG / DLA (estimate argmax == truth argmax, and within the truth's top 3)."""
    rows, hits = [], []
    nr = J.case_id.nunique()
    for ct in ("expert", "head"):
        s = J[J.ctype == ct]
        agg = s.groupby(["layer", "idx"])[["truth", "truth_hf", "atp", "atpstar", "ig8", "dla"]].sum() / nr
        rk = agg.rank(ascending=False, method="min")
        for (l, i), r in agg.sort_values("truth", ascending=False).head(n).iterrows():
            rows.append(dict(task=task, ctype=ct, comp=f"L{l}{'E' if ct == 'expert' else 'H'}{i}", truth=r.truth, atp=r.atp,
                             atpstar=r.atpstar, ig8=r.ig8, dla=r.dla, rank_truth=int(rk.loc[(l, i), "truth"]),
                             rank_atp=int(rk.loc[(l, i), "atp"]), rank_atpstar=int(rk.loc[(l, i), "atpstar"]),
                             rank_ig8=int(rk.loc[(l, i), "ig8"]), rank_dla=int(rk.loc[(l, i), "dla"])))
        for est in ("atp", "atpstar", "ig8", "dla", "truth_hf"):
            h1, h3 = [], []
            for _, d in s.groupby("case_id"):
                o = np.argsort(-d.truth.to_numpy(), kind="stable")
                a = int(np.argmax(d[est].to_numpy()))
                h1.append(a == o[0]); h3.append(a in o[:3])
            hits.append(dict(task=task, ctype=ct, est=est, top1_hit=float(np.mean(h1)), top1_in_truth_top3=float(np.mean(h3))))
    return pd.DataFrame(rows), pd.DataFrame(hits)


def by_layer(task, J):
    """Pooled Pearson r and OLS slope (estimate on engine truth) per layer band and component type."""
    out = []
    bands = {"L0-3": range(0, 4), "L4-7": range(4, 8), "L8-11": range(8, 12), "L12-15": range(12, 16)}
    for ct in ("expert", "head", "attn", "moe"):
        for b, ls in bands.items():
            d = J[(J.ctype == ct) & J.layer.isin(list(ls))]
            for est in ("atp", "atpstar", "ig8"):
                t, e = d.truth.to_numpy(), d[est].to_numpy()
                out.append(dict(task=task, ctype=ct, band=b, est=est, n=len(d), mean_abs_truth=np.abs(t).mean(),
                                pearson=np.corrcoef(t, e)[0, 1] if t.std() > 0 and e.std() > 0 else np.nan,
                                slope=np.polyfit(t, e, 1)[0] if t.std() > 0 else np.nan))
    return pd.DataFrame(out)


def scatter_fig(Js, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cts = ("expert", "head", "attn", "moe")
    fig, axes = plt.subplots(len(Js), 4, figsize=(16, 4 * len(Js)))
    axes = np.atleast_2d(axes)
    for i, (task, J) in enumerate(Js.items()):
        for j, ct in enumerate(cts):
            ax = axes[i, j]
            s = J[(J.ctype == ct) & J.in_hf]
            ns, sw = s[~s.switch], s[s.switch]
            ax.scatter(ns.truth, ns.atp, s=3, alpha=0.35, c="#4477aa", label=f"no routing switch ({len(ns)})", rasterized=True)
            ax.scatter(sw.truth, sw.atp, s=3, alpha=0.35, c="#ee6677", label=f"routing switch ({len(sw)})", rasterized=True)
            lim = np.nanpercentile(np.abs(np.r_[s.truth, s.atp]), 99.9) * 1.05 if len(s) else 1
            ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.6)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            r = np.corrcoef(s.truth, s.atp)[0, 1] if len(s) > 2 else np.nan
            ax.set_title(f"{A.TASKS[task]['label']}: {ct} (r = {r:.2f})", fontsize=10)
            ax.set_xlabel("activation patch rescue (engine, logits)")
            if j == 0:
                ax.set_ylabel("AtP estimate")
            ax.legend(fontsize=7, loc="upper left", markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cut_fig(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4))
    axes = np.atleast_1d(axes)
    for ax, (task, cc) in zip(axes, curves.items()):
        for g, d in cc.groupby("group"):
            ax.plot(d.s, d.pearson, marker="o", label=g)
        ax.set_xlabel("sublayers after the node computed exactly (0 = AtP, 2 = AtP*)")
        ax.set_ylabel("Pearson r with the HF activation patch")
        ax.set_title(A.TASKS[task]["label"])
        ax.set_ylim(0.8, 1.0)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def addback_fig(curs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pools = ("experts", "heads_experts")
    fig, axes = plt.subplots(len(curs), 2, figsize=(12, 4.2 * len(curs)))
    axes = np.atleast_2d(axes)
    col = {"oracle": "#000000", "atp": "#ee6677", "atpstar": "#aa3377", "ig8": "#228833", "dla": "#4477aa", "pop": "#ccbb44",
           "rand0": "#bbbbbb"}
    for i, (task, cur) in enumerate(curs.items()):
        for j, pool in enumerate(pools):
            ax = axes[i, j]
            for o, c in col.items():
                d = cur[(cur.pool == pool) & (cur.order == o)].sort_values("k")
                if len(d):
                    ax.plot(d.k, d["mean"], marker=".", color=c, label=o)
                    ax.fill_between(d.k, d.ci_lo, d.ci_hi, color=c, alpha=0.12)
            ce = cur[(cur.pool == "ceiling") & (cur.order == "all_moe")]
            if len(ce):
                ax.axhline(float(ce["mean"].iloc[0]), color="gray", ls="--", lw=0.8, label="all-MoE ceiling")
            ax.set_xscale("log")
            ax.set_xlabel("k components patched back (static ordering)")
            ax.set_ylabel("restored fraction of the drop r(k)")
            ax.set_title(f"{A.TASKS[task]['label']}: {pool.replace('_', ' + ')}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="wino,ioi")
    args = ap.parse_args()
    tasks = args.tasks.split(",")
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    S = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tasks": {}}
    Js, curves, curs = {}, {}, {}
    agree_rows, pool_rows, popu_rows, route_rows, ab_auc_rows = [], [], [], [], []
    pop_tops, pop_hits, bl_rows = [], [], []
    for task in tasks:
        J, cases = N.load_joined(task)
        Js[task] = J
        st = {"sanity": sanity(task, J, cases), "fp32": fp32_check(task)}
        meta = json.load(open(os.path.join(A.run_dir(task), "run_meta.json")))
        st["gpu_s"] = {k: v.get("gpu_s", v.get("gpu_s_total")) for k, v in meta.items() if isinstance(v, dict)}
        for truth_col in ("truth", "truth_hf"):
            ests = N.EST + (("truth_hf", "frozen") if truth_col == "truth" else ("truth", "frozen"))
            R = N.per_row_agreement(J, truth_col, ests)
            for metric in ("spearman", "kendall", "top5", "top10", "top3", "rec3", "rec5", "rec10"):
                if metric not in R or R[metric].notna().sum() == 0:
                    continue
                s = N.summarise_rows(R.dropna(subset=[metric]), metric)
                s.insert(0, "truth", truth_col); s.insert(0, "task", task)
                agree_rows.append(s)
            p = N.pooled_agreement(J, truth_col, ests)
            p.insert(0, "truth", truth_col); p.insert(0, "task", task)
            pool_rows.append(p)
            for split in (None,):
                q = N.population_agreement(J, truth_col, ests, split=split)
                q.insert(0, "truth", truth_col); q.insert(0, "task", task)
                popu_rows.append(q)
        rt = N.routing_table(J)
        rt.insert(0, "task", task)
        route_rows.append(rt)
        cc = N.cut_curve(J)
        curves[task] = cc
        cc.assign(task=task).to_csv(os.path.join(TAB, f"ext13atp_cut_curve_{task}.csv"), index=False)
        if os.path.exists(os.path.join(A.run_dir(task), "addback_rows.parquet")):
            cur, auc = N.addback_summary(task)
            curs[task] = cur
            cur.assign(task=task).to_csv(os.path.join(TAB, f"ext13atp_addback_curves_{task}.csv"), index=False)
            auc.insert(0, "task", task)
            ab_auc_rows.append(auc)
        pt, ph = population_top(task, J)
        pop_tops.append(pt); pop_hits.append(ph)
        bl_rows.append(by_layer(task, J))
        # switch statistics by layer (experts and heads)
        sw = J[J.in_hf].groupby(["ctype", "layer"]).switch.mean().unstack(0)
        st["switch_by_layer"] = {ct: {int(l): float(v) for l, v in sw[ct].items()} for ct in sw.columns}
        S["tasks"][task] = st
    AG = pd.concat(agree_rows)
    md(AG, os.path.join(TAB, "ext13atp_agreement_rows"))
    PO = pd.concat(pool_rows)
    md(PO, os.path.join(TAB, "ext13atp_agreement_pooled"))
    PP = pd.concat(popu_rows)
    md(PP, os.path.join(TAB, "ext13atp_agreement_population"))
    RT = pd.concat(route_rows)
    # share of AtP's squared error (vs the HF truth) removed by freezing the downstream top-k
    RT["sse_share_removed_by_freeze"] = 1 - RT.sse_smooth / RT.sse_total
    md(RT, os.path.join(TAB, "ext13atp_routing"))
    PT = pd.concat(pop_tops); md(PT, os.path.join(TAB, "ext13atp_population_top"))
    PH = pd.concat(pop_hits); md(PH, os.path.join(TAB, "ext13atp_top1_hits"))
    BL = pd.concat(bl_rows); md(BL, os.path.join(TAB, "ext13atp_by_layer"))
    S["population_top"] = PT.to_dict(orient="records")
    S["top1_hits"] = PH.to_dict(orient="records")
    S["by_layer"] = BL.to_dict(orient="records")
    # compact main table: per task x group x estimator; per-row Spearman median (engine truth), mean Spearman CI,
    # top-10 overlap (top-3 for layers), recovery@10, pooled r, population Spearman
    comp = []
    for task in tasks:
        for g in N.CT_GROUPS:
            for est in N.EST + ("truth_hf", "frozen"):
                a = AG[(AG.task == task) & (AG.group == g) & (AG.est == est) & (AG.truth == "truth")]
                ah = AG[(AG.task == task) & (AG.group == g) & (AG.est == est) & (AG.truth == "truth_hf")]
                kk = "top3" if g in ("attn", "moe", "layer (attn+moe)") else "top10"
                rk = "rec3" if kk == "top3" else "rec10"

                def get(df, metric, col="median"):
                    x = df[df.metric == metric]
                    return float(x[col].iloc[0]) if len(x) else np.nan
                po = PO[(PO.task == task) & (PO.group == g) & (PO.est == est) & (PO.truth == "truth")]
                pp = PP[(PP.task == task) & (PP.group == g) & (PP.est == est) & (PP.truth == "truth")]
                comp.append(dict(task=task, group=g, est=est,
                                 spearman_row_median=get(a, "spearman"),
                                 spearman_row_mean_ci=fmt_ci(get(a, "spearman", "mean"), get(a, "spearman", "ci_lo"), get(a, "spearman", "ci_hi")),
                                 kendall_row_median=get(a, "kendall"),
                                 topk_overlap=get(a, kk, "mean"), recovery=get(a, rk, "mean"),
                                 pearson_pooled=float(po.pearson.iloc[0]) if len(po) else np.nan,
                                 slope=float(po.slope.iloc[0]) if len(po) else np.nan,
                                 spearman_population=float(pp.spearman.iloc[0]) if len(pp) else np.nan,
                                 top10_population=float(pp[("top3" if kk == "top3" else "top10")].iloc[0]) if len(pp) else np.nan,
                                 spearman_row_median_hf=get(ah, "spearman")))
    CM = pd.DataFrame(comp)
    md(CM, os.path.join(TAB, "ext13atp_agreement_main"))
    S["main"] = CM.to_dict(orient="records")
    S["routing"] = RT.to_dict(orient="records")
    if ab_auc_rows:
        AU = pd.concat(ab_auc_rows)
        md(AU, os.path.join(TAB, "ext13atp_addback_auc"))
        S["addback_auc"] = AU.to_dict(orient="records")
        S["addback_curves"] = {t: c.to_dict(orient="records") for t, c in curs.items()}
        addback_fig(curs, os.path.join(FIG, "ext13atp_addback.png"))
    S["cut_curves"] = {t: c.to_dict(orient="records") for t, c in curves.items()}
    scatter_fig(Js, os.path.join(FIG, "ext13atp_scatter_atp.png"))
    cut_fig(curves, os.path.join(FIG, "ext13atp_cut_curve.png"))
    json.dump(S, open(os.path.join(A.RESULTS, "ext13atp_summary.json"), "w"), indent=1, default=float)
    print(CM[CM.est.isin(MAIN_EST)].to_string())
    print(RT.to_string())


if __name__ == "__main__":
    main()
