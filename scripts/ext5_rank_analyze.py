"""Phase 2 / F1.1-F1.2 driver (ext5-analysis): expert rankings under several metrics and population-level minimal sets
under the additive approximation, from the existing all-layer expert passes (zero GPU).

Usage: python scripts/ext5_rank_analyze.py [--runs qwen3_bos_alllayers,mixtral_bos_alllayers,mixtral_nobos_alllayers] [--section-only]
Writes results/tables/ext5_rank_*.{csv,md}, results/figures/ext5_rank_minimal_<run>.{png,pdf}, results/ext5_rank_summary.json and
results/sections/ext5_f1_rankings.md (embedding results/tables/ext5_rank_subsets_section.md from ext5_rank_subsets.py when present).
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import analysis as A
from moetrace import ext1_analysis as X
from moetrace import ext5_rank as R
from moetrace.models import RESULTS
from moetrace.report import md_table

TAB, FIG, SEC = (os.path.join(RESULTS, d) for d in ("tables", "figures", "sections"))
for d in (TAB, FIG, SEC):
    os.makedirs(d, exist_ok=True)
SUMMARY = os.path.join(RESULTS, "ext5_rank_summary.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def f2(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def ci(m, lo, hi):
    return "n/a" if m is None or (isinstance(m, float) and np.isnan(m)) else f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def cis(s):
    return ci(s["mean"], s["ci_lo"], s["ci_hi"])


def rk(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else str(int(x))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ---------------------------------------------------------------------------------------------------------------
def analyze_run(run: str) -> dict:
    cfg = R.RUNS[run]
    short, nc = cfg["short"], cfg["n_controls"]
    md = A.load_model(run)
    cache = X.ExpertCache(md)
    disc, val = md.ids("paper", "discovery"), md.ids("paper", "validation")
    th = len(disc) // 2
    out = dict(run=run, short=short, label=cfg["label"], model=cfg["model"], n_controls=nc, n_disc=len(disc), n_val=len(val), threshold=th,
               two_stage=R.pair(*cfg["two_stage"]), second=R.pair(*cfg["second"]), n_layers=len(cache.layers))

    # ---- F1.1 full ranking
    log(run, "full ranking over", len(cache.layers), "layers")
    df = R.full_ranking(md, cache, "paper", nc, th)
    df.to_csv(os.path.join(TAB, f"ext5_rank_all_{short}.csv"), index=False)
    rec = df[df.recurrent]
    out["n_pairs"] = int(len(df)); out["n_recurrent"] = int(len(rec)); out["n_recurrent_block_pos"] = int(rec.block_clearly_positive.sum())
    top = rec.sort_values("val_rescue", ascending=False).head(30)
    rows = []
    for _, r in top.iterrows():
        rows.append([int(r.rank_val_rescue), r.pair, f"{int(r.disc_active)}/{int(r.disc_active) and out['n_disc']}", f"{int(r.val_active)}/{out['n_val']}",
                     ci(r.val_rescue, r.val_rescue_lo, r.val_rescue_hi), f3(r.val_active_only), ci(r.val_spec, r.val_spec_lo, r.val_spec_hi),
                     f"{f3(r.block_val)} / " + ("n/a" if pd.isna(r.block_share) else f"{100 * r.block_share:.0f}%"), f"{r.mean_rank:.2f} / {100 * r.mean_percentile:.0f}% / {100 * r.top1_frac:.0f}%",
                     " / ".join(("n/a" if pd.isna(r[f'rank_{m}']) else f"{int(r[f'rank_{m}'])}") for m in ("val_active_only", "val_spec", "block_share", "mean_percentile", "disc_allcase"))])
    md_table(["Rank (val. rescue)", "Pair", "Disc. active", "Val. active", "Val. rescue [95% CI]", "Active-only", "Spec [95% CI]", "Block rescue / share",
              "Mean rank / percentile / top-1", "Rank under: active-only / Spec / share / percentile / disc."],
             rows, os.path.join(TAB, f"ext5_rank_top30_{short}"),
             caption=f"{cfg['label']}: top-30 recurrent (layer, expert) pairs by validation all-case rescue, paper set ({len(rec)} recurrent pairs of {len(df)} with any activity; "
                     f"threshold {th}/{out['n_disc']} discovery cases). Ranks are among the recurrent pairs (1 = best); 'mean rank' is the mean per-case rank of the expert among "
                     f"that case's clean-active experts on validation.")
    # top-30 by Spec too (short)
    tops = rec.sort_values("val_spec", ascending=False).head(15)
    md_table(["Rank (Spec)", "Pair", "Val. active", "Spec [95% CI]", "Val. rescue [95% CI]", "Rank (val. rescue)", "Block share"],
             [[int(r.rank_val_spec), r.pair, f"{int(r.val_active)}/{out['n_val']}", ci(r.val_spec, r.val_spec_lo, r.val_spec_hi), ci(r.val_rescue, r.val_rescue_lo, r.val_rescue_hi),
               int(r.rank_val_rescue), ("n/a" if pd.isna(r.block_share) else f"{100 * r.block_share:.0f}%")] for _, r in tops.iterrows()],
             os.path.join(TAB, f"ext5_rank_top15_spec_{short}"), caption=f"{cfg['label']}: top-15 recurrent pairs by validation Spec")

    # ---- Kendall tau
    T_all, P_all = R.kendall_matrix(rec)
    T_pos, _ = R.kendall_matrix(rec[rec.block_clearly_positive])
    T_eff, _ = R.kendall_matrix(rec[rec.rescue_clearly_positive], [m for m in R.METRICS if m != "block_share"])
    out["n_recurrent_rescue_pos"] = int(rec.rescue_clearly_positive.sum())
    T_all.to_csv(os.path.join(TAB, f"ext5_rank_kendall_{short}.csv")); T_pos.to_csv(os.path.join(TAB, f"ext5_rank_kendall_blockpos_{short}.csv"))
    T_eff.to_csv(os.path.join(TAB, f"ext5_rank_kendall_rescuepos_{short}.csv"))
    within = {}
    for L in (cfg["two_stage"][0], cfg["second"][0]):
        sub = df[(df.layer == L) & (df.val_active >= max(5, out["n_val"] // 25))]
        T_L, _ = R.kendall_matrix(sub, [m for m in R.METRICS if m != "block_share"])
        within[L] = dict(n=int(len(sub)), tau=T_L)
    rows = []
    labels = {m: R.METRIC_LABEL[m] for m in R.METRICS}
    for a in R.METRICS:
        rows.append([labels[a]] + [f2(T_all.loc[a, b]) for b in R.METRICS] + [f2(T_pos.loc[a, b]) for b in R.METRICS if b in ("val_rescue", "val_spec", "block_share")]
                    + [f2(T_eff.loc[a, "val_rescue"]) if a != "block_share" else "n/a", f2(T_eff.loc[a, "val_spec"]) if a != "block_share" else "n/a"]
                    + [f2(within[cfg['two_stage'][0]]["tau"].loc[a, "val_rescue"]) if a != "block_share" else "n/a",
                       f2(within[cfg['two_stage'][0]]["tau"].loc[a, "val_spec"]) if a != "block_share" else "n/a"])
    md_table(["Metric"] + [f"tau vs {m}" for m in ("rescue", "active-only", "Spec", "share", "percentile", "disc.")] + ["tau vs rescue (block>0 layers)", "tau vs Spec (block>0)", "tau vs share (block>0)",
              "tau vs rescue (rescue CI>0)", "tau vs Spec (rescue CI>0)", f"within L{cfg['two_stage'][0]}: tau vs rescue", f"within L{cfg['two_stage'][0]}: tau vs Spec"],
             rows, os.path.join(TAB, f"ext5_rank_kendall_{short}"),
             caption=f"{cfg['label']}: Kendall tau-b between metric orderings over the {len(rec)} recurrent pairs (all layers; block share is defined only in layers whose block rescue CI excludes zero), "
                     f"over the {int(rec.block_clearly_positive.sum())} recurrent pairs in such layers, over the {int(rec.rescue_clearly_positive.sum())} recurrent pairs whose own rescue CI excludes zero, "
                     f"and within L{cfg['two_stage'][0]} over its {within[cfg['two_stage'][0]]['n']} experts with >= {max(5, out['n_val'] // 25)} validation-active cases")
    out["kendall_all"] = T_all.to_dict(); out["kendall_blockpos"] = T_pos.to_dict(); out["kendall_rescuepos"] = T_eff.to_dict()
    out["kendall_within"] = {str(L): dict(n=v["n"], tau=v["tau"].to_dict()) for L, v in within.items()}
    dis = R.disagreements(df)
    dis.to_csv(os.path.join(TAB, f"ext5_rank_disagreements_{short}.csv"), index=False)
    rows = []
    for _, r in dis.iterrows():
        rows.append([r.pair, f"{int(r.val_active)}/{out['n_val']}", f3(r.val_rescue), f3(r.val_active_only), f3(r.val_spec), f"{f3(r.block_val)} / " + ("n/a" if pd.isna(r.block_share) else f"{100 * r.block_share:.0f}%"), f"{100 * r.mean_percentile:.0f}%"]
                    + [("n/a" if pd.isna(r[f"rank_{m}"]) else int(r[f"rank_{m}"])) for m in R.METRICS] + [int(r.rank_spread), r.best_metric, r.worst_metric])
    md_table(["Pair", "Val. active", "Rescue", "Active-only", "Spec", "Block / share", "Percentile", "rk rescue", "rk active-only", "rk Spec", "rk share", "rk percentile", "rk disc.", "Spread", "Best under", "Worst under"],
             rows, os.path.join(TAB, f"ext5_rank_disagreements_{short}"),
             caption=f"{cfg['label']}: union of the top-10 recurrent pairs under each metric, with their rank under every metric (among {len(rec)} recurrent pairs)")
    out["disagreements"] = dis[["pair", "val_active", "val_rescue", "val_active_only", "val_spec", "block_val", "block_share", "mean_percentile", "rank_spread", "best_metric", "worst_metric"] + [f"rank_{m}" for m in R.METRICS]].to_dict("records")
    out["top_by_metric"] = {m: rec.sort_values(m, ascending=False).head(5).pair.tolist() for m in R.METRICS}
    ts = df[(df.layer == cfg["two_stage"][0]) & (df.expert == cfg["two_stage"][1])].iloc[0]
    sc = df[(df.layer == cfg["second"][0]) & (df.expert == cfg["second"][1])].iloc[0]
    out["two_stage_row"] = ts.to_dict(); out["second_row"] = sc.to_dict()

    # ---- F1.2 additive minimal sets
    log(run, "additive minimal sets")
    add_rows, size_rows, cov_rows, curves = [], [], [], {}
    out["additivity"] = {}; out["sizes"] = {}
    for L in cfg["minimal_layers"]:
        ac = R.additivity_check(md, cache, L, val)
        out["additivity"][str(L)] = ac
        add_rows.append([f"L{L}", f3(ac["sum_singles"]), f3(ac["coalition_clean"]), f3(ac["coalition_union"]), f3(ac["block"]), f2(ac["r_sum_vs_coalition"]), f2(ac["r_sum_vs_block"]),
                         f"{ac['mean_abs_diff']:.3f} / {ac['median_abs_diff']:.3f}", f"{100 * ac['frac_within_025']:.0f}% / {100 * ac['frac_within_05']:.0f}%", cis(ac["diff_summary"])])
        cur = R.additive_curve(cache, L, disc, val)
        cov = R.coverage_curve(cache, L, disc, val, 0.8)
        cur.to_csv(os.path.join(TAB, f"ext5_rank_minimal_curve_{short}_L{L}.csv"), index=False)
        cov.to_csv(os.path.join(TAB, f"ext5_rank_coverage_curve_{short}_L{L}.csv"), index=False)
        curves[L] = dict(additive=cur, coverage=cov)
        sz = R.size_for_targets(cur, "frac_val"); szi = R.size_for_targets(cur, "frac_val_insample"); szd = R.size_for_targets(cur, "frac_disc")
        cvs = R.size_for_targets(cov, "cov_val", (0.5, 0.8)); cvd = R.size_for_targets(cov, "cov_disc", (0.5, 0.8))
        peak = cur.loc[cur.frac_val.idxmax()]
        out["sizes"][str(L)] = dict(val=sz, insample=szi, disc=szd, coverage_val=cvs, coverage_disc=cvd, n_experts=int(len(cur)), peak_k=int(peak.k), peak_frac=float(peak.frac_val),
                                    order_top8=cur.head(8).pair.tolist(), coverage_top8=cov.head(8).pair.tolist(), frac_at_1=float(cur.frac_val.iloc[0]), frac_at_2=float(cur.frac_val.iloc[1]) if len(cur) > 1 else np.nan,
                                    frac_at_4=float(cur.frac_val.iloc[3]) if len(cur) > 3 else np.nan, frac_at_8=float(cur.frac_val.iloc[7]) if len(cur) > 7 else np.nan,
                                    cov_at_1=float(cov.cov_val.iloc[0]), cov_at_2=float(cov.cov_val.iloc[1]) if len(cov) > 1 else np.nan, cov_at_4=float(cov.cov_val.iloc[3]) if len(cov) > 3 else np.nan,
                                    cov_at_8=float(cov.cov_val.iloc[7]) if len(cov) > 7 else np.nan, n_eligible_val=int(cov.n_eligible_val.iloc[0]))
        fmt_s = lambda d: " / ".join(str(d[t]) if d[t] is not None else "never" for t in R.TARGETS)
        size_rows.append([f"L{L}", f3(ac["block"]), int(len(cur)), ", ".join(cur.head(4).pair.str.replace(f"L{L}", "")), f"{100 * cur.frac_val.iloc[0]:.0f}% / {100 * cur.frac_val.iloc[min(1, len(cur) - 1)]:.0f}% / "
                          f"{100 * cur.frac_val.iloc[min(3, len(cur) - 1)]:.0f}% / {100 * cur.frac_val.iloc[min(7, len(cur) - 1)]:.0f}%", fmt_s(sz), fmt_s(szi), f"{int(peak.k)} ({100 * peak.frac_val:.0f}%)"])
        cov_rows.append([f"L{L}", f"{int(cov.n_eligible_val.iloc[0])}/{len(val)}", ", ".join(cov.head(4).pair.str.replace(f"L{L}", "")), f"{100 * cov.cov_val.iloc[0]:.0f}% / {100 * cov.cov_val.iloc[min(1, len(cov) - 1)]:.0f}% / "
                         f"{100 * cov.cov_val.iloc[min(3, len(cov) - 1)]:.0f}% / {100 * cov.cov_val.iloc[min(7, len(cov) - 1)]:.0f}%", " / ".join(str(cvs[t]) if cvs[t] is not None else "never" for t in (0.5, 0.8)),
                         f"{100 * cov.cov_val.max():.0f}%"])
    md_table(["Layer", "Sum of singles", "Coalition (clean top-k)", "Coalition (union)", "Block", "r(sum, coalition)", "r(sum, block)", "mean / median |sum - coalition|", "within 0.25 / 0.5", "sum - coalition [95% CI]"],
             add_rows, os.path.join(TAB, f"ext5_rank_additivity_{short}"), caption=f"{cfg['label']}: additivity of single-expert rescues on the paper validation split (same pass; the exact end point of the additive curve is the clean-top-k coalition)")
    md_table(["Layer", "Block rescue (val)", "# experts ever active", "Greedy order (first 4)", "Fraction of block at |S| = 1 / 2 / 4 / 8", "|S| for 50 / 80 / 90% (disc. order, val. fraction)", "|S| for 50 / 80 / 90% (val. order, in-sample)", "Peak |S| (fraction)"],
             size_rows, os.path.join(TAB, f"ext5_rank_minimal_sizes_{short}"), caption=f"{cfg['label']}: population-level minimal sets under the additive approximation (greedy = descending all-case mean discovery rescue; fraction = cumulative validation all-case rescue / validation block rescue)")
    md_table(["Layer", "Eligible cases (block > 0)", "Coverage-greedy order (first 4)", "Cases covered at |S| = 1 / 2 / 4 / 8", "|S| covering 50 / 80% of cases", "Max coverage"],
             cov_rows, os.path.join(TAB, f"ext5_rank_coverage_{short}"), caption=f"{cfg['label']}: per-case coverage variant (case covered when its additive sum over S reaches 80% of its own block rescue; greedy on discovery, evaluated on validation)")

    # ---- cross-layer additive curve and the two-locus comparison
    xl = R.cross_layer_curve(cache, disc, val, cfg["ref_layer"])
    xl.to_csv(os.path.join(TAB, f"ext5_rank_crosslayer_curve_{short}.csv"), index=False)
    curves["cross"] = xl
    out["cross_layer"] = dict(ref_block=float(xl.ref_block_val.iloc[0]), order_top10=xl.head(10).pair.tolist(), frac_top10=xl.frac_val_of_ref_block.head(10).round(3).tolist(),
                              sizes=R.size_for_targets(xl.rename(columns={"frac_val_of_ref_block": "f"}), "f"), peak_k=int(xl.loc[xl.frac_val_of_ref_block.idxmax(), "k"]), peak_frac=float(xl.frac_val_of_ref_block.max()),
                              n_layers_in_top10=int(xl.head(10).layer.nunique()))
    tl = R.two_locus_comparison(md, cache, cfg["two_stage"], cfg["second"], val, nc)
    out["two_locus"] = tl
    out["cross_layer"]["frac_max_top10"] = xl.frac_max_of_ref_block.head(10).round(3).tolist()
    md_table(["k", "Pair", "Disc. active", "Disc. all-case", "Val. all-case", "Cumulative sum (val)", f"Sum, % of L{cfg['ref_layer']} block", "Per-case max (val)", f"Max, % of L{cfg['ref_layer']} block"],
             [[int(r.k), r.pair, f"{int(r.disc_active)}/{len(disc)}", f3(r.disc_allcase_mean), f3(r.val_allcase), f3(r.cum_val), f"{100 * r.frac_val_of_ref_block:.0f}%", f3(r.cum_max_val), f"{100 * r.frac_max_of_ref_block:.0f}%"]
              for _, r in xl.head(12).iterrows()],
             os.path.join(TAB, f"ext5_rank_crosslayer_{short}"), caption=f"{cfg['label']}: cross-layer greedy over all (layer, expert) pairs in descending discovery all-case rescue, first 12 pairs. "
                     f"'Sum' assumes full additivity across layers (upper bound, over-counts information that several layers restore); 'per-case max' assumes full redundancy (lower bound). Exact multi-layer patches are F1.4.")
    figure(cfg, curves, out, os.path.join(FIG, f"ext5_rank_minimal_{short}"))
    return out


# ---------------------------------------------------------------------------------------------------------------
def figure(cfg: dict, curves: dict, out: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"qwen3": "#2a78d6", "mixtral": "#eb6834"}[cfg["model"]]
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    layers = list(cfg["minimal_layers"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    shades = np.linspace(1.0, 0.35, len(layers))
    for L, a in zip(layers, shades):
        cur = curves[L]["additive"]
        kmax = min(len(cur), 32 if cfg["model"] == "qwen3" else 8)
        lw = 2.0 if L == cfg["two_stage"][0] else 1.3
        ax.plot(cur.k[:kmax], 100 * cur.frac_val[:kmax], color=C, alpha=a, lw=lw, marker="o", ms=2.5, label=f"L{L} (block {out['additivity'][str(L)]['block']:+.2f})")
    for t in (50, 80, 90):
        ax.axhline(t, color=GRID, lw=0.8, ls="--")
    ax.axhline(100, color=INK2, lw=0.6)
    ax.set_xlabel("|S| (experts, added in descending discovery all-case rescue)")
    ax.set_ylabel("cumulative validation rescue, % of block rescue")
    ax.set_title("A. additive curve per layer (fixed set, validation)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax = axes[1]
    for L, a in zip(layers, shades):
        cov = curves[L]["coverage"]
        kmax = min(len(cov), 32 if cfg["model"] == "qwen3" else 8)
        lw = 2.0 if L == cfg["two_stage"][0] else 1.3
        ax.plot(cov.k[:kmax], 100 * cov.cov_val[:kmax], color=C, alpha=a, lw=lw, marker="s", ms=2.5, label=f"L{L} ({int(cov.n_eligible_val.iloc[0])} cases)")
    ax.axhline(50, color=GRID, lw=0.8, ls="--"); ax.axhline(80, color=GRID, lw=0.8, ls="--")
    ax.set_xlabel("|S| (coverage-greedy on discovery)")
    ax.set_ylabel("% of validation cases with additive sum >= 80% of own block")
    ax.set_title("B. per-case coverage variant", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax = axes[2]
    xl = curves["cross"]
    kmax = min(len(xl), 12)
    ax.plot(xl.k[:kmax], 100 * xl.frac_val_of_ref_block[:kmax], color=INK2, lw=1.2, ls=":", marker="o", ms=2.5, label="sum over pairs (full additivity, upper bound)")
    ax.plot(xl.k[:kmax], 100 * xl.frac_max_of_ref_block[:kmax], color=INK, lw=1.6, marker="o", ms=2.5, label="per-case max over pairs (full redundancy, lower bound)")
    for _, r in xl.head(kmax).iterrows():
        ax.annotate(r.pair, (r.k, 100 * r.frac_max_of_ref_block), textcoords="offset points", xytext=(0, -11), fontsize=6.5, color=INK2, rotation=60, ha="center")
    ax.axhline(100, color=INK2, lw=0.6)
    for t in (50, 80, 90):
        ax.axhline(t, color=GRID, lw=0.8, ls="--")
    ax.set_xlabel("|S| over all (layer, expert) pairs (descending discovery rescue)")
    ax.set_ylabel(f"cumulative validation rescue, % of L{cfg['ref_layer']} block")
    ax.set_ylim(0, min(260, 100 * xl.frac_val_of_ref_block[:kmax].max() + 15))
    ax.set_title("C. cross-layer: sum vs per-case max (joint patches = F1.4)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.suptitle(f"{cfg['label']}: population-level minimal expert sets under the additive approximation (paper set; selection on discovery, evaluation on validation)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=170); fig.savefig(path + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
SHORT = {"val_rescue": "all-case rescue (val)", "val_active_only": "active-only rescue", "val_spec": "Spec", "block_share": "block share",
         "mean_percentile": "per-case percentile", "disc_allcase": "discovery statistic"}


def write_section(summary: dict, path: str):
    from ext5_rank_text import INTERPRETATION as INT, OVERVIEW, F12_READING, SUMMARY  # noqa
    L = []
    L.append("## Extension 5 / F1: expert rankings and minimal sufficient sets\n")
    L.append(SUMMARY)
    L.append(OVERVIEW)
    for run, o in summary["runs"].items():
        cfg = R.RUNS[run]
        L.append(f"### {o['label']} (`results/{run}`)\n")
        L.append(f"#### F1.1 rankings\n")
        ts, sc = o["two_stage_row"], o["second_row"]
        L.append(f"- {o['n_pairs']} (layer, expert) pairs have at least one clean-active paper case; {o['n_recurrent']} are recurrent (>= {o['threshold']}/{o['n_disc']} discovery cases), "
                 f"{o['n_recurrent_block_pos']} of them in layers whose block rescue CI excludes zero. Full table: `results/tables/ext5_rank_all_{o['short']}.csv`.")
        L.append(f"- Two-stage winner {o['two_stage']}: rank {int(ts['rank_val_rescue'])} by validation all-case rescue, {int(ts['rank_val_active_only'])} by active-only rescue, {int(ts['rank_val_spec'])} by Spec, "
                 f"{rk(ts['rank_block_share'])} by block share, {rk(ts['rank_mean_percentile'])} by mean per-case percentile, {rk(ts['rank_disc_allcase'])} by the discovery selection statistic "
                 f"(val. rescue {f3(ts['val_rescue'])}, active-only {f3(ts['val_active_only'])}, Spec {f3(ts['val_spec'])}, share {100 * ts['block_share']:.0f}%, mean rank {ts['mean_rank']:.2f} of "
                 f"{'8' if cfg['model'] == 'qwen3' else '2'} active, top-1 in {100 * ts['top1_frac']:.0f}% of its active cases).")
        L.append(f"- Second locus {o['second']}: ranks {rk(sc['rank_val_rescue'])} / {rk(sc['rank_val_active_only'])} / {rk(sc['rank_val_spec'])} / {rk(sc['rank_block_share'])} / {rk(sc['rank_mean_percentile'])} / {rk(sc['rank_disc_allcase'])} "
                 f"under the same six metrics (val. rescue {f3(sc['val_rescue'])}, Spec {f3(sc['val_spec'])}, share {100 * sc['block_share']:.0f}%, top-1 in {100 * sc['top1_frac']:.0f}%).")
        K = o["kendall_all"]
        L.append(f"- Kendall tau over the {o['n_recurrent']} recurrent pairs: rescue vs Spec {K['val_rescue']['val_spec']:.2f}, rescue vs active-only {K['val_rescue']['val_active_only']:.2f}, "
                 f"rescue vs block share {K['val_rescue']['block_share']:.2f}, rescue vs per-case percentile {K['val_rescue']['mean_percentile']:.2f}, discovery statistic vs validation rescue {K['disc_allcase']['val_rescue']:.2f}. "
                 f"Top-5 by each metric: " + "; ".join(f"{SHORT[m]}: {', '.join(v)}" for m, v in o["top_by_metric"].items()) + ".")
        L.append("")
        L.append(open(os.path.join(TAB, f"ext5_rank_top30_{o['short']}.md")).read())
        L.append(open(os.path.join(TAB, f"ext5_rank_kendall_{o['short']}.md")).read())
        L.append(open(os.path.join(TAB, f"ext5_rank_disagreements_{o['short']}.md")).read())
        if INT.get(run, {}).get("f11"):
            L.append(INT[run]["f11"] + "\n")
        L.append(f"#### F1.2 population-level minimal sets (additive approximation)\n")
        L.append(open(os.path.join(TAB, f"ext5_rank_additivity_{o['short']}.md")).read())
        L.append(open(os.path.join(TAB, f"ext5_rank_minimal_sizes_{o['short']}.md")).read())
        L.append(open(os.path.join(TAB, f"ext5_rank_coverage_{o['short']}.md")).read())
        L.append(f"![ext5 minimal sets {o['short']}](../figures/ext5_rank_minimal_{o['short']}.png)\n")
        L.append(f"Figure E5-F1-{o['short']}: A, cumulative validation all-case rescue of the greedy set as a fraction of the layer's block rescue (experts added in descending "
                 f"discovery all-case rescue; dashed lines 50/80/90%); B, fraction of validation cases whose additive sum over the set reaches 80% of their own block rescue "
                 f"(coverage-greedy); C, cumulative validation rescue over all (layer, expert) pairs in descending discovery rescue, relative to the L{cfg['ref_layer']} block rescue: dotted = additive sum "
                 f"(assumes independence across layers), solid = per-case maximum (assumes full redundancy); exact multi-layer patches are F1.4.\n")
        xl, tl = o["cross_layer"], o["two_locus"]
        L.append(f"- Cross-layer greedy (first 10 pairs, {xl['n_layers_in_top10']} layers): {', '.join(xl['order_top10'])}. Cumulative validation rescue as a fraction of the L{cfg['ref_layer']} block rescue "
                 f"({f3(xl['ref_block'])}): additive sum {', '.join(f'{100 * f:.0f}%' for f in xl['frac_top10'][:6])} ... (the sum keeps growing without bound, {100 * xl['peak_frac']:.0f}% at |S| = {xl['peak_k']}, "
                 f"which is impossible for a real joint patch and shows that different layers restore the same information); per-case max {', '.join(f'{100 * f:.0f}%' for f in xl['frac_max_top10'][:6])} ....")
        L.append(f"- {tl['a']} + {tl['b']} on validation: alone {cis(tl['a_alone'])} and {cis(tl['b_alone'])}; additive sum {cis(tl['additive_sum'])} = {100 * tl['sum_over_block'][0]:.0f}% "
                 f"[{100 * tl['sum_over_block'][1]:.0f}, {100 * tl['sum_over_block'][2]:.0f}] of the L{cfg['two_stage'][0]} block rescue {cis(tl['block_a_layer'])}; per-case max (union, a lower bound on a joint patch) "
                 f"{cis(tl['per_case_max'])} = {100 * tl['max_over_block'][0]:.0f}%; both positive in {100 * tl['both_positive']:.0f}% of cases, either in {100 * tl['either_positive']:.0f}%.")
        L.append("")
        if INT.get(run, {}).get("f12"):
            L.append(INT[run]["f12"] + "\n")
    L.append(F12_READING)
    frag = os.path.join(TAB, "ext5_rank_subsets_section.md")
    if os.path.exists(frag):
        L.append(open(frag).read())
    else:
        L.append("### F1.3 per-case minimal sets (exact subset patches)\n\n_Pending: `results/<run>_subsets/subset_rows.parquet` from the ext5-engine agent; analysis code in "
                 "`moetrace/ext5_rank.py` (per_case_minimal_sets, interaction_matrix, nonadditivity_decomposition) and `scripts/ext5_rank_subsets.py`._\n")
    L.append(f"_Generated {summary['generated_utc']} by scripts/ext5_rank_analyze.py._\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(R.RUNS))
    ap.add_argument("--section-only", action="store_true", help="rebuild the section from results/ext5_rank_summary.json and the saved tables")
    args = ap.parse_args()
    if args.section_only:
        summary = json.load(open(SUMMARY))
    else:
        summary = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": {}}
        for run in args.runs.split(","):
            if not os.path.exists(os.path.join(RESULTS, run, "expert_rows.parquet")):
                log(run, "missing expert_rows.parquet, skipped"); continue
            summary["runs"][run] = analyze_run(run)
        with open(SUMMARY, "w") as f:
            json.dump(jsonable(summary), f, indent=1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    write_section(summary, os.path.join(SEC, "ext5_f1_rankings.md"))
    log("section written")


if __name__ == "__main__":
    main()
