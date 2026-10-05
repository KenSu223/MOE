"""ext7-controls task 1 analysis (CPU): CounterFact STR attention / MoE / block curves, peaks, attention share of the
positive rescue (Direction-2b definition), additivity, first-donor sensitivity, and the descriptive comparison with the
GN attention sweeps of Direction 2b on the same cases and split.

Usage: python scripts/ext7_cf_analyze.py [--models qwen3,mixtral_bos]
Outputs results/tables/ext7_controls_cf_{peaks,share,additivity}.{md,csv}, results/tables/ext7_controls_cf_curves.csv,
results/figures/ext7_controls_cf_curves.png, key "cf" of results/ext7_controls_summary.json
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from moetrace import ext7_controls as C
from moetrace.models import RESULTS

TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
SUMMARY = os.path.join(RESULTS, "ext7_controls_summary.json")
KL = {"attn_layer": "attention", "layer": "MoE", "block": "block"}


def ci(m, lo, hi, d=3):
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def update_summary(key, val):
    s = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {}
    s[key] = val
    json.dump(s, open(SUMMARY, "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3,mixtral_bos")
    args = ap.parse_args()
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    peaks, shares, adds, curves, summ = [], [], [], [], {}
    models = [m for m in args.models.split(",") if os.path.exists(os.path.join(RESULTS, C.CF_RUNS[m]["str"], "str_attn_rows.parquet"))]
    for m in models:
        cfg = C.CF_RUNS[m]
        rows = pd.read_parquet(os.path.join(RESULTS, cfg["str"], "str_attn_rows.parquet"))
        runs = {"STR (donor mean)": C.StrAttnRun(cfg["str"], "mean", rows), "STR (first donor)": C.StrAttnRun(cfg["str"], "first", rows)}
        base = runs["STR (donor mean)"]
        runs["GN (same cases)"] = C.gn_run_on(cfg["gn"], base.disc, base.val)
        sm = {"n_disc": len(base.disc), "n_val": len(base.val), "n_donor_rows": int((rows.kind == "corrupt").sum() / rows.chunk.nunique())}
        for corr, r in runs.items():
            pk = C.peak_rows(r)
            pk.insert(0, "model", cfg["label"])
            pk.insert(1, "corruption", corr)
            peaks.append(pk)
            sh = C.share_ci(r, r.val)
            pkd = pk.set_index("kind")
            l_moe, l_attn = int(pkd.loc["layer", "L_disc"]), int(pkd.loc["attn_layer", "L_disc"])
            from moetrace.ext2_attn import share_at_layer
            s_moe, s_attn = share_at_layer(r, l_moe, r.val), share_at_layer(r, l_attn, r.val)
            s_pl = share_at_layer(r, cfg["paper_layer"], r.val)
            shares.append({"model": cfg["label"], "corruption": corr, "auc_attn": sh["auc_attn"], "auc_moe": sh["auc_moe"],
                           "share": sh["share"], "share_lo": sh["share_lo"], "share_hi": sh["share_hi"],
                           "moe_peak_layer": l_moe, "share_at_moe_peak": s_moe["share"], "share_at_moe_peak_lo": s_moe["share_lo"],
                           "share_at_moe_peak_hi": s_moe["share_hi"], "attn_peak_layer": l_attn, "share_at_attn_peak": s_attn["share"],
                           "share_at_attn_peak_lo": s_attn["share_lo"], "share_at_attn_peak_hi": s_attn["share_hi"],
                           "paper_layer": cfg["paper_layer"], "share_at_paper_layer": s_pl["share"],
                           "val_mean_drop": float(r.drop.loc[r.val].mean())})
            ad = C.additivity_at(r, sorted({l_moe, l_attn, int(pkd.loc["block", "L_disc"])}))
            ad.insert(0, "model", cfg["label"])
            ad.insert(1, "corruption", corr)
            adds.append(ad)
            curves.append(C.curve_table(r, cfg["label"], corr))
            sm[corr] = {"peaks": {k: {"L_disc": int(pkd.loc[k, "L_disc"]), "val": float(pkd.loc[k, "val_at_L_disc"]),
                                      "val_ci": [float(pkd.loc[k, "val_at_L_disc_lo"]), float(pkd.loc[k, "val_at_L_disc_hi"])],
                                      "val_norm": float(pkd.loc[k, "val_at_L_disc_norm"]), "L_val": int(pkd.loc[k, "L_val"]),
                                      "auc_pos": float(pkd.loc[k, "auc_pos"]), "auc_pos_norm": float(pkd.loc[k, "auc_pos_norm"])}
                                  for k in pkd.index},
                        "share": sh, "share_at_moe_peak": s_moe, "share_at_attn_peak": s_attn, "share_at_paper_layer": s_pl,
                        "val_mean_drop": float(r.drop.loc[r.val].mean()),
                        "layer_sum_norm": {k: float(r.mat(k, r.val).mean(0).sum() / r.drop.loc[r.val].mean()) for k in C.SUBLAYER_KINDS}}
        sm["layer_consistency_vs_ext6"] = C.layer_consistency(cfg["str"], cfg["ext6"])
        # curve correlation STR vs GN (normalised validation means) per kind
        sm["curve_r_str_vs_gn"] = {k: float(np.corrcoef(runs["STR (donor mean)"].mat(k, base.val).mean(0),
                                                        runs["GN (same cases)"].mat(k, base.val).mean(0))[0, 1]) for k in C.SUBLAYER_KINDS}
        summ[m] = sm
    P = pd.concat(peaks, ignore_index=True)
    S = pd.DataFrame(shares)
    A = pd.concat(adds, ignore_index=True)
    CV = pd.concat(curves, ignore_index=True)
    P.to_csv(os.path.join(TAB, "ext7_controls_cf_peaks.csv"), index=False)
    S.to_csv(os.path.join(TAB, "ext7_controls_cf_share.csv"), index=False)
    A.to_csv(os.path.join(TAB, "ext7_controls_cf_additivity.csv"), index=False)
    CV.to_csv(os.path.join(TAB, "ext7_controls_cf_curves.csv"), index=False)
    # markdown tables
    L = ["| Model | Corruption | Component | L* (disc.) | Val. rescue at L* [95% CI] | / drop | Val. argmax | AUC+ (val.) | AUC+ / drop | Mean val. drop |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in P.itertuples():
        L.append(f"| {r.model} | {r.corruption} | {KL[r.kind]} | L{r.L_disc} | {ci(r.val_at_L_disc, r.val_at_L_disc_lo, r.val_at_L_disc_hi)} | "
                 f"{r.val_at_L_disc_norm:.3f} | L{r.L_val} | {r.auc_pos:.2f} [{r.auc_lo:.2f}, {r.auc_hi:.2f}] | {r.auc_pos_norm:.3f} | {r.val_mean_drop:+.2f} |")
    open(os.path.join(TAB, "ext7_controls_cf_peaks.md"), "w").write("\n".join(L) + "\n")
    L = ["| Model | Corruption | Attention share of the positive rescue (AUC+) [95% CI] | AUC+ attention / MoE | Share at MoE peak | Share at attention peak | Share at paper layer |",
         "|---|---|---|---|---|---|---|"]
    for r in S.itertuples():
        L.append(f"| {r.model} | {r.corruption} | {r.share:.3f} [{r.share_lo:.3f}, {r.share_hi:.3f}] | {r.auc_attn:.2f} / {r.auc_moe:.2f} | "
                 f"L{r.moe_peak_layer}: {r.share_at_moe_peak:+.3f} [{r.share_at_moe_peak_lo:+.3f}, {r.share_at_moe_peak_hi:+.3f}] | "
                 f"L{r.attn_peak_layer}: {r.share_at_attn_peak:+.3f} [{r.share_at_attn_peak_lo:+.3f}, {r.share_at_attn_peak_hi:+.3f}] | "
                 f"L{r.paper_layer}: {r.share_at_paper_layer:+.3f} |")
    open(os.path.join(TAB, "ext7_controls_cf_share.md"), "w").write("\n".join(L) + "\n")
    L = ["| Model | Corruption | Layer | Attention | MoE | Sum | Block | Gap block − sum [95% CI] | Per-case r | Block > sum |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in A.itertuples():
        L.append(f"| {r.model} | {r.corruption} | L{r.layer} | {r.attn:+.3f} | {r.moe:+.3f} | {r.sum:+.3f} | {r.block:+.3f} | "
                 f"{ci(r.gap, r.gap_lo, r.gap_hi)} | {r.r_case:.2f} | {r.frac_block_gt_sum:.0%} |")
    open(os.path.join(TAB, "ext7_controls_cf_additivity.md"), "w").write("\n".join(L) + "\n")
    # figure: drop-normalised validation curves, STR donor mean (solid) vs GN same cases (dashed)
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 3.8), squeeze=False)
    col = {"attn_layer": "#1f77b4", "layer": "#ff7f0e", "block": "#2ca02c"}
    for ax, m in zip(axes[0], models):
        lab = C.CF_RUNS[m]["label"]
        for k in C.SUBLAYER_KINDS:
            for corr, ls in (("STR (donor mean)", "-"), ("GN (same cases)", "--")):
                c = CV[(CV.model == lab) & (CV.corruption == corr) & (CV.kind == k)]
                ax.plot(c.layer, c.mean_norm, ls, color=col[k], lw=1.6 if ls == "-" else 1.1,
                        label=f"{KL[k]} ({corr.split(' ')[0]})")
                if ls == "-":
                    ax.fill_between(c.layer, c.ci_lo_norm, c.ci_hi_norm, color=col[k], alpha=0.15, lw=0)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"{lab}: CounterFact, final position")
        ax.set_xlabel("layer")
        ax.set_ylabel("rescue / mean drop (validation)")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ext7_controls_cf_curves.png"), dpi=130)
    update_summary("cf", summ)
    print(open(os.path.join(TAB, "ext7_controls_cf_peaks.md")).read())
    print(open(os.path.join(TAB, "ext7_controls_cf_share.md")).read())
    print(open(os.path.join(TAB, "ext7_controls_cf_additivity.md")).read())
    print(json.dumps({m: {"consistency": summ[m]["layer_consistency_vs_ext6"], "curve_r": summ[m]["curve_r_str_vs_gn"]} for m in summ}, indent=1))


if __name__ == "__main__":
    main()
