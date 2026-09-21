"""ext5 F2 analysis driver: head rankings, specificity, additivity, minimal head sets, attention distributions; figures,
tables and the section results/sections/ext5_f2_heads.md.

Usage: python scripts/ext5_heads_analyze.py [--runs qwen3_heads,mixtral_nobos_heads,mixtral_bos_heads] [--no-section]
Outputs: results/tables/ext5_heads_*.{csv,md}, results/figures/ext5_heads_<run>.png, results/ext5_heads_summary.json,
         results/sections/ext5_f2_heads.md (unless --no-section)
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from moetrace.models import RESULTS
from moetrace.ext5_heads import (HeadRun, POS_CLASSES, POS_LABEL, head_ranking, rank_stability, additivity, minimal_set_population,
                                 minimal_set_per_case, attention_table, rescue_vs_subject_attention, md_table, fmt_ci)

TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
SEC = os.path.join(RESULTS, "sections")
# validated categorical palette (dataviz reference instance), fixed slot order
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_table(df: pd.DataFrame, name: str, cols=None, md=True, csv=True):
    if csv:
        df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    if md and cols:
        with open(os.path.join(TAB, name + ".md"), "w") as f:
            f.write(md_table(df, cols) + "\n")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TXT2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure(run: HeadRun, rk: dict, add: dict, att: dict, path: str):
    L = run.layers
    fig, axes = plt.subplots(len(L), 3, figsize=(15, 3.3 * len(L)), squeeze=False)
    for r, l in enumerate(L):
        df = rk[l].sort_values("head")
        ax = axes[r, 0]
        style(ax)
        x = df["head"].to_numpy()
        ax.bar(x, df.val_mean, width=0.7, color=PAL[0], edgecolor="white", linewidth=1)
        ax.errorbar(x, df.val_mean, yerr=[df.val_mean - df.val_ci_lo, df.val_ci_hi - df.val_mean], fmt="none", ecolor=TXT2, elinewidth=0.8, capsize=0)
        ax.axhline(0, color=TXT2, linewidth=0.8)
        top = rk[l].head(3)
        for _, t in top.iterrows():
            ax.annotate(f"h{int(t['head'])}", (t["head"], t.val_ci_hi), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=TXT)
        ax.set_title(f"L{l}: validation rescue per head (attention output {add[l]['attn_layer_mean']:+.2f}, MoE {add[l]['moe_layer_mean']:+.2f})",
                     fontsize=9, color=TXT, loc="left")
        ax.set_xlabel("head", fontsize=8, color=TXT2)
        ax.set_ylabel("rescue (logit diff.)", fontsize=8, color=TXT2)
        # additivity scatter
        ax = axes[r, 1]
        style(ax)
        M = run.mat(l, run.val)
        ssum = M.sum(1)
        attn = run.refv(l, "attn_layer", run.val)
        lim = [min(ssum.min(), attn.min()) - 0.2, max(ssum.max(), attn.max()) + 0.2]
        ax.plot(lim, lim, color=GRID, linewidth=1)
        ax.scatter(attn, ssum, s=14, color=PAL[0], alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("attention-output patch rescue", fontsize=8, color=TXT2)
        ax.set_ylabel("sum of single-head rescues", fontsize=8, color=TXT2)
        ax.set_title(f"L{l}: additivity, r = {add[l]['per_case_r']:.2f}, gap {add[l]['gap_mean']:+.2f}", fontsize=9, color=TXT, loc="left")
        # attention distribution of the top-3 heads, clean vs noised
        ax = axes[r, 2]
        style(ax)
        if l in att:
            at = att[l]
            labels, bottoms_c = [], []
            xs = np.arange(len(at))
            w = 0.38
            for k, cls in enumerate(POS_CLASSES):
                cvals = at[f"clean_{cls}"].to_numpy()
                nvals = at[f"noised_{cls}"].to_numpy()
                bc = np.zeros(len(at)) if k == 0 else np.sum([at[f"clean_{c}"].to_numpy() for c in POS_CLASSES[:k]], axis=0)
                bn = np.zeros(len(at)) if k == 0 else np.sum([at[f"noised_{c}"].to_numpy() for c in POS_CLASSES[:k]], axis=0)
                ax.bar(xs - w / 2, cvals, w, bottom=bc, color=PAL[k], edgecolor="white", linewidth=1, label=POS_LABEL[cls])
                ax.bar(xs + w / 2, nvals, w, bottom=bn, color=PAL[k], edgecolor="white", linewidth=1, hatch="///" if k == 0 else None)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"h{int(h)}\nclean | noised" for h in at["head"]], fontsize=8)
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("attention mass (final position)", fontsize=8, color=TXT2)
            ax.set_title(f"L{l}: where the top-3 heads attend (left clean, right noised)", fontsize=9, color=TXT, loc="left")
            if r == 0:
                ax.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{run.cfg['label']}: per-head attention patching at the final position ({run.case_set} set, {len(run.val)} validation cases)",
                 fontsize=10, color=TXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140, facecolor="white")
    fig.savefig(path.replace(".png", ".pdf"), facecolor="white")
    plt.close(fig)


def analyze(run_name: str) -> dict:
    run = HeadRun(run_name)
    short = run.cfg["short"]
    log(f"{run_name}: layers {run.layers}, {run.nH} heads, {len(run.disc)}/{len(run.val)} disc/val cases ({run.case_set})")
    out = {"run": run_name, "label": run.cfg["label"], "layers": run.layers, "n_heads": run.nH, "case_set": run.case_set,
           "n_disc": len(run.disc), "n_val": len(run.val), "per_layer": {}}
    rk, add, att = {}, {}, {}
    rank_cols = [("head", "Head"), ("val_mean", "Val. rescue"), ("val_ci_lo", "CI lo"), ("val_ci_hi", "CI hi"), ("val_pos_frac", "Pos. frac."),
                 ("spec", "Spec"), ("spec_ci_lo", "Spec CI lo"), ("spec_ci_hi", "Spec CI hi"), ("share_of_attn_layer", "Share of attn"),
                 ("disc_mean", "Disc. rescue"), ("disc_rank", "Disc. rank"), ("vnorm_mean", "|v_h|")]
    all_rank = []
    for l in run.layers:
        r = head_ranking(run, l)
        rk[l] = r
        all_rank.append(r)
        save_table(r.head(8), f"ext5_heads_ranking_{short}_L{l}", rank_cols)
        r.to_csv(os.path.join(TAB, f"ext5_heads_ranking_{short}_L{l}.csv"), index=False)
        a = additivity(run, l)
        add[l] = a
        pop = minimal_set_population(run, l)
        pc = minimal_set_per_case(run, l)
        st = rank_stability(r)
        ent = {"ranking_top5": r.head(5)[["head", "val_mean", "val_ci_lo", "val_ci_hi", "spec", "disc_rank"]].to_dict("records"),
               "stability": st, "additivity": a, "minimal_population": pop, "minimal_per_case": pc}
        if run.attn is not None:
            heads = [int(h) for h in r.head(3)["head"]]
            at = attention_table(run, l, heads)
            att[l] = at
            ent["attention_top3"] = at.to_dict("records")
            ent["attention_corr"] = rescue_vs_subject_attention(run, l)
        out["per_layer"][l] = ent
        t = r.iloc[0]
        log(f"  L{l}: top head h{int(t['head'])} {t.val_mean:+.3f} [{t.val_ci_lo:+.3f}, {t.val_ci_hi:+.3f}] Spec {t.spec:+.3f}; "
            f"attn_layer {a['attn_layer_mean']:+.3f} sum heads {a['sum_heads_mean']:+.3f} r {a['per_case_r']:.2f}; "
            f"k80 pop {pop['k_attn_layer']} per-case median {pc['k_median']}; disc/val rho {st['spearman_disc_val']:.2f}")
    pd.concat(all_rank).to_csv(os.path.join(TAB, f"ext5_heads_ranking_{short}_all.csv"), index=False)
    # additivity table
    ad = pd.DataFrame([{"layer": f"L{l}", "attn_layer": a["attn_layer_mean"], "attn_ci": fmt_ci(a["attn_layer_mean"], *a["attn_layer_ci"]),
                        "sum_heads": a["sum_heads_mean"], "sum_ci": fmt_ci(a["sum_heads_mean"], *a["sum_heads_ci"]),
                        "gap": fmt_ci(a["gap_mean"], *a["gap_ci"]), "gap_sd": a["gap_sd"], "r": a["per_case_r"], "frac_sum_gt": a["frac_sum_gt_attn"],
                        "moe": a["moe_layer_mean"], "block": a["block_mean"], "max_head": a["max_single_head_mean"], "n_pos": a["n_heads_positive_mean"]}
                       for l, a in add.items()])
    save_table(ad, f"ext5_heads_additivity_{short}", [("layer", "Layer"), ("attn_ci", "Attention output [CI]"), ("sum_ci", "Sum of heads [CI]"),
                                                       ("gap", "Sum − attention [CI]"), ("gap_sd", "Gap SD (per case)"), ("r", "Per-case r"),
                                                       ("frac_sum_gt", "Cases sum > attn"), ("moe", "MoE output"), ("block", "Block"),
                                                       ("max_head", "Best single head"), ("n_pos", "Heads with mean > 0")])
    # minimal sets table
    ms = pd.DataFrame([{"layer": f"L{l}", "k_attn": e["minimal_population"]["k_attn_layer"], "set_attn": str(e["minimal_population"].get("set_attn_layer")),
                        "k_sum": e["minimal_population"]["k_sum_heads"], "top3": str(e["minimal_population"]["top3_disc"]),
                        "top3_share": e["minimal_population"]["top3_share_of_attn"], "pc_median": e["minimal_per_case"]["k_median"],
                        "pc_q25": e["minimal_per_case"]["k_q25"], "pc_q75": e["minimal_per_case"]["k_q75"], "pc_frac1": e["minimal_per_case"]["frac_k_eq_1"],
                        "pc_frac3": e["minimal_per_case"]["frac_k_le_3"], "pc_n": e["minimal_per_case"]["n_cases_used"], "pc_unreach": e["minimal_per_case"]["n_unreachable"]}
                       for l, e in out["per_layer"].items()])
    ms.to_csv(os.path.join(TAB, f"ext5_heads_minimal_{short}.csv"), index=False)
    ms = ms.copy()
    for c in ("k_attn", "k_sum", "pc_n", "pc_unreach"):
        ms[c] = ms[c].map(lambda v: "not reached" if v is None or (isinstance(v, float) and np.isnan(v)) else str(int(v)))
    for c in ("pc_median", "pc_q25", "pc_q75"):
        ms[c] = ms[c].map(lambda v: f"{v:.0f}" if not (isinstance(v, float) and np.isnan(v)) else "n/a")
    for c in ("top3_share", "pc_frac1", "pc_frac3"):
        ms[c] = ms[c].map(lambda v: f"{v * 100:.0f}%" if not (isinstance(v, float) and np.isnan(v)) else "n/a")
    save_table(ms, f"ext5_heads_minimal_{short}", csv=False, cols=[("layer", "Layer"), ("k_attn", "k for 80% of attn (pop.)"), ("set_attn", "Set"),
                                                    ("k_sum", "k for 80% of Σ heads"), ("top3", "Top-3 (disc.)"), ("top3_share", "Top-3 share of attn"),
                                                    ("pc_median", "Per-case k median"), ("pc_q25", "q25"), ("pc_q75", "q75"), ("pc_frac1", "Frac k = 1"),
                                                    ("pc_frac3", "Frac k ≤ 3"), ("pc_n", "Cases used"), ("pc_unreach", "Unreachable")])
    if att:
        at_all = pd.concat(att.values())
        cols = [("layer", "Layer"), ("head", "Head")] + [(f"clean_{c}", f"clean: {POS_LABEL[c]}") for c in POS_CLASSES] + \
               [(f"noised_{c}", f"noised: {POS_LABEL[c]}") for c in POS_CLASSES] + [("subject_shift", "Subject-mass shift (noised − clean)")]
        save_table(at_all, f"ext5_heads_attention_{short}", cols)
        figure(run, rk, add, att, os.path.join(FIG, f"ext5_heads_{short}.png"))
    else:
        figure(run, rk, add, {}, os.path.join(FIG, f"ext5_heads_{short}.png"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="qwen3_heads,mixtral_nobos_heads,mixtral_bos_heads")
    ap.add_argument("--no-section", action="store_true")
    args = ap.parse_args()
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(SEC, exist_ok=True)
    runs = [r for r in args.runs.split(",") if os.path.exists(os.path.join(RESULTS, r, "head_rows.parquet"))]
    summ = {}
    for r in runs:
        summ[r] = analyze(r)
    with open(os.path.join(RESULTS, "ext5_heads_summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if not args.no_section:
        from moetrace.ext5_heads_section import write_section
        write_section(summ, os.path.join(SEC, "ext5_f2_heads.md"))
    log("done")


if __name__ == "__main__":
    main()
