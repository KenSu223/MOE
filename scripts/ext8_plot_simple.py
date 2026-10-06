"""ext8: simplified add-back and deletion figures with only the most important curves.

Add-back (results/figures/ext8_a1_curves_simple): adaptive greedy, per-case oracle, random order. Same data as
results/figures/ext8_a1_curves.png (validation, donor mean): oracle and random from results/tables/ext8_a1_curves.csv,
greedy from results/ext8_addback_summary.json.
Deletion (results/figures/ext8_a3_curves_simple): per-case deletion oracle (each case's own single-expert noising
ranking, `noise_oracle`) and random order, from results/tables/ext8_a3_curves.csv. Greedy, beam and exact search were
run for add-back only, so the deletion figure has no greedy curve.
Both: 2 x 2 small multiples (task x model), shared y axis, the all-MoE ceiling as a reference line, k for 80 % of the
ceiling per curve under each panel title.

Usage: python scripts/ext8_plot_simple.py
Outputs results/figures/ext8_a1_curves_simple.{png,pdf}, results/figures/ext8_a3_curves_simple.{png,pdf}
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = "/home/ubuntu/MOE/results"
PANELS = [("cf_qwen3", "CounterFact · Qwen3-30B-A3B"), ("cf_mixtral", "CounterFact · Mixtral-8x7B"),
          ("wino_qwen3", "WinoGrande · Qwen3-30B-A3B"), ("wino_mixtral", "WinoGrande · Mixtral-8x7B")]
# reference palette (dataviz skill, categorical slots 2 and 1) + a neutral for the baseline
COL = {"greedy": "#eb6834", "oracle": "#2a78d6", "noise_oracle": "#2a78d6", "rand": "#8a8984"}
INK, INK2, GRID = "#1f2328", "#52514e", "#e4e4e0"
FIGS = {
    "addback": dict(table="ext8_a1_curves.csv", block="a1", out="ext8_a1_curves_simple", ceiling_key="oracle",
                    order=("greedy", "oracle", "rand"),
                    label={"greedy": "adaptive greedy", "oracle": "oracle (each case's own single-expert ranking)",
                           "rand": "random order (mean of 5)"},
                    short={"greedy": "greedy", "oracle": "oracle", "rand": "random"},
                    ceiling_text="all MoE outputs patched", xlabel="experts patched back together, k (log scale)",
                    ylabel="fraction of the drop restored, r(k)"),
    "deletion": dict(table="ext8_a3_curves.csv", block="a3", out="ext8_a3_curves_simple", ceiling_key="noise_oracle",
                     order=("noise_oracle", "rand"),
                     label={"noise_oracle": "oracle (each case's own single-expert deletion ranking)",
                            "rand": "random order (mean of 5)"},
                     short={"noise_oracle": "oracle", "rand": "random"},
                     ceiling_text="all MoE outputs corrupted", xlabel="experts set to their corrupted value, k (log scale)",
                     ylabel="fraction of the drop caused, damage / drop"),
}


def fk(v):
    return str(int(v)) if v == v else "–"


def plot(kind, summ):
    spec = FIGS[kind]
    cur = pd.read_csv(f"{R}/tables/{spec['table']}")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharey=True)
    for ax, (run, title) in zip(axes.flat, PANELS):
        s = summ[run]
        K = s["K"]
        blk = s[spec["block"]]
        ceil = blk[spec["ceiling_key"]]["ceiling"]
        curves, k80 = {}, {}
        for o in spec["order"]:
            if o == "greedy":
                g = s["greedy"]
                curves[o] = (g["k"], g["r"], g["lo"], g["hi"])
                k80[o] = g["k80_ceiling"]
            else:
                c = cur[(cur.run == run) & (cur.donors == "mean") & (cur.ordering == o)].sort_values("k")
                curves[o] = (c.k.values, c.r.values, c.lo.values, c.hi.values)
                k80[o] = blk[o]["k80_ceiling"]
        for o in reversed(spec["order"]):
            k, r, lo, hi = curves[o]
            ax.fill_between(k, lo, hi, color=COL[o], alpha=0.14, linewidth=0)
            ax.plot(k, r, color=COL[o], linewidth=2.2 if o == "greedy" else 2.0, solid_capstyle="round",
                    solid_joinstyle="round", label=spec["label"][o], zorder=3 if o == "greedy" else 2)
        ax.axhline(ceil, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.text(1.05, ceil + 0.015, f"{spec['ceiling_text']}: {ceil:.2f}", color=INK2, fontsize=8.5, va="bottom")
        ax.text(0.0, 1.02, "experts for 80 % of the ceiling: "
                + " · ".join(f"{spec['short'][o]} {fk(k80[o])}" for o in spec["order"]),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(0.9, K * 1.1)
        ax.set_ylim(-0.03, 1.0)
        ax.set_title(f"{title}  (K = {K})", fontsize=10.5, loc="left", pad=20)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel(spec["xlabel"])
    for ax in axes[:, 0]:
        ax.set_ylabel(spec["ylabel"])
    h, l = axes[0, 0].get_legend_handles_labels()
    idx = [l.index(spec["label"][o]) for o in spec["order"]]
    fig.legend([h[i] for i in idx], [l[i] for i in idx], loc="upper center", ncol=len(idx), frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{R}/figures/{spec['out']}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"written results/figures/{spec['out']}.{{png,pdf}}")


def main():
    summ = json.load(open(f"{R}/ext8_addback_summary.json"))["runs"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK})
    plot("addback", summ)
    plot("deletion", summ)


if __name__ == "__main__":
    main()
