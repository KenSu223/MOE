#!/usr/bin/env python3
"""Phase 8 stage 3 figures (CounterFact, two-hop capitals) from results/relp_p8/ (scripts/relp_moe_p8_analyze3.py first).
Same palette and conventions as scripts/relp_moe_p8_figures.py. Writes results/figures/relp_moe_p8s3_*.png."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from relp_moe_p8_figures import AQUA, BLUE, FIG, GRID, INK, INK2, MODEL_NAME, NL, ORANGE, SEQ, SURF, VIOLET  # noqa: E402

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/relp_p8"
TASKS = ("counterfact", "twohop")
TN = {"counterfact": "CounterFact", "twohop": "two-hop capitals"}
CLS = ("subject_other", "subject", "between", "final")
CLS_LAB = ("subject,\nother", "subject,\nlast", "between", "final")


def save(fig, name):
    fig.savefig(FIG / f"relp_moe_p8s3_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def cells():
    return [(m, t) for t in TASKS for m in ("qwen3", "mixtral") if (BASE / m / t / "units.parquet").exists()]


def fig_sites():
    """Exact whole-MoE-output patches and RelP by layer x position class."""
    cs = cells()
    if not cs:
        return
    fig, axs = plt.subplots(2, len(cs), figsize=(3.6 * len(cs), 7.2), squeeze=False)
    for j, (model, task) in enumerate(cs):
        b = BASE / model / task
        rows = pd.read_parquet(b / "patch/rows.parquet")
        nd = int((rows.split == "discovery").sum())
        p = pd.read_parquet(b / "patch/patch.parquet", columns=["row", "direction", "kind", "layer", "cls", "effect"])
        p = p[(p.row < nd) & (p.kind == "position") & (p.direction == "denoise")]
        ex = (p.groupby(["layer", "cls"]).effect.sum().unstack(fill_value=0.0) / nd).reindex(range(NL[model]), fill_value=0.0)
        rl = np.load(b / "rules/scores_B.npz")["v_relp_cls"].mean(0)
        present = [k for k in range(4) if k in ex.columns]
        vmax = max(float(ex.values.max()), 1e-6)
        for i, (M, lab) in enumerate(((ex[present].values, "exact patch"), (rl[:, present], "RelP"))):
            ax = axs[i, j]
            im = ax.imshow(M, aspect="auto", origin="lower", cmap=SEQ, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks(range(len(present)), [CLS_LAB[k] for k in present], fontsize=7.5)
            ax.grid(False)
            ax.set_title(f"{MODEL_NAME[model]}, {TN[task]}\n{lab}", fontsize=9)
            if j == 0:
                ax.set_ylabel("layer")
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.subplots_adjust(hspace=0.38, wspace=0.45)
    fig.suptitle("Whole-MoE-output patches by layer and token position (denoise, logits restored; RelP on the colour scale of exact)", x=0.01, ha="left")
    save(fig, "sites")


def fig_scatter():
    """RelP rank vs exact rank of the role units, coloured by layer; circles = final token, triangles = subject token."""
    cs = cells()
    if not cs:
        return
    fig, axs = plt.subplots(1, len(cs), figsize=(3.6 * len(cs), 3.9), squeeze=False)
    for j, (model, task) in enumerate(cs):
        u = pd.read_parquet(BASE / model / task / "units.parquet")
        ax = axs[0, j]
        ns = ~u.sig_exact
        ax.scatter(u.rank_relp[ns], u.rank_exact[ns], s=5, c="#c9c8c3", lw=0, zorder=2)
        for r, mk in ((1, "o"), (0, "^")):
            s = u[u.sig_exact & (u.role == r)]
            sc = ax.scatter(s.rank_relp, s.rank_exact, s=14, c=s.layer, cmap=SEQ, vmin=0, vmax=NL[model] - 1, marker=mk, lw=0.3,
                            edgecolors=SURF, zorder=3)
        lim = len(u) * 1.3
        ax.plot([0.8, lim], [0.8, lim], color=INK2, lw=0.8, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, lim)
        ax.set_ylim(0.8, lim)
        ax.set_title(f"{MODEL_NAME[model]}, {TN[task]}", fontsize=9)
        ax.set_xlabel("RelP rank")
        if j == 0:
            ax.set_ylabel("exact patching rank")
        fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02).set_label("layer")
    fig.suptitle("Rank of every role unit under both methods (circle = final token, triangle = last subject token; grey = exact mean within 2 SE of 0)",
                 x=0.01, ha="left")
    save(fig, "rankscatter")


def fig_addback():
    f = BASE / "analysis/addback.parquet"
    if not f.exists():
        return
    S_ = pd.read_parquet(f)
    S_.loc[S_.ranking == "random", "ranking"] = "random"
    lines = [("relp", BLUE, "RelP ranking", "-"), ("exact", ORANGE, "activation-patching ranking", "-"),
             ("path_grad_rw", VIOLET, "gradient, 8-step path", "-"), ("lu", "#eda100", "Lu's layer-first order", "-"),
             ("random", "#9a9994", "random (3 seeds)", "--")]
    cs = sorted({(m, t) for m, t in zip(S_.model, S_.task)}, key=lambda x: (TASKS.index(x[1]), x[0] != "qwen3"))
    for unit, ulab in (("role", "expert-positions (subject and final token)"), ("all", "whole experts")):
        fig, axs = plt.subplots(2, len(cs), figsize=(3.6 * len(cs), 6.2), squeeze=False, sharex=True)
        for j, (model, task) in enumerate(cs):
            for i, (iv, lab) in enumerate((("denoise", "Rescue\n(share restored)"), ("noise", "Disruption\n(share lost)"))):
                ax = axs[i, j]
                x = S_[(S_.model == model) & (S_.task == task) & (S_.unit == unit) & (S_.intervention == iv)]
                allv = x[x.ranking == "all"].value.mean()
                for rk, colr, name, ls in lines:
                    a = x[x.ranking == rk].groupby("budget").value.mean()
                    if len(a):
                        ax.plot(a.index, a.values, color=colr, ls=ls, marker="o", ms=3.5, label=name)
                if np.isfinite(allv):
                    ax.axhline(allv, color=INK2, lw=0.8, ls=":", label="all units of the pool")
                ax.set_xscale("log")
                ax.set_ylim(-0.08, 1.1)
                if i == 0:
                    ax.set_title(f"{MODEL_NAME[model]}, {TN[task]}", fontsize=9)
                if j == 0:
                    ax.set_ylabel(lab)
                if i == 1:
                    ax.set_xlabel("number of units (K)")
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=8.5)
        fig.suptitle(f"Experts added back vs rescue and disruption on held-out pairs: {ulab}", x=0.5, ha="center")
        save(fig, f"addback_{unit}")


def fig_freeze():
    rows = []
    terms = [("t_set", "later top-k set switches", BLUE), ("t_weights", "later routing weights", "#86b6ef"),
             ("t_attention", "later attention patterns", ORANGE),
             ("t_normnl", "later norms + expert nonlinearity + higher order", AQUA), ("t_half", "half rule", VIOLET)]
    for model, task in cells():
        f = BASE / model / task / "freeze_units.parquet"
        if not f.exists():
            continue
        U = pd.read_parquet(f)
        L = NL[model]
        pos = U[U.kind == "position"]
        for lab, s in ((f"{MODEL_NAME[model]}, {TN[task]}: MoE output, layers 0-2 at the last subject token", pos[(pos.layer <= 2) & (pos.role == 0)]),
                       (f"{MODEL_NAME[model]}, {TN[task]}: MoE output, last 8 layers at the final token", pos[(pos.layer >= L - 8) & (pos.role == 1)])):
            rows.append((lab, s.sum(numeric_only=True)))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(11.5, 0.5 * len(rows) + 2.2))
    for gi, (lab, s) in enumerate(rows):
        lp, ln = 0.0, 0.0
        for t_, name, colr in terms:
            v = float(s[t_])
            ax.barh(gi, v, left=lp if v >= 0 else ln, color=colr, height=0.65, label=name if gi == 0 else None, edgecolor=SURF, lw=1)
            if v >= 0:
                lp += v
            else:
                ln += v
        ax.plot([float(s["gap"])], [gi], marker="|", color=INK, ms=16, mew=2.5, label="total gap (exact - RelP)" if gi == 0 else None)
        ax.text(max(lp, float(s["gap"])) + 0.08, gi, f"exact {float(s['effect_natural']):.2f}, RelP {float(s['relp']):.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("contribution to exact - RelP (logits; denoise, sums over the layers, means over the first 100 discovery pairs)")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.25 + 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, fontsize=8)
    fig.suptitle("Stage 3: where RelP's error comes from (patches re-run with later quantities held at the corrupted run's values)", x=0.01, ha="left")
    save(fig, "freeze")


def main():
    for f in (fig_sites, fig_scatter, fig_addback, fig_freeze):
        f()


if __name__ == "__main__":
    main()
