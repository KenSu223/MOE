#!/usr/bin/env python3
"""Phase 8 figures (docs/relp_moe_phase8_plan.md) from the analysis outputs of scripts/relp_moe_p8_analyze.py.
Palette: the dataviz reference instance (categorical slots 1-3 blue / orange / aqua, validated all-pairs in light mode;
sequential blue ramp for layer; diverging blue <-> red with a gray midpoint). Writes results/figures/relp_moe_p8_*.png."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import relp_moe_p8_data as D  # noqa: E402

P8 = ROOT / "results/relp_sva_moe/p8"
FIG = ROOT / "results/figures"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#c13332"])
TASKS = list(D.TASKS)
MODEL_NAME = {"qwen3": "Qwen3-30B-A3B", "mixtral": "Mixtral-8x7B"}
NL = {"qwen3": 48, "mixtral": 32}

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
                     "lines.linewidth": 2})


def qlabel(r):
    """'(-0.001, 3.0]' -> '0-3' (integer counts)."""
    lo, hi = (float(x) for x in r.strip("([]").split(","))
    lo = int(np.floor(lo)) + 1
    return f"{lo}-{int(hi)}" if int(hi) > lo else f"{lo}"


def save(fig, name):
    fig.savefig(FIG / f"relp_moe_p8_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def units(model, task):
    return pd.read_parquet(P8 / f"units_{model}_{task}.parquet")


def fig_layerdist(models):
    """S1.1: layers of the top-20 expert-positions, RelP (B) vs exact (denoise), per task."""
    for model in models:
        L = NL[model]
        fig, axs = plt.subplots(len(TASKS), 1, figsize=(8, 7.2), sharex=True)
        for ax, task in zip(axs, TASKS):
            u = units(model, task)
            for k, (col, colr, lab) in enumerate((("rank_relp_B", BLUE, "RelP (endpoint B)"), ("rank_exact_denoise", ORANGE, "exact patching (denoise)"))):
                top = u[u[col] <= 20]
                c = top.layer.value_counts().reindex(range(L), fill_value=0)
                ax.bar(np.arange(L) + (k - 0.5) * 0.4, c.values, width=0.38, color=colr, label=lab, zorder=3)
            ax.set_ylabel(task, rotation=0, ha="right", va="center")
            ax.set_yticks([0, 5, 10])
            ax.axvspan(L - 8.5, L - 0.5, color=GRID, alpha=0.5, zorder=0, lw=0)
        axs[0].legend(loc="upper left", ncol=2)
        axs[-1].set_xlabel("layer (shaded: last 8 layers)")
        fig.suptitle(f"{MODEL_NAME[model]}: layers of the top-20 expert-positions (count per layer)", x=0.01, ha="left")
        save(fig, f"layerdist_{model}")


def fig_rankscatter(models):
    """S1.2: RelP rank vs exact rank of every expert-position unit, coloured by layer."""
    for model in models:
        L = NL[model]
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
        for ax, task in zip(axs, TASKS):
            u = units(model, task)
            ns = ~u.sig_exact
            ax.scatter(u.rank_relp_B[ns], u.rank_exact_denoise[ns], s=6, c="#c9c8c3", lw=0, zorder=2, label="exact mean within 2 SE of 0")
            mk = {"final": "o", "subject=final": "o", "subject": "^", "between": "s"}
            for pc, m_ in mk.items():
                s = u[u.sig_exact & (u.pos_class == pc)]
                if len(s):
                    sc = ax.scatter(s.rank_relp_B, s.rank_exact_denoise, s=14, c=s.layer, cmap=SEQ, vmin=0, vmax=L - 1, marker=m_,
                                    lw=0.3, edgecolors=SURF, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            lim = max(u.rank_relp_B.max(), u.rank_exact_denoise.max()) * 1.3
            ax.plot([0.8, lim], [0.8, lim], color=INK2, lw=0.8, ls="--", zorder=1)
            ax.set_xlim(0.8, lim)
            ax.set_ylim(0.8, lim)
            ax.set_title(task)
            ax.set_xlabel("RelP rank (endpoint B)")
        axs[0].set_ylabel("exact patching rank (denoise)")
        cb = fig.colorbar(sc, ax=axs, fraction=0.015, pad=0.01)
        cb.set_label("layer")
        fig.suptitle(f"{MODEL_NAME[model]}: rank of every expert-position under both methods (circle = final position, triangle = subject, square = in between)",
                     x=0.01, ha="left")
        save(fig, f"rankscatter_{model}")
        # value-value version
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
        for ax, task in zip(axs, TASKS):
            u = units(model, task)
            sc = ax.scatter(u.relp_B, u.exact_denoise, s=8, c=u.layer, cmap=SEQ, vmin=0, vmax=L - 1, lw=0.2, edgecolors=SURF)
            ax.set_xscale("symlog", linthresh=0.01)
            ax.set_yscale("symlog", linthresh=0.01)
            lim = max(np.abs(u.relp_B).max(), np.abs(u.exact_denoise).max()) * 1.5
            ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls="--")
            ax.set_title(task)
            ax.set_xlabel("RelP mean score (endpoint B)")
        axs[0].set_ylabel("exact mean effect (denoise)")
        cb = fig.colorbar(sc, ax=axs, fraction=0.015, pad=0.01)
        cb.set_label("layer")
        fig.suptitle(f"{MODEL_NAME[model]}: expert-position means, RelP vs exact patching (symmetric log axes)", x=0.01, ha="left")
        save(fig, f"valscatter_{model}")


def fig_depth(models):
    """S1.3: overlap of the two rankings and split-half reliability of each, by K."""
    dp = pd.read_parquet(P8 / "s1_depth.parquet")
    sh = pd.read_parquet(P8 / "s1_splithalf.parquet")
    fig, axs = plt.subplots(len(models), 4, figsize=(13, 3.2 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        for j, task in enumerate(TASKS):
            ax = axs[i, j]
            a = dp[(dp.model == model) & (dp.task == task)]
            ax.plot(a.K, a.overlap, color=INK, marker="o", ms=4, label="RelP vs exact")
            for meth, colr in (("RelP", BLUE), ("exact", ORANGE)):
                b = sh[(sh.model == model) & (sh.task == task) & (sh.method == meth)]
                ax.plot(b.K, b.overlap_mean, color=colr, marker="o", ms=4, label=f"{meth}: half vs half")
            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{MODEL_NAME[model]} / {task}", fontsize=9)
            if i == len(models) - 1:
                ax.set_xlabel("K (top-K expert-positions)")
            if j == 0:
                ax.set_ylabel("top-K overlap")
    axs[0, 0].legend(loc="lower left")
    fig.suptitle("Agreement down the ranked list: between the methods, and within each method between two halves of the discovery pairs", x=0.01, ha="left")
    save(fig, "depth")


def fig_biasmap(models):
    """S1.4: exact minus RelP for whole-MoE-output units, layer x position."""
    bp = pd.read_parquet(P8 / "s1_bias_pos.parquet")
    for model in models:
        fig, axs = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
        L = NL[model]
        vmax = 1.0
        for ax, task in zip(axs, TASKS):
            b = bp[(bp.model == model) & (bp.task == task)]
            P = sorted(b.pos.unique())
            M = np.full((L, len(P)), np.nan)
            for r in b.itertuples(index=False):
                M[r.layer, P.index(r.pos)] = r.exact - r.relp
            im = ax.imshow(M, aspect="auto", origin="lower", cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="nearest")
            labs = [b[b.pos == p_].pos_class.iloc[0].replace("subject=final", "subj=final") for p_ in P]
            ax.set_xticks(range(len(P)), labs, rotation=30, ha="right")
            ax.set_title(task)
            ax.grid(False)
        axs[0].set_ylabel("layer")
        cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01, extend="both")
        cb.set_label("exact - RelP (logits; red = RelP underestimates)")
        fig.suptitle(f"{MODEL_NAME[model]}: whole-MoE-output patches, exact effect minus RelP (denoise vs endpoint B; clipped at ±1)", x=0.01, ha="left")
        save(fig, f"biasmap_{model}")


def fig_classes(models):
    """S1.5 / S1.6: per pair-unit agreement by routing class and by later route switches, per layer band."""
    cl = pd.read_parquet(P8 / "s1_classes.parquet")
    sw = pd.read_parquet(P8 / "s1_switches.parquet")
    bands = ["L0", "L1-2", "early-mid", "mid-late", "last8"]
    fig, axs = plt.subplots(2, len(models), figsize=(6.2 * len(models), 6.4), squeeze=False)
    for j, model in enumerate(models):
        ax = axs[0, j]
        c = cl[(cl.model == model) & (cl.band != "all")].groupby(["band", "class"]).pearson.mean().unstack()
        x = np.arange(len(bands))
        for k, (cls, colr, lab) in enumerate((("both", BLUE, "routed in both runs"), ("clean_only", ORANGE, "clean run only (injected)"),
                                             ("corrupt_only", AQUA, "corrupted run only (removed)"))):
            ax.bar(x + (k - 1) * 0.27, c.reindex(bands)[cls].values, width=0.25, color=colr, label=lab, zorder=3)
        ax.set_xticks(x, bands)
        ax.set_ylim(-0.1, 1)
        ax.set_ylabel("Pearson r, RelP vs exact\n(single pair-units, mean of 4 tasks)")
        ax.set_title(MODEL_NAME[model])
        ax.legend(loc="upper left", fontsize=8)
        ax = axs[1, j]
        s = sw[(sw.model == model) & (sw.band == "all")]
        for k, task in enumerate(TASKS):
            a = s[s.task == task]
            ax.plot(range(len(a)), a.pearson.values, marker="o", ms=4, color=[BLUE, ORANGE, AQUA, VIOLET][k], label=task)
            ax.set_xticks(range(len(a)), [qlabel(r) for r in a.route_changes])
        ax.set_ylim(-0.1, 1)
        ax.set_xlabel("later top-k sets switched by the exact patch (quintile)")
        ax.set_ylabel("Pearson r, RelP vs exact")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Where RelP's frozen router matters: agreement by routing class (top) and by downstream routing switches (bottom)", x=0.01, ha="left")
    save(fig, "classes")


def fig_addback(models):
    """S2.1: experts added back vs rescue / disruption (swap with the other sentence) and the RelP paper's faithfulness / completeness
    (mean baseline), one line per ranking (mean of the 4 tasks)."""
    f = P8 / "s21_addback.parquet"
    if not f.exists():
        return
    S_ = pd.read_parquet(f)
    S_ = S_[(S_.split == "all")]
    S_["value"] = S_.normalized          # noise: disruption = share lost (the protocol's own normalisation)
    S_.loc[S_.ranking == "random", "ranking"] = "random"
    lines = [("relp", BLUE, "RelP (protocol)", "-"), ("exact", ORANGE, "exact patching", "-"),
             ("exact_noL0-2", AQUA, "exact, layers 0-2 excluded", "-"), ("lu", "#eda100", "Lu's layer-first order", "-"),
             ("random", "#9a9994", "random (3 seeds)", "--")]
    for unit, ulab in (("expert_position", "expert-positions"), ("expert_all_positions", "whole experts")):
        panels = [("denoise", "Rescue: share restored\n(these units from the clean run)"),
                  ("noise", "Disruption: share lost\n(these units from the corrupted run)"),
                  ("keep", "Faithfulness (RelP paper): keep these\nunits, all others set to the mean"),
                  ("remove", "Completeness (RelP paper): these units\nset to the mean; share left, ideal 0")]
        fig, axs = plt.subplots(len(models), 4, figsize=(14, 3.4 * len(models)), squeeze=False)
        for i, model in enumerate(models):
            for j, (iv, lab) in enumerate(panels):
                ax = axs[i, j]
                x = S_[(S_.model == model) & (S_.unit == unit) & (S_.intervention == iv)]
                allv = x[x.ranking == "all"].value.mean()
                ntask = x.task.nunique()
                for rk, colr, name, ls in lines:
                    xr = x[x.ranking == rk]
                    cnt = xr.groupby("budget").task.nunique()
                    a = xr[xr.budget.isin(cnt[cnt == ntask].index)].groupby("budget").value.mean()   # budgets every task has
                    if len(a):
                        ax.plot(a.index, a.values, color=colr, ls=ls, marker="o", ms=3.5, label=name)
                if np.isfinite(allv):
                    ax.axhline(allv, color=INK2, lw=0.8, ls=":", label="all units of the pool")
                ax.set_xscale("log")
                if i == 0:
                    ax.set_title(lab, fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(MODEL_NAME[model], fontsize=10)
                if i == len(models) - 1:
                    ax.set_xlabel(f"number of {ulab} (K)")
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=8.5)
        fig.suptitle(f"Add-back curves on held-out SVA pairs (validation + test, mean of 4 tasks): {ulab}\n"
                     "rescue and disruption swap units with the other sentence; faithfulness and completeness (RelP paper) use the mean baseline",
                     x=0.01, y=1.03, ha="left")
        save(fig, f"addback_{unit}")


def fig_freeze(models):
    """S2.3: decomposition of exact - RelP (logits) for the units where the gap is large."""
    f = P8 / "s23_freeze_units.parquet"
    if not f.exists():
        return
    U = pd.read_parquet(f)
    terms = [("t_set", "later top-k set switches", BLUE), ("t_weights", "later routing weights", "#86b6ef"),
             ("t_attention", "later attention patterns", ORANGE),
             ("t_normnl", "later norms + expert nonlinearity + higher order", AQUA), ("t_half", "half rule", VIOLET)]
    rows = []
    for model in models:
        u = U[U.model == model]
        sub = u[(u.kind == "position") & (u.layer == 0) & u.pos_class.str.startswith("subject")]
        rows.append((f"{MODEL_NAME[model]}: L0 MoE output at the subject (mean of tasks)", sub.groupby("task").sum(numeric_only=True).mean()))
        if model == "mixtral":
            e = u[(u.kind == "expert") & (u.layer == 0) & (u.expert == 2) & u.pos_class.str.startswith("subject")]
            rows.append(("Mixtral-8x7B: L0E002 at the subject (mean of tasks)", e.groupby("task").sum(numeric_only=True).mean()))
        else:
            e = u[(u.kind == "expert") & (u.layer == 41) & (u.expert == 36) & (u.pos_class == "final")]
            if len(e):
                rows.append(("Qwen3-30B-A3B: L41E036 at the final position (nounpp, rc)", e.groupby("task").sum(numeric_only=True).mean()))
        l8 = u[(u.kind == "expert") & (u.band == "last8") & u.pos_class.isin(["final", "subject=final"])]
        rows.append((f"{MODEL_NAME[model]}: last-8-layer experts at the final position (sum per task, mean)", l8.groupby("task").sum(numeric_only=True).mean()))
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 2.2))
    for gi, (lab, s) in enumerate(rows):
        lp, ln = 0.0, 0.0
        for t_, name, colr in terms:
            v = float(s[t_])
            left = lp if v >= 0 else ln
            ax.barh(gi, v, left=left, color=colr, height=0.65, label=name if gi == 0 else None, edgecolor=SURF, lw=1)
            if v >= 0:
                lp += v
            else:
                ln += v
        ax.plot([float(s["gap"])], [gi], marker="|", color=INK, ms=16, mew=2.5, label="total gap (exact - RelP)" if gi == 0 else None)
        ax.text(max(lp, float(s["gap"])) + 0.08, gi, f"exact {float(s['effect_natural']):.2f}, RelP {float(s['relp']):.2f}",
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("contribution to exact - RelP (logits; denoise, means over the first 100 discovery pairs)")
    ax.set_xlim(right=ax.get_xlim()[1] + 2.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3, fontsize=8)
    fig.suptitle("Where RelP's error comes from: each exact patch re-run with later quantities held at the corrupted run's values", x=0.01, ha="left")
    save(fig, "freeze")


def fig_neuron(models):
    rows = []
    for model in models:
        for task in TASKS:
            d = D.task_dir(model, task)
            if not (d / "p8_neuron/run_meta.json").exists():
                continue
            nr = pd.read_parquet(d / "p8_neuron/neuron_patch.parquet")
            z = np.load(d / "p8_neuron/relp_neuron_rows.npz")
            info = D.load_json(d / "p8_neuron/neurons.json")
            n = z["v_relp"].shape[0]
            eff = nr.effect.values - D.patch_offsets(nr, info)
            ex = np.zeros(len(info))
            np.add.at(ex, nr.neuron.values, eff)
            rows.append((model, task, ex / n, z["v_relp"].mean(0), np.array(["random" in x["sets"] for x in info])))
    if not rows:
        return
    fig, axs = plt.subplots(len(models), 4, figsize=(13, 3.2 * len(models)), squeeze=False)
    for (model, task, ex, rl, rnd) in rows:
        ax = axs[models.index(model), TASKS.index(task)]
        ax.scatter(rl[rnd], ex[rnd], s=5, color="#b0afa9", lw=0, label="matched random neurons")
        ax.scatter(rl[~rnd], ex[~rnd], s=6, color=BLUE, lw=0, label="RelP top neurons + top expert")
        lim = max(np.abs(rl).max(), np.abs(ex).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls="--")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(f"{MODEL_NAME[model]} / {task}", fontsize=9)
        ax.set_xlabel("RelP (endpoint B), mean")
    for i in range(len(models)):
        axs[i, 0].set_ylabel("exact single-neuron patch, mean")
    axs[0, 0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Single neurons: RelP score vs exact patch effect (denoise, 300 discovery pairs, per-pair offset removed; symmetric log axes)", x=0.01, ha="left")
    save(fig, "neuron")


def fig_rules(models):
    """S2.4 / S2.6: share of the exact effect recovered by each rule at the two sites with large exact effects."""
    f = P8 / "s24_posunits.parquet"
    if not f.exists():
        return
    P = pd.read_parquet(f)
    vs = [("relp", "RelP"), ("relp_nohalf", "RelP without half rule"), ("grad", "plain gradient"), ("relp_rw", "RelP + routing weights"),
          ("relp_soft1", "RelP + soft top-k"), ("path_relp", "RelP, 8-step path*"), ("path_grad_rw", "gradient, 8-step path*")]
    cols = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", VIOLET]
    cats = [("L0", "subject", "layer 0, subject position"), ("last8", "final", "last 8 layers, final position")]
    fig, axs = plt.subplots(1, len(models), figsize=(6.2 * len(models), 3.8), squeeze=False, sharey=True)
    for i, model in enumerate(models):
        ax = axs[0, i]
        p = P[P.model == model]
        x = np.arange(len(cats))
        w = 0.8 / len(vs)
        for k, ((v, lab), colr) in enumerate(zip(vs, cols)):
            vals = []
            for lay, pos, _ in cats:
                b = p[(p.layers == lay) & (p.position.str.startswith(pos) | (p.position == "subject=final")) & (p.variant == v)]
                if v.startswith("path"):
                    b = b[b.task.isin(["simple", "nounpp"])]
                vals.append(b.score.sum() / b.exact.sum() if len(b) and abs(b.exact.sum()) > 1e-6 else np.nan)
            ax.bar(x + (k - len(vs) / 2 + 0.5) * w, vals, width=w * 0.95, color=colr, label=lab, zorder=3)
        ax.axhline(1, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(x, [c[2] for c in cats])
        ax.set_title(MODEL_NAME[model])
    axs[0, 0].set_ylabel("score / exact effect\n(whole-MoE-output units, summed over tasks)")
    axs[0, 0].set_ylim(0, 1.15)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
    fig.suptitle("Share of the exact patch effect recovered by each attribution rule (1 = exact; * simple and nounpp only)", x=0.01, ha="left")
    save(fig, "rules")

def fig_addback_tasks(models):
    """S2.1 per task: rescue and disruption vs K under RelP's and exact patching's ranking (expert-positions)."""
    f = P8 / "s21_addback.parquet"
    if not f.exists():
        return
    S_ = pd.read_parquet(f)
    S_ = S_[(S_.split == "all") & (S_.unit == "expert_position")].copy()
    S_["value"] = S_.normalized          # noise: disruption = share lost
    lines = [("relp", BLUE, "RelP ranking", "-"), ("exact", ORANGE, "activation-patching ranking", "-"),
             ("lu", "#eda100", "Lu's layer-first order", "-"), ("random", "#9a9994", "random (3 seeds)", "--")]
    for model in models:
        fig, axs = plt.subplots(2, 4, figsize=(13.5, 6.2), sharex=True)
        for j, task in enumerate(TASKS):
            for i, (iv, lab) in enumerate((("denoise", "Rescue\n(share restored)"), ("noise", "Disruption\n(share lost)"))):
                ax = axs[i, j]
                x = S_[(S_.model == model) & (S_.task == task) & (S_.intervention == iv)]
                for rk, colr, name, ls in lines:
                    a = x[x.ranking == rk].groupby("budget").value.mean()
                    if len(a):
                        ax.plot(a.index, a.values, color=colr, ls=ls, marker="o", ms=3.5, label=name)
                ax.set_xscale("log")
                ax.set_ylim(-0.05, 1.08)
                if i == 0:
                    ax.set_title(task)
                if j == 0:
                    ax.set_ylabel(lab)
                if i == 1:
                    ax.set_xlabel("number of expert-positions (K)")
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=9)
        fig.suptitle(f"{MODEL_NAME[model]}: experts added back vs rescue and disruption, per task (held-out pairs)", x=0.01, ha="left")
        save(fig, f"addback_tasks_{model}")


def main():
    models = [m for m in ("qwen3", "mixtral") if (P8 / f"units_{m}_simple.parquet").exists()]
    for f in (fig_layerdist, fig_rankscatter, fig_depth, fig_biasmap, fig_classes, fig_rules, fig_freeze, fig_neuron, fig_addback, fig_addback_tasks):
        f(models)


if __name__ == "__main__":
    main()
