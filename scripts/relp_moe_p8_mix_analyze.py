#!/usr/bin/env python3
"""Analysis of scripts/relp_moe_p8_mix.py (SVA add-back under five patch semantics): rescue (denoise, share of the paired
difference restored; ideal 1) and disruption (noise, share lost; ideal 1) per model, task, ranking, budget and mode
(ratio of means over the 80 held-out pairs, as relp_sva.summarize); random = mean of 3 seeds. Also: the routing-weight sum
at the patched sites under the full swap (from the p8_probe router logits), and the full-swap check against S2.1.
Writes results/tables/relp_moe_p8mix.md, results/relp_sva_moe/p8/mix_summary.parquet,
results/figures/relp_moe_p8mix_{modes,rankings}_<model>.png."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/relp_sva_moe"
OUTP = BASE / "p8"
FIG = ROOT / "results/figures"
TAB = ROOT / "results/tables/relp_moe_p8mix.md"
TASKS = ["simple", "nounpp", "rc", "within_rc"]
MODELS = {"qwen3": "Qwen3-30B-A3B", "mixtral": "Mixtral-8x7B"}
MODES = ["full", "content_both", "content_all", "renorm", "route"]
MODE_NAME = {"full": "full swap (S2.1)", "content_both": "content only, both-routed", "content_all": "content only, all live-routed",
             "renorm": "full swap, renormalised", "route": "routing decision moved too"}
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#9a9994"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2})


def load():
    parts = []
    for model in MODELS:
        for task in TASKS:
            f = BASE / model / "paper" / task / "p8_mix/margins.parquet"
            meta = BASE / model / "paper" / task / "p8_mix/run_meta.json"
            if f.exists() and json.loads(meta.read_text()).get("status") == "complete":
                parts.append(pd.read_parquet(f).assign(model=model, task=task))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarise(d):
    g = d.groupby(["model", "task", "ranking", "seed", "budget", "mode", "intervention"])
    s = g.agg(m=("margin", "mean"), a=("margin_A", "mean"), b=("margin_B", "mean"), n=("margin", "size")).reset_index()
    den = s.a - s.b
    s["value"] = np.where(s.intervention == "denoise", (s.m - s.b) / den, (s.a - s.m) / den)   # rescue / disruption
    s = s.groupby(["model", "task", "ranking", "budget", "mode", "intervention"]).value.mean().reset_index()   # random: mean of seeds
    return s


def weight_sums(models):
    """Routing-weight sum at the patched (pair, layer, position) sites under the full swap (denoise: run B, C := A's)."""
    out = []
    for model in models:
        for task in TASKS:
            d = BASE / model / "paper" / task
            if not (d / "p8_probe/router_logits.npy").exists():
                continue
            z = np.load(d / "p8_probe/probe.npz")
            topi = z["topi"].astype(np.int64)
            rl = np.load(d / "p8_probe/router_logits.npy").astype(np.float32)
            _, n, L, T, E = rl.shape
            lg = np.take_along_axis(rl, topi, axis=-1)
            w = np.exp(lg - lg.max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            G = np.zeros_like(rl)
            np.put_along_axis(G, topi, w, axis=-1)
            G = G[:, z["split"] != "discovery"]
            rk = json.loads((d / "p8_addback/rankings.json").read_text())["expert_position"]
            for rname in ("relp", "exact"):
                for K in (5, 20, 50):
                    l, p, e = np.unravel_index(np.array(rk[rname][:K]), (L, T, E))
                    gs, gd = G[0], G[1]
                    S = gd.sum(-1).copy()
                    for li, pi, ei in zip(l, p, e):
                        S[:, li, pi] += gs[:, li, pi, ei] - gd[:, li, pi, ei]
                    v = np.concatenate([S[:, li, pi] for li, pi in sorted(set(zip(l, p)))])
                    a_, b_ = gs[:, l, p, e] > 0, gd[:, l, p, e] > 0
                    out.append(dict(model=model, task=task, ranking=rname, K=K, off=float(np.mean(np.abs(v - 1) > 0.02)),
                                    p05=float(np.quantile(v, 0.05)), p95=float(np.quantile(v, 0.95)),
                                    injected=float((a_ & ~b_).sum() / max((a_ | b_).sum(), 1)),
                                    removed=float((~a_ & b_).sum() / max((a_ | b_).sum(), 1))))
    return pd.DataFrame(out)


def check_full(d):
    """The full-swap rows repeat the S2.1 add-back (other batch composition)."""
    out = []
    for (model, task), x in d[d["mode"] == "full"].groupby(["model", "task"]):
        old = pd.read_parquet(BASE / model / "paper" / task / "p8_addback/interventions.parquet")
        old = old[(old.unit == "expert_position") & old.intervention.isin(["noise", "denoise"])]
        f = x.merge(old[["row_id", "ranking", "seed", "budget", "intervention", "margin"]],
                    on=["row_id", "ranking", "seed", "budget", "intervention"], suffixes=("", "_old"))
        out.append(dict(model=model, task=task, rows=len(f), pearson=float(np.corrcoef(f.margin, f.margin_old)[0, 1]),
                        median_abs_diff=float((f.margin - f.margin_old).abs().median())))
    return pd.DataFrame(out)


def md(df, digits=2):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def figures(s):
    Ks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    mean = s[s.ranking != "all"].groupby(["model", "ranking", "budget", "mode", "intervention"]).value.mean().reset_index()
    ntask = s.groupby(["model", "ranking", "budget", "mode", "intervention"]).task.nunique().reset_index(name="nt")
    mean = mean.merge(ntask, on=["model", "ranking", "budget", "mode", "intervention"])
    ceil = s[s.ranking == "all"].groupby(["model", "mode", "intervention"]).value.mean()
    mstyle = {"full": (INK2, "--", "o"), "content_both": (BLUE, "-", "s"), "content_all": (AQUA, "-", "^"),
              "renorm": (ORANGE, "-", "D"), "route": (YELLOW, "-", "v")}
    rstyle = [("relp", BLUE, "RelP ranking", "-"), ("exact", ORANGE, "activation-patching ranking", "-"),
              ("lu", YELLOW, "Lu's layer-first order", "-"), ("random", GREY, "random (3 seeds)", "--")]
    labs = (("denoise", "rescue\n(share restored; ideal 1)"), ("noise", "disruption\n(share lost; ideal 1)"))
    for model in MODELS:
        x = mean[(mean.model == model) & (mean.nt == mean[mean.model == model].nt.max())]
        if x.empty:
            continue
        # 1: one panel per ranking, lines = patch semantics
        fig, axs = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True, sharey=True)
        for j, (rk, _, rname, _) in enumerate(rstyle):
            for i, (iv, lab) in enumerate(labs):
                ax = axs[i, j]
                for mode in MODES[1:] + MODES[:1]:          # the full swap last, on top (it overlaps renorm / route)
                    c, ls, mk = mstyle[mode]
                    a = x[(x.ranking == rk) & (x["mode"] == mode) & (x.intervention == iv)].sort_values("budget")
                    ax.plot(a.budget, a.value, color=c, ls=ls, marker=mk, ms=3.5, lw=1.6, label=MODE_NAME[mode],
                            zorder=4 if mode == "full" else 2)
                ax.set_xscale("log")
                ax.set_ylim(-0.05, 1.08)
                if i == 0:
                    ax.set_title(rname)
                if j == 0:
                    ax.set_ylabel(lab)
                if i == 1:
                    ax.set_xlabel("number of expert-positions (K)")
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=8.5)
        fig.suptitle(f"{MODELS[model]}: add-back under five patch semantics at the patched MoE sites (held-out SVA pairs, mean of 4 tasks)",
                     x=0.01, ha="left")
        fig.savefig(FIG / f"relp_moe_p8mix_modes_{model}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        # 2: one panel per patch semantics, lines = rankings (the layout of relp_moe_p8_addback_tasks_*)
        fig, axs = plt.subplots(2, 5, figsize=(16.5, 6.4), sharex=True, sharey=True)
        for j, mode in enumerate(MODES):
            for i, (iv, lab) in enumerate(labs):
                ax = axs[i, j]
                for rk, c, rname, ls in rstyle:
                    a = x[(x.ranking == rk) & (x["mode"] == mode) & (x.intervention == iv)].sort_values("budget")
                    ax.plot(a.budget, a.value, color=c, ls=ls, marker="o", ms=3.5, lw=1.6, label=rname)
                if (model, mode, iv) in ceil.index:
                    ax.axhline(ceil[(model, mode, iv)], color=INK2, lw=0.8, ls=":", label="all units of the pool")
                ax.set_xscale("log")
                ax.set_ylim(-0.05, 1.08)
                if i == 0:
                    ax.set_title(MODE_NAME[mode], fontsize=9)
                if j == 0:
                    ax.set_ylabel(lab)
                if i == 1:
                    ax.set_xlabel("number of expert-positions (K)")
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=8.5)
        fig.suptitle(f"{MODELS[model]}: rankings compared under each patch semantics (held-out SVA pairs, mean of 4 tasks)", x=0.01, ha="left")
        fig.savefig(FIG / f"relp_moe_p8mix_rankings_{model}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("figures", model)


def main():
    d = load()
    if d.empty:
        print("no complete p8_mix run")
        return
    s = summarise(d)
    OUTP.mkdir(parents=True, exist_ok=True)
    s.to_parquet(OUTP / "mix_summary.parquet", index=False)
    cells = d.groupby("model").task.nunique().to_dict()
    sec = [f"# Add-back under five patch semantics at the patched MoE sites (SVA, held-out pairs)\n",
           f"Cells: {cells} tasks per model. Values: ratio of means over the held-out pairs; mean of the tasks; random = mean of 3 seeds.",
           "Rescue = (F_patched - F_B) / (F_A - F_B) on the B input (denoise; ideal 1); disruption = (F_A - F_patched) / (F_A - F_B) "
           "on the A input (noise; ideal 1). These are the swap measures (named rescue / disruption since 2026-11-16; faithfulness / "
           "completeness are reserved for the RelP paper's keep / remove). Modes: " + "; ".join(f"{k} = {v}" for k, v in MODE_NAME.items()) + ".\n"]
    mt = s[s.ranking != "all"].groupby(["model", "ranking", "budget", "mode", "intervention"]).value.mean()
    for model in MODELS:
        if model not in cells:
            continue
        for iv, name in (("denoise", "rescue"), ("noise", "disruption")):
            rows = []
            for rk in ("relp", "exact", "lu", "random"):
                for mode in MODES:
                    r = {"ranking": rk, "mode": mode}
                    for K in (1, 5, 10, 20, 50, 100):
                        r[f"K={K}"] = mt.get((model, rk, K, mode, iv), np.nan)
                    rows.append(r)
            sec += [f"## {MODELS[model]}: {name}\n", md(pd.DataFrame(rows)), ""]
        al = s[(s.model == model) & (s.ranking == "all")].groupby(["mode", "intervention"]).value.mean().unstack()
        sec += [f"### {MODELS[model]}: every unit of the pool (ceiling)\n", md(al.reset_index().rename(columns={"denoise": "rescue", "noise": "disruption"})), ""]
        # RelP - exact under each mode
        diff = []
        for mode in MODES:
            r = {"mode": mode}
            for K in (5, 10, 20, 50, 100):
                for iv, nm in (("denoise", "R"), ("noise", "D")):
                    r[f"{nm} K={K}"] = mt.get((model, "relp", K, mode, iv), np.nan) - mt.get((model, "exact", K, mode, iv), np.nan)
            diff.append(r)
        sec += [f"### {MODELS[model]}: RelP ranking minus activation-patching ranking (R = rescue, D = disruption; positive = RelP better)\n",
                md(pd.DataFrame(diff)), ""]
    ws = weight_sums([m_ for m_ in MODELS if m_ in cells])
    if not ws.empty:
        w2 = ws.groupby(["model", "ranking", "K"]).agg(sites_off=("off", "mean"), p05=("p05", "min"), p95=("p95", "max"),
                                                      injected=("injected", "mean"), removed=("removed", "mean")).reset_index()
        sec += ["## Routing-weight sum at the patched sites under the full swap (denoise, held-out pairs)\n",
                "sites_off = share of patched (pair, layer, position) sites whose weight sum differs from 1 by more than 0.02; p05 / p95 = "
                "5th / 95th percentile over sites (extreme over tasks); injected / removed = share of patched (pair, unit) instances whose "
                "expert is routed only in the source run / only in the patched run. The other four modes keep a total of 1 by construction "
                "(content_*: the live weights; renorm: rescaled; route: the source run's weights).\n", md(w2), ""]
    ck = check_full(d)
    sec += ["## Check: the full-swap rows repeat the S2.1 add-back (other batch composition)\n", md(ck, 4), ""]
    TAB.write_text("\n".join(sec) + "\n")
    figures(s)
    print("wrote", TAB)


if __name__ == "__main__":
    main()
