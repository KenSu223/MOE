#!/usr/bin/env python3
"""Phase 7 report figures (CPU). Re-runnable; draws whatever data exists.

fig_relp_curves_<unit>_<measure>.png  held-out (test split, paper profile) intervention curves: RelP list vs the mean of the
  matched random sets, normalised as in the protocol (denoise = recovery of the paired difference, remove = remaining
  margin, keep = faithfulness), budgets on a log axis; small multiples rows = model, columns = task.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/relp_sva_moe"
FIG = ROOT / "results/figures"
MODELS = {"qwen3": "Qwen3-30B-A3B", "mixtral": "Mixtral-8x7B"}
TASKS = ("simple", "nounpp", "rc", "within_rc")
C_RELP, C_RAND = "#2a78d6", "#eb6834"          # reference categorical slots 1, 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MEASURE = {"denoise": "recovered fraction (denoise)", "remove": "remaining fraction (remove)", "keep": "faithfulness (keep)"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def curves(unit, measure, profile="paper", split="test", baseline="zero"):
    fig, axes = plt.subplots(len(MODELS), len(TASKS), figsize=(11, 5.2), sharex=True, sharey=True)
    any_data = False
    for i, (mk, mname) in enumerate(MODELS.items()):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            style(ax)
            p = RES / mk / profile / task / split / f"{split}_{unit}_{baseline}" / "summary.json"
            if not p.exists():
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", color=INK2, fontsize=8)
                continue
            any_data = True
            relp, rnd = {}, defaultdict(list)
            for r in json.loads(p.read_text()):
                if r["intervention"] != measure or r["normalized"] is None:
                    continue
                (relp.__setitem__(r["budget"], r["normalized"]) if r["ranking"] == "relp" else rnd[r["budget"]].append(r["normalized"]))
            b = sorted(relp)
            ax.plot(b, [relp[x] for x in b], color=C_RELP, lw=2, marker="o", ms=4, label="RelP list")
            br = sorted(rnd)
            ax.plot(br, [np.mean(rnd[x]) for x in br], color=C_RAND, lw=2, marker="o", ms=4, label="matched random (mean of 3)")
            ax.axhline(1.0 if measure != "remove" else 0.0, color=INK2, lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.set_xticks(b)
            ax.set_xticklabels([str(x) for x in b])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(task, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"{mname}\n{MEASURE[measure]}", fontsize=8.5, color=INK)
            if i == len(MODELS) - 1:
                ax.set_xlabel(f"{unit.replace('_', '-')} budget", fontsize=8.5, color=INK2)
    if not any_data:
        plt.close(fig)
        return None
    h, l = next((ax.get_legend_handles_labels() for ax in axes.flat if ax.get_legend_handles_labels()[0]), ([], []))
    fig.suptitle(f"Held-out {split} split, {profile} profile, {baseline} baseline: {MEASURE[measure]}", y=0.985,
                 fontsize=10.5, color=INK)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"relp_moe_curves_{unit}_{measure}.png"
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return out


ROLES = {"simple": ["the", "SUBJ"], "nounpp": ["the", "SUBJ", "prep", "the", "noun2"],
         "rc": ["the", "SUBJ", "that", "the", "noun2", "verb2"], "within_rc": ["the", "noun1", "that", "the", "SUBJ"]}
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"     # diverging blue <-> red, neutral gray midpoint
C_POS = {"subject": "#2a78d6", "final": "#eb6834", "other": "#1baf7a"}


def _patch_means(b):
    import pandas as pd
    parts = []
    for name in ("patch_full", "patch_layerpos", "patch_final"):
        d = b / name
        if (d / "run_meta.json").exists() and json.loads((d / "run_meta.json").read_text()).get("status") == "complete":
            parts.append(pd.concat([pd.read_parquet(q) for q in sorted(d.glob("part_*.parquet"))]))
    if not parts:
        return None
    df = pd.concat(parts).drop_duplicates(["split", "row", "direction", "kind", "layer", "pos", "expert"])
    d = df[(df.split == "discovery") & (df.direction == "denoise")]
    n = d.row.nunique()
    return (d.groupby(["kind", "layer", "pos", "expert"]).effect.sum() / n).rename("exact").reset_index()


def _relp_B(b):
    import torch
    p = b / "score_endpointB/aggregate.pt"
    return torch.load(p, map_location="cpu", weights_only=True)["mean"].sum(-1).numpy() if p.exists() else None


def layerpos_maps(profile):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    out = []
    for mk, mname in MODELS.items():
        fig, axes = plt.subplots(2, len(TASKS), figsize=(12, 7.2), gridspec_kw={"hspace": 0.18, "wspace": 0.45})
        ok = False
        for j, task in enumerate(TASKS):
            b = RES / mk / profile / task
            um, ep = _patch_means(b), _relp_B(b)
            if um is None or ep is None:
                for i in range(2):
                    axes[i, j].axis("off")
                continue
            ok = True
            L, T = ep.shape[0], ep.shape[1]
            ex = np.zeros((L, T))
            pu = um[um.kind == "position"]
            ex[pu.layer.values, pu.pos.values] = pu.exact.values
            rp = ep.sum(-1)
            first = int(pu.pos.min())
            roles = (["<s>"] if mk == "mixtral" else []) + ROLES[task]
            vmax = max(np.abs(ex).max(), np.abs(rp[:, first:]).max(), 1e-3)
            for i, M in enumerate((ex, rp)):
                ax = axes[i, j]
                im = ax.imshow(M[:, first:], aspect="auto", origin="lower", cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax),
                               interpolation="nearest")
                ax.set_xticks(range(T - first))
                ax.set_xticklabels(roles[first:], fontsize=7.5, color=INK2)
                ax.tick_params(colors=INK2, labelsize=7.5)
                if i == 0:
                    ax.set_title(task, fontsize=10, color=INK)
                if j == 0:
                    ax.set_ylabel(("exact patch (denoise rescue)" if i == 0 else "RelP endpoint B, sum over experts") + "\nlayer",
                                  fontsize=8.5, color=INK)
                top = np.unravel_index(np.argmax(M[:, first:]), M[:, first:].shape)
                ax.annotate(f"{M[top[0], top[1] + first]:.2f}", (top[1], top[0]), color=INK, fontsize=7, ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
            fig.colorbar(im, ax=axes[:, j], fraction=0.05, pad=0.02).ax.tick_params(labelsize=7, colors=INK2)
        if not ok:
            plt.close(fig)
            continue
        fig.suptitle(f"{mname}, {profile} profile, discovery rows: MoE-output patch per (layer, position) vs RelP prediction "
                     "(shared scale per task; labelled cell = maximum)", fontsize=10, color=INK)
        o = FIG / f"relp_moe_layerpos_{mk}_{profile}.png"
        fig.savefig(o, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
        plt.close(fig)
        out.append(o)
    return out


def expert_scatter(profile):
    fig, axes = plt.subplots(len(MODELS), len(TASKS), figsize=(12, 6.4))
    ok = False
    for i, (mk, mname) in enumerate(MODELS.items()):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            style(ax)
            b = RES / mk / profile / task
            um, ep = _patch_means(b), _relp_B(b)
            if um is None or ep is None:
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", color=INK2, fontsize=8)
                continue
            ok = True
            u = um[um.kind == "expert"].copy()
            u["relp"] = ep[u.layer.values, u.pos.values, u.expert.values]
            T = ep.shape[1]
            subj = (["<s>"] if mk == "mixtral" else []) + ROLES[task]
            sp = subj.index("SUBJ")
            u["role"] = np.where(u.pos == T - 1, "final", np.where(u.pos == sp, "subject", "other"))
            if sp == T - 1:
                u.loc[u.pos == sp, "role"] = "subject"   # the subject is the final token
            lim = max(u.exact.abs().max(), u.relp.abs().max()) * 1.08
            ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls=":")
            for role in ("other", "subject", "final"):
                v = u[u.role == role]
                ax.scatter(v.relp, v.exact, s=12, color=C_POS[role], edgecolor="#fcfcfb", linewidth=0.5,
                           label={"subject": "subject position", "final": "final position", "other": "other position"}[role])
            for r in u.reindex(u.exact.abs().sort_values(ascending=False).index).head(2).itertuples():
                ax.annotate(f"L{r.layer}E{r.expert}", (r.relp, r.exact), fontsize=7, color=INK, xytext=(4, 2),
                            textcoords="offset points")
            ax.set_xlim(-lim * 0.25, lim)
            ax.set_ylim(-lim * 0.25, lim)
            if i == 0:
                ax.set_title(task, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"{mname}\nexact rescue (patch)", fontsize=8.5, color=INK)
            if i == len(MODELS) - 1:
                ax.set_xlabel("RelP endpoint B", fontsize=8.5, color=INK2)
    if not ok:
        plt.close(fig)
        return None
    h, l = next((ax.get_legend_handles_labels() for ax in axes.flat if ax.get_legend_handles_labels()[0]), ([], []))
    fig.suptitle(f"Single-expert patches ({profile} profile, discovery means): RelP prediction vs exact rescue; dotted = identity",
                 y=0.985, fontsize=10.5, color=INK)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    o = FIG / f"relp_moe_scatter_{profile}.png"
    fig.savefig(o, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return o


def main():
    made = []
    for profile in ("paper", "grouped"):
        for o in layerpos_maps(profile):
            made.append(str(o.relative_to(ROOT)))
        o = expert_scatter(profile)
        if o:
            made.append(str(o.relative_to(ROOT)))
    for unit in ("neuron", "expert_position", "expert_all_positions"):
        for measure in ("denoise", "remove", "keep"):
            o = curves(unit, measure)
            if o:
                made.append(str(o.relative_to(ROOT)))
    print("\n".join(made))


if __name__ == "__main__":
    main()
