"""ext13ab (Phase 5) figures from results/ext13ab_summary.json (scripts/ext13ab_analyze.py).

Per condition (and replication family), one panel per model, style of scripts/ext8_plot_simple.py:
  ext13ab_<cond>[_rep]_addback  adaptive greedy / per-case oracle / random order, all-MoE ceiling line, k for 80 % of the
                                ceiling under the panel title (k for 80 % of each curve's own maximum when the ceiling is
                                <= 0.1 of the drop);
  ext13ab_<cond>[_rep]_deletion deletion oracle (each case's own single-expert noising ranking) / random order;
  ext13ab_<cond>[_rep]_split    all-MoE add-back, all-MoE deletion, MoE <= L-5 and the direct-path split (p5-data's run).
Cross-condition: ext13ab_overview (ceiling, oracle overshoot and k80 per condition and model along the task ladder, with the
Phase-3 CounterFact and WinoGrande option-swap runs), ext13ab_split_all (the split bars of every condition).

Usage: python scripts/ext13ab_plot.py [--summary PATH] [--outdir DIR]
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/home/ubuntu/MOE/results"
COL = {"greedy": "#eb6834", "oracle": "#2a78d6", "noise_oracle": "#2a78d6", "rand": "#8a8984"}
MCOL = {"qwen3": "#2a78d6", "mixtral": "#eb6834"}
BARCOL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, INK2, GRID = "#1f2328", "#52514e", "#e4e4e0"
CEIL_MIN = 0.1
LADDER = [("P3_cf", "CounterFact\n(Phase 3)"), ("P3_wino", "WG option\nswap (P3)"), ("dir_opt", "option swap\n(dir. twins)"),
          ("vt_e2a_orig", "templates\ne→a orig"), ("vt_e2a_neu", "templates\ne→a neutral"),
          ("cp_nat", "copy prefix\ntwins"), ("cp_vt_orig", "copy prefix\ntempl. orig"), ("cp_vt_neu", "copy prefix\ntempl. neutral"),
          ("f2e", "F2E natural\na→e"), ("f1", "F1 a→e"), ("dir_f1", "F1 on dir.\ntwins"), ("vt_a2e_orig", "templates\na→e orig"),
          ("vt_a2e_neu", "templates\na→e neutral")]
GROUPS = [(0, 4, "entity → attribute (answer generated)"), (5, 7, "entity → attribute, answer in a copy prefix"),
          (8, 12, "attribute → entity (answer = a context entity)")]


def fk(v):
    return "–" if v is None or not np.isfinite(v) else str(int(v))


def style():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK})


def clean_ax(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200, facecolor="white")
    plt.close(fig)
    print(f"written {name}")


def num(x):
    if x is None:
        return np.nan
    if isinstance(x, str):
        return np.inf if x in ("inf", "Infinity") else (-np.inf if x == "-inf" else float(x))
    return float(x)


def arr(v):
    return np.array([num(x) for x in v], dtype=float)


def curve_panel(ax, r, kind):
    if kind == "addback":
        blk, orders, ceil = r["boot"], ("greedy", "oracle", "rand"), num(r["boot"]["ceiling"][0])
        short = {"greedy": "greedy", "oracle": "oracle", "rand": "random"}
        ctext = "all MoE outputs patched"
    else:
        blk, orders, ceil = r["del"], ("noise_oracle", "rand"), num(r["del"]["ceiling"][0])
        short = {"noise_oracle": "oracle", "rand": "random"}
        ctext = "all MoE outputs corrupted"
    lo_all, hi_all = [0.0], [1.0]
    k80 = {}
    for o in reversed(orders):
        e = blk.get(o)
        if not e:
            continue
        k, rr, lo, hi = arr(e["k"]), arr(e["r_curve"]), arr(e["r_curve_lo"]), arr(e["r_curve_hi"])
        ax.fill_between(k, lo, hi, color=COL[o], alpha=0.14, linewidth=0)
        ax.plot(k, rr, color=COL[o], linewidth=2.2 if o == "greedy" else 2.0, solid_capstyle="round", zorder=3 if o == "greedy" else 2)
        lo_all += [np.nanmin(np.r_[rr, lo, ceil])]
        hi_all += [np.nanmax(np.r_[rr, hi, ceil])]
        k80[o] = (num(e["k80"][0]), num(e.get("k80max", [np.nan])[0]))
    ax.axhline(ceil, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(1.05, ceil + 0.015, f"{ctext}: {ceil:.2f}", color=INK2, fontsize=8.5, va="bottom", zorder=4)
    if ceil > CEIL_MIN:
        txt = "experts for 80 % of the ceiling: " + " · ".join(f"{short[o]} {fk(k80[o][0])}" for o in orders if o in k80)
    else:
        txt = f"ceiling ≤ {CEIL_MIN}; experts for 80 % of each curve's maximum: " + " · ".join(f"{short[o]} {fk(k80[o][1])}" for o in orders if o in k80)
    ax.text(0.0, 1.02, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.3, color=INK2)
    K = r["K"]
    ax.set_xscale("log")
    ax.set_xlim(0.9, K * 1.1)
    clean_ax(ax)
    return min(-0.03, min(lo_all) - 0.03), max(1.0, max(hi_all) + 0.03)


def cond_figs(S, outdir):
    runs = S["runs"]
    conds = {}
    for k, r in runs.items():
        conds.setdefault((r["cond"], r["family"]), []).append(r)
    for (cond, fam), rs in conds.items():
        rs = sorted(rs, key=lambda r: r["model"] != "qwen3")
        name = f"ext13ab_{cond}" + ("_rep" if fam == "rep" else "")
        for kind, ylab, xlab, labels in (
                ("addback", "fraction of the drop restored, r(k)", "experts patched back together, k (log scale)",
                 {"greedy": "adaptive greedy", "oracle": "oracle (each case's own single-expert ranking)", "rand": "random order (mean of 5)"}),
                ("deletion", "fraction of the drop caused, damage / drop", "experts set to their corrupted value, k (log scale)",
                 {"noise_oracle": "oracle (each case's own single-expert deletion ranking)", "rand": "random order (mean of 2)"})):
            one = len(rs) == 1
            fig, axes = plt.subplots(1, len(rs), figsize=(6.4 if one else 5.6 * len(rs), 5.2 if one else 4.4), sharey=True, squeeze=False)
            lims = []
            for ax, r in zip(axes[0], rs):
                lims.append(curve_panel(ax, r, kind))
                ax.set_title(f"{r['label']}  (K = {r['K']}, {r['n_units']} pairs)", fontsize=10, loc="left", pad=20)
                ax.set_xlabel(xlab)
            axes[0][0].set_ylim(min(a for a, _ in lims), max(b for _, b in lims))
            axes[0][0].set_ylabel(ylab)
            hs = [plt.Line2D([], [], color=COL[o], linewidth=2.2) for o in labels]
            fig.suptitle(f"{rs[0]['tlabel']}: {'add-back' if kind == 'addback' else 'deletion'} curves (validation)", x=0.01, ha="left",
                         y=0.995, fontsize=10.5, color=INK)
            fig.legend(hs, list(labels.values()), loc="upper left" if one else "upper center", ncol=1 if one else len(labels), frameon=False,
                       fontsize=9, bbox_to_anchor=(0.01, 0.95) if one else (0.5, 0.95))
            fig.tight_layout(rect=(0, 0, 1, 0.87 if one else 0.9))
            save(fig, outdir, f"{name}_{kind}")
        fig, ax = plt.subplots(figsize=(max(6.8, 2.2 + 2.4 * len(rs)), 4.2))
        split_bars(ax, rs, [r["label"] for r in rs])
        ax.set_title(f"{rs[0]['tlabel']}:\nfinal-position MoE ceilings and direct-path split (validation)", fontsize=9, loc="left")
        fig.tight_layout()
        save(fig, outdir, f"{name}_split")


BARS = (("M add-back", lambda r: _ci(r, "d", "all_moe")), ("M deletion", lambda r: _ci(r, "n", "all_moe")),
        ("MoE ≤ L−5 (add-back)", lambda r: _ci(r, "d", "int_moe")), ("direct MoE share", lambda r: _dir(r, "m_dir")),
        ("direct attention share", lambda r: _dir(r, "a_dir")))


def _ci(r, dr, k):
    try:
        v = r["ext8"]["a0"]["validation"][dr][k]
        return num(v["ratio"]), num(v["lo"]), num(v["hi"])
    except (KeyError, TypeError):
        return None


def _dir(r, k):
    d = r.get("direct") or {}
    if k not in d:
        return None
    return tuple(num(x) for x in d[k][:3])


def split_bars(ax, rs, xt):
    xs = np.arange(len(rs))
    w = 0.16
    for j, (lab, f) in enumerate(BARS):
        vals, err = [], []
        for r in rs:
            v = f(r)
            vals.append(v[0] if v else np.nan)
            err.append([v[0] - v[1], v[2] - v[0]] if v and np.isfinite(v[1]) else [0, 0])
        ax.bar(xs + (j - 2) * w, vals, w * 0.92, color=BARCOL[j], label=lab, yerr=np.array(err).T, capsize=2, ecolor=INK2,
               error_kw={"linewidth": 0.8})
    ax.set_xticks(xs)
    ax.set_xticklabels(xt, fontsize=8)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_ylabel("fraction of the drop (validation)")
    ax.legend(fontsize=7.5, ncol=5 if len(rs) >= 4 else 3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13 if len(rs) < 5 else -0.2))
    clean_ax(ax)


def ladder_runs(S):
    out = {}
    for k, r in S.get("phase3", {}).items():
        out[(("P3_cf" if k.startswith("cf") else "P3_wino"), r["model"])] = r
    for k, r in S["runs"].items():
        if r["family"] == "main":
            out[(r["cond"], r["model"])] = r
    return out


def overview(S, outdir):
    lr = ladder_runs(S)
    cols = [(c, lab) for c, lab in LADDER if any((c, m) in lr for m in MCOL)]
    if not cols:
        return
    idx = {c: i for i, (c, _) in enumerate(cols)}
    fig, axes = plt.subplots(3, 1, figsize=(max(8, 1.05 * len(cols) + 2), 10), sharex=True)
    off = {"qwen3": -0.12, "mixtral": 0.12}
    for m in MCOL:
        for c, _ in cols:
            r = lr.get((c, m))
            if not r:
                continue
            x = idx[c] + off[m]
            b = r["boot"]
            ce = arr(b["ceiling"])
            axes[0].errorbar(x, ce[0], yerr=[[ce[0] - ce[1]], [ce[2] - ce[0]]], fmt="o", color=MCOL[m], ms=6, capsize=2, lw=1)
            ov = arr(b["oracle"]["overshoot"])
            axes[1].errorbar(x, ov[0], yerr=[[ov[0] - ov[1]], [ov[2] - ov[0]]], fmt="o", color=MCOL[m], ms=6, capsize=2, lw=1)
            if ce[0] > CEIL_MIN:
                for o, mk, fc in (("greedy", "o", MCOL[m]), ("oracle", "s", "white")):
                    k = arr(b[o]["k80"])
                    if np.isfinite(k[0]):
                        lo, hi = k[1], (k[2] if np.isfinite(k[2]) else k[0])
                        axes[2].errorbar(x + (0.0 if o == "greedy" else 0.05), k[0], yerr=[[k[0] - lo], [max(hi - k[0], 0)]], fmt=mk,
                                         color=MCOL[m], mfc=fc, ms=6, capsize=2, lw=1)
                    else:
                        axes[2].text(x, 400, "never", color=MCOL[m], fontsize=7, ha="center", rotation=90)
            else:
                axes[2].text(x, 1.15, "M≤0.1", color=MCOL[m], fontsize=7, ha="center", rotation=90, va="bottom")
    axes[0].set_ylabel("all-MoE ceiling M\n(fraction of the drop)")
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[1].set_ylabel("oracle overshoot\n(max r(k) − M)")
    axes[1].axhline(0, color=INK2, lw=0.8)
    axes[2].set_ylabel("experts for 80 % of M\n(● greedy, ■ oracle)")
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.8, 700)
    for ax in axes:
        clean_ax(ax)
    # group shading along the ladder
    for gi, (a, b, lab) in enumerate(GROUPS):
        xs = [idx[c] for c, _ in LADDER[a:b + 1] if c in idx]
        if not xs:
            continue
        for ax in axes:
            if gi % 2 == 1:
                ax.axvspan(min(xs) - 0.5, max(xs) + 0.5, color="#f3f3f0", zorder=0)
        axes[0].text((min(xs) + max(xs)) / 2, 1.06, lab, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5,
                     color=INK2)
    axes[2].set_xticks(range(len(cols)))
    axes[2].set_xticklabels([lab for _, lab in cols], fontsize=8)
    hs = [plt.Line2D([], [], marker="o", color=MCOL[m], linestyle="", ms=6) for m in MCOL]
    fig.legend(hs, ["Qwen3-30B-A3B-Base", "Mixtral-8x7B (BOS)"], loc="upper center", ncol=2, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, outdir, "ext13ab_overview")
    # all split bars
    for m in MCOL:
        rs = [lr[(c, m)] for c, _ in cols if (c, m) in lr]
        if not rs:
            continue
        fig, ax = plt.subplots(figsize=(1.6 + 1.15 * len(rs), 4.0))
        split_bars(ax, rs, [lab for c, lab in cols if (c, m) in lr])
        ax.set_title(f"{rs[0]['label']}: all-MoE ceilings and direct-path split per condition (validation)", fontsize=9.5, loc="left")
        fig.tight_layout()
        save(fig, outdir, f"ext13ab_split_all_{m}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(R, "ext13ab_summary.json"))
    ap.add_argument("--outdir", default=os.path.join(R, "figures"))
    ap.add_argument("--no-section", dest="section", action="store_false", help="do not rewrite results/sections/ext13_addback.md")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    style()
    S = json.load(open(args.summary))
    cond_figs(S, args.outdir)
    overview(S, args.outdir)
    if args.section and args.summary == os.path.join(R, "ext13ab_summary.json"):
        # refresh results/sections/ext13_addback.md (the chains call this script after every job)
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import ext13ab_text
        sys.argv = [sys.argv[0]]
        ext13ab_text.main()


if __name__ == "__main__":
    main()
