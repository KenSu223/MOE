"""ext6 STR layer x position grid analysis (CPU only): Zhang & Nanda (2024) Section 4.1 / Figure 4 heatmaps for the
paper's MoE-output patch under symmetric token replacement.

Per run (results/<run>/str_grid_w<W>_rows.parquet): donor mean per (case, position, layer); mean over the positions of
each token group of a case (first / middle / last subject token, first subsequent token, further tokens, last token);
mean over cases. Two metrics: the logit difference rescue normalised by the mean drop (mean rescue / mean drop, the
normalisation Zhang & Nanda use) and the probability rescue Δp = p_patched(true) - p_corrupt(true) (their
probability metric). Also: their ratio (sum over layers of the effect at the last subject token) / (the same at the middle
subject tokens) under both metrics, on cases with >= 3 subject tokens (paired bootstrap); peaks per group; the last-token
column against the final-position STR sweep (consistency); and, when both windows exist, sliding window (joint patch of
5 layers) vs the sum of the single-layer effects over the same window (their Section 5).

Usage: python scripts/ext6_str_grid_analyze.py
Outputs results/tables/ext6_str_grid_*.{md,csv}, results/figures/ext6_str_grid_<run>.{png,pdf}, ext6_str_grid_curves.{png,pdf},
        results/ext6_str_grid_summary.json
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.ext6_str import POS_CATS
from moetrace.stats import bootstrap_ci, ratio_ci

ROOT = "/home/ubuntu/MOE/results"
TAB, FIG = os.path.join(ROOT, "tables"), os.path.join(ROOT, "figures")
RUNS = [("qwen3_str", "Qwen3-30B-A3B-Base"), ("mixtral_bos_str", "Mixtral-8x7B, BOS"), ("mixtral_nobos_str", "Mixtral-8x7B, no BOS")]
SHORT_CAT = {"first subject token": "first subj.", "middle subject tokens": "middle subj.", "last subject token": "last subj.",
             "first subsequent token": "first subseq.", "further tokens": "further", "last token": "last token"}


def md_table(header, rows, path, caption=None):
    lines = ([f"**{caption}**", ""] if caption else []) + ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    txt = "\n".join(lines) + "\n"
    with open(path + ".md", "w") as f:
        f.write(txt)
    pd.DataFrame(rows, columns=header).to_csv(path + ".csv", index=False)
    return txt


def load(run, w):
    p = os.path.join(ROOT, run, f"str_grid_w{w}_rows.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df["drop"] = df.delta_clean - df.delta_corrupt
    df["pdrop"] = df.p_true_clean - df.p_true_corrupt
    g = df.groupby(["case_id", "pos", "cat", "layer"], as_index=False)[["rescue", "dp", "drop", "pdrop"]].mean()  # donor mean
    cc = g.groupby(["case_id", "cat", "layer"], as_index=False)[["rescue", "dp"]].mean()  # mean over positions of a group
    drop = g.groupby("case_id")["drop"].mean()
    pdrop = g.groupby("case_id")["pdrop"].mean()
    L = int(df.layer.max()) + 1
    mats = {}
    for cat in POS_CATS:
        sub = cc[cc.cat == cat]
        if len(sub) == 0:
            continue
        mats[cat] = {k: sub.pivot(index="case_id", columns="layer", values=k).reindex(columns=range(L)) for k in ("rescue", "dp")}
    return {"L": L, "mats": mats, "drop": drop, "pdrop": pdrop, "n_cases": int(df.case_id.nunique()), "rows": len(df)}


def heat(G, metric):
    """[L, n_cats] mean over cases; metric 'ld' = mean rescue / mean drop of the same cases, 'dp' = mean Δp."""
    H = np.full((G["L"], len(POS_CATS)), np.nan)
    for j, cat in enumerate(POS_CATS):
        if cat not in G["mats"]:
            continue
        M = G["mats"][cat]["rescue" if metric == "ld" else "dp"]
        H[:, j] = M.mean(axis=0).values / (G["drop"].loc[M.index].mean() if metric == "ld" else 1.0)
    return H


def peaks(G, metric):
    out = []
    for cat in POS_CATS:
        if cat not in G["mats"]:
            continue
        M = G["mats"][cat]["rescue" if metric == "ld" else "dp"]
        den = G["drop"].loc[M.index].values if metric == "ld" else np.ones(len(M))
        curve = M.mean(axis=0).values / den.mean()
        l = int(np.nanargmax(curve))
        if metric == "ld":
            r, lo, hi = ratio_ci(M[l].values, den)
        else:
            r = float(M[l].mean()); lo, hi = bootstrap_ci(M[l].values)
        out.append({"cat": cat, "n": int(len(M)), "peak_layer": l, "peak": float(r), "lo": float(lo), "hi": float(hi),
                    "sum_over_layers": float(np.nansum(curve)), "curve": [float(x) for x in curve]})
    return out


def bp_ratio(G, metric):
    key = "rescue" if metric == "ld" else "dp"
    if "middle subject tokens" not in G["mats"]:
        return None
    Ml, Mm = G["mats"]["last subject token"][key], G["mats"]["middle subject tokens"][key]
    ids = Ml.index.intersection(Mm.index)
    num, den = Ml.loc[ids].sum(axis=1).values, Mm.loc[ids].sum(axis=1).values
    r, lo, hi = ratio_ci(num, den)
    return {"n": int(len(ids)), "ratio": float(r), "lo": float(lo), "hi": float(hi), "sum_last": float(num.mean()), "sum_middle": float(den.mean())}


def sweep_consistency(run, G):
    sw = pd.read_parquet(os.path.join(ROOT, run, "str_sweep_rows.parquet"))
    lay = sw[sw.kind == "layer"].groupby(["case_id", "layer"]).rescue.mean().unstack()
    M = G["mats"]["last token"]["rescue"]
    ids = M.index.intersection(lay.index)
    a, b = M.loc[ids].mean(axis=0).values, lay.loc[ids].mean(axis=0).values
    pc = np.corrcoef(M.loc[ids].values.ravel(), lay.loc[ids].values.ravel())[0, 1]
    return {"n": int(len(ids)), "curve_corr": float(np.corrcoef(a, b)[0, 1]), "curve_maxabsdiff": float(np.abs(a - b).max()),
            "percase_corr": float(pc)}


def sliding_vs_adding(G1, G5, metric, half=2):
    out = []
    for cat in POS_CATS:
        if cat not in G1["mats"] or cat not in G5["mats"]:
            continue
        c1 = np.array(next(p for p in peaks(G1, metric) if p["cat"] == cat)["curve"])
        c5 = np.array(next(p for p in peaks(G5, metric) if p["cat"] == cat)["curve"])
        L = len(c1)
        add = np.array([c1[max(0, l - half):min(L, l + half + 1)].sum() for l in range(L)])
        out.append({"cat": cat, "sliding_peak": float(c5.max()), "sliding_layer": int(c5.argmax()), "adding_peak": float(add.max()),
                    "adding_layer": int(add.argmax()), "ratio": float(c5.max() / add.max()) if add.max() > 0 else float("nan")})
    return out


GN_SUBJECT = {"qwen3_str": "qwen3_bos_subject", "mixtral_bos_str": "mixtral_bos_subject", "mixtral_nobos_str": "mixtral_nobos_subject"}


def gn_subject(run, G):
    """F4 (GN) MoE-output patch at the last subject token on the same cases, normalised by its own mean drop, vs STR."""
    gp = os.path.join(ROOT, GN_SUBJECT.get(run, ""), "sweep_rows.parquet")
    if run not in GN_SUBJECT or not os.path.exists(gp):
        return None
    r = pd.read_parquet(gp)
    M = G["mats"]["last subject token"]["rescue"]
    r = r[r.case_id.isin(set(M.index))]
    lay = r[r.kind == "layer"].groupby(["case_id", "layer"]).rescue.mean().unstack()
    drop = (r[r.kind == "clean"].groupby("case_id").delta.mean() - r[r.kind == "noised"].groupby("case_id").delta.mean()).loc[lay.index]
    g = lay.mean(axis=0).values / drop.mean()
    st = M.loc[lay.index].mean(axis=0).values / G["drop"].loc[lay.index].mean()
    lg = int(np.argmax(g)); ls = int(np.argmax(st))
    rg = ratio_ci(lay[lg].values, drop.values); rs = ratio_ci(M.loc[lay.index][ls].values, G["drop"].loc[lay.index].values)
    return {"n": int(len(lay)), "gn_drop": float(drop.mean()), "str_drop": float(G["drop"].loc[lay.index].mean()),
            "gn_peak_layer": lg, "gn_peak": [float(x) for x in rg], "str_peak_layer": ls, "str_peak": [float(x) for x in rs],
            "gn_sum": float(g.sum()), "str_sum": float(st.sum()), "gn_curve": [float(x) for x in g], "str_curve": [float(x) for x in st]}


def gn_figure(summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [r for r in summ if summ[r].get("gn_last_subject")]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        x = summ[run]["gn_last_subject"]
        ax.plot(x["str_curve"], color="#1f77b4", lw=2, label="STR (this grid)")
        ax.plot(x["gn_curve"], color="#d62728", lw=1.6, label="GN (F4 run, same cases)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(summ[run]["label"] + ": MoE patch at the last subject token", fontsize=8.5)
        ax.set_xlabel("MoE layer"); ax.set_ylabel("rescue / mean drop"); ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext6_str_grid_gn_subject.{ext}"), dpi=150)
    plt.close(fig)


def figure(run, label, Gs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ws = [w for w in (1, 5) if Gs.get(w)]
    fig, axes = plt.subplots(len(ws), 2, figsize=(9.5, 4.2 * len(ws)), squeeze=False)
    for i, w in enumerate(ws):
        for j, (metric, title) in enumerate((("ld", "logit difference / drop"), ("dp", "Δp(true)"))):
            ax = axes[i][j]
            H = heat(Gs[w], metric)
            vmax = np.nanmax(np.abs(H))
            im = ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_xticks(range(len(POS_CATS)))
            ax.set_xticklabels([SHORT_CAT[c] for c in POS_CATS], rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("MoE layer")
            ax.set_title(f"{title}, window {w}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"{label}: MoE-output patch under STR, layer x token group", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext6_str_grid_{run}.{ext}"), dpi=150)
    plt.close(fig)


def curves_figure(allG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [r for r in allG if allG[r].get(1)]
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.6), squeeze=False)
    colors = dict(zip(POS_CATS, ("#9ecae1", "#4292c6", "#08306b", "#fdae6b", "#e6550d", "#a50f15")))
    for ax, run in zip(axes[0], runs):
        for p in peaks(allG[run][1], "ld"):
            ax.plot(p["curve"], label=f"{SHORT_CAT[p['cat']]} (n={p['n']})", color=colors[p["cat"]], lw=2 if p["cat"] in ("last subject token", "last token") else 1.2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(dict(RUNS)[run] + ": single-layer MoE patch at each token group", fontsize=8.5)
        ax.set_xlabel("MoE layer")
        ax.set_ylabel("rescue / mean drop")
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext6_str_grid_curves.{ext}"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    allG, summ = {}, {}
    for run, label in RUNS:
        Gs = {w: load(run, w) for w in (1, 5)}
        Gs = {w: g for w, g in Gs.items() if g is not None}
        if not Gs:
            continue
        allG[run] = Gs
        s = {"label": label}
        for w, G in Gs.items():
            s[f"w{w}"] = {"n_cases": G["n_cases"], "rows": G["rows"], "mean_drop": float(G["drop"].mean()), "mean_pdrop": float(G["pdrop"].mean()),
                          "peaks_ld": peaks(G, "ld"), "peaks_dp": peaks(G, "dp"), "bp_ratio_ld": bp_ratio(G, "ld"), "bp_ratio_dp": bp_ratio(G, "dp")}
        if 1 in Gs:
            s["sweep_consistency_w1"] = sweep_consistency(run, Gs[1])
            s["gn_last_subject"] = gn_subject(run, Gs[1])
        if 1 in Gs and 5 in Gs:
            s["sliding_vs_adding_ld"] = sliding_vs_adding(Gs[1], Gs[5], "ld")
            s["sliding_vs_adding_dp"] = sliding_vs_adding(Gs[1], Gs[5], "dp")
        summ[run] = s
        figure(run, label, Gs)
        print(f"{run}: windows {sorted(Gs)}")
    if allG:
        curves_figure(allG)
        gn_figure(summ)
    rows = []
    for run, s in summ.items():
        for w in (1, 5):
            if f"w{w}" not in s:
                continue
            for metric, lab in (("ld", "LD / drop"), ("dp", "Δp")):
                for p in s[f"w{w}"][f"peaks_{metric}"]:
                    d = 3 if metric == "ld" else 4
                    rows.append([s["label"], w, lab, p["cat"], p["n"], f"L{p['peak_layer']}",
                                 f"{p['peak']:+.{d}f} [{p['lo']:+.{d}f}, {p['hi']:+.{d}f}]", f"{p['sum_over_layers']:+.{d}f}"])
    t1 = md_table(["Model", "Window", "Metric", "Token group", "n cases", "Peak layer", "Peak value [95% CI]", "Sum over layers"], rows,
                  os.path.join(TAB, "ext6_str_grid_peaks"), "STR layer x position grid: peak of each token group's layer curve (mean over retained cases)")
    rows = []
    for run, s in summ.items():
        for w in (1, 5):
            if f"w{w}" not in s:
                continue
            a, b = s[f"w{w}"]["bp_ratio_ld"], s[f"w{w}"]["bp_ratio_dp"]
            if a and b:
                rows.append([s["label"], w, a["n"], f"{a['ratio']:.2f} [{a['lo']:.2f}, {a['hi']:.2f}]", f"{b['ratio']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}]"])
    t2 = md_table(["Model", "Window", "Cases with >= 3 subject tokens", "Ratio last / middle subject tokens, LD", "Ratio, probability"], rows,
                  os.path.join(TAB, "ext6_str_grid_ratio"),
                  "Zhang & Nanda's statistic: sum over layers of the effect at the last subject token / at the middle subject tokens (GPT-2 XL, STR, window 5: LD 1.22x, probability 4.33x)")
    rows = []
    for run, s in summ.items():
        for metric in ("ld", "dp"):
            for x in s.get(f"sliding_vs_adding_{metric}", []):
                rows.append([s["label"], "LD / drop" if metric == "ld" else "Δp", x["cat"], f"{x['sliding_peak']:+.4f} (L{x['sliding_layer']})",
                             f"{x['adding_peak']:+.4f} (L{x['adding_layer']})", f"{x['ratio']:.2f}"])
    t3 = md_table(["Model", "Metric", "Token group", "Sliding window 5 peak", "Sum of single layers over the window, peak", "Sliding / adding"], rows,
                  os.path.join(TAB, "ext6_str_grid_sliding"),
                  "Sliding-window (joint) patch vs the sum of single-layer patches over the same 5-layer window (Zhang & Nanda Section 5: 1.40-1.75x in GPT-2 XL)") if rows else ""
    rows = []
    for run, s in summ.items():
        x = s.get("gn_last_subject")
        if x:
            rows.append([s["label"], x["n"], f"L{x['str_peak_layer']} {x['str_peak'][0]:+.3f} [{x['str_peak'][1]:+.3f}, {x['str_peak'][2]:+.3f}]",
                         f"L{x['gn_peak_layer']} {x['gn_peak'][0]:+.3f} [{x['gn_peak'][1]:+.3f}, {x['gn_peak'][2]:+.3f}]",
                         f"{x['str_sum']:+.2f}", f"{x['gn_sum']:+.2f}", f"{x['gn_sum'] / x['str_sum']:.2f}", f"{x['str_drop']:+.2f} / {x['gn_drop']:+.2f}"])
    t4 = md_table(["Model", "n cases", "STR peak (rescue / drop)", "GN peak (rescue / drop)", "STR sum over layers", "GN sum over layers", "GN / STR (sum)",
                   "Mean drop STR / GN"], rows, os.path.join(TAB, "ext6_str_grid_gn_subject"),
                  "Single-layer MoE patch at the last subject token: STR (this grid) vs GN (F4 runs *_subject) on the same cases, each normalised by its own drop") if rows else ""
    print(t4)
    with open(os.path.join(ROOT, "ext6_str_grid_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(t1, t2, t3, sep="\n")
    for run, s in summ.items():
        if "sweep_consistency_w1" in s:
            print(run, "last-token column vs final-position sweep:", s["sweep_consistency_w1"])


if __name__ == "__main__":
    main()
