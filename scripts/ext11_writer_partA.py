"""ext11 Part A (CPU): total vs direct vs indirect effect of single experts at the final position under STR.

total T = single-expert patch rescue (ext6 / ext7 source runs), direct D = ext8 DLA (final norm frozen at the corrupted
run), indirect I = T - D. Writes results/tables/ext11_A_*.md|csv, results/figures/ext11_A_*.png|pdf and key "A" of
results/ext11_writer_summary.json.

Usage: python scripts/ext11_writer_partA.py [--runs cf_qwen3,...]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
import numpy as np, pandas as pd
from moetrace.models import RESULTS
from moetrace import ext11_writer as W

TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ci(d, k="v", digits=3, sign=True):
    if d is None or not isinstance(d, dict) or d.get(k) is None or not np.isfinite(d.get(k, np.nan)):
        return "n/a"
    f = f"{{:{'+' if sign else ''}.{digits}f}}"
    return f"{f.format(d[k])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


def rci(c, digits=2):
    if not c or not np.isfinite(c.get("r", np.nan)):
        return "n/a"
    return f"{c['r']:.{digits}f} [{c['r_lo']:.{digits}f}, {c['r_hi']:.{digits}f}]"


def sci(c, digits=2):
    """OLS slope of T on D (x = T, y = D in corr_ci, so slope_xy): T carries the pass noise, D is nearly noise-free."""
    if not c or not np.isfinite(c.get("slope_xy", np.nan)):
        return "n/a"
    return f"{c['slope_xy']:.{digits}f} [{c['slope_xy_lo']:.{digits}f}, {c['slope_xy_hi']:.{digits}f}]"


def write_table(name, df, note=""):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    with open(os.path.join(TAB, f"{name}.md"), "w") as f:
        f.write(W.md_table(df))
        if note:
            f.write("\n\n" + note + "\n")


def analyse(key):
    c = W.RUNS[key]
    t0 = time.time()
    t = W.load_task(key)
    m = W.pair_table(key, t)
    L = t.L
    log(f"{key}: {len(m)} (row, expert) pairs, {m.case_id.nunique()} cases, load {time.time() - t0:.1f}s")
    res = {"model": c["model"], "task": c["task"], "label": c["label"], "tlabel": c["tlabel"], "L": L, "K": t.K,
           "n_rows": int(m.row_id.nunique()), "n_cases": int(m.case_id.nunique()),
           "n_val_cases": int(m[m.split == "validation"].case_id.nunique())}
    res["agreement_all"] = W.band_agreement(m, L)
    res["agreement_val"] = W.band_agreement(m, L, splits=("validation",))
    log(f"  agreement done {time.time() - t0:.1f}s")
    res["layers_all"] = W.layer_profile(m, L)
    log(f"  layer profile done {time.time() - t0:.1f}s")
    own = W.TARGETS[c["model"]][c["task"]]
    other = [le for le in W.model_targets(c["model"]) if le not in own]
    pop = W.pop_top(t, 10)
    res["own_targets"] = [W.ename(x) for x in own]
    res["other_targets"] = [W.ename(x) for x in other]
    res["pop_top10"] = [W.ename(x) for x in pop]
    res["pop_top10_values"] = {W.ename(x): float(t.pop[x]) for x in pop}
    exps = list(dict.fromkeys(own + other + pop))
    res["experts_val"] = W.expert_stats(m, exps, splits=("validation",))
    res["experts_all"] = W.expert_stats(m, exps, splits=("discovery", "validation"))
    res["set_pop10_val"] = W.set_stats(m, pop, splits=("validation",))
    res["set_own_val"] = W.set_stats(m, own, splits=("validation",))
    log(f"  experts done {time.time() - t0:.1f}s")
    res["ranking_val"] = W.ranking_agreement(m, splits=("validation",))
    res["frozen_norm_aggregate"] = W.frozen_norm_aggregate(key)
    res["inpass_noise"] = W.inpass_noise(key, m)
    # per-expert population over all experts with a sizeable total effect (for the layer-vs-share scatter)
    v = m[m.split == "validation"]
    cl = W.case_level(v, ["layer", "expert"])
    ncv = v.case_id.nunique()
    dropv = W.case_drops(v).mean()
    g = cl.groupby(["layer", "expert"]).agg(T=("T", "sum"), D=("D", "sum"), n=("T", "size")).reset_index()
    g["T_allcase_frac"] = g["T"] / ncv / dropv
    g["D_allcase_frac"] = g["D"] / ncv / dropv
    big = g[g["T_allcase_frac"].abs() >= 0.005].copy()
    big["share"] = big["D"] / big["T"]
    res["population_experts"] = big.sort_values("T_allcase_frac", ascending=False).head(60).round(5).to_dict(orient="records")
    return res, m


def figures(all_res, ms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    keys = [k for k in W.RUNS if k in all_res]
    # 1. layer profiles
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        r = all_res[k]
        Ls = [x["layer"] for x in r["layers_all"]]
        T = [x["T_frac"]["v"] for x in r["layers_all"]]
        D = [x["D_frac"]["v"] for x in r["layers_all"]]
        ax.bar(np.array(Ls) - 0.2, T, width=0.4, label="total (single patches)", color="#4472c4")
        ax.bar(np.array(Ls) + 0.2, D, width=0.4, label="direct (DLA)", color="#ed7d31")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{r['label']}\n{r['tlabel']}", fontsize=9)
        ax.set_xlabel("layer")
        ax.set_ylabel("sum over experts / drop")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext11_A_layers.{ext}"), dpi=150)
    plt.close(fig)
    # 2. scatter D vs T (validation, case level), targets highlighted
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.0), squeeze=False)
    for ax, k in zip(axes[0], keys):
        r = all_res[k]
        m = ms[k]
        v = m[m.split == "validation"]
        cl = W.case_level(v, ["layer", "expert"])
        L = r["L"]
        sc = ax.scatter(cl["T"], cl["D"], c=cl["layer"], s=2, cmap="viridis", vmin=0, vmax=L - 1, alpha=0.4, rasterized=True)
        for name, col in zip(r["own_targets"], ["red", "magenta", "orange"]):
            le = W.parse_ename(name)
            s = cl[(cl.layer == le[0]) & (cl.expert == le[1])]
            ax.scatter(s["T"], s["D"], s=10, color=col, label=name, edgecolor="k", linewidth=0.2)
        lim = np.nanpercentile(np.abs(np.r_[cl["T"], cl["D"]]), 99.9)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("total T (logits)")
        ax.set_ylabel("direct D = DLA (logits)")
        ax.set_title(f"{r['label']} {r['tlabel']}\nr = {r['agreement_val']['all']['corr']['r']:.2f}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        fig.colorbar(sc, ax=ax, label="layer")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext11_A_scatter.{ext}"), dpi=150)
    plt.close(fig)
    # 3. direct share vs layer for population experts with a sizeable total
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        r = all_res[k]
        pe = pd.DataFrame(r["population_experts"])
        pe = pe[pe["T_allcase_frac"] > 0]
        ax.scatter(pe["layer"], pe["D_allcase_frac"] / pe["T_allcase_frac"], s=400 * pe["T_allcase_frac"], alpha=0.6)
        for x in r["experts_val"]:
            if x["expert"] in r["own_targets"] and "share" in x:
                ax.annotate(x["expert"], (x["layer"], x["share"]["v"]), fontsize=7, color="red")
        ax.axhline(1, color="k", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-1, 2)
        ax.set_xlabel("layer")
        ax.set_ylabel("direct share D / T")
        ax.set_title(f"{r['label']} {r['tlabel']}", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext11_A_share_by_layer.{ext}"), dpi=150)
    plt.close(fig)


def tables(all_res):
    keys = [k for k in W.RUNS if k in all_res]
    # experts
    rows = []
    for k in keys:
        r = all_res[k]
        for x in r["experts_val"]:
            role = "own target" if x["expert"] in r["own_targets"] else ("other-task target" if x["expert"] in r["other_targets"] else "")
            if x["expert"] in r["pop_top10"]:
                role = (role + ", " if role else "") + f"pop #{r['pop_top10'].index(x['expert']) + 1}"
            rows.append({"model": r["label"], "task": r["tlabel"], "expert": x["expert"], "role": role,
                         "active cases": f"{x['n_active']}/{x['n_cases']}",
                         "total T (logits)": ci(x.get("T"), digits=2), "direct D (logits)": ci(x.get("D"), digits=2),
                         "indirect I (logits)": ci(x.get("I"), digits=2), "T / drop": ci(x.get("T_frac")),
                         "D / drop": ci(x.get("D_frac")), "direct share D/T": ci(x.get("share"), digits=2, sign=False),
                         "r(D, T)": rci(x.get("corr")), "slope T on D": sci(x.get("corr"))})
    write_table("ext11_A_experts", pd.DataFrame(rows),
                "Validation cases where the expert is clean-active at the final position (CounterFact: donor mean per case; "
                "WinoGrande: directed cases, bootstrap over pairs). T = single-expert STR patch rescue (source run), D = DLA of "
                "delta_e with the final norm frozen at the corrupted run (ext8), I = T - D. Fractions = ratio of means over the "
                "active cases. pop #k = rank in the task's ext8 population ranking (discovery all-case single rescue).")
    # bands
    rows = []
    for k in keys:
        r = all_res[k]
        for b, x in r["agreement_all"].items():
            rows.append({"model": r["label"], "task": r["tlabel"], "band": b, "layers": f"{x['layers'][0]}-{x['layers'][1]}",
                         "pairs": x["n_pairs"], "sum T / drop": ci(x["T_frac"]), "sum D / drop": ci(x["D_frac"]),
                         "sum I / drop": ci(x["I_frac"]), "share D/T": ci(x["share"], digits=2, sign=False),
                         "r(D, T)": rci(x["corr"]), "slope T on D": sci(x["corr"]),
                         "mean T - D (logits)": f"{x['mean_T_minus_D_logits']:+.3f}"})
    write_table("ext11_A_bands", pd.DataFrame(rows),
                "All cases (discovery + validation; no selection is involved). Sums over every clean-active expert of the band per "
                "case, mean over cases / mean drop. r and slope over (case, expert) pairs. 'last' = the last layer only, where "
                "T - D is the frozen-norm error plus pass noise (no downstream computation).")
    # layers (compact)
    rows = []
    for k in keys:
        r = all_res[k]
        for x in r["layers_all"]:
            rows.append({"model": r["label"], "task": r["tlabel"], "layer": x["layer"], "sum T / drop": round(x["T_frac"]["v"], 4),
                         "T lo": round(x["T_frac"]["lo"], 4), "T hi": round(x["T_frac"]["hi"], 4),
                         "sum D / drop": round(x["D_frac"]["v"], 4), "D lo": round(x["D_frac"]["lo"], 4),
                         "D hi": round(x["D_frac"]["hi"], 4), "sum I / drop": round(x["I_frac"]["v"], 4),
                         "sum |T| / drop": round(x["absT_frac"]["v"], 4), "sum |D| / drop": round(x["absD_frac"]["v"], 4),
                         "share D/T": round(x["share"]["v"], 3), "r(D,T)": round(x["corr"]["r"], 3),
                         "slope T on D": round(x["corr"]["slope_xy"], 3)})
    write_table("ext11_A_layers", pd.DataFrame(rows), "All cases; per layer sums over clean-active experts (see ext11_A_bands).")
    # sets + norm
    rows = []
    for k in keys:
        r = all_res[k]
        fn = r["frozen_norm_aggregate"]
        nz = r["inpass_noise"]
        rows.append({"model": r["label"], "task": r["tlabel"], "own targets": ", ".join(r["own_targets"]),
                     "own: T / drop": ci(r["set_own_val"]["T_frac"]), "own: share": ci(r["set_own_val"]["share"], digits=2, sign=False),
                     "pop top-10: T / drop": ci(r["set_pop10_val"]["T_frac"]), "pop top-10: D / drop": ci(r["set_pop10_val"]["D_frac"]),
                     "pop top-10: share": ci(r["set_pop10_val"]["share"], digits=2, sign=False),
                     "all-MoE direct exact / frozen": f"{fn['moe']['exact_frac']:.3f} / {fn['moe']['frozen_frac']:.3f}",
                     "frozen-norm err, median |err| / drop (MoE sum)": f"{fn['moe']['median_abs_err_frac_of_drop']:.3f}",
                     "single-patch pass noise SD (logits)": f"{nz['sd_single']:.3f} (n={nz['n']})",
                     "r(D, T) / r(D, mean of 2 passes) / r(T, T')": f"{nz['r_D_T']:.3f} / {nz['r_D_meanT']:.3f} / {nz['r_T_T8']:.3f}"})
    write_table("ext11_A_sets_norm", pd.DataFrame(rows),
                "Validation. Sets: per case sum over the listed experts that are clean-active. Frozen-norm error of the summed MoE "
                "writes (ext8 addback_direct: exact final RMSNorm vs frozen at the corrupted run, fp32, same path). Pass noise: ext8 "
                "in-pass singles (exact family, k = 1, top-10 of each first-donor row) vs the source run's single patches.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(W.RUNS))
    args = ap.parse_args()
    all_res, ms = {}, {}
    for k in args.runs.split(","):
        all_res[k], ms[k] = analyse(k)
    tables(all_res)
    figures(all_res, ms)
    W.update_summary("A", all_res)
    log("written results/ext11_writer_summary.json key A, tables ext11_A_*, figures ext11_A_*")


if __name__ == "__main__":
    main()
