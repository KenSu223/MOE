"""ext14 C2 RelP pilot analysis (CPU; p6-circuit). Reads results/p6ci_relp_<model>_<task>/ (scripts/ext14ci_relp_run.py),
writes results/tables/ext14ci_relp_*.md, results/figures/ext14ci_relp_*.png and the 'c2' key of results/ext14ci_summary.json.

Usage: python scripts/ext14ci_relp_analyze.py [--smoke]
"""
import argparse, glob, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext14_circuit as C
from moetrace.ext14_circuit_an import interp_k
from moetrace.models import RESULTS

TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
N_BOOT = 1000


def r_(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return None
    return float(np.corrcoef(a[m], b[m])[0, 1])


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0:
        return None
    return float(np.polyfit(x[m], y[m], 1)[0])


def boot_ratio(num: pd.Series, den: pd.Series, n=N_BOOT, seed=0):
    """ratio of sums over cases with a case bootstrap CI."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n, len(num)))
    b = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def analyse(run_dir):
    meta = json.load(open(os.path.join(run_dir, "run_meta.json"))) if os.path.exists(os.path.join(run_dir, "run_meta.json")) else {}
    rd = lambda n: pd.read_parquet(os.path.join(run_dir, f"{n}.parquet")) if os.path.exists(os.path.join(run_dir, f"{n}.parquet")) else None
    cases, grid, focal, multi, circ, top = (rd(n) for n in ("relp_cases", "relp_grid", "relp_focal", "relp_multi", "relp_circuits", "relp_nodes_top"))
    if cases is None:
        return None
    out = {"run": os.path.basename(run_dir), "n_cases": int(len(cases)), "meta": {k: meta.get(k) for k in ("model", "task", "focal", "seconds")}}
    cases = cases.assign(drp=cases.m_clean - cases.m_corrupt)
    out["drop_mean"] = float(cases.drp.mean())
    out["relp_total_over_drop"] = float(cases.relp_o_total.sum() / cases.drp.sum())
    out["atp_total_over_drop"] = float(cases.atp_o_total.sum() / cases.drp.sum())
    out["relp_c_total_over_drop"] = float(cases.relp_c_total.sum() / cases.drp.sum())
    # V3 sublayer grid
    if grid is not None and len(grid):
        g = grid.copy()
        g["relp_mean"] = 0.5 * (g.relp_o + g.relp_c)
        res = {}
        for nm, sub in (("all", g), ("final", g[g.is_final]), ("nonfinal", g[~g.is_final]), ("swap", g[g.is_swap & ~g.is_final])):
            res[nm] = {"n": int(len(sub)), **{f"r_{k}": r_(sub[k], sub.rescue) for k in ("relp_o", "relp_c", "relp_mean", "atp_o", "dla")},
                       **{f"slope_{k}": slope(sub[k], sub.rescue) for k in ("relp_o", "relp_mean", "atp_o")},
                       "exact_abs_mass": float(sub.rescue.abs().sum())}
        tot = g.rescue.abs().sum()
        res["exact_abs_share_final"] = float(g[g.is_final].rescue.abs().sum() / tot) if tot > 0 else None
        res["exact_abs_share_swap"] = float(g[g.is_swap & ~g.is_final].rescue.abs().sum() / tot) if tot > 0 else None
        if g.n_switch.notna().any():
            res["switch_rate"] = float((g.n_switch > 0).mean())
            res["switch_rate_final_cells"] = float((g[g.is_final].n_switch > 0).mean())
            sw = g[g.n_switch > 0]
            ns = g[g.n_switch == 0]
            res["r_relp_o_switch"] = r_(sw.relp_o, sw.rescue)
            res["r_relp_o_noswitch"] = r_(ns.relp_o, ns.rescue)
            res["abs_err_relp_o_switch"] = float((sw.relp_o - sw.rescue).abs().mean()) if len(sw) else None
            res["abs_err_relp_o_noswitch"] = float((ns.relp_o - ns.rescue).abs().mean()) if len(ns) else None
        # per-layer curve at the final position (mean over cases)
        fin = g[g.is_final].groupby("layer")[["rescue", "relp_o", "relp_c", "atp_o", "dla"]].mean()
        res["final_curve"] = {k: fin[k].round(4).tolist() for k in fin.columns}
        res["final_curve_r"] = {k: r_(fin[k], fin.rescue) for k in ("relp_o", "relp_c", "atp_o", "dla")}
        out["grid"] = res
    # V1 focal curves
    if focal is not None and len(focal):
        fr = {}
        for (l, e), f in focal.groupby(["layer", "expert"]):
            ref = f[f.ranking == "ref_all"].set_index("case_id").rescue
            fd = {"T_mean": float(ref.mean()), "n_cases": int(len(ref))}
            curves = {}
            for rk in ("relp_o", "relp_c", "atp_o", "dla", "mag", "rand"):
                q = f[f.ranking == rk]
                if not len(q):
                    continue
                piv = q.pivot_table(index="case_id", columns="k", values="rescue").reindex(ref.index)
                ks = np.array(list(piv.columns) + [int(f[f.ranking == "ref_all"].k.iloc[0])], float)
                r = np.append(piv.sum(0).to_numpy() / ref.sum(), 1.0)
                curves[rk] = {"k": ks.tolist(), "r": r.round(4).tolist(), "k50": interp_k(ks, r, 0.5), "k80": interp_k(ks, r, 0.8)}
            fd["curves"] = curves
            sg = f[f.ranking.str.startswith("single")]
            fd["single_r_relp_o"] = r_(sg.pred_relp_o, sg.rescue)
            fd["single_r_relp_c"] = r_(sg.pred_relp_c, sg.rescue)
            fd["single_r_atp_o"] = r_(sg.pred_atp_o, sg.rescue)
            fd["single_r_dla"] = r_(sg.pred_dla, sg.rescue)
            sub = f[f.ranking.isin(["relp_o", "dla", "atp_o", "mag"])]
            fd["subset_r_relp_o"] = r_(sub.pred_relp_o, sub.rescue)
            fd["subset_r_dla"] = r_(sub.pred_dla, sub.rescue)
            fd["direct_share"] = float(f[f.ranking == "ref_all"].pred_dla.sum() / ref.sum()) if ref.sum() != 0 else None
            fd["relp_share"] = float(f[f.ranking == "ref_all"].pred_relp_o.sum() / ref.sum()) if ref.sum() != 0 else None
            fr[f"L{l}E{e:03d}"] = fd
        out["focal"] = fr
    # V5 multi-layer final-position sets
    if multi is not None and len(multi):
        mm = multi.merge(cases[["case_id", "drp"]], on="case_id")
        mr = {}
        for rk, q in mm.groupby("ranking"):
            piv = q.pivot_table(index="case_id", columns="k", values="rescue")
            dr = cases.set_index("case_id").drp.reindex(piv.index)
            ks = np.array(piv.columns, float)
            r = piv.sum(0).to_numpy() / dr.sum()
            nl = q.groupby("k").n_layers.mean()
            mr[rk] = {"k": ks.tolist(), "r_of_drop": r.round(4).tolist(), "n_layers": nl.round(2).tolist(),
                      "k50_drop": interp_k(ks, r, 0.5), "k80_drop": interp_k(ks, r, 0.8)}
        out["multi"] = mr
    # circuits
    if circ is not None and len(circ):
        cc = circ.merge(cases[["case_id", "m_clean", "m_corrupt", "drp"]], on="case_id")
        cr = {}
        for (mode, pool), q in cc.groupby(["mode", "pool"]):
            d = {}
            if mode == "str":
                for kind in ("denoise", "noise"):
                    k = q[q.kind == kind]
                    piv = (k.assign(v=k.m - k.m_corrupt).pivot_table(index="case_id", columns="N", values="v"))
                    dr = cases.set_index("case_id").drp.reindex(piv.index)
                    Ns = np.array(piv.columns, float)
                    full = piv.notna().all(0)
                    r = (piv.loc[:, full].sum(0) / dr.sum()).to_numpy()
                    d[kind] = {"N": Ns[full.to_numpy()].tolist(), "frac_of_drop": r.round(4).tolist(),
                               "N80": interp_k(Ns[full.to_numpy()], r, 0.8) if kind == "denoise" else None}
            else:
                emp = q[q.kind == "empty"].set_index("case_id").m
                ful = q[q.kind == "full"].set_index("case_id").m
                den = (ful - emp)
                d["m_empty_mean"] = float(emp.mean())
                d["m_full_mean"] = float(ful.mean())
                for kind in ("faithful", "complete"):
                    k = q[q.kind == kind]
                    piv = k.pivot_table(index="case_id", columns="N", values="m")
                    full = piv.notna().all(0)
                    piv = piv.loc[:, full]
                    e_ = emp.reindex(piv.index)
                    dd = den.reindex(piv.index)
                    r = ((piv.sub(e_, axis=0)).sum(0) / dd.sum()).to_numpy()
                    Ns = np.array(piv.columns, float)
                    d[kind] = {"N": Ns.tolist(), "score": r.round(4).tolist(),
                               "N80": interp_k(Ns, r, 0.8) if kind == "faithful" else None}
            cr[f"{mode}|{pool}"] = d
        out["circuits"] = cr
    if top is not None and len(top):
        out["top_nodes"] = top.groupby(["mode", "N"])[["frac_final", "frac_swap", "frac_between", "frac_before", "mean_layer", "mass_share"]].mean().round(4).reset_index().to_dict("records")
    return out


def fig_circuits(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = [r for r in res if r.get("circuits")]
    if not rs:
        return
    fig, axs = plt.subplots(2, len(rs), figsize=(3.8 * len(rs), 5.6), squeeze=False)
    col = {"all": "#1f5fa8", "final": "#d07a1f", "final_swap": "#5aa05a"}
    for j, r in enumerate(rs):
        for pool in ("all", "final", "final_swap"):
            c = r["circuits"].get(f"str|{pool}")
            if c and "denoise" in c:
                axs[0, j].plot(c["denoise"]["N"], c["denoise"]["frac_of_drop"], color=col[pool], label=pool, marker="o", ms=3)
            c = r["circuits"].get(f"mean|{pool}")
            if c and "faithful" in c:
                axs[1, j].plot(c["faithful"]["N"], c["faithful"]["score"], color=col[pool], label=pool, marker="o", ms=3)
        for i, yl in enumerate(("STR denoise: fraction of drop", "mean-ablation faithfulness")):
            axs[i, j].set_xscale("log")
            axs[i, j].axhline(0.8, color="k", lw=0.6, ls=":")
            axs[i, j].axhline(1.0, color="k", lw=0.6)
            axs[i, j].set_ylabel(yl, fontsize=8)
            axs[i, j].set_xlabel("circuit size N (neurons x positions)", fontsize=8)
            axs[i, j].tick_params(labelsize=7)
        axs[0, j].set_title(f"{r['meta']['model']} / {r['meta']['task']} (n={r['n_cases']})", fontsize=9)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    pat = "p6ci_relp_*_smoke" if a.smoke else "p6ci_relp_*"
    runs = sorted(d for d in glob.glob(os.path.join(RESULTS, pat)) if os.path.isdir(d) and (a.smoke or not d.endswith("_smoke"))
                  and "verify" not in d)
    res = [r for r in (analyse(d) for d in runs) if r]
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    sfx = "_smoke" if a.smoke else ""
    L = ["| model / task | cases | exact vs RelP r (all cells; final; non-final) | vs AtP r (all; non-final) | RelP slope | "
         "routing switch rate | final-pos share of exact sublayer mass | focal k80: RelP / DLA / AtP / mag / rand | "
         "single-neuron r RelP / DLA | multi-layer final-pos k80 of drop (RelP / DLA) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        g = r.get("grid", {})
        foc = r.get("focal", {})
        fs = []
        for nm, fd in foc.items():
            c = fd["curves"]
            ks = [c.get(k, {}).get("k80") for k in ("relp_o", "dla", "atp_o", "mag", "rand")]
            fs.append(f"{nm}: " + " / ".join("never" if (v is None or np.isnan(v)) else f"{v:.0f}" for v in ks))
        sr = "; ".join(f"{nm}: {_f(fd['single_r_relp_o'])} / {_f(fd['single_r_dla'])}" for nm, fd in foc.items())
        mu = r.get("multi", {})
        L.append(f"| {r['meta']['model']} / {r['meta']['task']} | {r['n_cases']} | {_f(g.get('all', {}).get('r_relp_o'))}; "
                 f"{_f(g.get('final', {}).get('r_relp_o'))}; {_f(g.get('nonfinal', {}).get('r_relp_o'))} | "
                 f"{_f(g.get('all', {}).get('r_atp_o'))}; {_f(g.get('nonfinal', {}).get('r_atp_o'))} | {_f(g.get('all', {}).get('slope_relp_o'))} | "
                 f"{_f(g.get('switch_rate'))} | {_f(g.get('exact_abs_share_final'))} | {'<br>'.join(fs)} | {sr} | "
                 f"{_k(mu.get('relp_o', {}).get('k80_drop'))} / {_k(mu.get('dla', {}).get('k80_drop'))} |")
    L2 = ["| model / task | pool | STR denoise frac. of drop at N = 10 / 100 / 1000 / 10000 | N for 80 % (denoise) | "
          "mean-ablation faithfulness at N = 10 / 100 / 1000 / 10000 | N for F = 0.8 | completeness at N = 100 / 1000 |", "|---|---|---|---|---|---|---|"]
    for r in res:
        for pool in ("all", "final_swap", "final"):
            s = r.get("circuits", {}).get(f"str|{pool}", {})
            m = r.get("circuits", {}).get(f"mean|{pool}", {})
            L2.append(f"| {r['meta']['model']} / {r['meta']['task']} | {pool} | {_at(s.get('denoise'), 'frac_of_drop', (10, 100, 1000, 10000))} | "
                      f"{_k(s.get('denoise', {}).get('N80'))} | {_at(m.get('faithful'), 'score', (10, 100, 1000, 10000))} | "
                      f"{_k(m.get('faithful', {}).get('N80'))} | {_at(m.get('complete'), 'score', (100, 1000))} |")
    L3 = ["| model / task | mode | top-N | at final position | at swap position | in between | before swap | mean layer | share of |attr| |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        for t in r.get("top_nodes", []):
            L3.append(f"| {r['meta']['model']} / {r['meta']['task']} | {t['mode']} | {t['N']} | {t['frac_final']:.2f} | {t['frac_swap']:.2f} | "
                      f"{t['frac_between']:.2f} | {t['frac_before']:.2f} | {t['mean_layer']:.1f} | {t['mass_share']:.2f} |")
    open(os.path.join(TAB, f"ext14ci_relp_validation{sfx}.md"), "w").write("\n".join(L) + "\n")
    open(os.path.join(TAB, f"ext14ci_relp_circuits{sfx}.md"), "w").write("\n".join(L2) + "\n")
    open(os.path.join(TAB, f"ext14ci_relp_positions{sfx}.md"), "w").write("\n".join(L3) + "\n")
    fig_circuits(res, os.path.join(FIG, f"ext14ci_relp_circuits{sfx}.png"))
    if a.smoke:
        json.dump(res, open(os.path.join(RESULTS, "ext14ci_summary_relp_smoke.json"), "w"), indent=1, default=C._json_default)
    else:
        C.update_summary("c2", {"runs": res})
    print("relp analysis:", [r["run"] for r in res])


def _f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _k(x):
    return "never" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.0f}" if x >= 10 else f"{x:.1f}")


def _at(d, key, Ns):
    if not d:
        return "–"
    m = dict(zip(d["N"], d[key]))
    return " / ".join(_f(m.get(N)) for N in Ns)


if __name__ == "__main__":
    main()
