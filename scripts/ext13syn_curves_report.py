"""ext13syn Task B (P5-4) computation, tables and figures; called by scripts/ext13syn_all.py (agent p5-synth)."""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext13syn as S
from moetrace.models import RESULTS

TAB, FIG = os.path.join(RESULTS, "tables"), os.path.join(RESULTS, "figures")
GPU_DIR = os.path.join(RESULTS, "ext13syn_gpucheck")
BANDS = ("first half", "L/2..L-9", "L-8..L-5", "last 4")


def level_of(key: str):
    """add-back task key -> (ladder level, proto, family)."""
    if key.startswith("cf_"):
        return "cf", ("qwen3" if key.endswith("qwen3") else "mixtral_bos"), "main"
    if key.startswith("wino_"):
        return "p3_opt", ("qwen3" if key.endswith("qwen3") else "mixtral_bos"), "main"
    fam = "rep" if key.endswith("_rep") else "main"
    k = key[:-4] if fam == "rep" else key
    proto = "mixtral_bos" if k.endswith("mixtral_bos") else "qwen3"
    cond = k[: -len("_" + proto)]
    return cond, proto, fam


def ladder_vals(lad, level, proto):
    if not lad:
        return {}
    for r in lad["rows"]:
        if r["level"] == level and r["proto"] == proto and r["stratum"] == "all":
            return {"share": r.get("share"), "A_direct": r.get("A_direct"), "M_all": r.get("M_all")}
    return {}


def case_direct(c, cases):
    """Per validation case: direct attention share (A_direct / drop, exact final norm): base run direct_split, else the
    add-back run's addback_direct (CounterFact; donor rows averaged)."""
    p = os.path.join(RESULTS, c["src"], "direct_split.parquet")
    if os.path.exists(p):
        x = pd.read_parquet(p).drop_duplicates("case_id").set_index("case_id")
        if set(cases) <= set(x.index):
            x = x.loc[cases]
            return ((x.d_attn_direct - x.d_corrupt_fp32) / (x.d_clean_fp32 - x.d_corrupt_fp32)).to_numpy(float)
    d = c["R"].get("direct")
    if d is None:
        return None
    g = d.groupby("case_id")[["d_attn_direct", "d_corrupt_fp32", "d_clean_fp32"]].mean().reindex(cases)
    return ((g.d_attn_direct - g.d_corrupt_fp32) / (g.d_clean_fp32 - g.d_corrupt_fp32)).to_numpy(float)


def gpucheck(C, key, c, nb):
    p = os.path.join(GPU_DIR, f"rows_{key}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    piv_r = df.pivot_table(index="case_id", columns="variant", values="rescue", aggfunc="mean")
    drop = df.groupby("case_id")["drop"].mean().reindex(piv_r.index)
    unit = pd.Series(piv_r.index // 2, index=piv_r.index)
    from moetrace import ext8_addback as X
    rc = X.ratio_cols(piv_r, drop, unit)
    out = {"n_cases": int(len(piv_r)), "r": {v: [float(x.ratio), float(x.lo), float(x.hi)] for v, x in rc.iterrows()}}
    diffs = {"noneg - all_moe": ("noneg", "all_moe"), "noneg_late - all_moe": ("noneg_late", "all_moe"),
             "noneg_early - all_moe": ("noneg_early", "all_moe"), "noneg_row - all_moe": ("noneg_row", "all_moe"),
             "late mismatch | MoE upstream (int_moe - all_moe)": ("int_moe", "all_moe"),
             "late mismatch | attention upstream (intattn - intattn_latemoe)": ("intattn", "intattn_latemoe"),
             "late mismatch | both upstream (intblock - intblock_latemoe)": ("intblock", "intblock_latemoe")}
    D = pd.DataFrame({k: piv_r[a] - piv_r[b] for k, (a, b) in diffs.items() if a in piv_r and b in piv_r})
    rd = X.ratio_cols(D, drop, unit)
    out["diff"] = {v: [float(x.ratio), float(x.lo), float(x.hi)] for v, x in rd.iterrows()}
    if "negonly" in piv_r and "noneg" in piv_r:
        add = pd.DataFrame({"negonly + noneg - all_moe": piv_r["negonly"] + piv_r["noneg"] - piv_r["all_moe"]})
        ra = X.ratio_cols(add, drop, unit)
        out["additivity"] = [float(ra.ratio.iloc[0]), float(ra.lo.iloc[0]), float(ra.hi.iloc[0])]
    meta_p = os.path.join(GPU_DIR, "run_meta.json")
    if os.path.exists(meta_p):
        out["meta"] = json.load(open(meta_p)).get("tasks", {}).get(key)
    return out


def run(C, lad, nb, log):
    out = {"conditions": {}, "definitions": {
        "overshoot": "max_k r_oracle(k) - M (population curve: sum of rescues / sum of drops; oracle = each case's own single-expert ranking)",
        "late": "M(<= L-5) - M: A0 'int_moe' minus 'all_moe'; > 0 = the last four MoE layers (at their clean values) work against the answer",
        "N": "cross-fitted negative single-expert mass: validation sum of the singles of the (layer, expert) pairs whose discovery all-case population effect is < 0, / drop",
        "D": "DLA of delta_e (final norm frozen at the corrupted run), same sets / bands, / drop",
        "compensation": "per case OLS y = DLA(last-4 MoE outputs) ~ x_A = DLA(all attention outputs) + x_M = DLA(MoE outputs below L-4 + embedding), logits; base run dla_rows (exact); CounterFact: addback_direct + clean-active expert DLA (approximation)"}}
    keys = C.condition_keys()
    for key in keys:
        level, proto, fam = level_of(key)
        log(f"curves {key}")
        try:
            c = C.load(key)
        except Exception as ex:  # noqa
            log(f"  load failed: {ex}")
            continue
        df = C.case_table(c)
        ob = C.overshoot_boot(c, nb)
        cols = ["N", "P", "U", "DN", "DP", "Nrow", "Prow"] + [f"{g}|{b}" for g in ("N", "DN", "P", "DP", "T", "DT") for b in BANDS]
        rat = C.ratios(df, cols)
        drop = df["drop"]
        e = {"key": key, "level": level, "proto": proto, "model": S.MSHORT[proto], "family": fam, "tlabel": c["tlabel"], "L": c["L"], "K": c["K"],
             "n_cases": int(len(df)), "n_units": int(df.unit.nunique()), "mean_drop": float(drop.mean()), "kstar": ob["kstar"],
             "n_neg_pairs": df.attrs["n_neg"], "n_pos_pairs": df.attrs["n_pos"], "curve": ob["curve"],
             **{k: ob[k] for k in ("M", "M_int", "late", "max_r", "overshoot", "r_end")}, "ratios": rat}
        # additivity of the singles: P + N + U against the joint all-MoE patch
        sm = df[["N", "P", "U"]].sum(1)
        e["sum_singles"] = float(sm.sum() / drop.sum())
        # direct share of the negative set by band
        e["neg_direct_share"] = {b: (rat[f"DN|{b}"][0] / rat[f"N|{b}"][0] if abs(rat[f"N|{b}"][0]) > 0.02 else None) for b in BANDS}
        e["neg_direct_share_all"] = rat["DN"][0] / rat["N"][0] if abs(rat["N"][0]) > 0.02 else None
        e["neg_late_frac"] = rat["N|last 4"][0] / rat["N"][0] if rat["N"][0] < -0.02 else None
        # layer profile (point estimates)
        L = c["L"]
        e["layers"] = {"N": [float(df[f"N@{l}"].sum() / drop.sum()) for l in range(L)], "P": [float(df[f"P@{l}"].sum() / drop.sum()) for l in range(L)],
                       "DN": [float(df[f"DN@{l}"].sum() / drop.sum()) for l in range(L)]}
        # per-case relations
        os_c = (df.O_kstar - df.M) / drop
        negm = -df.N / drop
        negl = -df["N|last 4"] / drop
        late_c = (df.Mint - df.M) / drop
        tlate = -df["T|last 4"] / drop
        units = df.unit.to_numpy()
        adir = case_direct(c, list(df.index))
        e["per_case"] = {"r(overshoot_c, A_direct_c)": C.corr_boot(os_c.to_numpy(), adir, units) if adir is not None else None,
                         "r(late_c, A_direct_c)": C.corr_boot(late_c.to_numpy(), adir, units) if adir is not None else None,
                         "r(overshoot_c, -N_c)": C.corr_boot(os_c.to_numpy(), negm.to_numpy(), units),
                         "r(overshoot_c, -N_late_c)": C.corr_boot(os_c.to_numpy(), negl.to_numpy(), units),
                         "r(overshoot_c, late_c)": C.corr_boot(os_c.to_numpy(), late_c.to_numpy(), units),
                         "r(late_c, -N_late_c)": C.corr_boot(late_c.to_numpy(), negl.to_numpy(), units),
                         "r(late_c, -T_late_c)": C.corr_boot(late_c.to_numpy(), tlate.to_numpy(), units),
                         "sd_overshoot_c": float(os_c.std()), "mean_overshoot_c": float(os_c.mean())}
        # compensation regression
        val = list(df.index)
        dl = C.dla_layers(c["src"], val) if level != "cf" else None
        src_lab = "dla_rows (exact)"
        if dl is None:
            dl = C.dla_from_addback(c, val)
            src_lab = "addback DLA (approximation)"
        if dl is not None:
            ok = dl[["y_late_moe", "x_attn", "x_moe_early"]].notna().all(1).to_numpy()
            Xm = dl[["x_attn", "x_moe_early"]].to_numpy(float)
            y = dl.y_late_moe.to_numpy(float)
            co = C.ols_boot(Xm[ok], y[ok], units[ok])
            e["compensation"] = {"source": src_lab, "n": int(ok.sum()), "intercept": co[0], "b_attn": co[1], "b_moe_early": co[2],
                                 "mean_y": float(np.nanmean(y)), "mean_x_attn": float(np.nanmean(Xm[:, 0])), "mean_x_moe_early": float(np.nanmean(Xm[:, 1])),
                                 "r_y_xattn": C.corr_boot(y[ok], Xm[ok, 0], units[ok]), "r_y_xmoe": C.corr_boot(y[ok], Xm[ok, 1], units[ok])}
            if "y_late_moe_clean" in dl:
                Xc = dl[["x_attn_clean", "x_moe_early_clean"]].to_numpy(float)
                yc = dl.y_late_moe_clean.to_numpy(float)
                okc = np.isfinite(yc) & np.isfinite(Xc).all(1)
                cc = C.ols_boot(Xc[okc], yc[okc], units[okc])
                e["compensation_clean_only"] = {"intercept": cc[0], "b_attn": cc[1], "b_moe_early": cc[2]}
        # attention side of the same rows
        if fam == "main":
            e["ladder"] = ladder_vals(lad, level, proto)
        else:
            try:
                d = S.PairData(S.run_of(level, proto), "rep", ("validation",), S.pairs_file(level, proto), proto)
                s = S.summarise(d, nb=1000)
                e["ladder"] = {"share": s.get("share"), "A_direct": s.get("A_direct"), "M_all": s.get("M_all")}
            except Exception:  # noqa
                e["ladder"] = {}
        g = gpucheck(C, key, c, nb)
        if g:
            e["gpucheck"] = g
        out["conditions"][key] = e
    # ---------------- across conditions (main family)
    main = [e for e in out["conditions"].values() if e["family"] == "main"]
    feats = {"-N": lambda e: -e["ratios"]["N"][0], "-N_late": lambda e: -e["ratios"]["N|last 4"][0], "late": lambda e: e["late"][0],
             "-T_late": lambda e: -e["ratios"]["T|last 4"][0], "-DT_late": lambda e: -e["ratios"]["DT|last 4"][0],
             "A_direct": lambda e: (e.get("ladder") or {}).get("A_direct", [np.nan])[0] if (e.get("ladder") or {}).get("A_direct") else np.nan,
             "share": lambda e: (e.get("ladder") or {}).get("share", [np.nan])[0] if (e.get("ladder") or {}).get("share") else np.nan,
             "M": lambda e: e["M"][0]}
    rng = np.random.default_rng(0)
    xc = {}
    for scope in ("both", "qwen3", "mixtral_bos"):
        ee = [e for e in main if scope == "both" or e["proto"] == scope]
        if len(ee) < 5:
            continue
        y = np.array([e["overshoot"][0] for e in ee])
        res = {"n": len(ee)}
        for name, f in feats.items():
            x = np.array([f(e) for e in ee], dtype=float)
            ok = np.isfinite(x)
            if ok.sum() < 5:
                continue
            xx, yy = x[ok], y[ok]
            r0 = float(np.corrcoef(xx, yy)[0, 1])
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(xx), len(xx))
                if np.std(xx[i]) > 0 and np.std(yy[i]) > 0:
                    bs.append(np.corrcoef(xx[i], yy[i])[0, 1])
            sl = np.polyfit(xx, yy, 1)
            res[name] = {"r": [r0, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "slope": float(sl[0]), "intercept": float(sl[1]),
                         "spearman": float(pd.Series(xx).corr(pd.Series(yy), method="spearman"))}
        xc[scope] = res
    out["across"] = xc
    # pooled within-condition per-case relation (condition-demeaned), per model
    return out


# ------------------------------------------------------------------------------------------------------------------
def fci(x, d=2, sign=True):
    if x is None or x[0] is None or not np.isfinite(x[0]):
        return "n/a"
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    if len(x) < 3 or x[1] is None or not np.isfinite(x[1]):
        return f.format(x[0])
    return f"{f.format(x[0])} [{f.format(x[1])}, {f.format(x[2])}]"


def save(df, name, caption):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    cols = list(df.columns)
    lines = [f"**{caption}**", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    open(os.path.join(TAB, name + ".md"), "w").write("\n".join(lines) + "\n")


ORDER_HINT = ("cf", "p3_opt", "dir_opt", "vt_e2a_orig", "vt_e2a_neu", "cp_nat", "cp_vt_orig", "cp_vt_neu", "dir_f1", "f1", "vt_a2e_orig", "f2e")


def sorted_conditions(cur):
    cs = list(cur["conditions"].values())
    oi = {k: i for i, k in enumerate(ORDER_HINT)}
    return sorted(cs, key=lambda e: (e["proto"] != "qwen3", e["family"] != "main", oi.get(e["level"], 99)))


def tables_and_figures(summ):
    cur = summ["curves"]
    cs = sorted_conditions(cur)
    rows, rows2, rows3, rows4 = [], [], [], []
    for e in cs:
        R = e["ratios"]
        lab = e["level"] + (" [rep]" if e["family"] == "rep" else "")
        lad = e.get("ladder") or {}
        nds = e.get("neg_direct_share") or {}
        rows.append({"Model": e["model"], "Condition": lab, "Cases (units)": f"{e['n_cases']} ({e['n_units']})",
                     "All-MoE M": fci(e["M"], 3, False), "Oracle max r (k*)": f"{e['max_r'][0]:.3f} ({e['kstar']})",
                     "Overshoot": fci(e["overshoot"], 3), "Late MoE: M(≤L−5) − M": fci(e["late"], 3),
                     "−N (cross-fitted neg. singles)": fci([-R["N"][0], -R["N"][2], -R["N"][1]], 3),
                     "−N in last 4 layers": fci([-R["N|last 4"][0], -R["N|last 4"][2], -R["N|last 4"][1]], 3),
                     "D/T of neg. set, last 4": f"{nds.get('last 4'):.2f}" if nds.get("last 4") is not None else "n/a",
                     "All experts, last 4: T / D": f"{R['T|last 4'][0]:+.3f} / {R['DT|last 4'][0]:+.3f}",
                     "Σ singles (P+N+U)": f"{e['sum_singles']:.2f}",
                     "A_direct": fci(lad.get("A_direct"), 2, False) if lad.get("A_direct") else "n/a",
                     "Attention share": fci(lad.get("share"), 2, False) if lad.get("share") else "n/a"})
        pc = e["per_case"]
        rows2.append({"Model": e["model"], "Condition": lab, "Cases": e["n_cases"], "SD of case overshoot at k*": f"{pc['sd_overshoot_c']:.3f}",
                      "r(overshoot, −N)": fci(pc["r(overshoot_c, -N_c)"]), "r(overshoot, −N late)": fci(pc["r(overshoot_c, -N_late_c)"]),
                      "r(overshoot, late)": fci(pc["r(overshoot_c, late_c)"]), "r(late, −N late)": fci(pc["r(late_c, -N_late_c)"]),
                      "r(late, −T late)": fci(pc["r(late_c, -T_late_c)"]),
                      "r(overshoot, A_direct)": fci(pc.get("r(overshoot_c, A_direct_c)")), "r(late, A_direct)": fci(pc.get("r(late_c, A_direct_c)"))})
        cp = e.get("compensation")
        if cp:
            rows3.append({"Model": e["model"], "Condition": lab, "Source": cp["source"], "Cases": cp["n"],
                          "Mean late-MoE DLA (logits)": f"{cp['mean_y']:+.2f}", "Mean attention DLA": f"{cp['mean_x_attn']:+.2f}",
                          "Mean early-MoE DLA": f"{cp['mean_x_moe_early']:+.2f}",
                          "b(attention)": fci(cp["b_attn"], 3), "b(early MoE)": fci(cp["b_moe_early"], 3),
                          "clean-only b(attention) / b(early MoE)": (f"{cp['compensation_clean_only']['b_attn'][0]:+.3f} / {cp['compensation_clean_only']['b_moe_early'][0]:+.3f}"
                                                                     if cp.get("compensation_clean_only") else "n/a")})
        for b in BANDS:
            rows4.append({"Model": e["model"], "Condition": lab, "Band": b, "N / drop": f"{R[f'N|{b}'][0]:+.3f}", "DN / drop": f"{R[f'DN|{b}'][0]:+.3f}",
                          "P / drop": f"{R[f'P|{b}'][0]:+.3f}", "T (all) / drop": f"{R[f'T|{b}'][0]:+.3f}", "DT (all) / drop": f"{R[f'DT|{b}'][0]:+.3f}"})
    save(pd.DataFrame(rows), "ext13syn_curves_summary",
         "Add-back curve shape per condition (validation cases; fraction of the drop; unit-cluster bootstrap 95% CIs)")
    save(pd.DataFrame(rows2), "ext13syn_curves_rows", "Per-case relations within each condition (Pearson r, unit bootstrap 95% CI)")
    save(pd.DataFrame(rows3), "ext13syn_curves_compensation",
         "Compensation regression per case: DLA of the last-4 MoE outputs on the DLA of all attention outputs and of the earlier MoE outputs (logits; clean minus corrupted)")
    save(pd.DataFrame(rows4), "ext13syn_curves_bands", "Cross-fitted negative (N) / positive (P) single-expert mass and all experts (T) by depth band, with their direct parts (D)")
    xr = []
    for scope, res in (cur.get("across") or {}).items():
        for name, v in res.items():
            if name == "n":
                continue
            xr.append({"Scope": scope, "Conditions": res["n"], "Predictor of overshoot": name, "Pearson r [bootstrap over conditions]": fci(v["r"]),
                       "Spearman": f"{v['spearman']:+.2f}", "Slope": f"{v['slope']:+.2f}", "Intercept": f"{v['intercept']:+.3f}"})
    save(pd.DataFrame(xr), "ext13syn_curves_across", "Across conditions (main families): overshoot against candidate predictors")
    gr = []
    for e in cs:
        g = e.get("gpucheck")
        if not g:
            continue
        for v, x in g["r"].items():
            gr.append({"Condition": e["key"], "Row": v, "Value / drop": fci(x, 3), "Kind": "variant"})
        for v, x in g.get("diff", {}).items():
            gr.append({"Condition": e["key"], "Row": v, "Value / drop": fci(x, 3), "Kind": "difference"})
        if g.get("additivity"):
            gr.append({"Condition": e["key"], "Row": "negonly + noneg - all_moe (additivity)", "Value / drop": fci(g["additivity"], 3), "Kind": "difference"})
        gr.append({"Condition": e["key"], "Row": "oracle overshoot (add-back run)", "Value / drop": fci(e["overshoot"], 3), "Kind": "reference"})
    if gr:
        save(pd.DataFrame(gr), "ext13syn_curves_gpucheck", "GPU check (Qwen3, validation rows; same-pass values): add back all experts except the cross-fitted negative ones; late-MoE reaction test")
    figures(cur, cs)


def figures(cur, cs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"qwen3": "#1f6fb4", "mixtral_bos": "#d1612a"}
    dmk = {"cf": "D", "p3_opt": "v", "dir_opt": "v", "vt_e2a_orig": "v", "vt_e2a_neu": "v", "cp_nat": "^", "cp_vt_orig": "^", "cp_vt_neu": "^",
           "dir_f1": "o", "f1": "o", "vt_a2e_orig": "o", "f2e": "o"}
    main = [e for e in cs if e["family"] == "main"]
    # 1. overshoot against predictors
    preds = [("late", "late MoE: M(≤L−5) − M", lambda e: e["late"]),
             ("-N_late", "−N, last 4 layers (cross-fitted)", lambda e: [-e["ratios"]["N|last 4"][0], -e["ratios"]["N|last 4"][2], -e["ratios"]["N|last 4"][1]]),
             ("A_direct", "direct attention share A_direct", lambda e: (e.get("ladder") or {}).get("A_direct"))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (nm, lab, f) in zip(axes, preds):
        for e in main:
            x = f(e)
            if not x or x[0] is None:
                continue
            y = e["overshoot"]
            ax.errorbar(x[0], y[0], xerr=None if x[1] is None else [[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                        fmt=dmk.get(e["level"], "o"), color=col[e["proto"]], ms=6, lw=0.8, alpha=0.9)
            ax.annotate(e["level"], (x[0], y[0]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        r = (cur.get("across", {}).get("both", {}).get(nm) or {}).get("r")
        ax.set_xlabel(lab + (f"   (r = {r[0]:+.2f})" if r else ""))
        ax.set_ylabel("oracle overshoot: max r − M")
        ax.axhline(0, color="grey", lw=0.6)
        if nm != "A_direct":
            lim = [min(-0.05, ax.get_xlim()[0]), max(0.45, ax.get_xlim()[1])]
            ax.plot(lim, lim, ls=":", color="grey", lw=0.8)
        ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=col["qwen3"], marker="o", ls="", label="Qwen3"), Line2D([], [], color=col["mixtral_bos"], marker="o", ls="", label="Mixtral"),
          Line2D([], [], color="k", marker="o", ls="", label="attribute → entity"), Line2D([], [], color="k", marker="v", ls="", label="entity → attribute"),
          Line2D([], [], color="k", marker="^", ls="", label="+ copy prefix"), Line2D([], [], color="k", marker="D", ls="", label="CounterFact")]
    axes[0].legend(handles=hs, fontsize=7, loc="upper left")
    fig.suptitle("Why the add-back curves overshoot: overshoot vs the late MoE term, the late negative experts and the direct attention share (main families)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext13syn_curves_overshoot.{ext}"), dpi=150)
    plt.close(fig)
    # 2. layer profiles of the cross-fitted negative mass (and its DLA)
    pick = ("cf", "p3_opt", "dir_opt", "cp_nat", "cp_vt_orig", "dir_f1", "f1", "vt_e2a_orig")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    cmap = plt.get_cmap("tab10")
    for j, proto in enumerate(("qwen3", "mixtral_bos")):
        ee = [e for e in main if e["proto"] == proto and e["level"] in pick]
        for i, e in enumerate(ee):
            L = e["L"]
            x = (np.arange(L) + 0.5) / L
            st = "-" if e["level"] in ("dir_f1", "f1", "vt_a2e_orig") else ("--" if e["level"] in ("cf",) else ":")
            axes[0, j].plot(x, e["layers"]["N"], st, color=cmap(i), lw=1.3, label=e["level"])
            axes[1, j].plot(x, e["layers"]["P"], st, color=cmap(i), lw=1.3, label=e["level"])
        axes[0, j].set_title(f"{S.MSHORT[proto]}: cross-fitted NEGATIVE single-expert mass per layer / drop", fontsize=9)
        axes[1, j].set_title(f"{S.MSHORT[proto]}: cross-fitted POSITIVE single-expert mass per layer / drop", fontsize=9)
        for a in axes[:, j]:
            a.axhline(0, color="grey", lw=0.6)
            a.axvline((L - 4) / L, color="grey", lw=0.6, ls=":")
            a.grid(alpha=0.3)
        axes[1, j].set_xlabel("relative depth (layer + 0.5) / L; dotted line = L − 4")
        axes[0, j].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext13syn_curves_layers.{ext}"), dpi=150)
    plt.close(fig)
    # 3. compensation coefficients
    ee = [e for e in main if e.get("compensation")]
    if ee:
        fig, ax = plt.subplots(1, 2, figsize=(12, 0.32 * len(ee) + 1.6), sharey=True)
        for i, e in enumerate(ee):
            for a, k in zip(ax, ("b_attn", "b_moe_early")):
                v = e["compensation"][k]
                a.errorbar(v[0], i, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=dmk.get(e["level"], "o"), color=col[e["proto"]], ms=5, lw=1)
        for a, lab in zip(ax, ("b(attention DLA)", "b(early-MoE DLA)")):
            a.axvline(0, color="grey", lw=0.8)
            a.set_xlabel(lab + " in  DLA(last-4 MoE) ~ attention + early MoE")
            a.grid(axis="x", alpha=0.3)
        ax[0].set_yticks(range(len(ee)))
        ax[0].set_yticklabels([f"{e['level']} ({e['model']})" for e in ee], fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"ext13syn_curves_compensation.{ext}"), dpi=150)
        plt.close(fig)
    # 4. GPU check
    gg = [e for e in cs if e.get("gpucheck")]
    if gg:
        vs = ["all_moe", "noneg", "noneg_late", "noneg_early", "noneg_row", "negonly", "int_moe", "lateonly", "intattn", "intattn_latemoe", "intblock", "intblock_latemoe"]
        fig, ax = plt.subplots(1, 1, figsize=(13, 4.2))
        w = 0.8 / len(gg)
        for i, e in enumerate(gg):
            r = e["gpucheck"]["r"]
            xs = np.arange(len(vs)) + i * w
            vals = [r.get(v, [np.nan] * 3) for v in vs]
            ax.bar(xs, [v[0] for v in vals], w, yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], label=e["level"], capsize=2)
            ax.plot([xs[0] - w / 2, xs[0] + w * len(gg)], [e["max_r"][0]] * 2, ls=":", lw=0.8, color=f"C{i}")
        ax.set_xticks(np.arange(len(vs)) + 0.4 - w / 2)
        ax.set_xticklabels(vs, rotation=30, fontsize=8)
        ax.set_ylabel("rescue / drop (same pass)")
        ax.axhline(0, color="grey", lw=0.6)
        ax.legend(fontsize=8)
        ax.set_title("GPU check, Qwen3 validation rows: dotted = the add-back run's oracle maximum", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"ext13syn_curves_gpucheck.{ext}"), dpi=150)
        plt.close(fig)
