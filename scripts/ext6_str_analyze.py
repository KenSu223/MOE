"""ext6 STR analysis (CPU only): the paper's two-stage procedure under symmetric token replacement vs Gaussian noise
on the same cases.

For each protocol (Qwen3; Mixtral no BOS = paper protocol; Mixtral BOS):
  STR   results/<proto>_str        donor-mean per case (primary) and first donor (sensitivity)
  GN    results/<gn all-layer run> the base sweep + Direction-1 all-layer expert pass, restricted to the STR case subset
        (same split, same recurrence threshold = half the retained discovery cases), and on the full paper set
Quantities: layer curve, discovery L*, validation rescue at L*, sharpness; normalised rescue (Zhang & Nanda:
rescue / (Delta_clean - Delta_corrupt)); recurrence-first expert at L* and at the paper's layer with validation rescue
and Spec; fixed-hypothesis rows for the named experts of the earlier directions; joint (layer, expert) search over the
layers of the STR expert pass; coalitions; donor dispersion.

Usage: python scripts/ext6_str_analyze.py [--protocols qwen3,mixtral_nobos,mixtral_bos]
Outputs results/tables/ext6_str_*.{md,csv}, results/figures/ext6_str_layers.{png,pdf}, results/ext6_str_summary.json,
        results/sections/ext6_str.md
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import analysis as A
from moetrace import ext6_str as S
from moetrace.stats import summarize, ratio_ci, fmt

ROOT = "/home/ubuntu/MOE/results"
TAB, FIG = os.path.join(ROOT, "tables"), os.path.join(ROOT, "figures")
PROTO = {
    "qwen3": dict(str_run="qwen3_str", gn_run="qwen3_bos_alllayers", label="Qwen3-30B-A3B-Base", paper_layer=44,
                  named=[(44, 69), (42, 115)], n_controls=3),
    "mixtral_nobos": dict(str_run="mixtral_nobos_str", gn_run="mixtral_nobos_alllayers", label="Mixtral-8x7B, no BOS (paper protocol)",
                          paper_layer=19, named=[(19, 6), (19, 2), (18, 1)], n_controls=1),
    "mixtral_bos": dict(str_run="mixtral_bos_str", gn_run="mixtral_bos_alllayers", label="Mixtral-8x7B, BOS (tokenizer default)",
                        paper_layer=19, named=[(19, 2), (19, 6), (18, 1)], n_controls=1),
}


def pe(l, e):
    return f"L{l}E{e:03d}"


def md_table(header, rows, path, caption=None):
    lines = ([f"**{caption}**", ""] if caption else []) + ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    txt = "\n".join(lines) + "\n"
    with open(path + ".md", "w") as f:
        f.write(txt)
    pd.DataFrame(rows, columns=header).to_csv(path + ".csv", index=False)
    return txt


def restrict(md, sets):
    md.sets = {"paper": sets["paper"]}
    return md


def layers_with_experts(md):
    return sorted(md.expert_rows.layer.unique().tolist()) if md.expert_rows is not None else []


def norm_of(md, ev, key):
    """Population-level normalised value: mean(key) / mean(drop) over the validation cases, paired bootstrap."""
    pc = ev["per_case"].dropna(subset=[key])
    d = md.cases.set_index("case_id").loc[pc.case_id, "drop"].values
    return ratio_ci(pc[key].values, d)


def select_eval(md, layer, n_controls):
    disc, val = md.ids("paper", "discovery"), md.ids("paper", "validation")
    et = A.expert_table(md, layer)
    sel = A.select_expert(et, disc, len(disc) // 2)
    if sel["e_star"] is None:
        return sel, None
    ev = A.evaluate_expert(md, layer, sel["e_star"], val, n_controls)
    return sel, ev


def fixed(md, layer, e, n_controls):
    if layer not in layers_with_experts(md):
        return None
    val = md.ids("paper", "validation")
    disc = md.ids("paper", "discovery")
    ev = A.evaluate_expert(md, layer, e, val, n_controls)
    et = A.expert_table(md, layer)
    dact = int(et[et.case_id.isin(disc) & (et.expert == e)].shape[0])
    return {"disc_active": dact, "n_disc": len(disc), "val_active": ev["val_active"], "rescue": ev["rescue_all"], "spec": ev["spec_all"],
            "rescue_norm": norm_of(md, ev, "rescue"), "spec_norm": norm_of(md, ev, "spec")}


def joint(md, n_controls, top=10):
    disc, val = md.ids("paper", "discovery"), md.ids("paper", "validation")
    thr = len(disc) // 2
    e = md.expert_rows
    e = e[(e.kind == "expert") & e.clean_active & e.case_id.isin(disc)]
    g = e.groupby(["layer", "expert"])
    tab = pd.DataFrame({"act": g.size(), "allcase": g.rescue.sum() / len(disc)}).reset_index()
    tab = tab[tab.act >= thr].sort_values("allcase", ascending=False).head(top)
    out = []
    for r in tab.itertuples():
        ev = A.evaluate_expert(md, int(r.layer), int(r.expert), val, n_controls)
        out.append({"pair": pe(int(r.layer), int(r.expert)), "disc_active": int(r.act), "disc_allcase": float(r.allcase),
                    "val_active": ev["val_active"], "val_rescue": ev["rescue_all"], "val_spec": ev["spec_all"]})
    return out


def donor_dispersion(run, layer, val):
    sw = pd.read_parquet(os.path.join(ROOT, run, "str_sweep_rows.parquet"))
    x = sw[(sw.kind == "layer") & (sw.layer == layer) & sw.case_id.isin(val)]
    g = x.groupby("case_id").rescue
    multi = g.size() >= 2
    sd = g.std()[multi]
    mean = g.mean()[multi]
    sign = x.groupby("case_id").rescue.apply(lambda v: float(max((v > 0).mean(), (v <= 0).mean())))[multi]
    return {"n_cases_multi_donor": int(multi.sum()), "median_sd_across_donors": float(sd.median()),
            "median_abs_case_mean": float(mean.abs().median()), "mean_sign_agreement": float(sign.mean())}


def analyse(key):
    cfg = PROTO[key]
    sets = json.load(open(os.path.join(ROOT, cfg["str_run"], "case_sets.json")))
    out = {"label": cfg["label"], "str_run": cfg["str_run"], "gn_run": cfg["gn_run"]}
    fs = json.load(open(os.path.join(ROOT, cfg["str_run"], "str_filter_summary.json")))
    out["filter"] = {k: fs[k] for k in ("n_cases", "n_candidates", "cases_with_candidate", "cases_with_qualifying", "cases_kept",
                                        "kept_discovery", "kept_validation", "qualify_rate", "n_selected", "selected_per_case",
                                        "donor_top1_is_foil_rate_selected", "delta_donor_selected_mean")}
    mds = {"STR (donor mean)": S.model_data(cfg["str_run"], "mean"), "STR (first donor)": S.model_data(cfg["str_run"], "first"),
           "GN (same cases)": restrict(A.load_model(cfg["gn_run"]), sets), "GN (paper set)": A.load_model(cfg["gn_run"])}
    mds["GN (paper set)"].sets = {"paper": mds["GN (paper set)"].sets["paper"]}
    # ---- layer level
    lay = {}
    for name, md in mds.items():
        la = A.layer_analysis(md, "paper")
        val = md.ids("paper", "validation")
        Ls = la["L_star"]
        norm = S.normalised(md, Ls, val)
        norm_paper = S.normalised(md, cfg["paper_layer"], val)
        lay[name] = {"n_disc": la["n_disc"], "n_val": la["n_val"], "L_star": Ls, "disc_at_Lstar": la["disc_mean_at_Lstar"],
                     "val_at_Lstar": la["val_at_Lstar"], "sharpness": la["sharpness"], "disc_top5": la["disc_curve_top5"],
                     "val_at_paper_layer": summarize(md.R.loc[val, cfg["paper_layer"]].values),
                     "normalised_at_Lstar": norm, "normalised_at_paper_layer": norm_paper,
                     "val_curve": [c["val_mean"] for c in la["curve"]], "val_ci": [(c["ci_lo"], c["ci_hi"]) for c in la["curve"]],
                     "mean_delta_clean": float(md.cases.set_index("case_id").loc[md.ids("paper", "discovery") + val, "delta_clean"].mean()),
                     "mean_drop": float(md.cases.set_index("case_id").loc[md.ids("paper", "discovery") + val, "drop"].mean())}
    out["layers"] = lay
    s_cur, g_cur = np.array(lay["STR (donor mean)"]["val_curve"]), np.array(lay["GN (same cases)"]["val_curve"])
    out["curve_corr_str_gn"] = float(np.corrcoef(s_cur, g_cur)[0, 1])
    # per-case correlation at the paper layer
    Rs, Rg = mds["STR (donor mean)"].R, mds["GN (same cases)"].R
    val = mds["STR (donor mean)"].ids("paper", "validation")
    out["percase_corr_paper_layer"] = float(np.corrcoef(Rs.loc[val, cfg["paper_layer"]], Rg.loc[val, cfg["paper_layer"]])[0, 1])
    # ---- expert level
    str_layers = layers_with_experts(mds["STR (donor mean)"])
    out["str_expert_layers"] = str_layers
    exp = {}
    for name, md in mds.items():
        cand_layers = sorted(set([lay[name]["L_star"], cfg["paper_layer"]]) | {l for l, _ in cfg["named"]})
        res = {}
        for l in cand_layers:
            if l not in layers_with_experts(md):
                continue
            sel, ev = select_eval(md, l, cfg["n_controls"])
            if ev is None:
                res[l] = {"e_star": None, "n_candidates": 0, "max_activity": sel.get("max_activity")}
                continue
            co = A.coalitions(md, l, md.ids("paper", "validation"))
            res[l] = {"e_star": sel["e_star"], "n_candidates": sel["n_candidates"], "threshold": sel["threshold"],
                      "disc_active": sel["disc_active"], "disc_allcase": sel["disc_allcase_mean"], "top_candidates": sel["top_candidates"],
                      "val_active": ev["val_active"], "rescue": ev["rescue_all"], "spec": ev["spec_all"],
                      "rescue_norm": norm_of(md, ev, "rescue"), "spec_norm": norm_of(md, ev, "spec"),
                      "coalition_clean": co["coalition_clean"], "coalition_union": co["coalition_union"], "layer_same_pass": co["layer"]}
        exp[name] = res
    out["experts"] = exp
    out["named"] = {name: {pe(l, e): fixed(md, l, e, cfg["n_controls"]) for l, e in cfg["named"]} for name, md in mds.items()}
    # joint search over the layers of the STR expert pass (GN restricted to the same layers)
    js = {}
    for name in ("STR (donor mean)", "STR (first donor)", "GN (same cases)"):
        md = mds[name]
        if name.startswith("GN"):
            md = restrict(A.load_model(cfg["gn_run"]), sets)
            md.expert_rows = md.expert_rows[md.expert_rows.layer.isin(str_layers)]
        js[name] = joint(md, cfg["n_controls"])
    out["joint"] = js
    # activity of each joint top-1 on the FULL paper discovery set under GN (is it gated in only because the set shrank?)
    full = A.load_model(cfg["gn_run"])
    fd = full.sets["paper"]["discovery"]
    fe = full.expert_rows
    fe = fe[(fe.kind == "expert") & fe.clean_active & fe.case_id.isin(fd)]
    gate = {}
    for name, lst in js.items():
        if not lst:
            continue
        l, e = int(lst[0]["pair"][1:].split("E")[0]), int(lst[0]["pair"].split("E")[1])
        gate[name] = {"pair": lst[0]["pair"], "full_disc_active": int(((fe.layer == l) & (fe.expert == e)).sum()), "full_n_disc": len(fd),
                      "full_gate": len(fd) // 2}
    out["joint_top1_full_set_gate"] = gate
    # Table 11 analogue (Mixtral, top-2): selected expert vs the unique other clean-active expert, raw and at equal norm
    if cfg["n_controls"] == 1:
        eqn = {}
        for l, e in cfg["named"]:
            if l != cfg["paper_layer"]:
                continue
            for name in ("STR (donor mean)", "GN (same cases)"):
                md = mds[name]
                if name.startswith("GN"):
                    md = restrict(A.load_model(cfg["gn_run"].replace("_alllayers", "") if cfg["gn_run"] != "mixtral_bos_alllayers" else "mixtral"), sets)
                if md.expert_rows is None or not (md.expert_rows.kind == "expert_scaled").any():
                    continue
                r = A.active_pair_equal_norm(md, l, e, md.ids("paper", "validation"))
                eqn.setdefault(pe(l, e), {})[name] = {k: r[k] for k in ("n", "spec_raw", "selected_eq", "other_eq", "spec_eq")}
        out["equal_norm_pair"] = eqn
    out["donor_dispersion_Lstar"] = donor_dispersion(cfg["str_run"], lay["STR (donor mean)"]["L_star"], val)
    out["donor_dispersion_paper_layer"] = donor_dispersion(cfg["str_run"], cfg["paper_layer"], val)
    return out


def nf(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def write_tables(allres):
    os.makedirs(TAB, exist_ok=True)
    txt = []
    rows = []
    for key, r in allres.items():
        f = r["filter"]
        ls, lg = r["layers"]["STR (donor mean)"], r["layers"]["GN (same cases)"]
        dd = r["donor_dispersion_paper_layer"]
        rows.append([r["label"], f["n_candidates"], f"{f['cases_with_candidate']}/{f['n_cases']}", f"{f['cases_kept']} ({f['kept_discovery']}/{f['kept_validation']})",
                     f"{f['qualify_rate']:.2f}", f"{f['n_selected']} ({f['n_selected'] / f['cases_kept']:.1f}/case)",
                     f"{f['donor_top1_is_foil_rate_selected']:.2f}", f"{ls['mean_delta_clean']:+.2f}", f"{ls['mean_delta_clean'] - ls['mean_drop']:+.2f}",
                     f"{ls['mean_drop']:+.2f} / {lg['mean_drop']:+.2f}", f"{r['curve_corr_str_gn']:.3f}", f"{r['percase_corr_paper_layer']:.2f}",
                     f"{dd['median_sd_across_donors']:.2f} (|mean| {dd['median_abs_case_mean']:.2f}); sign agreement {dd['mean_sign_agreement']:.2f}"])
    txt.append(md_table(["Model / protocol", "Symmetric candidates", "Cases with a candidate", "Cases kept (disc/val)", "Qualify rate",
                         "Donor rows", "Donor top-1 = foil", "Mean Δ clean", "Mean Δ corrupt (STR)", "Mean drop STR / GN",
                         "r(val curve STR, GN)", "Per-case r at paper layer", "Donor dispersion at paper layer (median SD)"],
                        rows, os.path.join(TAB, "ext6_str_overview"), "STR construction and descriptors (same cases for STR and GN)"))
    rows = []
    for key, r in allres.items():
        for name, l in r["layers"].items():
            n = l["normalised_at_Lstar"]["ratio"]
            npl = l["normalised_at_paper_layer"]["ratio"]
            rows.append([r["label"], name, f"{l['n_disc']}/{l['n_val']}", f"L{l['L_star']}", fmt(l["val_at_Lstar"]),
                         f"L{l['sharpness']['top_layer']} (2nd L{l['sharpness']['next_layer']}, gap {l['sharpness']['gap']:+.3f})", fmt(l["val_at_paper_layer"]),
                         f"{l['mean_drop']:+.2f}", f"{n[0]:.3f} [{n[1]:.3f}, {n[2]:.3f}]", f"{npl[0]:.3f} [{npl[1]:.3f}, {npl[2]:.3f}]",
                         ", ".join(f"L{a} {b:+.2f}" for a, b in l["disc_top5"])])
    txt.append(md_table(["Model / protocol", "Corruption", "n disc/val", "L* (disc)", "Val rescue at L*", "Val argmax (2nd, gap)",
                         "Val rescue at paper layer", "Mean drop", "Normalised at L*", "Normalised at paper layer", "Discovery top 5"],
                        rows, os.path.join(TAB, "ext6_str_layers"),
                        "Layer-level tracing under STR vs GN (validation; normalised = mean rescue / mean drop, paired bootstrap)"))
    rows = []
    for key, r in allres.items():
        for name, res in r["experts"].items():
            for l, x in sorted(res.items()):
                if x.get("e_star") is None:
                    rows.append([r["label"], name, f"L{l}", "none", "", "", "", "", "", "", "", ""])
                    continue
                rows.append([r["label"], name, f"L{l}", pe(l, x["e_star"]), f"{x['disc_active']}/{r['layers'][name]['n_disc']}",
                             f"{x['n_candidates']} (>= {x['threshold']})", fmt(x["rescue"]), fmt(x["spec"]), nf(x["spec_norm"]),
                             fmt(x["coalition_clean"]), fmt(x["layer_same_pass"]),
                             ", ".join(f"E{e:03d} {v:+.2f} ({a})" for e, v, a in x["top_candidates"][:4])])
    txt.append(md_table(["Model / protocol", "Corruption", "Layer", "Selected", "Disc. active", "Candidates", "Val rescue", "Spec", "Spec / drop",
                         "Clean top-k coalition", "Layer (same pass)", "Top candidates (disc. all-case, active)"],
                        rows, os.path.join(TAB, "ext6_str_experts"),
                        "Recurrence-first expert selection at L* and at the paper's layer (validation rescue and Spec)"))
    rows = []
    for key, r in allres.items():
        for name, d in r["named"].items():
            for p, x in d.items():
                if x is None:
                    continue
                rows.append([r["label"], name, p, f"{x['disc_active']}/{x['n_disc']}", x["val_active"], fmt(x["rescue"]), fmt(x["spec"]),
                             nf(x["rescue_norm"]), nf(x["spec_norm"])])
    txt.append(md_table(["Model / protocol", "Corruption", "Expert", "Disc. active", "Val active", "Val rescue", "Spec", "Rescue / drop", "Spec / drop"], rows,
                        os.path.join(TAB, "ext6_str_named"), "Fixed-hypothesis validation of the experts named in the earlier directions"))
    rows = []
    for key, r in allres.items():
        for name, lst in r["joint"].items():
            for i, x in enumerate(lst[:5]):
                rows.append([r["label"], name, i + 1, x["pair"], x["disc_active"], f"{x['disc_allcase']:+.3f}", fmt(x["val_rescue"]), fmt(x["val_spec"])])
    txt.append(md_table(["Model / protocol", "Corruption", "Rank", "(layer, expert)", "Disc. active", "Disc. all-case", "Val rescue", "Spec"], rows,
                        os.path.join(TAB, "ext6_str_joint"), "Joint (layer, expert) search over the layers of the STR expert pass (top 5, recurrence gate = half of discovery)"))
    rows = []
    for key, r in allres.items():
        for p, d in r.get("equal_norm_pair", {}).items():
            for name, x in d.items():
                rows.append([r["label"], name, p, x["n"], fmt(x["spec_raw"]), fmt(x["selected_eq"]), fmt(x["other_eq"]), fmt(x["spec_eq"])])
    if rows:
        txt.append(md_table(["Model / protocol", "Corruption", "Expert", "n (anchor-active val)", "Raw active-pair Spec", "Selected, equal norm",
                             "Other active, equal norm", "Equal-norm Spec"], rows, os.path.join(TAB, "ext6_str_eqnorm"),
                            "Mixtral active-pair equal-norm check at L19 (paper Table 11 analogue): both patch vectors scaled to the smaller norm"))
    return "\n".join(txt)


def figure(allres):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    n = len(allres)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 6.6), squeeze=False)
    styles = (("STR (donor mean)", dict(color="#1f77b4", lw=2)), ("GN (same cases)", dict(color="#d62728", lw=1.6)),
              ("STR (first donor)", dict(color="#1f77b4", lw=1, ls="--")))
    for j, (key, r) in enumerate(allres.items()):
        for row in (0, 1):
            ax = axes[row][j]
            for name, style in styles:
                l = r["layers"][name]
                scale = 1.0 if row == 0 else 1.0 / l["normalised_at_Lstar"]["mean_drop"]
                y = np.array(l["val_curve"]) * scale
                ax.plot(np.arange(len(y)), y, label=name, **style)
                if name != "STR (first donor)":
                    lo, hi = np.array(l["val_ci"]).T * scale
                    ax.fill_between(np.arange(len(y)), lo, hi, color=style["color"], alpha=0.12)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(PROTO[key]["paper_layer"], color="grey", lw=0.8, ls=":")
            ax.set_xlabel("MoE layer")
            if row == 0:
                ax.set_title(r["label"], fontsize=9)
                ax.set_ylabel("validation rescue (logits)")
                ax.legend(fontsize=7)
            else:
                ax.set_ylabel("rescue / mean validation drop")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ext6_str_layers.{ext}"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="qwen3,mixtral_nobos,mixtral_bos")
    args = ap.parse_args()
    allres = {}
    for key in args.protocols.split(","):
        if not os.path.exists(os.path.join(ROOT, PROTO[key]["str_run"], "str_sweep_rows.parquet")):
            print(f"skip {key}: no STR sweep yet")
            continue
        allres[key] = analyse(key)
        print(f"{key}: done")
    tables = write_tables(allres)
    figure(allres)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if k != "per_case"}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.generic,)):
            return o.item()
        return o
    with open(os.path.join(ROOT, "ext6_str_summary.json"), "w") as f:
        json.dump(clean(allres), f, indent=1, default=str)
    print(tables)


if __name__ == "__main__":
    main()
