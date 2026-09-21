"""ext5 F4 analysis: causal tracing at the last subject token versus the final token.

Inputs: results/<short>_subject/ (scripts/ext5_subject_sweep.py + ext5_subject_expert.py) and the final-token curves of
results/<short>_attnsweep/ (ext2). Writes results/tables/ext5_subject_*.{md,csv}, results/figures/ext5_subject_*.{png,pdf},
results/ext5_subject_summary.json and results/sections/ext5_f4_subject.md (appends
results/sections/ext5_f4_subject_interpretation.md if present).

Usage: python scripts/ext5_subject_analyze.py [--runs qwen3_bos,mixtral_nobos,mixtral_bos]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from moetrace.models import MODELS, RESULTS
from moetrace.stats import summarize, ratio_ci, bootstrap_ci
from moetrace.ext2_attn import auc_positive, centre_of_mass, KIND_LABEL
from moetrace import analysis as A
from moetrace.report import md_table

TAB, FIG, SEC = (os.path.join(RESULTS, d) for d in ("tables", "figures", "sections"))
for d in (TAB, FIG, SEC):
    os.makedirs(d, exist_ok=True)
CFG = {
    "qwen3_bos": {"model": "qwen3", "subject": "qwen3_bos_subject", "final": "qwen3_bos_attnsweep", "base": "qwen3",
                  "label": "Qwen3-30B-A3B-Base (tokenizer defaults)", "fixed": [(44, 69), (42, 115)]},
    "mixtral_nobos": {"model": "mixtral", "subject": "mixtral_nobos_subject", "final": "mixtral_nobos_attnsweep", "base": "mixtral_nobos",
                      "label": "Mixtral-8x7B-v0.1 (no BOS, paper protocol)", "fixed": [(19, 2), (19, 6), (18, 1)]},
    "mixtral_bos": {"model": "mixtral", "subject": "mixtral_bos_subject", "final": "mixtral_bos_attnsweep", "base": "mixtral",
                    "label": "Mixtral-8x7B-v0.1 (BOS, tokenizer default)", "fixed": [(19, 2), (19, 6), (18, 1)]},
}
KINDS = ("layer", "attn_layer", "resid")
COL = {"attn_layer": "#2a78d6", "layer": "#eb6834", "resid": "#52514e"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
SITE_LABEL = {"subject": "last subject token", "final": "final token"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def ci(s):
    return "n/a" if s is None or s["n"] == 0 else f"{s['mean']:+.3f} [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------------------------------------------------------
class Site:
    """One sweep (subject or final): rescue matrices per kind, discovery/validation ids, curves."""

    def __init__(self, run: str, kinds=KINDS, case_set="paper"):
        d = os.path.join(RESULTS, run)
        self.run, self.dir = run, d
        self.rows = pd.read_parquet(os.path.join(d, "sweep_rows.parquet"))
        self.sets = json.load(open(os.path.join(d, "case_sets.json")))
        self.cases = pd.read_parquet(os.path.join(d, "sweep_cases.parquet"))
        self.kinds = [k for k in kinds if k in set(self.rows.kind)]
        self.R = {k: self.rows[self.rows.kind == k].pivot(index="case_id", columns="layer", values="rescue") for k in self.kinds}
        self.layers = list(self.R[self.kinds[0]].columns)
        present = set(self.R[self.kinds[0]].index)
        self.disc = [c for c in self.sets[case_set]["discovery"] if c in present]
        self.val = [c for c in self.sets[case_set]["validation"] if c in present]
        self.dclean = self.rows[self.rows.kind == "clean"].set_index("case_id").delta
        self.dnoised = self.rows[self.rows.kind == "noised"].set_index("case_id").delta

    def curve(self, kind: str, ids) -> pd.DataFrame:
        R = self.R[kind].loc[ids]
        out = []
        for l in R.columns:
            s = summarize(R[l].values, with_p=False)
            out.append({"layer": int(l), "mean": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "pos_frac": s["pos_frac"]})
        return pd.DataFrame(out)

    def peak(self, kind: str) -> dict:
        R = self.R[kind]
        md_ = R.loc[self.disc].mean(0)
        lstar = int(md_.idxmax())
        val = R.loc[self.val]
        mv = val.mean(0)
        s_star = summarize(val[lstar].values)
        lv = int(mv.idxmax())
        s_max = summarize(val[lv].values, with_p=False)
        drop = (self.dclean.loc[self.val] - self.dnoised.loc[self.val]).values
        frac, flo, fhi = ratio_ci(val[lstar].values, drop)
        return {"kind": kind, "L_disc": lstar, "disc_mean": float(md_[lstar]), "val": s_star, "L_val": lv, "val_max": s_max,
                "auc_pos": auc_positive(mv.values), "com": centre_of_mass(mv.values, list(R.columns)),
                "frac_of_drop": frac, "frac_lo": flo, "frac_hi": fhi, "n_layers": len(R.columns)}


def routing_overlap(run: str, ids: list[int]) -> list[float]:
    """Per layer: mean fraction of the clean top-k experts kept in the noised top-k at the recorded position."""
    rt = pd.read_parquet(os.path.join(RESULTS, run, "sweep_routing.parquet"))
    rt = rt[rt.case_id.isin(ids)]
    k = int(rt.slot.max()) + 1
    out = []
    for l in sorted(rt.layer.unique()):
        g = rt[rt.layer == l]
        c = g[g.run == "clean"].groupby("case_id").expert.apply(set)
        n = g[g.run == "noised"].groupby("case_id").expert.apply(set)
        out.append(float(np.mean([len(c[i] & n[i]) / k for i in c.index])))
    return out


def contiguous_band(curve: np.ndarray, frac: float = 0.25) -> tuple[int, int]:
    """Layers around the argmax whose mean rescue stays >= frac x the peak (contiguous), as (first, last)."""
    m = int(np.argmax(curve))
    thr = frac * curve[m]
    lo = m
    while lo > 0 and curve[lo - 1] >= thr:
        lo -= 1
    hi = m
    while hi < len(curve) - 1 and curve[hi + 1] >= thr:
        hi += 1
    return (lo, hi)


def noise_attribution(sub: Site) -> dict:
    ct = sub.cases.set_index("case_id").loc[sub.disc + sub.val]
    d_full, d_last, d_rest = ct["drop"].values, ct.drop_lastonly.values, ct.drop_exceptlast.values
    multi = ct.n_subject_tokens.values > 1
    out = {"n": len(ct), "n_multi_token_subject": int(multi.sum()), "drop_full": summarize(d_full, with_p=False),
           "drop_lastonly": summarize(d_last, with_p=False), "drop_exceptlast": summarize(d_rest, with_p=False),
           "interaction": summarize(d_full - d_last - d_rest, with_p=False)}
    for name, num in (("frac_lastonly", d_last), ("frac_exceptlast", d_rest)):
        r, lo, hi = ratio_ci(num, d_full)
        out[name] = {"ratio": r, "lo": lo, "hi": hi}
    r, lo, hi = ratio_ci(d_last[multi], d_full[multi])
    out["frac_lastonly_multi"] = {"ratio": r, "lo": lo, "hi": hi, "n": int(multi.sum())}
    out["lastonly_ge_half_of_full_frac"] = float((d_last >= 0.5 * d_full).mean())
    out["lastonly_passes_strict_drop_frac"] = float((d_last >= 0.5).mean())
    return out


# ---------------------------------------------------------------------------------------------------------------
def figure_model(short: str, cfg: dict, sub: Site, fin: Site, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, kind in zip(axes, KINDS):
        for site, obj, ls, alpha in (("subject", sub, "-", 0.18), ("final", fin, "--", 0.0)):
            c = obj.curve(kind, obj.val)
            col = COL[kind] if site == "subject" else INK2
            if alpha:
                ax.fill_between(c.layer, c.ci_lo, c.ci_hi, color=col, alpha=alpha, linewidth=0)
            ax.plot(c.layer, c["mean"], color=col, linewidth=2 if site == "subject" else 1.4, linestyle=ls,
                    label=f"{SITE_LABEL[site]}", solid_joinstyle="round")
            pk = obj.peak(kind)
            ax.plot([pk["L_val"]], [pk["val_max"]["mean"]], marker="o", markersize=6, color=col, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
            ax.annotate(f"L{pk['L_val']} {pk['val_max']['mean']:+.2f}", (pk["L_val"], pk["val_max"]["mean"]), textcoords="offset points",
                        xytext=(5, 4 if site == "subject" else -11), fontsize=8, color=col)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(KIND_LABEL[kind], fontsize=10, color=INK)
        ax.set_xlabel("layer", color=INK2)
        ax.tick_params(colors=INK2, labelsize=9)
    axes[0].set_ylabel("validation mean rescue (Δ logit)", color=INK2)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(f"{cfg['label']}: patch at the last subject token (solid, 95% band) vs the final token (dashed)", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=160)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def figure_protocols(sa: Site, sb: Site, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, kind in zip(axes, KINDS):
        for obj, lab, col, ls in ((sa, "BOS", COL[kind], "-"), (sb, "no BOS", INK2, "--")):
            c = obj.curve(kind, obj.val)
            ax.fill_between(c.layer, c.ci_lo, c.ci_hi, color=col, alpha=0.12, linewidth=0)
            ax.plot(c.layer, c["mean"], color=col, linewidth=1.8, linestyle=ls, label=lab)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(KIND_LABEL[kind] + " at the last subject token", fontsize=10, color=INK)
        ax.set_xlabel("layer", color=INK2)
        ax.tick_params(colors=INK2, labelsize=9)
    axes[0].set_ylabel("validation mean rescue (Δ logit)", color=INK2)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Mixtral-8x7B-v0.1: subject-site curves under the BOS and no-BOS protocols", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=160)
    fig.savefig(path + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def expert_analysis(short: str, cfg: dict, sub: Site) -> dict | None:
    od = sub.dir
    if not os.path.exists(os.path.join(od, "expert_rows.parquet")):
        return None
    md = A.load_model(cfg["subject"])
    sel_meta = json.load(open(os.path.join(od, "expert_layers.json"))) if os.path.exists(os.path.join(od, "expert_layers.json")) else {}
    n_controls = MODELS[cfg["model"]]["n_controls"]
    disc, val = sub.disc, sub.val
    thr = len(disc) // 2
    out = {"layers": sorted(md.expert_rows.layer.unique().tolist()), "selection_meta": sel_meta, "threshold": thr, "per_layer": {}, "fixed": []}
    er = md.expert_rows
    for l in out["layers"]:
        et = A.expert_table(md, l)
        sel = A.select_expert(et, disc, thr)
        row = {"layer": l, "selection": sel}
        # sanity: how concentrated is routing at p (distinct experts, max activity)
        act = et[et.case_id.isin(disc)].groupby("expert").size()
        row["n_distinct_active_disc"] = int(len(act))
        row["max_activity_disc"] = int(act.max()) if len(act) else 0
        if sel.get("e_star") is not None:
            ev = A.evaluate_expert(md, l, sel["e_star"], val, n_controls)
            row["eval"] = {k: v for k, v in ev.items() if k != "per_case"}
        # best all-case expert regardless of recurrence (joint-search style)
        sub_d = et[et.case_id.isin(disc)]
        allcase = sub_d.groupby("expert").rescue.sum() / len(disc)
        if len(allcase):
            eb = int(allcase.idxmax())
            evb = A.evaluate_expert(md, l, eb, val, n_controls)
            row["best_unrestricted"] = {"expert": eb, "disc_active": int(act[eb]), "disc_allcase_mean": float(allcase[eb]),
                                        "rescue_all": evb["rescue_all"], "spec_all": evb["spec_all"], "val_active": evb["val_active"]}
        row["coalitions"] = A.coalitions(md, l, val)
        # concentration: is the layer effect carried by one expert per case (and is it the same expert across cases)?
        ev_ = er[(er.layer == l) & er.case_id.isin(val)]
        sing = ev_[ev_.kind == "expert"]
        coal = ev_[ev_.kind == "coalition_clean"].set_index("case_id").rescue.loc[val]
        g = sing.groupby("case_id").rescue
        ssum, smax = g.sum().loc[val], g.max().loc[val]
        need = []
        for cid, grp in sing.groupby("case_id"):
            r_ = np.sort(grp.rescue.values)[::-1]
            cs = np.cumsum(r_)
            need.append(int(np.argmax(cs >= 0.5 * coal[cid])) + 1 if coal[cid] > 0 and (cs >= 0.5 * coal[cid]).any() else np.nan)
        top_e = sing.loc[sing.groupby("case_id").rescue.idxmax()].expert.value_counts()
        rmax, lo, hi = ratio_ci(smax.values, coal.values)
        row["concentration"] = {"coalition": summarize(coal.values, with_p=False), "sum_singles": summarize(ssum.values, with_p=False),
                                "corr_sum_vs_coalition": float(np.corrcoef(ssum.values, coal.values)[0, 1]),
                                "max_single": summarize(smax.values, with_p=False), "max_single_over_coalition": (rmax, lo, hi),
                                "mean_single": float(sing.rescue.mean()), "frac_singles_positive": float((sing.rescue > 0).mean()),
                                "median_experts_for_half": float(np.nanmedian(need)), "frac_one_expert_for_half": float(np.nanmean(np.array(need) == 1)),
                                "n_distinct_top_experts": int(len(top_e)), "top_expert_most_common": (int(top_e.index[0]), int(top_e.iloc[0])),
                                "top3_top_experts": [(int(e_), int(c_)) for e_, c_ in top_e.head(3).items()]}
        for kind in ("block", "attn_layer", "resid"):
            sub_k = er[(er.layer == l) & (er.kind == kind)].set_index("case_id")
            if len(sub_k):
                row[kind] = summarize(sub_k.loc[val].rescue.values, with_p=False)
        out["per_layer"][l] = row
    for (l, e) in cfg["fixed"]:
        fx = er[(er.layer == l) & (er.kind == "expert_fixed") & (er.expert == e)]
        if not len(fx):
            continue
        fx = fx.set_index("case_id")
        v = fx.loc[[c for c in val if c in fx.index]]
        d_ = fx.loc[[c for c in disc if c in fx.index]]
        act_v = v[v.clean_active]
        rec = {"layer": l, "expert": e, "val_active": int(v.clean_active.sum()), "disc_active": int(d_.clean_active.sum()),
               "val_noised_active": int(v.noised_active.sum()), "rescue_all": summarize(v.rescue.values),
               "rescue_active": summarize(act_v.rescue.values) if len(act_v) else None}
        if len(act_v) and l in out["per_layer"]:
            ev = A.evaluate_expert(md, l, e, val, n_controls)
            rec["spec_all"] = ev["spec_all"]
            rec["spec_active"] = ev["spec_active"]
        out["fixed"].append(rec)
    return out


# ---------------------------------------------------------------------------------------------------------------
def analyze(short: str) -> dict:
    cfg = CFG[short]
    sub, fin = Site(cfg["subject"]), Site(cfg["final"])
    assert sub.val == fin.val and sub.disc == fin.disc, "case sets differ between the subject and the final-token runs"
    res = {"short": short, "label": cfg["label"], "model": cfg["model"], "n_disc": len(sub.disc), "n_val": len(sub.val),
           "subject_run": cfg["subject"], "final_run": cfg["final"], "meta": json.load(open(os.path.join(sub.dir, "run_meta.json")))}
    ct = sub.cases.set_index("case_id").loc[sub.disc + sub.val]
    res["pos_stats"] = {"T_mean": float(ct.n_tokens.mean()), "p_mean": float(ct.pos.mean()), "suffix_mean": float((ct.n_tokens - ct.pos).mean()),
                        "suffix_min": int((ct.n_tokens - ct.pos).min()), "suffix_max": int((ct.n_tokens - ct.pos).max()),
                        "n_subject_tokens_mean": float(ct.n_subject_tokens.mean()), "single_token_subjects": int((ct.n_subject_tokens == 1).sum())}
    # ---- peaks per kind and site
    peaks, prow = {}, []
    for kind in KINDS:
        for site, obj in (("subject", sub), ("final", fin)):
            pk = obj.peak(kind)
            peaks[(kind, site)] = pk
            prow.append([KIND_LABEL[kind], SITE_LABEL[site], f"L{pk['L_disc']}", f3(pk["disc_mean"]), ci(pk["val"]), f"{pk['val']['pos_frac']:.0%}",
                         f"L{pk['L_val']}", ci(pk["val_max"]), f"{pk['frac_of_drop']:+.2f} [{pk['frac_lo']:+.2f}, {pk['frac_hi']:+.2f}]",
                         f"{pk['auc_pos']:.2f}", f"{pk['com']:.1f}"])
    res["peaks"] = {f"{k}_{s}": v for (k, s), v in peaks.items()}
    res["tables"] = {}
    res["tables"]["peaks"] = md_table(
        ["Patched component", "Site", "L* (disc.)", "Disc. mean at L*", "Val. rescue at L* [95% CI]", "Val. pos. frac.", "Val. argmax",
         "Val. max [95% CI]", "Rescue / drop at L* [CI]", "AUC+ (val.)", "Centre of mass"], prow,
        os.path.join(TAB, f"ext5_subject_peaks_{short}"),
        caption=f"{cfg['label']}: peaks of the rescue curves when the patch is applied at the last subject token versus the final token "
                f"(paper set, {len(sub.disc)} discovery / {len(sub.val)} validation cases)")
    # ---- paired comparison at the same layers: subject vs final at each site's peak
    comp = []
    for kind in KINDS:
        for site_ref in ("subject", "final"):
            l = peaks[(kind, site_ref)]["L_disc"]
            a, b = sub.R[kind].loc[sub.val, l].values, fin.R[kind].loc[fin.val, l].values
            s = summarize(a - b, with_p=True)
            comp.append({"kind": kind, "layer": l, "at": site_ref + "-site L*", "subject": summarize(a, with_p=False),
                         "final": summarize(b, with_p=False), "diff": s})
    res["paired"] = comp
    # ---- curve-level comparison: correlation of the validation mean curves and the layer of the maximum ratio
    cc = {}
    for kind in KINDS:
        a = sub.curve(kind, sub.val)["mean"].values
        b = fin.curve(kind, fin.val)["mean"].values
        cc[kind] = {"corr": float(np.corrcoef(a, b)[0, 1]), "subject_share_of_final_max": float(a.max() / b.max()) if b.max() > 0 else np.nan,
                    "first_layer_subject_ge_half_max": int(np.argmax(a >= 0.5 * a.max())), "first_layer_final_ge_half_max": int(np.argmax(b >= 0.5 * b.max())),
                    "band_subject": contiguous_band(a), "band_final": contiguous_band(b)}
    res["curve_compare"] = cc
    # ---- noise attribution
    res["noise"] = noise_attribution(sub)
    # ---- routing disruption: clean-vs-noised top-k overlap at p (subject run) and at the final token (base run)
    ov_s = routing_overlap(cfg["subject"], sub.disc + sub.val)
    ov_f = routing_overlap(cfg["base"], sub.disc + sub.val) if os.path.exists(os.path.join(RESULTS, cfg["base"], "sweep_routing.parquet")) else None
    res["routing_overlap"] = {"subject": ov_s, "final": ov_f, "subject_mean": float(np.mean(ov_s)), "subject_min": float(np.min(ov_s)),
                              "subject_max": float(np.max(ov_s)), "final_mean": float(np.mean(ov_f)) if ov_f else None,
                              "final_min": float(np.min(ov_f)) if ov_f else None, "final_max": float(np.max(ov_f)) if ov_f else None}
    # ---- experts at the subject site
    res["experts"] = expert_analysis(short, cfg, sub)
    # ---- figure
    figure_model(short, cfg, sub, fin, os.path.join(FIG, f"ext5_subject_curves_{short}"))
    # ---- expert tables
    if res["experts"]:
        ex = res["experts"]
        rows = []
        for l, r in ex["per_layer"].items():
            sel = r["selection"]
            if sel.get("e_star") is not None:
                ev = r["eval"]
                rows.append([f"L{l}", f"E{sel['e_star']:03d}", f"{sel['disc_active']}/{sel['n_disc']}", f3(sel["disc_allcase_mean"]),
                             f"{ev['val_active']}/{ev['n_val']}", ci(ev["rescue_all"]), ci(ev["rescue_active"]), ci(ev["spec_all"]), ci(ev["spec_active"]),
                             ci(r["coalitions"]["coalition_clean"]), ci(r["coalitions"]["layer"]), ci(r.get("block")), ci(r.get("attn_layer"))])
            else:
                rows.append([f"L{l}", f"none (max activity {sel.get('max_activity', 0)}/{len(sub.disc)})", "-", "-", "-", "-", "-", "-", "-",
                             ci(r["coalitions"]["coalition_clean"]), ci(r["coalitions"]["layer"]), ci(r.get("block")), ci(r.get("attn_layer"))])
        res["tables"]["experts"] = md_table(
            ["Layer", "Selected expert (recurrence-first)", "Disc. active", "Disc. all-case mean", "Val. active", "Val. rescue (all) [CI]",
             "Val. rescue (active) [CI]", "Spec (all) [CI]", "Spec (active) [CI]", "Coalition (clean top-k) [CI]", "MoE output [CI]", "Block [CI]", "Attention [CI]"],
            rows, os.path.join(TAB, f"ext5_subject_experts_{short}"),
            caption=f"{cfg['label']}: expert-level tracing at the last subject token (paper set; recurrence threshold {ex['threshold']}/{len(sub.disc)} "
                    f"discovery cases; controls = active-random, {MODELS[cfg['model']]['n_controls']} per case)")
        frows = []
        for r in ex["fixed"]:
            frows.append([f"L{r['layer']}E{r['expert']:03d}", f"{r['disc_active']}/{len(sub.disc)}", f"{r['val_active']}/{len(sub.val)}",
                          ci(r["rescue_all"]), ci(r["rescue_active"]) if r["rescue_active"] else "n/a",
                          ci(r.get("spec_all")) if r.get("spec_all") else "n/a"])
        crow = []
        for l, r in ex["per_layer"].items():
            c = r["concentration"]
            crow.append([f"L{l}", ci(c["coalition"]), ci(c["sum_singles"]), f"{c['corr_sum_vs_coalition']:.2f}", ci(c["max_single"]),
                         (f"{c['max_single_over_coalition'][0]:.2f} [{c['max_single_over_coalition'][1]:.2f}, {c['max_single_over_coalition'][2]:.2f}]"
                          if c["coalition"]["mean"] > 0.1 else "n/a (no layer effect)"),
                         f"{c['mean_single']:+.3f}", f"{c['frac_singles_positive']:.0%}", f"{c['median_experts_for_half']:.0f} ({c['frac_one_expert_for_half']:.0%} of cases: 1)",
                         f"{c['n_distinct_top_experts']} (most common E{c['top_expert_most_common'][0]:03d} in {c['top_expert_most_common'][1]}/{len(sub.val)})"])
        res["tables"]["concentration"] = md_table(
            ["Layer", "Coalition (clean top-k) [CI]", "Σ single-expert rescues [CI]", "r(Σ, coalition)", "Best single expert per case [CI]",
             "Best single / coalition [CI]", "Mean single", "Singles > 0", "Experts needed for 50% of the coalition (median)",
             "Distinct per-case best experts"], crow, os.path.join(TAB, f"ext5_subject_concentration_{short}"),
            caption=f"{cfg['label']}: is the subject-site layer effect carried by one expert, and by the same one across prompts? (validation cases)")
        res["tables"]["fixed"] = md_table(
            ["Final-token expert", "Disc. clean-active at p", "Val. clean-active at p", "Val. rescue at p (all cases) [CI]",
             "Val. rescue at p (active cases) [CI]", "Spec at p (all) [CI]"], frows, os.path.join(TAB, f"ext5_subject_fixed_{short}"),
            caption=f"{cfg['label']}: the final-token experts patched at the last subject token")
    log(f"{short}: " + "; ".join(f"{k} subj L{peaks[(k, 'subject')]['L_disc']} {peaks[(k, 'subject')]['val']['mean']:+.3f} / final L{peaks[(k, 'final')]['L_disc']} "
                                 f"{peaks[(k, 'final')]['val']['mean']:+.3f}" for k in KINDS))
    return res


def noise_table(results: list[dict]) -> str:
    rows = []
    for r in results:
        nz = r["noise"]
        rows.append([r["label"], nz["n"], nz["n_multi_token_subject"], ci(nz["drop_full"]), ci(nz["drop_lastonly"]), ci(nz["drop_exceptlast"]),
                     f"{nz['frac_lastonly']['ratio']:.2f} [{nz['frac_lastonly']['lo']:.2f}, {nz['frac_lastonly']['hi']:.2f}]",
                     f"{nz['frac_lastonly_multi']['ratio']:.2f} [{nz['frac_lastonly_multi']['lo']:.2f}, {nz['frac_lastonly_multi']['hi']:.2f}]",
                     f"{nz['frac_exceptlast']['ratio']:.2f} [{nz['frac_exceptlast']['lo']:.2f}, {nz['frac_exceptlast']['hi']:.2f}]",
                     ci(nz["interaction"]), f"{nz['lastonly_passes_strict_drop_frac']:.0%}"])
    return md_table(["Model / protocol", "n", "multi-token subjects", "Drop, whole span [CI]", "Drop, last subject token only [CI]",
                     "Drop, span minus last token [CI]", "Last-only / whole [CI]", "Last-only / whole, multi-token subjects [CI]",
                     "Rest / whole [CI]", "Whole − last − rest [CI]", "Cases with last-only drop ≥ 0.5"], rows,
                    os.path.join(TAB, "ext5_subject_noise_attribution"),
                    caption="Noise attribution: drop Δ_clean − Δ_noised when the same Gaussian draw is applied to the whole subject span, to the last "
                            "subject token only, or to the rest of the span (paper sets, discovery + validation)")


def comparison_table(results: list[dict]) -> str:
    """MoE-output peak at the last subject token vs at the final token for every run, with the noise attribution."""
    rows = []
    for r in results:
        a, b = r["peaks"]["layer_subject"], r["peaks"]["layer_final"]
        nz = r["noise"]
        rows.append([r["label"], f"L{a['L_disc']}", ci(a["val"]), f"{a['frac_of_drop']:.2f} [{a['frac_lo']:.2f}, {a['frac_hi']:.2f}]",
                     f"L{b['L_disc']}", ci(b["val"]), f"{b['frac_of_drop']:.2f} [{b['frac_lo']:.2f}, {b['frac_hi']:.2f}]",
                     f"{a['val']['mean'] / b['val']['mean']:.1f}x",
                     "L{}–L{} vs L{}–L{}".format(*r["curve_compare"]["layer"]["band_subject"], *r["curve_compare"]["layer"]["band_final"]),
                     ci(nz["drop_full"]), f"{nz['frac_lastonly']['ratio']:.2f} [{nz['frac_lastonly']['lo']:.2f}, {nz['frac_lastonly']['hi']:.2f}]",
                     f"{nz['frac_exceptlast']['ratio']:.2f}"])
    return md_table(["Model / protocol", "Subject site L*", "Val. rescue [CI]", "Share of drop [CI]", "Final-token L*", "Val. rescue [CI]",
                     "Share of drop [CI]", "Subject / final", "Band ≥ 25% of the peak, contiguous (subject vs final)", "Whole-span drop [CI]",
                     "Drop from the last subject token alone / whole", "Rest of span / whole"], rows,
                    os.path.join(TAB, "ext5_subject_comparison"),
                    caption="MoE-output patch: peak at the last subject token versus at the final token (paper sets, validation means; share of drop = "
                            "mean rescue / mean drop with paired bootstrap CI), with the noise attribution to the last subject token")


def protocol_table(ra: dict, rb: dict, sa: Site, sb: Site) -> tuple[str, dict]:
    rows, out = [], {}
    for kind in KINDS:
        a, b = sa.curve(kind, sa.val)["mean"].values, sb.curve(kind, sb.val)["mean"].values
        corr = float(np.corrcoef(a, b)[0, 1])
        pa, pb = ra["peaks"][f"{kind}_subject"], rb["peaks"][f"{kind}_subject"]
        la = pa["L_disc"]
        d = summarize(sa.R[kind].loc[sa.val, la].values - sb.R[kind].loc[sb.val, la].values)
        out[kind] = {"corr": corr, "bos_peak": (la, pa["val"]["mean"]), "nobos_peak": (pb["L_disc"], pb["val"]["mean"]), "paired_diff_at_bos_Lstar": d}
        rows.append([KIND_LABEL[kind], f"{corr:.3f}", f"L{la} {ci(pa['val'])}", f"L{pb['L_disc']} {ci(pb['val'])}", f"L{la}: {ci(d)} (p={d['p']:.3f})"])
    return md_table(["Patched component (at the last subject token)", "Corr. of validation curves", "BOS: L* and val. rescue [CI]",
                     "no BOS: L* and val. rescue [CI]", "Paired BOS − no BOS at the BOS L* [CI]"], rows,
                    os.path.join(TAB, "ext5_subject_protocol_mixtral"),
                    caption="Mixtral-8x7B-v0.1: subject-site curves under the BOS (tokenizer default) and no-BOS (paper) protocols"), out


# ---------------------------------------------------------------------------------------------------------------
def write_section(results: list[dict], noise_md: str, proto: tuple | None, verify: dict | None, comp_md: str = ""):
    by = {r["short"]: r for r in results}
    S = []
    S.append("## Extension 5 / F4: causal tracing at the last subject token (the \"early site\")\n")
    summ = os.path.join(SEC, "ext5_f4_subject_summary.md")
    if os.path.exists(summ):
        S.append(open(summ).read().rstrip() + "\n")
    S.append("**Question.** The paper (and every run of this reproduction so far) patches the *final* position of the prompt. Classic "
             "causal tracing (Meng et al., 2022) locates a second, earlier site: restoring the corrupted subject's *last token* in early-to-"
             "middle MLP layers recovers the fact. Does an MoE model show that early site, which layers and which experts carry it, and how does "
             "it compare with the final-token curves the paper reports?\n")
    S.append("**Method.** New executor `moetrace/ext5_subject.py` (`SubjectEngine.run_subject`; engine.py untouched, its helpers reused). A patch "
             "at position p and layer l changes positions p..T−1 for all layers ≥ l, so a *suffix wavefront row* starts after layer l with the noised "
             "run's residuals at positions p..T−1, the intervention applied at p only, and runs layers l+1..L−1 for those T−p tokens, attending to the "
             "parent noised run's K/V for positions < p and to its own K/V for ≥ p. Δ = logit(true) − logit(foil) is read at the final position as "
             "usual; rescue = Δ_patched − Δ_noised. p = last subject token (`Case.subject_pos[-1]`; CounterFact suffixes are 2–8 tokens, mean 4.2). "
             "Kinds mirror ext2 at the final position: `layer` (MoE output at p ← clean), `attn_layer` (attention-sublayer output at p ← clean, the "
             "MoE of that layer recomputes), `resid` (whole residual after layer l at p ← clean). Paper case sets (128/128), σ = 3, all layers, "
             "two layer-chunked passes per model; expert pass at the subject-site MoE peaks plus the final-token hypothesis layers "
             "(`ext5_subject_expert.py`: every clean-active expert at p, coalitions, active-random controls as in the paper protocol).\n")
    S.append("**Design caveat (stated up front).** The noise sits on the subject tokens, so `resid` at p restores, from layer 0 on, the corrupted "
             "token's own residual: its layer profile (where the curve rises and falls) is the information, not its absolute level, and it is a "
             "*cumulative* quantity like the final-token `resid`. The MoE-output and attention-output patches do not restore the corrupted embedding "
             "and are the primary curves. As a reference for how much of the corruption the last subject token carries, every sweep also ran "
             "prefill rows with the same Gaussian draw restricted to the last subject token only, and to the rest of the span only.\n")
    if verify:
        e = verify["final_token_consistency_in_process"]
        S.append("**Verification (OLMoE-1B-7B-0125, 20 cases × 16 layers, `scripts/ext5_subject_verify.py`, `results/verify_ext5_subject_olmoe.json`).** "
                 "Against transformers forward hooks that replace the self_attn output / MoE output / decoder-layer output at position p: per-case "
                 f"|Δ_engine − Δ_HF| mean {verify['attn_layer_vs_hf_meanabsdiff']:.3f} / {verify['layer_vs_hf_meanabsdiff']:.3f} / "
                 f"{verify['resid_vs_hf_meanabsdiff']:.3f} (max {verify['attn_layer_vs_hf_maxdiff']:.2f} / {verify['layer_vs_hf_maxdiff']:.2f} / "
                 f"{verify['resid_vs_hf_maxdiff']:.2f}) for attn_layer / layer / resid, the same size as the final-token floor of ext2 (0.19 / 0.15 / 0.13); "
                 f"20-case mean-curve max deviation {verify['attn_layer_mean_curve_maxdiff_vs_hf']:.3f} / {verify['layer_mean_curve_maxdiff_vs_hf']:.3f} / "
                 f"{verify['resid_mean_curve_maxdiff_vs_hf']:.3f}; per-case rescue correlation {verify['attn_layer_rescue_corr_vs_hf']:.3f} / "
                 f"{verify['layer_rescue_corr_vs_hf']:.3f} / {verify['resid_rescue_corr_vs_hf']:.3f} (the attention rescue at p is tiny in OLMoE, "
                 f"|mean| < 0.2, so its per-case correlation is floor-limited). Null invariant (zero vector at p = noised run): max {verify['zero_on_noised_maxdiff']:.3f}, "
                 f"mean {verify['zero_on_noised_meanabsdiff']:.3f}; identity (clean donor on the clean run): max {verify['identity_resid_maxdiff']:.3f}. "
                 f"Consistency with the final-token machinery: the same prefill batch through `run_subject` with p = T−1 and through `Engine.run` agrees "
                 f"to max |ΔΔ| {e['delta_maxdiff']:.3f}, mean {e['delta_meanabsdiff']:.4f} ({e['delta_frac_equal']:.0%} of 960 rows bit-identical); "
                 f"against the *stored* ext2 rows of the same cases (different batch composition) mean |ΔΔ| "
                 f"{verify['final_token_consistency_vs_olmoe_attnsweep']['layer']['delta_meanabsdiff']:.3f}, mean-curve max deviation "
                 f"{max(v['mean_curve_maxdiff'] for v in verify['final_token_consistency_vs_olmoe_attnsweep'].values()):.3f}. Patched-vector norms agree with HF "
                 f"to {max(v['max_reldiff'] for v in verify['vnorm_vs_hf'].values()):.1%}.\n")
    for r in results:
        pk = r["peaks"]
        S.append(f"### {r['label']} (`results/{r['subject_run']}` vs `results/{r['final_run']}`)\n")
        ps = r["pos_stats"]
        bul = []
        for kind in KINDS:
            a, b = pk[f"{kind}_subject"], pk[f"{kind}_final"]
            bul.append(f"{KIND_LABEL[kind]}: subject site L{a['L_disc']} {ci(a['val'])} (positive in {a['val']['pos_frac']:.0%}, {a['frac_of_drop']:+.2f} of the drop) "
                       f"vs final token L{b['L_disc']} {ci(b['val'])} ({b['frac_of_drop']:+.2f} of the drop); validation argmax L{a['L_val']} vs L{b['L_val']}; "
                       f"centre of mass {a['com']:.1f} vs {b['com']:.1f}")
        S.append(f"- Prompts: T = {ps['T_mean']:.1f} tokens, last subject token at p = {ps['p_mean']:.1f}, suffix {ps['suffix_mean']:.1f} tokens "
                 f"({ps['suffix_min']}–{ps['suffix_max']}); {ps['single_token_subjects']} single-token subjects.")
        S.append("- Peaks (discovery argmax, validation value): " + "; ".join(bul) + ".")
        cc = r["curve_compare"]
        S.append("- Curve shape: correlation of the subject-site and final-token validation curves " +
                 ", ".join(f"{KIND_LABEL[k]} {cc[k]['corr']:.2f}" for k in KINDS) + "; the subject-site MoE curve first reaches half its maximum at "
                 f"L{cc['layer']['first_layer_subject_ge_half_max']} (final token: L{cc['layer']['first_layer_final_ge_half_max']}).")
        for c in r["paired"]:
            if c["at"].startswith("subject") and c["kind"] in ("layer", "attn_layer"):
                S.append(f"- At the subject-site {KIND_LABEL[c['kind']]} peak L{c['layer']}: subject {ci(c['subject'])} vs final {ci(c['final'])}, paired "
                         f"difference {ci(c['diff'])} (sign-flip p = {c['diff']['p']:.3f}).")
        ov = r["routing_overlap"]
        S.append(f"- Routing at the patched token: the noise scrambles the last subject token's own routing at every layer (clean top-k kept in the "
                 f"noised top-k: mean {ov['subject_mean']:.2f}, range {ov['subject_min']:.2f}–{ov['subject_max']:.2f})" +
                 (f", whereas at the final token the overlap is {ov['final_mean']:.2f} ({ov['final_min']:.2f}–{ov['final_max']:.2f})." if ov["final_mean"] is not None else "."))
        nz = r["noise"]
        S.append(f"- Noise attribution: whole-span drop {ci(nz['drop_full'])}; the same draw on the last subject token only gives {ci(nz['drop_lastonly'])} "
                 f"({nz['frac_lastonly']['ratio']:.2f} [{nz['frac_lastonly']['lo']:.2f}, {nz['frac_lastonly']['hi']:.2f}] of the whole-span drop; "
                 f"{nz['frac_lastonly_multi']['ratio']:.2f} on the {nz['n_multi_token_subject']} multi-token subjects), the rest of the span "
                 f"{ci(nz['drop_exceptlast'])} ({nz['frac_exceptlast']['ratio']:.2f}); interaction whole − last − rest {ci(nz['interaction'])}.")
        if r["experts"]:
            ex = r["experts"]
            for l, row in ex["per_layer"].items():
                sel = row["selection"]
                if sel.get("e_star") is not None:
                    ev = row["eval"]
                    S.append(f"- Experts at p, L{l}: {sel['n_candidates']} recurrent candidates (≥ {ex['threshold']}/{r['n_disc']} discovery cases); "
                             f"selected E{sel['e_star']:03d} (active {sel['disc_active']}/{r['n_disc']} disc., {ev['val_active']}/{r['n_val']} val.), validation rescue "
                             f"{ci(ev['rescue_all'])}, Spec {ci(ev['spec_all'])} (active cases {ci(ev['spec_active'])}); clean top-k coalition {ci(row['coalitions']['coalition_clean'])} "
                             f"vs MoE-output patch {ci(row['coalitions']['layer'])}" +
                             (f"; unrestricted argmax E{row['best_unrestricted']['expert']:03d} (active {row['best_unrestricted']['disc_active']}) rescue "
                              f"{ci(row['best_unrestricted']['rescue_all'])}, Spec {ci(row['best_unrestricted']['spec_all'])}" if row.get("best_unrestricted") and
                              row["best_unrestricted"]["expert"] != sel["e_star"] else "") + ".")
                else:
                    S.append(f"- Experts at p, L{l}: no expert meets the recurrence threshold (max activity {sel.get('max_activity', 0)}/{r['n_disc']}; "
                             f"{row['n_distinct_active_disc']} distinct clean-active experts); clean top-k coalition {ci(row['coalitions']['coalition_clean'])} vs "
                             f"MoE-output patch {ci(row['coalitions']['layer'])}" +
                             (f"; unrestricted argmax E{row['best_unrestricted']['expert']:03d} (active {row['best_unrestricted']['disc_active']}) rescue "
                              f"{ci(row['best_unrestricted']['rescue_all'])}, Spec {ci(row['best_unrestricted']['spec_all'])}" if row.get("best_unrestricted") else "") + ".")
            for l, row in ex["per_layer"].items():
                c = row["concentration"]
                if c["coalition"]["mean"] > 0.2:
                    S.append(f"- Concentration at p, L{l}: the best single expert of each case recovers {ci(c['max_single'])} = "
                             f"{c['max_single_over_coalition'][0]:.0%} [{c['max_single_over_coalition'][1]:.0%}, {c['max_single_over_coalition'][2]:.0%}] of the "
                             f"coalition {ci(c['coalition'])} (one expert reaches half the coalition in {c['frac_one_expert_for_half']:.0%} of cases), but that expert "
                             f"differs across prompts: {c['n_distinct_top_experts']} distinct per-case winners, the most common (E{c['top_expert_most_common'][0]:03d}) "
                             f"in only {c['top_expert_most_common'][1]}/{len(sub.val) if False else r['n_val']} cases; singles are sub-additive "
                             f"(Σ singles {ci(c['sum_singles'])}, {c['frac_singles_positive']:.0%} of single patches positive).")
            if ex["fixed"]:
                S.append("- Final-token experts patched at p: " + "; ".join(
                    f"L{f['layer']}E{f['expert']:03d} clean-active at p in {f['val_active']}/{r['n_val']} validation cases, rescue {ci(f['rescue_all'])}" +
                    (f" (Spec {ci(f['spec_all'])})" if f.get("spec_all") else "") for f in ex["fixed"]) + ".")
        S.append(f"\n![ext5 subject-site curves {r['short']}](../figures/ext5_subject_curves_{r['short']}.png)\n")
        S.append(f"Figure E5-F4-{r['short']}: validation mean rescue by layer when the MoE output (left), the attention output (middle) or the whole residual "
                 f"(right) is replaced by the clean run's at the last subject token (solid, 95% bootstrap band) versus at the final token (dashed; ext2 runs). "
                 f"Tables: `results/tables/ext5_subject_peaks_{r['short']}.md`" +
                 (f", `ext5_subject_experts_{r['short']}.md`, `ext5_subject_concentration_{r['short']}.md`, `ext5_subject_fixed_{r['short']}.md`" if r["experts"] else "") + ".\n")
        S.append(r["tables"]["peaks"])
        if r["experts"]:
            S.append(r["tables"]["experts"])
            S.append(r["tables"]["concentration"])
            S.append(r["tables"]["fixed"])
    S.append("### Noise attribution to the last subject token\n")
    S.append(noise_md)
    if proto:
        S.append("### Mixtral: BOS versus no BOS at the subject site\n")
        S.append(proto[0])
        S.append("\n![ext5 subject-site protocols](../figures/ext5_subject_mixtral_protocols.png)\n")
    S.append("### Subject site versus final token: the MoE-output peak in all three runs\n")
    S.append(comp_md)
    interp = os.path.join(SEC, "ext5_f4_subject_interpretation.md")
    if os.path.exists(interp):
        S.append(open(interp).read())
    with open(os.path.join(SEC, "ext5_f4_subject.md"), "w") as f:
        f.write("\n".join(S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="qwen3_bos,mixtral_nobos,mixtral_bos")
    args = ap.parse_args()
    shorts = [s for s in args.runs.split(",") if os.path.exists(os.path.join(RESULTS, CFG[s]["subject"], "sweep_summary.json"))]
    results = [analyze(s) for s in shorts]
    noise_md = noise_table(results)
    proto = None
    if "mixtral_bos" in shorts and "mixtral_nobos" in shorts:
        by = {r["short"]: r for r in results}
        sa, sb = Site(CFG["mixtral_bos"]["subject"]), Site(CFG["mixtral_nobos"]["subject"])
        proto = protocol_table(by["mixtral_bos"], by["mixtral_nobos"], sa, sb)
        figure_protocols(sa, sb, os.path.join(FIG, "ext5_subject_mixtral_protocols"))
    vpath = os.path.join(RESULTS, "verify_ext5_subject_olmoe.json")
    verify = json.load(open(vpath)) if os.path.exists(vpath) else None
    comp_md = comparison_table(results)
    write_section(results, noise_md, proto, verify, comp_md)
    summ = {"runs": {r["short"]: jsonable({k: v for k, v in r.items() if k not in ("tables",)}) for r in results},
            "protocol_mixtral": jsonable(proto[1]) if proto else None, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(RESULTS, "ext5_subject_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    log("wrote section, tables, figures, summary")


if __name__ == "__main__":
    main()
