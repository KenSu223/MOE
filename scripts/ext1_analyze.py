"""Direction 1 driver: joint layer x expert search vs the two-stage selection, for every all-layer expert run.

Usage: python scripts/ext1_analyze.py [--runs qwen3_bos_alllayers,mixtral_bos_alllayers,mixtral_nobos_alllayers] [--allow-incomplete]
Writes results/tables/ext1_*.{md,csv}, results/figures/ext1_curves_<run>.{png,pdf}, results/ext1_summary.json and
results/sections/ext1_joint_search.md.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import analysis as A
from moetrace import ext1_analysis as X
from moetrace.models import MODELS, RESULTS
from moetrace.report import md_table

TAB, FIG, SEC = (os.path.join(RESULTS, d) for d in ("tables", "figures", "sections"))
for d in (TAB, FIG, SEC):
    os.makedirs(d, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def ci(m, lo, hi):
    return "n/a" if m is None or (isinstance(m, float) and np.isnan(m)) else f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def pct(a, b):
    return f"{a}/{b}"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if not isinstance(v, pd.DataFrame)}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ---------------------------------------------------------------------------------------------------------------
def analyze_run(run: str, allow_incomplete: bool) -> dict:
    cfg = X.RUNS[run]
    m = MODELS[cfg["model"]]
    nc = m["n_controls"]
    md = A.load_model(run)
    if md.expert_rows is None:
        log(f"{run}: no expert_rows.parquet yet, skipped")
        return None
    cache = X.ExpertCache(md)
    n_layers = int(md.routing.layer.nunique())
    complete = len(cache.layers) == n_layers
    if not complete and not allow_incomplete:
        log(f"{run}: only {len(cache.layers)}/{n_layers} layers present, skipped (use --allow-incomplete)")
        return None
    short = cfg["short"]
    out = dict(run=run, model=cfg["model"], label=cfg["label"], short=short, n_controls=nc, two_stage=list(cfg["two_stage"]),
               two_stage_pair=X.pair(*cfg["two_stage"]), neighbours=list(cfg["neighbours"]), n_layers_present=len(cache.layers),
               n_layers=n_layers, complete=complete, n_expert_rows=int(len(md.expert_rows)), sets={})
    top_rows, stab_rows, nb_rows, conc_rows, curve_rows = [], [], [], [], []
    for s in cfg["sets"]:
        log(f"{run} / {s}: per-layer curve")
        curve = X.per_layer_curve(md, cache, s, nc)
        curve.to_csv(os.path.join(TAB, f"ext1_layer_curve_{short}_{s}.csv"), index=False)
        log(f"{run} / {s}: joint search")
        js = X.joint_search(md, cache, s, nc, cfg["two_stage"])
        log(f"{run} / {s}: stability grid")
        st = X.joint_stability(md, cache, s, nc, cfg["two_stage"])
        log(f"{run} / {s}: neighbour layers {cfg['neighbours']}")
        nb = X.neighbour_check(md, cache, s, nc, cfg["neighbours"], cfg["two_stage"])
        conc = X.concentration_table(curve)
        cons = X.consistency_with_base(md, cfg["base_run"], s, cfg["two_stage"], nc)
        ts = js["two_stage"]
        w = js["winner"]
        other = (w["layer"], w["expert"]) if not js["same_as_two_stage"] else ((js["challenger"]["layer"], js["challenger"]["expert"]) if js["challenger"] else None)
        overlap = X.pair_overlap(md, cache, tuple(cfg["two_stage"]), other, md.ids(s, "validation"), nc) if other else None
        out["sets"][s] = dict(n_disc=js["n_disc"], n_val=js["n_val"], threshold=js["threshold"], n_candidates=js["n_candidates"],
                              n_layers_with_candidates=js["n_layers_with_candidates"], winner=w, two_stage=ts, challenger=js["challenger"],
                              same_as_two_stage=js["same_as_two_stage"], winner_in_two_stage_layer=js["winner_in_two_stage_layer"], vs_reference=js["vs_reference"],
                              shrinkage=js["shrinkage"], top10=js["top"].to_dict("records"),
                              stability={k: v for k, v in st.items() if k != "grid"},
                              neighbours={k: v for k, v in nb.items() if k != "table"},
                              consistency=cons, overlap=overlap, concentration=conc[["layer", "e_star", "layer_samepass_val_mean", "val_rescue", "val_spec", "concentration",
                                                                    "concentration_lo", "concentration_hi"]].to_dict("records"),
                              curve=curve.to_dict("records"),
                              layer_curve_max=dict(layer=int(curve.loc[curve.layer_val_mean.idxmax(), "layer"]), val=float(curve.layer_val_mean.max())),
                              best_expert_curve_max=dict(layer=int(curve.loc[curve.val_rescue.idxmax(), "layer"]), e_star=int(curve.loc[curve.val_rescue.idxmax(), "e_star"]),
                                                         val=float(curve.val_rescue.max())) if curve.val_rescue.notna().any() else None,
                              best_spec_curve_max=dict(layer=int(curve.loc[curve.val_spec.idxmax(), "layer"]), e_star=int(curve.loc[curve.val_spec.idxmax(), "e_star"]),
                                                       val=float(curve.val_spec.max())) if curve.val_spec.notna().any() else None)
        for _, r in js["top"].iterrows():
            top_rows.append([s, int(r["rank"]), r["pair"], pct(r.disc_active, js["n_disc"]), f3(r.disc_allcase_mean), pct(r.val_active, js["n_val"]),
                             ci(r.val_rescue, r.val_rescue_lo, r.val_rescue_hi), ci(r.val_spec, r.val_spec_lo, r.val_spec_hi), "yes" if r.is_two_stage else ""])
        top_rows.append([s, ts["rank"] if ts["rank"] is not None else "not recurrent", ts["pair"] + " (two-stage)", pct(ts["disc_active"], js["n_disc"]), f3(ts["disc_allcase_mean"]),
                         pct(ts["val_active"], js["n_val"]), ci(ts["val_rescue"], ts["val_rescue_lo"], ts["val_rescue_hi"]), ci(ts["val_spec"], ts["val_spec_lo"], ts["val_spec_hi"]), "reference"])
        for _, r in st["grid"].iterrows():
            stab_rows.append([s, r.seed, r.threshold, r.threshold_eff, r.get("joint_pair", "none"), f3(r.get("joint_disc_mean", np.nan)), f3(r.get("joint_val_rescue", np.nan)),
                              f3(r.get("joint_val_spec", np.nan)), r.n_candidates, f"L{r.split_two_stage_layer}" + (f"E{int(r.split_two_stage_expert):03d}" if r.split_two_stage_expert is not None and not pd.isna(r.split_two_stage_expert) else " (none)"),
                              "yes" if r.get("equals_reference", False) else "", "yes" if r.get("equals_split_two_stage", False) else ""])
        nbt = nb["table"]
        for _, r in nbt.head(10).iterrows():
            nb_rows.append([s, r.pair, pct(r.disc_active, js["n_disc"]), f3(r.disc_allcase_mean), "yes" if r.recurrent else "", pct(r.val_active, js["n_val"]), f3(r.val_rescue), f3(r.val_spec),
                            "yes" if r.beats_rescue else "", "yes" if r.beats_spec else ""])
        nb_rows.append([s, nb["reference"]["pair"] + " (two-stage)", pct(ts["disc_active"], js["n_disc"]), f3(ts["disc_allcase_mean"]), "yes" if ts["recurrent"] else "",
                        pct(ts["val_active"], js["n_val"]), f3(nb["reference"]["val_rescue"]), f3(nb["reference"]["val_spec"]), "ref", "ref"])
        for _, r in conc.iterrows():
            conc_rows.append([s, f"L{int(r.layer)}", f"E{int(r.e_star):03d}", ci(r.layer_samepass_val_mean, r.layer_samepass_val_lo, r.layer_samepass_val_hi),
                              ci(r.val_rescue, r.val_rescue_lo, r.val_rescue_hi), ci(r.val_spec, r.val_spec_lo, r.val_spec_hi), ci(r.concentration, r.concentration_lo, r.concentration_hi),
                              pct(int(r.val_active), js["n_val"])])
        for _, r in curve.iterrows():
            curve_rows.append([s, int(r.layer), f3(r.layer_val_mean), int(r.n_candidates), (f"E{int(r.e_star):03d}" if not pd.isna(r.e_star) else "none"),
                               (pct(int(r.disc_active), js["n_disc"]) if not pd.isna(r.disc_active) else ""), f3(r.disc_allcase_mean),
                               ci(r.val_rescue, r.val_rescue_lo, r.val_rescue_hi) if not pd.isna(r.val_rescue) else "", ci(r.val_spec, r.val_spec_lo, r.val_spec_hi) if not pd.isna(r.val_spec) else "",
                               f3(r.concentration) if not pd.isna(r.concentration) else ""])
        nbt.to_csv(os.path.join(TAB, f"ext1_neighbours_all_{short}_{s}.csv"), index=False)
        js["all_evaluated"].to_csv(os.path.join(TAB, f"ext1_all_candidates_{short}_{s}.csv"), index=False)
    md_table(["Set", "Rank (disc.)", "Pair", "Disc. active", "Disc. rescue", "Val. active", "Val. rescue [95% CI]", "Spec [95% CI]", "Two-stage winner"],
             top_rows, os.path.join(TAB, f"ext1_top10_{short}"), caption=f"{cfg['label']}: top-10 (layer, expert) pairs by discovery all-case mean rescue among recurrent candidates")
    md_table(["Set", "Seed", "Threshold", "Threshold (eff.)", "Joint winner", "Disc. rescue", "Val. rescue", "Val. Spec", "# candidates", "Split two-stage", "= reference", "= split two-stage"],
             stab_rows, os.path.join(TAB, f"ext1_stability_{short}"), caption=f"{cfg['label']}: joint-search stability grid (split seeds x recurrence thresholds)")
    md_table(["Set", "Pair", "Disc. active", "Disc. rescue", "Recurrent", "Val. active", "Val. rescue", "Val. Spec", "> ref rescue", "> ref Spec"],
             nb_rows, os.path.join(TAB, f"ext1_neighbours_{short}"), caption=f"{cfg['label']}: experts of layers {list(cfg['neighbours'])} ranked by validation rescue (top 10 per set)")
    md_table(["Set", "Layer", "Best expert", "Layer rescue (same pass) [CI]", "Expert rescue [CI]", "Spec [CI]", "Expert/layer [CI]", "Val. active"],
             conc_rows, os.path.join(TAB, f"ext1_concentration_{short}"), caption=f"{cfg['label']}: concentration Rescue(best expert)/Rescue(layer) on validation, layers with a clearly positive block rescue")
    md_table(["Set", "Layer", "Layer rescue (val)", "# candidates", "Best expert", "Disc. active", "Disc. rescue", "Val. rescue [CI]", "Spec [CI]", "Expert/layer"],
             curve_rows, os.path.join(TAB, f"ext1_best_expert_by_layer_{short}"), caption=f"{cfg['label']}: recurrence-first best expert at every layer")
    figure(out, cfg, os.path.join(FIG, f"ext1_curves_{short}"))
    return out


# ---------------------------------------------------------------------------------------------------------------
def figure(out: dict, cfg: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"qwen3": "#2a78d6", "mixtral": "#eb6834"}[cfg["model"]]
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    sets = list(out["sets"].keys())
    fig, axes = plt.subplots(1, len(sets), figsize=(max(4.6 * len(sets), 9.0), 3.6), squeeze=False, sharey=True)
    for ax, s in zip(axes[0], sets):
        d = out["sets"][s]
        cur = pd.DataFrame(d["curve"])
        ax.fill_between(cur.layer, cur.layer_val_lo, cur.layer_val_hi, color=INK2, alpha=0.12, lw=0)
        ax.plot(cur.layer, cur.layer_val_mean, color=INK2, lw=1.4, label="MoE-block rescue (sweep)")
        ax.plot(cur.layer, cur.val_rescue, color=C, lw=1.6, marker="o", ms=2.8, label="best-expert rescue (recurrence-first)")
        ax.plot(cur.layer, cur.val_spec, color=C, lw=1.2, ls="--", marker="s", ms=2.2, alpha=0.8, label="best-expert Spec (active-random)")
        L, E = cfg["two_stage"]
        ts = d["two_stage"]
        ax.plot([L], [ts["val_rescue"]], marker="*", ms=12, color=INK, mec=INK, mfc="none", ls="none", label=f"two-stage winner {X.pair(L, E)}")
        w = d["winner"]
        if w is not None and (w["layer"], w["expert"]) != (L, E):
            ax.plot([w["layer"]], [w["val_rescue"]], marker="D", ms=7, color=INK, mfc="none", ls="none", label=f"joint winner {w['pair']}")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xlabel("MoE layer")
        ax.set_title(f"{s} set (n_val={d['n_val']}, threshold {d['threshold']})", fontsize=9, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0][0].set_ylabel("Validation rescue (Delta patched - Delta noised)")
    handles = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    axes[0][0].legend(list(handles.values()), list(handles.keys()), frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(f"{cfg['label']}: MoE-block rescue vs best single expert per layer (validation means; 95% band for the block)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=170)
    fig.savefig(path + ".pdf")
    plt.close(fig)


INTERPRETATION = {
    "qwen3_bos_alllayers": (
        "**Reading.** The two-stage procedure does not miss a *better* single expert: L44E069 is the joint discovery argmax on the paper set and has "
        "the highest validation point estimate in all three case sets. It does miss a *second locus of the same kind*. L42E115 is clean-active in "
        "126/128 discovery and 123/128 validation cases (more recurrent than E069's 114/116), its validation rescue and Spec are within the CIs of "
        "E069's (and its own CIs are about half as wide), it is the joint discovery argmax on the strict and relaxed sets and in 5/25 grid settings, "
        "and L42 concentrates 72% of its block rescue in this one expert against 53% at L44. The two experts rescue partly different cases (per-case "
        "correlation r = 0.27; 26% of validation cases are rescued only by E115, 11% only by E069) and the per-case maximum of the two (+0.75) approaches "
        "the L44 block rescue (+0.94). The paper's Qwen3 picture 'one specific expert' should therefore read 'one specific expert per layer in the "
        "L42/L44 band', with L44E069 the stronger of two; the two-stage search sees only the layer with the higher block rescue and never examines "
        "L42, whose block rescue (+0.62) is the second highest. Stated plainly: **L42E115 (recurrent in 126/128 discovery cases, validation rescue "
        "+0.447 [+0.363, +0.537], Spec +0.423 [+0.339, +0.510]) is a second, near-equivalent factual-recall expert in the second-strongest layer; it "
        "wins the discovery argmax on the strict and relaxed sets and in 30 of the 75 grid cells (5/25 paper, 10/25 strict, 15/25 relaxed), while "
        "L44E069 keeps the higher validation point estimate in every set.** No expert in L43 or L45 comes close (best +0.15 and -0.03)."),
    "mixtral_bos_alllayers": (
        "**Reading.** With BOS, the joint search and the two-stage selection agree on L19E002 on all three sets and in 17/25 grid settings; every one "
        "of the 8 disagreements is a threshold-80/96 setting in which L19 has no recurrent expert at all (E002 is clean-active in only 76/128 discovery "
        "cases), so the two-stage procedure returns nothing while the joint search falls back to L21E001 or L18E001 (validation +0.27 / +0.24, both "
        "clearly weaker than E002's +0.36). L18E001 is the most concentrated layer (77% of a +0.32 block rescue) and is the only other pair whose Spec "
        "CI excludes zero; the same expert index E001 is the best expert at L17, L18, L21 and L22, which is worth checking against the BOS-mechanism "
        "results (Direction 3) since expert indices are independent parameters across layers. The picture of Mixtral as a coalition model is unchanged: "
        "no single expert in any layer reaches half of the L19 block rescue except E002 itself (63%)."),
    "mixtral_nobos_alllayers": (
        "**Reading.** Under the paper's own protocol the two-stage selection is the one that misses. L19 is the discovery block-rescue peak, but its "
        "signal is not concentrated in any expert: E006 carries 15% of the L19 block rescue and is negatively specific, exactly as the paper reports. One "
        "layer below, L18E001 carries 75% of a smaller block rescue (+0.19) and is positively specific (Spec +0.098 [+0.040, +0.162]); it is the joint "
        "discovery argmax at the paper's threshold; on the held-out validation split its rescue is higher than L19E006's but not significantly so "
        "(paired difference +0.077, p = 0.08), whereas its Spec is decisively higher (+0.257, p < 0.0001). Because L19E006's Spec is negative, 27 of the "
        "30 recurrent pairs beat it on validation Spec (14 with non-overlapping CIs), but L18E001 is the only pair anywhere whose Spec CI excludes zero, "
        "and no pair beats L19E006 on rescue with a CI clear of its CI. Its absolute effect is still "
        "small, a third of the L19 block rescue and a quarter of the L21 block, so the layer-level conclusion 'Mixtral's factual signal is spread over a "
        "coalition' stands; the expert-level statement 'the recurrent expert is non-specific' is a property of the L19 choice, not of the model. The "
        "selection is also fragile in this regime: over the 25 grid settings the joint winner is L21E001 (12x, at thresholds 32-48 where it becomes "
        "recurrent; validation +0.29, the highest of any expert evaluated in this run, but clean-active in only 59/128 paper discovery cases), L22E001 (8x, "
        "thresholds 80-96), L18E001 (2x) and never L19E006, and the per-split two-stage layer itself flips to L21 in 4 of 5 seeds, where no expert is "
        "recurrent at threshold >= 64. With 8 experts and top-2 routing, recurrence in half the discovery cases is a demanding requirement that the "
        "strongest experts fail, so for Mixtral without BOS the recurrence threshold, not the rescue, decides which expert is reported. The E001 pattern "
        "is not a BOS artefact: with and without BOS, E001 is the best expert by validation rescue at L17, L18, L21 and L22 (table below), and even "
        "without BOS the strongest L19 expert on validation is E002 (+0.22), which fails recurrence there and so is never reported under the paper's "
        "protocol; the BOS-induced change is that E002's L19 activity rises above the threshold."),
}


# ---------------------------------------------------------------------------------------------------------------
def verdict(out: dict) -> str:
    s = out["sets"]["paper"]
    ts, w, ch, st, nb = s["two_stage"], s["winner"], s["challenger"], s["stability"], s["neighbours"]
    lines = []
    if s["same_as_two_stage"]:
        lines.append(f"**Verdict: the joint search returns the two-stage winner.** Over all {s['n_candidates']} recurrent (layer, expert) pairs in "
                     f"{s['n_layers_with_candidates']} layers, the discovery argmax is {w['pair']} (discovery {f3(w['disc_allcase_mean'])}, validation "
                     f"{ci(w['val_rescue'], w['val_rescue_lo'], w['val_rescue_hi'])}, Spec {ci(w['val_spec'], w['val_spec_lo'], w['val_spec_hi'])}).")
    else:
        beats = w["val_rescue"] > ts["val_rescue"]
        lines.append(f"**Verdict: the joint search picks {w['pair']}, not the two-stage winner {ts['pair']}"
                     f"{' (rank ' + str(ts['rank']) + ' on discovery)' if ts['rank'] else ' (not recurrent)'}.** On validation {w['pair']} gives "
                     f"{ci(w['val_rescue'], w['val_rescue_lo'], w['val_rescue_hi'])} vs {ci(ts['val_rescue'], ts['val_rescue_lo'], ts['val_rescue_hi'])} for {ts['pair']} "
                     f"(Spec {ci(w['val_spec'], w['val_spec_lo'], w['val_spec_hi'])} vs {ci(ts['val_spec'], ts['val_spec_lo'], ts['val_spec_hi'])}); "
                     f"the joint winner {'does' if beats else 'does not'} beat the two-stage winner on validation rescue"
                     f"{' (rescue CIs overlap' if (w['val_rescue_lo'] < ts['val_rescue_hi'] and ts['val_rescue_lo'] < w['val_rescue_hi']) else ' (rescue CIs do not overlap'}"
                     f"{'; Spec CIs overlap)' if (w['val_spec_lo'] < ts['val_spec_hi'] and ts['val_spec_lo'] < w['val_spec_hi']) else '; Spec CIs do not overlap)'}.")
        ov = s.get("overlap")
        if ov:
            dr, ds = ov["diff_rescue_b_minus_a"], ov["diff_spec_b_minus_a"]
            lines.append(f"Paired per-case comparison on validation ({ov['b']} minus {ov['a']}): rescue {ci(dr['mean'], dr['ci_lo'], dr['ci_hi'])}, sign-flip p = {dr['p']:.4f}; "
                         f"Spec {ci(ds['mean'], ds['ci_lo'], ds['ci_hi'])} over the {ov['n_spec_pairs']} cases where both have controls, p = {ds['p']:.4f}.")
    if ch is not None:
        lines.append(f"The strongest recurrent pair outside L{ts['layer']} is {ch['pair']} (discovery rank {ch['rank']}, discovery {f3(ch['disc_allcase_mean'])}, validation "
                     f"{ci(ch['val_rescue'], ch['val_rescue_lo'], ch['val_rescue_hi'])}, Spec {ci(ch['val_spec'], ch['val_spec_lo'], ch['val_spec_hi'])}).")
    sh = s["shrinkage"]
    chg = sh["val_of_winner"] - sh["disc_max"]
    txt = (f"Discovery maximum vs its validation value (joint winner {w['pair']}): {f3(sh['disc_max'])} -> {f3(sh['val_of_winner'])} "
           f"(change {f3(chg)}, {100 * chg / sh['disc_max']:+.0f}% of the discovery value; a negative change is the winner's-curse shrinkage)")
    if not s["same_as_two_stage"]:
        txt += f"; the two-stage winner {ts['pair']}: {f3(sh['two_stage_disc'])} -> {f3(sh['two_stage_val'])} (change {f3(sh['two_stage_val'] - sh['two_stage_disc'])})"
    lines.append(txt + ".")
    ov = s.get("overlap")
    if ov:
        lines.append(f"Overlap of {ov['a']} and {ov['b']} on the {ov['n']} validation cases: per-case rescue correlation Pearson r = {ov['pearson_r']:.2f} "
                     f"(Spearman {ov['spearman_r']:.2f}); both positive in {100 * ov['both_positive']:.0f}% of cases, only {ov['a']} in {100 * ov['a_only_positive']:.0f}%, only {ov['b']} in "
                     f"{100 * ov['b_only_positive']:.0f}%, neither in {100 * (1 - ov['either_positive']):.0f}%; mean per-case max(rescue) {f3(ov['mean_max'])} vs {f3(ov['mean_a'])} / {f3(ov['mean_b'])} alone.")
    lines.append(f"Robustness (Appendix D grid, {st['n_settings']} split-seed x threshold settings): the joint winner equals {ts['pair']} in "
                 f"{st['n_equals_reference']}/{st['n_with_winner']} settings with a winner, lies outside L{ts['layer']} in {st['n_outside_reference_layer']}/{st['n_with_winner']}, "
                 f"and equals the per-split two-stage selection in {st['n_equals_split_two_stage']}/{st['n_with_winner']}. In {st['n_split_two_stage_none']} settings the per-split "
                 f"two-stage procedure finds no recurrent expert at its layer at all (the joint search still returns a pair); the joint winner leaves L{ts['layer']} in "
                 f"{st['n_outside_with_two_stage_candidate']} settings where the two-stage procedure did have a candidate. Winners: "
                 + ", ".join(f"{k} x{v}" for k, v in st["winner_counts"].items()) + ".")
    vr = s["vs_reference"]
    lines.append(f"All {vr['n_pairs']} recurrent pairs evaluated on validation against {ts['pair']}: {vr['n_beat_ref_rescue']} have a higher rescue "
                 f"({vr['n_rescue_ci_above_ref_ci']} with a rescue CI entirely above {ts['pair']}'s CI), {vr['n_beat_ref_spec']} a higher Spec ({vr['n_spec_ci_above_ref_ci']} with a Spec CI "
                 f"entirely above {ts['pair']}'s CI); {vr['n_spec_ci_above_zero']} pairs have a Spec CI above zero" + (f": {', '.join(vr['pairs_spec_ci_above_zero'][:12])}" if vr['pairs_spec_ci_above_zero'] else "") + ".")
    br, bs = nb["best_by_rescue"], nb["best_by_spec"]
    lines.append(f"Neighbouring layers {nb['layers']}: {nb['n_experts']} experts with any validation activity were evaluated; {nb['n_beat_rescue']} beat {ts['pair']} on validation rescue "
                 f"({f3(nb['reference']['val_rescue'])}) and {nb['n_beat_spec']} on Spec ({f3(nb['reference']['val_spec'])}); among the {nb['n_recurrent']} recurrent ones, "
                 f"{nb['n_recurrent_beat_rescue']} and {nb['n_recurrent_beat_spec']} respectively. Best by rescue: {br['pair']} "
                 f"{ci(br['val_rescue'], br['val_rescue_lo'], br['val_rescue_hi'])} (Spec {f3(br['val_spec'])}, active {br['val_active']}/{s['n_val']}"
                 f"{', recurrent' if br['recurrent'] else ', NOT recurrent'}); best by Spec: {bs['pair']} Spec {ci(bs['val_spec'], bs['val_spec_lo'], bs['val_spec_hi'])} "
                 f"(rescue {f3(bs['val_rescue'])}, active {bs['val_active']}/{s['n_val']}{', recurrent' if bs['recurrent'] else ', NOT recurrent'}).")
    return "\n".join(f"- {l}" for l in lines)


def mixtral_e001_table(summary: dict) -> str:
    """Layers 15-25 of both Mixtral runs: best recurrent expert, best expert by validation rescue, and E001's validation numbers (paper set)."""
    runs = [r for r in ("mixtral_bos_alllayers", "mixtral_nobos_alllayers") if r in summary["runs"]]
    if len(runs) < 2:
        return ""
    rows = []
    data = {}
    for run in runs:
        md = A.load_model(run); cache = X.ExpertCache(md); d, v = md.ids("paper", "discovery"), md.ids("paper", "validation")
        for l in range(15, 26):
            best_rec, _ = X.select_at_layer(cache, l, d, 64)
            pcs = {int(e): cache.per_case(l, int(e), v, 1) for e in cache.piv[l].columns}
            be = max(pcs, key=lambda e: pcs[e].rescue.mean())
            p1 = pcs.get(1)
            data[(run, l)] = dict(best_rec=(f"E{best_rec:03d}" if best_rec is not None else "none"), best_val=f"E{be:03d} {pcs[be].rescue.mean():+.3f}",
                                 e1=(f"{p1.rescue.mean():+.3f} / {p1.spec.mean():+.3f} ({int(p1.active.sum())}/{len(v)})" if p1 is not None else "n/a"))
    for l in range(15, 26):
        b, n = data[(runs[0], l)], data[(runs[1], l)]
        rows.append([f"L{l}", b["best_rec"], b["best_val"], b["e1"], n["best_rec"], n["best_val"], n["e1"]])
    return md_table(["Layer", "BOS: best recurrent", "BOS: best by val. rescue", "BOS: E001 rescue / Spec (active)", "no BOS: best recurrent", "no BOS: best by val. rescue", "no BOS: E001 rescue / Spec (active)"],
                    rows, os.path.join(TAB, "ext1_mixtral_E001_by_layer"), caption="Mixtral layers 15-25, paper set: best recurrent expert (threshold 64), best expert by validation rescue, and E001's validation rescue / Spec, with and without BOS")


def write_section(summary: dict, path: str):
    L = []
    L.append("## Extension 1: layer-then-expert versus joint layer x expert search\n")
    L.append("**Question.** The paper selects the layer L* by MoE-block rescue on discovery cases and then searches for a recurrent expert inside L* only. "
             "Could a single expert in another layer be a better (higher validation rescue, more specific) locus that the two-stage procedure never sees?\n")
    L.append("**Method.** We ran the expert pass at *every* MoE layer (48 for Qwen3, 32 for Mixtral; `run_expert.py --layers 0..L-1 --no-pairs`, split into "
             "layer chunks so that at most ~90k wavefront rows are resident) for the three base runs: Qwen3-30B-A3B-Base with tokenizer defaults "
             "(paper, strict and relaxed case sets), Mixtral-8x7B-v0.1 with BOS (all three sets) and Mixtral without BOS (paper set; the protocol that "
             "reproduces the paper). For every (layer, expert) pair we compute on the paper's discovery split the clean-active count and the all-case mean "
             "rescue (zero where the expert is not clean-active), keep pairs meeting the recurrence threshold (half the discovery split: 64 of 128, 128 of "
             "256) and (a) per layer take the best recurrent expert (the paper's rule applied at every layer), (b) take the global argmax over all pairs "
             "(joint search). Every selection is then evaluated on the untouched validation split with the same statistics as Table 1 (all-case rescue, "
             "active-random specificity with 3 controls for Qwen3 and 1 for Mixtral, 5,000-resample bootstrap CIs). Concentration is Rescue(best expert)/"
             "Rescue(MoE block) on validation from the same pass (paired bootstrap CI). Robustness repeats the joint search over the Appendix D grid "
             "(split seeds 0-4 x thresholds 32, 48, 64, 80, 96, doubled for the 512-case relaxed set; `random.Random(seed).shuffle` of the set's ids as in `analysis.stability_grid`; the per-split two-stage selection re-selects the layer by discovery block rescue and then the expert inside it). Code: "
             "`moetrace/ext1_analysis.py`, `scripts/ext1_analyze.py`; rows: `results/<run>/expert_rows.parquet` for runs "
             "`qwen3_bos_alllayers`, `mixtral_bos_alllayers`, `mixtral_nobos_alllayers`.\n")
    L.append("**Multiple-comparison caveat.** The joint search compares 48 x 128 = 6,144 (Qwen3) or 32 x 8 = 256 (Mixtral) pairs on discovery; the discovery "
             "maximum is therefore biased upward and only the validation numbers of a selected pair are unbiased estimates of its effect. We report the "
             "discovery-to-validation shrinkage of the maximum for that reason. The neighbouring-layer question (e) below scans every expert of 2-3 layers on "
             "*validation* directly and is a post-hoc comparison of ~100-400 experts against one pre-registered expert; a single expert exceeding the "
             "reference there is expected by chance and is not evidence unless it is also recurrent and its CI excludes the reference.\n")
    for run, out in summary["runs"].items():
        cfg = X.RUNS[run]
        s = out["sets"]["paper"]
        L.append(f"### {out['label']} (`results/{run}`)\n")
        L.append(verdict(out) + "\n")
        if INTERPRETATION.get(run):
            L.append(INTERPRETATION[run] + "\n")
        cons = s["consistency"]
        L.append(f"Consistency with the original single-layer pass (bf16 fingerprint): {cons['pair']} on the paper validation split has rescue "
                 f"{f3(cons['all_layer_pass']['val_rescue'])} / Spec {f3(cons['all_layer_pass']['val_spec'])} in the all-layer pass vs "
                 f"{f3(cons['base_pass']['val_rescue'])} / {f3(cons['base_pass']['val_spec'])} in `results/{cfg['base_run']}` "
                 f"(active {cons['all_layer_pass']['val_active']} vs {cons['base_pass']['val_active']}); the difference ({f3(cons['diff_rescue'])}) is bf16 noise.\n")
        L.append(f"![ext1 curves {out['short']}](../figures/ext1_curves_{out['short']}.png)\n")
        L.append(f"Figure E1-{out['short']}: validation MoE-block rescue by layer (grey, 95% band) with the recurrence-first best expert's validation rescue "
                 f"(solid) and active-random Spec (dashed) at every layer; star = two-stage winner, diamond = joint winner if different. Gaps = layers with "
                 f"no recurrent expert. Full per-layer table: `results/tables/ext1_best_expert_by_layer_{out['short']}.md`.\n")
        L.append(open(os.path.join(TAB, f"ext1_top10_{out['short']}.md")).read())
        # other sets one-liners
        for sname, d in out["sets"].items():
            if sname == "paper":
                continue
            w, ts = d["winner"], d["two_stage"]
            L.append(f"- {sname} set (n_disc={d['n_disc']}, threshold {d['threshold']}, {d['n_candidates']} recurrent pairs): joint winner {w['pair']} "
                     f"(validation {ci(w['val_rescue'], w['val_rescue_lo'], w['val_rescue_hi'])}, Spec {f3(w['val_spec'])}); two-stage winner {ts['pair']} "
                     f"{'(same)' if d['same_as_two_stage'] else '(validation ' + ci(ts['val_rescue'], ts['val_rescue_lo'], ts['val_rescue_hi']) + ')'}; "
                     f"grid: joint = reference in {d['stability']['n_equals_reference']}/{d['stability']['n_with_winner']}, outside L{ts['layer']} in "
                     f"{d['stability']['n_outside_reference_layer']}/{d['stability']['n_with_winner']}.")
        L.append("")
        L.append(open(os.path.join(TAB, f"ext1_concentration_{out['short']}.md")).read())
        conc = pd.DataFrame(s["concentration"])
        if len(conc):
            top = conc.iloc[0]
            L.append(f"Concentration (paper set): among the {len(conc)} layers whose block rescue is clearly positive, the signal is most concentrated in one "
                     f"expert at L{int(top.layer)} (E{int(top.e_star):03d}: {100 * top.concentration:.0f}% of the block rescue); at the two-stage layer L{cfg['two_stage'][0]} the ratio is "
                     f"{100 * float(conc[conc.layer == cfg['two_stage'][0]].concentration.iloc[0]):.0f}%" + (" (the maximum)." if int(top.layer) == cfg["two_stage"][0] else ".") + "\n")
        L.append(open(os.path.join(TAB, f"ext1_neighbours_{out['short']}.md")).read())
        L.append(f"Stability grid rows: `results/tables/ext1_stability_{out['short']}.md`; all evaluated neighbour-layer experts: "
                 f"`results/tables/ext1_neighbours_all_{out['short']}_<set>.csv`.\n")
    e1 = mixtral_e001_table(summary)
    if e1:
        L.append("### The E001 band in Mixtral, with and without BOS\n")
        L.append(e1)
        L.append("Expert indices are independent parameters in different layers, so the recurrence of index 1 as the strongest expert of L17, L18, L21 and L22 under both protocols is a property of the trained model (or of how the routers were initialised), not of the BOS token; it is an open observation for Direction 3.\n")
    L.append("### Summary across models\n")
    rows = []
    for run, out in summary["runs"].items():
        s = out["sets"]["paper"]
        w, ts, st = s["winner"], s["two_stage"], s["stability"]
        rows.append([out["label"], ts["pair"], ci(ts["val_rescue"], ts["val_rescue_lo"], ts["val_rescue_hi"]), ci(ts["val_spec"], ts["val_spec_lo"], ts["val_spec_hi"]),
                     w["pair"], ci(w["val_rescue"], w["val_rescue_lo"], w["val_rescue_hi"]), ci(w["val_spec"], w["val_spec_lo"], w["val_spec_hi"]),
                     f"{st['n_equals_reference']}/{st['n_with_winner']}", f"{st['n_outside_reference_layer']}/{st['n_with_winner']}",
                     f"{f3(s['shrinkage']['disc_max'])} -> {f3(s['shrinkage']['val_of_winner'])}"])
    L.append(md_table(["Model / protocol", "Two-stage winner", "Val. rescue", "Val. Spec", "Joint winner", "Val. rescue", "Val. Spec", "Grid: joint = two-stage", "Grid: outside L*", "Disc. max -> val."],
                      rows, os.path.join(TAB, "ext1_summary"), caption="Two-stage vs joint selection on the paper case set (validation split)"))
    L.append(f"_Generated {summary['generated_utc']} by scripts/ext1_analyze.py._\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(X.RUNS))
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--no-section", action="store_true")
    args = ap.parse_args()
    summary = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": {}}
    for run in args.runs.split(","):
        out = analyze_run(run, args.allow_incomplete)
        if out is not None:
            summary["runs"][run] = out
    js = jsonable(summary)
    for run in js["runs"]:
        for s in js["runs"][run]["sets"].values():
            s.pop("curve", None)
    with open(os.path.join(RESULTS, "ext1_summary.json"), "w") as f:
        json.dump(js, f, indent=1)
    if not args.no_section and summary["runs"]:
        write_section(summary, os.path.join(SEC, "ext1_joint_search.md"))
        log("section written")
    log("done")


if __name__ == "__main__":
    main()
