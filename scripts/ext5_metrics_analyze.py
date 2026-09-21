"""Phase 2 / F5 driver (ext5-analysis): re-derive the paper's layer / expert selections under probability-scale metrics.

Input: results/<run>_metrics/{sweep_rows,expert_rows,sweep_cases}.parquet + case_sets.json (ext5-engine, `--metrics`), with the
agreed extra columns logp_true, logp_foil, p_true, p_foil, rank_true, kl_to_clean on every row.

Usage: python scripts/ext5_metrics_analyze.py [--runs qwen3_metrics,mixtral_nobos_metrics,mixtral_bos_metrics] [--dirs d1,d2] [--section-only]
       python scripts/ext5_metrics_analyze.py --synthetic qwen3 --out /tmp/x     (schema / plumbing test on fabricated columns)
Writes results/tables/ext5_metrics_*.{md,csv}, results/figures/ext5_metrics_{curves,percase}_<short>.{png,pdf}, results/ext5_metrics_summary.json
and results/sections/ext5_f5_metrics.md. With no data present it writes the method part of the section with a 'pending' note.
"""
import argparse, json, os, shutil, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import analysis as A
from moetrace import ext5_metrics as M
from moetrace.models import RESULTS
from moetrace.report import md_table
from moetrace.stats import summarize

TAB, FIG, SEC = (os.path.join(RESULTS, d) for d in ("tables", "figures", "sections"))
SUMMARY = os.path.join(RESULTS, "ext5_metrics_summary.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def cis(s, d=3):
    return "n/a" if s is None or s.get("n", 0) == 0 else f"{s['mean']:+.{d}f} [{s['ci_lo']:+.{d}f}, {s['ci_hi']:+.{d}f}]"


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
def make_synthetic(base_run: str, out_dir: str, seed: int = 0) -> str:
    """Fabricate the six metric columns on a copy of an existing run so that the plumbing can be tested. A per-case
    'rest of the vocabulary' log-mass makes p_true + p_foil < 1 and the identity Δ = logp_true - logp_foil exact."""
    src = os.path.join(RESULTS, base_run)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(src, "case_sets.json"), os.path.join(out_dir, "case_sets.json"))
    shutil.copy(os.path.join(src, "sweep_cases.parquet"), os.path.join(out_dir, "sweep_cases.parquet"))
    rng = np.random.default_rng(seed)
    sr = pd.read_parquet(os.path.join(src, "sweep_rows.parquet"))
    cl = sr[sr.kind == "clean"].set_index("case_id")
    rest = (np.maximum(cl.logit_true, cl.logit_foil) - 0.5 + rng.normal(0, 1.0, len(cl))).to_dict()
    clean_logp = {}

    def fab(df):
        df = df.copy()
        r = df.case_id.map(rest).values
        lse = np.logaddexp(np.logaddexp(df.logit_true.values, df.logit_foil.values), r)
        df["logp_true"] = df.logit_true.values - lse
        df["logp_foil"] = df.logit_foil.values - lse
        df["p_true"] = np.exp(df.logp_true); df["p_foil"] = np.exp(df.logp_foil)
        best = np.maximum(df.logit_foil.values, r)
        second = np.minimum(df.logit_foil.values, r)
        df["rank_true"] = np.where(df.logit_true.values >= best, 1, np.where(df.logit_true.values >= second, 2, 3 + rng.poisson(3, len(df)))).astype(int)
        return df

    sr = fab(sr)
    for cid, g in sr[sr.kind == "clean"].groupby("case_id"):
        clean_logp[cid] = float(g.logp_true.iloc[0])
    sr["kl_to_clean"] = np.where(sr.kind == "clean", 0.0, 0.5 * (sr.logp_true - sr.case_id.map(clean_logp)) ** 2 + 0.01)
    sr.to_parquet(os.path.join(out_dir, "sweep_rows.parquet"), index=False)
    er = pd.read_parquet(os.path.join(src, "expert_rows.parquet"))
    er = fab(er)
    er["kl_to_clean"] = 0.5 * (er.logp_true - er.case_id.map(clean_logp)) ** 2 + 0.01
    er.to_parquet(os.path.join(out_dir, "expert_rows.parquet"), index=False)
    return out_dir


# ---------------------------------------------------------------------------------------------------------------
def analyze_run(d: str, cfg: dict) -> dict:
    short, nc, ts = cfg["short"], cfg["n_controls"], tuple(cfg["two_stage"])
    md0 = M.load_run(d)
    schema = M.check_schema(md0)
    out = dict(dir=d, short=short, label=cfg["label"], model=cfg["model"], two_stage=list(ts), second=list(cfg["second"]), schema=schema)
    if not schema["sweep_rows"].get("complete"):
        log(d, "sweep_rows lacks the metric columns:", schema["sweep_rows"].get("missing")); out["skipped"] = True; return out
    out["identity"] = M.identity_check(md0)
    md, ref = M.enrich(md0, d)
    disc, val = md.ids("paper", "discovery"), md.ids("paper", "validation")
    mds = {m: M.with_metric(md, m, ref) for m in M.METRICS}
    # ---- layers
    curves, arg = M.layer_curves(mds, "paper")
    curves.to_csv(os.path.join(TAB, f"ext5_metrics_layer_curves_{short}.csv"), index=False)
    out["layers"] = arg
    Ld = arg["delta"]["L_star"]
    rows = []
    for m, a in arg.items():
        s = a["sharpness"]
        rows.append([M.METRICS[m]["short"], f"{a['n_disc']} / {a['n_val']}", f"L{a['L_star']}", f3(a["disc_mean"]), cis(a["val"]), f"L{a['val_argmax_layer']}", f3(a["val_argmax"]),
                     f"L{s['top_layer']} vs L{s['next_layer']}: {f3(s['gap'])}", ", ".join(f"L{l} {v:+.3f}" for l, v in a["disc_top5"]), "same" if a["L_star"] == Ld else f"differs from Δ (L{Ld})"])
    md_table(["Metric", "n disc / val", "L* (disc. argmax)", "Disc. mean at L*", "Val. at L* [95% CI]", "Val. argmax", "Val. max", "Sharpness (top vs next, val)", "Disc. top-5", "vs Δ"],
             rows, os.path.join(TAB, f"ext5_metrics_layers_{short}"), caption=f"{cfg['label']}: layer selection under each metric (paper set; 'ratio' uses only cases with drop >= 1)")
    # ---- normalised rescue at the Δ layer and at every layer in the expert pass
    nr = {str(Ld): M.normalised_rescue(md, ref, Ld, val)}
    out["normalised"] = nr
    # ---- experts
    layers_present = sorted(int(l) for l in md.expert_rows.layer.unique()) if md.expert_rows is not None else []
    out["layers_present"] = layers_present
    exp_rows, fixed_rows, joint_rows = [], [], []
    out["experts"] = {}; out["fixed"] = {}; out["joint"] = {}
    for m, mdm in mds.items():
        dm, vm = mdm.ids("paper", "discovery"), mdm.ids("paper", "validation")
        res = {}
        for L in sorted(set([Ld] + ([arg[m]["L_star"]] if arg[m]["L_star"] in layers_present else []))):
            if L not in layers_present:
                continue
            sel = M.expert_selection(mdm, L, dm, vm, nc)
            res[str(L)] = {k: v for k, v in sel.items() if k != "per_case"}
            if sel.get("e_star") is not None:
                exp_rows.append([M.METRICS[m]["short"], f"L{L}" + (" (= Δ L*)" if L == Ld else " (own L*)"), f"E{sel['e_star']:03d}", f"{sel['selection']['disc_active']}/{len(dm)}", f3(sel["selection"]["disc_allcase_mean"]),
                                 f"{sel['val_active']}/{len(vm)}", cis(sel["rescue"]), cis(sel["spec"]), sel["spec_sign"], "same" if (L, sel["e_star"]) == ts else "differs"])
            else:
                exp_rows.append([M.METRICS[m]["short"], f"L{L}", "none recurrent", "", "", "", "", "", "", ""])
        out["experts"][m] = res
        fx = {}
        for tag, (L, e) in (("two_stage", ts), ("second", tuple(cfg["second"]))):
            if L in layers_present:
                ev = M.evaluate_fixed(mdm, L, e, vm, nc)
                fx[tag] = {k: v for k, v in ev.items() if k != "per_case"}
                if ev.get("available"):
                    fixed_rows.append([M.METRICS[m]["short"], M.pair(L, e) if hasattr(M, "pair") else f"L{L}E{e:03d}", f"{ev['val_active']}/{len(vm)}", cis(ev["rescue"]), cis(ev["spec"]), ev["spec_sign"]])
        out["fixed"][m] = fx
        if layers_present:
            try:
                js = M.joint_top(mdm, dm, vm, nc, ts, top_k=10)
                out["joint"][m] = dict(n_candidates=js["n_candidates"], layers_present=js["layers_present"], winner=js["winner"], same_as_two_stage=js["same_as_two_stage"],
                                       top=js["top"].to_dict("records"))
                for _, r in js["top"].iterrows():
                    joint_rows.append([M.METRICS[m]["short"], int(r["rank"]), r["pair"], f"{int(r.disc_active)}/{len(dm)}", f3(r.disc_allcase_mean), f"{int(r.val_active)}/{len(vm)}",
                                       f"{r.val_rescue:+.3f} [{r.val_rescue_lo:+.3f}, {r.val_rescue_hi:+.3f}]", f"{r.val_spec:+.3f} [{r.val_spec_lo:+.3f}, {r.val_spec_hi:+.3f}]", "yes" if r.is_two_stage else ""])
            except Exception as ex:  # noqa
                out["joint"][m] = dict(error=str(ex))
    md_table(["Metric", "Layer", "Selected expert", "Disc. active", "Disc. all-case", "Val. active", "Val. rescue [95% CI]", "Spec [95% CI]", "Spec sign", "vs paper (Δ)"],
             exp_rows, os.path.join(TAB, f"ext5_metrics_experts_{short}"), caption=f"{cfg['label']}: recurrence-first expert selection under each metric (threshold half the discovery split; all quantities in the metric's own units)")
    md_table(["Metric", "Pair", "Val. active", "Val. rescue [95% CI]", "Spec [95% CI]", "Spec sign"],
             fixed_rows, os.path.join(TAB, f"ext5_metrics_fixed_{short}"), caption=f"{cfg['label']}: the paper's expert and the second locus evaluated under each metric (validation split)")
    md_table(["Metric", "Rank", "Pair", "Disc. active", "Disc. all-case", "Val. active", "Val. rescue [95% CI]", "Spec [95% CI]", "Two-stage"],
             joint_rows, os.path.join(TAB, f"ext5_metrics_joint_{short}"), caption=f"{cfg['label']}: joint (layer, expert) search restricted to the layers of the metrics expert pass {layers_present}, top-10 per metric")
    n = nr[str(Ld)]
    md_table(["Layer", "n", "Mean Δ rescue", "Mean Δ drop", "Normalised rescue (mean/mean) [95% CI]", "Cases with drop >= 1", "Per-case Δ rescue/drop (drop >= 1) [95% CI]", "Mean Δp rescue", "Mean p drop", "Normalised Δp (mean/mean) [95% CI]"],
             [[f"L{Ld}", n["n"], f3(n["mean_rescue"]), f3(n["mean_drop"]), f"{n['ratio']:.3f} [{n['ratio_lo']:.3f}, {n['ratio_hi']:.3f}]", n["n_drop_ge_1"], cis(n["per_case_ratio"]),
               f3(n.get("mean_rescue_dp")), f3(n.get("mean_p_drop")), f"{n.get('ratio_dp', np.nan):.3f} [{n.get('ratio_dp_lo', np.nan):.3f}, {n.get('ratio_dp_hi', np.nan):.3f}]"]],
             os.path.join(TAB, f"ext5_metrics_normalised_{short}"), caption=f"{cfg['label']}: normalised rescue of the block patch at the paper's layer (validation)")
    # ---- funnel
    fn = M.funnel_alternative(md, ref)
    out["funnel"] = fn
    rows = []
    for name, o in (("strict Δ (>= 1.0, drop >= 0.5) vs p_clean >= 0.5", fn["strict_vs_p"]), ("relaxed Δ (>= 0.5, drop >= 0.25) vs p_clean >= 0.5", fn["relaxed_vs_p"]),
                    ("strict Δ vs p_clean >= 0.5 & p drop >= 0.25", fn["strict_vs_p_drop"])):
        rows.append([name, o["n_a"], o["n_b"], o["both"], o["a_only"], o["b_only"], o["neither"], f"{o['jaccard']:.2f}"])
    md_table(["Funnels (A vs B)", "pass A", "pass B", "both", "A only", "B only", "neither", "Jaccard"], rows, os.path.join(TAB, f"ext5_metrics_funnel_{short}"),
             caption=f"{cfg['label']}: overlap of the paper's Δ funnel with the alternative p_clean(true) >= 0.5 over the {fn['n']} cases of the run (the funnel scan itself was run on Δ; these are the cases that entered any case set)")
    rows = [[s, v["n"], v["pass_strict"], v["pass_p"], v["top1_clean"], v["top1_flipped"], f"{v['p_clean_median']:.3f}", v["p_clean_ge_09"]] for s, v in fn["per_set"].items()]
    md_table(["Set", "n", "pass strict Δ", "pass p_clean >= 0.5", "clean top-1", "noise flips top-1", "median p_clean", "p_clean >= 0.9"], rows, os.path.join(TAB, f"ext5_metrics_funnel_sets_{short}"),
             caption=f"{cfg['label']}: per case set")
    # ---- clean top-1 token analysis and tail concentration of the metric argmax layers
    try:
        out["top1"] = M.clean_top1_table(md, ref, cfg["model"])
    except Exception as ex:  # tokenizer missing etc.
        out["top1"] = dict(error=str(ex))
    tc = {}
    for m, a in arg.items():
        for L in sorted(set([a["L_star"], Ld])):
            tc[f"{m}@L{L}"] = M.tail_concentration(md, ref, L, disc + val, M.METRICS[m]["col"], "kl_noised", 0.9)
    out["tail"] = tc
    rows = [[M.METRICS[t.split("@")[0]]["short"], t.split("@")[1], f3(r["mean"]), f3(r["mean_top"]), f3(r["mean_rest"]), f"{100 * r['share_top']:.0f}%" if not np.isnan(r["share_top"]) else "n/a", f"{r['corr']:.2f}"] for t, r in tc.items()]
    md_table(["Metric", "Layer", "Mean rescue (256 cases)", "Mean, top-10% KL(noised‖clean) cases", "Mean, other 90%", "Share of summed rescue from the top-10%", "r(rescue, KL noised)"],
             rows, os.path.join(TAB, f"ext5_metrics_tail_{short}"), caption=f"{cfg['label']}: concentration of each metric's block rescue (at its own discovery argmax and at the Δ layer) in the 26 cases whose noised distribution is farthest from the clean one")
    # ---- per-case frame and figures
    e_star_d = out["experts"]["delta"].get(str(Ld), {}).get("e_star")
    pc = M.per_case_frame(md, ref, Ld, val, e_star_d)
    pc.to_csv(os.path.join(TAB, f"ext5_metrics_percase_{short}_L{Ld}.csv"), index=False)
    corr = {}
    for m in M.METRICS:
        if f"block_{m}" in pc.columns and m != "delta":
            x, y = pc.block_delta.values, pc[f"block_{m}"].values
            ok = ~(np.isnan(x) | np.isnan(y))
            corr[m] = dict(pearson=float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else np.nan, spearman=float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman")) if ok.sum() > 2 else np.nan, n=int(ok.sum()))
    out["percase_corr_block"] = corr
    fl = pc[pc.top1_flipped.astype(bool)]; nf = pc[~pc.top1_flipped.astype(bool)]
    out["flip_split"] = dict(n_flipped=int(len(fl)), n_not=int(len(nf)), block_delta_flipped=summarize(fl.block_delta.values, with_p=False), block_delta_not=summarize(nf.block_delta.values, with_p=False),
                             block_dp_flipped=summarize(fl.block_dp.values, with_p=False) if "block_dp" in pc else None, block_dp_not=summarize(nf.block_dp.values, with_p=False) if "block_dp" in pc else None,
                             share_of_dp_from_flipped=float(fl.block_dp.sum() / pc.block_dp.sum()) if "block_dp" in pc and pc.block_dp.sum() else np.nan,
                             share_of_delta_from_flipped=float(fl.block_delta.sum() / pc.block_delta.sum()) if pc.block_delta.sum() else np.nan)
    figures(cfg, curves, arg, pc, Ld, e_star_d, short)
    return out


def figures(cfg, curves, arg, pc, Ld, e_star, short):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"qwen3": "#2a78d6", "mixtral": "#eb6834"}[cfg["model"]]
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 1, figsize=(9, 3.8))
    styles = {"delta": (INK, 2.0, "-"), "dp": (C, 1.5, "-"), "logp": (C, 1.2, "--"), "rank": (INK2, 1.2, ":"), "kl": (INK2, 1.2, "-."), "ratio": (INK, 1.0, ":")}
    for m, g in curves.groupby("metric"):
        g = g.sort_values("layer")
        scale = np.nanmax(np.abs(g.val_mean.values)) or 1.0
        col, lw, ls = styles[m]
        ax.plot(g.layer, g.val_mean / scale, color=col, lw=lw, ls=ls, label=f"{M.METRICS[m]['short']} (max {scale:+.3f}, L* = L{arg[m]['L_star']})")
        ax.plot([arg[m]["L_star"]], [g.set_index('layer').val_mean[arg[m]["L_star"]] / scale], marker="*", ms=9, color=col, ls="none")
    ax.axhline(0, color=INK2, lw=0.6); ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_xlabel("MoE layer"); ax.set_ylabel("validation mean rescue / max |mean|")
    ax.set_title(f"{cfg['label']}: layer curves under each metric, each scaled by its own maximum (star = discovery argmax)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"ext5_metrics_curves_{short}.png"), dpi=170); fig.savefig(os.path.join(FIG, f"ext5_metrics_curves_{short}.pdf")); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    fl = pc.top1_flipped.astype(bool).values
    ax = axes[0]
    ax.scatter(pc.block_delta[~fl], pc.block_dp[~fl], s=14, color=C, alpha=0.7, label=f"top-1 kept under noise (n={int((~fl).sum())})")
    ax.scatter(pc.block_delta[fl], pc.block_dp[fl], s=18, color=INK, alpha=0.8, marker="D", label=f"noise flipped the top-1 (n={int(fl.sum())})")
    ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel(f"Δ rescue of the L{Ld} block patch"); ax.set_ylabel("Δp rescue (p_true patched − noised)")
    ax.set_title(f"A. block patch at L{Ld}: Δ vs Δp per validation case", loc="left", fontsize=9); ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    sc = ax.scatter(pc.p_true_clean, pc.block_dp, c=pc.block_delta, cmap="viridis", s=16, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="Δ rescue")
    ax.set_xlabel("p_true (clean)"); ax.set_ylabel("Δp rescue"); ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title("B. Δp vs the clean probability (saturation)", loc="left", fontsize=9)
    ax = axes[2]
    if e_star is not None and "expert_delta" in pc.columns:
        act = pc.expert_active.astype(bool).values
        ax.scatter(pc.expert_delta[act & ~fl], pc.expert_dp[act & ~fl], s=14, color=C, alpha=0.7, label="top-1 kept")
        ax.scatter(pc.expert_delta[act & fl], pc.expert_dp[act & fl], s=18, color=INK, alpha=0.8, marker="D", label="top-1 flipped")
        ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlabel(f"Δ rescue of L{Ld}E{e_star:03d}"); ax.set_ylabel("Δp rescue")
        ax.set_title(f"C. selected expert L{Ld}E{e_star:03d}: Δ vs Δp (active cases)", loc="left", fontsize=9); ax.legend(frameon=False, fontsize=7.5)
    for a in axes:
        a.grid(color=GRID, lw=0.5)
    fig.suptitle(f"{cfg['label']}: logit-difference vs probability-scale rescue per case (paper validation split)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"ext5_metrics_percase_{short}.png"), dpi=170); fig.savefig(os.path.join(FIG, f"ext5_metrics_percase_{short}.pdf")); plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
METHOD = (
    "## Extension 5 / F5: probability-scale metrics alongside the logit difference\n\n"
    "**Identity first.** The paper's effect measure is the logit difference Δ = logit(true) − logit(foil) at the final position. Because the "
    "softmax normaliser is common to both tokens, Δ = log p(true) − log p(foil) exactly: Δ *is* the log-odds of the two-way contrast, and "
    "'rescue' = Δ_patched − Δ_noised is a change in log-odds. What Δ does not carry is the absolute probability of the true token, its rank in "
    "the full vocabulary, or how far the whole next-token distribution is from the clean one. The engine (ext5-engine, `--metrics`) now stores "
    "for every prefill and wavefront row the full-vocabulary log-sum-exp derived quantities `logp_true`, `logp_foil`, `p_true`, `p_foil`, "
    "`rank_true` and `kl_to_clean` = KL(row ‖ clean) so that every table can be re-derived. The identity is checked numerically on every row below: "
    "max |Δ − (log p_true − log p_foil)| = 0.125 in every run, which is one bf16 ulp of the stored Δ (the logits are bf16 numbers of magnitude 16-32 "
    "and the log-softmax is computed from them in fp32), i.e. the identity holds to rounding. Expert rows use the noised reference of their own pass "
    "(`expert_prefill_L*.parquet`): cross-pass bf16 noise moves Δ_noised by up to 1.4 logits in 79% of rows, so mixing passes would corrupt every "
    "per-case rescue.\n\n"
    "**Metrics.** For a patched row (block, expert, coalition) relative to the case's noised run: Δ rescue (paper); Δp = p_true(patched) − "
    "p_true(noised); Δlog p = log p_true(patched) − log p_true(noised) (Δ without the foil); rank recovery = log2 rank(noised) − log2 rank(patched) "
    "(and the top-1 recovery indicator); KL reduction = KL(noised‖clean) − KL(patched‖clean). Normalised rescue is reported at the population level "
    "as mean rescue / mean drop with a paired bootstrap (a rescaling that cannot change any selection) and per case as rescue/drop on the cases with "
    "drop ≥ 1 (which re-weights cases and can). Each metric is substituted for the `rescue` column and the paper's procedure is re-run unchanged "
    "(`analysis.layer_analysis`, `select_expert`, `evaluate_expert`, `ext1_analysis.joint_search`): layer curve and discovery argmax on the paper "
    "set, recurrence-first expert at the Δ layer L* (and at the metric's own argmax when the expert pass covers it), validation rescue and Spec "
    "with 5,000-resample bootstrap CIs, and the joint search restricted to the layers of the metrics expert pass. The alternative funnel "
    "p_clean(true) ≥ 0.5 is compared with the paper's Δ funnel on the cases of the run. Code: `moetrace/ext5_metrics.py`, `scripts/ext5_metrics_analyze.py`.\n"
)


def write_section(summary: dict, path: str):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ext5_metrics_text import INTERPRETATION, RECOMMENDATION
    except Exception:
        INTERPRETATION, RECOMMENDATION = {}, ""
    L = [METHOD]
    runs = {k: v for k, v in summary.get("runs", {}).items() if not v.get("skipped")}
    if not runs:
        L.append("### Results\n\n_Pending: `results/{qwen3,mixtral_nobos,mixtral_bos}_metrics/{sweep_rows,expert_rows}.parquet` with the metric columns from the ext5-engine agent. "
                 "The analysis code has been exercised end-to-end on a synthetic fixture (`scripts/ext5_metrics_analyze.py --synthetic qwen3 --out <dir>`)._\n")
    for key, o in runs.items():
        L.append(f"### {o['label']} (`{os.path.relpath(o['dir'], '/home/ubuntu/MOE') if os.path.isabs(o['dir']) else o['dir']}`)\n")
        idn = o["identity"].get("sweep_rows", {})
        ide = o["identity"].get("expert_rows", {})
        L.append(f"- Identity check: max |Δ − (log p_true − log p_foil)| = {idn.get('max_abs_delta_minus_logodds', float('nan')):.2e} over {idn.get('n', 0)} sweep rows and "
                 f"{ide.get('max_abs_delta_minus_logodds', float('nan')):.2e} over {ide.get('n', 0)} expert rows (max |Δ − (logit_true − logit_foil)| {idn.get('max_abs_delta_minus_logitdiff', float('nan')):.2e}); "
                 f"p_true ranges {idn.get('p_range', ('?', '?'))[0]:.3g}–{idn.get('p_range', ('?', '?'))[1]:.3g}, max p_true + p_foil {idn.get('p_sum_max', float('nan')):.3f}, min KL {idn.get('kl_min', float('nan')):.2e}.")
        a = o["layers"]
        same = [m for m in a if a[m]["L_star"] == a["delta"]["L_star"]]
        diff = [f"{M.METRICS[m]['short']} → L{a[m]['L_star']}" for m in a if a[m]["L_star"] != a["delta"]["L_star"]]
        L.append(f"- Layer selection: Δ picks L{a['delta']['L_star']} (validation {cis(a['delta']['val'])}); the same layer under {', '.join(M.METRICS[m]['short'] for m in same if m != 'delta')}"
                 + (f"; different under {', '.join(diff)}" if diff else "; no metric changes the layer") + ".")
        ex = o["experts"]
        Ld = str(a["delta"]["L_star"])
        parts = []
        for m, r in ex.items():
            s = r.get(Ld)
            if s and s.get("e_star") is not None:
                parts.append(f"{M.METRICS[m]['short']}: E{s['e_star']:03d} (rescue {cis(s['rescue'])}, Spec {cis(s['spec'])}, {s['spec_sign']})")
            elif s:
                parts.append(f"{M.METRICS[m]['short']}: no recurrent expert")
        L.append(f"- Expert selection at L{Ld} (recurrence-first): " + "; ".join(parts) + ".")
        fs = o["flip_split"]
        L.append(f"- Cases where noise flipped the clean top-1: {fs['n_flipped']}/{fs['n_flipped'] + fs['n_not']} of the validation split; they carry "
                 f"{100 * fs['share_of_delta_from_flipped']:.0f}% of the summed Δ rescue of the L{Ld} block and {100 * fs['share_of_dp_from_flipped']:.0f}% of the summed Δp rescue "
                 f"(mean Δ {fs['block_delta_flipped']['mean']:+.3f} vs {fs['block_delta_not']['mean']:+.3f}; mean Δp {fs['block_dp_flipped']['mean']:+.3f} vs {fs['block_dp_not']['mean']:+.3f}). "
                 f"Per-case correlation of the block's Δ rescue with " + ", ".join(f"{M.METRICS[m]['short']} r = {c['pearson']:.2f} (Spearman {c['spearman']:.2f})" for m, c in o["percase_corr_block"].items()) + ".")
        n = o["normalised"][Ld]
        L.append(f"- Normalised rescue at L{Ld}: mean Δ rescue / mean drop = {n['ratio']:.3f} [{n['ratio_lo']:.3f}, {n['ratio_hi']:.3f}]; per-case ratio on the {n['n_drop_ge_1']} cases with drop ≥ 1: {cis(n['per_case_ratio'])}; "
                 f"on the probability scale mean Δp / mean p-drop = {n.get('ratio_dp', float('nan')):.3f} [{n.get('ratio_dp_lo', float('nan')):.3f}, {n.get('ratio_dp_hi', float('nan')):.3f}].")
        t1 = o.get("top1", {})
        if t1 and "n" in t1:
            L.append(f"- Clean run: the true object is the top-1 token in {t1['n_true_top1']}/{t1['n']} paper cases (median p_true {t1['median_p_true_when_top1']:.2f} there, {t1['median_p_true_when_not']:.3f} otherwise, median rank {t1['median_rank_when_not']:.0f}); "
                     f"when it is not, the top-1 is a function word or whitespace in {t1['n_function_word']}/{t1['n_not_top1']} cases ({100 * t1['frac_function_word_of_not_top1']:.0f}%): "
                     + ", ".join(f"{k} x{v}" for k, v in t1["top_tokens"][:6]) + ".")
        fn = o["funnel"]
        sv = fn["strict_vs_p"]
        L.append(f"- Alternative funnel p_clean(true) ≥ 0.5 over the {fn['n']} cases of the run: {sv['n_b']} pass vs {sv['n_a']} for the strict Δ funnel; both {sv['both']}, Δ only {sv['a_only']}, p only {sv['b_only']} "
                 f"(Jaccard {sv['jaccard']:.2f}). Clean p_true: median {fn['p_clean']['median']:.3f}, ≥ 0.9 in {100 * fn['p_clean']['frac_ge_09']:.0f}%, clean top-1 in {100 * fn['p_clean']['frac_top1']:.0f}%; "
                 f"noise flips the top-1 in {100 * fn['p_noised']['frac_flipped']:.0f}%. Paper set: {fn['per_set'].get('paper', {}).get('pass_p', '?')}/{fn['per_set'].get('paper', {}).get('n', '?')} pass p ≥ 0.5.")
        L.append("")
        for t in ("layers", "experts", "fixed", "joint", "normalised", "tail", "funnel", "funnel_sets"):
            p = os.path.join(TAB, f"ext5_metrics_{t}_{o['short']}.md")
            if os.path.exists(p):
                L.append(open(p).read())
        L.append(f"![ext5 metrics curves {o['short']}](../figures/ext5_metrics_curves_{o['short']}.png)\n")
        L.append(f"![ext5 metrics per case {o['short']}](../figures/ext5_metrics_percase_{o['short']}.png)\n")
        L.append(f"Figure E5-F5-{o['short']}: top, validation layer curves under each metric scaled by their own maximum (star = discovery argmax); bottom, per-case Δ vs Δp rescue of the "
                 f"L{Ld} block patch and of the selected expert, with cases whose clean top-1 was flipped by the noise marked, and Δp against the clean probability.\n")
        if INTERPRETATION.get(key):
            L.append(INTERPRETATION[key] + "\n")
    if runs and RECOMMENDATION:
        L.append("### Recommendation\n\n" + RECOMMENDATION + "\n")
    L.append(f"_Generated {summary['generated_utc']} by scripts/ext5_metrics_analyze.py._\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(M.RUNS))
    ap.add_argument("--dirs", default="")
    ap.add_argument("--synthetic", default="", help="base run name (e.g. qwen3) to fabricate a metrics run from, into --out")
    ap.add_argument("--out", default="")
    ap.add_argument("--section-only", action="store_true")
    args = ap.parse_args()
    global TAB, FIG, SEC
    if args.section_only:
        summary = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": {}}
        write_section(summary, os.path.join(SEC, "ext5_f5_metrics.md")); log("section written"); return
    summary = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": {}}
    if args.synthetic:
        TAB = FIG = SEC = args.out
        d = make_synthetic(args.synthetic, args.out)
        cfg = next(c for c in M.RUNS.values() if c["base"] == args.synthetic)
        cfg = dict(cfg, short=cfg["short"] + "_synthetic", label=cfg["label"] + " [SYNTHETIC TEST DATA]")
        summary["runs"]["synthetic"] = analyze_run(d, cfg)
        write_section(summary, os.path.join(args.out, "section_synthetic.md"))
        print(json.dumps(jsonable(summary), indent=1)[:2500]); log("synthetic test done; outputs in", args.out); return
    targets = [(os.path.join(RESULTS, r), M.RUNS[r]) for r in args.runs.split(",") if r in M.RUNS] if not args.dirs else \
              [(d, next(c for c in M.RUNS.values() if os.path.basename(d.rstrip("/")).startswith(c["base"]))) for d in args.dirs.split(",")]
    for d, cfg in targets:
        if not os.path.exists(os.path.join(d, "sweep_rows.parquet")):
            log(d, "not present, skipped"); continue
        log("analysing", d)
        summary["runs"][os.path.basename(d.rstrip("/"))] = analyze_run(d, cfg)
    with open(SUMMARY, "w") as f:
        json.dump(jsonable(summary), f, indent=1)
    write_section(summary, os.path.join(SEC, "ext5_f5_metrics.md"))
    log("done")


if __name__ == "__main__":
    main()
