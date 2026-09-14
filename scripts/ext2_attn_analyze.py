"""Direction 2 / Question B driver: attention-vs-MoE attribution curves from the ext2 attention sweeps.

Usage: python scripts/ext2_attn_analyze.py [--runs qwen3_bos_attnsweep,mixtral_bos_attnsweep,mixtral_nobos_attnsweep]
Writes results/tables/ext2_attn_*.{md,csv}, results/figures/ext2_attn_curves_<short>.{png,pdf},
results/figures/ext2_attn_curves_mixtral_bos_vs_nobos.{png,pdf}, results/ext2_attn_summary.json and
results/sections/ext2_attn_patch.md (appends results/sections/ext2_attn_interpretation.md if present).
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from moetrace import ext2_attn as X
from moetrace.models import MODELS, RESULTS
from moetrace.report import md_table

TAB, FIG, SEC = (os.path.join(RESULTS, d) for d in ("tables", "figures", "sections"))
for d in (TAB, FIG, SEC):
    os.makedirs(d, exist_ok=True)
DEFAULT_RUNS = ["qwen3_bos_attnsweep", "mixtral_bos_attnsweep", "mixtral_nobos_attnsweep"]
# categorical slots 1-3 of the reference palette (validated all-pairs); the residual curve is a de-emphasised
# reference series in the secondary text tone, not a fourth categorical hue
COL = {"attn_layer": "#2a78d6", "layer": "#eb6834", "block": "#1baf7a", "resid": "#52514e", "sum": "#52514e"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"


def ci(m, lo, hi):
    return "n/a" if m is None or (isinstance(m, float) and np.isnan(m)) else f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


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


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=INK2, linewidth=0.8)


# ---------------------------------------------------------------------------------------------------------------
def figure_run(run: X.Run, curves: pd.DataFrame, pk: pd.DataFrame, add: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for k in X.MAIN_KINDS:
        c = curves[curves.kind == k]
        ax.fill_between(c.layer, c.ci_lo, c.ci_hi, color=COL[k], alpha=0.15, linewidth=0)
        ax.plot(c.layer, c["mean"], color=COL[k], linewidth=2, solid_joinstyle="round", label=X.KIND_LABEL[k])
        p = pk[pk.kind == k].iloc[0]
        ax.plot([p.L_val], [p.val_max], marker="o", markersize=7, color=COL[k], markeredgecolor="white", markeredgewidth=2, zorder=5)
        ax.annotate(f"L{int(p.L_val)} {p.val_max:+.2f}", (p.L_val, p.val_max), textcoords="offset points", xytext=(6, 4),
                    fontsize=8.5, color=INK)
    if "resid" in run.kinds:
        c = curves[curves.kind == "resid"]
        ax.plot(c.layer, c["mean"], color=COL["resid"], linewidth=1.5, linestyle="--", label=X.KIND_LABEL["resid"])
    ax.set_xlabel("layer", color=INK2)
    ax.set_ylabel("rescue (validation mean, 95% CI)", color=INK2)
    ax.set_title(f"{run.cfg['label']}: rescue by patched component", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax)
    ax = axes[1]
    ax.plot(add.layer, add["sum"], color=COL["sum"], linewidth=1.5, linestyle="--", label="attention + MoE (sum of means)")
    ax.plot(add.layer, add["block"], color=COL["block"], linewidth=2, label="block (both patched together)")
    ax.fill_between(add.layer, add["block"] - add["sum"], 0, color=COL["block"], alpha=0.12, linewidth=0, label="block − sum")
    ax.set_xlabel("layer", color=INK2)
    ax.set_ylabel("rescue (validation mean)", color=INK2)
    ax.set_title("additivity: block vs attention + MoE", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=160)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def figure_protocols(ra: X.Run, rb: X.Run, ca: pd.DataFrame, cb: pd.DataFrame, path: str):
    kinds = [k for k in X.ALL_KINDS if k in ra.kinds and k in rb.kinds]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.4 * len(kinds), 3.8), sharey=False)
    for ax, k in zip(np.atleast_1d(axes), kinds):
        for r, c, col, lab in ((ra, ca, "#2a78d6", "BOS (tokenizer default)"), (rb, cb, "#eb6834", "no BOS (paper protocol)")):
            cc = c[c.kind == k]
            ax.fill_between(cc.layer, cc.ci_lo, cc.ci_hi, color=col, alpha=0.15, linewidth=0)
            ax.plot(cc.layer, cc["mean"], color=col, linewidth=2, label=lab)
        ax.set_title(X.KIND_LABEL[k], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("layer", color=INK2)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("rescue (validation mean, 95% CI)", color=INK2)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(f"{MODELS[ra.cfg['model']]['label']}: BOS vs no-BOS tokenisation", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=160)
    fig.savefig(path + ".pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def analyze_run(run_name: str) -> dict:
    run = X.Run(run_name)
    sh = run.cfg["short"]
    log(f"{run_name}: {len(run.disc)} disc / {len(run.val)} val cases, {run.L} layers, kinds {run.kinds}")
    kinds = [k for k in X.ALL_KINDS if k in run.kinds]
    cv = X.all_curves(run, run.val, kinds)
    cd = X.all_curves(run, run.disc, kinds)
    cv["split"], cd["split"] = "validation", "discovery"
    inc = X.resid_increments(cv) if "resid" in kinds else None
    curves = pd.concat([cv, cd], ignore_index=True)
    if inc is not None:
        curves = curves.merge(inc.rename(columns={"resid": "_r"})[["layer", "increment"]], on="layer", how="left")
        curves.loc[curves.kind != "resid", "increment"] = np.nan
    curves.to_csv(os.path.join(TAB, f"ext2_attn_layer_curves_{sh}.csv"), index=False)
    pk = X.peaks(run, kinds)
    rows = [[X.KIND_LABEL[r.kind], f"L{r.L_disc}", f3(r.disc_mean), ci(r.val_at_L_disc, r.val_at_L_disc_lo, r.val_at_L_disc_hi),
             f"{100 * r.val_pos_frac_at_L_disc:.0f}%", f"L{r.L_val}", ci(r.val_max, r.val_max_lo, r.val_max_hi),
             f"{r.auc_pos:.2f} [{r.auc_lo:.2f}, {r.auc_hi:.2f}]", f"{r.com:.1f}"] for r in pk.itertuples()]
    md_table(["Patched component", "L* (disc.)", "Disc. mean at L*", "Val. rescue at L* [95% CI]", "Val. pos. frac.",
              "Val. argmax", "Val. max [95% CI]", "AUC+ (val.) [CI]", "Centre of mass (val.)"], rows,
             os.path.join(TAB, f"ext2_attn_peaks_{sh}"),
             f"{run.cfg['label']}: peaks of the rescue curves per patched component ({run.case_set} set, "
             f"{len(run.disc)} discovery / {len(run.val)} validation cases)")
    L_moe = int(pk[pk.kind == "layer"].L_disc.iloc[0])
    L_attn = int(pk[pk.kind == "attn_layer"].L_disc.iloc[0])
    L_block = int(pk[pk.kind == "block"].L_disc.iloc[0])
    share = {"at_moe_peak": X.share_at_layer(run, L_moe, run.val), "at_attn_peak": X.share_at_layer(run, L_attn, run.val),
             "at_block_peak": X.share_at_layer(run, L_block, run.val), "overall": X.share_overall(run, run.val)}
    rows = []
    for key, lab in (("at_moe_peak", "at the MoE-peak layer"), ("at_attn_peak", "at the attention-peak layer"), ("at_block_peak", "at the block-peak layer")):
        s = share[key]
        rows.append([lab, f"L{s['layer']}", f3(s["attn_mean"]), f3(s["moe_mean"]), ci(s["share"], s["share_lo"], s["share_hi"])])
    s = share["overall"]
    rows.append(["overall (AUC+ of the validation curves)", "all", f"{s['auc_attn']:.2f}", f"{s['auc_moe']:.2f}", ci(s["share"], s["share_lo"], s["share_hi"])])
    md_table(["Where", "Layer", "Attention rescue", "MoE rescue", "Attention share attn/(attn+moe) [95% CI]"], rows,
             os.path.join(TAB, f"ext2_attn_share_{sh}"), f"{run.cfg['label']}: attention share of the rescue (validation)")
    add = X.additivity(run, run.val)
    add.to_csv(os.path.join(TAB, f"ext2_attn_additivity_{sh}_all_layers.csv"), index=False)
    top = add.reindex(add.block.abs().sort_values(ascending=False).index).head(8).sort_values("layer")
    rows = [[f"L{int(r.layer)}", f3(r.attn), f3(r.moe), f3(r["sum"]), f3(r.block), ci(r.gap, r.gap_lo, r.gap_hi), f"{r.r_case:.2f}",
             f"{100 * r.frac_block_gt_sum:.0f}%"] for _, r in top.iterrows()]
    md_table(["Layer", "Attention", "MoE", "Attn + MoE", "Block", "Block − (attn + MoE) [95% CI]", "Per-case r(block, attn+MoE)",
              "Cases block > sum"], rows, os.path.join(TAB, f"ext2_attn_additivity_{sh}"),
             f"{run.cfg['label']}: additivity at the 8 layers with the largest |block| rescue (validation means)")
    figure_run(run, cv, pk, add, os.path.join(FIG, f"ext2_attn_curves_{sh}"))
    # summary
    a_peak = add[add.layer == L_block].iloc[0]
    pooled = np.corrcoef(run.mat("block", run.val).ravel(), (run.mat("attn_layer", run.val) + run.mat("layer", run.val)).ravel())[0, 1]
    out = {"run": run_name, "short": sh, "label": run.cfg["label"], "model": run.cfg["model"], "protocol": run.cfg["protocol"],
           "case_set": run.case_set, "n_disc": len(run.disc), "n_val": len(run.val), "n_layers": run.L, "kinds": kinds,
           "meta": {k: run.meta.get(k) for k in ("pass_time_s", "spawn_rows", "prefill_rows", "completed_utc")},
           "peaks": pk.to_dict("records"), "share": share,
           "additivity": {"at_block_peak": a_peak.to_dict(), "pooled_case_layer_r": float(pooled),
                          "layers_block_gt_sum": int((add.gap > 0).sum()), "mean_gap_all_layers": float(add.gap.mean()),
                          "max_abs_gap": float(add.gap.abs().max()), "L_max_abs_gap": int(add.layer[add.gap.abs().idxmax()])},
           "curves_val": {k: cv[cv.kind == k]["mean"].round(3).tolist() for k in kinds},
           "resid_increments": None if inc is None else inc.increment.round(3).tolist(),
           "delta_clean_val_mean": float(run.delta_clean.loc[run.val].mean()), "delta_noised_val_mean": float(run.delta_noised.loc[run.val].mean())}
    if inc is not None:
        j = int(np.argmax(inc.increment.to_numpy()))
        out["resid_max_increment"] = {"layer": int(inc.layer.iloc[j]), "increment": float(inc.increment.iloc[j])}
        out["resid_half_layer"] = int(inc.layer[(inc.resid >= 0.5 * inc.resid.max()).idxmax()])
    for r in pk.itertuples():
        log(f"  {r.kind:10s} L*disc={r.L_disc:2d} val@L*={r.val_at_L_disc:+.3f} [{r.val_at_L_disc_lo:+.3f},{r.val_at_L_disc_hi:+.3f}]"
            f"  val argmax L{r.L_val} {r.val_max:+.3f}  AUC+ {r.auc_pos:.2f}  CoM {r.com:.1f}")
    log(f"  share at MoE peak L{L_moe}: {share['at_moe_peak']['share']:+.3f}; overall {share['overall']['share']:+.3f}; "
        f"block-sum gap at block peak L{L_block}: {a_peak.gap:+.3f}, r={a_peak.r_case:.2f}")
    return out, run, cv


# ---------------------------------------------------------------------------------------------------------------
def write_section(results: list[dict], proto: dict | None, verify: dict | None):
    P = []
    P.append("## Extension 2b: attention-sublayer versus MoE-sublayer patching\n")
    P.append("**Question.** The paper's causal tracing patches only the MoE-block output at the final position. How much of the "
             "factual-recall rescue at each layer sits in the attention sublayer (which moves information from the subject tokens "
             "to the final position) versus the MoE sublayer (which the paper reads as the retrieval site)?\n")
    P.append("**Method.** Three new intervention kinds in `moetrace/engine.py` (the previous kinds and their numerics are unchanged; "
             "`results/verify_olmoe.json` is bit-identical to `results/verify_olmoe_before_ext2.json` on every non-timing metric). "
             "Writing the decoder layer as h_attn = h_pre + Attn_l(h_pre), h_out = h_attn + MoE_l(h_attn), all at the final position of "
             "the noised run unless marked clean: `attn_layer` starts a wavefront row *before* the MoE of layer l as h_pre_noised + "
             "Attn_l^clean, so the MoE of that layer recomputes (with its router) on the patched residual; `layer` (the paper's patch) "
             "replaces the MoE output; `block` replaces both sublayer outputs of layer l, h = (h_pre_noised + Attn_l^clean) + "
             "MoE_l^clean; `resid` restores the clean residual h_out^clean after layer l (classic hidden-state causal tracing, which "
             "additionally carries the upstream difference h_pre^clean − h_pre^noised and is therefore cumulative). Rescue = "
             "Δ_patched − Δ_noised as in Table 1. One pass per run with clean, noised and all four kinds at every layer on the paper's "
             "256 cases (`scripts/ext2_attn_sweep.py`; rows in `results/<run>/sweep_rows.parquet` with the fp32 norm of the patched "
             "vector in `vnorm`). Layers are selected on the discovery split and evaluated on validation (5,000-resample bootstrap "
             "CIs); the attention share is attn/(attn + moe) as a ratio of validation means with a paired case bootstrap, overall as the "
             "ratio of the areas under the positive parts of the validation curves (AUC+). Additivity compares `block` with "
             "`attn_layer` + `layer` per layer (mean gap with CI, per-case Pearson r). Code: `moetrace/ext2_attn.py`, "
             "`scripts/ext2_attn_analyze.py`.\n")
    if verify:
        v = verify
        P.append("**Verification (OLMoE-1B-7B-0125, 20 cases x 16 layers, `scripts/ext2_attn_verify.py`, "
                 "`results/verify_ext2_attn_olmoe.json`).** Against transformers forward hooks (self_attn output, MoE output, decoder-layer "
                 "output replaced at the final position): per-case |Δ_engine − Δ_HF| mean "
                 f"{v['attn_layer_vs_hf_meanabsdiff']:.3f} / {v['block_vs_hf_meanabsdiff']:.3f} / {v['resid_vs_hf_meanabsdiff']:.3f} "
                 f"(max {v['attn_layer_vs_hf_maxdiff']:.2f} / {v['block_vs_hf_maxdiff']:.2f} / {v['resid_vs_hf_maxdiff']:.2f}) for "
                 f"attn_layer / block / resid against {v['layer_vs_hf_meanabsdiff']:.3f} (max {v['layer_vs_hf_maxdiff']:.2f}) for the "
                 f"already-verified `layer` kind on the same rows, i.e. the same bf16 noise floor; per-case rescue correlation with HF "
                 f"{v['attn_layer_rescue_corr_vs_hf']:.3f} / {v['block_rescue_corr_vs_hf']:.3f} / {v['resid_rescue_corr_vs_hf']:.3f} "
                 f"({v['layer_rescue_corr_vs_hf']:.3f} for `layer`); 20-case mean curves agree within "
                 f"{v['attn_layer_mean_curve_maxdiff_vs_hf']:.3f} / {v['block_mean_curve_maxdiff_vs_hf']:.3f} / "
                 f"{v['resid_mean_curve_maxdiff_vs_hf']:.3f} ({v['layer_mean_curve_maxdiff_vs_hf']:.3f} for `layer`). Invariants: (a) each "
                 f"kind spawned on the clean run with itself as donor reproduces the clean logits (max |Δ| {v['identity_attn_layer_maxdiff']:.3f}, "
                 f"mean {v['identity_attn_layer_meanabsdiff']:.3f}; `zero` on the clean run in verify_olmoe: 0.125); (b) `block` equals the "
                 f"difference form h_noised + (Attn^clean − Attn^noised) + (MoE^clean − MoE^noised) (`block_diff`) to bf16 rounding: mean "
                 f"|Δ| {v['block_vs_block_diff_meanabsdiff']:.3f}, {100 * v['block_vs_block_diff_frac_within_0.1']:.0f}% within 0.1, max "
                 f"{v['block_vs_block_diff_maxdiff']:.2f}; (c) the recorded norms of the patched vectors match the HF norms of the "
                 f"final-position differences (mean within 0.3%).\n")
    for r in results:
        sh = r["short"]
        P.append(f"### {r['label']} (`results/{r['run']}`)\n")
        pk = {p["kind"]: p for p in r["peaks"]}
        a, m, b = pk["attn_layer"], pk["layer"], pk["block"]
        sm, so = r["share"]["at_moe_peak"], r["share"]["overall"]
        ad = r["additivity"]
        P.append(f"- Peaks (discovery argmax, validation value): attention output L{a['L_disc']} "
                 f"{ci(a['val_at_L_disc'], a['val_at_L_disc_lo'], a['val_at_L_disc_hi'])}; MoE output L{m['L_disc']} "
                 f"{ci(m['val_at_L_disc'], m['val_at_L_disc_lo'], m['val_at_L_disc_hi'])}; block L{b['L_disc']} "
                 f"{ci(b['val_at_L_disc'], b['val_at_L_disc_lo'], b['val_at_L_disc_hi'])}. Validation argmax: L{a['L_val']} / L{m['L_val']} / "
                 f"L{b['L_val']}. Centre of mass of the positive part: {a['com']:.1f} / {m['com']:.1f} / {b['com']:.1f}.")
        P.append(f"- Areas under the positive part (validation): attention {a['auc_pos']:.2f} [{a['auc_lo']:.2f}, {a['auc_hi']:.2f}], "
                 f"MoE {m['auc_pos']:.2f} [{m['auc_lo']:.2f}, {m['auc_hi']:.2f}], block {b['auc_pos']:.2f} [{b['auc_lo']:.2f}, {b['auc_hi']:.2f}]. "
                 f"Attention share attn/(attn+moe): {ci(sm['share'], sm['share_lo'], sm['share_hi'])} at the MoE-peak layer L{sm['layer']} "
                 f"(attention {f3(sm['attn_mean'])}, MoE {f3(sm['moe_mean'])}), {ci(so['share'], so['share_lo'], so['share_hi'])} overall.")
        ap = ad["at_block_peak"]
        P.append(f"- Additivity: at the block-peak layer L{int(ap['layer'])} block {f3(ap['block'])} vs attention + MoE {f3(ap['sum'])} "
                 f"(gap {ci(ap['gap'], ap['gap_lo'], ap['gap_hi'])}, per-case r = {ap['r_case']:.2f}, block exceeds the sum in "
                 f"{100 * ap['frac_block_gt_sum']:.0f}% of cases); across all layers the mean gap is {f3(ad['mean_gap_all_layers'])} "
                 f"(largest |gap| {ad['max_abs_gap']:.3f} at L{ad['L_max_abs_gap']}), block > sum in {ad['layers_block_gt_sum']}/{r['n_layers']} "
                 f"layers, pooled per-(case, layer) r = {ad['pooled_case_layer_r']:.2f}.")
        if "resid" in pk:
            rs = pk["resid"]
            P.append(f"- Residual restoration (cumulative): reaches half its maximum at L{r['resid_half_layer']}, maximum "
                     f"{ci(rs['val_max'], rs['val_max_lo'], rs['val_max_hi'])} at L{rs['L_val']}; largest layer-to-layer increment "
                     f"{f3(r['resid_max_increment']['increment'])} at L{r['resid_max_increment']['layer']}. Noise drop on validation: "
                     f"Δ_clean {r['delta_clean_val_mean']:+.2f} → Δ_noised {r['delta_noised_val_mean']:+.2f}.")
        P.append("")
        P.append(f"![ext2 attention curves {sh}](../figures/ext2_attn_curves_{sh}.png)\n")
        P.append(f"Figure E2b-{sh}: left, validation mean rescue by layer when the final position's attention output (blue), MoE output "
                 f"(orange) or both (green) are replaced by the clean run's, with 95% bootstrap bands; dashed grey = clean residual restored "
                 f"after the layer (cumulative). Right, block versus the sum of the two single-sublayer rescues. Tables: "
                 f"`results/tables/ext2_attn_peaks_{sh}.md`, `ext2_attn_share_{sh}.md`, `ext2_attn_additivity_{sh}.md`, per-layer curves "
                 f"`ext2_attn_layer_curves_{sh}.csv`.\n")
        for t in ("peaks", "share", "additivity"):
            with open(os.path.join(TAB, f"ext2_attn_{t}_{sh}.md")) as f:
                P.append(f.read())
    if proto:
        P.append("### Mixtral: BOS versus no-BOS tokenisation\n")
        P.append(proto["text"])
        P.append("![ext2 Mixtral BOS vs no BOS](../figures/ext2_attn_curves_mixtral_bos_vs_nobos.png)\n")
        P.append("Figure E2b-protocol: the same 256 paper cases tokenised with and without `<s>`; one panel per patched component.\n")
        P.append(proto["table"])
    # data-driven reading
    P.append("### Reading\n")
    for r in results:
        pk = {p["kind"]: p for p in r["peaks"]}
        a, m, b = pk["attn_layer"], pk["layer"], pk["block"]
        so, sm = r["share"]["overall"], r["share"]["at_moe_peak"]
        ad = r["additivity"]
        earlier = "earlier than" if a["com"] < m["com"] - 0.5 else ("later than" if a["com"] > m["com"] + 0.5 else "at about the same depth as")
        P.append(f"- **{r['label']}.** The attention-output rescue is centred {earlier} the MoE-output rescue (centre of mass "
                 f"{a['com']:.1f} vs {m['com']:.1f}; discovery peaks L{a['L_disc']} vs L{m['L_disc']}). Over all layers attention carries "
                 f"{100 * so['share']:.0f}% [{100 * so['share_lo']:.0f}%, {100 * so['share_hi']:.0f}%] of the positive rescue area; at the "
                 f"MoE-peak layer L{sm['layer']} the attention patch alone gives {f3(sm['attn_mean'])} against {f3(sm['moe_mean'])} for "
                 f"the MoE patch. Patching both sublayers of one layer ({f3(b['val_max'])} at L{b['L_val']}) "
                 + ("exceeds" if ad["at_block_peak"]["gap"] > 0 else "falls short of")
                 + f" the sum of the two single-sublayer rescues by {abs(ad['at_block_peak']['gap']):.3f} at the block peak; "
                 f"the per-case correlation between block and sum is {ad['at_block_peak']['r_case']:.2f}.")
    P.append("")
    interp = os.path.join(SEC, "ext2_attn_interpretation.md")
    if os.path.exists(interp):
        with open(interp) as f:
            P.append(f.read())
    with open(os.path.join(SEC, "ext2_attn_patch.md"), "w") as f:
        f.write("\n".join(P))


# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    args = ap.parse_args()
    results, runs, curves = [], {}, {}
    for r in args.runs.split(","):
        if not os.path.exists(os.path.join(RESULTS, r, "sweep_rows.parquet")):
            log(f"{r}: no sweep_rows.parquet, skipped")
            continue
        out, run, cv = analyze_run(r)
        results.append(out)
        runs[r], curves[r] = run, cv
    # cross-run summary table
    rows = []
    for r in results:
        for p in r["peaks"]:
            rows.append([r["label"], X.KIND_LABEL[p["kind"]], f"L{p['L_disc']}", ci(p["val_at_L_disc"], p["val_at_L_disc_lo"], p["val_at_L_disc_hi"]),
                         f"L{p['L_val']}", ci(p["val_max"], p["val_max_lo"], p["val_max_hi"]), f"{p['auc_pos']:.2f}", f"{p['com']:.1f}"])
    md_table(["Model / protocol", "Patched component", "L* (disc.)", "Val. rescue at L* [CI]", "Val. argmax", "Val. max [CI]", "AUC+", "CoM"],
             rows, os.path.join(TAB, "ext2_attn_summary"), "Attention vs MoE attribution: peaks per model and patched component")
    rows = []
    for r in results:
        sm, so, ad = r["share"]["at_moe_peak"], r["share"]["overall"], r["additivity"]["at_block_peak"]
        rows.append([r["label"], f"L{sm['layer']}", ci(sm["share"], sm["share_lo"], sm["share_hi"]), ci(so["share"], so["share_lo"], so["share_hi"]),
                     f"L{int(ad['layer'])}", f3(ad["block"]), f3(ad["sum"]), ci(ad["gap"], ad["gap_lo"], ad["gap_hi"]), f"{ad['r_case']:.2f}"])
    md_table(["Model / protocol", "MoE-peak layer", "Attention share there [CI]", "Overall share (AUC+) [CI]", "Block peak", "Block", "Attn + MoE",
              "Gap [CI]", "Per-case r"], rows, os.path.join(TAB, "ext2_attn_shares"), "Attention share and additivity per model")
    proto = None
    if "mixtral_bos_attnsweep" in runs and "mixtral_nobos_attnsweep" in runs:
        ra, rb = runs["mixtral_bos_attnsweep"], runs["mixtral_nobos_attnsweep"]
        cmp = X.compare_runs(ra, rb)
        rows = [[X.KIND_LABEL[r.kind], f"L{r.L_a} {f3(r.max_a)}", f"L{r.L_b} {f3(r.max_b)}", f"{r.curve_r:.3f}", f"{r.max_absdiff:.3f} (L{r.L_max_absdiff})",
                 ci(r.paired_diff_at_L_a, r.paired_lo, r.paired_hi)] for r in cmp.itertuples()]
        table = md_table(["Patched component", "BOS: val. argmax, max", "no BOS: val. argmax, max", "Curve correlation (layers)",
                          "Max |BOS − noBOS| (layer)", "Paired BOS − noBOS at BOS argmax [CI]"], rows,
                         os.path.join(TAB, "ext2_attn_protocol_mixtral"), "Mixtral-8x7B-v0.1: attention / MoE / block curves with and without BOS")
        figure_protocols(ra, rb, curves["mixtral_bos_attnsweep"], curves["mixtral_nobos_attnsweep"], os.path.join(FIG, "ext2_attn_curves_mixtral_bos_vs_nobos"))
        ka = cmp.set_index("kind")
        text = (f"The layer curves of the two protocols are highly correlated across layers (r = {ka.loc['attn_layer', 'curve_r']:.2f} attention, "
                f"{ka.loc['layer', 'curve_r']:.2f} MoE, {ka.loc['block', 'curve_r']:.2f} block); the validation argmax layers are "
                f"L{ka.loc['attn_layer', 'L_a']}/L{ka.loc['attn_layer', 'L_b']} (attention), L{ka.loc['layer', 'L_a']}/L{ka.loc['layer', 'L_b']} (MoE) and "
                f"L{ka.loc['block', 'L_a']}/L{ka.loc['block', 'L_b']} (block) with/without BOS. The largest protocol difference is "
                f"{ka.loc['layer', 'max_absdiff']:.3f} at L{ka.loc['layer', 'L_max_absdiff']} for the MoE curve and {ka.loc['attn_layer', 'max_absdiff']:.3f} at "
                f"L{ka.loc['attn_layer', 'L_max_absdiff']} for the attention curve; paired per-case differences at the BOS argmax: MoE "
                f"{ci(ka.loc['layer', 'paired_diff_at_L_a'], ka.loc['layer', 'paired_lo'], ka.loc['layer', 'paired_hi'])}, attention "
                f"{ci(ka.loc['attn_layer', 'paired_diff_at_L_a'], ka.loc['attn_layer', 'paired_lo'], ka.loc['attn_layer', 'paired_hi'])}.\n")
        proto = {"table": table, "text": text, "rows": cmp.to_dict("records")}
    verify = None
    vp = os.path.join(RESULTS, "verify_ext2_attn_olmoe.json")
    if os.path.exists(vp):
        verify = json.load(open(vp))
    write_section(results, proto, verify)
    with open(os.path.join(RESULTS, "ext2_attn_summary.json"), "w") as f:
        json.dump(jsonable({"runs": results, "protocol_mixtral": proto["rows"] if proto else None,
                            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
