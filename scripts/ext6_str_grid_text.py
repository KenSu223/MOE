"""ext6: write results/sections/ext6_str_grid.md from results/ext6_str_grid_summary.json and the ext6_str_grid tables.

Usage: python scripts/ext6_str_grid_text.py
"""
import json, os

ROOT = "/home/ubuntu/MOE/results"
S = json.load(open(os.path.join(ROOT, "ext6_str_grid_summary.json")))
V = json.load(open(os.path.join(ROOT, "verify_ext6_str_grid_olmoe.json")))
RUNS = [r for r in ("qwen3_str", "mixtral_bos_str", "mixtral_nobos_str") if r in S]
SH = {"qwen3_str": "Qwen3", "mixtral_bos_str": "Mixtral BOS", "mixtral_nobos_str": "Mixtral no BOS"}
HAS5 = [r for r in RUNS if "w5" in S[r]]


def tab(name):
    p = os.path.join(ROOT, "tables", f"ext6_str_grid_{name}.md")
    return open(p).read().strip() if os.path.exists(p) else ""


def pk(run, cat, metric="ld", w=1):
    return next(p for p in S[run][f"w{w}"][f"peaks_{metric}"] if p["cat"] == cat)


def ci(p, d=3):
    return f"{p['peak']:+.{d}f} [{p['lo']:+.{d}f}, {p['hi']:+.{d}f}]"


def last_above(run, cat, thr=0.05):
    c = pk(run, cat)["curve"]
    ls = [i for i, x in enumerate(c) if x > thr]
    return max(ls) if ls else None


def sva(run, metric):
    """Range of the sliding / adding peak ratio over the last subject token and the final position."""
    sv = {x["cat"]: x["ratio"] for x in S[run].get(f"sliding_vs_adding_{metric}", [])}
    a, b = sv.get("last subject token"), sv.get("last token")
    return f"{min(a, b):.2f}–{max(a, b):.2f}" if a is not None and b is not None else "n/a"


def bpr(run, metric, w=1):
    x = S[run][f"w{w}"][f"bp_ratio_{metric}"]
    return f"{x['ratio']:.2f}× [{x['lo']:.2f}, {x['hi']:.2f}]"


out = []
w = out.append

# ---------------------------------------------------------------- summary
last = {r: pk(r, "last subject token") for r in RUNS}
fin = {r: pk(r, "last token") for r in RUNS}
mid = {r: pk(r, "middle subject tokens") for r in RUNS}
between = {r: max(pk(r, "first subsequent token")["peak"], pk(r, "further tokens")["peak"]) for r in RUNS}
gn = {r: S[r].get("gn_last_subject") for r in RUNS}
w("**Summary.** Zhang & Nanda (2024, Section 4.1 / Figure 4) extend the MLP patch from the last subject token to every "
  "position and plot layer × position heatmaps under STR, to test whether the last subject token is special and whether logit "
  "difference and probability agree. We ran the same grid for the paper's MoE-output patch with the STR corruption of Direction 6 "
  f"(every selected donor; {', '.join(f'{SH[r]} {S[r]['w1']['n_cases']} cases' for r in RUNS)}), single-layer patches at every "
  "position from the first subject token to the final token and every layer"
  + (", and the 5-layer sliding window of their Figure 4" if HAS5 else "") + ". The picture is the two-site pattern of Meng et al.: "
  + "; ".join(f"{SH[r]} early site at the last subject token (peak L{last[r]['peak_layer']} {last[r]['peak']:+.3f} of the drop, "
              f"{last[r]['sum_over_layers']:.2f} drops summed over layers) and late site at the final position (peak L{fin[r]['peak_layer']} "
              f"{fin[r]['peak']:+.3f}, the paper's layer band)" for r in RUNS)
  + f"; the tokens between the subject and the final position carry almost nothing (≤ {max(between.values()):.3f}). Among subject "
  "tokens the last one dominates (Zhang & Nanda's last / middle ratio: "
  + "; ".join(f"{SH[r]} logit difference {bpr(r, 'ld')}, probability {bpr(r, 'dp')}" for r in RUNS)
  + "); with single-layer patches the two metrics draw the same map, unlike GPT-2 XL (1.22× vs 4.33×)"
  + (". With the 5-layer window of their Figure 4, probability does emphasise the last subject token more ("
     + "; ".join(f"{SH[r]} logit difference {bpr(r, 'ld', 5)}, probability {bpr(r, 'dp', 5)}" for r in HAS5)
     + ") and the window is super-additive only on the probability scale (joint / summed single-layer peak "
     + "; ".join(f"{SH[r]} {sva(r, 'ld')} on the logit difference vs {sva(r, 'dp')} on Δp" for r in HAS5)
     + "): Zhang & Nanda's two GPT-2 XL observations reappear exactly where a window and the probability metric are combined"
     if HAS5 else "")
  + ". The early site is where the corruption matters: "
  "compared with the GN runs of F4 at the same position and cases, "
  + "; ".join(f"{SH[r]} GN peaks at L{gn[r]['gn_peak_layer']} instead of L{gn[r]['str_peak_layer']} and its layer-summed rescue is "
              f"{gn[r]['gn_sum'] / gn[r]['str_sum']:.2f}× STR's" for r in RUNS if gn[r])
  + " — Zhang & Nanda's GN inflation, which at the paper's final-position site (Direction 6) was absent or small.\n")

# ---------------------------------------------------------------- method
w("### What was run\n")
w("For every retained STR case (Direction 6: paper IDs and split, donors = same-relation subjects whose true object is the foil, up "
  "to five per case) and every position p from the first subject token to the final token, a clean prefill row and one row per "
  "donor record at p; per donor row and layer l one suffix wavefront row (`moetrace/ext5_subject.py`) carries the MoE output at "
  "p set to the clean one (window 1) or at layers l−2..l+2 clipped to the network (window 5, centred, the setting of Zhang & "
  "Nanda's Figure 4 and Meng et al.'s causal-tracing plots), and the model runs on from there. Positions before the first "
  "subject token are not patched: clean and donor prompts share that prefix, so the patch is exactly zero. Metrics at the final "
  "position: the paper's Δ (its rescue normalised by the drop Δ_clean − Δ_corrupt, the normalised logit difference of Zhang & "
  "Nanda) and Δp = p_patched(true) − p_corrupt(true) (their probability metric; full softmax). Aggregation as in ROME: donor mean "
  "per (case, position, layer), then mean over the positions of each token group of a case, then mean over the cases that have "
  "the group (one-token subjects have no first/middle token; the first subsequent token counts as \"last token\" when it is the "
  "final position).\n")
w("Engine additions (backward compatible, in `moetrace/ext5_subject.py`): `SubjectSpawn.window` (joint restoration of the MoE "
  "output at p over consecutive layers) and `run_subject(metrics=True)`. The F4 verification re-run gives results identical to the "
  "saved copy in all 70 non-timing fields. New verification against transformers hooks on OLMoE "
  f"(`results/verify_ext6_str_grid_olmoe.json`, {V['n_pairs']} STR pairs, {V['n_units']} (case, position) units, all layers): "
  f"window 1 max |ΔΔ| {V['w1_delta_maxdiff']:.2f}, mean {V['w1_delta_meanabsdiff']:.3f}, {100 * V['w1_delta_frac_within_0.25']:.0f} % "
  f"within 0.25, rescue r = {V['w1_rescue_corr_vs_hf']:.3f}; window 5 max {V['w5_delta_maxdiff']:.2f}, r = {V['w5_rescue_corr_vs_hf']:.3f}; "
  f"p(true) mean |diff| {V['w1_p_true_meanabsdiff']:.4f} / {V['w5_p_true_meanabsdiff']:.4f}; null invariant {V['zero_on_donor_maxdiff']:.3f} "
  f"(the GN patch at p in F4: max {V['calibration_ext5_layer_at_p_gn']['maxdiff']:.2f}, r = {V['calibration_ext5_layer_at_p_gn']['rescue_corr']:.3f}). "
  "Consistency: the grid's last-token column reproduces the final-position STR sweep of Direction 6 "
  + "; ".join(f"({SH[r]}: curve r = {S[r]['sweep_consistency_w1']['curve_corr']:.4f}, max |diff| {S[r]['sweep_consistency_w1']['curve_maxabsdiff']:.3f}, "
              f"per-case r = {S[r]['sweep_consistency_w1']['percase_corr']:.3f})" for r in RUNS) + ".\n")

# ---------------------------------------------------------------- heatmaps
w("### Heatmaps\n")
for r in RUNS:
    w(f"![{SH[r]} STR layer x token-group heatmaps](figures/ext6_str_grid_{r}.png)\n")
w("![Single-layer layer curves per token group](figures/ext6_str_grid_curves.png)\n")
w(tab("peaks") + "\n")

# ---------------------------------------------------------------- last subject token
w("### Is the last subject token special?\n")
w(tab("ratio") + "\n")
w("Within the subject, yes: " + "; ".join(
    f"in {SH[r]} the last subject token's layer-summed rescue is {last[r]['sum_over_layers']:.2f} drops against "
    f"{mid[r]['sum_over_layers']:.2f} for the middle and {pk(r, 'first subject token')['sum_over_layers']:.2f} for the first subject "
    f"token; above 0.05 of the drop the last subject token stays up to L{last_above(r, 'last subject token')}, the middle tokens up to "
    f"L{last_above(r, 'middle subject tokens')} and the first up to L{last_above(r, 'first subject token')}" for r in RUNS)
  + ". The first and middle subject tokens matter only in the first MoE layers (peak L0), where the MoE output at a subject "
  "position still carries that token's identity — the analogue of the MLP0 effect Zhang & Nanda set aside for GPT-2. Across the "
  "whole prompt, the last subject token and the final position are the two loci; per layer the final-position peak is "
  + ", ".join(f"{fin[r]['peak'] / last[r]['peak']:.2f}× the last-subject-token peak in {SH[r]}" for r in RUNS)
  + ". With single-layer patches, unlike GPT-2 XL (probability 4.33×, logit difference 1.22× more effect on the last than on the "
  "middle subject tokens), both metrics give the same ordering; the probability ratio is noisier because Δp is tiny (clean "
  "p(true) has a median of about 0.04, Direction 5-F5)."
  + (" With the 5-layer window, Zhang & Nanda's setting, the probability ratio rises above the logit-difference ratio ("
     + "; ".join(f"{SH[r]} {bpr(r, 'dp', 5)} vs {bpr(r, 'ld', 5)}" for r in HAS5)
     + "): a joint patch at the last subject token moves the logit difference far enough to reach the steep part of the softmax, "
     "a middle-token patch does not." if HAS5 else "") + "\n")

# ---------------------------------------------------------------- GN vs STR
if any(gn.values()):
    w("### STR vs GN at the last subject token\n")
    w(tab("gn_subject") + "\n")
    w("![STR vs GN at the last subject token](figures/ext6_str_grid_gn_subject.png)\n")
    w("Normalised by their own drops, " + "; ".join(
        f"{SH[r]}: GN peaks at L{gn[r]['gn_peak_layer']} ({gn[r]['gn_peak'][0]:+.3f}), STR at L{gn[r]['str_peak_layer']} "
        f"({gn[r]['str_peak'][0]:+.3f}); layer sums {gn[r]['gn_sum']:.2f} vs {gn[r]['str_sum']:.2f}" for r in RUNS if gn[r])
      + ". At L0 the patch rescues " + "; ".join(f"{gn[r]['str_curve'][0]:+.3f} under STR vs {gn[r]['gn_curve'][0]:+.3f} under GN in {SH[r]}"
                                               for r in RUNS if gn[r])
      + ". GN noises the subject embeddings, so the residual at the subject keeps the off-distribution embedding and restoring "
      "the first MoE output repairs less, with the repair peaking a few layers later; STR swaps in another real subject, so the "
      "first MoE layer's output already restores much of the token identity. Layer-summed GN / STR: "
      + "; ".join(f"{SH[r]} {gn[r]['gn_sum'] / gn[r]['str_sum']:.2f}×" for r in RUNS if gn[r])
      + ". Where this ratio is well above 1 it is the inflation Zhang & Nanda report for GPT-2 XL (GN peaks 2–5× STR's); near 1 the "
      "total is unchanged and only the layer profile shifts.\n")

# ---------------------------------------------------------------- window
if HAS5:
    w("### Sliding window vs single layer\n")
    w(tab("sliding") + "\n")
    rows = []
    for r in HAS5:
        sv = {x["cat"]: x for x in S[r].get("sliding_vs_adding_ld", [])}
        l5, f5 = pk(r, "last subject token", w=5), pk(r, "last token", w=5)
        rows.append(f"{SH[r]}: window-5 peaks last subject token L{l5['peak_layer']} {l5['peak']:+.3f}, final position "
                    f"L{f5['peak_layer']} {f5['peak']:+.3f}; joint / summed single-layer peak "
                    f"{sv['last subject token']['ratio']:.2f} (last subject token) and {sv['last token']['ratio']:.2f} (final position)")
    w("- " + "\n- ".join(rows) + "\n")
    w("A ratio below 1 means the five layers restore overlapping information (the joint patch is smaller than the sum of its "
      "parts), above 1 that they act together (Zhang & Nanda's GPT-2 XL: 1.40–1.75 with probability as the metric; they suspect "
      "non-linear effects). On the logit-difference scale the window is additive or sub-additive here ("
      + "; ".join(f"{SH[r]} {sva(r, 'ld')}" for r in HAS5)
      + ": adjacent layers at the subject restore overlapping information, as the cross-layer redundancy of Direction 5-F1 "
      "suggested), so sliding windows do not inflate the localisation in log-odds. On the probability scale the same windows are "
      "strongly super-additive (" + "; ".join(f"{SH[r]} {sva(r, 'dp')}" for r in HAS5)
      + "): p(true) is a convex function of the log-odds near p ≈ 0, so five small single-layer moves add up to much less than one "
      "large joint move. The non-linearity Zhang & Nanda suspected is, in these models, the softmax of the metric rather than the "
      "network. The window heatmaps are smoothed versions of the single-layer ones (same two sites), so the single-layer grid is "
      "the one to quote for layer-level claims.\n")

# ---------------------------------------------------------------- reading
w("### Reading\n")
w("1. The paper's choice of patch site (the final position) sees only the late site. Under the best-practice corruption the "
  "per-layer peak at the last subject token relative to the final-position peak is "
  + ", ".join(f"{last[r]['peak'] / fin[r]['peak']:.2f}× in {SH[r]}" for r in RUNS)
  + ", and the early site is spread over several layers instead of one. This is the STR version of Direction 5-F4, where "
  "patching at the last subject token found the early site under GN and no shared expert there.")
w("2. Zhang & Nanda's warnings, checked one by one in these MoE models. Metric: with single-layer patches logit difference and "
  "probability draw the same map; probability over-weights the last subject token only once windows are used, because joint "
  "patches reach the steep part of the softmax. Window: sliding windows barely inflate the logit difference (joint / summed "
  "singles " + "; ".join(f"{SH[r]} {sva(r, 'ld')}" for r in HAS5) + " at the two sites) but strongly inflate probability. Corruption: GN inflates the early site in Mixtral, changes only its "
  "layer profile in Qwen3, and barely matters at the final position, which is where the paper's claims live (Direction 6). "
  "Following their recommendations (STR, logit difference, single layer first) therefore changes the picture at the subject, "
  "not at the paper's site.")
w("3. Nothing between the subject and the final position matters for the MoE output: relation tokens are not a third site, "
  "consistent with attention moving the subject information to the final position (Directions 2b and 5-F2: mover heads read the "
  "last subject token).\n")
w("Caveats. MoE-output patches only (Meng et al. also plot hidden states and attention; the executor supports `resid` and "
  "`attn_layer` at p, not run here). Token groups have different case counts; means are over the cases that have the group. The "
  "layer-summed values add single-layer effects and over-count shared information (Direction 5-F1); they rank groups, they are not "
  "joint effects. Mixtral without BOS (the paper's protocol) is not in this grid yet.\n")
w("Files: `scripts/ext6_str_grid.py` (grid passes), `scripts/ext6_str_grid_verify.py`, `scripts/ext6_str_grid_analyze.py`, "
  "`scripts/ext6_str_grid_text.py`, `scripts/ext6_str_grid_chain.sh`; rows `results/{qwen3_str,mixtral_bos_str}/str_grid_w{1,5}_rows.parquet` "
  "(donor level: case, position, token group, layer, window bounds, Δ, rescue, p(true), rank) and `str_grid_w*_prefill.parquet`; tables "
  "`results/tables/ext6_str_grid_*`; figures `results/figures/ext6_str_grid_*`; numbers `results/ext6_str_grid_summary.json`.\n")

with open(os.path.join(ROOT, "sections", "ext6_str_grid.md"), "w") as fh:
    fh.write("\n".join(out))
print("wrote results/sections/ext6_str_grid.md")
