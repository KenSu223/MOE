"""Interpretive text for results/sections/ext5_f5_metrics.md (ext5-analysis). Numbers were read from results/ext5_metrics_summary.json
and the ext5_metrics_* tables of 2026-09-21; regenerate the section with scripts/ext5_metrics_analyze.py --section-only after editing."""

INTERPRETATION = {
    "qwen3_metrics": (
        "**Reading.** Nothing the paper selects changes: L44 is the discovery argmax under all six metrics and E069 the recurrence-first expert at "
        "L44 under all six, with a positive Spec under five of them. The second locus of ext1 is reinforced rather than weakened: under Δlog p and "
        "rank L42E115 is the joint top-1 on discovery (validation within E069's CI), under Δp its effect is five times E069's (+0.005 vs +0.001) and "
        "under KL the two tie (+0.108 vs +0.097). What changes is scale and weighting. The clean probability of the true object is small (median "
        "0.039); it is the top-1 token in 75/256 paper cases, and otherwise the model's top-1 is ' the', ' a', ' of' or whitespace in 90% of cases: "
        "the Δ funnel selects prompts on which the model prefers the true object to the counterfactual, not prompts it completes correctly. On the "
        "probability scale the L44 block patch therefore restores 2.6% [1.5, 3.8] of the lost probability mass (mean Δp +0.004 against a mean "
        "p-drop of 0.145) while restoring 17% [15, 19] of the lost log-odds; the 32 validation cases whose top-1 the noise flipped carry 34% of the "
        "Δ rescue but 83% of the Δp rescue, and the per-case correlation between Δ and Δp is 0.11 (Spearman 0.46). Δp is a saturating, top-1-"
        "dominated metric that is underpowered for expert-level contrasts (E069's Spec CI includes zero only under Δp). The rank metric tracks Δ "
        "best per case (r 0.87), the one-sided Δlog p almost as well (0.79); KL agrees on the layer and the expert but weights a different tail "
        "(r 0.45 with Δ; 21% of its rescue from the 10% most disrupted cases, against 15% for Δ)."),
    "mixtral_nobos_metrics": (
        "**Reading.** Under the paper's protocol the metric matters. Δ and Δ/drop select L19 and then E006 (Spec negative, as in the paper); Δp and "
        "rank select L18 and then E001 (Spec positive), the pair ext1's joint search found; Δlog p and KL select L0. The L0 argmax is a heavy-tail "
        "artefact: 66% of the summed KL rescue at L0 (57% of the Δlog p rescue) comes from the 26 cases whose noised distribution is farthest from "
        "the clean one (KL(noised‖clean) ≥ 4.2, true-token rank in the noised run in the hundreds to 16,000s; r(rescue, KL noised) = 0.87), where "
        "restoring the L0 or L1 MoE output of the final token alone brings the whole distribution back (mean KL rescue +4.4 on those cases, +0.26 "
        "on the other 90%). With BOS the L0 KL rescue is +0.001. These are the prompts in which, without a BOS sink, the noised final token itself "
        "collapses into a sink-like state (ext3: 24% of no-BOS prompts); Δ is blind to them because the true and the foil logit fall together "
        "(L0 Δ rescue +0.05; the per-case sink flags of ext3 live in its raw diagnostics and were not joined here). For the selection question the "
        "reading is: every metric that is not dominated by this tail (Δp, rank) or by the foil (Δ) prefers L18E001 to L19E006, and at L19 E006 is "
        "negatively specific under all six metrics (Spec −0.001 to −0.42), so the paper's negative-Spec finding is metric-independent while its "
        "layer choice is not."),
    "mixtral_bos_metrics": (
        "**Reading.** As in Qwen3, nothing selected changes: L19 under all six metrics, E002 at L19 under all six, Spec positive under five "
        "(indeterminate under Δp). L18E001 is the joint top-1 under Δp (+0.006 vs +0.004) and ties E002 under KL (+0.098 vs +0.100), so the two-"
        "locus reading (L19E002 / L18E001) holds on the probability scale too. There is no early-layer tail (L0 KL rescue +0.001) and the noised "
        "distributions are far less degenerate than without BOS (90th percentile of KL(noised‖clean) 2.9 vs 4.2). Normalised rescue at L19: 11% "
        "[9, 13] of the lost log-odds vs 4.7% [3.5, 6.2] of the lost probability mass; the 31 top-1-flipped validation cases carry 41% of the Δ "
        "rescue and 52% of the Δp rescue."),
}

RECOMMENDATION = (
    "The user's decision was to report probability-scale metrics *alongside* Δ. Which ones: (1) the **normalised rescue** mean rescue / mean "
    "drop with its paired bootstrap CI (Qwen3 L44 17% [15, 19]; Mixtral L19 11% [9, 13] with BOS, 9% [7, 11] without) — scale-free, cannot change "
    "any selection, and makes runs with different drops comparable; (2) the **rank recovery** log2 rank(noised) − log2 rank(patched) with the "
    "top-1 recovery count — the metric closest to Δ per case (r 0.80-0.87 under the clean protocols) that also answers whether the patch brings "
    "the answer back to the top (L44E069 +0.62 log2 units, L19E002 +0.60); (3) the **clean top-1 rate and median p_clean of the case set** as "
    "dataset descriptors (29-32% and 0.02-0.04 here), because they say what 'factual recall' means for these cloze prompts. Use **Δp only "
    "descriptively** (the normalised Δp: 2.6-4.7% of the lost probability mass), never for selection or Spec: it saturates, is dominated by the "
    "top-1-flip cases (52-83% of its mass from 24-25% of cases) and is underpowered (every Spec CI includes zero under Δp). Use **KL and Δlog p as "
    "protocol diagnostics**: where they disagree with Δ they flag degenerate noised runs (the no-BOS Mixtral sink states) that Δ cannot see. Do "
    "**not** adopt p_clean ≥ 0.5 as the funnel: it keeps 26-45 of the 256 paper cases (Jaccard 0.10-0.19 with the Δ funnel, and every case it "
    "keeps already passes Δ) and would turn the study into one about the minority of prompts the base model completes correctly; report the "
    "overlap instead. Bottom line: the paper's Qwen3 result (L44E069, Spec > 0) and the BOS Mixtral result (L19E002, Spec > 0) are metric-"
    "independent; the paper's no-BOS Mixtral result is the one where a probability- or rank-based selection replaces L19E006 (Spec < 0 under "
    "every metric) with L18E001 (Spec > 0), in agreement with ext1's joint search."
)
