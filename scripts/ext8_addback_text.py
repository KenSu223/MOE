"""ext8: write results/sections/ext8_addback.md from results/ext8_addback_summary.json and the ext8 tables.

Usage: python scripts/ext8_addback_text.py
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
from moetrace.models import RESULTS

SEC = os.path.join(RESULTS, "sections", "ext8_addback.md")
TAB = os.path.join(RESULTS, "tables")


def r3(x, d=3):
    return f"{x['ratio']:.{d}f} [{x['lo']:.{d}f}, {x['hi']:.{d}f}]"


def fk(x):
    return "never" if x is None or (isinstance(x, float) and not np.isfinite(x)) or x == "Infinity" else str(int(float(x)))


def tab(name):
    p = os.path.join(TAB, f"{name}.md")
    return open(p).read() if os.path.exists(p) else f"(table {name} not available)"


def T(r):
    return "CounterFact" if r["task"] == "cf" else "WinoGrande"


def M(r, k="all_moe", d="d"):
    return r["a0"]["validation"][d][k]


def ka(e, k):
    return e["r"][e["k"].index(k)] if k in e["k"] else float("nan")


def short(r):
    return ("Qwen3" if r["model"] == "qwen3" else "Mixtral") + " " + T(r)


def lst(rs, f):
    return ", ".join(f"{short(r)} {f(r)}" for r in rs)


def direct(r):
    a = r["a0"]["validation"]["direct"]
    return f"{a['a_dir']['ratio']:.2f} / {a['m_dir']['ratio']:.2f}"


def summary_text(runs, keys):
    rs = [runs[k] for k in keys]
    return (
        "**Summary.** Patching expert outputs back JOINTLY at the final position (exact multi-layer patches, nothing summed) shows that "
        "experts alone cannot repair CounterFact STR but largely repair WinoGrande STR. Ceiling = all MoE outputs at the final position: "
        + lst(rs, lambda r: f"**{M(r)['ratio']:.2f}** [{M(r)['lo']:.2f}, {M(r)['hi']:.2f}]") + " of the drop (deletion "
        + ", ".join(f"{M(r, d='n')['ratio']:.2f}" for r in rs) + "). All attention outputs restore 1.00 by construction (the MoE is per-token "
        "and the final token is shared), so the two-player Shapley split degenerates to φ_MoE = M/2; the direct-path split of the final "
        "residual difference gives attention / MoE " + lst(rs, direct) + ": on WinoGrande the logit difference at the final position is "
        "written mostly by expert outputs, the opposite of the IOI-like hypothesis at this position. A handful of experts carries most of "
        "the expert repair except in Qwen3 WinoGrande: 80 % of the ceiling with k = " + lst(rs, lambda r: f"{fk(r['a1']['oracle']['k80_ceiling'])} (greedy {fk(r['greedy']['k80_ceiling'])}, "
                                                    f"random {fk(r['a1']['rand']['k80_ceiling'])} of {r['K']})")
        + " experts (per-case oracle ordering). Subsets overshoot the ceiling (oracle maximum " + ", ".join(f"{r['a1']['oracle']['max_r']:.2f}" for r in rs)
        + "): some clean expert outputs work against the answer. Greedy is good enough: beam search (width 4) adds ≤ "
        + f"{max(max(r['beam']['beam_minus_greedy_mean']) for r in rs):.3f} of the drop and the exact optimum over each case's top-10 experts "
        + f"beats greedy-within-top-10 by ≤ {max(max(r['exact']['greedy_gap_mean']) for r in rs):.3f} (below bf16 run-to-run noise), while "
        "adaptive greedy beats the static single-expert ranking by " + ", ".join(f"{r['greedy']['r'][9] - ka(r['a1']['oracle'], 10):+.2f}" for r in rs)
        + " at k = 10. The patch-free direct-logit-attribution ranking is within 0.03 of the single-patch oracle or better (AUC over log k "
        + lst(rs, lambda r: f"{r['a1']['dla']['auc_log']:.2f} vs {r['a1']['oracle']['auc_log']:.2f}")
        + "); the paper's layer-wise order is below every ranking that uses each expert's effect on the answer (oracle, DLA, "
        "population) but above routing weight and random.")


def reading_text(runs, keys):
    r = runs
    cf = [k for k in keys if r[k]["task"] == "cf"]
    wi = [k for k in keys if r[k]["task"] == "wino"]
    rs = [r[k] for k in keys]
    L = []
    L.append("**1. The ceiling.** On CounterFact the all-MoE ceiling is about half the drop or less ("
             + "; ".join(f"{short(r[k])} {M(r[k])['ratio']:.2f}, {r[k]['a0']['validation']['d']['frac_cases_all_moe_ge_0.8']:.0%} of cases reach 0.8, "
                         f"the answer flips back (Δ > 0) in {r[k]['ceiling_restores_answer_frac_cases']:.0%}" for k in cf)
             + "), so full repair from experts is impossible for most facts; the remainder needs the final position's attention outputs, which "
             "bring the subject information in. On WinoGrande the ceiling is much higher ("
             + "; ".join(f"{short(r[k])} {M(r[k])['ratio']:.2f}, {r[k]['a0']['validation']['d']['frac_cases_all_moe_ge_0.8']:.0%} of cases ≥ 0.8, answer restored in "
                         f"{r[k]['ceiling_restores_answer_frac_cases']:.0%}" for k in wi)
             + "). Add-back (sufficiency) and deletion (necessity) ceilings agree within 0.02. Restricted to interior layers (≤ L−5) the "
             "ceilings are " + lst(rs, lambda x: f"{M(x, 'int_moe')['ratio']:.2f}")
             + ": in Mixtral the last four layers' expert outputs at the final position work against the answer on CounterFact.")
    L.append("")
    L.append("**2. Attention vs MoE (W4).** Setting every attention output at the final position to its clean value restores the clean "
             "final residual exactly (bit for bit on OLMoE; 1.00 here), because the MoE is a per-token function and the final token is shared. "
             "So A ≡ 1, the two-player split reduces to φ_MoE = M/2 (" + lst(rs, lambda x: f"{M(x, 'phi_moe')['ratio']:.2f}")
             + ") and the redundancy A + M − 1 is M itself: the requested decomposition only measures M. The informative split is the direct "
             "path: h_clean − h_corrupt at the final position is the sum of the sublayer writes, and the logit difference evaluated on "
             "h_corrupt + Σ dAttn vs h_corrupt + Σ dMoE (exact final norm) gives attention / MoE " + lst(rs, direct)
             + " (the linear DLA split agrees to 0.01). On CounterFact attention and experts write comparable parts of the answer (Mixtral "
             "attention-heavier); on WinoGrande the final position's attention outputs carry information whose direct effect on LD(r, r′) is "
             "small, and the experts write it. At the final position, WinoGrande is therefore not attention-driven in the IOI sense; whether "
             "attention carries the decisive information upstream (option position, mover heads) is the W3 / W5 question.")
    L.append("")
    L.append("**3. How many experts.** k for 80 % of the ceiling with the per-case oracle / adaptive greedy / population / layer-wise / random "
             "orderings: " + "; ".join(f"{short(x)} {fk(x['a1']['oracle']['k80_ceiling'])} / {fk(x['greedy']['k80_ceiling'])} / {fk(x['a1']['pop']['k80_ceiling'])} / "
                                       f"{fk(x['a1']['layerwise']['k80_ceiling'])} / {fk(x['a1']['rand']['k80_ceiling'])} of {x['K']}" for x in rs)
             + ". One expert restores " + lst(rs, lambda x: f"{x['a1']['oracle']['r'][0]:.2f}") + " of the drop, ten (oracle) "
             + ", ".join(f"{ka(x['a1']['oracle'], 10):.2f}" for x in rs) + ", fifteen (greedy) " + ", ".join(f"{x['greedy']['r'][14]:.2f}" for x in rs)
             + ". CounterFact is concentrated on a handful of the 384 / 64 clean-active experts; WinoGrande in Qwen3 needs tens of experts for "
             "80 % of its (higher) ceiling, while Mixtral WinoGrande is nearly as concentrated as Mixtral CounterFact. The curves are not monotone and overshoot the ceiling (oracle maximum "
             + lst(rs, lambda x: f"{x['a1']['oracle']['max_r']:.2f} at k = {x['a1']['oracle']['k_max_r']}")
             + "; the all-clean-active endpoint returns to the ceiling): the clean outputs of low-ranked experts lower LD, so 'fraction of the "
             "ceiling' exceeds 1 for good subsets, and the brief's greedy stop at 95 % of the ceiling was replaced by 15 steps for every row "
             "(it would have truncated the curves at the ceiling). Answer restored (smallest k with donor-mean Δ_k > 0, oracle order; median, "
             "fraction of cases where some k achieves it): " + lst(rs, lambda x: f"{fk(x['a1']['oracle']['case_k_restored']['median'])}, "
                                                                                 f"{x['a1']['oracle']['case_k_restored']['frac_finite']:.0%}")
             + ". The true object becomes top-1 in at most " + ", ".join(f"{max(x['a1']['oracle']['top1_frac']):.2f}" for x in rs)
             + " of rows at any k (clean top-1 rate " + ", ".join(f"{x['clean_top1_frac_val']:.2f}" for x in rs) + ").")
    L.append("")
    sh = [x for x in rs if "shapley" in x]
    L.append("**4. Is greedy good enough?** Yes. Beam search (width 4, sizes ≤ 6) exceeds greedy on the same rows by at most "
             + lst(rs, lambda x: f"{max(x['beam']['beam_minus_greedy_mean']):+.3f}")
             + " of the drop; over each case's top-10 singles the exact optimum (all 1,023 subsets) exceeds greedy-within-top-10 by at most "
             + ", ".join(f"{max(x['exact']['greedy_gap_mean']):.3f}" for x in rs) + " on average, and greedy finds the optimum in "
             + ", ".join(f"{min(x['exact']['greedy_optimal_frac']):.0%}" for x in rs)
             + " or more of rows at every size. These gaps are below the run-to-run bf16 noise (Caveats). What matters is adaptivity: the "
             "static single-expert prefix loses up to " + ", ".join(f"{max(x['exact']['oracle_prefix_gap_mean']):.3f}" for x in rs)
             + " to the top-10 optimum, and the full greedy (pool 32 / 64) is above the static oracle by "
             + lst(rs, lambda x: f"{x['greedy']['r'][9] - ka(x['a1']['oracle'], 10):+.2f}")
             + " at k = 10, because the experts interact (Shapley: Σ single / Σ φ, > 1 = redundancy, < 1 = synergy, "
             + lst(sh, lambda x: f"{x['shapley']['sum_single_over_sum_phi']['median']:.2f}") + "; r(φ, single) "
             + ", ".join(f"{x['shapley']['corr_phi_single']['median']:.2f}" for x in sh) + "; experts for 80 % of Σφ "
             + ", ".join(fk(x['shapley']['n_experts_80pct_phi']['median']) for x in sh)
             + "). Interactions matter most on WinoGrande in Qwen3 (synergy; single-expert rescue the poorest guide, r = 0.46) and in Mixtral "
             "WinoGrande (redundancy: the singles over-count by 39 %).")
    L.append("")
    L.append("**5. Cheap rankings.** The direct logit attribution of δ_e (no patching; final norm frozen at the corrupted run) is better "
             "than the single-patch oracle in Qwen3 (both tasks), equal on Mixtral CounterFact and slightly worse on Mixtral WinoGrande "
             "(AUC over log k, DLA / oracle / population / layer-wise: "
             + "; ".join(f"{short(x)} {x['a1']['dla']['auc_log']:.2f} / {x['a1']['oracle']['auc_log']:.2f} / {x['a1']['pop']['auc_log']:.2f} / "
                         f"{x['a1']['layerwise']['auc_log']:.2f}" for x in rs)
             + "). Routing weight and |δ_e| are poor (AUC " + ", ".join(f"{x['a1']['weight']['auc_log']:.2f}" for x in rs) + " and "
             + ", ".join(f"{x['a1']['vnorm']['auc_log']:.2f}" for x in rs) + "; random " + ", ".join(f"{x['a1']['rand']['auc_log']:.2f}" for x in rs)
             + "). The paper's layer-wise order (best layer's experts first) is below every ranking that uses each expert's effect on the "
             "answer (oracle, DLA, population) because the repair is spread over a band of layers; it is above routing weight and random "
             "and comparable to |δ_e| (better in three of four runs).")
    try:
        import pandas as pd
        pa = pd.read_csv(os.path.join(RESULTS, "tables", "ext8_a1_partial_auc_k15.csv")).set_index("run")
        L.append("")
        L.append("Restricted to k = 1..15, where adaptive greedy is also evaluated (`scripts/ext8_partial_auc.py`, "
                 "`results/tables/ext8_a1_partial_auc_k15.md`; static orderings interpolated at k = 15), AUC over log k greedy / oracle / "
                 "DLA / population / layer-wise / |δ_e| / weight / random: "
                 + "; ".join(f"{short(x)} " + " / ".join(f"{pa.loc[k, c]:.2f}" for c in ("greedy", "oracle", "dla", "pop", "layerwise",
                                                                                    "vnorm", "weight", "rand"))
                             for k, x in zip(keys, rs) if k in pa.index) + ".")
    except FileNotFoundError:
        pass
    L.append("")
    L.append("**6. Which experts.** Greedy's first five picks contain (share of rows) "
             + "; ".join(f"{short(x)}: " + ", ".join(f"{a} {b:.0%}" for a, b in x['greedy']['in_first5_top'][:4]) for x in rs)
             + ". CounterFact recovers the Direction-1 / 6 loci (Qwen3 L42E115 and L44E069; Mixtral E001 of L18–L22 and L19E002); "
             "WinoGrande uses different experts: Qwen3 L41E117, L43E081, L39E071 just below the CounterFact band, and in Mixtral two experts "
             "in the first five picks of about 80 % of rows, L20E000 and L19E006 (the paper's Mixtral expert and the Direction-3 'sink "
             "expert'; on CounterFact with BOS it is in the first five picks of only 14 % of rows).")
    L.append("")
    L.append("**7. Deletion (noising).** Swapping clean-active experts to their corrupted values in the clean run mirrors add-back: k for 80 % "
             "of the deletion ceiling with the noising-oracle / add-back-oracle / DLA / random orderings "
             + "; ".join(f"{short(x)} {fk(x['a3']['noise_oracle']['k80_ceiling'])} / {fk(x['a3']['oracle']['k80_ceiling'])} / "
                         f"{fk(x['a3']['dla']['k80_ceiling'])} / {fk(x['a3']['rand']['k80_ceiling'])}" for x in rs)
             + " (AUC over log k " + "; ".join(f"{short(x)} {x['a3']['noise_oracle']['auc_log']:.2f} / {x['a3']['oracle']['auc_log']:.2f} / "
                                              f"{x['a3']['dla']['auc_log']:.2f} / {x['a3']['rand']['auc_log']:.2f}" for x in rs)
             + "). Necessity and sufficiency rank largely the same experts; DLA is the best or within 0.02 of the best deletion order, and "
             "the noising singles beat the add-back singles as a deletion order (equal in Mixtral WinoGrande).")
    return "\n".join(L)


def caveats_text(runs, keys):
    rs = [runs[k] for k in keys]
    sh = [x for x in rs if "shapley" in x]
    L = [
        "- Final position only. Expert patches at the subject / option positions (F4, 6b, W3) are a different question; the direct-path "
        "split says who writes LD at the final position, not where the information is computed.",
        "- Candidates are the clean run's routed experts at the final position, taken from the source runs' routing; experts routed only in "
        "the corrupted (or patched) run keep their own outputs, so the all-clean-active endpoint ≈ the all-MoE ceiling ("
        + lst(rs, lambda x: f"{x['a1']['oracle']['r'][-1]:.2f} vs {M(x)['ratio']:.2f}") + "). Near-tie routing differs between the source pass "
        "and the add-back passes for 0.3–1.9 % of (row, candidate) pairs (run logs); those experts are patched to their in-pass clean value (0 if "
        "not routed).",
        "- bf16 batch-composition noise: identical prefill rows give different Δ in different passes (expert GEMMs batch prefill and wavefront "
        "tokens; routing near-ties amplify it): per-row SD across passes, median " + lst(rs, lambda x: f"{x['prefill_range']['corrupt_sd_median']:.2f}")
        + " logits (99th percentile " + ", ".join(f"{x['prefill_range']['corrupt_sd_p99']:.2f}" for x in rs) + "). Every curve of a row (one "
        "ordering, all k), every exact-subset table and every Shapley permutation is evaluated within one pass, and rescue uses the same pass's "
        "Δ_corrupt; greedy / beam steps span passes. Strategy differences below ~0.01 of the drop are not resolved.",
        "- Subsets: Shapley values from 5 permutations on first-donor rows (" + lst(sh, lambda x: f"{x['shapley']['n_rows']}") + "; mean SE of φ "
        + ", ".join(f"{x['shapley']['mean_se_of_phi']:.3f}" for x in sh) + " logits); beam rows " + lst(rs, lambda x: f"{x['beam']['n_rows']}")
        + "; exact top-10 rows " + ", ".join(f"{x['exact']['n_rows']}" for x in rs) + ". The exact optimum is over each row's top-10 singles "
        "only; the full greedy (pool 32 / 64) exceeds it from k ≈ 4 on.",
        "- Greedy and the oracle ordering use the row's own single-expert patches (in-sample); the population ranking (discovery → validation) "
        "is the out-of-sample comparison.",
        "- WinoGrande units are directed cases (one corrupted run each); CIs resample pairs. CounterFact: donor means per case; the first-donor "
        "rows are in the tables as the sensitivity run.",
        "- Gradient rankings (attribution patching, AtP*, EAP-IG) need F3 and were not run.",
    ]
    return "\n".join(L)


def main():
    S = json.load(open(os.path.join(RESULTS, "ext8_addback_summary.json")))
    runs = S["runs"]
    ver = json.load(open(os.path.join(RESULTS, "verify_ext8_engine_olmoe.json")))
    keys = [k for k in ("cf_qwen3", "cf_mixtral", "wino_qwen3", "wino_mixtral") if k in runs]
    L = []
    w = L.append

    # ---------------- summary paragraph (templated from the summary JSON)
    w(summary_text(runs, keys))
    w("")
    # ---------------- what was run
    w("### What was run")
    w("")
    w("**Engine (step E).** `multi` accepts `attn_layer` and `block` steps (first step = the single-layer kind; later step: the live row's "
      "attention output at the final position is replaced by the source run's, h_mid = h_in_own + Attn_source, and for `block` also the MoE "
      "output, h_out = h_mid + MoE_source), every kind works in the noising direction (parent = clean row, source = corrupted row), "
      "`SpawnSpec.kl_ref` sets the KL reference row, `DiagSpec.contrib_dla` records the per-expert direct logit attribution of the prefill rows, "
      "and the later-step vectors are computed in row chunks. Verification on OLMoE against transformers hooks "
      f"(`scripts/ext8_engine_verify.py`, {ver['n_units']} STR units: CounterFact donors and WinoGrande twins in both directions, 21 patch "
      f"configurations x 2 directions): all |ΔΔ| max {ver['vs_hf_all']['maxabs']:.2f}, mean {ver['vs_hf_all']['meanabs']:.3f}, "
      f"{100 * ver['vs_hf_all']['frac_within_0.25']:.0f} % within 0.25, effect r = {ver['vs_hf_all_effect_corr']:.4f} (the single-layer kinds on the "
      "same units are in the same envelope); all-layer `block` reproduces the source run's Δ bit for bit in both directions, in the engine and in "
      "transformers; single-step `multi` attn_layer / block equals the single-layer kinds exactly; all-layer coalition_set(all experts) = "
      f"all-layer `layer` to fp32 summation order; the DLA diagnostic equals the spawn-vector computation to {ver['dla']['diag_vs_vector_max_rel']:.0e} "
      f"(relative). Stress: {ver['stress']['rows']:,} all-layer multi rows in one OLMoE pass, {ver['stress']['peak_mem_GB']:.1f} GB peak. "
      "`results/verify_olmoe.json` and `results/verify_ext5_engine_olmoe.json` are identical to the pre-merge copies in every non-timing field.")
    w("")
    w("**Tasks and units.** CounterFact STR (Direction 6 runs `qwen3_str`, `mixtral_bos_str`: paper IDs and split, up to five known donors "
      "per case, donor mean primary, first donor sensitivity); WinoGrande STR (ext7 runs, one row per directed case, bootstrap over pairs). "
      "Candidates = the clean run's routed (layer, expert) pairs at the final position (Qwen3 384, Mixtral 64), the set of the ext6 / ext7 "
      "single-expert rows. Every curve point is an exact joint patch (`multi`, one `coalition_set` step per layer, parent = corrupted run): "
      "later layers see the effect of earlier patches, so nothing is summed.")
    w("")
    w("**Normalisation.** r(k) = (Δ_k − Δ_corrupt) / (Δ_clean − Δ_corrupt) with Δ_corrupt from the same pass and the drop from pass 0; "
      "deletion: (Δ_clean − Δ_k) / drop. Population values are ratios of means over validation cases with a percentile bootstrap over cases "
      "(WinoGrande: pairs); 'per case' = case ratios. AUC = trapezoid of the population r(k) over log k (k = 1 … K) divided by log K, "
      "and over k/K. k50/80/90 = the smallest grid k with r(k) ≥ q × ceiling (or ≥ q of the drop).")
    w("")
    w("**Orderings.** pop = discovery all-case single-expert rescue (evaluated on validation); oracle = the row's own single-expert rescue; "
      "layerwise = layers by discovery MoE-layer rescue, experts within a layer by pop (the paper's way); rand = mean of 5 per-case "
      "permutations; weight = clean routing weight; vnorm = |δ_e|; dla = (δ_e ⊙ γ)·(W_U[r] − W_U[r′]) / rms(h_final, corrupted run), "
      "i.e. the final RMSNorm frozen at the corrupted run's scale; noise_oracle (deletion only) = the row's own noising single-expert effect. "
      "Greedy: at every step every remaining candidate of the pool (Qwen3: the row's top-32 singles, Mixtral: all 64) is evaluated jointly with "
      "the current set, 15 steps for every row (the brief's optional stop at 95 % of the row's all-MoE ceiling was not used because subsets overshoot the ceiling; Reading 3). Beam: width 4, sizes ≤ 6. Exact: all 1,023 subsets of the row's top-10 "
      "singles. Shapley: marginal gains along the full prefix sweeps of the 5 random permutations. Gradient rankings need F3 (not built): not done.")
    w("")
    # run table
    w("| Run | Validation cases / rows | Passes | Greedy rows (pool) | Beam rows | Exact rows | Shapley rows | Prefill Δ of identical rows across passes |")
    w("|---|---|---|---|---|---|---|---|")
    for k in keys:
        r = runs[k]
        w(f"| {r['label']}, {r['task']} | {r['n_cases_val']} / {r['n_rows_val']} | {r['passes']} | {r.get('greedy', {}).get('n_rows', '')} "
          f"({r.get('greedy', {}).get('pool', '')}) | {r.get('beam', {}).get('n_rows', '')} | {r.get('exact', {}).get('n_rows', '')} | "
          f"{r.get('shapley', {}).get('n_rows', '')} | SD median {r['prefill_range']['clean_sd_median']:.2f} / {r['prefill_range']['corrupt_sd_median']:.2f}, "
          f"99th pct {r['prefill_range']['clean_sd_p99']:.2f} / {r['prefill_range']['corrupt_sd_p99']:.2f} (clean / corrupted) |")
    w("")
    # ---------------- A0
    w("### A0. Ceilings and the attention / MoE decomposition at the final position (W4 for these tasks)")
    w("")
    w(tab("ext8_a0_ceilings"))
    w("")
    w("A = all attention outputs, M = all MoE outputs, both = A and M at every layer (sanity: the clean final residual); φ = two-player "
      "Shapley split ½[A + (1 − M)] / ½[M + (1 − A)], redundancy A + M − 1; 'direct A / M' = Δ evaluated on h_corrupt + Σ_l dAttn_l and "
      "h_corrupt + Σ_l dMoE_l (exact final RMSNorm, fp32 offline from the recorded final-position sublayer outputs), as fractions of the "
      "same fp32 drop; 'DLA share' = the linear version with the norm frozen at the corrupted run.")
    w("")
    w("![A0 ceilings and direct-path split](figures/ext8_a0_split.png)")
    w("")
    w("![Add-back curves](figures/ext8_a1_curves.png)")
    w("")
    w("### A1. Static orderings (add-back)")
    w("")
    w(tab("ext8_a1_summary"))
    w("")
    w("### A2. Adaptive strategies")
    w("")
    w(tab("ext8_a2_adaptive"))
    w("")
    w("![Adaptive strategies, k ≤ 10](figures/ext8_a2_adaptive.png)")
    w("")
    w("Beam rows are the first donors of the first validation cases (Mixtral CounterFact 64 of 106; WinoGrande: first 128 / 64 directed "
      "cases); the beam's pool is the greedy pool (32 / 64), so it can exceed the top-10 optimum.")
    w("")
    w("**Shapley values** (5 permutations, full prefix sweeps, first donor)")
    w("")
    w(tab("ext8_shapley"))
    w("")
    w("### A3. Deletion curves (noising)")
    w("")
    w(tab("ext8_a3_summary"))
    w("")
    w("![Deletion curves](figures/ext8_a3_curves.png)")
    w("")
    w("### CounterFact vs WinoGrande")
    w("")
    w(tab("ext8_saturation_cf_vs_wino"))
    w("")
    w("### Reading")
    w("")
    w(reading_text(runs, keys))
    w("")
    w("### Caveats")
    w("")
    w(caveats_text(runs, keys))
    w("")
    w("### Files")
    w("")
    w("- Engine: `moetrace/engine.py` (ext8 additions documented in the module docstring); dev copy `moetrace/engine_ext8_dev.py`; "
      "verification `scripts/ext8_engine_verify.py` → `results/verify_ext8_engine_olmoe.json`; regression `scripts/ext8_regress_compare.py` "
      "against `results/verify_olmoe_before_ext8.json`, `results/verify_ext5_engine_olmoe_before_ext8.json`.")
    w("- Study: `moetrace/ext8_addback.py` (task loaders, orderings, work items, greedy / beam state, curve metrics), "
      "`scripts/ext8_addback_run.py` (resumable driver), `scripts/ext8_addback_analyze.py`, `scripts/ext8_addback_text.py`, "
      "`scripts/ext8_addback_chain.sh`, `scripts/ext8_addback_chain_mixtral.sh`; smoke test `scripts/ext8_smoke_src.py`, "
      "`scripts/ext8_smoke_chain.sh` (OLMoE: `results/olmoe_addback_src` = ext6-schema STR source on the Qwen3 paper IDs, "
      "`results/olmoe_addback_smoke`). GPU time of Phase-3 ext8 jobs: ≈ 82 min (add-back passes 71, verification / regression 7, smoke 4).")
    w("- Runs: " + ", ".join(f"`results/{runs[k]['label'] and ''}{x}`" for k, x in
                             (("cf_qwen3", "qwen3_str_addback"), ("cf_mixtral", "mixtral_bos_str_addback"),
                              ("wino_qwen3", "wino_qwen3_str_addback"), ("wino_mixtral", "wino_mixtral_bos_str_addback")) if k in runs)
      + " (`addback_rows_pNN.parquet` spawn rows with fam / order / dir / k / row / Δ / metrics, `addback_prefill_pNN.parquet`, "
      "`addback_dla.parquet`, `addback_direct.parquet`, `addback_state.pkl` = greedy / beam paths, `run_meta.json`).")
    w("- Tables `results/tables/ext8_*.md|csv` (curves in `ext8_a1_curves.csv`, `ext8_a3_curves.csv`); figures `results/figures/ext8_*`; "
      "numbers `results/ext8_addback_summary.json` (W4 keys `w4_counterfact`, `w4_winogrande`).")
    txt = "\n".join(L)
    os.makedirs(os.path.dirname(SEC), exist_ok=True)
    with open(SEC, "w") as f:
        f.write(txt + "\n")
    print(f"wrote {SEC}")


if __name__ == "__main__":
    main()
