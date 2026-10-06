"""ext10: write results/sections/ext10_circuit.md from results/ext10_circuit_summary.json and the ext10 tables.

Usage: python scripts/ext10_circuit_text.py
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
from moetrace.models import RESULTS
from moetrace import ext10_circuit as C

SEC = os.path.join(RESULTS, "sections", "ext10_circuit.md")
TAB = os.path.join(RESULTS, "tables")
Q = ("0.5", "0.8", "0.9")


def tab(name):
    p = os.path.join(TAB, f"{name}.md")
    return open(p).read() if os.path.exists(p) else f"(table {name} not available)\n"


def fk(x):
    if x is None:
        return "never"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "never"
    return "never" if not np.isfinite(x) else str(int(x))


def ci(e, d=2):
    return f"{e['ratio']:.{d}f} [{e['lo']:.{d}f}, {e['hi']:.{d}f}]"


def short(r):
    m = "Qwen3" if r["label"].startswith("Qwen3") else "Mixtral"
    t = "CounterFact" if r["task"].startswith("Counter") else ("IOI" if r["task"].startswith("IOI") else "WinoGrande")
    return f"{m} {t}"


def lst(rs, f, sep=", "):
    return sep.join(f"{short(r)} {f(r)}" for r in rs)


def step1_text(s1):
    rs = [s1[k] for k in C.RUN_ORDER if k in s1]
    out = []
    out.append("### Step 1. Single heads at every layer and the patch-free head DLA\n")
    out.append("Every (layer, head) of the model was patched alone at the final position (attn_head: the head's pre-o_proj output "
               "set to the clean run's, the rest of the corrupted attention output kept) on every ext8 validation row, with one "
               "attn_layer row per layer in the same pass. The head DLA uses the same per-head output differences without any "
               "patch: dla_h = (W_o[:, h] ΔH_h ⊙ γ) · (W_U[r] − W_U[r′]) / rms(h_final, corrupted run). Z8 = population effect "
               "≥ 2 SD from the mean over all heads of the model, on all validation units and separately on two disjoint halves.\n")
    out.append(tab("ext10_heads_summary"))
    out.append("\n'Σ single heads' = the sum of all single-head population effects (all heads at all layers jointly restore 1.00: the "
               "attention outputs are the only path of the corruption to the final position); 'Pop. heads for q' = the number of heads, "
               "taken in order of their population effect, whose single effects add up to q of the drop (an additive estimate; Step 2 "
               "measures joint patches); 'Heads in joint top-32' = how many of a row's 32 best single components (heads ∪ the "
               "row's clean-active experts) are heads. IOI (S2 → IO, Qwen3 only, the attention-pole reference): no single-expert patch "
               "table exists, so its expert columns use the expert DLA of the same pass ('experts: row r' = 1 by construction).\n")
    out.append("**Top and bottom heads (validation; donor mean for CounterFact)**\n")
    out.append(tab("ext10_heads_top"))
    out.append("\n**Heads found in earlier directions (F2 under GN on CounterFact, W5 under STR on WinoGrande at the W2 attention layers)**\n")
    out.append(tab("ext10_heads_known"))
    out.append("\n**The six layers with the largest attention-output patch: sum of single heads vs the layer patch**\n")
    out.append(tab("ext10_heads_layers"))
    out.append("\n**Additive top-k sums of single components per row (median, fraction of the drop; NOT joint patches)**\n")
    out.append(tab("ext10_heads_additive"))
    out.append("\n![Single-head rescue, layer x head](figures/ext10_heads_heatmap.png)\n")
    out.append("![Head DLA vs single-head patch; layer curves](figures/ext10_heads_dla_layers.png)\n")
    return "\n".join(out)


def step2_text(s2):
    rs = [s2[k] for k in C.RUN_ORDER if k in s2]
    out = ["### Step 2. Joint add-back over heads + experts\n"]
    out.append("Every curve point is ONE exact joint patch (`multi` spawn, one step per layer carrying that layer's heads and experts: "
               "heads patched before the MoE, the row's own MoE recomputed on the head-patched input, the listed experts' "
               "contributions set to the clean run's), so later layers see earlier patches. Mixed greedy: pool = the row's top-32 "
               "heads by single-head rescue ∪ ext8's expert pool (Qwen3: the row's top-32 experts, Mixtral: all 64), 20 steps; "
               "head-only greedy: pool = the 32 heads; static orderings over ALL heads ∪ all clean-active experts by the row's "
               "single-patch rescue (oracle) or by DLA, and the same over heads only, on k = 1..10, 12, 16, 24, 32, 48, 64, 96, 128. "
               "ext8's expert-only greedy (15 steps, same rows) and its static expert curves are the comparison. r(k) = rescue / drop, "
               "ceiling 1 (all heads = all attention outputs = the clean final residual); the all-MoE ceiling is drawn for reference.\n")
    out.append("**Sanity checks (inside the passes)**\n")
    out.append(tab("ext10_circuit_sanity"))
    out.append("\n**Curves (population r(k), validation; k50/80/90 = smallest k with r ≥ q of the drop; AUC over log k on the same "
               "k range for every curve; static curves interpolated in log k)**\n")
    out.append(tab("ext10_circuit_curves"))
    out.append("\n**Adaptive greedy: per-case thresholds, answer restored, composition, recurring components**\n")
    out.append(tab("ext10_circuit_greedy"))
    out.append("\n![Joint add-back curves over heads + experts](figures/ext10_circuit_curves.png)\n")
    out.append("![Composition of the mixed greedy picks](figures/ext10_circuit_composition.png)\n")
    return "\n".join(out)


def summary_text(s1, s2):
    r1 = [s1[k] for k in C.MAIN_RUNS if k in s1]
    parts = []
    if s2:
        r2 = [s2[k] for k in C.MAIN_RUNS if k in s2]
        g = lambda r: r["curves"]["greedy_mix"]
        parts.append(
            "**Summary.** With single attention heads as candidates next to experts, a handful of components restores the "
            "corrupted answer at the final position, and the ceiling is 1 instead of the all-MoE ceiling (" +
            lst(r2, lambda r: f"{r['ceilings']['all_moe']['ratio']:.2f}") + " of the drop). Adaptive greedy over heads ∪ experts "
            "(exact joint patches, 20 steps) restores, at k = 20, " +
            lst(r2, lambda r: f"**{ci(g(r)['r20'])}**") + " of the drop (k = 10: " +
            lst(r2, lambda r: f"{g(r)['r10']['ratio']:.2f}") + "); k for 50 / 80 / 90 % of the drop: " +
            lst(r2, lambda r: "/".join(fk(g(r)["k_q"][q]) for q in Q)) + ". Experts alone (ext8 greedy) reach " +
            lst(r2, lambda r: f"{r['curves']['greedy_expert_ext8']['r15']['ratio']:.2f}") + " at k = 15, heads alone (head-only greedy) " +
            lst(r2, lambda r: f"{r['curves']['greedy_head']['r20']['ratio']:.2f}") + " at k = 20. The answer is restored (Δ > 0) at k = 20 in " +
            lst(r2, lambda r: f"{g(r)['frac_cases_restored']['20']:.2f}") + " of cases; rows with r ≥ 0.9 at k = 20: " +
            lst(r2, lambda r: f"{g(r)['frac_rows_r_ge_0.9']['20']:.2f}") + ". Heads among the first 10 mixed-greedy picks: " +
            lst(r2, lambda r: f"{g(r)['composition']['10']['mean_heads']:.1f}") + ". The patch-free DLA ordering over heads ∪ experts reaches AUC "
            "(log k, 1..20) " + lst(r2, lambda r: f"{r['curves']['static_mix_dla']['auc_log_k20']:.2f}") + " vs the single-patch oracle " +
            lst(r2, lambda r: f"{r['curves']['static_mix_oracle']['auc_log_k20']:.2f}") + " and greedy " +
            lst(r2, lambda r: f"{g(r)['auc_log_k20']:.2f}") + ".")
        io = [s2[k] for k in C.RUN_ORDER if k in s2 and k.startswith("ioi")]
        if io:
            parts.append(" IOI (S2 → IO, Qwen3; attention-pole reference): " + lst(io, lambda r: (
                f"the mixed greedy picks heads almost exclusively ({g(r)['composition']['10']['mean_heads']:.1f} of the first 10), "
                f"r(10) / r(20) = {g(r)['r10']['ratio']:.2f} / {ci(g(r)['r20'])}, 80 % at k = {fk(g(r)['k_q']['0.8'])}, "
                f"while all MoE outputs give {r['ceilings']['all_moe']['ratio']:.2f}")) + ".")
    else:
        parts.append("**Summary (Step 1 only; Step 2 waits for the ext9 engine).**")
    top = lambda r: f"{r['top'][0]['head']} ({r['top'][0]['rescue_over_drop']['ratio']:+.2f} [{r['top'][0]['rescue_over_drop']['lo']:+.2f}, {r['top'][0]['rescue_over_drop']['hi']:+.2f}])"
    parts.append(" Single-head patches at every layer (Step 1): the best single head per task is " + lst(r1, top) +
                 " of the drop; per row the best head matches or beats the best expert on CounterFact and is weaker on WinoGrande (median "
                 "best head / best expert " +
                 lst(r1, lambda r: f"{r['experts']['best_head_over_drop']['median']:.2f} / {r['experts']['best_expert_over_drop']['median']:.2f}") +
                 "); Z8 (≥ 2 SD over all heads) detects " + lst(r1, lambda r: f"{r['z8']['n_pos']} positive / {r['z8']['n_neg']} negative of {r['L'] * r['nH']:,}") +
                 " heads. The head DLA agrees with the single-head patches at the population "
                 "level (r over heads " + lst(r1, lambda r: f"{r['dla_vs_single']['pop_pearson']:.2f}") + "; top-1 head agrees in " +
                 lst(r1, lambda r: f"{r['dla_vs_single']['top1_agree']:.2f}") + " of rows).")
    return "".join(parts)


def gpu_minutes():
    tot, by = 0.0, {}
    for line in open(os.path.join(os.path.dirname(RESULTS), "logs", "gpu_queue.log")):
        if " END   ext10" in line:
            name = line.split("END   ")[1].split()[0]
            sec = float(line.rsplit("(", 1)[1].split("s)")[0])
            tot += sec
            by[name] = by.get(name, 0.0) + sec
    return tot / 60.0, by


def runs_text(S):
    rows = []
    for k in C.RUN_ORDER:
        od = C.out_dir(k)
        sp = os.path.join(od, "heads_state.json")
        if not os.path.exists(sp):
            continue
        st = json.load(open(sp))
        notes = [n for n in st["notes"] if "chunk" in n]
        r1 = S.get("step1", {}).get(k)
        if not r1:
            continue
        rows.append(f"| {r1['label']} | {r1['task']} | {r1['n_rows']} / {r1['n_cases']} | {len(st['chunks'])} | "
                    f"{sum(n['spawn_rows'] for n in notes):,} | {max(n['peak_GB'] or 0 for n in notes):.1f} | "
                    f"{sum(n['pass_s'] or 0 for n in notes) / 60:.1f} |")
    out = ["**Step 1 passes (one chunk of validation rows per pass; every (layer, head) + one attn_layer row per layer per row)**\n",
           "| Model | Task | Rows / cases | Passes | Spawn rows | Peak GB | Pass time (min) |", "|---|---|---|---|---|---|---|"] + rows
    sr = S.get("step2_runs", {})
    if sr:
        out += ["", "**Step 2 passes (all tasks of a model share every pass: Qwen3 CounterFact + WinoGrande + IOI, Mixtral CounterFact + "
                "WinoGrande; ≤ 100k / 95k spawn rows per pass)**\n", "| Model | Passes | Pass time (min) | Peak GB |", "|---|---|---|---|"]
        for m, v in sr.items():
            out.append(f"| {m} | {v['passes']} | {v['pass_s_sum'] / 60:.1f} | {v['peak_GB_max']:.1f} |")
    tot, by = gpu_minutes()
    out.append(f"\nGPU time of all ext10 queue jobs (incl. engine load, smoke tests): {tot:.0f} min.\n")
    return "\n".join(out) + "\n"


def g_(r, name="greedy_mix"):
    return r["curves"][name]


def reading_text(s1, s2):
    r1 = [s1[k] for k in C.RUN_ORDER if k in s1]
    m1 = [s1[k] for k in C.MAIN_RUNS if k in s1]
    out = ["### Reading\n"]
    n = 1
    if s2:
        r2 = [s2[k] for k in C.MAIN_RUNS if k in s2]
        io = [s2[k] for k in C.RUN_ORDER if k in s2 and k.startswith("ioi")]
        out.append(f"**{n}. Full repair is reachable with few components.** Experts alone cannot exceed the all-MoE ceiling (" +
                   lst(r2, lambda r: f"{r['ceilings']['all_moe']['ratio']:.2f}") + "; ext8 greedy at k = 15: " +
                   lst(r2, lambda r: f"{r['curves']['greedy_expert_ext8']['r15']['ratio']:.2f}") + "). Admitting heads, adaptive greedy "
                   "reaches " + lst(r2, lambda r: f"{g_(r)['r10']['ratio']:.2f} / {g_(r)['r20']['ratio']:.2f}") + " of the drop at k = 10 / 20, "
                   "i.e. 50 / 80 / 90 % of the drop with k = " + lst(r2, lambda r: "/".join(fk(g_(r)["k_q"][q]) for q in Q)) +
                   " components (population curve; per case, median k for 80 %: " +
                   lst(r2, lambda r: f"{fk(g_(r)['case_k_q']['0.8']['median'])} (reached in {g_(r)['case_k_q']['0.8']['frac_finite']:.2f})") +
                   "). The answer itself (Δ > 0, donor mean) is restored at k = 20 in " +
                   lst(r2, lambda r: f"{g_(r)['frac_cases_restored']['20']:.2f}") + " of cases (median k " +
                   lst(r2, lambda r: fk(g_(r)['case_k_restored']['median'])) + "), and r ≥ 0.9 at k = 20 holds for " +
                   lst(r2, lambda r: f"{g_(r)['frac_rows_r_ge_0.9']['20']:.2f}") + " of rows; the true object is top-1 in " +
                   lst(r2, lambda r: f"{g_(r)['frac_rows_top1_k20']:.2f}") + " of rows at k = 20 (clean top-1 rate " +
                   lst(r2, lambda r: f"{r['clean_top1_rate']:.2f}") + ").\n")
        n += 1
        out.append(f"**{n}. Heads and experts are complements, not substitutes.** Head-only greedy (pool 32) reaches " +
                   lst(r2, lambda r: f"{g_(r, 'greedy_head')['r20']['ratio']:.2f}") + " at k = 20, the mixed greedy " +
                   lst(r2, lambda r: f"{g_(r)['r20']['ratio']:.2f}") + ", experts alone (ext8, k = 15) " +
                   lst(r2, lambda r: f"{r['curves']['greedy_expert_ext8']['r15']['ratio']:.2f}") + " (mixed at k = 15: " +
                   lst(r2, lambda r: f"{g_(r)['r15']['ratio']:.2f}") + "). Among the first 5 / 10 / 20 mixed picks the mean number of heads is " +
                   lst(r2, lambda r: "/".join(f"{g_(r)['composition'][str(k)]['mean_heads']:.1f}" for k in (5, 10, 20))) +
                   "; the first pick is a head in " + lst(r2, lambda r: f"{g_(r)['composition']['1']['frac_rows_first_pick_head']:.2f}") +
                   " of rows. Median layer of the heads / experts among the first 10 picks: " +
                   lst(r2, lambda r: f"L{g_(r)['composition']['layers_first10']['heads']['median']:.0f} / L{g_(r)['composition']['layers_first10']['experts']['median']:.0f}"
                       if g_(r)['composition']['layers_first10']['heads'] and g_(r)['composition']['layers_first10']['experts'] else "n/a") +
                   ". The order differs by task: on CounterFact the greedy mostly starts with the mover heads and adds experts later (heads are "
                   "about half of the first 10-20 picks), on WinoGrande it starts with the W6 experts and adds a few late heads (roughly a "
                   "third of the picks); the median marginal gain of a head / expert step (mixed greedy, steps 2-20, fraction of the drop) is " +
                   lst(r2, lambda r: f"{g_(r)['marginal_gain']['heads']['median']:.3f} / {g_(r)['marginal_gain']['experts']['median']:.3f}") +
                   ". Neither component class alone gets there: the head-only curve flattens below the drop (most clearly on WinoGrande, "
                   "where the final-position attention writes only ~5-30 % of the logit difference directly), the expert-only curve below the "
                   "all-MoE ceiling.\n")
        n += 1
        out.append(f"**{n}. Which components recur.** Most frequent in the first five mixed-greedy picks (share of rows): " +
                   "; ".join(f"{short(r)}: " + ", ".join(f"{x['comp']} {x['share']:.2f}" for x in g_(r)["recurring_first5"][:6]) for r in r2) +
                   ". The recurring components are the loci of earlier directions: the CounterFact mover heads of F2 (Qwen3 L40H13 / H14 / H15, "
                   "L43H11; Mixtral L18H4, L24H22, L19H29) with the CounterFact experts (Qwen3 L44E069, Mixtral L21E001), and the WinoGrande "
                   "experts of W6 / ext8 (Qwen3 L41E117, L43E081, L39E071; Mixtral L20E000, L19E006, L21E006) with the late WinoGrande heads "
                   "found in Step 1 (Qwen3 L46H24, L41H27; Mixtral L25H9, L22H20).\n")
        n += 1
        out.append(f"**{n}. Greedy vs static and patch-free orderings.** AUC of r(k) over log k (k = 1..20; static orderings interpolated) "
                   "mixed greedy / head-only greedy / static oracle (heads ∪ experts) / static DLA (heads ∪ experts) / static head oracle / static head DLA: " +
                   "; ".join(f"{short(r)} " + " / ".join(f"{r['curves'][nm]['auc_log_k20']:.2f}" for nm in
                             ("greedy_mix", "greedy_head", "static_mix_oracle", "static_mix_dla", "static_head_oracle", "static_head_dla")) for r in r2) +
                   ". Over k = 1..15, against ext8's expert-only greedy / static expert oracle / static expert DLA: " +
                   "; ".join(f"{short(r)} mixed greedy {r['curves']['greedy_mix']['auc_log_k15']:.2f} vs " +
                             " / ".join(f"{r['curves'][nm]['auc_log_k15']:.2f}" for nm in ("greedy_expert_ext8", "static_expert_oracle_ext8", "static_expert_dla_ext8"))
                             for r in r2) + ". Largest k on the static grid: r(128) oracle / DLA " +
                   lst(r2, lambda r: f"{r['curves']['static_mix_oracle'].get('r128', {}).get('ratio', float('nan')):.2f} / {r['curves']['static_mix_dla'].get('r128', {}).get('ratio', float('nan')):.2f}") +
                   ". Adaptivity is worth 0.04-0.09 of AUC over the best static ordering; the patch-free DLA ordering over heads ∪ experts is "
                   "as good as the single-patch oracle on CounterFact and Qwen3 WinoGrande and slightly worse on Mixtral WinoGrande, and it "
                   "overshoots the drop at large k (r(128) > 1: clean components that lower LD are left out), whereas the oracle saturates "
                   "below 1. For heads alone the DLA is a poor guide on WinoGrande (the decisive heads act indirectly, Reading 7).\n")
        n += 1
        if io:
            out.append(f"**{n}. IOI (attention pole).** " + lst(io, lambda r: (
                f"mixed greedy r(10) / r(20) {g_(r)['r10']['ratio']:.2f} / {g_(r)['r20']['ratio']:.2f}, head-only {g_(r, 'greedy_head')['r20']['ratio']:.2f}, "
                f"heads among the first 10 picks {g_(r)['composition']['10']['mean_heads']:.1f}, all-MoE {r['ceilings']['all_moe']['ratio']:.2f}; first-5 picks "
                + ", ".join(f"{x['comp']} {x['share']:.2f}" for x in g_(r)['recurring_first5'][:4]))) +
                ". Expert orderings / pools on IOI use the expert DLA (no single-expert patch table exists for IOI).\n")
            n += 1
    out.append(f"**{n}. Single heads.** The best single head per row is as strong as the best single expert on CounterFact and weaker on "
               "WinoGrande (median best head / best expert " +
               lst(m1, lambda r: f"{r['experts']['best_head_over_drop']['median']:.2f} / {r['experts']['best_expert_over_drop']['median']:.2f}") +
               "); a row's 32 best single components contain " + lst(m1, lambda r: f"{r['experts']['n_heads_in_joint_top32']['median']:.0f}") +
               " heads (median). The top heads are those of earlier directions: CounterFact STR recovers the F2 GN heads (Qwen3 L40H13, L43H11; "
               "Mixtral L18H4, L24H22, L19H29, L15H1), WinoGrande recovers every W5 head (Qwen3 L38H18 / L38H21 positive, L38H16 negative; Mixtral "
               "L25H9, L19H13, L13H18 / H11 / H4) and adds heads in layers W5 did not scan (" +
               lst([r for r in m1 if r['key'].startswith('wino') and r.get('w5_rowmatch')],
                   lambda r: ", ".join(h["head"] + f" {h['rescue_over_drop']['ratio']:+.2f}" for h in (r["top"][:8] + r["bottom"][:1])
                                       if h["layer"] not in r["w5_rowmatch"]["layers"])) +
               "). Opposing heads of one KV group cancel inside a layer (Qwen3 WinoGrande L46H24 / L46H25), so the layer patch hides them.\n")
    n += 1
    out.append(f"**{n}. The patch-free head DLA.** The head DLA ranks heads like the single-head patch at the population level (r over heads " +
               lst(r1, lambda r: f"{r['dla_vs_single']['pop_pearson']:.2f}") + "; per row r " +
               lst(r1, lambda r: f"{r['dla_vs_single']['row_pearson']['median']:.2f}") + ", top-1 agreement " +
               lst(r1, lambda r: f"{r['dla_vs_single']['top1_agree']:.2f}") + ", top-10 overlap " +
               lst(r1, lambda r: f"{r['dla_vs_single']['top10_overlap']:.2f}") + "). The direct share of the top heads differs by task: "
               "CounterFact mover heads act partly through the downstream MoE (Qwen3 L40H13 single 0.19 vs DLA 0.12; Mixtral L18H4 0.15 vs "
               "0.06), the WinoGrande heads of the early hand-off layers act almost entirely indirectly (Mixtral L19H13, L13 heads: DLA ≈ 0), "
               "late heads write directly (Mixtral L24H22, L25H9, Qwen3 L46H24: DLA ≈ single). Σ head DLA over all heads and layers equals "
               "ext8's linear attention DLA (ratio " + lst(r1, lambda r: f"{r['head_dla_total_vs_ext8_direct']['ratio_of_sums']:.3f}") + ").\n")
    return "\n".join(out)


def caveats_text(s1, s2, S):
    r1 = [s1[k] for k in C.RUN_ORDER if k in s1]
    out = ["### Caveats\n"]
    out.append("- Final position only, denoising (sufficiency) only; components at the subject / option positions are not candidates. The "
               "ceiling 1 is reached by construction once all heads at all layers are patched (the corruption reaches the final position "
               "only through attention), so 'full repair' means: which small subset of final-position heads and experts suffices.")
    out.append("- bf16: single-head rescues are quantised at the logit spacing (one step = 0.125 at |logit| 16-32); identical single-head "
               "patches in different passes differ by a median of one step (this run vs the W5 head rows of ext7: " +
               lst([r for r in r1 if r.get("w5_rowmatch")], lambda r: f"r {r['w5_rowmatch']['r']:.2f}, median |diff| {r['w5_rowmatch']['median_abs_diff']:.3f}") +
               "). Sums over the 32 heads of a layer therefore carry ~0.5 logit of noise per row (row-level r(Σ heads, attn_layer) " +
               lst(r1, lambda r: f"{r['layer_additivity']['r_rows_all_layers']:.2f}") + "), while population means agree with the layer patch.")
    if s2:
        r2 = [s2[k] for k in C.RUN_ORDER if k in s2]
        out.append("- All heads at all layers as one multi patch reproduce the clean Δ in the population (ratio " +
                   lst(r2, lambda r: f"{r['ceilings']['all_heads']['ratio']:.3f}") + ") and per row up to bf16 noise (median / max |Δ − Δ_clean| " +
                   lst(r2, lambda r: f"{r['sanity_all_heads']['median_abs_dev']:.3f} / {r['sanity_all_heads']['max_abs_dev']:.2f}") +
                   "; the same-pass all-attention patch (attn_layer steps) shows the same envelope: o_proj rounding at every layer plus "
                   "batch-composition noise amplified by routing near-ties); all heads of one layer vs the attn_layer kind: max |ΔΔ| " +
                   lst([r for r in r2 if 'sanity_layer_heads' in r], lambda r: f"{r['sanity_layer_heads']['max_abs_dev']:.3f}") +
                   ". Prefill Δ of identical rows across passes: SD median " +
                   lst(r2, lambda r: f"{r['prefill_sd']['clean_median']:.2f}") + " logits. Greedy steps span passes (each step compares its "
                   "candidates within one pass).")
        out.append("- Greedy pools and the oracle ordering use the row's own single-component patches (in-sample, as ext8); the pools are "
                   "truncated (32 heads; Qwen3 32 experts, Mixtral all 64), so components outside a row's top-32 single heads can enter "
                   "only through the static orderings (which use all heads and all clean-active experts).")
        out.append("- The ext8 comparison curves are ext8's own passes on the same rows (its drop reference); static ext8 curves come from "
                   "results/tables/ext8_a1_curves.csv. Differences below ~0.02 of the drop are within bf16 pass-to-pass noise.")
    v = None
    vp = os.path.join(RESULTS, "verify_ext9_engine_olmoe.json")
    if os.path.exists(vp):
        vj = json.load(open(vp))["head_steps_vs_hf"]
        v = {"mixed_r": vj["denoise:mixed"]["effect_corr"], "mixed_max": vj["denoise:mixed"]["maxabs"],
             "all_r": vj["denoise:allheads_all_layers"]["effect_corr"]}
    out.append("- Engine: Step 1 used the attn_head kind (unchanged by ext9; regression identical); Step 2 uses the ext9 multi steps "
               "attn_head / heads_experts, verified on OLMoE against transformers hooks (results/verify_ext9_engine_olmoe.json" +
               (f": mixed head + expert sets effect r {v['mixed_r']:.3f}, max |ΔΔ| {v['mixed_max']:.3f}; all heads at all layers r {v['all_r']:.4f}; "
                f"one head step = attn_head kind bit for bit" if v else "") + ").")
    fd = ""
    if s2:
        cf = [s2[k] for k in ("cf_qwen3", "cf_mixtral") if k in s2]
        fd = (" First-donor rows only (sensitivity), mixed greedy r(10) / r(20): " +
              lst(cf, lambda r: f"{r['curves']['greedy_mix']['first_donor']['r10']:.2f} / {r['curves']['greedy_mix']['first_donor']['r20']:.2f}") + ".")
    out.append("- CounterFact rows are (case, donor) pairs (donor mean per case for population values); WinoGrande / IOI rows are "
               "directed cases with bootstrap over pairs." + fd)
    out.append("- IOI was run for Qwen3 only (its rows share the Qwen3 Step-2 passes at little extra cost); Mixtral IOI would have "
               "needed ~25 extra GPU minutes of separate passes and was left out of the budget.")
    return "\n".join(out) + "\n"


def files_text(S):
    gpu = gpu_minutes()[0]
    return ("### Files\n\n"
            "- Code: `moetrace/ext10_circuit.py` (run configs, task loading via `moetrace.ext8_addback`, IOI loader, step-2 candidates / "
            "pools / greedy state), `scripts/ext10_heads_run.py` (Step 1 driver: single heads + head / expert DLA, resumable, raw dump and "
            "CPU post-processing fallback, OOM chunk splitting), `scripts/ext10_heads_analyze.py`, `scripts/ext10_circuit_run.py` (Step 2 "
            "driver, both tasks of a model in every pass), `scripts/ext10_circuit_analyze.py`, `scripts/ext10_circuit_text.py`, chains "
            "`scripts/ext10_chain1.sh`, `scripts/ext10_chain2.sh`.\n"
            "- Step 1 runs: `results/{qwen3_str,mixtral_bos_str,wino_qwen3_str,wino_mixtral_bos_str,ioi_qwen3_s2io}_circuit/` "
            "(`head_rows_pNN.parquet` single-head + attn_layer rows, `head_dla_pNN.parquet`, `expert_dla_pNN.parquet`, "
            "`head_prefill_pNN.parquet`, `heads_state.json`, `run_meta.json`).\n"
            "- Step 2 runs: `results/{qwen3,mixtral}_circuit/` (`circuit_rows_pNN.parquet` spawn rows with task / fam / order / k / set / "
            "Δ / metrics, `circuit_prefill_pNN.parquet`, `circuit_state.pkl` = greedy paths, `run_meta.json`); smoke "
            "`results/qwen3_circuit_smoke/`.\n"
            "- Tables `results/tables/ext10_heads_*`, `results/tables/ext10_circuit_*`; figures `results/figures/ext10_heads_*`, "
            "`results/figures/ext10_circuit_*`; numbers `results/ext10_circuit_summary.json` (keys step1, step2)." +
            (f" GPU time of ext10 jobs ≈ {gpu:.0f} min." if gpu else "") + "\n")


def main():
    S = json.load(open(os.path.join(RESULTS, "ext10_circuit_summary.json")))
    s1 = S.get("step1", {})
    s2 = S.get("step2", {})
    txt = [summary_text(s1, s2), ""]
    txt.append("### What was run\n")
    txt.append("**Rows.** Exactly the ext8 validation rows: CounterFact STR (Direction 6 donors, paper IDs and split; (case, donor) rows, "
               "donor mean primary, first donor sensitivity) and WinoGrande option swap (ext7, directed cases, bootstrap over pairs), "
               "Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS, bf16; plus IOI (i) S2 → IO (ext7-controls pairs, main validation family = "
               "256 directed cases / 128 pairs) for Qwen3 as the attention-pole reference. Corruption = STR only; denoising (parent = "
               "corrupted run, source = clean run) at the final position.\n")
    txt.append(runs_text(S))
    txt.append(step1_text(s1))
    if s2:
        txt.append(step2_text(s2))
    txt.append(reading_text(s1, s2))
    txt.append(caveats_text(s1, s2, S))
    txt.append(files_text(S))
    os.makedirs(os.path.dirname(SEC), exist_ok=True)
    with open(SEC, "w") as f:
        f.write("\n".join(txt))
    print(f"wrote {SEC}")


if __name__ == "__main__":
    main()
