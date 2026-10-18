"""ext14 circuit (p6-circuit): writes results/sections/ext14_circuit.md from results/ext14ci_summary.json and the
ext14ci tables (C1 neurons in experts + Mistral-7B analogue + upcycling, C2 RelP pilot, C3 concept circuits).
Every number is read from disk; parts whose inputs are missing are marked pending.

Usage: python scripts/ext14ci_text.py
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
from moetrace.models import RESULTS

TAB = os.path.join(RESULTS, "tables")
OUT = os.path.join(RESULTS, "sections", "ext14_circuit.md")
# hand-written readings of C2 / C3 (filled in after the runs; the numbers they quote come from the tables included below)
NOTES = {
    "summary_c2": [
        "- **C2 (RelP pilot, OLMoE and Mistral-7B, 48 STR cases per task): the port works, and its value is at positions C1 cannot see.** "
        "At the final position RelP ranks a focal expert's / MLP's neurons about as well as forward DLA (k80 1–8 vs 1–19 neurons; it is "
        "clearly better only where the MLP acts indirectly, Mistral WinoGrande subset r 0.95–0.96 vs 0.61–0.78). But on CounterFact "
        "three quarters of the exact MLP / MoE-output effect sits at the subject and relation positions (final-position share 0.24 / 0.26), "
        "and neuron circuits restricted to the final (+ swapped) position recover at most 0.23 (OLMoE) / 0.50 (Mistral) of the drop at "
        "10,000 nodes, while all-position circuits reach 80 % with 997 / 478 nodes. On WinoGrande the final position dominates "
        "(share 0.60–0.67; final-only circuits 0.80–0.87). RelP is only moderately accurate off the final position (r vs exact 0.17–0.65), "
        "and in OLMoE 54–56 % of single sublayer patches switch a later top-k set (errors double there).",
        "- **DP3 evidence (user decision):** a gradient method adds (i) the non-final positions, essential for factual recall, and (ii) "
        "joint multi-layer sets; it does not add much at the final position. A forward alternative for (i) exists without F3: neuron-level "
        "patches at non-final positions in the position executor (`ext5_subject`), ranked by forward scores. Recommendation in the "
        "C2 reading below."],
    "c2_reading": [
        "", "**Reading and DP3 evidence.**",
        "- *Validity of the port.* The functional forward equals transformers (tiny fp32 exact; bf16 max |ΔΔ| 0.05 Mistral, 0.16 OLMoE "
        "(one routing flip)); RelP leaves the forward values unchanged. Against exact sublayer patches, RelP is accurate at the final "
        "position (r 0.88–0.93; per-layer curves r 0.96–0.99) and moderately accurate elsewhere (r 0.60–0.65 CounterFact, 0.17–0.61 "
        "WinoGrande; slopes 1.0–1.9, i.e. it over-states). Plain AtP is as good at the final position and worse off it on CounterFact "
        "(0.20 / 0.47 vs 0.60 / 0.65). Freezing routing costs little on average (OLMoE r 0.87 on cells that switch a later top-k set vs "
        "0.93 on cells that do not) although 54–56 % of cells switch.",
        "- *What C1 already gives.* For one expert / MLP layer at the final position the forward scores are nearly enough: RelP k80 1–8, "
        "DLA 1–19, magnitude 9–46 (random 740 / 10,500). RelP helps where the component acts through later layers (Mistral WinoGrande: DLA "
        "covers 0.32–0.50 of the MLP's patch, RelP 0.59–0.78; neuron-subset r 0.95–0.96 vs 0.61–0.78). For the most indirect expert we have "
        "(Qwen3 L37E032) the exact single-neuron oracle does not beat the magnitude ranking (k80 36 vs 17 on 96 rows): single-neuron "
        "effects are not additive there (their sum is −3.2× the expert's patch), so no per-neuron score, gradient or exact, ranks them well.",
        "- *What C1 cannot give.* (a) Non-final positions: CounterFact's neuron circuit lives mostly at the subject (top-1000 RelP nodes: "
        "31–38 % between the swapped token and the final token, mean layer 10 / 20 of 16 / 32); final-position circuits cap at 0.23 / 0.50 "
        "of the drop. (b) Joint multi-layer sets at the final position: 43–296 nodes for 80 % of the drop, with RelP and DLA rankings "
        "within 10 % of each other.",
        "- *Mean-ablation faithfulness (Arora et al.'s criterion)* is not reached by per-case circuits of practical size in either model "
        "(F ≤ 0.41 at 10,000 nodes; 0.8 at 155k–382k nodes or never), while completeness is high (removing the top 1,000 nodes leaves 0.03–0.23 "
        "of the effect). Mean-ablating every other neuron at every position takes the model off-distribution; the STR criterion, which "
        "only replaces what differs between the twins, is the one our project uses and there the circuits are compact.",
        "- *Recommendation for DP3.* Building F3 (reverse layer streaming for Qwen3-30B / Mixtral) is justified only if the goal is "
        "gradient-ranked circuits over ALL positions at scale. For the questions asked so far (which neurons of a known expert, which "
        "position carries what), a forward extension is cheaper and exact: neuron-subset patches at a chosen non-final position (the "
        "`expert_neurons` vector applied by the position executor), with candidates from forward scores (DLA / magnitude) or from layer-level "
        "position grids that already exist (Direction 6b, Phase-6 T4). F3 would add speed of discovery, not new kinds of evidence, and its "
        "estimates off the final position need exact validation anyway (r 0.17–0.65 here)."],
    "c3_reading": [
        "", "**Reading.** The knockout matrix is the held-out test: circuits are chosen on the source validation split and knocked out on "
        "the discovery split, every circuit in every task. A circuit is concept-specific when its own tasks lose much of the clean margin "
        "and the other concepts' tasks (and the control) lose about what matched random sets lose. N3 is tested by knocking out the heads "
        "and the experts of each circuit separately (selectivity table). Edges are small compared with the components' own effects: the "
        "components act mostly in parallel on the final residual (a mover head → late expert path, e.g. Qwen3 L40H13 → L44E069, carries "
        "≈ 0.03 of the drop)."],
    "caveats": [
        "**Caveats.**",
        "- C1 is final-position only (as the engine's `expert_neurons` kind); CounterFact values are donor means; Mistral-7B uses the Mixtral items "
        "filtered by its own margin, so its case sets are subsets of Mixtral's. Neuron rankings are per case; a fixed population set is much less "
        "efficient (case-specific neurons).",
        "- C2 is a pilot (48 cases per task, two small models): the RelP rules are our re-implementation from the paper's description, not "
        "Transluce's code; circuits are per case (no dataset-level aggregation, prompts differ in length); MoE mean ablation uses the routed-token "
        "mean (unconditional means wreck OLMoE); gradients use frozen routing while every exact check routes live.",
        "- C3 uses existing rows (scope cut): head layers measured differ between runs (IOI, CounterFact Phase-4, option swap: all layers; Phase-5/6 "
        "cells and the new COUNTRY sets: 5–6 top attention layers), IOI has no expert singles, COUNTRY experts are scored by DLA, selection is on the "
        "source validation split and evaluation on the discovery split; some sets are small (Phase-6 E1 names 16–22 pairs); 2 random draws per set; "
        "knockouts act on the clean run only (heads zeroed, experts rerouted at every position). The concept rule excludes components that also "
        "rank high for another concept, which can remove a task's strongest components (IOI's L42H10 / L43H24 are shared with the size / state "
        "tasks, so the PERSON-ROLE circuit is the name-attribute part, not IOI's)."],
}


def tab(name):
    p = os.path.join(TAB, name)
    return open(p).read().strip() if os.path.exists(p) else "_(pending)_"


def k(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "never"
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def f2(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def c1_part(s):
    c1 = s.get("c1")
    if not c1:
        return ["### C1 Neurons inside the selected experts", "", "_(pending)_"], []
    E = {(e["task"], e["expert"]): e for e in c1["experts"]}
    moe = [e for e in c1["experts"] if not e["task"].startswith("mistral")]
    dense = [e for e in c1["experts"] if e["task"].startswith("mistral")]
    qw = [e for e in moe if e["task"].startswith("qwen3")]
    mx = [e for e in moe if e["task"].startswith("mixtral")]
    best = lambda e: min(e["curves"]["dla_case"]["k80"], e["curves"]["mag_case"]["k80"])
    n1 = all(best(e) <= 50 for e in qw) if qw else None
    n2 = all(best(e) <= 200 for e in mx) if mx else None
    ind = E.get(("qwen3_f1", "L37E032"))
    orc = c1.get("oracle", {}).get("qwen3_f1|L37E032")
    sums = []
    sums.append(f"- **C1: a handful of neurons carry a selected expert's effect.** Patching the top neurons of an expert (by their own direct "
                f"logit attribution, final position) restores 80 % of the expert's own patch with "
                + ", ".join(f"{e['expert']} {k(e['curves']['dla_case']['k80'])}" for e in qw if e["expert"] != "L37E032")
                + f" of 768 neurons (Qwen3) and " + ", ".join(f"{e['expert']} {k(e['curves']['dla_case']['k80'])}" for e in mx)
                + " of 14,336 (Mixtral); a random order needs ~590 / ~10,300. Pre-registered N1 (≤ 50 of 768, Qwen3) "
                + ("holds" if n1 else "fails") + " (for L37E032 only with the magnitude score), N2 (≤ 200 of 14,336, Mixtral) "
                + ("holds" if n2 else "fails") + ".")
    if ind:
        sums.append(f"- **The indirect expert is the exception.** Qwen3 L37E032 (attribute → entity) writes almost nothing to the logits "
                    f"(direct share {f2(ind['direct_share'])}), so its neuron DLA is uninformative (k80 {k(ind['curves']['dla_case']['k80'])}); "
                    f"ranking by activation change × |W_down| needs {k(ind['curves']['mag_case']['k80'])}"
                    + (f", the exact single-neuron oracle {k(orc['curves']['oracle']['k80'])}" if orc else "") + ".")
    if dense:
        sums.append(f"- **Dense analogue: Mistral-7B's MLP is as sparse.** On the same items, 80 % of an MLP layer's rescue (L18–L21) takes "
                    + ", ".join(f"{e['expert'][:3]} {k(e['curves']['dla_case']['k80'])}" for e in dense if e["task"] == "mistral7b_cf")
                    + " neurons on CounterFact and " + ", ".join(f"{e['expert'][:3]} {k(e['curves']['dla_case']['k80'])}" for e in dense if e["task"] == "mistral7b_wino")
                    + " on WinoGrande.")
    mp = c1.get("mapping", {})
    if mp:
        m20 = mp.get("L20E000~L20E000")
        sums.append(f"- **Mixtral's experts are upcycled Mistral-7B MLPs, neuron for neuron.** Every Mixtral L20 expert's neuron i best matches "
                    f"Mistral-7B L20 MLP neuron i (100 % of 14,336, mutual), and the experts' top task neurons are the dense model's top neurons: "
                    + (f"L20E000's top-10 WinoGrande neurons hold Mistral ranks {', '.join(map(str, m20['mixtral_top10_rank_in_mistral']))}" if m20 else "")
                    + ".")
    L = ["### C1 Neurons inside the selected experts", "",
         "**Method.** For each selected expert on its own task (STR rows, validation split; CounterFact rows = (case, donor), averaged per "
         "case; WinoGrande rows = directed cases), the engine records the expert's intermediate activations a = SiLU(W_gate x) ⊙ (W_up x) "
         "at the final position in the clean and corrupted runs (`DiagSpec.expert_neurons`). Scores per neuron: "
         "d_i = w_clean a_i,clean − w_corrupt a_i,corrupt (w = the routing weight, 0 when not routed); DLA_i = d_i (W_down[:, i] · u) / "
         "rms(h_final, corrupted), u = γ ⊙ (W_U[answer] − W_U[contrast]) (final norm frozen, as the expert DLA of Phase 3/4; Σ_i DLA_i = the "
         "slot DLA, r = 1.0000); magnitude_i = |d_i| · ‖W_down[:, i]‖. Validation is exact: kind `expert_neurons` sets a neuron subset to its "
         "clean values inside the corrupted run, everything downstream recomputed; the expert's own single-expert patch is 100 %. "
         "Rankings: the row's own DLA (per case), magnitude, one fixed order from the discovery-split mean DLA, two random orders. "
         "Rows where the expert is routed in neither run contribute 0 to numerator and denominator.", "",
         tab("ext14ci_c1_neurons.md"), "", "![neurons per expert](../figures/ext14ci_c1_curves.png)", "",
         "**DLA vs exact patching, and how shared the neurons are across cases.**", "", tab("ext14ci_c1_agreement.md"), ""]
    rows = []
    for e in c1["experts"]:
        rows.append(e)
    L += ["Reading:",
          "- One or two neurons already carry half of a direct expert's effect (k50 ≈ 1–4); a positive-DLA ordering then overshoots the "
          "expert's net effect (curves above 1) because the expert's remaining neurons write against the answer — the expert's patch is a "
          "sum of a few large positive and many small negative terms.",
          "- The top neurons are case-specific: the mean Jaccard overlap of two cases' top-10 sets is 0.01–0.14 (random 0.0003–0.007), and "
          "one fixed neuron set chosen on the discovery split needs far more neurons for 80 % (column 'fixed set'). The exception is "
          "Qwen3 L41E117 on the option swap (fixed set 7.5; three neurons sit in at least half of the cases' top-10), the most "
          "'single-function' expert of the set.",
          "- Neuron DLA predicts subset patches well for experts that write (r 0.77–0.95, slopes 0.7–1.1) and not for the indirect "
          "L37E032 (r 0.14)."]
    if c1.get("oracle"):
        L += ["", "**Exact single-neuron oracle (Qwen3 L37E032, the indirect expert).** Every one of the 768 neurons patched alone on 96 "
              "validation rows, then top-k sets ordered by these exact single effects ('oracle'); the other rankings on the same rows were "
              "measured up to k = 400 only ('never' = below 80 % at k = 400).", "", tab("ext14ci_c1_oracle.md"), "",
              "The exact single-neuron effects of L37E032 do not add up (their sum is −3.2× the expert's own patch) and are spread over "
              "~300 neurons per row, so ordering by them is no better than ordering by the activation change × |W_down| (k80 36 vs 17): an "
              "expert that acts through later layers is a collective, not a sum of independent neurons."]
    ds = c1.get("dense_sweep", {})
    if ds:
        L += ["", "**Dense analogue (Mistral-7B, BOS).** Items: the Mixtral CounterFact / WinoGrande STR rows (identical tokenizer), kept where "
              "Mistral passes its own margin (clean Δ ≥ 1, corrupted Δ ≤ −1; " + ", ".join(
                  f"{t.split('_')[1]} {d['n_cases']} of {d['n_val_cases_total']} validation cases" for t, d in ds.items()) + "). The MLP-layer "
              "patch peaks at " + ", ".join(f"L{d['argmax_layer']} ({t.split('_')[1]})" for t, d in ds.items()) + " (normalised MLP rescue at L18–L21: "
              + "; ".join(f"{t.split('_')[1]} " + ", ".join(f2(x) for x in d["mlp_norm_by_layer"][18:22]) for t, d in ds.items())
              + "). The neuron counts are in the table above (rows Mistral-7B; 'E000' = the dense MLP): the dense MLP of one layer is as sparse as a Mixtral expert on "
              "the same items (CounterFact 1.2–3.4 neurons for 80 % vs 3.8–9.9; WinoGrande 7–19 vs 17)."]
    up = s.get("upcycle")
    if up:
        mt = up.get("match", [])
        L += ["", "**Upcycling cross-check (weights only, CPU).**",
              f"- Attention: per-layer cosine Mistral-7B ↔ Mixtral q {f2(up['attn_cos_median']['q'])}, k {f2(up['attn_cos_median']['k'])}, "
              f"v {f2(up['attn_cos_median']['v'])}, o {f2(up['attn_cos_median']['o'])} (medians over 32 layers; minima "
              f"{f2(up['attn_cos_min']['q'])}–{f2(up['attn_cos_min']['v'])}; highest in L7–L16, lowest in the first and last layers); RMSNorm "
              f"weights {f2(up['attn_cos_median']['ln1'], 3)} / {f2(up['attn_cos_median']['ln2'], 3)}; embeddings {f2(up['globals']['embed']['cos'])}, "
              f"LM head {f2(up['globals']['head']['cos'])}, final norm {f2(up['globals']['final_norm']['cos'], 3)}. Mixtral's attention matrices are "
              "≈ 2.8× larger in norm (relative difference ≈ 2.1) but point the same way: continued training from a common parent, not independent models.",
              "- Expert FFNs: the cosine between expert e's neuron i and the dense MLP's neuron i (gate rows, up rows, down columns) is "
              f"{min(up['ffn_gate_median_by_layer'].values()):.2f}–{max(up['ffn_gate_median_by_layer'].values()):.2f} (median over neurons, gate rows, "
              "by layer; highest mid-depth). At L20, for all 8 experts the best-matching Mistral neuron of every Mixtral neuron is the SAME index "
              f"({', '.join(f2(m['identity_rate']) for m in mt)}; mutual best matches {', '.join(f2(m['mutual_rate']) for m in mt)}), matched cosine "
              f"{min(m['match_cos_median'] for m in mt):.2f}–{max(m['match_cos_median'] for m in mt):.2f} vs {min(m['shuffled_match_cos_median'] for m in mt):.2f}–"
              f"{max(m['shuffled_match_cos_median'] for m in mt):.2f} for shuffled weights. Every expert is a copy of the dense MLP that drifted, with the neuron "
              "order kept (Lo et al. 2024 report cos ≈ 0.6 at the matrix level).",
              "- Function: under that identity map the experts' top task neurons are the dense model's top neurons (per-neuron population DLA "
              "on the same items):", "",
              "| Mixtral expert ~ Mistral MLP | common cases | Pearson r (Spearman) | top-10 / top-50 / top-200 overlap (chance) | Mistral rank of the expert's top-10 |",
              "|---|---|---|---|---|"]
        for kk, m in mp.items():
            L.append(f"| {kk.replace('~', ' ~ ')} | {m['n_cases_common']} | {f2(m['pearson'])} ({f2(m['spearman'])}) | {m['top10_overlap']} / "
                     f"{m['top50_overlap']} / {m['top200_overlap']} ({m['top10_expected']} / {m['top50_expected']} / {m['top200_expected']}) | "
                     f"{', '.join(map(str, m['mixtral_top10_rank_in_mistral']))} |")
        L += ["", "Low Spearman with high Pearson: most neurons have DLA ≈ 0 in both models (ties in the bulk); the large-effect neurons coincide."]
    return L, sums


def c2_part(s):
    c2 = s.get("c2", {}).get("runs", [])
    if not c2:
        return ["### C2 RelP on the neuron basis (pilot)", "", "_(pending)_"], []
    L = ["### C2 RelP on the neuron basis (pilot: OLMoE-1B-7B, Mistral-7B)", "",
         "**Method.** `moetrace/ext14_circuit_relp.py` re-implements the forward pass over the transformers weights (verified: = HF in fp32 "
         "on tiny checkpoints; plain gradients = finite differences; bf16 Δ within the engine's noise of HF) with RelP's linearisation for the "
         "backward pass (Arora et al. 2026; Rezaei Jafari et al. 2025): RMSNorm denominators frozen (incl. OLMoE's q/k norms), attention "
         "pattern frozen, SiLU's sigmoid gate frozen (identity rule), half rule on SiLU(g) ⊙ u. MoE port: top-k set and routing weights "
         "frozen (q_e = w_e a_e, w_e a constant), so experts routed in only one run of a pair are scored through the residual gradient "
         "W_down_eᵀ ∂Δ/∂h (the gradient exists for every expert). Nodes = neuron × position (positions from the swapped token on for STR; "
         "OLMoE: all 64 experts, Mistral: the MLP). Scores: relp_o = (q_clean − q_corrupt) · ∇_RelP Δ at the corrupted run, relp_c at the "
         "clean run, atp_o = plain gradient, mean_c = (q_clean − q̄) · ∇_RelP Δ at the clean run (Arora's mean-ablation baseline; MoE: q̄ = "
         "the mean where the expert is routed, applied only to experts routed in the live run). Exact checks in the same forward: "
         "(V3) the MLP / MoE output at every (layer, position ≥ swap) set to its clean value; (V1) top-k neuron sets of a focal "
         "expert / MLP layer at the final position (OLMoE WinoGrande L12E011 / L13E040, CounterFact L13E056; Mistral L18–L21); (V5) top-k "
         "final-position nodes over all layers patched jointly; circuits = top-N nodes by |score| per case, evaluated by STR denoising "
         "(corrupted input, circuit := clean) and Arora's mean-ablation faithfulness / completeness, with node pools all positions / "
         "final + swap position / final position only. Routing switches (OLMoE) = a later layer's final-position top-k set differs from "
         "the unpatched corrupted run.", "",
         tab("ext14ci_relp_validation.md"), "", tab("ext14ci_relp_circuits.md"), "", "Where the top nodes sit:", "", tab("ext14ci_relp_positions.md"),
         "", "![RelP circuits](../figures/ext14ci_relp_circuits.png)"]
    return L, []


def c3_bullets(s):
    """summary bullets + reading for C3 computed from the summary (knockout specificity, add-back, AUROC)."""
    c3 = s.get("c3", {})
    out_sum, out_read = [], []
    n3 = []
    for proto, lab in (("qwen3", "Qwen3"), ("mixtral_bos", "Mixtral")):
        r = c3.get(proto)
        if not r or "knockout_specificity" not in r:
            continue
        S = {(x["concept"], x["part"]): x for x in r["knockout_specificity"]}
        circ = r["circuits"]
        ab = {}
        for x in r.get("addback", []):
            if x.get("task_concept") == x["concept"]:
                ab.setdefault((x["concept"], x["part"]), []).append(x["R"])
        parts = []
        for conc in ("SIZE_STATE", "COUNTRY", "PERSON_ROLE"):
            c = circ.get(conc, {})
            if c.get("insufficient") or (conc, "all") not in S:
                parts.append(f"{conc} insufficient")
                continue
            a, h, e = S[(conc, "all")], S.get((conc, "heads")), S.get((conc, "experts"))
            ctl = a.get("F_control")
            ctl_txt = ""
            if ctl is not None and np.isfinite(ctl):
                ctl_txt = (f", the CounterFact control {ctl:.2f}" + (" (as much as its own tasks: a CounterFact-recall circuit, not a country "
                           "circuit)" if ctl >= 0.5 * a["F_own"] else ""))
            parts.append(f"{conc} ({len(c['heads'])} heads + {len(c['experts'])} experts): own tasks lose {a['F_own']:.2f} of the clean margin "
                         f"(net of random sets), other tasks {a['F_other_abs']:.2f} on average{ctl_txt}; add-back restores "
                         + "/".join(f"{v:.2f}" for v in ab.get((conc, "all"), [])) + " of the drop")
            if h and e and np.isfinite(h["selectivity"]) and np.isfinite(e["selectivity"]):
                n3.append((lab, conc, e["selectivity"], h["selectivity"], e["F_own"], h["F_own"]))
        out_sum.append(f"- **C3 {lab}:** " + "; ".join(parts) + ".")
    au = [(lab, c3[p]["auroc_home"]) for p, lab in (("qwen3", "Qwen3"), ("mixtral_bos", "Mixtral")) if c3.get(p, {}).get("auroc_home")]
    if n3:
        more = sum(1 for x in n3 if x[2] > x[3])
        out_sum.append(f"- **N3 (experts more concept-specific than heads): {'holds' if more == len(n3) else ('mostly holds' if more > len(n3) / 2 else 'fails')}** "
                       f"in {more} of {len(n3)} circuit comparisons (selectivity F_own / (|F_own| + mean |F_other|), experts vs heads: "
                       + "; ".join(f"{l} {c} {e:.2f} vs {h:.2f}" for l, c, e, h, _, _ in n3) + "). The descriptive AUROC of single-patch effects "
                       "(in-sample, only where a component was measured) points the other way: "
                       + "; ".join(f"{l} heads {a['heads_median']:.2f} vs experts {a['experts_median']:.2f}" for l, a in au if a.get("heads_median") is not None)
                       + " — heads rank higher within their home tasks, but knocking them out also hurts other tasks.")
    return out_sum, out_read


def c3_part(s):
    c3 = s.get("c3", {})
    if not c3:
        return ["### C3 Concept circuits", "", "_(pending)_"], []
    L = ["### C3 Concept circuits (existing rows; scope cut 2026-10-17)", "",
         "**Design.** Concepts and tasks (≥ 2 tasks per concept, the smallest set the existing rows allow): SIZE / STATE = the Phase-3 "
         "WinoGrande option swap (≈ Dial-E E3 entity → attribute) and Phase-6 E1 entity → attribute, both restricted to size / state "
         "answers; COUNTRY = CounterFact P17 and P27 + P495 (new STR pair sets, NEED HEADS cf_country); PERSON-ROLE = IOI (S2 → IO), Phase-5 "
         "F1 and Phase-6 E1 attribute → entity on name items; control = CounterFact language relations. Component effects = existing single "
         "head / expert patches at the final position (rescue / drop) on the source VALIDATION split (where the Phase-4 all-layer head runs "
         "live; COUNTRY experts by per-expert DLA); top set of a task = top-10 heads + top-10 experts; concept circuit = in the top sets of ≥ 2 "
         "of the concept's tasks and in no top set of another concept's task. Held-out test on the source DISCOVERY split (GPU): knockout of "
         "the circuit (all / heads only / experts only; heads zeroed by `head_mask`, experts rerouted by `route_mask`) in the clean run of every "
         "task, F = fraction of the clean margin lost, minus the mean of 2 matched random sets (same layers, same counts); add-back of the set "
         "(one multi spawn per case, denoising); edges A → B = rescue(A) + rescue(B) − rescue(A ∪ B) (the part of A's effect that runs "
         "through B). Head layers measured differ between runs (IOI, CounterFact, option swap: all layers; Phase-5/6 cells: their 5 top "
         "attention layers), and IOI has no expert singles, so a component can only enter the top sets where it was measured."]
    for proto in ("qwen3", "mixtral_bos"):
        if proto in c3:
            L += ["", tab(f"ext14ci_c3_{proto}.md")]
    return L, []


def main():
    s = json.load(open(os.path.join(RESULTS, "ext14ci_summary.json"))) if os.path.exists(os.path.join(RESULTS, "ext14ci_summary.json")) else {}
    l1, s1 = c1_part(s)
    l2, s2 = c2_part(s)
    l3, s3 = c3_part(s)
    b3, r3 = c3_bullets(s)
    extra = NOTES
    head = ["**Summary.** P6-5 asks whether the expert-level picture goes down to neurons, whether gradient circuit discovery on the "
            "neuron basis (RelP; Arora et al. 2026) can be ported to an MoE and adds something, and whether heads and experts form circuits "
            "that belong to a concept rather than a task."] + s1 + extra.get("summary_c2", []) + b3 + extra.get("summary_c3", [])
    body = head + [""] + l1 + [""] + l2 + extra.get("c2_reading", []) + [""] + l3 + extra.get("c3_reading", []) + [""] + extra.get("caveats", []) + [
        "", "**Files.** `moetrace/ext14_circuit.py`, `ext14_circuit_an.py`, `ext14_circuit_relp.py`, `ext14_circuit_c3.py`; "
        "`scripts/ext14ci_*` (`ext14ci_all.py` re-runs every analysis and this section); runs `results/p6ci_c1_<task>/`, "
        "`results/p6ci_relp_<model>_<task>/`, `results/p6ci_c3_*`, `results/p6ci_upcycle/`; tables `results/tables/ext14ci_*`; "
        "summary `results/ext14ci_summary.json`."]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(body) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
