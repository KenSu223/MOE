**Summary.** Phase 4 tests the Phase-3 picture from four sides, on Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS, under symmetric token replacement (STR). The four questions are necessity, full repair, how the experts act, and replication.

**(1) Are the localised experts necessary, and only for their own task? Yes, but each carries only a few per cent of the clean margin, because the rest of the network compensates.**
- **Single-expert knockout costs.** Removing one expert from the router's menu at every position costs its own task:
  - Qwen3 L41E117 (WinoGrande): 0.033 [0.028, 0.037] of the clean margin;
  - Qwen3 L44E069 (CounterFact): 0.033;
  - Qwen3 L42E115 (CounterFact): 0.024;
  - Mixtral L20E000 (WinoGrande): 0.036.

  Each costs the other task nothing (|F| ≤ 0.001), and each ranks first among the experts of its layer. Double-dissociation contrasts are +0.065 (Qwen3) and +0.041 (Mixtral).
- **Joint knockout of each task's top 10 experts** costs 0.14–0.15 of the own task's margin, and 0.00 of the other task in Qwen3. Frequency-matched random sets cost ≈ 0.
- **Mixtral's CounterFact selections L19E002 and L21E001** are sufficient in patching but barely necessary (0.004).
- **Sufficiency is not necessity.** On the same prompts a knockout removes only 0.05–0.47 of the expert's own direct write to the logit difference. The rest of the network makes up the remaining 0.53–0.95 (self-repair).

**(2) How few components restore the answer fully? About ten, once attention heads are candidates.**
- **Experts alone** stay below the all-MoE ceiling: 0.41–0.84 of the drop.
- **Heads plus experts.** Adaptive greedy over heads ∪ experts at the final position restores 0.97 / 1.00 (CounterFact, Qwen3 / Mixtral) and 0.93 / 0.94 (WinoGrande) of the drop with 20 components. 80 % of the drop takes 8–10 components.
- **The tasks differ in what greedy picks first.**
  - **CounterFact:** mover heads first (Qwen3 L40H13, Mixtral L18H4 / L24H22), the CounterFact experts later; 6 of the first 10 picks are heads.
  - **WinoGrande:** the WinoGrande experts first (L41E117, L20E000), late heads later; 3 of 10 picks are heads.
  - **IOI:** heads only, 10 of 10.

**(3) Do the experts write the answer or compute it?**
- **Qwen3's experts are writers.** Their direct-logit share of the patch effect is 0.96–1.32. L44E069 writes 1.59 logits and later layers undo 0.39 of them.
- **Mixtral's are partly writers** (0.54–0.87).
- **The direct share rises with depth.** First-half experts act almost only indirectly and contribute ≤ 0.03 of the drop. This is why the patch-free DLA ranking matches the single-patch oracle.
- **Routing follows the local slot; the write needs the full context.** L41E117 fires at 66 % of copulas and 59 % of degree adverbs in wikitext, and at the context-free "The bag was too" prompts, but writes the right trigger only with the full sentence (DLA +0.54 vs +0.08). In vocabulary space its write is a property axis.
- **CounterFact experts route broadly but write relation-specifically.** L44E069 fires before 72 % of capitalised words, yet writes +1.08 logits for place relations vs +0.01 for occupations.

**(4) Do the Phase-3 claims replicate? The expert claims do; one attention claim does not.**
- **Add-back replicates** on held-out splits: ceilings, k80, greedy ≈ optimal, DLA ≈ oracle and overshoot are all within CI of Phase 3.
- **Mixtral without BOS** re-selects L20E000 and leaves the WinoGrande attention/MoE balance unchanged.
- **Revised: the role swap.** Phase 3's "the role swap moves the balance towards attention" was an item effect. On identical name items, role swap and option swap give the same attention share (Qwen3 0.304 vs 0.306, Mixtral 0.406 vs 0.414). The role-swapped prompt is exactly the option-swapped prompt with the two names exchanged. Name items lean towards attention, object items towards the MoE.

### What was run

Phase 4 of RESEARCH_PLAN.md used four sub-agents plus the coordinator, ≈ 4.4 GPU-h in total:
- ext9-knockout ≈ 105 min, including the engine work;
- ext10-circuit ≈ 79 min;
- ext11-writer ≈ 12 min;
- ext12-complete ≈ 70 min.

Engine E4 (ext9, additive, verified on OLMoE against transformers hooks, `results/verify_ext9_engine_olmoe.json`) added three things:
- **Route masks on prefill rows** (`PrefillSpec.route_mask`; reroute = router logit −inf before the softmax, zero = contribution dropped; all positions or the final position only). Agreement with transformers: effect r 0.95–0.98; masked-layer routing identical in 0.94 of (row, layer).
- **`attn_head` and `heads_experts` steps in `multi`.** One head step equals the `attn_head` kind bit for bit. Mixed head/expert sets agree with transformers at effect r 0.9992.
- **`DiagSpec.contrib_final_vectors`**, which sum to the MoE output to 8e-8.

`verify_olmoe.json`, `verify_ext5_engine_olmoe.json` and `verify_ext8_engine_olmoe.json` are identical to their pre-merge copies in every result field.

The four studies:
- **Direction 9 (knockout).** Targets = the patching selections plus each task's population top-1/3/5/10. Controls: same-layer experts and frequency-matched random sets. Evaluation, with selection items excluded:
  - the WinoGrande margin pool and all W1–W6 pairs;
  - the CounterFact clean scan;
  - IOI;
  - wikitext NLL;
  - variants: final-position-only and zero mode.
- **Direction 10 (circuits).** Every head at every layer as a single patch, plus a patch-free head DLA. Then exact joint patches: greedy over heads ∪ experts (20 steps), head-only greedy, and static mixed orderings, on the ext8 validation rows. IOI was run as the attention-pole reference (Qwen3 only).
- **Direction 11 (writers).** For every clean-active expert, the patch effect was split into direct and indirect parts. Routing and DLA were measured on WinoGrande, context-free, CounterFact, IOI and wikitext prompts, plus vocabulary projections of the expert writes.
- **Direction 12 (completeness).** Three checks:
  - 4a: add-back on the WinoGrande replication split and on a CounterFact fold swap;
  - 4b: the option swap on the role-swap items;
  - 4c: Mixtral without BOS on the 776 WinoGrande pairs (W2, W6, W4, add-back, sink flags).

### One table

**Final position, validation; fractions of the drop unless marked.**

| Task | Model | All-MoE ceiling | Heads + experts, greedy r(20) | k for 80 % of drop (heads + experts) | Heads among first 10 picks | Selected expert | Direct share of its patch effect | Knockout F own task / other task | Share of its direct write lost on knockout (zero mode) |
|---|---|---|---|---|---|---|---|---|---|
| CounterFact | Qwen3 | 0.53 | 0.97 [0.95, 1.00] | 8 | 6.2 | L44E069 | 1.32 [1.20, 1.43] | 0.033 / 0.001 | 0.30 [0.24, 0.36] |
| CounterFact | Qwen3 | | | | | L42E115 | 0.96 [0.88, 1.06] | 0.024 / −0.001 | 0.38 [0.29, 0.47] |
| CounterFact | Mixtral BOS | 0.41 | 1.00 [0.98, 1.03] | 9 | 6.4 | L21E001 | 0.87 [0.75, 1.03] | 0.004 / 0.000 | 0.05 [−0.07, 0.15] |
| CounterFact | Mixtral BOS | | | | | L19E002 | 0.71 [0.55, 0.94] | 0.004 / −0.001 | 0.19 [0.05, 0.34] |
| CounterFact | Mixtral BOS | | | | | L18E001 | 0.54 [0.43, 0.69] | 0.015 / 0.002 | (indirect expert; not a compensation measure) |
| WinoGrande | Qwen3 | 0.84 | 0.93 [0.91, 0.96] | 10 | 3.6 | L41E117 | 1.04 [0.99, 1.10] | 0.033 / −0.001 | 0.47 [0.37, 0.56] |
| WinoGrande | Mixtral BOS | 0.79 | 0.94 [0.93, 0.95] | 10 | 2.8 | L20E000 | 0.56 [0.52, 0.62] | 0.036 / −0.001 | 0.45 [0.38, 0.51] |
| IOI (i) | Qwen3 | −0.27 | 0.94 [0.93, 0.96] | 11 | 9.9 | – | – | – | – |

Sources:
- `results/sections/ext10_circuit.md` (ceilings from ext8; ext10's in-pass recomputation gives Qwen3 WinoGrande 0.86);
- `results/sections/ext11_writer.md` (direct share = Σ direct / Σ total over validation cases where the expert is routed);
- `results/sections/ext9_knockout.md` (F = fraction of the clean margin lost, all-position reroute, full item set);
- `results/tables/ext9_synthesis_selfrepair.md`.

### Reading

1. **Necessity, specificity and self-repair.**
   - **The specificity claim holds as necessity, not only as patching.** The experts that the final-position STR patches localise are necessary for their own task and not for the other. Each ranks first among the experts of its layer, and joint knockouts of the population top-10 cost 14–15 % of the margin while matched random sets cost nothing. The paper's "expert-aware" localisation therefore passes a test the paper did not run.
   - **The cost of a single expert is small** (2–4 % of the margin), much smaller than its patch effect: a knockout moves Δ by 0.04–0.25 of what the same expert restores under STR.
   - **The gap is compensation, not a weak write.** On the same prompts, removing the expert's contribution would cost its direct write (DLA of c_e: Qwen3 L41E117 +0.32, L44E069 +0.76, Mixtral L20E000 +0.32 logits). The actual knockout cost is 0.30–0.47 of that for the Qwen3 experts and Mixtral L20E000, and 0.05–0.19 for Mixtral's CounterFact writers L21E001 / L19E002. Downstream layers make up the rest, the self-repair / "Hydra effect" described for dense transformers (McGrath et al. 2023; Rushing & Nanda 2024).
   - **Zero mode costs as much as reroute,** so the replacement expert is not what compensates.
   - **The one expert that acts mostly indirectly is not compensated.** Mixtral L18E001 (direct share 0.54) loses at least its whole direct write, and it is the most necessary of Mixtral's CounterFact experts: compensation acts on direct writes to the logit.
   - **Practical consequence:** knockout and patching answer different questions, and the patch-based selections should be quoted with both.
   - **One knockout that is not specific:** Mixtral's population top-10 for CounterFact hurts WinoGrande more than CounterFact (0.047 vs 0.038). It contains two WinoGrande experts, L22E005 and L19E006; L19E006 is the BOS-sink expert, and sets containing it also do the most generic damage (wikitext NLL).

2. **Full repair is a small head + expert circuit at the final position.** The team asked how many experts restore the answer. Experts alone cannot restore it on CounterFact (ceiling 0.41–0.53). With heads as candidates, 8–10 components restore 80 % of the drop and 20 components restore 93–100 %, flipping the answer back in 94–100 % of cases.
   - **The composition places each task on the attention–MoE axis** without the degenerate joint game of Phase 3: IOI picks only heads, CounterFact mostly heads (the known mover heads first, then the CounterFact experts), WinoGrande mostly experts (its expert first, then late heads).
   - **Greedy beats the static orderings again** (AUC over log k 0.58–0.64 vs 0.49–0.57).
   - **The patch-free DLA ordering over heads ∪ experts is as good as the single-patch oracle on CounterFact but not on WinoGrande.** WinoGrande's decisive early heads act indirectly (DLA ≈ 0), the head-side counterpart of the writer/computer split in reading 3.

3. **Late experts are answer writers whose routing is set by the local slot.**
   - **Direct share and depth.** The direct share of an expert's patch effect rises with depth. At and above a crossover layer (Qwen3 L41–L42, Mixtral WinoGrande L22) experts write more than their net effect, and the last 2–3 layers' experts write against the answer. This is the MoE side of what Phase 3 called the late read-out.
   - **Routing vs content.** Routing is decided by the local token context: copula and degree-adverb slots for L41E117 and L20E000, "a name follows" for the CounterFact experts, which also fire at every IOI final token and write nothing there. The content of the write is decided by the full context: L41E117's DLA for the right trigger is +0.54 with the sentence and +0.08 without it, and L44E069 writes for place relations, not for occupations.
   - **"Expert specificity" is therefore a property of the write given the slot, not of the routing.** A routing-frequency criterion (the paper's recurrence gate) finds these experts because the slot recurs across a task's prompts.

4. **What replicates and what is revised.**
   - **Replicates:**
     - every add-back claim of Phase 3 on held-out splits (4a);
     - the WinoGrande expert and the attention/MoE balance without BOS (4c; the final token carries the sink state in 2 of 1,024 WinoGrande prompts, against 63 of 256 CounterFact prompts in Direction 3);
     - the W5 / F2 mover heads, found again by the all-layer head scan (Direction 10).
   - **Revised:** the Phase-3 ordering "WinoGrande role swap > option swap" in attention share. On identical items the two corruptions give the same decomposition (4b), so within WinoGrande the axis is names vs objects, not binding vs reference. The text in `ext7_controls.md`, `ext7_synthesis.md` and CLAUDE.md now says so. A genuinely different second corruption site for WinoGrande (Zhang & Nanda's Z7) remains open.

### Caveats and open items

- **Different passes.** Knockout and writer measurements are cross-pass comparisons, with bf16 per-item SD 0.2–0.4 logits and a knockout noise floor of F ≤ 0.008. The self-repair ratios are means over 300–3,300 prompts with bootstrap CIs. DLA freezes the final norm (single-expert error ≤ 1 %).
- **The indirect part is not decomposed.** "Indirect" cannot yet be split into later attention vs later MoE: the engine has no downstream freezing (path patching).
- **Mixtral IOI circuit not run.** The IOI circuit run is Qwen3 only (Mixtral would need about 25 GPU-min); IOI uses expert DLA as the expert pool proxy.
- **BOS-sink confound.** Mixtral sets containing L19E006 (the BOS-sink expert) may do generic damage through position 0. A mask that spares position 0 would need a new engine option.
- **Still not done:**
  - F3 (gradient attribution);
  - two-site add-back (subject position + final position);
  - a second WinoGrande corruption site on object items.

Files:
- Sections: `results/sections/ext9_knockout.md`, `ext10_circuit.md`, `ext11_writer.md`, `ext12_complete.md`, this file.
- Numbers: `results/ext9_knockout_summary.json`, `ext10_circuit_summary.json`, `ext11_writer_summary.json`, `ext12_complete_summary.json`.
- Self-repair comparison: `scripts/ext9_synthesis_selfrepair.py` → `results/tables/ext9_synthesis_selfrepair.md`.
- Plan: RESEARCH_PLAN.md "Phase 4".
