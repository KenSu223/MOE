**Summary.** Phase 6 answers the team's feedback on Phase 5 with one pre-registered dial on the same WinoGrande items, a ladder of different tasks graded in advance, dense twins of both MoE families, a dense capacity series, and circuits down to neurons. Everything is Stage R, i.e. before the gated Mixtral-8x22B stage.

**Setup**
- 8 models, STR activation patching at the final token. Readout: the attention share of the repair = AUC+(attention) / (AUC+ attention + AUC+ MoE or MLP), the Phase-5 measure.
  - MoE: Qwen3-30B-A3B, Mixtral-8x7B, Qwen2-57B-A14B (fp32).
  - Dense: Mistral-7B, Qwen2-7B (fp32), Qwen3-4B / 8B / 14B.
  - Qwen3-1.7B: behaviour only.
- 102 analysed cells.
- 45 predictions committed before any GPU run (4143e09).

**1. The answer type sets the split, and stating the binding moves it — in every model except Qwen3-30B-A3B.** Same twins, frame held fixed:
- **attribute → entity** (answer = an entity): attention-heavy at every level of evidence (0.48–0.83).
- **entity → attribute** (answer = an attribute): MoE / MLP-heavy when the attribute must be inferred from the event (E3: 0.18–0.34).
- When the attribute is stated next to its entity (E1), the entity → attribute share rises in 7 of 8 models:
  - Qwen3-4B 0.31 → 0.58, Qwen3-8B 0.28 → 0.55, Qwen3-14B 0.21 → 0.46;
  - Qwen2-7B 0.21 → 0.37, Qwen2-57B 0.21 → 0.38;
  - Mistral-7B 0.31 → 0.49, Mixtral 0.34 → 0.50.
  - The direction gap halves (e.g. Qwen3-4B +0.37 → +0.12).
- Qwen3-30B-A3B does not move (0.18 → 0.22; gap +0.60 at E3, +0.59 at E1, own sets; +0.51 → +0.57 on common twins; H-type).
- So the Phase-5 "direction" effect is partly answer type and partly the inference the attribute needs. The pre-registered best guess (the gap closes) fails for Qwen3-30B-A3B, where it was testable, and is approached by the other models.

**2. Dense parent vs MoE child: no difference at matched architecture.**
- Mistral-7B → Mixtral-8x7B: identical prompts in all 10 cells, |MoE − dense| ≤ 0.10, cell order Spearman 0.98.
- Qwen2-7B → Qwen2-57B-A14B: 6 of 7 cells (antonym −0.11), Spearman 0.89.
- MoE ≥ dense + 0.10 in no cell.
- The division of labour between attention and the FFN is a property of the task, not of dense vs sparse FFNs (H-invariant, D0 / D1 / D4 held; D2 / D3 failed).
- Inside Qwen2-57B the always-on shared expert carries 0.43–0.77 of the MoE rescue.
- Mixtral's experts are Mistral-7B's MLP neuron for neuron (same index, top task neurons shared).

**3. Capacity.**
- The dense Qwen3 series 4B → 8B → 14B is flat (D5 held).
- Qwen3-30B-A3B departs from it in both directions: copy cells higher (IOI 0.92 vs 0.73–0.74; E1 attribute → entity 0.81 vs 0.66–0.70), entity → attribute lower (0.18 vs 0.21–0.31).
- With flat dense twins and no Qwen3 dense twin, this is a property of that model, not a capacity trend. The 8x22B test of the Mistral series is the gated Stage 3.

**4. The ladder.**
- The top of the predicted order holds: IOI 0.71–0.92, induction 0.67–0.77, explicit binding 0.65–0.79.
- The bottom does not: antonyms (grade 1) sit at 0.35–0.59, above object property in 7 of 8 models. The cue word is one token before the answer, so "answer stored in the weights" does not imply a low final-token attention share.
- Spearman(grade, share) is 0.55–0.87 (L9 failed); without antonyms 0.79–0.93.

**5. Associativity is a weak dial.**
- Qwen3 attribute → entity: conflict 0.83 ≥ neutral 0.81 > agree 0.79 > prior only 0.73 (A1 held, spread 0.10).
- In conflict items the MoE writes the prior answer against the stated one (−0.30 of the drop) and attention overrides it.
- An explicit statement beats event inference in 8 of 9 models (Set C; Mistral 0.57).

**6. Circuits.**
- **Neurons:** a handful of neurons carry 80 % of a selected expert's effect (pooled over cases: Qwen3 2.1–4.3 of 768, Mixtral 3.8–17 of 14,336; per-case medians Qwen3 2.1–5.5, Mixtral 4.0–23; N1 / N2 held), but different neurons for different cases. The indirect Qwen3 L37E032 needs a magnitude ranking.
- **RelP on MoE (OLMoE, Mistral pilot):** works, but adds little at the final position. Its value lies at the subject / relation positions (three quarters of CounterFact's effect).
- **Concept circuits (C3):** knocking out a concept's heads + experts removes, on its own tasks vs other tasks:
  - Qwen3: size / state 0.43 / 0.09, country 0.70 / 0.16, person role 0.21 / 0.04;
  - Mixtral: 0.32 / 0.11, 0.69 / 0.25, 0.39 / 0.02.
  - Experts are more concept-specific than heads in 3 of 6 concept × model comparisons (N3 partly): both size / state circuits and Qwen3 person role. In the country circuits and in Mixtral's person-role circuit (0.82 vs 0.88) the experts are less specific than the heads.

**7. Robustness and causal checks.**
- Shares are metric-invariant (R1, Spearman ≥ 0.97).
- **Knockouts.** On E1 attribute → entity, knocking out the top-3 heads / experts lowers the attention / MoE repair 21–31 % beyond matched random sets. On E3 entity → attribute, where attention does little, they lower it 12–19 %, just under the pre-registered 20 % bar. R3 held 1 / 4, R4 2 / 4.
- **Swapped evidence (R2 held):** moving the corruption into the evidence changes the share by only 0.03 / 0.05.
- The share itself moves only −0.02 / −0.07: the split is a property of the computation, not of a few components.

**8. Behaviour.**
- The first explicit frame was at chance (a frame artefact; the frame pilot fixed it before any patching).
- Negation fails everywhere (E2, behaviour only). In attribute → entity the copied word beats the correct entity: the Qwen3 family by margin in 0.41–0.60 of pairs, Mistral / Mixtral by sign in 0.29–0.30 (target ≤ 0.06).
- Qwen2-57B-A14B is unusable in bf16 (flat routers flip under rounding; transformers bf16 vs fp32 routing agreement 0.08). It runs in fp32.

**Feedback → answers**

| # | Team feedback | Answer (section) |
|---|---|---|
| 1 | Expert-level circuits; can the ICML paper (Arora et al. 2026, RelP on neurons) be applied to MoE? | Yes, ported and verified (frozen routing). At the final token it adds little beyond forward neuron DLA. Its value is at non-final positions (C2, 14e). Neurons inside experts: 2–4 of 768 (Qwen3) / 4–17 of 14,336 (Mixtral) for 80 % of the expert (C1). |
| 2 | Bigger capacity in the same family? | Dense Qwen3 4B → 14B flat. Qwen3-30B-A3B departs in both directions. Mixtral-8x22B is the gated Stage 3 (14d). |
| 3 | Dense counterparts? | Mistral-7B ↔ Mixtral and Qwen2-7B ↔ Qwen2-57B: invariant within ±0.10 on identical prompts (14c). |
| 4 | Add-back plots: random ≈ greedy first k | Added to every Phase-3/4/5/6 add-back figure plus tables (P6-0). |
| 5 | One principled dial, graded tasks, predictions before the experiment | Dial E / A on the same twins, ladder v2; 45 predictions committed before the runs; check table below (14a, 14b). |
| 6 | Parallel MoE / dense plots: does attention's work move into a larger FFN? | Not between twins. Within Qwen3, the MoE model is more attention-heavy on copy tasks, not less (14c, 14d). |
| 7 | Concept circuits of experts + heads; do we need the ICML paper? | Not for experts + heads: forward patching, DLA and knockout suffice (C3). RelP matters only for neurons at non-final positions (14e). |
| 8 | Same instance, variations that change the distribution: novel? | Yes, as far as the literature scan found (`docs/phase6_literature_scan.md`): binary context-vs-memory contrasts exist in dense models, but not a ≥ 3-level dial on the same items with the attention / FFN share measured, nor in MoE. |

**What was run.** Plan, decisions and deviations: RESEARCH_PLAN.md "Phase 6", its "Scope cut", and logs/PROGRESS.md.

Agents:
- p6-data: items, scans, case sets, tier 1;
- p6-engine: dense / Qwen2 / Qwen2-MoE families, head masks, neuron kind, fp32 protocol;
- p6-analysis: tier 2 on E1 attribute → entity, R-checks, cross-cell sections;
- p6-circuit: C1–C3.

Coordinator:
- frame and induction pilots (rules fixed before the results);
- QA blockers: P0 frame artefact; e3p3 truncation fixed before any scan;
- pooled / behaviour-only rule; all-strata twin cells; Qwen2 fp32 decision;
- GPU queue priorities and bundled jobs;
- P6-0 figures and Phase-5 page fixes.

Three usage limits stopped the agents. The detached chains kept the GPU busy, and nothing was lost.

| Item | Models | Cells |
|---|---|---|
| Dial E (E1 stated, E1p paraphrase, E2 negation, E3 event; frame P3 for attribute → entity, frame-matched E3p3) | all 8 | 2–8 per model |
| Dial A (agree / conflict / prior only) | Qwen3 (Mixtral behaviour only: too few strong-prior items) | 6 |
| Set C (event + statement) | behaviour all, patching Qwen3 / Mixtral | 2 |
| Ladder v2 (antonym, property, binding, induction + 4 anchors) | all 8 | 8 |
| Twins / series on identical prompts (all-strata E1 cells where needed) | Mistral ↔ Mixtral, Qwen2-7B ↔ 57B, Qwen3 4B / 8B / 14B / 30B | — |
| Tier 2 (experts, add-back with random-equivalent marks, mixed greedy, knockout, writers) | E1 attribute → entity, Qwen3 / Mixtral | 2 |
| R-metric, R-causal, R-site | Qwen3 / Mixtral | — |
| C1 neurons, C2 RelP pilot, C3 concept circuits | Qwen3, Mixtral, Mistral-7B, OLMoE | — |

**Pre-registration check** (`results/tables/ext14_prereg_check.md`).
- **Counts:** of 45 predictions, 21 held, 15 failed, 1 held partly (N3), 8 had too little data (pooled or behaviour-only cells).
- **Informative failures:**
  - L1 / L2 antonyms are not low;
  - E7 / E13 entity → attribute stays MoE-heavy at E1 in Qwen3-30B-A3B (H-type), the opposite of our best guess;
  - E2 the Mixtral share is lower than predicted;
  - A3 prior-only attribute → entity is 0.73, not ≤ 0.35: retrieving an entity is attention-heavy even without evidence;
  - D2 / D3 neither capacity nor sparsity moves the share between twins;
  - L9 the ladder's rank correlation reaches 0.8 in 3 of 8 models;
  - R3 / R4 at E3 entity → attribute: knockouts of the top heads / experts stay below the 20 % bar.
- **Held:** the copy / look-up top of the ladder, E3 levels as in Phase 5, H-invariant for the twins, D5, D6, R1, R2, N1, N2, A1 / A2 (Qwen3), Set C in 8 of 9 models.

**Results by question.**

*Dial and ladder (sections 14a / 14b).* The figure `results/figures/ext14syn_answer_type.png` shows, per model and direction, the share with the binding implied by the event (E3) and stated (E1). A paraphrase (E1p) equals E1 in every model (|Δ| ≤ 0.07), so the copied word is not what matters; the stated binding is. Negation (E2) cannot be traced: models copy the stated word ("The jug was not full …, the _ was empty" → the jug's partner), the copy trap winning 0.41–0.60 of attribute → entity pairs by margin in the Qwen3 family and 0.29–0.30 by sign in Mistral / Mixtral (target ≤ 0.06). This reading explains the Phase-5 copy control. "Options: t1, t2" made the attribute copyable but not bound to an entity, and moved the share by only +0.04–0.07. Stating "the jug was empty" binds it, and the share rises (except in Qwen3-30B-A3B).

*Dense vs MoE (14c).* Both twins are within bf16 / fp32 noise of each other on almost every cell, with the same cell order. MoE-ification of a dense model keeps attention and FFN doing the same share of the work. Prediction D6 held: Qwen2-57B's always-on shared expert carries most of the MoE rescue in every cell (0.67–0.77: entity answers, antonyms, object property, CounterFact, IOI) except the WinoGrande attribute answers, where it carries about half (0.43 E1, 0.51 E3). Upcycling evidence (C1): every Mixtral L20 expert matches the Mistral-7B L20 MLP neuron by neuron; L20E000's top-10 WinoGrande neurons are Mistral's top-ranked neurons at the same index (8 / 10).

*Capacity (14d).* See the summary.
- The paired values on common items agree with the own-set values: Qwen3 dense 14B − 4B within ±0.08 in every cell (largest: E3 entity → attribute −0.08); 30B-A3B vs dense on IOI +0.18 [0.16, 0.20].
- Precision is part of the protocol: Qwen2-57B in bf16 gives an E1 attribute → entity share of 0.68, in fp32 0.58, with rescue correlation bf16 vs fp32 of −0.04.

*Circuits (14e).* See the summary and `ext14_circuit.md`.
- C2 / DP3: a reverse-streaming gradient engine (F3) for the 30B / 47B models is not needed for the questions asked. Neuron patches at non-final positions can be done forward. Building F3 is justified only for gradient-ranked all-position circuits at scale.
- C3 (both models): see the summary. The country circuits overlap with general CounterFact machinery: the other-task loss is 0.16 / 0.25, the highest of the three concepts.

*Tier 2 on E1 attribute → entity.*
- Qwen3 selects L37E032 again, the Phase-5 attribute → entity expert: the same expert whether the binding is stated or inferred. Spec +0.21; knockout F +0.065; top-10 0.145 vs random 0.008. The all-MoE ceiling is −0.17, mixed greedy r(20) 0.93 with 9.6 heads among the first 10 picks.
- Mixtral selects L17E007 (ceiling 0.33; 80 % of it with 2 greedy / 3 oracle / 48 random experts).

*Robustness and causal (R1–R4).*
- R1 held.
- R3 / R4 on E1 attribute → entity are as in the summary.
- R-causal on E3 entity → attribute: top heads lower the attention repair 12–16 % beyond random, top experts the MoE repair 15–19 %; the share moves ≤ 0.05.
- R-site: the share changes by 0.03 (Qwen3) / 0.05 (Mixtral) when the corruption sits in the evidence instead of the question.

**Method lessons**
- **Frames decide competence.** The first explicit frame was at chance: Qwen3 0.39, Mixtral 0.05 correct in both directions. A one-word change of layout (P3) gave 0.99 / 0.80. Behaviour pilots with a pre-set decision rule caught it, and the induction format, before any patching.
- **One bad cut is enough.** Cutting a WinoGrande sentence at the connective left "… because the one on." when the blank sat inside a prepositional phrase. String-wise reconstruction checks on 100 % of items are now part of every builder.
- **Precision is model-specific.** Qwen2-57B-A14B's routers are nearly uniform; bf16 rounding flips most routed sets even in transformers. Every new MoE needs a bf16-vs-fp32 routing check before use.
- **Logistics:**
  - Small cells: a fixed pooled / behaviour-only rule instead of ad hoc relaxing.
  - Priority classes and one GPU job per cell raised the tier-1 share of GPU time from 13 % to ≈ 80 %.
  - Handshake watchers must anchor on the writer's tag at the start of the line (a quoted string once triggered one).

**Caveats**
- The Qwen3 dense series is not an architecture twin of Qwen3-30B-A3B.
- E1-family cells of Mistral-7B, and the twin E1 comparisons, use all association strata, because the neutral sets were too small (labelled).
- Several Mixtral / Mistral cells are pooled (marked †).
- Dial A is Qwen3-only.
- Qwen2-57B ran the trimmed 10-cell list in fp32.
- Ladder grades are confounded with cue distance (partial Spearman reported).
- All readouts are at the final token. Name-site attention (Phase 5 post-hoc) is not re-measured.

**Decisions for the user**
- **DP6:** run Stage 3 (Mixtral-8x22B: ladder + E1 / E3 tier 1, ≈ 14 GPU-h, needs deleting this phase's Qwen2 / Qwen3-dense checkpoints for disk), or stop here.
- **DP3:** build F3 (reverse layer streaming), or not. Recommendation: not now.

**Files**
- Sections: `results/sections/ext14_{ladder,dial,dense,scale,circuit}.md`.
- Tables: `results/tables/ext14_*` (`ext14_prereg_check.md`, `ext14_cells.md` with every cell's n and status).
- Figures: `results/figures/ext14_*`, `ext14syn_answer_type.png`.
- Numbers: `results/ext14_summary.json`, `ext14ci_summary.json`.
- Data and QA: `data/dial/` (pre-registration, frames, completeness, build report).
- Engine API: `logs/ext14_engine_api.md`.
- Synthesis figure and check table: `scripts/ext14_synth.py`.
