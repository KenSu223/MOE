# Brief: p6-circuit (P6-5: neurons inside experts, RelP pilot, concept circuits)

Read logs/agent_brief_p6_common.md first. Its rules apply. Then read RESEARCH_PLAN.md "P6-5", logs/ext14_engine_api.md (new fields and kinds: `DiagSpec.expert_neurons`, kind `expert_neurons`, `PrefillSpec.head_mask`, `route_mask_inherit`), and the predictions N1–N3 in data/dial/preregistration.md.

## Files you own

- `moetrace/ext14_circuit*.py`
- `scripts/ext14ci_*.py` and `.sh`
- `results/p6ci_*/`
- `results/figures/ext14ci_*`, `results/tables/ext14ci_*`, `results/ext14ci_summary.json`
- `results/sections/ext14_circuit.md`
- `logs/ext14ci_*.log`, `logs/agent_status_p6-circuit.md`

## C1 Neurons inside selected experts (forward only; start now)

**Experts and their own tasks**, using the existing case sets and pair files of the runs where each expert was selected (see CLAUDE.md section 2b and results/ext13*_summary.json):
- Qwen3:
  - L44E069 and L42E115 on CounterFact STR (`results/qwen3_str`);
  - L41E117 on WinoGrande option swap (Phase 3 `wino_qwen3_str` / Phase 5 `p5_dir_opt_qwen3`);
  - L37E032 on F1 attribute → entity (`p5_f1_qwen3`).
- Mixtral BOS:
  - L19E002 / L21E001 and L18E001 on CounterFact STR (`mixtral_bos_str`);
  - L20E000 on WinoGrande (`wino_mixtral_bos_str` / `p5_dir_opt_mixtral_bos`).
- Later add the experts that p6-analysis selects on the Phase-6 E1 cells (registry / `results/p6_e1_*_t2`), once available.

**Per expert:**
1. Record the intermediate activations a at the final token (clean and corrupted) on the validation pairs.
2. Score each neuron by its direct effect: Δa_i · w_e · (W_down[:, i] · ∇LD), with the final norm frozen at the corrupted run, as for expert DLA. Also score by Δa_i · w_e · ‖W_down[:, i]‖ as a magnitude baseline.
3. Validate with `expert_neurons` subset patches: top-k by score for k ∈ {1, 2, 5, 10, 20, 50, 100, 200, …, all}; matched random subsets; the expert's own single-expert patch as the 100 % reference.

Report:
- neurons needed for 50 / 80 % of the expert's rescue (N1 / N2 predictions);
- the overlap of top neurons across cases;
- the DLA-vs-patch agreement.

**Dense analogue:** Mistral-7B (`MODEL READY mistral7b_bos`) on the same CounterFact / WinoGrande STR items where the tokenizer allows (same 32k tokenizer as Mixtral):
- the MLP neurons of the layers matching Mixtral's experts (L18–L21);
- neurons needed for 80 % of the MLP layer's rescue there.

**Upcycling cross-check.** Mixtral L20E000 vs Mistral-7B L20 MLP:
- per-neuron weight correlation (gate / up / down rows), best-match assignment;
- whether the top Mixtral expert neurons map to top Mistral neurons;
- also the layer-wise cosine of the attention weights Mistral ↔ Mixtral (the upcycling evidence; CPU, from the checkpoints).

## C2 RelP pilot (Arora et al. 2026, arXiv 2601.22594; code github.com/TransluceAI/circuits)

Do not install their package into the project venv without need. Re-implement the linearisation rules (frozen RMSNorm denominator, frozen SiLU gate, frozen attention pattern, half rule on the gated product) with transformers autograd, as `moetrace/ext13atp.py` did for OLMoE.

Pilot models and port:
- OLMoE-1B-7B and Mistral-7B, which fit on the GPU.
- MoE port: freeze the top-k set and routing weights; generalise the half rule to w_e · SiLU(gate) ⊙ up; score both runs for experts active in only one; report the routing-switch rate.

Tasks: CounterFact STR and the WinoGrande option swap / F1 (OLMoE competence permitting, `results/wino_olmoe`); Mistral on its Phase-6 E1 / E3 cells once BASE READY.

Validate against exact patching (`expert_neurons`, `layer`) with faithfulness / completeness by mean ablation over the case set.

Deliverable: the question for **DP3 (user decision)**: do neuron circuits from RelP add something beyond C1 (e.g. multi-layer neuron paths, non-final positions) that would justify building reverse layer streaming (F3) for Qwen3-30B / Mixtral? State the evidence; do not build F3.

## C3 Concept circuits (needs Stage-1 rows; start when the ladder / Dial-E cells are BASE READY)

**Concepts:**
- SIZE / STATE: WinoGrande E1 / E3 big–small, empty–full… items; l_prop where applicable;
- COUNTRY: CounterFact P17 / P36 relations; l_bind if it uses countries — otherwise only CounterFact relations plus l_prop;
- PERSON-ROLE: IOI + Phase-5 name templates.

**Per task:**
1. Rank heads + experts (DLA + single patches from the existing tier-1 / tier-2 rows).
2. Concept scoring = AUROC of per-case exact effects against concept labels (CPU on existing rows).
3. Concept circuit = components in the top set of ≥ 2 tasks of a concept and not of the other concepts.
4. Edges among them by path patching at the final token (`multi` steps; ask p6-engine via a PROGRESS REQUEST if a kind is missing).
5. Validate with a concept × task knockout matrix (heads by `head_mask`, experts by `route_mask`), matched random sets, and add-back of the concept set.

Prediction N3: experts are more concept-specific than heads. Li & Subramani 2026 warn that component circuits are often not task-specific, so the specificity matrix is mandatory.

## GPU

Every job goes through the queue, ≤ 25 min each. Stage-1 and Stage-2 tier-1 jobs have priority: keep your jobs small enough that they interleave.

## Handshakes

- `[p6-circuit] C1 DONE`
- `[p6-circuit] C2 PILOT DONE (DP3 evidence in …)`
- `[p6-circuit] C3 DONE`

## Done when

C1, the C2 pilot and C3 are written up in `results/sections/ext14_circuit.md`, with `scripts/ext14ci_all.py` re-running the analyses from disk. Then send your final report.
