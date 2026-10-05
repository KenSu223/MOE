**Summary.** Phase 3 answers the two questions from the team's notes, both under symmetric token replacement (STR) only, on Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS.

**(1) How many experts restore the answer?** Patching every MoE output at the final position restores only part of the drop on CounterFact (Qwen3 0.53, Mixtral 0.41) but most of it on WinoGrande (0.84, 0.79).
- With all experts patched, the answer flips back in 44 % / 37 % of CounterFact cases and in 95 % / 89 % of WinoGrande cases.
- A handful of experts carries the expert repair: 80 % of that ceiling takes k = 4–6 experts on CounterFact and 7–10 on WinoGrande with adaptive greedy, out of 384 (Qwen3) or 64 (Mixtral).
- Greedy is effectively optimal: beam search and the exact optimum within each case's top 10 add ≤ 0.016 of the drop, below bf16 run-to-run noise.
- A patch-free ranking by direct logit attribution is as good as the single-patch oracle. The paper's layer-first order is below every ranking that uses each expert's effect on the answer (oracle, DLA, population), though above routing weight and random.

**(2) Attention or MoE?** At the final position the corruption can only enter through attention: the final token is shared and the MoE is per-token, so patching all attention outputs restores the clean state exactly. The question is therefore asked of the single-layer patches, of the all-MoE patch and of the direct paths to the logit difference. On every measure the tasks order the same way: IOI ≫ CounterFact > WinoGrande role swap > WinoGrande option swap. As shares of the positive single-layer rescue, attention carries:
- IOI 0.92 / 0.80;
- CounterFact 0.54 / 0.58;
- WinoGrande role swap 0.32 / 0.39;
- WinoGrande option swap 0.16 / 0.34.

The hypothesis "WinoGrande is IOI-like (attention-driven)" is rejected at the final position. WinoGrande is the most MoE-heavy of the three tasks, with one specific late expert per model (Qwen3 L41E117, Mixtral L20E000) that is not a CounterFact expert. Attention still moves the option's identity to the final token, gradually over L19–L41 in Qwen3 and in one step at L13 in Mixtral, through heads that look coreference-like.

### What was run

Phase 3 of RESEARCH_PLAN.md, three sub-agents plus coordinator, ≈ 3.9 GPU-h in total (ext8 ≈ 82 min, ext7-wino ≈ 70 min, ext7-controls ≈ 83 min, coordinator scans and checks ≈ 25 min). All runs follow Zhang & Nanda's recommendations:
- STR only, with no Gaussian noise anywhere (user decision; Gaussian noise on one option token does not hide which of two in-context candidates is meant);
- logit difference LD(r, r′) normalised by the drop;
- single-layer patches first;
- both directions of every symmetric pair;
- head detections at ≥ 2 SD;
- several corruption sites.

They also keep the criteria inherited from Direction 6: the clean margin, the donor margin, token symmetry, single-token continuations, a non-final STR site, fixed splits and the recurrence gate.

Tasks:
- **CounterFact STR**: Direction-6 donors.
- **WinoGrande option swap**: the blank filled with each twin's answer, predicting the sentence-final single-token trigger. 776 pairs pass the margin under all three scanned protocols; 128 / 128 pairs are used plus 128 / 128 for replication.
- **WinoGrande role swap**: the two candidates' first mentions exchanged; name pairs only.
- **IOI**: Wang et al. templates. Corruption (i) replaces S2 by IO; corruption (ii) replaces S1 and IO by other names.

New engine capability: `multi` spawns with `attn_layer` / `block` steps, and every kind in the noising direction. These were verified against transformers hooks on OLMoE (r 0.9994); `verify_olmoe.json` is unchanged. Details are in the three sections that follow:
- 7: WinoGrande, `ext7_wino.md`;
- 7b: role swap, IOI and the three-task table, `ext7_controls.md`;
- 8: add-back curves, `ext8_addback.md`.

### One table

**Final position, validation. Columns: attention share of the positive single-layer rescue; all-MoE patch M (denoising); direct-path attention / MoE writes; experts to 80 % of the all-MoE ceiling. Rescue is a fraction of the drop.**

| Task (STR site) | Model | Drop (logits) | Attention share | All-MoE M | Direct path attn / MoE | k for 80 % of ceiling (greedy / random, of K) | Answer restored by all MoE |
|---|---|---|---|---|---|---|---|
| IOI (i) S2 → IO | Qwen3 | 12.65 | 0.92 [0.90, 0.94] | −0.27 | 1.40 / −0.40 | – | – |
| IOI (i) S2 → IO | Mixtral BOS | 11.28 | 0.80 [0.79, 0.81] | −0.02 | 1.17 / −0.17 | – | – |
| CounterFact | Qwen3 | 11.67 | 0.54 [0.51, 0.57] | 0.53 [0.49, 0.57] | 0.50 / 0.51 | 5 / 320 of 384 | 44 % |
| CounterFact | Mixtral BOS | 12.98 | 0.58 [0.55, 0.60] | 0.41 [0.37, 0.45] | 0.63 / 0.36 | 4 / 64 of 64 | 37 % |
| WinoGrande role swap | Qwen3 | 6.94 | 0.32 [0.29, 0.35] | 0.77 | 0.19 / 0.81 | – | – |
| WinoGrande role swap | Mixtral BOS | 6.70 | 0.39 [0.38, 0.41] | 0.70 | 0.38 / 0.62 | – | – |
| WinoGrande option swap | Qwen3 | 8.14 | 0.16 [0.13, 0.20] | 0.84 [0.83, 0.86] | 0.05 / 0.95 | 10 / 320 of 384 | 95 % |
| WinoGrande option swap | Mixtral BOS | 7.69 | 0.34 [0.33, 0.36] | 0.79 [0.76, 0.81] | 0.29 / 0.71 | 7 / 48 of 64 | 89 % |

Values are from `results/tables/ext7_controls_three_task.md` and the ext8 comparison table. Notes:
- The all-attention patch gives A = 1.00 in every row; it is a sanity check, not a measurement.
- A direct-path share above 1 means the MoE writes against the answer (IOI).
- Add-back curves were run for CounterFact and the WinoGrande option swap, as planned.

### Reading

1. **How many experts does it take, and is greedy good enough?** For the curve of remaining gap against the number of experts added back, the ceiling is the all-MoE patch.
   - **The ceiling.** On CounterFact it is about half the drop: the rest of the repair needs attention, i.e. the subject's identity re-read from context. Even with all experts patched back, the answer flips back in fewer than half the cases. On WinoGrande the all-MoE patch restores 0.79–0.84 and flips 89–95 % of the answers back.
   - **How many experts reach 80 % of the ceiling.** It takes 4–6 experts (CounterFact) and 7–10 (WinoGrande) with adaptive greedy; random orders need 48–320.
   - **Overshoot.** Subsets can exceed the all-MoE ceiling (max r 0.61 vs 0.53 in Qwen3 CounterFact), because some clean expert outputs work against the answer. "Maximum rescue = patching all MoEs" is therefore not an upper bound for subsets.
   - **Greedy is good enough.** Adaptive greedy beats every static ranking by +0.04 to +0.12 at k = 10. Beam search (width 4) and the exact optimum within the top 10 are within bf16 noise of greedy.
   - **Static rankings.** Static rankings fail mainly where experts interact: Qwen3 WinoGrande needs 48 experts with the per-case static oracle but 10 with greedy.
   - **A cheap ranking that works.** The patch-free direct-logit-attribution ranking matches the oracle, so a good expert ranking does not need one patch per expert.
   - **The paper's layer-first order is below every effect-based ranking** (oracle, DLA, population; k for 80 %: 48 / 9 / 64 / 16 vs greedy 5 / 4 / 10 / 7), though above routing weight and random. Over k = 1..15, where greedy is also evaluated, AUC over log k greedy / oracle / DLA / layer-first is 0.39 / 0.37 / 0.37 / 0.17 (Qwen3 CounterFact), 0.35 / 0.31 / 0.29 / 0.20 (Mixtral CounterFact), 0.49 / 0.42 / 0.42 / 0.21 (Qwen3 WinoGrande) and 0.49 / 0.47 / 0.42 / 0.35 (Mixtral WinoGrande) (`results/tables/ext8_a1_partial_auc_k15.md`).

2. **Attention vs MoE.**
   - **IOI validates the method.** Attention carries the rescue, the MoE writes against the answer at the last layers, and the head patches recover IOI's known classes: an S2-reading head (Qwen3 L42H11), name-mover-like heads (Qwen3 L42H10, Mixtral L19H8) and negative movers (Qwen3 L42H14, Mixtral L22H29). With corruption (ii) the name movers come first, the corruption-site effect Zhang & Nanda report in their App. F.
   - **WinoGrande sits at the other end.** In our formulation (predict the property word given the filled-in entity, which is what lm-eval's partial scoring compares), the answer is not a copy of a context token. It is generated from the entity and the relation stated earlier in the sentence. Attention brings the entity's identity to the final position and the MoE writes the property: 95 % (Qwen3) and 71 % (Mixtral) of the direct path. The expert localisation is cleaner than on CounterFact: Qwen3 L41E117 has Spec +0.90 and equal-norm Spec +0.45, and is re-selected on the replication set.
   - **Role swap.** It tests which entity has which attribute (binding) instead of which entity is referred to. It moves the balance towards attention, with a Qwen3 attention peak at L38 (0.13 of the drop) and a Mixtral attention effect at the option position, but WinoGrande stays on the MoE side of CounterFact. Name pairs lean further towards attention than object pairs (direct attention share 0.26 vs 0.04 in Qwen3).
   - **The small single-layer attention effects in Qwen3 do not mean that attention is unimportant.** The option's identity reaches the final token over 20 layers, and heads of the same KV group cancel within a layer: L38H18 +0.34 and H21 +0.29 against H16 −0.55. Single-layer patches undercount distributed attention; the direct-path split and the position grid are the views to quote.

3. **Methodological findings (apply to every future patching study here).**
   - **The joint attention/MoE game at a shared final token is degenerate.** Patching all attention outputs restores the clean state (A = 1), so a Shapley split built on it is meaningless. Use the all-MoE patch and the direct-path split (exact final norm) instead.
   - **Noising adds nothing for symmetric pairs.** Noising in direction d equals denoising in direction 1 − d; noising adds information only for asymmetric donors (CounterFact).
   - **bf16 batch noise.** Identical rows vary across passes by a median of 0.06–0.17 logits, so differences below about 0.01 of the drop between strategies are not resolved.
   - **Gaussian noise fails as a corruption for two-candidate in-context choices.** On one option token it leaves Qwen3's answer largely intact (median noised margin 2.6 of 3.6). It is a further reason for STR.
   - **A latent bug in `ext5_subject.py` was fixed:** window > 1 rows and `attn_layer` rows in the same pass turned `attn_layer` into `block`. No result was affected (production passes never mixed them). Regression checks on OLMoE are in `results/verify_ext5_subject_olmoe.json` and `results/verify_ext6_str_grid_olmoe.json`, compared with the `*_before_ext7fix.json` copies.

### Caveats and open items

- **Gradient rankings (attribution patching, AtP\*, EAP-IG)** need F3 (reverse layer streaming), which is not built.
- **Mixtral without BOS** (the base reproduction's paper protocol) was not run in Phase 3 (user decision (e)). The WinoGrande case set is shared with that protocol, so it can be added on identical pairs.
- **Head patches** are final-position only; duplicate-token and induction heads acting at S2 (IOI) are not visible.
- **Position grids** for the role swap and IOI use 64 validation pairs.
- **Role swap and option swap** are compared on different items.
- **Subsets used for parts of the add-back study.** Add-back curves use the main validation set only, not the replication set. Shapley values, beam search and the exact optimum use subsets of rows (listed in section 8).
- **The WinoGrande direction.** The formulation is necessarily "entity → property". The original WinoGrande question ("which entity has the property") cannot be posed as a single next-token prediction for 86 % of the twins, because their trigger follows the blank.
- **Memorisation.** WinoGrande train_xl is public. Pass rates on the AfLite-filtered items match the dev twins, so there is no sign that memorised items inflate the pool.

Files:
- Sections: `results/sections/ext7_wino.md`, `ext7_controls.md`, `ext8_addback.md`, this file.
- Numbers: `results/ext7_wino_summary.json`, `results/ext7_controls_summary.json`, `results/ext8_addback_summary.json`, `results/ext7_wino_funnel.json`.
- Plan and decisions: RESEARCH_PLAN.md "Phase 3".
