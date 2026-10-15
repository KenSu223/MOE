**Summary.** Phase 5 answers the team meeting's critique of the Phase-3/4 WinoGrande experiments.
- **Setup.** Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS, bf16, STR activation patching at the final token, every condition with the full Phase-3/4 output set.
- **The direction of the question decides the attention / MoE split, not the copyability of the answer.** The option swap of Phase 3 ("the mirror was too" → " dirty") asks entity → attribute. The original WinoGrande question asks attribute → entity ("Fill in the blank: … the _ was too dirty. Answer: the" → " mirror"). On identical twins, switching to the original direction:
  - raises the attention share of the rescue from 0.19 to 0.71 (Qwen3) and from 0.35 to 0.52 (Mixtral);
  - makes attention write the answer directly (direct attention share 1.08 / 0.86 against 0.05 / 0.31);
  - leaves all MoE outputs together able to restore only 0.11 / 0.41 of the drop (0.81 / 0.75 in the option swap).
- **Copyability and the verb matter little.** Putting the two attributes in the prompt, so that the answer could be copied, moves the attention share by only +0.07 / +0.04. A neutral verb moves it by nothing measurable.
- **The tasks fall into two blocks, not a smooth gradient.** Every entity → attribute level sits at 0.16–0.31 / 0.34–0.42, CounterFact at 0.54 / 0.58, every attribute → entity level at 0.69–0.79 / 0.52–0.65, and IOI at 0.92 / 0.80.
- **Attribute → entity has its own expert in Qwen3, L37E032.** It is necessary (knockout 0.042 of the clean margin) but writes almost nothing to the logits (direct share 0.01): it computes, and later attention heads (L40H2, L42H10) move the entity to the answer position.
- **The add-back curve shapes are explained.** Experts in the last four layers write against the answer in proportion to the evidence already in the residual. An expert-only add-back restores the MoE-written evidence but not the attention-written evidence. So curves overshoot the all-MoE ceiling exactly when attention writes much of the answer: across 23 conditions, overshoot vs late-MoE term r = 0.98.
- **Attribution patching (OLMoE pilot)** matches exact patching for experts and MoE layers, but underestimates early heads.

**Questions.** The meeting asked for the following (RESEARCH_PLAN.md "Phase 5"):
- (a) WinoGrande in its original reference-resolution direction;
- (b) a balanced noun–attribute association;
- (c) neutral / unrelated verb controls for the name items, and memorisation;
- (d) a data-centric ladder and a subset that needs attention;
- (e) why the add-back curves differ;
- (f) attribution patching;
- (g) scale and MIB, for discussion.

Execution followed "Phase 5 execution plan". Agents:
- p5-data: data, probes, base analyses; section `ext13_base.md`;
- p5-addback: add-back curves; `ext13_addback.md`;
- p5-circuit: heads + experts; `ext13_circuit.md`;
- p5-expert: knockout and writers; `ext13_knockout.md`, `ext13_writer.md`;
- p5-synth: ladder and curve shapes; `ext13_ladder.md`, `ext13_curves.md`;
- p5-atp: attribution patching; `ext13_atp.md`.

GPU ≈ 25 h in 263 queued jobs (incl. the scans and the pre-neutral fill-in-the-blank runs).

### Data and conditions

**Attribute → entity pairs.** The STR corruption is the twin's own trigger (dirty ↔ weak). The readout is Δ = logit(entity_r) − logit(entity_r′) at the final token. Every prompt carries the whole WinoGrande sentence; only the answer is withheld. An earlier truncation at the blank dropped deciding words and was voided.

**Filters.** On top of the Phase-3 pair rules (twin integrity, single-token entities, token alignment ≤ 3 differing tokens, STR site ≠ final token, dedup, connective swaps as their own stratum), each pair must pass:
- **Margin:** ≥ 1 logit in both directions.
- **Association balance.** Pre-registered, literature-based rule:
  - Two probes give each trigger alone plus both candidates, with no sentence. v1: `Word: dirty. Which fits better, the mirror or the spray?`. v2: `Fill in the blank: Between the mirror and the spray, the _ goes with "dirty".`.
  - A probe "responds" in a model if ≥ 10 % of the margin pairs reach |probe| ≥ 1. Qwen3: v1 23.0 %, v2 22.7 %. Mixtral: v1 2.0 %, v2 37.3 %, so Mixtral uses v2 only.
  - The main set keeps the pairs where both triggers are neutral (|probe| < 1) under every responding probe. Margin pairs by stratum, associative / one-sided / reverse / neutral: Qwen3 86 / 299 / 394 / 790, Mixtral 12 / 111 / 135 / 235.

**Conditions.** Discovery / validation pairs (Qwen3 | Mixtral).

| Task | Condition | Example (clean prompt → answer) | Pairs |
|---|---|---|---|
| 1 Original direction, fill-in-the-blank | `f1` | `Fill in the blank: The janitor tried to clean the mirror with the spray but the _ was too dirty.\nAnswer: the` → " mirror" | 128/128 + rep. \| 101/101 |
| 2 Direction on identical twins | `dir_f1` vs `dir_opt` | the same twins as `f1` prompts and as Phase-3 option swaps | 90/90 \| 94/94 |
| 3 Natural sentence, blank at the end | `f2e` | `I liked music from the piano but not the keyboard because music sounded synthetic coming from the` → " keyboard" | 32/32 \| 18/18 |
| 4 Verb control, entity → attribute | `vt_e2a_orig`, `vt_e2a_neu` | `Sarah gave Laura a hand … because Sarah was feeling` → " helpful" (neutral: "supported Laura") | 128/128 each (55 hand-checked templates × 48 IOI name pairs) |
| 5 Verb control, attribute → entity | `vt_a2e_orig` | `Fill in the blank: Sarah gave Laura a hand … because _ was feeling helpful.\nAnswer:` → " Sarah" | Qwen3 60/37 only (5 validation templates) |
| 6 Copy vs generate | `cp_nat`, `cp_vt_orig`, `cp_vt_neu` | `Options: helpful, overwhelmed.\n` + the task-4 / Phase-3 prompt; both attribute orders; an item counts only if both orders pass | cp_nat 146/140 + rep \| 216/204 + rep; cp_vt 128/128 |
| 7 Memorisation | `mem` | the 55 templates with their original WinoGrande names × orig / neutral / unrelated verb | behaviour only |

### (a) Original direction: attention writes the answer

**Direction on identical twins** (dir_f1 − dir_opt, validation, 95 % cluster-bootstrap CIs):

| Quantity | Qwen3 | Mixtral |
|---|---|---|
| Attention share | +0.52 [0.46, 0.56] | +0.18 [0.15, 0.20] |
| Direct attention share | +1.03 [0.94, 1.13] | +0.55 [0.46, 0.64] |
| All-MoE M | −0.70 [−0.77, −0.64] | −0.35 [−0.40, −0.29] |
| Heads among the first 10 head + expert greedy picks | +4.6 [4.2, 4.9] | +3.3 [2.8, 3.7] |

**The original-direction conditions agree.**
- f1: attention share 0.76 / 0.56, direct attention 1.20 / 1.01, all-MoE 0.24 / 0.23.
- f2e, which has no instruction wrapper: 0.69 / 0.65, heads among the first 10 picks 7.4 / 6.8.
- vt_a2e_orig: 0.79.

So the "Fill in the blank" wrapper does not cause the shift. The same heads carry it in every attribute → entity condition: Qwen3 L40H2 and L42H10, Mixtral L18H10 / L20H21 / L21H11.

**Experts.**
- Qwen3 selects **L37E032** in f1 and in vt_a2e_orig:
  - f1: Spec +0.47 [0.39, 0.56], pattern A, re-selected on replication;
  - knockout F 0.042 [0.029, 0.055] (f1) and 0.034 (vt_a2e_orig).
- It is a computer, not a writer: its direct share of the patch effect is 0.01, and its write does not promote the entity (vocabulary rank of the answer ≈ 74,000).
- Mixtral has no specific attribute → entity expert in f1 (pattern B).
- The Phase-3 WinoGrande experts do nothing in the original direction: L41E117 F ≤ 0.004 on f1 / dir_f1 / f2e; L20E000 ≤ 0.005.

### (b) Association balance

- The neutral rule removes the pairs whose trigger alone points to an answer. Inside each level, the association strata move the attention share by at most 0.11, in opposite directions for the two task directions.
- In the Phase-3 option swap the pairs solvable from the local clause alone ("the mirror was too" → dirty) have the same attention share as the rest (0.21 vs 0.17 Qwen3, 0.36 vs 0.35 Mixtral). The MoE dominance there was not an association artefact.
- Names raise the share by 0.08–0.16 in every level.
- Mixtral has few name items that pass the attribute → entity margin (f1: 12 of 101 validation pairs).

### (c) Verbs and memorisation

- **Neutral verb** (bought → got and similar, hand-curated per template): drop 4.25 / 3.91 vs 4.97 / 4.71 logits. Attention share +0.01, CI includes 0.
- **Unrelated, role-symmetric verb** (e.g. "talked with"): the effect disappears (drop 0.22 / 0.00). The models read the verb.
- **Original names do no better than new IOI names.** Margin rate 0.31 vs 0.38 (Qwen3), 0.31 vs 0.29 (Mixtral). The unrelated verb removes the effect with either name set, so the name items are not solved by recall of the WinoGrande sentence.
- **The attribute → entity question on the same templates mostly fails** the margin: 3.7 % / 1.9 % (Qwen3 orig / neutral), 0.1 % / 0.2 % (Mixtral), against 39 % / 33 % entity → attribute.
- **The restated format** ("The one who … was") passes 9 %, with a first-mentioned-name bias (0.64 / 0.58). It is reported as behaviour only.

### (d) Ladder and the copy control

**Two blocks** (attention share Qwen3 / Mixtral; `ext13_ladder.md`):
- every entity → attribute level, with or without the answer in the prompt: 0.16–0.31 / 0.34–0.42;
- CounterFact: 0.54 / 0.58;
- every attribute → entity level: 0.69–0.79 / 0.52–0.65;
- IOI: 0.92 / 0.80.

The two models order the 12 shared levels alike (Spearman 0.90).

**Copy prefix (P5-8).**
- On the natural Phase-3 twins the `Options:` prefix moves the attention share by +0.07 [0.01, 0.11] / +0.04 [0.02, 0.05] against the option swap. Heads among the first 10 picks: 4.3 / 2.9 against 3.6 / 2.6. On the templates it moves the share by ≈ 0.
- The WinoGrande expert stays necessary when the answer is copyable on natural items: Qwen3 L41E117 F 0.027, Mixtral L20E000 0.045.
- On the name templates the WinoGrande expert is unused with or without the prefix. Mixtral uses L19E006 there (0.035–0.042).
- Design note: a first prefix ("Of Sarah and Laura, one … and the other …") let the models map names to attributes by position; Mixtral passed 0.43 vs 0.00 by attribute order even with an unrelated verb. It was replaced before any patching ran. The `Options:` prefix passes both orders alike (0.73 / 0.74, 0.90 / 0.89).

**What needs attention.** The direction of the question, not the availability of the answer.
- The attention-heavy subset is attribute → entity on name items: Qwen3 f1 names share 0.78, vt_a2e_orig 0.79, 9 of the first 10 picks are heads.
- In linguistic terms, the original task is reference resolution (pick the referent the attribute licenses).
- The option swap is attribute recall about an already identified entity, much like factual recall, and is done by the MoE.
- The ladder confounds answer type (context entity vs generated attribute), prompt length (17–19 vs 27–32 tokens) and item mix. The instruction wrapper is ruled out by f2e.

### (e) Why the add-back curves differ

`ext13_curves.md`.
- **Overshoot is the late-MoE term.** The oracle curve's overshoot over the all-MoE ceiling equals, condition by condition, M(≤ L−5) − M: r = 0.98 [0.93, 1.00] across 23 conditions; per case r 0.88–1.00 inside overshooting conditions.
- **Negative experts are everywhere.** Every task has experts with negative single-expert effects, concentrated in the last four layers and writing against the answer directly. The option swap has them too (Mixtral 0.24 of the drop) but overshoots only +0.03.
- **What decides the shape is what they react to.** The late MoE writes against the answer in proportion to the evidence already in the residual (per logit of attention DLA −0.40; per logit of earlier-MoE DLA −0.51). An expert-only add-back restores MoE-written evidence but not attention-written evidence. When attention writes much of the answer (attribute → entity, CounterFact), the ceiling M contains the late experts' reaction to evidence the patched run lacks, so subsets that skip them overshoot: f1 +0.28 / +0.40, option swap +0.00.
- **GPU check (Qwen3 f1).** All experts except the negative ones restore 0.49, against M 0.24 and the best oracle subset 0.51. The whole gain comes from the last-four-layer negatives.

### (f) Attribution patching

OLMoE-1B-7B, WinoGrande option swap and IOI, 64 + 64 pairs each; `ext13_atp.md`.

| Component | AtP vs engine patching, r (WinoGrande / IOI) | Noise ceiling |
|---|---|---|
| Experts | 0.96 / 0.97 | 0.97 / 0.96 |
| MoE layers | 0.96 / 0.99 | 1.00 / 0.99 |
| Heads | 0.83 / 0.95 | 0.92 / 0.94 |

- Early components are underestimated (slope 0.01–0.32 in L0–3).
- 32–34 % of single-expert patches switch downstream top-k slots. Freezing the downstream routing removes about half of AtP's squared error.
- As an add-back ordering, AtP equals the oracle (AUC 0.669 vs 0.670).
- Recommendation: no reverse layer streaming for final-position work; exact patching costs about 5 GPU-min per 256 rows. Worth building only for all-position or edge-level attribution.

### (g) Not run

- Scale: the same-family Qwen3-235B does not fit on disk. The dense Qwen3 family is an option.
- MIB: discussion only.
- The letter A/B format and synonym averaging were not run, by user decision.

### Caveats

- Small or template-clustered samples:
  - f2e (32 / 18 pairs);
  - vt_a2e_orig (5 validation templates);
  - template conditions (8–18 templates per split; CIs are seed-clustered).
- Mixtral attribute → entity has few name items.
- Cross-condition correlations rest on about 20 points.
- The curve-shape GPU check covers Qwen3 only.
- The ladder's two blocks also differ in answer type and prompt length.
- Expert selections on templates are often pattern B and below the knockout necessity criterion.

### Files

- Sections: `results/sections/ext13_{base,addback,circuit,knockout,writer,ladder,curves,atp}.md`.
- Summaries: `results/ext13{base,ab,ci,ko,syn,atp}_summary.json`.
- Condition registry: `data/wino_ref/p5_conditions.json`.
- Run dirs: `results/p5_<cond>_<proto>[_ab|_ci]`, `results/p5_<proto>_{ko,wr}*`, `results/p5_atp_*`.
- Pre-neutral fill-in-the-blank runs, kept as a record: `results/wino_ref_F1_*`.

### Post-hoc analyses (2026-10-13/14; CPU, existing rows; `scripts/ext13_posthoc.py`)

**Position × layer grid by site** (`results/tables/ext13_posthoc_sites.md`).
- The attention / MoE split of the main text is a final-token measure.
- At the swapped word's own position the MoE dominates in every condition and direction, in the first layers: L0–L2, Qwen3 0.13–0.22 of the drop, Mixtral 0.51–0.95. This is the WinoGrande analogue of the early subject site of CounterFact (Directions 5-F4, 6b).
- The tokens in between carry ≤ 0.05 per layer.
- The information moves from the swapped word to the final token at a hand-off layer: the swapped-word residual still restores ≥ 0.5 up to a layer, and the final-token residual restores ≥ 0.5 from a layer on.
  - Qwen3: L28 → L29 when asking for the attribute, L34 → L36 when asking for the entity.
  - Mixtral: L12 → L13 and L14 → L16.
  - The entity question hands off later; final-token attention then writes the entity (Qwen3 L39–L43, Mixtral L16).

**Name items: the look-up happens at the name, not at the final token** (`results/tables/ext13_posthoc_namesite.md`).
- On the verb templates (entity → attribute), attention at the swapped name restores much more of the drop than any final-token attention layer:
  - Qwen3 L6: 0.18–0.32 (final token ≤ 0.10);
  - Mixtral L7–L9: 0.36–0.46, or 0.65–0.84 summed over L4–L9 (final token ≤ 0.15).
- This holds in all four template conditions.
- Object nouns show ≤ 0.05 at their own position. The few natural name items (10 pairs) agree in Mixtral (0.21) but not clearly in Qwen3 (0.04).
- Reading: the name attends back to its earlier mention to retrieve its role ("find what the person did"), and the final-token MoE turns the role into the attribute. The final-token attention share therefore understates attention on name items.

**Competence of the entity question** (`results/tables/ext13_posthoc_f1_competence.md`).
- Pass margin, fill-in-the-blank format:
  - Qwen3: objects 20 %, natural names 10 %, name templates 4 %;
  - Mixtral: 15 %, 1 %, 0 %.
- On the templates the models ignore the trigger:
  - correct in both twins 0.27 / 0.23, about the 0.25 chance level;
  - the same name in both twins in 68–69 % of pairs;
  - Qwen3 prefers the first-mentioned name in 65 % of prompts.
- On the items they do pass, the entity question is attention-dominated, more so than on objects: share 0.78–0.79 (Qwen3 names, templates), 0.68 (Mixtral names, 12 pairs).
- Failing items have no drop and cannot be traced with STR.

**Template stability** (`results/tables/ext13_posthoc_templates.md`).
- Outcomes are mostly all-or-none per template across its 48 name pairs:
  - Qwen3: 16 templates pass for > 80 % of the name pairs, 30 for ≤ 20 %;
  - Mixtral: 12 and 31.
- 89 % / 91 % of the drop variance lies between templates. The sentence decides, not the names.

**Margin rule** (`results/tables/ext13_posthoc_margins.md`).
- "≥ 1 logit" everywhere means logit(answer) − logit(contrast answer). That is the log-odds of the two candidates: 1 logit = a probability ratio of e ≈ 2.7, i.e. 73 % in a two-way choice. It is not a top-1 rule.
- Top-1 in both twins:
  - entity question (f1, validation): 0.81 / 0.61;
  - Phase-3 option swap: 0.25 / 0.33;
  - CounterFact: the true object is top-1 in 29–32 % of cases.
- f1 validation clean margins: median 2.0 / 1.75 logits, 10th percentile 1.12.

**Suggested follow-ups (not run).**
- Head-level patches at the swapped name (Qwen3 L4–L9, Mixtral L5–L10).
- Expert-level patches at the swapped word (L0–L2).
- A clean-prompt DLA of the name preference on the failing name items (which components write the first-mention bias).
- About 30–60 GPU-min per model in total.
