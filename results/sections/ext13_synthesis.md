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
