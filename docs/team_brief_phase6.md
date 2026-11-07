# WinoGrande Phase 6 Brief

Follow-up to the Phase 5 brief (`docs/team_brief_phase5.md`). You asked for four things:
- one principled dial on the same WinoGrande items, with the expected result written down first;
- different tasks graded in advance;
- dense counterparts and bigger models in the same family;
- circuits down to experts and neurons.

We fixed 45 predictions in a commit before any GPU run. Eight models were run.
- MoE: Qwen3-30B-A3B, Mixtral-8x7B, Qwen2-57B-A14B.
- Dense: Mistral-7B, Qwen2-7B, Qwen3-4B / 8B / 14B.
- The Qwen2 pair runs in fp32. Mixtral-8x22B waits for the team's go.

The HTML version, with figures, is `docs/pages/team_brief_phase6/index.html`.

## Answers at a glance

| Question | Answer |
|---|---|
| What moves the attention / FFN split on the same items? | Mainly the answer type. Entity answers are attention-heavy and attribute answers FFN-heavy. Stating the attribute next to its entity moves attribute answers toward attention in 7 of 8 models. Qwen3-30B-A3B ignores it. |
| Dense vs MoE? | No difference at matched architecture. Mistral-7B and Mixtral agree within ±0.10 on all 10 tasks; the Qwen2 twins on 6 of 7. |
| More capacity? | The dense Qwen3 series is flat from 4B to 14B. Qwen3-30B-A3B departs in both directions, which is not a size trend. 8x22B is pending. |
| Graded tasks, predicted in advance? | The top of the order held (IOI, induction, binding look-up). The bottom failed: antonyms sit near 0.5. |
| Associativity? | A weak dial (spread 0.10). In conflicts, the experts write the prior answer and attention overrides it. |
| Experts down to neurons? | A handful of neurons carry 80 % of an expert's effect (2–4 of 768 in Qwen3, 4–17 of 14,336 in Mixtral), but different neurons for different prompts. Mixtral's experts are Mistral-7B's MLP, neuron for neuron. |
| The ICML paper (RelP on neurons) on MoE? | It ports and works. At the last token it adds little; its value is at earlier positions. It is not needed for expert + head circuits. |
| Concept circuits? | In both models, the circuits hurt their own tasks much more than other tasks. Experts are more concept-specific than heads in 3 of 6 cases (size / state in both models, person role in Qwen3); not for country, nor for person role in Mixtral. |
| Is the dial new? | As far as our literature scan found, yes for a 3+ level dial on the same items with an attention / FFN split, and for MoE. |

## 1. The dial

The readout is attention's share of the repair at the last token. The question, the swapped word and the format stay fixed; only the evidence changes.

```
E1 stated      The jug was empty and the pot was full.  Fill in the blank: The _ was empty.  Answer: the → " jug"
E1p paraphrase The jug was drained and the pot was filled. …                              → " jug"
E2 negation    The jug was not full and the pot was not empty. …                            → " jug"
E3 event       The restaurant worker poured water from the jug into the pot. …              → " jug"
entity → attribute, same levels:  … , so the jug was → " empty"
```

Before any patching, a behaviour pilot replaced our first format (`…, so the _ was empty`), which gave chance-level answers. The decision rule for the pilot was written down beforehand.

## 2. Answer type vs stated binding

- **Entity answers are attention-heavy at every level** (0.48–0.83).
- **Attribute answers are FFN-heavy when inferred** (0.18–0.34).
- **Stating the attribute raises attribute answers toward attention:**
  - Qwen3-4B 0.31 → 0.58, Qwen3-8B 0.28 → 0.55, Qwen3-14B 0.21 → 0.46;
  - the Qwen2 twins 0.21 → 0.37–0.38;
  - Mistral 0.31 → 0.49, Mixtral 0.34 → 0.50.
- **Qwen3-30B-A3B is the exception** (0.18 → 0.22).
- **A paraphrase works as well as the exact words.** The binding matters, not copying the word.
- **Negation fails in every model** (copy trap). It is reported as behaviour only.

## 3. Dense vs MoE, capacity

- **Mistral-7B → Mixtral:** every task within ±0.10, task order preserved (Spearman 0.98).
- **Qwen2-7B → Qwen2-57B:** 6 of 7 tasks within ±0.10; antonyms −0.11.
- **Qwen2-57B's shared expert** carries 0.67–0.77 of the expert repair on every task (entity answers, antonyms, object property, CounterFact, IOI) except the WinoGrande attribute answers, where it carries about half (0.43 stated, 0.51 inferred).
- **The dense Qwen3 series is flat.** Qwen3-30B-A3B is more attention-heavy on copy tasks (IOI 0.92 vs 0.73) and less on attribute answers.
- **Qwen2-57B must run in fp32.** In bf16 its nearly uniform routers flip; the share would read 0.68 instead of 0.58.

## 4. Ladder

| Grade | Task | Attention share, 8 models |
|---|---|---|
| 1 | antonym | 0.35–0.59 |
| 1.5 | object property | 0.33–0.42 |
| 2 | WinoGrande entity → attribute | 0.18–0.34 |
| 2 | CounterFact | 0.41–0.58 |
| 3 | WinoGrande attribute → entity | 0.48–0.73 |
| 4 | binding look-up | 0.65–0.79 |
| 5 | induction | 0.67–0.77 |
| 5 | IOI | 0.71–0.92 |

## 5. Circuits

| Expert | Neurons | Neurons for 80 % of its effect (pooled over prompts) | Median per prompt |
|---|---|---|---|
| Qwen3 L44E069 | 768 | 2.1 | 2.1 |
| Qwen3 L41E117 | 768 | 2.7 | 2.9 |
| Qwen3 L37E032 (indirect) | 768 | 33 (magnitude ranking) | 142 (direct-effect ranking) |
| Mixtral L19E002 | 14,336 | 6.7 | 11 |
| Mixtral L20E000 | 14,336 | 17 | 17 |

Pooled = the patch of the top-k neurons restores 80 % of the expert's patch on the case means. Per-prompt medians use each prompt's own direct-effect ranking.

**RelP (neuron circuits).** Ported to MoE with frozen routing. At the last token it ranks neurons about as well as our forward scores. On factual recall, three quarters of the neuron effect is at the subject and relation positions.

**Concept circuits.** Margin lost on own / other tasks:

| Concept | Qwen3 | Mixtral |
|---|---|---|
| size / state | 0.43 / 0.09 | 0.32 / 0.11 |
| country | 0.70 / 0.16 | 0.69 / 0.25 |
| person role | 0.21 / 0.04 | 0.39 / 0.02 |

## 6. Predictions

| Outcome (45) | Count |
|---|---|
| Held | 21 |
| Held partly | 1 |
| Failed | 15 |
| Not testable | 8 |

**Informative failures:**
- antonyms are not FFN-only;
- in Qwen3-30B-A3B, stating the binding does not move attribute answers;
- finding an entity is attention-heavy even with no evidence;
- neither capacity nor sparsity moved the twins.

## 7. Open

- **Mixtral-8x22B:** about 14 GPU hours. The team's call.
- **Gradient engine for the 30B / 47B models:** not recommended now.
- **Caveats:**
  - Mistral and the twin comparisons of the stated level use all association strata.
  - Several Mixtral / Mistral cells are pooled.
  - Associativity is measured in Qwen3 only.
  - All readouts are at the last token.
