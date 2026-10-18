# Phase 6 build report (p6-data)

## Status matrix (cell x model)

BASE n = tier 1 done (BASE READY), n validation pairs; 'pooled n' = discovery = validation = all n passing pairs (< 64 validation, >= 64 together); beh (n < 64) = behaviour only, too few pairs (scan-level behaviour recorded); beh (design) = Stage-2 Set C / Qwen3-1.7B competence only; beh (trimmed) = Qwen2-57B tier 1 trimmed to the twin-comparison cells; anchor = existing Phase-3/6 runs; '-' = not built for that model.

| Cell | qwen3 | mixtral_bos | mistral7b_bos | qwen2_7b | qwen2_57b | qwen3_1p7b | qwen3_4b | qwen3_8b | qwen3_14b |
|---|---|---|---|---|---|---|---|---|---|
| e1_a2e | BASE 128 | BASE 90 | beh (38 < 64) | BASE 128 | BASE 128 | beh (design) | BASE 128 | BASE 128 | BASE 128 |
| e1_e2a | BASE 128 | BASE pooled 80 | beh (34 < 64) | BASE 128 | BASE 128 | beh (design) | BASE 128 | BASE 118 | BASE 128 |
| e1p_a2e | BASE 128 | beh (44 < 64) | beh (4 < 64) | BASE 128 | beh (trimmed) | beh (design) | BASE 128 | BASE 128 | BASE 128 |
| e1p_e2a | BASE 128 | BASE pooled 110 | beh (50 < 64) | BASE 128 | beh (trimmed) | beh (design) | BASE 120 | BASE 120 | BASE 128 |
| e2_a2e | beh (0 < 64) | beh (0 < 64) | beh (0 < 64) | beh (0 < 64) | beh (0 < 64) | beh (design) | beh (0 < 64) | beh (0 < 64) | beh (2 < 64) |
| e2_e2a | beh (6 < 64) | beh (18 < 64) | beh (20 < 64) | beh (22 < 64) | beh (12 < 64) | beh (design) | beh (2 < 64) | beh (8 < 64) | beh (12 < 64) |
| e3p3_a2e | BASE 70 | BASE 80 | BASE pooled 106 | BASE 96 | BASE 128 | beh (design) | BASE 76 | BASE 110 | BASE 128 |
| e3_a2e | BASE 84 | BASE pooled 113 | BASE pooled 111 | BASE 68 | beh (trimmed) | beh (design) | BASE 78 | BASE 128 | BASE 128 |
| e3_e2a | BASE 128 | BASE 128 | BASE 128 | BASE 128 | BASE 128 | beh (design) | BASE 128 | BASE 128 | BASE 128 |
| e1_a2e_all | BASE 128 | BASE 128 | BASE pooled 94 | - | - | - | - | - | - |
| e1_e2a_all | BASE 128 | BASE 110 | BASE pooled 104 | - | - | - | - | - | - |
| e1p_a2e_all | - | - | beh (10 < 64) | - | beh (trimmed) | - | - | - | - |
| e1p_e2a_all | - | - | BASE pooled 116 | - | beh (trimmed) | - | - | - | - |
| e0_a2e | BASE 128 | beh (10 < 64) | - | - | - | - | - | - | - |
| e0_e2a | BASE pooled 100 | beh (8 < 64) | - | - | - | - | - | - | - |
| aagr_a2e | BASE 128 | beh (20 < 64) | - | - | - | - | - | - | - |
| aagr_e2a | BASE 74 | beh (12 < 64) | - | - | - | - | - | - | - |
| acon_a2e | BASE 128 | beh (0 < 64) | - | - | - | - | - | - | - |
| acon_e2a | beh (44 < 64) | beh (0 < 64) | - | - | - | - | - | - | - |
| sc_agree_a2e | BASE 128 | BASE 86 | beh (design) | beh (design) | beh (trimmed) | beh (design) | beh (design) | beh (design) | beh (design) |
| sc_conflict_stmt_a2e | BASE 72 | beh (20 < 64) | beh (design) | beh (design) | beh (design) | beh (design) | beh (design) | beh (design) | beh (design) |
| sc_conflict_event_a2e | beh (12 < 64) | beh (4 < 64) | beh (design) | beh (design) | beh (design) | beh (design) | beh (design) | beh (design) | beh (design) |
| rsite_e1_a2e | BASE 128 | BASE 128 | - | - | - | - | - | - | - |
| l_ant | BASE 126 | BASE 126 | BASE 126 | BASE 126 | BASE 126 | beh (design) | BASE 126 | BASE 126 | BASE 126 |
| l_prop | BASE 127 | BASE 128 | BASE 128 | BASE 128 | BASE 128 | beh (design) | BASE 128 | BASE 128 | BASE 128 |
| l_cf | anchor | anchor | BASE 98 | BASE 88 | BASE 98 | beh (design) | BASE 83 | BASE 95 | BASE 88 |
| l_ioi | anchor | anchor | BASE 128 | BASE 128 | BASE 128 | beh (design) | BASE 128 | BASE 128 | BASE 128 |
| l_bind | BASE 128 | BASE 128 | BASE 128 | beh (62 < 64) | beh (10 < 64) | beh (design) | BASE 128 | BASE pooled 80 | BASE 128 |
| l_ind | BASE pooled 66 | BASE 128 | BASE 128 | BASE pooled 92 | BASE 128 | beh (design) | BASE 128 | beh (18 < 64) | BASE pooled 64 |

# Dial-E yields (p6-data)

Blank-clause recount (before any split). 'strict' = `the _ (was|is) (too) X` at the end of the sentence (the other session's pattern, 82 % = 615 / 749); 'BC_RE' = the builder's pattern `(det) _ (was|is|were|are) [degree adverbs] TRIGGER` (moetrace/ext14_data.py), the Dial-E item rule.

| Twin set | Twins | strict pattern | BC_RE (item rule) |
|---|---|---|---|
| core 776 | 776 | 542 (70%) | 654 (84%) |
| 749 dir_f1 twins | 749 | 526 (70%) | 648 (87%) |

## Items and pairs per cell (after R1 / R2 / R4 dedup / completeness; before the scan)

| Cell | Family | Items | of which core 776 | names | Pairs | Main losses (items or pairs) |
|---|---|---|---|---|---|---|
| e1_a2e | qwen | 2973 | 589 | 528 | 5946 | R1_answer_multi_token 1258; item:not_both_orders_built 1222; R2_length_mismatch 1076; item:R4_duplicate_prompt 106 |
| e1_e2a | qwen | 2969 | 589 | 528 | 5938 | R1_answer_multi_token 1430; item:not_both_orders_built 1226; R2_length_mismatch 964; item:R4_duplicate_prompt 106 |
| e3_a2e | qwen | 3049 | 643 | 525 | 3049 | R1_answer_multi_token 629; R2_length_mismatch 541; R2_2_differing_tokens 66; item:R4_duplicate_prompt 10 |
| e3_e2a | qwen | 3061 | 648 | 528 | 3061 | W3_trigger_multi_token 715; W4_length_mismatch 482; R2_option_not_one_token 30; item:R4_duplicate_prompt 13 |
| e1p_a2e | qwen | 1874 | 485 | 275 | 3748 | item:P2_not_in_hand_list 1313; R1_answer_multi_token 746; item:P1_hand_list_no_paraphrase 509; item:not_both_orders_built 448 |
| e1p_e2a | qwen | 1874 | 485 | 275 | 3748 | item:P2_not_in_hand_list 1313; R2_length_mismatch 684; item:P1_hand_list_no_paraphrase 509; item:not_both_orders_built 448 |
| e2_a2e | qwen | 2973 | 589 | 528 | 5946 | R1_answer_multi_token 1258; item:not_both_orders_built 1222; R2_length_mismatch 1076; item:R4_duplicate_prompt 106 |
| e2_e2a | qwen | 2969 | 589 | 528 | 5938 | R1_answer_multi_token 1430; item:not_both_orders_built 1226; R2_length_mismatch 964; item:R4_duplicate_prompt 106 |
| sc_agree_a2e | qwen | 3051 | 643 | 525 | 6102 | R1_answer_multi_token 1258; item:not_both_orders_built 1242; R2_length_mismatch 1082; R2_2_differing_tokens 132 |
| sc_conflict_a2e | qwen | 3051 | 643 | 525 | 6102 | R1_answer_multi_token 1258; item:not_both_orders_built 1242; R2_length_mismatch 1082; R2_2_differing_tokens 132 |
| rsite_e1_a2e | qwen | 3030 | 575 | 240 | 6060 | R1_answer_multi_token 1258; item:not_both_orders_built 1165; R2_length_mismatch 818; R2_3_differing_tokens 254 |
| e3p3_a2e | qwen | 2923 | 624 | 506 | 2923 | R1_answer_multi_token 602; R2_length_mismatch 517; E3P3_complement_clause 105; E3P3_text_before_blank_np 53 |
| e1_a2e | mistral | 1586 | 571 | 156 | 3172 | R1_answer_multi_token 3702; item:not_both_orders_built 2642; R2_length_mismatch 1350; R2_2_differing_tokens 210 |
| e1_e2a | mistral | 1572 | 571 | 153 | 3144 | R1_answer_multi_token 3178; item:not_both_orders_built 2656; R2_length_mismatch 1880; R2_2_differing_tokens 244 |
| e3_a2e | mistral | 1642 | 626 | 154 | 1642 | R1_answer_multi_token 1851; R2_length_mismatch 675; R2_2_differing_tokens 114; R2_3_differing_tokens 11 |
| e3_e2a | mistral | 1636 | 630 | 153 | 1636 | W3_trigger_multi_token 1589; W4_length_mismatch 940; R2_option_not_one_token 129; item:R4_duplicate_prompt 7 |
| e1p_a2e | mistral | 1110 | 468 | 96 | 2220 | R1_answer_multi_token 2062; item:P2_not_in_hand_list 1313; item:not_both_orders_built 1243; item:P1_hand_list_no_paraphrase 509 |
| e1p_e2a | mistral | 1108 | 468 | 96 | 2216 | R2_length_mismatch 1418; item:P2_not_in_hand_list 1313; item:not_both_orders_built 1245; R1_answer_multi_token 884 |
| e2_a2e | mistral | 1586 | 571 | 156 | 3172 | R1_answer_multi_token 3702; item:not_both_orders_built 2642; R2_length_mismatch 1350; R2_2_differing_tokens 210 |
| e2_e2a | mistral | 1572 | 571 | 153 | 3144 | R1_answer_multi_token 3178; item:not_both_orders_built 2656; R2_length_mismatch 1880; R2_2_differing_tokens 244 |
| sc_agree_a2e | mistral | 1644 | 626 | 154 | 3288 | R1_answer_multi_token 3702; item:not_both_orders_built 2652; R2_length_mismatch 1350; R2_2_differing_tokens 228 |
| sc_conflict_a2e | mistral | 1644 | 626 | 154 | 3288 | R1_answer_multi_token 3702; item:not_both_orders_built 2652; R2_length_mismatch 1350; R2_2_differing_tokens 228 |
| rsite_e1_a2e | mistral | 2360 | 571 | 374 | 4720 | R1_answer_multi_token 3702; item:not_both_orders_built 1851; item:R4_duplicate_prompt 90 |
| e3p3_a2e | mistral | 1572 | 606 | 149 | 1572 | R1_answer_multi_token 1777; R2_length_mismatch 648; E3P3_complement_clause 105; R2_2_differing_tokens 101 |


## Design notes and caveats

- **Item pool (Dial E, Set C, R-site).** WinoGrande train_xl twins in Phase-3 geometry (the two sentences differ only in their final
  word) with a copular blank clause `(det) _ (was|is|were|are) [degree adverbs] TRIGGER`; core = the 776 Phase-3 shared twins;
  extension twins are scanned only for cells with < 256 core margin pairs. One discovery / validation assignment per twin
  (data/dial/twin_split.json) for every cell; cap 128 pairs per split filled in priority order (core, number of Dial-E cells the twin
  passes in the model, built in both tokenizer families, twin rank), so that the paired intersections across levels are large.
- **Attribute -> entity frame P3** (FRAME PILOT, data/dial/qa/frame_pilot.md; coordinator rule, confirmed 2026-10-16): the E1-family
  a2e cells (e1, e1p, e2, rsite, e0, Dial A) use `{statements}. Fill in the blank: The _ was X.\nAnswer: the` (P0 ', so the _' was at
  chance on objects in both models). New cell e3p3_a2e = E3 content in the same frame (`{event clause}. Fill in the blank: The _ was X.`;
  event clause = the sentence before the blank clause minus the connective; audit in data/dial/qa/completeness.md), so that
  E1 -> E1p -> E2 -> E3p3 is frame-matched; e3_a2e (F1 sentence) stays as the Phase-5 anchor. Set C uses the P3 layout
  (`{statement}. Fill in the blank: <original sentence>`). e2a cells keep the Phase-3 option-swap frame. E0 in P3 reads
  `Of the X and the Y. Fill in the blank: The _ was X.` (a fragment; frame-matched prior-only control).
- **PP blanks excluded** from the whole pool (D5_blank_in_pp: the word before the blank NP is a preposition, e.g. 'the sauce on the _ was',
  'the job of _ was'): the blank is not the subject of the blank clause. e3p3 additionally excludes complement clauses ('I thought the _
  was') and items where anything other than connective words / punctuation precedes the determiner + blank.
- **Association-neutral main sets** (coordinator 2026-10-16): WinoGrande cells keep only items whose two triggers have |probe| < 1
  under the model's responding Phase-5 probes (Qwen3 v1 and v2, Mixtral v2). Strong-prior items are used only in Dial A / E0.
- **Small cells (coordinator rule 2026-10-16).** < 64 validation pairs but >= 64 pairs in discovery + validation -> `pooled`
  (registry flag; family 'main' with discovery = validation = all passing pairs, so layers / heads for T5 are chosen on the same items);
  < 64 pooled -> behaviour only (`insufficient`, no tier 1), with the behaviour record (items whose clean prompts follow the target
  answer in every order vs the competing answer: the prior for acon, the event / statement for Set C, the copied word for E2).
- **All-strata E1-family cells (`<cell>_all`; coordinator decision 2026-10-17 03:15Z).** For the dense vs MoE and capacity
  comparisons only: e1_a2e / e1_e2a / e1p_* of the Stage-2 models whose association-neutral set is insufficient run on all association
  strata (e.g. Mistral-7B e1_a2e: 47 items pass both orders, 19 of them neutral); matching Stage-1 cells e1_a2e_all / e1_e2a_all for
  mixtral_bos (twin of Mistral-7B) and qwen3 (anchor of the Qwen3 dense series), minimal tier 1. Items that also pass in the twin /
  series (Stage 1) or that the counterpart selected (Stage 2) go first, so the comparisons are on identical items where both pass
  (Mixtral-Mistral: 27 shared items in e1_a2e and e1_e2a). Each pair carries the association class by the model's own probes
  (assoc_own, stratum_neutral_own) and by the counterpart's probes (assoc_counterpart, stratum_neutral_counterpart). The neutral cells
  stay primary for the pre-registered dial predictions.
- **Mixtral Dial A / E0 are behaviour only (why).** (1) 135 of the 378 controlled noun-pair items have a multi-token noun in the 32k
  vocabulary (elephant, whale, feather, ...), leaving 243 (batch 2 was added for this); (2) Mixtral expresses weak and often incoherent
  lexical priors for them: under the responding probe v2 only 38 % of the items are coherent (big -> elephant AND small -> mouse), the
  median |v2| is 0.4-0.6, and 12 items reach strength >= 1 on both triggers (Qwen3: 165 of 328); the in-frame prior (E0 margin, no
  evidence) agrees: 20 items with E0 strength >= 1, 8 pass E0 in both orders (Qwen3: 227 / 179); (3) WinoGrande twins add < 20 items
  (adversarially filtered against such priors). A lower threshold (strength >= 0.5: 33 controlled items) would stop being a strong-
  prior set. So Mixtral Dial A / E0 stay behaviour only; the Qwen3 cells carry Dial A.
- **Dedup.** E3 / Set C: Phase-3 placeholder key (same sentence template with other options = duplicate). E0-E2 / R-site / Dial A:
  exact normalised text (the event is gone, so the same trigger pair with other entities is a different item; `trig_key` keeps the
  trigger pair for clustering).
- **E1p paraphrases**: hand-checked list (data/dial/paraphrase_hand.json, 372 triggers, 34 without an acceptable paraphrase);
  WordNet candidates were often wrong in a 40-item look (rusty -> chromatic, retiring -> past), so they are not used. The paraphrase
  must differ from both triggers and share no 4-letter stem with either.
- **E2 caveat (coordinator QA).** With gradable adjectives and degree adverbs, negation does not entail the antonym ("the bowl is not
  too big" does not entail "too small"; "not very young" does not entail "very old"). E2-a2e still works by elimination ("the pot was
  not empty" excludes the pot); E2-e2a relies on pragmatic contrast only. Per pair: `degree_adv` (a degree adverb in the blank
  clause) and `antonym_type` (complementary for the hand list moetrace/ext14_data.COMPLEMENTARY: open / closed, alive / dead,
  present / absent, ...; full / empty and every other pair count as gradable); both are strata in the registry, so E2 can be reported
  overall and on the no-degree-adverb / complementary strata.
- **R-site.** Corrupted prompt = E1-a2e with the two entity mentions in the evidence swapped (two differing tokens, a documented
  exception to R2), same question trigger; the clean prompt is E1-a2e's prompt A. (The plan's alternative, swapping the two
  attributes, states the same binding; the brief's entity swap was built.)
- **Set C.** The statement is placed before the whole original sentence; Delta is signed toward the statement's answer.
  sc_conflict is split into sc_conflict_stmt_a2e (clean prefers the statement, margin both ways) and sc_conflict_event_a2e
  (prefers the event; trig_a / trig_b flipped so that the clean answer is 'true'); both orders of an item must fall in the same subset.
- **Ladder.** l_ant: LRE adjective_antonym (100 samples) + curated antonyms (data/dial/curated_antonyms.json); 3 templates ('The
  opposite of X is', 'The antonym of X is', 'Q: What is the opposite of X?\nA:'); split by word pair. l_prop: LRE commonsense
  relations with single-word subjects (fruit colours, substance phase, object superclass, work location) + curated object colour /
  taste / habitat / material (data/dial/curated_props.json); split by subject halves per relation. l_bind / l_ind: synthetic, word
  lists restricted to words that are single tokens in both families (identical texts in both); 400 items x 2 orders.
- **Dial A / E0.** Controlled noun pairs with a designed prior (data/dial/curated_dialA_nouns.json, batch 1 + batch 2 with nouns
  that are single tokens in both families) plus WinoGrande twins with a coherent prior (strength >= 1); the agree / conflict label and
  the E0 answer orientation come from each model's own probe values.


## FRAME PILOT (p6-data, E1 evidence, attribute -> entity; behaviour only)

200 core twins, the same for every frame and both models, both statement orders (pairs = items x 2). Neutral = Phase-5 association probes |v| < 1 for both triggers (Qwen3 v1 and v2, Mixtral v2). Frames: P0 'Fill in the blank: {A} and {B}, so the _ was X.\nAnswer: the'; P1 'Fill in the blank: {A}. {B}. The _ was X.\nAnswer: the'; P2 'Fill in the blank: {A} and {B}. The _ was X.\nAnswer: the'; P3 '{A} and {B}. Fill in the blank: The _ was X.\nAnswer: the'; P4 '{A} and {B}. The one that was X was the' (no wrapper, reference). First-mention bias = mean logit(first-mentioned entity) - logit(second) over all prompts (answers balanced).

| Model | Frame | Stratum | Items | Margin both ways | Correct both directions | Items both orders margin | Order gap (o1 - o2) | First-mention bias | Median drop |
|---|---|---|---|---|---|---|---|---|---|
| qwen3 | P0 | objects | 174 | 0.10 | 0.40 | 0.03 | +0.01 | +0.60 | 1.50 |
| qwen3 | P0 | objects (neutral) | 152 | 0.08 | 0.39 | 0.01 | +0.00 | +0.65 | 1.38 |
| qwen3 | P0 | names | 26 | 0.62 | 0.88 | 0.46 | +0.00 | +0.15 | 3.56 |
| qwen3 | P0 | names (neutral) | 26 | 0.62 | 0.88 | 0.46 | +0.00 | +0.15 | 3.56 |
| qwen3 | P0 | all | 200 | 0.17 | 0.47 | 0.09 | +0.01 | +0.54 | 1.62 |
| qwen3 | P0 | all (neutral) | 178 | 0.16 | 0.47 | 0.07 | +0.00 | +0.58 | 1.62 |
| qwen3 | P1 | objects | 174 | 0.09 | 0.46 | 0.01 | +0.02 | +0.03 | 1.25 |
| qwen3 | P1 | objects (neutral) | 152 | 0.07 | 0.46 | 0.00 | +0.01 | +0.07 | 1.12 |
| qwen3 | P1 | names | 26 | 0.46 | 0.75 | 0.23 | +0.00 | -0.08 | 3.31 |
| qwen3 | P1 | names (neutral) | 26 | 0.46 | 0.75 | 0.23 | +0.00 | -0.08 | 3.31 |
| qwen3 | P1 | all | 200 | 0.14 | 0.50 | 0.04 | +0.02 | +0.01 | 1.38 |
| qwen3 | P1 | all (neutral) | 178 | 0.12 | 0.50 | 0.03 | +0.01 | +0.05 | 1.25 |
| qwen3 | P2 | objects | 174 | 0.81 | 0.95 | 0.68 | -0.06 | +0.06 | 4.50 |
| qwen3 | P2 | objects (neutral) | 152 | 0.81 | 0.95 | 0.68 | -0.07 | +0.09 | 4.38 |
| qwen3 | P2 | names | 26 | 0.79 | 0.94 | 0.69 | -0.19 | +0.39 | 5.75 |
| qwen3 | P2 | names (neutral) | 26 | 0.79 | 0.94 | 0.69 | -0.19 | +0.39 | 5.75 |
| qwen3 | P2 | all | 200 | 0.81 | 0.95 | 0.68 | -0.08 | +0.10 | 4.62 |
| qwen3 | P2 | all (neutral) | 178 | 0.81 | 0.95 | 0.69 | -0.09 | +0.13 | 4.50 |
| qwen3 | P3 | objects | 174 | 0.97 | 0.99 | 0.94 | +0.01 | +0.17 | 6.88 |
| qwen3 | P3 | objects (neutral) | 152 | 0.97 | 0.99 | 0.94 | +0.00 | +0.14 | 6.81 |
| qwen3 | P3 | names | 26 | 0.90 | 0.96 | 0.85 | -0.04 | -0.25 | 8.56 |
| qwen3 | P3 | names (neutral) | 26 | 0.90 | 0.96 | 0.85 | -0.04 | -0.25 | 8.56 |
| qwen3 | P3 | all | 200 | 0.96 | 0.98 | 0.93 | +0.00 | +0.11 | 7.06 |
| qwen3 | P3 | all (neutral) | 178 | 0.96 | 0.98 | 0.93 | -0.01 | +0.09 | 7.00 |
| qwen3 | P4 | objects | 174 | 0.49 | 0.81 | 0.33 | -0.02 | -0.13 | 3.50 |
| qwen3 | P4 | objects (neutral) | 152 | 0.47 | 0.81 | 0.30 | -0.04 | -0.16 | 3.38 |
| qwen3 | P4 | names | 26 | 0.38 | 0.54 | 0.19 | -0.08 | +0.41 | 4.31 |
| qwen3 | P4 | names (neutral) | 26 | 0.38 | 0.54 | 0.19 | -0.08 | +0.41 | 4.31 |
| qwen3 | P4 | all | 200 | 0.47 | 0.78 | 0.31 | -0.03 | -0.06 | 3.50 |
| qwen3 | P4 | all (neutral) | 178 | 0.46 | 0.77 | 0.29 | -0.04 | -0.08 | 3.50 |
| mixtral_bos | P0 | objects | 174 | 0.02 | 0.06 | 0.01 | -0.01 | +0.17 | -0.38 |
| mixtral_bos | P0 | objects (neutral) | 153 | 0.01 | 0.05 | 0.00 | -0.02 | +0.19 | -0.38 |
| mixtral_bos | P0 | names | 26 | 0.00 | 0.00 | 0.00 | +0.00 | +0.22 | -0.50 |
| mixtral_bos | P0 | names (neutral) | 25 | 0.00 | 0.00 | 0.00 | +0.00 | +0.25 | -0.50 |
| mixtral_bos | P0 | all | 200 | 0.01 | 0.05 | 0.01 | -0.01 | +0.18 | -0.38 |
| mixtral_bos | P0 | all (neutral) | 178 | 0.01 | 0.04 | 0.00 | -0.02 | +0.20 | -0.38 |
| mixtral_bos | P1 | objects | 174 | 0.01 | 0.06 | 0.01 | +0.01 | +0.11 | -0.44 |
| mixtral_bos | P1 | objects (neutral) | 153 | 0.02 | 0.06 | 0.01 | +0.01 | +0.14 | -0.50 |
| mixtral_bos | P1 | names | 26 | 0.00 | 0.04 | 0.00 | +0.00 | -0.01 | -0.50 |
| mixtral_bos | P1 | names (neutral) | 25 | 0.00 | 0.04 | 0.00 | +0.00 | +0.00 | -0.50 |
| mixtral_bos | P1 | all | 200 | 0.01 | 0.06 | 0.01 | +0.00 | +0.09 | -0.50 |
| mixtral_bos | P1 | all (neutral) | 178 | 0.01 | 0.06 | 0.01 | +0.01 | +0.12 | -0.50 |
| mixtral_bos | P2 | objects | 174 | 0.03 | 0.23 | 0.01 | -0.01 | -0.20 | 0.38 |
| mixtral_bos | P2 | objects (neutral) | 153 | 0.02 | 0.22 | 0.01 | -0.01 | -0.20 | 0.25 |
| mixtral_bos | P2 | names | 26 | 0.00 | 0.12 | 0.00 | +0.00 | -0.05 | -0.25 |
| mixtral_bos | P2 | names (neutral) | 25 | 0.00 | 0.12 | 0.00 | +0.00 | -0.04 | -0.19 |
| mixtral_bos | P2 | all | 200 | 0.03 | 0.21 | 0.01 | -0.01 | -0.18 | 0.25 |
| mixtral_bos | P2 | all (neutral) | 178 | 0.02 | 0.21 | 0.01 | -0.01 | -0.17 | 0.25 |
| mixtral_bos | P3 | objects | 174 | 0.24 | 0.79 | 0.14 | -0.01 | +0.12 | 2.00 |
| mixtral_bos | P3 | objects (neutral) | 153 | 0.24 | 0.80 | 0.14 | -0.01 | +0.12 | 2.00 |
| mixtral_bos | P3 | names | 26 | 0.25 | 0.62 | 0.15 | -0.04 | +0.23 | 1.50 |
| mixtral_bos | P3 | names (neutral) | 25 | 0.26 | 0.62 | 0.16 | -0.04 | +0.26 | 1.50 |
| mixtral_bos | P3 | all | 200 | 0.24 | 0.77 | 0.14 | -0.01 | +0.13 | 2.00 |
| mixtral_bos | P3 | all (neutral) | 178 | 0.24 | 0.77 | 0.14 | -0.02 | +0.14 | 1.88 |
| mixtral_bos | P4 | objects | 174 | 0.32 | 0.63 | 0.18 | +0.07 | -0.46 | 2.25 |
| mixtral_bos | P4 | objects (neutral) | 153 | 0.31 | 0.62 | 0.17 | +0.05 | -0.47 | 2.25 |
| mixtral_bos | P4 | names | 26 | 0.10 | 0.48 | 0.04 | -0.04 | +0.29 | 1.38 |
| mixtral_bos | P4 | names (neutral) | 25 | 0.10 | 0.50 | 0.04 | -0.04 | +0.31 | 1.44 |
| mixtral_bos | P4 | all | 200 | 0.29 | 0.61 | 0.16 | +0.05 | -0.37 | 2.12 |
| mixtral_bos | P4 | all (neutral) | 178 | 0.28 | 0.60 | 0.15 | +0.04 | -0.36 | 2.12 |

## Decision rule (coordinator, pre-registered)

Highest neutral-objects correct-both among P0-P3 with |order gap| <= 10 pp; if none > 60 % in a model, keep P0.

- qwen3: {"frame": "P3", "neutral_objects_correct_both": 0.9868421052631579, "order_gap": 0.0}
- mixtral_bos: {"frame": "P3", "neutral_objects_correct_both": 0.7973856209150327, "order_gap": 0.01307189542483661}


## Cells (registry data/dial/p6_conditions.json)

| Cell | Model | Pool items | Items passing (all strata) | Items passing (main set) | Order pass rates | Top-1 (selected) | Disc / val pairs | Names / objects (val) | Status |
|---|---|---|---|---|---|---|---|---|---|
| l_cf | qwen3 | anchor | | | | | | | known share 0.54, runs results/qwen3_str |
| l_cf | mixtral_bos | anchor | | | | | | | known share 0.58, runs results/mixtral_bos_str |
| l_cf | mistral7b_bos | 212 | 196 | 196 | {} | 0.2194 | 98 / 98 | 0 / 98 | BASE READY |
| l_cf | qwen3_4b | 208 | 173 | 173 | {} | 0.1618 | 90 / 83 | 0 / 83 | BASE READY |
| l_cf | qwen3_8b | 212 | 188 | 188 | {} | 0.1543 | 93 / 95 | 0 / 95 | BASE READY |
| l_cf | qwen2_7b | 212 | 181 | 181 | {} | 0.2376 | 93 / 88 | 0 / 88 | BASE READY |
| l_cf | qwen3_1p7b | 198 | 146 | 146 | {} | 0.2123 | 79 / 67 | 0 / 67 | cases ready |
| l_cf | qwen3_14b | 212 | 185 | 185 | {} | 0.1351 | 97 / 88 | 0 / 88 | BASE READY |
| l_cf | qwen2_57b | 217 | 202 | 202 | {} | 0.2723 | 104 / 98 | 0 / 98 | BASE READY |
| l_ioi | qwen3 | anchor | | | | | | | known share 0.92, runs results/ioi_qwen3_s2io |
| l_ioi | mixtral_bos | anchor | | | | | | | known share 0.8, runs results/ioi_mixtral_bos_s2io |
| l_ioi | mistral7b_bos | 1600 | 1598 | 1598 | {} | 0.6484 | 128 / 128 | 128 / 0 | BASE READY |
| l_ioi | qwen3_4b | 1600 | 1599 | 1599 | {} | 0.6172 | 128 / 128 | 128 / 0 | BASE READY |
| l_ioi | qwen3_8b | 1600 | 1599 | 1599 | {} | 0.668 | 128 / 128 | 128 / 0 | BASE READY |
| l_ioi | qwen2_7b | 1600 | 1573 | 1573 | {} | 0.5938 | 128 / 128 | 128 / 0 | BASE READY |
| l_ioi | qwen3_1p7b | 1600 | 1587 | 1587 | {} | 0.5898 | 128 / 128 | 128 / 0 | cases ready |
| l_ioi | qwen3_14b | 1600 | 1598 | 1598 | {} | 0.7461 | 128 / 128 | 128 / 0 | BASE READY |
| l_ioi | qwen2_57b | 1600 | 1592 | 1592 | {} | 0.7031 | 128 / 128 | 128 / 0 | BASE READY |
| l_bind | qwen3 | 1600 | 130 | 130 | {'1': 0.2419, '2': 0.2462} | 1.0 | 128 / 128 | 0 / 128 | BASE READY |
| l_bind | mixtral_bos | 1600 | 502 | 502 | {'1': 0.5156, '2': 0.4975} | 0.9922 | 128 / 128 | 0 / 128 | BASE READY |
| l_bind | mistral7b_bos | 1600 | 145 | 145 | {'1': 0.2756, '2': 0.2775} | 0.9883 | 128 / 128 | 0 / 128 | BASE READY |
| l_bind | qwen3_4b | 1600 | 213 | 213 | {'1': 0.3481, '2': 0.3475} | 0.9961 | 128 / 128 | 0 / 128 | BASE READY |
| l_bind | qwen3_8b | 1600 | 40 | 40 | {'1': 0.2112, '2': 0.1963} | 1.0 | 80 / 80 | 0 / 80 | BASE READY |
| l_bind | qwen2_7b | 1600 | 31 | 31 | {'1': 0.1244, '2': 0.1231} | 1.0 | 28 / 34 | 0 / 34 | insufficient |
| l_bind | qwen3_1p7b | 1600 | 75 | 75 | {'1': 0.275, '2': 0.2825} | 0.98 | 72 / 78 | 0 / 78 | cases ready |
| l_bind | qwen3_14b | 1600 | 344 | 344 | {'1': 0.4881, '2': 0.4888} | 1.0 | 128 / 128 | 0 / 128 | BASE READY |
| l_bind | qwen2_57b | 1600 | 5 | 5 | {'1': 0.0219, '2': 0.0138} | 1.0 | 6 / 4 | 0 / 4 | insufficient |
| l_ant | qwen3 | 210 | 161 | 161 | {} | 0.5714 | 126 / 126 | 0 / 126 | BASE READY |
| l_ant | mixtral_bos | 179 | 160 | 160 | {} | 0.6111 | 126 / 126 | 0 / 126 | BASE READY |
| l_ant | mistral7b_bos | 179 | 158 | 158 | {} | 0.6468 | 126 / 126 | 0 / 126 | BASE READY |
| l_ant | qwen3_4b | 210 | 175 | 175 | {} | 0.4603 | 126 / 126 | 0 / 126 | BASE READY |
| l_ant | qwen3_8b | 210 | 182 | 182 | {} | 0.381 | 126 / 126 | 0 / 126 | BASE READY |
| l_ant | qwen2_7b | 210 | 140 | 140 | {} | 0.4524 | 126 / 126 | 0 / 126 | BASE READY |
| l_ant | qwen3_1p7b | 210 | 164 | 164 | {} | 0.4484 | 126 / 126 | 0 / 126 | cases ready |
| l_ant | qwen3_14b | 210 | 161 | 161 | {} | 0.4762 | 126 / 126 | 0 / 126 | BASE READY |
| l_ant | qwen2_57b | 210 | 179 | 179 | {} | 0.4087 | 126 / 126 | 0 / 126 | BASE READY |
| l_prop | qwen3 | 723 | 593 | 593 | {} | 0.3608 | 128 / 127 | 0 / 127 | BASE READY |
| l_prop | mixtral_bos | 305 | 245 | 245 | {} | 0.324 | 122 / 128 | 0 / 128 | BASE READY |
| l_prop | mistral7b_bos | 305 | 238 | 238 | {} | 0.3625 | 123 / 128 | 0 / 128 | BASE READY |
| l_prop | qwen3_4b | 723 | 541 | 541 | {} | 0.3438 | 128 / 128 | 0 / 128 | BASE READY |
| l_prop | qwen3_8b | 723 | 559 | 559 | {} | 0.3672 | 128 / 128 | 0 / 128 | BASE READY |
| l_prop | qwen2_7b | 723 | 564 | 564 | {} | 0.4102 | 128 / 128 | 0 / 128 | BASE READY |
| l_prop | qwen3_1p7b | 723 | 496 | 496 | {} | 0.3725 | 127 / 128 | 0 / 128 | cases ready |
| l_prop | qwen3_14b | 723 | 584 | 584 | {} | 0.3633 | 128 / 128 | 0 / 128 | BASE READY |
| l_prop | qwen2_57b | 723 | 576 | 576 | {} | 0.3984 | 128 / 128 | 0 / 128 | BASE READY |
| e1_a2e | qwen3 | 589 | 530 | 224 | {'1': 0.9406, '2': 0.9542} | 0.9336 | 128 / 128 | 16 / 112 | BASE READY |
| e1_a2e | mixtral_bos | 1586 | 211 | 98 | {'1': 0.2535, '2': 0.2453} | 0.9541 | 106 / 90 | 16 / 74 | BASE READY |
| e1_a2e | mistral7b_bos | 1586 | 47 | 19 | {'1': 0.1759, '2': 0.1564} | 0.9474 | 20 / 18 | 6 / 12 | insufficient |
| e1_a2e | qwen3_4b | 589 | 533 | 235 | {'1': 0.9593, '2': 0.9389} | 0.8008 | 128 / 128 | 12 / 116 | BASE READY |
| e1_a2e | qwen3_8b | 589 | 542 | 197 | {'1': 0.961, '2': 0.9457} | 0.9922 | 128 / 128 | 6 / 122 | BASE READY |
| e1_a2e | qwen2_7b | 1789 | 247 | 174 | {'1': 0.2359, '2': 0.2241} | 1.0 | 128 / 128 | 108 / 20 | BASE READY |
| e1_a2e | qwen3_1p7b | 589 | 53 | 23 | {'1': 0.1511, '2': 0.1698} | 0.8696 | 18 / 28 | 16 / 12 | insufficient |
| e1_a2e | qwen3_14b | 589 | 582 | 261 | {'1': 0.9949, '2': 0.9932} | 1.0 | 128 / 128 | 12 / 116 | BASE READY |
| e1_a2e | qwen2_57b | 589 | 374 | 210 | {'1': 0.7759, '2': 0.7691} | 0.9844 | 128 / 128 | 18 / 110 | BASE READY |
| e3_e2a | qwen3 | 648 | 636 | 293 | {} | 0.3711 | 128 / 128 | 13 / 115 | BASE READY |
| e3_e2a | mixtral_bos | 630 | 627 | 300 | {} | 0.3672 | 128 / 128 | 12 / 116 | BASE READY |
| e3_e2a | mistral7b_bos | 1636 | 765 | 316 | {} | 0.3359 | 128 / 128 | 12 / 116 | BASE READY |
| e3_e2a | qwen3_4b | 1848 | 869 | 379 | {} | 0.332 | 128 / 128 | 8 / 120 | BASE READY |
| e3_e2a | qwen3_8b | 1848 | 931 | 341 | {} | 0.3047 | 128 / 128 | 14 / 114 | BASE READY |
| e3_e2a | qwen2_7b | 648 | 534 | 369 | {} | 0.3164 | 128 / 128 | 8 / 120 | BASE READY |
| e3_e2a | qwen3_1p7b | 648 | 474 | 211 | {} | 0.2986 | 90 / 121 | 7 / 114 | cases ready |
| e3_e2a | qwen3_14b | 648 | 559 | 266 | {} | 0.3858 | 126 / 128 | 7 / 121 | BASE READY |
| e3_e2a | qwen2_57b | 648 | 544 | 306 | {} | 0.2852 | 128 / 128 | 13 / 115 | BASE READY |
| rsite_e1_a2e | qwen3 | 575 | 505 | 213 | {'1': 0.9391, '2': 0.9217} | 0.9414 | 128 / 128 | 10 / 118 | BASE READY |
| rsite_e1_a2e | mixtral_bos | 2360 | 242 | 144 | {'1': 0.2703, '2': 0.1987} | 1.0 | 128 / 128 | 42 / 86 | BASE READY |
| e0_a2e | qwen3 | 208 | 146 | 146 | {'1': 0.8077, '2': 0.8077} | 1.0 | 128 / 128 | 0 / 128 | BASE READY |
| e0_a2e | mixtral_bos | 32 | 5 | 5 | {'1': 0.3438, '2': 0.25} | 0.8 | 6 / 4 | 0 / 4 | insufficient |
| e0_e2a | qwen3 | 208 | 50 | 50 | {'1': 0.3413, '2': 0.3413} | 0.02 | 100 / 100 | 0 / 100 | BASE READY |
| e0_e2a | mixtral_bos | 31 | 4 | 4 | {'1': 0.1613, '2': 0.1935} | 0.0 | 4 / 4 | 0 / 4 | insufficient |
| aagr_a2e | qwen3 | 203 | 200 | 200 | {'1': 0.9901, '2': 0.9951} | 0.9805 | 128 / 128 | 0 / 128 | BASE READY |
| aagr_a2e | mixtral_bos | 31 | 10 | 10 | {'1': 0.4516, '2': 0.4194} | 0.9 | 4 / 16 | 0 / 16 | insufficient |
| aagr_e2a | qwen3 | 203 | 89 | 89 | {'1': 0.6108, '2': 0.67} | 0.0337 | 104 / 74 | 0 / 74 | BASE READY |
| aagr_e2a | mixtral_bos | 30 | 6 | 6 | {'1': 0.2, '2': 0.4333} | 0.3333 | 4 / 8 | 0 / 8 | insufficient; REVIEW: order gap 0.23 > 0.10 (pass rates by order {1: 0.2, 2: 0.4333}) |
| acon_a2e | qwen3 | 170 | 157 | 157 | {'1': 0.9588, '2': 0.9647} | 1.0 | 128 / 128 | 0 / 128 | BASE READY |
| acon_a2e | mixtral_bos | 12 | 0 | 0 | {'1': 0.0, '2': 0.0} | None | 0 / 0 | 0 / 0 | insufficient |
| acon_e2a | qwen3 | 170 | 22 | 22 | {'1': 0.3059, '2': 0.2941} | 0.0 | 20 / 24 | 0 / 24 | insufficient |
| acon_e2a | mixtral_bos | 12 | 0 | 0 | {'1': 0.0833, '2': 0.0} | None | 0 / 0 | 0 / 0 | insufficient |
| e1_e2a | qwen3 | 1789 | 795 | 305 | {'1': 0.5824, '2': 0.5623} | 0.1953 | 128 / 128 | 10 / 118 | BASE READY |
| e1_e2a | mixtral_bos | 1572 | 103 | 40 | {'1': 0.1349, '2': 0.1559} | 0.65 | 80 / 80 | 10 / 70 | BASE READY |
| e1_e2a | mistral7b_bos | 1572 | 52 | 17 | {'1': 0.0865, '2': 0.0935} | 0.3529 | 18 / 16 | 2 / 14 | insufficient |
| e1_e2a | qwen3_4b | 1789 | 286 | 124 | {'1': 0.3309, '2': 0.3208} | 0.3445 | 110 / 128 | 10 / 118 | BASE READY |
| e1_e2a | qwen3_8b | 1789 | 300 | 101 | {'1': 0.2946, '2': 0.3248} | 0.5792 | 84 / 118 | 2 / 116 | BASE READY |
| e1_e2a | qwen2_7b | 1789 | 325 | 230 | {'1': 0.2996, '2': 0.2974} | 0.4883 | 128 / 128 | 32 / 96 | BASE READY |
| e1_e2a | qwen3_1p7b | 589 | 85 | 43 | {'1': 0.2445, '2': 0.2309} | 0.2442 | 86 / 86 | 0 / 86 | cases ready |
| e1_e2a | qwen3_14b | 1789 | 559 | 236 | {'1': 0.4528, '2': 0.4522} | 0.3164 | 128 / 128 | 4 / 124 | BASE READY |
| e1_e2a | qwen2_57b | 1789 | 479 | 271 | {'1': 0.3935, '2': 0.403} | 0.0859 | 128 / 128 | 10 / 118 | BASE READY |
| e3p3_a2e | mixtral_bos | 1572 | 322 | 160 | {} | 0.75 | 80 / 80 | 14 / 66 | BASE READY |
| e3p3_a2e | qwen3 | 1824 | 328 | 128 | {} | 0.75 | 58 / 70 | 20 / 50 | BASE READY |
| e3p3_a2e | mistral7b_bos | 1572 | 252 | 106 | {} | 0.8302 | 106 / 106 | 9 / 97 | BASE READY |
| e3p3_a2e | qwen3_4b | 1824 | 355 | 159 | {} | 0.673 | 83 / 76 | 26 / 50 | BASE READY |
| e3p3_a2e | qwen3_8b | 1824 | 578 | 226 | {} | 0.8186 | 116 / 110 | 27 / 83 | BASE READY |
| e3p3_a2e | qwen2_7b | 1824 | 314 | 202 | {} | 0.8515 | 106 / 96 | 41 / 55 | BASE READY |
| e3p3_a2e | qwen3_1p7b | 624 | 32 | 15 | {} | 0.4667 | 8 / 7 | 1 / 6 | insufficient |
| e3p3_a2e | qwen3_14b | 1824 | 832 | 386 | {} | 0.7656 | 128 / 128 | 30 / 98 | BASE READY |
| e3p3_a2e | qwen2_57b | 1824 | 654 | 384 | {} | 0.8047 | 128 / 128 | 29 / 99 | BASE READY |
| e3_a2e | qwen3 | 1843 | 353 | 158 | {} | 0.6962 | 74 / 84 | 19 / 65 | BASE READY |
| e3_a2e | mixtral_bos | 1642 | 263 | 113 | {} | 0.5398 | 113 / 113 | 3 / 110 | BASE READY |
| e3_a2e | mistral7b_bos | 1642 | 258 | 111 | {} | 0.5856 | 111 / 111 | 4 / 107 | BASE READY |
| e3_a2e | qwen3_4b | 1843 | 426 | 168 | {} | 0.7083 | 90 / 78 | 6 / 72 | BASE READY |
| e3_a2e | qwen3_8b | 1843 | 671 | 251 | {} | 0.7377 | 116 / 128 | 21 / 107 | BASE READY |
| e3_a2e | qwen2_7b | 1843 | 265 | 160 | {} | 0.6875 | 92 / 68 | 9 / 59 | BASE READY |
| e3_a2e | qwen3_1p7b | 643 | 30 | 10 | {} | 0.3 | 7 / 3 | 0 / 3 | insufficient |
| e3_a2e | qwen3_14b | 1843 | 902 | 419 | {} | 0.7188 | 128 / 128 | 14 / 114 | BASE READY |
| e3_a2e | qwen2_57b | 643 | 330 | 187 | {} | 0.6738 | 97 / 90 | 7 / 83 | cases ready |
| e1p_a2e | qwen3 | 1685 | 1078 | 435 | {'1': 0.7662, '2': 0.768} | 0.9688 | 128 / 128 | 24 / 104 | BASE READY |
| e1p_a2e | mixtral_bos | 1110 | 55 | 22 | {'1': 0.1216, '2': 0.1279} | 0.8636 | 16 / 28 | 6 / 22 | insufficient |
| e1p_a2e | mistral7b_bos | 1110 | 5 | 2 | {'1': 0.0784, '2': 0.0874} | 1.0 | 2 / 2 | 0 / 2 | insufficient |
| e1p_a2e | qwen3_4b | 485 | 295 | 148 | {'1': 0.7588, '2': 0.7402} | 0.9648 | 128 / 128 | 14 / 114 | BASE READY |
| e1p_a2e | qwen3_8b | 1685 | 1113 | 444 | {'1': 0.7887, '2': 0.7786} | 1.0 | 128 / 128 | 10 / 118 | BASE READY |
| e1p_a2e | qwen2_7b | 1685 | 269 | 196 | {'1': 0.2605, '2': 0.27} | 1.0 | 128 / 128 | 86 / 42 | BASE READY |
| e1p_a2e | qwen3_1p7b | 485 | 25 | 12 | {'1': 0.1258, '2': 0.0948} | 0.8333 | 8 / 16 | 10 / 6 | insufficient |
| e1p_a2e | qwen3_14b | 485 | 458 | 205 | {'1': 0.9649, '2': 0.9649} | 0.9961 | 128 / 128 | 14 / 114 | BASE READY |
| e1p_a2e | qwen2_57b | 485 | 164 | 98 | {'1': 0.5196, '2': 0.5175} | 0.9949 | 102 / 94 | 16 / 78 | cases ready |
| e1p_e2a | qwen3 | 1685 | 689 | 268 | {'1': 0.5561, '2': 0.527} | 0.0938 | 128 / 128 | 18 / 110 | BASE READY |
| e1p_e2a | mixtral_bos | 1108 | 112 | 55 | {'1': 0.1661, '2': 0.1634} | 0.0273 | 110 / 110 | 38 / 72 | BASE READY |
| e1p_e2a | mistral7b_bos | 1108 | 58 | 25 | {'1': 0.0921, '2': 0.0903} | 0.02 | 30 / 20 | 6 / 14 | insufficient |
| e1p_e2a | qwen3_4b | 1685 | 258 | 122 | {'1': 0.2599, '2': 0.2398} | 0.0492 | 124 / 120 | 32 / 88 | BASE READY |
| e1p_e2a | qwen3_8b | 1685 | 300 | 121 | {'1': 0.3086, '2': 0.2926} | 0.0248 | 122 / 120 | 38 / 82 | BASE READY |
| e1p_e2a | qwen2_7b | 1685 | 288 | 198 | {'1': 0.2582, '2': 0.2487} | 0.0703 | 128 / 128 | 50 / 78 | BASE READY |
| e1p_e2a | qwen3_1p7b | 485 | 69 | 27 | {'1': 0.2371, '2': 0.2021} | 0.1111 | 24 / 30 | 2 / 28 | insufficient |
| e1p_e2a | qwen3_14b | 1685 | 587 | 242 | {'1': 0.4872, '2': 0.4777} | 0.0742 | 128 / 128 | 10 / 118 | BASE READY |
| e1p_e2a | qwen2_57b | 485 | 191 | 104 | {'1': 0.5464, '2': 0.5155} | 0.024 | 108 / 100 | 8 / 92 | cases ready |
| e2_a2e | mixtral_bos | 1586 | 2 | 0 | {'1': 0.0044, '2': 0.0025} | None | 0 / 0 | 0 / 0 | insufficient |
| e2_a2e | qwen3 | 1789 | 0 | 0 | {'1': 0.0, '2': 0.0} | None | 0 / 0 | 0 / 0 | insufficient |
| e2_a2e | mistral7b_bos | 1586 | 0 | 0 | {'1': 0.0006, '2': 0.0013} | None | 0 / 0 | 0 / 0 | insufficient |
| e2_a2e | qwen3_4b | 1789 | 0 | 0 | {'1': 0.0, '2': 0.0006} | None | 0 / 0 | 0 / 0 | insufficient |
| e2_a2e | qwen3_8b | 1789 | 0 | 0 | {'1': 0.0017, '2': 0.0006} | None | 0 / 0 | 0 / 0 | insufficient |
| e2_a2e | qwen2_7b | 1789 | 1 | 0 | {'1': 0.0006, '2': 0.0011} | None | 0 / 0 | 0 / 0 | insufficient |
| e2_a2e | qwen3_1p7b | 589 | 0 | 0 | {'1': 0.0, '2': 0.0} | None | 0 / 0 | 0 / 0 | insufficient |
| e2_a2e | qwen3_14b | 1789 | 2 | 1 | {'1': 0.0045, '2': 0.005} | 1.0 | 0 / 2 | 0 / 2 | insufficient |
| e2_a2e | qwen2_57b | 589 | 0 | 0 | {'1': 0.0017, '2': 0.0034} | None | 0 / 0 | 0 / 0 | insufficient |
| e2_e2a | mixtral_bos | 1572 | 11 | 9 | {'1': 0.0229, '2': 0.0216} | 0.6667 | 8 / 10 | 0 / 10 | insufficient |
| e2_e2a | qwen3 | 1789 | 12 | 3 | {'1': 0.0324, '2': 0.0358} | 0.0 | 2 / 4 | 2 / 2 | insufficient |
| e2_e2a | mistral7b_bos | 1572 | 25 | 10 | {'1': 0.0426, '2': 0.0433} | 0.25 | 8 / 12 | 2 / 10 | insufficient |
| e2_e2a | qwen3_4b | 1789 | 4 | 1 | {'1': 0.0151, '2': 0.0129} | 0.0 | 2 / 0 | 0 / 0 | insufficient |
| e2_e2a | qwen3_8b | 1789 | 6 | 4 | {'1': 0.0235, '2': 0.0307} | 0.125 | 6 / 2 | 2 / 0 | insufficient |
| e2_e2a | qwen2_7b | 1789 | 13 | 11 | {'1': 0.0458, '2': 0.0458} | 0.3636 | 12 / 10 | 8 / 2 | insufficient |
| e2_e2a | qwen3_1p7b | 589 | 15 | 5 | {'1': 0.0815, '2': 0.09} | 0.2 | 2 / 8 | 4 / 4 | insufficient |
| e2_e2a | qwen3_14b | 1789 | 19 | 6 | {'1': 0.0565, '2': 0.0604} | 0.1667 | 6 / 6 | 4 / 2 | insufficient |
| e2_e2a | qwen2_57b | 589 | 13 | 6 | {'1': 0.0764, '2': 0.0747} | 0.0 | 10 / 2 | 0 / 2 | insufficient |
| sc_agree_a2e | mixtral_bos | 1644 | 179 | 80 | {'1': 0.1977, '2': 0.191} | 0.9563 | 74 / 86 | 2 / 84 | BASE READY |
| sc_agree_a2e | qwen3 | 1843 | 552 | 219 | {'1': 0.4504, '2': 0.4227} | 0.9336 | 128 / 128 | 24 / 104 | BASE READY |
| sc_agree_a2e | mistral7b_bos | 1644 | 69 | 32 | {'1': 0.1204, '2': 0.1034} | 0.9375 | 64 / 64 | 0 / 64 | cases ready |
| sc_agree_a2e | qwen3_4b | 1843 | 784 | 329 | {'1': 0.6028, '2': 0.5437} | 0.9492 | 128 / 128 | 10 / 118 | cases ready |
| sc_agree_a2e | qwen3_8b | 1843 | 927 | 331 | {'1': 0.6571, '2': 0.6093} | 0.9805 | 128 / 128 | 4 / 124 | cases ready |
| sc_agree_a2e | qwen2_7b | 1843 | 24 | 18 | {'1': 0.0494, '2': 0.0288} | 0.9722 | 22 / 14 | 12 / 2 | insufficient |
| sc_agree_a2e | qwen3_1p7b | 643 | 4 | 3 | {'1': 0.0202, '2': 0.0156} | 1.0 | 6 / 0 | 0 / 0 | insufficient |
| sc_agree_a2e | qwen3_14b | 643 | 512 | 231 | {'1': 0.8585, '2': 0.8647} | 0.9531 | 128 / 128 | 8 / 120 | cases ready |
| sc_agree_a2e | qwen2_57b | 643 | 261 | 145 | {'1': 0.5739, '2': 0.535} | 0.9766 | 128 / 128 | 16 / 112 | cases ready |
| sc_conflict_stmt_a2e | mixtral_bos | 1644 | 19 | 10 | {'1': 0.0328, '2': 0.0182} | 1.0 | 12 / 8 | 0 / 8 | insufficient |
| sc_conflict_stmt_a2e | qwen3 | 1843 | 167 | 67 | {'1': 0.216, '2': 0.1769} | 0.9552 | 62 / 72 | 16 / 56 | BASE READY |
| sc_conflict_stmt_a2e | mistral7b_bos | 1644 | 4 | 2 | {'1': 0.0152, '2': 0.0176} | 1.0 | 2 / 2 | 0 / 2 | insufficient |
| sc_conflict_stmt_a2e | qwen3_4b | 1843 | 350 | 152 | {'1': 0.3559, '2': 0.2827} | 0.9844 | 128 / 128 | 30 / 98 | cases ready |
| sc_conflict_stmt_a2e | qwen3_8b | 1843 | 306 | 105 | {'1': 0.3163, '2': 0.2642} | 0.981 | 118 / 92 | 22 / 70 | cases ready |
| sc_conflict_stmt_a2e | qwen2_7b | 1843 | 3 | 3 | {'1': 0.006, '2': 0.006} | 1.0 | 4 / 2 | 2 / 0 | insufficient |
| sc_conflict_stmt_a2e | qwen3_1p7b | 643 | 1 | 0 | {'1': 0.0062, '2': 0.0016} | None | 0 / 0 | 0 / 0 | insufficient |
| sc_conflict_stmt_a2e | qwen3_14b | 1843 | 768 | 337 | {'1': 0.5475, '2': 0.5035} | 0.9805 | 128 / 128 | 22 / 106 | cases ready |
| sc_conflict_stmt_a2e | qwen2_57b | 643 | 11 | 8 | {'1': 0.0747, '2': 0.0653} | 1.0 | 8 / 8 | 4 / 4 | insufficient |
| sc_conflict_event_a2e | mixtral_bos | 1644 | 6 | 2 | {'1': 0.0091, '2': 0.0091} | 1.0 | 4 / 0 | 0 / 0 | insufficient |
| sc_conflict_event_a2e | qwen3 | 1843 | 10 | 6 | {'1': 0.0114, '2': 0.0157} | 1.0 | 4 / 8 | 0 / 8 | insufficient |
| sc_conflict_event_a2e | mistral7b_bos | 1644 | 3 | 2 | {'1': 0.0079, '2': 0.0067} | 1.0 | 4 / 0 | 0 / 0 | insufficient |
| sc_conflict_event_a2e | qwen3_4b | 1843 | 1 | 0 | {'1': 0.0033, '2': 0.0011} | None | 0 / 0 | 0 / 0 | insufficient |
| sc_conflict_event_a2e | qwen3_8b | 1843 | 28 | 16 | {'1': 0.0266, '2': 0.026} | 1.0 | 22 / 10 | 0 / 10 | insufficient |
| sc_conflict_event_a2e | qwen2_7b | 1843 | 0 | 0 | {'1': 0.0005, '2': 0.0} | None | 0 / 0 | 0 / 0 | insufficient |
| sc_conflict_event_a2e | qwen3_1p7b | 643 | 0 | 0 | {'1': 0.0, '2': 0.0016} | None | 0 / 0 | 0 / 0 | insufficient |
| sc_conflict_event_a2e | qwen3_14b | 1843 | 88 | 76 | {'1': 0.0543, '2': 0.0532} | 0.9868 | 88 / 64 | 0 / 64 | cases ready |
| sc_conflict_event_a2e | qwen2_57b | 643 | 7 | 3 | {'1': 0.0249, '2': 0.0311} | 0.5 | 2 / 4 | 0 / 4 | insufficient |
| l_ind | qwen3 | 800 | 33 | 33 | {'1': 0.1163, '2': 0.1263} | 0.9091 | 66 / 66 | 0 / 66 | BASE READY |
| l_ind | mixtral_bos | 800 | 144 | 144 | {'1': 0.3837, '2': 0.3488} | 0.8398 | 128 / 128 | 0 / 128 | BASE READY |
| l_ind | mistral7b_bos | 800 | 148 | 148 | {'1': 0.39, '2': 0.3588} | 0.9219 | 128 / 128 | 0 / 128 | BASE READY |
| l_ind | qwen3_4b | 800 | 132 | 132 | {'1': 0.3275, '2': 0.3063} | 0.7063 | 124 / 128 | 0 / 128 | BASE READY |
| l_ind | qwen3_8b | 800 | 9 | 9 | {'1': 0.0525, '2': 0.0488} | 0.7778 | 6 / 12 | 0 / 12 | insufficient |
| l_ind | qwen2_7b | 800 | 46 | 46 | {'1': 0.1263, '2': 0.1425} | 0.8804 | 92 / 92 | 0 / 92 | BASE READY |
| l_ind | qwen3_1p7b | 800 | 88 | 88 | {'1': 0.2137, '2': 0.2238} | 0.5568 | 82 / 94 | 0 / 94 | cases ready |
| l_ind | qwen3_14b | 800 | 32 | 32 | {'1': 0.14, '2': 0.135} | 0.8125 | 64 / 64 | 0 / 64 | BASE READY |
| l_ind | qwen2_57b | 800 | 272 | 272 | {'1': 0.5162, '2': 0.4925} | 0.9531 | 128 / 128 | 0 / 128 | BASE READY |
| e1_a2e_all | mixtral_bos | 1586 | 211 | 211 | {'1': 0.2535, '2': 0.2453} | 0.9492 | 128 / 128 | 14 / 114 | BASE READY |
| e1_a2e_all | qwen3 | 589 | 530 | 530 | {'1': 0.9406, '2': 0.9542} | 0.918 | 128 / 128 | 12 / 116 | BASE READY |
| e1_a2e_all | mistral7b_bos | 1586 | 47 | 47 | {'1': 0.1759, '2': 0.1564} | 0.9468 | 94 / 94 | 16 / 78 | BASE READY |
| e1_e2a_all | mixtral_bos | 1572 | 103 | 103 | {'1': 0.1349, '2': 0.1559} | 0.665 | 96 / 110 | 6 / 104 | BASE READY |
| e1_e2a_all | qwen3 | 1789 | 795 | 795 | {'1': 0.5824, '2': 0.5623} | 0.207 | 128 / 128 | 2 / 126 | BASE READY |
| e1_e2a_all | mistral7b_bos | 1572 | 52 | 52 | {'1': 0.0865, '2': 0.0935} | 0.4904 | 104 / 104 | 8 / 96 | BASE READY |
| e1p_a2e_all | mistral7b_bos | 1110 | 5 | 5 | {'1': 0.0784, '2': 0.0874} | 1.0 | 6 / 4 | 0 / 4 | insufficient |
| e1p_a2e_all | qwen2_57b | 485 | 164 | 164 | {'1': 0.5196, '2': 0.5175} | 0.9922 | 128 / 128 | 16 / 112 | cases ready |
| e1p_e2a_all | mistral7b_bos | 1108 | 58 | 58 | {'1': 0.0921, '2': 0.0903} | 0.0431 | 116 / 116 | 30 / 86 | BASE READY |
| e1p_e2a_all | qwen2_57b | 485 | 191 | 191 | {'1': 0.5464, '2': 0.5155} | 0.0117 | 128 / 128 | 10 / 118 | cases ready |

## Problems met and how they were handled

- **Qwen3 at chance on objects in the E1 a2e cloze** (P0 ', so the _ was X'; first-mention bias ~1 logit) -> FRAME PILOT (P0-P4, 200 core
  twins, both models, pre-set rule) -> P3 '{statements}. Fill in the blank: The _ was X.' for every E1-family a2e cell (Qwen3 neutral objects
  correct-both 0.39 -> 0.99, Mixtral 0.05 -> 0.80); new cell e3p3_a2e (E3 content in P3) for a frame-matched dial; Set C moved to the P3
  layout; old-frame scans renamed VOID_oldframe_*.
- **e3p3 truncation bug** (blank NP longer than a determiner: 'because the one on. Fill in the blank: The _ ...') -> PP-blank items excluded
  from the whole pool (D5_blank_in_pp), e3p3 eligibility = only connective word(s) / punctuation before a determiner + blank, complement
  clauses excluded, string-wise reconstruction checks on 100 % of e3p3 / Set C / R-site items (0 failures).
- **E2 semantics** (gradable adjectives + degree adverbs: negation does not entail the antonym) -> strata degree_adv / antonym_type.
- **Mixtral at chance on 'Pairs: a = b, c = d. a ='** -> IND PILOT (I0-I4, pre-set rule) -> I2 (newline 'a: b' lines + two distractor
  pairs); I3 (flat sequence) behaviour only because its STR is the final token (R2).
- **l_bind / l_ind too few passing items** (Qwen3 first-mention bias) -> more items (l_bind 1,600, l_ind 800); 'orange' removed from the
  objects (also a colour).
- **Extension scans were split-unbalanced** (the twin rank is split-ordered, so 'first 1,200 extension items by rank' took discovery twins
  only) -> interleaved selection that skips scanned items (scan_<cell>_ext<k>); affected Mixtral cells withdrawn and rescanned.
- **Small cells** -> coordinator rule: pooled (< 64 validation, >= 64 together) / behaviour only (< 64 pooled), REVIEW lines only for other
  problems.
- **Mixtral Dial A / E0 tiny pools** -> diagnosed (multi-token nouns, weak / incoherent v2 priors, in-frame E0 prior agrees); behaviour only.
- **Neutral rule leaves the dense twins too few E1 items** -> `_all` cells (all association strata, own + counterpart association labels,
  twin-preferred items first) for the dense vs MoE / capacity comparisons only.
- **Scope cut** (only what the attention vs MoE / dense question needs) -> tier 1 = sweep + joint + DLA + direct; heads only for e1_a2e,
  e3_e2a and NEED HEADS cells; no new grids; BASE READY = W2 + W4.
- **Queue throughput** (~15 chains, ~25-60 min wait per job) -> one GPU job per cell (stages back-to-back, .done markers kept), bundles of
  2 (Stage 1) / 4 (dense) cells.
- **Qwen2-57B / Qwen2-7B bf16 fragility** -> fp32 protocol applied by Engine(repo) from moetrace/models.py; Qwen2-57B held until
  p6-engine's PROTOCOL line (a watcher pattern that matched a quoted mention started it once; cancelled before any GPU work; watchers now
  only count lines that start with the engine tag); Qwen2-57B scan batches capped at 50k tokens, tier-1 budgets at 0.5x (two OOMs retried
  at 0.75x automatically); its tier 1 trimmed to the 10 twin-comparison cells.
- **Missed NEED HEADS e1_e2a/mixtral_bos** (stage list fetched before the request; done cells are not revisited) -> heads run as one job
  afterwards and the base analysis re-run.
- **Three agent stops (session limits)** -> every chain nohup setsid; resumed from disk, no GPU work redone.
- **Accidental smoke-test write** into results/p5_dir_f1_qwen3/base_summary.json (19:32Z on 10-16) -> restored with git checkout; test mode
  writes to the scratch dir only.


## IND PILOT (p6-data; l_ind formats, behaviour only)

200 items (4 key / value pairs of 8 distinct random words, single tokens in both families; the same for every frame and both models), both query orders (pairs = items x 2); STR = the query key (k1 <-> k2). I0 'Pairs: a = b, c = d. a ='; I1 'a: b\\nc: d\\na:'; I2 'a: b\\nc: d\\ne: f\\ng: h\\na:'; I3 'List: a b c d e f a' (behaviour reference only: the query key is the final token = STR at the readout position, which R2 forbids for patching); I1 / I2 start with a newline so that every key, also the first, is the same line-start token; I4 'The code for a is b. The code for c is d. The code for a is' (reference). First-mention bias = mean logit(value of the first-listed key) - logit(value of the second) over all prompts.

| Model | Frame | Margin both ways | Correct both directions | Items both orders margin | Order gap (o1 - o2) | First-mention bias | Median drop |
|---|---|---|---|---|---|---|---|
| qwen3 | I0 | 0.35 | 0.77 | 0.15 | +0.03 | +0.58 | 2.75 |
| qwen3 | I1 | 0.03 | 0.18 | 0.01 | +0.00 | +0.17 | -0.25 |
| qwen3 | I2 | 0.12 | 0.47 | 0.04 | -0.00 | +0.26 | 1.25 |
| qwen3 | I3 | 0.34 | 0.69 | 0.14 | -0.03 | +0.77 | 3.25 |
| qwen3 | I4 | 0.00 | 0.05 | 0.00 | +0.00 | -0.11 | -0.62 |
| mixtral_bos | I0 | 0.10 | 0.43 | 0.01 | -0.01 | +0.74 | 1.38 |
| mixtral_bos | I1 | 0.35 | 0.77 | 0.20 | -0.03 | +0.24 | 2.25 |
| mixtral_bos | I2 | 0.34 | 0.77 | 0.14 | -0.02 | +0.55 | 2.50 |
| mixtral_bos | I3 | 0.33 | 0.68 | 0.14 | +0.02 | +0.61 | 2.75 |
| mixtral_bos | I4 | 0.55 | 0.86 | 0.32 | -0.06 | +0.03 | 3.12 |

Decision (pre-set rule): **I2** (no I1-I3 frame reaches 0.8 in both models: best I1-I3 frame anyway (highest minimum over models, |gap| <= 0.10 where possible)).


## Stage-2 tokenizer identity (p6-data)

| Model | Family | Vocab identical | Same default special tokens | Prompts checked | Mismatches | Reuse family pair files |
|---|---|---|---|---|---|---|
| Mistral-7B-v0.1 (mistral7b_bos) | mistral | True | True | 80954 | 0 | True |
| Qwen2-7B (qwen2_7b) | qwen | False | True | 133010 | 0 | False |
| Qwen2-57B-A14B (qwen2_57b) | qwen | False | True | 133010 | 0 | False |
| Qwen3-1.7B-Base (qwen3_1p7b) | qwen | True | True | 133010 | 0 | True |
| Qwen3-4B-Base (qwen3_4b) | qwen | True | True | 133010 | 0 | True |
| Qwen3-8B-Base (qwen3_8b) | qwen | True | True | 133010 | 0 | True |
| Qwen3-14B-Base (qwen3_14b) | qwen | True | True | 133010 | 0 | True |

