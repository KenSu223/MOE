# Completeness audit (p6-data; Phase-5 F2-cut guard)

Checks are string-level on 100 % of built items, before any scan (moetrace/ext14_data.completeness). E3 a2e / Set C: the whole WinoGrande sentence (with the blank) is in the prompt. E3 e2a: prompt + ' ' + trigger = the filled sentence (only punctuation after it). E1 / E1p / E2 / R-site / E0 (WinoGrande items): the whole blank clause is in the prompt (a2e: `<det> _ <cop> [adv] TRIGGER.`; e2a: the prompt ends with the filled clause up to the trigger). E1: both filled blank clauses are in the evidence. E1p: neither trigger appears in the evidence, both paraphrases do. E2: each entity is stated as `not [adv] <other trigger>`. Whitespace is normalised (some WinoGrande sentences have double spaces). Items failing any check are dropped (column 'dropped').

| Cell | Family | Items built (both orders) | Check | Checked pairs | Failed | Items dropped |
|---|---|---|---|---|---|---|
| e1_a2e | qwen | 2973 | both_filled_clauses_in_evidence | 5946 | 0 | 0 |
| e1_a2e | qwen |  | whole_blank_clause_a | 5946 | 0 |  |
| e1_a2e | qwen |  | whole_blank_clause_b | 5946 | 0 |  |
| e1_e2a | qwen | 2969 | both_filled_clauses_in_evidence | 5938 | 0 | 0 |
| e1_e2a | qwen |  | whole_blank_clause_a | 5938 | 0 |  |
| e1_e2a | qwen |  | whole_blank_clause_b | 5938 | 0 |  |
| e3_a2e | qwen | 3049 | whole_sentence_a | 3049 | 0 | 0 |
| e3_a2e | qwen |  | whole_sentence_b | 3049 | 0 |  |
| e3_e2a | qwen | 3061 | prompt_plus_trigger_is_sentence_a | 3061 | 0 | 0 |
| e3_e2a | qwen |  | prompt_plus_trigger_is_sentence_b | 3061 | 0 |  |
| e1p_a2e | qwen | 1874 | no_verbatim_trigger_in_evidence | 3748 | 0 | 0 |
| e1p_a2e | qwen |  | paraphrases_in_evidence | 3748 | 0 |  |
| e1p_a2e | qwen |  | whole_blank_clause_a | 3748 | 0 |  |
| e1p_a2e | qwen |  | whole_blank_clause_b | 3748 | 0 |  |
| e1p_e2a | qwen | 1874 | no_verbatim_trigger_in_evidence | 3748 | 0 | 0 |
| e1p_e2a | qwen |  | paraphrases_in_evidence | 3748 | 0 |  |
| e1p_e2a | qwen |  | whole_blank_clause_a | 3748 | 0 |  |
| e1p_e2a | qwen |  | whole_blank_clause_b | 3748 | 0 |  |
| e2_a2e | qwen | 2973 | negated_other_trigger | 5946 | 0 | 0 |
| e2_a2e | qwen |  | whole_blank_clause_a | 5946 | 0 |  |
| e2_a2e | qwen |  | whole_blank_clause_b | 5946 | 0 |  |
| e2_e2a | qwen | 2969 | negated_other_trigger | 5938 | 0 | 0 |
| e2_e2a | qwen |  | whole_blank_clause_a | 5938 | 0 |  |
| e2_e2a | qwen |  | whole_blank_clause_b | 5938 | 0 |  |
| sc_agree_a2e | qwen | 3051 | reconstructs_prompt_a | 6102 | 0 | 0 |
| sc_agree_a2e | qwen |  | reconstructs_prompt_b | 6102 | 0 |  |
| sc_agree_a2e | qwen |  | whole_sentence_a | 6102 | 0 |  |
| sc_agree_a2e | qwen |  | whole_sentence_b | 6102 | 0 |  |
| sc_conflict_a2e | qwen | 3051 | reconstructs_prompt_a | 6102 | 0 | 0 |
| sc_conflict_a2e | qwen |  | reconstructs_prompt_b | 6102 | 0 |  |
| sc_conflict_a2e | qwen |  | whole_sentence_a | 6102 | 0 |  |
| sc_conflict_a2e | qwen |  | whole_sentence_b | 6102 | 0 |  |
| rsite_e1_a2e | qwen | 3030 | evidence_mentions_swapped | 6060 | 0 | 0 |
| rsite_e1_a2e | qwen |  | question_identical | 6060 | 0 |  |
| rsite_e1_a2e | qwen |  | whole_blank_clause_a | 6060 | 0 |  |
| rsite_e1_a2e | qwen |  | whole_blank_clause_b | 6060 | 0 |  |
| e3p3_a2e | qwen | 2923 | blank_np_is_det_only | 2964 | 0 | 41 |
| e3p3_a2e | qwen |  | both_entities_in_event_clause | 2964 | 41 |  |
| e3p3_a2e | qwen |  | only_connective_dropped | 2964 | 0 |  |
| e3p3_a2e | qwen |  | reconstructs_sentence_a | 2964 | 0 |  |
| e3p3_a2e | qwen |  | reconstructs_sentence_b | 2964 | 0 |  |
| e3p3_a2e | qwen |  | whole_blank_clause_a | 2964 | 0 |  |
| e3p3_a2e | qwen |  | whole_blank_clause_b | 2964 | 0 |  |
| e1_a2e | mistral | 1586 | both_filled_clauses_in_evidence | 3172 | 0 | 0 |
| e1_a2e | mistral |  | whole_blank_clause_a | 3172 | 0 |  |
| e1_a2e | mistral |  | whole_blank_clause_b | 3172 | 0 |  |
| e1_e2a | mistral | 1572 | both_filled_clauses_in_evidence | 3144 | 0 | 0 |
| e1_e2a | mistral |  | whole_blank_clause_a | 3144 | 0 |  |
| e1_e2a | mistral |  | whole_blank_clause_b | 3144 | 0 |  |
| e3_a2e | mistral | 1642 | whole_sentence_a | 1642 | 0 | 0 |
| e3_a2e | mistral |  | whole_sentence_b | 1642 | 0 |  |
| e3_e2a | mistral | 1636 | prompt_plus_trigger_is_sentence_a | 1636 | 0 | 0 |
| e3_e2a | mistral |  | prompt_plus_trigger_is_sentence_b | 1636 | 0 |  |
| e1p_a2e | mistral | 1110 | no_verbatim_trigger_in_evidence | 2220 | 0 | 0 |
| e1p_a2e | mistral |  | paraphrases_in_evidence | 2220 | 0 |  |
| e1p_a2e | mistral |  | whole_blank_clause_a | 2220 | 0 |  |
| e1p_a2e | mistral |  | whole_blank_clause_b | 2220 | 0 |  |
| e1p_e2a | mistral | 1108 | no_verbatim_trigger_in_evidence | 2216 | 0 | 0 |
| e1p_e2a | mistral |  | paraphrases_in_evidence | 2216 | 0 |  |
| e1p_e2a | mistral |  | whole_blank_clause_a | 2216 | 0 |  |
| e1p_e2a | mistral |  | whole_blank_clause_b | 2216 | 0 |  |
| e2_a2e | mistral | 1586 | negated_other_trigger | 3172 | 0 | 0 |
| e2_a2e | mistral |  | whole_blank_clause_a | 3172 | 0 |  |
| e2_a2e | mistral |  | whole_blank_clause_b | 3172 | 0 |  |
| e2_e2a | mistral | 1572 | negated_other_trigger | 3144 | 0 | 0 |
| e2_e2a | mistral |  | whole_blank_clause_a | 3144 | 0 |  |
| e2_e2a | mistral |  | whole_blank_clause_b | 3144 | 0 |  |
| sc_agree_a2e | mistral | 1644 | reconstructs_prompt_a | 3288 | 0 | 0 |
| sc_agree_a2e | mistral |  | reconstructs_prompt_b | 3288 | 0 |  |
| sc_agree_a2e | mistral |  | whole_sentence_a | 3288 | 0 |  |
| sc_agree_a2e | mistral |  | whole_sentence_b | 3288 | 0 |  |
| sc_conflict_a2e | mistral | 1644 | reconstructs_prompt_a | 3288 | 0 | 0 |
| sc_conflict_a2e | mistral |  | reconstructs_prompt_b | 3288 | 0 |  |
| sc_conflict_a2e | mistral |  | whole_sentence_a | 3288 | 0 |  |
| sc_conflict_a2e | mistral |  | whole_sentence_b | 3288 | 0 |  |
| rsite_e1_a2e | mistral | 2360 | evidence_mentions_swapped | 4720 | 0 | 0 |
| rsite_e1_a2e | mistral |  | question_identical | 4720 | 0 |  |
| rsite_e1_a2e | mistral |  | whole_blank_clause_a | 4720 | 0 |  |
| rsite_e1_a2e | mistral |  | whole_blank_clause_b | 4720 | 0 |  |
| e3p3_a2e | mistral | 1572 | blank_np_is_det_only | 1600 | 0 | 28 |
| e3p3_a2e | mistral |  | both_entities_in_event_clause | 1600 | 28 |  |
| e3p3_a2e | mistral |  | only_connective_dropped | 1600 | 0 |  |
| e3p3_a2e | mistral |  | reconstructs_sentence_a | 1600 | 0 |  |
| e3p3_a2e | mistral |  | reconstructs_sentence_b | 1600 | 0 |  |
| e3p3_a2e | mistral |  | whole_blank_clause_a | 1600 | 0 |  |
| e3p3_a2e | mistral |  | whole_blank_clause_b | 1600 | 0 |  |

## e3p3_a2e (E3 content in the P3 frame): event clause audit

Prompt `{event clause}. Fill in the blank: The _ was X.\nAnswer: the`. Event clause = the WinoGrande sentence before the blank clause with the connective word(s) introducing it dropped (moetrace/ext14_data.e3p3_parts), final punctuation normalised to '.'. Checks: both entities in the event clause; the whole blank clause in the question (capitalised); the event clause equals the raw text before the blank clause minus exactly the dropped connective words (punctuation ignored). Excluded: blank inside a PP ('the sauce on the _ was', all cells: D5_blank_in_pp), blank clause = object of a verb of saying / thinking ('I thought the _ was', 'realized that the _ was': E3P3_complement_clause), no event clause. Blank clauses are sentence-final by construction (Phase-3 geometry: the trigger is the last word).

**qwen**: 2923 items (core 624); excluded: E3P3_complement_clause 105, E3P3_text_before_blank_np 53, check_fail:both_entities_in_event_clause 41, item:R4_duplicate_prompt 10, item:incomplete_dropped 41

| Dropped connective | Items |
|---|---|
| because | 1422 |
| (sentence break) | 476 |
| but | 343 |
| (comma only) | 177 |
| until | 159 |
| since | 121 |
| as | 85 |
| so | 49 |
| and | 20 |
| instead because | 10 |
| even though | 7 |
| however | 7 |
| therefore | 7 |
| why | 3 |
| so because | 3 |
| when | 3 |
| instead since | 3 |
| instead as | 3 |
| now | 3 |
| before because | 2 |
| although | 2 |
| though | 2 |
| yet because | 2 |
| and now | 1 |
| and luckily | 1 |
| after because | 1 |
| so now | 1 |
| so then | 1 |
| yet | 1 |
| now because | 1 |
| for | 1 |
| where | 1 |
| afterwards | 1 |
| and then | 1 |
| because because | 1 |
| still because | 1 |
| for because | 1 |

**mistral**: 1572 items (core 606); excluded: E3P3_complement_clause 105, E3P3_text_before_blank_np 53, check_fail:both_entities_in_event_clause 28, item:R4_duplicate_prompt 7, item:incomplete_dropped 28

| Dropped connective | Items |
|---|---|
| because | 740 |
| but | 247 |
| (sentence break) | 240 |
| until | 125 |
| (comma only) | 94 |
| since | 43 |
| as | 25 |
| so | 17 |
| and | 13 |
| instead because | 4 |
| however | 3 |
| even though | 3 |
| now | 3 |
| before because | 2 |
| so because | 2 |
| though | 2 |
| and luckily | 1 |
| after because | 1 |
| therefore | 1 |
| yet | 1 |
| now because | 1 |
| for | 1 |
| and then | 1 |
| yet because | 1 |
| for because | 1 |

