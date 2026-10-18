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
