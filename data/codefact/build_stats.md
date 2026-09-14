# CodeFact build statistics (2026-09-14 05:31 UTC)

Command: `python scripts/ext4_build_codefact.py --csn-sample 6000 --cap-per-unit 2 --max-per-category 1200 --max-prefix-tokens 160 --csn-extra-sample 16000 --extra-categories R3`

## Sources and licences

| Source | Units used | Licence | Notes |
|---|---|---|---|
| HumanEval (openai/openai_humaneval) | 164 | MIT | prompt + canonical_solution, docstrings removed |
| MBPP full (google-research-datasets/mbpp; train/test/validation/prompt) | 974 | CC-BY-4.0 | `code` field; CRLF normalised |
| CodeSearchNet Python, test split (code-search-net/code_search_net) | 5956 | per repository; the corpus was restricted at collection time to repositories whose licence permits redistribution (Husain et al. 2019); per-function repo + URL in items.jsonl and csn_sample_repos.csv | seed-0 sample of 6000 functions <= 1500 chars, ASCII, docstrings removed |
| CodeSearchNet Python, validation split (extra units for R3 only) | 15901 | as above | seed-7 sample of 16000 functions, cap 3 per unit |

The Stack was requested as the volume source but is gated on the Hub (needs an authenticated account that accepted the terms); no token is available on this machine, so CodeSearchNet (the documented fallback) was used.

## Yields per category

| Category | Raw candidates | After per-unit cap | Written (single-token in >= 1 tokenizer) | Qwen3 ok | Mixtral ok | both ok | Qwen3 rejects | Mixtral rejects |
|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 46223 | 13801 | 1200 | 964 | 1127 | 891 | {'subject_is_final': 243, 'too_long': 44, 'multi_token': 31} | {'too_long': 108, 'multi_token': 39, 'subject_is_final': 6, 'copy': 2} |
| S2 block keyword | 3235 | 2775 | 1200 | 1200 | 1070 | 1070 | {'too_long': 118} | {'too_long': 247, 'multi_token': 1} |
| S3 keyword completion | 6921 | 5437 | 1200 | 1195 | 1124 | 1119 | {'too_long': 56, 'multi_token': 5} | {'too_long': 132} |
| R1 variable recall | 27866 | 10352 | 1200 | 1132 | 955 | 887 | {'multi_token': 1862, 'too_long': 30, 'copy': 20} | {'too_long': 71, 'multi_token': 1973, 'copy': 45} |
| R2 attribute / API recall | 1133 | 933 | 795 | 794 | 671 | 670 | {'too_long': 88, 'multi_token': 51} | {'too_long': 183, 'multi_token': 76, 'subject_is_final': 3} |
| R3 constant recall | 5429 | 3756 | 1200 | 1163 | 824 | 787 | {'multi_token': 1972, 'too_long': 318, 'copy': 31} | {'too_long': 472, 'multi_token': 2155, 'copy': 33} |

Rejects: `multi_token` = true or foil is not a single token as a continuation of the prefix (after up to 4 characters of boundary back-off), `copy` = the true token occurs in the last 3 prefix tokens, `subject_is_final` = the subject token is the last prefix token, `too_long` = prefix > max tokens, `same_token` = true and foil map to the same token.

## Sub-categories (written items)

| Category | Sub-category | n | Qwen3 ok | Mixtral ok |
|---|---|---|---|---|
| R1 | import | 4 | 4 | 2 |
| R1 | param | 539 | 505 | 444 |
| R1 | store | 657 | 623 | 509 |
| R2 | local_dict | 94 | 94 | 69 |
| R2 | local_list | 503 | 503 | 433 |
| R2 | local_set | 49 | 49 | 40 |
| R2 | local_str | 30 | 30 | 26 |
| R2 | module_cmath | 3 | 3 | 3 |
| R2 | module_datetime | 1 | 1 | 1 |
| R2 | module_itertools | 1 | 1 | 0 |
| R2 | module_math | 41 | 40 | 41 |
| R2 | module_re | 73 | 73 | 58 |
| R3 | int | 308 | 302 | 243 |
| R3 | str | 892 | 861 | 581 |
| S1 | ) | 997 | 798 | 942 |
| S1 | ] | 185 | 149 | 169 |
| S1 | } | 18 | 17 | 16 |
| S2 | elif_after_if | 228 | 228 | 199 |
| S2 | else_after_for | 3 | 3 | 2 |
| S2 | else_after_if | 631 | 631 | 567 |
| S2 | else_after_try | 20 | 20 | 18 |
| S2 | except_after_try | 291 | 291 | 260 |
| S2 | finally_after_try | 27 | 27 | 24 |
| S3 | elif_colon | 35 | 35 | 33 |
| S3 | for_in | 459 | 459 | 433 |
| S3 | if_colon | 690 | 685 | 644 |
| S3 | while_colon | 16 | 16 | 14 |

## Prefix length (Qwen3 tokens, items ok for Qwen3)

| Category | n | min | 25% | median | 75% | max | mean subject tokens |
|---|---|---|---|---|---|---|---|
| S1 | 964 | 5 | 18 | 36 | 64 | 159 | 1.00 |
| S2 | 1200 | 14 | 39 | 62 | 94 | 160 | 1.00 |
| S3 | 1195 | 7 | 22 | 40 | 72 | 158 | 1.00 |
| R1 | 1132 | 9 | 25 | 40 | 67 | 160 | 1.02 |
| R2 | 794 | 10 | 35 | 61 | 95 | 160 | 3.30 |
| R3 | 1163 | 18 | 59 | 84 | 116 | 160 | 1.00 |

## Boundary back-off (Qwen3 / Mixtral, items ok): how many characters the prefix/answer boundary moved back

| Category | Qwen3 backoff 0 / 1 / 2+ | Mixtral backoff 0 / 1 / 2+ |
|---|---|---|
| S1 | 699 / 227 / 38 | 902 / 209 / 16 |
| S2 | 0 / 1200 / 0 | 18 / 1052 / 0 |
| S3 | 1195 / 0 / 0 | 1124 / 0 / 0 |
| R1 | 134 / 998 / 0 | 399 / 556 / 0 |
| R2 | 0 / 794 / 0 | 671 / 0 / 0 |
| R3 | 1076 / 0 / 87 | 806 / 1 / 17 |

## Source mix of written items

| Category | humaneval | mbpp | csn |
|---|---|---|---|
| S1 | 23 | 161 | 1016 |
| S2 | 23 | 102 | 1075 |
| S3 | 45 | 155 | 1000 |
| R1 | 27 | 113 | 1060 |
| R2 | 37 | 203 | 555 |
| R3 | 17 | 32 | 1151 |
