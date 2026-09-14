## Direction 4: expert-aware tracing on code (CodeFact)

**Question.** Does the paper's expert-aware causal tracing transfer from factual recall to code, where the 'fact' is syntactic (matching bracket, block keyword) or semantic (variable, API, constant recall)? Do the categories localise to the same layers and experts, and do the factual-recall experts (Qwen3 L44E069 / L42E115, Mixtral L19E006 / L19E002 / L18E001) play any role?

**Dataset (Phase A, `data/codefact/`).** CounterFact-style next-token counterfactuals built from Python with `ast`/`tokenize` (`scripts/ext4_build_codefact.py`; items in `items.jsonl`, yields and licences in `build_stats.md`, 20 eyeballed items per category in `samples.md`). Sources: HumanEval canonical solutions (MIT), MBPP (CC-BY-4.0) and a seed-0 sample of CodeSearchNet Python functions (The Stack is gated on the Hub and no token is available; CodeSearchNet was collected from repositories whose licences permit redistribution, per-function repo and URL are recorded). Docstrings are removed, CRLF normalised. Each item = (prefix, true next token, foil next token of the same category, subject span). Categories: **S1** closing bracket (`)` `]` `}` vs another closer; subject = the matching opener), **S2** block keyword (`else`/`elif`/`except`/`finally` vs a sibling keyword; subject = the `if`/`for`/`while`/`try` head), **S3** keyword completion (` in` after `for x` vs `,`; `:` after an `if`/`elif`/`while` header vs ` and`; subject = the `for`/`if` token), **R1** variable recall (a name bound earlier, foil = the most recently bound other name; subject = the definition site; the name occurred at most twice before), **R2** attribute/API recall (`.append` after `x = []`, `re.search` after `import re`; foil = another attribute of the same type/module; subject = the literal / module name), **R3** constant recall (a string or single-digit literal that occurred earlier, foil = another earlier literal; subject = the first occurrence). True and foil must each be ONE token as a continuation of the prefix in the model tokenizer (`moetrace/ext4_data.py`: the boundary backs off over up to 4 punctuation/whitespace characters when the tokenizer merges, e.g. Qwen's `.append`, ` else`); items whose true token occurs in the last 3 prefix tokens are excluded (copying). Prefixes are capped at 160 tokens. Noise, thresholds, split (128/128, seed 0), recurrence (64/128), active-random controls (3 Qwen3, 1 Mixtral), bootstrap CIs: exactly as in the paper protocol; the only change to the pipeline is the case loader.

**Yields per category (candidates -> single-token items per tokenizer):**
| Category | Raw candidates | After per-unit cap | Written (single-token in >= 1 tokenizer) | Qwen3 ok | Mixtral ok | both ok | Qwen3 rejects | Mixtral rejects |
|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 46223 | 13801 | 1200 | 964 | 1127 | 891 | {'subject_is_final': 243, 'too_long': 44, 'multi_token': 31} | {'too_long': 108, 'multi_token': 39, 'subject_is_final': 6, 'copy': 2} |
| S2 block keyword | 3235 | 2775 | 1200 | 1200 | 1070 | 1070 | {'too_long': 118} | {'too_long': 247, 'multi_token': 1} |
| S3 keyword completion | 6921 | 5437 | 1200 | 1195 | 1124 | 1119 | {'too_long': 56, 'multi_token': 5} | {'too_long': 132} |
| R1 variable recall | 27866 | 10352 | 1200 | 1132 | 955 | 887 | {'multi_token': 1862, 'too_long': 30, 'copy': 20} | {'too_long': 71, 'multi_token': 1973, 'copy': 45} |
| R2 attribute / API recall | 1133 | 933 | 795 | 794 | 671 | 670 | {'too_long': 88, 'multi_token': 51} | {'too_long': 183, 'multi_token': 76, 'subject_is_final': 3} |
| R3 constant recall | 5429 | 3756 | 1200 | 1163 | 824 | 787 | {'multi_token': 1972, 'too_long': 318, 'copy': 31} | {'too_long': 472, 'multi_token': 2155, 'copy': 33} |

### Threshold calibration (Qwen3-30B-A3B-Base, all scanned items)

One GPU scan per model/protocol runs, for every item, the clean and subject-noised prefill rows AND the MoE-block patch at every layer (`scripts/ext4_scan.py`, chunks of up to 1,024 items sorted by length), so the filter and the layer sweep are the same pass. The paper's absolute filter (Δ_clean ≥ 1.0, drop ≥ 0.5) is primary; per-category pass rates are results in their own right. The relative rule (drop ≥ 25 % of Δ_clean, Δ_clean ≥ 1) is reported in the appendix table only.

**qwen3 (raw): per-category calibration of the paper's filter on CodeFact (clean vs subject-noised Δ = logit(true) − logit(foil))**

| Category | n scanned | median Δ_clean | median drop | paper filter (Δ≥1, drop≥0.5) | relaxed (Δ≥0.5, drop≥0.25) | relative (drop ≥ 25 % Δ_clean, Δ≥1) | relative 50 % | top-1 = true / starts with true (clean) | top-1 = true / starts with true (noised) | Δ_clean ≥ 1 | drop ≥ 0.5 | final token carries max norm (L5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 964 | +13.56 | +4.38 | 871 (90 %) | 910 (94 %) | 593 (62 %) | 290 (30 %) | 18 % / 87 % | 16 % / 69 % | 100 % | 90 % | 0.0 % |
| S2 block keyword | 1200 | +3.25 | +0.25 | 399 (33 %) | 555 (46 %) | 179 (15 %) | 36 (3 %) | 81 % / 81 % | 80 % / 80 % | 78 % | 37 % | 0.0 % |
| S3 keyword completion | 1195 | +3.75 | +0.12 | 430 (36 %) | 543 (45 %) | 244 (20 %) | 71 (6 %) | 38 % / 96 % | 36 % / 93 % | 88 % | 37 % | 0.0 % |
| R1 variable recall | 1132 | +8.12 | +1.88 | 896 (79 %) | 971 (86 %) | 562 (50 %) | 278 (25 %) | 89 % / 89 % | 62 % / 63 % | 96 % | 82 % | 0.0 % |
| R2 attribute / API recall | 794 | +7.38 | +0.75 | 451 (57 %) | 528 (66 %) | 189 (24 %) | 62 (8 %) | 90 % / 90 % | 84 % / 84 % | 94 % | 58 % | 0.0 % |
| R3 constant recall | 1163 | +7.62 | +0.88 | 690 (59 %) | 797 (69 %) | 338 (29 %) | 159 (14 %) | 85 % / 89 % | 78 % / 82 % | 94 % | 61 % | 0.0 % |
| ALL | 6448 | +7.00 | +0.88 | 3737 (58 %) | 4304 (67 %) | 2105 (33 %) | 896 (14 %) | 67 % / 89 % | 59 % / 78 % | 91 % | 60 % | 0.0 % |

**mixtral (nobos): per-category calibration of the paper's filter on CodeFact (clean vs subject-noised Δ = logit(true) − logit(foil))**

| Category | n scanned | median Δ_clean | median drop | paper filter (Δ≥1, drop≥0.5) | relaxed (Δ≥0.5, drop≥0.25) | relative (drop ≥ 25 % Δ_clean, Δ≥1) | relative 50 % | top-1 = true / starts with true (clean) | top-1 = true / starts with true (noised) | Δ_clean ≥ 1 | drop ≥ 0.5 | final token carries max norm (L5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 800 | +13.19 | +2.91 | 690 (86 %) | 730 (91 %) | 368 (46 %) | 92 (12 %) | 52 % / 87 % | 49 % / 78 % | 100 % | 86 % | 0.0 % |
| S2 block keyword | 800 | +2.75 | +0.25 | 316 (40 %) | 402 (50 %) | 215 (27 %) | 70 (9 %) | 79 % / 79 % | 77 % / 77 % | 76 % | 44 % | 0.0 % |
| S3 keyword completion | 800 | +4.62 | +0.00 | 241 (30 %) | 325 (41 %) | 95 (12 %) | 30 (4 %) | 95 % / 95 % | 92 % / 92 % | 97 % | 31 % | 0.0 % |
| R1 variable recall | 800 | +7.12 | +0.44 | 379 (47 %) | 467 (58 %) | 171 (21 %) | 82 (10 %) | 88 % / 88 % | 76 % / 76 % | 94 % | 49 % | 0.0 % |
| R2 attribute / API recall | 671 | +5.88 | +0.50 | 321 (48 %) | 410 (61 %) | 88 (13 %) | 9 (1 %) | 90 % / 90 % | 89 % / 89 % | 89 % | 50 % | 0.0 % |
| R3 constant recall | 800 | +6.75 | +0.25 | 318 (40 %) | 404 (50 %) | 120 (15 %) | 42 (5 %) | 86 % / 87 % | 81 % / 82 % | 93 % | 41 % | 0.0 % |
| ALL | 4671 | +6.38 | +0.50 | 2265 (48 %) | 2738 (59 %) | 1057 (23 %) | 325 (7 %) | 82 % / 88 % | 77 % / 82 % | 92 % | 50 % | 0.0 % |

**qwen3_coder (raw): per-category calibration of the paper's filter on CodeFact (clean vs subject-noised Δ = logit(true) − logit(foil))**

| Category | n scanned | median Δ_clean | median drop | paper filter (Δ≥1, drop≥0.5) | relaxed (Δ≥0.5, drop≥0.25) | relative (drop ≥ 25 % Δ_clean, Δ≥1) | relative 50 % | top-1 = true / starts with true (clean) | top-1 = true / starts with true (noised) | Δ_clean ≥ 1 | drop ≥ 0.5 | final token carries max norm (L5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 964 | +13.88 | +3.03 | 810 (84 %) | 852 (88 %) | 442 (46 %) | 167 (17 %) | 16 % / 89 % | 15 % / 77 % | 100 % | 84 % | 0.0 % |
| S2 block keyword | 1200 | +5.38 | +0.12 | 395 (33 %) | 517 (43 %) | 125 (10 %) | 27 (2 %) | 80 % / 80 % | 79 % / 79 % | 82 % | 38 % | 0.0 % |
| S3 keyword completion | 1195 | +4.50 | +0.12 | 474 (40 %) | 538 (45 %) | 262 (22 %) | 68 (6 %) | 37 % / 96 % | 37 % / 94 % | 79 % | 42 % | 0.0 % |
| R1 variable recall | 1132 | +12.75 | +2.62 | 899 (79 %) | 945 (83 %) | 504 (45 %) | 257 (23 %) | 88 % / 88 % | 63 % / 63 % | 97 % | 81 % | 0.0 % |
| R2 attribute / API recall | 794 | +11.25 | +1.62 | 576 (73 %) | 617 (78 %) | 232 (29 %) | 58 (7 %) | 89 % / 89 % | 83 % / 83 % | 95 % | 74 % | 0.0 % |
| R3 constant recall | 1163 | +12.25 | +0.94 | 696 (60 %) | 761 (65 %) | 291 (25 %) | 121 (10 %) | 87 % / 90 % | 77 % / 81 % | 96 % | 61 % | 0.0 % |
| ALL | 6448 | +10.84 | +1.00 | 3850 (60 %) | 4230 (66 %) | 1856 (29 %) | 698 (11 %) | 66 % / 89 % | 59 % / 79 % | 91 % | 62 % | 0.0 % |

**qwen3_coder (chat): per-category calibration of the paper's filter on CodeFact (clean vs subject-noised Δ = logit(true) − logit(foil))**

| Category | n scanned | median Δ_clean | median drop | paper filter (Δ≥1, drop≥0.5) | relaxed (Δ≥0.5, drop≥0.25) | relative (drop ≥ 25 % Δ_clean, Δ≥1) | relative 50 % | top-1 = true / starts with true (clean) | top-1 = true / starts with true (noised) | Δ_clean ≥ 1 | drop ≥ 0.5 | final token carries max norm (L5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 964 | +16.19 | +4.11 | 876 (91 %) | 901 (93 %) | 501 (52 %) | 160 (17 %) | 9 % / 89 % | 8 % / 80 % | 100 % | 91 % | 0.0 % |
| S2 block keyword | 1200 | +6.25 | +0.25 | 484 (40 %) | 578 (48 %) | 207 (17 %) | 36 (3 %) | 80 % / 80 % | 78 % / 78 % | 82 % | 45 % | 0.0 % |
| S3 keyword completion | 1195 | +3.12 | -0.38 | 409 (34 %) | 449 (38 %) | 231 (19 %) | 51 (4 %) | 38 % / 96 % | 37 % / 95 % | 61 % | 36 % | 0.0 % |
| R1 variable recall | 1132 | +15.50 | +3.09 | 899 (79 %) | 932 (82 %) | 499 (44 %) | 254 (22 %) | 87 % / 87 % | 62 % / 62 % | 97 % | 81 % | 0.0 % |
| R2 attribute / API recall | 794 | +14.00 | +2.00 | 606 (76 %) | 640 (81 %) | 270 (34 %) | 92 (12 %) | 89 % / 89 % | 82 % / 82 % | 95 % | 78 % | 0.0 % |
| R3 constant recall | 1163 | +15.75 | +1.50 | 756 (65 %) | 818 (70 %) | 331 (28 %) | 134 (12 %) | 86 % / 90 % | 77 % / 80 % | 96 % | 66 % | 0.0 % |
| ALL | 6448 | +13.14 | +1.50 | 4030 (62 %) | 4318 (67 %) | 2039 (32 %) | 727 (11 %) | 65 % / 88 % | 57 % / 80 % | 88 % | 65 % | 0.0 % |

![CodeFact calibration](figures/ext4_calibration.png)

**Reading the calibration.** Syntax items have large clean margins (the model is nearly always right, top-1 = true in the great majority) but the subject noise often does not move them: the answer is redundantly determined by the rest of the context (a `)` after `foo(bar` is predicted from `foo` being a call even when the `(` embedding is destroyed; `else` is predicted from the dedent and the block content). A low pass rate under the paper's filter is therefore the expected signature of a *non-recall* category, not a construction error (the 20 eyeballed items per category in `data/codefact/samples.md` have the right subject). Recall categories carry their information in one place (the definition site) and are noise-sensitive like CounterFact.

### Qwen3-30B-A3B-Base (tokenizer defaults)

Scanned 6448 items; passing the paper filter per category: S1 871, S2 399, S3 430, R1 896, R2 451, R3 690. Sets with fewer than 256 passing items use all passing items (marked partial; recurrence threshold = half the discovery split).

**Qwen3-30B-A3B-Base (tokenizer defaults): per-category localisation on CodeFact (validation split; recurrence threshold 64 of 128)**

| Set | n (disc+val) | top-1 = true | mean Δ_clean / drop | L* | block rescue (val) | 2nd layer | e* (two-stage) | active disc / val | expert rescue (val) | Spec (active-random) | rank-1 among active | coalition clean / union | joint winner (all layers) | recurrent pairs (all layers) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 128+128 | 19 % | +13.48 / +5.65 | L47 | +2.420 [+2.069, +2.803] | L46 +1.271 | L47E025 | 119/128 / 124/128 | +1.153 [+0.951, +1.368] | +0.973 [+0.784, +1.175] | 65/124 | +2.036 / +2.418 | L47E025 (= two-stage) | 201 |
| S2 block keyword | 128+128 | 95 % | +6.47 / +1.41 | L47 | +0.642 [+0.505, +0.780] | L41 +0.488 | L47E062 | 111/128 / 110/128 | +0.336 [+0.237, +0.438] | +0.343 [+0.237, +0.449] | 70/110 | +0.109 / +0.645 | L41E041 +0.551 [+0.452, +0.652], Spec +0.549 | 269 |
| S3 keyword completion | 128+128 | 68 % | +7.15 / +2.29 | L47 | +1.054 [+0.836, +1.277] | L42 +0.441 | L47E025 | 101/128 / 104/128 | +0.188 [+0.143, +0.233] | +0.154 [+0.098, +0.209] | 39/104 | +0.242 / +1.050 | L47E025 (= two-stage) | 131 |
| R1 variable recall | 128+128 | 94 % | +8.75 / +3.44 | L43 | +0.391 [+0.299, +0.490] | L42 +0.188 | L43E126 | 121/128 / 118/128 | +0.326 [+0.252, +0.408] | +0.316 [+0.242, +0.400] | 87/118 | +0.370 / +0.400 | L43E126 (= two-stage) | 206 |
| R2 attribute / API recall | 128+128 | 92 % | +9.22 / +2.42 | L47 | +0.935 [+0.679, +1.223] | L43 +0.353 | L47E005 | 124/128 / 125/128 | +0.118 [+0.040, +0.202] | +0.042 [-0.028, +0.114] | 37/125 | +0.594 / +0.922 | L47E005 (= two-stage) | 205 |
| R3 constant recall | 128+128 | 92 % | +9.15 / +2.68 | L47 | +0.828 [+0.600, +1.056] | L42 +0.238 | L47E101 | 98/128 / 92/128 | +0.117 [+0.077, +0.159] | +0.079 [+0.032, +0.128] | 37/92 | +0.513 / +0.863 | L47E101 (= two-stage) | 163 |
| all mixed | 128+128 | 73 % | +9.13 / +2.89 | L47 | +0.962 [+0.728, +1.215] | L42 +0.409 | none (max activity 63/128 < 64) |  |  |  |  | +0.590 / +0.988 | L42E048 +0.256 [+0.174, +0.349], Spec +0.258 | 104 |

![layer curves qwen3](figures/ext4_curves_qwen3.png)

**Qwen3-30B-A3B-Base (tokenizer defaults): the same selection restricted to interior layers (excluding the last 4 MoE blocks, whose patch acts as a read-out on code)**

| Set | last-layer block rescue (val) | L* interior (≤ L−5) | block rescue (val) | e* interior | active disc / val | expert rescue (val) | Spec (active-random) | rank-1 among active | coalition clean / union | joint winner (interior layers) |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | L47: +2.420 [+2.069, +2.803] | L42 | +1.261 [+1.044, +1.506] | L42E048 | 125/128 / 125/128 | +0.916 [+0.737, +1.117] | +0.873 [+0.700, +1.068] | 99/125 | +1.267 / +1.265 | L43E084 +0.940 [+0.723, +1.189], Spec +0.930 [+0.710, +1.174] (rank 2 over all layers) |
| S2 | L47: +0.642 [+0.505, +0.780] | L41 | +0.488 [+0.400, +0.575] | L41E041 | 128/128 / 128/128 | +0.551 [+0.452, +0.652] | +0.549 [+0.451, +0.649] | 106/128 | +0.542 / +0.511 | L41E041 +0.551 [+0.452, +0.652], Spec +0.549 [+0.451, +0.649] (rank 1 over all layers) |
| S3 | L47: +1.054 [+0.836, +1.277] | L42 | +0.441 [+0.325, +0.566] | L42E044 | 96/128 / 97/128 | +0.125 [+0.078, +0.176] | +0.090 [+0.042, +0.139] | 36/97 | +0.357 / +0.422 | L42E044 +0.125 [+0.078, +0.176], Spec +0.090 [+0.042, +0.139] (rank 2 over all layers) |
| R1 | L47: +0.160 [-0.138, +0.458] | L43 | +0.391 [+0.299, +0.490] | L43E126 | 121/128 / 118/128 | +0.326 [+0.252, +0.408] | +0.316 [+0.242, +0.400] | 87/118 | +0.370 / +0.400 | L43E126 +0.326 [+0.252, +0.408], Spec +0.316 [+0.242, +0.400] (rank 1 over all layers) |
| R2 | L47: +0.935 [+0.679, +1.223] | L43 | +0.353 [+0.252, +0.453] | L43E051 | 124/128 / 126/128 | +0.230 [+0.163, +0.301] | +0.234 [+0.166, +0.306] | 74/126 | +0.333 / +0.330 | L43E051 +0.230 [+0.163, +0.301], Spec +0.234 [+0.166, +0.306] (rank 2 over all layers) |
| R3 | L47: +0.828 [+0.600, +1.056] | L43 | +0.216 [+0.129, +0.305] | L43E126 | 64/128 / 74/128 | +0.070 [+0.036, +0.106] | +0.055 [+0.020, +0.092] | 40/74 | +0.168 / +0.195 | L6E113 -0.015 [-0.032, +0.002], Spec -0.015 [-0.033, +0.003] (rank 4 over all layers) |
| all | L47: +0.962 [+0.728, +1.215] | L41 | +0.381 [+0.264, +0.505] | L41E023 | 121/128 / 125/128 | -0.000 [-0.033, +0.035] | -0.039 [-0.079, +0.002] | 34/125 | +0.373 / +0.363 | L42E048 +0.256 [+0.174, +0.349], Spec +0.258 [+0.177, +0.348] (rank 1 over all layers) |

**Qwen3-30B-A3B-Base (tokenizer defaults): the factual-recall experts (L44E069 (paper), L42E115 (ext1 second locus)) on the code categories**

| Set | factual expert | disc active | recurrent (≥ threshold) | joint rank | val active | val rescue | val Spec |
|---|---|---|---|---|---|---|---|
| S1 | L44E069 | 0/128 | no | - | 2/128 | +0.001 [+0.000, +0.003] | -0.059 [-0.087, -0.034] |
| S1 | L42E115 | 1/128 | no | - | 1/128 | +0.000 [+0.000, +0.000] | -0.174 [-0.224, -0.128] |
| S2 | L44E069 | 4/128 | no | - | 1/128 | +0.001 [+0.000, +0.003] | -0.006 [-0.024, +0.011] |
| S2 | L42E115 | 17/128 | no | - | 14/128 | -0.003 [-0.011, +0.005] | -0.015 [-0.032, +0.003] |
| S3 | L44E069 | 2/128 | no | - | 0/128 | 0 (never active) | n/a |
| S3 | L42E115 | 1/128 | no | - | 1/128 | +0.000 [+0.000, +0.001] | -0.039 [-0.067, -0.013] |
| R1 | L44E069 | 1/128 | no | - | 0/128 | 0 (never active) | n/a |
| R1 | L42E115 | 4/128 | no | - | 2/128 | -0.001 [-0.003, +0.000] | -0.036 [-0.056, -0.017] |
| R2 | L44E069 | 1/128 | no | - | 2/128 | +0.000 [+0.000, +0.000] | +0.007 [-0.010, +0.025] |
| R2 | L42E115 | 0/128 | no | - | 0/128 | 0 (never active) | n/a |
| R3 | L44E069 | 1/128 | no | - | 2/128 | +0.000 [-0.001, +0.003] | +0.009 [-0.013, +0.030] |
| R3 | L42E115 | 3/128 | no | - | 5/128 | +0.008 [+0.000, +0.023] | -0.026 [-0.053, +0.002] |
| all | L44E069 | 2/128 | no | - | 1/128 | +0.000 [+0.000, +0.000] | -0.013 [-0.032, +0.006] |
| all | L42E115 | 4/128 | no | - | 5/128 | +0.002 [+0.000, +0.006] | -0.028 [-0.060, +0.002] |

**Qwen3-30B-A3B-Base (tokenizer defaults): selected layers, experts and recurrent sets per category**

| Set | L* (paper rule) | two-stage e* | joint winner (all layers) | L* interior | two-stage e* interior | joint winner (interior) | recurrent pairs (all layers) | recurrent pairs (interior) | recurrent experts at L* |
|---|---|---|---|---|---|---|---|---|---|
| S1 | L47 | L47E025 | L47E025 | L42 | L42E048 | L43E084 | 201 | 180 | E005, E016, E025, E050, E098, E116, E122 |
| S2 | L47 | L47E062 | L41E041 | L41 | L41E041 | L41E041 | 269 | 249 | E033, E060, E062, E077, E097, E122 |
| S3 | L47 | L47E025 | L47E025 | L42 | L42E044 | L42E044 | 131 | 120 | E005, E016, E025, E050, E116 |
| R1 | L43 | L43E126 | L43E126 | L43 | L43E126 | L43E126 | 206 | 183 | E025, E067, E086, E090, E105, E126 |
| R2 | L47 | L47E005 | L47E005 | L43 | L43E051 | L43E051 | 205 | 187 | E005, E016, E025, E050, E098, E116 |
| R3 | L47 | L47E101 | L47E101 | L43 | L43E126 | L6E113 | 163 | 151 | E001, E002, E046, E060, E062, E101, E125 |
| all | L47 | none | L42E048 | L41 | L41E023 | L42E048 | 104 | 99 | - |

**Qwen3-30B-A3B-Base (tokenizer defaults): Jaccard overlap of the recurrent (layer, expert) pairs (all layers, discovery activity ≥ threshold) between categories**

| recurrent pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.14 | 0.39 | 0.37 | 0.38 | 0.24 | 0.41 |
| S2 | 0.14 | 1.00 | 0.12 | 0.23 | 0.17 | 0.09 | 0.20 |
| S3 | 0.39 | 0.12 | 1.00 | 0.18 | 0.41 | 0.08 | 0.28 |
| R1 | 0.37 | 0.23 | 0.18 | 1.00 | 0.33 | 0.32 | 0.47 |
| R2 | 0.38 | 0.17 | 0.41 | 0.33 | 1.00 | 0.14 | 0.40 |
| R3 | 0.24 | 0.09 | 0.08 | 0.32 | 0.14 | 1.00 | 0.25 |
| all | 0.41 | 0.20 | 0.28 | 0.47 | 0.40 | 0.25 | 1.00 |

**Qwen3-30B-A3B-Base (tokenizer defaults): Jaccard overlap of the recurrent (layer, expert) pairs restricted to interior layers (≤ L−5)**

| recurrent pairs, interior layers: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.15 | 0.38 | 0.40 | 0.36 | 0.26 | 0.42 |
| S2 | 0.15 | 1.00 | 0.12 | 0.23 | 0.18 | 0.09 | 0.21 |
| S3 | 0.38 | 0.12 | 1.00 | 0.19 | 0.41 | 0.08 | 0.27 |
| R1 | 0.40 | 0.23 | 0.19 | 1.00 | 0.36 | 0.34 | 0.50 |
| R2 | 0.36 | 0.18 | 0.41 | 0.36 | 1.00 | 0.14 | 0.41 |
| R3 | 0.26 | 0.09 | 0.08 | 0.34 | 0.14 | 1.00 | 0.26 |
| all | 0.42 | 0.21 | 0.27 | 0.50 | 0.41 | 0.26 | 1.00 |

**Qwen3-30B-A3B-Base (tokenizer defaults): Jaccard overlap of the top-10 joint (layer, expert) pairs between categories**

| top-10 joint pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.00 | 0.11 | 0.05 | 0.11 | 0.00 | 0.11 |
| S2 | 0.00 | 1.00 | 0.00 | 0.11 | 0.00 | 0.00 | 0.05 |
| S3 | 0.11 | 0.00 | 1.00 | 0.00 | 0.18 | 0.00 | 0.11 |
| R1 | 0.05 | 0.11 | 0.00 | 1.00 | 0.05 | 0.05 | 0.11 |
| R2 | 0.11 | 0.00 | 0.18 | 0.05 | 1.00 | 0.00 | 0.25 |
| R3 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 | 1.00 | 0.00 |
| all | 0.11 | 0.05 | 0.11 | 0.11 | 0.25 | 0.00 | 1.00 |

**Qwen3-30B-A3B-Base (tokenizer defaults): Jaccard overlap of the top-10 joint pairs restricted to interior layers**

| top-10 interior joint pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.00 | 0.18 | 0.05 | 0.18 | 0.00 | 0.11 |
| S2 | 0.00 | 1.00 | 0.05 | 0.00 | 0.00 | 0.00 | 0.05 |
| S3 | 0.18 | 0.05 | 1.00 | 0.00 | 0.11 | 0.00 | 0.18 |
| R1 | 0.05 | 0.00 | 0.00 | 1.00 | 0.11 | 0.11 | 0.11 |
| R2 | 0.18 | 0.00 | 0.11 | 0.11 | 1.00 | 0.00 | 0.33 |
| R3 | 0.00 | 0.00 | 0.00 | 0.11 | 0.00 | 1.00 | 0.00 |
| all | 0.11 | 0.05 | 0.18 | 0.11 | 0.33 | 0.00 | 1.00 |

**Qwen3-30B-A3B-Base (tokenizer defaults): (layer, expert) pairs recurrent in ≥ 2 categories**

| pair | n categories | categories |
|---|---|---|
| L2E026 | 6 | S1, S2, S3, R1, R2, R3 |
| L4E084 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E025 | 6 | S1, S2, S3, R1, R2, R3 |
| L26E065 | 6 | S1, S2, S3, R1, R2, R3 |
| L29E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L31E057 | 6 | S1, S2, S3, R1, R2, R3 |
| L35E119 | 6 | S1, S2, S3, R1, R2, R3 |
| L41E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L43E086 | 6 | S1, S2, S3, R1, R2, R3 |
| L44E091 | 6 | S1, S2, S3, R1, R2, R3 |
| L45E014 | 6 | S1, S2, S3, R1, R2, R3 |
| L1E110 | 5 | S1, S3, R1, R2, R3 |
| L8E007 | 5 | S1, S3, R1, R2, R3 |
| L12E088 | 5 | S1, S2, R1, R2, R3 |
| L13E022 | 5 | S1, S2, S3, R1, R2 |
| L13E046 | 5 | S1, S2, S3, R1, R3 |
| L16E061 | 5 | S1, S2, R1, R2, R3 |
| L16E120 | 5 | S1, S3, R1, R2, R3 |
| L19E057 | 5 | S1, S2, R1, R2, R3 |
| L21E003 | 5 | S1, S2, R1, R2, R3 |
| L22E044 | 5 | S1, S2, R1, R2, R3 |
| L22E079 | 5 | S1, S2, R1, R2, R3 |
| L24E088 | 5 | S1, S2, R1, R2, R3 |
| L25E022 | 5 | S1, S2, S3, R1, R2 |
| L25E046 | 5 | S1, S2, R1, R2, R3 |
| L26E056 | 5 | S1, S2, S3, R1, R2 |
| L28E061 | 5 | S1, S2, S3, R1, R2 |
| L28E125 | 5 | S1, S2, S3, R1, R2 |
| L29E025 | 5 | S1, S2, S3, R1, R2 |
| L33E003 | 5 | S1, S2, S3, R1, R2 |
| L34E044 | 5 | S1, S2, S3, R1, R2 |
| L34E049 | 5 | S1, S3, R1, R2, R3 |
| L34E079 | 5 | S1, S2, S3, R1, R2 |
| L35E091 | 5 | S1, S2, S3, R1, R2 |
| L36E088 | 5 | S1, S2, S3, R1, R2 |
| L37E022 | 5 | S1, S2, S3, R1, R2 |
| L37E046 | 5 | S1, S2, S3, R1, R2 |
| L37E054 | 5 | S1, S3, R1, R2, R3 |
| L38E056 | 5 | S1, S2, S3, R1, R2 |

**Qwen3-30B-A3B-Base (tokenizer defaults): interior (layer, expert) pairs recurrent in ≥ 2 categories**

| pair (interior) | n categories | categories |
|---|---|---|
| L2E026 | 6 | S1, S2, S3, R1, R2, R3 |
| L4E084 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E025 | 6 | S1, S2, S3, R1, R2, R3 |
| L26E065 | 6 | S1, S2, S3, R1, R2, R3 |
| L29E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L31E057 | 6 | S1, S2, S3, R1, R2, R3 |
| L35E119 | 6 | S1, S2, S3, R1, R2, R3 |
| L41E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L43E086 | 6 | S1, S2, S3, R1, R2, R3 |
| L1E110 | 5 | S1, S3, R1, R2, R3 |
| L8E007 | 5 | S1, S3, R1, R2, R3 |
| L12E088 | 5 | S1, S2, R1, R2, R3 |
| L13E022 | 5 | S1, S2, S3, R1, R2 |
| L13E046 | 5 | S1, S2, S3, R1, R3 |
| L16E061 | 5 | S1, S2, R1, R2, R3 |
| L16E120 | 5 | S1, S3, R1, R2, R3 |
| L19E057 | 5 | S1, S2, R1, R2, R3 |
| L21E003 | 5 | S1, S2, R1, R2, R3 |
| L22E044 | 5 | S1, S2, R1, R2, R3 |
| L22E079 | 5 | S1, S2, R1, R2, R3 |
| L24E088 | 5 | S1, S2, R1, R2, R3 |
| L25E022 | 5 | S1, S2, S3, R1, R2 |
| L25E046 | 5 | S1, S2, R1, R2, R3 |
| L26E056 | 5 | S1, S2, S3, R1, R2 |
| L28E061 | 5 | S1, S2, S3, R1, R2 |
| L28E125 | 5 | S1, S2, S3, R1, R2 |
| L29E025 | 5 | S1, S2, S3, R1, R2 |
| L33E003 | 5 | S1, S2, S3, R1, R2 |
| L34E044 | 5 | S1, S2, S3, R1, R2 |
| L34E049 | 5 | S1, S3, R1, R2, R3 |
| L34E079 | 5 | S1, S2, S3, R1, R2 |
| L35E091 | 5 | S1, S2, S3, R1, R2 |
| L36E088 | 5 | S1, S2, S3, R1, R2 |
| L37E022 | 5 | S1, S2, S3, R1, R2 |
| L37E046 | 5 | S1, S2, S3, R1, R2 |
| L37E054 | 5 | S1, S3, R1, R2, R3 |
| L38E056 | 5 | S1, S2, S3, R1, R2 |
| L38E065 | 5 | S1, S2, S3, R1, R2 |
| L39E042 | 5 | S1, S2, S3, R1, R2 |

![expert overlap qwen3](figures/ext4_overlap_qwen3.png)

- **S1** (closing bracket, n=128+128): L*=47, block rescue +2.420 [+2.069, +2.803]; interior L*=42 +1.261 [+1.044, +1.506] -> L42E048 rescue +0.916 [+0.737, +1.117] Spec +0.873 [+0.700, +1.068], interior joint winner L43E084 rescue +0.940 Spec +0.930; two-stage expert L47E025 (active 119/128 disc), rescue +1.153 [+0.951, +1.368], Spec +0.973 [+0.784, +1.175], rank-1 among active in 65/124; coalition (clean top-k) +2.036; joint winner L47E025 (same).
- **S2** (block keyword, n=128+128): L*=47, block rescue +0.642 [+0.505, +0.780]; interior L*=41 +0.488 [+0.400, +0.575] -> L41E041 rescue +0.551 [+0.452, +0.652] Spec +0.549 [+0.451, +0.649], interior joint winner L41E041 rescue +0.551 Spec +0.549; two-stage expert L47E062 (active 111/128 disc), rescue +0.336 [+0.237, +0.438], Spec +0.343 [+0.237, +0.449], rank-1 among active in 70/110; coalition (clean top-k) +0.109; joint winner L41E041 rescue +0.551 Spec +0.549.
- **S3** (keyword completion, n=128+128): L*=47, block rescue +1.054 [+0.836, +1.277]; interior L*=42 +0.441 [+0.325, +0.566] -> L42E044 rescue +0.125 [+0.078, +0.176] Spec +0.090 [+0.042, +0.139], interior joint winner L42E044 rescue +0.125 Spec +0.090; two-stage expert L47E025 (active 101/128 disc), rescue +0.188 [+0.143, +0.233], Spec +0.154 [+0.098, +0.209], rank-1 among active in 39/104; coalition (clean top-k) +0.242; joint winner L47E025 (same).
- **R1** (variable recall, n=128+128): L*=43, block rescue +0.391 [+0.299, +0.490]; interior L*=43 +0.391 [+0.299, +0.490] -> L43E126 rescue +0.326 [+0.252, +0.408] Spec +0.316 [+0.242, +0.400], interior joint winner L43E126 rescue +0.326 Spec +0.316; two-stage expert L43E126 (active 121/128 disc), rescue +0.326 [+0.252, +0.408], Spec +0.316 [+0.242, +0.400], rank-1 among active in 87/118; coalition (clean top-k) +0.370; joint winner L43E126 (same).
- **R2** (attribute / API recall, n=128+128): L*=47, block rescue +0.935 [+0.679, +1.223]; interior L*=43 +0.353 [+0.252, +0.453] -> L43E051 rescue +0.230 [+0.163, +0.301] Spec +0.234 [+0.166, +0.306], interior joint winner L43E051 rescue +0.230 Spec +0.234; two-stage expert L47E005 (active 124/128 disc), rescue +0.118 [+0.040, +0.202], Spec +0.042 [-0.028, +0.114], rank-1 among active in 37/125; coalition (clean top-k) +0.594; joint winner L47E005 (same).
- **R3** (constant recall, n=128+128): L*=47, block rescue +0.828 [+0.600, +1.056]; interior L*=43 +0.216 [+0.129, +0.305] -> L43E126 rescue +0.070 [+0.036, +0.106] Spec +0.055 [+0.020, +0.092], interior joint winner L6E113 rescue -0.015 Spec -0.015; two-stage expert L47E101 (active 98/128 disc), rescue +0.117 [+0.077, +0.159], Spec +0.079 [+0.032, +0.128], rank-1 among active in 37/92; coalition (clean top-k) +0.513; joint winner L47E101 (same).
- **Cross-category overlap**: mean pairwise Jaccard of the recurrent (layer, expert) sets 0.24 (within syntax 0.22, within recall 0.27, syntax-recall 0.24); of the top-10 joint pairs 0.04. Selected layers: S1 L47, S2 L47, S3 L47, R1 L43, R2 L47, R3 L47; pairs recurrent in every category: 12.
- **Factual-recall experts on code**: none of them is recurrent or rescues > 0.1 on any code category.

### Mixtral-8x7B-v0.1 (no BOS, paper protocol)

Scanned 4671 items; passing the paper filter per category: S1 690, S2 316, S3 241, R1 379, R2 321, R3 318. Sets with fewer than 256 passing items use all passing items (marked partial; recurrence threshold = half the discovery split).

**Mixtral-8x7B-v0.1 (no BOS, paper protocol): per-category localisation on CodeFact (validation split; recurrence threshold 64 of 128)**

| Set | n (disc+val) | top-1 = true | mean Δ_clean / drop | L* | block rescue (val) | 2nd layer | e* (two-stage) | active disc / val | expert rescue (val) | Spec (active-random) | rank-1 among active | coalition clean / union | joint winner (all layers) | recurrent pairs (all layers) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 128+128 | 51 % | +13.21 / +4.01 | L31 | +1.886 [+1.638, +2.143] | L30 +1.013 | L31E000 | 128/128 / 128/128 | +1.659 [+1.411, +1.916] | +1.443 [+1.195, +1.704] | 115/128 | +1.865 / +1.892 | L31E000 (= two-stage) | 45 |
| S2 block keyword | 128+128 | 94 % | +5.83 / +1.88 | L17 | +0.406 [+0.318, +0.506] | L18 +0.337 | L17E003 | 128/128 / 128/128 | +0.356 [+0.275, +0.446] | +0.322 [+0.244, +0.408] | 107/128 | +0.382 / +0.427 | L17E003 (= two-stage) | 53 |
| S3 keyword completion (partial) | 120+121 of 241 pass | 98 % | +6.37 / +1.89 | L31 | +0.582 [+0.431, +0.736] | L29 +0.278 | L31E000 | 102/120 / 109/121 | +0.361 [+0.268, +0.459] | +0.263 [+0.168, +0.356] | 93/109 | +0.436 / +0.574 | L18E002 +0.186 [+0.108, +0.270], Spec +0.126 | 45 |
| R1 variable recall | 128+128 | 94 % | +8.04 / +2.45 | L31 | +0.455 [+0.208, +0.713] | L18 +0.140 | L31E006 | 121/128 / 116/128 | +0.028 [-0.133, +0.197] | -0.242 [-0.440, -0.055] | 50/116 | +0.299 / +0.469 | L31E006 (= two-stage) | 41 |
| R2 attribute / API recall | 128+128 | 96 % | +7.75 / +1.58 | L31 | +0.458 [+0.347, +0.573] | L18 +0.205 | L31E005 | 76/128 / 70/128 | +0.312 [+0.231, +0.400] | +0.496 [+0.392, +0.601] | 67/70 | -0.007 / +0.464 | L31E005 (= two-stage) | 60 |
| R3 constant recall | 128+128 | 93 % | +8.35 / +1.83 | L31 | +0.607 [+0.421, +0.803] | L28 +0.195 | L31E005 | 74/128 / 76/128 | +0.264 [+0.134, +0.409] | +0.155 [+0.012, +0.309] | 54/76 | +0.481 / +0.622 | L31E005 (= two-stage) | 38 |
| all mixed | 128+128 | 89 % | +8.27 / +2.06 | L31 | +0.744 [+0.553, +0.953] | L30 +0.276 | none (max activity 60/128 < 64) |  |  |  |  | +0.599 / +0.738 | L23E005 +0.019 [-0.010, +0.052], Spec +0.010 | 3 |

![layer curves mixtral](figures/ext4_curves_mixtral.png)

**Mixtral-8x7B-v0.1 (no BOS, paper protocol): the same selection restricted to interior layers (excluding the last 4 MoE blocks, whose patch acts as a read-out on code)**

| Set | last-layer block rescue (val) | L* interior (≤ L−5) | block rescue (val) | e* interior | active disc / val | expert rescue (val) | Spec (active-random) | rank-1 among active | coalition clean / union | joint winner (interior layers) |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | L31: +1.886 [+1.638, +2.143] | L0 | +0.362 [+0.191, +0.589] | L0E005 | 84/128 / 75/128 | +0.185 [+0.096, +0.293] | +0.064 [-0.037, +0.171] | 56/75 | +0.342 / +0.380 | L0E005 +0.185 [+0.096, +0.293], Spec +0.064 [-0.037, +0.171] (rank 5 over all layers) |
| S2 | L31: +0.006 [-0.083, +0.097] | L17 | +0.406 [+0.318, +0.506] | L17E003 | 128/128 / 128/128 | +0.356 [+0.275, +0.446] | +0.322 [+0.244, +0.408] | 107/128 | +0.382 / +0.427 | L17E003 +0.356 [+0.275, +0.446], Spec +0.322 [+0.244, +0.408] (rank 1 over all layers) |
| S3 | L31: +0.582 [+0.431, +0.736] | L19 | +0.260 [+0.176, +0.346] | L19E005 | 88/120 / 88/121 | +0.147 [+0.082, +0.212] | +0.057 [-0.020, +0.136] | 61/88 | +0.259 / +0.257 | L18E002 +0.186 [+0.108, +0.270], Spec +0.126 [+0.044, +0.210] (rank 1 over all layers) |
| R1 | L31: +0.455 [+0.208, +0.713] | L27 | +0.120 [+0.078, +0.165] | L27E006 | 92/128 / 94/128 | +0.031 [+0.004, +0.058] | -0.005 [-0.038, +0.031] | 63/94 | +0.097 / +0.099 | L27E006 +0.031 [+0.004, +0.058], Spec -0.005 [-0.038, +0.031] (rank 3 over all layers) |
| R2 | L31: +0.458 [+0.347, +0.573] | L20 | +0.190 [+0.155, +0.226] | L20E006 | 95/128 / 97/128 | +0.095 [+0.069, +0.122] | +0.065 [+0.027, +0.104] | 80/97 | +0.158 / +0.176 | L19E004 +0.094 [+0.051, +0.139], Spec +0.069 [+0.030, +0.110] (rank 2 over all layers) |
| R3 | L31: +0.607 [+0.421, +0.803] | L20 | +0.123 [+0.073, +0.175] | L20E000 | 89/128 / 96/128 | +0.051 [+0.020, +0.081] | -0.001 [-0.042, +0.038] | 67/96 | +0.119 / +0.119 | L20E000 +0.051 [+0.020, +0.081], Spec -0.001 [-0.042, +0.038] (rank 5 over all layers) |
| all | L31: +0.744 [+0.553, +0.953] | L19 | +0.168 [+0.113, +0.225] | none (max activity 47/128) |  |  |  |  | +0.148 / +0.166 | L23E005 +0.019 [-0.010, +0.052], Spec +0.010 [-0.028, +0.048] (rank 1 over all layers) |

**Mixtral-8x7B-v0.1 (no BOS, paper protocol): the factual-recall experts (L19E006 (paper), L19E002 (BOS run), L18E001 (ext1 joint winner)) on the code categories**

| Set | factual expert | disc active | recurrent (≥ threshold) | joint rank | val active | val rescue | val Spec |
|---|---|---|---|---|---|---|---|
| S1 | L19E006 | 13/128 | no | - | 13/128 | +0.003 [-0.006, +0.012] | -0.032 [-0.071, +0.005] |
| S1 | L19E002 | 7/128 | no | - | 9/128 | +0.004 [-0.016, +0.026] | -0.022 [-0.060, +0.016] |
| S1 | L18E001 | 4/128 | no | - | 5/128 | -0.000 [-0.003, +0.001] | -0.113 [-0.160, -0.066] |
| S2 | L19E006 | 73/128 | yes | 42 | 67/128 | +0.005 [-0.008, +0.019] | -0.268 [-0.348, -0.194] |
| S2 | L19E002 | 128/128 | yes | 3 | 128/128 | +0.322 [+0.247, +0.403] | +0.313 [+0.238, +0.395] |
| S2 | L18E001 | 0/128 | no | - | 0/128 | 0 (never active) | n/a |
| S3 | L19E006 | 9/120 | no | - | 10/121 | +0.013 [+0.002, +0.030] | -0.137 [-0.203, -0.071] |
| S3 | L19E002 | 2/120 | no | - | 3/121 | +0.002 [+0.000, +0.005] | -0.148 [-0.213, -0.084] |
| S3 | L18E001 | 2/120 | no | - | 0/121 | 0 (never active) | n/a |
| R1 | L19E006 | 0/128 | no | - | 0/128 | 0 (never active) | n/a |
| R1 | L19E002 | 9/128 | no | - | 5/128 | +0.009 [-0.001, +0.025] | -0.050 [-0.084, -0.016] |
| R1 | L18E001 | 120/128 | yes | 5 | 123/128 | +0.141 [+0.082, +0.207] | +0.118 [+0.060, +0.185] |
| R2 | L19E006 | 26/128 | no | - | 18/128 | -0.001 [-0.008, +0.005] | -0.067 [-0.104, -0.031] |
| R2 | L19E002 | 11/128 | no | - | 3/128 | +0.002 [-0.001, +0.008] | -0.053 [-0.090, -0.017] |
| R2 | L18E001 | 0/128 | no | - | 0/128 | 0 (never active) | n/a |
| R3 | L19E006 | 32/128 | no | - | 33/128 | +0.021 [-0.001, +0.045] | -0.038 [-0.073, -0.001] |
| R3 | L19E002 | 15/128 | no | - | 21/128 | +0.008 [-0.004, +0.020] | -0.047 [-0.083, -0.011] |
| R3 | L18E001 | 67/128 | yes | 7 | 56/128 | +0.021 [-0.004, +0.049] | -0.011 [-0.043, +0.021] |
| all | L19E006 | 20/128 | no | - | 33/128 | +0.008 [-0.002, +0.020] | -0.088 [-0.137, -0.041] |
| all | L19E002 | 25/128 | no | - | 32/128 | +0.044 [+0.011, +0.081] | -0.002 [-0.051, +0.049] |
| all | L18E001 | 37/128 | no | - | 31/128 | +0.034 [+0.001, +0.072] | -0.049 [-0.100, +0.005] |

**Mixtral-8x7B-v0.1 (no BOS, paper protocol): selected layers, experts and recurrent sets per category**

| Set | L* (paper rule) | two-stage e* | joint winner (all layers) | L* interior | two-stage e* interior | joint winner (interior) | recurrent pairs (all layers) | recurrent pairs (interior) | recurrent experts at L* |
|---|---|---|---|---|---|---|---|---|---|
| S1 | L31 | L31E000 | L31E000 | L0 | L0E005 | L0E005 | 45 | 37 | E000, E001 |
| S2 | L17 | L17E003 | L17E003 | L17 | L17E003 | L17E003 | 53 | 47 | E003 |
| S3 | L31 | L31E000 | L18E002 | L19 | L19E005 | L18E002 | 45 | 38 | E000, E001 |
| R1 | L31 | L31E006 | L31E006 | L27 | L27E006 | L27E006 | 41 | 33 | E003, E006 |
| R2 | L31 | L31E005 | L31E005 | L20 | L20E006 | L19E004 | 60 | 51 | E003, E005, E006 |
| R3 | L31 | L31E005 | L31E005 | L20 | L20E000 | L20E000 | 38 | 33 | E005, E007 |
| all | L31 | none | L23E005 | L19 | none | L23E005 | 3 | 3 | - |

**Mixtral-8x7B-v0.1 (no BOS, paper protocol): Jaccard overlap of the recurrent (layer, expert) pairs (all layers, discovery activity ≥ threshold) between categories**

| recurrent pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.05 | 0.50 | 0.00 | 0.07 | 0.01 | 0.00 |
| S2 | 0.05 | 1.00 | 0.01 | 0.06 | 0.11 | 0.03 | 0.06 |
| S3 | 0.50 | 0.01 | 1.00 | 0.01 | 0.07 | 0.02 | 0.00 |
| R1 | 0.00 | 0.06 | 0.01 | 1.00 | 0.10 | 0.18 | 0.02 |
| R2 | 0.07 | 0.11 | 0.07 | 0.10 | 1.00 | 0.11 | 0.03 |
| R3 | 0.01 | 0.03 | 0.02 | 0.18 | 0.11 | 1.00 | 0.00 |
| all | 0.00 | 0.06 | 0.00 | 0.02 | 0.03 | 0.00 | 1.00 |

**Mixtral-8x7B-v0.1 (no BOS, paper protocol): Jaccard overlap of the recurrent (layer, expert) pairs restricted to interior layers (≤ L−5)**

| recurrent pairs, interior layers: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.05 | 0.50 | 0.00 | 0.09 | 0.01 | 0.00 |
| S2 | 0.05 | 1.00 | 0.01 | 0.03 | 0.09 | 0.04 | 0.06 |
| S3 | 0.50 | 0.01 | 1.00 | 0.01 | 0.09 | 0.03 | 0.00 |
| R1 | 0.00 | 0.03 | 0.01 | 1.00 | 0.08 | 0.18 | 0.03 |
| R2 | 0.09 | 0.09 | 0.09 | 0.08 | 1.00 | 0.11 | 0.04 |
| R3 | 0.01 | 0.04 | 0.03 | 0.18 | 0.11 | 1.00 | 0.00 |
| all | 0.00 | 0.06 | 0.00 | 0.03 | 0.04 | 0.00 | 1.00 |

**Mixtral-8x7B-v0.1 (no BOS, paper protocol): Jaccard overlap of the top-10 joint (layer, expert) pairs between categories**

| top-10 joint pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.00 | 0.11 | 0.00 | 0.00 | 0.00 | 0.00 |
| S2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 |
| S3 | 0.11 | 0.00 | 1.00 | 0.00 | 0.05 | 0.00 | 0.00 |
| R1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.25 | 0.08 |
| R2 | 0.00 | 0.05 | 0.05 | 0.00 | 1.00 | 0.11 | 0.08 |
| R3 | 0.00 | 0.00 | 0.00 | 0.25 | 0.11 | 1.00 | 0.00 |
| all | 0.00 | 0.00 | 0.00 | 0.08 | 0.08 | 0.00 | 1.00 |

**Mixtral-8x7B-v0.1 (no BOS, paper protocol): Jaccard overlap of the top-10 joint pairs restricted to interior layers**

| top-10 interior joint pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.00 | 0.11 | 0.00 | 0.00 | 0.00 | 0.00 |
| S2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 |
| S3 | 0.11 | 0.00 | 1.00 | 0.00 | 0.05 | 0.00 | 0.00 |
| R1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.05 | 0.11 | 0.08 |
| R2 | 0.00 | 0.05 | 0.05 | 0.05 | 1.00 | 0.11 | 0.08 |
| R3 | 0.00 | 0.00 | 0.00 | 0.11 | 0.11 | 1.00 | 0.00 |
| all | 0.00 | 0.00 | 0.00 | 0.08 | 0.08 | 0.00 | 1.00 |

**Mixtral-8x7B-v0.1 (no BOS, paper protocol): (layer, expert) pairs recurrent in ≥ 2 categories**

| pair | n categories | categories |
|---|---|---|
| L0E005 | 3 | S1, S2, S3 |
| L2E004 | 3 | S1, S3, R3 |
| L5E006 | 3 | S2, R2, R3 |
| L7E007 | 3 | S1, S3, R2 |
| L11E000 | 3 | R1, R2, R3 |
| L12E006 | 3 | R1, R2, R3 |
| L14E003 | 3 | S1, S3, R2 |
| L15E007 | 3 | S1, S3, R2 |
| L16E002 | 3 | S1, S3, R2 |
| L26E001 | 3 | S1, S3, R2 |
| L30E001 | 3 | S2, R1, R2 |
| L31E006 | 3 | S2, R1, R2 |
| L0E006 | 2 | S2, R2 |
| L0E007 | 2 | S3, R2 |
| L1E004 | 2 | S2, R3 |
| L2E003 | 2 | S2, R2 |
| L3E001 | 2 | S1, S3 |
| L3E005 | 2 | R1, R2 |
| L5E005 | 2 | S1, S3 |
| L6E006 | 2 | S1, S3 |
| L7E003 | 2 | S3, R3 |
| L8E003 | 2 | S1, S3 |
| L8E005 | 2 | R2, R3 |
| L9E001 | 2 | R2, R3 |
| L10E001 | 2 | S1, S3 |
| L10E002 | 2 | S2, R2 |
| L10E003 | 2 | S1, S3 |
| L10E004 | 2 | R1, R3 |
| L10E007 | 2 | R1, R3 |
| L11E001 | 2 | R1, R3 |
| L11E002 | 2 | S1, S3 |
| L12E000 | 2 | S1, R2 |
| L12E004 | 2 | S1, S3 |
| L12E005 | 2 | S2, R2 |
| L13E006 | 2 | S3, R1 |
| L14E000 | 2 | S1, S2 |
| L14E001 | 2 | R1, R2 |
| L15E006 | 2 | R1, R3 |
| L16E006 | 2 | R2, R3 |
| L17E005 | 2 | S1, S3 |

**Mixtral-8x7B-v0.1 (no BOS, paper protocol): interior (layer, expert) pairs recurrent in ≥ 2 categories**

| pair (interior) | n categories | categories |
|---|---|---|
| L0E005 | 3 | S1, S2, S3 |
| L2E004 | 3 | S1, S3, R3 |
| L5E006 | 3 | S2, R2, R3 |
| L7E007 | 3 | S1, S3, R2 |
| L11E000 | 3 | R1, R2, R3 |
| L12E006 | 3 | R1, R2, R3 |
| L14E003 | 3 | S1, S3, R2 |
| L15E007 | 3 | S1, S3, R2 |
| L16E002 | 3 | S1, S3, R2 |
| L26E001 | 3 | S1, S3, R2 |
| L0E006 | 2 | S2, R2 |
| L0E007 | 2 | S3, R2 |
| L1E004 | 2 | S2, R3 |
| L2E003 | 2 | S2, R2 |
| L3E001 | 2 | S1, S3 |
| L3E005 | 2 | R1, R2 |
| L5E005 | 2 | S1, S3 |
| L6E006 | 2 | S1, S3 |
| L7E003 | 2 | S3, R3 |
| L8E003 | 2 | S1, S3 |
| L8E005 | 2 | R2, R3 |
| L9E001 | 2 | R2, R3 |
| L10E001 | 2 | S1, S3 |
| L10E002 | 2 | S2, R2 |
| L10E003 | 2 | S1, S3 |
| L10E004 | 2 | R1, R3 |
| L10E007 | 2 | R1, R3 |
| L11E001 | 2 | R1, R3 |
| L11E002 | 2 | S1, S3 |
| L12E000 | 2 | S1, R2 |
| L12E004 | 2 | S1, S3 |
| L12E005 | 2 | S2, R2 |
| L13E006 | 2 | S3, R1 |
| L14E000 | 2 | S1, S2 |
| L14E001 | 2 | R1, R2 |
| L15E006 | 2 | R1, R3 |
| L16E006 | 2 | R2, R3 |
| L17E005 | 2 | S1, S3 |
| L18E000 | 2 | S2, R3 |
| L18E001 | 2 | R1, R3 |

![expert overlap mixtral](figures/ext4_overlap_mixtral.png)

- **S1** (closing bracket, n=128+128): L*=31, block rescue +1.886 [+1.638, +2.143]; interior L*=0 +0.362 [+0.191, +0.589] -> L0E005 rescue +0.185 [+0.096, +0.293] Spec +0.064 [-0.037, +0.171], interior joint winner L0E005 rescue +0.185 Spec +0.064; two-stage expert L31E000 (active 128/128 disc), rescue +1.659 [+1.411, +1.916], Spec +1.443 [+1.195, +1.704], rank-1 among active in 115/128; coalition (clean top-k) +1.865; joint winner L31E000 (same).
- **S2** (block keyword, n=128+128): L*=17, block rescue +0.406 [+0.318, +0.506]; interior L*=17 +0.406 [+0.318, +0.506] -> L17E003 rescue +0.356 [+0.275, +0.446] Spec +0.322 [+0.244, +0.408], interior joint winner L17E003 rescue +0.356 Spec +0.322; two-stage expert L17E003 (active 128/128 disc), rescue +0.356 [+0.275, +0.446], Spec +0.322 [+0.244, +0.408], rank-1 among active in 107/128; coalition (clean top-k) +0.382; joint winner L17E003 (same).
- **S3** (keyword completion, n=120+121, partial): L*=31, block rescue +0.582 [+0.431, +0.736]; interior L*=19 +0.260 [+0.176, +0.346] -> L19E005 rescue +0.147 [+0.082, +0.212] Spec +0.057 [-0.020, +0.136], interior joint winner L18E002 rescue +0.186 Spec +0.126; two-stage expert L31E000 (active 102/120 disc), rescue +0.361 [+0.268, +0.459], Spec +0.263 [+0.168, +0.356], rank-1 among active in 93/109; coalition (clean top-k) +0.436; joint winner L18E002 rescue +0.186 Spec +0.126.
- **R1** (variable recall, n=128+128): L*=31, block rescue +0.455 [+0.208, +0.713]; interior L*=27 +0.120 [+0.078, +0.165] -> L27E006 rescue +0.031 [+0.004, +0.058] Spec -0.005 [-0.038, +0.031], interior joint winner L27E006 rescue +0.031 Spec -0.005; two-stage expert L31E006 (active 121/128 disc), rescue +0.028 [-0.133, +0.197], Spec -0.242 [-0.440, -0.055], rank-1 among active in 50/116; coalition (clean top-k) +0.299; joint winner L31E006 (same).
- **R2** (attribute / API recall, n=128+128): L*=31, block rescue +0.458 [+0.347, +0.573]; interior L*=20 +0.190 [+0.155, +0.226] -> L20E006 rescue +0.095 [+0.069, +0.122] Spec +0.065 [+0.027, +0.104], interior joint winner L19E004 rescue +0.094 Spec +0.069; two-stage expert L31E005 (active 76/128 disc), rescue +0.312 [+0.231, +0.400], Spec +0.496 [+0.392, +0.601], rank-1 among active in 67/70; coalition (clean top-k) -0.007; joint winner L31E005 (same).
- **R3** (constant recall, n=128+128): L*=31, block rescue +0.607 [+0.421, +0.803]; interior L*=20 +0.123 [+0.073, +0.175] -> L20E000 rescue +0.051 [+0.020, +0.081] Spec -0.001 [-0.042, +0.038], interior joint winner L20E000 rescue +0.051 Spec -0.001; two-stage expert L31E005 (active 74/128 disc), rescue +0.264 [+0.134, +0.409], Spec +0.155 [+0.012, +0.309], rank-1 among active in 54/76; coalition (clean top-k) +0.481; joint winner L31E005 (same).
- **Cross-category overlap**: mean pairwise Jaccard of the recurrent (layer, expert) sets 0.09 (within syntax 0.19, within recall 0.13, syntax-recall 0.04); of the top-10 joint pairs 0.04. Selected layers: S1 L31, S2 L17, S3 L31, R1 L31, R2 L31, R3 L31; pairs recurrent in every category: 0.
- **Factual-recall experts on code**: L19E006 on S2 (active 73/128, rescue +0.005, Spec -0.268); L19E002 on S2 (active 128/128, rescue +0.322, Spec +0.313); L18E001 on R1 (active 120/128, rescue +0.141, Spec +0.118); L18E001 on R3 (active 67/128, rescue +0.021, Spec -0.011).

### Qwen3-Coder-30B-A3B-Instruct (raw code prefix)

Scanned 6448 items; passing the paper filter per category: S1 810, S2 395, S3 474, R1 899, R2 576, R3 696. Sets with fewer than 256 passing items use all passing items (marked partial; recurrence threshold = half the discovery split).

**Qwen3-Coder-30B-A3B-Instruct (raw code prefix): per-category localisation on CodeFact (validation split; recurrence threshold 64 of 128)**

| Set | n (disc+val) | top-1 = true | mean Δ_clean / drop | L* | block rescue (val) | 2nd layer | e* (two-stage) | active disc / val | expert rescue (val) | Spec (active-random) | rank-1 among active | coalition clean / union | joint winner (all layers) | recurrent pairs (all layers) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 128+128 | 14 % | +13.81 / +4.39 | L47 | +1.869 [+1.563, +2.206] | L46 +0.819 | L47E025 | 125/128 / 124/128 | +1.208 [+0.973, +1.473] | +1.113 [+0.884, +1.370] | 98/124 | +1.684 / +1.855 | L47E025 (= two-stage) | 92 |
| S2 block keyword | 128+128 | 92 % | +10.17 / +2.01 | L47 | +0.644 [+0.477, +0.841] | L41 +0.395 | L47E077 | 68/128 / 64/128 | +0.055 [+0.019, +0.091] | +0.057 [+0.016, +0.098] | 30/64 | -0.071 / +0.642 | L41E041 +0.362 [+0.274, +0.459], Spec +0.343 | 227 |
| S3 keyword completion | 128+128 | 68 % | +11.20 / +3.55 | L47 | +1.547 [+1.183, +1.941] | L42 +0.439 | L47E014 | 111/128 / 120/128 | +0.773 [+0.585, +0.981] | +0.703 [+0.519, +0.908] | 74/120 | +1.170 / +1.537 | L47E014 (= two-stage) | 74 |
| R1 variable recall | 128+128 | 94 % | +13.46 / +4.85 | L47 | +0.085 [-0.231, +0.408] | L43 +0.217 | L47E060 | 88/128 / 80/128 | +0.055 [+0.004, +0.108] | +0.122 [+0.057, +0.190] | 19/80 | -0.306 / +0.108 | L43E126 +0.147 [+0.080, +0.219], Spec +0.135 | 82 |
| R2 attribute / API recall | 128+128 | 94 % | +13.94 / +3.57 | L47 | +1.995 [+1.594, +2.431] | L43 +0.468 | L47E116 | 128/128 / 126/128 | +0.427 [+0.321, +0.540] | +0.278 [+0.187, +0.375] | 53/126 | +1.804 / +1.985 | L47E116 (= two-stage) | 148 |
| R3 constant recall | 128+128 | 91 % | +13.57 / +3.73 | L47 | +0.860 [+0.535, +1.191] | L42 +0.102 | L47E002 | 93/128 / 87/128 | +0.128 [+0.062, +0.201] | +0.047 [-0.021, +0.115] | 27/87 | +0.669 / +0.848 | L47E002 (= two-stage) | 88 |
| all mixed | 128+128 | 75 % | +12.46 / +3.62 | L47 | +1.041 [+0.717, +1.375] | L42 +0.308 | none (max activity 50/128 < 64) |  |  |  |  | +0.593 / +1.031 | L45E014 +0.074 [+0.018, +0.136], Spec +0.071 | 20 |

![layer curves coder_raw](figures/ext4_curves_coder_raw.png)

**Qwen3-Coder-30B-A3B-Instruct (raw code prefix): the same selection restricted to interior layers (excluding the last 4 MoE blocks, whose patch acts as a read-out on code)**

| Set | last-layer block rescue (val) | L* interior (≤ L−5) | block rescue (val) | e* interior | active disc / val | expert rescue (val) | Spec (active-random) | rank-1 among active | coalition clean / union | joint winner (interior layers) |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | L47: +1.869 [+1.563, +2.206] | L41 | +0.785 [+0.570, +1.066] | L41E108 | 125/128 / 121/128 | +0.670 [+0.484, +0.911] | +0.664 [+0.479, +0.903] | 88/121 | +0.719 / +0.755 | L41E108 +0.670 [+0.484, +0.911], Spec +0.664 [+0.479, +0.903] (rank 2 over all layers) |
| S2 | L47: +0.644 [+0.477, +0.841] | L39 | +0.338 [+0.246, +0.436] | L39E092 | 127/128 / 128/128 | +0.224 [+0.152, +0.301] | +0.202 [+0.132, +0.277] | 72/128 | +0.302 / +0.305 | L41E041 +0.362 [+0.274, +0.459], Spec +0.343 [+0.255, +0.441] (rank 1 over all layers) |
| S3 | L47: +1.547 [+1.183, +1.941] | L42 | +0.439 [+0.238, +0.677] | L42E048 | 71/128 / 80/128 | +0.151 [+0.094, +0.211] | +0.137 [+0.069, +0.206] | 35/80 | +0.438 / +0.456 | L3E057 +0.146 [+0.044, +0.266], Spec +0.111 [+0.025, +0.209] (rank 2 over all layers) |
| R1 | L47: +0.085 [-0.231, +0.408] | L43 | +0.217 [+0.099, +0.337] | L43E126 | 84/128 / 77/128 | +0.147 [+0.080, +0.219] | +0.135 [+0.064, +0.209] | 51/77 | +0.233 / +0.262 | L43E126 +0.147 [+0.080, +0.219], Spec +0.135 [+0.064, +0.209] (rank 1 over all layers) |
| R2 | L47: +1.995 [+1.594, +2.431] | L39 | +0.443 [+0.335, +0.551] | L39E065 | 106/128 / 108/128 | +0.150 [+0.073, +0.228] | +0.122 [+0.035, +0.207] | 52/108 | +0.386 / +0.406 | L43E051 +0.408 [+0.298, +0.525], Spec +0.407 [+0.298, +0.525] (rank 2 over all layers) |
| R3 | L47: +0.860 [+0.535, +1.191] | L42 | +0.102 [+0.024, +0.181] | L42E039 | 72/128 / 69/128 | -0.009 [-0.028, +0.010] | -0.033 [-0.060, -0.005] | 13/69 | +0.072 / +0.084 | L40E069 -0.007 [-0.047, +0.031], Spec -0.004 [-0.043, +0.034] (rank 5 over all layers) |
| all | L47: +1.041 [+0.717, +1.375] | L43 | +0.228 [+0.113, +0.353] | L43E086 | 91/128 / 99/128 | -0.012 [-0.037, +0.013] | -0.060 [-0.095, -0.028] | 22/99 | +0.202 / +0.239 | L2E026 +0.049 [-0.008, +0.114], Spec +0.031 [-0.018, +0.086] (rank 2 over all layers) |

**Qwen3-Coder-30B-A3B-Instruct (raw code prefix): the factual-recall experts (L44E069 / L42E115 (Qwen3-Base factual experts)) on the code categories**

| Set | factual expert | disc active | recurrent (≥ threshold) | joint rank | val active | val rescue | val Spec |
|---|---|---|---|---|---|---|---|
| S1 | L44E069 | 2/128 | no | - | 1/128 | -0.000 [-0.001, +0.000] | -0.054 [-0.079, -0.032] |
| S1 | L42E115 | 1/128 | no | - | 1/128 | +0.000 [+0.000, +0.000] | -0.063 [-0.095, -0.035] |
| S2 | L44E069 | 4/128 | no | - | 3/128 | +0.000 [+0.000, +0.000] | -0.014 [-0.031, +0.003] |
| S2 | L42E115 | 3/128 | no | - | 6/128 | +0.001 [-0.002, +0.005] | -0.018 [-0.042, +0.006] |
| S3 | L44E069 | 4/128 | no | - | 3/128 | -0.001 [-0.003, +0.001] | -0.014 [-0.031, +0.001] |
| S3 | L42E115 | 4/128 | no | - | 4/128 | +0.000 [+0.000, +0.000] | -0.061 [-0.123, -0.013] |
| R1 | L44E069 | 1/128 | no | - | 2/128 | +0.000 [+0.000, +0.001] | -0.028 [-0.062, +0.003] |
| R1 | L42E115 | 2/128 | no | - | 4/128 | +0.001 [+0.000, +0.002] | -0.024 [-0.051, +0.005] |
| R2 | L44E069 | 4/128 | no | - | 7/128 | -0.002 [-0.007, +0.001] | +0.011 [-0.006, +0.029] |
| R2 | L42E115 | 0/128 | no | - | 0/128 | 0 (never active) | n/a |
| R3 | L44E069 | 1/128 | no | - | 4/128 | -0.002 [-0.005, +0.000] | -0.018 [-0.041, +0.004] |
| R3 | L42E115 | 7/128 | no | - | 6/128 | -0.000 [-0.006, +0.006] | -0.010 [-0.032, +0.010] |
| all | L44E069 | 3/128 | no | - | 2/128 | -0.002 [-0.005, +0.000] | -0.021 [-0.037, -0.005] |
| all | L42E115 | 4/128 | no | - | 4/128 | -0.001 [-0.006, +0.003] | -0.044 [-0.071, -0.020] |

**Qwen3-Coder-30B-A3B-Instruct (raw code prefix): selected layers, experts and recurrent sets per category**

| Set | L* (paper rule) | two-stage e* | joint winner (all layers) | L* interior | two-stage e* interior | joint winner (interior) | recurrent pairs (all layers) | recurrent pairs (interior) | recurrent experts at L* |
|---|---|---|---|---|---|---|---|---|---|
| S1 | L47 | L47E025 | L47E025 | L41 | L41E108 | L41E108 | 92 | 76 | E011, E014, E025, E122 |
| S2 | L47 | L47E077 | L41E041 | L39 | L39E092 | L41E041 | 227 | 215 | E045, E065, E077 |
| S3 | L47 | L47E014 | L47E014 | L42 | L42E048 | L3E057 | 74 | 64 | E011, E014, E025 |
| R1 | L47 | L47E060 | L43E126 | L43 | L43E126 | L43E126 | 82 | 67 | E001, E033, E060, E062, E066, E097, E115 |
| R2 | L47 | L47E116 | L47E116 | L39 | L39E065 | L43E051 | 148 | 133 | E005, E016, E025, E052, E098, E116 |
| R3 | L47 | L47E002 | L47E002 | L42 | L42E039 | L40E069 | 88 | 84 | E002, E046, E101, E125 |
| all | L47 | none | L45E014 | L43 | L43E086 | L2E026 | 20 | 17 | - |

**Qwen3-Coder-30B-A3B-Instruct (raw code prefix): Jaccard overlap of the recurrent (layer, expert) pairs (all layers, discovery activity ≥ threshold) between categories**

| recurrent pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.06 | 0.20 | 0.15 | 0.17 | 0.10 | 0.19 |
| S2 | 0.06 | 1.00 | 0.03 | 0.08 | 0.09 | 0.03 | 0.07 |
| S3 | 0.20 | 0.03 | 1.00 | 0.08 | 0.21 | 0.05 | 0.15 |
| R1 | 0.15 | 0.08 | 0.08 | 1.00 | 0.14 | 0.10 | 0.21 |
| R2 | 0.17 | 0.09 | 0.21 | 0.14 | 1.00 | 0.05 | 0.11 |
| R3 | 0.10 | 0.03 | 0.05 | 0.10 | 0.05 | 1.00 | 0.10 |
| all | 0.19 | 0.07 | 0.15 | 0.21 | 0.11 | 0.10 | 1.00 |

**Qwen3-Coder-30B-A3B-Instruct (raw code prefix): Jaccard overlap of the recurrent (layer, expert) pairs restricted to interior layers (≤ L−5)**

| recurrent pairs, interior layers: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.05 | 0.18 | 0.16 | 0.16 | 0.12 | 0.19 |
| S2 | 0.05 | 1.00 | 0.03 | 0.07 | 0.08 | 0.03 | 0.06 |
| S3 | 0.18 | 0.03 | 1.00 | 0.07 | 0.22 | 0.05 | 0.14 |
| R1 | 0.16 | 0.07 | 0.07 | 1.00 | 0.15 | 0.12 | 0.22 |
| R2 | 0.16 | 0.08 | 0.22 | 0.15 | 1.00 | 0.06 | 0.10 |
| R3 | 0.12 | 0.03 | 0.05 | 0.12 | 0.06 | 1.00 | 0.11 |
| all | 0.19 | 0.06 | 0.14 | 0.22 | 0.10 | 0.11 | 1.00 |

**Qwen3-Coder-30B-A3B-Instruct (raw code prefix): Jaccard overlap of the top-10 joint (layer, expert) pairs between categories**

| top-10 joint pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.05 | 0.05 | 0.00 | 0.11 | 0.00 | 0.05 |
| S2 | 0.05 | 1.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.05 |
| S3 | 0.05 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.05 |
| R1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.05 | 0.00 | 0.00 |
| R2 | 0.11 | 0.05 | 0.00 | 0.05 | 1.00 | 0.00 | 0.11 |
| R3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.05 |
| all | 0.05 | 0.05 | 0.05 | 0.00 | 0.11 | 0.05 | 1.00 |

**Qwen3-Coder-30B-A3B-Instruct (raw code prefix): Jaccard overlap of the top-10 joint pairs restricted to interior layers**

| top-10 interior joint pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.00 | 0.05 | 0.05 | 0.18 | 0.00 | 0.00 |
| S2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| S3 | 0.05 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.05 |
| R1 | 0.05 | 0.00 | 0.00 | 1.00 | 0.11 | 0.00 | 0.00 |
| R2 | 0.18 | 0.00 | 0.00 | 0.11 | 1.00 | 0.00 | 0.05 |
| R3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.05 |
| all | 0.00 | 0.00 | 0.05 | 0.00 | 0.05 | 0.05 | 1.00 |

**Qwen3-Coder-30B-A3B-Instruct (raw code prefix): (layer, expert) pairs recurrent in ≥ 2 categories**

| pair | n categories | categories |
|---|---|---|
| L2E026 | 6 | S1, S2, S3, R1, R2, R3 |
| L4E084 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E025 | 6 | S1, S2, S3, R1, R2, R3 |
| L29E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L41E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L8E007 | 5 | S1, S3, R1, R2, R3 |
| L43E086 | 5 | S1, S2, R1, R2, R3 |
| L44E091 | 5 | S1, S2, S3, R1, R2 |
| L45E014 | 5 | S1, S2, S3, R1, R2 |
| L8E038 | 4 | S1, S2, R1, R3 |
| L20E038 | 4 | S1, S2, R1, R3 |
| L29E025 | 4 | S1, S2, R1, R2 |
| L37E046 | 4 | S1, S2, S3, R2 |
| L45E001 | 4 | S1, S2, R1, R2 |
| L2E096 | 3 | S1, S3, R2 |
| L4E036 | 3 | S1, S3, R2 |
| L9E075 | 3 | S2, S3, R2 |
| L12E101 | 3 | S1, S3, R2 |
| L13E046 | 3 | S1, S2, R3 |
| L16E038 | 3 | S2, R1, R2 |
| L22E106 | 3 | S1, S3, R2 |
| L23E035 | 3 | S1, S3, R2 |
| L26E065 | 3 | S2, R1, R2 |
| L28E038 | 3 | S2, R1, R2 |
| L28E120 | 3 | S1, R1, R3 |
| L31E055 | 3 | S1, S3, R2 |
| L31E057 | 3 | S1, R1, R2 |
| L33E003 | 3 | S1, R1, R2 |
| L34E044 | 3 | S2, R1, R2 |
| L34E079 | 3 | S2, R1, R2 |
| L34E106 | 3 | S1, S3, R2 |
| L35E035 | 3 | S1, S3, R2 |
| L35E093 | 3 | R1, R2, R3 |
| L36E088 | 3 | S1, S2, R2 |
| L37E022 | 3 | S3, R1, R2 |
| L38E056 | 3 | S2, R1, R2 |
| L38E106 | 3 | S1, S3, R2 |
| L39E065 | 3 | S1, S3, R2 |
| L40E038 | 3 | S1, S2, R1 |

**Qwen3-Coder-30B-A3B-Instruct (raw code prefix): interior (layer, expert) pairs recurrent in ≥ 2 categories**

| pair (interior) | n categories | categories |
|---|---|---|
| L2E026 | 6 | S1, S2, S3, R1, R2, R3 |
| L4E084 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E025 | 6 | S1, S2, S3, R1, R2, R3 |
| L29E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L41E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L8E007 | 5 | S1, S3, R1, R2, R3 |
| L43E086 | 5 | S1, S2, R1, R2, R3 |
| L8E038 | 4 | S1, S2, R1, R3 |
| L20E038 | 4 | S1, S2, R1, R3 |
| L29E025 | 4 | S1, S2, R1, R2 |
| L37E046 | 4 | S1, S2, S3, R2 |
| L2E096 | 3 | S1, S3, R2 |
| L4E036 | 3 | S1, S3, R2 |
| L9E075 | 3 | S2, S3, R2 |
| L12E101 | 3 | S1, S3, R2 |
| L13E046 | 3 | S1, S2, R3 |
| L16E038 | 3 | S2, R1, R2 |
| L22E106 | 3 | S1, S3, R2 |
| L23E035 | 3 | S1, S3, R2 |
| L26E065 | 3 | S2, R1, R2 |
| L28E038 | 3 | S2, R1, R2 |
| L28E120 | 3 | S1, R1, R3 |
| L31E055 | 3 | S1, S3, R2 |
| L31E057 | 3 | S1, R1, R2 |
| L33E003 | 3 | S1, R1, R2 |
| L34E044 | 3 | S2, R1, R2 |
| L34E079 | 3 | S2, R1, R2 |
| L34E106 | 3 | S1, S3, R2 |
| L35E035 | 3 | S1, S3, R2 |
| L35E093 | 3 | R1, R2, R3 |
| L36E088 | 3 | S1, S2, R2 |
| L37E022 | 3 | S3, R1, R2 |
| L38E056 | 3 | S2, R1, R2 |
| L38E106 | 3 | S1, S3, R2 |
| L39E065 | 3 | S1, S3, R2 |
| L40E038 | 3 | S1, S2, R1 |
| L40E061 | 3 | S1, R1, R2 |
| L41E025 | 3 | S2, R1, R2 |
| L42E045 | 3 | S1, R1, R2 |

![expert overlap coder_raw](figures/ext4_overlap_coder_raw.png)

- **S1** (closing bracket, n=128+128): L*=47, block rescue +1.869 [+1.563, +2.206]; interior L*=41 +0.785 [+0.570, +1.066] -> L41E108 rescue +0.670 [+0.484, +0.911] Spec +0.664 [+0.479, +0.903], interior joint winner L41E108 rescue +0.670 Spec +0.664; two-stage expert L47E025 (active 125/128 disc), rescue +1.208 [+0.973, +1.473], Spec +1.113 [+0.884, +1.370], rank-1 among active in 98/124; coalition (clean top-k) +1.684; joint winner L47E025 (same).
- **S2** (block keyword, n=128+128): L*=47, block rescue +0.644 [+0.477, +0.841]; interior L*=39 +0.338 [+0.246, +0.436] -> L39E092 rescue +0.224 [+0.152, +0.301] Spec +0.202 [+0.132, +0.277], interior joint winner L41E041 rescue +0.362 Spec +0.343; two-stage expert L47E077 (active 68/128 disc), rescue +0.055 [+0.019, +0.091], Spec +0.057 [+0.016, +0.098], rank-1 among active in 30/64; coalition (clean top-k) -0.071; joint winner L41E041 rescue +0.362 Spec +0.343.
- **S3** (keyword completion, n=128+128): L*=47, block rescue +1.547 [+1.183, +1.941]; interior L*=42 +0.439 [+0.238, +0.677] -> L42E048 rescue +0.151 [+0.094, +0.211] Spec +0.137 [+0.069, +0.206], interior joint winner L3E057 rescue +0.146 Spec +0.111; two-stage expert L47E014 (active 111/128 disc), rescue +0.773 [+0.585, +0.981], Spec +0.703 [+0.519, +0.908], rank-1 among active in 74/120; coalition (clean top-k) +1.170; joint winner L47E014 (same).
- **R1** (variable recall, n=128+128): L*=47, block rescue +0.085 [-0.231, +0.408]; interior L*=43 +0.217 [+0.099, +0.337] -> L43E126 rescue +0.147 [+0.080, +0.219] Spec +0.135 [+0.064, +0.209], interior joint winner L43E126 rescue +0.147 Spec +0.135; two-stage expert L47E060 (active 88/128 disc), rescue +0.055 [+0.004, +0.108], Spec +0.122 [+0.057, +0.190], rank-1 among active in 19/80; coalition (clean top-k) -0.306; joint winner L43E126 rescue +0.147 Spec +0.135.
- **R2** (attribute / API recall, n=128+128): L*=47, block rescue +1.995 [+1.594, +2.431]; interior L*=39 +0.443 [+0.335, +0.551] -> L39E065 rescue +0.150 [+0.073, +0.228] Spec +0.122 [+0.035, +0.207], interior joint winner L43E051 rescue +0.408 Spec +0.407; two-stage expert L47E116 (active 128/128 disc), rescue +0.427 [+0.321, +0.540], Spec +0.278 [+0.187, +0.375], rank-1 among active in 53/126; coalition (clean top-k) +1.804; joint winner L47E116 (same).
- **R3** (constant recall, n=128+128): L*=47, block rescue +0.860 [+0.535, +1.191]; interior L*=42 +0.102 [+0.024, +0.181] -> L42E039 rescue -0.009 [-0.028, +0.010] Spec -0.033 [-0.060, -0.005], interior joint winner L40E069 rescue -0.007 Spec -0.004; two-stage expert L47E002 (active 93/128 disc), rescue +0.128 [+0.062, +0.201], Spec +0.047 [-0.021, +0.115], rank-1 among active in 27/87; coalition (clean top-k) +0.669; joint winner L47E002 (same).
- **Cross-category overlap**: mean pairwise Jaccard of the recurrent (layer, expert) sets 0.10 (within syntax 0.10, within recall 0.10, syntax-recall 0.11); of the top-10 joint pairs 0.02. Selected layers: S1 L47, S2 L47, S3 L47, R1 L47, R2 L47, R3 L47; pairs recurrent in every category: 6.
- **Factual-recall experts on code**: none of them is recurrent or rescues > 0.1 on any code category.

### Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence)

Scanned 6448 items; passing the paper filter per category: S1 876, S2 484, S3 409, R1 899, R2 606, R3 756. Sets with fewer than 256 passing items use all passing items (marked partial; recurrence threshold = half the discovery split).

**Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): per-category localisation on CodeFact (validation split; recurrence threshold 64 of 128)**

| Set | n (disc+val) | top-1 = true | mean Δ_clean / drop | L* | block rescue (val) | 2nd layer | e* (two-stage) | active disc / val | expert rescue (val) | Spec (active-random) | rank-1 among active | coalition clean / union | joint winner (all layers) | recurrent pairs (all layers) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 128+128 | 8 % | +16.59 / +5.50 | L47 | +2.270 [+1.932, +2.631] | L46 +0.844 | L47E025 | 126/128 / 121/128 | +1.028 [+0.827, +1.246] | +0.869 [+0.678, +1.079] | 84/121 | +2.256 / +2.288 | L47E025 (= two-stage) | 108 |
| S2 block keyword | 128+128 | 94 % | +13.33 / +3.30 | L47 | +2.032 [+1.653, +2.432] | L42 +0.851 | L47E077 | 72/128 / 67/128 | +0.137 [+0.091, +0.187] | +0.039 [-0.013, +0.092] | 29/67 | +0.780 / +1.992 | L41E041 +0.513 [+0.370, +0.664], Spec +0.514 | 253 |
| S3 keyword completion | 128+128 | 82 % | +15.14 / +4.36 | L47 | +2.709 [+2.290, +3.135] | L45 +0.473 | L47E014 | 116/128 / 115/128 | +0.985 [+0.758, +1.214] | +0.778 [+0.566, +0.998] | 69/115 | +2.733 / +2.656 | L47E014 (= two-stage) | 102 |
| R1 variable recall | 128+128 | 90 % | +17.00 / +6.10 | L43 | +0.141 [-0.004, +0.283] | L44 +0.251 | L43E126 | 80/128 / 85/128 | +0.160 [+0.083, +0.239] | +0.160 [+0.079, +0.245] | 49/85 | +0.141 / +0.120 | L43E126 (= two-stage) | 97 |
| R2 attribute / API recall | 128+128 | 93 % | +16.14 / +4.50 | L47 | +2.248 [+1.839, +2.711] | L39 +0.600 | L47E005 | 110/128 / 101/128 | +0.413 [+0.316, +0.513] | +0.195 [+0.098, +0.298] | 54/101 | +2.219 / +2.180 | L47E005 (= two-stage) | 176 |
| R3 constant recall | 128+128 | 90 % | +17.15 / +4.72 | L47 | +1.033 [+0.588, +1.471] | L41 +0.203 | L47E101 | 83/128 / 84/128 | +0.094 [+0.002, +0.186] | -0.016 [-0.108, +0.078] | 25/84 | +1.210 / +1.066 | L47E101 (= two-stage) | 64 |
| all mixed | 128+128 | 76 % | +15.41 / +4.53 | L47 | +1.740 [+1.304, +2.201] | L42 +0.349 | none (max activity 45/128 < 64) |  |  |  |  | +1.396 / +1.569 | L4E036 +0.034 [+0.001, +0.071], Spec -0.007 | 22 |

![layer curves coder_chat](figures/ext4_curves_coder_chat.png)

**Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): the same selection restricted to interior layers (excluding the last 4 MoE blocks, whose patch acts as a read-out on code)**

| Set | last-layer block rescue (val) | L* interior (≤ L−5) | block rescue (val) | e* interior | active disc / val | expert rescue (val) | Spec (active-random) | rank-1 among active | coalition clean / union | joint winner (interior layers) |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | L47: +2.270 [+1.932, +2.631] | L41 | +0.784 [+0.604, +0.985] | L41E108 | 122/128 / 124/128 | +0.716 [+0.548, +0.909] | +0.711 [+0.541, +0.907] | 89/124 | +0.765 / +0.791 | L41E108 +0.716 [+0.548, +0.909], Spec +0.711 [+0.541, +0.907] (rank 3 over all layers) |
| S2 | L47: +2.032 [+1.653, +2.432] | L42 | +0.851 [+0.671, +1.033] | L42E006 | 128/128 / 126/128 | +0.203 [+0.134, +0.274] | +0.136 [+0.072, +0.205] | 40/126 | +0.654 / +0.821 | L41E041 +0.513 [+0.370, +0.664], Spec +0.514 [+0.373, +0.663] (rank 1 over all layers) |
| S3 | L47: +2.709 [+2.290, +3.135] | L42 | +0.368 [+0.229, +0.514] | L42E048 | 75/128 / 65/128 | +0.204 [+0.144, +0.273] | +0.216 [+0.150, +0.289] | 43/65 | +0.318 / +0.357 | L42E048 +0.204 [+0.144, +0.273], Spec +0.216 [+0.150, +0.289] (rank 4 over all layers) |
| R1 | L47: +0.358 [+0.024, +0.701] | L43 | +0.141 [-0.004, +0.283] | L43E126 | 80/128 / 85/128 | +0.160 [+0.083, +0.239] | +0.160 [+0.079, +0.245] | 49/85 | +0.141 / +0.120 | L43E126 +0.160 [+0.083, +0.239], Spec +0.160 [+0.079, +0.245] (rank 1 over all layers) |
| R2 | L47: +2.248 [+1.839, +2.711] | L39 | +0.600 [+0.437, +0.785] | L39E065 | 114/128 / 112/128 | +0.259 [+0.157, +0.370] | +0.238 [+0.135, +0.348] | 53/112 | +0.459 / +0.564 | L43E051 +0.396 [+0.263, +0.545], Spec +0.385 [+0.256, +0.529] (rank 3 over all layers) |
| R3 | L47: +1.033 [+0.588, +1.471] | L42 | +0.195 [+0.066, +0.330] | L42E021 | 74/128 / 76/128 | +0.008 [-0.024, +0.040] | -0.023 [-0.074, +0.026] | 16/76 | +0.143 / +0.236 | L2E077 -0.005 [-0.058, +0.051], Spec -0.005 [-0.051, +0.042] (rank 4 over all layers) |
| all | L47: +1.740 [+1.304, +2.201] | L41 | +0.263 [+0.117, +0.413] | L41E023 | 109/128 / 114/128 | +0.027 [-0.016, +0.073] | +0.001 [-0.046, +0.050] | 25/114 | +0.254 / +0.230 | L4E036 +0.034 [+0.001, +0.071], Spec -0.007 [-0.044, +0.032] (rank 1 over all layers) |

**Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): the factual-recall experts (L44E069 / L42E115 (Qwen3-Base factual experts)) on the code categories**

| Set | factual expert | disc active | recurrent (≥ threshold) | joint rank | val active | val rescue | val Spec |
|---|---|---|---|---|---|---|---|
| S1 | L44E069 | 3/128 | no | - | 2/128 | -0.000 [-0.001, +0.000] | -0.043 [-0.069, -0.018] |
| S1 | L42E115 | 1/128 | no | - | 0/128 | 0 (never active) | n/a |
| S2 | L44E069 | 5/128 | no | - | 4/128 | -0.003 [-0.008, +0.000] | -0.004 [-0.028, +0.020] |
| S2 | L42E115 | 4/128 | no | - | 6/128 | +0.001 [-0.006, +0.009] | -0.071 [-0.115, -0.030] |
| S3 | L44E069 | 1/128 | no | - | 5/128 | +0.003 [+0.000, +0.007] | -0.031 [-0.054, -0.008] |
| S3 | L42E115 | 2/128 | no | - | 5/128 | +0.000 [-0.004, +0.004] | -0.012 [-0.055, +0.031] |
| R1 | L44E069 | 2/128 | no | - | 2/128 | -0.001 [-0.004, +0.000] | -0.040 [-0.096, +0.014] |
| R1 | L42E115 | 2/128 | no | - | 1/128 | -0.003 [-0.009, +0.000] | -0.017 [-0.054, +0.017] |
| R2 | L44E069 | 12/128 | no | - | 9/128 | +0.002 [-0.002, +0.009] | -0.006 [-0.031, +0.020] |
| R2 | L42E115 | 0/128 | no | - | 0/128 | 0 (never active) | n/a |
| R3 | L44E069 | 4/128 | no | - | 6/128 | +0.003 [+0.000, +0.009] | -0.011 [-0.050, +0.026] |
| R3 | L42E115 | 8/128 | no | - | 4/128 | +0.000 [-0.008, +0.011] | -0.023 [-0.067, +0.018] |
| all | L44E069 | 4/128 | no | - | 2/128 | +0.002 [+0.000, +0.006] | -0.008 [-0.043, +0.023] |
| all | L42E115 | 5/128 | no | - | 3/128 | +0.000 [-0.003, +0.003] | -0.022 [-0.056, +0.013] |

**Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): selected layers, experts and recurrent sets per category**

| Set | L* (paper rule) | two-stage e* | joint winner (all layers) | L* interior | two-stage e* interior | joint winner (interior) | recurrent pairs (all layers) | recurrent pairs (interior) | recurrent experts at L* |
|---|---|---|---|---|---|---|---|---|---|
| S1 | L47 | L47E025 | L47E025 | L41 | L41E108 | L41E108 | 108 | 89 | E014, E025, E122 |
| S2 | L47 | L47E077 | L41E041 | L42 | L42E006 | L41E041 | 253 | 237 | E045, E052, E065, E077, E108 |
| S3 | L47 | L47E014 | L47E014 | L42 | L42E048 | L42E048 | 102 | 90 | E011, E012, E014, E025 |
| R1 | L43 | L43E126 | L43E126 | L43 | L43E126 | L43E126 | 97 | 78 | E045, E067, E086, E126 |
| R2 | L47 | L47E005 | L47E005 | L39 | L39E065 | L43E051 | 176 | 160 | E005, E016, E025, E052, E098, E116 |
| R3 | L47 | L47E101 | L47E101 | L42 | L42E021 | L2E077 | 64 | 61 | E002, E101, E125 |
| all | L47 | none | L4E036 | L41 | L41E023 | L4E036 | 22 | 19 | - |

**Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): Jaccard overlap of the recurrent (layer, expert) pairs (all layers, discovery activity ≥ threshold) between categories**

| recurrent pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.05 | 0.17 | 0.13 | 0.19 | 0.06 | 0.13 |
| S2 | 0.05 | 1.00 | 0.03 | 0.09 | 0.09 | 0.05 | 0.07 |
| S3 | 0.17 | 0.03 | 1.00 | 0.06 | 0.19 | 0.04 | 0.11 |
| R1 | 0.13 | 0.09 | 0.06 | 1.00 | 0.19 | 0.09 | 0.20 |
| R2 | 0.19 | 0.09 | 0.19 | 0.19 | 1.00 | 0.04 | 0.10 |
| R3 | 0.06 | 0.05 | 0.04 | 0.09 | 0.04 | 1.00 | 0.15 |
| all | 0.13 | 0.07 | 0.11 | 0.20 | 0.10 | 0.15 | 1.00 |

**Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): Jaccard overlap of the recurrent (layer, expert) pairs restricted to interior layers (≤ L−5)**

| recurrent pairs, interior layers: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.04 | 0.15 | 0.14 | 0.19 | 0.07 | 0.12 |
| S2 | 0.04 | 1.00 | 0.03 | 0.09 | 0.08 | 0.05 | 0.06 |
| S3 | 0.15 | 0.03 | 1.00 | 0.06 | 0.20 | 0.04 | 0.10 |
| R1 | 0.14 | 0.09 | 0.06 | 1.00 | 0.20 | 0.10 | 0.21 |
| R2 | 0.19 | 0.08 | 0.20 | 0.20 | 1.00 | 0.05 | 0.09 |
| R3 | 0.07 | 0.05 | 0.04 | 0.10 | 0.05 | 1.00 | 0.16 |
| all | 0.12 | 0.06 | 0.10 | 0.21 | 0.09 | 0.16 | 1.00 |

**Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): Jaccard overlap of the top-10 joint (layer, expert) pairs between categories**

| top-10 joint pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.00 | 0.18 | 0.00 | 0.05 | 0.00 | 0.00 |
| S2 | 0.00 | 1.00 | 0.05 | 0.00 | 0.00 | 0.00 | 0.05 |
| S3 | 0.18 | 0.05 | 1.00 | 0.00 | 0.05 | 0.00 | 0.05 |
| R1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| R2 | 0.05 | 0.00 | 0.05 | 0.00 | 1.00 | 0.00 | 0.05 |
| R3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| all | 0.00 | 0.05 | 0.05 | 0.00 | 0.05 | 0.00 | 1.00 |

**Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): Jaccard overlap of the top-10 joint pairs restricted to interior layers**

| top-10 interior joint pairs: Jaccard | S1 | S2 | S3 | R1 | R2 | R3 | all |
|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 0.00 | 0.05 | 0.00 | 0.18 | 0.00 | 0.05 |
| S2 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| S3 | 0.05 | 0.00 | 1.00 | 0.00 | 0.05 | 0.00 | 0.05 |
| R1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.05 | 0.00 | 0.05 |
| R2 | 0.18 | 0.00 | 0.05 | 0.05 | 1.00 | 0.00 | 0.05 |
| R3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| all | 0.05 | 0.00 | 0.05 | 0.05 | 0.05 | 0.00 | 1.00 |

**Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): (layer, expert) pairs recurrent in ≥ 2 categories**

| pair | n categories | categories |
|---|---|---|
| L2E026 | 6 | S1, S2, S3, R1, R2, R3 |
| L4E084 | 6 | S1, S2, S3, R1, R2, R3 |
| L15E031 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L29E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E025 | 5 | S2, S3, R1, R2, R3 |
| L26E107 | 5 | S1, S2, R1, R2, R3 |
| L41E023 | 5 | S1, S2, S3, R1, R2 |
| L44E091 | 5 | S1, S2, S3, R1, R2 |
| L45E014 | 5 | S1, S2, S3, R1, R2 |
| L8E007 | 4 | S1, S3, R1, R2 |
| L20E038 | 4 | S1, S2, R1, R3 |
| L40E038 | 4 | S1, S2, R1, R2 |
| L43E086 | 4 | S1, S2, R1, R2 |
| L45E001 | 4 | S1, S2, R1, R2 |
| L2E096 | 3 | S1, S3, R2 |
| L4E036 | 3 | S1, S3, R2 |
| L8E038 | 3 | S2, R1, R3 |
| L9E003 | 3 | S1, R2, R3 |
| L9E075 | 3 | S2, S3, R2 |
| L12E082 | 3 | S1, S2, R1 |
| L14E005 | 3 | S2, R1, R3 |
| L15E065 | 3 | S1, S3, R2 |
| L16E038 | 3 | S2, R1, R2 |
| L16E115 | 3 | S2, R1, R3 |
| L21E000 | 3 | S2, R1, R3 |
| L22E079 | 3 | S2, R1, R2 |
| L22E106 | 3 | S1, S3, R2 |
| L23E035 | 3 | S1, S3, R2 |
| L24E082 | 3 | S1, S2, R1 |
| L25E046 | 3 | S1, S2, R2 |
| L27E031 | 3 | S2, R1, R2 |
| L28E038 | 3 | S2, R1, R2 |
| L29E025 | 3 | S2, R1, R2 |
| L31E055 | 3 | S1, S3, R2 |
| L31E057 | 3 | S1, R1, R2 |
| L33E003 | 3 | S1, R1, R2 |
| L33E031 | 3 | S1, S3, R2 |
| L34E044 | 3 | S2, R1, R2 |
| L34E079 | 3 | S2, R1, R2 |

**Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): interior (layer, expert) pairs recurrent in ≥ 2 categories**

| pair (interior) | n categories | categories |
|---|---|---|
| L2E026 | 6 | S1, S2, S3, R1, R2, R3 |
| L4E084 | 6 | S1, S2, S3, R1, R2, R3 |
| L15E031 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L29E023 | 6 | S1, S2, S3, R1, R2, R3 |
| L17E025 | 5 | S2, S3, R1, R2, R3 |
| L26E107 | 5 | S1, S2, R1, R2, R3 |
| L41E023 | 5 | S1, S2, S3, R1, R2 |
| L8E007 | 4 | S1, S3, R1, R2 |
| L20E038 | 4 | S1, S2, R1, R3 |
| L40E038 | 4 | S1, S2, R1, R2 |
| L43E086 | 4 | S1, S2, R1, R2 |
| L2E096 | 3 | S1, S3, R2 |
| L4E036 | 3 | S1, S3, R2 |
| L8E038 | 3 | S2, R1, R3 |
| L9E003 | 3 | S1, R2, R3 |
| L9E075 | 3 | S2, S3, R2 |
| L12E082 | 3 | S1, S2, R1 |
| L14E005 | 3 | S2, R1, R3 |
| L15E065 | 3 | S1, S3, R2 |
| L16E038 | 3 | S2, R1, R2 |
| L16E115 | 3 | S2, R1, R3 |
| L21E000 | 3 | S2, R1, R3 |
| L22E079 | 3 | S2, R1, R2 |
| L22E106 | 3 | S1, S3, R2 |
| L23E035 | 3 | S1, S3, R2 |
| L24E082 | 3 | S1, S2, R1 |
| L25E046 | 3 | S1, S2, R2 |
| L27E031 | 3 | S2, R1, R2 |
| L28E038 | 3 | S2, R1, R2 |
| L29E025 | 3 | S2, R1, R2 |
| L31E055 | 3 | S1, S3, R2 |
| L31E057 | 3 | S1, R1, R2 |
| L33E003 | 3 | S1, R1, R2 |
| L33E031 | 3 | S1, S3, R2 |
| L34E044 | 3 | S2, R1, R2 |
| L34E079 | 3 | S2, R1, R2 |
| L34E106 | 3 | S1, S3, R2 |
| L35E035 | 3 | S1, S3, R2 |
| L35E091 | 3 | S2, R1, R2 |

![expert overlap coder_chat](figures/ext4_overlap_coder_chat.png)

- **S1** (closing bracket, n=128+128): L*=47, block rescue +2.270 [+1.932, +2.631]; interior L*=41 +0.784 [+0.604, +0.985] -> L41E108 rescue +0.716 [+0.548, +0.909] Spec +0.711 [+0.541, +0.907], interior joint winner L41E108 rescue +0.716 Spec +0.711; two-stage expert L47E025 (active 126/128 disc), rescue +1.028 [+0.827, +1.246], Spec +0.869 [+0.678, +1.079], rank-1 among active in 84/121; coalition (clean top-k) +2.256; joint winner L47E025 (same).
- **S2** (block keyword, n=128+128): L*=47, block rescue +2.032 [+1.653, +2.432]; interior L*=42 +0.851 [+0.671, +1.033] -> L42E006 rescue +0.203 [+0.134, +0.274] Spec +0.136 [+0.072, +0.205], interior joint winner L41E041 rescue +0.513 Spec +0.514; two-stage expert L47E077 (active 72/128 disc), rescue +0.137 [+0.091, +0.187], Spec +0.039 [-0.013, +0.092], rank-1 among active in 29/67; coalition (clean top-k) +0.780; joint winner L41E041 rescue +0.513 Spec +0.514.
- **S3** (keyword completion, n=128+128): L*=47, block rescue +2.709 [+2.290, +3.135]; interior L*=42 +0.368 [+0.229, +0.514] -> L42E048 rescue +0.204 [+0.144, +0.273] Spec +0.216 [+0.150, +0.289], interior joint winner L42E048 rescue +0.204 Spec +0.216; two-stage expert L47E014 (active 116/128 disc), rescue +0.985 [+0.758, +1.214], Spec +0.778 [+0.566, +0.998], rank-1 among active in 69/115; coalition (clean top-k) +2.733; joint winner L47E014 (same).
- **R1** (variable recall, n=128+128): L*=43, block rescue +0.141 [-0.004, +0.283]; interior L*=43 +0.141 [-0.004, +0.283] -> L43E126 rescue +0.160 [+0.083, +0.239] Spec +0.160 [+0.079, +0.245], interior joint winner L43E126 rescue +0.160 Spec +0.160; two-stage expert L43E126 (active 80/128 disc), rescue +0.160 [+0.083, +0.239], Spec +0.160 [+0.079, +0.245], rank-1 among active in 49/85; coalition (clean top-k) +0.141; joint winner L43E126 (same).
- **R2** (attribute / API recall, n=128+128): L*=47, block rescue +2.248 [+1.839, +2.711]; interior L*=39 +0.600 [+0.437, +0.785] -> L39E065 rescue +0.259 [+0.157, +0.370] Spec +0.238 [+0.135, +0.348], interior joint winner L43E051 rescue +0.396 Spec +0.385; two-stage expert L47E005 (active 110/128 disc), rescue +0.413 [+0.316, +0.513], Spec +0.195 [+0.098, +0.298], rank-1 among active in 54/101; coalition (clean top-k) +2.219; joint winner L47E005 (same).
- **R3** (constant recall, n=128+128): L*=47, block rescue +1.033 [+0.588, +1.471]; interior L*=42 +0.195 [+0.066, +0.330] -> L42E021 rescue +0.008 [-0.024, +0.040] Spec -0.023 [-0.074, +0.026], interior joint winner L2E077 rescue -0.005 Spec -0.005; two-stage expert L47E101 (active 83/128 disc), rescue +0.094 [+0.002, +0.186], Spec -0.016 [-0.108, +0.078], rank-1 among active in 25/84; coalition (clean top-k) +1.210; joint winner L47E101 (same).
- **Cross-category overlap**: mean pairwise Jaccard of the recurrent (layer, expert) sets 0.10 (within syntax 0.08, within recall 0.11, syntax-recall 0.10); of the top-10 joint pairs 0.02. Selected layers: S1 L47, S2 L47, S3 L47, R1 L43, R2 L47, R3 L47; pairs recurrent in every category: 5.
- **Factual-recall experts on code**: none of them is recurrent or rescues > 0.1 on any code category.

### Summary across models and protocols

**All runs: layer L* and MoE-block validation rescue -> two-stage expert with validation rescue / active-random Spec; then the same restricted to interior layers (≤ L−5)**

| Set | Qwen3-30B-A3B-Base (tokenizer defaults) | Mixtral-8x7B-v0.1 (no BOS, paper protocol) | Qwen3-Coder-30B-A3B-Instruct (raw code prefix) | Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence) |
|---|---|---|---|---|
| S1 closing bracket | L47 +2.420 -> L47E025 +1.153 / Spec +0.973; interior L42 +1.261 -> L42E048 +0.916 / Spec +0.873 | L31 +1.886 -> L31E000 +1.659 / Spec +1.443; interior L0 +0.362 -> L0E005 +0.185 / Spec +0.064 | L47 +1.869 -> L47E025 +1.208 / Spec +1.113; interior L41 +0.785 -> L41E108 +0.670 / Spec +0.664 | L47 +2.270 -> L47E025 +1.028 / Spec +0.869; interior L41 +0.784 -> L41E108 +0.716 / Spec +0.711 |
| S2 block keyword | L47 +0.642 -> L47E062 +0.336 / Spec +0.343; interior L41 +0.488 -> L41E041 +0.551 / Spec +0.549 | L17 +0.406 -> L17E003 +0.356 / Spec +0.322; interior L17 +0.406 -> L17E003 +0.356 / Spec +0.322 | L47 +0.644 -> L47E077 +0.055 / Spec +0.057; interior L39 +0.338 -> L39E092 +0.224 / Spec +0.202 | L47 +2.032 -> L47E077 +0.137 / Spec +0.039; interior L42 +0.851 -> L42E006 +0.203 / Spec +0.136 |
| S3 keyword completion | L47 +1.054 -> L47E025 +0.188 / Spec +0.154; interior L42 +0.441 -> L42E044 +0.125 / Spec +0.090 | L31 +0.582 -> L31E000 +0.361 / Spec +0.263; interior L19 +0.260 -> L19E005 +0.147 / Spec +0.057 (partial n=120+121) | L47 +1.547 -> L47E014 +0.773 / Spec +0.703; interior L42 +0.439 -> L42E048 +0.151 / Spec +0.137 | L47 +2.709 -> L47E014 +0.985 / Spec +0.778; interior L42 +0.368 -> L42E048 +0.204 / Spec +0.216 |
| R1 variable recall | L43 +0.391 -> L43E126 +0.326 / Spec +0.316; interior L43 +0.391 -> L43E126 +0.326 / Spec +0.316 | L31 +0.455 -> L31E006 +0.028 / Spec -0.242; interior L27 +0.120 -> L27E006 +0.031 / Spec -0.005 | L47 +0.085 -> L47E060 +0.055 / Spec +0.122; interior L43 +0.217 -> L43E126 +0.147 / Spec +0.135 | L43 +0.141 -> L43E126 +0.160 / Spec +0.160; interior L43 +0.141 -> L43E126 +0.160 / Spec +0.160 |
| R2 attribute / API recall | L47 +0.935 -> L47E005 +0.118 / Spec +0.042; interior L43 +0.353 -> L43E051 +0.230 / Spec +0.234 | L31 +0.458 -> L31E005 +0.312 / Spec +0.496; interior L20 +0.190 -> L20E006 +0.095 / Spec +0.065 | L47 +1.995 -> L47E116 +0.427 / Spec +0.278; interior L39 +0.443 -> L39E065 +0.150 / Spec +0.122 | L47 +2.248 -> L47E005 +0.413 / Spec +0.195; interior L39 +0.600 -> L39E065 +0.259 / Spec +0.238 |
| R3 constant recall | L47 +0.828 -> L47E101 +0.117 / Spec +0.079; interior L43 +0.216 -> L43E126 +0.070 / Spec +0.055 | L31 +0.607 -> L31E005 +0.264 / Spec +0.155; interior L20 +0.123 -> L20E000 +0.051 / Spec -0.001 | L47 +0.860 -> L47E002 +0.128 / Spec +0.047; interior L42 +0.102 -> L42E039 -0.009 / Spec -0.033 | L47 +1.033 -> L47E101 +0.094 / Spec -0.016; interior L42 +0.195 -> L42E021 +0.008 / Spec -0.023 |
| all mixed | L47 +0.962 -> no recurrent expert; interior L41 +0.381 -> L41E023 -0.000 / Spec -0.039 | L31 +0.744 -> no recurrent expert; interior L19 +0.168 | L47 +1.041 -> no recurrent expert; interior L43 +0.228 -> L43E086 -0.012 / Spec -0.060 | L47 +1.740 -> no recurrent expert; interior L41 +0.263 -> L41E023 +0.027 / Spec +0.001 |


### Findings

**1. The paper's filter separates "recall-like" from "redundantly determined" code categories, and S1 is on the recall side.** Under the paper's absolute thresholds (Δ_clean ≥ 1, drop ≥ 0.5) the pass rates on Qwen3-30B-A3B-Base are S1 90 %, S2 33 %, S3 36 %, R1 79 %, R2 57 %, R3 59 % (Mixtral, no BOS: 86 / 40 / 30 / 47 / 48 / 40 %). The clean margins are large everywhere (median Δ_clean +3 to +14) but the subject-noise drop is tiny for block keywords and keyword completion (median drop +0.25 and +0.12 in Qwen3): `else`, ` in` and `:` are determined by the whole context, so destroying the `if`/`for` embedding barely moves them, and the paper's method has (correctly) little to trace. Closing brackets are the exception among the syntax categories: noising the opener alone moves the closer by 3-6 logits (median drop +4.4 / +2.9), i.e. the bracket type is read from one token, exactly like a fact from its subject. The relative rule (drop ≥ 25 % of Δ_clean) passes far fewer items (62 / 15 / 20 / 50 / 24 / 29 % in Qwen3) because code margins are 5-15 logits; it changes the pass counts, not the ordering of the categories (appendix). Top-1 agreement is high for the recall categories (R1 89 %, R2 90 %, R3 85 %) and low for S1 (18 %) only because the model prefers merged tokens such as `):` or `)\n`; counting predictions that start with the true string gives 87 %.

**2. On code the last MoE block acts as a read-out, so the paper's layer rule selects the final layer.** The MoE-block patch at the last layer (L47 in Qwen3, L31 in Mixtral) has the largest validation rescue for every category except R1 in Qwen3 (L43) and S2 in Mixtral (L17): S1 +2.42 / +1.89, mixed set +0.96 / +0.74 (Qwen3 / Mixtral). On CounterFact the same patch at the last layer rescues nothing (−0.13 Qwen3, −0.08 Mixtral, paper set) and the curve peaks at L44 / L19. The code curves also have an interior peak in a narrow band shared by all categories: Qwen3 L41-43 (S1 +1.26 at L42, S2 +0.49 at L41, R1 +0.39 at L43), Mixtral L17-22 (S2 +0.41 at L17, R2 +0.21 at L18, S3 +0.26 at L22), the band in which Direction 1 found the E001 experts. We therefore report both the paper's rule and the same rule restricted to interior layers (≤ L−5). The final-layer experts that the paper's rule selects are determined by the token class at the final position, not by the category: in Qwen3 the S1, S3 and R2 prefixes end in an identifier and are routed at L47 to {E025, E050, E005, E016, E116} (S1 → L47E025, S3 → L47E025, R2 → L47E005), the S2 prefixes end in indentation whitespace and are routed to {E077, E060, E033, E062} (→ L47E062), the R3 prefixes end in a quote and go to {E002, E101, E060, E125} (→ L47E101). The mixed set, whose final tokens are of all classes, has no expert above the 64/128 recurrence threshold at the last layer in either model (max activity 63 and 60 of 128).

**3. Per category (validation split; two-stage expert at the paper's L*, interior selection in brackets).**
- *S1 closing bracket* is the one code category that localises like a fact, in both models: Qwen3 L47E025 rescue +1.15 [+0.95, +1.37], Spec +0.97 [+0.78, +1.18], active 119/128, rank-1 among the eight active experts in 65/124 cases; Mixtral L31E000 active in all 128 discovery cases, rescue +1.66 [+1.41, +1.92], Spec +1.44 [+1.20, +1.70], rank-1 in 115/128, 88 % of the block rescue. Qwen3 also has an interior bracket expert, L42E048 (rescue +0.92, Spec +0.87, active 125/128; joint runner-up L43E084 +0.94 / +0.93), so the bracket information is carried by a single expert in two layers (cf. L44E069 and L42E115 for facts). Mixtral has no interior bracket expert (best L0E005 +0.19, Spec +0.06).
- *S2 block keyword*: Qwen3's paper-rule expert L47E062 (+0.34, Spec +0.34) is beaten by the interior expert L41E041 (rescue +0.55 [+0.45, +0.65], Spec +0.55, active 128/128, joint winner over all layers). Mixtral selects L17E003 (+0.36 [+0.28, +0.45], Spec +0.32, active 128/128) directly, its only non-final L*. The CounterFact BOS-run expert L19E002 is recurrent on S2 (128/128; +0.32, Spec +0.31, joint rank 3) and the paper's L19E006 is active in 73/128 with Spec −0.27: with the final position on indentation whitespace, E002 is the L19 expert that carries content and E006 again the one that does not.
- *S3 keyword completion*: weak everywhere (Qwen3 L47E025 +0.19, Spec +0.15; interior L42E044 +0.13; Mixtral L31E000 +0.36, Spec +0.26, joint winner L18E002 +0.19).
- *R1 variable recall* is the category closest to CounterFact in Qwen3: L* = L43 (not the last layer), a single expert L43E126 with rescue +0.33 [+0.25, +0.41], Spec +0.32 [+0.24, +0.40], active 121/128, rank-1 in 87/118 and 83 % of the block rescue, the joint winner over all layers. In Mixtral the paper's rule gives L31E006 with rescue +0.03 and *negative* specificity (−0.24 [−0.44, −0.06]) while the coalition rescues +0.30: variable recall in Mixtral is a coalition effect, and E006 — the paper's factual expert index at L19 — is again the negatively specific expert, now at L31 (the sink diagnostic is irrelevant here: no code prompt has a sink-carrying final token, 0/1,521). The interior selection (L27E006, +0.03) finds nothing either.
- *R2 attribute / API recall*: Qwen3 L47E005 +0.12 with Spec ≈ 0 (+0.04 [−0.03, +0.11]); interior L43E051 +0.23, Spec +0.23 (rank-1 in 74/126). Mixtral L31E005 +0.31, Spec +0.50 [+0.39, +0.60] on the 70/128 cases where it is active (the coalition of the two active experts rescues −0.01: the whole effect is E005).
- *R3 constant recall*: Qwen3 L47E101 +0.12, Spec +0.08; interior L43E126 (the R1 variable-recall expert) +0.07. Mixtral L31E005 +0.26, Spec +0.16.

**4. Across categories the selected experts do not overlap; the recurrent sets do, through always-on experts.** The top-10 joint (layer, expert) pairs are almost disjoint between categories (mean pairwise Jaccard 0.05 in Qwen3, 0.04 in Mixtral; the only overlaps are the shared final-layer identifier experts S1-S3-R2 in Qwen3 and R1-R3 in Mixtral). The recurrent sets (activity ≥ 64/128 at any layer) overlap moderately in Qwen3 (Jaccard 0.08-0.47; 10 pairs such as L2E026, L17E023, L41E023, L43E086 are recurrent in all six categories and none of them rescues anything) and barely in Mixtral (3 pairs recurrent on the mixed set). Syntax and recall categories do not form two blocks: S1 is as close to R1/R2 (Jaccard 0.37 / 0.38) as to S3 (0.39) and far from S2 (0.14). Every category has its own layer × expert locus; what is shared is the final-layer read-out and the interior band.

**5. The factual-recall experts play no role on code.** Qwen3's L44E069 is clean-active in 0-4 of 128 discovery cases in every category (L42E115 in 0-17) and rescues nothing (|rescue| ≤ 0.01). Mixtral's L18E001 is inactive on code (0-5/128) and L19E006 / L19E002 are recurrent only on S2 (above), where E002 rescues (+0.32) and E006 does not.

**6. Syntax vs recall.** The plan's hypothesis was "R categories behave like CounterFact, S categories show little subject-noise sensitivity and diffuse rescue". Half of it holds: S2 and S3 pass the filter rarely and their experts are weak (Spec ≤ 0.35), and R1 in Qwen3 localises to a mid-late single expert like a fact. The other half does not: S1 is the most localised category of all (a bracket expert with Spec +1.0 / +1.4), and R2 / R3 in Qwen3 are weak and diffuse (Spec ≤ 0.08 at L*, +0.23 interior). The determinant is not syntax vs semantics but whether the answer is read from one token of the context (the opener, the definition site) or from many.

**7. Coder-Instruct vs Base on the same items.** The 6,448 Qwen3-valid items were scanned with Qwen3-Coder-30B-A3B-Instruct under both protocols (raw code prefix; chat template with the user turn "Complete the following Python code." and the prefix inside a ```python fence in the open assistant turn, 17 template tokens). The Coder model sees the items the way the base model does: per-item Δ_clean correlates at r = 0.83-0.92 (raw) / 0.73-0.89 (chat) with the base model's, the drop at 0.37-0.65, and the pass rates are within a few points (raw: S1 84 %, S2 33 %, S3 40 %, R1 79 %, R2 73 %, R3 60 %; chat: 91 / 40 / 34 / 79 / 76 / 65 %); 219-816 items per category pass in both models. The localisation transfers where the base model localises: **S1 selects the same bracket expert L47E025 in all three runs** (rescue +1.15 base, +1.21 Coder raw, +1.03 Coder chat; Spec +0.97 / +1.11 / +0.87), **R1 selects L43E126** in the base and the chat run (+0.33 / +0.16, Spec +0.32 / +0.16) and L43E126 is still the all-layer joint winner in the raw run, where the paper's rule moves L* to the (uninformative) last layer (block +0.09 [−0.23, +0.41]); R2 and R3 select the base experts L47E005 / L47E101 in the chat run and neighbours (L47E116, L47E002) in the raw run, all weak except L47E005 / L47E116 on R2 in the Coder model (+0.41 / +0.43, Spec +0.20 / +0.28 vs +0.12 in the base). Two categories change: S3 gains a strong final-layer expert in the Coder model (L47E014, rescue +0.77 raw / +0.99 chat, Spec +0.70 / +0.78, vs L47E025 +0.19 in the base), and S2's paper-rule expert is weak in the Coder model (L47E077 +0.06 / +0.14) while the interior expert **L41E041 is the joint winner in all three runs** (+0.55 base, +0.36 raw, +0.51 chat). The interior bracket expert differs: L42E048 in the base, L41E108 in both Coder runs (+0.67 / +0.72, Spec +0.66 / +0.71). Protocol matters for the block, not for the experts: the chat template raises the last-layer block rescue (S2 +2.03 vs +0.64, S3 +2.71 vs +1.55, mixed +1.74 vs +1.04) because the fenced assistant turn makes the model far more confident, but the selected experts are the same under both protocols for S1, S2, S3 and R2/R3 up to neighbours, and the factual experts stay inactive (L44E069 / L42E115 clean-active in ≤ 7 of 128 cases in every Coder category). Code post-training therefore re-uses the base model's code experts (bracket L47E025, variable-recall L43E126, keyword L41E041) and adds or strengthens final-layer experts for keyword completion and attribute recall.

### Coder-Instruct vs Base on the same items

**Qwen3-Coder-30B-A3B-Instruct vs Qwen3-30B-A3B-Base per category (each model on its own passing items)**

| Set | Base L* / block | Base e* | Base rescue | Base Spec | Coder raw L* / block | Coder raw e* | Coder raw rescue | Coder raw Spec | Coder chat L* / block | Coder chat e* | Coder chat rescue | Coder chat Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | L47 +2.420 [+2.069, +2.803] | L47E025 | +1.153 [+0.951, +1.368] | +0.973 [+0.784, +1.175] | L47 +1.869 [+1.563, +2.206] | L47E025 | +1.208 [+0.973, +1.473] | +1.113 [+0.884, +1.370] | L47 +2.270 [+1.932, +2.631] | L47E025 | +1.028 [+0.827, +1.246] | +0.869 [+0.678, +1.079] |
| S2 | L47 +0.642 [+0.505, +0.780] | L47E062 | +0.336 [+0.237, +0.438] | +0.343 [+0.237, +0.449] | L47 +0.644 [+0.477, +0.841] | L47E077 | +0.055 [+0.019, +0.091] | +0.057 [+0.016, +0.098] | L47 +2.032 [+1.653, +2.432] | L47E077 | +0.137 [+0.091, +0.187] | +0.039 [-0.013, +0.092] |
| S3 | L47 +1.054 [+0.836, +1.277] | L47E025 | +0.188 [+0.143, +0.233] | +0.154 [+0.098, +0.209] | L47 +1.547 [+1.183, +1.941] | L47E014 | +0.773 [+0.585, +0.981] | +0.703 [+0.519, +0.908] | L47 +2.709 [+2.290, +3.135] | L47E014 | +0.985 [+0.758, +1.214] | +0.778 [+0.566, +0.998] |
| R1 | L43 +0.391 [+0.299, +0.490] | L43E126 | +0.326 [+0.252, +0.408] | +0.316 [+0.242, +0.400] | L47 +0.085 [-0.231, +0.408] | L47E060 | +0.055 [+0.004, +0.108] | +0.122 [+0.057, +0.190] | L43 +0.141 [-0.004, +0.283] | L43E126 | +0.160 [+0.083, +0.239] | +0.160 [+0.079, +0.245] |
| R2 | L47 +0.935 [+0.679, +1.223] | L47E005 | +0.118 [+0.040, +0.202] | +0.042 [-0.028, +0.114] | L47 +1.995 [+1.594, +2.431] | L47E116 | +0.427 [+0.321, +0.540] | +0.278 [+0.187, +0.375] | L47 +2.248 [+1.839, +2.711] | L47E005 | +0.413 [+0.316, +0.513] | +0.195 [+0.098, +0.298] |
| R3 | L47 +0.828 [+0.600, +1.056] | L47E101 | +0.117 [+0.077, +0.159] | +0.079 [+0.032, +0.128] | L47 +0.860 [+0.535, +1.191] | L47E002 | +0.128 [+0.062, +0.201] | +0.047 [-0.021, +0.115] | L47 +1.033 [+0.588, +1.471] | L47E101 | +0.094 [+0.002, +0.186] | -0.016 [-0.108, +0.078] |
| all | L47 +0.962 [+0.728, +1.215] | none | n/a | n/a | L47 +1.041 [+0.717, +1.375] | none | n/a | n/a | L47 +1.740 [+1.304, +2.201] | none | n/a | n/a |

**Items scanned by both Base and Qwen3-Coder-30B-A3B-Instruct (raw code prefix): filter agreement**

| Category | items scanned by both | pass Base | pass coder_raw | pass both | corr Δ_clean | corr drop |
|---|---|---|---|---|---|---|
| S1 | 964 | 871 | 810 | 761 | 0.83 | 0.56 |
| S2 | 1200 | 399 | 395 | 219 | 0.92 | 0.37 |
| S3 | 1195 | 430 | 474 | 308 | 0.91 | 0.63 |
| R1 | 1132 | 896 | 899 | 783 | 0.88 | 0.65 |
| R2 | 794 | 451 | 576 | 369 | 0.84 | 0.54 |
| R3 | 1163 | 690 | 696 | 512 | 0.87 | 0.61 |

Selected experts per category, Qwen3-Coder-30B-A3B-Instruct (raw code prefix): S1: Base L47E025 vs Coder L47E025 (same expert); S2: Base L47E062 vs Coder L47E077; S3: Base L47E025 vs Coder L47E014; R1: Base L43E126 vs Coder L47E060; R2: Base L47E005 vs Coder L47E116; R3: Base L47E101 vs Coder L47E002; all: Base none vs Coder none.

**Items scanned by both Base and Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): filter agreement**

| Category | items scanned by both | pass Base | pass coder_chat | pass both | corr Δ_clean | corr drop |
|---|---|---|---|---|---|---|
| S1 | 964 | 871 | 876 | 816 | 0.73 | 0.54 |
| S2 | 1200 | 399 | 484 | 274 | 0.89 | 0.45 |
| S3 | 1195 | 430 | 409 | 282 | 0.88 | 0.62 |
| R1 | 1132 | 896 | 899 | 767 | 0.85 | 0.60 |
| R2 | 794 | 451 | 606 | 382 | 0.80 | 0.52 |
| R3 | 1163 | 690 | 756 | 540 | 0.85 | 0.58 |

Selected experts per category, Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): S1: Base L47E025 vs Coder L47E025 (same expert); S2: Base L47E062 vs Coder L47E077; S3: Base L47E025 vs Coder L47E014; R1: Base L43E126 vs Coder L43E126 (same expert); R2: Base L47E005 vs Coder L47E005 (same expert); R3: Base L47E101 vs Coder L47E101 (same expert); all: Base none vs Coder none.


### Verdict

- Qwen3-30B-A3B-Base (tokenizer defaults): pass counts syntax vs recall 567 vs 679 of the scanned items per category; mean block rescue at L* +1.372 (S) vs +0.718 (R); mean two-stage expert rescue +0.559 (S) vs +0.187 (R); mean Spec +0.490 (S) vs +0.145 (R); layers S1 L47, S2 L47, S3 L47, R1 L43, R2 L47, R3 L47.
- Mixtral-8x7B-v0.1 (no BOS, paper protocol): pass counts syntax vs recall 416 vs 339 of the scanned items per category; mean block rescue at L* +0.958 (S) vs +0.507 (R); mean two-stage expert rescue +0.792 (S) vs +0.201 (R); mean Spec +0.676 (S) vs +0.136 (R); layers S1 L31, S2 L17, S3 L31, R1 L31, R2 L31, R3 L31.

### Appendix: relative threshold rule and quantiles

**qwen3 (raw): quantiles of Δ_clean and of the subject-noise drop per category**

| Category | n | quantity | 5 % | 25 % | median | 75 % | 95 % | mean |
|---|---|---|---|---|---|---|---|---|
| S1 | 964 | Δ_clean | +8.77 | +11.75 | +13.56 | +15.31 | +17.50 | +13.45 |
| S1 | 964 | drop | +0.12 | +1.75 | +4.38 | +7.56 | +12.38 | +5.04 |
| S1 | 964 | drop / Δ_clean (Δ_clean ≥ 1) | +0.01 | +0.14 | +0.32 | +0.55 | +0.92 | +0.38 |
| S2 | 1200 | Δ_clean | -1.00 | +1.25 | +3.25 | +6.12 | +9.62 | +3.75 |
| S2 | 1200 | drop | -1.00 | -0.25 | +0.25 | +0.75 | +2.50 | +0.37 |
| S2 | 940 | drop / Δ_clean (Δ_clean ≥ 1) | -0.33 | -0.05 | +0.07 | +0.20 | +0.43 | +0.06 |
| S3 | 1195 | Δ_clean | +0.00 | +2.00 | +3.75 | +6.50 | +10.75 | +4.46 |
| S3 | 1195 | drop | -1.12 | -0.50 | +0.12 | +1.12 | +4.25 | +0.62 |
| S3 | 1050 | drop / Δ_clean (Δ_clean ≥ 1) | -0.46 | -0.12 | +0.05 | +0.24 | +0.56 | +0.06 |
| R1 | 1132 | Δ_clean | +1.38 | +5.38 | +8.12 | +10.62 | +14.43 | +8.05 |
| R1 | 1132 | drop | -0.38 | +0.75 | +1.88 | +4.03 | +7.93 | +2.69 |
| R1 | 1083 | drop / Δ_clean (Δ_clean ≥ 1) | -0.05 | +0.11 | +0.26 | +0.50 | +1.15 | +0.37 |
| R2 | 794 | Δ_clean | +0.75 | +5.25 | +7.38 | +10.12 | +14.81 | +7.65 |
| R2 | 794 | drop | -0.75 | +0.00 | +0.75 | +1.81 | +5.38 | +1.27 |
| R2 | 750 | drop / Δ_clean (Δ_clean ≥ 1) | -0.14 | +0.02 | +0.11 | +0.25 | +0.57 | +0.15 |
| R3 | 1163 | Δ_clean | +0.76 | +5.00 | +7.62 | +10.38 | +16.37 | +7.84 |
| R3 | 1163 | drop | -0.88 | +0.12 | +0.88 | +2.50 | +6.12 | +1.60 |
| R3 | 1099 | drop / Δ_clean (Δ_clean ≥ 1) | -0.13 | +0.02 | +0.12 | +0.33 | +0.82 | +0.22 |

**mixtral (nobos): quantiles of Δ_clean and of the subject-noise drop per category**

| Category | n | quantity | 5 % | 25 % | median | 75 % | 95 % | mean |
|---|---|---|---|---|---|---|---|---|
| S1 | 800 | Δ_clean | +8.81 | +11.61 | +13.19 | +14.75 | +16.38 | +13.02 |
| S1 | 800 | drop | +0.00 | +1.12 | +2.91 | +4.66 | +8.44 | +3.30 |
| S1 | 800 | drop / Δ_clean (Δ_clean ≥ 1) | +0.00 | +0.09 | +0.22 | +0.35 | +0.68 | +0.26 |
| S2 | 800 | Δ_clean | -1.38 | +1.00 | +2.75 | +5.25 | +8.88 | +3.24 |
| S2 | 800 | drop | -0.75 | -0.12 | +0.25 | +1.16 | +3.88 | +0.73 |
| S2 | 607 | drop / Δ_clean (Δ_clean ≥ 1) | -0.27 | +0.00 | +0.14 | +0.33 | +0.62 | +0.16 |
| S3 | 800 | Δ_clean | +1.38 | +3.22 | +4.62 | +6.25 | +9.38 | +4.87 |
| S3 | 800 | drop | -1.12 | -0.38 | +0.00 | +0.62 | +3.00 | +0.36 |
| S3 | 774 | drop / Δ_clean (Δ_clean ≥ 1) | -0.30 | -0.10 | +0.00 | +0.13 | +0.43 | +0.04 |
| R1 | 800 | Δ_clean | +0.75 | +4.62 | +7.12 | +9.56 | +13.12 | +7.06 |
| R1 | 800 | drop | -1.00 | -0.12 | +0.44 | +1.50 | +5.00 | +1.01 |
| R1 | 756 | drop / Δ_clean (Δ_clean ≥ 1) | -0.19 | -0.02 | +0.07 | +0.21 | +0.82 | +0.16 |
| R2 | 671 | Δ_clean | -0.50 | +3.62 | +5.88 | +7.88 | +12.50 | +5.87 |
| R2 | 671 | drop | -0.62 | +0.00 | +0.50 | +1.12 | +2.88 | +0.74 |
| R2 | 598 | drop / Δ_clean (Δ_clean ≥ 1) | -0.16 | +0.02 | +0.09 | +0.19 | +0.36 | +0.10 |
| R3 | 800 | Δ_clean | +0.49 | +4.12 | +6.75 | +9.12 | +13.82 | +6.75 |
| R3 | 800 | drop | -1.12 | -0.25 | +0.25 | +1.00 | +3.50 | +0.61 |
| R3 | 744 | drop / Δ_clean (Δ_clean ≥ 1) | -0.23 | -0.03 | +0.04 | +0.17 | +0.51 | +0.09 |

**qwen3_coder (raw): quantiles of Δ_clean and of the subject-noise drop per category**

| Category | n | quantity | 5 % | 25 % | median | 75 % | 95 % | mean |
|---|---|---|---|---|---|---|---|---|
| S1 | 964 | Δ_clean | +8.64 | +12.19 | +13.88 | +15.38 | +17.50 | +13.70 |
| S1 | 964 | drop | -0.19 | +1.05 | +3.03 | +5.57 | +10.50 | +3.77 |
| S1 | 964 | drop / Δ_clean (Δ_clean ≥ 1) | -0.01 | +0.08 | +0.22 | +0.41 | +0.72 | +0.28 |
| S2 | 1200 | Δ_clean | -1.76 | +2.00 | +5.38 | +10.38 | +15.82 | +6.22 |
| S2 | 1200 | drop | -1.25 | -0.38 | +0.12 | +0.88 | +3.38 | +0.46 |
| S2 | 985 | drop / Δ_clean (Δ_clean ≥ 1) | -0.36 | -0.06 | +0.03 | +0.15 | +0.37 | +0.03 |
| S3 | 1195 | Δ_clean | -1.38 | +1.50 | +4.50 | +9.75 | +18.33 | +6.09 |
| S3 | 1195 | drop | -1.75 | -0.62 | +0.12 | +1.62 | +6.88 | +0.99 |
| S3 | 947 | drop / Δ_clean (Δ_clean ≥ 1) | -0.57 | -0.11 | +0.09 | +0.26 | +0.56 | +0.07 |
| R1 | 1132 | Δ_clean | +2.25 | +8.62 | +12.75 | +16.38 | +22.24 | +12.47 |
| R1 | 1132 | drop | -0.65 | +0.81 | +2.62 | +5.75 | +11.62 | +3.75 |
| R1 | 1097 | drop / Δ_clean (Δ_clean ≥ 1) | -0.06 | +0.08 | +0.22 | +0.48 | +1.14 | +0.35 |
| R2 | 794 | Δ_clean | +0.71 | +7.88 | +11.25 | +16.44 | +24.33 | +12.05 |
| R2 | 794 | drop | -1.12 | +0.38 | +1.62 | +3.56 | +8.34 | +2.39 |
| R2 | 752 | drop / Δ_clean (Δ_clean ≥ 1) | -0.11 | +0.05 | +0.16 | +0.28 | +0.59 | +0.18 |
| R3 | 1163 | Δ_clean | +1.62 | +8.19 | +12.25 | +16.38 | +23.87 | +12.32 |
| R3 | 1163 | drop | -1.25 | -0.06 | +0.94 | +3.25 | +9.31 | +2.13 |
| R3 | 1112 | drop / Δ_clean (Δ_clean ≥ 1) | -0.12 | +0.00 | +0.09 | +0.27 | +0.69 | +0.17 |

**qwen3_coder (chat): quantiles of Δ_clean and of the subject-noise drop per category**

| Category | n | quantity | 5 % | 25 % | median | 75 % | 95 % | mean |
|---|---|---|---|---|---|---|---|---|
| S1 | 964 | Δ_clean | +10.39 | +14.02 | +16.19 | +18.38 | +22.24 | +16.27 |
| S1 | 964 | drop | +0.06 | +1.94 | +4.11 | +7.02 | +12.11 | +4.89 |
| S1 | 964 | drop / Δ_clean (Δ_clean ≥ 1) | +0.00 | +0.12 | +0.26 | +0.43 | +0.70 | +0.30 |
| S2 | 1200 | Δ_clean | -2.38 | +2.12 | +6.25 | +12.88 | +20.88 | +7.59 |
| S2 | 1200 | drop | -1.75 | -0.50 | +0.25 | +1.75 | +6.50 | +0.99 |
| S2 | 986 | drop / Δ_clean (Δ_clean ≥ 1) | -0.41 | -0.07 | +0.06 | +0.21 | +0.44 | +0.06 |
| S3 | 1195 | Δ_clean | -5.25 | -1.25 | +3.12 | +12.12 | +23.91 | +5.86 |
| S3 | 1195 | drop | -3.04 | -1.50 | -0.38 | +2.12 | +8.38 | +0.71 |
| S3 | 726 | drop / Δ_clean (Δ_clean ≥ 1) | -0.89 | -0.09 | +0.12 | +0.29 | +0.55 | +0.02 |
| R1 | 1132 | Δ_clean | +2.62 | +10.62 | +15.50 | +20.88 | +28.93 | +15.67 |
| R1 | 1132 | drop | -1.12 | +0.88 | +3.09 | +7.12 | +14.74 | +4.54 |
| R1 | 1102 | drop / Δ_clean (Δ_clean ≥ 1) | -0.10 | +0.07 | +0.22 | +0.47 | +1.06 | +0.33 |
| R2 | 794 | Δ_clean | +0.96 | +8.75 | +14.00 | +20.69 | +30.25 | +14.70 |
| R2 | 794 | drop | -1.38 | +0.62 | +2.00 | +4.88 | +12.34 | +3.26 |
| R2 | 754 | drop / Δ_clean (Δ_clean ≥ 1) | -0.10 | +0.06 | +0.17 | +0.32 | +0.65 | +0.21 |
| R3 | 1163 | Δ_clean | +1.62 | +9.88 | +15.75 | +21.53 | +29.34 | +15.74 |
| R3 | 1163 | drop | -1.81 | +0.03 | +1.50 | +4.50 | +11.93 | +2.83 |
| R3 | 1114 | drop / Δ_clean (Δ_clean ≥ 1) | -0.13 | +0.01 | +0.11 | +0.30 | +0.71 | +0.19 |

**Qwen3-30B-A3B-Base (tokenizer defaults): calibration per sub-category (sub-categories with >= 10 scanned items)**

| Category | sub-category | n | median Δ_clean | median drop | paper filter pass | relative-25 % pass | top-1 = true | top-1 starts with true | mean prefix tokens | mean subject tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | param | 505 | +8.50 | +2.38 | 78 % | 52 % | 87 % | 87 % | 36 | 1.0 |
| R1 | store | 623 | +8.00 | +1.75 | 80 % | 48 % | 91 % | 91 % | 61 | 1.0 |
| R2 | local_dict | 94 | +9.94 | +1.66 | 78 % | 34 % | 68 % | 68 % | 76 | 9.1 |
| R2 | local_list | 503 | +6.62 | +0.38 | 48 % | 17 % | 93 % | 93 % | 75 | 2.1 |
| R2 | local_set | 49 | +10.75 | +2.62 | 86 % | 45 % | 92 % | 92 % | 65 | 3.2 |
| R2 | local_str | 30 | +12.06 | +1.88 | 87 % | 43 % | 83 % | 83 % | 86 | 13.7 |
| R2 | module_math | 40 | +8.44 | +0.22 | 42 % | 12 % | 88 % | 88 % | 29 | 1.0 |
| R2 | module_re | 73 | +8.00 | +1.12 | 68 % | 41 % | 97 % | 97 % | 24 | 1.0 |
| R3 | int | 302 | +6.75 | +0.25 | 42 % | 12 % | 82 % | 82 % | 92 | 1.0 |
| R3 | str | 861 | +7.88 | +1.25 | 66 % | 35 % | 86 % | 92 % | 87 | 1.0 |
| S1 | ) | 798 | +13.62 | +4.03 | 90 % | 59 % | 18 % | 86 % | 43 | 1.0 |
| S1 | ] | 149 | +13.03 | +5.94 | 92 % | 74 % | 23 % | 95 % | 58 | 1.0 |
| S1 | } | 17 | +12.25 | +6.88 | 100 % | 94 % | 6 % | 94 % | 62 | 1.0 |
| S2 | elif_after_if | 228 | +1.25 | +0.12 | 24 % | 14 % | 57 % | 57 % | 74 | 1.0 |
| S2 | else_after_if | 631 | +2.62 | +0.12 | 21 % | 11 % | 83 % | 83 % | 70 | 1.0 |
| S2 | else_after_try | 20 | +5.00 | +0.44 | 50 % | 25 % | 55 % | 55 % | 75 | 1.0 |
| S2 | except_after_try | 291 | +8.00 | +1.00 | 67 % | 23 % | 98 % | 98 % | 63 | 1.0 |
| S2 | finally_after_try | 27 | +1.75 | +0.25 | 26 % | 22 % | 78 % | 78 % | 70 | 1.0 |
| S3 | elif_colon | 35 | +3.75 | -0.12 | 29 % | 9 % | 3 % | 100 % | 74 | 1.0 |
| S3 | for_in | 459 | +7.00 | +1.25 | 64 % | 40 % | 98 % | 98 % | 51 | 1.0 |
| S3 | if_colon | 685 | +2.50 | -0.12 | 18 % | 8 % | 0 % | 94 % | 50 | 1.0 |
| S3 | while_colon | 16 | +3.19 | -0.19 | 25 % | 12 % | 0 % | 100 % | 58 | 1.0 |

**Mixtral-8x7B-v0.1 (no BOS, paper protocol): calibration per sub-category (sub-categories with >= 10 scanned items)**

| Category | sub-category | n | median Δ_clean | median drop | paper filter pass | relative-25 % pass | top-1 = true | top-1 starts with true | mean prefix tokens | mean subject tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | param | 356 | +7.12 | +0.38 | 45 % | 18 % | 85 % | 85 % | 47 | 1.0 |
| R1 | store | 442 | +7.12 | +0.50 | 49 % | 24 % | 91 % | 91 % | 73 | 1.0 |
| R2 | local_dict | 69 | +10.25 | +1.00 | 70 % | 20 % | 88 % | 88 % | 89 | 9.9 |
| R2 | local_list | 433 | +5.38 | +0.38 | 46 % | 12 % | 92 % | 92 % | 88 | 2.4 |
| R2 | local_set | 40 | +10.12 | +1.75 | 90 % | 22 % | 95 % | 95 % | 69 | 2.7 |
| R2 | local_str | 26 | +12.00 | +1.66 | 81 % | 31 % | 85 % | 85 % | 103 | 15.2 |
| R2 | module_math | 41 | +5.50 | -0.12 | 12 % | 7 % | 85 % | 85 % | 35 | 1.0 |
| R2 | module_re | 58 | +2.38 | +0.00 | 16 % | 5 % | 84 % | 84 % | 31 | 1.0 |
| R3 | int | 235 | +6.00 | +0.12 | 33 % | 11 % | 81 % | 81 % | 95 | 1.0 |
| R3 | str | 565 | +7.12 | +0.31 | 42 % | 17 % | 88 % | 89 % | 96 | 1.0 |
| S1 | ) | 668 | +13.25 | +3.00 | 88 % | 47 % | 50 % | 85 % | 51 | 1.0 |
| S1 | ] | 123 | +12.56 | +2.25 | 78 % | 41 % | 63 % | 94 % | 67 | 1.0 |
| S2 | elif_after_if | 152 | +0.38 | -0.12 | 6 % | 4 % | 50 % | 50 % | 82 | 1.0 |
| S2 | else_after_if | 417 | +2.50 | +0.12 | 28 % | 19 % | 82 % | 82 % | 82 | 1.0 |
| S2 | else_after_try | 15 | +5.75 | +1.38 | 87 % | 33 % | 60 % | 60 % | 92 | 1.0 |
| S2 | except_after_try | 198 | +7.31 | +2.06 | 86 % | 59 % | 97 % | 97 % | 75 | 1.0 |
| S2 | finally_after_try | 18 | +0.88 | +0.44 | 39 % | 39 % | 56 % | 56 % | 91 | 1.0 |
| S3 | elif_colon | 26 | +5.38 | +0.25 | 31 % | 15 % | 96 % | 96 % | 90 | 1.0 |
| S3 | for_in | 324 | +5.62 | +0.38 | 48 % | 19 % | 98 % | 98 % | 59 | 1.0 |
| S3 | if_colon | 440 | +4.00 | -0.12 | 17 % | 6 % | 92 % | 92 % | 57 | 1.0 |
| S3 | while_colon | 10 | +4.69 | +0.00 | 40 % | 20 % | 100 % | 100 % | 61 | 1.0 |

**Qwen3-Coder-30B-A3B-Instruct (raw code prefix): calibration per sub-category (sub-categories with >= 10 scanned items)**

| Category | sub-category | n | median Δ_clean | median drop | paper filter pass | relative-25 % pass | top-1 = true | top-1 starts with true | mean prefix tokens | mean subject tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | param | 505 | +13.00 | +2.88 | 79 % | 44 % | 87 % | 87 % | 36 | 1.0 |
| R1 | store | 623 | +12.38 | +2.50 | 80 % | 45 % | 90 % | 90 % | 61 | 1.0 |
| R2 | local_dict | 94 | +14.25 | +1.94 | 79 % | 30 % | 67 % | 67 % | 76 | 9.1 |
| R2 | local_list | 503 | +10.00 | +1.25 | 68 % | 24 % | 92 % | 92 % | 75 | 2.1 |
| R2 | local_set | 49 | +17.69 | +4.56 | 94 % | 51 % | 88 % | 88 % | 65 | 3.2 |
| R2 | local_str | 30 | +15.94 | +3.77 | 87 % | 53 % | 80 % | 80 % | 86 | 13.7 |
| R2 | module_math | 40 | +16.47 | +2.09 | 75 % | 22 % | 90 % | 90 % | 29 | 1.0 |
| R2 | module_re | 73 | +11.25 | +2.25 | 75 % | 45 % | 99 % | 99 % | 24 | 1.0 |
| R3 | int | 302 | +11.06 | +0.38 | 46 % | 11 % | 86 % | 86 % | 92 | 1.0 |
| R3 | str | 861 | +12.75 | +1.31 | 65 % | 30 % | 87 % | 92 % | 87 | 1.0 |
| S1 | ) | 798 | +13.88 | +2.81 | 84 % | 43 % | 14 % | 88 % | 43 | 1.0 |
| S1 | ] | 149 | +13.88 | +4.12 | 85 % | 60 % | 24 % | 95 % | 58 | 1.0 |
| S1 | } | 17 | +12.12 | +3.50 | 94 % | 71 % | 6 % | 88 % | 62 | 1.0 |
| S2 | elif_after_if | 228 | +2.50 | +0.25 | 27 % | 12 % | 59 % | 59 % | 74 | 1.0 |
| S2 | else_after_if | 631 | +4.25 | +0.00 | 26 % | 8 % | 80 % | 80 % | 70 | 1.0 |
| S2 | else_after_try | 20 | +7.06 | +0.19 | 45 % | 15 % | 45 % | 45 % | 75 | 1.0 |
| S2 | except_after_try | 291 | +12.62 | +0.62 | 52 % | 13 % | 98 % | 98 % | 63 | 1.0 |
| S2 | finally_after_try | 27 | +3.50 | +0.25 | 15 % | 7 % | 70 % | 70 % | 70 | 1.0 |
| S3 | elif_colon | 35 | +3.50 | -0.12 | 29 % | 14 % | 3 % | 100 % | 74 | 1.0 |
| S3 | for_in | 459 | +11.75 | +2.19 | 73 % | 41 % | 97 % | 97 % | 51 | 1.0 |
| S3 | if_colon | 685 | +2.00 | -0.25 | 19 % | 10 % | 0 % | 95 % | 50 | 1.0 |
| S3 | while_colon | 16 | +2.44 | +0.12 | 25 % | 12 % | 0 % | 94 % | 58 | 1.0 |

**Qwen3-Coder-30B-A3B-Instruct (chat template + ```python fence): calibration per sub-category (sub-categories with >= 10 scanned items)**

| Category | sub-category | n | median Δ_clean | median drop | paper filter pass | relative-25 % pass | top-1 = true | top-1 starts with true | mean prefix tokens | mean subject tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | param | 505 | +16.50 | +3.31 | 78 % | 45 % | 85 % | 85 % | 53 | 1.0 |
| R1 | store | 623 | +15.00 | +3.00 | 81 % | 43 % | 89 % | 89 % | 78 | 1.0 |
| R2 | local_dict | 94 | +19.00 | +2.88 | 86 % | 43 % | 69 % | 69 % | 93 | 9.1 |
| R2 | local_list | 503 | +12.12 | +1.88 | 73 % | 31 % | 92 % | 92 % | 92 | 2.1 |
| R2 | local_set | 49 | +23.25 | +6.75 | 92 % | 65 % | 90 % | 90 % | 82 | 3.2 |
| R2 | local_str | 30 | +19.12 | +7.31 | 90 % | 73 % | 80 % | 80 % | 103 | 13.7 |
| R2 | module_math | 40 | +21.75 | +1.25 | 70 % | 12 % | 85 % | 85 % | 46 | 1.0 |
| R2 | module_re | 73 | +10.62 | +1.00 | 71 % | 21 % | 93 % | 93 % | 41 | 1.0 |
| R3 | int | 302 | +14.69 | +1.00 | 62 % | 20 % | 85 % | 85 % | 109 | 1.0 |
| R3 | str | 861 | +16.00 | +1.88 | 66 % | 31 % | 86 % | 91 % | 104 | 1.0 |
| S1 | ) | 798 | +16.19 | +3.94 | 91 % | 50 % | 6 % | 88 % | 60 | 1.0 |
| S1 | ] | 149 | +16.38 | +5.88 | 91 % | 63 % | 23 % | 95 % | 75 | 1.0 |
| S1 | } | 17 | +14.44 | +4.06 | 100 % | 59 % | 0 % | 82 % | 79 | 1.0 |
| S2 | elif_after_if | 228 | +3.06 | +0.31 | 32 % | 16 % | 62 % | 62 % | 91 | 1.0 |
| S2 | else_after_if | 631 | +4.62 | -0.12 | 25 % | 9 % | 79 % | 79 % | 87 | 1.0 |
| S2 | else_after_try | 20 | +7.38 | +0.94 | 60 % | 20 % | 45 % | 45 % | 92 | 1.0 |
| S2 | except_after_try | 291 | +16.12 | +2.50 | 77 % | 34 % | 99 % | 99 % | 80 | 1.0 |
| S2 | finally_after_try | 27 | +4.38 | +1.50 | 56 % | 41 % | 70 % | 70 % | 87 | 1.0 |
| S3 | elif_colon | 35 | +1.75 | -0.12 | 26 % | 17 % | 3 % | 100 % | 91 | 1.0 |
| S3 | for_in | 459 | +14.50 | +2.88 | 73 % | 41 % | 97 % | 97 % | 68 | 1.0 |
| S3 | if_colon | 685 | -0.38 | -1.12 | 8 % | 5 % | 0 % | 94 % | 67 | 1.0 |
| S3 | while_colon | 16 | -0.12 | +0.62 | 38 % | 19 % | 0 % | 94 % | 75 | 1.0 |

The relative rule (drop ≥ 25 % of Δ_clean) admits the syntax items whose absolute drop is small relative to a very large margin only when the drop is also a quarter of that margin, so it is stricter than the paper's rule for high-margin items and looser for low-margin ones; the pass rates above show where the two rules disagree. It is not used for any selection in this section.

**Deviations and open questions.** (1) The Stack replaced by CodeSearchNet (gated dataset, no token). (2) Tokenizer merges force a boundary back-off for many items (all S2 and R2 items under Qwen: ` else`, `.append` are single tokens), so the 'true token' sometimes contains the preceding punctuation; the contrast between true and foil is unchanged. (3) In Qwen's tokenizer an opener often merges with the following identifier (`(bar`), so the S1 subject token carries the first argument too. (4) R3 integer items (single digits) are weak by construction; string items are the informative part. (5) Per-category sets share one expert pass (union of cases), so a category's cases can appear in the mixed `all` set. (6) Coder-Instruct runs depend on the ext2 download (marked pending if absent).

_Generated 2026-09-14T15:55:30Z by scripts/ext4_analyze.py._
