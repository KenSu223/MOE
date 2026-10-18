## qwen3: concept circuits from existing rows (selection: source validation split; evaluation: source discovery split)

| task | concept | selection units | evaluation pairs | head layers measured | expert rows |
|---|---|---|---|---|---|
| ss_e3 | SIZE_STATE | 172 | 48 | all | yes |
| ss_e1 | SIZE_STATE | 188 | 48 | 36,38,42,43,45 | yes |
| co_p17 | COUNTRY | 96 | 48 | 28,38,39,40,42,43 | yes |
| co_nat | COUNTRY | 192 | 48 | 28,38,39,40,42,43 | yes |
| pr_ioi | PERSON_ROLE | 256 | 48 | all | no |
| pr_f1n | PERSON_ROLE | 140 | 48 | 33,39,42,43,45 | yes |
| pr_e1n | PERSON_ROLE | 32 | 22 | 27,39,42,43,45 | yes |
| ct_lang | CONTROL | 96 | 48 | 28,38,39,40,42,43 | yes |

| concept | tasks | circuit heads | circuit experts | in top sets of other concepts (excluded) | also in the control's top set |
|---|---|---|---|---|---|
| SIZE_STATE | ss_e3, ss_e1 | L38H18, L38H20, L38H21 | L34E119, L38E021, L39E071, L41E053, L41E117, L43E081, L43E114, L44E122 | – | – |
| COUNTRY | co_p17, co_nat | L28H13, L28H15, L40H11, L40H12, L40H13, L40H14, L40H15, L43H11 | L40E030, L40E127, L41E001, L42E115, L43E005, L44E052, L44E069 | L43H15 | L28H13, L28H15, L40E127, L40H11, L40H12, L40H13, L40H14, L40H15, L41E001, L42E115, L43H11, L44E052, L44E069 |
| PERSON_ROLE | pr_ioi, pr_f1n, pr_e1n | L39H27, L39H29, L39H31, L43H22 | L37E032, L37E033, L39E036 | L42H10, L43H15, L43H24 | – |

| task | top heads (mean effect = rescue / drop) | top experts |
|---|---|---|
| ss_e3 | L46H24 +0.119, L41H27 +0.074, L38H18 +0.047, L29H0 +0.043, L38H21 +0.037 | L41E117 +0.145, L43E081 +0.098, L39E071 +0.076, L34E119 +0.054, L44E122 +0.043 |
| ss_e1 | L38H18 +0.051, L36H28 +0.034, L38H21 +0.025, L42H18 +0.022, L43H24 +0.013 | L39E071 +0.115, L38E021 +0.107, L43E081 +0.103, L41E117 +0.100, L34E119 +0.074 |
| co_p17 | L40H13 +0.219, L43H15 +0.067, L43H11 +0.067, L40H15 +0.065, L40H14 +0.057 | L44E069 +0.112, L42E115 +0.083, L43E005 +0.055, L44E052 +0.029, L40E127 +0.023 |
| co_nat | L40H13 +0.267, L43H11 +0.090, L40H15 +0.090, L40H14 +0.069, L43H15 +0.062 | L44E069 +0.186, L42E115 +0.088, L44E052 +0.032, L43E005 +0.029, L41E001 +0.023 |
| pr_ioi | L42H11 +0.180, L43H24 +0.102, L40H2 +0.071, L42H10 +0.069, L43H29 +0.063 | – (no expert rows) |
| pr_f1n | L42H10 +0.283, L39H29 +0.166, L43H24 +0.145, L39H27 +0.093, L39H31 +0.087 | L37E032 +0.176, L39E103 +0.044, L40E115 +0.021, L42E031 +0.018, L33E045 +0.015 |
| pr_e1n | L42H10 +0.294, L43H24 +0.160, L39H27 +0.136, L39H29 +0.119, L39H31 +0.116 | L37E032 +0.098, L37E033 +0.018, L39E036 +0.016, L43E041 +0.014, L45E002 +0.014 |
| ct_lang | L40H13 +0.299, L43H11 +0.098, L40H15 +0.094, L43H15 +0.080, L40H14 +0.069 | L42E115 +0.156, L44E069 +0.129, L41E001 +0.021, L40E127 +0.017, L33E076 +0.015 |

AUROC of each top component for its home concept (selection cases of the tasks where it was measured): heads median 0.786 (n 44), experts median 0.665 (n 42); Mann-Whitney experts > heads p = 0.982

### Knockout matrix: fraction of the clean margin lost F (circuit) minus the mean of 2 matched random sets; evaluation cases

| circuit (part) | ss_e3 (SIZE) | ss_e1 (SIZE) | co_p17 (COUN) | co_nat (COUN) | pr_ioi (PERS) | pr_f1n (PERS) | pr_e1n (PERS) | ct_lang (CONT) | own − other |
|---|---|---|---|---|---|---|---|---|---|
| COUNTRY (all) | -0.037 | -0.042 | +0.664 | +0.735 | +0.042 | +0.061 | +0.034 | +0.726 | +0.569 |
| COUNTRY (experts) | -0.022 | -0.006 | +0.126 | +0.102 | -0.032 | +0.000 | -0.010 | +0.210 | +0.091 |
| COUNTRY (heads) | -0.013 | -0.041 | +0.605 | +0.687 | +0.086 | +0.043 | +0.052 | +0.641 | +0.518 |
| PERSON_ROLE (all) | +0.018 | +0.081 | +0.011 | +0.036 | -0.012 | +0.327 | +0.321 | +0.040 | +0.174 |
| PERSON_ROLE (experts) | -0.011 | -0.002 | -0.002 | -0.002 | -0.004 | +0.099 | +0.104 | +0.001 | +0.070 |
| PERSON_ROLE (heads) | +0.023 | +0.083 | +0.013 | +0.034 | -0.009 | +0.287 | +0.253 | +0.038 | +0.139 |
| SIZE_STATE (all) | +0.369 | +0.488 | +0.003 | +0.000 | -0.007 | +0.234 | +0.241 | +0.048 | +0.342 |
| SIZE_STATE (experts) | +0.246 | +0.362 | -0.002 | -0.003 | -0.003 | -0.005 | +0.006 | +0.022 | +0.301 |
| SIZE_STATE (heads) | +0.125 | +0.101 | +0.005 | +0.003 | -0.007 | +0.239 | +0.246 | +0.026 | +0.027 |

Raw F of the circuits (draw 0) and of the random sets (mean of draws 1-2):

| circuit (part) | ss_e3 | ss_e1 | co_p17 | co_nat | pr_ioi | pr_f1n | pr_e1n | ct_lang |
|---|---|---|---|---|---|---|---|---|
| COUNTRY (all) | -0.001 (0.035) | 0.009 (0.051) | 0.666 (0.002) | 0.740 (0.005) | -0.035 (-0.077) | -0.013 (-0.074) | -0.013 (-0.047) | 0.721 (-0.005) |
| COUNTRY (experts) | -0.010 (0.012) | 0.001 (0.007) | 0.126 (-0.001) | 0.106 (0.005) | -0.025 (0.007) | -0.012 (-0.012) | 0.003 (0.014) | 0.208 (-0.002) |
| COUNTRY (heads) | 0.002 (0.015) | 0.007 (0.048) | 0.608 (0.004) | 0.690 (0.003) | 0.003 (-0.083) | -0.011 (-0.054) | -0.006 (-0.058) | 0.640 (-0.002) |
| PERSON_ROLE (all) | 0.015 (-0.003) | 0.067 (-0.013) | 0.012 (0.001) | 0.033 (-0.002) | -0.008 (0.004) | 0.319 (-0.008) | 0.334 (0.013) | 0.040 (-0.000) |
| PERSON_ROLE (experts) | -0.008 (0.003) | 0.008 (0.010) | -0.001 (0.001) | 0.000 (0.002) | -0.001 (0.003) | 0.089 (-0.010) | 0.109 (0.005) | 0.002 (0.001) |
| PERSON_ROLE (heads) | 0.014 (-0.009) | 0.062 (-0.020) | 0.015 (0.002) | 0.033 (-0.001) | -0.009 (0.000) | 0.283 (-0.003) | 0.268 (0.016) | 0.037 (-0.002) |
| SIZE_STATE (all) | 0.364 (-0.005) | 0.474 (-0.013) | 0.002 (-0.002) | -0.005 (-0.005) | -0.011 (-0.004) | 0.226 (-0.008) | 0.239 (-0.002) | 0.032 (-0.016) |
| SIZE_STATE (experts) | 0.242 (-0.003) | 0.359 (-0.002) | -0.001 (0.001) | -0.000 (0.003) | -0.001 (0.002) | 0.002 (0.007) | 0.002 (-0.004) | 0.020 (-0.002) |
| SIZE_STATE (heads) | 0.116 (-0.008) | 0.086 (-0.015) | 0.001 (-0.004) | -0.003 (-0.006) | -0.010 (-0.004) | 0.226 (-0.014) | 0.242 (-0.003) | 0.012 (-0.014) |

Specificity per circuit part (N3): F_own = mean net F on the concept's own tasks, F_other = mean |net F| on every other task (other concepts and the control), selectivity = F_own / (|F_own| + F_other):

| circuit (part) | F_own | F_other (mean abs) | F on the control task | selectivity |
|---|---|---|---|---|
| COUNTRY (all) | +0.700 | 0.157 | +0.726 | 0.82 |
| COUNTRY (experts) | +0.114 | 0.047 | +0.210 | 0.71 |
| COUNTRY (heads) | +0.646 | 0.146 | +0.641 | 0.82 |
| PERSON_ROLE (all) | +0.212 | 0.037 | +0.040 | 0.85 |
| PERSON_ROLE (experts) | +0.066 | 0.004 | +0.001 | 0.95 |
| PERSON_ROLE (heads) | +0.177 | 0.038 | +0.038 | 0.82 |
| SIZE_STATE (all) | +0.429 | 0.089 | +0.048 | 0.83 |
| SIZE_STATE (experts) | +0.304 | 0.007 | +0.022 | 0.98 |
| SIZE_STATE (heads) | +0.113 | 0.087 | +0.026 | 0.56 |

### Add-back of the concept set (fraction of the drop restored; matched random sets in parentheses)

| circuit (part) | ss_e3 | ss_e1 | co_p17 | co_nat | pr_ioi | pr_f1n | pr_e1n | ct_lang |
|---|---|---|---|---|---|---|---|---|
| COUNTRY (all) | 0.015 (0.031) | 0.005 (0.015) | 0.801 (0.020) | 0.852 (0.013) | 0.004 (-0.020) | 0.006 (-0.043) | 0.008 (-0.013) | 0.814 (0.026) |
| COUNTRY (experts) | -0.001 (0.028) | -0.001 (0.010) | 0.276 (0.001) | 0.329 (0.000) | 0.006 (0.002) | 0.010 (0.002) | 0.008 (0.006) | 0.329 (0.005) |
| COUNTRY (heads) | 0.015 (0.003) | 0.005 (0.006) | 0.795 (0.018) | 0.823 (0.011) | -0.001 (-0.024) | -0.004 (-0.048) | 0.002 (-0.015) | 0.783 (0.021) |
| PERSON_ROLE (all) | 0.003 (0.007) | 0.050 (-0.007) | 0.003 (-0.000) | 0.007 (-0.000) | 0.043 (0.006) | 0.554 (-0.007) | 0.557 (-0.000) | 0.010 (0.001) |
| PERSON_ROLE (experts) | -0.001 (0.005) | 0.001 (0.003) | -0.001 (-0.000) | -0.001 (-0.001) | 0.001 (0.002) | 0.173 (-0.003) | 0.154 (0.001) | -0.000 (0.001) |
| PERSON_ROLE (heads) | 0.004 (0.002) | 0.049 (-0.010) | 0.004 (-0.000) | 0.008 (-0.000) | 0.043 (0.004) | 0.443 (-0.005) | 0.455 (0.001) | 0.012 (0.002) |
| SIZE_STATE (all) | 0.523 (-0.002) | 0.626 (-0.002) | 0.001 (0.001) | 0.001 (0.001) | -0.003 (0.001) | 0.280 (-0.021) | 0.290 (-0.014) | 0.004 (0.000) |
| SIZE_STATE (experts) | 0.446 (0.000) | 0.527 (0.001) | -0.000 (0.001) | -0.001 (0.002) | 0.000 (0.000) | -0.002 (-0.004) | 0.001 (0.003) | 0.002 (0.002) |
| SIZE_STATE (heads) | 0.102 (-0.001) | 0.100 (-0.002) | 0.001 (0.000) | -0.000 (-0.003) | -0.002 (0.001) | 0.275 (-0.018) | 0.293 (-0.015) | 0.003 (-0.001) |

### Edges A -> B among circuit components (mediated effect rescue(A) + rescue(B) - rescue(A u B) / drop; top 12)

| concept | A | B | edge / drop | share of A's rescue through B |
|---|---|---|---|---|
| COUNTRY | L40H13 | L44E069 | +0.034 | 0.14 |
| SIZE_STATE | L34E119 | L43E081 | +0.011 | 0.15 |
| PERSON_ROLE | L39H29 | L43H22 | +0.009 | 0.14 |
| PERSON_ROLE | L39H27 | L43H22 | +0.009 | 0.19 |
| SIZE_STATE | L34E119 | L41E117 | +0.009 | 0.12 |
| PERSON_ROLE | L39H31 | L43H22 | +0.008 | 0.18 |
| COUNTRY | L40H13 | L42E115 | +0.008 | 0.03 |
| PERSON_ROLE | L37E032 | L43H22 | +0.008 | 0.15 |
| COUNTRY | L40H15 | L44E069 | +0.006 | 0.09 |
| SIZE_STATE | L38E021 | L43E081 | +0.006 | 0.10 |
| SIZE_STATE | L39E071 | L43E081 | +0.005 | 0.05 |
| COUNTRY | L40H14 | L44E069 | +0.005 | 0.08 |
