## mixtral_bos: concept circuits from existing rows (selection: source validation split; evaluation: source discovery split)

| task | concept | selection units | evaluation pairs | head layers measured | expert rows |
|---|---|---|---|---|---|
| ss_e3 | SIZE_STATE | 172 | 48 | all | yes |
| ss_e1 | SIZE_STATE | 72 | 36 | 13,15,16,18,19 | yes |
| co_p17 | COUNTRY | 96 | 48 | 13,15,16,18,19,24 | yes |
| co_nat | COUNTRY | 192 | 48 | 13,15,16,18,19,24 | yes |
| pr_ioi | PERSON_ROLE | 256 | 48 | 16,19,21,22,31 | no |
| pr_f1n | PERSON_ROLE | 24 | 11 | 15,16,19,21,27 | yes |
| pr_e1n | PERSON_ROLE | 32 | 14 | 13,16,18,19,23 | yes |
| ct_lang | CONTROL | 96 | 48 | 13,15,16,18,19,24 | yes |

| concept | tasks | circuit heads | circuit experts | in top sets of other concepts (excluded) | also in the control's top set |
|---|---|---|---|---|---|
| SIZE_STATE | ss_e3, ss_e1 | L13H18, L13H4, L15H7, L19H13 | L21E004, L21E006, L24E002, L25E002, L26E002, L26E007 | L15H8, L19E006, L19H12, L20E000 | L15H7, L21E006, L26E007 |
| COUNTRY | co_p17, co_nat | L15H1, L15H3, L18H12, L18H14, L18H4, L19H29, L19H30, L24H21, L24H22, L24H23 | L19E002, L21E001, L22E001, L23E002, L25E003, L28E002 | L18E001 | L15H1, L15H3, L18H12, L18H4, L19H29, L19H30, L21E001, L22E001, L24H21, L24H22, L24H23 |
| PERSON_ROLE | pr_ioi, pr_f1n, pr_e1n | L16H1, L16H12, L16H14 | L15E002, L15E006, L16E005, L17E007, L19E007, L20E006, L28E003 | L16H0, L19H8 | – |

| task | top heads (mean effect = rescue / drop) | top experts |
|---|---|---|
| ss_e3 | L25H9 +0.084, L22H20 +0.068, L19H13 +0.059, L13H18 +0.034, L19H12 +0.029 | L20E000 +0.154, L19E006 +0.134, L21E006 +0.101, L24E002 +0.091, L26E002 +0.088 |
| ss_e1 | L19H13 +0.108, L15H8 +0.085, L19H12 +0.079, L16H0 +0.075, L15H7 +0.068 | L21E004 +0.112, L25E002 +0.104, L26E002 +0.101, L20E000 +0.083, L22E000 +0.080 |
| co_p17 | L24H22 +0.168, L18H4 +0.168, L19H29 +0.064, L24H23 +0.041, L24H21 +0.019 | L19E002 +0.065, L21E001 +0.045, L22E001 +0.028, L20E000 +0.025, L18E001 +0.018 |
| co_nat | L18H4 +0.273, L24H22 +0.233, L19H29 +0.111, L24H23 +0.071, L24H21 +0.061 | L21E001 +0.063, L19E002 +0.051, L25E003 +0.036, L22E001 +0.036, L20E005 +0.026 |
| pr_ioi | L21H6 +0.108, L19H10 +0.065, L31H5 +0.064, L16H15 +0.051, L19H8 +0.050 | – (no expert rows) |
| pr_f1n | L16H1 +0.145, L19H8 +0.136, L19H18 +0.121, L16H14 +0.116, L19H14 +0.090 | L17E007 +0.125, L16E005 +0.115, L18E001 +0.057, L15E002 +0.053, L15E006 +0.044 |
| pr_e1n | L16H1 +0.481, L19H8 +0.332, L13H11 +0.299, L16H0 +0.230, L18H10 +0.116 | L17E007 +0.239, L16E005 +0.101, L15E006 +0.096, L28E003 +0.084, L19E007 +0.043 |
| ct_lang | L18H4 +0.307, L24H22 +0.248, L19H29 +0.105, L24H21 +0.080, L24H23 +0.059 | L22E005 +0.036, L20E005 +0.035, L19E006 +0.029, L21E006 +0.025, L21E001 +0.023 |

AUROC of each top component for its home concept (selection cases of the tasks where it was measured): heads median 0.792 (n 41), experts median 0.794 (n 35); Mann-Whitney experts > heads p = 0.426

### Knockout matrix: fraction of the clean margin lost F (circuit) minus the mean of 2 matched random sets; evaluation cases

| circuit (part) | ss_e3 (SIZE) | ss_e1 (SIZE) | co_p17 (COUN) | co_nat (COUN) | pr_ioi (PERS) | pr_f1n (PERS) | pr_e1n (PERS) | ct_lang (CONT) | own − other |
|---|---|---|---|---|---|---|---|---|---|
| COUNTRY (all) | -0.059 | -0.159 | +0.613 | +0.776 | -0.132 | -0.241 | -0.164 | +0.754 | +0.695 |
| COUNTRY (experts) | -0.036 | -0.066 | -0.016 | -0.007 | +0.004 | -0.012 | +0.002 | +0.005 | +0.006 |
| COUNTRY (heads) | -0.029 | -0.106 | +0.610 | +0.766 | -0.140 | -0.256 | -0.170 | +0.743 | +0.681 |
| PERSON_ROLE (all) | +0.003 | +0.054 | +0.005 | +0.010 | +0.007 | +0.509 | +0.658 | +0.015 | +0.374 |
| PERSON_ROLE (experts) | -0.011 | -0.095 | -0.020 | -0.012 | -0.014 | +0.179 | +0.226 | -0.003 | +0.158 |
| PERSON_ROLE (heads) | +0.017 | +0.173 | +0.024 | +0.016 | +0.032 | +0.483 | +0.586 | +0.022 | +0.317 |
| SIZE_STATE (all) | +0.228 | +0.409 | +0.040 | +0.062 | +0.026 | +0.307 | +0.134 | +0.065 | +0.212 |
| SIZE_STATE (experts) | +0.107 | +0.162 | +0.005 | +0.005 | -0.001 | +0.044 | +0.000 | -0.004 | +0.126 |
| SIZE_STATE (heads) | +0.125 | +0.303 | +0.038 | +0.055 | +0.027 | +0.254 | +0.135 | +0.066 | +0.118 |

Raw F of the circuits (draw 0) and of the random sets (mean of draws 1-2):

| circuit (part) | ss_e3 | ss_e1 | co_p17 | co_nat | pr_ioi | pr_f1n | pr_e1n | ct_lang |
|---|---|---|---|---|---|---|---|---|
| COUNTRY (all) | 0.004 (0.063) | 0.003 (0.162) | 0.613 (0.000) | 0.768 (-0.008) | -0.030 (0.102) | -0.014 (0.227) | -0.011 (0.153) | 0.746 (-0.008) |
| COUNTRY (experts) | -0.002 (0.035) | -0.001 (0.065) | -0.011 (0.005) | -0.003 (0.004) | 0.009 (0.005) | 0.007 (0.019) | 0.016 (0.014) | 0.004 (-0.001) |
| COUNTRY (heads) | 0.008 (0.037) | -0.008 (0.098) | 0.606 (-0.004) | 0.757 (-0.009) | -0.042 (0.098) | -0.038 (0.218) | -0.021 (0.149) | 0.734 (-0.008) |
| PERSON_ROLE (all) | 0.031 (0.027) | 0.071 (0.018) | 0.028 (0.023) | 0.009 (-0.000) | 0.020 (0.013) | 0.601 (0.092) | 0.642 (-0.016) | 0.021 (0.006) |
| PERSON_ROLE (experts) | 0.028 (0.039) | 0.002 (0.097) | 0.004 (0.024) | -0.002 (0.010) | -0.012 (0.002) | 0.300 (0.121) | 0.232 (0.006) | 0.004 (0.007) |
| PERSON_ROLE (heads) | 0.007 (-0.010) | 0.074 (-0.098) | 0.024 (0.000) | 0.009 (-0.007) | 0.037 (0.005) | 0.454 (-0.029) | 0.585 (-0.001) | 0.022 (0.000) |
| SIZE_STATE (all) | 0.242 (0.015) | 0.452 (0.043) | 0.037 (-0.004) | 0.061 (-0.000) | 0.055 (0.028) | 0.328 (0.020) | 0.188 (0.054) | 0.068 (0.003) |
| SIZE_STATE (experts) | 0.106 (-0.001) | 0.149 (-0.013) | -0.000 (-0.006) | 0.001 (-0.004) | 0.003 (0.004) | 0.044 (0.000) | 0.014 (0.014) | -0.003 (0.001) |
| SIZE_STATE (heads) | 0.141 (0.016) | 0.352 (0.049) | 0.038 (0.000) | 0.060 (0.005) | 0.053 (0.027) | 0.276 (0.022) | 0.183 (0.048) | 0.070 (0.004) |

Specificity per circuit part (N3): F_own = mean net F on the concept's own tasks, F_other = mean |net F| on every other task (other concepts and the control), selectivity = F_own / (|F_own| + F_other):

| circuit (part) | F_own | F_other (mean abs) | F on the control task | selectivity |
|---|---|---|---|---|
| COUNTRY (all) | +0.695 | 0.251 | +0.754 | 0.73 |
| COUNTRY (experts) | -0.011 | 0.021 | +0.005 | -0.35 |
| COUNTRY (heads) | +0.688 | 0.240 | +0.743 | 0.74 |
| PERSON_ROLE (all) | +0.391 | 0.017 | +0.015 | 0.96 |
| PERSON_ROLE (experts) | +0.130 | 0.028 | -0.003 | 0.82 |
| PERSON_ROLE (heads) | +0.367 | 0.050 | +0.022 | 0.88 |
| SIZE_STATE (all) | +0.318 | 0.106 | +0.065 | 0.75 |
| SIZE_STATE (experts) | +0.135 | 0.010 | -0.004 | 0.93 |
| SIZE_STATE (heads) | +0.214 | 0.096 | +0.066 | 0.69 |

### Add-back of the concept set (fraction of the drop restored; matched random sets in parentheses)

| circuit (part) | ss_e3 | ss_e1 | co_p17 | co_nat | pr_ioi | pr_f1n | pr_e1n | ct_lang |
|---|---|---|---|---|---|---|---|---|
| COUNTRY (all) | 0.009 (0.158) | 0.030 (0.145) | 0.806 (-0.002) | 0.911 (-0.001) | 0.041 (0.058) | 0.060 (0.140) | 0.048 (0.125) | 0.824 (0.061) |
| COUNTRY (experts) | 0.006 (0.124) | 0.019 (0.106) | 0.180 (0.006) | 0.269 (0.013) | 0.033 (0.013) | 0.033 (0.037) | 0.017 (0.048) | 0.079 (0.073) |
| COUNTRY (heads) | 0.003 (0.037) | 0.013 (0.051) | 0.799 (-0.007) | 0.891 (-0.014) | 0.009 (0.046) | 0.041 (0.109) | 0.027 (0.075) | 0.815 (-0.012) |
| PERSON_ROLE (all) | 0.034 (0.095) | -0.012 (0.132) | 0.012 (0.041) | 0.014 (0.073) | 0.027 (0.013) | 0.519 (0.029) | 0.899 (0.022) | 0.013 (0.037) |
| PERSON_ROLE (experts) | 0.035 (0.087) | -0.026 (0.113) | 0.009 (0.041) | 0.013 (0.073) | 0.033 (0.012) | 0.368 (0.027) | 0.628 (0.002) | 0.010 (0.038) |
| PERSON_ROLE (heads) | 0.001 (0.009) | 0.015 (0.016) | 0.003 (-0.001) | 0.003 (-0.001) | -0.006 (0.001) | 0.313 (0.010) | 0.619 (0.019) | 0.004 (0.001) |
| SIZE_STATE (all) | 0.481 (0.004) | 0.587 (0.018) | 0.006 (0.019) | 0.026 (0.083) | 0.009 (0.011) | 0.129 (0.025) | 0.101 (0.028) | 0.050 (0.036) |
| SIZE_STATE (experts) | 0.369 (0.002) | 0.350 (0.005) | -0.000 (0.014) | 0.005 (0.073) | 0.003 (0.012) | 0.012 (0.025) | 0.008 (0.028) | 0.031 (0.031) |
| SIZE_STATE (heads) | 0.206 (0.003) | 0.391 (0.013) | 0.003 (0.003) | 0.020 (0.008) | 0.007 (-0.000) | 0.113 (0.003) | 0.093 (-0.002) | 0.020 (0.006) |

### Edges A -> B among circuit components (mediated effect rescue(A) + rescue(B) - rescue(A u B) / drop; top 12)

| concept | A | B | edge / drop | share of A's rescue through B |
|---|---|---|---|---|
| COUNTRY | L18H4 | L24H22 | +0.035 | 0.15 |
| COUNTRY | L18H4 | L24H23 | +0.013 | 0.05 |
| COUNTRY | L18H4 | L21E001 | +0.011 | 0.05 |
| SIZE_STATE | L21E006 | L26E002 | +0.009 | 0.09 |
| COUNTRY | L18H4 | L23E002 | +0.009 | 0.04 |
| COUNTRY | L18H4 | L25E003 | +0.008 | 0.03 |
| COUNTRY | L18H4 | L28E002 | +0.007 | 0.03 |
| SIZE_STATE | L19H13 | L21E004 | +0.006 | 0.09 |
| SIZE_STATE | L19H13 | L24E002 | +0.006 | 0.09 |
| SIZE_STATE | L19H13 | L25E002 | +0.005 | 0.08 |
| SIZE_STATE | L24E002 | L26E002 | +0.005 | 0.06 |
| SIZE_STATE | L19H13 | L21E006 | +0.005 | 0.07 |
