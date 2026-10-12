**Competence scans of the Phase-5 builds (cluster bootstrap over seeds / Phase-3 twins); verbs_ent2attr = the existing IOI-name entity → attribute scan**

| Model | Build | Subset | Pairs | Units (seeds / twins) | Margin ≥ 1 both ways | Correct in both twins | Mean drop |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | vt_a2e | variant neutral | 2640 | 55 | 0.02 [0.00, 0.05] | 0.22 [0.16, 0.28] | 0.72 [0.53, 0.92] |
| Qwen3-30B-A3B-Base | vt_a2e | variant orig | 2640 | 55 | 0.04 [0.01, 0.07] | 0.27 [0.20, 0.34] | 0.97 [0.73, 1.22] |
| Qwen3-30B-A3B-Base | vt_a2e | variant unrelated | 2640 | 55 | 0.00 [0.00, 0.00] | 0.05 [0.03, 0.08] | 0.01 [-0.14, 0.16] |
| Qwen3-30B-A3B-Base | cp_vt | variant neutral | 5280 | 55 | 0.22 [0.14, 0.30] | 0.47 [0.36, 0.57] | 3.11 [2.53, 3.68] |
| Qwen3-30B-A3B-Base | cp_vt | variant orig | 5280 | 55 | 0.31 [0.21, 0.41] | 0.53 [0.43, 0.63] | 3.95 [3.40, 4.48] |
| Qwen3-30B-A3B-Base | cp_vt | variant unrelated | 5280 | 55 | 0.03 [0.00, 0.07] | 0.08 [0.04, 0.14] | 0.20 [-0.11, 0.53] |
| Qwen3-30B-A3B-Base | cp_nat | order 1 | 776 | 776 | 0.73 [0.70, 0.77] | 0.92 [0.90, 0.94] | 5.40 [5.24, 5.56] |
| Qwen3-30B-A3B-Base | cp_nat | order 2 | 776 | 776 | 0.74 [0.71, 0.77] | 0.93 [0.91, 0.95] | 5.39 [5.24, 5.54] |
| Qwen3-30B-A3B-Base | verbs_ent2attr | variant neutral | 2640 | 55 | 0.27 [0.18, 0.37] | 0.42 [0.31, 0.53] | 4.25 [3.48, 4.99] |
| Qwen3-30B-A3B-Base | verbs_ent2attr | variant orig | 2640 | 55 | 0.39 [0.28, 0.50] | 0.53 [0.42, 0.65] | 4.97 [4.26, 5.65] |
| Qwen3-30B-A3B-Base | verbs_ent2attr | variant unrelated | 2640 | 55 | 0.05 [0.01, 0.09] | 0.16 [0.09, 0.23] | 0.22 [-0.20, 0.61] |
| Mixtral-8x7B, BOS | vt_a2e | variant neutral | 2640 | 55 | 0.00 [0.00, 0.00] | 0.20 [0.15, 0.25] | 0.37 [0.26, 0.47] |
| Mixtral-8x7B, BOS | vt_a2e | variant orig | 2640 | 55 | 0.00 [0.00, 0.00] | 0.23 [0.17, 0.30] | 0.46 [0.33, 0.61] |
| Mixtral-8x7B, BOS | vt_a2e | variant unrelated | 2640 | 55 | 0.00 [0.00, 0.00] | 0.06 [0.04, 0.09] | -0.02 [-0.10, 0.05] |
| Mixtral-8x7B, BOS | cp_vt | variant neutral | 5280 | 55 | 0.19 [0.11, 0.26] | 0.43 [0.34, 0.52] | 2.92 [2.44, 3.41] |
| Mixtral-8x7B, BOS | cp_vt | variant orig | 5280 | 55 | 0.29 [0.21, 0.38] | 0.48 [0.38, 0.59] | 3.49 [2.98, 4.02] |
| Mixtral-8x7B, BOS | cp_vt | variant unrelated | 5280 | 55 | 0.02 [0.00, 0.05] | 0.08 [0.03, 0.13] | 0.23 [-0.23, 0.64] |
| Mixtral-8x7B, BOS | cp_nat | order 1 | 776 | 776 | 0.90 [0.88, 0.92] | 0.98 [0.97, 0.99] | 6.91 [6.75, 7.06] |
| Mixtral-8x7B, BOS | cp_nat | order 2 | 776 | 776 | 0.89 [0.87, 0.91] | 0.97 [0.96, 0.98] | 6.74 [6.58, 6.91] |
| Mixtral-8x7B, BOS | verbs_ent2attr | variant neutral | 2640 | 55 | 0.21 [0.12, 0.30] | 0.39 [0.28, 0.50] | 3.91 [3.31, 4.52] |
| Mixtral-8x7B, BOS | verbs_ent2attr | variant orig | 2640 | 55 | 0.33 [0.22, 0.43] | 0.51 [0.39, 0.63] | 4.71 [4.09, 5.34] |
| Mixtral-8x7B, BOS | verbs_ent2attr | variant unrelated | 2640 | 55 | 0.02 [0.00, 0.04] | 0.08 [0.03, 0.15] | 0.00 [-0.61, 0.50] |
