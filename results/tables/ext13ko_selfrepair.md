**Knockout loss vs the expert's own direct write on the same clean prompts (eval items whose final position routes the expert; all-position reroute). W = DLA of c_e (baseline), N = net change of the layer's MoE direct write (baseline - knockout; includes the replacement expert), K = knockout loss of Delta (logits). 1 - K/W = compensation in the ext9 synthesis convention; K/N > 1 means the downstream network amplifies the loss.**

| model | condition | expert | routed items | direct write W | net layer direct change N | knockout K | K / W [95% CI] | K / N [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1 | L37E032 | 754 | 0.0021 | 0.0022 | 0.1016 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | f1 | L41E117 | 17 | 0.019 | 0.0125 | 0.0368 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | f1 | L44E069 | 71 | 0.0245 | 0.0139 | 0.0458 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | f1 | L42E115 | 274 | 0.0046 | 0.0005 | -0.0287 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | dir_f1 | L44E116 | 153 | 0.2842 | 0.2486 | 0.0596 | 0.21 [0.07, 0.34] | 0.24 [0.08, 0.39] |
| Qwen3-30B-A3B-Base | dir_f1 | L44E069 | 15 | 0.0495 | -0.0024 | 0.0167 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | dir_f1 | L42E115 | 24 | 0.015 | 0.009 | -0.0729 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | dir_opt | L41E117 | 139 | 0.3776 | 0.3351 | 0.1425 | 0.38 [0.19, 0.55] | 0.43 [0.22, 0.62] |
| Qwen3-30B-A3B-Base | f2e | L43E036 | 63 | 0.1373 | 0.0975 | 0.0605 | 0.44 [-0.18, 1.18] | n/a |
| Qwen3-30B-A3B-Base | f2e | L44E069 | 28 | 0.0205 | 0.0101 | -0.029 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | f2e | L42E115 | 40 | 0.0225 | 0.0124 | 0.0359 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L39E121 | 184 | 0.1801 | 0.1505 | 0.0421 | 0.23 [-1.01, 0.47] | 0.28 [-1.61, 0.75] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L41E117 | 119 | 0.1313 | 0.1056 | -0.011 | -0.08 [-12.79, 6.94] | -0.10 [-5.04, 8.49] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L44E069 | 61 | 0.1769 | 0.1258 | 0.1762 | 1.00 [-3.23, 1.59] | 1.40 [-1.07, 3.07] |
| Qwen3-30B-A3B-Base | vt_e2a_orig | L42E115 | 20 | -0.0082 | 0.0735 | -0.0187 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L42E123 | 256 | 0.1896 | 0.1751 | 0.0608 | 0.32 [-0.10, 0.81] | 0.35 [-0.11, 0.94] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L41E117 | 99 | 0.2296 | 0.2044 | 0.0051 | 0.02 [-0.14, 2.99] | 0.02 [-0.15, 1.09] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L44E069 | 52 | 0.2133 | 0.155 | 0.1875 | 0.88 [-19.91, 1.22] | 1.21 [-0.10, 2.73] |
| Qwen3-30B-A3B-Base | vt_e2a_neu | L42E115 | 24 | -0.0063 | 0.0285 | 0.0104 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L37E032 | 74 | -0.0002 | -0.0017 | 0.0608 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | vt_a2e_orig | L42E115 | 48 | -0.003 | 0.0027 | -0.0052 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_orig | L39E001 | 182 | 0.1589 | 0.1257 | 0.0515 | 0.32 [-0.87, 0.84] | 0.41 [-1.85, 1.36] |
| Qwen3-30B-A3B-Base | cp_vt_orig | L41E117 | 109 | 0.012 | 0.0042 | -0.0235 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_orig | L44E069 | 98 | 0.0112 | -0.0036 | 0.0561 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_orig | L42E115 | 55 | -0.0044 | -0.0339 | 0.0091 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_neu | L42E123 | 256 | 0.0784 | 0.0478 | -0.0186 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_neu | L41E117 | 139 | 0.0097 | 0.0289 | 0.0261 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_neu | L44E069 | 67 | 0.003 | -0.015 | -0.0075 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_vt_neu | L42E115 | 35 | 0.0101 | -0.0213 | 0.025 | n/a (W < 0.1: acts indirectly) | n/a |
| Qwen3-30B-A3B-Base | cp_nat | L41E117 | 821 | 0.327 | 0.3086 | 0.093 | 0.28 [0.22, 0.35] | 0.30 [0.23, 0.37] |
| Qwen3-30B-A3B-Base | cp_nat | L44E069 | 11 | 0.0384 | -0.0686 | -0.1364 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | f1 | L20E001 | 150 | 0.1337 | 0.0527 | 0.0158 | 0.12 [-0.19, 0.42] | n/a |
| Mixtral-8x7B (BOS) | f1 | L20E000 | 38 | 0.0542 | 0.0238 | 0.0296 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | f1 | L19E002 | 8 | 0.0132 | 0.0081 | 0.0469 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | f1 | L21E001 | 71 | 0.0745 | 0.0045 | 0.0123 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | f1 | L18E001 | 24 | -0.0113 | -0.0175 | 0.0312 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | dir_f1 | L20E001 | 165 | 0.1975 | 0.0838 | -0.0121 | -0.06 [-0.28, 0.15] | n/a |
| Mixtral-8x7B (BOS) | dir_f1 | L20E000 | 14 | 0.0608 | 0.0026 | 0.0446 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | dir_f1 | L21E001 | 70 | 0.1155 | 0.0013 | -0.0107 | -0.09 [-0.69, 0.38] | n/a |
| Mixtral-8x7B (BOS) | dir_opt | L20E000 | 185 | 0.4258 | 0.2481 | 0.1243 | 0.29 [0.19, 0.40] | 0.50 [0.34, 0.67] |
| Mixtral-8x7B (BOS) | f2e | L19E004 | 36 | 0.2181 | 0.0864 | 0.1042 | 0.48 [0.08, 0.86] | n/a |
| Mixtral-8x7B (BOS) | f2e | L20E000 | 10 | 0.0251 | -0.0381 | 0.0 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | f2e | L21E001 | 16 | 0.1489 | 0.0321 | 0.0625 | 0.42 [0.15, 2.20] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E006 | 240 | 0.2876 | 0.1198 | 0.1417 | 0.49 [0.29, 0.82] | 1.18 [0.78, 2.22] |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L20E000 | 238 | 0.2722 | 0.088 | 0.0399 | 0.15 [-0.31, 0.40] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L19E002 | 48 | 0.1367 | -0.0533 | -0.0755 | -0.55 [-2.17, 0.09] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L21E001 | 67 | 0.1565 | 0.0287 | 0.0373 | 0.24 [0.00, 0.79] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_orig | L18E001 | 20 | 0.038 | -0.033 | -0.0375 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E006 | 232 | 0.1773 | 0.056 | 0.1056 | 0.60 [0.17, 1.20] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L20E000 | 220 | 0.2496 | 0.1018 | 0.0756 | 0.30 [-0.22, 0.55] | 0.74 [-1.19, 2.36] |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L19E002 | 17 | 0.215 | 0.0174 | -0.0147 | -0.07 [-0.34, 0.62] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L21E001 | 91 | 0.1334 | 0.0359 | 0.0357 | 0.27 [0.11, 1.17] | n/a |
| Mixtral-8x7B (BOS) | vt_e2a_neu | L18E001 | 28 | 0.0352 | -0.0224 | -0.058 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E006 | 228 | 0.2751 | 0.092 | 0.0471 | 0.17 [-0.04, 0.44] | n/a |
| Mixtral-8x7B (BOS) | cp_vt_orig | L20E000 | 200 | 0.336 | 0.1366 | 0.0688 | 0.20 [-0.19, 0.37] | 0.50 [-2.09, 1.38] |
| Mixtral-8x7B (BOS) | cp_vt_orig | L19E002 | 14 | 0.2276 | -0.089 | -0.0804 | -0.35 [-0.35, -0.35] | n/a |
| Mixtral-8x7B (BOS) | cp_vt_orig | L21E001 | 85 | 0.1456 | 0.0216 | 0.0206 | 0.14 [-0.08, 0.65] | n/a |
| Mixtral-8x7B (BOS) | cp_vt_neu | L19E006 | 178 | 0.176 | 0.0857 | 0.1074 | 0.61 [0.06, 0.95] | n/a |
| Mixtral-8x7B (BOS) | cp_vt_neu | L20E000 | 174 | 0.1434 | -0.041 | -0.0065 | -0.05 [-1.52, 0.45] | n/a |
| Mixtral-8x7B (BOS) | cp_vt_neu | L21E001 | 100 | 0.1026 | 0.0325 | 0.0887 | 0.87 [0.36, 1.43] | n/a |
| Mixtral-8x7B (BOS) | cp_nat | L21E006 | 1180 | 0.2631 | 0.0675 | 0.0907 | 0.34 [0.28, 0.41] | n/a |
| Mixtral-8x7B (BOS) | cp_nat | L20E000 | 1246 | 0.3608 | 0.2145 | 0.1681 | 0.47 [0.41, 0.52] | 0.78 [0.70, 0.88] |
| Mixtral-8x7B (BOS) | cp_nat | L19E002 | 28 | 0.0515 | 0.0018 | 0.0045 | n/a (W < 0.1: acts indirectly) | n/a |
| Mixtral-8x7B (BOS) | cp_nat | L21E001 | 28 | 0.0775 | 0.015 | 0.0134 | n/a (W < 0.1: acts indirectly) | n/a |
