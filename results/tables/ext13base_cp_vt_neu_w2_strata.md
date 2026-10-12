**cp_vt_neu W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 256 | 5.51 | 0.136 [0.125, 0.148] | 0.069 [0.061, 0.077] | 0.28 [0.26, 0.31] |
| Qwen3-30B-A3B-Base | person-name options | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 186 | 5.36 | 0.133 [0.119, 0.146] | 0.068 [0.059, 0.078] | 0.27 [0.25, 0.30] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 70 | 5.90 | 0.145 [0.125, 0.165] | 0.070 [0.055, 0.085] | 0.32 [0.28, 0.36] |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 256 | 5.51 | 0.136 [0.125, 0.148] | 0.069 [0.061, 0.077] | 0.28 [0.26, 0.31] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 22 | 6.30 | 0.177 [0.152, 0.206] | 0.074 [0.053, 0.095] | 0.32 [0.25, 0.40] |
| Qwen3-30B-A3B-Base | probe one-sided | False | 234 | 5.43 | 0.132 [0.120, 0.144] | 0.068 [0.060, 0.077] | 0.28 [0.26, 0.30] |
| Qwen3-30B-A3B-Base | probe reverse | True | 48 | 5.71 | 0.129 [0.102, 0.155] | 0.068 [0.048, 0.089] | 0.32 [0.28, 0.36] |
| Qwen3-30B-A3B-Base | probe reverse | False | 208 | 5.46 | 0.138 [0.126, 0.150] | 0.069 [0.060, 0.077] | 0.28 [0.26, 0.30] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 17 | 6.82 | 0.059 [0.029, 0.101] | 0.057 [0.025, 0.085] | 0.45 [0.39, 0.49] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 239 | 5.41 | 0.143 [0.132, 0.154] | 0.070 [0.061, 0.078] | 0.28 [0.25, 0.30] |
| Mixtral-8x7B, BOS | person-name options | True | 256 | 5.67 | 0.155 [0.150, 0.161] | 0.165 [0.154, 0.177] | 0.41 [0.40, 0.41] |
| Mixtral-8x7B, BOS | person-name options | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 234 | 5.65 | 0.156 [0.150, 0.162] | 0.168 [0.157, 0.181] | 0.41 [0.40, 0.41] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 22 | 5.80 | 0.145 [0.129, 0.160] | 0.133 [0.099, 0.172] | 0.42 [0.39, 0.45] |
| Mixtral-8x7B, BOS | probe-associative | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 256 | 5.67 | 0.155 [0.150, 0.161] | 0.165 [0.154, 0.177] | 0.41 [0.40, 0.41] |
| Mixtral-8x7B, BOS | probe one-sided | True | 10 | 5.89 | 0.154 [0.144, 0.167] | 0.148 [0.085, 0.229] | 0.39 [0.35, 0.42] |
| Mixtral-8x7B, BOS | probe one-sided | False | 246 | 5.66 | 0.156 [0.150, 0.161] | 0.166 [0.154, 0.178] | 0.41 [0.40, 0.42] |
| Mixtral-8x7B, BOS | probe reverse | True | 12 | 5.73 | 0.137 [0.106, 0.164] | 0.119 [0.093, 0.145] | 0.45 [0.40, 0.49] |
| Mixtral-8x7B, BOS | probe reverse | False | 244 | 5.66 | 0.156 [0.151, 0.162] | 0.167 [0.155, 0.180] | 0.40 [0.40, 0.41] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 148 | 6.01 | 0.159 [0.153, 0.164] | 0.173 [0.156, 0.191] | 0.41 [0.40, 0.42] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 108 | 5.19 | 0.151 [0.139, 0.162] | 0.152 [0.140, 0.164] | 0.40 [0.39, 0.42] |
