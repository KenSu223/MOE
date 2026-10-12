**vt_e2a_neu W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 256 | 6.32 | 0.147 [0.134, 0.160] | 0.059 [0.052, 0.066] | 0.25 [0.23, 0.27] |
| Qwen3-30B-A3B-Base | person-name options | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 197 | 6.14 | 0.142 [0.125, 0.158] | 0.056 [0.049, 0.064] | 0.24 [0.22, 0.27] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 59 | 6.94 | 0.163 [0.140, 0.185] | 0.067 [0.051, 0.084] | 0.27 [0.22, 0.32] |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 256 | 6.32 | 0.147 [0.134, 0.160] | 0.059 [0.052, 0.066] | 0.25 [0.23, 0.27] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 19 | 7.81 | 0.151 [0.120, 0.186] | 0.063 [0.037, 0.091] | 0.26 [0.19, 0.35] |
| Qwen3-30B-A3B-Base | probe one-sided | False | 237 | 6.21 | 0.146 [0.132, 0.160] | 0.059 [0.051, 0.066] | 0.25 [0.23, 0.27] |
| Qwen3-30B-A3B-Base | probe reverse | True | 40 | 6.52 | 0.169 [0.141, 0.198] | 0.070 [0.048, 0.090] | 0.28 [0.23, 0.34] |
| Qwen3-30B-A3B-Base | probe reverse | False | 216 | 6.29 | 0.143 [0.127, 0.157] | 0.057 [0.050, 0.065] | 0.24 [0.22, 0.27] |
| Mixtral-8x7B, BOS | person-name options | True | 256 | 5.70 | 0.178 [0.169, 0.187] | 0.152 [0.143, 0.161] | 0.40 [0.39, 0.41] |
| Mixtral-8x7B, BOS | person-name options | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 228 | 5.63 | 0.181 [0.172, 0.190] | 0.154 [0.144, 0.164] | 0.40 [0.39, 0.41] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 28 | 6.32 | 0.156 [0.126, 0.185] | 0.131 [0.110, 0.153] | 0.41 [0.38, 0.44] |
| Mixtral-8x7B, BOS | probe-associative | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 256 | 5.70 | 0.178 [0.169, 0.187] | 0.152 [0.143, 0.161] | 0.40 [0.39, 0.41] |
| Mixtral-8x7B, BOS | probe one-sided | True | 16 | 6.15 | 0.135 [0.101, 0.168] | 0.134 [0.103, 0.171] | 0.41 [0.37, 0.46] |
| Mixtral-8x7B, BOS | probe one-sided | False | 240 | 5.67 | 0.181 [0.172, 0.190] | 0.153 [0.144, 0.163] | 0.40 [0.39, 0.41] |
| Mixtral-8x7B, BOS | probe reverse | True | 12 | 6.56 | 0.181 [0.130, 0.225] | 0.128 [0.102, 0.152] | 0.41 [0.37, 0.46] |
| Mixtral-8x7B, BOS | probe reverse | False | 244 | 5.66 | 0.178 [0.169, 0.187] | 0.153 [0.143, 0.162] | 0.40 [0.39, 0.41] |
