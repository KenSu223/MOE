**cp_nat W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 22 | 7.08 | 0.112 [0.079, 0.146] | 0.072 [0.041, 0.104] | 0.35 [0.31, 0.39] |
| Qwen3-30B-A3B-Base | person-name options | False | 264 | 6.22 | 0.160 [0.150, 0.169] | 0.056 [0.048, 0.063] | 0.19 [0.18, 0.22] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 124 | 5.75 | 0.153 [0.138, 0.168] | 0.051 [0.040, 0.063] | 0.19 [0.17, 0.23] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 162 | 6.69 | 0.157 [0.146, 0.168] | 0.061 [0.052, 0.071] | 0.22 [0.21, 0.25] |
| Qwen3-30B-A3B-Base | probe-associative | True | 22 | 6.95 | 0.150 [0.118, 0.183] | 0.043 [0.016, 0.071] | 0.29 [0.25, 0.34] |
| Qwen3-30B-A3B-Base | probe-associative | False | 264 | 6.23 | 0.156 [0.147, 0.165] | 0.058 [0.051, 0.066] | 0.20 [0.18, 0.22] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 40 | 7.47 | 0.165 [0.144, 0.188] | 0.048 [0.031, 0.065] | 0.21 [0.20, 0.27] |
| Qwen3-30B-A3B-Base | probe one-sided | False | 246 | 6.09 | 0.154 [0.144, 0.164] | 0.059 [0.051, 0.067] | 0.20 [0.19, 0.23] |
| Qwen3-30B-A3B-Base | probe reverse | True | 90 | 6.41 | 0.161 [0.146, 0.175] | 0.076 [0.064, 0.089] | 0.23 [0.21, 0.26] |
| Qwen3-30B-A3B-Base | probe reverse | False | 196 | 6.22 | 0.153 [0.141, 0.164] | 0.048 [0.039, 0.057] | 0.20 [0.18, 0.23] |
| Qwen3-30B-A3B-Base | Phase-3 local-context assoc | True | 34 | 7.61 | 0.160 [0.135, 0.188] | 0.044 [0.030, 0.059] | 0.27 [0.24, 0.32] |
| Qwen3-30B-A3B-Base | Phase-3 local-context assoc | False | 252 | 6.10 | 0.155 [0.145, 0.164] | 0.059 [0.051, 0.068] | 0.20 [0.18, 0.22] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 34 | 6.60 | 0.172 [0.144, 0.199] | 0.057 [0.042, 0.071] | 0.19 [0.16, 0.25] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 252 | 6.24 | 0.153 [0.144, 0.163] | 0.057 [0.049, 0.065] | 0.21 [0.19, 0.23] |
| Mixtral-8x7B, BOS | person-name options | True | 36 | 7.23 | 0.087 [0.070, 0.105] | 0.140 [0.111, 0.169] | 0.51 [0.47, 0.55] |
| Mixtral-8x7B, BOS | person-name options | False | 384 | 7.12 | 0.164 [0.157, 0.170] | 0.159 [0.150, 0.168] | 0.37 [0.36, 0.38] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 172 | 7.10 | 0.156 [0.147, 0.166] | 0.163 [0.150, 0.176] | 0.38 [0.36, 0.39] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 248 | 7.16 | 0.158 [0.149, 0.166] | 0.154 [0.143, 0.166] | 0.39 [0.38, 0.40] |
| Mixtral-8x7B, BOS | probe-associative | True | 6 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 414 | 7.12 | 0.157 [0.150, 0.164] | 0.158 [0.149, 0.167] | 0.38 [0.37, 0.39] |
| Mixtral-8x7B, BOS | probe one-sided | True | 84 | 7.56 | 0.156 [0.140, 0.172] | 0.131 [0.116, 0.146] | 0.38 [0.37, 0.40] |
| Mixtral-8x7B, BOS | probe one-sided | False | 336 | 7.02 | 0.157 [0.150, 0.165] | 0.165 [0.155, 0.175] | 0.38 [0.37, 0.39] |
| Mixtral-8x7B, BOS | probe reverse | True | 124 | 6.95 | 0.166 [0.155, 0.178] | 0.175 [0.159, 0.192] | 0.37 [0.36, 0.38] |
| Mixtral-8x7B, BOS | probe reverse | False | 296 | 7.21 | 0.154 [0.146, 0.161] | 0.150 [0.141, 0.161] | 0.39 [0.38, 0.40] |
| Mixtral-8x7B, BOS | Phase-3 local-context assoc | True | 44 | 7.49 | 0.187 [0.171, 0.202] | 0.114 [0.093, 0.135] | 0.37 [0.35, 0.40] |
| Mixtral-8x7B, BOS | Phase-3 local-context assoc | False | 376 | 7.09 | 0.153 [0.147, 0.161] | 0.163 [0.154, 0.172] | 0.38 [0.37, 0.39] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 263 | 7.43 | 0.156 [0.148, 0.164] | 0.170 [0.159, 0.182] | 0.38 [0.37, 0.39] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 157 | 6.63 | 0.160 [0.148, 0.171] | 0.133 [0.122, 0.145] | 0.38 [0.37, 0.39] |
