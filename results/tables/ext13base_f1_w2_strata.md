**f1 W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 132 | 4.06 | 0.181 [0.160, 0.205] | 0.400 [0.373, 0.428] | 0.79 [0.77, 0.81] |
| Qwen3-30B-A3B-Base | person-name options | False | 124 | 5.31 | 0.068 [0.055, 0.082] | 0.215 [0.192, 0.239] | 0.73 [0.71, 0.75] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 256 | 4.67 | 0.119 [0.104, 0.134] | 0.298 [0.276, 0.320] | 0.76 [0.75, 0.78] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 256 | 4.67 | 0.119 [0.104, 0.134] | 0.298 [0.276, 0.320] | 0.76 [0.75, 0.78] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe one-sided | False | 256 | 4.67 | 0.119 [0.104, 0.134] | 0.298 [0.276, 0.320] | 0.76 [0.75, 0.78] |
| Qwen3-30B-A3B-Base | probe reverse | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe reverse | False | 256 | 4.67 | 0.119 [0.104, 0.134] | 0.298 [0.276, 0.320] | 0.76 [0.75, 0.78] |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | True | 15 | 4.28 | 0.084 [0.045, 0.131] | 0.205 [0.145, 0.276] | 0.68 [0.60, 0.74] |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | False | 241 | 4.69 | 0.121 [0.106, 0.136] | 0.303 [0.281, 0.326] | 0.77 [0.75, 0.78] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 201 | 4.68 | 0.137 [0.121, 0.156] | 0.318 [0.293, 0.343] | 0.77 [0.75, 0.78] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 55 | 4.60 | 0.048 [0.030, 0.069] | 0.224 [0.186, 0.265] | 0.73 [0.69, 0.75] |
| Mixtral-8x7B, BOS | person-name options | True | 23 | 3.55 | 0.056 [0.035, 0.077] | 0.317 [0.236, 0.407] | 0.67 [0.63, 0.72] |
| Mixtral-8x7B, BOS | person-name options | False | 179 | 4.42 | 0.141 [0.128, 0.153] | 0.274 [0.257, 0.292] | 0.54 [0.52, 0.55] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 202 | 4.32 | 0.133 [0.121, 0.145] | 0.278 [0.261, 0.296] | 0.55 [0.53, 0.57] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 202 | 4.32 | 0.133 [0.121, 0.145] | 0.278 [0.261, 0.296] | 0.55 [0.53, 0.57] |
| Mixtral-8x7B, BOS | probe one-sided | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe one-sided | False | 202 | 4.32 | 0.133 [0.121, 0.145] | 0.278 [0.261, 0.296] | 0.55 [0.53, 0.57] |
| Mixtral-8x7B, BOS | probe reverse | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe reverse | False | 202 | 4.32 | 0.133 [0.121, 0.145] | 0.278 [0.261, 0.296] | 0.55 [0.53, 0.57] |
| Mixtral-8x7B, BOS | AfLite survivor (train_debiased) | True | 8 |  |  |  |  |
| Mixtral-8x7B, BOS | AfLite survivor (train_debiased) | False | 194 | 4.31 | 0.133 [0.121, 0.146] | 0.279 [0.261, 0.297] | 0.55 [0.53, 0.57] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 117 | 4.27 | 0.125 [0.109, 0.141] | 0.296 [0.273, 0.321] | 0.56 [0.53, 0.58] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 85 | 4.39 | 0.143 [0.125, 0.160] | 0.255 [0.229, 0.280] | 0.54 [0.51, 0.57] |
