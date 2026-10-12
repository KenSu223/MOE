**f2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 9 |  |  |  |  |
| Qwen3-30B-A3B-Base | person-name options | False | 55 | 8.37 | 0.081 [0.063, 0.100] | 0.206 [0.178, 0.236] | 0.67 [0.60, 0.72] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 64 | 8.22 | 0.071 [0.054, 0.089] | 0.205 [0.179, 0.231] | 0.68 [0.62, 0.72] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 64 | 8.22 | 0.071 [0.054, 0.089] | 0.205 [0.179, 0.231] | 0.68 [0.62, 0.72] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe one-sided | False | 64 | 8.22 | 0.071 [0.054, 0.089] | 0.205 [0.179, 0.231] | 0.68 [0.62, 0.72] |
| Qwen3-30B-A3B-Base | probe reverse | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe reverse | False | 64 | 8.22 | 0.071 [0.054, 0.089] | 0.205 [0.179, 0.231] | 0.68 [0.62, 0.72] |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | True | 3 |  |  |  |  |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | False | 61 | 8.24 | 0.070 [0.051, 0.088] | 0.208 [0.181, 0.235] | 0.69 [0.62, 0.74] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 41 | 8.53 | 0.078 [0.058, 0.099] | 0.211 [0.182, 0.239] | 0.64 [0.58, 0.70] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 23 | 7.67 | 0.058 [0.026, 0.089] | 0.194 [0.143, 0.250] | 0.73 [0.62, 0.79] |
| Mixtral-8x7B, BOS | person-name options | True | 5 |  |  |  |  |
| Mixtral-8x7B, BOS | person-name options | False | 31 | 7.46 | 0.087 [0.070, 0.104] | 0.206 [0.181, 0.232] | 0.67 [0.63, 0.70] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 36 | 7.57 | 0.084 [0.068, 0.099] | 0.247 [0.206, 0.294] | 0.67 [0.64, 0.70] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 36 | 7.57 | 0.084 [0.068, 0.099] | 0.247 [0.206, 0.294] | 0.67 [0.64, 0.70] |
| Mixtral-8x7B, BOS | probe one-sided | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe one-sided | False | 36 | 7.57 | 0.084 [0.068, 0.099] | 0.247 [0.206, 0.294] | 0.67 [0.64, 0.70] |
| Mixtral-8x7B, BOS | probe reverse | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe reverse | False | 36 | 7.57 | 0.084 [0.068, 0.099] | 0.247 [0.206, 0.294] | 0.67 [0.64, 0.70] |
| Mixtral-8x7B, BOS | AfLite survivor (train_debiased) | True | 2 |  |  |  |  |
| Mixtral-8x7B, BOS | AfLite survivor (train_debiased) | False | 34 | 7.68 | 0.084 [0.068, 0.099] | 0.249 [0.206, 0.295] | 0.67 [0.64, 0.70] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 26 | 7.38 | 0.080 [0.063, 0.097] | 0.237 [0.196, 0.282] | 0.66 [0.62, 0.70] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 10 | 8.07 | 0.092 [0.058, 0.124] | 0.269 [0.175, 0.376] | 0.68 [0.63, 0.73] |
