**e1_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen2_7b | person-name options | True | 68 | 6.34 | 0.383 [0.334, 0.431] | 0.278 [0.232, 0.330] | 0.26 [0.24, 0.28] |
| qwen2_7b | person-name options | False | 188 | 5.35 | 0.240 [0.225, 0.256] | 0.166 [0.141, 0.192] | 0.42 [0.41, 0.43] |
| qwen2_7b | statement order 1 (entity of A mentioned first) | True | 128 | 5.77 | 0.288 [0.260, 0.318] | 0.202 [0.167, 0.239] | 0.36 [0.34, 0.39] |
| qwen2_7b | statement order 1 (entity of A mentioned first) | False | 128 | 5.46 | 0.278 [0.251, 0.306] | 0.197 [0.166, 0.228] | 0.38 [0.36, 0.40] |
| qwen2_7b | core-776 twin | True | 186 | 5.36 | 0.278 [0.256, 0.299] | 0.171 [0.144, 0.200] | 0.40 [0.39, 0.42] |
| qwen2_7b | core-776 twin | False | 70 | 6.30 | 0.296 [0.255, 0.339] | 0.263 [0.223, 0.307] | 0.30 [0.28, 0.33] |
| qwen2_7b | degree adverb in the blank clause (too / very / more ...) | True | 164 | 5.52 | 0.275 [0.256, 0.294] | 0.149 [0.125, 0.176] | 0.41 [0.39, 0.43] |
| qwen2_7b | degree adverb in the blank clause (too / very / more ...) | False | 92 | 5.78 | 0.298 [0.255, 0.341] | 0.286 [0.245, 0.330] | 0.31 [0.29, 0.33] |
| qwen2_7b | complementary antonym pair | True | 2 |  |  |  |  |
| qwen2_7b | complementary antonym pair | False | 254 | 5.62 | 0.282 [0.262, 0.302] | 0.197 [0.174, 0.222] | 0.37 [0.36, 0.39] |
| qwen2_7b | item built in both tokenizer families | True | 204 | 5.39 | 0.279 [0.258, 0.301] | 0.180 [0.154, 0.206] | 0.40 [0.38, 0.41] |
| qwen2_7b | item built in both tokenizer families | False | 52 | 6.50 | 0.298 [0.250, 0.347] | 0.264 [0.211, 0.320] | 0.30 [0.27, 0.33] |
| qwen2_7b | probe-neutral (both triggers |probe| < 1) | True | 256 | 5.61 | 0.283 [0.264, 0.303] | 0.200 [0.176, 0.224] | 0.37 [0.36, 0.39] |
| qwen2_7b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen2_7b | probe-associative | True | 0 |  |  |  |  |
| qwen2_7b | probe-associative | False | 256 | 5.61 | 0.283 [0.264, 0.303] | 0.200 [0.176, 0.224] | 0.37 [0.36, 0.39] |
| qwen2_7b | probe one-sided | True | 0 |  |  |  |  |
| qwen2_7b | probe one-sided | False | 256 | 5.61 | 0.283 [0.264, 0.303] | 0.200 [0.176, 0.224] | 0.37 [0.36, 0.39] |
| qwen2_7b | probe reverse | True | 0 |  |  |  |  |
| qwen2_7b | probe reverse | False | 256 | 5.61 | 0.283 [0.264, 0.303] | 0.200 [0.176, 0.224] | 0.37 [0.36, 0.39] |
| qwen2_7b | top-1 in both directions | True | 125 | 5.35 | 0.264 [0.246, 0.281] | 0.116 [0.097, 0.138] | 0.44 [0.42, 0.46] |
| qwen2_7b | top-1 in both directions | False | 131 | 5.86 | 0.300 [0.267, 0.333] | 0.272 [0.236, 0.310] | 0.31 [0.29, 0.33] |
