**e3_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen2_57b | person-name options | True | 17 | 7.98 | 0.283 [0.226, 0.336] | 0.073 [0.025, 0.117] | 0.32 [0.26, 0.38] |
| qwen2_57b | person-name options | False | 239 | 9.18 | 0.337 [0.319, 0.353] | 0.153 [0.137, 0.169] | 0.20 [0.19, 0.21] |
| qwen2_57b | statement order 1 (entity of A mentioned first) | True | 169 | 9.28 | 0.335 [0.314, 0.354] | 0.137 [0.118, 0.156] | 0.21 [0.19, 0.23] |
| qwen2_57b | statement order 1 (entity of A mentioned first) | False | 87 | 8.74 | 0.331 [0.303, 0.357] | 0.171 [0.141, 0.201] | 0.21 [0.19, 0.23] |
| qwen2_57b | core-776 twin | True | 256 | 9.10 | 0.334 [0.317, 0.349] | 0.148 [0.133, 0.164] | 0.21 [0.19, 0.22] |
| qwen2_57b | core-776 twin | False | 0 |  |  |  |  |
| qwen2_57b | degree adverb in the blank clause (too / very / more ...) | True | 149 | 8.45 | 0.385 [0.365, 0.403] | 0.223 [0.205, 0.242] | 0.24 [0.22, 0.25] |
| qwen2_57b | degree adverb in the blank clause (too / very / more ...) | False | 107 | 10.00 | 0.273 [0.251, 0.294] | 0.060 [0.045, 0.074] | 0.18 [0.16, 0.20] |
| qwen2_57b | complementary antonym pair | True | 2 |  |  |  |  |
| qwen2_57b | complementary antonym pair | False | 254 | 9.13 | 0.334 [0.318, 0.350] | 0.148 [0.133, 0.164] | 0.21 [0.19, 0.22] |
| qwen2_57b | item built in both tokenizer families | True | 254 | 9.10 | 0.335 [0.318, 0.351] | 0.149 [0.132, 0.165] | 0.21 [0.19, 0.22] |
| qwen2_57b | item built in both tokenizer families | False | 2 |  |  |  |  |
| qwen2_57b | probe-neutral (both triggers |probe| < 1) | True | 256 | 9.10 | 0.334 [0.317, 0.349] | 0.148 [0.133, 0.164] | 0.21 [0.19, 0.22] |
| qwen2_57b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen2_57b | probe-associative | True | 0 |  |  |  |  |
| qwen2_57b | probe-associative | False | 256 | 9.10 | 0.334 [0.317, 0.349] | 0.148 [0.133, 0.164] | 0.21 [0.19, 0.22] |
| qwen2_57b | probe one-sided | True | 0 |  |  |  |  |
| qwen2_57b | probe one-sided | False | 256 | 9.10 | 0.334 [0.317, 0.349] | 0.148 [0.133, 0.164] | 0.21 [0.19, 0.22] |
| qwen2_57b | probe reverse | True | 0 |  |  |  |  |
| qwen2_57b | probe reverse | False | 256 | 9.10 | 0.334 [0.317, 0.349] | 0.148 [0.133, 0.164] | 0.21 [0.19, 0.22] |
| qwen2_57b | top-1 in both directions | True | 73 | 9.29 | 0.317 [0.296, 0.338] | 0.105 [0.080, 0.132] | 0.18 [0.16, 0.20] |
| qwen2_57b | top-1 in both directions | False | 183 | 9.02 | 0.340 [0.319, 0.361] | 0.166 [0.146, 0.185] | 0.22 [0.20, 0.24] |
