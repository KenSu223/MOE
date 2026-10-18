**e1p_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen2_7b | person-name options | True | 172 | 6.74 | 0.352 [0.345, 0.358] | 0.464 [0.453, 0.475] | 0.64 [0.63, 0.64] |
| qwen2_7b | person-name options | False | 84 | 3.81 | 0.239 [0.222, 0.256] | 0.317 [0.294, 0.340] | 0.54 [0.53, 0.55] |
| qwen2_7b | statement order 1 (entity of A mentioned first) | True | 128 | 5.90 | 0.327 [0.316, 0.337] | 0.435 [0.418, 0.452] | 0.62 [0.61, 0.63] |
| qwen2_7b | statement order 1 (entity of A mentioned first) | False | 128 | 5.66 | 0.328 [0.315, 0.339] | 0.429 [0.413, 0.446] | 0.61 [0.60, 0.62] |
| qwen2_7b | core-776 twin | True | 76 | 5.08 | 0.296 [0.276, 0.315] | 0.393 [0.361, 0.424] | 0.60 [0.59, 0.62] |
| qwen2_7b | core-776 twin | False | 180 | 6.07 | 0.338 [0.330, 0.346] | 0.446 [0.434, 0.459] | 0.62 [0.62, 0.63] |
| qwen2_7b | degree adverb in the blank clause (too / very / more ...) | True | 98 | 5.63 | 0.302 [0.286, 0.316] | 0.409 [0.387, 0.431] | 0.61 [0.60, 0.62] |
| qwen2_7b | degree adverb in the blank clause (too / very / more ...) | False | 158 | 5.87 | 0.343 [0.333, 0.351] | 0.446 [0.432, 0.460] | 0.62 [0.62, 0.63] |
| qwen2_7b | complementary antonym pair | True | 2 |  |  |  |  |
| qwen2_7b | complementary antonym pair | False | 254 | 5.77 | 0.327 [0.318, 0.335] | 0.431 [0.418, 0.443] | 0.62 [0.61, 0.63] |
| qwen2_7b | item built in both tokenizer families | True | 130 | 5.64 | 0.320 [0.307, 0.332] | 0.421 [0.401, 0.439] | 0.62 [0.61, 0.63] |
| qwen2_7b | item built in both tokenizer families | False | 126 | 5.92 | 0.334 [0.323, 0.344] | 0.443 [0.428, 0.458] | 0.62 [0.61, 0.63] |
| qwen2_7b | probe-neutral (both triggers |probe| < 1) | True | 256 | 5.78 | 0.327 [0.319, 0.335] | 0.432 [0.420, 0.445] | 0.62 [0.61, 0.63] |
| qwen2_7b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen2_7b | probe-associative | True | 0 |  |  |  |  |
| qwen2_7b | probe-associative | False | 256 | 5.78 | 0.327 [0.319, 0.335] | 0.432 [0.420, 0.445] | 0.62 [0.61, 0.63] |
| qwen2_7b | probe one-sided | True | 0 |  |  |  |  |
| qwen2_7b | probe one-sided | False | 256 | 5.78 | 0.327 [0.319, 0.335] | 0.432 [0.420, 0.445] | 0.62 [0.61, 0.63] |
| qwen2_7b | probe reverse | True | 0 |  |  |  |  |
| qwen2_7b | probe reverse | False | 256 | 5.78 | 0.327 [0.319, 0.335] | 0.432 [0.420, 0.445] | 0.62 [0.61, 0.63] |
| qwen2_7b | top-1 in both directions | True | 256 | 5.78 | 0.327 [0.319, 0.335] | 0.432 [0.420, 0.445] | 0.62 [0.61, 0.63] |
| qwen2_7b | top-1 in both directions | False | 0 |  |  |  |  |
