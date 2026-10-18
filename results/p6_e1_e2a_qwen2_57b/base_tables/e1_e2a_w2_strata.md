**e1_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen2_57b | person-name options | True | 18 | 5.37 | 0.279 [0.217, 0.339] | 0.034 [-0.036, 0.098] | 0.39 [0.33, 0.44] |
| qwen2_57b | person-name options | False | 238 | 6.77 | 0.289 [0.273, 0.304] | 0.298 [0.276, 0.320] | 0.38 [0.36, 0.39] |
| qwen2_57b | statement order 1 (entity of A mentioned first) | True | 128 | 6.63 | 0.287 [0.264, 0.308] | 0.287 [0.258, 0.316] | 0.38 [0.36, 0.40] |
| qwen2_57b | statement order 1 (entity of A mentioned first) | False | 128 | 6.71 | 0.290 [0.268, 0.310] | 0.279 [0.245, 0.314] | 0.38 [0.36, 0.40] |
| qwen2_57b | core-776 twin | True | 234 | 6.63 | 0.291 [0.274, 0.306] | 0.292 [0.268, 0.315] | 0.38 [0.37, 0.39] |
| qwen2_57b | core-776 twin | False | 22 | 7.10 | 0.265 [0.215, 0.311] | 0.196 [0.131, 0.264] | 0.34 [0.30, 0.39] |
| qwen2_57b | degree adverb in the blank clause (too / very / more ...) | True | 224 | 6.74 | 0.298 [0.285, 0.312] | 0.325 [0.307, 0.344] | 0.37 [0.36, 0.39] |
| qwen2_57b | degree adverb in the blank clause (too / very / more ...) | False | 32 | 6.15 | 0.211 [0.127, 0.278] | -0.037 [-0.082, 0.005] | 0.40 [0.32, 0.49] |
| qwen2_57b | complementary antonym pair | True | 0 |  |  |  |  |
| qwen2_57b | complementary antonym pair | False | 256 | 6.67 | 0.288 [0.273, 0.303] | 0.283 [0.261, 0.305] | 0.38 [0.36, 0.39] |
| qwen2_57b | item built in both tokenizer families | True | 250 | 6.68 | 0.286 [0.271, 0.301] | 0.285 [0.263, 0.307] | 0.38 [0.36, 0.39] |
| qwen2_57b | item built in both tokenizer families | False | 6 |  |  |  |  |
| qwen2_57b | probe-neutral (both triggers |probe| < 1) | True | 256 | 6.67 | 0.288 [0.273, 0.303] | 0.283 [0.261, 0.305] | 0.38 [0.36, 0.39] |
| qwen2_57b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen2_57b | probe-associative | True | 0 |  |  |  |  |
| qwen2_57b | probe-associative | False | 256 | 6.67 | 0.288 [0.273, 0.303] | 0.283 [0.261, 0.305] | 0.38 [0.36, 0.39] |
| qwen2_57b | probe one-sided | True | 0 |  |  |  |  |
| qwen2_57b | probe one-sided | False | 256 | 6.67 | 0.288 [0.273, 0.303] | 0.283 [0.261, 0.305] | 0.38 [0.36, 0.39] |
| qwen2_57b | probe reverse | True | 0 |  |  |  |  |
| qwen2_57b | probe reverse | False | 256 | 6.67 | 0.288 [0.273, 0.303] | 0.283 [0.261, 0.305] | 0.38 [0.36, 0.39] |
| qwen2_57b | top-1 in both directions | True | 22 | 6.09 | 0.339 [0.313, 0.368] | 0.419 [0.373, 0.466] | 0.40 [0.38, 0.43] |
| qwen2_57b | top-1 in both directions | False | 234 | 6.72 | 0.284 [0.267, 0.300] | 0.271 [0.248, 0.294] | 0.37 [0.36, 0.39] |
