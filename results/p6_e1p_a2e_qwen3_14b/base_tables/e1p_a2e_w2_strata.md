**e1p_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_14b | person-name options | True | 28 | 12.72 | 0.228 [0.216, 0.241] | 0.428 [0.395, 0.458] | 0.75 [0.74, 0.76] |
| qwen3_14b | person-name options | False | 228 | 8.15 | 0.197 [0.193, 0.202] | 0.294 [0.278, 0.311] | 0.64 [0.63, 0.65] |
| qwen3_14b | statement order 1 (entity of A mentioned first) | True | 128 | 8.67 | 0.203 [0.196, 0.210] | 0.309 [0.286, 0.331] | 0.66 [0.65, 0.68] |
| qwen3_14b | statement order 1 (entity of A mentioned first) | False | 128 | 8.62 | 0.202 [0.195, 0.208] | 0.323 [0.299, 0.346] | 0.67 [0.65, 0.68] |
| qwen3_14b | core-776 twin | True | 256 | 8.65 | 0.202 [0.197, 0.207] | 0.316 [0.300, 0.332] | 0.66 [0.65, 0.68] |
| qwen3_14b | core-776 twin | False | 0 |  |  |  |  |
| qwen3_14b | degree adverb in the blank clause (too / very / more ...) | True | 172 | 8.36 | 0.199 [0.193, 0.205] | 0.278 [0.263, 0.292] | 0.64 [0.63, 0.65] |
| qwen3_14b | degree adverb in the blank clause (too / very / more ...) | False | 84 | 9.23 | 0.209 [0.201, 0.217] | 0.387 [0.356, 0.418] | 0.70 [0.69, 0.71] |
| qwen3_14b | complementary antonym pair | True | 2 |  |  |  |  |
| qwen3_14b | complementary antonym pair | False | 254 | 8.67 | 0.202 [0.198, 0.207] | 0.317 [0.300, 0.334] | 0.66 [0.65, 0.68] |
| qwen3_14b | item built in both tokenizer families | True | 254 | 8.60 | 0.202 [0.198, 0.207] | 0.314 [0.298, 0.329] | 0.66 [0.65, 0.67] |
| qwen3_14b | item built in both tokenizer families | False | 2 |  |  |  |  |
| qwen3_14b | probe-neutral (both triggers |probe| < 1) | True | 256 | 8.65 | 0.202 [0.197, 0.207] | 0.316 [0.300, 0.332] | 0.66 [0.65, 0.68] |
| qwen3_14b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen3_14b | probe-associative | True | 0 |  |  |  |  |
| qwen3_14b | probe-associative | False | 256 | 8.65 | 0.202 [0.197, 0.207] | 0.316 [0.300, 0.332] | 0.66 [0.65, 0.68] |
| qwen3_14b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_14b | probe one-sided | False | 256 | 8.65 | 0.202 [0.197, 0.207] | 0.316 [0.300, 0.332] | 0.66 [0.65, 0.68] |
| qwen3_14b | probe reverse | True | 0 |  |  |  |  |
| qwen3_14b | probe reverse | False | 256 | 8.65 | 0.202 [0.197, 0.207] | 0.316 [0.300, 0.332] | 0.66 [0.65, 0.68] |
| qwen3_14b | top-1 in both directions | True | 255 | 8.65 | 0.202 [0.197, 0.207] | 0.316 [0.300, 0.333] | 0.66 [0.65, 0.68] |
| qwen3_14b | top-1 in both directions | False | 1 |  |  |  |  |
