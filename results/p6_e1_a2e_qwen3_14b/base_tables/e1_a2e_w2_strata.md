**e1_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_14b | person-name options | True | 32 | 12.59 | 0.236 [0.224, 0.249] | 0.369 [0.354, 0.384] | 0.73 [0.72, 0.73] |
| qwen3_14b | person-name options | False | 224 | 9.07 | 0.194 [0.190, 0.199] | 0.297 [0.291, 0.303] | 0.64 [0.63, 0.65] |
| qwen3_14b | statement order 1 (entity of A mentioned first) | True | 128 | 9.48 | 0.202 [0.195, 0.209] | 0.311 [0.301, 0.320] | 0.66 [0.65, 0.67] |
| qwen3_14b | statement order 1 (entity of A mentioned first) | False | 128 | 9.55 | 0.201 [0.195, 0.208] | 0.307 [0.297, 0.317] | 0.66 [0.65, 0.67] |
| qwen3_14b | core-776 twin | True | 256 | 9.51 | 0.201 [0.197, 0.206] | 0.309 [0.302, 0.315] | 0.66 [0.65, 0.67] |
| qwen3_14b | core-776 twin | False | 0 |  |  |  |  |
| qwen3_14b | degree adverb in the blank clause (too / very / more ...) | True | 178 | 9.37 | 0.198 [0.192, 0.204] | 0.309 [0.300, 0.317] | 0.65 [0.64, 0.66] |
| qwen3_14b | degree adverb in the blank clause (too / very / more ...) | False | 78 | 9.83 | 0.209 [0.200, 0.218] | 0.309 [0.298, 0.319] | 0.68 [0.67, 0.70] |
| qwen3_14b | complementary antonym pair | True | 2 |  |  |  |  |
| qwen3_14b | complementary antonym pair | False | 254 | 9.50 | 0.202 [0.197, 0.207] | 0.310 [0.303, 0.316] | 0.66 [0.65, 0.67] |
| qwen3_14b | item built in both tokenizer families | True | 254 | 9.47 | 0.201 [0.197, 0.206] | 0.308 [0.301, 0.315] | 0.66 [0.65, 0.67] |
| qwen3_14b | item built in both tokenizer families | False | 2 |  |  |  |  |
| qwen3_14b | probe-neutral (both triggers |probe| < 1) | True | 256 | 9.51 | 0.201 [0.197, 0.206] | 0.309 [0.302, 0.315] | 0.66 [0.65, 0.67] |
| qwen3_14b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen3_14b | probe-associative | True | 0 |  |  |  |  |
| qwen3_14b | probe-associative | False | 256 | 9.51 | 0.201 [0.197, 0.206] | 0.309 [0.302, 0.315] | 0.66 [0.65, 0.67] |
| qwen3_14b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_14b | probe one-sided | False | 256 | 9.51 | 0.201 [0.197, 0.206] | 0.309 [0.302, 0.315] | 0.66 [0.65, 0.67] |
| qwen3_14b | probe reverse | True | 0 |  |  |  |  |
| qwen3_14b | probe reverse | False | 256 | 9.51 | 0.201 [0.197, 0.206] | 0.309 [0.302, 0.315] | 0.66 [0.65, 0.67] |
| qwen3_14b | top-1 in both directions | True | 256 | 9.51 | 0.201 [0.197, 0.206] | 0.309 [0.302, 0.315] | 0.66 [0.65, 0.67] |
| qwen3_14b | top-1 in both directions | False | 0 |  |  |  |  |
