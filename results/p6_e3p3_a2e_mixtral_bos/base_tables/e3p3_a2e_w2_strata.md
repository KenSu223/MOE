**e3p3_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | person-name options | True | 25 | 3.92 | 0.045 [0.026, 0.065] | 0.540 [0.508, 0.572] | 0.64 [0.62, 0.65] |
| Mixtral-8x7B, BOS | person-name options | False | 135 | 5.48 | 0.149 [0.137, 0.160] | 0.193 [0.177, 0.209] | 0.53 [0.52, 0.55] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | True | 109 | 5.13 | 0.131 [0.116, 0.145] | 0.248 [0.220, 0.281] | 0.55 [0.54, 0.57] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | False | 51 | 5.46 | 0.148 [0.130, 0.167] | 0.205 [0.180, 0.231] | 0.53 [0.51, 0.55] |
| Mixtral-8x7B, BOS | core-776 twin | True | 100 | 5.57 | 0.147 [0.135, 0.160] | 0.211 [0.190, 0.232] | 0.54 [0.53, 0.56] |
| Mixtral-8x7B, BOS | core-776 twin | False | 60 | 4.68 | 0.115 [0.093, 0.137] | 0.279 [0.231, 0.330] | 0.56 [0.53, 0.59] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | True | 98 | 5.74 | 0.148 [0.135, 0.162] | 0.210 [0.188, 0.234] | 0.54 [0.52, 0.55] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | False | 62 | 4.45 | 0.112 [0.094, 0.131] | 0.281 [0.237, 0.324] | 0.57 [0.54, 0.59] |
| Mixtral-8x7B, BOS | complementary antonym pair | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | complementary antonym pair | False | 160 | 5.24 | 0.136 [0.125, 0.148] | 0.233 [0.212, 0.256] | 0.55 [0.53, 0.56] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | True | 160 | 5.24 | 0.136 [0.125, 0.148] | 0.233 [0.212, 0.256] | 0.55 [0.53, 0.56] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 160 | 5.24 | 0.136 [0.125, 0.148] | 0.233 [0.212, 0.256] | 0.55 [0.53, 0.56] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 160 | 5.24 | 0.136 [0.125, 0.148] | 0.233 [0.212, 0.256] | 0.55 [0.53, 0.56] |
| Mixtral-8x7B, BOS | probe one-sided | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe one-sided | False | 160 | 5.24 | 0.136 [0.125, 0.148] | 0.233 [0.212, 0.256] | 0.55 [0.53, 0.56] |
| Mixtral-8x7B, BOS | probe reverse | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe reverse | False | 160 | 5.24 | 0.136 [0.125, 0.148] | 0.233 [0.212, 0.256] | 0.55 [0.53, 0.56] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 120 | 5.29 | 0.136 [0.122, 0.149] | 0.247 [0.221, 0.275] | 0.55 [0.53, 0.56] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 40 | 5.09 | 0.139 [0.116, 0.162] | 0.193 [0.159, 0.227] | 0.55 [0.51, 0.58] |
