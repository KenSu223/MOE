**e1_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_4b | person-name options | True | 28 | 6.19 | 0.201 [0.168, 0.238] | 0.184 [0.135, 0.238] | 0.40 [0.36, 0.46] |
| qwen3_4b | person-name options | False | 210 | 5.14 | 0.144 [0.135, 0.154] | 0.396 [0.368, 0.423] | 0.59 [0.57, 0.61] |
| qwen3_4b | statement order 1 (entity of A mentioned first) | True | 119 | 5.31 | 0.143 [0.130, 0.156] | 0.363 [0.328, 0.400] | 0.58 [0.55, 0.60] |
| qwen3_4b | statement order 1 (entity of A mentioned first) | False | 119 | 5.21 | 0.162 [0.148, 0.176] | 0.370 [0.332, 0.411] | 0.55 [0.52, 0.58] |
| qwen3_4b | core-776 twin | True | 74 | 4.86 | 0.161 [0.147, 0.175] | 0.377 [0.330, 0.426] | 0.56 [0.53, 0.60] |
| qwen3_4b | core-776 twin | False | 164 | 5.44 | 0.149 [0.137, 0.161] | 0.362 [0.331, 0.395] | 0.56 [0.54, 0.58] |
| qwen3_4b | degree adverb in the blank clause (too / very / more ...) | True | 100 | 5.59 | 0.145 [0.134, 0.158] | 0.311 [0.277, 0.347] | 0.57 [0.54, 0.60] |
| qwen3_4b | degree adverb in the blank clause (too / very / more ...) | False | 138 | 5.02 | 0.158 [0.144, 0.172] | 0.411 [0.375, 0.448] | 0.56 [0.53, 0.58] |
| qwen3_4b | complementary antonym pair | True | 4 |  |  |  |  |
| qwen3_4b | complementary antonym pair | False | 234 | 5.28 | 0.153 [0.143, 0.163] | 0.368 [0.342, 0.395] | 0.56 [0.54, 0.58] |
| qwen3_4b | item built in both tokenizer families | True | 144 | 4.94 | 0.156 [0.145, 0.168] | 0.376 [0.345, 0.409] | 0.56 [0.54, 0.59] |
| qwen3_4b | item built in both tokenizer families | False | 94 | 5.74 | 0.147 [0.131, 0.164] | 0.354 [0.309, 0.400] | 0.56 [0.53, 0.59] |
| qwen3_4b | probe-neutral (both triggers |probe| < 1) | True | 238 | 5.26 | 0.152 [0.143, 0.162] | 0.367 [0.340, 0.395] | 0.56 [0.54, 0.58] |
| qwen3_4b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen3_4b | probe-associative | True | 0 |  |  |  |  |
| qwen3_4b | probe-associative | False | 238 | 5.26 | 0.152 [0.143, 0.162] | 0.367 [0.340, 0.395] | 0.56 [0.54, 0.58] |
| qwen3_4b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_4b | probe one-sided | False | 238 | 5.26 | 0.152 [0.143, 0.162] | 0.367 [0.340, 0.395] | 0.56 [0.54, 0.58] |
| qwen3_4b | probe reverse | True | 0 |  |  |  |  |
| qwen3_4b | probe reverse | False | 238 | 5.26 | 0.152 [0.143, 0.162] | 0.367 [0.340, 0.395] | 0.56 [0.54, 0.58] |
| qwen3_4b | top-1 in both directions | True | 82 | 5.50 | 0.132 [0.122, 0.142] | 0.349 [0.317, 0.381] | 0.59 [0.56, 0.61] |
| qwen3_4b | top-1 in both directions | False | 156 | 5.13 | 0.164 [0.151, 0.178] | 0.377 [0.341, 0.416] | 0.55 [0.52, 0.58] |
