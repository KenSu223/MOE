**e1p_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_4b | person-name options | True | 28 | 6.90 | 0.113 [0.103, 0.121] | 0.400 [0.383, 0.417] | 0.74 [0.72, 0.75] |
| qwen3_4b | person-name options | False | 228 | 6.21 | 0.116 [0.111, 0.120] | 0.292 [0.283, 0.300] | 0.67 [0.67, 0.68] |
| qwen3_4b | statement order 1 (entity of A mentioned first) | True | 128 | 6.37 | 0.115 [0.109, 0.122] | 0.312 [0.299, 0.325] | 0.69 [0.68, 0.70] |
| qwen3_4b | statement order 1 (entity of A mentioned first) | False | 128 | 6.21 | 0.115 [0.110, 0.121] | 0.297 [0.286, 0.309] | 0.68 [0.67, 0.69] |
| qwen3_4b | core-776 twin | True | 256 | 6.29 | 0.115 [0.111, 0.120] | 0.305 [0.296, 0.313] | 0.68 [0.68, 0.69] |
| qwen3_4b | core-776 twin | False | 0 |  |  |  |  |
| qwen3_4b | degree adverb in the blank clause (too / very / more ...) | True | 126 | 6.91 | 0.121 [0.115, 0.127] | 0.308 [0.294, 0.321] | 0.68 [0.67, 0.69] |
| qwen3_4b | degree adverb in the blank clause (too / very / more ...) | False | 130 | 5.69 | 0.108 [0.102, 0.114] | 0.301 [0.290, 0.313] | 0.69 [0.68, 0.70] |
| qwen3_4b | complementary antonym pair | True | 2 |  |  |  |  |
| qwen3_4b | complementary antonym pair | False | 254 | 6.30 | 0.115 [0.111, 0.120] | 0.305 [0.297, 0.314] | 0.68 [0.68, 0.69] |
| qwen3_4b | item built in both tokenizer families | True | 250 | 6.27 | 0.115 [0.110, 0.119] | 0.304 [0.295, 0.313] | 0.68 [0.68, 0.69] |
| qwen3_4b | item built in both tokenizer families | False | 6 |  |  |  |  |
| qwen3_4b | probe-neutral (both triggers |probe| < 1) | True | 256 | 6.29 | 0.115 [0.111, 0.120] | 0.305 [0.296, 0.313] | 0.68 [0.68, 0.69] |
| qwen3_4b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen3_4b | probe-associative | True | 0 |  |  |  |  |
| qwen3_4b | probe-associative | False | 256 | 6.29 | 0.115 [0.111, 0.120] | 0.305 [0.296, 0.313] | 0.68 [0.68, 0.69] |
| qwen3_4b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_4b | probe one-sided | False | 256 | 6.29 | 0.115 [0.111, 0.120] | 0.305 [0.296, 0.313] | 0.68 [0.68, 0.69] |
| qwen3_4b | probe reverse | True | 0 |  |  |  |  |
| qwen3_4b | probe reverse | False | 256 | 6.29 | 0.115 [0.111, 0.120] | 0.305 [0.296, 0.313] | 0.68 [0.68, 0.69] |
| qwen3_4b | top-1 in both directions | True | 247 | 6.31 | 0.116 [0.112, 0.120] | 0.303 [0.294, 0.312] | 0.68 [0.68, 0.69] |
| qwen3_4b | top-1 in both directions | False | 9 |  |  |  |  |
