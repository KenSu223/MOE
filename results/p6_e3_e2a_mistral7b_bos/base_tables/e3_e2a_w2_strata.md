**e3_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | person-name options | True | 22 | 6.47 | 0.101 [0.073, 0.133] | 0.140 [0.114, 0.171] | 0.38 [0.35, 0.41] |
| mistral7b_bos | person-name options | False | 234 | 7.69 | 0.178 [0.160, 0.195] | 0.159 [0.148, 0.170] | 0.30 [0.30, 0.31] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | True | 182 | 7.64 | 0.185 [0.165, 0.205] | 0.152 [0.139, 0.164] | 0.31 [0.30, 0.32] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | False | 74 | 7.44 | 0.138 [0.112, 0.168] | 0.173 [0.152, 0.193] | 0.31 [0.29, 0.32] |
| mistral7b_bos | core-776 twin | True | 225 | 7.87 | 0.177 [0.159, 0.194] | 0.159 [0.147, 0.170] | 0.31 [0.30, 0.32] |
| mistral7b_bos | core-776 twin | False | 31 | 5.48 | 0.121 [0.083, 0.170] | 0.149 [0.120, 0.181] | 0.34 [0.31, 0.36] |
| mistral7b_bos | degree adverb in the blank clause (too / very / more ...) | True | 134 | 7.22 | 0.098 [0.087, 0.109] | 0.174 [0.160, 0.187] | 0.32 [0.30, 0.33] |
| mistral7b_bos | degree adverb in the blank clause (too / very / more ...) | False | 122 | 7.99 | 0.246 [0.223, 0.267] | 0.142 [0.126, 0.158] | 0.30 [0.29, 0.32] |
| mistral7b_bos | complementary antonym pair | True | 2 |  |  |  |  |
| mistral7b_bos | complementary antonym pair | False | 254 | 7.60 | 0.172 [0.155, 0.188] | 0.158 [0.147, 0.169] | 0.31 [0.30, 0.32] |
| mistral7b_bos | item built in both tokenizer families | True | 256 | 7.58 | 0.172 [0.155, 0.188] | 0.158 [0.147, 0.168] | 0.31 [0.30, 0.32] |
| mistral7b_bos | item built in both tokenizer families | False | 0 |  |  |  |  |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | True | 256 | 7.58 | 0.172 [0.155, 0.188] | 0.158 [0.147, 0.168] | 0.31 [0.30, 0.32] |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| mistral7b_bos | probe-associative | True | 0 |  |  |  |  |
| mistral7b_bos | probe-associative | False | 256 | 7.58 | 0.172 [0.155, 0.188] | 0.158 [0.147, 0.168] | 0.31 [0.30, 0.32] |
| mistral7b_bos | probe one-sided | True | 0 |  |  |  |  |
| mistral7b_bos | probe one-sided | False | 256 | 7.58 | 0.172 [0.155, 0.188] | 0.158 [0.147, 0.168] | 0.31 [0.30, 0.32] |
| mistral7b_bos | probe reverse | True | 0 |  |  |  |  |
| mistral7b_bos | probe reverse | False | 256 | 7.58 | 0.172 [0.155, 0.188] | 0.158 [0.147, 0.168] | 0.31 [0.30, 0.32] |
| mistral7b_bos | top-1 in both directions | True | 86 | 9.16 | 0.264 [0.238, 0.288] | 0.164 [0.148, 0.182] | 0.29 [0.28, 0.31] |
| mistral7b_bos | top-1 in both directions | False | 170 | 6.79 | 0.109 [0.098, 0.121] | 0.154 [0.140, 0.167] | 0.32 [0.31, 0.33] |
