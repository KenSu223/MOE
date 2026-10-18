**e1p_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | person-name options | True | 76 | 4.23 | 0.153 [0.137, 0.170] | 0.191 [0.163, 0.219] | 0.56 [0.53, 0.59] |
| Mixtral-8x7B, BOS | person-name options | False | 144 | 4.15 | 0.219 [0.199, 0.238] | 0.231 [0.207, 0.252] | 0.43 [0.41, 0.46] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | True | 110 | 4.11 | 0.193 [0.170, 0.215] | 0.217 [0.191, 0.243] | 0.47 [0.44, 0.51] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | False | 110 | 4.25 | 0.199 [0.178, 0.220] | 0.216 [0.190, 0.242] | 0.48 [0.44, 0.51] |
| Mixtral-8x7B, BOS | core-776 twin | True | 88 | 4.44 | 0.189 [0.165, 0.214] | 0.218 [0.189, 0.247] | 0.46 [0.42, 0.50] |
| Mixtral-8x7B, BOS | core-776 twin | False | 132 | 4.01 | 0.201 [0.182, 0.220] | 0.216 [0.192, 0.239] | 0.49 [0.46, 0.51] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | True | 148 | 4.38 | 0.201 [0.181, 0.222] | 0.241 [0.221, 0.260] | 0.46 [0.44, 0.49] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | False | 72 | 3.77 | 0.185 [0.164, 0.206] | 0.159 [0.128, 0.194] | 0.49 [0.46, 0.53] |
| Mixtral-8x7B, BOS | complementary antonym pair | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | complementary antonym pair | False | 220 | 4.18 | 0.196 [0.180, 0.211] | 0.217 [0.199, 0.235] | 0.47 [0.45, 0.50] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | True | 220 | 4.18 | 0.196 [0.180, 0.211] | 0.217 [0.199, 0.235] | 0.47 [0.45, 0.50] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 220 | 4.18 | 0.196 [0.180, 0.211] | 0.217 [0.199, 0.235] | 0.47 [0.45, 0.50] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 220 | 4.18 | 0.196 [0.180, 0.211] | 0.217 [0.199, 0.235] | 0.47 [0.45, 0.50] |
| Mixtral-8x7B, BOS | probe one-sided | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe one-sided | False | 220 | 4.18 | 0.196 [0.180, 0.211] | 0.217 [0.199, 0.235] | 0.47 [0.45, 0.50] |
| Mixtral-8x7B, BOS | probe reverse | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe reverse | False | 220 | 4.18 | 0.196 [0.180, 0.211] | 0.217 [0.199, 0.235] | 0.47 [0.45, 0.50] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 6 |  |  |  |  |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 214 | 4.18 | 0.197 [0.181, 0.213] | 0.219 [0.200, 0.237] | 0.47 [0.45, 0.50] |
