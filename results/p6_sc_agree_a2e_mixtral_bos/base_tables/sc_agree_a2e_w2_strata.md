**sc_agree_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | person-name options | True | 2 |  |  |  |  |
| Mixtral-8x7B, BOS | person-name options | False | 158 | 4.21 | 0.166 [0.149, 0.182] | 0.383 [0.364, 0.401] | 0.61 [0.59, 0.62] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | True | 80 | 4.35 | 0.163 [0.141, 0.189] | 0.378 [0.353, 0.404] | 0.61 [0.59, 0.63] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | False | 80 | 4.05 | 0.169 [0.147, 0.192] | 0.398 [0.369, 0.428] | 0.60 [0.59, 0.62] |
| Mixtral-8x7B, BOS | core-776 twin | True | 64 | 3.99 | 0.148 [0.126, 0.171] | 0.357 [0.325, 0.393] | 0.60 [0.58, 0.61] |
| Mixtral-8x7B, BOS | core-776 twin | False | 96 | 4.34 | 0.177 [0.155, 0.201] | 0.406 [0.383, 0.428] | 0.61 [0.60, 0.63] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | True | 80 | 4.27 | 0.138 [0.119, 0.159] | 0.343 [0.318, 0.370] | 0.59 [0.57, 0.60] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | False | 80 | 4.13 | 0.195 [0.171, 0.221] | 0.433 [0.409, 0.457] | 0.63 [0.61, 0.65] |
| Mixtral-8x7B, BOS | complementary antonym pair | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | complementary antonym pair | False | 160 | 4.20 | 0.166 [0.150, 0.183] | 0.387 [0.368, 0.407] | 0.61 [0.60, 0.62] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | True | 160 | 4.20 | 0.166 [0.150, 0.183] | 0.387 [0.368, 0.407] | 0.61 [0.60, 0.62] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 160 | 4.20 | 0.166 [0.150, 0.183] | 0.387 [0.368, 0.407] | 0.61 [0.60, 0.62] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 160 | 4.20 | 0.166 [0.150, 0.183] | 0.387 [0.368, 0.407] | 0.61 [0.60, 0.62] |
| Mixtral-8x7B, BOS | probe one-sided | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe one-sided | False | 160 | 4.20 | 0.166 [0.150, 0.183] | 0.387 [0.368, 0.407] | 0.61 [0.60, 0.62] |
| Mixtral-8x7B, BOS | probe reverse | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe reverse | False | 160 | 4.20 | 0.166 [0.150, 0.183] | 0.387 [0.368, 0.407] | 0.61 [0.60, 0.62] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 153 | 4.23 | 0.167 [0.150, 0.184] | 0.387 [0.368, 0.407] | 0.61 [0.60, 0.62] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 7 |  |  |  |  |
