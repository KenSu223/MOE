**e1_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | person-name options | True | 20 | 3.54 | 0.147 [0.086, 0.198] | 0.228 [0.178, 0.292] | 0.56 [0.52, 0.59] |
| Mixtral-8x7B, BOS | person-name options | False | 140 | 3.97 | 0.190 [0.171, 0.209] | 0.182 [0.162, 0.203] | 0.50 [0.47, 0.52] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | True | 80 | 3.98 | 0.186 [0.164, 0.208] | 0.188 [0.162, 0.215] | 0.49 [0.46, 0.53] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | False | 80 | 3.86 | 0.184 [0.157, 0.213] | 0.187 [0.157, 0.216] | 0.51 [0.48, 0.54] |
| Mixtral-8x7B, BOS | core-776 twin | True | 96 | 4.02 | 0.194 [0.173, 0.216] | 0.178 [0.152, 0.204] | 0.50 [0.46, 0.53] |
| Mixtral-8x7B, BOS | core-776 twin | False | 64 | 3.76 | 0.171 [0.138, 0.201] | 0.203 [0.176, 0.230] | 0.51 [0.49, 0.53] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | True | 136 | 3.71 | 0.190 [0.170, 0.210] | 0.201 [0.179, 0.223] | 0.50 [0.47, 0.52] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | False | 24 | 5.11 | 0.165 [0.118, 0.201] | 0.133 [0.090, 0.171] | 0.51 [0.48, 0.55] |
| Mixtral-8x7B, BOS | complementary antonym pair | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | complementary antonym pair | False | 160 | 3.92 | 0.185 [0.168, 0.204] | 0.187 [0.167, 0.208] | 0.50 [0.48, 0.52] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | True | 160 | 3.92 | 0.185 [0.168, 0.204] | 0.187 [0.167, 0.208] | 0.50 [0.48, 0.52] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 160 | 3.92 | 0.185 [0.168, 0.204] | 0.187 [0.167, 0.208] | 0.50 [0.48, 0.52] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 160 | 3.92 | 0.185 [0.168, 0.204] | 0.187 [0.167, 0.208] | 0.50 [0.48, 0.52] |
| Mixtral-8x7B, BOS | probe one-sided | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe one-sided | False | 160 | 3.92 | 0.185 [0.168, 0.204] | 0.187 [0.167, 0.208] | 0.50 [0.48, 0.52] |
| Mixtral-8x7B, BOS | probe reverse | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe reverse | False | 160 | 3.92 | 0.185 [0.168, 0.204] | 0.187 [0.167, 0.208] | 0.50 [0.48, 0.52] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 104 | 3.79 | 0.203 [0.182, 0.226] | 0.188 [0.162, 0.213] | 0.49 [0.46, 0.52] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 56 | 4.17 | 0.155 [0.125, 0.182] | 0.186 [0.157, 0.220] | 0.52 [0.50, 0.55] |
