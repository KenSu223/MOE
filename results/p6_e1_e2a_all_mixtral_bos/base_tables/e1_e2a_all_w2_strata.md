**e1_e2a_all W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | person-name options | True | 12 | 3.78 | 0.147 [0.106, 0.183] | 0.218 [0.172, 0.276] | 0.55 [0.52, 0.57] |
| Mixtral-8x7B, BOS | person-name options | False | 194 | 4.20 | 0.179 [0.168, 0.189] | 0.174 [0.162, 0.185] | 0.52 [0.50, 0.53] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | True | 103 | 4.14 | 0.182 [0.168, 0.196] | 0.177 [0.160, 0.193] | 0.51 [0.49, 0.54] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | False | 103 | 4.22 | 0.172 [0.157, 0.188] | 0.175 [0.159, 0.191] | 0.53 [0.51, 0.54] |
| Mixtral-8x7B, BOS | core-776 twin | True | 134 | 4.26 | 0.184 [0.172, 0.196] | 0.172 [0.157, 0.186] | 0.51 [0.50, 0.53] |
| Mixtral-8x7B, BOS | core-776 twin | False | 72 | 4.01 | 0.165 [0.145, 0.184] | 0.185 [0.164, 0.205] | 0.53 [0.51, 0.55] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | True | 164 | 4.00 | 0.186 [0.175, 0.197] | 0.197 [0.185, 0.209] | 0.52 [0.51, 0.53] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | False | 42 | 4.88 | 0.150 [0.128, 0.173] | 0.108 [0.089, 0.128] | 0.52 [0.49, 0.55] |
| Mixtral-8x7B, BOS | complementary antonym pair | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | complementary antonym pair | False | 206 | 4.18 | 0.177 [0.167, 0.188] | 0.176 [0.164, 0.188] | 0.52 [0.51, 0.53] |
| Mixtral-8x7B, BOS | association-neutral by the model's own probes (_all cells) | True | 80 | 3.93 | 0.186 [0.170, 0.204] | 0.189 [0.168, 0.211] | 0.50 [0.48, 0.52] |
| Mixtral-8x7B, BOS | association-neutral by the model's own probes (_all cells) | False | 126 | 4.34 | 0.172 [0.159, 0.185] | 0.168 [0.155, 0.182] | 0.53 [0.51, 0.55] |
| Mixtral-8x7B, BOS | association-neutral by the twin / counterpart's probes (_all cells) | True | 76 | 3.95 | 0.183 [0.165, 0.202] | 0.187 [0.168, 0.205] | 0.52 [0.50, 0.54] |
| Mixtral-8x7B, BOS | association-neutral by the twin / counterpart's probes (_all cells) | False | 130 | 4.31 | 0.174 [0.161, 0.186] | 0.170 [0.156, 0.184] | 0.52 [0.51, 0.54] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | True | 206 | 4.18 | 0.177 [0.167, 0.188] | 0.176 [0.164, 0.188] | 0.52 [0.51, 0.53] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 80 | 3.93 | 0.186 [0.170, 0.204] | 0.189 [0.168, 0.211] | 0.50 [0.48, 0.52] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 126 | 4.34 | 0.172 [0.159, 0.185] | 0.168 [0.155, 0.182] | 0.53 [0.51, 0.55] |
| Mixtral-8x7B, BOS | probe-associative | True | 4 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 202 | 4.19 | 0.176 [0.166, 0.187] | 0.177 [0.165, 0.189] | 0.52 [0.51, 0.53] |
| Mixtral-8x7B, BOS | probe one-sided | True | 54 | 4.80 | 0.147 [0.132, 0.163] | 0.157 [0.140, 0.174] | 0.55 [0.53, 0.57] |
| Mixtral-8x7B, BOS | probe one-sided | False | 152 | 3.96 | 0.190 [0.178, 0.202] | 0.184 [0.170, 0.199] | 0.51 [0.49, 0.53] |
| Mixtral-8x7B, BOS | probe reverse | True | 68 | 4.01 | 0.192 [0.174, 0.211] | 0.181 [0.161, 0.201] | 0.52 [0.49, 0.54] |
| Mixtral-8x7B, BOS | probe reverse | False | 138 | 4.26 | 0.170 [0.158, 0.183] | 0.173 [0.159, 0.188] | 0.52 [0.51, 0.54] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 137 | 4.08 | 0.188 [0.176, 0.200] | 0.190 [0.176, 0.203] | 0.52 [0.50, 0.53] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 69 | 4.36 | 0.156 [0.138, 0.175] | 0.150 [0.131, 0.171] | 0.52 [0.50, 0.55] |
