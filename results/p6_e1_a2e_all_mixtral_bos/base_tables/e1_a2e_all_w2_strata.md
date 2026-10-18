**e1_a2e_all W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | person-name options | True | 22 | 3.93 | 0.300 [0.280, 0.322] | 0.542 [0.484, 0.595] | 0.59 [0.56, 0.62] |
| Mixtral-8x7B, BOS | person-name options | False | 234 | 4.08 | 0.098 [0.091, 0.105] | 0.244 [0.234, 0.255] | 0.57 [0.56, 0.58] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | True | 128 | 4.09 | 0.114 [0.100, 0.129] | 0.267 [0.246, 0.288] | 0.58 [0.56, 0.59] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | False | 128 | 4.05 | 0.115 [0.102, 0.129] | 0.271 [0.251, 0.294] | 0.57 [0.56, 0.59] |
| Mixtral-8x7B, BOS | core-776 twin | True | 170 | 4.06 | 0.101 [0.091, 0.111] | 0.253 [0.237, 0.270] | 0.57 [0.56, 0.59] |
| Mixtral-8x7B, BOS | core-776 twin | False | 86 | 4.08 | 0.142 [0.123, 0.162] | 0.301 [0.273, 0.331] | 0.58 [0.56, 0.60] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | True | 184 | 4.05 | 0.122 [0.110, 0.134] | 0.276 [0.256, 0.296] | 0.58 [0.57, 0.60] |
| Mixtral-8x7B, BOS | degree adverb in the blank clause (too / very / more ...) | False | 72 | 4.14 | 0.098 [0.084, 0.112] | 0.252 [0.234, 0.271] | 0.55 [0.53, 0.57] |
| Mixtral-8x7B, BOS | complementary antonym pair | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | complementary antonym pair | False | 256 | 4.07 | 0.115 [0.105, 0.125] | 0.269 [0.255, 0.284] | 0.57 [0.56, 0.59] |
| Mixtral-8x7B, BOS | association-neutral by the model's own probes (_all cells) | True | 106 | 4.01 | 0.147 [0.128, 0.164] | 0.313 [0.285, 0.344] | 0.57 [0.55, 0.59] |
| Mixtral-8x7B, BOS | association-neutral by the model's own probes (_all cells) | False | 150 | 4.11 | 0.093 [0.084, 0.102] | 0.239 [0.226, 0.252] | 0.57 [0.56, 0.59] |
| Mixtral-8x7B, BOS | association-neutral by the twin / counterpart's probes (_all cells) | True | 94 | 3.94 | 0.146 [0.128, 0.165] | 0.316 [0.287, 0.349] | 0.59 [0.57, 0.61] |
| Mixtral-8x7B, BOS | association-neutral by the twin / counterpart's probes (_all cells) | False | 162 | 4.15 | 0.098 [0.088, 0.108] | 0.243 [0.229, 0.258] | 0.56 [0.55, 0.58] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | True | 254 | 4.07 | 0.115 [0.105, 0.125] | 0.270 [0.255, 0.285] | 0.57 [0.56, 0.59] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | False | 2 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 106 | 4.01 | 0.147 [0.128, 0.164] | 0.313 [0.285, 0.344] | 0.57 [0.55, 0.59] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 150 | 4.11 | 0.093 [0.084, 0.102] | 0.239 [0.226, 0.252] | 0.57 [0.56, 0.59] |
| Mixtral-8x7B, BOS | probe-associative | True | 14 | 4.30 | 0.076 [0.047, 0.108] | 0.182 [0.149, 0.217] | 0.53 [0.48, 0.58] |
| Mixtral-8x7B, BOS | probe-associative | False | 242 | 4.06 | 0.117 [0.107, 0.127] | 0.274 [0.259, 0.291] | 0.58 [0.56, 0.59] |
| Mixtral-8x7B, BOS | probe one-sided | True | 64 | 4.25 | 0.097 [0.082, 0.112] | 0.237 [0.220, 0.253] | 0.58 [0.56, 0.60] |
| Mixtral-8x7B, BOS | probe one-sided | False | 192 | 4.01 | 0.121 [0.109, 0.133] | 0.281 [0.262, 0.300] | 0.57 [0.56, 0.59] |
| Mixtral-8x7B, BOS | probe reverse | True | 72 | 3.96 | 0.092 [0.080, 0.105] | 0.253 [0.233, 0.274] | 0.57 [0.55, 0.60] |
| Mixtral-8x7B, BOS | probe reverse | False | 184 | 4.12 | 0.123 [0.111, 0.136] | 0.275 [0.256, 0.295] | 0.57 [0.56, 0.59] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 243 | 4.04 | 0.115 [0.105, 0.126] | 0.271 [0.256, 0.287] | 0.58 [0.57, 0.59] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 13 | 4.70 | 0.111 [0.089, 0.133] | 0.240 [0.193, 0.286] | 0.49 [0.47, 0.52] |
