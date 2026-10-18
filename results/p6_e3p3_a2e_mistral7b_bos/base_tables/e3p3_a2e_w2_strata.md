**e3p3_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | person-name options | True | 18 | 3.53 | 0.148 [0.115, 0.180] | 0.499 [0.442, 0.547] | 0.66 [0.63, 0.69] |
| mistral7b_bos | person-name options | False | 194 | 4.47 | 0.145 [0.130, 0.160] | 0.293 [0.273, 0.312] | 0.54 [0.53, 0.56] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | True | 158 | 4.26 | 0.144 [0.127, 0.163] | 0.323 [0.300, 0.348] | 0.56 [0.54, 0.58] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | False | 54 | 4.78 | 0.147 [0.121, 0.172] | 0.263 [0.227, 0.302] | 0.53 [0.50, 0.56] |
| mistral7b_bos | core-776 twin | True | 126 | 4.44 | 0.154 [0.134, 0.174] | 0.308 [0.282, 0.337] | 0.56 [0.54, 0.58] |
| mistral7b_bos | core-776 twin | False | 86 | 4.32 | 0.131 [0.110, 0.151] | 0.304 [0.272, 0.337] | 0.55 [0.51, 0.58] |
| mistral7b_bos | degree adverb in the blank clause (too / very / more ...) | True | 104 | 4.73 | 0.127 [0.106, 0.147] | 0.271 [0.244, 0.302] | 0.54 [0.52, 0.56] |
| mistral7b_bos | degree adverb in the blank clause (too / very / more ...) | False | 108 | 4.06 | 0.166 [0.146, 0.184] | 0.347 [0.322, 0.371] | 0.57 [0.54, 0.59] |
| mistral7b_bos | complementary antonym pair | True | 0 |  |  |  |  |
| mistral7b_bos | complementary antonym pair | False | 212 | 4.39 | 0.145 [0.131, 0.159] | 0.307 [0.286, 0.328] | 0.56 [0.54, 0.57] |
| mistral7b_bos | item built in both tokenizer families | True | 212 | 4.39 | 0.145 [0.131, 0.159] | 0.307 [0.286, 0.328] | 0.56 [0.54, 0.57] |
| mistral7b_bos | item built in both tokenizer families | False | 0 |  |  |  |  |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | True | 212 | 4.39 | 0.145 [0.131, 0.159] | 0.307 [0.286, 0.328] | 0.56 [0.54, 0.57] |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| mistral7b_bos | probe-associative | True | 0 |  |  |  |  |
| mistral7b_bos | probe-associative | False | 212 | 4.39 | 0.145 [0.131, 0.159] | 0.307 [0.286, 0.328] | 0.56 [0.54, 0.57] |
| mistral7b_bos | probe one-sided | True | 0 |  |  |  |  |
| mistral7b_bos | probe one-sided | False | 212 | 4.39 | 0.145 [0.131, 0.159] | 0.307 [0.286, 0.328] | 0.56 [0.54, 0.57] |
| mistral7b_bos | probe reverse | True | 0 |  |  |  |  |
| mistral7b_bos | probe reverse | False | 212 | 4.39 | 0.145 [0.131, 0.159] | 0.307 [0.286, 0.328] | 0.56 [0.54, 0.57] |
| mistral7b_bos | top-1 in both directions | True | 176 | 4.46 | 0.147 [0.132, 0.163] | 0.307 [0.284, 0.332] | 0.55 [0.54, 0.57] |
| mistral7b_bos | top-1 in both directions | False | 36 | 4.07 | 0.132 [0.088, 0.174] | 0.303 [0.259, 0.349] | 0.55 [0.49, 0.60] |
