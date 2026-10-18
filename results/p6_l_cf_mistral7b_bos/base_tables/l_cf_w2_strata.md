**l_cf W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | person-name options | True | 0 |  |  |  |  |
| mistral7b_bos | person-name options | False | 196 | 11.73 | 0.102 [0.092, 0.111] | 0.197 [0.181, 0.214] | 0.54 [0.52, 0.56] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | True | 196 | 11.73 | 0.102 [0.092, 0.111] | 0.197 [0.181, 0.214] | 0.54 [0.52, 0.56] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | False | 0 |  |  |  |  |
| mistral7b_bos | item built in both tokenizer families | True | 0 |  |  |  |  |
| mistral7b_bos | item built in both tokenizer families | False | 196 | 11.73 | 0.102 [0.092, 0.111] | 0.197 [0.181, 0.214] | 0.54 [0.52, 0.56] |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | False | 196 | 11.73 | 0.102 [0.092, 0.111] | 0.197 [0.181, 0.214] | 0.54 [0.52, 0.56] |
| mistral7b_bos | probe-associative | True | 0 |  |  |  |  |
| mistral7b_bos | probe-associative | False | 196 | 11.73 | 0.102 [0.092, 0.111] | 0.197 [0.181, 0.214] | 0.54 [0.52, 0.56] |
| mistral7b_bos | probe one-sided | True | 0 |  |  |  |  |
| mistral7b_bos | probe one-sided | False | 196 | 11.73 | 0.102 [0.092, 0.111] | 0.197 [0.181, 0.214] | 0.54 [0.52, 0.56] |
| mistral7b_bos | probe reverse | True | 0 |  |  |  |  |
| mistral7b_bos | probe reverse | False | 196 | 11.73 | 0.102 [0.092, 0.111] | 0.197 [0.181, 0.214] | 0.54 [0.52, 0.56] |
| mistral7b_bos | top-1 in both directions | True | 43 | 13.87 | 0.118 [0.099, 0.137] | 0.229 [0.196, 0.265] | 0.53 [0.49, 0.57] |
| mistral7b_bos | top-1 in both directions | False | 153 | 11.13 | 0.096 [0.086, 0.106] | 0.185 [0.167, 0.205] | 0.54 [0.52, 0.56] |
