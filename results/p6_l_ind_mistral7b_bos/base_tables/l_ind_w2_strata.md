**l_ind W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | person-name options | True | 0 |  |  |  |  |
| mistral7b_bos | person-name options | False | 256 | 4.67 | 0.109 [0.104, 0.114] | 0.355 [0.339, 0.370] | 0.68 [0.67, 0.69] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | True | 128 | 4.69 | 0.109 [0.102, 0.117] | 0.359 [0.338, 0.381] | 0.68 [0.67, 0.70] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | False | 128 | 4.65 | 0.109 [0.102, 0.116] | 0.350 [0.330, 0.371] | 0.68 [0.67, 0.69] |
| mistral7b_bos | item built in both tokenizer families | True | 256 | 4.67 | 0.109 [0.104, 0.114] | 0.355 [0.339, 0.370] | 0.68 [0.67, 0.69] |
| mistral7b_bos | item built in both tokenizer families | False | 0 |  |  |  |  |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | False | 256 | 4.67 | 0.109 [0.104, 0.114] | 0.355 [0.339, 0.370] | 0.68 [0.67, 0.69] |
| mistral7b_bos | probe-associative | True | 0 |  |  |  |  |
| mistral7b_bos | probe-associative | False | 256 | 4.67 | 0.109 [0.104, 0.114] | 0.355 [0.339, 0.370] | 0.68 [0.67, 0.69] |
| mistral7b_bos | probe one-sided | True | 0 |  |  |  |  |
| mistral7b_bos | probe one-sided | False | 256 | 4.67 | 0.109 [0.104, 0.114] | 0.355 [0.339, 0.370] | 0.68 [0.67, 0.69] |
| mistral7b_bos | probe reverse | True | 0 |  |  |  |  |
| mistral7b_bos | probe reverse | False | 256 | 4.67 | 0.109 [0.104, 0.114] | 0.355 [0.339, 0.370] | 0.68 [0.67, 0.69] |
| mistral7b_bos | top-1 in both directions | True | 236 | 4.65 | 0.106 [0.101, 0.112] | 0.354 [0.337, 0.369] | 0.68 [0.67, 0.69] |
| mistral7b_bos | top-1 in both directions | False | 20 | 4.92 | 0.138 [0.121, 0.155] | 0.365 [0.328, 0.403] | 0.66 [0.63, 0.69] |
