**l_bind W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | person-name options | True | 0 |  |  |  |  |
| mistral7b_bos | person-name options | False | 256 | 4.41 | 0.092 [0.088, 0.095] | 0.301 [0.294, 0.308] | 0.67 [0.66, 0.68] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | True | 128 | 4.43 | 0.092 [0.088, 0.097] | 0.301 [0.291, 0.310] | 0.67 [0.66, 0.68] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | False | 128 | 4.39 | 0.091 [0.086, 0.096] | 0.301 [0.292, 0.311] | 0.67 [0.66, 0.68] |
| mistral7b_bos | item built in both tokenizer families | True | 256 | 4.41 | 0.092 [0.088, 0.095] | 0.301 [0.294, 0.308] | 0.67 [0.66, 0.68] |
| mistral7b_bos | item built in both tokenizer families | False | 0 |  |  |  |  |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | False | 256 | 4.41 | 0.092 [0.088, 0.095] | 0.301 [0.294, 0.308] | 0.67 [0.66, 0.68] |
| mistral7b_bos | probe-associative | True | 0 |  |  |  |  |
| mistral7b_bos | probe-associative | False | 256 | 4.41 | 0.092 [0.088, 0.095] | 0.301 [0.294, 0.308] | 0.67 [0.66, 0.68] |
| mistral7b_bos | probe one-sided | True | 0 |  |  |  |  |
| mistral7b_bos | probe one-sided | False | 256 | 4.41 | 0.092 [0.088, 0.095] | 0.301 [0.294, 0.308] | 0.67 [0.66, 0.68] |
| mistral7b_bos | probe reverse | True | 0 |  |  |  |  |
| mistral7b_bos | probe reverse | False | 256 | 4.41 | 0.092 [0.088, 0.095] | 0.301 [0.294, 0.308] | 0.67 [0.66, 0.68] |
| mistral7b_bos | top-1 in both directions | True | 253 | 4.40 | 0.092 [0.089, 0.095] | 0.301 [0.295, 0.308] | 0.67 [0.66, 0.68] |
| mistral7b_bos | top-1 in both directions | False | 3 |  |  |  |  |
