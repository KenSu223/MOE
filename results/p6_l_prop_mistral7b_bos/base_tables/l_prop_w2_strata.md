**l_prop W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | person-name options | True | 0 |  |  |  |  |
| mistral7b_bos | person-name options | False | 251 | 10.14 | 0.124 [0.115, 0.133] | 0.159 [0.151, 0.168] | 0.38 [0.36, 0.39] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | True | 251 | 10.14 | 0.124 [0.115, 0.133] | 0.159 [0.151, 0.168] | 0.38 [0.36, 0.39] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | False | 0 |  |  |  |  |
| mistral7b_bos | item built in both tokenizer families | True | 173 | 10.21 | 0.121 [0.110, 0.133] | 0.157 [0.147, 0.168] | 0.37 [0.36, 0.38] |
| mistral7b_bos | item built in both tokenizer families | False | 78 | 9.98 | 0.130 [0.117, 0.143] | 0.164 [0.147, 0.181] | 0.40 [0.37, 0.42] |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | False | 251 | 10.14 | 0.124 [0.115, 0.133] | 0.159 [0.151, 0.168] | 0.38 [0.36, 0.39] |
| mistral7b_bos | probe-associative | True | 0 |  |  |  |  |
| mistral7b_bos | probe-associative | False | 251 | 10.14 | 0.124 [0.115, 0.133] | 0.159 [0.151, 0.168] | 0.38 [0.36, 0.39] |
| mistral7b_bos | probe one-sided | True | 0 |  |  |  |  |
| mistral7b_bos | probe one-sided | False | 251 | 10.14 | 0.124 [0.115, 0.133] | 0.159 [0.151, 0.168] | 0.38 [0.36, 0.39] |
| mistral7b_bos | probe reverse | True | 0 |  |  |  |  |
| mistral7b_bos | probe reverse | False | 251 | 10.14 | 0.124 [0.115, 0.133] | 0.159 [0.151, 0.168] | 0.38 [0.36, 0.39] |
| mistral7b_bos | top-1 in both directions | True | 91 | 10.62 | 0.132 [0.119, 0.146] | 0.182 [0.166, 0.199] | 0.38 [0.36, 0.40] |
| mistral7b_bos | top-1 in both directions | False | 160 | 9.87 | 0.119 [0.108, 0.130] | 0.145 [0.136, 0.155] | 0.37 [0.36, 0.39] |
