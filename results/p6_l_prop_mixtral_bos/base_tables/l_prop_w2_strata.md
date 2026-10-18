**l_prop W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | person-name options | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | person-name options | False | 250 | 10.18 | 0.111 [0.103, 0.119] | 0.133 [0.124, 0.143] | 0.42 [0.41, 0.43] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | True | 250 | 10.18 | 0.111 [0.103, 0.119] | 0.133 [0.124, 0.143] | 0.42 [0.41, 0.43] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | item built in both tokenizer families | True | 175 | 10.17 | 0.116 [0.107, 0.127] | 0.136 [0.125, 0.147] | 0.41 [0.39, 0.43] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | False | 75 | 10.23 | 0.099 [0.088, 0.111] | 0.127 [0.111, 0.144] | 0.45 [0.42, 0.47] |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 250 | 10.18 | 0.111 [0.103, 0.119] | 0.133 [0.124, 0.143] | 0.42 [0.41, 0.43] |
| Mixtral-8x7B, BOS | probe-associative | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 250 | 10.18 | 0.111 [0.103, 0.119] | 0.133 [0.124, 0.143] | 0.42 [0.41, 0.43] |
| Mixtral-8x7B, BOS | probe one-sided | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe one-sided | False | 250 | 10.18 | 0.111 [0.103, 0.119] | 0.133 [0.124, 0.143] | 0.42 [0.41, 0.43] |
| Mixtral-8x7B, BOS | probe reverse | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe reverse | False | 250 | 10.18 | 0.111 [0.103, 0.119] | 0.133 [0.124, 0.143] | 0.42 [0.41, 0.43] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 81 | 11.22 | 0.105 [0.096, 0.115] | 0.137 [0.122, 0.154] | 0.43 [0.41, 0.45] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 169 | 9.69 | 0.115 [0.103, 0.126] | 0.131 [0.120, 0.142] | 0.42 [0.40, 0.43] |
