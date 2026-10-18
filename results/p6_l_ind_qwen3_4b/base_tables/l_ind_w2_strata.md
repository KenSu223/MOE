**l_ind W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_4b | person-name options | True | 0 |  |  |  |  |
| qwen3_4b | person-name options | False | 252 | 4.84 | 0.104 [0.097, 0.110] | 0.349 [0.333, 0.365] | 0.69 [0.67, 0.69] |
| qwen3_4b | statement order 1 (entity of A mentioned first) | True | 126 | 4.81 | 0.103 [0.094, 0.112] | 0.349 [0.326, 0.372] | 0.69 [0.67, 0.70] |
| qwen3_4b | statement order 1 (entity of A mentioned first) | False | 126 | 4.87 | 0.105 [0.096, 0.113] | 0.349 [0.327, 0.372] | 0.68 [0.67, 0.70] |
| qwen3_4b | item built in both tokenizer families | True | 252 | 4.84 | 0.104 [0.097, 0.110] | 0.349 [0.333, 0.365] | 0.69 [0.67, 0.69] |
| qwen3_4b | item built in both tokenizer families | False | 0 |  |  |  |  |
| qwen3_4b | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| qwen3_4b | probe-neutral (both triggers |probe| < 1) | False | 252 | 4.84 | 0.104 [0.097, 0.110] | 0.349 [0.333, 0.365] | 0.69 [0.67, 0.69] |
| qwen3_4b | probe-associative | True | 0 |  |  |  |  |
| qwen3_4b | probe-associative | False | 252 | 4.84 | 0.104 [0.097, 0.110] | 0.349 [0.333, 0.365] | 0.69 [0.67, 0.69] |
| qwen3_4b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_4b | probe one-sided | False | 252 | 4.84 | 0.104 [0.097, 0.110] | 0.349 [0.333, 0.365] | 0.69 [0.67, 0.69] |
| qwen3_4b | probe reverse | True | 0 |  |  |  |  |
| qwen3_4b | probe reverse | False | 252 | 4.84 | 0.104 [0.097, 0.110] | 0.349 [0.333, 0.365] | 0.69 [0.67, 0.69] |
| qwen3_4b | top-1 in both directions | True | 178 | 4.80 | 0.111 [0.104, 0.119] | 0.365 [0.345, 0.384] | 0.68 [0.67, 0.69] |
| qwen3_4b | top-1 in both directions | False | 74 | 4.94 | 0.087 [0.075, 0.098] | 0.313 [0.287, 0.340] | 0.69 [0.66, 0.70] |
