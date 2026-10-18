**l_ant W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | person-name options | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | person-name options | False | 252 | 9.88 | 0.085 [0.073, 0.098] | 0.141 [0.130, 0.153] | 0.49 [0.48, 0.50] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | True | 252 | 9.88 | 0.085 [0.073, 0.098] | 0.141 [0.130, 0.153] | 0.49 [0.48, 0.50] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | item built in both tokenizer families | True | 252 | 9.88 | 0.085 [0.073, 0.098] | 0.141 [0.130, 0.153] | 0.49 [0.48, 0.50] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 252 | 9.88 | 0.085 [0.073, 0.098] | 0.141 [0.130, 0.153] | 0.49 [0.48, 0.50] |
| Mixtral-8x7B, BOS | probe-associative | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 252 | 9.88 | 0.085 [0.073, 0.098] | 0.141 [0.130, 0.153] | 0.49 [0.48, 0.50] |
| Mixtral-8x7B, BOS | probe one-sided | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe one-sided | False | 252 | 9.88 | 0.085 [0.073, 0.098] | 0.141 [0.130, 0.153] | 0.49 [0.48, 0.50] |
| Mixtral-8x7B, BOS | probe reverse | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe reverse | False | 252 | 9.88 | 0.085 [0.073, 0.098] | 0.141 [0.130, 0.153] | 0.49 [0.48, 0.50] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 154 | 11.14 | 0.086 [0.072, 0.101] | 0.144 [0.131, 0.159] | 0.49 [0.48, 0.51] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 98 | 7.90 | 0.085 [0.064, 0.108] | 0.134 [0.115, 0.155] | 0.48 [0.46, 0.51] |
