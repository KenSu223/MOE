**l_bind W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_8b | person-name options | True | 0 |  |  |  |  |
| qwen3_8b | person-name options | False | 160 | 4.64 | 0.155 [0.148, 0.162] | 0.305 [0.289, 0.321] | 0.68 [0.67, 0.69] |
| qwen3_8b | statement order 1 (entity of A mentioned first) | True | 80 | 4.48 | 0.160 [0.149, 0.171] | 0.316 [0.295, 0.336] | 0.68 [0.66, 0.70] |
| qwen3_8b | statement order 1 (entity of A mentioned first) | False | 80 | 4.79 | 0.151 [0.142, 0.159] | 0.295 [0.270, 0.320] | 0.68 [0.66, 0.69] |
| qwen3_8b | item built in both tokenizer families | True | 160 | 4.64 | 0.155 [0.148, 0.162] | 0.305 [0.289, 0.321] | 0.68 [0.67, 0.69] |
| qwen3_8b | item built in both tokenizer families | False | 0 |  |  |  |  |
| qwen3_8b | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| qwen3_8b | probe-neutral (both triggers |probe| < 1) | False | 160 | 4.64 | 0.155 [0.148, 0.162] | 0.305 [0.289, 0.321] | 0.68 [0.67, 0.69] |
| qwen3_8b | probe-associative | True | 0 |  |  |  |  |
| qwen3_8b | probe-associative | False | 160 | 4.64 | 0.155 [0.148, 0.162] | 0.305 [0.289, 0.321] | 0.68 [0.67, 0.69] |
| qwen3_8b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_8b | probe one-sided | False | 160 | 4.64 | 0.155 [0.148, 0.162] | 0.305 [0.289, 0.321] | 0.68 [0.67, 0.69] |
| qwen3_8b | probe reverse | True | 0 |  |  |  |  |
| qwen3_8b | probe reverse | False | 160 | 4.64 | 0.155 [0.148, 0.162] | 0.305 [0.289, 0.321] | 0.68 [0.67, 0.69] |
| qwen3_8b | top-1 in both directions | True | 160 | 4.64 | 0.155 [0.148, 0.162] | 0.305 [0.289, 0.321] | 0.68 [0.67, 0.69] |
| qwen3_8b | top-1 in both directions | False | 0 |  |  |  |  |
