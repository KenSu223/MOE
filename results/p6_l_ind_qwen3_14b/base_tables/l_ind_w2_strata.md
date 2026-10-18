**l_ind W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_14b | person-name options | True | 0 |  |  |  |  |
| qwen3_14b | person-name options | False | 128 | 5.13 | 0.113 [0.098, 0.129] | 0.420 [0.397, 0.443] | 0.67 [0.65, 0.69] |
| qwen3_14b | statement order 1 (entity of A mentioned first) | True | 64 | 5.18 | 0.123 [0.100, 0.145] | 0.433 [0.402, 0.463] | 0.68 [0.65, 0.70] |
| qwen3_14b | statement order 1 (entity of A mentioned first) | False | 64 | 5.08 | 0.103 [0.082, 0.123] | 0.407 [0.374, 0.442] | 0.66 [0.63, 0.69] |
| qwen3_14b | item built in both tokenizer families | True | 128 | 5.13 | 0.113 [0.098, 0.129] | 0.420 [0.397, 0.443] | 0.67 [0.65, 0.69] |
| qwen3_14b | item built in both tokenizer families | False | 0 |  |  |  |  |
| qwen3_14b | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| qwen3_14b | probe-neutral (both triggers |probe| < 1) | False | 128 | 5.13 | 0.113 [0.098, 0.129] | 0.420 [0.397, 0.443] | 0.67 [0.65, 0.69] |
| qwen3_14b | probe-associative | True | 0 |  |  |  |  |
| qwen3_14b | probe-associative | False | 128 | 5.13 | 0.113 [0.098, 0.129] | 0.420 [0.397, 0.443] | 0.67 [0.65, 0.69] |
| qwen3_14b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_14b | probe one-sided | False | 128 | 5.13 | 0.113 [0.098, 0.129] | 0.420 [0.397, 0.443] | 0.67 [0.65, 0.69] |
| qwen3_14b | probe reverse | True | 0 |  |  |  |  |
| qwen3_14b | probe reverse | False | 128 | 5.13 | 0.113 [0.098, 0.129] | 0.420 [0.397, 0.443] | 0.67 [0.65, 0.69] |
| qwen3_14b | top-1 in both directions | True | 104 | 5.01 | 0.118 [0.102, 0.133] | 0.436 [0.417, 0.457] | 0.68 [0.65, 0.70] |
| qwen3_14b | top-1 in both directions | False | 24 | 5.64 | 0.094 [0.058, 0.140] | 0.357 [0.299, 0.429] | 0.63 [0.59, 0.68] |
