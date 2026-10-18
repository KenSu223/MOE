**e1p_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_14b | person-name options | True | 36 | 8.10 | 0.147 [0.110, 0.189] | 0.079 [0.048, 0.112] | 0.37 [0.33, 0.42] |
| qwen3_14b | person-name options | False | 220 | 8.02 | 0.174 [0.156, 0.191] | 0.128 [0.118, 0.139] | 0.40 [0.39, 0.42] |
| qwen3_14b | statement order 1 (entity of A mentioned first) | True | 128 | 8.50 | 0.170 [0.144, 0.194] | 0.124 [0.110, 0.139] | 0.38 [0.37, 0.41] |
| qwen3_14b | statement order 1 (entity of A mentioned first) | False | 128 | 7.56 | 0.170 [0.147, 0.192] | 0.118 [0.104, 0.133] | 0.41 [0.39, 0.43] |
| qwen3_14b | core-776 twin | True | 170 | 8.27 | 0.176 [0.156, 0.196] | 0.118 [0.106, 0.130] | 0.40 [0.38, 0.42] |
| qwen3_14b | core-776 twin | False | 86 | 7.55 | 0.156 [0.126, 0.184] | 0.129 [0.110, 0.147] | 0.40 [0.38, 0.43] |
| qwen3_14b | degree adverb in the blank clause (too / very / more ...) | True | 160 | 9.11 | 0.194 [0.174, 0.213] | 0.133 [0.121, 0.145] | 0.40 [0.39, 0.42] |
| qwen3_14b | degree adverb in the blank clause (too / very / more ...) | False | 96 | 6.24 | 0.112 [0.087, 0.137] | 0.093 [0.074, 0.113] | 0.38 [0.35, 0.42] |
| qwen3_14b | complementary antonym pair | True | 0 |  |  |  |  |
| qwen3_14b | complementary antonym pair | False | 256 | 8.03 | 0.170 [0.153, 0.186] | 0.121 [0.111, 0.132] | 0.40 [0.38, 0.41] |
| qwen3_14b | item built in both tokenizer families | True | 210 | 8.04 | 0.176 [0.158, 0.193] | 0.121 [0.110, 0.133] | 0.40 [0.38, 0.42] |
| qwen3_14b | item built in both tokenizer families | False | 46 | 7.98 | 0.144 [0.103, 0.182] | 0.121 [0.093, 0.145] | 0.38 [0.34, 0.42] |
| qwen3_14b | probe-neutral (both triggers |probe| < 1) | True | 256 | 8.03 | 0.170 [0.153, 0.186] | 0.121 [0.111, 0.132] | 0.40 [0.38, 0.41] |
| qwen3_14b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen3_14b | probe-associative | True | 0 |  |  |  |  |
| qwen3_14b | probe-associative | False | 256 | 8.03 | 0.170 [0.153, 0.186] | 0.121 [0.111, 0.132] | 0.40 [0.38, 0.41] |
| qwen3_14b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_14b | probe one-sided | False | 256 | 8.03 | 0.170 [0.153, 0.186] | 0.121 [0.111, 0.132] | 0.40 [0.38, 0.41] |
| qwen3_14b | probe reverse | True | 0 |  |  |  |  |
| qwen3_14b | probe reverse | False | 256 | 8.03 | 0.170 [0.153, 0.186] | 0.121 [0.111, 0.132] | 0.40 [0.38, 0.41] |
| qwen3_14b | top-1 in both directions | True | 19 | 9.87 | 0.266 [0.206, 0.310] | 0.136 [0.106, 0.179] | 0.40 [0.37, 0.43] |
| qwen3_14b | top-1 in both directions | False | 237 | 7.88 | 0.160 [0.143, 0.177] | 0.120 [0.110, 0.130] | 0.40 [0.38, 0.41] |
