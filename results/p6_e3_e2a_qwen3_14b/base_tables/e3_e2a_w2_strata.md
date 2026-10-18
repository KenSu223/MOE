**e3_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_14b | person-name options | True | 16 | 7.85 | 0.146 [0.080, 0.209] | 0.021 [0.004, 0.040] | 0.32 [0.26, 0.38] |
| qwen3_14b | person-name options | False | 238 | 8.58 | 0.206 [0.194, 0.218] | 0.068 [0.060, 0.077] | 0.20 [0.19, 0.21] |
| qwen3_14b | statement order 1 (entity of A mentioned first) | True | 178 | 8.37 | 0.196 [0.181, 0.212] | 0.061 [0.051, 0.071] | 0.21 [0.20, 0.23] |
| qwen3_14b | statement order 1 (entity of A mentioned first) | False | 76 | 8.92 | 0.217 [0.198, 0.236] | 0.075 [0.060, 0.090] | 0.20 [0.17, 0.22] |
| qwen3_14b | core-776 twin | True | 254 | 8.54 | 0.203 [0.191, 0.215] | 0.065 [0.058, 0.074] | 0.20 [0.19, 0.22] |
| qwen3_14b | core-776 twin | False | 0 |  |  |  |  |
| qwen3_14b | degree adverb in the blank clause (too / very / more ...) | True | 124 | 9.29 | 0.208 [0.189, 0.229] | 0.085 [0.073, 0.099] | 0.21 [0.19, 0.23] |
| qwen3_14b | degree adverb in the blank clause (too / very / more ...) | False | 130 | 7.81 | 0.196 [0.185, 0.207] | 0.043 [0.035, 0.051] | 0.20 [0.19, 0.22] |
| qwen3_14b | complementary antonym pair | True | 0 |  |  |  |  |
| qwen3_14b | complementary antonym pair | False | 254 | 8.54 | 0.203 [0.191, 0.215] | 0.065 [0.058, 0.074] | 0.20 [0.19, 0.22] |
| qwen3_14b | item built in both tokenizer families | True | 252 | 8.51 | 0.203 [0.191, 0.216] | 0.066 [0.058, 0.074] | 0.20 [0.19, 0.22] |
| qwen3_14b | item built in both tokenizer families | False | 2 |  |  |  |  |
| qwen3_14b | probe-neutral (both triggers |probe| < 1) | True | 254 | 8.54 | 0.203 [0.191, 0.215] | 0.065 [0.058, 0.074] | 0.20 [0.19, 0.22] |
| qwen3_14b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen3_14b | probe-associative | True | 0 |  |  |  |  |
| qwen3_14b | probe-associative | False | 254 | 8.54 | 0.203 [0.191, 0.215] | 0.065 [0.058, 0.074] | 0.20 [0.19, 0.22] |
| qwen3_14b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_14b | probe one-sided | False | 254 | 8.54 | 0.203 [0.191, 0.215] | 0.065 [0.058, 0.074] | 0.20 [0.19, 0.22] |
| qwen3_14b | probe reverse | True | 0 |  |  |  |  |
| qwen3_14b | probe reverse | False | 254 | 8.54 | 0.203 [0.191, 0.215] | 0.065 [0.058, 0.074] | 0.20 [0.19, 0.22] |
| qwen3_14b | top-1 in both directions | True | 98 | 8.47 | 0.206 [0.192, 0.221] | 0.051 [0.043, 0.060] | 0.20 [0.19, 0.21] |
| qwen3_14b | top-1 in both directions | False | 156 | 8.58 | 0.201 [0.183, 0.218] | 0.074 [0.062, 0.086] | 0.21 [0.19, 0.23] |
