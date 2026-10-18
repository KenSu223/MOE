**e3_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen2_7b | person-name options | True | 14 | 5.48 | 0.216 [0.157, 0.277] | 0.041 [-0.002, 0.085] | 0.29 [0.24, 0.34] |
| qwen2_7b | person-name options | False | 242 | 8.11 | 0.306 [0.285, 0.325] | 0.089 [0.077, 0.102] | 0.21 [0.19, 0.22] |
| qwen2_7b | statement order 1 (entity of A mentioned first) | True | 191 | 7.87 | 0.314 [0.291, 0.338] | 0.081 [0.068, 0.095] | 0.20 [0.19, 0.21] |
| qwen2_7b | statement order 1 (entity of A mentioned first) | False | 65 | 8.25 | 0.269 [0.233, 0.303] | 0.104 [0.079, 0.129] | 0.23 [0.20, 0.26] |
| qwen2_7b | core-776 twin | True | 256 | 7.97 | 0.302 [0.282, 0.322] | 0.087 [0.075, 0.099] | 0.21 [0.20, 0.22] |
| qwen2_7b | core-776 twin | False | 0 |  |  |  |  |
| qwen2_7b | degree adverb in the blank clause (too / very / more ...) | True | 126 | 7.57 | 0.237 [0.221, 0.254] | 0.145 [0.130, 0.160] | 0.23 [0.22, 0.25] |
| qwen2_7b | degree adverb in the blank clause (too / very / more ...) | False | 130 | 8.36 | 0.359 [0.329, 0.388] | 0.036 [0.024, 0.049] | 0.19 [0.18, 0.21] |
| qwen2_7b | complementary antonym pair | True | 2 |  |  |  |  |
| qwen2_7b | complementary antonym pair | False | 254 | 8.00 | 0.303 [0.283, 0.322] | 0.087 [0.076, 0.099] | 0.21 [0.20, 0.22] |
| qwen2_7b | item built in both tokenizer families | True | 253 | 7.99 | 0.303 [0.283, 0.322] | 0.087 [0.076, 0.100] | 0.21 [0.20, 0.22] |
| qwen2_7b | item built in both tokenizer families | False | 3 |  |  |  |  |
| qwen2_7b | probe-neutral (both triggers |probe| < 1) | True | 256 | 7.97 | 0.302 [0.282, 0.322] | 0.087 [0.075, 0.099] | 0.21 [0.20, 0.22] |
| qwen2_7b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen2_7b | probe-associative | True | 0 |  |  |  |  |
| qwen2_7b | probe-associative | False | 256 | 7.97 | 0.302 [0.282, 0.322] | 0.087 [0.075, 0.099] | 0.21 [0.20, 0.22] |
| qwen2_7b | probe one-sided | True | 0 |  |  |  |  |
| qwen2_7b | probe one-sided | False | 256 | 7.97 | 0.302 [0.282, 0.322] | 0.087 [0.075, 0.099] | 0.21 [0.20, 0.22] |
| qwen2_7b | probe reverse | True | 0 |  |  |  |  |
| qwen2_7b | probe reverse | False | 256 | 7.97 | 0.302 [0.282, 0.322] | 0.087 [0.075, 0.099] | 0.21 [0.20, 0.22] |
| qwen2_7b | top-1 in both directions | True | 81 | 9.45 | 0.412 [0.382, 0.440] | 0.055 [0.038, 0.073] | 0.19 [0.17, 0.21] |
| qwen2_7b | top-1 in both directions | False | 175 | 7.28 | 0.236 [0.219, 0.254] | 0.106 [0.092, 0.121] | 0.23 [0.21, 0.24] |
