**e3_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 25 | 8.12 | 0.085 [0.061, 0.107] | 0.031 [0.018, 0.043] | 0.28 [0.24, 0.34] |
| Qwen3-30B-A3B-Base | person-name options | False | 231 | 8.51 | 0.225 [0.213, 0.237] | 0.034 [0.029, 0.039] | 0.15 [0.13, 0.17] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | True | 175 | 8.32 | 0.194 [0.180, 0.209] | 0.038 [0.033, 0.043] | 0.17 [0.16, 0.20] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | False | 81 | 8.81 | 0.248 [0.229, 0.269] | 0.026 [0.017, 0.035] | 0.13 [0.10, 0.19] |
| Qwen3-30B-A3B-Base | core-776 twin | True | 256 | 8.47 | 0.212 [0.200, 0.224] | 0.034 [0.029, 0.038] | 0.16 [0.14, 0.18] |
| Qwen3-30B-A3B-Base | core-776 twin | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | degree adverb in the blank clause (too / very / more ...) | True | 126 | 8.93 | 0.252 [0.233, 0.269] | 0.035 [0.028, 0.041] | 0.15 [0.13, 0.18] |
| Qwen3-30B-A3B-Base | degree adverb in the blank clause (too / very / more ...) | False | 130 | 8.03 | 0.169 [0.157, 0.181] | 0.033 [0.027, 0.039] | 0.18 [0.15, 0.21] |
| Qwen3-30B-A3B-Base | complementary antonym pair | True | 1 |  |  |  |  |
| Qwen3-30B-A3B-Base | complementary antonym pair | False | 255 | 8.48 | 0.212 [0.200, 0.224] | 0.034 [0.030, 0.038] | 0.16 [0.14, 0.18] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | True | 253 | 8.49 | 0.213 [0.201, 0.225] | 0.034 [0.029, 0.038] | 0.16 [0.14, 0.18] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | False | 3 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 256 | 8.47 | 0.212 [0.200, 0.224] | 0.034 [0.029, 0.038] | 0.16 [0.14, 0.18] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 256 | 8.47 | 0.212 [0.200, 0.224] | 0.034 [0.029, 0.038] | 0.16 [0.14, 0.18] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe one-sided | False | 256 | 8.47 | 0.212 [0.200, 0.224] | 0.034 [0.029, 0.038] | 0.16 [0.14, 0.18] |
| Qwen3-30B-A3B-Base | probe reverse | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe reverse | False | 256 | 8.47 | 0.212 [0.200, 0.224] | 0.034 [0.029, 0.038] | 0.16 [0.14, 0.18] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 95 | 8.93 | 0.204 [0.188, 0.218] | 0.034 [0.027, 0.041] | 0.14 [0.12, 0.17] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 161 | 8.20 | 0.217 [0.199, 0.234] | 0.034 [0.028, 0.040] | 0.17 [0.15, 0.21] |
