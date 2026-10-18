**l_ant W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | person-name options | False | 252 | 8.51 | 0.199 [0.187, 0.212] | 0.276 [0.262, 0.290] | 0.52 [0.51, 0.54] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | True | 252 | 8.51 | 0.199 [0.187, 0.212] | 0.276 [0.262, 0.290] | 0.52 [0.51, 0.54] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | True | 252 | 8.51 | 0.199 [0.187, 0.212] | 0.276 [0.262, 0.290] | 0.52 [0.51, 0.54] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 252 | 8.51 | 0.199 [0.187, 0.212] | 0.276 [0.262, 0.290] | 0.52 [0.51, 0.54] |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 252 | 8.51 | 0.199 [0.187, 0.212] | 0.276 [0.262, 0.290] | 0.52 [0.51, 0.54] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe one-sided | False | 252 | 8.51 | 0.199 [0.187, 0.212] | 0.276 [0.262, 0.290] | 0.52 [0.51, 0.54] |
| Qwen3-30B-A3B-Base | probe reverse | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe reverse | False | 252 | 8.51 | 0.199 [0.187, 0.212] | 0.276 [0.262, 0.290] | 0.52 [0.51, 0.54] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 144 | 9.29 | 0.199 [0.185, 0.214] | 0.273 [0.257, 0.290] | 0.53 [0.51, 0.54] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 108 | 7.46 | 0.199 [0.176, 0.222] | 0.280 [0.254, 0.305] | 0.52 [0.50, 0.54] |
