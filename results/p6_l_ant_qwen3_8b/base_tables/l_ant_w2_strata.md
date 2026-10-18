**l_ant W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_8b | person-name options | True | 0 |  |  |  |  |
| qwen3_8b | person-name options | False | 252 | 8.84 | 0.136 [0.122, 0.151] | 0.412 [0.395, 0.429] | 0.55 [0.53, 0.57] |
| qwen3_8b | statement order 1 (entity of A mentioned first) | True | 252 | 8.84 | 0.136 [0.122, 0.151] | 0.412 [0.395, 0.429] | 0.55 [0.53, 0.57] |
| qwen3_8b | statement order 1 (entity of A mentioned first) | False | 0 |  |  |  |  |
| qwen3_8b | item built in both tokenizer families | True | 252 | 8.84 | 0.136 [0.122, 0.151] | 0.412 [0.395, 0.429] | 0.55 [0.53, 0.57] |
| qwen3_8b | item built in both tokenizer families | False | 0 |  |  |  |  |
| qwen3_8b | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| qwen3_8b | probe-neutral (both triggers |probe| < 1) | False | 252 | 8.84 | 0.136 [0.122, 0.151] | 0.412 [0.395, 0.429] | 0.55 [0.53, 0.57] |
| qwen3_8b | probe-associative | True | 0 |  |  |  |  |
| qwen3_8b | probe-associative | False | 252 | 8.84 | 0.136 [0.122, 0.151] | 0.412 [0.395, 0.429] | 0.55 [0.53, 0.57] |
| qwen3_8b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_8b | probe one-sided | False | 252 | 8.84 | 0.136 [0.122, 0.151] | 0.412 [0.395, 0.429] | 0.55 [0.53, 0.57] |
| qwen3_8b | probe reverse | True | 0 |  |  |  |  |
| qwen3_8b | probe reverse | False | 252 | 8.84 | 0.136 [0.122, 0.151] | 0.412 [0.395, 0.429] | 0.55 [0.53, 0.57] |
| qwen3_8b | top-1 in both directions | True | 96 | 9.51 | 0.146 [0.120, 0.172] | 0.378 [0.353, 0.404] | 0.55 [0.52, 0.57] |
| qwen3_8b | top-1 in both directions | False | 156 | 8.42 | 0.130 [0.112, 0.148] | 0.435 [0.414, 0.456] | 0.55 [0.53, 0.58] |
