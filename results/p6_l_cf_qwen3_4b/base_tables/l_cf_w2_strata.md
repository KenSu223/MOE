**l_cf W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_4b | person-name options | True | 0 |  |  |  |  |
| qwen3_4b | person-name options | False | 173 | 11.11 | 0.185 [0.169, 0.201] | 0.265 [0.246, 0.285] | 0.56 [0.54, 0.57] |
| qwen3_4b | statement order 1 (entity of A mentioned first) | True | 173 | 11.11 | 0.185 [0.169, 0.201] | 0.265 [0.246, 0.285] | 0.56 [0.54, 0.57] |
| qwen3_4b | statement order 1 (entity of A mentioned first) | False | 0 |  |  |  |  |
| qwen3_4b | item built in both tokenizer families | True | 0 |  |  |  |  |
| qwen3_4b | item built in both tokenizer families | False | 173 | 11.11 | 0.185 [0.169, 0.201] | 0.265 [0.246, 0.285] | 0.56 [0.54, 0.57] |
| qwen3_4b | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| qwen3_4b | probe-neutral (both triggers |probe| < 1) | False | 173 | 11.11 | 0.185 [0.169, 0.201] | 0.265 [0.246, 0.285] | 0.56 [0.54, 0.57] |
| qwen3_4b | probe-associative | True | 0 |  |  |  |  |
| qwen3_4b | probe-associative | False | 173 | 11.11 | 0.185 [0.169, 0.201] | 0.265 [0.246, 0.285] | 0.56 [0.54, 0.57] |
| qwen3_4b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_4b | probe one-sided | False | 173 | 11.11 | 0.185 [0.169, 0.201] | 0.265 [0.246, 0.285] | 0.56 [0.54, 0.57] |
| qwen3_4b | probe reverse | True | 0 |  |  |  |  |
| qwen3_4b | probe reverse | False | 173 | 11.11 | 0.185 [0.169, 0.201] | 0.265 [0.246, 0.285] | 0.56 [0.54, 0.57] |
| qwen3_4b | top-1 in both directions | True | 28 | 15.07 | 0.189 [0.153, 0.228] | 0.294 [0.252, 0.342] | 0.56 [0.53, 0.58] |
| qwen3_4b | top-1 in both directions | False | 145 | 10.34 | 0.184 [0.167, 0.201] | 0.257 [0.236, 0.279] | 0.56 [0.54, 0.58] |
