**l_ant W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen2_57b | person-name options | True | 0 |  |  |  |  |
| qwen2_57b | person-name options | False | 252 | 8.92 | 0.303 [0.285, 0.322] | 0.404 [0.388, 0.421] | 0.35 [0.34, 0.36] |
| qwen2_57b | statement order 1 (entity of A mentioned first) | True | 252 | 8.92 | 0.303 [0.285, 0.322] | 0.404 [0.388, 0.421] | 0.35 [0.34, 0.36] |
| qwen2_57b | statement order 1 (entity of A mentioned first) | False | 0 |  |  |  |  |
| qwen2_57b | item built in both tokenizer families | True | 252 | 8.92 | 0.303 [0.285, 0.322] | 0.404 [0.388, 0.421] | 0.35 [0.34, 0.36] |
| qwen2_57b | item built in both tokenizer families | False | 0 |  |  |  |  |
| qwen2_57b | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| qwen2_57b | probe-neutral (both triggers |probe| < 1) | False | 252 | 8.92 | 0.303 [0.285, 0.322] | 0.404 [0.388, 0.421] | 0.35 [0.34, 0.36] |
| qwen2_57b | probe-associative | True | 0 |  |  |  |  |
| qwen2_57b | probe-associative | False | 252 | 8.92 | 0.303 [0.285, 0.322] | 0.404 [0.388, 0.421] | 0.35 [0.34, 0.36] |
| qwen2_57b | probe one-sided | True | 0 |  |  |  |  |
| qwen2_57b | probe one-sided | False | 252 | 8.92 | 0.303 [0.285, 0.322] | 0.404 [0.388, 0.421] | 0.35 [0.34, 0.36] |
| qwen2_57b | probe reverse | True | 0 |  |  |  |  |
| qwen2_57b | probe reverse | False | 252 | 8.92 | 0.303 [0.285, 0.322] | 0.404 [0.388, 0.421] | 0.35 [0.34, 0.36] |
| qwen2_57b | top-1 in both directions | True | 103 | 10.12 | 0.324 [0.296, 0.354] | 0.416 [0.391, 0.443] | 0.37 [0.35, 0.38] |
| qwen2_57b | top-1 in both directions | False | 149 | 8.09 | 0.285 [0.263, 0.309] | 0.394 [0.373, 0.416] | 0.35 [0.34, 0.36] |
