**l_cf W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen2_57b | person-name options | True | 0 |  |  |  |  |
| qwen2_57b | person-name options | False | 202 | 13.08 | 0.244 [0.229, 0.258] | 0.297 [0.275, 0.318] | 0.42 [0.40, 0.44] |
| qwen2_57b | statement order 1 (entity of A mentioned first) | True | 202 | 13.08 | 0.244 [0.229, 0.258] | 0.297 [0.275, 0.318] | 0.42 [0.40, 0.44] |
| qwen2_57b | statement order 1 (entity of A mentioned first) | False | 0 |  |  |  |  |
| qwen2_57b | item built in both tokenizer families | True | 0 |  |  |  |  |
| qwen2_57b | item built in both tokenizer families | False | 202 | 13.08 | 0.244 [0.229, 0.258] | 0.297 [0.275, 0.318] | 0.42 [0.40, 0.44] |
| qwen2_57b | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| qwen2_57b | probe-neutral (both triggers |probe| < 1) | False | 202 | 13.08 | 0.244 [0.229, 0.258] | 0.297 [0.275, 0.318] | 0.42 [0.40, 0.44] |
| qwen2_57b | probe-associative | True | 0 |  |  |  |  |
| qwen2_57b | probe-associative | False | 202 | 13.08 | 0.244 [0.229, 0.258] | 0.297 [0.275, 0.318] | 0.42 [0.40, 0.44] |
| qwen2_57b | probe one-sided | True | 0 |  |  |  |  |
| qwen2_57b | probe one-sided | False | 202 | 13.08 | 0.244 [0.229, 0.258] | 0.297 [0.275, 0.318] | 0.42 [0.40, 0.44] |
| qwen2_57b | probe reverse | True | 0 |  |  |  |  |
| qwen2_57b | probe reverse | False | 202 | 13.08 | 0.244 [0.229, 0.258] | 0.297 [0.275, 0.318] | 0.42 [0.40, 0.44] |
| qwen2_57b | top-1 in both directions | True | 55 | 16.00 | 0.288 [0.264, 0.310] | 0.320 [0.284, 0.357] | 0.42 [0.39, 0.44] |
| qwen2_57b | top-1 in both directions | False | 147 | 11.98 | 0.222 [0.205, 0.239] | 0.285 [0.259, 0.312] | 0.41 [0.39, 0.43] |
