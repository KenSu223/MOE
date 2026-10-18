**l_prop W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen2_7b | person-name options | True | 0 |  |  |  |  |
| qwen2_7b | person-name options | False | 256 | 9.43 | 0.321 [0.303, 0.341] | 0.234 [0.223, 0.246] | 0.35 [0.34, 0.36] |
| qwen2_7b | statement order 1 (entity of A mentioned first) | True | 256 | 9.43 | 0.321 [0.303, 0.341] | 0.234 [0.223, 0.246] | 0.35 [0.34, 0.36] |
| qwen2_7b | statement order 1 (entity of A mentioned first) | False | 0 |  |  |  |  |
| qwen2_7b | item built in both tokenizer families | True | 168 | 9.67 | 0.278 [0.255, 0.301] | 0.201 [0.188, 0.214] | 0.34 [0.33, 0.36] |
| qwen2_7b | item built in both tokenizer families | False | 88 | 8.99 | 0.410 [0.386, 0.436] | 0.302 [0.283, 0.321] | 0.36 [0.35, 0.37] |
| qwen2_7b | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| qwen2_7b | probe-neutral (both triggers |probe| < 1) | False | 256 | 9.43 | 0.321 [0.303, 0.341] | 0.234 [0.223, 0.246] | 0.35 [0.34, 0.36] |
| qwen2_7b | probe-associative | True | 0 |  |  |  |  |
| qwen2_7b | probe-associative | False | 256 | 9.43 | 0.321 [0.303, 0.341] | 0.234 [0.223, 0.246] | 0.35 [0.34, 0.36] |
| qwen2_7b | probe one-sided | True | 0 |  |  |  |  |
| qwen2_7b | probe one-sided | False | 256 | 9.43 | 0.321 [0.303, 0.341] | 0.234 [0.223, 0.246] | 0.35 [0.34, 0.36] |
| qwen2_7b | probe reverse | True | 0 |  |  |  |  |
| qwen2_7b | probe reverse | False | 256 | 9.43 | 0.321 [0.303, 0.341] | 0.234 [0.223, 0.246] | 0.35 [0.34, 0.36] |
| qwen2_7b | top-1 in both directions | True | 105 | 8.81 | 0.386 [0.364, 0.408] | 0.274 [0.256, 0.292] | 0.36 [0.35, 0.38] |
| qwen2_7b | top-1 in both directions | False | 151 | 9.87 | 0.281 [0.256, 0.306] | 0.209 [0.195, 0.224] | 0.34 [0.33, 0.35] |
