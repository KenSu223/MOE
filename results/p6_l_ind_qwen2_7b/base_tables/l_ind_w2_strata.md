**l_ind W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen2_7b | person-name options | True | 0 |  |  |  |  |
| qwen2_7b | person-name options | False | 184 | 4.30 | 0.146 [0.137, 0.155] | 0.334 [0.313, 0.356] | 0.68 [0.66, 0.70] |
| qwen2_7b | statement order 1 (entity of A mentioned first) | True | 92 | 4.50 | 0.154 [0.141, 0.166] | 0.335 [0.304, 0.369] | 0.67 [0.65, 0.69] |
| qwen2_7b | statement order 1 (entity of A mentioned first) | False | 92 | 4.10 | 0.138 [0.125, 0.151] | 0.333 [0.303, 0.362] | 0.69 [0.67, 0.71] |
| qwen2_7b | item built in both tokenizer families | True | 184 | 4.30 | 0.146 [0.137, 0.155] | 0.334 [0.313, 0.356] | 0.68 [0.66, 0.70] |
| qwen2_7b | item built in both tokenizer families | False | 0 |  |  |  |  |
| qwen2_7b | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| qwen2_7b | probe-neutral (both triggers |probe| < 1) | False | 184 | 4.30 | 0.146 [0.137, 0.155] | 0.334 [0.313, 0.356] | 0.68 [0.66, 0.70] |
| qwen2_7b | probe-associative | True | 0 |  |  |  |  |
| qwen2_7b | probe-associative | False | 184 | 4.30 | 0.146 [0.137, 0.155] | 0.334 [0.313, 0.356] | 0.68 [0.66, 0.70] |
| qwen2_7b | probe one-sided | True | 0 |  |  |  |  |
| qwen2_7b | probe one-sided | False | 184 | 4.30 | 0.146 [0.137, 0.155] | 0.334 [0.313, 0.356] | 0.68 [0.66, 0.70] |
| qwen2_7b | probe reverse | True | 0 |  |  |  |  |
| qwen2_7b | probe reverse | False | 184 | 4.30 | 0.146 [0.137, 0.155] | 0.334 [0.313, 0.356] | 0.68 [0.66, 0.70] |
| qwen2_7b | top-1 in both directions | True | 162 | 4.29 | 0.145 [0.135, 0.155] | 0.331 [0.308, 0.355] | 0.69 [0.67, 0.71] |
| qwen2_7b | top-1 in both directions | False | 22 | 4.37 | 0.153 [0.126, 0.181] | 0.360 [0.321, 0.403] | 0.65 [0.61, 0.68] |
