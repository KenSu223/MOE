**e0_e2a W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | person-name options | False | 200 | 4.63 | 0.141 [0.122, 0.161] | 0.222 [0.197, 0.247] | 0.44 [0.42, 0.46] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | True | 100 | 4.66 | 0.136 [0.109, 0.165] | 0.207 [0.174, 0.239] | 0.43 [0.41, 0.46] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | False | 100 | 4.62 | 0.146 [0.121, 0.174] | 0.238 [0.199, 0.276] | 0.45 [0.42, 0.48] |
| Qwen3-30B-A3B-Base | core-776 twin | True | 8 |  |  |  |  |
| Qwen3-30B-A3B-Base | core-776 twin | False | 192 | 4.58 | 0.143 [0.124, 0.164] | 0.230 [0.206, 0.256] | 0.45 [0.42, 0.47] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | True | 104 | 4.52 | 0.155 [0.128, 0.186] | 0.219 [0.186, 0.254] | 0.46 [0.43, 0.49] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | False | 96 | 4.76 | 0.126 [0.103, 0.153] | 0.226 [0.190, 0.263] | 0.42 [0.39, 0.45] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 200 | 4.63 | 0.141 [0.122, 0.161] | 0.222 [0.197, 0.247] | 0.44 [0.42, 0.46] |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 200 | 4.63 | 0.141 [0.122, 0.161] | 0.222 [0.197, 0.247] | 0.44 [0.42, 0.46] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe one-sided | False | 200 | 4.63 | 0.141 [0.122, 0.161] | 0.222 [0.197, 0.247] | 0.44 [0.42, 0.46] |
| Qwen3-30B-A3B-Base | probe reverse | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe reverse | False | 200 | 4.63 | 0.141 [0.122, 0.161] | 0.222 [0.197, 0.247] | 0.44 [0.42, 0.46] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 4 |  |  |  |  |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 196 | 4.56 | 0.142 [0.123, 0.162] | 0.227 [0.202, 0.253] | 0.44 [0.42, 0.47] |
