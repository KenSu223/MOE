**e0_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | person-name options | False | 256 | 7.34 | 0.108 [0.101, 0.114] | 0.219 [0.211, 0.227] | 0.73 [0.72, 0.74] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | True | 128 | 7.47 | 0.107 [0.098, 0.116] | 0.216 [0.205, 0.227] | 0.73 [0.71, 0.74] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | False | 128 | 7.21 | 0.108 [0.100, 0.117] | 0.222 [0.211, 0.233] | 0.74 [0.72, 0.75] |
| Qwen3-30B-A3B-Base | core-776 twin | True | 8 |  |  |  |  |
| Qwen3-30B-A3B-Base | core-776 twin | False | 248 | 7.37 | 0.109 [0.103, 0.115] | 0.221 [0.213, 0.228] | 0.73 [0.72, 0.74] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | True | 136 | 6.72 | 0.110 [0.101, 0.118] | 0.229 [0.219, 0.239] | 0.73 [0.72, 0.74] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | False | 120 | 8.04 | 0.106 [0.097, 0.115] | 0.210 [0.199, 0.221] | 0.73 [0.71, 0.74] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 256 | 7.34 | 0.108 [0.101, 0.114] | 0.219 [0.211, 0.227] | 0.73 [0.72, 0.74] |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 256 | 7.34 | 0.108 [0.101, 0.114] | 0.219 [0.211, 0.227] | 0.73 [0.72, 0.74] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe one-sided | False | 256 | 7.34 | 0.108 [0.101, 0.114] | 0.219 [0.211, 0.227] | 0.73 [0.72, 0.74] |
| Qwen3-30B-A3B-Base | probe reverse | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe reverse | False | 256 | 7.34 | 0.108 [0.101, 0.114] | 0.219 [0.211, 0.227] | 0.73 [0.72, 0.74] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 256 | 7.34 | 0.108 [0.101, 0.114] | 0.219 [0.211, 0.227] | 0.73 [0.72, 0.74] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 0 |  |  |  |  |
