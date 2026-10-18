**aagr_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | person-name options | False | 256 | 11.00 | 0.066 [0.061, 0.071] | 0.247 [0.241, 0.254] | 0.80 [0.79, 0.81] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | True | 128 | 11.09 | 0.067 [0.060, 0.074] | 0.245 [0.237, 0.254] | 0.79 [0.78, 0.80] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | False | 128 | 10.91 | 0.066 [0.059, 0.072] | 0.249 [0.240, 0.258] | 0.80 [0.79, 0.81] |
| Qwen3-30B-A3B-Base | core-776 twin | True | 18 | 9.30 | 0.055 [0.040, 0.073] | 0.248 [0.224, 0.270] | 0.82 [0.80, 0.83] |
| Qwen3-30B-A3B-Base | core-776 twin | False | 238 | 11.13 | 0.067 [0.062, 0.072] | 0.247 [0.241, 0.254] | 0.80 [0.79, 0.80] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | False | 256 | 11.00 | 0.066 [0.061, 0.071] | 0.247 [0.241, 0.254] | 0.80 [0.79, 0.81] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 256 | 11.00 | 0.066 [0.061, 0.071] | 0.247 [0.241, 0.254] | 0.80 [0.79, 0.81] |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 256 | 11.00 | 0.066 [0.061, 0.071] | 0.247 [0.241, 0.254] | 0.80 [0.79, 0.81] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe one-sided | False | 256 | 11.00 | 0.066 [0.061, 0.071] | 0.247 [0.241, 0.254] | 0.80 [0.79, 0.81] |
| Qwen3-30B-A3B-Base | probe reverse | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe reverse | False | 256 | 11.00 | 0.066 [0.061, 0.071] | 0.247 [0.241, 0.254] | 0.80 [0.79, 0.81] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 251 | 11.04 | 0.067 [0.062, 0.072] | 0.247 [0.241, 0.253] | 0.80 [0.79, 0.81] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 5 |  |  |  |  |
