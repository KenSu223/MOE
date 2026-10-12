**vt_a2e_orig W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 97 | 3.38 | 0.089 [0.079, 0.099] | 0.344 [0.330, 0.359] | 0.76 [0.74, 0.79] |
| Qwen3-30B-A3B-Base | person-name options | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 77 | 3.33 | 0.093 [0.081, 0.105] | 0.334 [0.318, 0.351] | 0.76 [0.73, 0.78] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 20 | 3.57 | 0.074 [0.055, 0.092] | 0.380 [0.353, 0.411] | 0.78 [0.73, 0.81] |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 97 | 3.38 | 0.089 [0.079, 0.099] | 0.344 [0.330, 0.359] | 0.76 [0.74, 0.79] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 8 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe one-sided | False | 89 | 3.37 | 0.091 [0.081, 0.102] | 0.343 [0.327, 0.358] | 0.76 [0.74, 0.79] |
| Qwen3-30B-A3B-Base | probe reverse | True | 12 | 3.61 | 0.081 [0.056, 0.106] | 0.391 [0.355, 0.438] | 0.78 [0.73, 0.83] |
| Qwen3-30B-A3B-Base | probe reverse | False | 85 | 3.35 | 0.090 [0.079, 0.101] | 0.337 [0.322, 0.352] | 0.76 [0.73, 0.79] |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | False | 97 | 3.38 | 0.089 [0.079, 0.099] | 0.344 [0.330, 0.359] | 0.76 [0.74, 0.79] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 87 | 3.44 | 0.086 [0.076, 0.096] | 0.348 [0.333, 0.363] | 0.77 [0.74, 0.79] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 10 | 2.84 | 0.123 [0.093, 0.159] | 0.305 [0.264, 0.353] | 0.70 [0.63, 0.77] |
