**e1p_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 54 | 8.56 | 0.137 [0.127, 0.147] | 0.317 [0.303, 0.332] | 0.90 [0.87, 0.91] |
| Qwen3-30B-A3B-Base | person-name options | False | 202 | 5.52 | 0.061 [0.055, 0.068] | 0.295 [0.285, 0.304] | 0.78 [0.77, 0.79] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | True | 128 | 6.14 | 0.087 [0.076, 0.097] | 0.304 [0.293, 0.315] | 0.83 [0.81, 0.85] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | False | 128 | 6.18 | 0.080 [0.071, 0.090] | 0.298 [0.287, 0.310] | 0.82 [0.79, 0.84] |
| Qwen3-30B-A3B-Base | core-776 twin | True | 210 | 5.94 | 0.080 [0.072, 0.087] | 0.303 [0.294, 0.312] | 0.82 [0.80, 0.84] |
| Qwen3-30B-A3B-Base | core-776 twin | False | 46 | 7.15 | 0.098 [0.082, 0.114] | 0.294 [0.276, 0.312] | 0.84 [0.81, 0.86] |
| Qwen3-30B-A3B-Base | degree adverb in the blank clause (too / very / more ...) | True | 158 | 5.86 | 0.067 [0.059, 0.076] | 0.303 [0.294, 0.312] | 0.83 [0.81, 0.84] |
| Qwen3-30B-A3B-Base | degree adverb in the blank clause (too / very / more ...) | False | 98 | 6.65 | 0.107 [0.096, 0.117] | 0.299 [0.284, 0.313] | 0.82 [0.79, 0.84] |
| Qwen3-30B-A3B-Base | complementary antonym pair | True | 2 |  |  |  |  |
| Qwen3-30B-A3B-Base | complementary antonym pair | False | 254 | 6.18 | 0.084 [0.077, 0.090] | 0.301 [0.293, 0.309] | 0.83 [0.81, 0.84] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | True | 230 | 6.05 | 0.080 [0.073, 0.087] | 0.304 [0.295, 0.312] | 0.83 [0.81, 0.84] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | False | 26 | 7.14 | 0.111 [0.088, 0.136] | 0.284 [0.259, 0.312] | 0.82 [0.78, 0.85] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 256 | 6.16 | 0.083 [0.076, 0.090] | 0.301 [0.293, 0.309] | 0.83 [0.81, 0.84] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 256 | 6.16 | 0.083 [0.076, 0.090] | 0.301 [0.293, 0.309] | 0.83 [0.81, 0.84] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe one-sided | False | 256 | 6.16 | 0.083 [0.076, 0.090] | 0.301 [0.293, 0.309] | 0.83 [0.81, 0.84] |
| Qwen3-30B-A3B-Base | probe reverse | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe reverse | False | 256 | 6.16 | 0.083 [0.076, 0.090] | 0.301 [0.293, 0.309] | 0.83 [0.81, 0.84] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 248 | 6.20 | 0.085 [0.078, 0.092] | 0.301 [0.292, 0.309] | 0.83 [0.81, 0.84] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 8 |  |  |  |  |
