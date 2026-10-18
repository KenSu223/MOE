**rsite_e1_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 20 | 9.30 | 0.016 [0.008, 0.023] | 0.448 [0.408, 0.491] | 0.96 [0.92, 0.97] |
| Qwen3-30B-A3B-Base | person-name options | False | 236 | 6.91 | 0.056 [0.051, 0.062] | 0.363 [0.353, 0.373] | 0.82 [0.80, 0.83] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | True | 128 | 7.56 | 0.053 [0.046, 0.061] | 0.386 [0.371, 0.402] | 0.84 [0.82, 0.85] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | False | 128 | 6.62 | 0.051 [0.044, 0.059] | 0.354 [0.341, 0.369] | 0.82 [0.80, 0.84] |
| Qwen3-30B-A3B-Base | core-776 twin | True | 256 | 7.09 | 0.052 [0.047, 0.058] | 0.372 [0.361, 0.382] | 0.83 [0.82, 0.84] |
| Qwen3-30B-A3B-Base | core-776 twin | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | degree adverb in the blank clause (too / very / more ...) | True | 158 | 6.86 | 0.047 [0.041, 0.052] | 0.353 [0.340, 0.367] | 0.82 [0.80, 0.83] |
| Qwen3-30B-A3B-Base | degree adverb in the blank clause (too / very / more ...) | False | 98 | 7.46 | 0.060 [0.051, 0.071] | 0.399 [0.382, 0.416] | 0.85 [0.81, 0.87] |
| Qwen3-30B-A3B-Base | complementary antonym pair | True | 2 |  |  |  |  |
| Qwen3-30B-A3B-Base | complementary antonym pair | False | 254 | 7.09 | 0.052 [0.047, 0.058] | 0.371 [0.361, 0.382] | 0.83 [0.82, 0.84] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | True | 252 | 7.06 | 0.050 [0.045, 0.055] | 0.371 [0.361, 0.382] | 0.83 [0.82, 0.84] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | False | 4 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 256 | 7.09 | 0.052 [0.047, 0.058] | 0.372 [0.361, 0.382] | 0.83 [0.82, 0.84] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 256 | 7.09 | 0.052 [0.047, 0.058] | 0.372 [0.361, 0.382] | 0.83 [0.82, 0.84] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe one-sided | False | 256 | 7.09 | 0.052 [0.047, 0.058] | 0.372 [0.361, 0.382] | 0.83 [0.82, 0.84] |
| Qwen3-30B-A3B-Base | probe reverse | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe reverse | False | 256 | 7.09 | 0.052 [0.047, 0.058] | 0.372 [0.361, 0.382] | 0.83 [0.82, 0.84] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 241 | 7.21 | 0.052 [0.047, 0.058] | 0.375 [0.364, 0.385] | 0.83 [0.82, 0.84] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 15 | 5.30 | 0.050 [0.036, 0.066] | 0.304 [0.276, 0.335] | 0.78 [0.71, 0.81] |
