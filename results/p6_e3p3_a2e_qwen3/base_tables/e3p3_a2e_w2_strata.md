**e3p3_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 34 | 4.19 | 0.170 [0.146, 0.194] | 0.346 [0.309, 0.385] | 0.87 [0.82, 0.89] |
| Qwen3-30B-A3B-Base | person-name options | False | 94 | 4.80 | 0.065 [0.053, 0.079] | 0.242 [0.218, 0.267] | 0.70 [0.68, 0.72] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | True | 97 | 4.63 | 0.103 [0.087, 0.120] | 0.283 [0.260, 0.306] | 0.77 [0.73, 0.79] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | False | 31 | 4.66 | 0.051 [0.032, 0.071] | 0.218 [0.173, 0.266] | 0.69 [0.64, 0.72] |
| Qwen3-30B-A3B-Base | core-776 twin | True | 58 | 4.82 | 0.074 [0.055, 0.094] | 0.240 [0.213, 0.267] | 0.72 [0.68, 0.75] |
| Qwen3-30B-A3B-Base | core-776 twin | False | 70 | 4.49 | 0.105 [0.086, 0.125] | 0.291 [0.258, 0.323] | 0.77 [0.73, 0.80] |
| Qwen3-30B-A3B-Base | degree adverb in the blank clause (too / very / more ...) | True | 57 | 4.88 | 0.073 [0.053, 0.094] | 0.232 [0.201, 0.264] | 0.75 [0.70, 0.78] |
| Qwen3-30B-A3B-Base | degree adverb in the blank clause (too / very / more ...) | False | 71 | 4.45 | 0.105 [0.087, 0.125] | 0.298 [0.271, 0.327] | 0.75 [0.72, 0.77] |
| Qwen3-30B-A3B-Base | complementary antonym pair | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | complementary antonym pair | False | 128 | 4.64 | 0.090 [0.077, 0.104] | 0.267 [0.245, 0.289] | 0.75 [0.72, 0.77] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | True | 79 | 4.61 | 0.074 [0.057, 0.092] | 0.242 [0.216, 0.268] | 0.74 [0.71, 0.76] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | False | 49 | 4.70 | 0.116 [0.094, 0.139] | 0.307 [0.274, 0.341] | 0.76 [0.71, 0.80] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 128 | 4.64 | 0.090 [0.077, 0.104] | 0.267 [0.245, 0.289] | 0.75 [0.72, 0.77] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-associative | False | 128 | 4.64 | 0.090 [0.077, 0.104] | 0.267 [0.245, 0.289] | 0.75 [0.72, 0.77] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe one-sided | False | 128 | 4.64 | 0.090 [0.077, 0.104] | 0.267 [0.245, 0.289] | 0.75 [0.72, 0.77] |
| Qwen3-30B-A3B-Base | probe reverse | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe reverse | False | 128 | 4.64 | 0.090 [0.077, 0.104] | 0.267 [0.245, 0.289] | 0.75 [0.72, 0.77] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 96 | 4.57 | 0.105 [0.090, 0.122] | 0.281 [0.258, 0.306] | 0.77 [0.75, 0.80] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 32 | 4.85 | 0.048 [0.031, 0.068] | 0.228 [0.184, 0.274] | 0.67 [0.63, 0.70] |
