**e1_e2a_all W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | person-name options | True | 6 |  |  |  |  |
| Qwen3-30B-A3B-Base | person-name options | False | 250 | 9.76 | 0.169 [0.156, 0.181] | 0.068 [0.059, 0.078] | 0.28 [0.25, 0.30] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | True | 128 | 9.54 | 0.175 [0.157, 0.191] | 0.070 [0.056, 0.085] | 0.27 [0.24, 0.31] |
| Qwen3-30B-A3B-Base | statement order 1 (entity of A mentioned first) | False | 128 | 9.75 | 0.159 [0.142, 0.176] | 0.069 [0.057, 0.082] | 0.29 [0.26, 0.33] |
| Qwen3-30B-A3B-Base | core-776 twin | True | 256 | 9.65 | 0.167 [0.155, 0.179] | 0.069 [0.060, 0.079] | 0.28 [0.26, 0.31] |
| Qwen3-30B-A3B-Base | core-776 twin | False | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | degree adverb in the blank clause (too / very / more ...) | True | 200 | 10.41 | 0.178 [0.165, 0.191] | 0.067 [0.057, 0.078] | 0.26 [0.24, 0.29] |
| Qwen3-30B-A3B-Base | degree adverb in the blank clause (too / very / more ...) | False | 56 | 6.93 | 0.104 [0.079, 0.129] | 0.082 [0.065, 0.100] | 0.37 [0.33, 0.42] |
| Qwen3-30B-A3B-Base | complementary antonym pair | True | 0 |  |  |  |  |
| Qwen3-30B-A3B-Base | complementary antonym pair | False | 256 | 9.65 | 0.167 [0.155, 0.179] | 0.069 [0.060, 0.079] | 0.28 [0.26, 0.31] |
| Qwen3-30B-A3B-Base | association-neutral by the model's own probes (_all cells) | True | 64 | 8.95 | 0.172 [0.149, 0.194] | 0.064 [0.048, 0.081] | 0.25 [0.22, 0.30] |
| Qwen3-30B-A3B-Base | association-neutral by the model's own probes (_all cells) | False | 192 | 9.88 | 0.165 [0.150, 0.179] | 0.071 [0.060, 0.083] | 0.29 [0.26, 0.32] |
| Qwen3-30B-A3B-Base | association-neutral by the twin / counterpart's probes (_all cells) | True | 92 | 8.67 | 0.147 [0.125, 0.169] | 0.082 [0.067, 0.099] | 0.33 [0.30, 0.37] |
| Qwen3-30B-A3B-Base | association-neutral by the twin / counterpart's probes (_all cells) | False | 164 | 10.19 | 0.176 [0.162, 0.190] | 0.063 [0.052, 0.075] | 0.26 [0.23, 0.29] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | True | 254 | 9.67 | 0.166 [0.154, 0.178] | 0.069 [0.060, 0.079] | 0.28 [0.26, 0.31] |
| Qwen3-30B-A3B-Base | item built in both tokenizer families | False | 2 |  |  |  |  |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | True | 64 | 8.95 | 0.172 [0.149, 0.194] | 0.064 [0.048, 0.081] | 0.25 [0.22, 0.30] |
| Qwen3-30B-A3B-Base | probe-neutral (both triggers |probe| < 1) | False | 192 | 9.88 | 0.165 [0.150, 0.179] | 0.071 [0.060, 0.083] | 0.29 [0.26, 0.32] |
| Qwen3-30B-A3B-Base | probe-associative | True | 16 | 9.97 | 0.171 [0.123, 0.216] | 0.083 [0.031, 0.149] | 0.32 [0.23, 0.43] |
| Qwen3-30B-A3B-Base | probe-associative | False | 240 | 9.63 | 0.166 [0.154, 0.179] | 0.068 [0.060, 0.078] | 0.28 [0.26, 0.31] |
| Qwen3-30B-A3B-Base | probe one-sided | True | 76 | 9.12 | 0.119 [0.099, 0.138] | 0.093 [0.073, 0.116] | 0.34 [0.30, 0.39] |
| Qwen3-30B-A3B-Base | probe one-sided | False | 180 | 9.87 | 0.185 [0.171, 0.199] | 0.060 [0.051, 0.070] | 0.26 [0.23, 0.29] |
| Qwen3-30B-A3B-Base | probe reverse | True | 100 | 10.44 | 0.195 [0.175, 0.214] | 0.054 [0.045, 0.065] | 0.26 [0.23, 0.30] |
| Qwen3-30B-A3B-Base | probe reverse | False | 156 | 9.14 | 0.146 [0.131, 0.161] | 0.080 [0.067, 0.095] | 0.30 [0.27, 0.33] |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 53 | 8.75 | 0.142 [0.120, 0.162] | 0.144 [0.120, 0.170] | 0.40 [0.37, 0.43] |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 203 | 9.88 | 0.172 [0.158, 0.186] | 0.052 [0.044, 0.060] | 0.24 [0.22, 0.27] |
