**e3p3_a2e W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| qwen3_4b | person-name options | True | 46 | 4.52 | 0.133 [0.121, 0.146] | 0.456 [0.431, 0.481] | 0.73 [0.72, 0.75] |
| qwen3_4b | person-name options | False | 113 | 5.44 | 0.131 [0.121, 0.141] | 0.236 [0.220, 0.252] | 0.65 [0.63, 0.66] |
| qwen3_4b | statement order 1 (entity of A mentioned first) | True | 132 | 5.09 | 0.130 [0.121, 0.139] | 0.311 [0.287, 0.335] | 0.69 [0.67, 0.70] |
| qwen3_4b | statement order 1 (entity of A mentioned first) | False | 27 | 5.58 | 0.138 [0.121, 0.156] | 0.206 [0.176, 0.238] | 0.62 [0.58, 0.65] |
| qwen3_4b | core-776 twin | True | 67 | 5.40 | 0.129 [0.116, 0.143] | 0.236 [0.212, 0.261] | 0.65 [0.63, 0.67] |
| qwen3_4b | core-776 twin | False | 92 | 5.01 | 0.133 [0.123, 0.144] | 0.335 [0.307, 0.364] | 0.69 [0.67, 0.70] |
| qwen3_4b | degree adverb in the blank clause (too / very / more ...) | True | 75 | 5.68 | 0.131 [0.119, 0.144] | 0.275 [0.245, 0.309] | 0.66 [0.64, 0.69] |
| qwen3_4b | degree adverb in the blank clause (too / very / more ...) | False | 84 | 4.72 | 0.132 [0.122, 0.142] | 0.310 [0.283, 0.338] | 0.68 [0.67, 0.70] |
| qwen3_4b | complementary antonym pair | True | 1 |  |  |  |  |
| qwen3_4b | complementary antonym pair | False | 158 | 5.19 | 0.131 [0.123, 0.139] | 0.291 [0.270, 0.314] | 0.67 [0.66, 0.69] |
| qwen3_4b | item built in both tokenizer families | True | 102 | 5.31 | 0.130 [0.119, 0.141] | 0.259 [0.235, 0.282] | 0.66 [0.64, 0.67] |
| qwen3_4b | item built in both tokenizer families | False | 57 | 4.93 | 0.135 [0.123, 0.147] | 0.355 [0.320, 0.389] | 0.70 [0.67, 0.72] |
| qwen3_4b | probe-neutral (both triggers |probe| < 1) | True | 159 | 5.17 | 0.131 [0.123, 0.139] | 0.292 [0.271, 0.313] | 0.67 [0.66, 0.69] |
| qwen3_4b | probe-neutral (both triggers |probe| < 1) | False | 0 |  |  |  |  |
| qwen3_4b | probe-associative | True | 0 |  |  |  |  |
| qwen3_4b | probe-associative | False | 159 | 5.17 | 0.131 [0.123, 0.139] | 0.292 [0.271, 0.313] | 0.67 [0.66, 0.69] |
| qwen3_4b | probe one-sided | True | 0 |  |  |  |  |
| qwen3_4b | probe one-sided | False | 159 | 5.17 | 0.131 [0.123, 0.139] | 0.292 [0.271, 0.313] | 0.67 [0.66, 0.69] |
| qwen3_4b | probe reverse | True | 0 |  |  |  |  |
| qwen3_4b | probe reverse | False | 159 | 5.17 | 0.131 [0.123, 0.139] | 0.292 [0.271, 0.313] | 0.67 [0.66, 0.69] |
| qwen3_4b | top-1 in both directions | True | 107 | 5.03 | 0.133 [0.124, 0.143] | 0.289 [0.266, 0.316] | 0.67 [0.65, 0.69] |
| qwen3_4b | top-1 in both directions | False | 52 | 5.47 | 0.129 [0.114, 0.143] | 0.296 [0.259, 0.335] | 0.68 [0.66, 0.70] |
