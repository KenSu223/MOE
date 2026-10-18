**l_ind W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | person-name options | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | person-name options | False | 256 | 4.54 | 0.086 [0.081, 0.090] | 0.400 [0.386, 0.414] | 0.72 [0.71, 0.73] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | True | 128 | 4.49 | 0.086 [0.079, 0.093] | 0.406 [0.384, 0.427] | 0.73 [0.71, 0.74] |
| Mixtral-8x7B, BOS | statement order 1 (entity of A mentioned first) | False | 128 | 4.60 | 0.086 [0.080, 0.092] | 0.395 [0.375, 0.414] | 0.72 [0.70, 0.73] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | True | 256 | 4.54 | 0.086 [0.081, 0.090] | 0.400 [0.386, 0.414] | 0.72 [0.71, 0.73] |
| Mixtral-8x7B, BOS | item built in both tokenizer families | False | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-neutral (both triggers |probe| < 1) | False | 256 | 4.54 | 0.086 [0.081, 0.090] | 0.400 [0.386, 0.414] | 0.72 [0.71, 0.73] |
| Mixtral-8x7B, BOS | probe-associative | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe-associative | False | 256 | 4.54 | 0.086 [0.081, 0.090] | 0.400 [0.386, 0.414] | 0.72 [0.71, 0.73] |
| Mixtral-8x7B, BOS | probe one-sided | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe one-sided | False | 256 | 4.54 | 0.086 [0.081, 0.090] | 0.400 [0.386, 0.414] | 0.72 [0.71, 0.73] |
| Mixtral-8x7B, BOS | probe reverse | True | 0 |  |  |  |  |
| Mixtral-8x7B, BOS | probe reverse | False | 256 | 4.54 | 0.086 [0.081, 0.090] | 0.400 [0.386, 0.414] | 0.72 [0.71, 0.73] |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 215 | 4.52 | 0.085 [0.081, 0.090] | 0.399 [0.384, 0.413] | 0.72 [0.71, 0.73] |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 41 | 4.65 | 0.090 [0.073, 0.105] | 0.408 [0.364, 0.451] | 0.73 [0.70, 0.76] |
