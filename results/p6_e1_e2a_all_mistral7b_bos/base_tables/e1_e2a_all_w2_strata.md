**e1_e2a_all W2 strata (main discovery + validation; layers fixed by the main discovery argmax; pair bootstrap)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Attention share |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | person-name options | True | 16 | 3.84 | 0.085 [0.055, 0.121] | 0.040 [0.006, 0.080] | 0.42 [0.36, 0.49] |
| mistral7b_bos | person-name options | False | 192 | 4.75 | 0.152 [0.140, 0.164] | 0.142 [0.120, 0.164] | 0.50 [0.47, 0.52] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | True | 104 | 4.73 | 0.145 [0.129, 0.161] | 0.137 [0.106, 0.170] | 0.50 [0.46, 0.53] |
| mistral7b_bos | statement order 1 (entity of A mentioned first) | False | 104 | 4.64 | 0.151 [0.134, 0.168] | 0.134 [0.106, 0.161] | 0.49 [0.45, 0.52] |
| mistral7b_bos | core-776 twin | True | 132 | 4.75 | 0.149 [0.136, 0.165] | 0.135 [0.111, 0.160] | 0.50 [0.47, 0.53] |
| mistral7b_bos | core-776 twin | False | 76 | 4.57 | 0.144 [0.125, 0.165] | 0.135 [0.097, 0.175] | 0.47 [0.43, 0.51] |
| mistral7b_bos | degree adverb in the blank clause (too / very / more ...) | True | 136 | 4.22 | 0.152 [0.137, 0.168] | 0.146 [0.117, 0.175] | 0.51 [0.48, 0.53] |
| mistral7b_bos | degree adverb in the blank clause (too / very / more ...) | False | 72 | 5.57 | 0.141 [0.124, 0.159] | 0.120 [0.091, 0.151] | 0.47 [0.43, 0.51] |
| mistral7b_bos | complementary antonym pair | True | 0 |  |  |  |  |
| mistral7b_bos | complementary antonym pair | False | 208 | 4.68 | 0.148 [0.136, 0.159] | 0.135 [0.114, 0.157] | 0.49 [0.47, 0.51] |
| mistral7b_bos | association-neutral by the model's own probes (_all cells) | True | 68 | 4.34 | 0.146 [0.121, 0.172] | 0.146 [0.110, 0.182] | 0.50 [0.46, 0.55] |
| mistral7b_bos | association-neutral by the model's own probes (_all cells) | False | 140 | 4.85 | 0.148 [0.136, 0.161] | 0.131 [0.106, 0.157] | 0.49 [0.46, 0.51] |
| mistral7b_bos | association-neutral by the twin / counterpart's probes (_all cells) | True | 80 | 4.54 | 0.147 [0.129, 0.168] | 0.126 [0.087, 0.168] | 0.48 [0.43, 0.52] |
| mistral7b_bos | association-neutral by the twin / counterpart's probes (_all cells) | False | 128 | 4.77 | 0.148 [0.133, 0.162] | 0.141 [0.118, 0.164] | 0.50 [0.47, 0.52] |
| mistral7b_bos | item built in both tokenizer families | True | 208 | 4.68 | 0.148 [0.136, 0.159] | 0.135 [0.114, 0.157] | 0.49 [0.47, 0.51] |
| mistral7b_bos | item built in both tokenizer families | False | 0 |  |  |  |  |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | True | 68 | 4.34 | 0.146 [0.121, 0.172] | 0.146 [0.110, 0.182] | 0.50 [0.46, 0.55] |
| mistral7b_bos | probe-neutral (both triggers |probe| < 1) | False | 140 | 4.85 | 0.148 [0.136, 0.161] | 0.131 [0.106, 0.157] | 0.49 [0.46, 0.51] |
| mistral7b_bos | probe-associative | True | 4 |  |  |  |  |
| mistral7b_bos | probe-associative | False | 204 | 4.68 | 0.148 [0.136, 0.160] | 0.134 [0.113, 0.156] | 0.49 [0.46, 0.51] |
| mistral7b_bos | probe one-sided | True | 80 | 5.28 | 0.150 [0.134, 0.167] | 0.115 [0.089, 0.141] | 0.48 [0.44, 0.51] |
| mistral7b_bos | probe one-sided | False | 128 | 4.31 | 0.145 [0.129, 0.162] | 0.151 [0.121, 0.181] | 0.50 [0.47, 0.53] |
| mistral7b_bos | probe reverse | True | 56 | 4.22 | 0.145 [0.124, 0.168] | 0.155 [0.103, 0.208] | 0.49 [0.44, 0.54] |
| mistral7b_bos | probe reverse | False | 152 | 4.86 | 0.148 [0.135, 0.162] | 0.129 [0.107, 0.150] | 0.49 [0.46, 0.52] |
| mistral7b_bos | top-1 in both directions | True | 102 | 4.47 | 0.147 [0.134, 0.161] | 0.176 [0.144, 0.207] | 0.53 [0.50, 0.56] |
| mistral7b_bos | top-1 in both directions | False | 106 | 4.89 | 0.148 [0.130, 0.167] | 0.099 [0.075, 0.125] | 0.46 [0.43, 0.49] |
