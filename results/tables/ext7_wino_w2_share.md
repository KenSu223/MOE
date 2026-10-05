**W2: attention share of the positive final-position rescue (Direction-2b definition: AUC+(attn) / (AUC+(attn) + AUC+(MoE)), sums over layers of the positive part of the mean curve)**

| Model | Set | Pairs | AUC+ attention | AUC+ MoE | AUC+ block | AUC+ attention / drop | AUC+ MoE / drop | Attention share [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main, validation | 128 | 1.86 | 9.81 | 11.34 | 0.228 | 1.205 | 0.16 [0.13, 0.20] |
| Qwen3-30B-A3B-Base | main, all 256 pairs | 256 | 2.07 | 9.79 | 11.57 | 0.249 | 1.177 | 0.17 [0.15, 0.20] |
| Qwen3-30B-A3B-Base | replication, validation | 128 | 2.60 | 11.20 | 13.15 | 0.303 | 1.302 | 0.19 [0.17, 0.22] |
| Qwen3-30B-A3B-Base | own pool, validation | 128 | 2.84 | 9.28 | 11.51 | 0.358 | 1.169 | 0.23 [0.20, 0.27] |
| Qwen3-30B-A3B-Base | main val., A→B only | 128 | 1.90 | 10.53 | 11.96 | 0.233 | 1.294 | 0.15 [0.13, 0.21] |
| Qwen3-30B-A3B-Base | main val., B→A only | 128 | 1.94 | 9.26 | 10.88 | 0.238 | 1.137 | 0.17 [0.14, 0.23] |
| Qwen3-30B-A3B-Base | CounterFact STR, validation (ext7-controls' qwen3_str_attnsweep, donor mean) | 108 |  |  |  | 0.800 | 0.670 | 0.54 |
| Mixtral-8x7B, BOS | main, validation | 128 | 5.59 | 10.68 | 15.56 | 0.728 | 1.389 | 0.34 [0.33, 0.36] |
| Mixtral-8x7B, BOS | main, all 256 pairs | 256 | 5.68 | 10.55 | 15.63 | 0.730 | 1.356 | 0.35 [0.34, 0.36] |
| Mixtral-8x7B, BOS | replication, validation | 128 | 6.22 | 11.57 | 16.91 | 0.735 | 1.366 | 0.35 [0.34, 0.36] |
| Mixtral-8x7B, BOS | own pool, validation | 128 | 5.78 | 10.01 | 14.83 | 0.780 | 1.352 | 0.37 [0.35, 0.38] |
| Mixtral-8x7B, BOS | main val., A→B only | 128 | 5.78 | 10.85 | 15.78 | 0.751 | 1.411 | 0.35 [0.33, 0.36] |
| Mixtral-8x7B, BOS | main val., B→A only | 128 | 5.43 | 10.55 | 15.35 | 0.707 | 1.373 | 0.34 [0.32, 0.36] |
| Mixtral-8x7B, BOS | CounterFact STR, validation (ext7-controls' mixtral_bos_str_attnsweep, donor mean) | 106 |  |  |  | 0.842 | 0.615 | 0.58 |
