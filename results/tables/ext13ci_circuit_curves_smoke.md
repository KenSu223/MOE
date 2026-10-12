| Run | Curve | r(1) | r(5) | r(10) | r(20) [95% CI] | r(128) | max r (k) | k50/80/90 of drop | AUC log k (1..20) | all-MoE ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, F1 pre-neutral (smoke) | greedy_mix | 0.307 |  |  |  |  | 0.580 (3) | 3/never/never |  | -0.040 |
| Qwen3-30B-A3B-Base, F1 pre-neutral (smoke) | greedy_head | 0.307 |  |  |  |  | 0.574 (3) | 3/never/never |  | -0.040 |
| Qwen3-30B-A3B-Base, F1 pre-neutral (smoke) | static_mix_oracle | 0.307 | 0.688 | 0.778 |  | 1.085 | 1.085 (128) | 3/12/24 | 0.629 | -0.040 |
| Qwen3-30B-A3B-Base, F1 pre-neutral (smoke) | static_mix_dla | 0.239 | 0.426 | 0.602 |  | 1.176 | 1.176 (128) | 7/24/32 | 0.453 | -0.040 |
| Qwen3-30B-A3B-Base, F1 pre-neutral (smoke) | static_head_oracle | 0.307 | 0.676 | 0.807 |  | 1.040 | 1.040 (128) | 3/10/24 | 0.633 | -0.040 |
| Qwen3-30B-A3B-Base, F1 pre-neutral (smoke) | static_head_dla | 0.239 | 0.415 | 0.597 |  | 1.153 | 1.153 (128) | 7/24/32 | 0.460 | -0.040 |
