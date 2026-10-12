| Run | Heads | str | mention (answer side / other) | blank | pos0 | final |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, F1 pre-neutral (smoke) | all heads | 0.02 | 0.02 / 0.02 | 0.04 | 0.50 | 0.12 |
| Qwen3-30B-A3B-Base, F1 pre-neutral (smoke) | positive heads, rescue-weighted | 0.03 | 0.07 / 0.03 | 0.04 | 0.37 | 0.12 |
| Qwen3-30B-A3B-Base, F1 pre-neutral (smoke) | top-10 heads, rescue-weighted | 0.04 | 0.21 / 0.07 | 0.03 | 0.16 | 0.19 |
| Qwen3-30B-A3B-Base, F1 pre-neutral (smoke) | best head of each row | 0.00 | 0.18 / 0.11 | 0.01 | 0.07 | 0.35 |
| Qwen3-30B-A3B-Base, F1 pre-neutral (smoke) | labels of the top-10 heads | other context tokens 6, mover (mention, clean side) 2, self (final token) 1, sink (pos0) 1 |  |  |  |  |
