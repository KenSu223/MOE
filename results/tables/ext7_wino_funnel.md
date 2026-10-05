| Split | Model / protocol | Twins | W2 sentence-final single-word trigger | W3 single-token trigger | W4–W5 token-symmetric | W6 dedup | Correct both ways | **Margin ≥ 1 both ways (primary)** | of which names / objects | assoc | top-1 both | debiased | mean Δ_A / Δ_B | mean drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train_xl | Qwen3-30B-A3B-Base | 20,199 | 6,737 | 5,501 | 4,532 | 4,514 | 2,783 | **2,099** | 535 / 1,564 | 183 | 313 | 495 of 1,549 | +3.89 / -3.79 | 7.68 |
| train_xl | Mixtral-8x7B, no BOS | 20,199 | 6,737 | 4,037 | 2,430 | 2,421 | 1,489 | **1,109** | 127 / 982 | 113 | 273 | 255 of 845 | +3.72 / -3.55 | 7.27 |
| train_xl | Mixtral-8x7B, BOS | 20,199 | 6,737 | 4,037 | 2,430 | 2,421 | 1,524 | **1,151** | 150 / 1,001 | 132 | 295 | 279 of 845 | +3.78 / -3.65 | 7.42 |
| train_xl | OLMoE-1B-7B (verification only) | 20,199 | 6,737 | 5,115 | 3,823 | 3,809 | 1,952 | **1,325** | 218 / 1,107 | 122 | 290 | 283 of 1,341 | +3.71 / -3.48 | 7.18 |
| dev | Qwen3-30B-A3B-Base | 284 | 90 | 79 | 68 | 68 | 35 | **27** | 3 / 24 | 5 | 2 | 0 of 0 | +3.83 / -3.00 | 6.83 |
| dev | Mixtral-8x7B, no BOS | 284 | 90 | 59 | 42 | 42 | 20 | **11** | 0 / 11 | 1 | 3 | 0 of 0 | +2.95 / -3.43 | 6.39 |
