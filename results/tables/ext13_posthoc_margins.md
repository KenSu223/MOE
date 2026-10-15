**Margin rule = logit(answer) - logit(twin's answer) >= 1 in both twins (1 logit = probability ratio e ≈ 2.7, 73 % in a two-way choice); top-1 is recorded, not required**

| Model | Set | Pairs | Answer is top-1 in both twins | Clean margin median (logits) | 10th percentile |
|---|---|---|---|---|---|
| Qwen3 | ask entity (f1), validation | 128 | 0.81 | 2.00 | 1.12 |
| Mixtral BOS | ask entity (f1), validation | 101 | 0.61 | 1.75 | 1.12 |
