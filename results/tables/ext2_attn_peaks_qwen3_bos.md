**Qwen3-30B-A3B-Base (tokenizer defaults): peaks of the rescue curves per patched component (paper set, 128 discovery / 128 validation cases)**

| Patched component | L* (disc.) | Disc. mean at L* | Val. rescue at L* [95% CI] | Val. pos. frac. | Val. argmax | Val. max [95% CI] | AUC+ (val.) [CI] | Centre of mass (val.) |
|---|---|---|---|---|---|---|---|---|
| attention output | L40 | +1.718 | +1.594 [+1.410, +1.791] | 96% | L40 | +1.594 [+1.410, +1.791] | 3.95 [3.43, 4.64] | 36.3 |
| MoE output | L44 | +0.989 | +0.925 [+0.774, +1.096] | 86% | L44 | +0.925 [+0.774, +1.096] | 3.76 [3.15, 4.68] | 38.6 |
| attention + MoE (block) | L40 | +2.089 | +1.946 [+1.734, +2.169] | 95% | L40 | +1.946 [+1.734, +2.169] | 7.30 [6.42, 8.47] | 37.6 |
| residual after layer (hidden state) | L47 | +6.241 | +5.639 [+5.020, +6.292] | 96% | L47 | +5.639 [+5.020, +6.292] | 94.13 [81.54, 107.88] | 36.0 |
