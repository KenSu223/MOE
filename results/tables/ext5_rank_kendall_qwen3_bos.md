**Qwen3-30B-A3B-Base (tokenizer defaults): Kendall tau-b between metric orderings over the 234 recurrent pairs (all layers; block share is defined only in layers whose block rescue CI excludes zero), over the 90 recurrent pairs in such layers, over the 37 recurrent pairs whose own rescue CI excludes zero, and within L44 over its 35 experts with >= 5 validation-active cases**

| Metric | tau vs rescue | tau vs active-only | tau vs Spec | tau vs share | tau vs percentile | tau vs disc. | tau vs rescue (block>0 layers) | tau vs Spec (block>0) | tau vs share (block>0) | tau vs rescue (rescue CI>0) | tau vs Spec (rescue CI>0) | within L44: tau vs rescue | within L44: tau vs Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all-case rescue (val) | 1.00 | 0.94 | 0.38 | 0.75 | 0.38 | 0.27 | 1.00 | 0.37 | 0.75 | 1.00 | 0.53 | 1.00 | 0.73 |
| active-only rescue (val) | 0.94 | 1.00 | 0.37 | 0.73 | 0.39 | 0.26 | 0.95 | 0.35 | 0.73 | 0.85 | 0.44 | 0.76 | 0.59 |
| Spec (val) | 0.38 | 0.37 | 1.00 | 0.44 | 0.45 | 0.21 | 0.37 | 1.00 | 0.44 | 0.53 | 1.00 | 0.73 | 1.00 |
| share of block rescue (val) | 0.75 | 0.73 | 0.44 | 1.00 | 0.44 | 0.21 | 0.75 | 0.44 | 1.00 | n/a | n/a | n/a | n/a |
| mean per-case percentile among active (val) | 0.38 | 0.39 | 0.45 | 0.44 | 1.00 | 0.16 | 0.33 | 0.63 | 0.44 | 0.11 | 0.38 | 0.49 | 0.47 |
| all-case rescue (disc; selection statistic) | 0.27 | 0.26 | 0.21 | 0.21 | 0.16 | 1.00 | 0.30 | 0.25 | 0.21 | 0.74 | 0.51 | 0.53 | 0.40 |
