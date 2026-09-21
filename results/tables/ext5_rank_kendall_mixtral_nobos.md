**Mixtral-8x7B-v0.1 (no BOS, paper protocol): Kendall tau-b between metric orderings over the 30 recurrent pairs (all layers; block share is defined only in layers whose block rescue CI excludes zero), over the 8 recurrent pairs in such layers, over the 7 recurrent pairs whose own rescue CI excludes zero, and within L19 over its 7 experts with >= 5 validation-active cases**

| Metric | tau vs rescue | tau vs active-only | tau vs Spec | tau vs share | tau vs percentile | tau vs disc. | tau vs rescue (block>0 layers) | tau vs Spec (block>0) | tau vs share (block>0) | tau vs rescue (rescue CI>0) | tau vs Spec (rescue CI>0) | within L19: tau vs rescue | within L19: tau vs Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all-case rescue (val) | 1.00 | 0.95 | 0.01 | 0.43 | 0.03 | 0.64 | 1.00 | 0.43 | 0.43 | 1.00 | 0.43 | 1.00 | 0.71 |
| active-only rescue (val) | 0.95 | 1.00 | 0.01 | 0.21 | 0.01 | 0.61 | 0.79 | 0.21 | 0.21 | 0.71 | 0.14 | 0.33 | 0.62 |
| Spec (val) | 0.01 | 0.01 | 1.00 | 0.86 | 0.05 | -0.08 | 0.43 | 1.00 | 0.86 | 0.43 | 1.00 | 0.71 | 1.00 |
| share of block rescue (val) | 0.43 | 0.21 | 0.86 | 1.00 | 0.29 | 0.57 | 0.43 | 0.86 | 1.00 | n/a | n/a | n/a | n/a |
| mean per-case percentile among active (val) | 0.03 | 0.01 | 0.05 | 0.29 | 1.00 | -0.06 | 0.29 | 0.29 | 0.29 | 0.43 | 0.24 | 0.43 | 0.52 |
| all-case rescue (disc; selection statistic) | 0.64 | 0.61 | -0.08 | 0.57 | -0.06 | 1.00 | 0.86 | 0.57 | 0.57 | 0.81 | 0.62 | 0.81 | 0.71 |
