**Mixtral-8x7B-v0.1 (BOS, tokenizer default): Kendall tau-b between metric orderings over the 38 recurrent pairs (all layers; block share is defined only in layers whose block rescue CI excludes zero), over the 19 recurrent pairs in such layers, over the 18 recurrent pairs whose own rescue CI excludes zero, and within L19 over its 8 experts with >= 5 validation-active cases**

| Metric | tau vs rescue | tau vs active-only | tau vs Spec | tau vs share | tau vs percentile | tau vs disc. | tau vs rescue (block>0 layers) | tau vs Spec (block>0) | tau vs share (block>0) | tau vs rescue (rescue CI>0) | tau vs Spec (rescue CI>0) | within L19: tau vs rescue | within L19: tau vs Spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all-case rescue (val) | 1.00 | 0.97 | 0.03 | 0.51 | 0.12 | 0.60 | 1.00 | 0.37 | 0.51 | 1.00 | 0.35 | 1.00 | 0.64 |
| active-only rescue (val) | 0.97 | 1.00 | 0.01 | 0.43 | 0.10 | 0.60 | 0.89 | 0.29 | 0.43 | 0.87 | 0.25 | 0.50 | 0.29 |
| Spec (val) | 0.03 | 0.01 | 1.00 | 0.72 | 0.62 | -0.03 | 0.37 | 1.00 | 0.72 | 0.35 | 1.00 | 0.64 | 1.00 |
| share of block rescue (val) | 0.51 | 0.43 | 0.72 | 1.00 | 0.51 | 0.53 | 0.51 | 0.72 | 1.00 | n/a | n/a | n/a | n/a |
| mean per-case percentile among active (val) | 0.12 | 0.10 | 0.62 | 0.51 | 1.00 | 0.03 | 0.43 | 0.62 | 0.51 | 0.35 | 0.60 | 0.64 | 0.86 |
| all-case rescue (disc; selection statistic) | 0.60 | 0.60 | -0.03 | 0.53 | 0.03 | 1.00 | 0.84 | 0.41 | 0.53 | 0.88 | 0.30 | 0.71 | 0.79 |
