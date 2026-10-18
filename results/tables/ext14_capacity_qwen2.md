**Capacity series 'qwen2' (Qwen2-7B → Qwen2-57B-A14B): attention share on each model's own validation set and on the pairs common to every member present (validation)**

| Cell | Qwen2-7B (own set) | Qwen2-57B-A14B (own set) | Common pairs | paired shares (common pairs) Qwen2-7B → Qwen2-57B-A14B | last − first (paired) | interior variant, paired |
|---|---|---|---|---|---|---|
| E1 a→e | 0.62 [0.61, 0.63] | 0.60 [0.58, 0.61] | 18 | 0.64 → 0.66 | +0.02 [+0.00, +0.05] | 0.64 → 0.68 |
| E3 a→e (P3 frame) | 0.56 [0.55, 0.58] | 0.57 [0.55, 0.58] | 35 | 0.56 → 0.58 | +0.02 [-0.01, +0.04] | 0.56 → 0.57 |
| E1 e→a | 0.37 [0.35, 0.40] | 0.38 [0.36, 0.41] | 32 | 0.41 → 0.38 | -0.03 [-0.07, +0.02] | 0.45 → 0.47 |
| E3 e→a | 0.21 [0.19, 0.23] | 0.21 [0.19, 0.24] | 65 | 0.22 → 0.22 | -0.01 [-0.04, +0.02] | 0.25 → 0.27 |
| L-ant antonym | 0.47 [0.45, 0.49] | 0.35 [0.33, 0.37] | 96 | 0.48 → 0.37 | -0.11 [-0.13, -0.09] | 0.49 → 0.46 |
| L-prop object property | 0.35 [0.33, 0.37] | 0.35 [0.34, 0.37] | 110 | 0.35 → 0.35 | +0.00 [-0.01, +0.02] | 0.40 → 0.44 |
| L-cf CounterFact | 0.41 [0.38, 0.44] | 0.42 [0.40, 0.45] | 85 | 0.41 → 0.42 | +0.01 [-0.00, +0.03] | 0.50 → 0.58 |
| L-ioi IOI | 0.71 [0.70, 0.72] | 0.77 [0.76, 0.78] | 127 | 0.71 → 0.77 | +0.06 [+0.05, +0.08] | 0.62 → 0.73 |
| L-ind induction † | 0.68 [0.66, 0.70] | 0.70 [0.68, 0.72] | 14 | 0.64 → 0.64 | +0.01 [-0.06, +0.09] | 0.68 → 0.69 |

Own sets differ in items between models; the paired columns use the common pairs only (few for cells whose models select different twins). † = pooled cell in at least one model.
