**Capacity series 'mistral' (Mistral-7B → Mixtral-8x7B): attention share on each model's own validation set and on the pairs common to every member present (validation)**

| Cell | Mistral-7B (own set) | Mixtral-8x7B (own set) | Common pairs | paired shares (common pairs) Mistral-7B → Mixtral-8x7B | last − first (paired) | interior variant, paired |
|---|---|---|---|---|---|---|
| E1 a→e (all strata) † | 0.61 [0.59, 0.63] | 0.58 [0.56, 0.60] | 34 | 0.61 → 0.59 | -0.03 [-0.06, +0.01] | 0.60 → 0.56 |
| E3 a→e (P3 frame) † | 0.56 [0.54, 0.57] | 0.55 [0.53, 0.57] | 19 | 0.57 → 0.57 | -0.01 [-0.05, +0.03] | 0.55 → 0.54 |
| E3 a→e (F1 frame, Phase-5 anchor) † | 0.48 [0.46, 0.50] | 0.52 [0.50, 0.54] | 36 | 0.50 → 0.52 | +0.02 [-0.00, +0.05] | 0.47 → 0.49 |
| E1 e→a (all strata) † | 0.49 [0.46, 0.52] | 0.51 [0.49, 0.53] | 26 | 0.54 → 0.54 | -0.00 [-0.03, +0.03] | 0.52 → 0.51 |
| E3 e→a | 0.31 [0.30, 0.33] | 0.34 [0.33, 0.35] | 59 | 0.31 → 0.34 | +0.03 [+0.01, +0.04] | 0.30 → 0.32 |
| L-ant antonym | 0.45 [0.42, 0.48] | 0.48 [0.45, 0.51] | 123 | 0.45 → 0.48 | +0.03 [+0.01, +0.05] | 0.37 → 0.42 |
| L-prop object property | 0.37 [0.35, 0.39] | 0.42 [0.40, 0.44] | 122 | 0.37 → 0.42 | +0.06 [+0.04, +0.07] | 0.33 → 0.36 |
| L-bind binding | 0.68 [0.66, 0.69] | 0.71 [0.70, 0.73] | 14 | 0.65 → 0.69 | +0.04 [+0.02, +0.08] | 0.63 → 0.67 |
| L-ioi IOI | 0.78 [0.78, 0.79] | 0.80 [0.79, 0.81] | 128 | 0.78 → 0.80 | +0.02 [+0.01, +0.03] | 0.72 → 0.74 |
| L-ind induction | 0.68 [0.67, 0.70] | 0.71 [0.69, 0.73] | 50 | 0.69 → 0.72 | +0.03 [-0.00, +0.07] | 0.66 → 0.69 |

Own sets differ in items between models; the paired columns use the common pairs only (few for cells whose models select different twins). † = pooled cell in at least one model.
