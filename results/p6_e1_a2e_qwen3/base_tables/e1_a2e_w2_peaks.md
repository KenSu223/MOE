**e1_a2e W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | layer | L37 | +0.44 | +0.403 [+0.342, +0.469] | 0.057 [0.049, 0.065] | L37 +0.44, L43 +0.41, L44 +0.39, L42 +0.35 |
| Qwen3-30B-A3B-Base | main | attn_layer | L42 | +2.07 | +2.134 [+2.014, +2.257] | 0.302 [0.290, 0.313] | L42 +2.07, L39 +1.72, L43 +1.57, L45 +1.15 |
| Qwen3-30B-A3B-Base | main | block | L42 | +2.28 | +2.329 [+2.211, +2.451] | 0.329 [0.318, 0.340] | L42 +2.28, L43 +1.92, L39 +1.79, L38 +1.16 |
| Qwen3-30B-A3B-Base | main_interior | layer | L37 | +0.44 | +0.403 [+0.342, +0.469] | 0.057 [0.049, 0.065] | L37 +0.44, L43 +0.41, L42 +0.35, L41 +0.31 |
| Qwen3-30B-A3B-Base | main_interior | attn_layer | L42 | +2.07 | +2.134 [+2.014, +2.257] | 0.302 [0.290, 0.313] | L42 +2.07, L39 +1.72, L43 +1.57, L38 +1.09 |
| Qwen3-30B-A3B-Base | main_interior | block | L42 | +2.28 | +2.329 [+2.211, +2.451] | 0.329 [0.318, 0.340] | L42 +2.28, L43 +1.92, L39 +1.79, L38 +1.16 |
