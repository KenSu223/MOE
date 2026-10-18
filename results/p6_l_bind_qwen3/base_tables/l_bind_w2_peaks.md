**l_bind W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | layer | L37 | +0.38 | +0.400 [+0.357, +0.444] | 0.077 [0.069, 0.084] | L37 +0.38, L39 +0.34, L44 +0.27, L41 +0.20 |
| Qwen3-30B-A3B-Base | main | attn_layer | L42 | +1.56 | +1.530 [+1.434, +1.626] | 0.292 [0.282, 0.302] | L42 +1.56, L43 +0.99, L45 +0.91, L39 +0.86 |
| Qwen3-30B-A3B-Base | main | block | L42 | +1.68 | +1.674 [+1.570, +1.776] | 0.320 [0.311, 0.330] | L42 +1.68, L43 +1.16, L39 +1.13, L41 +1.01 |
| Qwen3-30B-A3B-Base | main_interior | layer | L37 | +0.38 | +0.400 [+0.357, +0.444] | 0.077 [0.069, 0.084] | L37 +0.38, L39 +0.34, L41 +0.20, L43 +0.20 |
| Qwen3-30B-A3B-Base | main_interior | attn_layer | L42 | +1.56 | +1.530 [+1.434, +1.626] | 0.292 [0.282, 0.302] | L42 +1.56, L43 +0.99, L39 +0.86, L41 +0.79 |
| Qwen3-30B-A3B-Base | main_interior | block | L42 | +1.68 | +1.674 [+1.570, +1.776] | 0.320 [0.311, 0.330] | L42 +1.68, L43 +1.16, L39 +1.13, L41 +1.01 |
