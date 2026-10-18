**e1_e2a_all W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | layer | L39 | +1.64 | +1.571 [+1.357, +1.782] | 0.162 [0.145, 0.179] | L39 +1.64, L41 +1.53, L38 +1.43, L43 +1.39 |
| Qwen3-30B-A3B-Base | main | attn_layer | L42 | +0.73 | +0.607 [+0.491, +0.731] | 0.063 [0.050, 0.077] | L42 +0.73, L43 +0.41, L45 +0.40, L35 +0.28 |
| Qwen3-30B-A3B-Base | main | block | L39 | +1.92 | +1.753 [+1.518, +1.982] | 0.181 [0.161, 0.200] | L39 +1.92, L43 +1.78, L38 +1.67, L41 +1.66 |
| Qwen3-30B-A3B-Base | main_interior | layer | L39 | +1.64 | +1.571 [+1.357, +1.782] | 0.162 [0.145, 0.179] | L39 +1.64, L41 +1.53, L38 +1.43, L43 +1.39 |
| Qwen3-30B-A3B-Base | main_interior | attn_layer | L42 | +0.73 | +0.607 [+0.491, +0.731] | 0.063 [0.050, 0.077] | L42 +0.73, L43 +0.41, L35 +0.28, L38 +0.28 |
| Qwen3-30B-A3B-Base | main_interior | block | L39 | +1.92 | +1.753 [+1.518, +1.982] | 0.181 [0.161, 0.200] | L39 +1.92, L43 +1.78, L38 +1.67, L41 +1.66 |
