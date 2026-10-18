**l_ind W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main | layer | L19 | +0.40 | +0.377 [+0.340, +0.414] | 0.086 [0.079, 0.094] | L19 +0.40, L17 +0.38, L20 +0.22, L24 +0.20 |
| Mixtral-8x7B, BOS | main | attn_layer | L18 | +1.92 | +1.714 [+1.592, +1.839] | 0.393 [0.370, 0.416] | L18 +1.92, L17 +1.22, L13 +0.41, L31 +0.40 |
| Mixtral-8x7B, BOS | main | block | L18 | +2.07 | +1.894 [+1.771, +2.015] | 0.434 [0.413, 0.455] | L18 +2.07, L17 +1.57, L19 +0.50, L24 +0.48 |
| Mixtral-8x7B, BOS | main_interior | layer | L19 | +0.40 | +0.377 [+0.340, +0.414] | 0.086 [0.079, 0.094] | L19 +0.40, L17 +0.38, L20 +0.22, L24 +0.20 |
| Mixtral-8x7B, BOS | main_interior | attn_layer | L18 | +1.92 | +1.714 [+1.592, +1.839] | 0.393 [0.370, 0.416] | L18 +1.92, L17 +1.22, L13 +0.41, L15 +0.38 |
| Mixtral-8x7B, BOS | main_interior | block | L18 | +2.07 | +1.894 [+1.771, +2.015] | 0.434 [0.413, 0.455] | L18 +2.07, L17 +1.57, L19 +0.50, L24 +0.48 |
