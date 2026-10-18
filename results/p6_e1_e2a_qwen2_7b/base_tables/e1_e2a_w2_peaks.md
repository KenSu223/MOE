**e1_e2a W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen2_7b | main | layer | L25 | +1.70 | +1.476 [+1.308, +1.659] | 0.276 [0.249, 0.304] | L25 +1.71, L22 +1.26, L20 +1.22, L23 +1.07 |
| qwen2_7b | main | attn_layer | L19 | +1.24 | +1.002 [+0.826, +1.192] | 0.187 [0.155, 0.221] | L19 +1.24, L20 +0.99, L22 +0.73, L18 +0.40 |
| qwen2_7b | main | block | L22 | +1.90 | +1.746 [+1.612, +1.878] | 0.326 [0.305, 0.347] | L22 +1.90, L20 +1.84, L19 +1.55, L18 +1.54 |
| qwen2_7b | main_interior | layer | L22 | +1.26 | +1.196 [+1.092, +1.298] | 0.223 [0.206, 0.239] | L22 +1.26, L20 +1.22, L23 +1.07, L18 +0.91 |
| qwen2_7b | main_interior | attn_layer | L19 | +1.24 | +1.002 [+0.826, +1.192] | 0.187 [0.155, 0.221] | L19 +1.24, L20 +0.99, L22 +0.73, L18 +0.40 |
| qwen2_7b | main_interior | block | L22 | +1.90 | +1.746 [+1.612, +1.878] | 0.326 [0.305, 0.347] | L22 +1.90, L20 +1.84, L19 +1.55, L18 +1.54 |
