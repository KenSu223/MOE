**l_ind W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen2_7b | main | layer | L24 | +0.63 | +0.628 [+0.571, +0.687] | 0.146 [0.137, 0.155] | L24 +0.63, L25 +0.57, L23 +0.48, L22 +0.47 |
| qwen2_7b | main | attn_layer | L22 | +1.44 | +1.438 [+1.324, +1.554] | 0.334 [0.313, 0.356] | L22 +1.44, L20 +1.04, L21 +0.63, L17 +0.54 |
| qwen2_7b | main | block | L22 | +1.84 | +1.842 [+1.715, +1.972] | 0.428 [0.411, 0.446] | L22 +1.84, L20 +1.01, L23 +0.87, L21 +0.76 |
| qwen2_7b | main_interior | layer | L23 | +0.48 | +0.484 [+0.419, +0.556] | 0.113 [0.099, 0.127] | L23 +0.48, L22 +0.47, L21 +0.25, L14 +0.07 |
| qwen2_7b | main_interior | attn_layer | L22 | +1.44 | +1.438 [+1.324, +1.554] | 0.334 [0.313, 0.356] | L22 +1.44, L20 +1.04, L21 +0.63, L17 +0.54 |
| qwen2_7b | main_interior | block | L22 | +1.84 | +1.842 [+1.715, +1.972] | 0.428 [0.411, 0.446] | L22 +1.84, L20 +1.01, L23 +0.87, L21 +0.76 |
