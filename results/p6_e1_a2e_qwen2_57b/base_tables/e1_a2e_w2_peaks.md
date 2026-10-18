**e1_a2e W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen2_57b | main | layer | L20 | +1.79 | +1.736 [+1.614, +1.866] | 0.281 [0.268, 0.295] | L20 +1.79, L19 +0.97, L22 +0.85, L24 +0.74 |
| qwen2_57b | main | attn_layer | L20 | +2.82 | +2.798 [+2.614, +2.979] | 0.454 [0.430, 0.477] | L20 +2.82, L23 +1.57, L18 +1.19, L24 +1.11 |
| qwen2_57b | main | block | L20 | +3.93 | +3.820 [+3.594, +4.047] | 0.619 [0.598, 0.640] | L20 +3.93, L23 +1.93, L22 +1.86, L24 +1.77 |
| qwen2_57b | main_interior | layer | L20 | +1.79 | +1.736 [+1.614, +1.866] | 0.281 [0.268, 0.295] | L20 +1.79, L19 +0.97, L22 +0.85, L21 +0.69 |
| qwen2_57b | main_interior | attn_layer | L20 | +2.82 | +2.798 [+2.614, +2.979] | 0.454 [0.430, 0.477] | L20 +2.82, L23 +1.57, L18 +1.19, L22 +1.07 |
| qwen2_57b | main_interior | block | L20 | +3.93 | +3.820 [+3.594, +4.047] | 0.619 [0.598, 0.640] | L20 +3.93, L23 +1.93, L22 +1.86, L18 +1.59 |
