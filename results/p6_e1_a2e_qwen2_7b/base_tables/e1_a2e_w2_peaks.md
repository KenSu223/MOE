**e1_a2e W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen2_7b | main | layer | L19 | +2.27 | +2.427 [+2.262, +2.586] | 0.395 [0.377, 0.413] | L19 +2.27, L20 +2.15, L22 +0.67, L21 +0.67 |
| qwen2_7b | main | attn_layer | L19 | +3.57 | +3.696 [+3.442, +3.938] | 0.601 [0.581, 0.619] | L19 +3.57, L18 +3.33, L23 +1.34, L24 +1.31 |
| qwen2_7b | main | block | L19 | +4.13 | +4.295 [+4.017, +4.554] | 0.698 [0.678, 0.718] | L19 +4.13, L18 +3.61, L20 +2.33, L24 +1.77 |
| qwen2_7b | main_interior | layer | L19 | +2.27 | +2.427 [+2.262, +2.586] | 0.395 [0.377, 0.413] | L19 +2.27, L20 +2.15, L22 +0.67, L21 +0.67 |
| qwen2_7b | main_interior | attn_layer | L19 | +3.57 | +3.696 [+3.442, +3.938] | 0.601 [0.581, 0.619] | L19 +3.57, L18 +3.33, L23 +1.34, L22 +1.00 |
| qwen2_7b | main_interior | block | L19 | +4.13 | +4.295 [+4.017, +4.554] | 0.698 [0.678, 0.718] | L19 +4.13, L18 +3.61, L20 +2.33, L22 +1.59 |
