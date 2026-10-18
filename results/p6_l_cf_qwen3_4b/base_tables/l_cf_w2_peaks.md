**l_cf W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen3_4b | main | layer | L29 | +2.03 | +2.080 [+1.751, +2.421] | 0.200 [0.178, 0.222] | L29 +2.03, L30 +1.28, L28 +1.20, L27 +0.73 |
| qwen3_4b | main | attn_layer | L24 | +2.94 | +2.954 [+2.555, +3.378] | 0.284 [0.255, 0.313] | L24 +2.94, L31 +2.77, L26 +0.79, L27 +0.67 |
| qwen3_4b | main | block | L24 | +3.47 | +3.324 [+2.882, +3.788] | 0.320 [0.290, 0.349] | L24 +3.47, L31 +3.32, L29 +2.66, L30 +1.72 |
| qwen3_4b | main_interior | layer | L29 | +2.03 | +2.080 [+1.751, +2.421] | 0.200 [0.178, 0.222] | L29 +2.03, L30 +1.28, L28 +1.20, L27 +0.73 |
| qwen3_4b | main_interior | attn_layer | L24 | +2.94 | +2.954 [+2.555, +3.378] | 0.284 [0.255, 0.313] | L24 +2.94, L31 +2.77, L26 +0.79, L27 +0.67 |
| qwen3_4b | main_interior | block | L24 | +3.47 | +3.324 [+2.882, +3.788] | 0.320 [0.290, 0.349] | L24 +3.47, L31 +3.32, L29 +2.66, L30 +1.72 |
