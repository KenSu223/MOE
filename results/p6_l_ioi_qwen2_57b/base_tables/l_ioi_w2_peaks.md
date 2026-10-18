**l_ioi W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen2_57b | main | layer | L20 | +0.63 | +0.710 [+0.647, +0.773] | 0.060 [0.055, 0.065] | L20 +0.63, L26 +0.52, L22 +0.48, L21 +0.45 |
| qwen2_57b | main | attn_layer | L27 | +2.98 | +2.845 [+2.595, +3.103] | 0.240 [0.221, 0.259] | L27 +2.98, L24 +2.94, L20 +1.79, L26 +1.76 |
| qwen2_57b | main | block | L24 | +3.63 | +3.710 [+3.475, +3.955] | 0.313 [0.297, 0.330] | L24 +3.63, L26 +2.42, L20 +2.30, L27 +1.41 |
| qwen2_57b | main_interior | layer | L20 | +0.63 | +0.710 [+0.647, +0.773] | 0.060 [0.055, 0.065] | L20 +0.63, L22 +0.48, L21 +0.45, L23 +0.15 |
| qwen2_57b | main_interior | attn_layer | L20 | +1.79 | +1.937 [+1.789, +2.091] | 0.164 [0.151, 0.176] | L20 +1.79, L23 +0.74, L18 +0.42, L22 +0.30 |
| qwen2_57b | main_interior | block | L20 | +2.30 | +2.494 [+2.317, +2.672] | 0.211 [0.197, 0.224] | L20 +2.30, L23 +0.90, L22 +0.80, L18 +0.44 |
