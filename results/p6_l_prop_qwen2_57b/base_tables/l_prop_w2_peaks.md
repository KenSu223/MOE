**l_prop W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen2_57b | main | layer | L25 | +3.63 | +3.746 [+3.450, +4.037] | 0.380 [0.355, 0.403] | L25 +3.63, L24 +2.16, L23 +1.64, L22 +1.08 |
| qwen2_57b | main | attn_layer | L21 | +2.96 | +2.654 [+2.409, +2.923] | 0.269 [0.250, 0.288] | L21 +2.96, L23 +1.31, L22 +0.43, L19 +0.40 |
| qwen2_57b | main | block | L21 | +3.68 | +3.464 [+3.173, +3.763] | 0.351 [0.333, 0.369] | L21 +3.67, L25 +3.60, L23 +3.05, L24 +2.37 |
| qwen2_57b | main_interior | layer | L23 | +1.64 | +1.883 [+1.664, +2.104] | 0.191 [0.171, 0.210] | L23 +1.64, L22 +1.08, L21 +0.87, L20 +0.35 |
| qwen2_57b | main_interior | attn_layer | L21 | +2.96 | +2.654 [+2.409, +2.923] | 0.269 [0.250, 0.288] | L21 +2.96, L23 +1.31, L22 +0.43, L19 +0.40 |
| qwen2_57b | main_interior | block | L21 | +3.68 | +3.464 [+3.173, +3.763] | 0.351 [0.333, 0.369] | L21 +3.67, L23 +3.05, L22 +1.53, L20 +0.66 |
