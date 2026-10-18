**l_ioi W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen2_7b | main | layer | L22 | +1.45 | +1.476 [+1.389, +1.566] | 0.123 [0.115, 0.132] | L22 +1.45, L20 +1.18, L21 +1.00, L25 +0.63 |
| qwen2_7b | main | attn_layer | L27 | +3.34 | +3.459 [+3.202, +3.726] | 0.289 [0.269, 0.309] | L27 +3.34, L24 +2.94, L20 +2.20, L26 +1.76 |
| qwen2_7b | main | block | L24 | +3.45 | +3.678 [+3.453, +3.899] | 0.307 [0.296, 0.318] | L24 +3.46, L27 +3.32, L20 +3.21, L26 +1.98 |
| qwen2_7b | main_interior | layer | L22 | +1.45 | +1.476 [+1.389, +1.566] | 0.123 [0.115, 0.132] | L22 +1.45, L20 +1.18, L21 +1.00, L23 +0.13 |
| qwen2_7b | main_interior | attn_layer | L20 | +2.20 | +2.110 [+1.951, +2.272] | 0.176 [0.164, 0.189] | L20 +2.20, L23 +0.99, L18 +0.90, L22 +0.18 |
| qwen2_7b | main_interior | block | L20 | +3.21 | +3.204 [+2.963, +3.450] | 0.268 [0.251, 0.285] | L20 +3.21, L22 +1.75, L23 +1.17, L18 +0.94 |
