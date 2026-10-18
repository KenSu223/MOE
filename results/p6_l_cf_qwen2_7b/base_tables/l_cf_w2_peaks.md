**l_cf W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen2_7b | main | layer | L24 | +3.23 | +2.986 [+2.639, +3.360] | 0.258 [0.239, 0.278] | L24 +3.23, L23 +2.83, L25 +2.22, L26 +1.15 |
| qwen2_7b | main | attn_layer | L23 | +3.62 | +3.036 [+2.642, +3.436] | 0.263 [0.234, 0.293] | L23 +3.62, L22 +2.59, L19 +0.97, L21 +0.65 |
| qwen2_7b | main | block | L23 | +6.28 | +5.466 [+4.897, +6.053] | 0.473 [0.448, 0.498] | L23 +6.29, L22 +3.82, L24 +3.54, L25 +2.19 |
| qwen2_7b | main_interior | layer | L23 | +2.83 | +2.580 [+2.271, +2.916] | 0.223 [0.204, 0.243] | L23 +2.83, L22 +1.00, L21 +0.35, L19 +0.24 |
| qwen2_7b | main_interior | attn_layer | L23 | +3.62 | +3.036 [+2.642, +3.436] | 0.263 [0.234, 0.293] | L23 +3.62, L22 +2.59, L19 +0.97, L21 +0.65 |
| qwen2_7b | main_interior | block | L23 | +6.28 | +5.466 [+4.897, +6.053] | 0.473 [0.448, 0.498] | L23 +6.29, L22 +3.82, L19 +1.23, L21 +1.05 |
