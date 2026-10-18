**l_cf W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen2_57b | main | layer | L25 | +3.23 | +3.145 [+2.781, +3.530] | 0.249 [0.229, 0.269] | L25 +3.23, L24 +2.54, L26 +2.24, L23 +1.91 |
| qwen2_57b | main | attn_layer | L23 | +3.95 | +3.809 [+3.323, +4.319] | 0.301 [0.269, 0.335] | L23 +3.95, L22 +1.92, L19 +1.00, L21 +0.59 |
| qwen2_57b | main | block | L23 | +5.93 | +5.667 [+5.098, +6.274] | 0.448 [0.419, 0.478] | L23 +5.92, L25 +3.21, L22 +2.99, L24 +2.75 |
| qwen2_57b | main_interior | layer | L23 | +1.91 | +1.816 [+1.571, +2.073] | 0.144 [0.127, 0.161] | L23 +1.91, L22 +0.72, L19 +0.31, L20 +0.25 |
| qwen2_57b | main_interior | attn_layer | L23 | +3.95 | +3.809 [+3.323, +4.319] | 0.301 [0.269, 0.335] | L23 +3.95, L22 +1.92, L19 +1.00, L21 +0.59 |
| qwen2_57b | main_interior | block | L23 | +5.93 | +5.667 [+5.098, +6.274] | 0.448 [0.419, 0.478] | L23 +5.92, L22 +2.99, L19 +1.33, L21 +0.80 |
