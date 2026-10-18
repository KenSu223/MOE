**e1_e2a W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| qwen2_57b | main | layer | L25 | +1.99 | +1.852 [+1.693, +2.012] | 0.280 [0.262, 0.298] | L25 +1.99, L22 +1.72, L23 +1.41, L21 +1.24 |
| qwen2_57b | main | attn_layer | L20 | +1.91 | +1.869 [+1.654, +2.092] | 0.283 [0.253, 0.312] | L20 +1.91, L19 +1.06, L18 +0.43, L22 +0.40 |
| qwen2_57b | main | block | L20 | +2.83 | +2.803 [+2.529, +3.079] | 0.424 [0.391, 0.454] | L20 +2.83, L22 +2.30, L25 +1.81, L23 +1.72 |
| qwen2_57b | main_interior | layer | L22 | +1.72 | +1.611 [+1.400, +1.824] | 0.244 [0.214, 0.274] | L22 +1.72, L23 +1.41, L21 +1.24, L20 +1.21 |
| qwen2_57b | main_interior | attn_layer | L20 | +1.91 | +1.869 [+1.654, +2.092] | 0.283 [0.253, 0.312] | L20 +1.91, L19 +1.06, L18 +0.43, L22 +0.40 |
| qwen2_57b | main_interior | block | L20 | +2.83 | +2.803 [+2.529, +3.079] | 0.424 [0.391, 0.454] | L20 +2.83, L22 +2.30, L23 +1.72, L21 +1.40 |
