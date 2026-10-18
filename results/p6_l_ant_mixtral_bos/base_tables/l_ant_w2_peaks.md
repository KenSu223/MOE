**l_ant W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main | layer | L26 | +0.87 | +0.820 [+0.665, +0.991] | 0.081 [0.066, 0.097] | L26 +0.87, L18 +0.81, L25 +0.67, L19 +0.66 |
| Mixtral-8x7B, BOS | main | attn_layer | L21 | +1.29 | +1.498 [+1.334, +1.665] | 0.147 [0.132, 0.163] | L21 +1.29, L17 +1.12, L25 +0.82, L30 +0.78 |
| Mixtral-8x7B, BOS | main | block | L21 | +1.86 | +2.254 [+2.018, +2.497] | 0.222 [0.201, 0.244] | L21 +1.85, L17 +1.52, L25 +1.48, L19 +1.12 |
| Mixtral-8x7B, BOS | main_interior | layer | L26 | +0.87 | +0.820 [+0.665, +0.991] | 0.081 [0.066, 0.097] | L26 +0.87, L18 +0.81, L25 +0.67, L19 +0.66 |
| Mixtral-8x7B, BOS | main_interior | attn_layer | L21 | +1.29 | +1.498 [+1.334, +1.665] | 0.147 [0.132, 0.163] | L21 +1.29, L17 +1.12, L25 +0.82, L19 +0.41 |
| Mixtral-8x7B, BOS | main_interior | block | L21 | +1.86 | +2.254 [+2.018, +2.497] | 0.222 [0.201, 0.244] | L21 +1.85, L17 +1.52, L25 +1.48, L19 +1.12 |
