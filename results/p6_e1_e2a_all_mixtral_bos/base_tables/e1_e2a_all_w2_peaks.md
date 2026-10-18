**e1_e2a_all W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main | layer | L19 | +0.69 | +0.786 [+0.716, +0.857] | 0.197 [0.182, 0.213] | L19 +0.69, L20 +0.65, L21 +0.48, L22 +0.35 |
| Mixtral-8x7B, BOS | main | attn_layer | L19 | +0.69 | +0.774 [+0.707, +0.844] | 0.194 [0.178, 0.210] | L19 +0.69, L18 +0.64, L13 +0.59, L15 +0.57 |
| Mixtral-8x7B, BOS | main | block | L19 | +1.21 | +1.340 [+1.245, +1.437] | 0.336 [0.316, 0.356] | L19 +1.22, L20 +1.07, L18 +0.94, L17 +0.57 |
| Mixtral-8x7B, BOS | main_interior | layer | L19 | +0.69 | +0.786 [+0.716, +0.857] | 0.197 [0.182, 0.213] | L19 +0.69, L20 +0.65, L21 +0.48, L22 +0.35 |
| Mixtral-8x7B, BOS | main_interior | attn_layer | L19 | +0.69 | +0.774 [+0.707, +0.844] | 0.194 [0.178, 0.210] | L19 +0.69, L18 +0.64, L13 +0.59, L15 +0.57 |
| Mixtral-8x7B, BOS | main_interior | block | L19 | +1.21 | +1.340 [+1.245, +1.437] | 0.336 [0.316, 0.356] | L19 +1.22, L20 +1.07, L18 +0.94, L17 +0.57 |
