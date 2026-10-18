**l_bind W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | main | layer | L17 | +0.42 | +0.391 [+0.363, +0.418] | 0.091 [0.086, 0.095] | L17 +0.42, L21 +0.41, L18 +0.35, L19 +0.30 |
| mistral7b_bos | main | attn_layer | L19 | +1.35 | +1.304 [+1.227, +1.379] | 0.302 [0.292, 0.312] | L19 +1.35, L16 +0.82, L18 +0.65, L20 +0.49 |
| mistral7b_bos | main | block | L19 | +1.58 | +1.527 [+1.441, +1.612] | 0.354 [0.344, 0.364] | L19 +1.58, L18 +0.97, L16 +0.82, L20 +0.70 |
| mistral7b_bos | main_interior | layer | L17 | +0.42 | +0.391 [+0.363, +0.418] | 0.091 [0.086, 0.095] | L17 +0.42, L21 +0.41, L18 +0.35, L19 +0.30 |
| mistral7b_bos | main_interior | attn_layer | L19 | +1.35 | +1.304 [+1.227, +1.379] | 0.302 [0.292, 0.312] | L19 +1.35, L16 +0.82, L18 +0.65, L20 +0.49 |
| mistral7b_bos | main_interior | block | L19 | +1.58 | +1.527 [+1.441, +1.612] | 0.354 [0.344, 0.364] | L19 +1.58, L18 +0.97, L16 +0.82, L20 +0.70 |
