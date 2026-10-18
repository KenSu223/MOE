**l_prop W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | main | layer | L23 | +1.38 | +1.131 [+1.000, +1.272] | 0.109 [0.097, 0.120] | L23 +1.38, L20 +1.34, L22 +1.22, L21 +1.22 |
| mistral7b_bos | main | attn_layer | L19 | +1.59 | +1.641 [+1.524, +1.766] | 0.158 [0.146, 0.169] | L19 +1.59, L18 +0.81, L15 +0.72, L21 +0.53 |
| mistral7b_bos | main | block | L19 | +2.27 | +2.291 [+2.143, +2.448] | 0.220 [0.206, 0.233] | L19 +2.27, L20 +1.82, L21 +1.72, L24 +1.50 |
| mistral7b_bos | main_interior | layer | L23 | +1.38 | +1.131 [+1.000, +1.272] | 0.109 [0.097, 0.120] | L23 +1.38, L20 +1.34, L22 +1.22, L21 +1.22 |
| mistral7b_bos | main_interior | attn_layer | L19 | +1.59 | +1.641 [+1.524, +1.766] | 0.158 [0.146, 0.169] | L19 +1.59, L18 +0.81, L15 +0.72, L21 +0.53 |
| mistral7b_bos | main_interior | block | L19 | +2.27 | +2.291 [+2.143, +2.448] | 0.220 [0.206, 0.233] | L19 +2.27, L20 +1.82, L21 +1.72, L24 +1.50 |
