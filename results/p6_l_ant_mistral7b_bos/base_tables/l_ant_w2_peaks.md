**l_ant W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | main | layer | L18 | +1.23 | +1.043 [+0.901, +1.195] | 0.096 [0.082, 0.111] | L18 +1.23, L26 +1.09, L25 +0.81, L19 +0.69 |
| mistral7b_bos | main | attn_layer | L25 | +1.34 | +1.322 [+1.135, +1.518] | 0.122 [0.106, 0.137] | L25 +1.33, L31 +1.19, L30 +0.97, L17 +0.93 |
| mistral7b_bos | main | block | L25 | +2.14 | +2.026 [+1.803, +2.253] | 0.186 [0.169, 0.204] | L25 +2.14, L17 +1.80, L19 +1.29, L21 +1.16 |
| mistral7b_bos | main_interior | layer | L18 | +1.23 | +1.043 [+0.901, +1.195] | 0.096 [0.082, 0.111] | L18 +1.23, L26 +1.09, L25 +0.81, L19 +0.69 |
| mistral7b_bos | main_interior | attn_layer | L25 | +1.34 | +1.322 [+1.135, +1.518] | 0.122 [0.106, 0.137] | L25 +1.33, L17 +0.93, L15 +0.63, L21 +0.59 |
| mistral7b_bos | main_interior | block | L25 | +2.14 | +2.026 [+1.803, +2.253] | 0.186 [0.169, 0.204] | L25 +2.14, L17 +1.80, L19 +1.29, L21 +1.16 |
