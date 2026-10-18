**e1_e2a_all W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | main | layer | L20 | +0.69 | +0.691 [+0.619, +0.766] | 0.148 [0.136, 0.159] | L20 +0.69, L19 +0.61, L21 +0.59, L22 +0.47 |
| mistral7b_bos | main | attn_layer | L18 | +0.63 | +0.633 [+0.526, +0.754] | 0.135 [0.114, 0.157] | L18 +0.63, L19 +0.60, L20 +0.59, L17 +0.56 |
| mistral7b_bos | main | block | L20 | +1.20 | +1.202 [+1.091, +1.313] | 0.257 [0.241, 0.271] | L20 +1.20, L19 +1.16, L18 +0.95, L17 +0.87 |
| mistral7b_bos | main_interior | layer | L20 | +0.69 | +0.691 [+0.619, +0.766] | 0.148 [0.136, 0.159] | L20 +0.69, L19 +0.61, L21 +0.59, L22 +0.47 |
| mistral7b_bos | main_interior | attn_layer | L18 | +0.63 | +0.633 [+0.526, +0.754] | 0.135 [0.114, 0.157] | L18 +0.63, L19 +0.60, L20 +0.59, L17 +0.56 |
| mistral7b_bos | main_interior | block | L20 | +1.20 | +1.202 [+1.091, +1.313] | 0.257 [0.241, 0.271] | L20 +1.20, L19 +1.16, L18 +0.95, L17 +0.87 |
