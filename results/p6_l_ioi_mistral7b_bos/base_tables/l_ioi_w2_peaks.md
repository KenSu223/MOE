**l_ioi W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| mistral7b_bos | main | layer | L28 | +0.47 | +0.529 [+0.464, +0.598] | 0.047 [0.042, 0.053] | L28 +0.47, L27 +0.36, L17 +0.33, L20 +0.33 |
| mistral7b_bos | main | attn_layer | L19 | +2.76 | +2.899 [+2.727, +3.074] | 0.260 [0.249, 0.271] | L19 +2.76, L31 +2.08, L16 +1.77, L30 +1.38 |
| mistral7b_bos | main | block | L19 | +3.10 | +3.286 [+3.103, +3.474] | 0.295 [0.284, 0.305] | L19 +3.10, L16 +1.83, L21 +1.25, L20 +1.15 |
| mistral7b_bos | main_interior | layer | L27 | +0.37 | +0.392 [+0.355, +0.431] | 0.035 [0.032, 0.038] | L27 +0.36, L17 +0.33, L20 +0.33, L19 +0.27 |
| mistral7b_bos | main_interior | attn_layer | L19 | +2.76 | +2.899 [+2.727, +3.074] | 0.260 [0.249, 0.271] | L19 +2.76, L16 +1.77, L21 +0.95, L20 +0.83 |
| mistral7b_bos | main_interior | block | L19 | +3.10 | +3.286 [+3.103, +3.474] | 0.295 [0.284, 0.305] | L19 +3.10, L16 +1.83, L21 +1.25, L20 +1.15 |
