**l_ind W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | layer | L43 | +0.37 | +0.374 [+0.293, +0.460] | 0.087 [0.070, 0.106] | L43 +0.37, L44 +0.32, L42 +0.22, L40 +0.21 |
| Qwen3-30B-A3B-Base | main | attn_layer | L42 | +2.22 | +2.221 [+1.969, +2.483] | 0.517 [0.482, 0.552] | L42 +2.22, L45 +0.83, L40 +0.44, L37 +0.42 |
| Qwen3-30B-A3B-Base | main | block | L42 | +2.30 | +2.304 [+2.056, +2.570] | 0.536 [0.502, 0.572] | L42 +2.30, L45 +0.78, L43 +0.62, L40 +0.62 |
| Qwen3-30B-A3B-Base | main_interior | layer | L43 | +0.37 | +0.374 [+0.293, +0.460] | 0.087 [0.070, 0.106] | L43 +0.37, L42 +0.22, L40 +0.21, L41 +0.14 |
| Qwen3-30B-A3B-Base | main_interior | attn_layer | L42 | +2.22 | +2.221 [+1.969, +2.483] | 0.517 [0.482, 0.552] | L42 +2.22, L40 +0.44, L37 +0.42, L39 +0.41 |
| Qwen3-30B-A3B-Base | main_interior | block | L42 | +2.30 | +2.304 [+2.056, +2.570] | 0.536 [0.502, 0.572] | L42 +2.30, L43 +0.62, L40 +0.62, L39 +0.44 |
