**l_ant W2: final-position single-layer patches (pair bootstrap)**

| Model | Family | Patched output | L* (disc.) | Disc. mean | Validation rescue at L* [95% CI] | Normalised | Discovery top 4 |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main | layer | L41 | +1.59 | +1.794 [+1.628, +1.970] | 0.214 [0.196, 0.232] | L41 +1.59, L43 +1.43, L44 +1.06, L39 +0.81 |
| Qwen3-30B-A3B-Base | main | attn_layer | L43 | +2.51 | +2.174 [+1.985, +2.375] | 0.259 [0.241, 0.277] | L43 +2.52, L38 +1.44, L46 +1.18, L39 +0.66 |
| Qwen3-30B-A3B-Base | main | block | L43 | +3.48 | +3.309 [+3.053, +3.576] | 0.395 [0.372, 0.418] | L43 +3.48, L41 +1.81, L38 +1.57, L39 +1.46 |
| Qwen3-30B-A3B-Base | main_interior | layer | L41 | +1.59 | +1.794 [+1.628, +1.970] | 0.214 [0.196, 0.232] | L41 +1.59, L43 +1.43, L39 +0.81, L42 +0.57 |
| Qwen3-30B-A3B-Base | main_interior | attn_layer | L43 | +2.51 | +2.174 [+1.985, +2.375] | 0.259 [0.241, 0.277] | L43 +2.52, L38 +1.44, L39 +0.66, L26 +0.39 |
| Qwen3-30B-A3B-Base | main_interior | block | L43 | +3.48 | +3.309 [+3.053, +3.576] | 0.395 [0.372, 0.418] | L43 +3.48, L41 +1.81, L38 +1.57, L39 +1.46 |
