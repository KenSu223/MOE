| Model | Corruption | Attention share of the positive rescue (AUC+) [95% CI] | AUC+ attention / MoE | Share at MoE peak | Share at attention peak | Share at paper layer |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | STR (donor mean) | 0.544 [0.511, 0.572] | 9.30 / 7.79 | L44: +0.025 [+0.015, +0.035] | L40: +0.833 [+0.799, +0.864] | L44: +0.025 |
| Qwen3-30B-A3B-Base | STR (first donor) | 0.533 [0.499, 0.561] | 9.31 / 8.17 | L44: +0.016 [+0.002, +0.030] | L40: +0.843 [+0.807, +0.874] | L44: +0.016 |
| Qwen3-30B-A3B-Base | GN (same cases) | 0.504 [0.464, 0.543] | 3.79 / 3.73 | L44: +0.018 [-0.018, +0.054] | L40: +0.792 [+0.749, +0.835] | L44: +0.018 |
| Mixtral-8x7B, BOS | STR (donor mean) | 0.578 [0.552, 0.604] | 10.93 / 7.99 | L21: +0.144 [+0.084, +0.205] | L18: +0.831 [+0.794, +0.867] | L19: +0.653 |
| Mixtral-8x7B, BOS | STR (first donor) | 0.587 [0.557, 0.617] | 10.78 / 7.58 | L21: +0.122 [+0.051, +0.193] | L18: +0.826 [+0.782, +0.871] | L19: +0.667 |
| Mixtral-8x7B, BOS | GN (same cases) | 0.563 [0.527, 0.594] | 4.79 / 3.72 | L19: +0.617 [+0.574, +0.665] | L18: +0.775 [+0.719, +0.831] | L19: +0.617 |
