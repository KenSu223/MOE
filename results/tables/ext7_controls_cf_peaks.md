| Model | Corruption | Component | L* (disc.) | Val. rescue at L* [95% CI] | / drop | Val. argmax | AUC+ (val.) | AUC+ / drop | Mean val. drop |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | STR (donor mean) | attention | L40 | +3.955 [+3.526, +4.406] | 0.339 | L40 | 9.30 [8.39, 10.36] | 0.797 | +11.67 |
| Qwen3-30B-A3B-Base | STR (donor mean) | MoE | L44 | +2.077 [+1.775, +2.406] | 0.178 | L44 | 7.79 [6.96, 8.99] | 0.667 | +11.67 |
| Qwen3-30B-A3B-Base | STR (donor mean) | block | L40 | +4.735 [+4.252, +5.241] | 0.406 | L40 | 16.49 [14.97, 18.27] | 1.413 | +11.67 |
| Qwen3-30B-A3B-Base | STR (first donor) | attention | L40 | +3.877 [+3.409, +4.372] | 0.335 | L40 | 9.31 [8.24, 10.57] | 0.805 | +11.57 |
| Qwen3-30B-A3B-Base | STR (first donor) | MoE | L44 | +2.061 [+1.750, +2.394] | 0.178 | L44 | 8.17 [7.11, 9.71] | 0.706 | +11.57 |
| Qwen3-30B-A3B-Base | STR (first donor) | block | L40 | +4.663 [+4.133, +5.225] | 0.403 | L40 | 16.70 [14.89, 18.83] | 1.444 | +11.57 |
| Qwen3-30B-A3B-Base | GN (same cases) | attention | L40 | +1.556 [+1.354, +1.764] | 0.284 | L40 | 3.79 [3.30, 4.48] | 0.693 | +5.48 |
| Qwen3-30B-A3B-Base | GN (same cases) | MoE | L44 | +0.895 [+0.719, +1.084] | 0.163 | L44 | 3.73 [3.11, 4.68] | 0.681 | +5.48 |
| Qwen3-30B-A3B-Base | GN (same cases) | block | L40 | +1.889 [+1.663, +2.130] | 0.345 | L40 | 7.22 [6.30, 8.36] | 1.317 | +5.48 |
| Mixtral-8x7B, BOS | STR (donor mean) | attention | L18 | +2.329 [+2.016, +2.658] | 0.180 | L24 | 10.93 [10.05, 11.87] | 0.842 | +12.97 |
| Mixtral-8x7B, BOS | STR (donor mean) | MoE | L21 | +1.100 [+0.918, +1.286] | 0.085 | L19 | 7.99 [6.94, 9.18] | 0.616 | +12.97 |
| Mixtral-8x7B, BOS | STR (donor mean) | block | L19 | +3.468 [+3.076, +3.869] | 0.267 | L19 | 18.33 [16.59, 20.23] | 1.413 | +12.97 |
| Mixtral-8x7B, BOS | STR (first donor) | attention | L18 | +2.337 [+1.993, +2.703] | 0.180 | L24 | 10.78 [9.82, 11.87] | 0.832 | +12.96 |
| Mixtral-8x7B, BOS | STR (first donor) | MoE | L21 | +1.101 [+0.925, +1.279] | 0.085 | L21 | 7.58 [6.43, 8.94] | 0.585 | +12.96 |
| Mixtral-8x7B, BOS | STR (first donor) | block | L19 | +3.392 [+2.985, +3.802] | 0.262 | L19 | 17.85 [16.00, 19.77] | 1.378 | +12.96 |
| Mixtral-8x7B, BOS | GN (same cases) | attention | L18 | +0.953 [+0.779, +1.131] | 0.192 | L18 | 4.79 [4.03, 5.57] | 0.963 | +4.97 |
| Mixtral-8x7B, BOS | GN (same cases) | MoE | L19 | +0.561 [+0.430, +0.696] | 0.113 | L19 | 3.72 [3.05, 4.52] | 0.749 | +4.97 |
| Mixtral-8x7B, BOS | GN (same cases) | block | L19 | +1.356 [+1.140, +1.571] | 0.273 | L19 | 8.08 [6.85, 9.33] | 1.624 | +4.97 |
