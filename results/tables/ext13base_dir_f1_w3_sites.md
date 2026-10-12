**dir_f1 W3: position × layer grid by site (main family, window 1; site mean over its positions)**

| Model | Patched quantity | Site | Pairs | Positions / case | Peak layer | Peak rescue / drop [95% CI] |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | attn_layer | STR | 180 | 1.0 | L27 | +0.050 [+0.039, +0.062] |
| Qwen3-30B-A3B-Base | attn_layer | answer cue | 180 | 2.9 | L38 | +0.018 [+0.016, +0.021] |
| Qwen3-30B-A3B-Base | attn_layer | final | 180 | 1.0 | L43 | +0.211 [+0.200, +0.223] |
| Qwen3-30B-A3B-Base | layer | STR | 180 | 1.0 | L2 | +0.209 [+0.178, +0.241] |
| Qwen3-30B-A3B-Base | layer | answer cue | 180 | 2.9 | L37 | +0.005 [+0.003, +0.006] |
| Qwen3-30B-A3B-Base | layer | final | 180 | 1.0 | L44 | +0.088 [+0.077, +0.100] |
| Qwen3-30B-A3B-Base | resid | STR | 180 | 1.0 | L10 | +1.000 [+0.991, +1.010] |
| Qwen3-30B-A3B-Base | resid | answer cue | 180 | 2.9 | L36 | +0.058 [+0.053, +0.062] |
| Qwen3-30B-A3B-Base | resid | final | 180 | 1.0 | L47 | +0.997 [+0.987, +1.006] |
| Mixtral-8x7B, BOS | attn_layer | STR | 188 | 1.0 | L15 | +0.065 [+0.057, +0.073] |
| Mixtral-8x7B, BOS | attn_layer | other | 188 | 1.0 | L13 | +0.008 [+0.004, +0.012] |
| Mixtral-8x7B, BOS | attn_layer | answer cue | 188 | 4.0 | L16 | +0.011 [+0.009, +0.012] |
| Mixtral-8x7B, BOS | attn_layer | final | 188 | 1.0 | L16 | +0.265 [+0.252, +0.278] |
| Mixtral-8x7B, BOS | layer | STR | 188 | 1.0 | L0 | +0.535 [+0.507, +0.563] |
| Mixtral-8x7B, BOS | layer | other | 188 | 1.0 | L11 | +0.003 [-0.000, +0.006] |
| Mixtral-8x7B, BOS | layer | answer cue | 188 | 4.0 | L20 | +0.004 [+0.003, +0.005] |
| Mixtral-8x7B, BOS | layer | final | 188 | 1.0 | L20 | +0.157 [+0.146, +0.168] |
| Mixtral-8x7B, BOS | resid | STR | 188 | 1.0 | L0 | +0.999 [+0.994, +1.005] |
| Mixtral-8x7B, BOS | resid | other | 188 | 1.0 | L13 | +0.023 [+0.016, +0.030] |
| Mixtral-8x7B, BOS | resid | answer cue | 188 | 4.0 | L20 | +0.025 [+0.023, +0.027] |
| Mixtral-8x7B, BOS | resid | final | 188 | 1.0 | L31 | +1.000 [+0.994, +1.005] |
