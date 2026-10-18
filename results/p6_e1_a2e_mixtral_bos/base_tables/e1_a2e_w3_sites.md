**e1_a2e W3: position × layer grid by site (main family, window 1; site mean over its positions)**

| Model | Patched quantity | Site | Pairs | Positions / case | Peak layer | Peak rescue / drop [95% CI] |
|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | attn_layer | STR | 196 | 1.0 | L12 | +0.155 [+0.136, +0.175] |
| Mixtral-8x7B, BOS | attn_layer | question | 2 | 1.0 | L0 | +0.481 [+0.365, +0.651] |
| Mixtral-8x7B, BOS | attn_layer | other | 196 | 1.0 | L13 | +0.031 [+0.027, +0.035] |
| Mixtral-8x7B, BOS | attn_layer | answer cue | 196 | 3.8 | L13 | +0.022 [+0.020, +0.025] |
| Mixtral-8x7B, BOS | attn_layer | final | 196 | 1.0 | L16 | +0.302 [+0.283, +0.323] |
| Mixtral-8x7B, BOS | layer | STR | 196 | 1.0 | L0 | +0.624 [+0.592, +0.655] |
| Mixtral-8x7B, BOS | layer | question | 2 | 1.0 | L0 | +0.415 [+0.349, +0.512] |
| Mixtral-8x7B, BOS | layer | other | 196 | 1.0 | L5 | +0.008 [+0.004, +0.012] |
| Mixtral-8x7B, BOS | layer | answer cue | 196 | 3.8 | L16 | +0.009 [+0.007, +0.010] |
| Mixtral-8x7B, BOS | layer | final | 196 | 1.0 | L17 | +0.135 [+0.122, +0.148] |
| Mixtral-8x7B, BOS | resid | STR | 196 | 1.0 | L0 | +0.992 [+0.984, +1.001] |
| Mixtral-8x7B, BOS | resid | question | 2 | 1.0 | L8 | +0.764 [+0.721, +0.794] |
| Mixtral-8x7B, BOS | resid | other | 196 | 1.0 | L13 | +0.041 [+0.036, +0.046] |
| Mixtral-8x7B, BOS | resid | answer cue | 196 | 3.8 | L16 | +0.033 [+0.030, +0.036] |
| Mixtral-8x7B, BOS | resid | final | 196 | 1.0 | L31 | +0.994 [+0.987, +1.001] |
