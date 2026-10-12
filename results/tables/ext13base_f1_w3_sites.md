**f1 W3: position × layer grid by site (main family, window 1; site mean over its positions)**

| Model | Patched quantity | Site | Pairs | Positions / case | Peak layer | Peak rescue / drop [95% CI] |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | attn_layer | STR | 256 | 1.2 | L34 | +0.036 [+0.029, +0.045] |
| Qwen3-30B-A3B-Base | attn_layer | blank | 35 | 1.0 | L35 | +0.022 [+0.015, +0.029] |
| Qwen3-30B-A3B-Base | attn_layer | mention | 11 | 1.1 | L10 | +0.045 [+0.022, +0.078] |
| Qwen3-30B-A3B-Base | attn_layer | other | 137 | 3.3 | L34 | +0.033 [+0.023, +0.043] |
| Qwen3-30B-A3B-Base | attn_layer | answer cue | 256 | 2.5 | L39 | +0.008 [+0.006, +0.010] |
| Qwen3-30B-A3B-Base | attn_layer | final | 256 | 1.0 | L39 | +0.296 [+0.274, +0.318] |
| Qwen3-30B-A3B-Base | layer | STR | 256 | 1.2 | L0 | +0.151 [+0.121, +0.181] |
| Qwen3-30B-A3B-Base | layer | blank | 35 | 1.0 | L21 | +0.003 [-0.003, +0.007] |
| Qwen3-30B-A3B-Base | layer | mention | 11 | 1.1 | L35 | +0.060 [+0.035, +0.092] |
| Qwen3-30B-A3B-Base | layer | other | 137 | 3.3 | L24 | +0.022 [+0.015, +0.029] |
| Qwen3-30B-A3B-Base | layer | answer cue | 256 | 2.5 | L25 | +0.004 [+0.002, +0.006] |
| Qwen3-30B-A3B-Base | layer | final | 256 | 1.0 | L37 | +0.119 [+0.105, +0.134] |
| Qwen3-30B-A3B-Base | resid | STR | 256 | 1.2 | L0 | +0.919 [+0.893, +0.944] |
| Qwen3-30B-A3B-Base | resid | blank | 35 | 1.0 | L35 | +0.085 [+0.063, +0.107] |
| Qwen3-30B-A3B-Base | resid | mention | 11 | 1.1 | L33 | +0.382 [+0.272, +0.487] |
| Qwen3-30B-A3B-Base | resid | other | 137 | 3.3 | L24 | +0.268 [+0.227, +0.312] |
| Qwen3-30B-A3B-Base | resid | answer cue | 256 | 2.5 | L36 | +0.038 [+0.033, +0.042] |
| Qwen3-30B-A3B-Base | resid | final | 256 | 1.0 | L47 | +0.999 [+0.989, +1.008] |
| Mixtral-8x7B, BOS | attn_layer | STR | 202 | 1.1 | L15 | +0.048 [+0.040, +0.057] |
| Mixtral-8x7B, BOS | attn_layer | blank | 35 | 1.0 | L13 | +0.009 [+0.001, +0.018] |
| Mixtral-8x7B, BOS | attn_layer | mention | 4 | 1.2 | L10 | +0.041 [+0.020, +0.067] |
| Mixtral-8x7B, BOS | attn_layer | other | 202 | 2.2 | L13 | +0.015 [+0.011, +0.020] |
| Mixtral-8x7B, BOS | attn_layer | answer cue | 202 | 3.9 | L16 | +0.011 [+0.010, +0.013] |
| Mixtral-8x7B, BOS | attn_layer | final | 202 | 1.0 | L16 | +0.278 [+0.261, +0.295] |
| Mixtral-8x7B, BOS | layer | STR | 202 | 1.1 | L0 | +0.485 [+0.454, +0.515] |
| Mixtral-8x7B, BOS | layer | blank | 35 | 1.0 | L10 | +0.003 [-0.001, +0.007] |
| Mixtral-8x7B, BOS | layer | mention | 4 | 1.2 | L17 | +0.043 [+0.004, +0.100] |
| Mixtral-8x7B, BOS | layer | other | 202 | 2.2 | L15 | +0.006 [+0.003, +0.009] |
| Mixtral-8x7B, BOS | layer | answer cue | 202 | 3.9 | L17 | +0.003 [+0.002, +0.005] |
| Mixtral-8x7B, BOS | layer | final | 202 | 1.0 | L20 | +0.131 [+0.119, +0.144] |
| Mixtral-8x7B, BOS | resid | STR | 202 | 1.1 | L0 | +0.933 [+0.907, +0.957] |
| Mixtral-8x7B, BOS | resid | blank | 35 | 1.0 | L13 | +0.010 [+0.001, +0.021] |
| Mixtral-8x7B, BOS | resid | mention | 4 | 1.2 | L12 | +0.264 [+0.130, +0.395] |
| Mixtral-8x7B, BOS | resid | other | 202 | 2.2 | L13 | +0.069 [+0.055, +0.083] |
| Mixtral-8x7B, BOS | resid | answer cue | 202 | 3.9 | L20 | +0.021 [+0.019, +0.024] |
| Mixtral-8x7B, BOS | resid | final | 202 | 1.0 | L31 | +0.998 [+0.991, +1.005] |
