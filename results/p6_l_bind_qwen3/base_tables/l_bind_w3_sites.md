**l_bind W3: position × layer grid by site (main family, window 1; site mean over its positions)**

| Model | Patched quantity | Site | Pairs | Positions / case | Peak layer | Peak rescue / drop [95% CI] |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | attn_layer | STR | 256 | 1.0 | L30 | +0.048 [+0.044, +0.052] |
| Qwen3-30B-A3B-Base | attn_layer | other | 256 | 2.0 | L35 | +0.047 [+0.045, +0.049] |
| Qwen3-30B-A3B-Base | attn_layer | final | 256 | 1.0 | L42 | +0.295 [+0.288, +0.303] |
| Qwen3-30B-A3B-Base | layer | STR | 256 | 1.0 | L0 | +0.146 [+0.119, +0.172] |
| Qwen3-30B-A3B-Base | layer | other | 256 | 2.0 | L37 | +0.021 [+0.019, +0.022] |
| Qwen3-30B-A3B-Base | layer | final | 256 | 1.0 | L37 | +0.073 [+0.068, +0.078] |
| Qwen3-30B-A3B-Base | resid | STR | 256 | 1.0 | L0 | +0.999 [+0.990, +1.009] |
| Qwen3-30B-A3B-Base | resid | other | 256 | 2.0 | L36 | +0.142 [+0.138, +0.146] |
| Qwen3-30B-A3B-Base | resid | final | 256 | 1.0 | L47 | +0.994 [+0.985, +1.004] |
