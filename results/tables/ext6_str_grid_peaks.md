**STR layer x position grid: peak of each token group's layer curve (mean over retained cases)**

| Model | Window | Metric | Token group | n cases | Peak layer | Peak value [95% CI] | Sum over layers |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 1 | LD / drop | first subject token | 213 | L0 | +0.079 [+0.062, +0.097] | +0.068 |
| Qwen3-30B-A3B-Base | 1 | LD / drop | middle subject tokens | 188 | L0 | +0.167 [+0.147, +0.188] | +0.426 |
| Qwen3-30B-A3B-Base | 1 | LD / drop | last subject token | 215 | L0 | +0.187 [+0.165, +0.211] | +1.267 |
| Qwen3-30B-A3B-Base | 1 | LD / drop | first subsequent token | 176 | L27 | +0.003 [+0.002, +0.005] | +0.029 |
| Qwen3-30B-A3B-Base | 1 | LD / drop | further tokens | 139 | L28 | +0.003 [+0.002, +0.005] | +0.043 |
| Qwen3-30B-A3B-Base | 1 | LD / drop | last token | 215 | L44 | +0.176 [+0.161, +0.192] | +0.578 |
| Qwen3-30B-A3B-Base | 1 | Δp | first subject token | 213 | L0 | +0.0060 [+0.0021, +0.0109] | +0.0059 |
| Qwen3-30B-A3B-Base | 1 | Δp | middle subject tokens | 188 | L2 | +0.0023 [+0.0006, +0.0047] | +0.0090 |
| Qwen3-30B-A3B-Base | 1 | Δp | last subject token | 215 | L3 | +0.0062 [+0.0023, +0.0115] | +0.0327 |
| Qwen3-30B-A3B-Base | 1 | Δp | first subsequent token | 176 | L24 | +0.0001 [-0.0000, +0.0002] | -0.0004 |
| Qwen3-30B-A3B-Base | 1 | Δp | further tokens | 139 | L28 | +0.0001 [+0.0000, +0.0002] | +0.0005 |
| Qwen3-30B-A3B-Base | 1 | Δp | last token | 215 | L44 | +0.0059 [+0.0034, +0.0090] | +0.0176 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | first subject token | 213 | L0 | +0.096 [+0.077, +0.116] | +0.329 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | middle subject tokens | 188 | L2 | +0.341 [+0.308, +0.378] | +1.757 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | last subject token | 215 | L2 | +0.592 [+0.549, +0.635] | +5.906 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | first subsequent token | 176 | L26 | +0.011 [+0.008, +0.014] | +0.223 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | further tokens | 139 | L28 | +0.009 [+0.006, +0.013] | +0.181 |
| Qwen3-30B-A3B-Base | 5 | LD / drop | last token | 215 | L42 | +0.475 [+0.455, +0.495] | +3.092 |
| Qwen3-30B-A3B-Base | 5 | Δp | first subject token | 213 | L1 | +0.0071 [+0.0032, +0.0121] | +0.0219 |
| Qwen3-30B-A3B-Base | 5 | Δp | middle subject tokens | 188 | L1 | +0.0192 [+0.0113, +0.0286] | +0.0937 |
| Qwen3-30B-A3B-Base | 5 | Δp | last subject token | 215 | L2 | +0.0715 [+0.0524, +0.0921] | +0.5028 |
| Qwen3-30B-A3B-Base | 5 | Δp | first subsequent token | 176 | L26 | +0.0001 [+0.0000, +0.0002] | +0.0008 |
| Qwen3-30B-A3B-Base | 5 | Δp | further tokens | 139 | L32 | +0.0002 [+0.0000, +0.0004] | +0.0027 |
| Qwen3-30B-A3B-Base | 5 | Δp | last token | 215 | L42 | +0.0484 [+0.0352, +0.0630] | +0.1686 |
| Mixtral-8x7B, BOS | 1 | LD / drop | first subject token | 212 | L0 | +0.197 [+0.171, +0.224] | +0.254 |
| Mixtral-8x7B, BOS | 1 | LD / drop | middle subject tokens | 196 | L0 | +0.181 [+0.165, +0.198] | +0.327 |
| Mixtral-8x7B, BOS | 1 | LD / drop | last subject token | 213 | L0 | +0.304 [+0.279, +0.330] | +1.599 |
| Mixtral-8x7B, BOS | 1 | LD / drop | first subsequent token | 165 | L21 | +0.005 [+0.003, +0.006] | +0.049 |
| Mixtral-8x7B, BOS | 1 | LD / drop | further tokens | 139 | L19 | +0.007 [+0.005, +0.009] | +0.047 |
| Mixtral-8x7B, BOS | 1 | LD / drop | last token | 213 | L21 | +0.086 [+0.077, +0.095] | +0.500 |
| Mixtral-8x7B, BOS | 1 | Δp | first subject token | 212 | L0 | +0.0132 [+0.0061, +0.0225] | +0.0165 |
| Mixtral-8x7B, BOS | 1 | Δp | middle subject tokens | 196 | L0 | +0.0024 [+0.0015, +0.0035] | +0.0082 |
| Mixtral-8x7B, BOS | 1 | Δp | last subject token | 213 | L3 | +0.0156 [+0.0090, +0.0235] | +0.0601 |
| Mixtral-8x7B, BOS | 1 | Δp | first subsequent token | 165 | L6 | +0.0000 [-0.0000, +0.0001] | +0.0001 |
| Mixtral-8x7B, BOS | 1 | Δp | further tokens | 139 | L20 | +0.0001 [+0.0000, +0.0002] | +0.0004 |
| Mixtral-8x7B, BOS | 1 | Δp | last token | 213 | L20 | +0.0032 [+0.0017, +0.0055] | +0.0145 |
| Mixtral-8x7B, BOS | 5 | LD / drop | first subject token | 212 | L2 | +0.211 [+0.184, +0.239] | +1.024 |
| Mixtral-8x7B, BOS | 5 | LD / drop | middle subject tokens | 196 | L2 | +0.239 [+0.218, +0.260] | +1.362 |
| Mixtral-8x7B, BOS | 5 | LD / drop | last subject token | 213 | L4 | +0.663 [+0.622, +0.704] | +6.933 |
| Mixtral-8x7B, BOS | 5 | LD / drop | first subsequent token | 165 | L20 | +0.025 [+0.020, +0.031] | +0.252 |
| Mixtral-8x7B, BOS | 5 | LD / drop | further tokens | 139 | L19 | +0.026 [+0.022, +0.031] | +0.227 |
| Mixtral-8x7B, BOS | 5 | LD / drop | last token | 213 | L20 | +0.383 [+0.364, +0.403] | +2.798 |
| Mixtral-8x7B, BOS | 5 | Δp | first subject token | 212 | L2 | +0.0138 [+0.0065, +0.0230] | +0.0764 |
| Mixtral-8x7B, BOS | 5 | Δp | middle subject tokens | 196 | L2 | +0.0094 [+0.0063, +0.0135] | +0.0565 |
| Mixtral-8x7B, BOS | 5 | Δp | last subject token | 213 | L4 | +0.1188 [+0.0926, +0.1463] | +0.9203 |
| Mixtral-8x7B, BOS | 5 | Δp | first subsequent token | 165 | L20 | +0.0002 [+0.0001, +0.0002] | +0.0014 |
| Mixtral-8x7B, BOS | 5 | Δp | further tokens | 139 | L19 | +0.0004 [+0.0002, +0.0006] | +0.0026 |
| Mixtral-8x7B, BOS | 5 | Δp | last token | 213 | L20 | +0.0572 [+0.0420, +0.0743] | +0.2930 |
