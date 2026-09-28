**Joint (layer, expert) search over the layers of the STR expert pass (top 5, recurrence gate = half of discovery)**

| Model / protocol | Corruption | Rank | (layer, expert) | Disc. active | Disc. all-case | Val rescue | Spec |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | STR (donor mean) | 1 | L44E069 | 94 | +1.117 | +1.071 [+0.802, +1.387] | +0.953 [+0.673, +1.276] |
| Qwen3-30B-A3B-Base | STR (donor mean) | 2 | L42E115 | 104 | +0.849 | +0.838 [+0.687, +0.998] | +0.782 [+0.624, +0.954] |
| Qwen3-30B-A3B-Base | STR (donor mean) | 3 | L43E046 | 72 | +0.395 | +0.238 [+0.114, +0.382] | +0.103 [-0.041, +0.258] |
| Qwen3-30B-A3B-Base | STR (donor mean) | 4 | L43E005 | 62 | +0.299 | +0.253 [+0.130, +0.414] | +0.180 [+0.044, +0.347] |
| Qwen3-30B-A3B-Base | STR (donor mean) | 5 | L40E127 | 76 | +0.262 | +0.215 [+0.143, +0.304] | +0.120 [+0.029, +0.219] |
| Qwen3-30B-A3B-Base | STR (first donor) | 1 | L44E069 | 94 | +1.192 | +1.113 [+0.848, +1.417] | +0.997 [+0.722, +1.309] |
| Qwen3-30B-A3B-Base | STR (first donor) | 2 | L42E115 | 104 | +0.851 | +0.803 [+0.652, +0.966] | +0.744 [+0.584, +0.913] |
| Qwen3-30B-A3B-Base | STR (first donor) | 3 | L43E046 | 72 | +0.425 | +0.281 [+0.148, +0.430] | +0.170 [+0.020, +0.325] |
| Qwen3-30B-A3B-Base | STR (first donor) | 4 | L41E001 | 94 | +0.270 | +0.203 [+0.137, +0.269] | +0.169 [+0.102, +0.235] |
| Qwen3-30B-A3B-Base | STR (first donor) | 5 | L40E127 | 76 | +0.268 | +0.215 [+0.142, +0.304] | +0.128 [+0.040, +0.230] |
| Qwen3-30B-A3B-Base | GN (same cases) | 1 | L42E115 | 105 | +0.430 | +0.449 [+0.358, +0.550] | +0.419 [+0.324, +0.521] |
| Qwen3-30B-A3B-Base | GN (same cases) | 2 | L44E069 | 96 | +0.420 | +0.472 [+0.314, +0.655] | +0.413 [+0.253, +0.595] |
| Qwen3-30B-A3B-Base | GN (same cases) | 3 | L43E046 | 73 | +0.161 | +0.070 [+0.027, +0.120] | -0.012 [-0.066, +0.045] |
| Qwen3-30B-A3B-Base | GN (same cases) | 4 | L43E005 | 62 | +0.157 | +0.157 [+0.077, +0.259] | +0.104 [+0.022, +0.208] |
| Qwen3-30B-A3B-Base | GN (same cases) | 5 | L40E127 | 76 | +0.150 | +0.133 [+0.082, +0.198] | +0.090 [+0.034, +0.157] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | 1 | L21E001 | 54 | +0.398 | +0.623 [+0.475, +0.778] | +0.419 [+0.271, +0.569] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | 2 | L20E005 | 55 | +0.238 | +0.247 [+0.170, +0.331] | -0.073 [-0.178, +0.033] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | 3 | L18E001 | 66 | +0.207 | +0.210 [+0.144, +0.279] | +0.151 [+0.083, +0.223] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | 4 | L19E006 | 72 | +0.179 | +0.127 [+0.078, +0.181] | -0.286 [-0.406, -0.168] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | 5 | L22E001 | 82 | +0.165 | +0.215 [+0.130, +0.308] | +0.032 [-0.090, +0.152] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | 1 | L21E001 | 54 | +0.376 | +0.626 [+0.480, +0.786] | +0.426 [+0.277, +0.581] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | 2 | L20E005 | 55 | +0.233 | +0.272 [+0.186, +0.361] | -0.064 [-0.177, +0.051] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | 3 | L18E001 | 66 | +0.225 | +0.212 [+0.131, +0.295] | +0.140 [+0.057, +0.224] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | 4 | L22E001 | 82 | +0.188 | +0.199 [+0.102, +0.302] | +0.023 [-0.103, +0.146] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (first donor) | 5 | L19E006 | 72 | +0.176 | +0.126 [+0.059, +0.203] | -0.296 [-0.420, -0.169] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | 1 | L21E001 | 53 | +0.268 | +0.306 [+0.195, +0.422] | +0.180 [+0.075, +0.289] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | 2 | L18E001 | 66 | +0.163 | +0.134 [+0.066, +0.212] | +0.094 [+0.025, +0.169] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | 3 | L19E006 | 72 | +0.127 | +0.054 [-0.026, +0.137] | -0.169 [-0.274, -0.064] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | 4 | L22E001 | 81 | +0.125 | +0.087 [+0.020, +0.159] | +0.041 [-0.031, +0.115] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | 5 | L20E005 | 56 | +0.096 | +0.125 [+0.072, +0.187] | -0.076 [-0.157, +0.003] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | 1 | L21E001 | 76 | +0.709 | +0.642 [+0.483, +0.802] | +0.396 [+0.232, +0.567] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | 2 | L19E002 | 61 | +0.561 | +0.597 [+0.397, +0.842] | +0.302 [+0.099, +0.547] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | 3 | L20E005 | 62 | +0.348 | +0.316 [+0.224, +0.416] | -0.074 [-0.201, +0.061] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | 4 | L18E001 | 81 | +0.274 | +0.373 [+0.277, +0.474] | +0.289 [+0.191, +0.387] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | 5 | L22E001 | 78 | +0.259 | +0.400 [+0.285, +0.526] | +0.122 [-0.027, +0.271] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | 1 | L21E001 | 76 | +0.700 | +0.617 [+0.463, +0.772] | +0.401 [+0.241, +0.558] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | 2 | L19E002 | 61 | +0.588 | +0.551 [+0.358, +0.781] | +0.263 [+0.059, +0.495] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | 3 | L20E005 | 62 | +0.340 | +0.292 [+0.196, +0.395] | -0.070 [-0.195, +0.065] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | 4 | L18E001 | 81 | +0.306 | +0.370 [+0.252, +0.486] | +0.292 [+0.175, +0.407] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (first donor) | 5 | L22E001 | 78 | +0.289 | +0.373 [+0.228, +0.524] | +0.096 [-0.067, +0.259] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | 1 | L19E002 | 61 | +0.388 | +0.371 [+0.259, +0.494] | +0.212 [+0.100, +0.328] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | 2 | L21E001 | 78 | +0.374 | +0.282 [+0.190, +0.380] | +0.139 [+0.047, +0.236] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | 3 | L18E001 | 81 | +0.204 | +0.225 [+0.148, +0.306] | +0.175 [+0.097, +0.257] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | 4 | L20E005 | 63 | +0.155 | +0.160 [+0.104, +0.223] | -0.047 [-0.121, +0.031] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | 5 | L22E001 | 78 | +0.136 | +0.167 [+0.101, +0.244] | +0.059 [-0.017, +0.142] |
