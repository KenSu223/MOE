**Mixtral active-pair equal-norm check at L19 (paper Table 11 analogue): both patch vectors scaled to the smaller norm**

| Model / protocol | Corruption | Expert | n (anchor-active val) | Raw active-pair Spec | Selected, equal norm | Other active, equal norm | Equal-norm Spec |
|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | L19E006 | 72 | -0.142 [-0.275, -0.019] | +0.144 [+0.090, +0.202] | +0.151 [+0.088, +0.223] | -0.008 [-0.055, +0.039] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | L19E006 | 70 | -0.131 [-0.267, +0.002] | +0.074 [+0.027, +0.121] | +0.141 [+0.059, +0.221] | -0.066 [-0.139, +0.007] |
| Mixtral-8x7B, no BOS (paper protocol) | STR (donor mean) | L19E002 | 56 | +0.436 [+0.247, +0.628] | +0.247 [+0.168, +0.328] | +0.249 [+0.164, +0.343] | -0.002 [-0.058, +0.052] |
| Mixtral-8x7B, no BOS (paper protocol) | GN (same cases) | L19E002 | 56 | +0.280 [+0.104, +0.459] | +0.226 [+0.147, +0.315] | +0.113 [+0.051, +0.177] | +0.114 [+0.035, +0.201] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | L19E002 | 68 | +0.640 [+0.366, +0.977] | +0.286 [+0.210, +0.371] | +0.236 [+0.167, +0.309] | +0.050 [-0.002, +0.106] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | L19E002 | 68 | +0.428 [+0.287, +0.586] | +0.182 [+0.116, +0.253] | +0.108 [+0.057, +0.160] | +0.074 [+0.026, +0.128] |
| Mixtral-8x7B, BOS (tokenizer default) | STR (donor mean) | L19E006 | 59 | -0.322 [-0.714, -0.053] | +0.231 [+0.153, +0.323] | +0.210 [+0.129, +0.306] | +0.021 [-0.036, +0.075] |
| Mixtral-8x7B, BOS (tokenizer default) | GN (same cases) | L19E006 | 59 | -0.119 [-0.276, +0.008] | +0.095 [+0.049, +0.145] | +0.100 [+0.039, +0.168] | -0.004 [-0.054, +0.041] |
