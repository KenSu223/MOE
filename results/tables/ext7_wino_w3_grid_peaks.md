**W3: position × layer grid under STR (main set, 256 pairs; positions before the option are exactly zero)**

| Model | Window | Patched quantity at p | Token group | Pairs | Peak layer | Peak rescue / drop [95% CI] | Sum over layers | Peak Δp (descriptive) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | 1 | attention output | first option tok. | 9 | L4 | +0.020 [-0.001, +0.042] | -0.027 | +0.0001 (L1) |
| Qwen3-30B-A3B-Base | 1 | attention output | last option tok. | 256 | L34 | +0.013 [+0.009, +0.016] | +0.054 | +0.0013 (L3) |
| Qwen3-30B-A3B-Base | 1 | attention output | first subseq. | 157 | L26 | +0.005 [+0.001, +0.011] | +0.033 | +0.0004 (L26) |
| Qwen3-30B-A3B-Base | 1 | attention output | further | 18 | L38 | +0.009 [+0.003, +0.017] | -0.069 | +0.0013 (L39) |
| Qwen3-30B-A3B-Base | 1 | attention output | final token | 256 | L42 | +0.032 [+0.027, +0.037] | +0.213 | +0.0020 (L39) |
| Qwen3-30B-A3B-Base | 1 | MoE output | first option tok. | 9 | L0 | +0.052 [-0.071, +0.160] | +0.106 | +0.0024 (L0) |
| Qwen3-30B-A3B-Base | 1 | MoE output | last option tok. | 256 | L0 | +0.171 [+0.141, +0.204] | +0.396 | +0.0372 (L0) |
| Qwen3-30B-A3B-Base | 1 | MoE output | first subseq. | 157 | L39 | +0.024 [+0.019, +0.028] | +0.151 | +0.0016 (L39) |
| Qwen3-30B-A3B-Base | 1 | MoE output | further | 18 | L38 | +0.008 [+0.004, +0.012] | -0.094 | +0.0009 (L14) |
| Qwen3-30B-A3B-Base | 1 | MoE output | final token | 256 | L41 | +0.214 [+0.201, +0.227] | +1.050 | +0.0234 (L41) |
| Qwen3-30B-A3B-Base | 1 | residual (hidden state) | first option tok. | 9 | L0 | +0.484 [+0.226, +0.741] | +5.893 | +0.0325 (L0) |
| Qwen3-30B-A3B-Base | 1 | residual (hidden state) | last option tok. | 256 | L2 | +0.990 [+0.973, +1.003] | +27.333 | +0.2472 (L9) |
| Qwen3-30B-A3B-Base | 1 | residual (hidden state) | first subseq. | 157 | L36 | +0.121 [+0.108, +0.136] | +2.355 | +0.0139 (L35) |
| Qwen3-30B-A3B-Base | 1 | residual (hidden state) | further | 18 | L35 | +0.107 [+0.066, +0.155] | +1.721 | +0.0173 (L35) |
| Qwen3-30B-A3B-Base | 1 | residual (hidden state) | final token | 256 | L47 | +1.000 [+1.000, +1.000] | +17.603 | +0.2443 (L47) |
| Qwen3-30B-A3B-Base | 5 | MoE output | first option tok. | 9 | L2 | +0.398 [+0.150, +0.666] | +2.379 | +0.0354 (L2) |
| Qwen3-30B-A3B-Base | 5 | MoE output | last option tok. | 256 | L2 | +0.973 [+0.956, +0.987] | +6.574 | +0.2398 (L2) |
| Qwen3-30B-A3B-Base | 5 | MoE output | first subseq. | 157 | L38 | +0.066 [+0.058, +0.074] | +0.706 | +0.0049 (L37) |
| Qwen3-30B-A3B-Base | 5 | MoE output | further | 18 | L38 | +0.038 [+0.020, +0.056] | +0.361 | +0.0036 (L39) |
| Qwen3-30B-A3B-Base | 5 | MoE output | final token | 256 | L41 | +0.569 [+0.557, +0.580] | +5.622 | +0.1346 (L41) |
| Mixtral-8x7B, BOS | 1 | attention output | first option tok. | 21 | L12 | +0.015 [+0.007, +0.023] | +0.080 | +0.0087 (L7) |
| Mixtral-8x7B, BOS | 1 | attention output | middle option tok. | 3 | L0 | +0.066 [-0.006, +0.163] | +0.093 | +0.0007 (L0) |
| Mixtral-8x7B, BOS | 1 | attention output | last option tok. | 256 | L12 | +0.052 [+0.044, +0.061] | +0.169 | +0.0067 (L12) |
| Mixtral-8x7B, BOS | 1 | attention output | first subseq. | 157 | L13 | +0.022 [+0.018, +0.026] | +0.117 | +0.0022 (L13) |
| Mixtral-8x7B, BOS | 1 | attention output | further | 18 | L13 | +0.035 [+0.021, +0.051] | +0.100 | +0.0083 (L13) |
| Mixtral-8x7B, BOS | 1 | attention output | final token | 256 | L13 | +0.147 [+0.136, +0.158] | +0.736 | +0.0222 (L13) |
| Mixtral-8x7B, BOS | 1 | MoE output | first option tok. | 21 | L0 | +0.190 [+0.065, +0.345] | +0.250 | +0.0245 (L0) |
| Mixtral-8x7B, BOS | 1 | MoE output | middle option tok. | 3 | L0 | +0.149 [-0.025, +0.500] | +0.328 | +0.0029 (L7) |
| Mixtral-8x7B, BOS | 1 | MoE output | last option tok. | 256 | L0 | +0.900 [+0.874, +0.924] | +1.419 | +0.2705 (L0) |
| Mixtral-8x7B, BOS | 1 | MoE output | first subseq. | 157 | L21 | +0.027 [+0.024, +0.031] | +0.181 | +0.0025 (L21) |
| Mixtral-8x7B, BOS | 1 | MoE output | further | 18 | L20 | +0.017 [+0.009, +0.027] | +0.104 | +0.0023 (L17) |
| Mixtral-8x7B, BOS | 1 | MoE output | final token | 256 | L20 | +0.166 [+0.159, +0.174] | +1.101 | +0.0183 (L20) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | first option tok. | 21 | L0 | +0.292 [+0.164, +0.437] | +1.851 | +0.0339 (L0) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | middle option tok. | 3 | L1 | +0.199 [-0.025, +0.674] | +2.474 | +0.0043 (L12) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | last option tok. | 256 | L2 | +0.985 [+0.970, +0.996] | +13.068 | +0.2863 (L2) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | first subseq. | 157 | L15 | +0.120 [+0.107, +0.134] | +1.940 | +0.0198 (L13) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | further | 18 | L16 | +0.106 [+0.079, +0.133] | +1.304 | +0.0206 (L13) |
| Mixtral-8x7B, BOS | 1 | residual (hidden state) | final token | 256 | L31 | +1.000 [+1.000, +1.000] | +16.609 | +0.2892 (L31) |
| Mixtral-8x7B, BOS | 5 | MoE output | first option tok. | 21 | L2 | +0.212 [+0.085, +0.365] | +1.410 | +0.0258 (L2) |
| Mixtral-8x7B, BOS | 5 | MoE output | middle option tok. | 3 | L2 | +0.212 [-0.050, +0.717] | +2.311 | +0.0081 (L10) |
| Mixtral-8x7B, BOS | 5 | MoE output | last option tok. | 256 | L2 | +0.984 [+0.968, +0.995] | +9.965 | +0.2870 (L2) |
| Mixtral-8x7B, BOS | 5 | MoE output | first subseq. | 157 | L18 | +0.091 [+0.082, +0.101] | +0.967 | +0.0121 (L18) |
| Mixtral-8x7B, BOS | 5 | MoE output | further | 18 | L19 | +0.064 [+0.046, +0.082] | +0.643 | +0.0079 (L19) |
| Mixtral-8x7B, BOS | 5 | MoE output | final token | 256 | L20 | +0.558 [+0.547, +0.568] | +5.406 | +0.1628 (L20) |
