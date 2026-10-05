**W2 strata (all 256 main pairs, layers fixed by the main discovery argmax; descriptive)**

| Model | Stratum | Value | Pairs | Mean drop | MoE at L*_MoE / drop | Attention at L*_attn / drop | Block at L*_block / drop | Attention share | Peak attention / peak MoE (normalised) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | context-free association passes (assoc) | True | 31 | 10.10 | 0.218 [0.186, 0.250] | 0.013 [-0.002, 0.028] | 0.246 [0.214, 0.276] | 0.21 [0.19, 0.26] | 0.047 / 0.218 |
| Qwen3-30B-A3B-Base | context-free association passes (assoc) | False | 225 | 8.07 | 0.211 [0.197, 0.226] | 0.026 [0.013, 0.039] | 0.231 [0.216, 0.245] | 0.17 [0.15, 0.20] | 0.032 / 0.211 |
| Qwen3-30B-A3B-Base | person-name options | True | 23 | 7.93 | 0.109 [0.078, 0.148] | 0.132 [0.071, 0.194] | 0.149 [0.111, 0.193] | 0.30 [0.26, 0.38] | 0.132 / 0.169 |
| Qwen3-30B-A3B-Base | person-name options | False | 233 | 8.35 | 0.222 [0.208, 0.236] | 0.014 [0.004, 0.024] | 0.241 [0.228, 0.254] | 0.16 [0.14, 0.19] | 0.030 / 0.222 |
| Qwen3-30B-A3B-Base | top-1 in both directions | True | 64 | 8.68 | 0.199 [0.177, 0.221] | 0.009 [-0.006, 0.025] | 0.213 [0.189, 0.238] | 0.17 [0.15, 0.21] | 0.036 / 0.199 |
| Qwen3-30B-A3B-Base | top-1 in both directions | False | 192 | 8.19 | 0.217 [0.200, 0.234] | 0.030 [0.015, 0.045] | 0.240 [0.224, 0.255] | 0.18 [0.15, 0.21] | 0.030 / 0.217 |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | True | 51 | 8.01 | 0.187 [0.161, 0.215] | 0.050 [0.025, 0.080] | 0.222 [0.196, 0.249] | 0.28 [0.24, 0.33] | 0.050 / 0.187 |
| Qwen3-30B-A3B-Base | AfLite survivor (train_debiased) | False | 205 | 8.39 | 0.218 [0.203, 0.233] | 0.018 [0.006, 0.031] | 0.235 [0.220, 0.251] | 0.16 [0.14, 0.19] | 0.031 / 0.218 |
| Qwen3-30B-A3B-Base | one-token option | True | 247 | 8.34 | 0.215 [0.201, 0.229] | 0.023 [0.012, 0.035] | 0.234 [0.220, 0.248] | 0.17 [0.15, 0.20] | 0.031 / 0.215 |
| Qwen3-30B-A3B-Base | one-token option | False | 9 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base | trigger word in context | True | 6 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base | trigger word in context | False | 250 | 8.32 | 0.215 [0.201, 0.229] | 0.022 [0.010, 0.033] | 0.235 [0.221, 0.249] | 0.17 [0.15, 0.20] | 0.031 / 0.215 |
| Mixtral-8x7B, BOS | context-free association passes (assoc) | True | 25 | 9.61 | 0.173 [0.149, 0.196] | 0.091 [0.068, 0.115] | 0.262 [0.241, 0.284] | 0.36 [0.34, 0.39] | 0.129 / 0.181 |
| Mixtral-8x7B, BOS | context-free association passes (assoc) | False | 231 | 7.58 | 0.165 [0.157, 0.172] | 0.154 [0.143, 0.166] | 0.246 [0.236, 0.256] | 0.35 [0.34, 0.36] | 0.154 / 0.165 |
| Mixtral-8x7B, BOS | person-name options | True | 23 | 6.63 | 0.124 [0.094, 0.155] | 0.146 [0.112, 0.180] | 0.255 [0.217, 0.295] | 0.43 [0.39, 0.48] | 0.146 / 0.169 |
| Mixtral-8x7B, BOS | person-name options | False | 233 | 7.90 | 0.169 [0.161, 0.176] | 0.147 [0.135, 0.159] | 0.247 [0.238, 0.257] | 0.34 [0.33, 0.35] | 0.147 / 0.169 |
| Mixtral-8x7B, BOS | top-1 in both directions | True | 85 | 8.88 | 0.152 [0.140, 0.164] | 0.174 [0.153, 0.195] | 0.222 [0.205, 0.238] | 0.34 [0.33, 0.36] | 0.174 / 0.153 |
| Mixtral-8x7B, BOS | top-1 in both directions | False | 171 | 7.23 | 0.174 [0.165, 0.183] | 0.130 [0.119, 0.142] | 0.264 [0.254, 0.274] | 0.35 [0.34, 0.36] | 0.130 / 0.174 |
| Mixtral-8x7B, BOS | AfLite survivor (train_debiased) | True | 51 | 7.15 | 0.163 [0.145, 0.181] | 0.125 [0.097, 0.159] | 0.246 [0.224, 0.267] | 0.37 [0.34, 0.39] | 0.125 / 0.163 |
| Mixtral-8x7B, BOS | AfLite survivor (train_debiased) | False | 205 | 7.94 | 0.166 [0.158, 0.174] | 0.151 [0.139, 0.163] | 0.248 [0.238, 0.259] | 0.35 [0.34, 0.36] | 0.151 / 0.166 |
| Mixtral-8x7B, BOS | one-token option | True | 235 | 7.83 | 0.167 [0.159, 0.175] | 0.146 [0.135, 0.157] | 0.248 [0.238, 0.258] | 0.35 [0.34, 0.36] | 0.146 / 0.167 |
| Mixtral-8x7B, BOS | one-token option | False | 21 | 7.28 | 0.152 [0.123, 0.180] | 0.154 [0.104, 0.212] | 0.247 [0.222, 0.274] | 0.37 [0.33, 0.41] | 0.154 / 0.160 |
| Mixtral-8x7B, BOS | trigger word in context | True | 6 |  |  |  |  |  |  |
| Mixtral-8x7B, BOS | trigger word in context | False | 250 | 7.83 | 0.167 [0.160, 0.175] | 0.147 [0.136, 0.158] | 0.249 [0.239, 0.258] | 0.35 [0.34, 0.36] | 0.147 / 0.167 |
