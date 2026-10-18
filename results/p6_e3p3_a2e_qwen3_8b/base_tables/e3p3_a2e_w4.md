**e3p3_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_8b | main_validation | 110 | 0.402 [0.371, 0.433] | 0.402 [0.371, 0.433] | 1.201 [1.143, 1.259] | -0.201 [-0.259, -0.143] | 1.196 [1.138, 1.256] | -0.196 [-0.256, -0.138] |
| qwen3_8b | main_all | 226 | 0.384 [0.361, 0.407] | 0.384 [0.361, 0.407] | 1.227 [1.186, 1.266] | -0.227 [-0.266, -0.186] | 1.222 [1.181, 1.262] | -0.222 [-0.262, -0.181] |
