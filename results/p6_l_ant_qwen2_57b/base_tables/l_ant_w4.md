**l_ant W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_57b | main_validation | 126 | 1.049 [1.003, 1.098] | 1.049 [1.003, 1.098] | -0.093 [-0.146, -0.044] | 1.093 [1.044, 1.146] | -0.095 [-0.148, -0.046] | 1.095 [1.046, 1.148] |
| qwen2_57b | main_all | 252 | 1.048 [1.018, 1.080] | 1.048 [1.018, 1.080] | -0.095 [-0.128, -0.063] | 1.095 [1.063, 1.128] | -0.097 [-0.130, -0.065] | 1.097 [1.065, 1.130] |
