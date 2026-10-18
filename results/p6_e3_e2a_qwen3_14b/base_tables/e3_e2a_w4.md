**e3_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_14b | main_validation | 128 | 0.847 [0.830, 0.863] | 0.847 [0.830, 0.863] | 0.134 [0.111, 0.156] | 0.866 [0.844, 0.889] | 0.132 [0.109, 0.155] | 0.868 [0.845, 0.891] |
| qwen3_14b | main_all | 254 | 0.845 [0.832, 0.857] | 0.845 [0.832, 0.857] | 0.131 [0.115, 0.147] | 0.869 [0.853, 0.885] | 0.129 [0.113, 0.146] | 0.871 [0.854, 0.887] |
