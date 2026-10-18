**e3_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_8b | main_validation | 128 | 0.824 [0.806, 0.840] | 0.824 [0.806, 0.840] | 0.209 [0.189, 0.231] | 0.791 [0.769, 0.811] | 0.209 [0.188, 0.230] | 0.791 [0.770, 0.812] |
| qwen3_8b | main_all | 256 | 0.820 [0.807, 0.832] | 0.820 [0.807, 0.832] | 0.211 [0.197, 0.227] | 0.789 [0.773, 0.803] | 0.211 [0.196, 0.226] | 0.789 [0.774, 0.804] |
