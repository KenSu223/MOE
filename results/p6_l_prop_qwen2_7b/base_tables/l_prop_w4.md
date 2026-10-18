**l_prop W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_7b | main_validation | 128 | 0.756 [0.739, 0.773] | 0.756 [0.739, 0.773] | 0.245 [0.227, 0.263] | 0.755 [0.737, 0.773] | 0.245 [0.228, 0.264] | 0.755 [0.736, 0.772] |
| qwen2_7b | main_all | 256 | 0.753 [0.742, 0.764] | 0.753 [0.742, 0.764] | 0.247 [0.236, 0.258] | 0.753 [0.742, 0.764] | 0.248 [0.236, 0.260] | 0.752 [0.740, 0.764] |
