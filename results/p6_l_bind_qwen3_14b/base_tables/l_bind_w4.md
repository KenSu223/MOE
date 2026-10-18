**l_bind W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_14b | main_validation | 128 | 0.107 [0.069, 0.141] | 0.107 [0.069, 0.141] | 1.434 [1.389, 1.481] | -0.434 [-0.481, -0.389] | 1.433 [1.388, 1.482] | -0.433 [-0.482, -0.388] |
| qwen3_14b | main_all | 256 | 0.070 [0.042, 0.096] | 0.070 [0.042, 0.096] | 1.486 [1.447, 1.525] | -0.486 [-0.525, -0.447] | 1.487 [1.448, 1.527] | -0.487 [-0.527, -0.448] |
