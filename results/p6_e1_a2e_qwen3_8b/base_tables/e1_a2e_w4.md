**e1_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_8b | main_validation | 128 | 0.141 [0.105, 0.176] | 0.141 [0.105, 0.176] | 1.344 [1.297, 1.392] | -0.344 [-0.392, -0.297] | 1.344 [1.297, 1.392] | -0.344 [-0.392, -0.297] |
| qwen3_8b | main_all | 256 | 0.158 [0.131, 0.184] | 0.158 [0.131, 0.184] | 1.329 [1.294, 1.364] | -0.329 [-0.364, -0.294] | 1.328 [1.293, 1.362] | -0.328 [-0.362, -0.293] |
