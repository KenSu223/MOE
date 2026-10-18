**e1_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_4b | main_validation | 128 | 0.117 [0.073, 0.158] | 0.117 [0.073, 0.158] | 1.485 [1.431, 1.539] | -0.485 [-0.539, -0.431] | 1.481 [1.428, 1.533] | -0.481 [-0.533, -0.428] |
| qwen3_4b | main_all | 256 | 0.176 [0.145, 0.207] | 0.176 [0.145, 0.207] | 1.413 [1.371, 1.454] | -0.413 [-0.454, -0.371] | 1.409 [1.368, 1.450] | -0.409 [-0.450, -0.368] |
