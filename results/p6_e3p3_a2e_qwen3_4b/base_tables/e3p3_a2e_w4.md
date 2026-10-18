**e3p3_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_4b | main_validation | 76 | 0.322 [0.272, 0.369] | 0.320 [0.270, 0.368] | 1.257 [1.191, 1.327] | -0.257 [-0.327, -0.191] | 1.254 [1.189, 1.323] | -0.254 [-0.323, -0.189] |
| qwen3_4b | main_all | 159 | 0.317 [0.286, 0.346] | 0.316 [0.285, 0.346] | 1.251 [1.204, 1.301] | -0.251 [-0.301, -0.204] | 1.250 [1.204, 1.299] | -0.250 [-0.299, -0.204] |
