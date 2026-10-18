**e1_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main_validation | 90 | 0.330 [0.273, 0.388] | 0.328 [0.269, 0.387] | 1.038 [0.938, 1.137] | -0.038 [-0.137, 0.062] | 1.038 [0.938, 1.136] | -0.038 [-0.136, 0.062] |
| Mixtral-8x7B, BOS | main_all | 196 | 0.325 [0.284, 0.364] | 0.323 [0.281, 0.362] | 0.962 [0.891, 1.035] | 0.038 [-0.035, 0.109] | 0.962 [0.891, 1.034] | 0.038 [-0.034, 0.109] |
