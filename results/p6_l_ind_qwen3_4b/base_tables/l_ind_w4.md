**l_ind W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_4b | main_validation | 128 | 0.179 [0.135, 0.221] | 0.179 [0.135, 0.221] | 1.018 [0.963, 1.073] | -0.018 [-0.073, 0.037] | 1.016 [0.962, 1.072] | -0.016 [-0.072, 0.038] |
| qwen3_4b | main_all | 252 | 0.168 [0.139, 0.197] | 0.168 [0.139, 0.197] | 1.049 [1.012, 1.087] | -0.049 [-0.087, -0.012] | 1.047 [1.010, 1.086] | -0.047 [-0.086, -0.010] |
