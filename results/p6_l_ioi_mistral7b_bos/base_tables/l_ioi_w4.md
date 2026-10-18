**l_ioi W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| mistral7b_bos | main_validation | 128 | 0.130 [0.111, 0.150] | 0.130 [0.111, 0.150] | 1.092 [1.071, 1.114] | -0.092 [-0.114, -0.071] | 1.092 [1.070, 1.114] | -0.092 [-0.114, -0.070] |
| mistral7b_bos | main_all | 256 | 0.119 [0.104, 0.135] | 0.119 [0.104, 0.135] | 1.106 [1.089, 1.123] | -0.106 [-0.123, -0.089] | 1.105 [1.088, 1.122] | -0.105 [-0.122, -0.088] |
