**l_ind W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main_validation | 128 | -0.035 [-0.078, 0.009] | -0.034 [-0.077, 0.010] | 1.391 [1.339, 1.443] | -0.391 [-0.443, -0.339] | 1.391 [1.339, 1.443] | -0.391 [-0.443, -0.339] |
| Mixtral-8x7B, BOS | main_all | 256 | -0.022 [-0.056, 0.010] | -0.022 [-0.056, 0.010] | 1.402 [1.367, 1.439] | -0.402 [-0.439, -0.367] | 1.402 [1.367, 1.439] | -0.402 [-0.439, -0.367] |
