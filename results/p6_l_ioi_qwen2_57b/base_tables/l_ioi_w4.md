**l_ioi W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_57b | main_validation | 128 | 0.088 [0.057, 0.119] | 0.088 [0.057, 0.119] | 1.282 [1.248, 1.316] | -0.282 [-0.316, -0.248] | 1.285 [1.251, 1.319] | -0.285 [-0.319, -0.251] |
| qwen2_57b | main_all | 256 | 0.076 [0.054, 0.098] | 0.076 [0.054, 0.098] | 1.286 [1.261, 1.310] | -0.286 [-0.310, -0.261] | 1.287 [1.263, 1.312] | -0.287 [-0.312, -0.263] |
