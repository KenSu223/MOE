**e3_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_14b | main_validation | 128 | 0.403 [0.372, 0.433] | 0.403 [0.372, 0.433] | 1.259 [1.202, 1.315] | -0.259 [-0.315, -0.202] | 1.256 [1.200, 1.312] | -0.256 [-0.312, -0.200] |
| qwen3_14b | main_all | 256 | 0.373 [0.349, 0.397] | 0.373 [0.349, 0.397] | 1.283 [1.243, 1.324] | -0.283 [-0.324, -0.243] | 1.278 [1.238, 1.319] | -0.278 [-0.319, -0.238] |
