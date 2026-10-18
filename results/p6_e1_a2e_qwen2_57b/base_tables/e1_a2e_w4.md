**e1_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_57b | main_validation | 128 | 0.517 [0.488, 0.546] | 0.517 [0.488, 0.546] | 1.082 [1.027, 1.141] | -0.082 [-0.141, -0.027] | 1.075 [1.020, 1.134] | -0.075 [-0.134, -0.020] |
| qwen2_57b | main_all | 256 | 0.535 [0.516, 0.553] | 0.535 [0.516, 0.553] | 1.033 [0.988, 1.077] | -0.033 [-0.077, 0.012] | 1.028 [0.983, 1.072] | -0.028 [-0.072, 0.017] |
