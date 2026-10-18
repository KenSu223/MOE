**e1p_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_7b | main_validation | 128 | 0.650 [0.627, 0.671] | 0.650 [0.627, 0.671] | 0.927 [0.889, 0.965] | 0.073 [0.035, 0.111] | 0.928 [0.889, 0.964] | 0.072 [0.036, 0.111] |
| qwen2_7b | main_all | 256 | 0.655 [0.640, 0.669] | 0.655 [0.640, 0.669] | 0.903 [0.875, 0.931] | 0.097 [0.069, 0.125] | 0.904 [0.875, 0.931] | 0.096 [0.069, 0.125] |
