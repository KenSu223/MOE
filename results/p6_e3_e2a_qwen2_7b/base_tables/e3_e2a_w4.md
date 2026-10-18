**e3_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_7b | main_validation | 128 | 0.938 [0.929, 0.946] | 0.938 [0.929, 0.946] | 0.060 [0.047, 0.073] | 0.940 [0.927, 0.953] | 0.060 [0.047, 0.073] | 0.940 [0.927, 0.953] |
| qwen2_7b | main_all | 256 | 0.931 [0.925, 0.938] | 0.931 [0.925, 0.938] | 0.070 [0.060, 0.081] | 0.930 [0.919, 0.940] | 0.070 [0.059, 0.080] | 0.930 [0.920, 0.941] |
