**e1p_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_7b | main_validation | 128 | 0.815 [0.797, 0.832] | 0.815 [0.797, 0.832] | 0.232 [0.209, 0.257] | 0.768 [0.743, 0.791] | 0.232 [0.209, 0.256] | 0.768 [0.744, 0.791] |
| qwen2_7b | main_all | 256 | 0.810 [0.795, 0.825] | 0.810 [0.795, 0.825] | 0.233 [0.213, 0.252] | 0.767 [0.748, 0.787] | 0.232 [0.212, 0.251] | 0.768 [0.749, 0.788] |
