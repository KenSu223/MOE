**l_ant W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main_validation | 126 | 0.834 [0.787, 0.884] | 0.831 [0.784, 0.880] | 0.137 [0.080, 0.192] | 0.863 [0.808, 0.920] | 0.137 [0.080, 0.191] | 0.863 [0.809, 0.920] |
| Qwen3-30B-A3B-Base | main_all | 252 | 0.819 [0.786, 0.854] | 0.817 [0.784, 0.852] | 0.152 [0.113, 0.188] | 0.848 [0.812, 0.887] | 0.152 [0.113, 0.188] | 0.848 [0.812, 0.887] |
