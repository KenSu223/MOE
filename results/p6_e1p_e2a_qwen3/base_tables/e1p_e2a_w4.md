**e1p_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main_validation | 128 | 0.838 [0.814, 0.860] | 0.837 [0.812, 0.860] | 0.139 [0.108, 0.170] | 0.861 [0.830, 0.892] | 0.136 [0.106, 0.168] | 0.864 [0.832, 0.894] |
| Qwen3-30B-A3B-Base | main_all | 256 | 0.805 [0.786, 0.824] | 0.804 [0.784, 0.823] | 0.176 [0.153, 0.198] | 0.824 [0.802, 0.847] | 0.174 [0.151, 0.196] | 0.826 [0.804, 0.849] |
