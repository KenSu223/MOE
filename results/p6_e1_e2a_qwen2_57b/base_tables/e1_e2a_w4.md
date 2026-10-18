**e1_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_57b | main_validation | 128 | 0.812 [0.793, 0.830] | 0.812 [0.793, 0.830] | 0.229 [0.193, 0.268] | 0.771 [0.732, 0.807] | 0.226 [0.189, 0.265] | 0.774 [0.735, 0.811] |
| qwen2_57b | main_all | 256 | 0.825 [0.811, 0.838] | 0.825 [0.811, 0.838] | 0.206 [0.180, 0.232] | 0.794 [0.768, 0.820] | 0.203 [0.177, 0.229] | 0.797 [0.771, 0.823] |
