**l_ant W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| mistral7b_bos | main_validation | 126 | 0.800 [0.762, 0.839] | 0.800 [0.762, 0.839] | 0.260 [0.221, 0.299] | 0.740 [0.701, 0.779] | 0.261 [0.223, 0.300] | 0.739 [0.700, 0.777] |
| mistral7b_bos | main_all | 252 | 0.792 [0.763, 0.822] | 0.792 [0.763, 0.822] | 0.276 [0.246, 0.306] | 0.724 [0.694, 0.754] | 0.278 [0.248, 0.307] | 0.722 [0.693, 0.752] |
