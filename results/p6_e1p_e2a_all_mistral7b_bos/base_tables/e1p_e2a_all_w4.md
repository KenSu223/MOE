**e1p_e2a_all W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| mistral7b_bos | main_validation | 116 | 0.599 [0.568, 0.629] | 0.599 [0.568, 0.629] | 0.457 [0.426, 0.490] | 0.543 [0.510, 0.574] | 0.457 [0.426, 0.489] | 0.543 [0.511, 0.574] |
| mistral7b_bos | main_all | 116 | 0.599 [0.568, 0.629] | 0.599 [0.568, 0.629] | 0.457 [0.426, 0.490] | 0.543 [0.510, 0.574] | 0.457 [0.426, 0.489] | 0.543 [0.511, 0.574] |
