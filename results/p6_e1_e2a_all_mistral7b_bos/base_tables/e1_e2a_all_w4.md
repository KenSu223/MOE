**e1_e2a_all W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| mistral7b_bos | main_validation | 104 | 0.388 [0.345, 0.431] | 0.388 [0.345, 0.431] | 0.747 [0.678, 0.815] | 0.253 [0.185, 0.322] | 0.745 [0.677, 0.813] | 0.255 [0.187, 0.323] |
| mistral7b_bos | main_all | 104 | 0.388 [0.345, 0.431] | 0.388 [0.345, 0.431] | 0.747 [0.678, 0.815] | 0.253 [0.185, 0.322] | 0.745 [0.677, 0.813] | 0.255 [0.187, 0.323] |
