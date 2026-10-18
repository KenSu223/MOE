**l_cf W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| mistral7b_bos | main_validation | 98 | 0.465 [0.426, 0.503] | 0.465 [0.426, 0.503] | 0.572 [0.541, 0.606] | 0.428 [0.394, 0.459] | 0.570 [0.540, 0.604] | 0.430 [0.396, 0.460] |
| mistral7b_bos | main_all | 196 | 0.471 [0.444, 0.497] | 0.471 [0.444, 0.497] | 0.571 [0.549, 0.594] | 0.429 [0.406, 0.451] | 0.570 [0.548, 0.592] | 0.430 [0.408, 0.452] |
