**l_ind W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| mistral7b_bos | main_validation | 128 | 0.249 [0.213, 0.286] | 0.249 [0.213, 0.286] | 1.321 [1.271, 1.371] | -0.321 [-0.371, -0.271] | 1.319 [1.269, 1.369] | -0.319 [-0.369, -0.269] |
| mistral7b_bos | main_all | 256 | 0.238 [0.212, 0.262] | 0.238 [0.212, 0.262] | 1.322 [1.288, 1.357] | -0.322 [-0.357, -0.288] | 1.319 [1.285, 1.355] | -0.319 [-0.355, -0.285] |
