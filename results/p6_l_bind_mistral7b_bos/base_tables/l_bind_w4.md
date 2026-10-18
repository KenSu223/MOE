**l_bind W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| mistral7b_bos | main_validation | 128 | -0.531 [-0.580, -0.482] | -0.531 [-0.580, -0.482] | 1.629 [1.570, 1.690] | -0.629 [-0.690, -0.570] | 1.622 [1.564, 1.684] | -0.622 [-0.684, -0.564] |
| mistral7b_bos | main_all | 256 | -0.529 [-0.563, -0.494] | -0.529 [-0.563, -0.494] | 1.622 [1.583, 1.662] | -0.622 [-0.662, -0.583] | 1.617 [1.577, 1.657] | -0.617 [-0.657, -0.577] |
