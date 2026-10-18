**l_bind W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main_validation | 128 | -0.725 [-0.779, -0.671] | -0.726 [-0.781, -0.672] | 1.748 [1.677, 1.817] | -0.748 [-0.817, -0.677] | 1.746 [1.676, 1.815] | -0.746 [-0.815, -0.676] |
| Mixtral-8x7B, BOS | main_all | 256 | -0.688 [-0.726, -0.650] | -0.688 [-0.727, -0.651] | 1.703 [1.656, 1.752] | -0.703 [-0.752, -0.656] | 1.701 [1.654, 1.750] | -0.701 [-0.750, -0.654] |
