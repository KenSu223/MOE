**sc_agree_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main_validation | 86 | 0.324 [0.266, 0.381] | 0.325 [0.267, 0.382] | 1.120 [1.010, 1.228] | -0.120 [-0.228, -0.010] | 1.118 [1.008, 1.228] | -0.118 [-0.228, -0.008] |
| Mixtral-8x7B, BOS | main_all | 160 | 0.349 [0.308, 0.389] | 0.350 [0.309, 0.390] | 1.056 [0.980, 1.127] | -0.056 [-0.127, 0.020] | 1.054 [0.979, 1.126] | -0.054 [-0.126, 0.021] |
