**e1_a2e_all W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| mistral7b_bos | main_validation | 94 | 0.421 [0.382, 0.460] | 0.421 [0.382, 0.460] | 1.021 [0.950, 1.093] | -0.021 [-0.093, 0.050] | 1.014 [0.942, 1.086] | -0.014 [-0.086, 0.058] |
| mistral7b_bos | main_all | 94 | 0.421 [0.382, 0.460] | 0.421 [0.382, 0.460] | 1.021 [0.950, 1.093] | -0.021 [-0.093, 0.050] | 1.014 [0.942, 1.086] | -0.014 [-0.086, 0.058] |
