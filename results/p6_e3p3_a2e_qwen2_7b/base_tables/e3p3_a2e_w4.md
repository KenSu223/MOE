**e3p3_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_7b | main_validation | 96 | 0.660 [0.629, 0.688] | 0.660 [0.629, 0.688] | 0.924 [0.865, 0.982] | 0.076 [0.018, 0.135] | 0.922 [0.864, 0.980] | 0.078 [0.020, 0.136] |
| qwen2_7b | main_all | 202 | 0.638 [0.614, 0.661] | 0.638 [0.614, 0.661] | 0.948 [0.903, 0.992] | 0.052 [0.008, 0.097] | 0.946 [0.901, 0.990] | 0.054 [0.010, 0.099] |
