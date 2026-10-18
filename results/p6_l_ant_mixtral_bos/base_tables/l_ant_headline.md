**l_ant: headline numbers with cluster-bootstrap 95% CIs (unit = item; 5,000 resamples)**

| Model | Set | Pairs | Units | Attention share (AUC+) | AUC+ attention / drop | AUC+ MoE / drop | A_direct / drop | M_direct / drop | All-MoE M / drop | MoE peak (layer, / drop) | Attention peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main, validation | 126 | 42 | 0.48 [0.45, 0.51] | 0.753 [0.709, 0.807] | 0.815 [0.732, 0.912] | 0.202 [0.124, 0.276] | 0.798 [0.724, 0.876] | 0.847 [0.766, 0.939] | L18 0.083 | L21 0.147 |
| Mixtral-8x7B, BOS | main, all | 252 | 84 | 0.49 [0.47, 0.51] | 0.728 [0.697, 0.765] | 0.753 [0.699, 0.821] | 0.220 [0.160, 0.278] | 0.780 [0.722, 0.840] | 0.837 [0.774, 0.906] | L26 0.085 | L21 0.141 |
| Mixtral-8x7B, BOS | objects (main, all) | 252 | 84 | 0.49 [0.47, 0.51] | 0.728 [0.697, 0.765] | 0.753 [0.699, 0.821] | 0.220 [0.160, 0.278] | 0.780 [0.722, 0.840] | 0.837 [0.774, 0.906] | L26 0.085 | L21 0.141 |
| Mixtral-8x7B, BOS | not neutral (main, all) | 252 | 84 | 0.49 [0.47, 0.51] | 0.728 [0.697, 0.765] | 0.753 [0.699, 0.821] | 0.220 [0.160, 0.278] | 0.780 [0.722, 0.840] | 0.837 [0.774, 0.906] | L26 0.085 | L21 0.141 |
