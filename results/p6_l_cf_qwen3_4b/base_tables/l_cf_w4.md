**l_cf W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_4b | main_validation | 83 | 0.376 [0.313, 0.436] | 0.376 [0.313, 0.436] | 0.734 [0.680, 0.789] | 0.266 [0.211, 0.320] | 0.734 [0.681, 0.790] | 0.266 [0.210, 0.319] |
| qwen3_4b | main_all | 173 | 0.382 [0.347, 0.416] | 0.382 [0.347, 0.416] | 0.733 [0.703, 0.764] | 0.267 [0.236, 0.297] | 0.733 [0.703, 0.764] | 0.267 [0.236, 0.297] |
