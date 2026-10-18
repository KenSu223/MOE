**e3p3_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| mistral7b_bos | main_validation | 106 | 0.517 [0.483, 0.550] | 0.517 [0.483, 0.550] | 0.921 [0.856, 0.987] | 0.079 [0.013, 0.144] | 0.918 [0.853, 0.985] | 0.082 [0.015, 0.147] |
| mistral7b_bos | main_all | 106 | 0.517 [0.483, 0.550] | 0.517 [0.483, 0.550] | 0.921 [0.856, 0.987] | 0.079 [0.013, 0.144] | 0.918 [0.853, 0.985] | 0.082 [0.015, 0.147] |
