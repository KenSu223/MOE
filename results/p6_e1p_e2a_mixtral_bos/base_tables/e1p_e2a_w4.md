**e1p_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main_validation | 110 | 0.512 [0.473, 0.548] | 0.512 [0.473, 0.548] | 0.517 [0.479, 0.557] | 0.483 [0.443, 0.521] | 0.517 [0.479, 0.558] | 0.483 [0.442, 0.521] |
| Mixtral-8x7B, BOS | main_all | 110 | 0.512 [0.473, 0.548] | 0.512 [0.473, 0.548] | 0.517 [0.479, 0.557] | 0.483 [0.443, 0.521] | 0.517 [0.479, 0.558] | 0.483 [0.442, 0.521] |
