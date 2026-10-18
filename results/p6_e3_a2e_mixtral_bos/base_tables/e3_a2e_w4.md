**e3_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main_validation | 113 | 0.424 [0.380, 0.467] | 0.425 [0.381, 0.468] | 0.862 [0.782, 0.943] | 0.138 [0.057, 0.218] | 0.861 [0.781, 0.943] | 0.139 [0.057, 0.219] |
| Mixtral-8x7B, BOS | main_all | 113 | 0.424 [0.380, 0.467] | 0.425 [0.381, 0.468] | 0.862 [0.782, 0.943] | 0.138 [0.057, 0.218] | 0.861 [0.781, 0.943] | 0.139 [0.057, 0.219] |
