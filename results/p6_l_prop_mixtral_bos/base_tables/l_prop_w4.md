**l_prop W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main_validation | 128 | 0.397 [0.358, 0.436] | 0.398 [0.359, 0.437] | 0.592 [0.558, 0.625] | 0.408 [0.375, 0.442] | 0.591 [0.558, 0.624] | 0.409 [0.376, 0.442] |
| Mixtral-8x7B, BOS | main_all | 250 | 0.429 [0.403, 0.454] | 0.429 [0.403, 0.455] | 0.559 [0.537, 0.581] | 0.441 [0.419, 0.463] | 0.558 [0.536, 0.580] | 0.442 [0.420, 0.464] |
