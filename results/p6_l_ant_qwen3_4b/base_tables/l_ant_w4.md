**l_ant W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_4b | main_validation | 126 | 0.556 [0.515, 0.598] | 0.556 [0.515, 0.598] | 0.457 [0.422, 0.491] | 0.543 [0.509, 0.578] | 0.457 [0.421, 0.491] | 0.543 [0.509, 0.579] |
| qwen3_4b | main_all | 252 | 0.532 [0.505, 0.559] | 0.532 [0.505, 0.559] | 0.480 [0.454, 0.504] | 0.520 [0.496, 0.546] | 0.479 [0.454, 0.504] | 0.521 [0.496, 0.546] |
