**l_prop W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_14b | main_validation | 128 | 0.618 [0.585, 0.648] | 0.618 [0.585, 0.648] | 0.446 [0.420, 0.474] | 0.554 [0.526, 0.580] | 0.444 [0.418, 0.472] | 0.556 [0.528, 0.582] |
| qwen3_14b | main_all | 256 | 0.608 [0.587, 0.628] | 0.608 [0.587, 0.628] | 0.444 [0.426, 0.463] | 0.556 [0.537, 0.574] | 0.442 [0.424, 0.461] | 0.558 [0.539, 0.576] |
