**e1_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_4b | main_validation | 128 | 0.325 [0.277, 0.369] | 0.325 [0.277, 0.369] | 0.666 [0.599, 0.737] | 0.334 [0.263, 0.401] | 0.666 [0.599, 0.737] | 0.334 [0.263, 0.401] |
| qwen3_4b | main_all | 238 | 0.345 [0.308, 0.381] | 0.345 [0.308, 0.381] | 0.664 [0.615, 0.715] | 0.336 [0.285, 0.385] | 0.663 [0.614, 0.714] | 0.337 [0.286, 0.386] |
