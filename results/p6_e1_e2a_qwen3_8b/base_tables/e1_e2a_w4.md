**e1_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_8b | main_validation | 118 | 0.282 [0.230, 0.333] | 0.282 [0.230, 0.333] | 0.644 [0.571, 0.717] | 0.356 [0.283, 0.429] | 0.644 [0.570, 0.715] | 0.356 [0.285, 0.430] |
| qwen3_8b | main_all | 202 | 0.259 [0.218, 0.298] | 0.259 [0.218, 0.298] | 0.662 [0.605, 0.720] | 0.338 [0.280, 0.395] | 0.661 [0.604, 0.718] | 0.339 [0.282, 0.396] |
