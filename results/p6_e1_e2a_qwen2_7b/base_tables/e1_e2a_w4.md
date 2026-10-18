**e1_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_7b | main_validation | 128 | 0.683 [0.645, 0.720] | 0.683 [0.645, 0.720] | 0.335 [0.282, 0.391] | 0.665 [0.609, 0.718] | 0.334 [0.281, 0.390] | 0.666 [0.610, 0.719] |
| qwen2_7b | main_all | 256 | 0.698 [0.672, 0.724] | 0.698 [0.672, 0.724] | 0.353 [0.311, 0.397] | 0.647 [0.603, 0.689] | 0.352 [0.309, 0.395] | 0.648 [0.605, 0.691] |
