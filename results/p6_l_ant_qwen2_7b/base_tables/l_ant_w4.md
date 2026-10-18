**l_ant W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_7b | main_validation | 126 | 0.763 [0.721, 0.805] | 0.763 [0.721, 0.805] | 0.196 [0.149, 0.243] | 0.804 [0.757, 0.851] | 0.196 [0.149, 0.243] | 0.804 [0.757, 0.851] |
| qwen2_7b | main_all | 252 | 0.776 [0.747, 0.807] | 0.776 [0.747, 0.807] | 0.186 [0.153, 0.217] | 0.814 [0.783, 0.847] | 0.186 [0.153, 0.217] | 0.814 [0.783, 0.847] |
