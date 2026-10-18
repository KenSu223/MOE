**l_cf W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_7b | main_validation | 88 | 0.744 [0.721, 0.765] | 0.744 [0.721, 0.765] | 0.275 [0.252, 0.298] | 0.725 [0.702, 0.748] | 0.275 [0.253, 0.299] | 0.725 [0.701, 0.747] |
| qwen2_7b | main_all | 181 | 0.730 [0.715, 0.745] | 0.730 [0.715, 0.745] | 0.288 [0.274, 0.303] | 0.712 [0.697, 0.726] | 0.288 [0.274, 0.303] | 0.712 [0.697, 0.726] |
