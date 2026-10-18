**aagr_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main_validation | 74 | 0.673 [0.635, 0.708] | 0.678 [0.641, 0.713] | 0.192 [0.144, 0.244] | 0.808 [0.756, 0.856] | 0.188 [0.139, 0.240] | 0.812 [0.760, 0.861] |
| Qwen3-30B-A3B-Base | main_all | 178 | 0.666 [0.640, 0.690] | 0.669 [0.643, 0.693] | 0.225 [0.191, 0.259] | 0.775 [0.741, 0.809] | 0.222 [0.188, 0.257] | 0.778 [0.743, 0.812] |
