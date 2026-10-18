**e3p3_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Mixtral-8x7B, BOS | main_validation | 80 | 0.464 [0.425, 0.502] | 0.464 [0.425, 0.501] | 0.836 [0.761, 0.911] | 0.164 [0.089, 0.239] | 0.833 [0.759, 0.908] | 0.167 [0.092, 0.241] |
| Mixtral-8x7B, BOS | main_all | 160 | 0.458 [0.429, 0.488] | 0.457 [0.428, 0.486] | 0.798 [0.746, 0.850] | 0.202 [0.150, 0.254] | 0.795 [0.743, 0.847] | 0.205 [0.153, 0.257] |
