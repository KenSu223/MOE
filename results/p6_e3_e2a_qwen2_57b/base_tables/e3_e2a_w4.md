**e3_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_57b | main_validation | 128 | 0.923 [0.913, 0.932] | 0.923 [0.913, 0.932] | 0.047 [0.033, 0.063] | 0.953 [0.937, 0.967] | 0.043 [0.028, 0.059] | 0.957 [0.941, 0.972] |
| qwen2_57b | main_all | 256 | 0.924 [0.917, 0.930] | 0.924 [0.917, 0.930] | 0.047 [0.036, 0.059] | 0.953 [0.941, 0.964] | 0.043 [0.031, 0.055] | 0.957 [0.945, 0.969] |
