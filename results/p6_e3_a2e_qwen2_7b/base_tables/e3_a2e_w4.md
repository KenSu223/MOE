**e3_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_7b | main_validation | 68 | 0.709 [0.675, 0.741] | 0.709 [0.675, 0.741] | 0.778 [0.684, 0.876] | 0.222 [0.124, 0.316] | 0.778 [0.684, 0.875] | 0.222 [0.125, 0.316] |
| qwen2_7b | main_all | 160 | 0.701 [0.679, 0.724] | 0.701 [0.679, 0.724] | 0.773 [0.712, 0.833] | 0.227 [0.167, 0.288] | 0.771 [0.711, 0.832] | 0.229 [0.168, 0.289] |
