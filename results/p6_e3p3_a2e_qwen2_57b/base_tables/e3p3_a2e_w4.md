**e3p3_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen2_57b | main_validation | 128 | 0.674 [0.650, 0.697] | 0.674 [0.650, 0.697] | 0.998 [0.947, 1.051] | 0.002 [-0.051, 0.053] | 0.993 [0.942, 1.045] | 0.007 [-0.045, 0.058] |
| qwen2_57b | main_all | 256 | 0.674 [0.656, 0.691] | 0.674 [0.656, 0.691] | 0.991 [0.954, 1.027] | 0.009 [-0.027, 0.046] | 0.986 [0.950, 1.022] | 0.014 [-0.022, 0.050] |
