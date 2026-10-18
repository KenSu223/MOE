**e3_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main_validation | 128 | 0.823 [0.806, 0.838] | 0.823 [0.806, 0.839] | 0.028 [-0.000, 0.056] | 0.972 [0.944, 1.000] | 0.025 [-0.003, 0.054] | 0.975 [0.946, 1.003] |
| Qwen3-30B-A3B-Base | main_all | 256 | 0.830 [0.818, 0.841] | 0.830 [0.818, 0.841] | 0.017 [-0.003, 0.037] | 0.983 [0.963, 1.003] | 0.014 [-0.006, 0.034] | 0.986 [0.966, 1.006] |
