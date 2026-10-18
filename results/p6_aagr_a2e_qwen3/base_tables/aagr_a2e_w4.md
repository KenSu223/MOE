**aagr_a2e W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main_validation | 128 | 0.023 [0.001, 0.046] | 0.025 [0.002, 0.048] | 1.093 [1.070, 1.115] | -0.093 [-0.115, -0.070] | 1.090 [1.068, 1.113] | -0.090 [-0.113, -0.068] |
| Qwen3-30B-A3B-Base | main_all | 256 | 0.019 [0.002, 0.036] | 0.020 [0.002, 0.037] | 1.108 [1.091, 1.126] | -0.108 [-0.126, -0.091] | 1.105 [1.089, 1.123] | -0.105 [-0.123, -0.089] |
