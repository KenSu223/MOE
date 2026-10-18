**l_bind W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_4b | main_validation | 128 | 0.127 [0.091, 0.161] | 0.127 [0.091, 0.161] | 1.537 [1.487, 1.588] | -0.537 [-0.588, -0.487] | 1.544 [1.493, 1.595] | -0.544 [-0.595, -0.493] |
| qwen3_4b | main_all | 256 | 0.119 [0.095, 0.144] | 0.119 [0.095, 0.144] | 1.545 [1.508, 1.581] | -0.545 [-0.581, -0.508] | 1.551 [1.514, 1.588] | -0.551 [-0.588, -0.514] |
