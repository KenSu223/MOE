**l_prop W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_8b | main_validation | 128 | 0.606 [0.575, 0.634] | 0.606 [0.575, 0.634] | 0.440 [0.413, 0.470] | 0.560 [0.530, 0.587] | 0.437 [0.411, 0.467] | 0.563 [0.533, 0.589] |
| qwen3_8b | main_all | 256 | 0.579 [0.559, 0.599] | 0.579 [0.559, 0.599] | 0.456 [0.437, 0.476] | 0.544 [0.524, 0.563] | 0.453 [0.434, 0.473] | 0.547 [0.527, 0.566] |
