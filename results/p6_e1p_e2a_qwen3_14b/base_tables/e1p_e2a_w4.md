**e1p_e2a W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| qwen3_14b | main_validation | 128 | 0.718 [0.690, 0.743] | 0.718 [0.690, 0.743] | 0.307 [0.277, 0.339] | 0.693 [0.661, 0.723] | 0.304 [0.275, 0.336] | 0.696 [0.664, 0.725] |
| qwen3_14b | main_all | 256 | 0.696 [0.676, 0.715] | 0.696 [0.676, 0.715] | 0.317 [0.296, 0.340] | 0.683 [0.660, 0.704] | 0.315 [0.294, 0.338] | 0.685 [0.662, 0.706] |
