**l_prop W4: all-MoE patch and direct-path split at the final position (/ drop; pair bootstrap)**

| Model | Set | Pairs | All-MoE M denoise | All-MoE M noise | A_direct | M_direct | DLA share attention | DLA share MoE |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | main_validation | 127 | 0.678 [0.636, 0.715] | 0.678 [0.637, 0.715] | 0.279 [0.241, 0.322] | 0.721 [0.678, 0.759] | 0.275 [0.236, 0.318] | 0.725 [0.682, 0.764] |
| Qwen3-30B-A3B-Base | main_all | 255 | 0.664 [0.638, 0.688] | 0.664 [0.639, 0.688] | 0.281 [0.256, 0.307] | 0.719 [0.693, 0.744] | 0.278 [0.252, 0.304] | 0.722 [0.696, 0.748] |
