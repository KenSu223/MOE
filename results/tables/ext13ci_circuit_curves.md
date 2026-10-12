| Run | Curve | r(1) | r(5) | r(10) | r(20) [95% CI] | r(128) | max r (k) | k50/80/90 of drop | AUC log k (1..20) | all-MoE ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, F1 attr→entity | greedy_mix | 0.300 | 0.723 | 0.881 | 0.968 [0.942, 0.996] |  | 0.968 (20) | 3/7/12 | 0.677 | 0.243 |
| Qwen3-30B-A3B-Base, F1 attr→entity | greedy_head | 0.296 | 0.701 | 0.856 | 0.930 [0.904, 0.956] |  | 0.933 (19) | 3/8/14 | 0.657 | 0.243 |
| Qwen3-30B-A3B-Base, F1 attr→entity | greedy_expert_ab | 0.123 | 0.277 | 0.304 | 0.321 (k=15) |  | 0.321 (15) | never/never/never |  | 0.243 |
| Qwen3-30B-A3B-Base, F1 attr→entity | static_mix_oracle | 0.300 | 0.643 | 0.762 |  | 0.978 | 0.978 (128) | 3/16/48 | 0.603 | 0.243 |
| Qwen3-30B-A3B-Base, F1 attr→entity | static_mix_dla | 0.220 | 0.480 | 0.676 |  | 1.157 | 1.157 (128) | 6/16/24 | 0.493 | 0.243 |
| Qwen3-30B-A3B-Base, F1 attr→entity | static_head_oracle | 0.294 | 0.630 | 0.765 |  | 0.975 | 0.975 (128) | 3/16/48 | 0.601 | 0.243 |
| Qwen3-30B-A3B-Base, F1 attr→entity | static_head_dla | 0.215 | 0.455 | 0.639 |  | 1.110 | 1.110 (128) | 6/24/32 | 0.469 | 0.243 |
| Mixtral-8x7B (BOS), F1 attr→entity | greedy_mix | 0.229 | 0.668 | 0.871 | 1.031 [0.995, 1.072] |  | 1.031 (20) | 3/8/12 | 0.633 | 0.227 |
| Mixtral-8x7B (BOS), F1 attr→entity | greedy_head | 0.215 | 0.630 | 0.799 | 0.882 [0.853, 0.914] |  | 0.882 (20) | 4/11/never | 0.583 | 0.227 |
| Mixtral-8x7B (BOS), F1 attr→entity | greedy_expert_ab | 0.151 | 0.437 | 0.579 | 0.643 (k=15) |  | 0.643 (15) | 7/never/never |  | 0.227 |
| Mixtral-8x7B (BOS), F1 attr→entity | static_mix_oracle | 0.229 | 0.579 | 0.719 |  | 0.957 | 0.957 (128) | 4/24/48 | 0.543 | 0.227 |
| Mixtral-8x7B (BOS), F1 attr→entity | static_mix_dla | 0.110 | 0.348 | 0.522 |  | 1.080 | 1.080 (128) | 10/32/48 | 0.360 | 0.227 |
| Mixtral-8x7B (BOS), F1 attr→entity | static_head_oracle | 0.215 | 0.565 | 0.708 |  | 0.945 | 0.945 (128) | 4/24/96 | 0.530 | 0.227 |
| Mixtral-8x7B (BOS), F1 attr→entity | static_head_dla | 0.100 | 0.305 | 0.457 |  | 1.037 | 1.037 (128) | 12/48/48 | 0.320 | 0.227 |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | greedy_mix | 0.239 | 0.592 | 0.760 | 0.866 [0.843, 0.891] |  | 0.866 (20) | 4/13/never | 0.564 | 0.109 |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | greedy_head | 0.238 | 0.576 | 0.740 | 0.830 [0.802, 0.857] |  | 0.830 (20) | 4/15/never | 0.549 | 0.109 |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | greedy_expert_ab | 0.100 | 0.251 | 0.284 | 0.298 (k=15) |  | 0.299 (14) | never/never/never |  | 0.109 |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | static_mix_oracle | 0.236 | 0.535 | 0.656 |  | 0.922 | 0.922 (128) | 5/32/96 | 0.509 | 0.109 |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | static_mix_dla | 0.152 | 0.410 | 0.589 |  | 1.115 | 1.115 (128) | 8/24/32 | 0.419 | 0.109 |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | static_head_oracle | 0.236 | 0.526 | 0.655 |  | 0.928 | 0.928 (128) | 5/32/96 | 0.505 | 0.109 |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | static_head_dla | 0.149 | 0.381 | 0.547 |  | 1.099 | 1.099 (128) | 9/24/48 | 0.389 | 0.109 |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | greedy_mix | 0.230 | 0.662 | 0.855 | 1.003 [0.973, 1.038] |  | 1.003 (20) | 3/9/13 | 0.624 | 0.408 |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | greedy_head | 0.212 | 0.626 | 0.781 | 0.857 [0.829, 0.885] |  | 0.857 (20) | 4/12/never | 0.575 | 0.408 |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | greedy_expert_ab | 0.163 | 0.463 | 0.619 | 0.688 (k=15) |  | 0.688 (15) | 6/never/never |  | 0.408 |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | static_mix_oracle | 0.230 | 0.571 | 0.713 |  | 0.946 | 0.946 (128) | 4/24/48 | 0.538 | 0.408 |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | static_mix_dla | 0.118 | 0.367 | 0.543 |  | 1.060 | 1.060 (128) | 9/32/48 | 0.374 | 0.408 |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | static_head_oracle | 0.212 | 0.562 | 0.689 |  | 0.914 | 0.914 (128) | 4/24/96 | 0.519 | 0.408 |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | static_head_dla | 0.096 | 0.292 | 0.444 |  | 1.005 | 1.005 (128) | 16/48/64 | 0.306 | 0.408 |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | greedy_mix | 0.212 | 0.564 | 0.750 | 0.885 [0.857, 0.915] |  | 0.885 (20) | 4/13/never | 0.545 | 0.812 |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | greedy_head | 0.116 | 0.366 | 0.477 | 0.545 [0.511, 0.579] |  | 0.545 (20) | 12/never/never | 0.342 | 0.812 |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | greedy_expert_ab | 0.203 | 0.513 | 0.643 | 0.686 (k=15) |  | 0.686 (14) | 5/never/never |  | 0.812 |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | static_mix_oracle | 0.213 | 0.493 | 0.623 |  | 0.882 | 0.882 (128) | 6/48/never | 0.471 | 0.812 |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | static_mix_dla | 0.164 | 0.470 | 0.646 |  | 1.064 | 1.064 (128) | 6/24/48 | 0.460 | 0.812 |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | static_head_oracle | 0.117 | 0.300 | 0.382 |  | 0.630 | 0.630 (128) | 32/never/never | 0.287 | 0.812 |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | static_head_dla | 0.094 | 0.272 | 0.374 |  | 0.755 | 0.755 (128) | 24/never/never | 0.267 | 0.812 |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | greedy_mix | 0.193 | 0.549 | 0.732 | 0.888 [0.876, 0.900] |  | 0.888 (20) | 5/14/never | 0.529 | 0.756 |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | greedy_head | 0.093 | 0.358 | 0.536 | 0.619 [0.597, 0.641] |  | 0.619 (20) | 9/never/never | 0.348 | 0.756 |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | greedy_expert_ab | 0.191 | 0.521 | 0.676 | 0.753 (k=15) |  | 0.753 (15) | 5/never/never |  | 0.756 |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | static_mix_oracle | 0.193 | 0.522 | 0.677 |  | 0.934 | 0.934 (128) | 5/24/64 | 0.498 | 0.756 |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | static_mix_dla | 0.155 | 0.461 | 0.652 |  | 1.029 | 1.029 (128) | 6/24/32 | 0.459 | 0.756 |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | static_head_oracle | 0.092 | 0.317 | 0.460 |  | 0.744 | 0.744 (128) | 16/never/never | 0.310 | 0.756 |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | static_head_dla | 0.079 | 0.207 | 0.282 |  | 0.775 | 0.775 (128) | 48/never/never | 0.207 | 0.756 |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | greedy_mix | 0.222 | 0.593 | 0.762 | 0.865 [0.840, 0.890] |  | 0.865 (20) | 4/14/never | 0.558 | 0.786 |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | greedy_head | 0.181 | 0.461 | 0.559 | 0.598 [0.570, 0.628] |  | 0.598 (20) | 7/never/never | 0.424 | 0.786 |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | greedy_expert_ab | 0.171 | 0.461 | 0.569 | 0.598 (k=15) |  | 0.598 (15) | 6/never/never |  | 0.786 |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | static_mix_oracle | 0.220 | 0.505 | 0.598 |  | 0.868 | 0.868 (128) | 5/96/never | 0.467 | 0.786 |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | static_mix_dla | 0.171 | 0.486 | 0.645 |  | 1.077 | 1.077 (128) | 6/24/48 | 0.469 | 0.786 |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | static_head_oracle | 0.180 | 0.382 | 0.456 |  | 0.679 | 0.679 (128) | 24/never/never | 0.363 | 0.786 |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | static_head_dla | 0.158 | 0.369 | 0.468 |  | 0.853 | 0.853 (128) | 16/96/never | 0.357 | 0.786 |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | greedy_mix | 0.194 | 0.590 | 0.783 | 0.923 [0.910, 0.938] |  | 0.923 (20) | 4/11/18 | 0.560 | 0.732 |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | greedy_head | 0.117 | 0.427 | 0.595 | 0.657 [0.635, 0.680] |  | 0.657 (20) | 7/never/never | 0.399 | 0.732 |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | greedy_expert_ab | 0.182 | 0.537 | 0.699 | 0.772 (k=15) |  | 0.772 (15) | 5/never/never |  | 0.732 |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | static_mix_oracle | 0.194 | 0.548 | 0.715 |  | 0.950 | 0.950 (128) | 5/16/48 | 0.521 | 0.732 |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | static_mix_dla | 0.134 | 0.449 | 0.650 |  | 1.033 | 1.033 (128) | 6/24/32 | 0.444 | 0.732 |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | static_head_oracle | 0.117 | 0.369 | 0.506 |  | 0.758 | 0.758 (128) | 10/never/never | 0.353 | 0.732 |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | static_head_dla | 0.089 | 0.228 | 0.307 |  | 0.776 | 0.776 (128) | 32/never/never | 0.227 | 0.732 |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | greedy_mix | 0.207 | 0.586 | 0.795 | 0.939 [0.859, 1.035] |  | 0.939 (20) | 4/11/17 | 0.567 | 0.703 |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | greedy_head | 0.164 | 0.476 | 0.620 | 0.688 [0.610, 0.773] |  | 0.688 (20) | 6/never/never | 0.447 | 0.703 |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | greedy_expert_ab | 0.174 | 0.477 | 0.607 | 0.655 (k=15) |  | 0.655 (15) | 6/never/never |  | 0.703 |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | static_mix_oracle | 0.209 | 0.495 | 0.624 |  | 0.917 | 0.917 (128) | 6/48/128 | 0.472 | 0.703 |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | static_mix_dla | 0.176 | 0.418 | 0.598 |  | 1.112 | 1.112 (128) | 7/24/48 | 0.423 | 0.703 |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | static_head_oracle | 0.168 | 0.401 | 0.496 |  | 0.787 | 0.787 (128) | 12/never/never | 0.379 | 0.703 |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | static_head_dla | 0.124 | 0.360 | 0.485 |  | 0.979 | 0.979 (128) | 12/48/96 | 0.355 | 0.703 |
| Mixtral-8x7B (BOS), copy prefix, templates orig | greedy_mix | 0.240 | 0.626 | 0.809 | 0.936 [0.896, 0.975] |  | 0.936 (20) | 4/10/16 | 0.596 | 0.545 |
| Mixtral-8x7B (BOS), copy prefix, templates orig | greedy_head | 0.188 | 0.537 | 0.671 | 0.721 [0.640, 0.804] |  | 0.721 (20) | 5/never/never | 0.492 | 0.545 |
| Mixtral-8x7B (BOS), copy prefix, templates orig | greedy_expert_ab | 0.191 | 0.519 | 0.667 | 0.726 (k=15) |  | 0.726 (15) | 5/never/never |  | 0.545 |
| Mixtral-8x7B (BOS), copy prefix, templates orig | static_mix_oracle | 0.241 | 0.572 | 0.710 |  | 0.943 | 0.943 (128) | 4/24/48 | 0.541 | 0.545 |
| Mixtral-8x7B (BOS), copy prefix, templates orig | static_mix_dla | 0.174 | 0.470 | 0.632 |  | 1.050 | 1.050 (128) | 6/24/32 | 0.466 | 0.545 |
| Mixtral-8x7B (BOS), copy prefix, templates orig | static_head_oracle | 0.188 | 0.461 | 0.592 |  | 0.806 | 0.806 (128) | 7/128/never | 0.439 | 0.545 |
| Mixtral-8x7B (BOS), copy prefix, templates orig | static_head_dla | 0.096 | 0.238 | 0.371 |  | 0.922 | 0.922 (128) | 24/96/128 | 0.255 | 0.545 |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | greedy_mix | 0.208 | 0.584 | 0.787 | 0.930 [0.836, 1.059] |  | 0.930 (20) | 4/11/17 | 0.566 | 0.739 |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | greedy_head | 0.162 | 0.478 | 0.618 | 0.683 [0.603, 0.776] |  | 0.683 (20) | 6/never/never | 0.447 | 0.739 |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | greedy_expert_ab | 0.184 | 0.482 | 0.612 | 0.659 (k=15) |  | 0.659 (15) | 6/never/never |  | 0.739 |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | static_mix_oracle | 0.210 | 0.474 | 0.610 |  | 0.912 | 0.912 (128) | 6/48/128 | 0.461 | 0.739 |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | static_mix_dla | 0.155 | 0.425 | 0.592 |  | 1.078 | 1.078 (128) | 7/24/48 | 0.422 | 0.739 |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | static_head_oracle | 0.162 | 0.399 | 0.493 |  | 0.779 | 0.779 (128) | 12/never/never | 0.376 | 0.739 |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | static_head_dla | 0.098 | 0.343 | 0.466 |  | 0.995 | 0.995 (128) | 12/48/96 | 0.335 | 0.739 |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | greedy_mix | 0.174 | 0.543 | 0.735 | 0.885 [0.830, 0.967] |  | 0.885 (20) | 5/13/never | 0.523 | 0.463 |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | greedy_head | 0.129 | 0.442 | 0.620 | 0.706 [0.665, 0.823] |  | 0.706 (20) | 7/never/never | 0.424 | 0.463 |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | greedy_expert_ab | 0.160 | 0.488 | 0.637 | 0.701 (k=15) |  | 0.701 (15) | 6/never/never |  | 0.463 |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | static_mix_oracle | 0.174 | 0.504 | 0.662 |  | 0.936 | 0.936 (128) | 5/24/64 | 0.482 | 0.463 |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | static_mix_dla | 0.142 | 0.448 | 0.623 |  | 1.049 | 1.049 (128) | 7/24/48 | 0.436 | 0.463 |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | static_head_oracle | 0.128 | 0.390 | 0.544 |  | 0.813 | 0.813 (128) | 9/128/never | 0.381 | 0.463 |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | static_head_dla | 0.087 | 0.260 | 0.376 |  | 1.040 | 1.040 (128) | 24/48/64 | 0.258 | 0.463 |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | greedy_mix | 0.240 | 0.623 | 0.826 | 0.962 [0.888, 1.067] |  | 0.962 (20) | 4/10/15 | 0.600 | 0.823 |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | greedy_head | 0.147 | 0.440 | 0.574 | 0.627 [0.546, 0.722] |  | 0.627 (20) | 7/never/never | 0.411 | 0.823 |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | greedy_expert_ab | 0.221 | 0.539 | 0.676 | 0.722 (k=15) |  | 0.722 (15) | 5/never/never |  | 0.823 |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | static_mix_oracle | 0.239 | 0.551 | 0.692 |  | 0.955 | 0.955 (128) | 4/24/64 | 0.526 | 0.823 |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | static_mix_dla | 0.206 | 0.517 | 0.668 |  | 1.118 | 1.118 (128) | 5/24/32 | 0.498 | 0.823 |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | static_head_oracle | 0.147 | 0.373 | 0.457 |  | 0.726 | 0.726 (128) | 16/never/never | 0.348 | 0.823 |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | static_head_dla | 0.120 | 0.309 | 0.432 |  | 0.966 | 0.966 (128) | 16/64/96 | 0.313 | 0.823 |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | greedy_mix | 0.243 | 0.618 | 0.797 | 0.935 [0.899, 0.975] |  | 0.935 (20) | 4/11/17 | 0.592 | 0.677 |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | greedy_head | 0.183 | 0.519 | 0.671 | 0.737 [0.668, 0.799] |  | 0.737 (20) | 5/never/never | 0.484 | 0.677 |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | greedy_expert_ab | 0.205 | 0.517 | 0.652 | 0.718 (k=15) |  | 0.718 (15) | 5/never/never |  | 0.677 |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | static_mix_oracle | 0.243 | 0.580 | 0.716 |  | 0.957 | 0.957 (128) | 4/24/48 | 0.548 | 0.677 |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | static_mix_dla | 0.186 | 0.507 | 0.679 |  | 1.059 | 1.059 (128) | 5/24/32 | 0.495 | 0.677 |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | static_head_oracle | 0.183 | 0.459 | 0.601 |  | 0.831 | 0.831 (128) | 7/96/never | 0.441 | 0.677 |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | static_head_dla | 0.117 | 0.301 | 0.411 |  | 0.959 | 0.959 (128) | 16/64/96 | 0.298 | 0.677 |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | greedy_mix | 0.246 | 0.625 | 0.805 | 0.922 [0.882, 0.966] |  | 0.922 (20) | 4/10/18 | 0.593 | 0.813 |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | greedy_head | 0.149 | 0.424 | 0.530 | 0.578 [0.514, 0.646] |  | 0.578 (20) | 9/never/never | 0.391 | 0.813 |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | greedy_expert_ab | 0.226 | 0.542 | 0.674 | 0.712 (k=15) |  | 0.712 (15) | 5/never/never |  | 0.813 |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | static_mix_oracle | 0.246 | 0.552 | 0.673 |  | 0.924 | 0.924 (128) | 4/32/96 | 0.520 | 0.813 |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | static_mix_dla | 0.208 | 0.520 | 0.682 |  | 1.118 | 1.118 (128) | 5/24/32 | 0.504 | 0.813 |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | static_head_oracle | 0.149 | 0.346 | 0.425 |  | 0.694 | 0.694 (128) | 24/never/never | 0.330 | 0.813 |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | static_head_dla | 0.107 | 0.314 | 0.419 |  | 0.975 | 0.975 (128) | 16/64/96 | 0.306 | 0.813 |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | greedy_mix | 0.230 | 0.591 | 0.760 | 0.890 [0.853, 0.939] |  | 0.890 (20) | 4/12/never | 0.564 | 0.623 |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | greedy_head | 0.181 | 0.488 | 0.642 | 0.722 [0.664, 0.782] |  | 0.722 (20) | 6/never/never | 0.465 | 0.623 |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | greedy_expert_ab | 0.191 | 0.496 | 0.635 | 0.696 (k=15) |  | 0.696 (15) | 6/never/never |  | 0.623 |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | static_mix_oracle | 0.228 | 0.545 | 0.682 |  | 0.934 | 0.934 (128) | 4/24/64 | 0.519 | 0.623 |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | static_mix_dla | 0.171 | 0.475 | 0.652 |  | 1.059 | 1.059 (128) | 6/24/32 | 0.468 | 0.623 |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | static_head_oracle | 0.180 | 0.445 | 0.580 |  | 0.830 | 0.830 (128) | 7/96/never | 0.430 | 0.623 |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | static_head_dla | 0.110 | 0.284 | 0.393 |  | 0.994 | 0.994 (128) | 16/48/96 | 0.284 | 0.623 |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | greedy_mix | 0.329 | 0.758 | 0.920 | 1.027 [0.895, 1.066] |  | 1.027 (20) | 2/6/10 | 0.718 | 0.211 |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | greedy_head | 0.329 | 0.755 | 0.914 | 1.004 [0.870, 1.016] |  | 1.006 (19) | 2/7/10 | 0.714 | 0.211 |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | greedy_expert_ab | 0.038 | 0.168 | 0.217 | 0.233 (k=15) |  | 0.233 (15) | never/never/never |  | 0.211 |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | static_mix_oracle | 0.330 | 0.719 | 0.836 |  | 1.055 | 1.055 (128) | 2/8/16 | 0.673 | 0.211 |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | static_mix_dla | 0.307 | 0.602 | 0.797 |  | 1.258 | 1.258 (128) | 3/12/16 | 0.610 | 0.211 |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | static_head_oracle | 0.330 | 0.722 | 0.851 |  | 1.056 | 1.056 (128) | 2/8/16 | 0.678 | 0.211 |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | static_head_dla | 0.307 | 0.602 | 0.797 |  | 1.244 | 1.244 (128) | 3/12/16 | 0.610 | 0.211 |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | greedy_mix | 0.272 | 0.707 | 0.895 | 1.015 [0.949, 1.094] |  | 1.015 (20) | 3/8/12 | 0.658 | 0.227 |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | greedy_head | 0.270 | 0.676 | 0.822 | 0.894 [0.828, 0.961] |  | 0.894 (20) | 3/9/never | 0.614 | 0.227 |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | greedy_expert_ab | 0.126 | 0.347 | 0.418 | 0.457 (k=15) |  | 0.457 (15) | never/never/never |  | 0.227 |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | static_mix_oracle | 0.272 | 0.551 | 0.687 |  | 0.956 | 0.956 (128) | 4/24/64 | 0.536 | 0.227 |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | static_mix_dla | 0.139 | 0.471 | 0.641 |  | 1.148 | 1.148 (128) | 6/24/32 | 0.454 | 0.227 |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | static_head_oracle | 0.269 | 0.535 | 0.680 |  | 0.926 | 0.926 (128) | 5/24/96 | 0.528 | 0.227 |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | static_head_dla | 0.121 | 0.436 | 0.620 |  | 1.081 | 1.081 (128) | 7/24/32 | 0.422 | 0.227 |
| Mixtral-8x7B (BOS), F2E natural, blank at end | greedy_mix | 0.197 | 0.524 | 0.690 | 0.826 [0.769, 0.899] |  | 0.826 (20) | 5/17/never | 0.505 | -0.056 |
| Mixtral-8x7B (BOS), F2E natural, blank at end | greedy_head | 0.185 | 0.482 | 0.629 | 0.725 [0.669, 0.786] |  | 0.725 (20) | 6/never/never | 0.463 | -0.056 |
| Mixtral-8x7B (BOS), F2E natural, blank at end | greedy_expert_ab | 0.141 | 0.352 | 0.449 | 0.502 (k=15) |  | 0.502 (15) | 15/never/never |  | -0.056 |
| Mixtral-8x7B (BOS), F2E natural, blank at end | static_mix_oracle | 0.197 | 0.475 | 0.591 |  | 0.875 | 0.875 (128) | 6/48/never | 0.449 | -0.056 |
| Mixtral-8x7B (BOS), F2E natural, blank at end | static_mix_dla | 0.088 | 0.284 | 0.434 |  | 1.036 | 1.036 (128) | 16/48/64 | 0.294 | -0.056 |
| Mixtral-8x7B (BOS), F2E natural, blank at end | static_head_oracle | 0.184 | 0.440 | 0.556 |  | 0.816 | 0.816 (128) | 7/96/never | 0.424 | -0.056 |
| Mixtral-8x7B (BOS), F2E natural, blank at end | static_head_dla | 0.071 | 0.273 | 0.421 |  | 1.001 | 1.001 (128) | 16/48/64 | 0.279 | -0.056 |
