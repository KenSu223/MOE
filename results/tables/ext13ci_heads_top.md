| Run | Head | Rank | Rescue / drop [95% CI] | Rescue (logits) | z (all / halves) | Z8 both halves | DLA / drop | Row top-5 / top-1 share | Attention (clean run) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, F1 attr→entity | L40H2 | 1 | +0.257 [+0.242, +0.272] | +1.18 | +14.8 / +14.3, +15.0 | yes | +0.005 | 0.93 / 0.46 | self (final token) |
| Qwen3-30B-A3B-Base, F1 attr→entity | L42H10 | 2 | +0.212 [+0.191, +0.234] | +0.97 | +12.2 / +12.0, +12.2 | yes | +0.397 | 0.73 / 0.25 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, F1 attr→entity | L41H20 | 3 | +0.134 [+0.121, +0.148] | +0.62 | +7.7 / +7.8, +7.6 | yes | +0.052 | 0.41 / 0.00 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, F1 attr→entity | L39H29 | 4 | +0.119 [+0.108, +0.132] | +0.55 | +6.9 / +6.9, +6.8 | yes | +0.005 | 0.28 / 0.02 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, F1 attr→entity | L43H24 | 5 | +0.110 [+0.099, +0.122] | +0.51 | +6.4 / +6.0, +6.6 | yes | +0.090 | 0.25 / 0.00 | sink (pos0) |
| Qwen3-30B-A3B-Base, F1 attr→entity | L36H28 | 6 | +0.107 [+0.095, +0.121] | +0.49 | +6.2 / +6.1, +6.3 | yes | +0.002 | 0.21 / 0.02 | diffuse |
| Qwen3-30B-A3B-Base, F1 attr→entity | L36H23 | 7 | +0.096 [+0.082, +0.111] | +0.44 | +5.5 / +5.9, +5.1 | yes | +0.001 | 0.21 / 0.05 | STR site (trigger) |
| Qwen3-30B-A3B-Base, F1 attr→entity | L38H20 | 8 | +0.093 [+0.083, +0.103] | +0.43 | +5.4 / +5.2, +5.5 | yes | +0.001 | 0.13 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, F1 attr→entity | L44H15 | 1536 | -0.244 [-0.278, -0.212] | -1.12 | -14.0 / -14.2, -13.5 | yes | -0.104 | 0.00 / 0.00 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, F1 attr→entity | L44H31 | 1535 | -0.104 [-0.116, -0.092] | -0.48 | -5.9 / -6.0, -5.8 | yes | -0.064 | 0.00 / 0.00 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, F1 attr→entity | L36H31 | 1534 | -0.100 [-0.111, -0.090] | -0.46 | -5.7 / -5.6, -5.8 | yes | -0.001 | 0.00 / 0.00 | diffuse |
| Mixtral-8x7B (BOS), F1 attr→entity | L18H10 | 1 | +0.092 [+0.080, +0.105] | +0.40 | +8.3 / +8.4, +8.0 | yes | +0.003 | 0.37 / 0.07 | other context tokens |
| Mixtral-8x7B (BOS), F1 attr→entity | L20H21 | 2 | +0.083 [+0.066, +0.102] | +0.36 | +7.4 / +8.0, +6.6 | yes | +0.008 | 0.30 / 0.05 | self (final token) |
| Mixtral-8x7B (BOS), F1 attr→entity | L20H27 | 3 | +0.081 [+0.064, +0.100] | +0.36 | +7.3 / +5.6, +8.8 | yes | +0.024 | 0.38 / 0.15 | STR site (trigger) |
| Mixtral-8x7B (BOS), F1 attr→entity | L21H11 | 4 | +0.072 [+0.057, +0.089] | +0.32 | +6.4 / +6.0, +6.7 | yes | +0.082 | 0.30 / 0.08 | diffuse |
| Mixtral-8x7B (BOS), F1 attr→entity | L31H22 | 5 | +0.068 [+0.060, +0.077] | +0.30 | +6.1 / +5.4, +6.5 | yes | +0.057 | 0.19 / 0.02 | other context tokens |
| Mixtral-8x7B (BOS), F1 attr→entity | L19H12 | 6 | +0.068 [+0.057, +0.080] | +0.30 | +6.0 / +6.2, +5.7 | yes | +0.011 | 0.27 / 0.02 | STR site (trigger) |
| Mixtral-8x7B (BOS), F1 attr→entity | L12H21 | 7 | +0.066 [+0.051, +0.082] | +0.29 | +5.9 / +6.6, +4.9 | yes | +0.001 | 0.25 / 0.12 | STR site (trigger) |
| Mixtral-8x7B (BOS), F1 attr→entity | L16H1 | 8 | +0.063 [+0.050, +0.077] | +0.27 | +5.6 / +6.0, +5.0 | yes | +0.002 | 0.21 / 0.07 | other context tokens |
| Mixtral-8x7B (BOS), F1 attr→entity | L20H25 | 1024 | -0.058 [-0.069, -0.048] | -0.25 | -5.5 / -4.8, -6.2 | yes | -0.029 | 0.00 / 0.00 | diffuse |
| Mixtral-8x7B (BOS), F1 attr→entity | L28H2 | 1023 | -0.054 [-0.063, -0.046] | -0.24 | -5.2 / -4.9, -5.4 | yes | -0.035 | 0.00 / 0.00 | mover (mention: answer) |
| Mixtral-8x7B (BOS), F1 attr→entity | L18H11 | 1022 | -0.049 [-0.056, -0.042] | -0.21 | -4.7 / -4.7, -4.6 | yes | -0.002 | 0.00 / 0.00 | other context tokens |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | L40H2 | 1 | +0.210 [+0.194, +0.227] | +1.10 | +15.6 / +15.1, +15.8 | yes | +0.013 | 0.94 / 0.61 | self (final token) |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | L36H23 | 2 | +0.108 [+0.098, +0.120] | +0.57 | +8.0 / +8.4, +7.3 | yes | +0.002 | 0.42 / 0.04 | STR site (trigger) |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | L42H10 | 3 | +0.101 [+0.082, +0.121] | +0.53 | +7.4 / +7.6, +7.0 | yes | +0.176 | 0.35 / 0.06 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | L42H31 | 4 | +0.092 [+0.075, +0.110] | +0.48 | +6.7 / +7.7, +5.6 | yes | +0.054 | 0.37 / 0.18 | self (final token) |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | L39H29 | 5 | +0.089 [+0.076, +0.103] | +0.47 | +6.6 / +6.3, +6.6 | yes | +0.008 | 0.28 / 0.01 | diffuse |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | L39H27 | 6 | +0.074 [+0.066, +0.083] | +0.39 | +5.4 / +4.8, +5.8 | yes | +0.024 | 0.13 / 0.01 | other context tokens |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | L41H20 | 7 | +0.073 [+0.061, +0.087] | +0.38 | +5.4 / +4.8, +5.8 | yes | +0.023 | 0.16 / 0.00 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | L38H3 | 8 | +0.071 [+0.064, +0.078] | +0.37 | +5.2 / +5.5, +4.7 | yes | +0.010 | 0.25 / 0.01 | STR site (trigger) |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | L44H15 | 1536 | -0.100 [-0.119, -0.081] | -0.52 | -7.6 / -7.7, -7.4 | yes | -0.046 | 0.00 / 0.00 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | L36H31 | 1535 | -0.080 [-0.090, -0.071] | -0.42 | -6.1 / -5.8, -6.3 | yes | -0.003 | 0.00 / 0.00 | STR site (trigger) |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | L37H3 | 1534 | -0.080 [-0.088, -0.071] | -0.42 | -6.1 / -6.4, -5.7 | yes | -0.002 | 0.00 / 0.00 | STR site (trigger) |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | L20H27 | 1 | +0.121 [+0.103, +0.139] | +0.56 | +9.9 / +12.1, +7.5 | yes | +0.036 | 0.55 / 0.24 | STR site (trigger) |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | L21H11 | 2 | +0.106 [+0.090, +0.123] | +0.50 | +8.7 / +7.2, +10.1 | yes | +0.115 | 0.49 / 0.16 | mover (mention: answer) |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | L12H21 | 3 | +0.100 [+0.084, +0.118] | +0.47 | +8.2 / +7.1, +9.1 | yes | +0.002 | 0.39 / 0.20 | STR site (trigger) |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | L19H12 | 4 | +0.093 [+0.082, +0.104] | +0.43 | +7.6 / +8.0, +7.0 | yes | +0.014 | 0.36 / 0.05 | STR site (trigger) |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | L18H10 | 5 | +0.085 [+0.076, +0.095] | +0.40 | +6.9 / +6.6, +7.1 | yes | +0.003 | 0.35 / 0.02 | diffuse |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | L15H8 | 6 | +0.083 [+0.070, +0.096] | +0.39 | +6.7 / +6.9, +6.4 | yes | +0.002 | 0.30 / 0.09 | STR site (trigger) |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | L16H23 | 7 | +0.082 [+0.071, +0.092] | +0.38 | +6.6 / +6.9, +6.2 | yes | +0.005 | 0.38 / 0.11 | STR site (trigger) |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | L20H21 | 8 | +0.061 [+0.050, +0.072] | +0.28 | +4.9 / +4.2, +5.4 | yes | +0.005 | 0.19 / 0.01 | self (final token) |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | L20H25 | 1024 | -0.083 [-0.095, -0.071] | -0.39 | -7.2 / -7.8, -6.4 | yes | -0.037 | 0.00 / 0.00 | diffuse |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | L28H2 | 1023 | -0.060 [-0.069, -0.052] | -0.28 | -5.3 / -5.4, -5.0 | yes | -0.037 | 0.00 / 0.00 | mover (mention: answer) |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | L18H28 | 1022 | -0.056 [-0.065, -0.048] | -0.26 | -5.0 / -4.5, -5.3 | yes | -0.002 | 0.00 / 0.00 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | L46H24 | 1 | +0.074 [+0.064, +0.084] | +0.64 | +14.7 / +14.6, +13.7 | yes | +0.090 | 0.56 / 0.22 | diffuse |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | L41H27 | 2 | +0.056 [+0.047, +0.066] | +0.49 | +11.2 / +10.1, +11.6 | yes | +0.031 | 0.51 / 0.19 | sink (pos0) |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | L38H18 | 3 | +0.052 [+0.043, +0.062] | +0.46 | +10.5 / +9.7, +10.5 | yes | +0.028 | 0.46 / 0.13 | sink (pos0) |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | L38H21 | 4 | +0.036 [+0.028, +0.047] | +0.32 | +7.3 / +5.8, +8.3 | yes | +0.021 | 0.21 / 0.06 | sink (pos0) |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | L41H0 | 5 | +0.025 [+0.019, +0.031] | +0.22 | +5.0 / +4.6, +5.0 | yes | +0.003 | 0.17 / 0.03 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | L42H30 | 6 | +0.024 [+0.020, +0.028] | +0.21 | +4.9 / +5.4, +4.0 | yes | +0.014 | 0.11 / 0.01 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | L39H5 | 7 | +0.022 [+0.017, +0.028] | +0.19 | +4.4 / +3.6, +4.9 | yes | -0.001 | 0.12 / 0.02 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | L38H19 | 8 | +0.020 [+0.016, +0.024] | +0.17 | +4.0 / +2.9, +4.9 | yes | +0.010 | 0.04 / 0.01 | diffuse |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | L46H25 | 1536 | -0.085 [-0.094, -0.076] | -0.74 | -17.0 / -15.6, -17.2 | yes | -0.112 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | L38H16 | 1535 | -0.064 [-0.075, -0.055] | -0.56 | -12.8 / -12.9, -11.8 | yes | -0.049 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | L41H24 | 1534 | -0.037 [-0.042, -0.032] | -0.32 | -7.4 / -7.1, -7.1 | yes | -0.021 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | L25H9 | 1 | +0.064 [+0.058, +0.071] | +0.62 | +15.4 / +16.6, +13.8 | yes | +0.077 | 0.80 / 0.25 | STR site (filled entity) |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | L22H20 | 2 | +0.056 [+0.048, +0.064] | +0.54 | +13.4 / +10.6, +15.4 | yes | +0.036 | 0.66 / 0.27 | STR site (filled entity) |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | L19H13 | 3 | +0.043 [+0.037, +0.049] | +0.41 | +10.2 / +10.8, +9.4 | yes | -0.000 | 0.61 / 0.14 | STR site (filled entity) |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | L20H26 | 4 | +0.028 [+0.023, +0.034] | +0.27 | +6.7 / +8.2, +5.1 | yes | +0.001 | 0.39 / 0.07 | STR site (filled entity) |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | L13H4 | 5 | +0.021 [+0.017, +0.026] | +0.21 | +5.1 / +5.7, +4.4 | yes | +0.000 | 0.20 / 0.05 | STR site (filled entity) |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | L13H18 | 6 | +0.021 [+0.017, +0.025] | +0.20 | +5.0 / +4.8, +5.0 | yes | +0.000 | 0.23 / 0.03 | STR site (filled entity) |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | L15H9 | 7 | +0.019 [+0.015, +0.022] | +0.18 | +4.4 / +3.7, +4.9 | yes | +0.002 | 0.18 / 0.03 | mover (mention: answer) |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | L31H22 | 8 | +0.018 [+0.016, +0.020] | +0.18 | +4.4 / +4.5, +4.1 | yes | +0.019 | 0.09 / 0.00 | diffuse |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | L22H23 | 1024 | -0.030 [-0.033, -0.026] | -0.29 | -7.2 / -6.1, -7.9 | yes | -0.020 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | L25H11 | 1023 | -0.024 [-0.028, -0.021] | -0.23 | -5.8 / -5.7, -5.8 | yes | -0.027 | 0.00 / 0.00 | sink (pos0) |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | L18H18 | 1022 | -0.015 [-0.018, -0.013] | -0.15 | -3.7 / -3.3, -4.0 | yes | -0.003 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | L46H24 | 1 | +0.146 [+0.125, +0.167] | +0.87 | +18.8 / +18.6, +18.3 | yes | +0.169 | 0.72 / 0.51 | other context tokens |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | L41H27 | 2 | +0.099 [+0.086, +0.113] | +0.59 | +12.8 / +11.9, +13.1 | yes | +0.057 | 0.62 / 0.09 | diffuse |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | L38H18 | 3 | +0.062 [+0.053, +0.073] | +0.37 | +8.1 / +8.5, +7.4 | yes | +0.038 | 0.34 / 0.10 | sink (pos0) |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | L38H21 | 4 | +0.041 [+0.034, +0.049] | +0.25 | +5.4 / +5.4, +5.2 | yes | +0.030 | 0.19 / 0.01 | sink (pos0) |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | L29H0 | 5 | +0.032 [+0.021, +0.044] | +0.19 | +4.2 / +4.4, +4.0 | yes | +0.001 | 0.21 / 0.06 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | L40H2 | 6 | +0.031 [+0.025, +0.037] | +0.19 | +4.1 / +4.5, +3.5 | yes | +0.010 | 0.09 / 0.00 | sink (pos0) |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | L42H30 | 7 | +0.024 [+0.019, +0.031] | +0.15 | +3.2 / +3.4, +2.9 | yes | +0.012 | 0.05 / 0.00 | diffuse |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | L38H20 | 8 | +0.024 [+0.019, +0.029] | +0.14 | +3.1 / +2.7, +3.4 | yes | +0.010 | 0.04 / 0.00 | other context tokens |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | L46H25 | 1536 | -0.134 [-0.153, -0.116] | -0.80 | -17.2 / -16.8, -16.8 | yes | -0.159 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | L38H16 | 1535 | -0.104 [-0.120, -0.089] | -0.62 | -13.3 / -12.6, -13.4 | yes | -0.088 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | L41H24 | 1534 | -0.047 [-0.052, -0.043] | -0.28 | -6.0 / -6.6, -5.3 | yes | -0.028 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | L22H20 | 1 | +0.074 [+0.065, +0.084] | +0.53 | +14.9 / +15.1, +14.5 | yes | +0.050 | 0.75 / 0.20 | STR site (filled entity) |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | L25H9 | 2 | +0.074 [+0.067, +0.081] | +0.53 | +14.9 / +14.4, +15.2 | yes | +0.106 | 0.79 / 0.29 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | L19H13 | 3 | +0.050 [+0.043, +0.056] | +0.35 | +9.9 / +9.1, +10.6 | yes | +0.001 | 0.57 / 0.19 | STR site (filled entity) |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | L13H18 | 4 | +0.031 [+0.025, +0.037] | +0.22 | +6.1 / +6.1, +6.0 | yes | -0.000 | 0.25 / 0.06 | STR site (filled entity) |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | L31H22 | 5 | +0.029 [+0.026, +0.032] | +0.21 | +5.7 / +5.6, +5.7 | yes | +0.030 | 0.17 / 0.00 | diffuse |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | L19H12 | 6 | +0.027 [+0.022, +0.033] | +0.19 | +5.4 / +5.9, +4.8 | yes | +0.011 | 0.24 / 0.02 | STR site (filled entity) |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | L13H11 | 7 | +0.026 [+0.019, +0.035] | +0.19 | +5.2 / +5.8, +4.5 | yes | +0.001 | 0.14 / 0.04 | STR site (filled entity) |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | L16H10 | 8 | +0.023 [+0.018, +0.029] | +0.16 | +4.5 / +4.7, +4.3 | yes | +0.006 | 0.13 / 0.01 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | L22H23 | 1024 | -0.037 [-0.041, -0.033] | -0.26 | -7.6 / -7.4, -7.7 | yes | -0.029 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | L25H11 | 1023 | -0.020 [-0.022, -0.018] | -0.14 | -4.2 / -4.2, -4.2 | yes | -0.027 | 0.00 / 0.00 | diffuse |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | L31H20 | 1022 | -0.015 [-0.017, -0.013] | -0.11 | -3.2 / -3.2, -3.0 | yes | -0.015 | 0.00 / 0.00 | diffuse |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | L41H0 | 1 | +0.077 [+0.034, +0.116] | +0.43 | +11.4 / +7.7, +13.8 | yes | +0.023 | 0.36 / 0.18 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | L42H30 | 2 | +0.076 [+0.052, +0.099] | +0.42 | +11.2 / +9.9, +11.2 | yes | +0.062 | 0.46 / 0.05 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | L41H5 | 3 | +0.048 [+0.009, +0.111] | +0.27 | +7.1 / +9.7, +3.7 | yes | +0.035 | 0.18 / 0.10 | diffuse |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | L38H3 | 4 | +0.043 [+0.030, +0.057] | +0.24 | +6.4 / +6.7, +5.3 | yes | +0.022 | 0.13 / 0.01 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | L31H16 | 5 | +0.042 [+0.019, +0.068] | +0.23 | +6.2 / +5.4, +6.3 | yes | +0.016 | 0.29 / 0.16 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | L38H19 | 6 | +0.039 [+0.017, +0.069] | +0.21 | +5.7 / +7.7, +3.0 | yes | +0.009 | 0.11 / 0.06 | diffuse |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | L38H18 | 7 | +0.037 [+0.021, +0.056] | +0.21 | +5.5 / +6.5, +3.8 | yes | +0.017 | 0.11 / 0.00 | diffuse |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | L40H2 | 8 | +0.036 [+0.018, +0.062] | +0.20 | +5.3 / +5.9, +4.1 | yes | +0.014 | 0.15 / 0.03 | sink (pos0) |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | L42H26 | 1536 | -0.080 [-0.101, -0.057] | -0.44 | -11.5 / -9.2, -12.5 | yes | -0.068 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | L38H16 | 1535 | -0.060 [-0.083, -0.040] | -0.33 | -8.7 / -8.5, -7.9 | yes | -0.023 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | L36H18 | 1534 | -0.041 [-0.062, -0.022] | -0.23 | -5.8 / -5.1, -6.0 | yes | -0.008 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), copy prefix, templates orig | L13H18 | 1 | +0.077 [+0.031, +0.136] | +0.41 | +12.9 / +6.6, +15.9 | yes | +0.007 | 0.37 / 0.21 | STR site (filled entity) |
| Mixtral-8x7B (BOS), copy prefix, templates orig | L8H8 | 2 | +0.056 [+0.011, +0.119] | +0.29 | +9.3 / +11.3, +5.6 | yes | +0.000 | 0.25 / 0.11 | STR site (filled entity) |
| Mixtral-8x7B (BOS), copy prefix, templates orig | L22H20 | 3 | +0.048 [+0.014, +0.083] | +0.25 | +8.0 / +9.9, +4.6 | yes | +0.024 | 0.31 / 0.10 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates orig | L19H13 | 4 | +0.044 [+0.005, +0.097] | +0.23 | +7.3 / +1.2, +11.3 | no | +0.015 | 0.21 / 0.11 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates orig | L19H12 | 5 | +0.037 [+0.009, +0.080] | +0.19 | +6.1 / +10.7, +0.7 | no | +0.025 | 0.30 / 0.14 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates orig | L19H1 | 6 | +0.035 [+0.019, +0.053] | +0.19 | +5.9 / +5.6, +4.9 | yes | +0.003 | 0.19 / 0.02 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates orig | L18H10 | 7 | +0.031 [+0.014, +0.057] | +0.17 | +5.2 / +7.9, +1.7 | no | +0.007 | 0.24 / 0.03 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates orig | L27H19 | 8 | +0.029 [-0.003, +0.087] | +0.15 | +4.8 / -0.2, +8.3 | no | +0.026 | 0.11 / 0.04 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates orig | L22H23 | 1024 | -0.038 [-0.069, -0.008] | -0.20 | -6.5 / -5.8, -5.7 | yes | -0.025 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), copy prefix, templates orig | L18H18 | 1023 | -0.026 [-0.049, -0.008] | -0.14 | -4.4 / -2.4, -5.4 | yes | -0.010 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), copy prefix, templates orig | L31H20 | 1022 | -0.023 [-0.038, -0.011] | -0.12 | -3.9 / -2.3, -4.6 | yes | -0.023 | 0.00 / 0.00 | diffuse |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | L41H0 | 1 | +0.104 [+0.032, +0.164] | +0.58 | +15.8 / +21.2, +5.6 | yes | +0.027 | 0.49 / 0.23 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | L42H30 | 2 | +0.063 [+0.033, +0.093] | +0.35 | +9.5 / +7.3, +9.7 | yes | +0.045 | 0.40 / 0.07 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | L31H16 | 3 | +0.054 [+0.018, +0.089] | +0.30 | +8.2 / +10.6, +3.4 | yes | +0.018 | 0.30 / 0.15 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | L42H27 | 4 | +0.042 [+0.008, +0.079] | +0.23 | +6.3 / +6.9, +4.1 | yes | +0.022 | 0.24 / 0.02 | sink (pos0) |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | L38H3 | 5 | +0.037 [+0.020, +0.056] | +0.21 | +5.6 / +5.4, +4.4 | yes | +0.018 | 0.09 / 0.00 | other context tokens |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | L40H2 | 6 | +0.036 [+0.011, +0.081] | +0.20 | +5.4 / +4.8, +4.7 | yes | +0.012 | 0.10 / 0.04 | sink (pos0) |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | L38H18 | 7 | +0.030 [+0.015, +0.054] | +0.17 | +4.5 / +2.0, +6.2 | yes | +0.015 | 0.07 / 0.00 | diffuse |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | L41H5 | 8 | +0.028 [+0.006, +0.068] | +0.16 | +4.2 / +2.4, +5.2 | yes | +0.020 | 0.13 / 0.04 | diffuse |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | L42H26 | 1536 | -0.067 [-0.100, -0.043] | -0.38 | -10.4 / -7.4, -11.2 | yes | -0.059 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | L38H16 | 1535 | -0.051 [-0.081, -0.031] | -0.29 | -7.9 / -6.4, -7.6 | yes | -0.027 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | L41H7 | 1534 | -0.042 [-0.090, +0.016] | -0.24 | -6.5 / -12.2, +1.7 | no | -0.009 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | L13H18 | 1 | +0.069 [+0.020, +0.175] | +0.37 | +11.0 / +7.0, +10.9 | yes | +0.021 | 0.39 / 0.14 | STR site (filled entity) |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | L19H13 | 2 | +0.065 [+0.008, +0.198] | +0.35 | +10.4 / +0.4, +13.4 | no | +0.038 | 0.32 / 0.22 | STR site (filled entity) |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | L18H17 | 3 | +0.041 [+0.012, +0.058] | +0.22 | +6.4 / +1.5, +7.6 | no | +0.009 | 0.35 / 0.04 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | L31H22 | 4 | +0.040 [+0.016, +0.064] | +0.22 | +6.4 / +2.0, +7.3 | yes | +0.034 | 0.27 / 0.08 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | L18H10 | 5 | +0.034 [+0.012, +0.082] | +0.19 | +5.4 / +10.9, +1.7 | no | +0.009 | 0.28 / 0.15 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | L20H27 | 6 | +0.031 [+0.007, +0.052] | +0.17 | +4.8 / +2.1, +5.3 | yes | +0.011 | 0.25 / 0.02 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | L19H12 | 7 | +0.027 [+0.005, +0.074] | +0.15 | +4.2 / +9.7, +0.7 | no | +0.012 | 0.28 / 0.02 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | L15H9 | 8 | +0.027 [+0.003, +0.053] | +0.15 | +4.2 / +0.1, +5.4 | no | +0.000 | 0.18 / 0.09 | other context tokens |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | L18H18 | 1024 | -0.048 [-0.074, -0.018] | -0.26 | -8.1 / -3.5, -8.9 | yes | -0.016 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | L31H20 | 1023 | -0.038 [-0.056, -0.018] | -0.20 | -6.4 / -3.0, -6.9 | yes | -0.038 | 0.00 / 0.00 | diffuse |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | L16H2 | 1022 | -0.025 [-0.048, -0.009] | -0.14 | -4.4 / -7.1, -2.3 | yes | -0.005 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | L42H30 | 1 | +0.047 [+0.030, +0.066] | +0.31 | +10.0 / +9.4, +8.8 | yes | +0.030 | 0.38 / 0.06 | diffuse |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | L38H3 | 2 | +0.046 [+0.020, +0.073] | +0.30 | +9.8 / +6.4, +11.1 | yes | +0.016 | 0.32 / 0.06 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | L41H27 | 3 | +0.043 [+0.008, +0.087] | +0.28 | +9.2 / +8.6, +7.9 | yes | +0.019 | 0.24 / 0.13 | sink (pos0) |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | L19H16 | 4 | +0.031 [+0.006, +0.065] | +0.20 | +6.6 / +8.3, +3.8 | yes | +0.004 | 0.14 / 0.09 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | L43H28 | 5 | +0.030 [+0.007, +0.066] | +0.19 | +6.4 / +1.5, +9.5 | no | +0.025 | 0.11 / 0.04 | self (final token) |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | L31H16 | 6 | +0.028 [+0.013, +0.047] | +0.18 | +6.0 / +6.6, +4.4 | yes | +0.008 | 0.13 / 0.06 | diffuse |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | L38H19 | 7 | +0.027 [+0.016, +0.041] | +0.17 | +5.8 / +7.7, +3.0 | yes | +0.008 | 0.19 / 0.02 | diffuse |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | L40H15 | 8 | +0.026 [+0.006, +0.054] | +0.17 | +5.7 / +2.9, +7.1 | yes | +0.012 | 0.12 / 0.04 | diffuse |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | L42H26 | 1536 | -0.056 [-0.072, -0.041] | -0.36 | -11.5 / -8.8, -11.8 | yes | -0.035 | 0.00 / 0.00 | sink (pos0) |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | L38H16 | 1535 | -0.055 [-0.091, -0.026] | -0.35 | -11.3 / -17.4, -3.8 | yes | -0.024 | 0.02 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | L38H0 | 1534 | -0.040 [-0.067, -0.011] | -0.26 | -8.3 / -7.0, -7.9 | yes | -0.019 | 0.01 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | L13H18 | 1 | +0.066 [+0.039, +0.102] | +0.43 | +12.6 / +14.0, +9.2 | yes | +0.007 | 0.42 / 0.14 | STR site (filled entity) |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | L8H8 | 2 | +0.063 [+0.030, +0.109] | +0.41 | +12.0 / +11.3, +11.8 | yes | +0.000 | 0.31 / 0.11 | STR site (filled entity) |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | L19H13 | 3 | +0.057 [+0.026, +0.090] | +0.37 | +10.8 / +11.8, +8.2 | yes | +0.012 | 0.39 / 0.22 | STR site (filled entity) |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | L19H12 | 4 | +0.045 [+0.011, +0.088] | +0.29 | +8.4 / +7.5, +8.9 | yes | +0.023 | 0.25 / 0.18 | diffuse |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | L19H1 | 5 | +0.034 [+0.024, +0.043] | +0.22 | +6.3 / +5.8, +6.4 | yes | +0.005 | 0.24 / 0.01 | STR site (filled entity) |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | L22H20 | 6 | +0.030 [+0.010, +0.056] | +0.19 | +5.6 / +3.0, +8.8 | yes | +0.013 | 0.19 / 0.07 | STR site (filled entity) |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | L15H8 | 7 | +0.024 [+0.012, +0.041] | +0.16 | +4.5 / +4.9, +3.4 | yes | +0.006 | 0.14 / 0.02 | other context tokens |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | L18H10 | 8 | +0.023 [+0.010, +0.038] | +0.15 | +4.2 / +3.9, +4.3 | yes | +0.006 | 0.16 / 0.00 | diffuse |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | L18H18 | 1024 | -0.023 [-0.041, -0.009] | -0.15 | -4.6 / -4.6, -4.0 | yes | -0.008 | 0.02 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | L22H23 | 1023 | -0.021 [-0.039, -0.007] | -0.13 | -4.1 / -3.2, -4.9 | yes | -0.010 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | L19H3 | 1022 | -0.019 [-0.027, -0.012] | -0.13 | -3.9 / -4.2, -3.0 | yes | -0.003 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | L38H3 | 1 | +0.058 [+0.027, +0.093] | +0.35 | +11.1 / +14.1, +4.6 | yes | +0.016 | 0.30 / 0.10 | other context tokens |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | L42H30 | 2 | +0.048 [+0.029, +0.070] | +0.29 | +9.1 / +7.7, +8.5 | yes | +0.028 | 0.34 / 0.08 | diffuse |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | L19H16 | 3 | +0.044 [-0.000, +0.124] | +0.26 | +8.2 / +1.8, +13.9 | no | +0.004 | 0.19 / 0.12 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | L41H0 | 4 | +0.037 [+0.009, +0.069] | +0.22 | +7.0 / +5.2, +7.3 | yes | +0.009 | 0.26 / 0.14 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | L43H28 | 5 | +0.035 [+0.006, +0.084] | +0.21 | +6.5 / +9.9, +0.9 | no | +0.030 | 0.10 / 0.05 | self (final token) |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | L31H16 | 6 | +0.031 [+0.014, +0.054] | +0.19 | +5.8 / +3.9, +6.7 | yes | +0.010 | 0.18 / 0.05 | STR site (filled entity) |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | L41H27 | 7 | +0.031 [+0.004, +0.068] | +0.19 | +5.7 / +0.2, +10.9 | no | +0.019 | 0.29 / 0.15 | mover (mention: other candidate) |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | L38H19 | 8 | +0.029 [+0.015, +0.046] | +0.18 | +5.5 / +2.4, +7.8 | yes | +0.009 | 0.18 / 0.01 | diffuse |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | L38H16 | 1536 | -0.071 [-0.114, -0.039] | -0.43 | -13.8 / -8.7, -16.3 | yes | -0.027 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | L42H26 | 1535 | -0.054 [-0.070, -0.037] | -0.32 | -10.5 / -10.3, -8.1 | yes | -0.033 | 0.00 / 0.00 | sink (pos0) |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | L38H0 | 1534 | -0.053 [-0.086, -0.027] | -0.32 | -10.4 / -12.4, -5.4 | yes | -0.015 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | L13H18 | 1 | +0.068 [+0.027, +0.118] | +0.39 | +12.5 / +9.0, +13.2 | yes | +0.007 | 0.36 / 0.16 | STR site (filled entity) |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | L19H13 | 2 | +0.060 [+0.021, +0.109] | +0.35 | +11.1 / +7.8, +12.0 | yes | +0.011 | 0.44 / 0.17 | STR site (filled entity) |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | L8H8 | 3 | +0.044 [+0.010, +0.103] | +0.26 | +8.1 / +12.8, +2.2 | yes | +0.001 | 0.17 / 0.06 | STR site (filled entity) |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | L19H12 | 4 | +0.034 [+0.005, +0.080] | +0.20 | +6.3 / +11.3, +0.5 | no | +0.017 | 0.24 / 0.17 | diffuse |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | L21H20 | 5 | +0.031 [+0.003, +0.087] | +0.18 | +5.6 / +9.4, +1.0 | no | +0.022 | 0.10 / 0.09 | other context tokens |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | L15H8 | 6 | +0.028 [+0.011, +0.053] | +0.17 | +5.2 / +7.6, +2.0 | no | +0.006 | 0.17 / 0.01 | other context tokens |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | L19H1 | 7 | +0.028 [+0.014, +0.042] | +0.16 | +5.1 / +3.3, +5.8 | yes | +0.005 | 0.16 / 0.05 | STR site (filled entity) |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | L13H16 | 8 | +0.027 [+0.005, +0.051] | +0.16 | +4.9 / +1.3, +7.3 | no | +0.003 | 0.21 / 0.04 | diffuse |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | L18H18 | 1024 | -0.040 [-0.063, -0.020] | -0.23 | -7.4 / -5.6, -7.7 | yes | -0.009 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | L21H21 | 1023 | -0.026 [-0.060, -0.004] | -0.15 | -4.8 / -6.9, -1.9 | no | -0.015 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | L19H3 | 1022 | -0.018 [-0.026, -0.010] | -0.11 | -3.4 / -1.9, -4.3 | no | -0.003 | 0.00 / 0.00 | self (final token) |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | L40H2 | 1 | +0.315 [+0.264, +0.337] | +1.15 | +13.0 / +13.3, +12.8 | yes | +0.001 | 1.00 / 0.66 | self (final token) |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | L42H10 | 2 | +0.303 [+0.174, +0.312] | +1.11 | +12.5 / +9.1, +12.6 | yes | +0.482 | 0.95 / 0.30 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | L36H23 | 3 | +0.191 [+0.165, +0.254] | +0.70 | +7.8 / +8.4, +7.7 | yes | +0.000 | 0.81 / 0.01 | STR site (trigger) |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | L43H24 | 4 | +0.148 [+0.094, +0.152] | +0.54 | +6.1 / +4.2, +6.1 | yes | +0.128 | 0.26 / 0.00 | sink (pos0) |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | L36H28 | 5 | +0.145 [+0.050, +0.153] | +0.53 | +6.0 / +3.1, +6.1 | yes | +0.001 | 0.35 / 0.00 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | L39H27 | 6 | +0.138 [+0.078, +0.194] | +0.51 | +5.7 / +5.4, +5.6 | yes | +0.005 | 0.24 / 0.00 | STR site (trigger) |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | L39H29 | 7 | +0.137 [+0.132, +0.234] | +0.50 | +5.6 / +7.5, +5.4 | yes | +0.004 | 0.24 / 0.01 | diffuse |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | L33H20 | 8 | +0.136 [+0.009, +0.148] | +0.50 | +5.6 / +0.7, +5.9 | no | +0.001 | 0.31 / 0.00 | STR site (trigger) |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | L44H15 | 1536 | -0.332 [-0.336, -0.291] | -1.21 | -13.8 / -13.4, -13.6 | yes | -0.167 | 0.00 / 0.00 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | L40H30 | 1535 | -0.179 [-0.188, -0.064] | -0.66 | -7.5 / -4.6, -7.6 | yes | -0.004 | 0.00 / 0.00 | diffuse |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | L36H31 | 1534 | -0.171 [-0.177, -0.109] | -0.63 | -7.2 / -4.6, -7.3 | yes | -0.001 | 0.00 / 0.00 | STR site (trigger) |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | L40H2 | 1 | +0.243 [+0.208, +0.278] | +2.03 | +20.5 / +21.0, +17.9 | yes | +0.010 | 0.88 / 0.53 | other context tokens |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | L42H10 | 2 | +0.119 [+0.095, +0.143] | +1.00 | +10.0 / +8.3, +10.9 | yes | +0.199 | 0.44 / 0.05 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | L41H18 | 3 | +0.073 [+0.059, +0.089] | +0.62 | +6.2 / +5.4, +6.4 | yes | +0.017 | 0.16 / 0.00 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | L43H28 | 4 | +0.068 [+0.053, +0.084] | +0.57 | +5.7 / +5.9, +5.0 | yes | +0.058 | 0.25 / 0.05 | other context tokens |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | L41H20 | 5 | +0.061 [+0.042, +0.083] | +0.51 | +5.1 / +3.8, +6.1 | yes | +0.020 | 0.19 / 0.02 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | L42H12 | 6 | +0.059 [+0.043, +0.076] | +0.49 | +5.0 / +5.5, +3.9 | yes | +0.051 | 0.11 / 0.00 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | L39H29 | 7 | +0.057 [+0.040, +0.077] | +0.48 | +4.8 / +3.7, +5.6 | yes | +0.007 | 0.11 / 0.02 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | L43H24 | 8 | +0.056 [+0.042, +0.073] | +0.47 | +4.7 / +3.6, +5.4 | yes | +0.039 | 0.03 / 0.00 | sink (pos0) |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | L44H15 | 1536 | -0.118 [-0.155, -0.088] | -0.99 | -10.0 / -8.7, -10.5 | yes | -0.053 | 0.00 / 0.00 | mover (mention: answer) |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | L40H7 | 1535 | -0.080 [-0.094, -0.067] | -0.67 | -6.8 / -6.6, -6.4 | yes | -0.004 | 0.00 / 0.00 | sink (pos0) |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | L47H29 | 1534 | -0.079 [-0.095, -0.065] | -0.66 | -6.7 / -7.5, -5.3 | yes | -0.087 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), F2E natural, blank at end | L20H21 | 1 | +0.114 [+0.084, +0.144] | +0.79 | +11.7 / +11.2, +11.7 | yes | +0.015 | 0.67 / 0.19 | other context tokens |
| Mixtral-8x7B (BOS), F2E natural, blank at end | L18H10 | 2 | +0.103 [+0.085, +0.121] | +0.71 | +10.6 / +10.4, +10.0 | yes | +0.001 | 0.72 / 0.19 | other context tokens |
| Mixtral-8x7B (BOS), F2E natural, blank at end | L19H8 | 3 | +0.099 [+0.064, +0.146] | +0.69 | +10.2 / +11.3, +8.0 | yes | +0.045 | 0.47 / 0.14 | mover (mention: answer) |
| Mixtral-8x7B (BOS), F2E natural, blank at end | L16H0 | 4 | +0.095 [+0.060, +0.138] | +0.66 | +9.8 / +7.4, +12.3 | yes | +0.003 | 0.56 / 0.22 | other context tokens |
| Mixtral-8x7B (BOS), F2E natural, blank at end | L16H3 | 5 | +0.056 [+0.025, +0.092] | +0.39 | +5.6 / +5.1, +5.9 | yes | +0.005 | 0.28 / 0.11 | mover (mention: other candidate) |
| Mixtral-8x7B (BOS), F2E natural, blank at end | L17H21 | 6 | +0.042 [+0.031, +0.054] | +0.29 | +4.2 / +3.6, +4.6 | yes | +0.001 | 0.19 / 0.00 | other context tokens |
| Mixtral-8x7B (BOS), F2E natural, blank at end | L31H22 | 7 | +0.041 [+0.031, +0.051] | +0.28 | +4.1 / +3.9, +4.0 | yes | +0.041 | 0.06 / 0.00 | diffuse |
| Mixtral-8x7B (BOS), F2E natural, blank at end | L19H9 | 8 | +0.040 [+0.024, +0.058] | +0.27 | +4.0 / +4.1, +3.4 | yes | +0.028 | 0.06 / 0.00 | mover (mention: answer) |
| Mixtral-8x7B (BOS), F2E natural, blank at end | L20H22 | 1024 | -0.059 [-0.071, -0.048] | -0.41 | -6.4 / -5.8, -6.6 | yes | -0.022 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), F2E natural, blank at end | L21H21 | 1023 | -0.044 [-0.063, -0.026] | -0.31 | -4.8 / -4.7, -4.6 | yes | -0.051 | 0.00 / 0.00 | self (final token) |
| Mixtral-8x7B (BOS), F2E natural, blank at end | L17H24 | 1022 | -0.043 [-0.060, -0.031] | -0.30 | -4.7 / -4.1, -5.2 | yes | -0.002 | 0.00 / 0.00 | mover (mention: answer) |
