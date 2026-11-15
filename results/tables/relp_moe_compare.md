| model/profile/task | Lu: L*, e* (val rescue, Spec, test rescue) | joint best (final pos) | RelP same rule | RelP top-1 (own rule) | exact top-1 unit | agree top-1 | Pearson | rank corr (effect>=0.05) | slope exact~RelP | bootstrap top-1 exact / RelP |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3/paper/simple | L0 E91 (0.04, 0.02, 0.00) | L46E30 | L47E111 | L46 p1 E30 | L46 p1 E30 | True | 0.961 | 0.950 | 0.95 | L46E30 1.00 / L46E30 1.00 |
| qwen3/paper/nounpp | L41 E36 (2.69, 2.65, 2.38) | L41E36 | L41E36 | L41 p4 E36 | L41 p4 E36 | True | 0.968 | 0.971 | 1.45 | L41E36 1.00 / L41E36 1.00 |
| qwen3/paper/rc | L41 E36 (2.33, 2.27, 1.89) | L41E36 | L41E36 | L41 p5 E36 | L41 p5 E36 | True | 0.977 | 0.930 | 1.29 | L41E36 1.00 / L41E36 1.00 |
| qwen3/paper/within_rc | L0 E64 (-0.01, -0.02, -0.02) | L46E30 | L41E46 | L46 p4 E30 | L46 p4 E30 | True | 0.994 | 0.970 | 1.03 | L46E30 0.91 / L46E30 0.83 |
| qwen3/grouped/simple | L0 E105 (0.07, 0.03, 0.03) | L46E30 | L47E52 | L46 p1 E30 | L46 p1 E30 | True | 0.940 | 0.857 | 0.95 | L46E30 1.00 / L46E30 1.00 |
| qwen3/grouped/nounpp | L41 E36 (2.63, 2.59, 2.48) | L41E36 | L41E36 | L41 p4 E36 | L41 p4 E36 | True | 0.975 | 0.945 | 1.46 | L41E36 1.00 / L41E36 1.00 |
| qwen3/grouped/rc | L41 E36 (2.44, 2.39, 2.16) | L41E36 | L41E36 | L41 p5 E36 | L41 p5 E36 | True | 0.985 | 0.904 | 1.32 | L41E36 1.00 / L41E36 1.00 |
| qwen3/grouped/within_rc | L0 E28 (0.01, -0.02, -0.02) | L46E30 | L41E46 | L41 p4 E46 | L46 p4 E30 | True | 0.992 | 0.951 | 1.05 | L46E30 0.71 / L46E30 0.58 |
| mixtral/paper/simple | L0 E2 (9.93, 9.89, 10.51) | L0E2 | L31E0 | L31 p2 E0 | L0 p2 E2 | False | 0.294 | 0.791 | 1.15 | L0E2 1.00 / L31E0 0.72 |
| mixtral/paper/nounpp | L31 E0 (0.89, 0.22, 0.72) | L30E2 | L31E0 | L30 p5 E2 | L0 p2 E2 | False | 0.299 | 0.940 | 1.45 | L30E2 1.00 / L30E2 1.00 |
| mixtral/paper/rc | L30 E2 (0.50, 0.18, 0.67) | L30E2 | L30E2 | L30 p6 E2 | L0 p2 E2 | False | 0.305 | 0.928 | 1.74 | L30E2 1.00 / L30E2 1.00 |
| mixtral/paper/within_rc | L0 E2 (7.33, 7.31, 7.49) | L0E2 | L30E2 | L30 p5 E2 | L0 p5 E2 | False | 0.268 | 0.765 | 1.10 | L0E2 1.00 / L30E2 1.00 |
| mixtral/grouped/simple | L0 E2 (10.21, 10.19, 9.49) | L0E2 | L31E0 | L31 p2 E0 | L0 p2 E2 | False | 0.309 | 0.787 | 1.13 | L0E2 1.00 / L31E0 1.00 |
| mixtral/grouped/nounpp | L31 E0 (0.75, 0.22, 0.69) | L30E2 | L31E0 | L30 p5 E2 | L30 p5 E2 | True | 0.998 | 0.983 | 1.07 | L30E2 1.00 / L30E2 1.00 |
| mixtral/grouped/rc | L30 E2 (0.46, 0.10, 0.59) | L30E2 | L30E2 | L30 p6 E2 | L30 p6 E2 | True | 0.988 | 0.920 | 1.01 | L30E2 1.00 / L30E2 1.00 |
| mixtral/grouped/within_rc | L0 E2 (7.50, 7.50, 7.33) | L0E2 | L30E2 | L30 p5 E2 | L0 p5 E2 | False | 0.266 | 0.775 | 1.09 | L0E2 1.00 / L30E2 1.00 |
