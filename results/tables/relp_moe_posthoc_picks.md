| model/task | paired difference | RelP top-1 (layer, position, expert) | exact rescue disc / test | Lu rule (layer, expert) | exact rescue disc / test |
|---|---|---|---|---|---|
| qwen3/paper/simple | 10.48 | L46 p1 E30 | 1.10 (11%) / 1.24 (12%) | L0 E91 | 0.02 (0%) / 0.00 (0%) |
| qwen3/paper/nounpp | 7.73 | L41 p4 E36 | 2.40 (31%) / 2.38 (31%) | L41 E36 | 2.40 (31%) / 2.38 (31%) |
| qwen3/paper/rc | 7.01 | L41 p5 E36 | 2.16 (31%) / 1.89 (27%) | L41 E36 | 2.16 (31%) / 1.89 (27%) |
| qwen3/paper/within_rc | 8.49 | L46 p4 E30 | 0.84 (10%) / 0.84 (10%) | L0 E64 | 0.01 (0%) / -0.02 (-0%) |
| mixtral/paper/simple | 10.06 | L31 p2 E0 | 1.73 (17%) / 1.86 (18%) | L0 E2 | 9.88 (98%) / 10.51 (104%) |
| mixtral/paper/nounpp | 7.77 | L30 p5 E2 | 1.10 (14%) / 1.02 (13%) | L31 E0 | 0.84 (11%) / 0.77 (10%) |
| mixtral/paper/rc | 6.28 | L30 p6 E2 | 0.62 (10%) / 0.67 (11%) | L30 E2 | 0.62 (10%) / 0.67 (11%) |
| mixtral/paper/within_rc | 7.85 | L30 p5 E2 | 1.03 (13%) / 1.07 (14%) | L0 E2 | 7.47 (95%) / 7.49 (95%) |
