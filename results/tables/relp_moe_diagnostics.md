| model/profile/task | unit | natural (exact) | frozen routing | frozen attention | both frozen | direct | RelP | verdict |
|---|---|---|---|---|---|---|---|---|
| qwen3/paper/simple | expert L0 p1 E91 | -0.03 | 0.00 | -0.01 | -0.00 | 0.01 | 0.04 | agree |
| qwen3/paper/simple | expert L37 p1 E104 | 0.02 | 0.03 | -0.00 | 0.03 | -0.04 | -0.06 | agree |
| qwen3/paper/simple | expert L38 p1 E16 | -0.01 | 0.01 | 0.00 | 0.01 | 0.14 | 0.08 | agree |
| qwen3/paper/simple | expert L41 p1 E36 | 0.57 | 0.58 | 0.56 | 0.58 | 1.22 | 0.71 | agree |
| qwen3/paper/simple | expert L41 p1 E46 | 0.31 | 0.33 | 0.31 | 0.32 | 0.82 | 0.48 | nonlinear (frozen) |
| qwen3/paper/simple | expert L43 p1 E56 | 0.25 | 0.25 | 0.26 | 0.25 | 0.35 | 0.35 | agree |
| qwen3/paper/simple | expert L46 p1 E30 | 1.11 | 1.11 | 1.11 | 1.12 | 0.94 | 1.04 | agree |
| qwen3/paper/simple | position L0 p1 | 5.84 | 0.56 | 5.89 | 0.55 | 0.03 | 0.16 | routing-mediated |
| qwen3/paper/simple | position L2 p1 | 1.26 | 0.26 | 1.20 | 0.28 | 0.06 | 0.18 | routing-mediated |
| qwen3/paper/simple | position L41 p1 | 1.40 | 1.42 | 1.39 | 1.41 | 2.03 | 1.18 | agree |
| qwen3/paper/nounpp | expert L2 p1 E115 | 0.05 | 0.05 | 0.09 | 0.06 |  | 0.07 | agree |
| qwen3/paper/nounpp | expert L35 p2 E79 | 0.08 | 0.07 | 0.08 | 0.07 |  | 0.06 | agree |
| qwen3/paper/nounpp | expert L41 p4 E36 | 2.42 | 2.39 | 2.42 | 2.39 | 1.61 | 1.41 | nonlinear (frozen) |
| qwen3/paper/nounpp | expert L42 p4 E25 | 0.05 | 0.06 | 0.05 | 0.05 | 0.08 | 0.07 | agree |
| qwen3/paper/nounpp | expert L47 p4 E15 | 0.19 | 0.19 | 0.19 | 0.19 | 0.17 | 0.17 | agree |
| qwen3/paper/nounpp | position L0 p1 | 4.54 | 0.72 | 4.55 | 0.76 |  | 0.14 | routing-mediated |
| qwen3/paper/nounpp | position L2 p1 | 0.64 | 0.29 | 0.71 | 0.35 |  | 0.13 | routing-mediated |
| qwen3/paper/nounpp | position L41 p4 | 3.36 | 3.33 | 3.38 | 3.34 | 1.96 | 1.73 | nonlinear (frozen) |
| qwen3/paper/rc | expert L2 p1 E115 | 0.08 | 0.06 | 0.08 | 0.06 |  | 0.08 | agree |
| qwen3/paper/rc | expert L41 p5 E36 | 2.16 | 2.14 | 2.18 | 2.15 | 1.34 | 1.48 | nonlinear (frozen) |
| qwen3/paper/rc | expert L41 p5 E46 | 0.39 | 0.38 | 0.40 | 0.39 | 0.32 | 0.35 | agree |
| qwen3/paper/rc | expert L42 p5 E77 | 0.54 | 0.53 | 0.54 | 0.53 | 0.46 | 0.49 | agree |
| qwen3/paper/rc | expert L44 p5 E88 | 0.33 | 0.33 | 0.33 | 0.33 | 0.28 | 0.30 | agree |
| qwen3/paper/rc | position L0 p1 | 4.12 | 0.65 | 3.91 | 0.63 |  | 0.14 | routing-mediated |
| qwen3/paper/rc | position L2 p1 | 0.60 | 0.27 | 0.59 | 0.27 |  | 0.15 | routing-mediated |
| qwen3/paper/rc | position L41 p5 | 3.03 | 2.98 | 3.05 | 3.00 | 1.69 | 1.84 | nonlinear (frozen) |
| qwen3/paper/within_rc | expert L0 p4 E64 | -0.01 | -0.00 | -0.01 | -0.00 | 0.00 | 0.01 | agree |
| qwen3/paper/within_rc | expert L37 p4 E36 | 0.30 | 0.30 | 0.31 | 0.30 | 0.21 | 0.25 | agree |
| qwen3/paper/within_rc | expert L42 p4 E74 | -0.02 | -0.02 | -0.02 | -0.02 | -0.06 | -0.06 | agree |
| qwen3/paper/within_rc | expert L44 p4 E88 | 0.56 | 0.56 | 0.56 | 0.56 | 0.39 | 0.46 | agree |
| qwen3/paper/within_rc | expert L45 p4 E17 | 0.75 | 0.75 | 0.75 | 0.75 | 0.63 | 0.71 | agree |
| qwen3/paper/within_rc | expert L46 p4 E30 | 0.83 | 0.84 | 0.84 | 0.84 | 0.72 | 0.81 | agree |
| qwen3/paper/within_rc | expert L47 p4 E15 | 0.11 | 0.11 | 0.11 | 0.11 | 0.08 | 0.08 | agree |
| qwen3/paper/within_rc | position L0 p4 | 5.47 | 0.49 | 5.95 | 0.49 | 0.02 | 0.12 | routing-mediated |
| qwen3/paper/within_rc | position L37 p4 | 0.62 | 0.59 | 0.61 | 0.58 | 0.28 | 0.38 | nonlinear (frozen) |
| qwen3/paper/within_rc | position L41 p4 | 2.21 | 2.19 | 2.22 | 2.19 | 1.96 | 1.66 | nonlinear (frozen) |
| qwen3/grouped/simple | expert L0 p1 E105 | 0.03 | 0.00 | 0.05 | 0.01 | 0.00 | 0.02 | agree |
| qwen3/grouped/simple | expert L38 p1 E16 | -0.03 | -0.01 | -0.01 | 0.00 | 0.15 | 0.07 | nonlinear (frozen) |
| qwen3/grouped/simple | expert L41 p1 E36 | 0.51 | 0.50 | 0.50 | 0.50 | 1.19 | 0.68 | nonlinear (frozen) |
| qwen3/grouped/simple | expert L41 p1 E46 | 0.35 | 0.35 | 0.34 | 0.34 | 0.99 | 0.55 | nonlinear (frozen) |
| qwen3/grouped/simple | expert L43 p1 E56 | 0.24 | 0.24 | 0.22 | 0.23 | 0.38 | 0.35 | nonlinear (frozen) |
| qwen3/grouped/simple | expert L46 p1 E30 | 1.34 | 1.35 | 1.34 | 1.35 | 1.10 | 1.23 | agree |
| qwen3/grouped/simple | position L0 p1 | 6.63 | 0.39 | 6.80 | 0.40 | 0.03 | 0.15 | routing-mediated |
| qwen3/grouped/simple | position L2 p1 | 1.22 | 0.23 | 1.37 | 0.24 | 0.06 | 0.17 | routing-mediated |
| qwen3/grouped/simple | position L42 p1 | 0.34 | 0.26 | 0.35 | 0.28 | 0.06 | 0.10 | nonlinear (frozen) |
| qwen3/grouped/nounpp | expert L41 p4 E36 | 2.44 | 2.42 | 2.47 | 2.45 | 1.54 | 1.45 | nonlinear (frozen) |
| qwen3/grouped/nounpp | expert L42 p4 E25 | 0.07 | 0.07 | 0.06 | 0.07 | 0.09 | 0.08 | agree |
| qwen3/grouped/nounpp | expert L47 p4 E15 | 0.19 | 0.19 | 0.19 | 0.19 | 0.17 | 0.17 | agree |
| qwen3/grouped/nounpp | expert L47 p4 E95 | 0.06 | 0.06 | 0.06 | 0.06 | 0.08 | 0.08 | agree |
| qwen3/grouped/nounpp | expert L47 p4 E127 | 0.10 | 0.10 | 0.10 | 0.10 | 0.12 | 0.12 | agree |
| qwen3/grouped/nounpp | position L0 p1 | 4.61 | 0.57 | 4.69 | 0.59 |  | 0.14 | routing-mediated |
| qwen3/grouped/nounpp | position L2 p1 | 0.75 | 0.30 | 0.84 | 0.35 |  | 0.13 | routing-mediated |
| qwen3/grouped/nounpp | position L41 p4 | 3.30 | 3.27 | 3.32 | 3.29 | 1.86 | 1.77 | nonlinear (frozen) |
| qwen3/grouped/rc | expert L35 p5 E79 | 0.07 | 0.06 | 0.08 | 0.06 | 0.02 | 0.04 | agree |
| qwen3/grouped/rc | expert L41 p5 E36 | 2.19 | 2.14 | 2.21 | 2.16 | 1.32 | 1.47 | nonlinear (frozen) |
| qwen3/grouped/rc | expert L41 p5 E46 | 0.37 | 0.37 | 0.38 | 0.38 | 0.30 | 0.33 | agree |
| qwen3/grouped/rc | expert L42 p5 E77 | 0.59 | 0.58 | 0.59 | 0.58 | 0.50 | 0.54 | agree |
| qwen3/grouped/rc | expert L44 p5 E88 | 0.34 | 0.34 | 0.34 | 0.34 | 0.29 | 0.31 | agree |
| qwen3/grouped/rc | position L0 p1 | 3.99 | 0.59 | 3.85 | 0.56 |  | 0.14 | routing-mediated |
| qwen3/grouped/rc | position L2 p1 | 0.57 | 0.27 | 0.55 | 0.26 |  | 0.14 | routing-mediated |
| qwen3/grouped/rc | position L41 p5 | 3.02 | 2.96 | 3.05 | 3.00 | 1.66 | 1.82 | nonlinear (frozen) |
| qwen3/grouped/within_rc | expert L0 p4 E28 | 0.03 | 0.01 | -0.01 | 0.01 | 0.00 | 0.02 | agree |
| qwen3/grouped/within_rc | expert L37 p4 E36 | 0.31 | 0.31 | 0.32 | 0.31 | 0.23 | 0.27 | agree |
| qwen3/grouped/within_rc | expert L42 p4 E25 | 0.49 | 0.49 | 0.49 | 0.49 | 0.38 | 0.43 | agree |
| qwen3/grouped/within_rc | expert L43 p4 E19 | 0.23 | 0.23 | 0.23 | 0.23 | 0.16 | 0.18 | agree |
| qwen3/grouped/within_rc | expert L44 p4 E88 | 0.57 | 0.56 | 0.57 | 0.56 | 0.38 | 0.47 | agree |
| qwen3/grouped/within_rc | expert L45 p4 E17 | 0.76 | 0.76 | 0.77 | 0.77 | 0.62 | 0.71 | agree |
| qwen3/grouped/within_rc | expert L46 p4 E30 | 0.85 | 0.85 | 0.85 | 0.85 | 0.72 | 0.82 | agree |
| qwen3/grouped/within_rc | position L0 p4 | 5.20 | 0.35 | 5.49 | 0.34 | 0.02 | 0.12 | routing-mediated |
| qwen3/grouped/within_rc | position L37 p4 | 0.62 | 0.60 | 0.61 | 0.58 | 0.31 | 0.38 | nonlinear (frozen) |
| qwen3/grouped/within_rc | position L41 p4 | 2.36 | 2.33 | 2.37 | 2.34 | 2.00 | 1.72 | nonlinear (frozen) |
| mixtral/paper/simple | expert L0 p2 E2 | 9.89 | 9.71 | 9.73 | 9.13 | 0.00 | 0.13 | nonlinear (frozen) |
| mixtral/paper/simple | expert L27 p2 E2 | 0.76 | 0.74 | 0.76 | 0.74 | 0.65 | 0.66 | agree |
| mixtral/paper/simple | expert L30 p2 E2 | 1.81 | 1.80 | 1.82 | 1.81 | 1.62 | 1.66 | agree |
| mixtral/paper/simple | expert L31 p2 E0 | 1.74 | 1.74 | 1.74 | 1.74 | 1.72 | 1.72 | agree |
| mixtral/paper/simple | expert L31 p2 E1 | 0.68 | 0.68 | 0.68 | 0.68 | 0.47 | 0.47 | nonlinear (frozen) |
| mixtral/paper/simple | expert L31 p2 E6 | 0.44 | 0.44 | 0.44 | 0.44 | 0.32 | 0.32 | nonlinear (frozen) |
| mixtral/paper/simple | position L0 p2 | 9.94 | 9.81 | 9.83 | 9.71 | 0.00 | 0.15 | nonlinear (frozen) |
| mixtral/paper/simple | position L4 p2 | 0.43 | 0.33 | 0.40 | 0.32 | 0.01 | 0.13 | nonlinear (frozen) |
| mixtral/paper/simple | position L19 p2 | 0.83 | 0.79 | 0.83 | 0.79 | 0.48 | 0.62 | nonlinear (frozen) |
| mixtral/paper/nounpp | expert L0 p2 E2 | 7.60 | 7.51 | 7.30 | 6.78 |  | 0.14 | nonlinear (frozen) |
| mixtral/paper/nounpp | expert L2 p2 E5 | 0.04 | 0.02 | 0.05 | 0.04 |  | 0.06 | agree |
| mixtral/paper/nounpp | expert L29 p5 E1 | 0.55 | 0.56 | 0.55 | 0.56 | 0.38 | 0.46 | agree |
| mixtral/paper/nounpp | expert L30 p5 E2 | 1.10 | 1.10 | 1.10 | 1.10 | 0.97 | 1.04 | agree |
| mixtral/paper/nounpp | expert L31 p5 E0 | 0.84 | 0.84 | 0.84 | 0.84 | 0.83 | 0.83 | agree |
| mixtral/paper/nounpp | expert L31 p5 E1 | 0.58 | 0.58 | 0.58 | 0.58 | 0.47 | 0.47 | agree |
| mixtral/paper/nounpp | position L0 p2 | 7.68 | 7.64 | 7.47 | 7.40 |  | 0.16 | nonlinear (frozen) |
| mixtral/paper/nounpp | position L29 p5 | 0.70 | 0.70 | 0.70 | 0.70 | 0.48 | 0.58 | agree |
| mixtral/paper/nounpp | position L30 p5 | 1.34 | 1.34 | 1.33 | 1.33 | 1.15 | 1.24 | agree |
| mixtral/paper/rc | expert L0 p2 E2 | 5.80 | 5.65 | 5.44 | 5.19 |  | 0.14 | nonlinear (frozen) |
| mixtral/paper/rc | expert L13 p6 E5 | 0.06 | 0.06 | 0.07 | 0.07 | 0.01 | 0.04 | agree |
| mixtral/paper/rc | expert L17 p6 E0 | 0.11 | 0.12 | 0.12 | 0.13 | 0.03 | 0.08 | agree |
| mixtral/paper/rc | expert L19 p6 E5 | 0.07 | 0.07 | 0.07 | 0.08 | 0.01 | 0.04 | agree |
| mixtral/paper/rc | expert L20 p6 E5 | 0.09 | 0.09 | 0.09 | 0.09 | 0.04 | 0.07 | agree |
| mixtral/paper/rc | expert L30 p6 E2 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | agree |
| mixtral/paper/rc | position L0 p2 | 6.00 | 5.95 | 5.85 | 5.78 |  | 0.16 | nonlinear (frozen) |
| mixtral/paper/rc | position L2 p2 | 0.16 | 0.13 | 0.19 | 0.15 |  | 0.09 | agree |
| mixtral/paper/rc | position L26 p6 | 0.62 | 0.62 | 0.62 | 0.62 | 0.39 | 0.54 | agree |
| mixtral/paper/within_rc | expert L0 p5 E2 | 7.48 | 7.24 | 6.04 | 5.46 | 0.00 | 0.09 | nonlinear (frozen) |
| mixtral/paper/within_rc | expert L4 p5 E5 | 0.16 | 0.14 | 0.16 | 0.14 | 0.00 | 0.04 | nonlinear (frozen) |
| mixtral/paper/within_rc | expert L27 p5 E2 | 1.01 | 1.01 | 1.00 | 1.00 | 0.79 | 0.87 | agree |
| mixtral/paper/within_rc | expert L30 p5 E2 | 1.03 | 1.03 | 1.03 | 1.03 | 1.05 | 1.06 | agree |
| mixtral/paper/within_rc | expert L31 p5 E1 | 0.31 | 0.31 | 0.31 | 0.31 | 0.04 | 0.04 | nonlinear (frozen) |
| mixtral/paper/within_rc | expert L31 p5 E6 | 0.29 | 0.29 | 0.29 | 0.29 | 0.18 | 0.18 | nonlinear (frozen) |
| mixtral/paper/within_rc | position L0 p5 | 7.66 | 7.58 | 7.54 | 7.41 | 0.00 | 0.11 | nonlinear (frozen) |
| mixtral/paper/within_rc | position L2 p5 | 0.36 | 0.32 | 0.36 | 0.31 | 0.00 | 0.09 | nonlinear (frozen) |
| mixtral/paper/within_rc | position L19 p5 | 0.87 | 0.90 | 0.93 | 0.95 | 0.38 | 0.65 | nonlinear (frozen) |
| mixtral/grouped/simple | expert L0 p2 E2 | 9.64 | 9.38 | 9.50 | 8.50 | 0.00 | 0.15 | nonlinear (frozen) |
| mixtral/grouped/simple | expert L2 p2 E5 | 0.20 | 0.13 | 0.21 | 0.15 | 0.00 | 0.08 | routing-mediated |
| mixtral/grouped/simple | expert L4 p2 E5 | 0.17 | 0.14 | 0.17 | 0.14 | -0.00 | 0.06 | nonlinear (frozen) |
| mixtral/grouped/simple | expert L30 p2 E2 | 1.84 | 1.84 | 1.85 | 1.85 | 1.65 | 1.69 | agree |
| mixtral/grouped/simple | expert L31 p2 E0 | 1.86 | 1.86 | 1.86 | 1.86 | 1.88 | 1.88 | agree |
| mixtral/grouped/simple | expert L31 p2 E1 | 0.35 | 0.35 | 0.35 | 0.35 | 0.22 | 0.22 | nonlinear (frozen) |
| mixtral/grouped/simple | position L0 p2 | 9.68 | 9.48 | 9.59 | 9.39 | 0.00 | 0.17 | nonlinear (frozen) |
| mixtral/grouped/simple | position L2 p2 | 0.38 | 0.27 | 0.37 | 0.28 | 0.00 | 0.11 | nonlinear (frozen) |
| mixtral/grouped/simple | position L4 p2 | 0.55 | 0.45 | 0.52 | 0.43 | 0.01 | 0.14 | nonlinear (frozen) |
| mixtral/grouped/nounpp | expert L15 p5 E7 | 0.18 | 0.18 | 0.17 | 0.18 | 0.08 | 0.15 | agree |
| mixtral/grouped/nounpp | expert L29 p5 E1 | 0.56 | 0.56 | 0.56 | 0.57 | 0.39 | 0.47 | agree |
| mixtral/grouped/nounpp | expert L30 p5 E2 | 1.18 | 1.18 | 1.18 | 1.18 | 1.03 | 1.10 | agree |
| mixtral/grouped/nounpp | expert L31 p5 E0 | 0.93 | 0.93 | 0.93 | 0.93 | 0.90 | 0.90 | agree |
| mixtral/grouped/nounpp | expert L31 p5 E1 | 0.57 | 0.57 | 0.57 | 0.57 | 0.49 | 0.49 | agree |
| mixtral/grouped/nounpp | expert L31 p5 E6 | 0.26 | 0.26 | 0.26 | 0.26 | 0.23 | 0.23 | agree |
| mixtral/grouped/nounpp | position L0 p2 | 7.98 | 7.93 | 7.76 | 7.68 |  | 0.17 | nonlinear (frozen) |
| mixtral/grouped/nounpp | position L29 p5 | 0.72 | 0.72 | 0.72 | 0.73 | 0.50 | 0.60 | agree |
| mixtral/grouped/nounpp | position L30 p5 | 1.42 | 1.42 | 1.42 | 1.42 | 1.21 | 1.31 | agree |
| mixtral/grouped/rc | expert L13 p6 E5 | 0.09 | 0.08 | 0.08 | 0.08 | 0.01 | 0.04 | agree |
| mixtral/grouped/rc | expert L14 p6 E2 | 0.06 | 0.06 | 0.06 | 0.07 | 0.00 | 0.03 | agree |
| mixtral/grouped/rc | expert L17 p6 E0 | 0.13 | 0.13 | 0.14 | 0.13 | 0.04 | 0.08 | agree |
| mixtral/grouped/rc | expert L19 p6 E0 | 0.10 | 0.09 | 0.10 | 0.10 | 0.04 | 0.08 | agree |
| mixtral/grouped/rc | expert L19 p6 E5 | 0.09 | 0.09 | 0.09 | 0.09 | 0.02 | 0.06 | agree |
| mixtral/grouped/rc | expert L30 p6 E2 | 0.63 | 0.63 | 0.64 | 0.64 | 0.64 | 0.63 | agree |
| mixtral/grouped/rc | position L0 p2 | 6.04 | 5.99 | 5.84 | 5.80 |  | 0.16 | nonlinear (frozen) |
| mixtral/grouped/rc | position L2 p2 | 0.19 | 0.15 | 0.20 | 0.17 |  | 0.09 | nonlinear (frozen) |
| mixtral/grouped/rc | position L4 p2 | 0.18 | 0.13 | 0.19 | 0.15 |  | 0.09 | agree |
| mixtral/grouped/within_rc | expert L0 p5 E2 | 7.47 | 7.27 | 6.48 | 5.99 | 0.00 | 0.09 | nonlinear (frozen) |
| mixtral/grouped/within_rc | expert L4 p5 E5 | 0.14 | 0.13 | 0.14 | 0.12 | 0.00 | 0.04 | nonlinear (frozen) |
| mixtral/grouped/within_rc | expert L27 p5 E2 | 0.99 | 0.99 | 0.99 | 0.98 | 0.75 | 0.85 | agree |
| mixtral/grouped/within_rc | expert L30 p5 E2 | 1.04 | 1.04 | 1.04 | 1.04 | 1.04 | 1.06 | agree |
| mixtral/grouped/within_rc | expert L31 p5 E1 | 0.36 | 0.36 | 0.36 | 0.36 | 0.06 | 0.06 | nonlinear (frozen) |
| mixtral/grouped/within_rc | expert L31 p5 E6 | 0.32 | 0.32 | 0.32 | 0.32 | 0.21 | 0.21 | nonlinear (frozen) |
| mixtral/grouped/within_rc | position L0 p5 | 7.62 | 7.56 | 7.53 | 7.40 | 0.00 | 0.10 | nonlinear (frozen) |
| mixtral/grouped/within_rc | position L2 p5 | 0.35 | 0.33 | 0.34 | 0.31 | 0.00 | 0.09 | nonlinear (frozen) |
| mixtral/grouped/within_rc | position L19 p5 | 0.92 | 0.93 | 0.98 | 0.99 | 0.39 | 0.68 | nonlinear (frozen) |
