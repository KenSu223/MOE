| model | profile | task | verify | n disc | acc A | margin A | margin B | top expert-position (L, role, E, score) | top expert (L, E, score) |
|---|---|---|---|---|---|---|---|---|---|
| qwen3 | smoke | simple | 2148 ok | 20 | 1.00 | 4.89 | -5.43 | L46 SUBJ E30 +0.969 | L46E30 +0.969 |
| qwen3 | smoke | nounpp | 2148 ok | 20 | 0.95 | 3.97 | -4.10 | L41 noun2 E36 +1.667 | L41E36 +1.954 |
| qwen3 | paper | simple | 2148 ok | 300 | 1.00 | 5.24 | -5.24 | L46 SUBJ E30 +1.035 | L46E30 +1.035 |
| qwen3 | paper | nounpp | 2148 ok | 300 | 0.99 | 3.85 | -3.87 | L41 noun2 E36 +1.522 | L41E36 +1.769 |
| qwen3 | paper | rc | 2148 ok | 300 | 0.99 | 3.53 | -3.48 | L41 verb2 E36 +1.488 | L41E36 +1.685 |
| qwen3 | paper | within_rc | 2148 ok | 300 | 1.00 | 4.26 | -4.22 | L46 SUBJ E30 +0.819 | L46E30 +0.819 |
| qwen3 | grouped | simple | 2148 ok | 76 | 1.00 | 5.46 | -5.43 | L46 SUBJ E30 +1.221 | L46E30 +1.221 |
| qwen3 | grouped | nounpp | 2148 ok | 300 | 0.99 | 3.73 | -3.85 | L41 noun2 E36 +1.458 | L41E36 +1.690 |
| qwen3 | grouped | rc | 2148 ok | 300 | 1.00 | 3.48 | -3.61 | L41 verb2 E36 +1.538 | L41E36 +1.719 |
| qwen3 | grouped | within_rc | 2148 ok | 300 | 1.00 | 4.16 | -4.31 | L41 SUBJ E46 +0.800 | L41E46 +0.800 |
| mixtral | smoke | simple | 1444 ok | 20 | 0.95 | 5.25 | -5.11 | L31 SUBJ E0 +1.762 | L31E0 +1.762 |
| mixtral | smoke | nounpp | 1444 ok | 20 | 1.00 | 4.18 | -4.78 | L30 noun2 E2 +1.011 | L30E2 +1.027 |
| mixtral | paper | simple | 1444 ok | 236 | 0.98 | 5.03 | -5.03 | L31 SUBJ E0 +1.714 | L31E0 +1.714 |
| mixtral | paper | nounpp | 1444 ok | 300 | 0.98 | 3.86 | -3.90 | L30 noun2 E2 +1.035 | L30E2 +1.050 |
| mixtral | paper | rc | 1444 ok | 300 | 0.99 | 3.21 | -3.07 | L30 verb2 E2 +0.612 | L30E2 +0.617 |
| mixtral | paper | within_rc | 1444 ok | 300 | 0.99 | 3.88 | -3.97 | L30 SUBJ E2 +1.069 | L30E2 +1.069 |
| mixtral | grouped | simple | 1444 ok | 54 | 0.98 | 4.92 | -4.87 | L31 SUBJ E0 +1.882 | L31E0 +1.882 |
| mixtral | grouped | nounpp | 1444 ok | 300 | 0.99 | 3.98 | -4.10 | L30 noun2 E2 +1.089 | L30E2 +1.104 |
| mixtral | grouped | rc | 1444 ok | 300 | 0.98 | 3.17 | -3.16 | L30 verb2 E2 +0.625 | L30E2 +0.629 |
| mixtral | grouped | within_rc | 1444 ok | 300 | 0.99 | 3.88 | -3.92 | L30 SUBJ E2 +1.057 | L30E2 +1.057 |
