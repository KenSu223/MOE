**E2 (negation) and Dial-A conflict behaviour: target vs the other answer (copy trap / prior), all scanned pairs (twin-cluster bootstrap CIs)**

| Model | Cell | Stratum | Pairs (items) | Target wins, margin both ways | Other answer wins, margin both ways | Target, sign both ways | Other, sign both ways | Items: target in every order / other in every order | Other answer = |
|---|---|---|---|---|---|---|---|---|---|
| Mistral-7B | E2 a→e | all | 3172 (1586) | 0.00 [0.00, 0.00] | 0.02 [0.02, 0.03] | 0.02 [0.01, 0.02] | 0.30 [0.29, 0.32] | 0 / 3 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | antonym_type=complementary | 38 (19) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.21 [0.11, 0.32] | 0 / 0 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | antonym_type=gradable | 3134 (1567) | 0.00 [0.00, 0.00] | 0.02 [0.02, 0.03] | 0.02 [0.01, 0.03] | 0.30 [0.29, 0.32] | 0 / 3 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | degree_adv=False | 1622 (811) | 0.00 [0.00, 0.00] | 0.02 [0.01, 0.03] | 0.03 [0.02, 0.04] | 0.28 [0.26, 0.30] | 0 / 3 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | degree_adv=True | 1550 (775) | 0.00 [0.00, 0.00] | 0.02 [0.02, 0.03] | 0.01 [0.01, 0.01] | 0.32 [0.30, 0.34] | 0 / 0 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | all | 3144 (1572) | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.00] | 0.33 [0.31, 0.35] | 0.04 [0.04, 0.05] | 25 / 1 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | antonym_type=complementary | 38 (19) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.26 [0.13, 0.39] | 0.05 [0.00, 0.13] | 0 / 0 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | antonym_type=gradable | 3106 (1553) | 0.04 [0.04, 0.05] | 0.00 [0.00, 0.00] | 0.33 [0.31, 0.35] | 0.04 [0.04, 0.05] | 25 / 1 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | degree_adv=False | 1596 (798) | 0.05 [0.04, 0.06] | 0.00 [0.00, 0.01] | 0.35 [0.32, 0.37] | 0.03 [0.02, 0.04] | 13 / 1 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | degree_adv=True | 1548 (774) | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.00] | 0.31 [0.28, 0.34] | 0.06 [0.05, 0.07] | 12 / 0 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | all | 3172 (1586) | 0.00 [0.00, 0.01] | 0.03 [0.02, 0.04] | 0.06 [0.05, 0.07] | 0.29 [0.27, 0.31] | 2 / 15 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | antonym_type=complementary | 38 (19) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.13 [0.03, 0.24] | 0.13 [0.03, 0.24] | 0 / 0 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | antonym_type=gradable | 3134 (1567) | 0.00 [0.00, 0.01] | 0.03 [0.02, 0.04] | 0.06 [0.05, 0.07] | 0.29 [0.27, 0.31] | 2 / 15 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | degree_adv=False | 1622 (811) | 0.00 [0.00, 0.01] | 0.02 [0.01, 0.03] | 0.08 [0.07, 0.10] | 0.23 [0.21, 0.26] | 1 / 7 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | degree_adv=True | 1550 (775) | 0.00 [0.00, 0.01] | 0.04 [0.03, 0.05] | 0.04 [0.03, 0.05] | 0.34 [0.32, 0.37] | 1 / 8 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | A-conflict a→e | all | 24 (12) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.46 [0.21, 0.71] | 0.08 [0.00, 0.21] | 0 / 0 | prior |
| Mixtral-8x7B | E2 e→a | all | 3144 (1572) | 0.02 [0.02, 0.03] | 0.00 [0.00, 0.01] | 0.18 [0.16, 0.19] | 0.06 [0.05, 0.06] | 11 / 2 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | antonym_type=complementary | 38 (19) | 0.03 [0.00, 0.08] | 0.00 [0.00, 0.00] | 0.16 [0.05, 0.26] | 0.03 [0.00, 0.08] | 0 / 0 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | antonym_type=gradable | 3106 (1553) | 0.02 [0.02, 0.03] | 0.00 [0.00, 0.01] | 0.18 [0.16, 0.19] | 0.06 [0.05, 0.06] | 11 / 2 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | degree_adv=False | 1596 (798) | 0.02 [0.01, 0.02] | 0.00 [0.00, 0.01] | 0.19 [0.17, 0.21] | 0.03 [0.02, 0.04] | 1 / 0 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | degree_adv=True | 1548 (774) | 0.03 [0.02, 0.04] | 0.01 [0.00, 0.01] | 0.16 [0.13, 0.18] | 0.08 [0.07, 0.10] | 10 / 2 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | A-conflict e→a | all | 24 (12) | 0.04 [0.00, 0.12] | 0.00 [0.00, 0.00] | 0.21 [0.04, 0.38] | 0.08 [0.00, 0.21] | 0 / 0 | prior |
| Qwen2-57B-A14B | E2 a→e | all | 1178 (589) | 0.00 [0.00, 0.01] | 0.08 [0.07, 0.10] | 0.04 [0.03, 0.06] | 0.54 [0.51, 0.57] | 0 / 14 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | antonym_type=complementary | 10 (5) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.20 [0.00, 0.40] | 0.40 [0.00, 0.80] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | antonym_type=gradable | 1168 (584) | 0.00 [0.00, 0.01] | 0.08 [0.07, 0.10] | 0.04 [0.03, 0.05] | 0.54 [0.51, 0.58] | 0 / 14 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | degree_adv=False | 490 (245) | 0.00 [0.00, 0.01] | 0.07 [0.05, 0.10] | 0.06 [0.04, 0.08] | 0.46 [0.40, 0.51] | 0 / 5 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | degree_adv=True | 688 (344) | 0.00 [0.00, 0.01] | 0.09 [0.07, 0.12] | 0.03 [0.02, 0.05] | 0.60 [0.56, 0.64] | 0 / 9 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | all | 1178 (589) | 0.08 [0.06, 0.09] | 0.01 [0.00, 0.01] | 0.31 [0.28, 0.33] | 0.09 [0.07, 0.11] | 13 / 1 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | antonym_type=complementary | 10 (5) | 0.10 [0.00, 0.30] | 0.00 [0.00, 0.00] | 0.30 [0.00, 0.70] | 0.10 [0.00, 0.30] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | antonym_type=gradable | 1168 (584) | 0.08 [0.06, 0.09] | 0.01 [0.00, 0.01] | 0.31 [0.28, 0.33] | 0.09 [0.07, 0.11] | 13 / 1 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | degree_adv=False | 490 (245) | 0.08 [0.06, 0.11] | 0.00 [0.00, 0.00] | 0.34 [0.30, 0.39] | 0.03 [0.02, 0.05] | 7 / 0 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | degree_adv=True | 688 (344) | 0.07 [0.05, 0.09] | 0.01 [0.00, 0.02] | 0.28 [0.25, 0.32] | 0.13 [0.10, 0.16] | 6 / 1 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | all | 3578 (1789) | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.04] | 0.02 [0.01, 0.02] | 0.39 [0.38, 0.41] | 1 / 15 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.08 [0.00, 0.19] | 0.02 [0.00, 0.06] | 0.31 [0.17, 0.48] | 0 / 1 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | antonym_type=gradable | 3530 (1765) | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.04] | 0.02 [0.01, 0.02] | 0.39 [0.38, 0.41] | 1 / 14 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | degree_adv=False | 1986 (993) | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.04] | 0.03 [0.02, 0.03] | 0.37 [0.35, 0.40] | 1 / 7 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | degree_adv=True | 1592 (796) | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.04] | 0.01 [0.01, 0.02] | 0.42 [0.39, 0.44] | 0 / 8 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | all | 3578 (1789) | 0.05 [0.04, 0.05] | 0.00 [0.00, 0.01] | 0.31 [0.29, 0.32] | 0.05 [0.04, 0.06] | 13 / 1 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | antonym_type=complementary | 50 (25) | 0.04 [0.00, 0.10] | 0.00 [0.00, 0.00] | 0.50 [0.36, 0.62] | 0.02 [0.00, 0.06] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | antonym_type=gradable | 3528 (1764) | 0.05 [0.04, 0.05] | 0.00 [0.00, 0.01] | 0.30 [0.29, 0.32] | 0.05 [0.04, 0.06] | 13 / 1 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | degree_adv=False | 1986 (993) | 0.06 [0.05, 0.07] | 0.00 [0.00, 0.00] | 0.38 [0.35, 0.40] | 0.02 [0.01, 0.03] | 8 / 0 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | degree_adv=True | 1592 (796) | 0.02 [0.02, 0.03] | 0.01 [0.00, 0.01] | 0.22 [0.20, 0.24] | 0.09 [0.07, 0.10] | 5 / 1 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | all | 3578 (1789) | 0.00 [0.00, 0.00] | 0.59 [0.57, 0.61] | 0.00 [0.00, 0.00] | 0.83 [0.82, 0.85] | 0 / 827 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.25 [0.10, 0.40] | 0.00 [0.00, 0.00] | 0.46 [0.29, 0.65] | 0 / 4 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | antonym_type=gradable | 3530 (1765) | 0.00 [0.00, 0.00] | 0.60 [0.58, 0.62] | 0.00 [0.00, 0.00] | 0.84 [0.82, 0.85] | 0 / 823 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | degree_adv=False | 1986 (993) | 0.00 [0.00, 0.00] | 0.45 [0.43, 0.48] | 0.00 [0.00, 0.01] | 0.76 [0.73, 0.78] | 0 / 294 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | degree_adv=True | 1592 (796) | 0.00 [0.00, 0.00] | 0.77 [0.74, 0.79] | 0.00 [0.00, 0.00] | 0.92 [0.91, 0.94] | 0 / 533 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | A-conflict a→e | all | 340 (170) | 0.96 [0.94, 0.98] | 0.00 [0.00, 0.00] | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 157 / 0 | prior |
| Qwen3-30B-A3B | E2 e→a | all | 3578 (1789) | 0.03 [0.03, 0.04] | 0.02 [0.02, 0.03] | 0.15 [0.14, 0.17] | 0.13 [0.11, 0.14] | 12 / 9 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | antonym_type=complementary | 50 (25) | 0.06 [0.00, 0.12] | 0.00 [0.00, 0.00] | 0.22 [0.12, 0.34] | 0.00 [0.00, 0.00] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | antonym_type=gradable | 3528 (1764) | 0.03 [0.03, 0.04] | 0.02 [0.02, 0.03] | 0.15 [0.14, 0.16] | 0.13 [0.12, 0.14] | 12 / 9 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | degree_adv=False | 1986 (993) | 0.04 [0.03, 0.05] | 0.01 [0.01, 0.02] | 0.18 [0.16, 0.19] | 0.07 [0.06, 0.09] | 8 / 2 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | degree_adv=True | 1592 (796) | 0.02 [0.01, 0.03] | 0.04 [0.03, 0.05] | 0.13 [0.11, 0.14] | 0.19 [0.17, 0.21] | 4 / 7 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | A-conflict e→a | all | 340 (170) | 0.30 [0.25, 0.36] | 0.01 [0.00, 0.01] | 0.44 [0.39, 0.51] | 0.05 [0.03, 0.07] | 22 / 0 | prior |
| Qwen3-14B | E2 a→e | all | 3578 (1789) | 0.00 [0.00, 0.01] | 0.56 [0.54, 0.58] | 0.02 [0.01, 0.02] | 0.79 [0.78, 0.81] | 2 / 714 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | antonym_type=complementary | 48 (24) | 0.02 [0.00, 0.06] | 0.08 [0.00, 0.19] | 0.10 [0.02, 0.21] | 0.44 [0.29, 0.60] | 0 / 1 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | antonym_type=gradable | 3530 (1765) | 0.00 [0.00, 0.01] | 0.56 [0.55, 0.58] | 0.01 [0.01, 0.02] | 0.80 [0.78, 0.81] | 2 / 713 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | degree_adv=False | 1986 (993) | 0.01 [0.00, 0.01] | 0.40 [0.38, 0.42] | 0.02 [0.02, 0.03] | 0.70 [0.68, 0.73] | 2 / 213 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | degree_adv=True | 1592 (796) | 0.00 [0.00, 0.00] | 0.76 [0.73, 0.78] | 0.00 [0.00, 0.01] | 0.90 [0.89, 0.92] | 0 / 501 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | all | 3578 (1789) | 0.06 [0.05, 0.07] | 0.01 [0.01, 0.01] | 0.25 [0.23, 0.27] | 0.09 [0.08, 0.10] | 19 / 2 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | antonym_type=complementary | 50 (25) | 0.06 [0.00, 0.14] | 0.00 [0.00, 0.00] | 0.20 [0.08, 0.34] | 0.02 [0.00, 0.06] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | antonym_type=gradable | 3528 (1764) | 0.06 [0.05, 0.07] | 0.01 [0.01, 0.01] | 0.25 [0.24, 0.27] | 0.09 [0.08, 0.10] | 19 / 2 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | degree_adv=False | 1986 (993) | 0.07 [0.06, 0.08] | 0.00 [0.00, 0.01] | 0.28 [0.26, 0.30] | 0.04 [0.03, 0.05] | 11 / 0 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | degree_adv=True | 1592 (796) | 0.05 [0.04, 0.06] | 0.02 [0.01, 0.02] | 0.21 [0.19, 0.23] | 0.14 [0.12, 0.16] | 8 / 2 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | all | 1178 (589) | 0.00 [0.00, 0.00] | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.01] | 0.26 [0.23, 0.28] | 0 / 8 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | antonym_type=complementary | 10 (5) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.10 [0.00, 0.30] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | antonym_type=gradable | 1168 (584) | 0.00 [0.00, 0.00] | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.01] | 0.26 [0.23, 0.28] | 0 / 8 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | degree_adv=False | 490 (245) | 0.00 [0.00, 0.00] | 0.04 [0.02, 0.06] | 0.00 [0.00, 0.01] | 0.29 [0.25, 0.32] | 0 / 2 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | degree_adv=True | 688 (344) | 0.00 [0.00, 0.00] | 0.04 [0.02, 0.06] | 0.00 [0.00, 0.01] | 0.24 [0.20, 0.27] | 0 / 6 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | all | 1178 (589) | 0.09 [0.07, 0.10] | 0.00 [0.00, 0.00] | 0.36 [0.33, 0.39] | 0.02 [0.01, 0.03] | 15 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | antonym_type=complementary | 10 (5) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.20 [0.00, 0.40] | 0.00 [0.00, 0.00] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | antonym_type=gradable | 1168 (584) | 0.09 [0.07, 0.10] | 0.00 [0.00, 0.00] | 0.36 [0.33, 0.39] | 0.02 [0.01, 0.03] | 15 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | degree_adv=False | 490 (245) | 0.13 [0.10, 0.17] | 0.00 [0.00, 0.01] | 0.48 [0.44, 0.53] | 0.02 [0.01, 0.03] | 10 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | degree_adv=True | 688 (344) | 0.05 [0.03, 0.07] | 0.00 [0.00, 0.00] | 0.27 [0.24, 0.31] | 0.02 [0.01, 0.03] | 5 / 0 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | all | 3578 (1789) | 0.00 [0.00, 0.00] | 0.60 [0.58, 0.62] | 0.00 [0.00, 0.00] | 0.84 [0.83, 0.85] | 0 / 785 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.25 [0.12, 0.40] | 0.00 [0.00, 0.00] | 0.54 [0.40, 0.69] | 0 / 2 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | antonym_type=gradable | 3530 (1765) | 0.00 [0.00, 0.00] | 0.61 [0.59, 0.63] | 0.00 [0.00, 0.00] | 0.85 [0.83, 0.86] | 0 / 783 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | degree_adv=False | 1986 (993) | 0.00 [0.00, 0.00] | 0.50 [0.48, 0.53] | 0.00 [0.00, 0.00] | 0.79 [0.77, 0.81] | 0 / 307 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | degree_adv=True | 1592 (796) | 0.00 [0.00, 0.00] | 0.73 [0.71, 0.76] | 0.00 [0.00, 0.00] | 0.91 [0.89, 0.92] | 0 / 478 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | all | 3578 (1789) | 0.01 [0.01, 0.02] | 0.02 [0.02, 0.03] | 0.12 [0.10, 0.13] | 0.11 [0.09, 0.12] | 4 / 12 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | antonym_type=complementary | 50 (25) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.12] | 0.12 [0.04, 0.22] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | antonym_type=gradable | 3528 (1764) | 0.01 [0.01, 0.02] | 0.02 [0.02, 0.03] | 0.12 [0.10, 0.13] | 0.11 [0.09, 0.12] | 4 / 12 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | degree_adv=False | 1986 (993) | 0.02 [0.01, 0.02] | 0.02 [0.01, 0.02] | 0.12 [0.11, 0.14] | 0.10 [0.08, 0.11] | 2 / 3 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | degree_adv=True | 1592 (796) | 0.01 [0.01, 0.02] | 0.03 [0.02, 0.04] | 0.10 [0.09, 0.12] | 0.12 [0.10, 0.13] | 2 / 9 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | all | 3578 (1789) | 0.00 [0.00, 0.00] | 0.41 [0.39, 0.43] | 0.01 [0.01, 0.01] | 0.65 [0.63, 0.67] | 0 / 526 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.08 [0.02, 0.17] | 0.00 [0.00, 0.00] | 0.38 [0.23, 0.54] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | antonym_type=gradable | 3530 (1765) | 0.00 [0.00, 0.00] | 0.41 [0.39, 0.43] | 0.01 [0.01, 0.01] | 0.66 [0.64, 0.67] | 0 / 526 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | degree_adv=False | 1986 (993) | 0.00 [0.00, 0.00] | 0.18 [0.16, 0.20] | 0.01 [0.01, 0.02] | 0.50 [0.47, 0.52] | 0 / 64 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | degree_adv=True | 1592 (796) | 0.00 [0.00, 0.00] | 0.69 [0.66, 0.71] | 0.00 [0.00, 0.01] | 0.85 [0.83, 0.87] | 0 / 462 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | all | 3578 (1789) | 0.03 [0.02, 0.03] | 0.01 [0.01, 0.01] | 0.16 [0.15, 0.18] | 0.06 [0.05, 0.07] | 6 / 5 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | antonym_type=complementary | 50 (25) | 0.04 [0.00, 0.10] | 0.00 [0.00, 0.00] | 0.18 [0.08, 0.30] | 0.02 [0.00, 0.06] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | antonym_type=gradable | 3528 (1764) | 0.03 [0.02, 0.03] | 0.01 [0.01, 0.01] | 0.16 [0.15, 0.18] | 0.06 [0.05, 0.07] | 6 / 5 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | degree_adv=False | 1986 (993) | 0.04 [0.03, 0.05] | 0.01 [0.00, 0.01] | 0.21 [0.19, 0.22] | 0.05 [0.04, 0.06] | 5 / 1 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | degree_adv=True | 1592 (796) | 0.01 [0.01, 0.02] | 0.01 [0.01, 0.02] | 0.11 [0.10, 0.13] | 0.08 [0.06, 0.09] | 1 / 4 | copy trap (word stated with the other entity) |

Full table with every cell and stratum: ext14_behaviour_rates.md.
