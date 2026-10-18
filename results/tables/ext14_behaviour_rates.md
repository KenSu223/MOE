**Behaviour of every scanned pair (twin-cluster bootstrap CIs, 1,000 resamples)**

| Model | Cell | Stratum | Pairs (items) | Target wins, margin both ways | Other answer wins, margin both ways | Target, sign both ways | Other, sign both ways | Items: target in every order / other in every order | Other answer = |
|---|---|---|---|---|---|---|---|---|---|
| Mistral-7B | E1 a→e | all | 3172 (1586) | 0.17 [0.15, 0.18] | 0.00 [0.00, 0.00] | 0.52 [0.51, 0.54] | 0.00 [0.00, 0.00] | 47 / 0 | other answer |
| Mistral-7B | E1 a→e | antonym_type=complementary | 38 (19) | 0.05 [0.00, 0.13] | 0.00 [0.00, 0.00] | 0.45 [0.29, 0.61] | 0.00 [0.00, 0.00] | 0 / 0 | other answer |
| Mistral-7B | E1 a→e | antonym_type=gradable | 3134 (1567) | 0.17 [0.16, 0.18] | 0.00 [0.00, 0.00] | 0.52 [0.51, 0.54] | 0.00 [0.00, 0.00] | 47 / 0 | other answer |
| Mistral-7B | E1 a→e | degree_adv=False | 1622 (811) | 0.12 [0.10, 0.13] | 0.00 [0.00, 0.00] | 0.46 [0.43, 0.48] | 0.00 [0.00, 0.01] | 17 / 0 | other answer |
| Mistral-7B | E1 a→e | degree_adv=True | 1550 (775) | 0.22 [0.20, 0.24] | 0.00 [0.00, 0.00] | 0.59 [0.57, 0.61] | 0.00 [0.00, 0.00] | 30 / 0 | other answer |
| Mistral-7B | E1 a→e | names=False | 2860 (1430) | 0.18 [0.16, 0.19] | 0.00 [0.00, 0.00] | 0.53 [0.51, 0.54] | 0.00 [0.00, 0.00] | 39 / 0 | other answer |
| Mistral-7B | E1 a→e | names=True | 312 (156) | 0.08 [0.04, 0.12] | 0.00 [0.00, 0.00] | 0.48 [0.42, 0.55] | 0.01 [0.00, 0.02] | 8 / 0 | other answer |
| Mistral-7B | E1 a→e | order=1 | 1586 (1586) | 0.18 [0.16, 0.19] | 0.00 [0.00, 0.00] | 0.53 [0.50, 0.55] | 0.00 [0.00, 0.00] | 279 / 0 | other answer |
| Mistral-7B | E1 a→e | order=2 | 1586 (1586) | 0.16 [0.14, 0.18] | 0.00 [0.00, 0.00] | 0.52 [0.49, 0.54] | 0.00 [0.00, 0.01] | 248 / 0 | other answer |
| Mistral-7B | E1 a→e (all strata) | all | 3172 (1586) | 0.17 [0.15, 0.18] | 0.00 [0.00, 0.00] | 0.52 [0.51, 0.54] | 0.00 [0.00, 0.00] | 47 / 0 | other answer |
| Mistral-7B | E1 a→e (all strata) | antonym_type=complementary | 38 (19) | 0.05 [0.00, 0.13] | 0.00 [0.00, 0.00] | 0.45 [0.29, 0.61] | 0.00 [0.00, 0.00] | 0 / 0 | other answer |
| Mistral-7B | E1 a→e (all strata) | antonym_type=gradable | 3134 (1567) | 0.17 [0.16, 0.18] | 0.00 [0.00, 0.00] | 0.52 [0.51, 0.54] | 0.00 [0.00, 0.00] | 47 / 0 | other answer |
| Mistral-7B | E1 a→e (all strata) | degree_adv=False | 1622 (811) | 0.12 [0.10, 0.13] | 0.00 [0.00, 0.00] | 0.46 [0.43, 0.48] | 0.00 [0.00, 0.01] | 17 / 0 | other answer |
| Mistral-7B | E1 a→e (all strata) | degree_adv=True | 1550 (775) | 0.22 [0.20, 0.24] | 0.00 [0.00, 0.00] | 0.59 [0.57, 0.61] | 0.00 [0.00, 0.00] | 30 / 0 | other answer |
| Mistral-7B | E1 a→e (all strata) | names=False | 2860 (1430) | 0.18 [0.16, 0.19] | 0.00 [0.00, 0.00] | 0.53 [0.51, 0.54] | 0.00 [0.00, 0.00] | 39 / 0 | other answer |
| Mistral-7B | E1 a→e (all strata) | names=True | 312 (156) | 0.08 [0.04, 0.12] | 0.00 [0.00, 0.00] | 0.48 [0.42, 0.55] | 0.01 [0.00, 0.02] | 8 / 0 | other answer |
| Mistral-7B | E1 a→e (all strata) | order=1 | 1586 (1586) | 0.18 [0.16, 0.19] | 0.00 [0.00, 0.00] | 0.53 [0.50, 0.55] | 0.00 [0.00, 0.00] | 279 / 0 | other answer |
| Mistral-7B | E1 a→e (all strata) | order=2 | 1586 (1586) | 0.16 [0.14, 0.18] | 0.00 [0.00, 0.00] | 0.52 [0.49, 0.54] | 0.00 [0.00, 0.01] | 248 / 0 | other answer |
| Mistral-7B | E1p a→e | all | 2220 (1110) | 0.08 [0.07, 0.09] | 0.00 [0.00, 0.00] | 0.39 [0.37, 0.40] | 0.00 [0.00, 0.01] | 5 / 0 | other answer |
| Mistral-7B | E1p a→e | antonym_type=complementary | 18 (9) | 0.06 [0.00, 0.17] | 0.00 [0.00, 0.00] | 0.33 [0.11, 0.56] | 0.00 [0.00, 0.00] | 0 / 0 | other answer |
| Mistral-7B | E1p a→e | antonym_type=gradable | 2202 (1101) | 0.08 [0.07, 0.09] | 0.00 [0.00, 0.00] | 0.39 [0.37, 0.41] | 0.00 [0.00, 0.01] | 5 / 0 | other answer |
| Mistral-7B | E1p a→e | degree_adv=False | 1068 (534) | 0.07 [0.06, 0.09] | 0.00 [0.00, 0.00] | 0.37 [0.35, 0.40] | 0.00 [0.00, 0.01] | 1 / 0 | other answer |
| Mistral-7B | E1p a→e | degree_adv=True | 1152 (576) | 0.09 [0.08, 0.11] | 0.00 [0.00, 0.00] | 0.39 [0.37, 0.42] | 0.00 [0.00, 0.00] | 4 / 0 | other answer |
| Mistral-7B | E1p a→e | names=False | 2028 (1014) | 0.09 [0.08, 0.10] | 0.00 [0.00, 0.00] | 0.38 [0.37, 0.40] | 0.00 [0.00, 0.01] | 5 / 0 | other answer |
| Mistral-7B | E1p a→e | names=True | 192 (96) | 0.01 [0.00, 0.03] | 0.00 [0.00, 0.00] | 0.40 [0.33, 0.48] | 0.00 [0.00, 0.00] | 0 / 0 | other answer |
| Mistral-7B | E1p a→e | order=1 | 1110 (1110) | 0.08 [0.06, 0.09] | 0.00 [0.00, 0.00] | 0.39 [0.37, 0.42] | 0.00 [0.00, 0.00] | 87 / 0 | other answer |
| Mistral-7B | E1p a→e | order=2 | 1110 (1110) | 0.09 [0.07, 0.10] | 0.00 [0.00, 0.00] | 0.38 [0.35, 0.41] | 0.01 [0.00, 0.01] | 97 / 0 | other answer |
| Mistral-7B | E1p a→e (all strata) | all | 2220 (1110) | 0.08 [0.07, 0.09] | 0.00 [0.00, 0.00] | 0.39 [0.37, 0.40] | 0.00 [0.00, 0.01] | 5 / 0 | other answer |
| Mistral-7B | E1p a→e (all strata) | antonym_type=complementary | 18 (9) | 0.06 [0.00, 0.17] | 0.00 [0.00, 0.00] | 0.33 [0.11, 0.56] | 0.00 [0.00, 0.00] | 0 / 0 | other answer |
| Mistral-7B | E1p a→e (all strata) | antonym_type=gradable | 2202 (1101) | 0.08 [0.07, 0.09] | 0.00 [0.00, 0.00] | 0.39 [0.37, 0.41] | 0.00 [0.00, 0.01] | 5 / 0 | other answer |
| Mistral-7B | E1p a→e (all strata) | degree_adv=False | 1068 (534) | 0.07 [0.06, 0.09] | 0.00 [0.00, 0.00] | 0.37 [0.35, 0.40] | 0.00 [0.00, 0.01] | 1 / 0 | other answer |
| Mistral-7B | E1p a→e (all strata) | degree_adv=True | 1152 (576) | 0.09 [0.08, 0.11] | 0.00 [0.00, 0.00] | 0.39 [0.37, 0.42] | 0.00 [0.00, 0.00] | 4 / 0 | other answer |
| Mistral-7B | E1p a→e (all strata) | names=False | 2028 (1014) | 0.09 [0.08, 0.10] | 0.00 [0.00, 0.00] | 0.38 [0.37, 0.40] | 0.00 [0.00, 0.01] | 5 / 0 | other answer |
| Mistral-7B | E1p a→e (all strata) | names=True | 192 (96) | 0.01 [0.00, 0.03] | 0.00 [0.00, 0.00] | 0.40 [0.33, 0.48] | 0.00 [0.00, 0.00] | 0 / 0 | other answer |
| Mistral-7B | E1p a→e (all strata) | order=1 | 1110 (1110) | 0.08 [0.06, 0.09] | 0.00 [0.00, 0.00] | 0.39 [0.37, 0.42] | 0.00 [0.00, 0.00] | 87 / 0 | other answer |
| Mistral-7B | E1p a→e (all strata) | order=2 | 1110 (1110) | 0.09 [0.07, 0.10] | 0.00 [0.00, 0.00] | 0.38 [0.35, 0.41] | 0.01 [0.00, 0.01] | 97 / 0 | other answer |
| Mistral-7B | E2 a→e | all | 3172 (1586) | 0.00 [0.00, 0.00] | 0.02 [0.02, 0.03] | 0.02 [0.01, 0.02] | 0.30 [0.29, 0.32] | 0 / 3 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | antonym_type=complementary | 38 (19) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.21 [0.11, 0.32] | 0 / 0 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | antonym_type=gradable | 3134 (1567) | 0.00 [0.00, 0.00] | 0.02 [0.02, 0.03] | 0.02 [0.01, 0.03] | 0.30 [0.29, 0.32] | 0 / 3 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | degree_adv=False | 1622 (811) | 0.00 [0.00, 0.00] | 0.02 [0.01, 0.03] | 0.03 [0.02, 0.04] | 0.28 [0.26, 0.30] | 0 / 3 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | degree_adv=True | 1550 (775) | 0.00 [0.00, 0.00] | 0.02 [0.02, 0.03] | 0.01 [0.01, 0.01] | 0.32 [0.30, 0.34] | 0 / 0 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | names=False | 2860 (1430) | 0.00 [0.00, 0.00] | 0.02 [0.02, 0.03] | 0.01 [0.01, 0.02] | 0.31 [0.29, 0.32] | 0 / 3 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | names=True | 312 (156) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.07 [0.05, 0.11] | 0.26 [0.21, 0.32] | 0 / 0 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | order=1 | 1586 (1586) | 0.00 [0.00, 0.00] | 0.02 [0.01, 0.03] | 0.01 [0.01, 0.02] | 0.28 [0.26, 0.31] | 1 / 33 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 a→e | order=2 | 1586 (1586) | 0.00 [0.00, 0.00] | 0.02 [0.02, 0.03] | 0.02 [0.02, 0.03] | 0.32 [0.30, 0.34] | 2 / 34 | copy trap (word stated with the other entity) |
| Mistral-7B | E3 a→e (P3 frame) | all | 1572 (1572) | 0.16 [0.14, 0.18] | 0.00 [0.00, 0.01] | 0.36 [0.34, 0.38] | 0.02 [0.02, 0.03] | 252 / 4 | other answer |
| Mistral-7B | E3 a→e (P3 frame) | antonym_type=complementary | 16 (16) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.19] | 0.00 [0.00, 0.00] | 0 / 0 | other answer |
| Mistral-7B | E3 a→e (P3 frame) | antonym_type=gradable | 1556 (1556) | 0.16 [0.14, 0.18] | 0.00 [0.00, 0.01] | 0.36 [0.34, 0.39] | 0.02 [0.02, 0.03] | 252 / 4 | other answer |
| Mistral-7B | E3 a→e (P3 frame) | degree_adv=False | 814 (814) | 0.13 [0.11, 0.15] | 0.00 [0.00, 0.01] | 0.33 [0.29, 0.36] | 0.03 [0.02, 0.04] | 108 / 3 | other answer |
| Mistral-7B | E3 a→e (P3 frame) | degree_adv=True | 758 (758) | 0.19 [0.16, 0.22] | 0.00 [0.00, 0.00] | 0.40 [0.36, 0.43] | 0.02 [0.01, 0.03] | 144 / 1 | other answer |
| Mistral-7B | E3 a→e (P3 frame) | names=False | 1423 (1423) | 0.17 [0.15, 0.19] | 0.00 [0.00, 0.01] | 0.35 [0.33, 0.38] | 0.02 [0.02, 0.03] | 239 / 4 | other answer |
| Mistral-7B | E3 a→e (P3 frame) | names=True | 149 (149) | 0.09 [0.05, 0.14] | 0.00 [0.00, 0.00] | 0.45 [0.37, 0.53] | 0.01 [0.00, 0.02] | 13 / 0 | other answer |
| Mistral-7B | E3 a→e (P3 frame) | order=1 | 1095 (1095) | 0.15 [0.13, 0.18] | 0.00 [0.00, 0.01] | 0.35 [0.33, 0.38] | 0.03 [0.02, 0.04] | 168 / 4 | other answer |
| Mistral-7B | E3 a→e (P3 frame) | order=2 | 477 (477) | 0.18 [0.14, 0.21] | 0.00 [0.00, 0.00] | 0.38 [0.34, 0.43] | 0.01 [0.00, 0.02] | 84 / 0 | other answer |
| Mistral-7B | E3 a→e (F1 frame, Phase-5 anchor) | all | 1642 (1642) | 0.16 [0.14, 0.18] | 0.00 [0.00, 0.00] | 0.38 [0.35, 0.40] | 0.02 [0.01, 0.02] | 258 / 0 | other answer |
| Mistral-7B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=complementary | 19 (19) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.21 [0.05, 0.42] | 0.05 [0.00, 0.16] | 0 / 0 | other answer |
| Mistral-7B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=gradable | 1623 (1623) | 0.16 [0.14, 0.18] | 0.00 [0.00, 0.00] | 0.38 [0.35, 0.40] | 0.02 [0.01, 0.02] | 258 / 0 | other answer |
| Mistral-7B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=False | 854 (854) | 0.16 [0.14, 0.19] | 0.00 [0.00, 0.00] | 0.37 [0.34, 0.41] | 0.02 [0.01, 0.03] | 136 / 0 | other answer |
| Mistral-7B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=True | 788 (788) | 0.15 [0.13, 0.18] | 0.00 [0.00, 0.00] | 0.38 [0.35, 0.42] | 0.01 [0.00, 0.02] | 122 / 0 | other answer |
| Mistral-7B | E3 a→e (F1 frame, Phase-5 anchor) | names=False | 1488 (1488) | 0.17 [0.15, 0.19] | 0.00 [0.00, 0.00] | 0.39 [0.36, 0.41] | 0.02 [0.01, 0.03] | 254 / 0 | other answer |
| Mistral-7B | E3 a→e (F1 frame, Phase-5 anchor) | names=True | 154 (154) | 0.03 [0.01, 0.05] | 0.00 [0.00, 0.00] | 0.29 [0.22, 0.36] | 0.01 [0.00, 0.02] | 4 / 0 | other answer |
| Mistral-7B | E3 a→e (F1 frame, Phase-5 anchor) | order=1 | 1135 (1135) | 0.15 [0.13, 0.17] | 0.00 [0.00, 0.00] | 0.36 [0.33, 0.39] | 0.02 [0.01, 0.03] | 171 / 0 | other answer |
| Mistral-7B | E3 a→e (F1 frame, Phase-5 anchor) | order=2 | 507 (507) | 0.17 [0.14, 0.21] | 0.00 [0.00, 0.00] | 0.42 [0.38, 0.46] | 0.01 [0.00, 0.03] | 87 / 0 | other answer |
| Mistral-7B | E1 e→a | all | 3144 (1572) | 0.09 [0.08, 0.10] | 0.03 [0.02, 0.03] | 0.34 [0.32, 0.36] | 0.16 [0.15, 0.17] | 52 / 10 | other answer |
| Mistral-7B | E1 e→a | antonym_type=complementary | 38 (19) | 0.03 [0.00, 0.08] | 0.03 [0.00, 0.08] | 0.24 [0.08, 0.42] | 0.21 [0.08, 0.37] | 0 / 0 | other answer |
| Mistral-7B | E1 e→a | antonym_type=gradable | 3106 (1553) | 0.09 [0.08, 0.10] | 0.03 [0.02, 0.03] | 0.34 [0.32, 0.36] | 0.16 [0.14, 0.17] | 52 / 10 | other answer |
| Mistral-7B | E1 e→a | degree_adv=False | 1596 (798) | 0.06 [0.04, 0.07] | 0.05 [0.04, 0.06] | 0.19 [0.17, 0.21] | 0.26 [0.24, 0.29] | 18 / 9 | other answer |
| Mistral-7B | E1 e→a | degree_adv=True | 1548 (774) | 0.13 [0.11, 0.15] | 0.01 [0.00, 0.01] | 0.50 [0.47, 0.53] | 0.05 [0.04, 0.07] | 34 / 1 | other answer |
| Mistral-7B | E1 e→a | names=False | 2838 (1419) | 0.09 [0.08, 0.10] | 0.03 [0.02, 0.03] | 0.35 [0.33, 0.37] | 0.16 [0.15, 0.18] | 48 / 9 | other answer |
| Mistral-7B | E1 e→a | names=True | 306 (153) | 0.08 [0.05, 0.12] | 0.03 [0.01, 0.05] | 0.25 [0.19, 0.32] | 0.13 [0.09, 0.17] | 4 / 1 | other answer |
| Mistral-7B | E1 e→a | order=1 | 1572 (1572) | 0.09 [0.07, 0.10] | 0.03 [0.02, 0.04] | 0.34 [0.32, 0.37] | 0.15 [0.14, 0.17] | 136 / 43 | other answer |
| Mistral-7B | E1 e→a | order=2 | 1572 (1572) | 0.09 [0.08, 0.11] | 0.03 [0.02, 0.03] | 0.34 [0.32, 0.37] | 0.17 [0.15, 0.18] | 147 / 41 | other answer |
| Mistral-7B | E1 e→a (all strata) | all | 3144 (1572) | 0.09 [0.08, 0.10] | 0.03 [0.02, 0.03] | 0.34 [0.32, 0.36] | 0.16 [0.15, 0.17] | 52 / 10 | other answer |
| Mistral-7B | E1 e→a (all strata) | antonym_type=complementary | 38 (19) | 0.03 [0.00, 0.08] | 0.03 [0.00, 0.08] | 0.24 [0.08, 0.42] | 0.21 [0.08, 0.37] | 0 / 0 | other answer |
| Mistral-7B | E1 e→a (all strata) | antonym_type=gradable | 3106 (1553) | 0.09 [0.08, 0.10] | 0.03 [0.02, 0.03] | 0.34 [0.32, 0.36] | 0.16 [0.14, 0.17] | 52 / 10 | other answer |
| Mistral-7B | E1 e→a (all strata) | degree_adv=False | 1596 (798) | 0.06 [0.04, 0.07] | 0.05 [0.04, 0.06] | 0.19 [0.17, 0.21] | 0.26 [0.24, 0.29] | 18 / 9 | other answer |
| Mistral-7B | E1 e→a (all strata) | degree_adv=True | 1548 (774) | 0.13 [0.11, 0.15] | 0.01 [0.00, 0.01] | 0.50 [0.47, 0.53] | 0.05 [0.04, 0.07] | 34 / 1 | other answer |
| Mistral-7B | E1 e→a (all strata) | names=False | 2838 (1419) | 0.09 [0.08, 0.10] | 0.03 [0.02, 0.03] | 0.35 [0.33, 0.37] | 0.16 [0.15, 0.18] | 48 / 9 | other answer |
| Mistral-7B | E1 e→a (all strata) | names=True | 306 (153) | 0.08 [0.05, 0.12] | 0.03 [0.01, 0.05] | 0.25 [0.19, 0.32] | 0.13 [0.09, 0.17] | 4 / 1 | other answer |
| Mistral-7B | E1 e→a (all strata) | order=1 | 1572 (1572) | 0.09 [0.07, 0.10] | 0.03 [0.02, 0.04] | 0.34 [0.32, 0.37] | 0.15 [0.14, 0.17] | 136 / 43 | other answer |
| Mistral-7B | E1 e→a (all strata) | order=2 | 1572 (1572) | 0.09 [0.08, 0.11] | 0.03 [0.02, 0.03] | 0.34 [0.32, 0.37] | 0.17 [0.15, 0.18] | 147 / 41 | other answer |
| Mistral-7B | E1p e→a | all | 2216 (1108) | 0.09 [0.08, 0.11] | 0.00 [0.00, 0.00] | 0.33 [0.30, 0.35] | 0.04 [0.03, 0.05] | 58 / 1 | other answer |
| Mistral-7B | E1p e→a | antonym_type=complementary | 18 (9) | 0.17 [0.00, 0.39] | 0.00 [0.00, 0.00] | 0.17 [0.00, 0.39] | 0.00 [0.00, 0.00] | 1 / 0 | other answer |
| Mistral-7B | E1p e→a | antonym_type=gradable | 2198 (1099) | 0.09 [0.08, 0.11] | 0.00 [0.00, 0.00] | 0.33 [0.31, 0.35] | 0.04 [0.03, 0.05] | 57 / 1 | other answer |
| Mistral-7B | E1p e→a | degree_adv=False | 1064 (532) | 0.09 [0.07, 0.11] | 0.00 [0.00, 0.00] | 0.29 [0.26, 0.33] | 0.06 [0.04, 0.07] | 28 / 0 | other answer |
| Mistral-7B | E1p e→a | degree_adv=True | 1152 (576) | 0.10 [0.08, 0.12] | 0.00 [0.00, 0.01] | 0.36 [0.33, 0.39] | 0.02 [0.01, 0.03] | 30 / 1 | other answer |
| Mistral-7B | E1p e→a | names=False | 2024 (1012) | 0.08 [0.06, 0.09] | 0.00 [0.00, 0.00] | 0.31 [0.29, 0.34] | 0.04 [0.03, 0.05] | 43 / 1 | other answer |
| Mistral-7B | E1p e→a | names=True | 192 (96) | 0.22 [0.15, 0.29] | 0.00 [0.00, 0.00] | 0.47 [0.37, 0.57] | 0.01 [0.00, 0.02] | 15 / 0 | other answer |
| Mistral-7B | E1p e→a | order=1 | 1108 (1108) | 0.09 [0.07, 0.11] | 0.00 [0.00, 0.00] | 0.32 [0.30, 0.35] | 0.04 [0.03, 0.05] | 102 / 1 | other answer |
| Mistral-7B | E1p e→a | order=2 | 1108 (1108) | 0.09 [0.07, 0.11] | 0.00 [0.00, 0.01] | 0.33 [0.31, 0.36] | 0.04 [0.03, 0.05] | 100 / 3 | other answer |
| Mistral-7B | E1p e→a (all strata) | all | 2216 (1108) | 0.09 [0.08, 0.11] | 0.00 [0.00, 0.00] | 0.33 [0.30, 0.35] | 0.04 [0.03, 0.05] | 58 / 1 | other answer |
| Mistral-7B | E1p e→a (all strata) | antonym_type=complementary | 18 (9) | 0.17 [0.00, 0.39] | 0.00 [0.00, 0.00] | 0.17 [0.00, 0.39] | 0.00 [0.00, 0.00] | 1 / 0 | other answer |
| Mistral-7B | E1p e→a (all strata) | antonym_type=gradable | 2198 (1099) | 0.09 [0.08, 0.11] | 0.00 [0.00, 0.00] | 0.33 [0.31, 0.35] | 0.04 [0.03, 0.05] | 57 / 1 | other answer |
| Mistral-7B | E1p e→a (all strata) | degree_adv=False | 1064 (532) | 0.09 [0.07, 0.11] | 0.00 [0.00, 0.00] | 0.29 [0.26, 0.33] | 0.06 [0.04, 0.07] | 28 / 0 | other answer |
| Mistral-7B | E1p e→a (all strata) | degree_adv=True | 1152 (576) | 0.10 [0.08, 0.12] | 0.00 [0.00, 0.01] | 0.36 [0.33, 0.39] | 0.02 [0.01, 0.03] | 30 / 1 | other answer |
| Mistral-7B | E1p e→a (all strata) | names=False | 2024 (1012) | 0.08 [0.06, 0.09] | 0.00 [0.00, 0.00] | 0.31 [0.29, 0.34] | 0.04 [0.03, 0.05] | 43 / 1 | other answer |
| Mistral-7B | E1p e→a (all strata) | names=True | 192 (96) | 0.22 [0.15, 0.29] | 0.00 [0.00, 0.00] | 0.47 [0.37, 0.57] | 0.01 [0.00, 0.02] | 15 / 0 | other answer |
| Mistral-7B | E1p e→a (all strata) | order=1 | 1108 (1108) | 0.09 [0.07, 0.11] | 0.00 [0.00, 0.00] | 0.32 [0.30, 0.35] | 0.04 [0.03, 0.05] | 102 / 1 | other answer |
| Mistral-7B | E1p e→a (all strata) | order=2 | 1108 (1108) | 0.09 [0.07, 0.11] | 0.00 [0.00, 0.01] | 0.33 [0.31, 0.36] | 0.04 [0.03, 0.05] | 100 / 3 | other answer |
| Mistral-7B | E2 e→a | all | 3144 (1572) | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.00] | 0.33 [0.31, 0.35] | 0.04 [0.04, 0.05] | 25 / 1 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | antonym_type=complementary | 38 (19) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.26 [0.13, 0.39] | 0.05 [0.00, 0.13] | 0 / 0 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | antonym_type=gradable | 3106 (1553) | 0.04 [0.04, 0.05] | 0.00 [0.00, 0.00] | 0.33 [0.31, 0.35] | 0.04 [0.04, 0.05] | 25 / 1 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | degree_adv=False | 1596 (798) | 0.05 [0.04, 0.06] | 0.00 [0.00, 0.01] | 0.35 [0.32, 0.37] | 0.03 [0.02, 0.04] | 13 / 1 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | degree_adv=True | 1548 (774) | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.00] | 0.31 [0.28, 0.34] | 0.06 [0.05, 0.07] | 12 / 0 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | names=False | 2838 (1419) | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.00] | 0.33 [0.31, 0.35] | 0.04 [0.04, 0.05] | 22 / 1 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | names=True | 306 (153) | 0.05 [0.03, 0.08] | 0.01 [0.00, 0.02] | 0.32 [0.27, 0.38] | 0.05 [0.02, 0.08] | 3 / 0 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | order=1 | 1572 (1572) | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.00] | 0.33 [0.30, 0.35] | 0.04 [0.03, 0.05] | 67 / 2 | copy trap (word stated with the other entity) |
| Mistral-7B | E2 e→a | order=2 | 1572 (1572) | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.01] | 0.33 [0.31, 0.35] | 0.05 [0.04, 0.06] | 68 / 5 | copy trap (word stated with the other entity) |
| Mistral-7B | E3 e→a | all | 1636 (1636) | 0.47 [0.44, 0.49] | 0.00 [0.00, 0.00] | 0.64 [0.61, 0.66] | 0.01 [0.00, 0.01] | 765 / 1 | other answer |
| Mistral-7B | E3 e→a | antonym_type=complementary | 19 (19) | 0.37 [0.16, 0.58] | 0.00 [0.00, 0.00] | 0.53 [0.32, 0.74] | 0.05 [0.00, 0.16] | 7 / 0 | other answer |
| Mistral-7B | E3 e→a | antonym_type=gradable | 1617 (1617) | 0.47 [0.44, 0.49] | 0.00 [0.00, 0.00] | 0.64 [0.62, 0.66] | 0.01 [0.00, 0.01] | 758 / 1 | other answer |
| Mistral-7B | E3 e→a | degree_adv=False | 845 (845) | 0.41 [0.37, 0.44] | 0.00 [0.00, 0.00] | 0.59 [0.56, 0.62] | 0.01 [0.01, 0.02] | 343 / 1 | other answer |
| Mistral-7B | E3 e→a | degree_adv=True | 791 (791) | 0.53 [0.50, 0.57] | 0.00 [0.00, 0.00] | 0.69 [0.66, 0.72] | 0.00 [0.00, 0.00] | 422 / 0 | other answer |
| Mistral-7B | E3 e→a | names=False | 1483 (1483) | 0.49 [0.46, 0.51] | 0.00 [0.00, 0.00] | 0.66 [0.63, 0.68] | 0.01 [0.00, 0.01] | 724 / 0 | other answer |
| Mistral-7B | E3 e→a | names=True | 153 (153) | 0.27 [0.20, 0.33] | 0.01 [0.00, 0.02] | 0.46 [0.39, 0.55] | 0.01 [0.00, 0.03] | 41 / 1 | other answer |
| Mistral-7B | E3 e→a | order=1 | 1129 (1129) | 0.45 [0.42, 0.48] | 0.00 [0.00, 0.00] | 0.63 [0.60, 0.66] | 0.01 [0.00, 0.01] | 509 / 1 | other answer |
| Mistral-7B | E3 e→a | order=2 | 507 (507) | 0.50 [0.47, 0.55] | 0.00 [0.00, 0.00] | 0.66 [0.62, 0.70] | 0.00 [0.00, 0.01] | 256 / 0 | other answer |
| Mistral-7B | Set C agree (a→e) | all | 3288 (1644) | 0.11 [0.10, 0.12] | 0.00 [0.00, 0.00] | 0.44 [0.42, 0.45] | 0.01 [0.01, 0.02] | 69 / 0 | event / other |
| Mistral-7B | Set C agree (a→e) | antonym_type=complementary | 38 (19) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.24 [0.11, 0.39] | 0.00 [0.00, 0.00] | 0 / 0 | event / other |
| Mistral-7B | Set C agree (a→e) | antonym_type=gradable | 3250 (1625) | 0.11 [0.10, 0.13] | 0.00 [0.00, 0.00] | 0.44 [0.42, 0.46] | 0.01 [0.01, 0.02] | 69 / 0 | event / other |
| Mistral-7B | Set C agree (a→e) | degree_adv=False | 1708 (854) | 0.14 [0.12, 0.16] | 0.00 [0.00, 0.00] | 0.48 [0.46, 0.51] | 0.02 [0.01, 0.03] | 46 / 0 | event / other |
| Mistral-7B | Set C agree (a→e) | degree_adv=True | 1580 (790) | 0.08 [0.07, 0.10] | 0.00 [0.00, 0.00] | 0.38 [0.36, 0.41] | 0.01 [0.01, 0.02] | 23 / 0 | event / other |
| Mistral-7B | Set C agree (a→e) | names=False | 2980 (1490) | 0.12 [0.11, 0.13] | 0.00 [0.00, 0.00] | 0.45 [0.43, 0.47] | 0.01 [0.01, 0.02] | 69 / 0 | event / other |
| Mistral-7B | Set C agree (a→e) | names=True | 308 (154) | 0.03 [0.01, 0.05] | 0.00 [0.00, 0.00] | 0.27 [0.22, 0.33] | 0.05 [0.03, 0.07] | 0 / 0 | event / other |
| Mistral-7B | Set C agree (a→e) | order=1 | 1644 (1644) | 0.12 [0.10, 0.14] | 0.00 [0.00, 0.00] | 0.45 [0.42, 0.47] | 0.02 [0.01, 0.02] | 198 / 0 | event / other |
| Mistral-7B | Set C agree (a→e) | order=2 | 1644 (1644) | 0.10 [0.09, 0.12] | 0.00 [0.00, 0.00] | 0.43 [0.40, 0.45] | 0.01 [0.01, 0.02] | 170 / 0 | event / other |
| Mistral-7B | sc_conflict_event_a2e | all | 3288 (1644) | 0.01 [0.00, 0.01] | 0.02 [0.01, 0.02] | 0.12 [0.10, 0.13] | 0.17 [0.16, 0.19] | 3 / 4 | event / other |
| Mistral-7B | sc_conflict_event_a2e | antonym_type=complementary | 38 (19) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.08] | 0.03 [0.00, 0.08] | 0 / 0 | event / other |
| Mistral-7B | sc_conflict_event_a2e | antonym_type=gradable | 3250 (1625) | 0.01 [0.00, 0.01] | 0.02 [0.01, 0.02] | 0.12 [0.11, 0.13] | 0.18 [0.16, 0.19] | 3 / 4 | event / other |
| Mistral-7B | sc_conflict_event_a2e | degree_adv=False | 1708 (854) | 0.01 [0.01, 0.02] | 0.02 [0.02, 0.03] | 0.14 [0.12, 0.16] | 0.21 [0.19, 0.23] | 3 / 3 | event / other |
| Mistral-7B | sc_conflict_event_a2e | degree_adv=True | 1580 (790) | 0.00 [0.00, 0.01] | 0.01 [0.00, 0.01] | 0.09 [0.08, 0.11] | 0.14 [0.12, 0.16] | 0 / 1 | event / other |
| Mistral-7B | sc_conflict_event_a2e | names=False | 2980 (1490) | 0.01 [0.00, 0.01] | 0.02 [0.01, 0.02] | 0.12 [0.11, 0.13] | 0.18 [0.16, 0.19] | 3 / 4 | event / other |
| Mistral-7B | sc_conflict_event_a2e | names=True | 308 (154) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.11 [0.07, 0.15] | 0.12 [0.09, 0.16] | 0 / 0 | event / other |
| Mistral-7B | sc_conflict_event_a2e | order=1 | 1644 (1644) | 0.01 [0.00, 0.01] | 0.02 [0.01, 0.02] | 0.12 [0.11, 0.14] | 0.18 [0.16, 0.20] | 13 / 25 | event / other |
| Mistral-7B | sc_conflict_event_a2e | order=2 | 1644 (1644) | 0.01 [0.00, 0.01] | 0.02 [0.01, 0.02] | 0.11 [0.10, 0.13] | 0.17 [0.15, 0.19] | 11 / 29 | event / other |
| Mistral-7B | sc_conflict_stmt_a2e | all | 3288 (1644) | 0.02 [0.01, 0.02] | 0.01 [0.00, 0.01] | 0.17 [0.16, 0.19] | 0.12 [0.10, 0.13] | 4 / 3 | event / other |
| Mistral-7B | sc_conflict_stmt_a2e | antonym_type=complementary | 38 (19) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.08] | 0.03 [0.00, 0.08] | 0 / 0 | event / other |
| Mistral-7B | sc_conflict_stmt_a2e | antonym_type=gradable | 3250 (1625) | 0.02 [0.01, 0.02] | 0.01 [0.00, 0.01] | 0.18 [0.16, 0.19] | 0.12 [0.11, 0.13] | 4 / 3 | event / other |
| Mistral-7B | sc_conflict_stmt_a2e | degree_adv=False | 1708 (854) | 0.02 [0.02, 0.03] | 0.01 [0.01, 0.02] | 0.21 [0.19, 0.23] | 0.14 [0.12, 0.16] | 3 / 3 | event / other |
| Mistral-7B | sc_conflict_stmt_a2e | degree_adv=True | 1580 (790) | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.14 [0.12, 0.16] | 0.09 [0.08, 0.11] | 1 / 0 | event / other |
| Mistral-7B | sc_conflict_stmt_a2e | names=False | 2980 (1490) | 0.02 [0.01, 0.02] | 0.01 [0.00, 0.01] | 0.18 [0.16, 0.19] | 0.12 [0.11, 0.13] | 4 / 3 | event / other |
| Mistral-7B | sc_conflict_stmt_a2e | names=True | 308 (154) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.12 [0.09, 0.16] | 0.11 [0.07, 0.15] | 0 / 0 | event / other |
| Mistral-7B | sc_conflict_stmt_a2e | order=1 | 1644 (1644) | 0.02 [0.01, 0.02] | 0.01 [0.00, 0.01] | 0.18 [0.16, 0.20] | 0.12 [0.11, 0.14] | 25 / 13 | event / other |
| Mistral-7B | sc_conflict_stmt_a2e | order=2 | 1644 (1644) | 0.02 [0.01, 0.02] | 0.01 [0.00, 0.01] | 0.17 [0.15, 0.19] | 0.11 [0.10, 0.13] | 29 / 11 | event / other |
| Mixtral-8x7B | E1 a→e | all | 3172 (1586) | 0.25 [0.23, 0.27] | 0.00 [0.00, 0.00] | 0.68 [0.67, 0.70] | 0.01 [0.00, 0.01] | 211 / 0 | other answer |
| Mixtral-8x7B | E1 a→e | antonym_type=complementary | 38 (19) | 0.03 [0.00, 0.08] | 0.00 [0.00, 0.00] | 0.50 [0.34, 0.63] | 0.05 [0.00, 0.13] | 0 / 0 | other answer |
| Mixtral-8x7B | E1 a→e | antonym_type=gradable | 3134 (1567) | 0.25 [0.23, 0.27] | 0.00 [0.00, 0.00] | 0.69 [0.67, 0.70] | 0.01 [0.00, 0.01] | 211 / 0 | other answer |
| Mixtral-8x7B | E1 a→e | degree_adv=False | 1622 (811) | 0.21 [0.19, 0.24] | 0.00 [0.00, 0.00] | 0.66 [0.64, 0.69] | 0.01 [0.01, 0.02] | 83 / 0 | other answer |
| Mixtral-8x7B | E1 a→e | degree_adv=True | 1550 (775) | 0.29 [0.26, 0.31] | 0.00 [0.00, 0.00] | 0.71 [0.68, 0.73] | 0.00 [0.00, 0.00] | 128 / 0 | other answer |
| Mixtral-8x7B | E1 a→e | names=False | 2860 (1430) | 0.26 [0.24, 0.28] | 0.00 [0.00, 0.00] | 0.70 [0.68, 0.71] | 0.00 [0.00, 0.01] | 196 / 0 | other answer |
| Mixtral-8x7B | E1 a→e | names=True | 312 (156) | 0.17 [0.12, 0.22] | 0.00 [0.00, 0.00] | 0.58 [0.52, 0.64] | 0.02 [0.01, 0.04] | 15 / 0 | other answer |
| Mixtral-8x7B | E1 a→e | order=1 | 1586 (1586) | 0.25 [0.23, 0.27] | 0.00 [0.00, 0.00] | 0.68 [0.66, 0.71] | 0.01 [0.00, 0.01] | 402 / 0 | other answer |
| Mixtral-8x7B | E1 a→e | order=2 | 1586 (1586) | 0.25 [0.22, 0.27] | 0.00 [0.00, 0.00] | 0.68 [0.66, 0.71] | 0.01 [0.00, 0.01] | 389 / 0 | other answer |
| Mixtral-8x7B | E1 a→e (all strata) | all | 3172 (1586) | 0.25 [0.23, 0.27] | 0.00 [0.00, 0.00] | 0.68 [0.67, 0.70] | 0.01 [0.00, 0.01] | 211 / 0 | other answer |
| Mixtral-8x7B | E1 a→e (all strata) | antonym_type=complementary | 38 (19) | 0.03 [0.00, 0.08] | 0.00 [0.00, 0.00] | 0.50 [0.34, 0.63] | 0.05 [0.00, 0.13] | 0 / 0 | other answer |
| Mixtral-8x7B | E1 a→e (all strata) | antonym_type=gradable | 3134 (1567) | 0.25 [0.23, 0.27] | 0.00 [0.00, 0.00] | 0.69 [0.67, 0.70] | 0.01 [0.00, 0.01] | 211 / 0 | other answer |
| Mixtral-8x7B | E1 a→e (all strata) | degree_adv=False | 1622 (811) | 0.21 [0.19, 0.24] | 0.00 [0.00, 0.00] | 0.66 [0.64, 0.69] | 0.01 [0.01, 0.02] | 83 / 0 | other answer |
| Mixtral-8x7B | E1 a→e (all strata) | degree_adv=True | 1550 (775) | 0.29 [0.26, 0.31] | 0.00 [0.00, 0.00] | 0.71 [0.68, 0.73] | 0.00 [0.00, 0.00] | 128 / 0 | other answer |
| Mixtral-8x7B | E1 a→e (all strata) | names=False | 2860 (1430) | 0.26 [0.24, 0.28] | 0.00 [0.00, 0.00] | 0.70 [0.68, 0.71] | 0.00 [0.00, 0.01] | 196 / 0 | other answer |
| Mixtral-8x7B | E1 a→e (all strata) | names=True | 312 (156) | 0.17 [0.12, 0.22] | 0.00 [0.00, 0.00] | 0.58 [0.52, 0.64] | 0.02 [0.01, 0.04] | 15 / 0 | other answer |
| Mixtral-8x7B | E1 a→e (all strata) | order=1 | 1586 (1586) | 0.25 [0.23, 0.27] | 0.00 [0.00, 0.00] | 0.68 [0.66, 0.71] | 0.01 [0.00, 0.01] | 402 / 0 | other answer |
| Mixtral-8x7B | E1 a→e (all strata) | order=2 | 1586 (1586) | 0.25 [0.22, 0.27] | 0.00 [0.00, 0.00] | 0.68 [0.66, 0.71] | 0.01 [0.00, 0.01] | 389 / 0 | other answer |
| Mixtral-8x7B | E1p a→e | all | 2220 (1110) | 0.12 [0.11, 0.14] | 0.00 [0.00, 0.00] | 0.49 [0.47, 0.52] | 0.01 [0.00, 0.01] | 55 / 0 | other answer |
| Mixtral-8x7B | E1p a→e | antonym_type=complementary | 18 (9) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.28 [0.11, 0.44] | 0.00 [0.00, 0.00] | 0 / 0 | other answer |
| Mixtral-8x7B | E1p a→e | antonym_type=gradable | 2202 (1101) | 0.13 [0.11, 0.14] | 0.00 [0.00, 0.00] | 0.50 [0.47, 0.52] | 0.01 [0.00, 0.01] | 55 / 0 | other answer |
| Mixtral-8x7B | E1p a→e | degree_adv=False | 1068 (534) | 0.12 [0.10, 0.14] | 0.00 [0.00, 0.00] | 0.52 [0.49, 0.56] | 0.01 [0.00, 0.01] | 20 / 0 | other answer |
| Mixtral-8x7B | E1p a→e | degree_adv=True | 1152 (576) | 0.13 [0.10, 0.15] | 0.00 [0.00, 0.00] | 0.47 [0.43, 0.50] | 0.01 [0.00, 0.01] | 35 / 0 | other answer |
| Mixtral-8x7B | E1p a→e | names=False | 2028 (1014) | 0.13 [0.11, 0.14] | 0.00 [0.00, 0.00] | 0.50 [0.47, 0.52] | 0.01 [0.00, 0.01] | 50 / 0 | other answer |
| Mixtral-8x7B | E1p a→e | names=True | 192 (96) | 0.12 [0.07, 0.18] | 0.00 [0.00, 0.00] | 0.45 [0.36, 0.54] | 0.00 [0.00, 0.00] | 5 / 0 | other answer |
| Mixtral-8x7B | E1p a→e | order=1 | 1110 (1110) | 0.12 [0.10, 0.14] | 0.00 [0.00, 0.00] | 0.50 [0.47, 0.53] | 0.01 [0.00, 0.01] | 135 / 0 | other answer |
| Mixtral-8x7B | E1p a→e | order=2 | 1110 (1110) | 0.13 [0.11, 0.15] | 0.00 [0.00, 0.00] | 0.49 [0.46, 0.52] | 0.01 [0.00, 0.01] | 142 / 0 | other answer |
| Mixtral-8x7B | E2 a→e | all | 3172 (1586) | 0.00 [0.00, 0.01] | 0.03 [0.02, 0.04] | 0.06 [0.05, 0.07] | 0.29 [0.27, 0.31] | 2 / 15 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | antonym_type=complementary | 38 (19) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.13 [0.03, 0.24] | 0.13 [0.03, 0.24] | 0 / 0 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | antonym_type=gradable | 3134 (1567) | 0.00 [0.00, 0.01] | 0.03 [0.02, 0.04] | 0.06 [0.05, 0.07] | 0.29 [0.27, 0.31] | 2 / 15 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | degree_adv=False | 1622 (811) | 0.00 [0.00, 0.01] | 0.02 [0.01, 0.03] | 0.08 [0.07, 0.10] | 0.23 [0.21, 0.26] | 1 / 7 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | degree_adv=True | 1550 (775) | 0.00 [0.00, 0.01] | 0.04 [0.03, 0.05] | 0.04 [0.03, 0.05] | 0.34 [0.32, 0.37] | 1 / 8 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | names=False | 2860 (1430) | 0.00 [0.00, 0.01] | 0.03 [0.02, 0.04] | 0.06 [0.05, 0.07] | 0.27 [0.25, 0.29] | 2 / 14 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | names=True | 312 (156) | 0.00 [0.00, 0.00] | 0.03 [0.01, 0.05] | 0.04 [0.02, 0.06] | 0.44 [0.38, 0.49] | 0 / 1 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | order=1 | 1586 (1586) | 0.00 [0.00, 0.01] | 0.03 [0.02, 0.04] | 0.06 [0.05, 0.07] | 0.29 [0.27, 0.32] | 7 / 48 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 a→e | order=2 | 1586 (1586) | 0.00 [0.00, 0.01] | 0.03 [0.02, 0.04] | 0.06 [0.05, 0.08] | 0.28 [0.26, 0.30] | 4 / 49 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E3 a→e (P3 frame) | all | 1572 (1572) | 0.20 [0.19, 0.22] | 0.00 [0.00, 0.01] | 0.40 [0.38, 0.43] | 0.02 [0.01, 0.03] | 322 / 5 | other answer |
| Mixtral-8x7B | E3 a→e (P3 frame) | antonym_type=complementary | 16 (16) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.25 [0.06, 0.44] | 0.00 [0.00, 0.00] | 0 / 0 | other answer |
| Mixtral-8x7B | E3 a→e (P3 frame) | antonym_type=gradable | 1556 (1556) | 0.21 [0.19, 0.23] | 0.00 [0.00, 0.01] | 0.40 [0.38, 0.43] | 0.02 [0.01, 0.03] | 322 / 5 | other answer |
| Mixtral-8x7B | E3 a→e (P3 frame) | degree_adv=False | 814 (814) | 0.16 [0.14, 0.19] | 0.00 [0.00, 0.01] | 0.38 [0.35, 0.42] | 0.03 [0.02, 0.04] | 133 / 4 | other answer |
| Mixtral-8x7B | E3 a→e (P3 frame) | degree_adv=True | 758 (758) | 0.25 [0.22, 0.28] | 0.00 [0.00, 0.00] | 0.43 [0.39, 0.46] | 0.01 [0.00, 0.02] | 189 / 1 | other answer |
| Mixtral-8x7B | E3 a→e (P3 frame) | names=False | 1423 (1423) | 0.21 [0.19, 0.23] | 0.00 [0.00, 0.01] | 0.39 [0.37, 0.42] | 0.02 [0.01, 0.03] | 297 / 5 | other answer |
| Mixtral-8x7B | E3 a→e (P3 frame) | names=True | 149 (149) | 0.17 [0.11, 0.23] | 0.00 [0.00, 0.00] | 0.50 [0.42, 0.58] | 0.02 [0.00, 0.05] | 25 / 0 | other answer |
| Mixtral-8x7B | E3 a→e (P3 frame) | order=1 | 1095 (1095) | 0.19 [0.16, 0.21] | 0.00 [0.00, 0.01] | 0.40 [0.37, 0.43] | 0.02 [0.01, 0.03] | 206 / 3 | other answer |
| Mixtral-8x7B | E3 a→e (P3 frame) | order=2 | 477 (477) | 0.24 [0.21, 0.28] | 0.00 [0.00, 0.01] | 0.41 [0.36, 0.45] | 0.02 [0.01, 0.03] | 116 / 2 | other answer |
| Mixtral-8x7B | E3 a→e (F1 frame, Phase-5 anchor) | all | 1642 (1642) | 0.16 [0.14, 0.18] | 0.00 [0.00, 0.00] | 0.37 [0.34, 0.39] | 0.01 [0.01, 0.02] | 263 / 0 | other answer |
| Mixtral-8x7B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=complementary | 19 (19) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.11 [0.00, 0.26] | 0.00 [0.00, 0.00] | 0 / 0 | other answer |
| Mixtral-8x7B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=gradable | 1623 (1623) | 0.16 [0.14, 0.18] | 0.00 [0.00, 0.00] | 0.37 [0.34, 0.39] | 0.01 [0.01, 0.02] | 263 / 0 | other answer |
| Mixtral-8x7B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=False | 854 (854) | 0.17 [0.14, 0.19] | 0.00 [0.00, 0.00] | 0.37 [0.34, 0.40] | 0.01 [0.01, 0.02] | 143 / 0 | other answer |
| Mixtral-8x7B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=True | 788 (788) | 0.15 [0.13, 0.18] | 0.00 [0.00, 0.00] | 0.36 [0.32, 0.39] | 0.01 [0.00, 0.02] | 120 / 0 | other answer |
| Mixtral-8x7B | E3 a→e (F1 frame, Phase-5 anchor) | names=False | 1488 (1488) | 0.17 [0.15, 0.19] | 0.00 [0.00, 0.00] | 0.38 [0.36, 0.41] | 0.01 [0.01, 0.02] | 259 / 0 | other answer |
| Mixtral-8x7B | E3 a→e (F1 frame, Phase-5 anchor) | names=True | 154 (154) | 0.03 [0.01, 0.05] | 0.00 [0.00, 0.00] | 0.19 [0.14, 0.26] | 0.02 [0.00, 0.05] | 4 / 0 | other answer |
| Mixtral-8x7B | E3 a→e (F1 frame, Phase-5 anchor) | order=1 | 1135 (1135) | 0.16 [0.14, 0.18] | 0.00 [0.00, 0.00] | 0.36 [0.33, 0.39] | 0.01 [0.01, 0.02] | 179 / 0 | other answer |
| Mixtral-8x7B | E3 a→e (F1 frame, Phase-5 anchor) | order=2 | 507 (507) | 0.17 [0.13, 0.20] | 0.00 [0.00, 0.00] | 0.38 [0.34, 0.42] | 0.01 [0.00, 0.01] | 84 / 0 | other answer |
| Mixtral-8x7B | E0 prior only a→e | all | 64 (32) | 0.30 [0.19, 0.44] | 0.00 [0.00, 0.00] | 0.72 [0.59, 0.84] | 0.00 [0.00, 0.00] | 5 / 0 | other answer |
| Mixtral-8x7B | E0 prior only a→e | order=1 | 32 (32) | 0.34 [0.19, 0.53] | 0.00 [0.00, 0.00] | 0.72 [0.56, 0.88] | 0.00 [0.00, 0.00] | 11 / 0 | other answer |
| Mixtral-8x7B | E0 prior only a→e | order=2 | 32 (32) | 0.25 [0.09, 0.41] | 0.00 [0.00, 0.00] | 0.72 [0.56, 0.88] | 0.00 [0.00, 0.00] | 8 / 0 | other answer |
| Mixtral-8x7B | A-agree a→e | all | 62 (31) | 0.44 [0.27, 0.60] | 0.02 [0.00, 0.05] | 0.79 [0.65, 0.90] | 0.02 [0.00, 0.05] | 10 / 0 | other answer |
| Mixtral-8x7B | A-agree a→e | order=1 | 31 (31) | 0.45 [0.26, 0.65] | 0.03 [0.00, 0.10] | 0.81 [0.65, 0.94] | 0.03 [0.00, 0.10] | 14 / 1 | other answer |
| Mixtral-8x7B | A-agree a→e | order=2 | 31 (31) | 0.42 [0.23, 0.58] | 0.00 [0.00, 0.00] | 0.77 [0.61, 0.90] | 0.00 [0.00, 0.00] | 13 / 0 | other answer |
| Mixtral-8x7B | A-conflict a→e | all | 24 (12) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.46 [0.21, 0.71] | 0.08 [0.00, 0.21] | 0 / 0 | prior |
| Mixtral-8x7B | A-conflict a→e | order=1 | 12 (12) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.58 [0.33, 0.83] | 0.00 [0.00, 0.00] | 0 / 0 | prior |
| Mixtral-8x7B | A-conflict a→e | order=2 | 12 (12) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.33 [0.08, 0.58] | 0.17 [0.00, 0.42] | 0 / 0 | prior |
| Mixtral-8x7B | E1 e→a | all | 3144 (1572) | 0.15 [0.13, 0.16] | 0.01 [0.01, 0.02] | 0.42 [0.40, 0.44] | 0.08 [0.07, 0.09] | 103 / 1 | other answer |
| Mixtral-8x7B | E1 e→a | antonym_type=complementary | 38 (19) | 0.08 [0.00, 0.18] | 0.05 [0.00, 0.13] | 0.37 [0.21, 0.53] | 0.18 [0.05, 0.37] | 0 / 0 | other answer |
| Mixtral-8x7B | E1 e→a | antonym_type=gradable | 3106 (1553) | 0.15 [0.13, 0.16] | 0.01 [0.01, 0.01] | 0.42 [0.40, 0.44] | 0.08 [0.07, 0.09] | 103 / 1 | other answer |
| Mixtral-8x7B | E1 e→a | degree_adv=False | 1596 (798) | 0.08 [0.06, 0.09] | 0.02 [0.01, 0.03] | 0.27 [0.25, 0.30] | 0.13 [0.12, 0.15] | 21 / 1 | other answer |
| Mixtral-8x7B | E1 e→a | degree_adv=True | 1548 (774) | 0.21 [0.19, 0.24] | 0.00 [0.00, 0.01] | 0.57 [0.54, 0.60] | 0.03 [0.02, 0.04] | 82 / 0 | other answer |
| Mixtral-8x7B | E1 e→a | names=False | 2838 (1419) | 0.15 [0.13, 0.17] | 0.01 [0.01, 0.01] | 0.43 [0.41, 0.45] | 0.08 [0.07, 0.09] | 97 / 1 | other answer |
| Mixtral-8x7B | E1 e→a | names=True | 306 (153) | 0.10 [0.07, 0.15] | 0.02 [0.01, 0.04] | 0.30 [0.24, 0.37] | 0.13 [0.09, 0.17] | 6 / 0 | other answer |
| Mixtral-8x7B | E1 e→a | order=1 | 1572 (1572) | 0.13 [0.12, 0.15] | 0.01 [0.01, 0.01] | 0.40 [0.38, 0.43] | 0.08 [0.06, 0.09] | 212 / 15 | other answer |
| Mixtral-8x7B | E1 e→a | order=2 | 1572 (1572) | 0.16 [0.14, 0.17] | 0.01 [0.01, 0.02] | 0.43 [0.41, 0.46] | 0.09 [0.07, 0.10] | 245 / 20 | other answer |
| Mixtral-8x7B | E1 e→a (all strata) | all | 3144 (1572) | 0.15 [0.13, 0.16] | 0.01 [0.01, 0.02] | 0.42 [0.40, 0.44] | 0.08 [0.07, 0.09] | 103 / 1 | other answer |
| Mixtral-8x7B | E1 e→a (all strata) | antonym_type=complementary | 38 (19) | 0.08 [0.00, 0.18] | 0.05 [0.00, 0.13] | 0.37 [0.21, 0.53] | 0.18 [0.05, 0.37] | 0 / 0 | other answer |
| Mixtral-8x7B | E1 e→a (all strata) | antonym_type=gradable | 3106 (1553) | 0.15 [0.13, 0.16] | 0.01 [0.01, 0.01] | 0.42 [0.40, 0.44] | 0.08 [0.07, 0.09] | 103 / 1 | other answer |
| Mixtral-8x7B | E1 e→a (all strata) | degree_adv=False | 1596 (798) | 0.08 [0.06, 0.09] | 0.02 [0.01, 0.03] | 0.27 [0.25, 0.30] | 0.13 [0.12, 0.15] | 21 / 1 | other answer |
| Mixtral-8x7B | E1 e→a (all strata) | degree_adv=True | 1548 (774) | 0.21 [0.19, 0.24] | 0.00 [0.00, 0.01] | 0.57 [0.54, 0.60] | 0.03 [0.02, 0.04] | 82 / 0 | other answer |
| Mixtral-8x7B | E1 e→a (all strata) | names=False | 2838 (1419) | 0.15 [0.13, 0.17] | 0.01 [0.01, 0.01] | 0.43 [0.41, 0.45] | 0.08 [0.07, 0.09] | 97 / 1 | other answer |
| Mixtral-8x7B | E1 e→a (all strata) | names=True | 306 (153) | 0.10 [0.07, 0.15] | 0.02 [0.01, 0.04] | 0.30 [0.24, 0.37] | 0.13 [0.09, 0.17] | 6 / 0 | other answer |
| Mixtral-8x7B | E1 e→a (all strata) | order=1 | 1572 (1572) | 0.13 [0.12, 0.15] | 0.01 [0.01, 0.01] | 0.40 [0.38, 0.43] | 0.08 [0.06, 0.09] | 212 / 15 | other answer |
| Mixtral-8x7B | E1 e→a (all strata) | order=2 | 1572 (1572) | 0.16 [0.14, 0.17] | 0.01 [0.01, 0.02] | 0.43 [0.41, 0.46] | 0.09 [0.07, 0.10] | 245 / 20 | other answer |
| Mixtral-8x7B | E1p e→a | all | 2216 (1108) | 0.16 [0.14, 0.18] | 0.00 [0.00, 0.00] | 0.38 [0.36, 0.41] | 0.04 [0.03, 0.05] | 112 / 0 | other answer |
| Mixtral-8x7B | E1p e→a | antonym_type=complementary | 18 (9) | 0.17 [0.00, 0.39] | 0.00 [0.00, 0.00] | 0.28 [0.05, 0.56] | 0.00 [0.00, 0.00] | 1 / 0 | other answer |
| Mixtral-8x7B | E1p e→a | antonym_type=gradable | 2198 (1099) | 0.16 [0.15, 0.18] | 0.00 [0.00, 0.01] | 0.38 [0.36, 0.41] | 0.04 [0.03, 0.05] | 111 / 0 | other answer |
| Mixtral-8x7B | E1p e→a | degree_adv=False | 1064 (532) | 0.13 [0.10, 0.16] | 0.00 [0.00, 0.01] | 0.36 [0.32, 0.40] | 0.06 [0.05, 0.08] | 38 / 0 | other answer |
| Mixtral-8x7B | E1p e→a | degree_adv=True | 1152 (576) | 0.20 [0.17, 0.23] | 0.00 [0.00, 0.00] | 0.41 [0.37, 0.44] | 0.01 [0.01, 0.02] | 74 / 0 | other answer |
| Mixtral-8x7B | E1p e→a | names=False | 2024 (1012) | 0.16 [0.14, 0.17] | 0.00 [0.00, 0.01] | 0.37 [0.34, 0.40] | 0.04 [0.03, 0.05] | 92 / 0 | other answer |
| Mixtral-8x7B | E1p e→a | names=True | 192 (96) | 0.27 [0.19, 0.36] | 0.00 [0.00, 0.00] | 0.53 [0.43, 0.62] | 0.01 [0.00, 0.02] | 20 / 0 | other answer |
| Mixtral-8x7B | E1p e→a | order=1 | 1108 (1108) | 0.17 [0.14, 0.19] | 0.00 [0.00, 0.00] | 0.40 [0.37, 0.43] | 0.03 [0.02, 0.04] | 184 / 2 | other answer |
| Mixtral-8x7B | E1p e→a | order=2 | 1108 (1108) | 0.16 [0.14, 0.19] | 0.00 [0.00, 0.01] | 0.37 [0.34, 0.40] | 0.04 [0.03, 0.06] | 181 / 4 | other answer |
| Mixtral-8x7B | E2 e→a | all | 3144 (1572) | 0.02 [0.02, 0.03] | 0.00 [0.00, 0.01] | 0.18 [0.16, 0.19] | 0.06 [0.05, 0.06] | 11 / 2 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | antonym_type=complementary | 38 (19) | 0.03 [0.00, 0.08] | 0.00 [0.00, 0.00] | 0.16 [0.05, 0.26] | 0.03 [0.00, 0.08] | 0 / 0 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | antonym_type=gradable | 3106 (1553) | 0.02 [0.02, 0.03] | 0.00 [0.00, 0.01] | 0.18 [0.16, 0.19] | 0.06 [0.05, 0.06] | 11 / 2 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | degree_adv=False | 1596 (798) | 0.02 [0.01, 0.02] | 0.00 [0.00, 0.01] | 0.19 [0.17, 0.21] | 0.03 [0.02, 0.04] | 1 / 0 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | degree_adv=True | 1548 (774) | 0.03 [0.02, 0.04] | 0.01 [0.00, 0.01] | 0.16 [0.13, 0.18] | 0.08 [0.07, 0.10] | 10 / 2 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | names=False | 2838 (1419) | 0.02 [0.02, 0.03] | 0.00 [0.00, 0.01] | 0.17 [0.15, 0.18] | 0.06 [0.05, 0.07] | 11 / 2 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | names=True | 306 (153) | 0.04 [0.02, 0.06] | 0.01 [0.00, 0.03] | 0.25 [0.20, 0.29] | 0.05 [0.03, 0.08] | 0 / 0 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | order=1 | 1572 (1572) | 0.02 [0.02, 0.03] | 0.00 [0.00, 0.01] | 0.18 [0.16, 0.20] | 0.05 [0.04, 0.06] | 36 / 4 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E2 e→a | order=2 | 1572 (1572) | 0.02 [0.01, 0.03] | 0.01 [0.00, 0.01] | 0.17 [0.15, 0.18] | 0.06 [0.05, 0.07] | 34 / 11 | copy trap (word stated with the other entity) |
| Mixtral-8x7B | E3 e→a | all | 630 (630) | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 627 / 0 | other answer |
| Mixtral-8x7B | E3 e→a | antonym_type=gradable | 626 (626) | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 623 / 0 | other answer |
| Mixtral-8x7B | E3 e→a | degree_adv=False | 275 (275) | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 274 / 0 | other answer |
| Mixtral-8x7B | E3 e→a | degree_adv=True | 355 (355) | 0.99 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 353 / 0 | other answer |
| Mixtral-8x7B | E3 e→a | names=False | 602 (602) | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 599 / 0 | other answer |
| Mixtral-8x7B | E3 e→a | names=True | 28 (28) | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 28 / 0 | other answer |
| Mixtral-8x7B | E3 e→a | order=1 | 402 (402) | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 401 / 0 | other answer |
| Mixtral-8x7B | E3 e→a | order=2 | 228 (228) | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 226 / 0 | other answer |
| Mixtral-8x7B | E0 prior only e→a | all | 62 (31) | 0.18 [0.06, 0.31] | 0.00 [0.00, 0.00] | 0.48 [0.32, 0.63] | 0.00 [0.00, 0.00] | 4 / 0 | other answer |
| Mixtral-8x7B | E0 prior only e→a | order=1 | 31 (31) | 0.16 [0.03, 0.29] | 0.00 [0.00, 0.00] | 0.55 [0.39, 0.71] | 0.00 [0.00, 0.00] | 5 / 0 | other answer |
| Mixtral-8x7B | E0 prior only e→a | order=2 | 31 (31) | 0.19 [0.06, 0.35] | 0.00 [0.00, 0.00] | 0.42 [0.26, 0.58] | 0.00 [0.00, 0.00] | 6 / 0 | other answer |
| Mixtral-8x7B | A-agree e→a | all | 60 (30) | 0.32 [0.17, 0.47] | 0.00 [0.00, 0.00] | 0.67 [0.53, 0.80] | 0.07 [0.00, 0.15] | 6 / 0 | other answer |
| Mixtral-8x7B | A-agree e→a | order=1 | 30 (30) | 0.20 [0.07, 0.37] | 0.00 [0.00, 0.00] | 0.67 [0.50, 0.83] | 0.03 [0.00, 0.10] | 6 / 0 | other answer |
| Mixtral-8x7B | A-agree e→a | order=2 | 30 (30) | 0.43 [0.27, 0.63] | 0.00 [0.00, 0.00] | 0.67 [0.50, 0.83] | 0.10 [0.00, 0.23] | 13 / 0 | other answer |
| Mixtral-8x7B | A-conflict e→a | all | 24 (12) | 0.04 [0.00, 0.12] | 0.00 [0.00, 0.00] | 0.21 [0.04, 0.38] | 0.08 [0.00, 0.21] | 0 / 0 | prior |
| Mixtral-8x7B | A-conflict e→a | order=1 | 12 (12) | 0.08 [0.00, 0.25] | 0.00 [0.00, 0.00] | 0.33 [0.08, 0.58] | 0.17 [0.00, 0.42] | 1 / 0 | prior |
| Mixtral-8x7B | A-conflict e→a | order=2 | 12 (12) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.08 [0.00, 0.25] | 0.00 [0.00, 0.00] | 0 / 0 | prior |
| Mixtral-8x7B | Set C agree (a→e) | all | 3288 (1644) | 0.19 [0.18, 0.21] | 0.00 [0.00, 0.00] | 0.53 [0.51, 0.55] | 0.01 [0.00, 0.01] | 179 / 0 | event / other |
| Mixtral-8x7B | Set C agree (a→e) | antonym_type=complementary | 38 (19) | 0.05 [0.00, 0.16] | 0.00 [0.00, 0.00] | 0.39 [0.24, 0.58] | 0.00 [0.00, 0.00] | 1 / 0 | event / other |
| Mixtral-8x7B | Set C agree (a→e) | antonym_type=gradable | 3250 (1625) | 0.20 [0.18, 0.21] | 0.00 [0.00, 0.00] | 0.54 [0.52, 0.56] | 0.01 [0.00, 0.01] | 178 / 0 | event / other |
| Mixtral-8x7B | Set C agree (a→e) | degree_adv=False | 1708 (854) | 0.19 [0.17, 0.21] | 0.00 [0.00, 0.00] | 0.54 [0.51, 0.57] | 0.00 [0.00, 0.01] | 96 / 0 | event / other |
| Mixtral-8x7B | Set C agree (a→e) | degree_adv=True | 1580 (790) | 0.20 [0.18, 0.22] | 0.00 [0.00, 0.00] | 0.53 [0.50, 0.56] | 0.01 [0.00, 0.01] | 83 / 0 | event / other |
| Mixtral-8x7B | Set C agree (a→e) | names=False | 2980 (1490) | 0.21 [0.19, 0.23] | 0.00 [0.00, 0.00] | 0.55 [0.53, 0.57] | 0.00 [0.00, 0.01] | 178 / 0 | event / other |
| Mixtral-8x7B | Set C agree (a→e) | names=True | 308 (154) | 0.06 [0.04, 0.09] | 0.00 [0.00, 0.00] | 0.39 [0.33, 0.45] | 0.02 [0.01, 0.04] | 1 / 0 | event / other |
| Mixtral-8x7B | Set C agree (a→e) | order=1 | 1644 (1644) | 0.20 [0.18, 0.22] | 0.00 [0.00, 0.00] | 0.55 [0.52, 0.57] | 0.01 [0.00, 0.01] | 325 / 0 | event / other |
| Mixtral-8x7B | Set C agree (a→e) | order=2 | 1644 (1644) | 0.19 [0.17, 0.21] | 0.00 [0.00, 0.00] | 0.52 [0.50, 0.54] | 0.00 [0.00, 0.01] | 314 / 0 | event / other |
| Mixtral-8x7B | sc_conflict_event_a2e | all | 3288 (1644) | 0.01 [0.01, 0.01] | 0.03 [0.02, 0.03] | 0.10 [0.09, 0.11] | 0.16 [0.15, 0.18] | 6 / 19 | event / other |
| Mixtral-8x7B | sc_conflict_event_a2e | antonym_type=complementary | 38 (19) | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.08] | 0.11 [0.00, 0.24] | 0.16 [0.05, 0.29] | 0 / 0 | event / other |
| Mixtral-8x7B | sc_conflict_event_a2e | antonym_type=gradable | 3250 (1625) | 0.01 [0.01, 0.01] | 0.03 [0.02, 0.03] | 0.10 [0.09, 0.11] | 0.16 [0.15, 0.18] | 6 / 19 | event / other |
| Mixtral-8x7B | sc_conflict_event_a2e | degree_adv=False | 1708 (854) | 0.01 [0.01, 0.02] | 0.03 [0.02, 0.04] | 0.14 [0.12, 0.16] | 0.19 [0.17, 0.21] | 4 / 15 | event / other |
| Mixtral-8x7B | sc_conflict_event_a2e | degree_adv=True | 1580 (790) | 0.00 [0.00, 0.01] | 0.02 [0.01, 0.03] | 0.06 [0.04, 0.07] | 0.14 [0.12, 0.16] | 2 / 4 | event / other |
| Mixtral-8x7B | sc_conflict_event_a2e | names=False | 2980 (1490) | 0.01 [0.01, 0.01] | 0.03 [0.02, 0.03] | 0.10 [0.09, 0.12] | 0.16 [0.15, 0.18] | 6 / 18 | event / other |
| Mixtral-8x7B | sc_conflict_event_a2e | names=True | 308 (154) | 0.00 [0.00, 0.00] | 0.02 [0.00, 0.04] | 0.06 [0.04, 0.10] | 0.19 [0.15, 0.24] | 0 / 1 | event / other |
| Mixtral-8x7B | sc_conflict_event_a2e | order=1 | 1644 (1644) | 0.01 [0.00, 0.01] | 0.03 [0.02, 0.04] | 0.10 [0.09, 0.12] | 0.17 [0.15, 0.18] | 15 / 54 | event / other |
| Mixtral-8x7B | sc_conflict_event_a2e | order=2 | 1644 (1644) | 0.01 [0.00, 0.01] | 0.02 [0.01, 0.02] | 0.10 [0.08, 0.11] | 0.16 [0.14, 0.18] | 15 / 30 | event / other |
| Mixtral-8x7B | sc_conflict_stmt_a2e | all | 3288 (1644) | 0.03 [0.02, 0.03] | 0.01 [0.01, 0.01] | 0.16 [0.15, 0.18] | 0.10 [0.09, 0.11] | 19 / 6 | event / other |
| Mixtral-8x7B | sc_conflict_stmt_a2e | antonym_type=complementary | 38 (19) | 0.03 [0.00, 0.08] | 0.00 [0.00, 0.00] | 0.16 [0.05, 0.29] | 0.11 [0.00, 0.24] | 0 / 0 | event / other |
| Mixtral-8x7B | sc_conflict_stmt_a2e | antonym_type=gradable | 3250 (1625) | 0.03 [0.02, 0.03] | 0.01 [0.01, 0.01] | 0.16 [0.15, 0.18] | 0.10 [0.09, 0.11] | 19 / 6 | event / other |
| Mixtral-8x7B | sc_conflict_stmt_a2e | degree_adv=False | 1708 (854) | 0.03 [0.02, 0.04] | 0.01 [0.01, 0.02] | 0.19 [0.17, 0.21] | 0.14 [0.12, 0.16] | 15 / 4 | event / other |
| Mixtral-8x7B | sc_conflict_stmt_a2e | degree_adv=True | 1580 (790) | 0.02 [0.01, 0.03] | 0.00 [0.00, 0.01] | 0.14 [0.12, 0.16] | 0.06 [0.04, 0.07] | 4 / 2 | event / other |
| Mixtral-8x7B | sc_conflict_stmt_a2e | names=False | 2980 (1490) | 0.03 [0.02, 0.03] | 0.01 [0.01, 0.01] | 0.16 [0.15, 0.18] | 0.10 [0.09, 0.12] | 18 / 6 | event / other |
| Mixtral-8x7B | sc_conflict_stmt_a2e | names=True | 308 (154) | 0.02 [0.00, 0.04] | 0.00 [0.00, 0.00] | 0.19 [0.15, 0.24] | 0.06 [0.04, 0.10] | 1 / 0 | event / other |
| Mixtral-8x7B | sc_conflict_stmt_a2e | order=1 | 1644 (1644) | 0.03 [0.02, 0.04] | 0.01 [0.00, 0.01] | 0.17 [0.15, 0.18] | 0.10 [0.09, 0.12] | 54 / 15 | event / other |
| Mixtral-8x7B | sc_conflict_stmt_a2e | order=2 | 1644 (1644) | 0.02 [0.01, 0.02] | 0.01 [0.00, 0.01] | 0.16 [0.14, 0.18] | 0.10 [0.08, 0.11] | 30 / 15 | event / other |
| Qwen2-57B-A14B | E1 a→e | all | 1178 (589) | 0.77 [0.75, 0.80] | 0.00 [0.00, 0.00] | 0.96 [0.95, 0.97] | 0.00 [0.00, 0.00] | 374 / 0 | other answer |
| Qwen2-57B-A14B | E1 a→e | antonym_type=complementary | 10 (5) | 0.70 [0.30, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 3 / 0 | other answer |
| Qwen2-57B-A14B | E1 a→e | antonym_type=gradable | 1168 (584) | 0.77 [0.75, 0.80] | 0.00 [0.00, 0.00] | 0.96 [0.95, 0.97] | 0.00 [0.00, 0.00] | 371 / 0 | other answer |
| Qwen2-57B-A14B | E1 a→e | degree_adv=False | 490 (245) | 0.78 [0.74, 0.82] | 0.00 [0.00, 0.00] | 0.95 [0.93, 0.97] | 0.00 [0.00, 0.00] | 157 / 0 | other answer |
| Qwen2-57B-A14B | E1 a→e | degree_adv=True | 688 (344) | 0.76 [0.73, 0.80] | 0.00 [0.00, 0.00] | 0.96 [0.95, 0.98] | 0.00 [0.00, 0.00] | 217 / 0 | other answer |
| Qwen2-57B-A14B | E1 a→e | names=False | 1116 (558) | 0.76 [0.73, 0.79] | 0.00 [0.00, 0.00] | 0.96 [0.94, 0.97] | 0.00 [0.00, 0.00] | 343 / 0 | other answer |
| Qwen2-57B-A14B | E1 a→e | names=True | 62 (31) | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 31 / 0 | other answer |
| Qwen2-57B-A14B | E1 a→e | order=1 | 589 (589) | 0.78 [0.74, 0.81] | 0.00 [0.00, 0.00] | 0.95 [0.93, 0.97] | 0.00 [0.00, 0.00] | 457 / 0 | other answer |
| Qwen2-57B-A14B | E1 a→e | order=2 | 589 (589) | 0.77 [0.74, 0.80] | 0.00 [0.00, 0.00] | 0.97 [0.95, 0.98] | 0.00 [0.00, 0.00] | 453 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e | all | 970 (485) | 0.52 [0.48, 0.55] | 0.00 [0.00, 0.00] | 0.85 [0.83, 0.88] | 0.00 [0.00, 0.00] | 164 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e | antonym_type=gradable | 962 (481) | 0.52 [0.48, 0.55] | 0.00 [0.00, 0.00] | 0.85 [0.83, 0.88] | 0.00 [0.00, 0.00] | 162 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e | degree_adv=False | 392 (196) | 0.66 [0.61, 0.71] | 0.00 [0.00, 0.00] | 0.91 [0.88, 0.94] | 0.00 [0.00, 0.00] | 97 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e | degree_adv=True | 578 (289) | 0.42 [0.38, 0.47] | 0.00 [0.00, 0.00] | 0.81 [0.78, 0.85] | 0.00 [0.00, 0.00] | 67 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e | names=False | 918 (459) | 0.50 [0.46, 0.53] | 0.00 [0.00, 0.00] | 0.85 [0.82, 0.87] | 0.00 [0.00, 0.00] | 143 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e | names=True | 52 (26) | 0.90 [0.83, 0.96] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 21 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e | order=1 | 485 (485) | 0.52 [0.47, 0.56] | 0.00 [0.00, 0.00] | 0.86 [0.83, 0.89] | 0.00 [0.00, 0.00] | 252 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e | order=2 | 485 (485) | 0.52 [0.47, 0.56] | 0.00 [0.00, 0.00] | 0.85 [0.81, 0.88] | 0.00 [0.00, 0.00] | 251 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e (all strata) | all | 970 (485) | 0.52 [0.48, 0.55] | 0.00 [0.00, 0.00] | 0.85 [0.83, 0.88] | 0.00 [0.00, 0.00] | 164 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e (all strata) | antonym_type=gradable | 962 (481) | 0.52 [0.48, 0.55] | 0.00 [0.00, 0.00] | 0.85 [0.83, 0.88] | 0.00 [0.00, 0.00] | 162 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e (all strata) | degree_adv=False | 392 (196) | 0.66 [0.61, 0.71] | 0.00 [0.00, 0.00] | 0.91 [0.88, 0.94] | 0.00 [0.00, 0.00] | 97 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e (all strata) | degree_adv=True | 578 (289) | 0.42 [0.38, 0.47] | 0.00 [0.00, 0.00] | 0.81 [0.78, 0.85] | 0.00 [0.00, 0.00] | 67 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e (all strata) | names=False | 918 (459) | 0.50 [0.46, 0.53] | 0.00 [0.00, 0.00] | 0.85 [0.82, 0.87] | 0.00 [0.00, 0.00] | 143 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e (all strata) | names=True | 52 (26) | 0.90 [0.83, 0.96] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 21 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e (all strata) | order=1 | 485 (485) | 0.52 [0.47, 0.56] | 0.00 [0.00, 0.00] | 0.86 [0.83, 0.89] | 0.00 [0.00, 0.00] | 252 / 0 | other answer |
| Qwen2-57B-A14B | E1p a→e (all strata) | order=2 | 485 (485) | 0.52 [0.47, 0.56] | 0.00 [0.00, 0.00] | 0.85 [0.81, 0.88] | 0.00 [0.00, 0.00] | 251 / 0 | other answer |
| Qwen2-57B-A14B | E2 a→e | all | 1178 (589) | 0.00 [0.00, 0.01] | 0.08 [0.07, 0.10] | 0.04 [0.03, 0.06] | 0.54 [0.51, 0.57] | 0 / 14 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | antonym_type=complementary | 10 (5) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.20 [0.00, 0.40] | 0.40 [0.00, 0.80] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | antonym_type=gradable | 1168 (584) | 0.00 [0.00, 0.01] | 0.08 [0.07, 0.10] | 0.04 [0.03, 0.05] | 0.54 [0.51, 0.58] | 0 / 14 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | degree_adv=False | 490 (245) | 0.00 [0.00, 0.01] | 0.07 [0.05, 0.10] | 0.06 [0.04, 0.08] | 0.46 [0.40, 0.51] | 0 / 5 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | degree_adv=True | 688 (344) | 0.00 [0.00, 0.01] | 0.09 [0.07, 0.12] | 0.03 [0.02, 0.05] | 0.60 [0.56, 0.64] | 0 / 9 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | names=False | 1116 (558) | 0.00 [0.00, 0.01] | 0.06 [0.05, 0.08] | 0.05 [0.03, 0.06] | 0.53 [0.50, 0.56] | 0 / 8 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | names=True | 62 (31) | 0.00 [0.00, 0.00] | 0.45 [0.32, 0.56] | 0.00 [0.00, 0.00] | 0.82 [0.74, 0.90] | 0 / 6 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | order=1 | 589 (589) | 0.00 [0.00, 0.01] | 0.08 [0.06, 0.11] | 0.04 [0.02, 0.05] | 0.53 [0.49, 0.57] | 1 / 50 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 a→e | order=2 | 589 (589) | 0.00 [0.00, 0.01] | 0.08 [0.06, 0.10] | 0.05 [0.03, 0.07] | 0.56 [0.51, 0.59] | 2 / 48 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E3 a→e (P3 frame) | all | 1824 (1824) | 0.36 [0.34, 0.38] | 0.01 [0.00, 0.01] | 0.55 [0.53, 0.58] | 0.02 [0.01, 0.03] | 654 / 10 | other answer |
| Qwen2-57B-A14B | E3 a→e (P3 frame) | antonym_type=complementary | 24 (24) | 0.17 [0.04, 0.33] | 0.04 [0.00, 0.12] | 0.42 [0.21, 0.62] | 0.04 [0.00, 0.12] | 4 / 1 | other answer |
| Qwen2-57B-A14B | E3 a→e (P3 frame) | antonym_type=gradable | 1800 (1800) | 0.36 [0.34, 0.38] | 0.01 [0.00, 0.01] | 0.56 [0.53, 0.58] | 0.02 [0.01, 0.03] | 650 / 9 | other answer |
| Qwen2-57B-A14B | E3 a→e (P3 frame) | degree_adv=False | 1035 (1035) | 0.34 [0.31, 0.36] | 0.01 [0.00, 0.01] | 0.53 [0.50, 0.56] | 0.03 [0.02, 0.04] | 348 / 9 | other answer |
| Qwen2-57B-A14B | E3 a→e (P3 frame) | degree_adv=True | 789 (789) | 0.39 [0.36, 0.42] | 0.00 [0.00, 0.00] | 0.59 [0.55, 0.62] | 0.01 [0.01, 0.02] | 306 / 1 | other answer |
| Qwen2-57B-A14B | E3 a→e (P3 frame) | names=False | 1563 (1563) | 0.34 [0.32, 0.37] | 0.01 [0.00, 0.01] | 0.53 [0.50, 0.55] | 0.02 [0.01, 0.03] | 535 / 8 | other answer |
| Qwen2-57B-A14B | E3 a→e (P3 frame) | names=True | 261 (261) | 0.46 [0.39, 0.51] | 0.01 [0.00, 0.02] | 0.72 [0.67, 0.77] | 0.02 [0.01, 0.04] | 119 / 2 | other answer |
| Qwen2-57B-A14B | E3 a→e (P3 frame) | order=1 | 1323 (1323) | 0.36 [0.33, 0.39] | 0.01 [0.00, 0.01] | 0.55 [0.53, 0.58] | 0.02 [0.02, 0.03] | 478 / 10 | other answer |
| Qwen2-57B-A14B | E3 a→e (P3 frame) | order=2 | 501 (501) | 0.35 [0.31, 0.39] | 0.00 [0.00, 0.00] | 0.55 [0.51, 0.59] | 0.01 [0.00, 0.02] | 176 / 0 | other answer |
| Qwen2-57B-A14B | E3 a→e (F1 frame, Phase-5 anchor) | all | 643 (643) | 0.51 [0.47, 0.55] | 0.00 [0.00, 0.00] | 0.73 [0.69, 0.76] | 0.00 [0.00, 0.00] | 330 / 0 | other answer |
| Qwen2-57B-A14B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=gradable | 638 (638) | 0.51 [0.47, 0.55] | 0.00 [0.00, 0.00] | 0.73 [0.69, 0.76] | 0.00 [0.00, 0.00] | 328 / 0 | other answer |
| Qwen2-57B-A14B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=False | 285 (285) | 0.62 [0.56, 0.68] | 0.00 [0.00, 0.00] | 0.80 [0.75, 0.85] | 0.00 [0.00, 0.00] | 178 / 0 | other answer |
| Qwen2-57B-A14B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=True | 358 (358) | 0.42 [0.38, 0.47] | 0.00 [0.00, 0.00] | 0.66 [0.62, 0.71] | 0.00 [0.00, 0.00] | 152 / 0 | other answer |
| Qwen2-57B-A14B | E3 a→e (F1 frame, Phase-5 anchor) | names=False | 612 (612) | 0.52 [0.49, 0.56] | 0.00 [0.00, 0.00] | 0.74 [0.70, 0.77] | 0.00 [0.00, 0.00] | 321 / 0 | other answer |
| Qwen2-57B-A14B | E3 a→e (F1 frame, Phase-5 anchor) | names=True | 31 (31) | 0.29 [0.13, 0.45] | 0.00 [0.00, 0.00] | 0.55 [0.39, 0.74] | 0.00 [0.00, 0.00] | 9 / 0 | other answer |
| Qwen2-57B-A14B | E3 a→e (F1 frame, Phase-5 anchor) | order=1 | 415 (415) | 0.52 [0.47, 0.56] | 0.00 [0.00, 0.00] | 0.72 [0.68, 0.77] | 0.00 [0.00, 0.00] | 214 / 0 | other answer |
| Qwen2-57B-A14B | E3 a→e (F1 frame, Phase-5 anchor) | order=2 | 228 (228) | 0.51 [0.44, 0.57] | 0.00 [0.00, 0.00] | 0.73 [0.67, 0.79] | 0.00 [0.00, 0.00] | 116 / 0 | other answer |
| Qwen2-57B-A14B | E1 e→a | all | 3578 (1789) | 0.40 [0.38, 0.42] | 0.01 [0.00, 0.01] | 0.63 [0.61, 0.65] | 0.03 [0.02, 0.04] | 479 / 2 | other answer |
| Qwen2-57B-A14B | E1 e→a | antonym_type=complementary | 50 (25) | 0.22 [0.10, 0.36] | 0.00 [0.00, 0.00] | 0.48 [0.36, 0.62] | 0.10 [0.02, 0.22] | 3 / 0 | other answer |
| Qwen2-57B-A14B | E1 e→a | antonym_type=gradable | 3528 (1764) | 0.40 [0.38, 0.42] | 0.01 [0.00, 0.01] | 0.63 [0.61, 0.65] | 0.03 [0.02, 0.04] | 476 / 2 | other answer |
| Qwen2-57B-A14B | E1 e→a | degree_adv=False | 1986 (993) | 0.29 [0.27, 0.31] | 0.01 [0.01, 0.01] | 0.53 [0.50, 0.56] | 0.05 [0.04, 0.06] | 158 / 2 | other answer |
| Qwen2-57B-A14B | E1 e→a | degree_adv=True | 1592 (796) | 0.53 [0.50, 0.56] | 0.00 [0.00, 0.00] | 0.75 [0.73, 0.78] | 0.01 [0.00, 0.01] | 321 / 0 | other answer |
| Qwen2-57B-A14B | E1 e→a | names=False | 3060 (1530) | 0.41 [0.39, 0.44] | 0.00 [0.00, 0.01] | 0.65 [0.63, 0.67] | 0.02 [0.02, 0.03] | 429 / 0 | other answer |
| Qwen2-57B-A14B | E1 e→a | names=True | 518 (259) | 0.31 [0.26, 0.36] | 0.02 [0.01, 0.03] | 0.52 [0.47, 0.57] | 0.08 [0.05, 0.10] | 50 / 2 | other answer |
| Qwen2-57B-A14B | E1 e→a | order=1 | 1789 (1789) | 0.39 [0.37, 0.42] | 0.00 [0.00, 0.01] | 0.63 [0.61, 0.65] | 0.03 [0.02, 0.04] | 704 / 8 | other answer |
| Qwen2-57B-A14B | E1 e→a | order=2 | 1789 (1789) | 0.40 [0.38, 0.43] | 0.01 [0.00, 0.01] | 0.63 [0.61, 0.65] | 0.03 [0.02, 0.04] | 721 / 13 | other answer |
| Qwen2-57B-A14B | E1p e→a | all | 970 (485) | 0.53 [0.49, 0.57] | 0.00 [0.00, 0.00] | 0.73 [0.70, 0.77] | 0.01 [0.00, 0.01] | 191 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a | antonym_type=gradable | 962 (481) | 0.53 [0.49, 0.57] | 0.00 [0.00, 0.00] | 0.73 [0.70, 0.77] | 0.01 [0.00, 0.01] | 190 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a | degree_adv=False | 392 (196) | 0.47 [0.41, 0.53] | 0.00 [0.00, 0.01] | 0.69 [0.63, 0.74] | 0.02 [0.00, 0.03] | 66 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a | degree_adv=True | 578 (289) | 0.57 [0.52, 0.62] | 0.00 [0.00, 0.00] | 0.77 [0.72, 0.81] | 0.00 [0.00, 0.00] | 125 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a | names=False | 918 (459) | 0.52 [0.48, 0.56] | 0.00 [0.00, 0.00] | 0.73 [0.69, 0.76] | 0.01 [0.00, 0.01] | 175 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a | names=True | 52 (26) | 0.71 [0.56, 0.85] | 0.00 [0.00, 0.00] | 0.87 [0.73, 0.98] | 0.00 [0.00, 0.00] | 16 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a | order=1 | 485 (485) | 0.55 [0.50, 0.59] | 0.00 [0.00, 0.00] | 0.75 [0.71, 0.79] | 0.01 [0.00, 0.01] | 265 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a | order=2 | 485 (485) | 0.52 [0.47, 0.56] | 0.00 [0.00, 0.01] | 0.72 [0.68, 0.76] | 0.01 [0.00, 0.01] | 250 / 1 | other answer |
| Qwen2-57B-A14B | E1p e→a (all strata) | all | 970 (485) | 0.53 [0.49, 0.57] | 0.00 [0.00, 0.00] | 0.73 [0.70, 0.77] | 0.01 [0.00, 0.01] | 191 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a (all strata) | antonym_type=gradable | 962 (481) | 0.53 [0.49, 0.57] | 0.00 [0.00, 0.00] | 0.73 [0.70, 0.77] | 0.01 [0.00, 0.01] | 190 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a (all strata) | degree_adv=False | 392 (196) | 0.47 [0.41, 0.53] | 0.00 [0.00, 0.01] | 0.69 [0.63, 0.74] | 0.02 [0.00, 0.03] | 66 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a (all strata) | degree_adv=True | 578 (289) | 0.57 [0.52, 0.62] | 0.00 [0.00, 0.00] | 0.77 [0.72, 0.81] | 0.00 [0.00, 0.00] | 125 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a (all strata) | names=False | 918 (459) | 0.52 [0.48, 0.56] | 0.00 [0.00, 0.00] | 0.73 [0.69, 0.76] | 0.01 [0.00, 0.01] | 175 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a (all strata) | names=True | 52 (26) | 0.71 [0.56, 0.85] | 0.00 [0.00, 0.00] | 0.87 [0.73, 0.98] | 0.00 [0.00, 0.00] | 16 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a (all strata) | order=1 | 485 (485) | 0.55 [0.50, 0.59] | 0.00 [0.00, 0.00] | 0.75 [0.71, 0.79] | 0.01 [0.00, 0.01] | 265 / 0 | other answer |
| Qwen2-57B-A14B | E1p e→a (all strata) | order=2 | 485 (485) | 0.52 [0.47, 0.56] | 0.00 [0.00, 0.01] | 0.72 [0.68, 0.76] | 0.01 [0.00, 0.01] | 250 / 1 | other answer |
| Qwen2-57B-A14B | E2 e→a | all | 1178 (589) | 0.08 [0.06, 0.09] | 0.01 [0.00, 0.01] | 0.31 [0.28, 0.33] | 0.09 [0.07, 0.11] | 13 / 1 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | antonym_type=complementary | 10 (5) | 0.10 [0.00, 0.30] | 0.00 [0.00, 0.00] | 0.30 [0.00, 0.70] | 0.10 [0.00, 0.30] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | antonym_type=gradable | 1168 (584) | 0.08 [0.06, 0.09] | 0.01 [0.00, 0.01] | 0.31 [0.28, 0.33] | 0.09 [0.07, 0.11] | 13 / 1 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | degree_adv=False | 490 (245) | 0.08 [0.06, 0.11] | 0.00 [0.00, 0.00] | 0.34 [0.30, 0.39] | 0.03 [0.02, 0.05] | 7 / 0 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | degree_adv=True | 688 (344) | 0.07 [0.05, 0.09] | 0.01 [0.00, 0.02] | 0.28 [0.25, 0.32] | 0.13 [0.10, 0.16] | 6 / 1 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | names=False | 1116 (558) | 0.07 [0.06, 0.09] | 0.01 [0.00, 0.01] | 0.30 [0.27, 0.33] | 0.09 [0.07, 0.11] | 12 / 1 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | names=True | 62 (31) | 0.13 [0.05, 0.23] | 0.02 [0.00, 0.05] | 0.44 [0.32, 0.55] | 0.06 [0.02, 0.13] | 1 / 0 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | order=1 | 589 (589) | 0.08 [0.06, 0.10] | 0.01 [0.00, 0.02] | 0.30 [0.26, 0.33] | 0.10 [0.07, 0.12] | 45 / 5 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E2 e→a | order=2 | 589 (589) | 0.07 [0.05, 0.10] | 0.00 [0.00, 0.01] | 0.31 [0.28, 0.35] | 0.08 [0.06, 0.10] | 44 / 2 | copy trap (word stated with the other entity) |
| Qwen2-57B-A14B | E3 e→a | all | 648 (648) | 0.84 [0.81, 0.87] | 0.00 [0.00, 0.00] | 0.94 [0.92, 0.96] | 0.00 [0.00, 0.00] | 544 / 0 | other answer |
| Qwen2-57B-A14B | E3 e→a | antonym_type=gradable | 643 (643) | 0.84 [0.81, 0.87] | 0.00 [0.00, 0.00] | 0.94 [0.92, 0.96] | 0.00 [0.00, 0.00] | 540 / 0 | other answer |
| Qwen2-57B-A14B | E3 e→a | degree_adv=False | 286 (286) | 0.84 [0.80, 0.88] | 0.00 [0.00, 0.00] | 0.94 [0.92, 0.97] | 0.00 [0.00, 0.00] | 240 / 0 | other answer |
| Qwen2-57B-A14B | E3 e→a | degree_adv=True | 362 (362) | 0.84 [0.80, 0.88] | 0.00 [0.00, 0.00] | 0.94 [0.91, 0.96] | 0.00 [0.00, 0.00] | 304 / 0 | other answer |
| Qwen2-57B-A14B | E3 e→a | names=False | 617 (617) | 0.84 [0.81, 0.87] | 0.00 [0.00, 0.00] | 0.94 [0.92, 0.96] | 0.00 [0.00, 0.00] | 520 / 0 | other answer |
| Qwen2-57B-A14B | E3 e→a | names=True | 31 (31) | 0.77 [0.61, 0.90] | 0.00 [0.00, 0.00] | 0.97 [0.90, 1.00] | 0.00 [0.00, 0.00] | 24 / 0 | other answer |
| Qwen2-57B-A14B | E3 e→a | order=1 | 418 (418) | 0.86 [0.83, 0.89] | 0.00 [0.00, 0.00] | 0.95 [0.93, 0.97] | 0.00 [0.00, 0.00] | 358 / 0 | other answer |
| Qwen2-57B-A14B | E3 e→a | order=2 | 230 (230) | 0.81 [0.76, 0.86] | 0.00 [0.00, 0.00] | 0.91 [0.88, 0.95] | 0.00 [0.00, 0.00] | 186 / 0 | other answer |
| Qwen2-57B-A14B | Set C agree (a→e) | all | 1286 (643) | 0.55 [0.52, 0.59] | 0.00 [0.00, 0.00] | 0.83 [0.81, 0.85] | 0.00 [0.00, 0.00] | 261 / 0 | event / other |
| Qwen2-57B-A14B | Set C agree (a→e) | antonym_type=complementary | 10 (5) | 0.60 [0.30, 0.90] | 0.00 [0.00, 0.00] | 0.70 [0.30, 1.00] | 0.00 [0.00, 0.00] | 2 / 0 | event / other |
| Qwen2-57B-A14B | Set C agree (a→e) | antonym_type=gradable | 1276 (638) | 0.55 [0.52, 0.59] | 0.00 [0.00, 0.00] | 0.83 [0.81, 0.86] | 0.00 [0.00, 0.00] | 259 / 0 | event / other |
| Qwen2-57B-A14B | Set C agree (a→e) | degree_adv=False | 570 (285) | 0.55 [0.50, 0.59] | 0.00 [0.00, 0.00] | 0.79 [0.76, 0.83] | 0.00 [0.00, 0.00] | 115 / 0 | event / other |
| Qwen2-57B-A14B | Set C agree (a→e) | degree_adv=True | 716 (358) | 0.56 [0.52, 0.60] | 0.00 [0.00, 0.00] | 0.86 [0.83, 0.89] | 0.00 [0.00, 0.00] | 146 / 0 | event / other |
| Qwen2-57B-A14B | Set C agree (a→e) | names=False | 1224 (612) | 0.56 [0.53, 0.59] | 0.00 [0.00, 0.00] | 0.84 [0.81, 0.86] | 0.00 [0.00, 0.00] | 250 / 0 | event / other |
| Qwen2-57B-A14B | Set C agree (a→e) | names=True | 62 (31) | 0.48 [0.34, 0.65] | 0.00 [0.00, 0.00] | 0.69 [0.56, 0.81] | 0.00 [0.00, 0.00] | 11 / 0 | event / other |
| Qwen2-57B-A14B | Set C agree (a→e) | order=1 | 643 (643) | 0.57 [0.53, 0.61] | 0.00 [0.00, 0.00] | 0.87 [0.84, 0.89] | 0.00 [0.00, 0.00] | 369 / 0 | event / other |
| Qwen2-57B-A14B | Set C agree (a→e) | order=2 | 643 (643) | 0.53 [0.50, 0.57] | 0.00 [0.00, 0.00] | 0.79 [0.76, 0.83] | 0.00 [0.00, 0.00] | 344 / 0 | event / other |
| Qwen2-57B-A14B | sc_conflict_event_a2e | all | 1286 (643) | 0.03 [0.02, 0.04] | 0.07 [0.06, 0.08] | 0.12 [0.11, 0.15] | 0.28 [0.26, 0.31] | 7 / 11 | event / other |
| Qwen2-57B-A14B | sc_conflict_event_a2e | antonym_type=complementary | 10 (5) | 0.00 [0.00, 0.00] | 0.20 [0.00, 0.40] | 0.10 [0.00, 0.30] | 0.50 [0.10, 0.90] | 0 / 0 | event / other |
| Qwen2-57B-A14B | sc_conflict_event_a2e | antonym_type=gradable | 1276 (638) | 0.03 [0.02, 0.04] | 0.07 [0.05, 0.08] | 0.12 [0.10, 0.15] | 0.28 [0.25, 0.31] | 7 / 11 | event / other |
| Qwen2-57B-A14B | sc_conflict_event_a2e | degree_adv=False | 570 (285) | 0.04 [0.02, 0.06] | 0.07 [0.05, 0.09] | 0.19 [0.16, 0.23] | 0.19 [0.15, 0.23] | 5 / 7 | event / other |
| Qwen2-57B-A14B | sc_conflict_event_a2e | degree_adv=True | 716 (358) | 0.02 [0.01, 0.03] | 0.07 [0.05, 0.09] | 0.07 [0.05, 0.09] | 0.35 [0.32, 0.40] | 2 / 4 | event / other |
| Qwen2-57B-A14B | sc_conflict_event_a2e | names=False | 1224 (612) | 0.03 [0.02, 0.04] | 0.06 [0.05, 0.08] | 0.13 [0.11, 0.15] | 0.27 [0.24, 0.30] | 7 / 9 | event / other |
| Qwen2-57B-A14B | sc_conflict_event_a2e | names=True | 62 (31) | 0.00 [0.00, 0.00] | 0.23 [0.13, 0.34] | 0.00 [0.00, 0.00] | 0.52 [0.40, 0.65] | 0 / 2 | event / other |
| Qwen2-57B-A14B | sc_conflict_event_a2e | order=1 | 643 (643) | 0.02 [0.01, 0.04] | 0.07 [0.05, 0.09] | 0.12 [0.10, 0.15] | 0.29 [0.25, 0.32] | 16 / 48 | event / other |
| Qwen2-57B-A14B | sc_conflict_event_a2e | order=2 | 643 (643) | 0.03 [0.02, 0.05] | 0.07 [0.05, 0.09] | 0.12 [0.10, 0.15] | 0.27 [0.24, 0.31] | 20 / 42 | event / other |
| Qwen2-57B-A14B | sc_conflict_stmt_a2e | all | 1286 (643) | 0.07 [0.06, 0.08] | 0.03 [0.02, 0.04] | 0.28 [0.26, 0.31] | 0.12 [0.11, 0.15] | 11 / 7 | event / other |
| Qwen2-57B-A14B | sc_conflict_stmt_a2e | antonym_type=complementary | 10 (5) | 0.20 [0.00, 0.40] | 0.00 [0.00, 0.00] | 0.50 [0.10, 0.90] | 0.10 [0.00, 0.30] | 0 / 0 | event / other |
| Qwen2-57B-A14B | sc_conflict_stmt_a2e | antonym_type=gradable | 1276 (638) | 0.07 [0.05, 0.08] | 0.03 [0.02, 0.04] | 0.28 [0.25, 0.31] | 0.12 [0.10, 0.15] | 11 / 7 | event / other |
| Qwen2-57B-A14B | sc_conflict_stmt_a2e | degree_adv=False | 570 (285) | 0.07 [0.05, 0.09] | 0.04 [0.02, 0.06] | 0.19 [0.15, 0.23] | 0.19 [0.16, 0.23] | 7 / 5 | event / other |
| Qwen2-57B-A14B | sc_conflict_stmt_a2e | degree_adv=True | 716 (358) | 0.07 [0.05, 0.09] | 0.02 [0.01, 0.03] | 0.35 [0.32, 0.40] | 0.07 [0.05, 0.09] | 4 / 2 | event / other |
| Qwen2-57B-A14B | sc_conflict_stmt_a2e | names=False | 1224 (612) | 0.06 [0.05, 0.08] | 0.03 [0.02, 0.04] | 0.27 [0.24, 0.30] | 0.13 [0.11, 0.15] | 9 / 7 | event / other |
| Qwen2-57B-A14B | sc_conflict_stmt_a2e | names=True | 62 (31) | 0.23 [0.13, 0.34] | 0.00 [0.00, 0.00] | 0.52 [0.40, 0.65] | 0.00 [0.00, 0.00] | 2 / 0 | event / other |
| Qwen2-57B-A14B | sc_conflict_stmt_a2e | order=1 | 643 (643) | 0.07 [0.05, 0.09] | 0.02 [0.01, 0.04] | 0.29 [0.25, 0.32] | 0.12 [0.10, 0.15] | 48 / 16 | event / other |
| Qwen2-57B-A14B | sc_conflict_stmt_a2e | order=2 | 643 (643) | 0.07 [0.05, 0.09] | 0.03 [0.02, 0.05] | 0.27 [0.24, 0.31] | 0.12 [0.10, 0.15] | 42 / 20 | event / other |
| Qwen2-7B | E1 a→e | all | 3578 (1789) | 0.23 [0.21, 0.25] | 0.00 [0.00, 0.00] | 0.70 [0.69, 0.72] | 0.00 [0.00, 0.00] | 247 / 0 | other answer |
| Qwen2-7B | E1 a→e | antonym_type=complementary | 48 (24) | 0.19 [0.08, 0.33] | 0.00 [0.00, 0.00] | 0.73 [0.60, 0.85] | 0.00 [0.00, 0.00] | 2 / 0 | other answer |
| Qwen2-7B | E1 a→e | antonym_type=gradable | 3530 (1765) | 0.23 [0.21, 0.25] | 0.00 [0.00, 0.00] | 0.70 [0.69, 0.72] | 0.00 [0.00, 0.00] | 245 / 0 | other answer |
| Qwen2-7B | E1 a→e | degree_adv=False | 1986 (993) | 0.24 [0.22, 0.26] | 0.00 [0.00, 0.00] | 0.69 [0.67, 0.72] | 0.00 [0.00, 0.01] | 151 / 0 | other answer |
| Qwen2-7B | E1 a→e | degree_adv=True | 1592 (796) | 0.22 [0.19, 0.24] | 0.00 [0.00, 0.00] | 0.72 [0.69, 0.74] | 0.00 [0.00, 0.00] | 96 / 0 | other answer |
| Qwen2-7B | E1 a→e | names=False | 3060 (1530) | 0.14 [0.12, 0.15] | 0.00 [0.00, 0.00] | 0.66 [0.64, 0.68] | 0.00 [0.00, 0.00] | 73 / 0 | other answer |
| Qwen2-7B | E1 a→e | names=True | 518 (259) | 0.79 [0.75, 0.83] | 0.00 [0.00, 0.00] | 0.96 [0.94, 0.98] | 0.00 [0.00, 0.00] | 174 / 0 | other answer |
| Qwen2-7B | E1 a→e | order=1 | 1789 (1789) | 0.24 [0.22, 0.25] | 0.00 [0.00, 0.00] | 0.70 [0.68, 0.72] | 0.00 [0.00, 0.01] | 422 / 0 | other answer |
| Qwen2-7B | E1 a→e | order=2 | 1789 (1789) | 0.22 [0.20, 0.24] | 0.00 [0.00, 0.00] | 0.71 [0.69, 0.73] | 0.00 [0.00, 0.00] | 401 / 0 | other answer |
| Qwen2-7B | E1p a→e | all | 3370 (1685) | 0.27 [0.25, 0.28] | 0.00 [0.00, 0.00] | 0.72 [0.70, 0.74] | 0.00 [0.00, 0.00] | 269 / 0 | other answer |
| Qwen2-7B | E1p a→e | antonym_type=complementary | 32 (16) | 0.41 [0.19, 0.62] | 0.00 [0.00, 0.00] | 0.84 [0.69, 0.97] | 0.00 [0.00, 0.00] | 5 / 0 | other answer |
| Qwen2-7B | E1p a→e | antonym_type=gradable | 3338 (1669) | 0.26 [0.25, 0.28] | 0.00 [0.00, 0.00] | 0.72 [0.70, 0.74] | 0.00 [0.00, 0.00] | 264 / 0 | other answer |
| Qwen2-7B | E1p a→e | degree_adv=False | 1790 (895) | 0.32 [0.29, 0.34] | 0.00 [0.00, 0.00] | 0.78 [0.75, 0.80] | 0.00 [0.00, 0.00] | 172 / 0 | other answer |
| Qwen2-7B | E1p a→e | degree_adv=True | 1580 (790) | 0.21 [0.18, 0.23] | 0.00 [0.00, 0.00] | 0.66 [0.63, 0.68] | 0.00 [0.00, 0.00] | 97 / 0 | other answer |
| Qwen2-7B | E1p a→e | names=False | 2904 (1452) | 0.18 [0.16, 0.19] | 0.00 [0.00, 0.00] | 0.68 [0.66, 0.70] | 0.00 [0.00, 0.00] | 111 / 0 | other answer |
| Qwen2-7B | E1p a→e | names=True | 466 (233) | 0.81 [0.77, 0.85] | 0.00 [0.00, 0.00] | 0.96 [0.94, 0.97] | 0.00 [0.00, 0.00] | 158 / 0 | other answer |
| Qwen2-7B | E1p a→e | order=1 | 1685 (1685) | 0.26 [0.24, 0.28] | 0.00 [0.00, 0.00] | 0.72 [0.69, 0.74] | 0.00 [0.00, 0.00] | 439 / 0 | other answer |
| Qwen2-7B | E1p a→e | order=2 | 1685 (1685) | 0.27 [0.25, 0.29] | 0.00 [0.00, 0.00] | 0.73 [0.70, 0.75] | 0.00 [0.00, 0.00] | 455 / 0 | other answer |
| Qwen2-7B | E2 a→e | all | 3578 (1789) | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.04] | 0.02 [0.01, 0.02] | 0.39 [0.38, 0.41] | 1 / 15 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.08 [0.00, 0.19] | 0.02 [0.00, 0.06] | 0.31 [0.17, 0.48] | 0 / 1 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | antonym_type=gradable | 3530 (1765) | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.04] | 0.02 [0.01, 0.02] | 0.39 [0.38, 0.41] | 1 / 14 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | degree_adv=False | 1986 (993) | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.04] | 0.03 [0.02, 0.03] | 0.37 [0.35, 0.40] | 1 / 7 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | degree_adv=True | 1592 (796) | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.04] | 0.01 [0.01, 0.02] | 0.42 [0.39, 0.44] | 0 / 8 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | names=False | 3060 (1530) | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.01] | 0.02 [0.02, 0.03] | 0.35 [0.33, 0.37] | 1 / 1 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | names=True | 518 (259) | 0.00 [0.00, 0.00] | 0.17 [0.13, 0.20] | 0.01 [0.00, 0.02] | 0.65 [0.61, 0.70] | 0 / 14 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | order=1 | 1789 (1789) | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.04] | 0.01 [0.01, 0.02] | 0.39 [0.37, 0.41] | 1 / 55 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 a→e | order=2 | 1789 (1789) | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.04] | 0.03 [0.02, 0.03] | 0.39 [0.37, 0.41] | 2 / 51 | copy trap (word stated with the other entity) |
| Qwen2-7B | E3 a→e (P3 frame) | all | 1824 (1824) | 0.17 [0.16, 0.19] | 0.00 [0.00, 0.01] | 0.41 [0.38, 0.43] | 0.02 [0.01, 0.02] | 314 / 6 | other answer |
| Qwen2-7B | E3 a→e (P3 frame) | antonym_type=complementary | 24 (24) | 0.08 [0.00, 0.21] | 0.04 [0.00, 0.12] | 0.33 [0.17, 0.54] | 0.08 [0.00, 0.21] | 2 / 1 | other answer |
| Qwen2-7B | E3 a→e (P3 frame) | antonym_type=gradable | 1800 (1800) | 0.17 [0.15, 0.19] | 0.00 [0.00, 0.01] | 0.41 [0.38, 0.43] | 0.01 [0.01, 0.02] | 312 / 5 | other answer |
| Qwen2-7B | E3 a→e (P3 frame) | degree_adv=False | 1035 (1035) | 0.20 [0.17, 0.22] | 0.01 [0.00, 0.01] | 0.43 [0.40, 0.46] | 0.02 [0.01, 0.03] | 203 / 6 | other answer |
| Qwen2-7B | E3 a→e (P3 frame) | degree_adv=True | 789 (789) | 0.14 [0.12, 0.16] | 0.00 [0.00, 0.00] | 0.37 [0.34, 0.40] | 0.01 [0.00, 0.02] | 111 / 0 | other answer |
| Qwen2-7B | E3 a→e (P3 frame) | names=False | 1563 (1563) | 0.14 [0.13, 0.16] | 0.00 [0.00, 0.01] | 0.37 [0.35, 0.40] | 0.01 [0.01, 0.02] | 223 / 4 | other answer |
| Qwen2-7B | E3 a→e (P3 frame) | names=True | 261 (261) | 0.35 [0.29, 0.40] | 0.01 [0.00, 0.02] | 0.60 [0.54, 0.66] | 0.02 [0.00, 0.03] | 91 / 2 | other answer |
| Qwen2-7B | E3 a→e (P3 frame) | order=1 | 1323 (1323) | 0.20 [0.17, 0.22] | 0.00 [0.00, 0.01] | 0.43 [0.41, 0.46] | 0.02 [0.01, 0.02] | 260 / 5 | other answer |
| Qwen2-7B | E3 a→e (P3 frame) | order=2 | 501 (501) | 0.11 [0.08, 0.14] | 0.00 [0.00, 0.01] | 0.34 [0.29, 0.38] | 0.01 [0.00, 0.02] | 54 / 1 | other answer |
| Qwen2-7B | E3 a→e (F1 frame, Phase-5 anchor) | all | 1843 (1843) | 0.14 [0.13, 0.16] | 0.00 [0.00, 0.00] | 0.39 [0.37, 0.41] | 0.01 [0.01, 0.02] | 265 / 1 | other answer |
| Qwen2-7B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=complementary | 24 (24) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.25 [0.08, 0.42] | 0.04 [0.00, 0.12] | 0 / 0 | other answer |
| Qwen2-7B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=gradable | 1819 (1819) | 0.15 [0.13, 0.16] | 0.00 [0.00, 0.00] | 0.39 [0.37, 0.41] | 0.01 [0.01, 0.02] | 265 / 1 | other answer |
| Qwen2-7B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=False | 1038 (1038) | 0.15 [0.13, 0.17] | 0.00 [0.00, 0.00] | 0.39 [0.36, 0.42] | 0.01 [0.01, 0.02] | 156 / 1 | other answer |
| Qwen2-7B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=True | 805 (805) | 0.14 [0.11, 0.16] | 0.00 [0.00, 0.00] | 0.38 [0.35, 0.42] | 0.01 [0.00, 0.02] | 109 / 0 | other answer |
| Qwen2-7B | E3 a→e (F1 frame, Phase-5 anchor) | names=False | 1587 (1587) | 0.15 [0.13, 0.17] | 0.00 [0.00, 0.00] | 0.39 [0.36, 0.41] | 0.01 [0.01, 0.02] | 233 / 1 | other answer |
| Qwen2-7B | E3 a→e (F1 frame, Phase-5 anchor) | names=True | 256 (256) | 0.12 [0.09, 0.16] | 0.00 [0.00, 0.00] | 0.38 [0.32, 0.44] | 0.02 [0.00, 0.03] | 32 / 0 | other answer |
| Qwen2-7B | E3 a→e (F1 frame, Phase-5 anchor) | order=1 | 1324 (1324) | 0.14 [0.12, 0.16] | 0.00 [0.00, 0.00] | 0.39 [0.37, 0.42] | 0.01 [0.01, 0.02] | 186 / 1 | other answer |
| Qwen2-7B | E3 a→e (F1 frame, Phase-5 anchor) | order=2 | 519 (519) | 0.15 [0.12, 0.19] | 0.00 [0.00, 0.00] | 0.37 [0.33, 0.41] | 0.01 [0.00, 0.02] | 79 / 0 | other answer |
| Qwen2-7B | E1 e→a | all | 3578 (1789) | 0.30 [0.28, 0.32] | 0.00 [0.00, 0.01] | 0.61 [0.59, 0.63] | 0.03 [0.02, 0.03] | 325 / 1 | other answer |
| Qwen2-7B | E1 e→a | antonym_type=complementary | 50 (25) | 0.18 [0.08, 0.30] | 0.00 [0.00, 0.00] | 0.46 [0.30, 0.64] | 0.02 [0.00, 0.06] | 2 / 0 | other answer |
| Qwen2-7B | E1 e→a | antonym_type=gradable | 3528 (1764) | 0.30 [0.28, 0.32] | 0.00 [0.00, 0.01] | 0.61 [0.59, 0.63] | 0.03 [0.02, 0.03] | 323 / 1 | other answer |
| Qwen2-7B | E1 e→a | degree_adv=False | 1986 (993) | 0.25 [0.23, 0.27] | 0.01 [0.00, 0.01] | 0.51 [0.49, 0.53] | 0.04 [0.03, 0.05] | 134 / 1 | other answer |
| Qwen2-7B | E1 e→a | degree_adv=True | 1592 (796) | 0.36 [0.33, 0.39] | 0.00 [0.00, 0.00] | 0.73 [0.70, 0.75] | 0.01 [0.01, 0.02] | 191 / 0 | other answer |
| Qwen2-7B | E1 e→a | names=False | 3060 (1530) | 0.29 [0.27, 0.31] | 0.00 [0.00, 0.01] | 0.61 [0.59, 0.63] | 0.03 [0.02, 0.03] | 264 / 1 | other answer |
| Qwen2-7B | E1 e→a | names=True | 518 (259) | 0.37 [0.32, 0.42] | 0.00 [0.00, 0.01] | 0.58 [0.53, 0.63] | 0.02 [0.01, 0.04] | 61 / 0 | other answer |
| Qwen2-7B | E1 e→a | order=1 | 1789 (1789) | 0.30 [0.28, 0.32] | 0.00 [0.00, 0.00] | 0.61 [0.58, 0.63] | 0.02 [0.02, 0.03] | 536 / 2 | other answer |
| Qwen2-7B | E1 e→a | order=2 | 1789 (1789) | 0.30 [0.28, 0.32] | 0.01 [0.00, 0.01] | 0.60 [0.58, 0.63] | 0.03 [0.02, 0.04] | 532 / 10 | other answer |
| Qwen2-7B | E1p e→a | all | 3370 (1685) | 0.25 [0.23, 0.27] | 0.00 [0.00, 0.00] | 0.51 [0.49, 0.53] | 0.01 [0.01, 0.01] | 288 / 0 | other answer |
| Qwen2-7B | E1p e→a | antonym_type=complementary | 32 (16) | 0.12 [0.03, 0.22] | 0.00 [0.00, 0.00] | 0.44 [0.22, 0.63] | 0.03 [0.00, 0.09] | 0 / 0 | other answer |
| Qwen2-7B | E1p e→a | antonym_type=gradable | 3338 (1669) | 0.25 [0.24, 0.27] | 0.00 [0.00, 0.00] | 0.51 [0.49, 0.53] | 0.01 [0.01, 0.01] | 288 / 0 | other answer |
| Qwen2-7B | E1p e→a | degree_adv=False | 1790 (895) | 0.30 [0.28, 0.33] | 0.00 [0.00, 0.00] | 0.57 [0.55, 0.60] | 0.01 [0.01, 0.02] | 183 / 0 | other answer |
| Qwen2-7B | E1p e→a | degree_adv=True | 1580 (790) | 0.20 [0.17, 0.23] | 0.00 [0.00, 0.00] | 0.44 [0.41, 0.47] | 0.01 [0.00, 0.01] | 105 / 0 | other answer |
| Qwen2-7B | E1p e→a | names=False | 2904 (1452) | 0.22 [0.20, 0.24] | 0.00 [0.00, 0.00] | 0.49 [0.47, 0.51] | 0.01 [0.01, 0.02] | 207 / 0 | other answer |
| Qwen2-7B | E1p e→a | names=True | 466 (233) | 0.44 [0.39, 0.50] | 0.00 [0.00, 0.00] | 0.64 [0.58, 0.69] | 0.00 [0.00, 0.01] | 81 / 0 | other answer |
| Qwen2-7B | E1p e→a | order=1 | 1685 (1685) | 0.26 [0.24, 0.28] | 0.00 [0.00, 0.00] | 0.52 [0.50, 0.54] | 0.01 [0.01, 0.02] | 435 / 0 | other answer |
| Qwen2-7B | E1p e→a | order=2 | 1685 (1685) | 0.25 [0.23, 0.27] | 0.00 [0.00, 0.00] | 0.50 [0.48, 0.52] | 0.01 [0.00, 0.01] | 419 / 0 | other answer |
| Qwen2-7B | E2 e→a | all | 3578 (1789) | 0.05 [0.04, 0.05] | 0.00 [0.00, 0.01] | 0.31 [0.29, 0.32] | 0.05 [0.04, 0.06] | 13 / 1 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | antonym_type=complementary | 50 (25) | 0.04 [0.00, 0.10] | 0.00 [0.00, 0.00] | 0.50 [0.36, 0.62] | 0.02 [0.00, 0.06] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | antonym_type=gradable | 3528 (1764) | 0.05 [0.04, 0.05] | 0.00 [0.00, 0.01] | 0.30 [0.29, 0.32] | 0.05 [0.04, 0.06] | 13 / 1 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | degree_adv=False | 1986 (993) | 0.06 [0.05, 0.07] | 0.00 [0.00, 0.00] | 0.38 [0.35, 0.40] | 0.02 [0.01, 0.03] | 8 / 0 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | degree_adv=True | 1592 (796) | 0.02 [0.02, 0.03] | 0.01 [0.00, 0.01] | 0.22 [0.20, 0.24] | 0.09 [0.07, 0.10] | 5 / 1 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | names=False | 3060 (1530) | 0.03 [0.03, 0.04] | 0.00 [0.00, 0.00] | 0.29 [0.27, 0.31] | 0.05 [0.04, 0.06] | 4 / 1 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | names=True | 518 (259) | 0.13 [0.10, 0.16] | 0.01 [0.00, 0.01] | 0.40 [0.36, 0.44] | 0.04 [0.03, 0.06] | 9 / 0 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | order=1 | 1789 (1789) | 0.05 [0.04, 0.06] | 0.00 [0.00, 0.01] | 0.31 [0.29, 0.33] | 0.05 [0.04, 0.06] | 82 / 6 | copy trap (word stated with the other entity) |
| Qwen2-7B | E2 e→a | order=2 | 1789 (1789) | 0.05 [0.04, 0.06] | 0.00 [0.00, 0.01] | 0.31 [0.28, 0.33] | 0.05 [0.04, 0.06] | 82 / 5 | copy trap (word stated with the other entity) |
| Qwen2-7B | E3 e→a | all | 648 (648) | 0.82 [0.80, 0.85] | 0.00 [0.00, 0.00] | 0.95 [0.94, 0.97] | 0.00 [0.00, 0.00] | 534 / 0 | other answer |
| Qwen2-7B | E3 e→a | antonym_type=gradable | 643 (643) | 0.82 [0.79, 0.85] | 0.00 [0.00, 0.00] | 0.95 [0.93, 0.97] | 0.00 [0.00, 0.00] | 529 / 0 | other answer |
| Qwen2-7B | E3 e→a | degree_adv=False | 286 (286) | 0.83 [0.78, 0.87] | 0.00 [0.00, 0.00] | 0.94 [0.91, 0.97] | 0.00 [0.00, 0.00] | 237 / 0 | other answer |
| Qwen2-7B | E3 e→a | degree_adv=True | 362 (362) | 0.82 [0.78, 0.86] | 0.00 [0.00, 0.00] | 0.96 [0.94, 0.98] | 0.00 [0.00, 0.00] | 297 / 0 | other answer |
| Qwen2-7B | E3 e→a | names=False | 617 (617) | 0.83 [0.80, 0.86] | 0.00 [0.00, 0.00] | 0.96 [0.94, 0.97] | 0.00 [0.00, 0.00] | 515 / 0 | other answer |
| Qwen2-7B | E3 e→a | names=True | 31 (31) | 0.61 [0.42, 0.77] | 0.00 [0.00, 0.00] | 0.84 [0.71, 0.94] | 0.00 [0.00, 0.00] | 19 / 0 | other answer |
| Qwen2-7B | E3 e→a | order=1 | 418 (418) | 0.83 [0.79, 0.86] | 0.00 [0.00, 0.00] | 0.95 [0.93, 0.97] | 0.00 [0.00, 0.00] | 346 / 0 | other answer |
| Qwen2-7B | E3 e→a | order=2 | 230 (230) | 0.82 [0.77, 0.87] | 0.00 [0.00, 0.00] | 0.96 [0.93, 0.98] | 0.00 [0.00, 0.00] | 188 / 0 | other answer |
| Qwen2-7B | Set C agree (a→e) | all | 3686 (1843) | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.00] | 0.32 [0.30, 0.33] | 0.01 [0.01, 0.01] | 24 / 0 | event / other |
| Qwen2-7B | Set C agree (a→e) | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.44 [0.31, 0.56] | 0.02 [0.00, 0.06] | 0 / 0 | event / other |
| Qwen2-7B | Set C agree (a→e) | antonym_type=gradable | 3638 (1819) | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.00] | 0.32 [0.30, 0.33] | 0.01 [0.01, 0.01] | 24 / 0 | event / other |
| Qwen2-7B | Set C agree (a→e) | degree_adv=False | 2074 (1037) | 0.05 [0.04, 0.06] | 0.00 [0.00, 0.00] | 0.36 [0.34, 0.39] | 0.01 [0.01, 0.01] | 15 / 0 | event / other |
| Qwen2-7B | Set C agree (a→e) | degree_adv=True | 1612 (806) | 0.03 [0.02, 0.04] | 0.00 [0.00, 0.00] | 0.26 [0.24, 0.28] | 0.01 [0.01, 0.02] | 9 / 0 | event / other |
| Qwen2-7B | Set C agree (a→e) | names=False | 3174 (1587) | 0.03 [0.02, 0.03] | 0.00 [0.00, 0.00] | 0.30 [0.28, 0.31] | 0.01 [0.01, 0.02] | 11 / 0 | event / other |
| Qwen2-7B | Set C agree (a→e) | names=True | 512 (256) | 0.11 [0.08, 0.14] | 0.00 [0.00, 0.00] | 0.45 [0.40, 0.51] | 0.01 [0.00, 0.01] | 13 / 0 | event / other |
| Qwen2-7B | Set C agree (a→e) | order=1 | 1843 (1843) | 0.05 [0.04, 0.06] | 0.00 [0.00, 0.00] | 0.34 [0.32, 0.36] | 0.01 [0.01, 0.02] | 91 / 0 | event / other |
| Qwen2-7B | Set C agree (a→e) | order=2 | 1843 (1843) | 0.03 [0.02, 0.04] | 0.00 [0.00, 0.00] | 0.30 [0.28, 0.32] | 0.01 [0.01, 0.01] | 53 / 0 | event / other |
| Qwen2-7B | sc_conflict_event_a2e | all | 3686 (1843) | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.01] | 0.07 [0.06, 0.08] | 0.13 [0.12, 0.14] | 0 / 3 | event / other |
| Qwen2-7B | sc_conflict_event_a2e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.02 [0.00, 0.06] | 0.10 [0.02, 0.21] | 0.04 [0.00, 0.10] | 0 / 0 | event / other |
| Qwen2-7B | sc_conflict_event_a2e | antonym_type=gradable | 3638 (1819) | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.01] | 0.07 [0.06, 0.08] | 0.13 [0.12, 0.14] | 0 / 3 | event / other |
| Qwen2-7B | sc_conflict_event_a2e | degree_adv=False | 2074 (1037) | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.01] | 0.09 [0.07, 0.10] | 0.13 [0.11, 0.15] | 0 / 1 | event / other |
| Qwen2-7B | sc_conflict_event_a2e | degree_adv=True | 1612 (806) | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.01] | 0.04 [0.03, 0.06] | 0.13 [0.11, 0.15] | 0 / 2 | event / other |
| Qwen2-7B | sc_conflict_event_a2e | names=False | 3174 (1587) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.08 [0.07, 0.09] | 0.11 [0.09, 0.12] | 0 / 0 | event / other |
| Qwen2-7B | sc_conflict_event_a2e | names=True | 512 (256) | 0.00 [0.00, 0.00] | 0.04 [0.02, 0.06] | 0.01 [0.00, 0.02] | 0.28 [0.23, 0.33] | 0 / 3 | event / other |
| Qwen2-7B | sc_conflict_event_a2e | order=1 | 1843 (1843) | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.01] | 0.06 [0.05, 0.07] | 0.15 [0.13, 0.16] | 1 / 11 | event / other |
| Qwen2-7B | sc_conflict_event_a2e | order=2 | 1843 (1843) | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.01] | 0.07 [0.06, 0.08] | 0.11 [0.10, 0.12] | 0 / 11 | event / other |
| Qwen2-7B | sc_conflict_stmt_a2e | all | 3686 (1843) | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.13 [0.12, 0.14] | 0.07 [0.06, 0.08] | 3 / 0 | event / other |
| Qwen2-7B | sc_conflict_stmt_a2e | antonym_type=complementary | 48 (24) | 0.02 [0.00, 0.06] | 0.00 [0.00, 0.00] | 0.04 [0.00, 0.10] | 0.10 [0.02, 0.21] | 0 / 0 | event / other |
| Qwen2-7B | sc_conflict_stmt_a2e | antonym_type=gradable | 3638 (1819) | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.13 [0.12, 0.14] | 0.07 [0.06, 0.08] | 3 / 0 | event / other |
| Qwen2-7B | sc_conflict_stmt_a2e | degree_adv=False | 2074 (1037) | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.13 [0.11, 0.15] | 0.09 [0.07, 0.10] | 1 / 0 | event / other |
| Qwen2-7B | sc_conflict_stmt_a2e | degree_adv=True | 1612 (806) | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.13 [0.11, 0.15] | 0.04 [0.03, 0.06] | 2 / 0 | event / other |
| Qwen2-7B | sc_conflict_stmt_a2e | names=False | 3174 (1587) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.11 [0.09, 0.12] | 0.08 [0.07, 0.09] | 0 / 0 | event / other |
| Qwen2-7B | sc_conflict_stmt_a2e | names=True | 512 (256) | 0.04 [0.02, 0.06] | 0.00 [0.00, 0.00] | 0.28 [0.23, 0.33] | 0.01 [0.00, 0.02] | 3 / 0 | event / other |
| Qwen2-7B | sc_conflict_stmt_a2e | order=1 | 1843 (1843) | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.15 [0.13, 0.16] | 0.06 [0.05, 0.07] | 11 / 1 | event / other |
| Qwen2-7B | sc_conflict_stmt_a2e | order=2 | 1843 (1843) | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.11 [0.10, 0.12] | 0.07 [0.06, 0.08] | 11 / 0 | event / other |
| Qwen3-30B-A3B | E1 a→e | all | 1178 (589) | 0.95 [0.93, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 530 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e | antonym_type=complementary | 10 (5) | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 5 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e | antonym_type=gradable | 1168 (584) | 0.95 [0.93, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 525 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e | degree_adv=False | 490 (245) | 0.96 [0.94, 0.98] | 0.00 [0.00, 0.00] | 0.99 [0.99, 1.00] | 0.00 [0.00, 0.00] | 230 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e | degree_adv=True | 688 (344) | 0.94 [0.92, 0.95] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 300 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e | names=False | 1116 (558) | 0.95 [0.93, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 502 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e | names=True | 62 (31) | 0.94 [0.85, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 28 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e | order=1 | 589 (589) | 0.94 [0.92, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 554 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e | order=2 | 589 (589) | 0.95 [0.94, 0.97] | 0.00 [0.00, 0.00] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 562 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e (all strata) | all | 1178 (589) | 0.95 [0.93, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 530 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e (all strata) | antonym_type=complementary | 10 (5) | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 5 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e (all strata) | antonym_type=gradable | 1168 (584) | 0.95 [0.93, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 525 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e (all strata) | degree_adv=False | 490 (245) | 0.96 [0.94, 0.98] | 0.00 [0.00, 0.00] | 0.99 [0.99, 1.00] | 0.00 [0.00, 0.00] | 230 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e (all strata) | degree_adv=True | 688 (344) | 0.94 [0.92, 0.95] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 300 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e (all strata) | names=False | 1116 (558) | 0.95 [0.93, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 502 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e (all strata) | names=True | 62 (31) | 0.94 [0.85, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 28 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e (all strata) | order=1 | 589 (589) | 0.94 [0.92, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 554 / 0 | other answer |
| Qwen3-30B-A3B | E1 a→e (all strata) | order=2 | 589 (589) | 0.95 [0.94, 0.97] | 0.00 [0.00, 0.00] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 562 / 0 | other answer |
| Qwen3-30B-A3B | E1p a→e | all | 3370 (1685) | 0.77 [0.75, 0.78] | 0.00 [0.00, 0.00] | 0.92 [0.91, 0.93] | 0.00 [0.00, 0.00] | 1078 / 0 | other answer |
| Qwen3-30B-A3B | E1p a→e | antonym_type=complementary | 32 (16) | 0.62 [0.41, 0.84] | 0.00 [0.00, 0.00] | 0.88 [0.72, 1.00] | 0.00 [0.00, 0.00] | 8 / 0 | other answer |
| Qwen3-30B-A3B | E1p a→e | antonym_type=gradable | 3338 (1669) | 0.77 [0.75, 0.78] | 0.00 [0.00, 0.00] | 0.92 [0.91, 0.93] | 0.00 [0.00, 0.00] | 1070 / 0 | other answer |
| Qwen3-30B-A3B | E1p a→e | degree_adv=False | 1790 (895) | 0.74 [0.72, 0.76] | 0.00 [0.00, 0.00] | 0.91 [0.89, 0.92] | 0.00 [0.00, 0.00] | 546 / 0 | other answer |
| Qwen3-30B-A3B | E1p a→e | degree_adv=True | 1580 (790) | 0.80 [0.77, 0.82] | 0.00 [0.00, 0.00] | 0.94 [0.92, 0.95] | 0.00 [0.00, 0.00] | 532 / 0 | other answer |
| Qwen3-30B-A3B | E1p a→e | names=False | 2904 (1452) | 0.75 [0.73, 0.77] | 0.00 [0.00, 0.00] | 0.91 [0.90, 0.93] | 0.00 [0.00, 0.00] | 891 / 0 | other answer |
| Qwen3-30B-A3B | E1p a→e | names=True | 466 (233) | 0.88 [0.84, 0.91] | 0.00 [0.00, 0.00] | 0.96 [0.94, 0.97] | 0.00 [0.00, 0.00] | 187 / 0 | other answer |
| Qwen3-30B-A3B | E1p a→e | order=1 | 1685 (1685) | 0.77 [0.75, 0.79] | 0.00 [0.00, 0.00] | 0.92 [0.91, 0.93] | 0.00 [0.00, 0.00] | 1291 / 0 | other answer |
| Qwen3-30B-A3B | E1p a→e | order=2 | 1685 (1685) | 0.77 [0.75, 0.79] | 0.00 [0.00, 0.00] | 0.92 [0.91, 0.93] | 0.00 [0.00, 0.00] | 1294 / 0 | other answer |
| Qwen3-30B-A3B | E2 a→e | all | 3578 (1789) | 0.00 [0.00, 0.00] | 0.59 [0.57, 0.61] | 0.00 [0.00, 0.00] | 0.83 [0.82, 0.85] | 0 / 827 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.25 [0.10, 0.40] | 0.00 [0.00, 0.00] | 0.46 [0.29, 0.65] | 0 / 4 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | antonym_type=gradable | 3530 (1765) | 0.00 [0.00, 0.00] | 0.60 [0.58, 0.62] | 0.00 [0.00, 0.00] | 0.84 [0.82, 0.85] | 0 / 823 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | degree_adv=False | 1986 (993) | 0.00 [0.00, 0.00] | 0.45 [0.43, 0.48] | 0.00 [0.00, 0.01] | 0.76 [0.73, 0.78] | 0 / 294 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | degree_adv=True | 1592 (796) | 0.00 [0.00, 0.00] | 0.77 [0.74, 0.79] | 0.00 [0.00, 0.00] | 0.92 [0.91, 0.94] | 0 / 533 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | names=False | 3060 (1530) | 0.00 [0.00, 0.00] | 0.65 [0.63, 0.68] | 0.00 [0.00, 0.00] | 0.87 [0.86, 0.88] | 0 / 801 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | names=True | 518 (259) | 0.00 [0.00, 0.00] | 0.22 [0.18, 0.26] | 0.00 [0.00, 0.01] | 0.60 [0.55, 0.64] | 0 / 26 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | order=1 | 1789 (1789) | 0.00 [0.00, 0.00] | 0.60 [0.58, 0.62] | 0.00 [0.00, 0.00] | 0.84 [0.82, 0.85] | 0 / 1070 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 a→e | order=2 | 1789 (1789) | 0.00 [0.00, 0.00] | 0.59 [0.56, 0.61] | 0.00 [0.00, 0.01] | 0.83 [0.81, 0.84] | 0 / 1049 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E3 a→e (P3 frame) | all | 1824 (1824) | 0.18 [0.16, 0.20] | 0.00 [0.00, 0.01] | 0.40 [0.38, 0.42] | 0.02 [0.01, 0.02] | 328 / 6 | other answer |
| Qwen3-30B-A3B | E3 a→e (P3 frame) | antonym_type=complementary | 24 (24) | 0.00 [0.00, 0.00] | 0.04 [0.00, 0.12] | 0.21 [0.08, 0.38] | 0.04 [0.00, 0.12] | 0 / 1 | other answer |
| Qwen3-30B-A3B | E3 a→e (P3 frame) | antonym_type=gradable | 1800 (1800) | 0.18 [0.16, 0.20] | 0.00 [0.00, 0.01] | 0.40 [0.38, 0.42] | 0.02 [0.01, 0.02] | 328 / 5 | other answer |
| Qwen3-30B-A3B | E3 a→e (P3 frame) | degree_adv=False | 1035 (1035) | 0.18 [0.15, 0.20] | 0.00 [0.00, 0.01] | 0.39 [0.36, 0.42] | 0.02 [0.01, 0.03] | 184 / 5 | other answer |
| Qwen3-30B-A3B | E3 a→e (P3 frame) | degree_adv=True | 789 (789) | 0.18 [0.16, 0.21] | 0.00 [0.00, 0.00] | 0.41 [0.37, 0.44] | 0.01 [0.01, 0.02] | 144 / 1 | other answer |
| Qwen3-30B-A3B | E3 a→e (P3 frame) | names=False | 1563 (1563) | 0.18 [0.16, 0.20] | 0.00 [0.00, 0.01] | 0.39 [0.36, 0.41] | 0.02 [0.01, 0.02] | 279 / 5 | other answer |
| Qwen3-30B-A3B | E3 a→e (P3 frame) | names=True | 261 (261) | 0.19 [0.14, 0.23] | 0.00 [0.00, 0.01] | 0.46 [0.40, 0.52] | 0.01 [0.00, 0.03] | 49 / 1 | other answer |
| Qwen3-30B-A3B | E3 a→e (P3 frame) | order=1 | 1323 (1323) | 0.18 [0.16, 0.20] | 0.00 [0.00, 0.01] | 0.39 [0.36, 0.41] | 0.02 [0.01, 0.03] | 236 / 5 | other answer |
| Qwen3-30B-A3B | E3 a→e (P3 frame) | order=2 | 501 (501) | 0.18 [0.15, 0.22] | 0.00 [0.00, 0.01] | 0.43 [0.39, 0.47] | 0.01 [0.00, 0.02] | 92 / 1 | other answer |
| Qwen3-30B-A3B | E3 a→e (F1 frame, Phase-5 anchor) | all | 1843 (1843) | 0.19 [0.17, 0.21] | 0.00 [0.00, 0.00] | 0.40 [0.37, 0.42] | 0.01 [0.00, 0.01] | 353 / 2 | other answer |
| Qwen3-30B-A3B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=complementary | 24 (24) | 0.00 [0.00, 0.00] | 0.04 [0.00, 0.12] | 0.17 [0.04, 0.33] | 0.04 [0.00, 0.12] | 0 / 1 | other answer |
| Qwen3-30B-A3B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=gradable | 1819 (1819) | 0.19 [0.18, 0.21] | 0.00 [0.00, 0.00] | 0.40 [0.38, 0.42] | 0.00 [0.00, 0.01] | 353 / 1 | other answer |
| Qwen3-30B-A3B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=False | 1038 (1038) | 0.22 [0.19, 0.24] | 0.00 [0.00, 0.00] | 0.43 [0.40, 0.46] | 0.01 [0.00, 0.01] | 225 / 2 | other answer |
| Qwen3-30B-A3B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=True | 805 (805) | 0.16 [0.13, 0.19] | 0.00 [0.00, 0.00] | 0.36 [0.32, 0.39] | 0.00 [0.00, 0.01] | 128 / 0 | other answer |
| Qwen3-30B-A3B | E3 a→e (F1 frame, Phase-5 anchor) | names=False | 1587 (1587) | 0.20 [0.18, 0.22] | 0.00 [0.00, 0.00] | 0.40 [0.37, 0.42] | 0.00 [0.00, 0.01] | 316 / 1 | other answer |
| Qwen3-30B-A3B | E3 a→e (F1 frame, Phase-5 anchor) | names=True | 256 (256) | 0.14 [0.10, 0.19] | 0.00 [0.00, 0.01] | 0.39 [0.32, 0.45] | 0.01 [0.00, 0.03] | 37 / 1 | other answer |
| Qwen3-30B-A3B | E3 a→e (F1 frame, Phase-5 anchor) | order=1 | 1324 (1324) | 0.19 [0.17, 0.21] | 0.00 [0.00, 0.00] | 0.39 [0.36, 0.41] | 0.01 [0.00, 0.01] | 250 / 2 | other answer |
| Qwen3-30B-A3B | E3 a→e (F1 frame, Phase-5 anchor) | order=2 | 519 (519) | 0.20 [0.16, 0.23] | 0.00 [0.00, 0.00] | 0.42 [0.38, 0.46] | 0.00 [0.00, 0.00] | 103 / 0 | other answer |
| Qwen3-30B-A3B | E0 prior only a→e | all | 416 (208) | 0.81 [0.76, 0.85] | 0.00 [0.00, 0.00] | 0.93 [0.91, 0.96] | 0.00 [0.00, 0.01] | 146 / 0 | other answer |
| Qwen3-30B-A3B | E0 prior only a→e | order=1 | 208 (208) | 0.81 [0.75, 0.86] | 0.00 [0.00, 0.00] | 0.93 [0.89, 0.96] | 0.00 [0.00, 0.02] | 168 / 0 | other answer |
| Qwen3-30B-A3B | E0 prior only a→e | order=2 | 208 (208) | 0.81 [0.75, 0.86] | 0.00 [0.00, 0.00] | 0.94 [0.90, 0.97] | 0.00 [0.00, 0.00] | 168 / 0 | other answer |
| Qwen3-30B-A3B | A-agree a→e | all | 406 (203) | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 200 / 0 | other answer |
| Qwen3-30B-A3B | A-agree a→e | order=1 | 203 (203) | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 201 / 0 | other answer |
| Qwen3-30B-A3B | A-agree a→e | order=2 | 203 (203) | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 202 / 0 | other answer |
| Qwen3-30B-A3B | A-conflict a→e | all | 340 (170) | 0.96 [0.94, 0.98] | 0.00 [0.00, 0.00] | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 157 / 0 | prior |
| Qwen3-30B-A3B | A-conflict a→e | order=1 | 170 (170) | 0.96 [0.93, 0.98] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 163 / 0 | prior |
| Qwen3-30B-A3B | A-conflict a→e | order=2 | 170 (170) | 0.96 [0.94, 0.99] | 0.00 [0.00, 0.00] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 164 / 0 | prior |
| Qwen3-30B-A3B | E1 e→a | all | 3578 (1789) | 0.57 [0.55, 0.59] | 0.00 [0.00, 0.00] | 0.73 [0.71, 0.75] | 0.01 [0.01, 0.01] | 795 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a | antonym_type=complementary | 50 (25) | 0.30 [0.16, 0.44] | 0.00 [0.00, 0.00] | 0.66 [0.54, 0.78] | 0.00 [0.00, 0.00] | 3 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a | antonym_type=gradable | 3528 (1764) | 0.58 [0.56, 0.60] | 0.00 [0.00, 0.00] | 0.73 [0.71, 0.75] | 0.01 [0.01, 0.01] | 792 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a | degree_adv=False | 1986 (993) | 0.46 [0.43, 0.48] | 0.00 [0.00, 0.01] | 0.64 [0.61, 0.66] | 0.01 [0.01, 0.02] | 302 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a | degree_adv=True | 1592 (796) | 0.72 [0.69, 0.74] | 0.00 [0.00, 0.00] | 0.85 [0.83, 0.87] | 0.00 [0.00, 0.01] | 493 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a | names=False | 3060 (1530) | 0.59 [0.57, 0.61] | 0.00 [0.00, 0.00] | 0.75 [0.73, 0.76] | 0.01 [0.01, 0.01] | 709 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a | names=True | 518 (259) | 0.49 [0.43, 0.53] | 0.00 [0.00, 0.00] | 0.65 [0.60, 0.70] | 0.01 [0.00, 0.02] | 86 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a | order=1 | 1789 (1789) | 0.58 [0.56, 0.61] | 0.00 [0.00, 0.00] | 0.74 [0.71, 0.76] | 0.01 [0.01, 0.02] | 1042 / 4 | other answer |
| Qwen3-30B-A3B | E1 e→a | order=2 | 1789 (1789) | 0.56 [0.54, 0.59] | 0.00 [0.00, 0.00] | 0.73 [0.71, 0.75] | 0.01 [0.00, 0.01] | 1006 / 1 | other answer |
| Qwen3-30B-A3B | E1 e→a (all strata) | all | 3578 (1789) | 0.57 [0.55, 0.59] | 0.00 [0.00, 0.00] | 0.73 [0.71, 0.75] | 0.01 [0.01, 0.01] | 795 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a (all strata) | antonym_type=complementary | 50 (25) | 0.30 [0.16, 0.44] | 0.00 [0.00, 0.00] | 0.66 [0.54, 0.78] | 0.00 [0.00, 0.00] | 3 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a (all strata) | antonym_type=gradable | 3528 (1764) | 0.58 [0.56, 0.60] | 0.00 [0.00, 0.00] | 0.73 [0.71, 0.75] | 0.01 [0.01, 0.01] | 792 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a (all strata) | degree_adv=False | 1986 (993) | 0.46 [0.43, 0.48] | 0.00 [0.00, 0.01] | 0.64 [0.61, 0.66] | 0.01 [0.01, 0.02] | 302 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a (all strata) | degree_adv=True | 1592 (796) | 0.72 [0.69, 0.74] | 0.00 [0.00, 0.00] | 0.85 [0.83, 0.87] | 0.00 [0.00, 0.01] | 493 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a (all strata) | names=False | 3060 (1530) | 0.59 [0.57, 0.61] | 0.00 [0.00, 0.00] | 0.75 [0.73, 0.76] | 0.01 [0.01, 0.01] | 709 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a (all strata) | names=True | 518 (259) | 0.49 [0.43, 0.53] | 0.00 [0.00, 0.00] | 0.65 [0.60, 0.70] | 0.01 [0.00, 0.02] | 86 / 0 | other answer |
| Qwen3-30B-A3B | E1 e→a (all strata) | order=1 | 1789 (1789) | 0.58 [0.56, 0.61] | 0.00 [0.00, 0.00] | 0.74 [0.71, 0.76] | 0.01 [0.01, 0.02] | 1042 / 4 | other answer |
| Qwen3-30B-A3B | E1 e→a (all strata) | order=2 | 1789 (1789) | 0.56 [0.54, 0.59] | 0.00 [0.00, 0.00] | 0.73 [0.71, 0.75] | 0.01 [0.00, 0.01] | 1006 / 1 | other answer |
| Qwen3-30B-A3B | E1p e→a | all | 3370 (1685) | 0.54 [0.52, 0.56] | 0.00 [0.00, 0.00] | 0.70 [0.68, 0.72] | 0.01 [0.00, 0.01] | 689 / 1 | other answer |
| Qwen3-30B-A3B | E1p e→a | antonym_type=complementary | 32 (16) | 0.38 [0.22, 0.56] | 0.00 [0.00, 0.00] | 0.59 [0.44, 0.75] | 0.00 [0.00, 0.00] | 3 / 0 | other answer |
| Qwen3-30B-A3B | E1p e→a | antonym_type=gradable | 3338 (1669) | 0.54 [0.52, 0.56] | 0.00 [0.00, 0.00] | 0.70 [0.68, 0.72] | 0.01 [0.00, 0.01] | 686 / 1 | other answer |
| Qwen3-30B-A3B | E1p e→a | degree_adv=False | 1790 (895) | 0.48 [0.45, 0.51] | 0.00 [0.00, 0.01] | 0.65 [0.63, 0.68] | 0.01 [0.01, 0.01] | 305 / 1 | other answer |
| Qwen3-30B-A3B | E1p e→a | degree_adv=True | 1580 (790) | 0.61 [0.58, 0.64] | 0.00 [0.00, 0.00] | 0.74 [0.72, 0.77] | 0.00 [0.00, 0.01] | 384 / 0 | other answer |
| Qwen3-30B-A3B | E1p e→a | names=False | 2904 (1452) | 0.54 [0.52, 0.56] | 0.00 [0.00, 0.00] | 0.69 [0.67, 0.71] | 0.01 [0.00, 0.01] | 595 / 1 | other answer |
| Qwen3-30B-A3B | E1p e→a | names=True | 466 (233) | 0.54 [0.49, 0.59] | 0.00 [0.00, 0.00] | 0.71 [0.65, 0.76] | 0.00 [0.00, 0.00] | 94 / 0 | other answer |
| Qwen3-30B-A3B | E1p e→a | order=1 | 1685 (1685) | 0.56 [0.53, 0.58] | 0.00 [0.00, 0.00] | 0.71 [0.69, 0.73] | 0.01 [0.00, 0.01] | 937 / 4 | other answer |
| Qwen3-30B-A3B | E1p e→a | order=2 | 1685 (1685) | 0.53 [0.50, 0.55] | 0.00 [0.00, 0.00] | 0.68 [0.66, 0.71] | 0.01 [0.00, 0.01] | 888 / 1 | other answer |
| Qwen3-30B-A3B | E2 e→a | all | 3578 (1789) | 0.03 [0.03, 0.04] | 0.02 [0.02, 0.03] | 0.15 [0.14, 0.17] | 0.13 [0.11, 0.14] | 12 / 9 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | antonym_type=complementary | 50 (25) | 0.06 [0.00, 0.12] | 0.00 [0.00, 0.00] | 0.22 [0.12, 0.34] | 0.00 [0.00, 0.00] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | antonym_type=gradable | 3528 (1764) | 0.03 [0.03, 0.04] | 0.02 [0.02, 0.03] | 0.15 [0.14, 0.16] | 0.13 [0.12, 0.14] | 12 / 9 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | degree_adv=False | 1986 (993) | 0.04 [0.03, 0.05] | 0.01 [0.01, 0.02] | 0.18 [0.16, 0.19] | 0.07 [0.06, 0.09] | 8 / 2 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | degree_adv=True | 1592 (796) | 0.02 [0.01, 0.03] | 0.04 [0.03, 0.05] | 0.13 [0.11, 0.14] | 0.19 [0.17, 0.21] | 4 / 7 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | names=False | 3060 (1530) | 0.03 [0.03, 0.04] | 0.02 [0.01, 0.03] | 0.15 [0.14, 0.17] | 0.13 [0.11, 0.14] | 10 / 8 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | names=True | 518 (259) | 0.03 [0.02, 0.05] | 0.04 [0.03, 0.06] | 0.15 [0.12, 0.19] | 0.12 [0.09, 0.16] | 2 / 1 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | order=1 | 1789 (1789) | 0.03 [0.02, 0.04] | 0.02 [0.01, 0.03] | 0.15 [0.13, 0.17] | 0.13 [0.12, 0.15] | 58 / 37 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E2 e→a | order=2 | 1789 (1789) | 0.04 [0.03, 0.04] | 0.03 [0.02, 0.03] | 0.16 [0.14, 0.17] | 0.12 [0.11, 0.14] | 64 / 46 | copy trap (word stated with the other entity) |
| Qwen3-30B-A3B | E3 e→a | all | 648 (648) | 0.98 [0.97, 0.99] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 636 / 0 | other answer |
| Qwen3-30B-A3B | E3 e→a | antonym_type=gradable | 643 (643) | 0.98 [0.97, 0.99] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 632 / 0 | other answer |
| Qwen3-30B-A3B | E3 e→a | degree_adv=False | 286 (286) | 0.97 [0.95, 0.99] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 277 / 0 | other answer |
| Qwen3-30B-A3B | E3 e→a | degree_adv=True | 362 (362) | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 359 / 0 | other answer |
| Qwen3-30B-A3B | E3 e→a | names=False | 617 (617) | 0.98 [0.97, 0.99] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 607 / 0 | other answer |
| Qwen3-30B-A3B | E3 e→a | names=True | 31 (31) | 0.94 [0.84, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 29 / 0 | other answer |
| Qwen3-30B-A3B | E3 e→a | order=1 | 418 (418) | 0.98 [0.96, 0.99] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 409 / 0 | other answer |
| Qwen3-30B-A3B | E3 e→a | order=2 | 230 (230) | 0.99 [0.97, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 227 / 0 | other answer |
| Qwen3-30B-A3B | E0 prior only e→a | all | 416 (208) | 0.34 [0.29, 0.40] | 0.00 [0.00, 0.00] | 0.63 [0.57, 0.69] | 0.00 [0.00, 0.01] | 50 / 0 | other answer |
| Qwen3-30B-A3B | E0 prior only e→a | order=1 | 208 (208) | 0.34 [0.27, 0.41] | 0.00 [0.00, 0.00] | 0.63 [0.56, 0.69] | 0.00 [0.00, 0.00] | 71 / 0 | other answer |
| Qwen3-30B-A3B | E0 prior only e→a | order=2 | 208 (208) | 0.34 [0.28, 0.41] | 0.00 [0.00, 0.00] | 0.63 [0.56, 0.70] | 0.01 [0.00, 0.02] | 71 / 0 | other answer |
| Qwen3-30B-A3B | A-agree e→a | all | 406 (203) | 0.64 [0.59, 0.69] | 0.00 [0.00, 0.00] | 0.80 [0.76, 0.84] | 0.00 [0.00, 0.01] | 89 / 0 | other answer |
| Qwen3-30B-A3B | A-agree e→a | order=1 | 203 (203) | 0.61 [0.54, 0.68] | 0.00 [0.00, 0.00] | 0.77 [0.71, 0.82] | 0.00 [0.00, 0.01] | 124 / 0 | other answer |
| Qwen3-30B-A3B | A-agree e→a | order=2 | 203 (203) | 0.67 [0.61, 0.73] | 0.00 [0.00, 0.00] | 0.83 [0.78, 0.88] | 0.00 [0.00, 0.00] | 136 / 0 | other answer |
| Qwen3-30B-A3B | A-conflict e→a | all | 340 (170) | 0.30 [0.25, 0.36] | 0.01 [0.00, 0.01] | 0.44 [0.39, 0.51] | 0.05 [0.03, 0.07] | 22 / 0 | prior |
| Qwen3-30B-A3B | A-conflict e→a | order=1 | 170 (170) | 0.31 [0.24, 0.38] | 0.01 [0.00, 0.02] | 0.45 [0.38, 0.53] | 0.04 [0.02, 0.07] | 52 / 1 | prior |
| Qwen3-30B-A3B | A-conflict e→a | order=2 | 170 (170) | 0.29 [0.22, 0.36] | 0.01 [0.00, 0.02] | 0.44 [0.36, 0.51] | 0.06 [0.02, 0.10] | 50 / 1 | prior |
| Qwen3-30B-A3B | Set C agree (a→e) | all | 3686 (1843) | 0.44 [0.42, 0.45] | 0.00 [0.00, 0.00] | 0.76 [0.74, 0.77] | 0.00 [0.00, 0.00] | 552 / 0 | event / other |
| Qwen3-30B-A3B | Set C agree (a→e) | antonym_type=complementary | 48 (24) | 0.31 [0.17, 0.46] | 0.00 [0.00, 0.00] | 0.79 [0.67, 0.90] | 0.00 [0.00, 0.00] | 4 / 0 | event / other |
| Qwen3-30B-A3B | Set C agree (a→e) | antonym_type=gradable | 3638 (1819) | 0.44 [0.42, 0.46] | 0.00 [0.00, 0.00] | 0.76 [0.74, 0.78] | 0.00 [0.00, 0.00] | 548 / 0 | event / other |
| Qwen3-30B-A3B | Set C agree (a→e) | degree_adv=False | 2074 (1037) | 0.48 [0.45, 0.50] | 0.00 [0.00, 0.00] | 0.76 [0.74, 0.78] | 0.00 [0.00, 0.00] | 357 / 0 | event / other |
| Qwen3-30B-A3B | Set C agree (a→e) | degree_adv=True | 1612 (806) | 0.38 [0.36, 0.41] | 0.00 [0.00, 0.00] | 0.75 [0.73, 0.78] | 0.00 [0.00, 0.00] | 195 / 0 | event / other |
| Qwen3-30B-A3B | Set C agree (a→e) | names=False | 3174 (1587) | 0.44 [0.42, 0.47] | 0.00 [0.00, 0.00] | 0.78 [0.76, 0.79] | 0.00 [0.00, 0.00] | 486 / 0 | event / other |
| Qwen3-30B-A3B | Set C agree (a→e) | names=True | 512 (256) | 0.39 [0.33, 0.44] | 0.00 [0.00, 0.01] | 0.65 [0.60, 0.70] | 0.00 [0.00, 0.01] | 66 / 0 | event / other |
| Qwen3-30B-A3B | Set C agree (a→e) | order=1 | 1843 (1843) | 0.45 [0.43, 0.47] | 0.00 [0.00, 0.00] | 0.77 [0.75, 0.79] | 0.00 [0.00, 0.00] | 830 / 1 | event / other |
| Qwen3-30B-A3B | Set C agree (a→e) | order=2 | 1843 (1843) | 0.42 [0.40, 0.44] | 0.00 [0.00, 0.00] | 0.75 [0.73, 0.77] | 0.00 [0.00, 0.00] | 779 / 0 | event / other |
| Qwen3-30B-A3B | sc_conflict_event_a2e | all | 3686 (1843) | 0.01 [0.01, 0.02] | 0.20 [0.18, 0.21] | 0.06 [0.05, 0.06] | 0.49 [0.47, 0.51] | 10 / 167 | event / other |
| Qwen3-30B-A3B | sc_conflict_event_a2e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.08 [0.00, 0.19] | 0.02 [0.00, 0.06] | 0.40 [0.25, 0.56] | 0 / 1 | event / other |
| Qwen3-30B-A3B | sc_conflict_event_a2e | antonym_type=gradable | 3638 (1819) | 0.01 [0.01, 0.02] | 0.20 [0.18, 0.21] | 0.06 [0.05, 0.06] | 0.49 [0.47, 0.51] | 10 / 166 | event / other |
| Qwen3-30B-A3B | sc_conflict_event_a2e | degree_adv=False | 2074 (1037) | 0.02 [0.02, 0.03] | 0.19 [0.17, 0.21] | 0.09 [0.07, 0.11] | 0.44 [0.41, 0.46] | 10 / 103 | event / other |
| Qwen3-30B-A3B | sc_conflict_event_a2e | degree_adv=True | 1612 (806) | 0.00 [0.00, 0.00] | 0.20 [0.18, 0.23] | 0.01 [0.01, 0.02] | 0.55 [0.52, 0.58] | 0 / 64 | event / other |
| Qwen3-30B-A3B | sc_conflict_event_a2e | names=False | 3174 (1587) | 0.02 [0.01, 0.02] | 0.19 [0.18, 0.21] | 0.06 [0.05, 0.07] | 0.48 [0.46, 0.50] | 10 / 141 | event / other |
| Qwen3-30B-A3B | sc_conflict_event_a2e | names=True | 512 (256) | 0.00 [0.00, 0.00] | 0.21 [0.17, 0.25] | 0.00 [0.00, 0.01] | 0.53 [0.48, 0.58] | 0 / 26 | event / other |
| Qwen3-30B-A3B | sc_conflict_event_a2e | order=1 | 1843 (1843) | 0.01 [0.01, 0.02] | 0.22 [0.20, 0.24] | 0.05 [0.04, 0.06] | 0.52 [0.50, 0.54] | 21 / 398 | event / other |
| Qwen3-30B-A3B | sc_conflict_event_a2e | order=2 | 1843 (1843) | 0.02 [0.01, 0.02] | 0.18 [0.16, 0.20] | 0.06 [0.05, 0.07] | 0.45 [0.43, 0.47] | 29 / 326 | event / other |
| Qwen3-30B-A3B | sc_conflict_stmt_a2e | all | 3686 (1843) | 0.20 [0.18, 0.21] | 0.01 [0.01, 0.02] | 0.49 [0.47, 0.51] | 0.06 [0.05, 0.06] | 167 / 10 | event / other |
| Qwen3-30B-A3B | sc_conflict_stmt_a2e | antonym_type=complementary | 48 (24) | 0.08 [0.00, 0.19] | 0.00 [0.00, 0.00] | 0.40 [0.25, 0.56] | 0.02 [0.00, 0.06] | 1 / 0 | event / other |
| Qwen3-30B-A3B | sc_conflict_stmt_a2e | antonym_type=gradable | 3638 (1819) | 0.20 [0.18, 0.21] | 0.01 [0.01, 0.02] | 0.49 [0.47, 0.51] | 0.06 [0.05, 0.06] | 166 / 10 | event / other |
| Qwen3-30B-A3B | sc_conflict_stmt_a2e | degree_adv=False | 2074 (1037) | 0.19 [0.17, 0.21] | 0.02 [0.02, 0.03] | 0.44 [0.41, 0.46] | 0.09 [0.07, 0.11] | 103 / 10 | event / other |
| Qwen3-30B-A3B | sc_conflict_stmt_a2e | degree_adv=True | 1612 (806) | 0.20 [0.18, 0.23] | 0.00 [0.00, 0.00] | 0.55 [0.52, 0.58] | 0.01 [0.01, 0.02] | 64 / 0 | event / other |
| Qwen3-30B-A3B | sc_conflict_stmt_a2e | names=False | 3174 (1587) | 0.19 [0.18, 0.21] | 0.02 [0.01, 0.02] | 0.48 [0.46, 0.50] | 0.06 [0.05, 0.07] | 141 / 10 | event / other |
| Qwen3-30B-A3B | sc_conflict_stmt_a2e | names=True | 512 (256) | 0.21 [0.17, 0.25] | 0.00 [0.00, 0.00] | 0.53 [0.48, 0.58] | 0.00 [0.00, 0.01] | 26 / 0 | event / other |
| Qwen3-30B-A3B | sc_conflict_stmt_a2e | order=1 | 1843 (1843) | 0.22 [0.20, 0.24] | 0.01 [0.01, 0.02] | 0.52 [0.50, 0.54] | 0.05 [0.04, 0.06] | 398 / 21 | event / other |
| Qwen3-30B-A3B | sc_conflict_stmt_a2e | order=2 | 1843 (1843) | 0.18 [0.16, 0.20] | 0.02 [0.01, 0.02] | 0.45 [0.43, 0.47] | 0.06 [0.05, 0.07] | 326 / 29 | event / other |
| Qwen3-14B | E1 a→e | all | 1178 (589) | 0.99 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 582 / 0 | other answer |
| Qwen3-14B | E1 a→e | antonym_type=complementary | 10 (5) | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 5 / 0 | other answer |
| Qwen3-14B | E1 a→e | antonym_type=gradable | 1168 (584) | 0.99 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 577 / 0 | other answer |
| Qwen3-14B | E1 a→e | degree_adv=False | 490 (245) | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 240 / 0 | other answer |
| Qwen3-14B | E1 a→e | degree_adv=True | 688 (344) | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 342 / 0 | other answer |
| Qwen3-14B | E1 a→e | names=False | 1116 (558) | 0.99 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 551 / 0 | other answer |
| Qwen3-14B | E1 a→e | names=True | 62 (31) | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 31 / 0 | other answer |
| Qwen3-14B | E1 a→e | order=1 | 589 (589) | 0.99 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 586 / 0 | other answer |
| Qwen3-14B | E1 a→e | order=2 | 589 (589) | 0.99 [0.99, 1.00] | 0.00 [0.00, 0.00] | 1.00 [0.99, 1.00] | 0.00 [0.00, 0.00] | 585 / 0 | other answer |
| Qwen3-14B | E1p a→e | all | 970 (485) | 0.96 [0.95, 0.98] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 458 / 0 | other answer |
| Qwen3-14B | E1p a→e | antonym_type=gradable | 962 (481) | 0.96 [0.95, 0.98] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 454 / 0 | other answer |
| Qwen3-14B | E1p a→e | degree_adv=False | 392 (196) | 0.96 [0.94, 0.98] | 0.00 [0.00, 0.00] | 0.98 [0.97, 0.99] | 0.00 [0.00, 0.00] | 184 / 0 | other answer |
| Qwen3-14B | E1p a→e | degree_adv=True | 578 (289) | 0.97 [0.95, 0.98] | 0.00 [0.00, 0.00] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 274 / 0 | other answer |
| Qwen3-14B | E1p a→e | names=False | 918 (459) | 0.97 [0.95, 0.98] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 434 / 0 | other answer |
| Qwen3-14B | E1p a→e | names=True | 52 (26) | 0.96 [0.90, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 24 / 0 | other answer |
| Qwen3-14B | E1p a→e | order=1 | 485 (485) | 0.96 [0.95, 0.98] | 0.00 [0.00, 0.00] | 0.98 [0.97, 0.99] | 0.00 [0.00, 0.00] | 468 / 0 | other answer |
| Qwen3-14B | E1p a→e | order=2 | 485 (485) | 0.96 [0.95, 0.98] | 0.00 [0.00, 0.00] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 468 / 0 | other answer |
| Qwen3-14B | E2 a→e | all | 3578 (1789) | 0.00 [0.00, 0.01] | 0.56 [0.54, 0.58] | 0.02 [0.01, 0.02] | 0.79 [0.78, 0.81] | 2 / 714 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | antonym_type=complementary | 48 (24) | 0.02 [0.00, 0.06] | 0.08 [0.00, 0.19] | 0.10 [0.02, 0.21] | 0.44 [0.29, 0.60] | 0 / 1 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | antonym_type=gradable | 3530 (1765) | 0.00 [0.00, 0.01] | 0.56 [0.55, 0.58] | 0.01 [0.01, 0.02] | 0.80 [0.78, 0.81] | 2 / 713 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | degree_adv=False | 1986 (993) | 0.01 [0.00, 0.01] | 0.40 [0.38, 0.42] | 0.02 [0.02, 0.03] | 0.70 [0.68, 0.73] | 2 / 213 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | degree_adv=True | 1592 (796) | 0.00 [0.00, 0.00] | 0.76 [0.73, 0.78] | 0.00 [0.00, 0.01] | 0.90 [0.89, 0.92] | 0 / 501 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | names=False | 3060 (1530) | 0.00 [0.00, 0.01] | 0.58 [0.56, 0.60] | 0.01 [0.01, 0.02] | 0.81 [0.80, 0.83] | 2 / 647 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | names=True | 518 (259) | 0.01 [0.00, 0.02] | 0.43 [0.38, 0.48] | 0.04 [0.03, 0.06] | 0.66 [0.61, 0.70] | 0 / 67 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | order=1 | 1789 (1789) | 0.00 [0.00, 0.01] | 0.56 [0.54, 0.58] | 0.02 [0.01, 0.02] | 0.79 [0.77, 0.81] | 8 / 1001 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 a→e | order=2 | 1789 (1789) | 0.01 [0.00, 0.01] | 0.55 [0.53, 0.58] | 0.02 [0.01, 0.02] | 0.80 [0.78, 0.81] | 9 / 992 | copy trap (word stated with the other entity) |
| Qwen3-14B | E3 a→e (P3 frame) | all | 1824 (1824) | 0.46 [0.43, 0.48] | 0.01 [0.00, 0.01] | 0.63 [0.60, 0.65] | 0.02 [0.01, 0.02] | 832 / 13 | other answer |
| Qwen3-14B | E3 a→e (P3 frame) | antonym_type=complementary | 24 (24) | 0.33 [0.17, 0.54] | 0.00 [0.00, 0.00] | 0.54 [0.38, 0.75] | 0.08 [0.00, 0.21] | 8 / 0 | other answer |
| Qwen3-14B | E3 a→e (P3 frame) | antonym_type=gradable | 1800 (1800) | 0.46 [0.43, 0.48] | 0.01 [0.00, 0.01] | 0.63 [0.60, 0.65] | 0.02 [0.01, 0.02] | 824 / 13 | other answer |
| Qwen3-14B | E3 a→e (P3 frame) | degree_adv=False | 1035 (1035) | 0.43 [0.40, 0.46] | 0.01 [0.00, 0.01] | 0.60 [0.57, 0.63] | 0.02 [0.01, 0.03] | 441 / 8 | other answer |
| Qwen3-14B | E3 a→e (P3 frame) | degree_adv=True | 789 (789) | 0.50 [0.46, 0.53] | 0.01 [0.00, 0.01] | 0.65 [0.62, 0.69] | 0.01 [0.01, 0.02] | 391 / 5 | other answer |
| Qwen3-14B | E3 a→e (P3 frame) | names=False | 1563 (1563) | 0.43 [0.40, 0.45] | 0.01 [0.00, 0.01] | 0.60 [0.57, 0.62] | 0.02 [0.01, 0.02] | 670 / 10 | other answer |
| Qwen3-14B | E3 a→e (P3 frame) | names=True | 261 (261) | 0.62 [0.56, 0.68] | 0.01 [0.00, 0.03] | 0.79 [0.74, 0.84] | 0.02 [0.01, 0.04] | 162 / 3 | other answer |
| Qwen3-14B | E3 a→e (P3 frame) | order=1 | 1323 (1323) | 0.46 [0.44, 0.49] | 0.01 [0.00, 0.01] | 0.64 [0.61, 0.66] | 0.02 [0.01, 0.03] | 614 / 12 | other answer |
| Qwen3-14B | E3 a→e (P3 frame) | order=2 | 501 (501) | 0.44 [0.39, 0.48] | 0.00 [0.00, 0.01] | 0.60 [0.55, 0.64] | 0.01 [0.00, 0.02] | 218 / 1 | other answer |
| Qwen3-14B | E3 a→e (F1 frame, Phase-5 anchor) | all | 1843 (1843) | 0.49 [0.47, 0.51] | 0.00 [0.00, 0.00] | 0.67 [0.65, 0.69] | 0.01 [0.00, 0.01] | 902 / 3 | other answer |
| Qwen3-14B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=complementary | 24 (24) | 0.21 [0.04, 0.38] | 0.04 [0.00, 0.12] | 0.42 [0.21, 0.62] | 0.04 [0.00, 0.12] | 5 / 1 | other answer |
| Qwen3-14B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=gradable | 1819 (1819) | 0.49 [0.47, 0.52] | 0.00 [0.00, 0.00] | 0.67 [0.65, 0.69] | 0.01 [0.00, 0.01] | 897 / 2 | other answer |
| Qwen3-14B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=False | 1038 (1038) | 0.48 [0.45, 0.51] | 0.00 [0.00, 0.00] | 0.68 [0.65, 0.71] | 0.01 [0.00, 0.01] | 502 / 2 | other answer |
| Qwen3-14B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=True | 805 (805) | 0.50 [0.46, 0.53] | 0.00 [0.00, 0.00] | 0.66 [0.63, 0.69] | 0.01 [0.00, 0.01] | 400 / 1 | other answer |
| Qwen3-14B | E3 a→e (F1 frame, Phase-5 anchor) | names=False | 1587 (1587) | 0.50 [0.47, 0.52] | 0.00 [0.00, 0.00] | 0.66 [0.64, 0.68] | 0.01 [0.00, 0.01] | 786 / 2 | other answer |
| Qwen3-14B | E3 a→e (F1 frame, Phase-5 anchor) | names=True | 256 (256) | 0.45 [0.39, 0.51] | 0.00 [0.00, 0.01] | 0.74 [0.68, 0.79] | 0.00 [0.00, 0.01] | 116 / 1 | other answer |
| Qwen3-14B | E3 a→e (F1 frame, Phase-5 anchor) | order=1 | 1324 (1324) | 0.49 [0.46, 0.52] | 0.00 [0.00, 0.01] | 0.67 [0.65, 0.70] | 0.01 [0.00, 0.01] | 650 / 3 | other answer |
| Qwen3-14B | E3 a→e (F1 frame, Phase-5 anchor) | order=2 | 519 (519) | 0.49 [0.44, 0.53] | 0.00 [0.00, 0.00] | 0.66 [0.61, 0.70] | 0.01 [0.00, 0.01] | 252 / 0 | other answer |
| Qwen3-14B | E1 e→a | all | 3578 (1789) | 0.45 [0.43, 0.47] | 0.00 [0.00, 0.00] | 0.66 [0.64, 0.67] | 0.02 [0.01, 0.02] | 559 / 0 | other answer |
| Qwen3-14B | E1 e→a | antonym_type=complementary | 50 (25) | 0.18 [0.08, 0.30] | 0.02 [0.00, 0.06] | 0.52 [0.36, 0.66] | 0.04 [0.00, 0.10] | 1 / 0 | other answer |
| Qwen3-14B | E1 e→a | antonym_type=gradable | 3528 (1764) | 0.46 [0.44, 0.48] | 0.00 [0.00, 0.00] | 0.66 [0.64, 0.68] | 0.02 [0.01, 0.02] | 558 / 0 | other answer |
| Qwen3-14B | E1 e→a | degree_adv=False | 1986 (993) | 0.28 [0.26, 0.30] | 0.00 [0.00, 0.01] | 0.52 [0.50, 0.55] | 0.03 [0.02, 0.04] | 131 / 0 | other answer |
| Qwen3-14B | E1 e→a | degree_adv=True | 1592 (796) | 0.67 [0.64, 0.69] | 0.00 [0.00, 0.00] | 0.82 [0.80, 0.84] | 0.00 [0.00, 0.01] | 428 / 0 | other answer |
| Qwen3-14B | E1 e→a | names=False | 3060 (1530) | 0.48 [0.46, 0.50] | 0.00 [0.00, 0.00] | 0.70 [0.68, 0.72] | 0.01 [0.01, 0.01] | 514 / 0 | other answer |
| Qwen3-14B | E1 e→a | names=True | 518 (259) | 0.29 [0.24, 0.34] | 0.01 [0.00, 0.02] | 0.42 [0.38, 0.47] | 0.06 [0.04, 0.08] | 45 / 0 | other answer |
| Qwen3-14B | E1 e→a | order=1 | 1789 (1789) | 0.45 [0.43, 0.48] | 0.00 [0.00, 0.01] | 0.66 [0.64, 0.68] | 0.02 [0.01, 0.03] | 810 / 5 | other answer |
| Qwen3-14B | E1 e→a | order=2 | 1789 (1789) | 0.45 [0.43, 0.47] | 0.00 [0.00, 0.01] | 0.65 [0.63, 0.67] | 0.01 [0.01, 0.02] | 809 / 4 | other answer |
| Qwen3-14B | E1p e→a | all | 3370 (1685) | 0.48 [0.46, 0.50] | 0.00 [0.00, 0.00] | 0.70 [0.68, 0.71] | 0.01 [0.00, 0.01] | 587 / 0 | other answer |
| Qwen3-14B | E1p e→a | antonym_type=complementary | 32 (16) | 0.16 [0.03, 0.31] | 0.00 [0.00, 0.00] | 0.66 [0.47, 0.84] | 0.00 [0.00, 0.00] | 1 / 0 | other answer |
| Qwen3-14B | E1p e→a | antonym_type=gradable | 3338 (1669) | 0.49 [0.47, 0.51] | 0.00 [0.00, 0.00] | 0.70 [0.68, 0.71] | 0.01 [0.00, 0.01] | 586 / 0 | other answer |
| Qwen3-14B | E1p e→a | degree_adv=False | 1790 (895) | 0.43 [0.40, 0.46] | 0.00 [0.00, 0.00] | 0.67 [0.64, 0.69] | 0.01 [0.00, 0.01] | 253 / 0 | other answer |
| Qwen3-14B | E1p e→a | degree_adv=True | 1580 (790) | 0.54 [0.51, 0.57] | 0.00 [0.00, 0.00] | 0.73 [0.70, 0.75] | 0.00 [0.00, 0.00] | 334 / 0 | other answer |
| Qwen3-14B | E1p e→a | names=False | 2904 (1452) | 0.49 [0.47, 0.51] | 0.00 [0.00, 0.00] | 0.70 [0.68, 0.72] | 0.00 [0.00, 0.01] | 511 / 0 | other answer |
| Qwen3-14B | E1p e→a | names=True | 466 (233) | 0.44 [0.39, 0.50] | 0.00 [0.00, 0.00] | 0.64 [0.59, 0.69] | 0.02 [0.00, 0.03] | 76 / 0 | other answer |
| Qwen3-14B | E1p e→a | order=1 | 1685 (1685) | 0.49 [0.46, 0.51] | 0.00 [0.00, 0.00] | 0.70 [0.68, 0.72] | 0.00 [0.00, 0.01] | 821 / 0 | other answer |
| Qwen3-14B | E1p e→a | order=2 | 1685 (1685) | 0.48 [0.45, 0.50] | 0.00 [0.00, 0.00] | 0.69 [0.66, 0.71] | 0.01 [0.00, 0.01] | 805 / 2 | other answer |
| Qwen3-14B | E2 e→a | all | 3578 (1789) | 0.06 [0.05, 0.07] | 0.01 [0.01, 0.01] | 0.25 [0.23, 0.27] | 0.09 [0.08, 0.10] | 19 / 2 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | antonym_type=complementary | 50 (25) | 0.06 [0.00, 0.14] | 0.00 [0.00, 0.00] | 0.20 [0.08, 0.34] | 0.02 [0.00, 0.06] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | antonym_type=gradable | 3528 (1764) | 0.06 [0.05, 0.07] | 0.01 [0.01, 0.01] | 0.25 [0.24, 0.27] | 0.09 [0.08, 0.10] | 19 / 2 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | degree_adv=False | 1986 (993) | 0.07 [0.06, 0.08] | 0.00 [0.00, 0.01] | 0.28 [0.26, 0.30] | 0.04 [0.03, 0.05] | 11 / 0 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | degree_adv=True | 1592 (796) | 0.05 [0.04, 0.06] | 0.02 [0.01, 0.02] | 0.21 [0.19, 0.23] | 0.14 [0.12, 0.16] | 8 / 2 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | names=False | 3060 (1530) | 0.04 [0.04, 0.05] | 0.01 [0.01, 0.01] | 0.23 [0.21, 0.24] | 0.09 [0.08, 0.10] | 8 / 1 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | names=True | 518 (259) | 0.14 [0.11, 0.17] | 0.02 [0.01, 0.03] | 0.39 [0.34, 0.43] | 0.07 [0.04, 0.09] | 11 / 1 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | order=1 | 1789 (1789) | 0.06 [0.05, 0.07] | 0.01 [0.00, 0.01] | 0.26 [0.24, 0.28] | 0.09 [0.07, 0.10] | 101 / 14 | copy trap (word stated with the other entity) |
| Qwen3-14B | E2 e→a | order=2 | 1789 (1789) | 0.06 [0.05, 0.07] | 0.01 [0.01, 0.02] | 0.24 [0.23, 0.26] | 0.08 [0.07, 0.10] | 108 / 22 | copy trap (word stated with the other entity) |
| Qwen3-14B | E3 e→a | all | 648 (648) | 0.86 [0.83, 0.89] | 0.00 [0.00, 0.00] | 0.94 [0.92, 0.96] | 0.00 [0.00, 0.00] | 559 / 0 | other answer |
| Qwen3-14B | E3 e→a | antonym_type=gradable | 643 (643) | 0.86 [0.84, 0.89] | 0.00 [0.00, 0.00] | 0.94 [0.93, 0.96] | 0.00 [0.00, 0.00] | 556 / 0 | other answer |
| Qwen3-14B | E3 e→a | degree_adv=False | 286 (286) | 0.84 [0.79, 0.87] | 0.00 [0.00, 0.00] | 0.94 [0.92, 0.97] | 0.00 [0.00, 0.01] | 239 / 0 | other answer |
| Qwen3-14B | E3 e→a | degree_adv=True | 362 (362) | 0.88 [0.85, 0.91] | 0.00 [0.00, 0.00] | 0.94 [0.91, 0.96] | 0.00 [0.00, 0.00] | 320 / 0 | other answer |
| Qwen3-14B | E3 e→a | names=False | 617 (617) | 0.87 [0.85, 0.90] | 0.00 [0.00, 0.00] | 0.94 [0.93, 0.96] | 0.00 [0.00, 0.00] | 538 / 0 | other answer |
| Qwen3-14B | E3 e→a | names=True | 31 (31) | 0.68 [0.52, 0.84] | 0.00 [0.00, 0.00] | 0.87 [0.74, 0.97] | 0.00 [0.00, 0.00] | 21 / 0 | other answer |
| Qwen3-14B | E3 e→a | order=1 | 418 (418) | 0.88 [0.85, 0.91] | 0.00 [0.00, 0.00] | 0.95 [0.93, 0.97] | 0.00 [0.00, 0.00] | 368 / 0 | other answer |
| Qwen3-14B | E3 e→a | order=2 | 230 (230) | 0.83 [0.78, 0.88] | 0.00 [0.00, 0.00] | 0.93 [0.89, 0.96] | 0.00 [0.00, 0.01] | 191 / 0 | other answer |
| Qwen3-14B | Set C agree (a→e) | all | 1286 (643) | 0.86 [0.84, 0.88] | 0.00 [0.00, 0.00] | 0.96 [0.94, 0.97] | 0.00 [0.00, 0.00] | 512 / 0 | event / other |
| Qwen3-14B | Set C agree (a→e) | antonym_type=complementary | 10 (5) | 0.90 [0.70, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 4 / 0 | event / other |
| Qwen3-14B | Set C agree (a→e) | antonym_type=gradable | 1276 (638) | 0.86 [0.84, 0.88] | 0.00 [0.00, 0.00] | 0.96 [0.94, 0.97] | 0.00 [0.00, 0.00] | 508 / 0 | event / other |
| Qwen3-14B | Set C agree (a→e) | degree_adv=False | 570 (285) | 0.76 [0.72, 0.80] | 0.00 [0.00, 0.00] | 0.93 [0.90, 0.95] | 0.00 [0.00, 0.00] | 185 / 0 | event / other |
| Qwen3-14B | Set C agree (a→e) | degree_adv=True | 716 (358) | 0.94 [0.92, 0.96] | 0.00 [0.00, 0.00] | 0.98 [0.97, 0.99] | 0.00 [0.00, 0.00] | 327 / 0 | event / other |
| Qwen3-14B | Set C agree (a→e) | names=False | 1224 (612) | 0.86 [0.84, 0.88] | 0.00 [0.00, 0.00] | 0.96 [0.94, 0.97] | 0.00 [0.00, 0.00] | 485 / 0 | event / other |
| Qwen3-14B | Set C agree (a→e) | names=True | 62 (31) | 0.92 [0.84, 0.98] | 0.00 [0.00, 0.00] | 0.97 [0.92, 1.00] | 0.02 [0.00, 0.05] | 27 / 0 | event / other |
| Qwen3-14B | Set C agree (a→e) | order=1 | 643 (643) | 0.86 [0.83, 0.88] | 0.00 [0.00, 0.00] | 0.95 [0.93, 0.97] | 0.00 [0.00, 0.00] | 552 / 0 | event / other |
| Qwen3-14B | Set C agree (a→e) | order=2 | 643 (643) | 0.86 [0.84, 0.89] | 0.00 [0.00, 0.00] | 0.96 [0.95, 0.98] | 0.00 [0.00, 0.00] | 556 / 0 | event / other |
| Qwen3-14B | sc_conflict_event_a2e | all | 3686 (1843) | 0.05 [0.04, 0.06] | 0.53 [0.51, 0.55] | 0.09 [0.08, 0.10] | 0.70 [0.68, 0.72] | 88 / 768 | event / other |
| Qwen3-14B | sc_conflict_event_a2e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.31 [0.15, 0.48] | 0.00 [0.00, 0.00] | 0.65 [0.46, 0.81] | 0 / 5 | event / other |
| Qwen3-14B | sc_conflict_event_a2e | antonym_type=gradable | 3638 (1819) | 0.05 [0.04, 0.07] | 0.53 [0.51, 0.55] | 0.09 [0.08, 0.10] | 0.70 [0.68, 0.72] | 88 / 763 | event / other |
| Qwen3-14B | sc_conflict_event_a2e | degree_adv=False | 2074 (1037) | 0.09 [0.07, 0.11] | 0.48 [0.46, 0.51] | 0.13 [0.11, 0.16] | 0.66 [0.64, 0.69] | 85 / 392 | event / other |
| Qwen3-14B | sc_conflict_event_a2e | degree_adv=True | 1612 (806) | 0.01 [0.00, 0.01] | 0.58 [0.55, 0.61] | 0.03 [0.02, 0.04] | 0.75 [0.73, 0.78] | 3 / 376 | event / other |
| Qwen3-14B | sc_conflict_event_a2e | names=False | 3174 (1587) | 0.06 [0.05, 0.07] | 0.50 [0.47, 0.52] | 0.10 [0.09, 0.12] | 0.67 [0.65, 0.69] | 88 / 618 | event / other |
| Qwen3-14B | sc_conflict_event_a2e | names=True | 512 (256) | 0.00 [0.00, 0.00] | 0.71 [0.67, 0.76] | 0.00 [0.00, 0.01] | 0.87 [0.83, 0.90] | 0 / 150 | event / other |
| Qwen3-14B | sc_conflict_event_a2e | order=1 | 1843 (1843) | 0.05 [0.04, 0.07] | 0.55 [0.52, 0.57] | 0.08 [0.07, 0.10] | 0.71 [0.68, 0.72] | 100 / 1009 | event / other |
| Qwen3-14B | sc_conflict_event_a2e | order=2 | 1843 (1843) | 0.05 [0.04, 0.06] | 0.50 [0.48, 0.53] | 0.09 [0.08, 0.11] | 0.70 [0.68, 0.72] | 98 / 928 | event / other |
| Qwen3-14B | sc_conflict_stmt_a2e | all | 3686 (1843) | 0.53 [0.51, 0.55] | 0.05 [0.04, 0.06] | 0.70 [0.68, 0.72] | 0.09 [0.08, 0.10] | 768 / 88 | event / other |
| Qwen3-14B | sc_conflict_stmt_a2e | antonym_type=complementary | 48 (24) | 0.31 [0.15, 0.48] | 0.00 [0.00, 0.00] | 0.65 [0.46, 0.81] | 0.00 [0.00, 0.00] | 5 / 0 | event / other |
| Qwen3-14B | sc_conflict_stmt_a2e | antonym_type=gradable | 3638 (1819) | 0.53 [0.51, 0.55] | 0.05 [0.04, 0.07] | 0.70 [0.68, 0.72] | 0.09 [0.08, 0.10] | 763 / 88 | event / other |
| Qwen3-14B | sc_conflict_stmt_a2e | degree_adv=False | 2074 (1037) | 0.48 [0.46, 0.51] | 0.09 [0.07, 0.11] | 0.66 [0.64, 0.69] | 0.13 [0.11, 0.16] | 392 / 85 | event / other |
| Qwen3-14B | sc_conflict_stmt_a2e | degree_adv=True | 1612 (806) | 0.58 [0.55, 0.61] | 0.01 [0.00, 0.01] | 0.75 [0.73, 0.78] | 0.03 [0.02, 0.04] | 376 / 3 | event / other |
| Qwen3-14B | sc_conflict_stmt_a2e | names=False | 3174 (1587) | 0.50 [0.47, 0.52] | 0.06 [0.05, 0.07] | 0.67 [0.65, 0.69] | 0.10 [0.09, 0.12] | 618 / 88 | event / other |
| Qwen3-14B | sc_conflict_stmt_a2e | names=True | 512 (256) | 0.71 [0.67, 0.76] | 0.00 [0.00, 0.00] | 0.87 [0.83, 0.90] | 0.00 [0.00, 0.01] | 150 / 0 | event / other |
| Qwen3-14B | sc_conflict_stmt_a2e | order=1 | 1843 (1843) | 0.55 [0.52, 0.57] | 0.05 [0.04, 0.07] | 0.71 [0.68, 0.72] | 0.08 [0.07, 0.10] | 1009 / 100 | event / other |
| Qwen3-14B | sc_conflict_stmt_a2e | order=2 | 1843 (1843) | 0.50 [0.48, 0.53] | 0.05 [0.04, 0.06] | 0.70 [0.68, 0.72] | 0.09 [0.08, 0.11] | 928 / 98 | event / other |
| Qwen3-1.7B | E1 a→e | all | 1178 (589) | 0.16 [0.13, 0.19] | 0.00 [0.00, 0.00] | 0.43 [0.40, 0.47] | 0.00 [0.00, 0.00] | 53 / 0 | other answer |
| Qwen3-1.7B | E1 a→e | antonym_type=complementary | 10 (5) | 0.30 [0.00, 0.70] | 0.00 [0.00, 0.00] | 0.50 [0.10, 0.90] | 0.00 [0.00, 0.00] | 1 / 0 | other answer |
| Qwen3-1.7B | E1 a→e | antonym_type=gradable | 1168 (584) | 0.16 [0.13, 0.19] | 0.00 [0.00, 0.00] | 0.43 [0.40, 0.46] | 0.00 [0.00, 0.00] | 52 / 0 | other answer |
| Qwen3-1.7B | E1 a→e | degree_adv=False | 490 (245) | 0.18 [0.14, 0.22] | 0.00 [0.00, 0.00] | 0.40 [0.35, 0.45] | 0.00 [0.00, 0.01] | 24 / 0 | other answer |
| Qwen3-1.7B | E1 a→e | degree_adv=True | 688 (344) | 0.15 [0.12, 0.18] | 0.00 [0.00, 0.00] | 0.46 [0.42, 0.50] | 0.00 [0.00, 0.00] | 29 / 0 | other answer |
| Qwen3-1.7B | E1 a→e | names=False | 1116 (558) | 0.12 [0.10, 0.14] | 0.00 [0.00, 0.00] | 0.41 [0.38, 0.44] | 0.00 [0.00, 0.00] | 28 / 0 | other answer |
| Qwen3-1.7B | E1 a→e | names=True | 62 (31) | 0.87 [0.77, 0.95] | 0.00 [0.00, 0.00] | 0.95 [0.89, 1.00] | 0.00 [0.00, 0.00] | 25 / 0 | other answer |
| Qwen3-1.7B | E1 a→e | order=1 | 589 (589) | 0.15 [0.12, 0.18] | 0.00 [0.00, 0.00] | 0.43 [0.39, 0.47] | 0.00 [0.00, 0.01] | 89 / 0 | other answer |
| Qwen3-1.7B | E1 a→e | order=2 | 589 (589) | 0.17 [0.14, 0.20] | 0.00 [0.00, 0.00] | 0.44 [0.40, 0.48] | 0.00 [0.00, 0.00] | 100 / 0 | other answer |
| Qwen3-1.7B | E1p a→e | all | 970 (485) | 0.11 [0.09, 0.14] | 0.00 [0.00, 0.00] | 0.33 [0.30, 0.37] | 0.00 [0.00, 0.00] | 25 / 0 | other answer |
| Qwen3-1.7B | E1p a→e | antonym_type=gradable | 962 (481) | 0.11 [0.09, 0.13] | 0.00 [0.00, 0.00] | 0.33 [0.30, 0.36] | 0.00 [0.00, 0.00] | 24 / 0 | other answer |
| Qwen3-1.7B | E1p a→e | degree_adv=False | 392 (196) | 0.12 [0.08, 0.16] | 0.00 [0.00, 0.00] | 0.35 [0.31, 0.40] | 0.00 [0.00, 0.00] | 10 / 0 | other answer |
| Qwen3-1.7B | E1p a→e | degree_adv=True | 578 (289) | 0.11 [0.07, 0.14] | 0.00 [0.00, 0.00] | 0.32 [0.28, 0.37] | 0.00 [0.00, 0.00] | 15 / 0 | other answer |
| Qwen3-1.7B | E1p a→e | names=False | 918 (459) | 0.08 [0.06, 0.09] | 0.00 [0.00, 0.00] | 0.30 [0.27, 0.33] | 0.00 [0.00, 0.00] | 10 / 0 | other answer |
| Qwen3-1.7B | E1p a→e | names=True | 52 (26) | 0.71 [0.56, 0.85] | 0.00 [0.00, 0.00] | 0.85 [0.75, 0.94] | 0.00 [0.00, 0.00] | 15 / 0 | other answer |
| Qwen3-1.7B | E1p a→e | order=1 | 485 (485) | 0.13 [0.10, 0.16] | 0.00 [0.00, 0.00] | 0.35 [0.31, 0.39] | 0.00 [0.00, 0.00] | 61 / 0 | other answer |
| Qwen3-1.7B | E1p a→e | order=2 | 485 (485) | 0.09 [0.07, 0.12] | 0.00 [0.00, 0.00] | 0.32 [0.27, 0.36] | 0.00 [0.00, 0.00] | 46 / 0 | other answer |
| Qwen3-1.7B | E2 a→e | all | 1178 (589) | 0.00 [0.00, 0.00] | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.01] | 0.26 [0.23, 0.28] | 0 / 8 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | antonym_type=complementary | 10 (5) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.10 [0.00, 0.30] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | antonym_type=gradable | 1168 (584) | 0.00 [0.00, 0.00] | 0.04 [0.03, 0.05] | 0.00 [0.00, 0.01] | 0.26 [0.23, 0.28] | 0 / 8 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | degree_adv=False | 490 (245) | 0.00 [0.00, 0.00] | 0.04 [0.02, 0.06] | 0.00 [0.00, 0.01] | 0.29 [0.25, 0.32] | 0 / 2 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | degree_adv=True | 688 (344) | 0.00 [0.00, 0.00] | 0.04 [0.02, 0.06] | 0.00 [0.00, 0.01] | 0.24 [0.20, 0.27] | 0 / 6 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | names=False | 1116 (558) | 0.00 [0.00, 0.00] | 0.02 [0.01, 0.03] | 0.00 [0.00, 0.01] | 0.24 [0.22, 0.26] | 0 / 1 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | names=True | 62 (31) | 0.00 [0.00, 0.00] | 0.39 [0.26, 0.52] | 0.00 [0.00, 0.00] | 0.58 [0.47, 0.71] | 0 / 7 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | order=1 | 589 (589) | 0.00 [0.00, 0.00] | 0.04 [0.03, 0.06] | 0.00 [0.00, 0.00] | 0.24 [0.21, 0.28] | 0 / 25 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 a→e | order=2 | 589 (589) | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.05] | 0.01 [0.00, 0.01] | 0.27 [0.23, 0.31] | 0 / 20 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E3 a→e (P3 frame) | all | 624 (624) | 0.05 [0.04, 0.07] | 0.00 [0.00, 0.00] | 0.17 [0.14, 0.21] | 0.00 [0.00, 0.00] | 32 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (P3 frame) | antonym_type=gradable | 619 (619) | 0.05 [0.04, 0.07] | 0.00 [0.00, 0.00] | 0.17 [0.15, 0.20] | 0.00 [0.00, 0.00] | 32 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (P3 frame) | degree_adv=False | 276 (276) | 0.05 [0.03, 0.08] | 0.00 [0.00, 0.00] | 0.21 [0.16, 0.26] | 0.00 [0.00, 0.00] | 13 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (P3 frame) | degree_adv=True | 348 (348) | 0.05 [0.03, 0.08] | 0.00 [0.00, 0.00] | 0.14 [0.11, 0.18] | 0.00 [0.00, 0.01] | 19 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (P3 frame) | names=False | 593 (593) | 0.05 [0.03, 0.07] | 0.00 [0.00, 0.00] | 0.17 [0.14, 0.20] | 0.00 [0.00, 0.01] | 29 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (P3 frame) | names=True | 31 (31) | 0.10 [0.00, 0.19] | 0.00 [0.00, 0.00] | 0.29 [0.16, 0.45] | 0.00 [0.00, 0.00] | 3 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (P3 frame) | order=1 | 409 (409) | 0.04 [0.03, 0.07] | 0.00 [0.00, 0.00] | 0.17 [0.14, 0.21] | 0.00 [0.00, 0.01] | 18 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (P3 frame) | order=2 | 215 (215) | 0.07 [0.03, 0.10] | 0.00 [0.00, 0.00] | 0.17 [0.12, 0.22] | 0.00 [0.00, 0.00] | 14 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (F1 frame, Phase-5 anchor) | all | 643 (643) | 0.05 [0.03, 0.06] | 0.00 [0.00, 0.00] | 0.19 [0.16, 0.22] | 0.01 [0.00, 0.02] | 30 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=gradable | 638 (638) | 0.05 [0.03, 0.06] | 0.00 [0.00, 0.00] | 0.19 [0.16, 0.22] | 0.01 [0.00, 0.02] | 30 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=False | 285 (285) | 0.05 [0.02, 0.07] | 0.00 [0.00, 0.00] | 0.22 [0.18, 0.27] | 0.01 [0.00, 0.02] | 13 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=True | 358 (358) | 0.05 [0.03, 0.07] | 0.00 [0.00, 0.00] | 0.16 [0.12, 0.19] | 0.01 [0.00, 0.02] | 17 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (F1 frame, Phase-5 anchor) | names=False | 612 (612) | 0.05 [0.03, 0.07] | 0.00 [0.00, 0.00] | 0.19 [0.16, 0.22] | 0.01 [0.00, 0.01] | 30 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (F1 frame, Phase-5 anchor) | names=True | 31 (31) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.16] | 0.10 [0.00, 0.19] | 0 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (F1 frame, Phase-5 anchor) | order=1 | 415 (415) | 0.03 [0.02, 0.05] | 0.00 [0.00, 0.00] | 0.17 [0.14, 0.21] | 0.02 [0.00, 0.03] | 14 / 0 | other answer |
| Qwen3-1.7B | E3 a→e (F1 frame, Phase-5 anchor) | order=2 | 228 (228) | 0.07 [0.04, 0.11] | 0.00 [0.00, 0.00] | 0.21 [0.16, 0.26] | 0.00 [0.00, 0.00] | 16 / 0 | other answer |
| Qwen3-1.7B | E1 e→a | all | 1178 (589) | 0.24 [0.21, 0.27] | 0.00 [0.00, 0.01] | 0.52 [0.49, 0.56] | 0.05 [0.03, 0.06] | 85 / 0 | other answer |
| Qwen3-1.7B | E1 e→a | antonym_type=complementary | 10 (5) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.10 [0.00, 0.30] | 0.10 [0.00, 0.30] | 0 / 0 | other answer |
| Qwen3-1.7B | E1 e→a | antonym_type=gradable | 1168 (584) | 0.24 [0.21, 0.27] | 0.00 [0.00, 0.01] | 0.53 [0.49, 0.56] | 0.05 [0.03, 0.06] | 85 / 0 | other answer |
| Qwen3-1.7B | E1 e→a | degree_adv=False | 490 (245) | 0.15 [0.11, 0.19] | 0.01 [0.00, 0.01] | 0.32 [0.27, 0.37] | 0.10 [0.08, 0.13] | 25 / 0 | other answer |
| Qwen3-1.7B | E1 e→a | degree_adv=True | 688 (344) | 0.30 [0.26, 0.34] | 0.00 [0.00, 0.00] | 0.67 [0.63, 0.71] | 0.01 [0.00, 0.01] | 60 / 0 | other answer |
| Qwen3-1.7B | E1 e→a | names=False | 1116 (558) | 0.24 [0.22, 0.27] | 0.00 [0.00, 0.00] | 0.53 [0.50, 0.57] | 0.05 [0.03, 0.06] | 84 / 0 | other answer |
| Qwen3-1.7B | E1 e→a | names=True | 62 (31) | 0.11 [0.03, 0.21] | 0.02 [0.00, 0.05] | 0.37 [0.24, 0.50] | 0.03 [0.00, 0.08] | 1 / 0 | other answer |
| Qwen3-1.7B | E1 e→a | order=1 | 589 (589) | 0.24 [0.21, 0.28] | 0.00 [0.00, 0.01] | 0.53 [0.49, 0.57] | 0.05 [0.04, 0.07] | 144 / 1 | other answer |
| Qwen3-1.7B | E1 e→a | order=2 | 589 (589) | 0.23 [0.20, 0.26] | 0.00 [0.00, 0.01] | 0.51 [0.47, 0.56] | 0.04 [0.02, 0.06] | 136 / 2 | other answer |
| Qwen3-1.7B | E1p e→a | all | 970 (485) | 0.22 [0.19, 0.25] | 0.00 [0.00, 0.00] | 0.50 [0.47, 0.54] | 0.01 [0.00, 0.01] | 69 / 0 | other answer |
| Qwen3-1.7B | E1p e→a | antonym_type=gradable | 962 (481) | 0.22 [0.19, 0.25] | 0.00 [0.00, 0.00] | 0.51 [0.47, 0.55] | 0.01 [0.00, 0.01] | 69 / 0 | other answer |
| Qwen3-1.7B | E1p e→a | degree_adv=False | 392 (196) | 0.19 [0.14, 0.24] | 0.00 [0.00, 0.00] | 0.48 [0.43, 0.54] | 0.02 [0.01, 0.03] | 26 / 0 | other answer |
| Qwen3-1.7B | E1p e→a | degree_adv=True | 578 (289) | 0.24 [0.20, 0.28] | 0.00 [0.00, 0.00] | 0.52 [0.47, 0.57] | 0.00 [0.00, 0.01] | 43 / 0 | other answer |
| Qwen3-1.7B | E1p e→a | names=False | 918 (459) | 0.22 [0.18, 0.25] | 0.00 [0.00, 0.00] | 0.49 [0.46, 0.53] | 0.01 [0.00, 0.01] | 65 / 0 | other answer |
| Qwen3-1.7B | E1p e→a | names=True | 52 (26) | 0.29 [0.15, 0.44] | 0.00 [0.00, 0.00] | 0.65 [0.50, 0.79] | 0.00 [0.00, 0.00] | 4 / 0 | other answer |
| Qwen3-1.7B | E1p e→a | order=1 | 485 (485) | 0.24 [0.20, 0.27] | 0.00 [0.00, 0.00] | 0.51 [0.46, 0.55] | 0.01 [0.00, 0.02] | 115 / 0 | other answer |
| Qwen3-1.7B | E1p e→a | order=2 | 485 (485) | 0.20 [0.17, 0.24] | 0.00 [0.00, 0.00] | 0.50 [0.46, 0.54] | 0.00 [0.00, 0.01] | 98 / 0 | other answer |
| Qwen3-1.7B | E2 e→a | all | 1178 (589) | 0.09 [0.07, 0.10] | 0.00 [0.00, 0.00] | 0.36 [0.33, 0.39] | 0.02 [0.01, 0.03] | 15 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | antonym_type=complementary | 10 (5) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.20 [0.00, 0.40] | 0.00 [0.00, 0.00] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | antonym_type=gradable | 1168 (584) | 0.09 [0.07, 0.10] | 0.00 [0.00, 0.00] | 0.36 [0.33, 0.39] | 0.02 [0.01, 0.03] | 15 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | degree_adv=False | 490 (245) | 0.13 [0.10, 0.17] | 0.00 [0.00, 0.01] | 0.48 [0.44, 0.53] | 0.02 [0.01, 0.03] | 10 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | degree_adv=True | 688 (344) | 0.05 [0.03, 0.07] | 0.00 [0.00, 0.00] | 0.27 [0.24, 0.31] | 0.02 [0.01, 0.03] | 5 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | names=False | 1116 (558) | 0.07 [0.05, 0.09] | 0.00 [0.00, 0.00] | 0.35 [0.32, 0.38] | 0.02 [0.01, 0.03] | 9 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | names=True | 62 (31) | 0.37 [0.26, 0.52] | 0.00 [0.00, 0.00] | 0.55 [0.40, 0.71] | 0.02 [0.00, 0.05] | 6 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | order=1 | 589 (589) | 0.08 [0.06, 0.11] | 0.00 [0.00, 0.00] | 0.35 [0.31, 0.39] | 0.02 [0.01, 0.03] | 48 / 0 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E2 e→a | order=2 | 589 (589) | 0.09 [0.07, 0.11] | 0.00 [0.00, 0.01] | 0.37 [0.33, 0.40] | 0.02 [0.01, 0.03] | 53 / 1 | copy trap (word stated with the other entity) |
| Qwen3-1.7B | E3 e→a | all | 648 (648) | 0.73 [0.70, 0.76] | 0.00 [0.00, 0.00] | 0.90 [0.87, 0.92] | 0.00 [0.00, 0.01] | 474 / 0 | other answer |
| Qwen3-1.7B | E3 e→a | antonym_type=gradable | 643 (643) | 0.74 [0.70, 0.77] | 0.00 [0.00, 0.00] | 0.90 [0.88, 0.93] | 0.00 [0.00, 0.01] | 473 / 0 | other answer |
| Qwen3-1.7B | E3 e→a | degree_adv=False | 286 (286) | 0.75 [0.70, 0.80] | 0.00 [0.00, 0.00] | 0.90 [0.86, 0.93] | 0.01 [0.00, 0.02] | 214 / 0 | other answer |
| Qwen3-1.7B | E3 e→a | degree_adv=True | 362 (362) | 0.72 [0.67, 0.76] | 0.00 [0.00, 0.00] | 0.90 [0.87, 0.93] | 0.00 [0.00, 0.00] | 260 / 0 | other answer |
| Qwen3-1.7B | E3 e→a | names=False | 617 (617) | 0.74 [0.71, 0.78] | 0.00 [0.00, 0.00] | 0.91 [0.89, 0.93] | 0.00 [0.00, 0.00] | 459 / 0 | other answer |
| Qwen3-1.7B | E3 e→a | names=True | 31 (31) | 0.48 [0.32, 0.68] | 0.00 [0.00, 0.00] | 0.68 [0.52, 0.84] | 0.03 [0.00, 0.10] | 15 / 0 | other answer |
| Qwen3-1.7B | E3 e→a | order=1 | 418 (418) | 0.75 [0.71, 0.79] | 0.00 [0.00, 0.00] | 0.89 [0.86, 0.92] | 0.00 [0.00, 0.01] | 312 / 0 | other answer |
| Qwen3-1.7B | E3 e→a | order=2 | 230 (230) | 0.70 [0.64, 0.77] | 0.00 [0.00, 0.00] | 0.92 [0.88, 0.95] | 0.00 [0.00, 0.00] | 162 / 0 | other answer |
| Qwen3-1.7B | Set C agree (a→e) | all | 1286 (643) | 0.02 [0.01, 0.03] | 0.00 [0.00, 0.00] | 0.11 [0.09, 0.13] | 0.02 [0.01, 0.02] | 4 / 0 | event / other |
| Qwen3-1.7B | Set C agree (a→e) | antonym_type=complementary | 10 (5) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.20 [0.00, 0.60] | 0.00 [0.00, 0.00] | 0 / 0 | event / other |
| Qwen3-1.7B | Set C agree (a→e) | antonym_type=gradable | 1276 (638) | 0.02 [0.01, 0.03] | 0.00 [0.00, 0.00] | 0.11 [0.09, 0.12] | 0.02 [0.01, 0.02] | 4 / 0 | event / other |
| Qwen3-1.7B | Set C agree (a→e) | degree_adv=False | 570 (285) | 0.02 [0.01, 0.04] | 0.00 [0.00, 0.00] | 0.12 [0.09, 0.15] | 0.02 [0.01, 0.03] | 3 / 0 | event / other |
| Qwen3-1.7B | Set C agree (a→e) | degree_adv=True | 716 (358) | 0.01 [0.00, 0.02] | 0.00 [0.00, 0.00] | 0.09 [0.07, 0.12] | 0.02 [0.01, 0.03] | 1 / 0 | event / other |
| Qwen3-1.7B | Set C agree (a→e) | names=False | 1224 (612) | 0.01 [0.01, 0.02] | 0.00 [0.00, 0.00] | 0.10 [0.08, 0.12] | 0.02 [0.01, 0.02] | 3 / 0 | event / other |
| Qwen3-1.7B | Set C agree (a→e) | names=True | 62 (31) | 0.08 [0.02, 0.16] | 0.00 [0.00, 0.00] | 0.23 [0.11, 0.35] | 0.00 [0.00, 0.00] | 1 / 0 | event / other |
| Qwen3-1.7B | Set C agree (a→e) | order=1 | 643 (643) | 0.02 [0.01, 0.03] | 0.00 [0.00, 0.00] | 0.12 [0.10, 0.14] | 0.02 [0.01, 0.04] | 13 / 0 | event / other |
| Qwen3-1.7B | Set C agree (a→e) | order=2 | 643 (643) | 0.02 [0.01, 0.03] | 0.00 [0.00, 0.00] | 0.09 [0.07, 0.12] | 0.01 [0.00, 0.02] | 10 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_event_a2e | all | 1286 (643) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | 0.03 [0.02, 0.04] | 0.06 [0.05, 0.08] | 0 / 1 | event / other |
| Qwen3-1.7B | sc_conflict_event_a2e | antonym_type=complementary | 10 (5) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.10 [0.00, 0.30] | 0.00 [0.00, 0.00] | 0 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_event_a2e | antonym_type=gradable | 1276 (638) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | 0.03 [0.02, 0.04] | 0.07 [0.05, 0.08] | 0 / 1 | event / other |
| Qwen3-1.7B | sc_conflict_event_a2e | degree_adv=False | 570 (285) | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.06 [0.04, 0.08] | 0.05 [0.03, 0.07] | 0 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_event_a2e | degree_adv=True | 716 (358) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | 0.01 [0.00, 0.02] | 0.08 [0.05, 0.10] | 0 / 1 | event / other |
| Qwen3-1.7B | sc_conflict_event_a2e | names=False | 1224 (612) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.04] | 0.05 [0.04, 0.07] | 0 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_event_a2e | names=True | 62 (31) | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.15] | 0.03 [0.00, 0.08] | 0.27 [0.15, 0.40] | 0 / 1 | event / other |
| Qwen3-1.7B | sc_conflict_event_a2e | order=1 | 643 (643) | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.01] | 0.03 [0.02, 0.05] | 0.07 [0.05, 0.09] | 0 / 4 | event / other |
| Qwen3-1.7B | sc_conflict_event_a2e | order=2 | 643 (643) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 [0.02, 0.05] | 0.06 [0.04, 0.08] | 1 / 1 | event / other |
| Qwen3-1.7B | sc_conflict_stmt_a2e | all | 1286 (643) | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.06 [0.05, 0.08] | 0.03 [0.02, 0.04] | 1 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_stmt_a2e | antonym_type=complementary | 10 (5) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.10 [0.00, 0.30] | 0 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_stmt_a2e | antonym_type=gradable | 1276 (638) | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.07 [0.05, 0.08] | 0.03 [0.02, 0.04] | 1 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_stmt_a2e | degree_adv=False | 570 (285) | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.05 [0.03, 0.07] | 0.06 [0.04, 0.08] | 0 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_stmt_a2e | degree_adv=True | 716 (358) | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.08 [0.05, 0.10] | 0.01 [0.00, 0.02] | 1 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_stmt_a2e | names=False | 1224 (612) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.05 [0.04, 0.07] | 0.03 [0.02, 0.04] | 0 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_stmt_a2e | names=True | 62 (31) | 0.06 [0.00, 0.15] | 0.00 [0.00, 0.00] | 0.27 [0.15, 0.40] | 0.03 [0.00, 0.08] | 1 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_stmt_a2e | order=1 | 643 (643) | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.07 [0.05, 0.09] | 0.03 [0.02, 0.05] | 4 / 0 | event / other |
| Qwen3-1.7B | sc_conflict_stmt_a2e | order=2 | 643 (643) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.06 [0.04, 0.08] | 0.03 [0.02, 0.05] | 1 / 1 | event / other |
| Qwen3-4B | E1 a→e | all | 1178 (589) | 0.95 [0.94, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 533 / 0 | other answer |
| Qwen3-4B | E1 a→e | antonym_type=complementary | 10 (5) | 0.90 [0.70, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 4 / 0 | other answer |
| Qwen3-4B | E1 a→e | antonym_type=gradable | 1168 (584) | 0.95 [0.94, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 529 / 0 | other answer |
| Qwen3-4B | E1 a→e | degree_adv=False | 490 (245) | 0.93 [0.91, 0.96] | 0.00 [0.00, 0.00] | 0.98 [0.96, 0.99] | 0.00 [0.00, 0.00] | 215 / 0 | other answer |
| Qwen3-4B | E1 a→e | degree_adv=True | 688 (344) | 0.96 [0.95, 0.98] | 0.00 [0.00, 0.00] | 0.99 [0.99, 1.00] | 0.00 [0.00, 0.00] | 318 / 0 | other answer |
| Qwen3-4B | E1 a→e | names=False | 1116 (558) | 0.95 [0.94, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 504 / 0 | other answer |
| Qwen3-4B | E1 a→e | names=True | 62 (31) | 0.97 [0.92, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 29 / 0 | other answer |
| Qwen3-4B | E1 a→e | order=1 | 589 (589) | 0.96 [0.94, 0.97] | 0.00 [0.00, 0.00] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 565 / 0 | other answer |
| Qwen3-4B | E1 a→e | order=2 | 589 (589) | 0.94 [0.92, 0.96] | 0.00 [0.00, 0.00] | 0.98 [0.97, 0.99] | 0.00 [0.00, 0.00] | 553 / 0 | other answer |
| Qwen3-4B | E1p a→e | all | 970 (485) | 0.75 [0.72, 0.78] | 0.00 [0.00, 0.00] | 0.89 [0.87, 0.91] | 0.00 [0.00, 0.00] | 295 / 0 | other answer |
| Qwen3-4B | E1p a→e | antonym_type=gradable | 962 (481) | 0.75 [0.72, 0.78] | 0.00 [0.00, 0.00] | 0.89 [0.87, 0.91] | 0.00 [0.00, 0.00] | 292 / 0 | other answer |
| Qwen3-4B | E1p a→e | degree_adv=False | 392 (196) | 0.80 [0.76, 0.84] | 0.00 [0.00, 0.00] | 0.93 [0.90, 0.95] | 0.00 [0.00, 0.00] | 130 / 0 | other answer |
| Qwen3-4B | E1p a→e | degree_adv=True | 578 (289) | 0.71 [0.67, 0.76] | 0.00 [0.00, 0.00] | 0.86 [0.84, 0.89] | 0.00 [0.00, 0.00] | 165 / 0 | other answer |
| Qwen3-4B | E1p a→e | names=False | 918 (459) | 0.74 [0.71, 0.77] | 0.00 [0.00, 0.00] | 0.89 [0.86, 0.91] | 0.00 [0.00, 0.00] | 274 / 0 | other answer |
| Qwen3-4B | E1p a→e | names=True | 52 (26) | 0.90 [0.83, 0.98] | 0.00 [0.00, 0.00] | 0.96 [0.90, 1.00] | 0.00 [0.00, 0.00] | 21 / 0 | other answer |
| Qwen3-4B | E1p a→e | order=1 | 485 (485) | 0.76 [0.72, 0.80] | 0.00 [0.00, 0.00] | 0.90 [0.87, 0.93] | 0.00 [0.00, 0.00] | 368 / 0 | other answer |
| Qwen3-4B | E1p a→e | order=2 | 485 (485) | 0.74 [0.70, 0.78] | 0.00 [0.00, 0.00] | 0.88 [0.85, 0.90] | 0.00 [0.00, 0.00] | 359 / 0 | other answer |
| Qwen3-4B | E2 a→e | all | 3578 (1789) | 0.00 [0.00, 0.00] | 0.60 [0.58, 0.62] | 0.00 [0.00, 0.00] | 0.84 [0.83, 0.85] | 0 / 785 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.25 [0.12, 0.40] | 0.00 [0.00, 0.00] | 0.54 [0.40, 0.69] | 0 / 2 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | antonym_type=gradable | 3530 (1765) | 0.00 [0.00, 0.00] | 0.61 [0.59, 0.63] | 0.00 [0.00, 0.00] | 0.85 [0.83, 0.86] | 0 / 783 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | degree_adv=False | 1986 (993) | 0.00 [0.00, 0.00] | 0.50 [0.48, 0.53] | 0.00 [0.00, 0.00] | 0.79 [0.77, 0.81] | 0 / 307 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | degree_adv=True | 1592 (796) | 0.00 [0.00, 0.00] | 0.73 [0.71, 0.76] | 0.00 [0.00, 0.00] | 0.91 [0.89, 0.92] | 0 / 478 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | names=False | 3060 (1530) | 0.00 [0.00, 0.00] | 0.61 [0.59, 0.63] | 0.00 [0.00, 0.00] | 0.84 [0.83, 0.86] | 0 / 685 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | names=True | 518 (259) | 0.00 [0.00, 0.01] | 0.58 [0.54, 0.63] | 0.00 [0.00, 0.01] | 0.84 [0.81, 0.87] | 0 / 100 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | order=1 | 1789 (1789) | 0.00 [0.00, 0.00] | 0.61 [0.58, 0.63] | 0.00 [0.00, 0.00] | 0.84 [0.82, 0.85] | 0 / 1085 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 a→e | order=2 | 1789 (1789) | 0.00 [0.00, 0.00] | 0.60 [0.58, 0.62] | 0.00 [0.00, 0.00] | 0.85 [0.83, 0.86] | 1 / 1076 | copy trap (word stated with the other entity) |
| Qwen3-4B | E3 a→e (P3 frame) | all | 1824 (1824) | 0.19 [0.18, 0.21] | 0.00 [0.00, 0.01] | 0.39 [0.36, 0.41] | 0.02 [0.01, 0.03] | 355 / 7 | other answer |
| Qwen3-4B | E3 a→e (P3 frame) | antonym_type=complementary | 24 (24) | 0.04 [0.00, 0.12] | 0.04 [0.00, 0.12] | 0.29 [0.12, 0.46] | 0.04 [0.00, 0.12] | 1 / 1 | other answer |
| Qwen3-4B | E3 a→e (P3 frame) | antonym_type=gradable | 1800 (1800) | 0.20 [0.18, 0.21] | 0.00 [0.00, 0.01] | 0.39 [0.36, 0.41] | 0.02 [0.01, 0.03] | 354 / 6 | other answer |
| Qwen3-4B | E3 a→e (P3 frame) | degree_adv=False | 1035 (1035) | 0.17 [0.15, 0.20] | 0.00 [0.00, 0.01] | 0.38 [0.35, 0.41] | 0.02 [0.01, 0.03] | 179 / 5 | other answer |
| Qwen3-4B | E3 a→e (P3 frame) | degree_adv=True | 789 (789) | 0.22 [0.19, 0.25] | 0.00 [0.00, 0.01] | 0.39 [0.35, 0.42] | 0.02 [0.01, 0.03] | 176 / 2 | other answer |
| Qwen3-4B | E3 a→e (P3 frame) | names=False | 1563 (1563) | 0.18 [0.16, 0.19] | 0.00 [0.00, 0.01] | 0.36 [0.34, 0.38] | 0.02 [0.01, 0.03] | 275 / 5 | other answer |
| Qwen3-4B | E3 a→e (P3 frame) | names=True | 261 (261) | 0.31 [0.25, 0.36] | 0.01 [0.00, 0.02] | 0.55 [0.49, 0.61] | 0.03 [0.01, 0.05] | 80 / 2 | other answer |
| Qwen3-4B | E3 a→e (P3 frame) | order=1 | 1323 (1323) | 0.20 [0.18, 0.23] | 0.00 [0.00, 0.01] | 0.40 [0.38, 0.43] | 0.02 [0.01, 0.03] | 270 / 6 | other answer |
| Qwen3-4B | E3 a→e (P3 frame) | order=2 | 501 (501) | 0.17 [0.14, 0.20] | 0.00 [0.00, 0.01] | 0.34 [0.30, 0.38] | 0.02 [0.01, 0.03] | 85 / 1 | other answer |
| Qwen3-4B | E3 a→e (F1 frame, Phase-5 anchor) | all | 1843 (1843) | 0.23 [0.21, 0.25] | 0.00 [0.00, 0.00] | 0.40 [0.38, 0.43] | 0.01 [0.01, 0.02] | 426 / 3 | other answer |
| Qwen3-4B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=complementary | 24 (24) | 0.17 [0.04, 0.33] | 0.00 [0.00, 0.00] | 0.38 [0.17, 0.58] | 0.04 [0.00, 0.12] | 4 / 0 | other answer |
| Qwen3-4B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=gradable | 1819 (1819) | 0.23 [0.21, 0.25] | 0.00 [0.00, 0.00] | 0.40 [0.38, 0.43] | 0.01 [0.01, 0.02] | 422 / 3 | other answer |
| Qwen3-4B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=False | 1038 (1038) | 0.25 [0.23, 0.28] | 0.00 [0.00, 0.00] | 0.44 [0.41, 0.47] | 0.01 [0.01, 0.02] | 262 / 2 | other answer |
| Qwen3-4B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=True | 805 (805) | 0.20 [0.18, 0.23] | 0.00 [0.00, 0.00] | 0.36 [0.33, 0.39] | 0.01 [0.00, 0.02] | 164 / 1 | other answer |
| Qwen3-4B | E3 a→e (F1 frame, Phase-5 anchor) | names=False | 1587 (1587) | 0.24 [0.22, 0.26] | 0.00 [0.00, 0.00] | 0.40 [0.38, 0.43] | 0.01 [0.00, 0.01] | 385 / 3 | other answer |
| Qwen3-4B | E3 a→e (F1 frame, Phase-5 anchor) | names=True | 256 (256) | 0.16 [0.12, 0.20] | 0.00 [0.00, 0.00] | 0.41 [0.34, 0.46] | 0.02 [0.01, 0.04] | 41 / 0 | other answer |
| Qwen3-4B | E3 a→e (F1 frame, Phase-5 anchor) | order=1 | 1324 (1324) | 0.23 [0.21, 0.26] | 0.00 [0.00, 0.00] | 0.42 [0.39, 0.44] | 0.01 [0.01, 0.02] | 309 / 2 | other answer |
| Qwen3-4B | E3 a→e (F1 frame, Phase-5 anchor) | order=2 | 519 (519) | 0.23 [0.19, 0.26] | 0.00 [0.00, 0.01] | 0.38 [0.33, 0.41] | 0.01 [0.00, 0.02] | 117 / 1 | other answer |
| Qwen3-4B | E1 e→a | all | 3578 (1789) | 0.33 [0.31, 0.34] | 0.00 [0.00, 0.00] | 0.61 [0.59, 0.63] | 0.01 [0.01, 0.01] | 286 / 0 | other answer |
| Qwen3-4B | E1 e→a | antonym_type=complementary | 50 (25) | 0.34 [0.20, 0.48] | 0.00 [0.00, 0.00] | 0.52 [0.38, 0.66] | 0.00 [0.00, 0.00] | 4 / 0 | other answer |
| Qwen3-4B | E1 e→a | antonym_type=gradable | 3528 (1764) | 0.33 [0.31, 0.34] | 0.00 [0.00, 0.00] | 0.61 [0.60, 0.63] | 0.01 [0.01, 0.01] | 282 / 0 | other answer |
| Qwen3-4B | E1 e→a | degree_adv=False | 1986 (993) | 0.30 [0.28, 0.33] | 0.00 [0.00, 0.01] | 0.55 [0.53, 0.58] | 0.01 [0.01, 0.02] | 160 / 0 | other answer |
| Qwen3-4B | E1 e→a | degree_adv=True | 1592 (796) | 0.36 [0.33, 0.38] | 0.00 [0.00, 0.00] | 0.69 [0.66, 0.71] | 0.00 [0.00, 0.01] | 126 / 0 | other answer |
| Qwen3-4B | E1 e→a | names=False | 3060 (1530) | 0.33 [0.32, 0.35] | 0.00 [0.00, 0.00] | 0.63 [0.61, 0.65] | 0.01 [0.01, 0.01] | 258 / 0 | other answer |
| Qwen3-4B | E1 e→a | names=True | 518 (259) | 0.29 [0.25, 0.33] | 0.01 [0.00, 0.02] | 0.50 [0.46, 0.55] | 0.02 [0.01, 0.03] | 28 / 0 | other answer |
| Qwen3-4B | E1 e→a | order=1 | 1789 (1789) | 0.33 [0.31, 0.35] | 0.00 [0.00, 0.00] | 0.60 [0.58, 0.62] | 0.01 [0.01, 0.02] | 592 / 4 | other answer |
| Qwen3-4B | E1 e→a | order=2 | 1789 (1789) | 0.32 [0.30, 0.34] | 0.00 [0.00, 0.00] | 0.62 [0.60, 0.64] | 0.01 [0.00, 0.01] | 574 / 4 | other answer |
| Qwen3-4B | E1p e→a | all | 3370 (1685) | 0.25 [0.23, 0.27] | 0.00 [0.00, 0.00] | 0.51 [0.48, 0.53] | 0.01 [0.00, 0.01] | 258 / 0 | other answer |
| Qwen3-4B | E1p e→a | antonym_type=complementary | 32 (16) | 0.25 [0.12, 0.41] | 0.00 [0.00, 0.00] | 0.62 [0.44, 0.81] | 0.00 [0.00, 0.00] | 1 / 0 | other answer |
| Qwen3-4B | E1p e→a | antonym_type=gradable | 3338 (1669) | 0.25 [0.23, 0.27] | 0.00 [0.00, 0.00] | 0.50 [0.48, 0.52] | 0.01 [0.00, 0.01] | 257 / 0 | other answer |
| Qwen3-4B | E1p e→a | degree_adv=False | 1790 (895) | 0.29 [0.26, 0.31] | 0.00 [0.00, 0.00] | 0.56 [0.53, 0.58] | 0.00 [0.00, 0.01] | 161 / 0 | other answer |
| Qwen3-4B | E1p e→a | degree_adv=True | 1580 (790) | 0.21 [0.18, 0.23] | 0.00 [0.00, 0.00] | 0.45 [0.42, 0.48] | 0.01 [0.00, 0.01] | 97 / 0 | other answer |
| Qwen3-4B | E1p e→a | names=False | 2904 (1452) | 0.24 [0.22, 0.26] | 0.00 [0.00, 0.00] | 0.50 [0.47, 0.52] | 0.01 [0.00, 0.01] | 224 / 0 | other answer |
| Qwen3-4B | E1p e→a | names=True | 466 (233) | 0.30 [0.26, 0.34] | 0.00 [0.00, 0.00] | 0.56 [0.50, 0.61] | 0.00 [0.00, 0.00] | 34 / 0 | other answer |
| Qwen3-4B | E1p e→a | order=1 | 1685 (1685) | 0.26 [0.24, 0.28] | 0.00 [0.00, 0.00] | 0.52 [0.49, 0.54] | 0.00 [0.00, 0.01] | 438 / 0 | other answer |
| Qwen3-4B | E1p e→a | order=2 | 1685 (1685) | 0.24 [0.22, 0.26] | 0.00 [0.00, 0.00] | 0.49 [0.47, 0.52] | 0.01 [0.00, 0.01] | 404 / 0 | other answer |
| Qwen3-4B | E2 e→a | all | 3578 (1789) | 0.01 [0.01, 0.02] | 0.02 [0.02, 0.03] | 0.12 [0.10, 0.13] | 0.11 [0.09, 0.12] | 4 / 12 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | antonym_type=complementary | 50 (25) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.06 [0.00, 0.12] | 0.12 [0.04, 0.22] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | antonym_type=gradable | 3528 (1764) | 0.01 [0.01, 0.02] | 0.02 [0.02, 0.03] | 0.12 [0.10, 0.13] | 0.11 [0.09, 0.12] | 4 / 12 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | degree_adv=False | 1986 (993) | 0.02 [0.01, 0.02] | 0.02 [0.01, 0.02] | 0.12 [0.11, 0.14] | 0.10 [0.08, 0.11] | 2 / 3 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | degree_adv=True | 1592 (796) | 0.01 [0.01, 0.02] | 0.03 [0.02, 0.04] | 0.10 [0.09, 0.12] | 0.12 [0.10, 0.13] | 2 / 9 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | names=False | 3060 (1530) | 0.01 [0.01, 0.01] | 0.02 [0.02, 0.03] | 0.11 [0.10, 0.12] | 0.10 [0.09, 0.11] | 4 / 10 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | names=True | 518 (259) | 0.03 [0.02, 0.05] | 0.03 [0.02, 0.05] | 0.15 [0.12, 0.18] | 0.13 [0.09, 0.16] | 0 / 2 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | order=1 | 1789 (1789) | 0.02 [0.01, 0.02] | 0.02 [0.02, 0.03] | 0.12 [0.10, 0.14] | 0.11 [0.10, 0.12] | 27 / 43 | copy trap (word stated with the other entity) |
| Qwen3-4B | E2 e→a | order=2 | 1789 (1789) | 0.01 [0.01, 0.02] | 0.02 [0.01, 0.03] | 0.11 [0.10, 0.13] | 0.10 [0.09, 0.12] | 23 / 37 | copy trap (word stated with the other entity) |
| Qwen3-4B | E3 e→a | all | 1848 (1848) | 0.47 [0.45, 0.49] | 0.00 [0.00, 0.01] | 0.63 [0.61, 0.65] | 0.01 [0.00, 0.01] | 869 / 6 | other answer |
| Qwen3-4B | E3 e→a | antonym_type=complementary | 24 (24) | 0.38 [0.21, 0.58] | 0.00 [0.00, 0.00] | 0.67 [0.46, 0.83] | 0.04 [0.00, 0.12] | 9 / 0 | other answer |
| Qwen3-4B | E3 e→a | antonym_type=gradable | 1824 (1824) | 0.47 [0.45, 0.49] | 0.00 [0.00, 0.01] | 0.63 [0.61, 0.65] | 0.01 [0.00, 0.01] | 860 / 6 | other answer |
| Qwen3-4B | E3 e→a | degree_adv=False | 1037 (1037) | 0.42 [0.39, 0.45] | 0.00 [0.00, 0.01] | 0.58 [0.55, 0.61] | 0.01 [0.00, 0.01] | 435 / 3 | other answer |
| Qwen3-4B | E3 e→a | degree_adv=True | 811 (811) | 0.54 [0.50, 0.57] | 0.00 [0.00, 0.01] | 0.69 [0.65, 0.72] | 0.00 [0.00, 0.01] | 434 / 3 | other answer |
| Qwen3-4B | E3 e→a | names=False | 1590 (1590) | 0.50 [0.47, 0.52] | 0.00 [0.00, 0.01] | 0.65 [0.63, 0.67] | 0.01 [0.00, 0.01] | 791 / 5 | other answer |
| Qwen3-4B | E3 e→a | names=True | 258 (258) | 0.30 [0.25, 0.36] | 0.00 [0.00, 0.01] | 0.50 [0.44, 0.56] | 0.01 [0.00, 0.02] | 78 / 1 | other answer |
| Qwen3-4B | E3 e→a | order=1 | 1327 (1327) | 0.46 [0.43, 0.48] | 0.00 [0.00, 0.01] | 0.61 [0.58, 0.63] | 0.01 [0.00, 0.01] | 604 / 5 | other answer |
| Qwen3-4B | E3 e→a | order=2 | 521 (521) | 0.51 [0.47, 0.55] | 0.00 [0.00, 0.01] | 0.68 [0.64, 0.72] | 0.00 [0.00, 0.01] | 265 / 1 | other answer |
| Qwen3-4B | Set C agree (a→e) | all | 3686 (1843) | 0.57 [0.55, 0.59] | 0.00 [0.00, 0.00] | 0.78 [0.76, 0.79] | 0.00 [0.00, 0.00] | 784 / 0 | event / other |
| Qwen3-4B | Set C agree (a→e) | antonym_type=complementary | 48 (24) | 0.48 [0.31, 0.65] | 0.00 [0.00, 0.00] | 0.69 [0.50, 0.85] | 0.00 [0.00, 0.00] | 8 / 0 | event / other |
| Qwen3-4B | Set C agree (a→e) | antonym_type=gradable | 3638 (1819) | 0.57 [0.56, 0.59] | 0.00 [0.00, 0.00] | 0.78 [0.76, 0.80] | 0.00 [0.00, 0.00] | 776 / 0 | event / other |
| Qwen3-4B | Set C agree (a→e) | degree_adv=False | 2074 (1037) | 0.51 [0.49, 0.54] | 0.00 [0.00, 0.00] | 0.73 [0.71, 0.75] | 0.00 [0.00, 0.00] | 370 / 0 | event / other |
| Qwen3-4B | Set C agree (a→e) | degree_adv=True | 1612 (806) | 0.65 [0.63, 0.68] | 0.00 [0.00, 0.00] | 0.84 [0.82, 0.86] | 0.00 [0.00, 0.00] | 414 / 0 | event / other |
| Qwen3-4B | Set C agree (a→e) | names=False | 3174 (1587) | 0.58 [0.56, 0.60] | 0.00 [0.00, 0.00] | 0.77 [0.76, 0.79] | 0.00 [0.00, 0.00] | 682 / 0 | event / other |
| Qwen3-4B | Set C agree (a→e) | names=True | 512 (256) | 0.54 [0.49, 0.59] | 0.00 [0.00, 0.00] | 0.80 [0.77, 0.84] | 0.00 [0.00, 0.00] | 102 / 0 | event / other |
| Qwen3-4B | Set C agree (a→e) | order=1 | 1843 (1843) | 0.60 [0.58, 0.62] | 0.00 [0.00, 0.00] | 0.80 [0.78, 0.82] | 0.00 [0.00, 0.00] | 1111 / 0 | event / other |
| Qwen3-4B | Set C agree (a→e) | order=2 | 1843 (1843) | 0.54 [0.52, 0.56] | 0.00 [0.00, 0.00] | 0.76 [0.74, 0.78] | 0.00 [0.00, 0.00] | 1002 / 0 | event / other |
| Qwen3-4B | sc_conflict_event_a2e | all | 3686 (1843) | 0.00 [0.00, 0.00] | 0.32 [0.30, 0.34] | 0.01 [0.01, 0.02] | 0.56 [0.54, 0.57] | 1 / 350 | event / other |
| Qwen3-4B | sc_conflict_event_a2e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.15 [0.02, 0.29] | 0.02 [0.00, 0.06] | 0.42 [0.27, 0.58] | 0 / 3 | event / other |
| Qwen3-4B | sc_conflict_event_a2e | antonym_type=gradable | 3638 (1819) | 0.00 [0.00, 0.00] | 0.32 [0.30, 0.34] | 0.01 [0.01, 0.02] | 0.56 [0.54, 0.58] | 1 / 347 | event / other |
| Qwen3-4B | sc_conflict_event_a2e | degree_adv=False | 2074 (1037) | 0.00 [0.00, 0.00] | 0.31 [0.28, 0.33] | 0.01 [0.01, 0.02] | 0.54 [0.51, 0.56] | 0 / 181 | event / other |
| Qwen3-4B | sc_conflict_event_a2e | degree_adv=True | 1612 (806) | 0.00 [0.00, 0.01] | 0.33 [0.31, 0.36] | 0.01 [0.01, 0.02] | 0.58 [0.55, 0.60] | 1 / 169 | event / other |
| Qwen3-4B | sc_conflict_event_a2e | names=False | 3174 (1587) | 0.00 [0.00, 0.00] | 0.30 [0.28, 0.32] | 0.02 [0.01, 0.02] | 0.53 [0.51, 0.55] | 1 / 281 | event / other |
| Qwen3-4B | sc_conflict_event_a2e | names=True | 512 (256) | 0.00 [0.00, 0.00] | 0.42 [0.36, 0.47] | 0.00 [0.00, 0.00] | 0.71 [0.66, 0.75] | 0 / 69 | event / other |
| Qwen3-4B | sc_conflict_event_a2e | order=1 | 1843 (1843) | 0.00 [0.00, 0.01] | 0.36 [0.33, 0.38] | 0.01 [0.01, 0.02] | 0.58 [0.56, 0.60] | 6 / 656 | event / other |
| Qwen3-4B | sc_conflict_event_a2e | order=2 | 1843 (1843) | 0.00 [0.00, 0.00] | 0.28 [0.26, 0.30] | 0.02 [0.01, 0.02] | 0.53 [0.50, 0.55] | 2 / 521 | event / other |
| Qwen3-4B | sc_conflict_stmt_a2e | all | 3686 (1843) | 0.32 [0.30, 0.34] | 0.00 [0.00, 0.00] | 0.56 [0.54, 0.57] | 0.01 [0.01, 0.02] | 350 / 1 | event / other |
| Qwen3-4B | sc_conflict_stmt_a2e | antonym_type=complementary | 48 (24) | 0.15 [0.02, 0.29] | 0.00 [0.00, 0.00] | 0.42 [0.27, 0.58] | 0.02 [0.00, 0.06] | 3 / 0 | event / other |
| Qwen3-4B | sc_conflict_stmt_a2e | antonym_type=gradable | 3638 (1819) | 0.32 [0.30, 0.34] | 0.00 [0.00, 0.00] | 0.56 [0.54, 0.58] | 0.01 [0.01, 0.02] | 347 / 1 | event / other |
| Qwen3-4B | sc_conflict_stmt_a2e | degree_adv=False | 2074 (1037) | 0.31 [0.28, 0.33] | 0.00 [0.00, 0.00] | 0.54 [0.51, 0.56] | 0.01 [0.01, 0.02] | 181 / 0 | event / other |
| Qwen3-4B | sc_conflict_stmt_a2e | degree_adv=True | 1612 (806) | 0.33 [0.31, 0.36] | 0.00 [0.00, 0.01] | 0.58 [0.55, 0.60] | 0.01 [0.01, 0.02] | 169 / 1 | event / other |
| Qwen3-4B | sc_conflict_stmt_a2e | names=False | 3174 (1587) | 0.30 [0.28, 0.32] | 0.00 [0.00, 0.00] | 0.53 [0.51, 0.55] | 0.02 [0.01, 0.02] | 281 / 1 | event / other |
| Qwen3-4B | sc_conflict_stmt_a2e | names=True | 512 (256) | 0.42 [0.36, 0.47] | 0.00 [0.00, 0.00] | 0.71 [0.66, 0.75] | 0.00 [0.00, 0.00] | 69 / 0 | event / other |
| Qwen3-4B | sc_conflict_stmt_a2e | order=1 | 1843 (1843) | 0.36 [0.33, 0.38] | 0.00 [0.00, 0.01] | 0.58 [0.56, 0.60] | 0.01 [0.01, 0.02] | 656 / 6 | event / other |
| Qwen3-4B | sc_conflict_stmt_a2e | order=2 | 1843 (1843) | 0.28 [0.26, 0.30] | 0.00 [0.00, 0.00] | 0.53 [0.50, 0.55] | 0.02 [0.01, 0.02] | 521 / 2 | event / other |
| Qwen3-8B | E1 a→e | all | 1178 (589) | 0.95 [0.94, 0.97] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 542 / 0 | other answer |
| Qwen3-8B | E1 a→e | antonym_type=complementary | 10 (5) | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 5 / 0 | other answer |
| Qwen3-8B | E1 a→e | antonym_type=gradable | 1168 (584) | 0.95 [0.94, 0.97] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 537 / 0 | other answer |
| Qwen3-8B | E1 a→e | degree_adv=False | 490 (245) | 0.92 [0.89, 0.95] | 0.00 [0.00, 0.00] | 0.98 [0.97, 0.99] | 0.00 [0.00, 0.00] | 213 / 0 | other answer |
| Qwen3-8B | E1 a→e | degree_adv=True | 688 (344) | 0.98 [0.96, 0.99] | 0.00 [0.00, 0.00] | 0.99 [0.99, 1.00] | 0.00 [0.00, 0.00] | 329 / 0 | other answer |
| Qwen3-8B | E1 a→e | names=False | 1116 (558) | 0.96 [0.94, 0.97] | 0.00 [0.00, 0.00] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 516 / 0 | other answer |
| Qwen3-8B | E1 a→e | names=True | 62 (31) | 0.90 [0.81, 0.97] | 0.00 [0.00, 0.00] | 0.97 [0.92, 1.00] | 0.00 [0.00, 0.00] | 26 / 0 | other answer |
| Qwen3-8B | E1 a→e | order=1 | 589 (589) | 0.96 [0.94, 0.98] | 0.00 [0.00, 0.00] | 0.99 [0.98, 1.00] | 0.00 [0.00, 0.00] | 566 / 0 | other answer |
| Qwen3-8B | E1 a→e | order=2 | 589 (589) | 0.95 [0.93, 0.96] | 0.00 [0.00, 0.00] | 0.99 [0.98, 0.99] | 0.00 [0.00, 0.00] | 557 / 0 | other answer |
| Qwen3-8B | E1p a→e | all | 3370 (1685) | 0.78 [0.77, 0.80] | 0.00 [0.00, 0.00] | 0.91 [0.90, 0.92] | 0.00 [0.00, 0.00] | 1113 / 0 | other answer |
| Qwen3-8B | E1p a→e | antonym_type=complementary | 32 (16) | 0.66 [0.47, 0.81] | 0.00 [0.00, 0.00] | 0.91 [0.81, 1.00] | 0.00 [0.00, 0.00] | 8 / 0 | other answer |
| Qwen3-8B | E1p a→e | antonym_type=gradable | 3338 (1669) | 0.78 [0.77, 0.80] | 0.00 [0.00, 0.00] | 0.91 [0.90, 0.92] | 0.00 [0.00, 0.00] | 1105 / 0 | other answer |
| Qwen3-8B | E1p a→e | degree_adv=False | 1790 (895) | 0.80 [0.78, 0.82] | 0.00 [0.00, 0.00] | 0.92 [0.91, 0.94] | 0.00 [0.00, 0.00] | 612 / 0 | other answer |
| Qwen3-8B | E1p a→e | degree_adv=True | 1580 (790) | 0.76 [0.74, 0.79] | 0.00 [0.00, 0.00] | 0.89 [0.87, 0.91] | 0.00 [0.00, 0.00] | 501 / 0 | other answer |
| Qwen3-8B | E1p a→e | names=False | 2904 (1452) | 0.77 [0.76, 0.79] | 0.00 [0.00, 0.00] | 0.90 [0.89, 0.91] | 0.00 [0.00, 0.00] | 940 / 0 | other answer |
| Qwen3-8B | E1p a→e | names=True | 466 (233) | 0.86 [0.83, 0.89] | 0.00 [0.00, 0.00] | 0.96 [0.94, 0.97] | 0.00 [0.00, 0.00] | 173 / 0 | other answer |
| Qwen3-8B | E1p a→e | order=1 | 1685 (1685) | 0.79 [0.77, 0.81] | 0.00 [0.00, 0.00] | 0.91 [0.90, 0.93] | 0.00 [0.00, 0.00] | 1329 / 0 | other answer |
| Qwen3-8B | E1p a→e | order=2 | 1685 (1685) | 0.78 [0.76, 0.80] | 0.00 [0.00, 0.00] | 0.91 [0.89, 0.92] | 0.00 [0.00, 0.00] | 1312 / 0 | other answer |
| Qwen3-8B | E2 a→e | all | 3578 (1789) | 0.00 [0.00, 0.00] | 0.41 [0.39, 0.43] | 0.01 [0.01, 0.01] | 0.65 [0.63, 0.67] | 0 / 526 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.08 [0.02, 0.17] | 0.00 [0.00, 0.00] | 0.38 [0.23, 0.54] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | antonym_type=gradable | 3530 (1765) | 0.00 [0.00, 0.00] | 0.41 [0.39, 0.43] | 0.01 [0.01, 0.01] | 0.66 [0.64, 0.67] | 0 / 526 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | degree_adv=False | 1986 (993) | 0.00 [0.00, 0.00] | 0.18 [0.16, 0.20] | 0.01 [0.01, 0.02] | 0.50 [0.47, 0.52] | 0 / 64 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | degree_adv=True | 1592 (796) | 0.00 [0.00, 0.00] | 0.69 [0.66, 0.71] | 0.00 [0.00, 0.01] | 0.85 [0.83, 0.87] | 0 / 462 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | names=False | 3060 (1530) | 0.00 [0.00, 0.00] | 0.44 [0.42, 0.47] | 0.01 [0.01, 0.01] | 0.68 [0.66, 0.70] | 0 / 507 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | names=True | 518 (259) | 0.00 [0.00, 0.00] | 0.19 [0.15, 0.23] | 0.01 [0.00, 0.02] | 0.50 [0.45, 0.55] | 0 / 19 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | order=1 | 1789 (1789) | 0.00 [0.00, 0.00] | 0.41 [0.39, 0.43] | 0.01 [0.00, 0.01] | 0.65 [0.63, 0.67] | 3 / 731 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 a→e | order=2 | 1789 (1789) | 0.00 [0.00, 0.00] | 0.41 [0.38, 0.43] | 0.01 [0.01, 0.02] | 0.66 [0.63, 0.68] | 1 / 726 | copy trap (word stated with the other entity) |
| Qwen3-8B | E3 a→e (P3 frame) | all | 1824 (1824) | 0.32 [0.30, 0.34] | 0.01 [0.00, 0.01] | 0.49 [0.47, 0.52] | 0.02 [0.01, 0.02] | 578 / 10 | other answer |
| Qwen3-8B | E3 a→e (P3 frame) | antonym_type=complementary | 24 (24) | 0.08 [0.00, 0.21] | 0.04 [0.00, 0.12] | 0.25 [0.08, 0.42] | 0.12 [0.00, 0.25] | 2 / 1 | other answer |
| Qwen3-8B | E3 a→e (P3 frame) | antonym_type=gradable | 1800 (1800) | 0.32 [0.30, 0.34] | 0.01 [0.00, 0.01] | 0.50 [0.47, 0.52] | 0.01 [0.01, 0.02] | 576 / 9 | other answer |
| Qwen3-8B | E3 a→e (P3 frame) | degree_adv=False | 1035 (1035) | 0.31 [0.28, 0.34] | 0.01 [0.00, 0.01] | 0.49 [0.46, 0.52] | 0.02 [0.02, 0.03] | 324 / 9 | other answer |
| Qwen3-8B | E3 a→e (P3 frame) | degree_adv=True | 789 (789) | 0.32 [0.29, 0.35] | 0.00 [0.00, 0.00] | 0.50 [0.47, 0.53] | 0.01 [0.00, 0.01] | 254 / 1 | other answer |
| Qwen3-8B | E3 a→e (P3 frame) | names=False | 1563 (1563) | 0.31 [0.29, 0.33] | 0.00 [0.00, 0.01] | 0.46 [0.44, 0.49] | 0.02 [0.01, 0.02] | 481 / 7 | other answer |
| Qwen3-8B | E3 a→e (P3 frame) | names=True | 261 (261) | 0.37 [0.31, 0.43] | 0.01 [0.00, 0.03] | 0.67 [0.61, 0.72] | 0.02 [0.00, 0.03] | 97 / 3 | other answer |
| Qwen3-8B | E3 a→e (P3 frame) | order=1 | 1323 (1323) | 0.33 [0.30, 0.35] | 0.01 [0.00, 0.01] | 0.51 [0.49, 0.54] | 0.02 [0.01, 0.02] | 431 / 8 | other answer |
| Qwen3-8B | E3 a→e (P3 frame) | order=2 | 501 (501) | 0.29 [0.25, 0.33] | 0.00 [0.00, 0.01] | 0.44 [0.39, 0.48] | 0.02 [0.01, 0.03] | 147 / 2 | other answer |
| Qwen3-8B | E3 a→e (F1 frame, Phase-5 anchor) | all | 1843 (1843) | 0.36 [0.34, 0.39] | 0.00 [0.00, 0.00] | 0.54 [0.52, 0.56] | 0.01 [0.00, 0.01] | 671 / 2 | other answer |
| Qwen3-8B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=complementary | 24 (24) | 0.21 [0.08, 0.38] | 0.00 [0.00, 0.00] | 0.25 [0.08, 0.42] | 0.00 [0.00, 0.00] | 5 / 0 | other answer |
| Qwen3-8B | E3 a→e (F1 frame, Phase-5 anchor) | antonym_type=gradable | 1819 (1819) | 0.37 [0.34, 0.39] | 0.00 [0.00, 0.00] | 0.54 [0.52, 0.56] | 0.01 [0.00, 0.01] | 666 / 2 | other answer |
| Qwen3-8B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=False | 1038 (1038) | 0.40 [0.37, 0.43] | 0.00 [0.00, 0.01] | 0.57 [0.54, 0.60] | 0.01 [0.00, 0.02] | 411 / 2 | other answer |
| Qwen3-8B | E3 a→e (F1 frame, Phase-5 anchor) | degree_adv=True | 805 (805) | 0.32 [0.29, 0.36] | 0.00 [0.00, 0.00] | 0.50 [0.46, 0.54] | 0.00 [0.00, 0.01] | 260 / 0 | other answer |
| Qwen3-8B | E3 a→e (F1 frame, Phase-5 anchor) | names=False | 1587 (1587) | 0.38 [0.36, 0.41] | 0.00 [0.00, 0.00] | 0.54 [0.52, 0.57] | 0.01 [0.00, 0.01] | 606 / 1 | other answer |
| Qwen3-8B | E3 a→e (F1 frame, Phase-5 anchor) | names=True | 256 (256) | 0.25 [0.21, 0.31] | 0.00 [0.00, 0.01] | 0.51 [0.45, 0.57] | 0.00 [0.00, 0.01] | 65 / 1 | other answer |
| Qwen3-8B | E3 a→e (F1 frame, Phase-5 anchor) | order=1 | 1324 (1324) | 0.37 [0.35, 0.40] | 0.00 [0.00, 0.00] | 0.54 [0.52, 0.57] | 0.01 [0.00, 0.01] | 491 / 2 | other answer |
| Qwen3-8B | E3 a→e (F1 frame, Phase-5 anchor) | order=2 | 519 (519) | 0.35 [0.30, 0.39] | 0.00 [0.00, 0.00] | 0.52 [0.48, 0.57] | 0.00 [0.00, 0.01] | 180 / 0 | other answer |
| Qwen3-8B | E1 e→a | all | 3578 (1789) | 0.31 [0.29, 0.33] | 0.00 [0.00, 0.00] | 0.55 [0.53, 0.57] | 0.03 [0.02, 0.03] | 300 / 1 | other answer |
| Qwen3-8B | E1 e→a | antonym_type=complementary | 50 (25) | 0.14 [0.04, 0.24] | 0.00 [0.00, 0.00] | 0.46 [0.32, 0.62] | 0.06 [0.00, 0.12] | 1 / 0 | other answer |
| Qwen3-8B | E1 e→a | antonym_type=gradable | 3528 (1764) | 0.31 [0.30, 0.33] | 0.00 [0.00, 0.00] | 0.55 [0.53, 0.57] | 0.03 [0.02, 0.03] | 299 / 1 | other answer |
| Qwen3-8B | E1 e→a | degree_adv=False | 1986 (993) | 0.22 [0.20, 0.24] | 0.00 [0.00, 0.01] | 0.46 [0.43, 0.48] | 0.04 [0.03, 0.05] | 108 / 1 | other answer |
| Qwen3-8B | E1 e→a | degree_adv=True | 1592 (796) | 0.42 [0.39, 0.45] | 0.00 [0.00, 0.01] | 0.67 [0.65, 0.70] | 0.01 [0.01, 0.02] | 192 / 0 | other answer |
| Qwen3-8B | E1 e→a | names=False | 3060 (1530) | 0.33 [0.31, 0.35] | 0.00 [0.00, 0.00] | 0.59 [0.56, 0.61] | 0.03 [0.02, 0.03] | 291 / 1 | other answer |
| Qwen3-8B | E1 e→a | names=True | 518 (259) | 0.17 [0.13, 0.20] | 0.01 [0.00, 0.02] | 0.36 [0.32, 0.41] | 0.04 [0.03, 0.06] | 9 / 0 | other answer |
| Qwen3-8B | E1 e→a | order=1 | 1789 (1789) | 0.29 [0.27, 0.32] | 0.00 [0.00, 0.00] | 0.54 [0.52, 0.57] | 0.03 [0.02, 0.03] | 527 / 4 | other answer |
| Qwen3-8B | E1 e→a | order=2 | 1789 (1789) | 0.32 [0.30, 0.35] | 0.00 [0.00, 0.01] | 0.56 [0.54, 0.58] | 0.03 [0.02, 0.04] | 581 / 6 | other answer |
| Qwen3-8B | E1p e→a | all | 3370 (1685) | 0.30 [0.28, 0.32] | 0.00 [0.00, 0.00] | 0.54 [0.52, 0.56] | 0.01 [0.01, 0.01] | 300 / 0 | other answer |
| Qwen3-8B | E1p e→a | antonym_type=complementary | 32 (16) | 0.25 [0.09, 0.41] | 0.00 [0.00, 0.00] | 0.44 [0.25, 0.62] | 0.00 [0.00, 0.00] | 2 / 0 | other answer |
| Qwen3-8B | E1p e→a | antonym_type=gradable | 3338 (1669) | 0.30 [0.28, 0.32] | 0.00 [0.00, 0.00] | 0.54 [0.52, 0.56] | 0.01 [0.00, 0.01] | 298 / 0 | other answer |
| Qwen3-8B | E1p e→a | degree_adv=False | 1790 (895) | 0.34 [0.31, 0.36] | 0.00 [0.00, 0.00] | 0.58 [0.55, 0.61] | 0.01 [0.00, 0.01] | 174 / 0 | other answer |
| Qwen3-8B | E1p e→a | degree_adv=True | 1580 (790) | 0.26 [0.23, 0.28] | 0.00 [0.00, 0.00] | 0.50 [0.46, 0.53] | 0.01 [0.00, 0.01] | 126 / 0 | other answer |
| Qwen3-8B | E1p e→a | names=False | 2904 (1452) | 0.29 [0.27, 0.31] | 0.00 [0.00, 0.00] | 0.54 [0.52, 0.56] | 0.01 [0.00, 0.01] | 244 / 0 | other answer |
| Qwen3-8B | E1p e→a | names=True | 466 (233) | 0.36 [0.30, 0.41] | 0.00 [0.00, 0.01] | 0.56 [0.51, 0.62] | 0.01 [0.00, 0.02] | 56 / 0 | other answer |
| Qwen3-8B | E1p e→a | order=1 | 1685 (1685) | 0.31 [0.29, 0.33] | 0.00 [0.00, 0.00] | 0.55 [0.53, 0.58] | 0.00 [0.00, 0.01] | 520 / 0 | other answer |
| Qwen3-8B | E1p e→a | order=2 | 1685 (1685) | 0.29 [0.27, 0.31] | 0.00 [0.00, 0.00] | 0.53 [0.51, 0.56] | 0.01 [0.01, 0.02] | 493 / 2 | other answer |
| Qwen3-8B | E2 e→a | all | 3578 (1789) | 0.03 [0.02, 0.03] | 0.01 [0.01, 0.01] | 0.16 [0.15, 0.18] | 0.06 [0.05, 0.07] | 6 / 5 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | antonym_type=complementary | 50 (25) | 0.04 [0.00, 0.10] | 0.00 [0.00, 0.00] | 0.18 [0.08, 0.30] | 0.02 [0.00, 0.06] | 0 / 0 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | antonym_type=gradable | 3528 (1764) | 0.03 [0.02, 0.03] | 0.01 [0.01, 0.01] | 0.16 [0.15, 0.18] | 0.06 [0.05, 0.07] | 6 / 5 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | degree_adv=False | 1986 (993) | 0.04 [0.03, 0.05] | 0.01 [0.00, 0.01] | 0.21 [0.19, 0.22] | 0.05 [0.04, 0.06] | 5 / 1 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | degree_adv=True | 1592 (796) | 0.01 [0.01, 0.02] | 0.01 [0.01, 0.02] | 0.11 [0.10, 0.13] | 0.08 [0.06, 0.09] | 1 / 4 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | names=False | 3060 (1530) | 0.02 [0.02, 0.03] | 0.01 [0.00, 0.01] | 0.16 [0.14, 0.17] | 0.05 [0.05, 0.06] | 1 / 5 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | names=True | 518 (259) | 0.05 [0.03, 0.08] | 0.02 [0.01, 0.03] | 0.21 [0.18, 0.26] | 0.09 [0.07, 0.12] | 5 / 0 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | order=1 | 1789 (1789) | 0.02 [0.02, 0.03] | 0.01 [0.01, 0.02] | 0.18 [0.16, 0.20] | 0.06 [0.05, 0.07] | 42 / 19 | copy trap (word stated with the other entity) |
| Qwen3-8B | E2 e→a | order=2 | 1789 (1789) | 0.03 [0.02, 0.04] | 0.01 [0.01, 0.01] | 0.15 [0.13, 0.17] | 0.06 [0.05, 0.07] | 55 / 16 | copy trap (word stated with the other entity) |
| Qwen3-8B | E3 e→a | all | 1848 (1848) | 0.50 [0.48, 0.52] | 0.00 [0.00, 0.01] | 0.64 [0.62, 0.67] | 0.01 [0.01, 0.02] | 931 / 6 | other answer |
| Qwen3-8B | E3 e→a | antonym_type=complementary | 24 (24) | 0.54 [0.33, 0.75] | 0.04 [0.00, 0.12] | 0.79 [0.62, 0.92] | 0.04 [0.00, 0.12] | 13 / 1 | other answer |
| Qwen3-8B | E3 e→a | antonym_type=gradable | 1824 (1824) | 0.50 [0.48, 0.53] | 0.00 [0.00, 0.01] | 0.64 [0.62, 0.66] | 0.01 [0.01, 0.02] | 918 / 5 | other answer |
| Qwen3-8B | E3 e→a | degree_adv=False | 1037 (1037) | 0.50 [0.47, 0.53] | 0.00 [0.00, 0.01] | 0.64 [0.61, 0.67] | 0.01 [0.01, 0.02] | 518 / 4 | other answer |
| Qwen3-8B | E3 e→a | degree_adv=True | 811 (811) | 0.51 [0.47, 0.54] | 0.00 [0.00, 0.01] | 0.65 [0.62, 0.69] | 0.01 [0.00, 0.01] | 413 / 2 | other answer |
| Qwen3-8B | E3 e→a | names=False | 1590 (1590) | 0.52 [0.49, 0.54] | 0.00 [0.00, 0.00] | 0.66 [0.64, 0.68] | 0.01 [0.00, 0.01] | 820 / 3 | other answer |
| Qwen3-8B | E3 e→a | names=True | 258 (258) | 0.43 [0.37, 0.50] | 0.01 [0.00, 0.02] | 0.56 [0.50, 0.63] | 0.02 [0.01, 0.04] | 111 / 3 | other answer |
| Qwen3-8B | E3 e→a | order=1 | 1327 (1327) | 0.50 [0.47, 0.53] | 0.00 [0.00, 0.01] | 0.64 [0.62, 0.67] | 0.01 [0.01, 0.02] | 661 / 6 | other answer |
| Qwen3-8B | E3 e→a | order=2 | 521 (521) | 0.52 [0.47, 0.56] | 0.00 [0.00, 0.00] | 0.65 [0.61, 0.69] | 0.01 [0.00, 0.02] | 270 / 0 | other answer |
| Qwen3-8B | Set C agree (a→e) | all | 3686 (1843) | 0.63 [0.61, 0.65] | 0.00 [0.00, 0.00] | 0.83 [0.82, 0.85] | 0.00 [0.00, 0.00] | 927 / 0 | event / other |
| Qwen3-8B | Set C agree (a→e) | antonym_type=complementary | 48 (24) | 0.33 [0.19, 0.50] | 0.00 [0.00, 0.00] | 0.52 [0.33, 0.69] | 0.00 [0.00, 0.00] | 5 / 0 | event / other |
| Qwen3-8B | Set C agree (a→e) | antonym_type=gradable | 3638 (1819) | 0.64 [0.62, 0.65] | 0.00 [0.00, 0.00] | 0.84 [0.82, 0.85] | 0.00 [0.00, 0.00] | 922 / 0 | event / other |
| Qwen3-8B | Set C agree (a→e) | degree_adv=False | 2074 (1037) | 0.62 [0.59, 0.64] | 0.00 [0.00, 0.00] | 0.82 [0.80, 0.84] | 0.00 [0.00, 0.00] | 497 / 0 | event / other |
| Qwen3-8B | Set C agree (a→e) | degree_adv=True | 1612 (806) | 0.65 [0.62, 0.68] | 0.00 [0.00, 0.00] | 0.85 [0.83, 0.87] | 0.00 [0.00, 0.00] | 430 / 0 | event / other |
| Qwen3-8B | Set C agree (a→e) | names=False | 3174 (1587) | 0.65 [0.63, 0.67] | 0.00 [0.00, 0.00] | 0.84 [0.82, 0.86] | 0.00 [0.00, 0.00] | 831 / 0 | event / other |
| Qwen3-8B | Set C agree (a→e) | names=True | 512 (256) | 0.54 [0.49, 0.58] | 0.00 [0.00, 0.00] | 0.79 [0.75, 0.83] | 0.00 [0.00, 0.01] | 96 / 0 | event / other |
| Qwen3-8B | Set C agree (a→e) | order=1 | 1843 (1843) | 0.66 [0.64, 0.68] | 0.00 [0.00, 0.00] | 0.84 [0.83, 0.86] | 0.00 [0.00, 0.00] | 1211 / 0 | event / other |
| Qwen3-8B | Set C agree (a→e) | order=2 | 1843 (1843) | 0.61 [0.59, 0.63] | 0.00 [0.00, 0.00] | 0.82 [0.80, 0.84] | 0.00 [0.00, 0.00] | 1123 / 0 | event / other |
| Qwen3-8B | sc_conflict_event_a2e | all | 3686 (1843) | 0.03 [0.02, 0.03] | 0.29 [0.27, 0.31] | 0.07 [0.06, 0.08] | 0.52 [0.50, 0.54] | 28 / 306 | event / other |
| Qwen3-8B | sc_conflict_event_a2e | antonym_type=complementary | 48 (24) | 0.00 [0.00, 0.00] | 0.12 [0.02, 0.23] | 0.02 [0.00, 0.06] | 0.31 [0.17, 0.46] | 0 / 1 | event / other |
| Qwen3-8B | sc_conflict_event_a2e | antonym_type=gradable | 3638 (1819) | 0.03 [0.02, 0.03] | 0.29 [0.28, 0.31] | 0.07 [0.06, 0.08] | 0.53 [0.51, 0.55] | 28 / 305 | event / other |
| Qwen3-8B | sc_conflict_event_a2e | degree_adv=False | 2074 (1037) | 0.04 [0.03, 0.05] | 0.27 [0.25, 0.29] | 0.11 [0.09, 0.12] | 0.49 [0.46, 0.51] | 27 / 154 | event / other |
| Qwen3-8B | sc_conflict_event_a2e | degree_adv=True | 1612 (806) | 0.01 [0.00, 0.01] | 0.32 [0.29, 0.34] | 0.03 [0.02, 0.04] | 0.57 [0.54, 0.60] | 1 / 152 | event / other |
| Qwen3-8B | sc_conflict_event_a2e | names=False | 3174 (1587) | 0.03 [0.02, 0.04] | 0.28 [0.26, 0.30] | 0.08 [0.07, 0.10] | 0.50 [0.48, 0.52] | 28 / 256 | event / other |
| Qwen3-8B | sc_conflict_event_a2e | names=True | 512 (256) | 0.00 [0.00, 0.00] | 0.35 [0.30, 0.40] | 0.00 [0.00, 0.01] | 0.66 [0.61, 0.70] | 0 / 50 | event / other |
| Qwen3-8B | sc_conflict_event_a2e | order=1 | 1843 (1843) | 0.03 [0.02, 0.03] | 0.32 [0.30, 0.34] | 0.07 [0.06, 0.08] | 0.54 [0.52, 0.56] | 49 / 583 | event / other |
| Qwen3-8B | sc_conflict_event_a2e | order=2 | 1843 (1843) | 0.03 [0.02, 0.03] | 0.26 [0.24, 0.28] | 0.07 [0.06, 0.09] | 0.51 [0.49, 0.53] | 48 / 487 | event / other |
| Qwen3-8B | sc_conflict_stmt_a2e | all | 3686 (1843) | 0.29 [0.27, 0.31] | 0.03 [0.02, 0.03] | 0.52 [0.50, 0.54] | 0.07 [0.06, 0.08] | 306 / 28 | event / other |
| Qwen3-8B | sc_conflict_stmt_a2e | antonym_type=complementary | 48 (24) | 0.12 [0.02, 0.23] | 0.00 [0.00, 0.00] | 0.31 [0.17, 0.46] | 0.02 [0.00, 0.06] | 1 / 0 | event / other |
| Qwen3-8B | sc_conflict_stmt_a2e | antonym_type=gradable | 3638 (1819) | 0.29 [0.28, 0.31] | 0.03 [0.02, 0.03] | 0.53 [0.51, 0.55] | 0.07 [0.06, 0.08] | 305 / 28 | event / other |
| Qwen3-8B | sc_conflict_stmt_a2e | degree_adv=False | 2074 (1037) | 0.27 [0.25, 0.29] | 0.04 [0.03, 0.05] | 0.49 [0.46, 0.51] | 0.11 [0.09, 0.12] | 154 / 27 | event / other |
| Qwen3-8B | sc_conflict_stmt_a2e | degree_adv=True | 1612 (806) | 0.32 [0.29, 0.34] | 0.01 [0.00, 0.01] | 0.57 [0.54, 0.60] | 0.03 [0.02, 0.04] | 152 / 1 | event / other |
| Qwen3-8B | sc_conflict_stmt_a2e | names=False | 3174 (1587) | 0.28 [0.26, 0.30] | 0.03 [0.02, 0.04] | 0.50 [0.48, 0.52] | 0.08 [0.07, 0.10] | 256 / 28 | event / other |
| Qwen3-8B | sc_conflict_stmt_a2e | names=True | 512 (256) | 0.35 [0.30, 0.40] | 0.00 [0.00, 0.00] | 0.66 [0.61, 0.70] | 0.00 [0.00, 0.01] | 50 / 0 | event / other |
| Qwen3-8B | sc_conflict_stmt_a2e | order=1 | 1843 (1843) | 0.32 [0.30, 0.34] | 0.03 [0.02, 0.03] | 0.54 [0.52, 0.56] | 0.07 [0.06, 0.08] | 583 / 49 | event / other |
| Qwen3-8B | sc_conflict_stmt_a2e | order=2 | 1843 (1843) | 0.26 [0.24, 0.28] | 0.03 [0.02, 0.03] | 0.51 [0.49, 0.53] | 0.07 [0.06, 0.09] | 487 / 48 | event / other |

d = logit(target) − logit(other) on each prompt of the pair; 'margin both ways' = the reporting rule (≥ 1 in both directions). E2: the other answer is the copy trap (the attribute word stated next to the other entity); Dial A conflict: the prior; Set C conflict (base cell): the event's answer.
