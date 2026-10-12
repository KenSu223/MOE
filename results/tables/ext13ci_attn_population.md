| Run | Heads | str | mention (answer side / other) | blank | pos0 | final | attr (answer side / other) | prefix_ment (answer side / other) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base, F1 attr→entity | all heads | 0.02 | 0.02 / 0.01 | 0.03 | 0.51 | 0.12 |  |  |
| Qwen3-30B-A3B-Base, F1 attr→entity | positive heads, rescue-weighted | 0.06 | 0.10 / 0.04 | 0.03 | 0.28 | 0.15 |  |  |
| Qwen3-30B-A3B-Base, F1 attr→entity | top-10 heads, rescue-weighted | 0.06 | 0.17 / 0.04 | 0.02 | 0.20 | 0.24 |  |  |
| Qwen3-30B-A3B-Base, F1 attr→entity | best head of each row | 0.07 | 0.19 / 0.04 | 0.01 | 0.09 | 0.37 |  |  |
| Qwen3-30B-A3B-Base, F1 attr→entity | labels of the top-10 heads | mover (mention: answer) 3, diffuse 3, self (final token) 2, sink (pos0) 1, STR site (trigger) 1 |  |  |  |  |  |  |
| Mixtral-8x7B (BOS), F1 attr→entity | all heads | 0.03 | 0.02 / 0.02 | 0.02 | 0.36 | 0.06 |  |  |
| Mixtral-8x7B (BOS), F1 attr→entity | positive heads, rescue-weighted | 0.10 | 0.05 / 0.02 | 0.01 | 0.22 | 0.07 |  |  |
| Mixtral-8x7B (BOS), F1 attr→entity | top-10 heads, rescue-weighted | 0.24 | 0.05 / 0.02 | 0.01 | 0.08 | 0.11 |  |  |
| Mixtral-8x7B (BOS), F1 attr→entity | best head of each row | 0.37 | 0.11 / 0.05 | 0.01 | 0.05 | 0.06 |  |  |
| Mixtral-8x7B (BOS), F1 attr→entity | labels of the top-10 heads | STR site (trigger) 5, other context tokens 3, self (final token) 1, diffuse 1 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | all heads | 0.04 | 0.02 / 0.01 | 0.04 | 0.48 | 0.12 |  |  |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | positive heads, rescue-weighted | 0.09 | 0.05 / 0.02 | 0.03 | 0.33 | 0.14 |  |  |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | top-10 heads, rescue-weighted | 0.12 | 0.11 / 0.03 | 0.02 | 0.16 | 0.24 |  |  |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | best head of each row | 0.04 | 0.06 / 0.01 | 0.01 | 0.07 | 0.53 |  |  |
| Qwen3-30B-A3B-Base, dir: F1 on Phase-3 pairs | labels of the top-10 heads | mover (mention: answer) 3, self (final token) 2, STR site (trigger) 2, diffuse 2, other context tokens 1 |  |  |  |  |  |  |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | all heads | 0.04 | 0.02 / 0.01 | 0.01 | 0.36 | 0.06 |  |  |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | positive heads, rescue-weighted | 0.14 | 0.05 / 0.02 | 0.01 | 0.21 | 0.07 |  |  |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | top-10 heads, rescue-weighted | 0.35 | 0.07 / 0.02 | 0.01 | 0.07 | 0.09 |  |  |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | best head of each row | 0.51 | 0.12 / 0.02 | 0.00 | 0.04 | 0.03 |  |  |
| Mixtral-8x7B (BOS), dir: F1 on Phase-3 pairs | labels of the top-10 heads | STR site (trigger) 5, mover (mention: answer) 2, diffuse 1, self (final token) 1, other context tokens 1 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | all heads | 0.05 | 0.03 / 0.02 |  | 0.59 | 0.09 |  |  |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | positive heads, rescue-weighted | 0.14 | 0.04 / 0.02 |  | 0.45 | 0.11 |  |  |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | top-10 heads, rescue-weighted | 0.23 | 0.04 / 0.02 |  | 0.33 | 0.10 |  |  |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | best head of each row | 0.26 | 0.04 / 0.02 |  | 0.26 | 0.17 |  |  |
| Qwen3-30B-A3B-Base, dir: option swap on the same pairs | labels of the top-10 heads | STR site (filled entity) 5, sink (pos0) 3, diffuse 2 |  |  |  |  |  |  |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | all heads | 0.05 | 0.02 / 0.02 |  | 0.64 | 0.05 |  |  |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | positive heads, rescue-weighted | 0.26 | 0.06 / 0.02 |  | 0.27 | 0.10 |  |  |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | top-10 heads, rescue-weighted | 0.36 | 0.08 / 0.02 |  | 0.16 | 0.09 |  |  |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | best head of each row | 0.45 | 0.06 / 0.01 |  | 0.10 | 0.11 |  |  |
| Mixtral-8x7B (BOS), dir: option swap on the same pairs | labels of the top-10 heads | STR site (filled entity) 7, diffuse 2, mover (mention: answer) 1 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | all heads | 0.04 | 0.02 / 0.02 |  | 0.54 | 0.10 | 0.02 / 0.01 | nan / nan |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | positive heads, rescue-weighted | 0.08 | 0.03 / 0.02 |  | 0.38 | 0.14 | 0.02 / 0.02 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | top-10 heads, rescue-weighted | 0.12 | 0.02 / 0.01 |  | 0.28 | 0.16 | 0.01 / 0.01 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | best head of each row | 0.13 | 0.03 / 0.01 |  | 0.12 | 0.25 | 0.01 / 0.01 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, copy prefix, natural Phase-3 pairs | labels of the top-10 heads | other context tokens 3, sink (pos0) 3, diffuse 2, STR site (filled entity) 2 |  |  |  |  |  |  |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | all heads | 0.05 | 0.02 / 0.02 |  | 0.41 | 0.05 | 0.01 / 0.01 | nan / nan |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | positive heads, rescue-weighted | 0.17 | 0.05 / 0.02 |  | 0.21 | 0.09 | 0.02 / 0.01 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | top-10 heads, rescue-weighted | 0.28 | 0.07 / 0.02 |  | 0.11 | 0.09 | 0.01 / 0.00 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | best head of each row | 0.38 | 0.07 / 0.01 |  | 0.06 | 0.10 | 0.01 / 0.00 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), copy prefix, natural Phase-3 pairs | labels of the top-10 heads | STR site (filled entity) 6, other context tokens 2, diffuse 1, mover (mention: answer) 1 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | all heads | 0.04 | 0.02 / 0.02 |  | 0.52 | 0.10 | 0.02 / 0.01 | nan / nan |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | positive heads, rescue-weighted | 0.10 | 0.03 / 0.02 |  | 0.37 | 0.11 | 0.04 / 0.02 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | top-10 heads, rescue-weighted | 0.17 | 0.03 / 0.02 |  | 0.22 | 0.15 | 0.02 / 0.02 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | best head of each row | 0.31 | 0.02 / 0.02 |  | 0.17 | 0.18 | 0.05 / 0.01 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, copy prefix, templates orig | labels of the top-10 heads | STR site (filled entity) 4, diffuse 3, sink (pos0) 2, self (final token) 1 |  |  |  |  |  |  |
| Mixtral-8x7B (BOS), copy prefix, templates orig | all heads | 0.03 | 0.01 / 0.01 |  | 0.42 | 0.05 | 0.02 / 0.01 | nan / nan |
| Mixtral-8x7B (BOS), copy prefix, templates orig | positive heads, rescue-weighted | 0.09 | 0.02 / 0.01 |  | 0.24 | 0.08 | 0.03 / 0.02 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), copy prefix, templates orig | top-10 heads, rescue-weighted | 0.19 | 0.03 / 0.01 |  | 0.16 | 0.08 | 0.02 / 0.01 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), copy prefix, templates orig | best head of each row | 0.32 | 0.04 / 0.01 |  | 0.06 | 0.09 | 0.01 / 0.00 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), copy prefix, templates orig | labels of the top-10 heads | other context tokens 6, STR site (filled entity) 2, diffuse 1, copy (attr: answer) 1 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | all heads | 0.04 | 0.02 / 0.02 |  | 0.51 | 0.10 | 0.02 / 0.01 | nan / nan |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | positive heads, rescue-weighted | 0.09 | 0.02 / 0.02 |  | 0.39 | 0.11 | 0.03 / 0.02 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | top-10 heads, rescue-weighted | 0.19 | 0.02 / 0.02 |  | 0.25 | 0.14 | 0.02 / 0.01 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | best head of each row | 0.28 | 0.02 / 0.01 |  | 0.18 | 0.23 | 0.02 / 0.01 | 0.00 / 0.00 |
| Qwen3-30B-A3B-Base, copy prefix, templates neutral | labels of the top-10 heads | diffuse 4, STR site (filled entity) 3, sink (pos0) 2, other context tokens 1 |  |  |  |  |  |  |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | all heads | 0.03 | 0.01 / 0.01 |  | 0.42 | 0.05 | 0.01 / 0.01 | nan / nan |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | positive heads, rescue-weighted | 0.07 | 0.01 / 0.01 |  | 0.30 | 0.06 | 0.02 / 0.01 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | top-10 heads, rescue-weighted | 0.14 | 0.02 / 0.01 |  | 0.15 | 0.08 | 0.01 / 0.01 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | best head of each row | 0.26 | 0.03 / 0.01 |  | 0.10 | 0.09 | 0.01 / 0.01 | 0.00 / 0.00 |
| Mixtral-8x7B (BOS), copy prefix, templates neutral | labels of the top-10 heads | other context tokens 8, STR site (filled entity) 2 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | all heads | 0.04 | 0.22 / 0.22 |  | 0.59 | 0.10 |  |  |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | positive heads, rescue-weighted | 0.11 | 0.16 / 0.15 |  | 0.39 | 0.13 |  |  |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | top-10 heads, rescue-weighted | 0.19 | 0.10 / 0.09 |  | 0.24 | 0.15 |  |  |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | best head of each row | 0.31 | 0.09 / 0.07 |  | 0.20 | 0.14 |  |  |
| Qwen3-30B-A3B-Base, verb templates e→a, orig verb | labels of the top-10 heads | diffuse 4, STR site (filled entity) 3, sink (pos0) 2, self (final token) 1 |  |  |  |  |  |  |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | all heads | 0.04 | 0.01 / 0.01 |  | 0.65 | 0.05 |  |  |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | positive heads, rescue-weighted | 0.14 | 0.02 / 0.01 |  | 0.36 | 0.08 |  |  |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | top-10 heads, rescue-weighted | 0.25 | 0.02 / 0.01 |  | 0.25 | 0.07 |  |  |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | best head of each row | 0.41 | 0.03 / 0.01 |  | 0.09 | 0.07 |  |  |
| Mixtral-8x7B (BOS), verb templates e→a, orig verb | labels of the top-10 heads | STR site (filled entity) 6, diffuse 2, other context tokens 2 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | all heads | 0.04 | 0.24 / 0.24 |  | 0.58 | 0.10 |  |  |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | positive heads, rescue-weighted | 0.08 | 0.19 / 0.20 |  | 0.46 | 0.11 |  |  |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | top-10 heads, rescue-weighted | 0.19 | 0.11 / 0.11 |  | 0.24 | 0.15 |  |  |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | best head of each row | 0.25 | 0.10 / 0.07 |  | 0.20 | 0.17 |  |  |
| Qwen3-30B-A3B-Base, verb templates e→a, neutral verb | labels of the top-10 heads | diffuse 3, STR site (filled entity) 3, mover (mention: other candidate) 2, other context tokens 1, self (final token) 1 |  |  |  |  |  |  |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | all heads | 0.04 | 0.01 / 0.01 |  | 0.65 | 0.05 |  |  |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | positive heads, rescue-weighted | 0.13 | 0.02 / 0.01 |  | 0.33 | 0.08 |  |  |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | top-10 heads, rescue-weighted | 0.22 | 0.02 / 0.01 |  | 0.23 | 0.06 |  |  |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | best head of each row | 0.41 | 0.02 / 0.01 |  | 0.09 | 0.05 |  |  |
| Mixtral-8x7B (BOS), verb templates e→a, neutral verb | labels of the top-10 heads | STR site (filled entity) 4, other context tokens 4, diffuse 2 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | all heads | 0.03 | 0.02 / 0.01 | 0.03 | 0.55 | 0.13 |  |  |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | positive heads, rescue-weighted | 0.12 | 0.08 / 0.04 | 0.02 | 0.34 | 0.13 |  |  |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | top-10 heads, rescue-weighted | 0.17 | 0.19 / 0.08 | 0.01 | 0.20 | 0.18 |  |  |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | best head of each row | 0.02 | 0.18 / 0.06 | 0.01 | 0.17 | 0.50 |  |  |
| Qwen3-30B-A3B-Base, verb templates a→e, orig verb | labels of the top-10 heads | mover (mention: answer) 4, STR site (trigger) 3, self (final token) 1, sink (pos0) 1, diffuse 1 |  |  |  |  |  |  |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | all heads | 0.03 | 0.09 / 0.08 |  | 0.56 | 0.09 |  |  |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | positive heads, rescue-weighted | 0.05 | 0.12 / 0.07 |  | 0.42 | 0.11 |  |  |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | top-10 heads, rescue-weighted | 0.06 | 0.18 / 0.06 |  | 0.25 | 0.13 |  |  |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | best head of each row | 0.06 | 0.08 / 0.02 |  | 0.15 | 0.17 |  |  |
| Qwen3-30B-A3B-Base, F2E natural, blank at end | labels of the top-10 heads | mover (mention: answer) 5, other context tokens 2, sink (pos0) 1, self (final token) 1, diffuse 1 |  |  |  |  |  |  |
| Mixtral-8x7B (BOS), F2E natural, blank at end | all heads | 0.03 | 0.04 / 0.03 |  | 0.60 | 0.05 |  |  |
| Mixtral-8x7B (BOS), F2E natural, blank at end | positive heads, rescue-weighted | 0.04 | 0.09 / 0.04 |  | 0.42 | 0.06 |  |  |
| Mixtral-8x7B (BOS), F2E natural, blank at end | top-10 heads, rescue-weighted | 0.06 | 0.14 / 0.05 |  | 0.19 | 0.07 |  |  |
| Mixtral-8x7B (BOS), F2E natural, blank at end | best head of each row | 0.07 | 0.12 / 0.05 |  | 0.07 | 0.04 |  |  |
| Mixtral-8x7B (BOS), F2E natural, blank at end | labels of the top-10 heads | other context tokens 5, mover (mention: answer) 2, mover (mention: other candidate) 1, diffuse 1, sink (pos0) 1 |  |  |  |  |  |  |
