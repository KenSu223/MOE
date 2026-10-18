| model, task | expert | neurons | cases (active rows) | expert rescue T [95% CI] | T / drop | direct share (sum DLA / T) | k50 DLA | k80 DLA [95% CI] | k80 fixed set | k80 magnitude | k80 random | per-case k80 median (IQR; n) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B, CounterFact STR | L44E069 | 768 | 108 (406) | +1.084 [+0.822, +1.380] | 0.093 | 1.31 | 1.0 | 2.1 [2, 3] | 247 | 22 | 589 | 2.1 (1.5-3.7; 62) |
| Qwen3-30B-A3B, CounterFact STR | L42E115 | 768 | 108 (425) | +0.829 [+0.670, +0.987] | 0.071 | 0.97 | 1.3 | 4.3 [3, 6] | 79 | 38 | 596 | 5.5 (2.3-11; 75) |
| Qwen3-30B-A3B, WinoGrande option swap (Phase 3) | L41E117 | 768 | 256 (229) | +0.987 [+0.835, +1.153] | 0.121 | 1.05 | 1.1 | 2.7 [2, 3] | 7.5 | 11 | 597 | 2.9 (1.7-5.0; 190) |
| Qwen3-30B-A3B, WinoGrande attribute -> entity (F1) | L37E032 | 768 | 256 (254) | +0.502 [+0.423, +0.590] | 0.109 | 0.01 | 25 | 511 [431, 567] | 575 | 33 | 589 | 142 (11-592; 177) |
| Mixtral-8x7B (BOS), CounterFact STR | L19E002 | 14336 | 106 (288) | +0.593 [+0.389, +0.839] | 0.046 | 0.75 | 1.6 | 6.7 [4, 11] | 1239 | 80 | 10030 | 11 (3.2-23; 51) |
| Mixtral-8x7B (BOS), CounterFact STR | L21E001 | 14336 | 106 (309) | +0.663 [+0.506, +0.830] | 0.051 | 0.91 | 1.2 | 3.8 [3, 5] | 3323 | 34 | 10524 | 4.0 (1.8-10; 58) |
| Mixtral-8x7B (BOS), CounterFact STR | L18E001 | 14336 | 106 (345) | +0.396 [+0.304, +0.494] | 0.031 | 0.52 | 3.2 | 9.9 [6, 15] | 6145 | 565 | 10479 | 23 (5.6-55; 53) |
| Mixtral-8x7B (BOS), WinoGrande option swap (Phase 3) | L20E000 | 14336 | 256 (246) | +1.112 [+1.017, +1.211] | 0.145 | 0.56 | 4.1 | 17 [13, 21] | 291 | 60 | 10324 | 17 (5.7-43; 238) |
| Mistral-7B (BOS), CounterFact STR (Mixtral items) | L18E000 | 14336 | 98 (369) | +0.178 [+0.091, +0.275] | 0.015 | 0.64 | 1.0 | 1.2 [1, 2] | 26 | 15 | 10526 | 6.6 (2.6-28; 33) |
| Mistral-7B (BOS), CounterFact STR (Mixtral items) | L19E000 | 14336 | 98 (369) | +0.910 [+0.743, +1.083] | 0.074 | 0.67 | 1.0 | 2.9 [2, 4] | 271 | 17 | 10396 | 4.3 (1.8-8.4; 73) |
| Mistral-7B (BOS), CounterFact STR (Mixtral items) | L20E000 | 14336 | 98 (369) | +1.093 [+0.874, +1.324] | 0.089 | 0.75 | 1.2 | 3.4 [3, 4] | 220 | 17 | 10471 | 4.7 (2.1-9.2; 78) |
| Mistral-7B (BOS), CounterFact STR (Mixtral items) | L21E000 | 14336 | 98 (369) | +1.299 [+1.095, +1.500] | 0.106 | 0.81 | 1.0 | 3.0 [2, 4] | 269 | 13 | 10659 | 3.3 (1.7-6.8; 80) |
| Mistral-7B (BOS), WinoGrande option swap (Mixtral items) | L18E000 | 14336 | 228 (228) | +1.036 [+0.939, +1.130] | 0.143 | 0.35 | 4.4 | 19 [15, 23] | 298 | 52 | 10425 | 23 (8.5-50; 223) |
| Mistral-7B (BOS), WinoGrande option swap (Mixtral items) | L19E000 | 14336 | 228 (228) | +0.692 [+0.605, +0.779] | 0.096 | 0.36 | 3.5 | 10 [8, 12] | 92 | 49 | 10558 | 13 (5.7-25; 194) |
| Mistral-7B (BOS), WinoGrande option swap (Mixtral items) | L20E000 | 14336 | 228 (228) | +1.038 [+0.923, +1.150] | 0.144 | 0.35 | 3.6 | 14 [10, 20] | 441 | 71 | 10368 | 20 (6.1-42; 207) |
| Mistral-7B (BOS), WinoGrande option swap (Mixtral items) | L21E000 | 14336 | 228 (228) | +1.361 [+1.236, +1.495] | 0.189 | 0.54 | 2.2 | 7.0 [6, 8] | 53 | 36 | 10465 | 6.9 (3.8-12; 218) |

k50 / k80 = the number of neurons whose joint patch (set to the clean values inside the corrupted run, final position, downstream recomputed) restores 50 / 80 % of the expert's own single-expert patch (ratio of case means, validation split; log-k interpolation; 'never' = not reached before all neurons). DLA = the row's own neuron DLA ranking; fixed set = one neuron order from the discovery-split mean DLA; magnitude = |w a change| x |W_down column|.
Per-case k80 on cases with T >= 0.25 logits.
