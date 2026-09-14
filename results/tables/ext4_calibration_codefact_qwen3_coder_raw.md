**qwen3_coder (raw): per-category calibration of the paper's filter on CodeFact (clean vs subject-noised Δ = logit(true) − logit(foil))**

| Category | n scanned | median Δ_clean | median drop | paper filter (Δ≥1, drop≥0.5) | relaxed (Δ≥0.5, drop≥0.25) | relative (drop ≥ 25 % Δ_clean, Δ≥1) | relative 50 % | top-1 = true / starts with true (clean) | top-1 = true / starts with true (noised) | Δ_clean ≥ 1 | drop ≥ 0.5 | final token carries max norm (L5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 964 | +13.88 | +3.03 | 810 (84 %) | 852 (88 %) | 442 (46 %) | 167 (17 %) | 16 % / 89 % | 15 % / 77 % | 100 % | 84 % | 0.0 % |
| S2 block keyword | 1200 | +5.38 | +0.12 | 395 (33 %) | 517 (43 %) | 125 (10 %) | 27 (2 %) | 80 % / 80 % | 79 % / 79 % | 82 % | 38 % | 0.0 % |
| S3 keyword completion | 1195 | +4.50 | +0.12 | 474 (40 %) | 538 (45 %) | 262 (22 %) | 68 (6 %) | 37 % / 96 % | 37 % / 94 % | 79 % | 42 % | 0.0 % |
| R1 variable recall | 1132 | +12.75 | +2.62 | 899 (79 %) | 945 (83 %) | 504 (45 %) | 257 (23 %) | 88 % / 88 % | 63 % / 63 % | 97 % | 81 % | 0.0 % |
| R2 attribute / API recall | 794 | +11.25 | +1.62 | 576 (73 %) | 617 (78 %) | 232 (29 %) | 58 (7 %) | 89 % / 89 % | 83 % / 83 % | 95 % | 74 % | 0.0 % |
| R3 constant recall | 1163 | +12.25 | +0.94 | 696 (60 %) | 761 (65 %) | 291 (25 %) | 121 (10 %) | 87 % / 90 % | 77 % / 81 % | 96 % | 61 % | 0.0 % |
| ALL | 6448 | +10.84 | +1.00 | 3850 (60 %) | 4230 (66 %) | 1856 (29 %) | 698 (11 %) | 66 % / 89 % | 59 % / 79 % | 91 % | 62 % | 0.0 % |
