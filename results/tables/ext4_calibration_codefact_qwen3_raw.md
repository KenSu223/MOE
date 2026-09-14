**qwen3 (raw): per-category calibration of the paper's filter on CodeFact (clean vs subject-noised Δ = logit(true) − logit(foil))**

| Category | n scanned | median Δ_clean | median drop | paper filter (Δ≥1, drop≥0.5) | relaxed (Δ≥0.5, drop≥0.25) | relative (drop ≥ 25 % Δ_clean, Δ≥1) | relative 50 % | top-1 = true / starts with true (clean) | top-1 = true / starts with true (noised) | Δ_clean ≥ 1 | drop ≥ 0.5 | final token carries max norm (L5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 closing bracket | 964 | +13.56 | +4.38 | 871 (90 %) | 910 (94 %) | 593 (62 %) | 290 (30 %) | 18 % / 87 % | 16 % / 69 % | 100 % | 90 % | 0.0 % |
| S2 block keyword | 1200 | +3.25 | +0.25 | 399 (33 %) | 555 (46 %) | 179 (15 %) | 36 (3 %) | 81 % / 81 % | 80 % / 80 % | 78 % | 37 % | 0.0 % |
| S3 keyword completion | 1195 | +3.75 | +0.12 | 430 (36 %) | 543 (45 %) | 244 (20 %) | 71 (6 %) | 38 % / 96 % | 36 % / 93 % | 88 % | 37 % | 0.0 % |
| R1 variable recall | 1132 | +8.12 | +1.88 | 896 (79 %) | 971 (86 %) | 562 (50 %) | 278 (25 %) | 89 % / 89 % | 62 % / 63 % | 96 % | 82 % | 0.0 % |
| R2 attribute / API recall | 794 | +7.38 | +0.75 | 451 (57 %) | 528 (66 %) | 189 (24 %) | 62 (8 %) | 90 % / 90 % | 84 % / 84 % | 94 % | 58 % | 0.0 % |
| R3 constant recall | 1163 | +7.62 | +0.88 | 690 (59 %) | 797 (69 %) | 338 (29 %) | 159 (14 %) | 85 % / 89 % | 78 % / 82 % | 94 % | 61 % | 0.0 % |
| ALL | 6448 | +7.00 | +0.88 | 3737 (58 %) | 4304 (67 %) | 2105 (33 %) | 896 (14 %) | 67 % / 89 % | 59 % / 78 % | 91 % | 60 % | 0.0 % |
