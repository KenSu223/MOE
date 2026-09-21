**Mixtral-8x7B-v0.1 (no BOS, paper protocol): per-case coverage variant (case covered when its additive sum over S reaches 80% of its own block rescue; greedy on discovery, evaluated on validation)**

| Layer | Eligible cases (block > 0) | Coverage-greedy order (first 4) | Cases covered at |S| = 1 / 2 / 4 / 8 | |S| covering 50 / 80% of cases | Max coverage |
|---|---|---|---|---|---|
| L17 | 64/128 | E001, E000, E005, E007 | 27% / 34% / 59% / 62% | 3 / never | 62% |
| L18 | 65/128 | E001, E006, E003, E002 | 32% / 45% / 52% / 65% | 3 / never | 65% |
| L19 | 92/128 | E002, E006, E004, E007 | 22% / 45% / 64% / 77% | 3 / never | 77% |
| L20 | 83/128 | E005, E000, E006, E002 | 13% / 25% / 46% / 77% | 5 / never | 77% |
| L21 | 85/128 | E001, E006, E000, E005 | 28% / 51% / 69% / 86% | 2 / 6 | 86% |
| L22 | 72/128 | E001, E005, E000, E006 | 33% / 58% / 69% / 79% | 2 / never | 79% |
