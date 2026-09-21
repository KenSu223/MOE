**Qwen3-30B-A3B-Base (tokenizer defaults): per-case coverage variant (case covered when its additive sum over S reaches 80% of its own block rescue; greedy on discovery, evaluated on validation)**

| Layer | Eligible cases (block > 0) | Coverage-greedy order (first 4) | Cases covered at |S| = 1 / 2 / 4 / 8 | |S| covering 50 / 80% of cases | Max coverage |
|---|---|---|---|---|---|
| L44 | 112/128 | E069, E006, E098, E052 | 30% / 42% / 51% / 59% | 4 / never | 66% |
| L42 | 111/128 | E115, E016, E080, E071 | 52% / 52% / 60% / 62% | 1 / never | 72% |
| L43 | 96/128 | E104, E005, E046, E036 | 12% / 25% / 44% / 55% | 5 / never | 70% |
| L40 | 95/128 | E030, E127, E084, E026 | 14% / 33% / 40% / 59% | 5 / never | 62% |
