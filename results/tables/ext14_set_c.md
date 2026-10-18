**Set C (event + statement): behaviour and attention share per subset**

| Model | Cell | Behaviour | Pairs | Share | Reference | Δ share vs reference |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | Set C agree (a→e) | n_pool 3686, n_margin 1609, top1_rate 0.9336 | 128 | 0.83 [0.81, 0.85] | e1_a2e | +0.01 [-0.03, +0.05] |
| Qwen3-30B-A3B | sc_conflict_stmt_a2e | statement 167 / event 10 of 1843 items (statement rate among margin items 0.94) | 72 | 0.87 [0.84, 0.89] | e1_a2e | -0.04 [-0.12, +0.03] |
| Qwen3-30B-A3B | sc_conflict_event_a2e | statement 167 / event 10 of 1843 items (statement rate among margin items 0.94) |  |  |  |  |
| Mixtral-8x7B | Set C agree (a→e) | n_pool 3288, n_margin 639, top1_rate 0.9563 | 86 | 0.61 [0.59, 0.63] | e1_a2e | -0.03 [-0.13, +0.04] |
| Mixtral-8x7B | sc_conflict_stmt_a2e | statement 19 / event 6 of 1644 items (statement rate among margin items 0.76) |  |  |  |  |
| Mixtral-8x7B | sc_conflict_event_a2e | statement 19 / event 6 of 1644 items (statement rate among margin items 0.76) |  |  |  |  |
| Mistral-7B | Set C agree (a→e) | n_pool 3288, n_margin 368, top1_rate 0.9375 |  |  |  |  |
| Mistral-7B | sc_conflict_stmt_a2e | statement 4 / event 3 of 1644 items (statement rate among margin items 0.57) |  |  |  |  |
| Mistral-7B | sc_conflict_event_a2e | statement 4 / event 3 of 1644 items (statement rate among margin items 0.57) |  |  |  |  |
| Qwen2-7B | Set C agree (a→e) | n_pool 3686, n_margin 144, top1_rate 0.9722 |  |  |  |  |
| Qwen2-7B | sc_conflict_stmt_a2e | statement 3 / event 0 of 1843 items (statement rate among margin items 1.00) |  |  |  |  |
| Qwen2-7B | sc_conflict_event_a2e | statement 3 / event 0 of 1843 items (statement rate among margin items 1.00) |  |  |  |  |
| Qwen2-57B-A14B | Set C agree (a→e) | n_pool 1286, n_margin 713, top1_rate 0.9766 |  |  |  |  |
| Qwen2-57B-A14B | sc_conflict_stmt_a2e | statement 11 / event 7 of 643 items (statement rate among margin items 0.61) |  |  |  |  |
| Qwen2-57B-A14B | sc_conflict_event_a2e | statement 11 / event 7 of 643 items (statement rate among margin items 0.61) |  |  |  |  |
| Qwen3-4B | Set C agree (a→e) | n_pool 3686, n_margin 2113, top1_rate 0.9492 |  |  |  |  |
| Qwen3-4B | sc_conflict_stmt_a2e | statement 350 / event 1 of 1843 items (statement rate among margin items 1.00) |  |  |  |  |
| Qwen3-4B | sc_conflict_event_a2e | statement 350 / event 1 of 1843 items (statement rate among margin items 1.00) |  |  |  |  |
| Qwen3-8B | Set C agree (a→e) | n_pool 3686, n_margin 2334, top1_rate 0.9805 |  |  |  |  |
| Qwen3-8B | sc_conflict_stmt_a2e | statement 306 / event 28 of 1843 items (statement rate among margin items 0.92) |  |  |  |  |
| Qwen3-8B | sc_conflict_event_a2e | statement 306 / event 28 of 1843 items (statement rate among margin items 0.92) |  |  |  |  |
| Qwen3-14B | Set C agree (a→e) | n_pool 1286, n_margin 1108, top1_rate 0.9531 |  |  |  |  |
| Qwen3-14B | sc_conflict_stmt_a2e | statement 768 / event 88 of 1843 items (statement rate among margin items 0.90) |  |  |  |  |
| Qwen3-14B | sc_conflict_event_a2e | statement 768 / event 88 of 1843 items (statement rate among margin items 0.90) |  |  |  |  |
| Qwen3-1.7B | Set C agree (a→e) | n_pool 1286, n_margin 23, top1_rate 1.0 |  |  |  |  |
| Qwen3-1.7B | sc_conflict_stmt_a2e | statement 1 / event 0 of 643 items (statement rate among margin items 1.00) |  |  |  |  |
| Qwen3-1.7B | sc_conflict_event_a2e | statement 1 / event 0 of 643 items (statement rate among margin items 1.00) |  |  |  |  |
