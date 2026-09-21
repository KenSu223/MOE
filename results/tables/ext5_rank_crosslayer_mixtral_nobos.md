**Mixtral-8x7B-v0.1 (no BOS, paper protocol): cross-layer greedy over all (layer, expert) pairs in descending discovery all-case rescue, first 12 pairs. 'Sum' assumes full additivity across layers (upper bound, over-counts information that several layers restore); 'per-case max' assumes full redundancy (lower bound). Exact multi-layer patches are F1.4.**

| k | Pair | Disc. active | Disc. all-case | Val. all-case | Cumulative sum (val) | Sum, % of L19 block | Per-case max (val) | Max, % of L19 block |
|---|---|---|---|---|---|---|---|---|
| 1 | L21E001 | 59/128 | +0.237 | +0.290 | +0.290 | 68% | +0.290 | 68% |
| 2 | L19E002 | 59/128 | +0.223 | +0.218 | +0.508 | 119% | +0.402 | 94% |
| 3 | L18E001 | 76/128 | +0.158 | +0.139 | +0.647 | 152% | +0.462 | 108% |
| 4 | L22E001 | 98/128 | +0.134 | +0.100 | +0.747 | 175% | +0.495 | 116% |
| 5 | L22E005 | 58/128 | +0.103 | +0.079 | +0.826 | 194% | +0.537 | 126% |
| 6 | L19E004 | 44/128 | +0.089 | +0.043 | +0.869 | 204% | +0.548 | 128% |
| 7 | L20E005 | 65/128 | +0.085 | +0.123 | +0.992 | 232% | +0.569 | 133% |
| 8 | L21E006 | 47/128 | +0.080 | +0.089 | +1.081 | 253% | +0.580 | 136% |
| 9 | L0E006 | 57/128 | +0.079 | +0.016 | +1.097 | 257% | +0.660 | 155% |
| 10 | L19E006 | 91/128 | +0.074 | +0.063 | +1.160 | 272% | +0.699 | 164% |
| 11 | L20E006 | 23/128 | +0.071 | +0.081 | +1.241 | 291% | +0.714 | 167% |
| 12 | L23E007 | 45/128 | +0.060 | +0.053 | +1.294 | 303% | +0.716 | 168% |
