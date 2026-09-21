**Mixtral-8x7B-v0.1 (BOS, tokenizer default): cross-layer greedy over all (layer, expert) pairs in descending discovery all-case rescue, first 12 pairs. 'Sum' assumes full additivity across layers (upper bound, over-counts information that several layers restore); 'per-case max' assumes full redundancy (lower bound). Exact multi-layer patches are F1.4.**

| k | Pair | Disc. active | Disc. all-case | Val. all-case | Cumulative sum (val) | Sum, % of L19 block | Per-case max (val) | Max, % of L19 block |
|---|---|---|---|---|---|---|---|---|
| 1 | L19E002 | 76/128 | +0.384 | +0.363 | +0.363 | 63% | +0.363 | 63% |
| 2 | L21E001 | 91/128 | +0.336 | +0.273 | +0.636 | 110% | +0.512 | 88% |
| 3 | L18E001 | 98/128 | +0.192 | +0.244 | +0.880 | 152% | +0.609 | 105% |
| 4 | L22E001 | 95/128 | +0.155 | +0.179 | +1.060 | 183% | +0.662 | 114% |
| 5 | L20E005 | 75/128 | +0.134 | +0.155 | +1.215 | 210% | +0.701 | 121% |
| 6 | L19E006 | 71/128 | +0.100 | +0.079 | +1.293 | 223% | +0.723 | 125% |
| 7 | L28E002 | 88/128 | +0.097 | +0.086 | +1.380 | 238% | +0.729 | 126% |
| 8 | L20E006 | 34/128 | +0.090 | +0.114 | +1.494 | 258% | +0.758 | 131% |
| 9 | L20E000 | 52/128 | +0.079 | +0.041 | +1.534 | 265% | +0.763 | 132% |
| 10 | L22E005 | 62/128 | +0.079 | +0.120 | +1.654 | 285% | +0.791 | 136% |
| 11 | L28E005 | 36/128 | +0.069 | +0.048 | +1.702 | 294% | +0.792 | 137% |
| 12 | L23E002 | 60/128 | +0.064 | +0.116 | +1.818 | 314% | +0.801 | 138% |
