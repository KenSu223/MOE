# Random-order experts needed to match greedy's / the oracle's first k (Phase-3/4 add-back, validation)

k_rand(k) = smallest k' with r_random(k') >= r_greedy(k) (log-k interpolation); 'never' = not reached. Written by
scripts/ext8_plot_simple.py (Phase 6, P6-0).

| Task · model | K | greedy 1 | greedy 5 | greedy 10 | greedy 15 | oracle 1 | oracle 5 | oracle 10 | oracle 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CounterFact · Qwen3-30B-A3B | 384 | 137 | 325 | never | never | 138 | 296 | 344 | 365 |
| CounterFact · Mixtral-8x7B | 64 | 16 | 62 | never | never | 16 | 52 | never | never |
| WinoGrande · Qwen3-30B-A3B | 384 | 74 | 207 | 269 | 294 | 75 | 171 | 208 | 227 |
| WinoGrande · Mixtral-8x7B | 64 | 12 | 37 | 54 | never | 12 | 35 | 48 | 56 |
