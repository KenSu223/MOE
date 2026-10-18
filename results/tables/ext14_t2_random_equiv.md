**Random-order k that reaches greedy's (oracle's) r at k = 1, 5, 10, 15 (P6-0 definition, interpolated in log k)**

| Cell | Model | Curve | k=1 | k=5 | k=10 | k=15 |
|---|---|---|---|---|---|---|
| E1 a→e | Qwen3-30B-A3B-Base | greedy | never | never | never | never |
| E1 a→e | Qwen3-30B-A3B-Base | oracle | never | never | never | never |
| E1 a→e | Mixtral-8x7B (BOS) | greedy | 21 | never | never | never |
| E1 a→e | Mixtral-8x7B (BOS) | oracle | 21 | never | never | never |

'never' = the random order (which ends at the all-MoE ceiling) never reaches that value.
