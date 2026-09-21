**Mixtral-8x7B-v0.1 (no BOS, paper protocol): per case set**

| Set | n | pass strict Δ | pass p_clean >= 0.5 | clean top-1 | noise flips top-1 | median p_clean | p_clean >= 0.9 |
|---|---|---|---|---|---|---|---|
| paper | 256 | 249 | 26 | 74 | 70 | 0.024 | 2 |
| strict | 230 | 227 | 24 | 69 | 65 | 0.028 | 2 |
| relaxed | 241 | 238 | 26 | 74 | 70 | 0.028 | 2 |
