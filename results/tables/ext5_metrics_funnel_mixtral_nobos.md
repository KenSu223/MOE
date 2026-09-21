**Mixtral-8x7B-v0.1 (no BOS, paper protocol): overlap of the paper's Δ funnel with the alternative p_clean(true) >= 0.5 over the 256 cases of the run (the funnel scan itself was run on Δ; these are the cases that entered any case set)**

| Funnels (A vs B) | pass A | pass B | both | A only | B only | neither | Jaccard |
|---|---|---|---|---|---|---|---|
| strict Δ (>= 1.0, drop >= 0.5) vs p_clean >= 0.5 | 249 | 26 | 26 | 223 | 0 | 7 | 0.10 |
| relaxed Δ (>= 0.5, drop >= 0.25) vs p_clean >= 0.5 | 252 | 26 | 26 | 226 | 0 | 4 | 0.10 |
| strict Δ vs p_clean >= 0.5 & p drop >= 0.25 | 249 | 26 | 26 | 223 | 0 | 7 | 0.10 |
