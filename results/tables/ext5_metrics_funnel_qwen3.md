**Qwen3-30B-A3B-Base (tokenizer defaults): overlap of the paper's Δ funnel with the alternative p_clean(true) >= 0.5 over the 256 cases of the run (the funnel scan itself was run on Δ; these are the cases that entered any case set)**

| Funnels (A vs B) | pass A | pass B | both | A only | B only | neither | Jaccard |
|---|---|---|---|---|---|---|---|
| strict Δ (>= 1.0, drop >= 0.5) vs p_clean >= 0.5 | 235 | 35 | 35 | 200 | 0 | 21 | 0.15 |
| relaxed Δ (>= 0.5, drop >= 0.25) vs p_clean >= 0.5 | 241 | 35 | 35 | 206 | 0 | 15 | 0.15 |
| strict Δ vs p_clean >= 0.5 & p drop >= 0.25 | 235 | 35 | 35 | 200 | 0 | 21 | 0.15 |
