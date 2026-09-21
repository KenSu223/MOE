**Mixtral-8x7B-v0.1 (BOS, tokenizer default): overlap of the paper's Δ funnel with the alternative p_clean(true) >= 0.5 over the 256 cases of the run (the funnel scan itself was run on Δ; these are the cases that entered any case set)**

| Funnels (A vs B) | pass A | pass B | both | A only | B only | neither | Jaccard |
|---|---|---|---|---|---|---|---|
| strict Δ (>= 1.0, drop >= 0.5) vs p_clean >= 0.5 | 234 | 45 | 44 | 190 | 1 | 21 | 0.19 |
| relaxed Δ (>= 0.5, drop >= 0.25) vs p_clean >= 0.5 | 240 | 45 | 45 | 195 | 0 | 16 | 0.19 |
| strict Δ vs p_clean >= 0.5 & p drop >= 0.25 | 234 | 41 | 41 | 193 | 0 | 22 | 0.18 |
