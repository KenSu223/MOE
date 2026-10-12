**Phase 5 knockout (route mask at all positions, reroute), eval items (validation + replication pairs, both twins): fraction of the clean margin lost F [95% CI]; pair accuracy baseline -> knockout; null = baseline recomputed in other passes**

| model | condition | knockout | role | routed at final | F [95% CI] | pair acc base -> ko | F routed items | null F |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | f1pre | L37E032 | selected | 1.0 | +0.074 [+0.000, +0.158] | 1.000 -> 0.833 | 0.0738 | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | L41E117 | wg | 0.0 | -0.004 [-0.038, +0.026] | 1.000 -> 1.000 |  | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | L44E069 | cf | 0.1667 | -0.004 [-0.015, +0.009] | 1.000 -> 1.000 |  | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | L42E115 | cf | 0.2917 | -0.006 [-0.035, +0.029] | 1.000 -> 1.000 | 0.0431 | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | top-3 | pop |  | +0.091 [+0.019, +0.178] | 1.000 -> 0.833 |  | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | top-5 | pop |  | +0.097 [+0.024, +0.188] | 1.000 -> 0.833 |  | -0.019 |
| Qwen3-30B-A3B-Base | f1pre | top-10 | pop |  | +0.127 [+0.057, +0.212] | 1.000 -> 0.833 |  | -0.019 |
