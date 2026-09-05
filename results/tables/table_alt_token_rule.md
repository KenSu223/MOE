**Secondary run: paper_like object-token rule (no-space token if single, else leading-space), paper case IDs**

| Model | L* | Layer rescue (val) | Next layer | e* | Disc. active | Val. active | Expert rescue (val) | Spec (val) | Clean top-k coalition | Union coalition | Paper IDs passing strict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L44 | +0.924 [+0.765, +1.095] | L43 +0.619 | E069 (5 cand.) | 114/128 | 116/128 | +0.488 [+0.353, +0.641] | +0.438 [+0.302, +0.592] | +0.907 [+0.755, +1.077] | +0.925 [+0.771, +1.097] | 251/256 |
| Mixtral-8x7B-v0.1 | L19 | +0.557 [+0.447, +0.676] | L20 +0.494 | E002 (2 cand.) | 76/128 | 84/128 | +0.360 [+0.265, +0.466] | +0.185 [+0.085, +0.290] | +0.553 [+0.447, +0.669] | +0.566 [+0.457, +0.684] | 234/256 |
