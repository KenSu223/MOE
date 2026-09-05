**Table 1: main validation results (128 held-out cases per model; paper case IDs)**

| Model | Layer (ours) | Layer rescue (ours) | Layer rescue (paper) | Expert (ours) | Expert (paper) | Expert rescue (ours) | Expert rescue (paper) | Spec (ours) | Spec (paper) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L44 (paper L44) | +0.941 [+0.778, +1.124] | +0.901 [+0.752, +1.053] | L44E069 | L44E069 | +0.503 [+0.362, +0.661] | +0.463 [+0.344, +0.590] | +0.450 [+0.310, +0.608] | +0.400 [+0.276, +0.533] |
| Mixtral-8x7B-v0.1 | L19 (paper L19) | +0.571 [+0.461, +0.692] | +0.457 [+0.331, +0.579] | L19E002 | L19E006 | +0.352 [+0.257, +0.458] | +0.099 [+0.018, +0.175] | +0.205 [+0.109, +0.308] | -0.175 [-0.284, -0.072] |
