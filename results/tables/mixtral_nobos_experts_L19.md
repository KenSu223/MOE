**Mixtral without BOS (add_special_tokens=False), paper case IDs: validation results (Tables 1, 11 analogues)**

| Expert | Val. active | Expert rescue (all-case) | Active-random control | Spec | Spec sign-flip p | Anchor-active n | Raw active-pair spec | Selected equal-norm | Other equal-norm | Equal-norm spec |
|---|---|---|---|---|---|---|---|---|---|---|
| L19E006 (selected) (paper's expert) | 83/128 | +0.073 [+0.007, +0.140] | +0.244 [+0.173, +0.321] | -0.171 [-0.262, -0.082] | 0.0001 | 83 | -0.103 [-0.223, +0.011] | +0.090 [+0.046, +0.135] | +0.132 [+0.063, +0.201] | -0.042 [-0.109, +0.027] |
| Paper L19E006 | 83/128 | +0.099 [+0.018, +0.175] |  | -0.175 [-0.284, -0.072] |  | 83 | -0.081 [-0.214, +0.042] | +0.046 [-0.007, +0.096] | +0.108 [+0.047, +0.172] | -0.062 [-0.130, -0.003] |
