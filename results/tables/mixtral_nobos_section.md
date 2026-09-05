### Mixtral without BOS (add_special_tokens=False), paper case IDs

Run directory `results/mixtral_nobos`; paper case IDs (128 discovery / 128 validation); paper IDs passing the strict filter under this tokenisation: 249/256; mean clean Delta +5.499, mean subject-noise drop +4.797.

**Mixtral without BOS (add_special_tokens=False), paper case IDs: layer level**

| Quantity | Ours | Paper |
|---|---|---|
| Discovery-selected layer | L19 | L19 |
| Validation rescue at L19 | +0.446 [+0.318, +0.569] | +0.457 [+0.331, +0.579] |
| Validation top layer / rescue | L21 +0.531 | L21 +0.496 |
| Validation next layer / rescue | L19 +0.446 | L19 +0.457 |
| Discovery top-5 layers | L19 +0.436, L21 +0.398, L20 +0.301, L22 +0.280, L18 +0.274 |  |

**Mixtral without BOS (add_special_tokens=False), paper case IDs: every expert at L19 (paper: E006 active 91/128 disc, 83/128 val)**

| Expert | Disc. active | Val. active | Disc. all-case mean rescue | Val. all-case mean rescue | Val. active-only mean rescue |
|---|---|---|---|---|---|
| E000 | 9/128 | 12/128 | +0.011 | +0.016 | +0.167 |
| E001 | 2/128 | 2/128 | +0.001 | +0.000 | +0.000 |
| E002 | 59/128 | 71/128 | +0.225 | +0.239 | +0.430 |
| E003 | 10/128 | 9/128 | +0.002 | +0.005 | +0.066 |
| E004 | 44/128 | 35/128 | +0.087 | +0.047 | +0.174 |
| E005 | 10/128 | 14/128 | +0.015 | +0.031 | +0.281 |
| E006 | 91/128 | 83/128 | +0.071 | +0.073 | +0.113 |
| E007 | 31/128 | 30/128 | +0.048 | +0.046 | +0.196 |

Recurrence-first selection (>= 64 of 128 discovery cases, highest all-case mean rescue): **E006** (1 candidates; paper: E006).

**Mixtral without BOS (add_special_tokens=False), paper case IDs: validation results (Tables 1, 11 analogues)**

| Expert | Val. active | Expert rescue (all-case) | Active-random control | Spec | Spec sign-flip p | Anchor-active n | Raw active-pair spec | Selected equal-norm | Other equal-norm | Equal-norm spec |
|---|---|---|---|---|---|---|---|---|---|---|
| L19E006 (selected) (paper's expert) | 83/128 | +0.073 [+0.007, +0.140] | +0.244 [+0.173, +0.321] | -0.171 [-0.262, -0.082] | 0.0001 | 83 | -0.103 [-0.223, +0.011] | +0.090 [+0.046, +0.135] | +0.132 [+0.063, +0.201] | -0.042 [-0.109, +0.027] |
| Paper L19E006 | 83/128 | +0.099 [+0.018, +0.175] |  | -0.175 [-0.284, -0.072] |  | 83 | -0.081 [-0.214, +0.042] | +0.046 [-0.007, +0.096] | +0.108 [+0.047, +0.172] | -0.062 [-0.130, -0.003] |

**Mixtral without BOS (add_special_tokens=False), paper case IDs: coalition patching (Table 16 analogue)**

| Patch | Rescue (ours) | Paper |
|---|---|---|
| Clean top-2 coalition | +0.431 [+0.315, +0.547] | +0.461 [+0.343, +0.572] |
| Routing-union coalition | +0.454 [+0.335, +0.570] | +0.490 [+0.367, +0.613] |
| L19 MoE-block patch (same pass) | +0.454 [+0.335, +0.570] | +0.457 [+0.331, +0.579] |

**Mixtral without BOS (add_special_tokens=False), paper case IDs: relation-wise validation for L19E006 (Table 5 analogue)**

| Relation | n | E006 rescue | E006 spec | Pos. frac. | Paper rescue | Paper spec | Paper pos. |
|---|---|---|---|---|---|---|---|
| P103 | 12 | +0.095 | +0.034 | 0.50 | +0.312 | +0.206 | 0.67 |
| P17 | 11 | +0.220 | -0.220 | 0.36 | +0.187 | -0.272 | 0.09 |
| P495 | 9 | +0.104 | -0.250 | 0.44 | +0.094 | -0.358 | 0.11 |
| P136 | 9 | +0.111 | -0.396 | 0.22 | +0.111 | -0.562 | 0.11 |
| P1412 | 9 | +0.273 | +0.002 | 0.89 | +0.031 | -0.076 | 0.33 |
| P178 | 9 | +0.000 | -0.375 | 0.00 | +0.000 | -0.368 | 0.00 |
| P740 | 9 | -0.104 | -0.243 | 0.11 | +0.146 | -0.167 | 0.56 |
| P27 | 9 | -0.007 | -0.250 | 0.33 | +0.049 | -0.132 | 0.22 |
| P106 | 7 | +0.078 | +0.065 | 0.57 | +0.223 | +0.179 | 0.71 |
| P131 | 6 | +0.151 | -0.510 | 0.33 | -0.016 | -0.557 | 0.00 |

**Mixtral without BOS (add_special_tokens=False), paper case IDs: selection stability (Appendix D analogue)**

| Check | Ours | Paper |
|---|---|---|
| Stability grid: E006 selected | 8/25 (selected counts {2.0: 12, 6.0: 8}) | (Qwen3 only in the paper) |
| Stability grid: mean val rescue / spec | +0.171 / -0.055 |  |
| Relation-held-out: E006 selected | 2/5 folds; active 131/256 |  |
| Relation-held-out: rescue / spec | +0.123 [+0.081, +0.168] / -0.117 [-0.188, -0.047] |  |
