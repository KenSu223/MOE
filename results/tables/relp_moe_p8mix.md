# Add-back under five patch semantics at the patched MoE sites (SVA, held-out pairs)

Cells: {'mixtral': 4, 'qwen3': 4} tasks per model. Values: ratio of means over the held-out pairs; mean of the tasks; random = mean of 3 seeds.
Rescue = (F_patched - F_B) / (F_A - F_B) on the B input (denoise; ideal 1); disruption = (F_A - F_patched) / (F_A - F_B) on the A input (noise; ideal 1). These are the swap measures (named rescue / disruption since 2026-11-16; faithfulness / completeness are reserved for the RelP paper's keep / remove). Modes: full = full swap (S2.1); content_both = content only, both-routed; content_all = content only, all live-routed; renorm = full swap, renormalised; route = routing decision moved too.

## Qwen3-30B-A3B: rescue

| ranking | mode | K=1 | K=5 | K=10 | K=20 | K=50 | K=100 |
|---|---|---|---|---|---|---|---|
| relp | full | 0.21 | 0.45 | 0.64 | 0.78 | 0.96 | 0.99 |
| relp | content_both | 0.21 | 0.45 | 0.60 | 0.69 | 0.80 | 0.95 |
| relp | content_all | 0.21 | 0.45 | 0.61 | 0.74 | 0.91 | 0.97 |
| relp | renorm | 0.21 | 0.45 | 0.64 | 0.78 | 0.95 | 0.99 |
| relp | route | 0.21 | 0.45 | 0.65 | 0.80 | 0.96 | 0.99 |
| exact | full | 0.21 | 0.45 | 0.64 | 0.79 | 0.88 | 0.91 |
| exact | content_both | 0.21 | 0.45 | 0.59 | 0.69 | 0.76 | 0.79 |
| exact | content_all | 0.21 | 0.45 | 0.60 | 0.74 | 0.85 | 0.88 |
| exact | renorm | 0.21 | 0.45 | 0.64 | 0.79 | 0.87 | 0.91 |
| exact | route | 0.21 | 0.46 | 0.64 | 0.80 | 0.89 | 0.92 |
| lu | full | 0.16 | 0.25 | 0.45 | 0.50 | 0.62 | 0.77 |
| lu | content_both | 0.16 | 0.25 | 0.43 | 0.48 | 0.58 | 0.72 |
| lu | content_all | 0.16 | 0.25 | 0.45 | 0.49 | 0.62 | 0.76 |
| lu | renorm | 0.16 | 0.25 | 0.47 | 0.50 | 0.62 | 0.77 |
| lu | route | 0.16 | 0.25 | 0.46 | 0.50 | 0.62 | 0.77 |
| random | full | 0.03 | 0.05 | 0.04 | 0.12 | 0.13 | 0.25 |
| random | content_both | 0.03 | 0.05 | 0.04 | 0.11 | 0.10 | 0.22 |
| random | content_all | 0.03 | 0.05 | 0.05 | 0.12 | 0.13 | 0.24 |
| random | renorm | 0.03 | 0.05 | 0.04 | 0.12 | 0.13 | 0.25 |
| random | route | 0.03 | 0.06 | 0.08 | 0.18 | 0.21 | 0.41 |

## Qwen3-30B-A3B: disruption

| ranking | mode | K=1 | K=5 | K=10 | K=20 | K=50 | K=100 |
|---|---|---|---|---|---|---|---|
| relp | full | 0.22 | 0.45 | 0.64 | 0.79 | 0.95 | 0.99 |
| relp | content_both | 0.22 | 0.45 | 0.60 | 0.69 | 0.80 | 0.95 |
| relp | content_all | 0.22 | 0.45 | 0.61 | 0.74 | 0.91 | 0.98 |
| relp | renorm | 0.22 | 0.45 | 0.65 | 0.79 | 0.95 | 0.99 |
| relp | route | 0.22 | 0.45 | 0.65 | 0.80 | 0.96 | 0.99 |
| exact | full | 0.22 | 0.45 | 0.64 | 0.79 | 0.88 | 0.91 |
| exact | content_both | 0.22 | 0.45 | 0.59 | 0.70 | 0.76 | 0.78 |
| exact | content_all | 0.22 | 0.45 | 0.61 | 0.74 | 0.85 | 0.88 |
| exact | renorm | 0.22 | 0.45 | 0.64 | 0.79 | 0.88 | 0.91 |
| exact | route | 0.22 | 0.45 | 0.65 | 0.80 | 0.89 | 0.92 |
| lu | full | 0.16 | 0.26 | 0.44 | 0.46 | 0.60 | 0.76 |
| lu | content_both | 0.17 | 0.26 | 0.42 | 0.43 | 0.56 | 0.71 |
| lu | content_all | 0.17 | 0.26 | 0.43 | 0.45 | 0.59 | 0.76 |
| lu | renorm | 0.16 | 0.26 | 0.43 | 0.46 | 0.60 | 0.76 |
| lu | route | 0.16 | 0.26 | 0.43 | 0.46 | 0.60 | 0.76 |
| random | full | 0.03 | 0.05 | 0.04 | 0.12 | 0.13 | 0.25 |
| random | content_both | 0.03 | 0.05 | 0.04 | 0.11 | 0.10 | 0.22 |
| random | content_all | 0.03 | 0.05 | 0.04 | 0.12 | 0.13 | 0.24 |
| random | renorm | 0.03 | 0.05 | 0.04 | 0.12 | 0.13 | 0.25 |
| random | route | 0.03 | 0.06 | 0.08 | 0.18 | 0.21 | 0.39 |

### Qwen3-30B-A3B: every unit of the pool (ceiling)

| mode | rescue | disruption |
|---|---|---|
| content_all | 0.99 | 0.99 |
| content_both | 0.98 | 0.98 |
| full | 1.00 | 1.00 |
| renorm | 1.00 | 1.00 |
| route | 1.00 | 1.00 |

### Qwen3-30B-A3B: RelP ranking minus activation-patching ranking (R = rescue, D = disruption; positive = RelP better)

| mode | R K=5 | D K=5 | R K=10 | D K=10 | R K=20 | D K=20 | R K=50 | D K=50 | R K=100 | D K=100 |
|---|---|---|---|---|---|---|---|---|---|---|
| full | -0.00 | 0.00 | 0.01 | 0.01 | -0.00 | -0.00 | 0.08 | 0.07 | 0.08 | 0.08 |
| content_both | -0.00 | -0.00 | 0.01 | 0.01 | -0.00 | -0.00 | 0.05 | 0.04 | 0.16 | 0.17 |
| content_all | -0.00 | -0.00 | 0.01 | 0.01 | 0.00 | -0.00 | 0.06 | 0.06 | 0.09 | 0.10 |
| renorm | -0.00 | 0.00 | 0.01 | 0.01 | -0.00 | -0.00 | 0.08 | 0.08 | 0.09 | 0.09 |
| route | -0.00 | -0.00 | 0.01 | 0.01 | -0.00 | -0.00 | 0.07 | 0.07 | 0.07 | 0.07 |

## Mixtral-8x7B: rescue

| ranking | mode | K=1 | K=5 | K=10 | K=20 | K=50 | K=100 |
|---|---|---|---|---|---|---|---|
| relp | full | 0.14 | 0.40 | 0.70 | 0.94 | 0.99 | 1.00 |
| relp | content_both | 0.13 | 0.39 | 0.69 | 0.94 | 0.99 | 1.00 |
| relp | content_all | 0.13 | 0.40 | 0.70 | 0.94 | 0.99 | 1.00 |
| relp | renorm | 0.13 | 0.40 | 0.70 | 0.94 | 0.99 | 1.00 |
| relp | route | 0.14 | 0.40 | 0.70 | 0.94 | 0.99 | 1.00 |
| exact | full | 0.96 | 0.97 | 0.98 | 0.98 | 1.00 | 1.00 |
| exact | content_both | 0.96 | 0.97 | 0.98 | 0.98 | 1.00 | 1.00 |
| exact | content_all | 0.96 | 0.97 | 0.98 | 0.98 | 1.00 | 1.00 |
| exact | renorm | 0.96 | 0.97 | 0.98 | 0.98 | 1.00 | 1.00 |
| exact | route | 0.96 | 0.97 | 0.98 | 0.99 | 1.00 | 1.00 |
| lu | full | 0.54 | 0.59 | 0.65 | 0.70 | 0.78 | 0.82 |
| lu | content_both | 0.53 | 0.57 | 0.63 | 0.68 | 0.75 | 0.81 |
| lu | content_all | 0.53 | 0.58 | 0.65 | 0.69 | 0.78 | 0.82 |
| lu | renorm | 0.53 | 0.59 | 0.65 | 0.70 | 0.78 | 0.82 |
| lu | route | 0.54 | 0.59 | 0.65 | 0.70 | 0.78 | 0.82 |
| random | full | 0.03 | 0.12 | 0.30 | 0.31 | 0.86 | 0.96 |
| random | content_both | 0.03 | 0.10 | 0.27 | 0.27 | 0.84 | 0.95 |
| random | content_all | 0.03 | 0.12 | 0.30 | 0.31 | 0.86 | 0.96 |
| random | renorm | 0.03 | 0.11 | 0.29 | 0.31 | 0.86 | 0.96 |
| random | route | 0.03 | 0.12 | 0.30 | 0.32 | 0.86 | 0.96 |

## Mixtral-8x7B: disruption

| ranking | mode | K=1 | K=5 | K=10 | K=20 | K=50 | K=100 |
|---|---|---|---|---|---|---|---|
| relp | full | 0.13 | 0.40 | 0.69 | 0.93 | 0.99 | 1.00 |
| relp | content_both | 0.13 | 0.38 | 0.68 | 0.93 | 0.99 | 1.00 |
| relp | content_all | 0.13 | 0.39 | 0.69 | 0.94 | 0.99 | 1.00 |
| relp | renorm | 0.13 | 0.40 | 0.69 | 0.94 | 0.99 | 1.00 |
| relp | route | 0.13 | 0.40 | 0.69 | 0.93 | 0.99 | 1.00 |
| exact | full | 0.95 | 0.97 | 0.97 | 0.98 | 1.00 | 1.00 |
| exact | content_both | 0.94 | 0.96 | 0.97 | 0.98 | 1.00 | 1.00 |
| exact | content_all | 0.94 | 0.96 | 0.97 | 0.98 | 1.00 | 1.00 |
| exact | renorm | 0.94 | 0.96 | 0.97 | 0.98 | 1.00 | 1.00 |
| exact | route | 0.95 | 0.97 | 0.98 | 0.99 | 1.00 | 1.00 |
| lu | full | 0.53 | 0.58 | 0.64 | 0.69 | 0.78 | 0.82 |
| lu | content_both | 0.51 | 0.56 | 0.62 | 0.67 | 0.75 | 0.81 |
| lu | content_all | 0.51 | 0.58 | 0.64 | 0.69 | 0.78 | 0.82 |
| lu | renorm | 0.52 | 0.58 | 0.64 | 0.69 | 0.78 | 0.82 |
| lu | route | 0.53 | 0.58 | 0.64 | 0.69 | 0.78 | 0.82 |
| random | full | 0.03 | 0.12 | 0.30 | 0.31 | 0.86 | 0.96 |
| random | content_both | 0.03 | 0.10 | 0.27 | 0.26 | 0.84 | 0.95 |
| random | content_all | 0.03 | 0.11 | 0.29 | 0.30 | 0.86 | 0.96 |
| random | renorm | 0.03 | 0.11 | 0.29 | 0.30 | 0.86 | 0.96 |
| random | route | 0.03 | 0.12 | 0.30 | 0.32 | 0.87 | 0.96 |

### Mixtral-8x7B: every unit of the pool (ceiling)

| mode | rescue | disruption |
|---|---|---|
| content_all | 1.00 | 1.00 |
| content_both | 1.00 | 1.00 |
| full | 1.00 | 1.00 |
| renorm | 1.00 | 1.00 |
| route | 1.00 | 1.00 |

### Mixtral-8x7B: RelP ranking minus activation-patching ranking (R = rescue, D = disruption; positive = RelP better)

| mode | R K=5 | D K=5 | R K=10 | D K=10 | R K=20 | D K=20 | R K=50 | D K=50 | R K=100 | D K=100 |
|---|---|---|---|---|---|---|---|---|---|---|
| full | -0.57 | -0.57 | -0.28 | -0.28 | -0.05 | -0.05 | -0.00 | -0.00 | 0.00 | -0.00 |
| content_both | -0.59 | -0.58 | -0.29 | -0.29 | -0.05 | -0.05 | -0.00 | -0.00 | 0.00 | -0.00 |
| content_all | -0.57 | -0.57 | -0.28 | -0.28 | -0.04 | -0.04 | -0.00 | -0.00 | 0.00 | 0.00 |
| renorm | -0.57 | -0.57 | -0.28 | -0.28 | -0.04 | -0.05 | -0.00 | -0.00 | -0.00 | -0.00 |
| route | -0.57 | -0.57 | -0.28 | -0.29 | -0.05 | -0.05 | -0.00 | -0.00 | 0.00 | 0.00 |

## Routing-weight sum at the patched sites under the full swap (denoise, held-out pairs)

sites_off = share of patched (pair, layer, position) sites whose weight sum differs from 1 by more than 0.02; p05 / p95 = 5th / 95th percentile over sites (extreme over tasks); injected / removed = share of patched (pair, unit) instances whose expert is routed only in the source run / only in the patched run. The other four modes keep a total of 1 by construction (content_*: the live weights; renorm: rescaled; route: the source run's weights).

| model | ranking | K | sites_off | p05 | p95 | injected | removed |
|---|---|---|---|---|---|---|---|
| mixtral | exact | 5 | 0.68 | 0.65 | 1.33 | 0.03 | 0.05 |
| mixtral | exact | 20 | 0.68 | 0.44 | 1.59 | 0.09 | 0.09 |
| mixtral | exact | 50 | 0.62 | 0.65 | 1.38 | 0.11 | 0.11 |
| mixtral | relp | 5 | 0.75 | 0.54 | 1.45 | 0.05 | 0.07 |
| mixtral | relp | 20 | 0.70 | 0.38 | 1.66 | 0.09 | 0.09 |
| mixtral | relp | 50 | 0.60 | 0.59 | 1.45 | 0.11 | 0.11 |
| qwen3 | exact | 5 | 0.38 | 0.94 | 1.07 | 0.03 | 0.03 |
| qwen3 | exact | 20 | 0.67 | 0.77 | 1.25 | 0.11 | 0.11 |
| qwen3 | exact | 50 | 0.70 | 0.75 | 1.25 | 0.12 | 0.12 |
| qwen3 | relp | 5 | 0.38 | 0.91 | 1.09 | 0.03 | 0.03 |
| qwen3 | relp | 20 | 0.67 | 0.77 | 1.25 | 0.11 | 0.11 |
| qwen3 | relp | 50 | 0.70 | 0.75 | 1.24 | 0.11 | 0.11 |

## Check: the full-swap rows repeat the S2.1 add-back (other batch composition)

| model | task | rows | pearson | median_abs_diff |
|---|---|---|---|---|
| mixtral | nounpp | 8640 | 0.9999 | 0.0157 |
| mixtral | rc | 8640 | 0.9998 | 0.0171 |
| mixtral | simple | 7680 | 1.0000 | 0.0113 |
| mixtral | within_rc | 8480 | 1.0000 | 0.0136 |
| qwen3 | nounpp | 8800 | 0.9992 | 0.0590 |
| qwen3 | rc | 8800 | 0.9992 | 0.0550 |
| qwen3 | simple | 8800 | 0.9997 | 0.0396 |
| qwen3 | within_rc | 8800 | 0.9996 | 0.0444 |

