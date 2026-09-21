**Mixtral-8x7B-v0.1 (BOS, tokenizer default): union of the top-10 recurrent pairs under each metric, with their rank under every metric (among 38 recurrent pairs)**

| Pair | Val. active | Rescue | Active-only | Spec | Block / share | Percentile | rk rescue | rk active-only | rk Spec | rk share | rk percentile | rk disc. | Spread | Best under | Worst under |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L19E002 | 84/128 | +0.363 | +0.553 | +0.192 | +0.580 / 63% | 82% | 1 | 1 | 1 | 2 | 1 | 1 | 1 | val_rescue | block_share |
| L21E001 | 84/128 | +0.273 | +0.417 | +0.118 | +0.500 / 55% | 71% | 2 | 2 | 3 | 4 | 7 | 2 | 5 | val_rescue | mean_percentile |
| L18E001 | 103/128 | +0.244 | +0.303 | +0.182 | +0.315 / 77% | 69% | 3 | 3 | 2 | 1 | 11 | 3 | 10 | block_share | mean_percentile |
| L22E001 | 94/128 | +0.179 | +0.244 | +0.057 | +0.314 / 57% | 66% | 4 | 5 | 7 | 3 | 20 | 4 | 17 | block_share | mean_percentile |
| L20E005 | 73/128 | +0.155 | +0.272 | -0.074 | +0.504 / 31% | 71% | 5 | 4 | 36 | 14 | 8 | 5 | 32 | val_active_only | val_spec |
| L28E002 | 79/128 | +0.086 | +0.140 | +0.005 | +0.189 / 46% | 62% | 6 | 8 | 15 | 7 | 25 | 7 | 19 | val_rescue | mean_percentile |
| L17E001 | 107/128 | +0.082 | +0.098 | +0.001 | +0.155 / 53% | 66% | 7 | 10 | 17 | 5 | 19 | 10 | 14 | block_share | mean_percentile |
| L19E006 | 67/128 | +0.079 | +0.150 | -0.246 | +0.580 / 14% | 57% | 8 | 6 | 38 | 17 | 32 | 6 | 32 | val_active_only | val_spec |
| L25E003 | 67/128 | +0.078 | +0.149 | -0.025 | +0.233 / 34% | 57% | 9 | 7 | 33 | 11 | 32 | 9 | 26 | val_active_only | val_spec |
| L17E005 | 82/128 | +0.066 | +0.103 | -0.021 | +0.155 / 42% | 63% | 10 | 9 | 31 | 8 | 23 | 12 | 23 | block_share | val_spec |
| L15E005 | 76/128 | +0.054 | +0.090 | +0.010 | +0.107 / 50% | 68% | 11 | 11 | 12 | 6 | 14 | 11 | 8 | block_share | mean_percentile |
| L6E007 | 95/128 | +0.031 | +0.042 | +0.041 | +0.038 / n/a | 76% | 15 | 16 | 8 | n/a | 3 | 24 | 21 | mean_percentile | disc_allcase |
| L16E007 | 107/128 | +0.031 | +0.037 | -0.017 | +0.082 / 38% | 61% | 15 | 18 | 28 | 10 | 27 | 19 | 18 | block_share | val_spec |
| L4E002 | 67/128 | +0.020 | +0.037 | +0.024 | +0.027 / n/a | 69% | 19 | 19 | 9 | n/a | 12 | 30 | 21 | val_spec | disc_allcase |
| L14E001 | 98/128 | +0.019 | +0.025 | -0.000 | +0.046 / 41% | 64% | 21 | 21 | 20 | 9 | 21 | 14 | 12 | block_share | val_rescue |
| L9E004 | 75/128 | +0.013 | +0.022 | +0.021 | -0.009 / n/a | 75% | 22 | 22 | 10 | n/a | 5 | 35 | 30 | mean_percentile | disc_allcase |
| L2E006 | 72/128 | +0.007 | +0.013 | +0.015 | -0.021 / n/a | 75% | 23 | 23 | 11 | n/a | 4 | 22 | 19 | mean_percentile | val_rescue |
| L0E004 | 73/128 | +0.004 | +0.007 | +0.004 | -0.008 / n/a | 81% | 29 | 28 | 16 | n/a | 2 | 32 | 30 | mean_percentile | disc_allcase |
| L29E003 | 87/128 | -0.006 | -0.009 | +0.106 | -0.117 / n/a | 70% | 32 | 32 | 4 | n/a | 9 | 37 | 33 | val_spec | disc_allcase |
| L31E007 | 97/128 | -0.029 | -0.038 | +0.096 | -0.140 / n/a | 69% | 35 | 35 | 6 | n/a | 10 | 36 | 30 | val_spec | disc_allcase |
| L30E004 | 85/128 | -0.040 | -0.060 | +0.098 | -0.199 / n/a | 74% | 36 | 36 | 5 | n/a | 6 | 20 | 31 | val_spec | val_rescue |
| L29E004 | 108/128 | -0.094 | -0.112 | -0.063 | -0.117 / n/a | 49% | 38 | 38 | 35 | n/a | 36 | 8 | 30 | disc_allcase | val_rescue |
