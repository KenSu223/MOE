**Qwen3-30B-A3B-Base (tokenizer defaults): union of the top-10 recurrent pairs under each metric, with their rank under every metric (among 234 recurrent pairs)**

| Pair | Val. active | Rescue | Active-only | Spec | Block / share | Percentile | rk rescue | rk active-only | rk Spec | rk share | rk percentile | rk disc. | Spread | Best under | Worst under |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L44E069 | 116/128 | +0.499 | +0.550 | +0.443 | +0.941 / 53% | 79% | 1 | 1 | 1 | 7 | 11 | 1 | 10 | val_rescue | mean_percentile |
| L42E115 | 123/128 | +0.447 | +0.465 | +0.423 | +0.621 / 72% | 87% | 2 | 2 | 2 | 2 | 1 | 2 | 1 | mean_percentile | val_rescue |
| L43E005 | 82/128 | +0.146 | +0.229 | +0.087 | +0.606 / 24% | 73% | 3 | 3 | 4 | 22 | 67 | 5 | 64 | val_rescue | mean_percentile |
| L40E127 | 93/128 | +0.126 | +0.174 | +0.083 | +0.444 / 28% | 76% | 4 | 4 | 6 | 17 | 27 | 6 | 23 | val_rescue | mean_percentile |
| L41E001 | 109/128 | +0.125 | +0.147 | +0.106 | +0.289 / 43% | 79% | 5 | 7 | 3 | 12 | 9 | 4 | 9 | val_spec | block_share |
| L40E030 | 92/128 | +0.121 | +0.168 | +0.086 | +0.444 / 27% | 75% | 6 | 5 | 5 | 19 | 28 | 7 | 23 | val_active_only | mean_percentile |
| L43E104 | 87/128 | +0.102 | +0.149 | +0.041 | +0.606 / 17% | 72% | 7 | 6 | 13 | 29 | 77 | 8 | 71 | val_active_only | mean_percentile |
| L43E046 | 87/128 | +0.095 | +0.140 | +0.017 | +0.606 / 16% | 67% | 8 | 8 | 29 | 30 | 168 | 3 | 165 | disc_allcase | mean_percentile |
| L28E030 | 115/128 | +0.082 | +0.091 | +0.071 | +0.143 / 57% | 80% | 9 | 10 | 8 | 5 | 5 | 12 | 7 | block_share | disc_allcase |
| L33E076 | 102/128 | +0.079 | +0.099 | +0.083 | +0.098 / 81% | 85% | 10 | 9 | 7 | 1 | 2 | 9 | 9 | block_share | val_rescue |
| L28E127 | 99/128 | +0.052 | +0.068 | +0.034 | +0.143 / 37% | 79% | 11 | 12 | 18 | 15 | 10 | 17 | 8 | mean_percentile | val_spec |
| L38E038 | 107/128 | +0.052 | +0.062 | +0.043 | +0.111 / 47% | 78% | 11 | 15 | 12 | 9 | 15 | 16 | 7 | block_share | disc_allcase |
| L38E057 | 94/128 | +0.048 | +0.066 | +0.036 | +0.111 / 44% | 84% | 14 | 13 | 17 | 11 | 3 | 14 | 14 | mean_percentile | val_spec |
| L39E040 | 121/128 | +0.045 | +0.048 | +0.040 | +0.079 / 58% | 73% | 15 | 17 | 14 | 4 | 52 | 15 | 48 | block_share | mean_percentile |
| L37E104 | 127/128 | +0.039 | +0.039 | +0.038 | +0.073 / 53% | 69% | 16 | 22 | 15 | 6 | 150 | 10 | 144 | block_share | mean_percentile |
| L47E032 | 110/128 | +0.035 | +0.041 | +0.059 | -0.128 / n/a | 76% | 18 | 20 | 10 | n/a | 22 | 19 | 12 | val_spec | mean_percentile |
| L34E094 | 107/128 | +0.034 | +0.040 | +0.032 | +0.058 / 58% | 79% | 19 | 21 | 19 | 3 | 8 | 11 | 18 | block_share | val_active_only |
| L47E034 | 109/128 | +0.032 | +0.038 | +0.061 | -0.128 / n/a | 71% | 21 | 24 | 9 | n/a | 103 | 22 | 94 | val_spec | mean_percentile |
| L12E034 | 77/128 | +0.017 | +0.028 | +0.002 | +0.035 / 47% | 73% | 35 | 35 | 80 | 8 | 55 | 216 | 208 | block_share | disc_allcase |
| L12E069 | 67/128 | +0.016 | +0.031 | -0.003 | +0.035 / 46% | 75% | 37 | 29 | 121 | 10 | 33 | 203 | 193 | block_share | disc_allcase |
| L22E091 | 68/128 | +0.015 | +0.028 | +0.021 | -0.004 / n/a | 81% | 42 | 36 | 24 | n/a | 4 | 131 | 127 | mean_percentile | disc_allcase |
| L23E058 | 69/128 | +0.011 | +0.021 | +0.011 | -0.005 / n/a | 79% | 52 | 46 | 38 | n/a | 7 | 93 | 86 | mean_percentile | disc_allcase |
| L26E101 | 55/128 | +0.003 | +0.008 | +0.001 | +0.010 / n/a | 80% | 119 | 106 | 84 | n/a | 6 | 167 | 161 | mean_percentile | disc_allcase |
