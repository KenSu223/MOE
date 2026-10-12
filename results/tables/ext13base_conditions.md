**Phase-5 conditions: split sizes and strata (data/wino_ref/case_sets_p5_<cond>_<proto>.json)**

| Condition | Model | Split | Pairs | Names | Objects | Associative | One-sided | Reverse | Neutral | No probe | Local assoc | Seeds | Flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1 | Qwen3-30B-A3B-Base | discovery | 128 | 62 | 66 | 0 | 0 | 0 | 128 | 0 |  |  |  |
| f1 | Qwen3-30B-A3B-Base | validation | 128 | 70 | 58 | 0 | 0 | 0 | 128 | 0 |  |  |  |
| f1 | Qwen3-30B-A3B-Base | rep_discovery | 128 | 67 | 61 | 0 | 0 | 0 | 128 | 0 |  |  |  |
| f1 | Qwen3-30B-A3B-Base | rep_validation | 128 | 60 | 68 | 0 | 0 | 0 | 128 | 0 |  |  |  |
| f1 | Mixtral-8x7B, BOS | discovery | 101 | 11 | 90 | 0 | 0 | 0 | 101 | 0 |  |  |  |
| f1 | Mixtral-8x7B, BOS | validation | 101 | 12 | 89 | 0 | 0 | 0 | 101 | 0 |  |  |  |
| dir_f1 | Qwen3-30B-A3B-Base | discovery | 90 | 8 | 82 | 6 | 26 | 19 | 39 | 0 | 21 |  |  |
| dir_f1 | Qwen3-30B-A3B-Base | validation | 90 | 6 | 84 | 4 | 16 | 20 | 50 | 0 | 7 |  |  |
| dir_f1 | Mixtral-8x7B, BOS | discovery | 94 | 0 | 94 | 5 | 21 | 27 | 41 | 0 | 19 |  |  |
| dir_f1 | Mixtral-8x7B, BOS | validation | 94 | 1 | 93 | 2 | 20 | 27 | 45 | 0 | 20 |  |  |
| dir_opt | Qwen3-30B-A3B-Base | discovery | 90 | 8 | 82 | 6 | 26 | 19 | 39 | 0 | 21 |  |  |
| dir_opt | Qwen3-30B-A3B-Base | validation | 90 | 6 | 84 | 4 | 16 | 20 | 50 | 0 | 7 |  |  |
| dir_opt | Mixtral-8x7B, BOS | discovery | 94 | 0 | 94 | 5 | 21 | 27 | 41 | 0 | 19 |  |  |
| dir_opt | Mixtral-8x7B, BOS | validation | 94 | 1 | 93 | 2 | 20 | 27 | 45 | 0 | 20 |  |  |
| cp_nat | Qwen3-30B-A3B-Base | discovery | 146 | 12 | 134 | 14 | 26 | 34 | 64 | 8 | 22 |  |  |
| cp_nat | Qwen3-30B-A3B-Base | validation | 140 | 10 | 130 | 8 | 14 | 56 | 60 | 2 | 12 |  |  |
| cp_nat | Qwen3-30B-A3B-Base | rep_discovery | 162 | 2 | 160 | 2 | 32 | 48 | 70 | 10 | 20 |  |  |
| cp_nat | Qwen3-30B-A3B-Base | rep_validation | 158 | 14 | 144 | 4 | 26 | 60 | 64 | 4 | 24 |  |  |
| cp_nat | Mixtral-8x7B, BOS | discovery | 216 | 20 | 196 | 0 | 48 | 54 | 94 | 20 | 22 |  |  |
| cp_nat | Mixtral-8x7B, BOS | validation | 204 | 16 | 188 | 6 | 36 | 70 | 78 | 14 | 22 |  |  |
| cp_nat | Mixtral-8x7B, BOS | rep_discovery | 214 | 10 | 204 | 0 | 32 | 50 | 112 | 20 | 30 |  |  |
| cp_nat | Mixtral-8x7B, BOS | rep_validation | 222 | 20 | 202 | 4 | 34 | 68 | 106 | 10 | 38 |  |  |
| cp_vt_orig | Qwen3-30B-A3B-Base | discovery | 128 | 128 | 0 | 0 | 14 | 24 | 90 | 0 |  | 14 |  |
| cp_vt_orig | Qwen3-30B-A3B-Base | validation | 128 | 128 | 0 | 0 | 8 | 19 | 101 | 0 |  | 13 |  |
| cp_vt_orig | Mixtral-8x7B, BOS | discovery | 128 | 128 | 0 | 0 | 16 | 10 | 102 | 0 |  | 9 |  |
| cp_vt_orig | Mixtral-8x7B, BOS | validation | 128 | 128 | 0 | 0 | 8 | 4 | 116 | 0 |  | 10 |  |
| cp_vt_neu | Qwen3-30B-A3B-Base | discovery | 128 | 128 | 0 | 0 | 14 | 20 | 94 | 0 |  | 12 |  |
| cp_vt_neu | Qwen3-30B-A3B-Base | validation | 128 | 128 | 0 | 0 | 8 | 28 | 92 | 0 |  | 8 |  |
| cp_vt_neu | Mixtral-8x7B, BOS | discovery | 128 | 128 | 0 | 0 | 2 | 6 | 120 | 0 |  | 8 |  |
| cp_vt_neu | Mixtral-8x7B, BOS | validation | 128 | 128 | 0 | 0 | 8 | 6 | 114 | 0 |  | 8 |  |
| vt_e2a_orig | Qwen3-30B-A3B-Base | discovery | 128 | 128 | 0 | 0 | 7 | 19 | 102 | 0 |  | 16 |  |
| vt_e2a_orig | Qwen3-30B-A3B-Base | validation | 128 | 128 | 0 | 0 | 7 | 12 | 109 | 0 |  | 17 |  |
| vt_e2a_orig | Mixtral-8x7B, BOS | discovery | 128 | 128 | 0 | 0 | 8 | 6 | 114 | 0 |  | 14 |  |
| vt_e2a_orig | Mixtral-8x7B, BOS | validation | 128 | 128 | 0 | 0 | 4 | 7 | 117 | 0 |  | 18 |  |
| vt_e2a_neu | Qwen3-30B-A3B-Base | discovery | 128 | 128 | 0 | 0 | 9 | 22 | 97 | 0 |  | 16 |  |
| vt_e2a_neu | Qwen3-30B-A3B-Base | validation | 128 | 128 | 0 | 0 | 10 | 18 | 100 | 0 |  | 12 |  |
| vt_e2a_neu | Mixtral-8x7B, BOS | discovery | 128 | 128 | 0 | 0 | 7 | 4 | 117 | 0 |  | 11 |  |
| vt_e2a_neu | Mixtral-8x7B, BOS | validation | 128 | 128 | 0 | 0 | 9 | 8 | 111 | 0 |  | 12 |  |
| vt_a2e_orig | Qwen3-30B-A3B-Base | discovery | 60 | 60 | 0 | 0 | 4 | 5 | 51 | 0 |  | 8 |  |
| vt_a2e_orig | Qwen3-30B-A3B-Base | validation | 37 | 37 | 0 | 0 | 4 | 7 | 26 | 0 |  | 5 |  |
| vt_a2e_orig | Mixtral-8x7B, BOS | discovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | 0 | behaviour only |
| vt_a2e_orig | Mixtral-8x7B, BOS | validation | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 0 |  | 1 | behaviour only |
| vt_a2e_neu | Qwen3-30B-A3B-Base | discovery | 45 | 45 | 0 | 0 | 0 | 7 | 38 | 0 |  | 5 | behaviour only |
| vt_a2e_neu | Qwen3-30B-A3B-Base | validation | 5 | 5 | 0 | 0 | 1 | 2 | 2 | 0 |  | 3 | behaviour only |
| vt_a2e_neu | Mixtral-8x7B, BOS | discovery | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 0 |  | 1 | behaviour only |
| vt_a2e_neu | Mixtral-8x7B, BOS | validation | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 0 |  | 2 | behaviour only |
| f2e | Qwen3-30B-A3B-Base | discovery | 32 | 3 | 29 | 0 | 0 | 0 | 32 | 0 |  |  |  |
| f2e | Qwen3-30B-A3B-Base | validation | 32 | 6 | 26 | 0 | 0 | 0 | 32 | 0 |  |  |  |
| f2e | Mixtral-8x7B, BOS | discovery | 18 | 3 | 15 | 0 | 0 | 0 | 18 | 0 |  |  |  |
| f2e | Mixtral-8x7B, BOS | validation | 18 | 2 | 16 | 0 | 0 | 0 | 18 | 0 |  |  |  |
