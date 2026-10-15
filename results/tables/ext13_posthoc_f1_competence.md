**Competence by question direction, item type and pool (chance for 'correct in both twins' = 0.25 if the trigger is ignored and each twin is guessed independently, 0 if the same name is always chosen)**

| Model | Items | Pairs | Correct in both twins | Pass margin | Mean drop (logits) | Same name in both twins | Prefers first-mentioned name |
|---|---|---|---|---|---|---|---|
| Qwen3 | all twins · objects | 4764 | 0.40 | 0.20 | 2.66 |  |  |
| Qwen3 | all twins · names | 5472 | 0.31 | 0.10 | 1.55 |  |  |
| Qwen3 | Phase-3 776 · objects | 676 | 0.48 | 0.25 | 3.17 |  |  |
| Qwen3 | Phase-3 776 · names | 73 | 0.45 | 0.19 | 2.20 |  |  |
| Qwen3 | name templates, ask entity · neutral | 2640 | 0.22 | 0.02 | 0.72 | 0.71 | 0.66 |
| Qwen3 | name templates, ask entity · orig | 2640 | 0.27 | 0.04 | 0.97 | 0.68 | 0.65 |
| Qwen3 | name templates, ask entity · unrelated | 2640 | 0.05 | 0.00 | 0.01 | 0.85 | 0.71 |
| Qwen3 | name templates, ask attribute · neutral | 2640 | 0.42 | 0.27 | 4.25 |  |  |
| Qwen3 | name templates, ask attribute · orig | 2640 | 0.53 | 0.39 | 4.97 |  |  |
| Qwen3 | name templates, ask attribute · unrelated | 2640 | 0.16 | 0.05 | 0.22 |  |  |
| Mixtral BOS | all twins · objects | 2802 | 0.36 | 0.15 | 1.82 |  |  |
| Mixtral BOS | all twins · names | 2081 | 0.17 | 0.01 | 0.39 |  |  |
| Mixtral BOS | Phase-3 776 · objects | 657 | 0.54 | 0.29 | 2.88 |  |  |
| Mixtral BOS | Phase-3 776 · names | 61 | 0.23 | 0.02 | 0.74 |  |  |
| Mixtral BOS | name templates, ask entity · neutral | 2640 | 0.20 | 0.00 | 0.37 | 0.72 | 0.51 |
| Mixtral BOS | name templates, ask entity · orig | 2640 | 0.23 | 0.00 | 0.46 | 0.69 | 0.44 |
| Mixtral BOS | name templates, ask entity · unrelated | 2640 | 0.06 | 0.00 | -0.02 | 0.81 | 0.61 |
| Mixtral BOS | name templates, ask attribute · neutral | 2640 | 0.39 | 0.21 | 3.91 |  |  |
| Mixtral BOS | name templates, ask attribute · orig | 2640 | 0.51 | 0.33 | 4.71 |  |  |
| Mixtral BOS | name templates, ask attribute · unrelated | 2640 | 0.08 | 0.02 | 0.00 |  |  |
