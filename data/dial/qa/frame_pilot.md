# FRAME PILOT (p6-data, E1 evidence, attribute -> entity; behaviour only)

200 core twins, the same for every frame and both models, both statement orders (pairs = items x 2). Neutral = Phase-5 association probes |v| < 1 for both triggers (Qwen3 v1 and v2, Mixtral v2). Frames: P0 'Fill in the blank: {A} and {B}, so the _ was X.\nAnswer: the'; P1 'Fill in the blank: {A}. {B}. The _ was X.\nAnswer: the'; P2 'Fill in the blank: {A} and {B}. The _ was X.\nAnswer: the'; P3 '{A} and {B}. Fill in the blank: The _ was X.\nAnswer: the'; P4 '{A} and {B}. The one that was X was the' (no wrapper, reference). First-mention bias = mean logit(first-mentioned entity) - logit(second) over all prompts (answers balanced).

| Model | Frame | Stratum | Items | Margin both ways | Correct both directions | Items both orders margin | Order gap (o1 - o2) | First-mention bias | Median drop |
|---|---|---|---|---|---|---|---|---|---|
| qwen3 | P0 | objects | 174 | 0.10 | 0.40 | 0.03 | +0.01 | +0.60 | 1.50 |
| qwen3 | P0 | objects (neutral) | 152 | 0.08 | 0.39 | 0.01 | +0.00 | +0.65 | 1.38 |
| qwen3 | P0 | names | 26 | 0.62 | 0.88 | 0.46 | +0.00 | +0.15 | 3.56 |
| qwen3 | P0 | names (neutral) | 26 | 0.62 | 0.88 | 0.46 | +0.00 | +0.15 | 3.56 |
| qwen3 | P0 | all | 200 | 0.17 | 0.47 | 0.09 | +0.01 | +0.54 | 1.62 |
| qwen3 | P0 | all (neutral) | 178 | 0.16 | 0.47 | 0.07 | +0.00 | +0.58 | 1.62 |
| qwen3 | P1 | objects | 174 | 0.09 | 0.46 | 0.01 | +0.02 | +0.03 | 1.25 |
| qwen3 | P1 | objects (neutral) | 152 | 0.07 | 0.46 | 0.00 | +0.01 | +0.07 | 1.12 |
| qwen3 | P1 | names | 26 | 0.46 | 0.75 | 0.23 | +0.00 | -0.08 | 3.31 |
| qwen3 | P1 | names (neutral) | 26 | 0.46 | 0.75 | 0.23 | +0.00 | -0.08 | 3.31 |
| qwen3 | P1 | all | 200 | 0.14 | 0.50 | 0.04 | +0.02 | +0.01 | 1.38 |
| qwen3 | P1 | all (neutral) | 178 | 0.12 | 0.50 | 0.03 | +0.01 | +0.05 | 1.25 |
| qwen3 | P2 | objects | 174 | 0.81 | 0.95 | 0.68 | -0.06 | +0.06 | 4.50 |
| qwen3 | P2 | objects (neutral) | 152 | 0.81 | 0.95 | 0.68 | -0.07 | +0.09 | 4.38 |
| qwen3 | P2 | names | 26 | 0.79 | 0.94 | 0.69 | -0.19 | +0.39 | 5.75 |
| qwen3 | P2 | names (neutral) | 26 | 0.79 | 0.94 | 0.69 | -0.19 | +0.39 | 5.75 |
| qwen3 | P2 | all | 200 | 0.81 | 0.95 | 0.68 | -0.08 | +0.10 | 4.62 |
| qwen3 | P2 | all (neutral) | 178 | 0.81 | 0.95 | 0.69 | -0.09 | +0.13 | 4.50 |
| qwen3 | P3 | objects | 174 | 0.97 | 0.99 | 0.94 | +0.01 | +0.17 | 6.88 |
| qwen3 | P3 | objects (neutral) | 152 | 0.97 | 0.99 | 0.94 | +0.00 | +0.14 | 6.81 |
| qwen3 | P3 | names | 26 | 0.90 | 0.96 | 0.85 | -0.04 | -0.25 | 8.56 |
| qwen3 | P3 | names (neutral) | 26 | 0.90 | 0.96 | 0.85 | -0.04 | -0.25 | 8.56 |
| qwen3 | P3 | all | 200 | 0.96 | 0.98 | 0.93 | +0.00 | +0.11 | 7.06 |
| qwen3 | P3 | all (neutral) | 178 | 0.96 | 0.98 | 0.93 | -0.01 | +0.09 | 7.00 |
| qwen3 | P4 | objects | 174 | 0.49 | 0.81 | 0.33 | -0.02 | -0.13 | 3.50 |
| qwen3 | P4 | objects (neutral) | 152 | 0.47 | 0.81 | 0.30 | -0.04 | -0.16 | 3.38 |
| qwen3 | P4 | names | 26 | 0.38 | 0.54 | 0.19 | -0.08 | +0.41 | 4.31 |
| qwen3 | P4 | names (neutral) | 26 | 0.38 | 0.54 | 0.19 | -0.08 | +0.41 | 4.31 |
| qwen3 | P4 | all | 200 | 0.47 | 0.78 | 0.31 | -0.03 | -0.06 | 3.50 |
| qwen3 | P4 | all (neutral) | 178 | 0.46 | 0.77 | 0.29 | -0.04 | -0.08 | 3.50 |
| mixtral_bos | P0 | objects | 174 | 0.02 | 0.06 | 0.01 | -0.01 | +0.17 | -0.38 |
| mixtral_bos | P0 | objects (neutral) | 153 | 0.01 | 0.05 | 0.00 | -0.02 | +0.19 | -0.38 |
| mixtral_bos | P0 | names | 26 | 0.00 | 0.00 | 0.00 | +0.00 | +0.22 | -0.50 |
| mixtral_bos | P0 | names (neutral) | 25 | 0.00 | 0.00 | 0.00 | +0.00 | +0.25 | -0.50 |
| mixtral_bos | P0 | all | 200 | 0.01 | 0.05 | 0.01 | -0.01 | +0.18 | -0.38 |
| mixtral_bos | P0 | all (neutral) | 178 | 0.01 | 0.04 | 0.00 | -0.02 | +0.20 | -0.38 |
| mixtral_bos | P1 | objects | 174 | 0.01 | 0.06 | 0.01 | +0.01 | +0.11 | -0.44 |
| mixtral_bos | P1 | objects (neutral) | 153 | 0.02 | 0.06 | 0.01 | +0.01 | +0.14 | -0.50 |
| mixtral_bos | P1 | names | 26 | 0.00 | 0.04 | 0.00 | +0.00 | -0.01 | -0.50 |
| mixtral_bos | P1 | names (neutral) | 25 | 0.00 | 0.04 | 0.00 | +0.00 | +0.00 | -0.50 |
| mixtral_bos | P1 | all | 200 | 0.01 | 0.06 | 0.01 | +0.00 | +0.09 | -0.50 |
| mixtral_bos | P1 | all (neutral) | 178 | 0.01 | 0.06 | 0.01 | +0.01 | +0.12 | -0.50 |
| mixtral_bos | P2 | objects | 174 | 0.03 | 0.23 | 0.01 | -0.01 | -0.20 | 0.38 |
| mixtral_bos | P2 | objects (neutral) | 153 | 0.02 | 0.22 | 0.01 | -0.01 | -0.20 | 0.25 |
| mixtral_bos | P2 | names | 26 | 0.00 | 0.12 | 0.00 | +0.00 | -0.05 | -0.25 |
| mixtral_bos | P2 | names (neutral) | 25 | 0.00 | 0.12 | 0.00 | +0.00 | -0.04 | -0.19 |
| mixtral_bos | P2 | all | 200 | 0.03 | 0.21 | 0.01 | -0.01 | -0.18 | 0.25 |
| mixtral_bos | P2 | all (neutral) | 178 | 0.02 | 0.21 | 0.01 | -0.01 | -0.17 | 0.25 |
| mixtral_bos | P3 | objects | 174 | 0.24 | 0.79 | 0.14 | -0.01 | +0.12 | 2.00 |
| mixtral_bos | P3 | objects (neutral) | 153 | 0.24 | 0.80 | 0.14 | -0.01 | +0.12 | 2.00 |
| mixtral_bos | P3 | names | 26 | 0.25 | 0.62 | 0.15 | -0.04 | +0.23 | 1.50 |
| mixtral_bos | P3 | names (neutral) | 25 | 0.26 | 0.62 | 0.16 | -0.04 | +0.26 | 1.50 |
| mixtral_bos | P3 | all | 200 | 0.24 | 0.77 | 0.14 | -0.01 | +0.13 | 2.00 |
| mixtral_bos | P3 | all (neutral) | 178 | 0.24 | 0.77 | 0.14 | -0.02 | +0.14 | 1.88 |
| mixtral_bos | P4 | objects | 174 | 0.32 | 0.63 | 0.18 | +0.07 | -0.46 | 2.25 |
| mixtral_bos | P4 | objects (neutral) | 153 | 0.31 | 0.62 | 0.17 | +0.05 | -0.47 | 2.25 |
| mixtral_bos | P4 | names | 26 | 0.10 | 0.48 | 0.04 | -0.04 | +0.29 | 1.38 |
| mixtral_bos | P4 | names (neutral) | 25 | 0.10 | 0.50 | 0.04 | -0.04 | +0.31 | 1.44 |
| mixtral_bos | P4 | all | 200 | 0.29 | 0.61 | 0.16 | +0.05 | -0.37 | 2.12 |
| mixtral_bos | P4 | all (neutral) | 178 | 0.28 | 0.60 | 0.15 | +0.04 | -0.36 | 2.12 |

## Decision rule (coordinator, pre-registered)

Highest neutral-objects correct-both among P0-P3 with |order gap| <= 10 pp; if none > 60 % in a model, keep P0.

- qwen3: {"frame": "P3", "neutral_objects_correct_both": 0.9868421052631579, "order_gap": 0.0}
- mixtral_bos: {"frame": "P3", "neutral_objects_correct_both": 0.7973856209150327, "order_gap": 0.01307189542483661}
