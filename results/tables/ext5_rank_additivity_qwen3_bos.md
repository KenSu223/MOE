**Qwen3-30B-A3B-Base (tokenizer defaults): additivity of single-expert rescues on the paper validation split (same pass; the exact end point of the additive curve is the clean-top-k coalition)**

| Layer | Sum of singles | Coalition (clean top-k) | Coalition (union) | Block | r(sum, coalition) | r(sum, block) | mean / median |sum - coalition| | within 0.25 / 0.5 | sum - coalition [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| L44 | +0.935 | +0.916 | +0.941 | +0.941 | 0.93 | 0.92 | 0.356 / 0.281 | 50% / 75% | +0.019 [-0.059, +0.095] |
| L42 | +0.597 | +0.622 | +0.621 | +0.621 | 0.82 | 0.80 | 0.336 / 0.250 | 58% / 84% | -0.024 [-0.114, +0.060] |
| L43 | +0.631 | +0.607 | +0.606 | +0.606 | 0.89 | 0.90 | 0.313 / 0.250 | 62% / 80% | +0.023 [-0.049, +0.096] |
| L40 | +0.408 | +0.442 | +0.444 | +0.444 | 0.77 | 0.78 | 0.361 / 0.312 | 49% / 78% | -0.034 [-0.118, +0.049] |
