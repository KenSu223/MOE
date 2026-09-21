**Mixtral-8x7B-v0.1 (no BOS, paper protocol): additivity of single-expert rescues on the paper validation split (same pass; the exact end point of the additive curve is the clean-top-k coalition)**

| Layer | Sum of singles | Coalition (clean top-k) | Coalition (union) | Block | r(sum, coalition) | r(sum, block) | mean / median |sum - coalition| | within 0.25 / 0.5 | sum - coalition [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| L17 | +0.102 | +0.103 | +0.102 | +0.102 | 0.90 | 0.84 | 0.104 / 0.125 | 94% / 99% | -0.001 [-0.029, +0.029] |
| L18 | +0.184 | +0.199 | +0.186 | +0.187 | 0.94 | 0.90 | 0.117 / 0.125 | 92% / 98% | -0.015 [-0.046, +0.016] |
| L19 | +0.416 | +0.442 | +0.426 | +0.427 | 0.92 | 0.91 | 0.139 / 0.125 | 92% / 97% | -0.026 [-0.081, +0.023] |
| L20 | +0.368 | +0.385 | +0.369 | +0.368 | 0.95 | 0.95 | 0.112 / 0.125 | 94% / 98% | -0.018 [-0.051, +0.016] |
| L21 | +0.489 | +0.474 | +0.513 | +0.513 | 0.98 | 0.76 | 0.086 / 0.062 | 98% / 99% | +0.015 [-0.008, +0.040] |
| L22 | +0.216 | +0.223 | +0.245 | +0.246 | 0.96 | 0.89 | 0.089 / 0.062 | 96% / 100% | -0.007 [-0.030, +0.016] |
