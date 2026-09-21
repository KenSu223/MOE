**Mixtral-8x7B-v0.1 (no BOS, paper protocol): concentration of each metric's block rescue (at its own discovery argmax and at the Δ layer) in the 26 cases whose noised distribution is farthest from the clean one**

| Metric | Layer | Mean rescue (256 cases) | Mean, top-10% KL(noised‖clean) cases | Mean, other 90% | Share of summed rescue from the top-10% | r(rescue, KL noised) |
|---|---|---|---|---|---|---|
| Δ | L19 | +0.441 | +0.513 | +0.433 | 12% | 0.11 |
| Δp | L18 | +0.003 | +0.001 | +0.003 | 5% | 0.01 |
| Δp | L19 | +0.003 | +0.000 | +0.003 | 1% | 0.02 |
| Δlog p | L0 | +0.503 | +2.818 | +0.242 | 57% | 0.69 |
| Δlog p | L19 | +0.522 | +0.656 | +0.507 | 13% | 0.13 |
| rank | L18 | +0.645 | +1.571 | +0.541 | 25% | 0.43 |
| rank | L19 | +0.694 | +0.625 | +0.702 | 9% | 0.15 |
| KL | L0 | +0.677 | +4.383 | +0.258 | 66% | 0.87 |
| KL | L19 | +0.098 | +0.484 | +0.055 | 50% | 0.12 |
| Δ/drop | L19 | +0.091 | +0.080 | +0.092 | 9% | -0.00 |
