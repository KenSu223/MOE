**Knockout effect vs the expert's direct write on the same clean prompts (final position routed to the expert; own task; all-position knockout). Lost share = knockout change of Δ / DLA of the expert's clean output; 1 − lost share = compensation by the rest of the network.**

| Model | Expert | Task | Mode | Prompts | Direct write DLA(c_e) (logits) | Knockout ΔΔ (logits) | Lost share [95% CI] | Compensation |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base | L41E117 | WinoGrande | zero | 850 | +0.321 | -0.151 | 0.47 [0.37, 0.56] | 0.53 |
| Qwen3-30B-A3B-Base | L41E117 | WinoGrande | reroute | 3304 | +0.333 | -0.144 | 0.43 [0.38, 0.48] | 0.57 |
| Qwen3-30B-A3B-Base | L44E069 | CounterFact | zero | 468 | +0.757 | -0.226 | 0.30 [0.24, 0.36] | 0.70 |
| Qwen3-30B-A3B-Base | L44E069 | CounterFact | reroute | 674 | +0.707 | -0.220 | 0.31 [0.27, 0.36] | 0.69 |
| Qwen3-30B-A3B-Base | L42E115 | CounterFact | zero | 494 | +0.415 | -0.156 | 0.38 [0.29, 0.47] | 0.62 |
| Qwen3-30B-A3B-Base | L42E115 | CounterFact | reroute | 721 | +0.402 | -0.154 | 0.38 [0.31, 0.46] | 0.62 |
| Mixtral-8x7B (BOS) | L20E000 | WinoGrande | zero | 999 | +0.319 | -0.143 | 0.45 [0.38, 0.51] | 0.55 |
| Mixtral-8x7B (BOS) | L20E000 | WinoGrande | reroute | 1976 | +0.314 | -0.139 | 0.44 [0.40, 0.49] | 0.56 |
| Mixtral-8x7B (BOS) | L18E001 | CounterFact | zero | 402 | +0.124 | -0.149 | 1.20 [0.82, 1.65] | -0.20 |
| Mixtral-8x7B (BOS) | L18E001 | CounterFact | reroute | 650 | +0.132 | -0.116 | 0.88 [0.59, 1.22] | 0.12 |
| Mixtral-8x7B (BOS) | L19E002 | CounterFact | zero | 310 | +0.350 | -0.068 | 0.19 [0.05, 0.34] | 0.81 |
| Mixtral-8x7B (BOS) | L19E002 | CounterFact | reroute | 486 | +0.356 | -0.049 | 0.14 [0.03, 0.24] | 0.86 |
| Mixtral-8x7B (BOS) | L21E001 | CounterFact | zero | 332 | +0.405 | -0.018 | 0.05 [-0.07, 0.15] | 0.95 |
| Mixtral-8x7B (BOS) | L21E001 | CounterFact | reroute | 547 | +0.417 | -0.035 | 0.08 [-0.00, 0.16] | 0.92 |
