**Attention share and additivity per model**

| Model / protocol | MoE-peak layer | Attention share there [CI] | Overall share (AUC+) [CI] | Block peak | Block | Attn + MoE | Gap [CI] | Per-case r |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B-Base (tokenizer defaults) | L44 | +0.021 [-0.015, +0.055] | +0.512 [+0.475, +0.547] | L40 | +1.946 | +2.022 | -0.076 [-0.123, -0.029] | 0.98 |
| Mixtral-8x7B-v0.1 (BOS, tokenizer default) | L19 | +0.622 [+0.583, +0.661] | +0.555 [+0.524, +0.582] | L19 | +1.374 | +1.482 | -0.108 [-0.175, -0.047] | 0.97 |
| Mixtral-8x7B-v0.1 (no BOS, paper protocol) | L21 | +0.177 [+0.107, +0.244] | +0.601 [+0.539, +0.654] | L19 | +1.183 | +1.290 | -0.107 [-0.167, -0.048] | 0.96 |
| OLMoE-1B-7B-0125 (pilot) | L12 | +0.706 [+0.585, +0.810] | +0.743 [+0.687, +0.779] | L12 | +3.955 | +4.164 | -0.209 [-0.571, +0.122] | 0.97 |
