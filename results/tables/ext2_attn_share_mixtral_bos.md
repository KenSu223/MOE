**Mixtral-8x7B-v0.1 (BOS, tokenizer default): attention share of the rescue (validation)**

| Where | Layer | Attention rescue | MoE rescue | Attention share attn/(attn+moe) [95% CI] |
|---|---|---|---|---|
| at the MoE-peak layer | L19 | +0.921 | +0.561 | +0.622 [+0.583, +0.661] |
| at the attention-peak layer | L18 | +0.988 | +0.318 | +0.757 [+0.706, +0.804] |
| at the block-peak layer | L19 | +0.921 | +0.561 | +0.622 [+0.583, +0.661] |
| overall (AUC+ of the validation curves) | all | 4.93 | 3.95 | +0.555 [+0.524, +0.582] |
