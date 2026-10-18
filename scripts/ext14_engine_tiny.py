"""ext14 (p6-engine): build tiny randomly initialised checkpoints of the Phase-6 families for exact fp32 checks.

Families: mistral (MistralForCausalLM), qwen2 (Qwen2ForCausalLM, q/k/v biases), qwen3 (Qwen3ForCausalLM, per-head q/k
norm, tied embeddings), qwen2_moe (Qwen2MoeForCausalLM: 8 routed experts top-2 without renormalisation + shared expert).
Weights are drawn with larger scales than the HF init (std 0.08 matrices, norms 1 + 0.1 N(0, 1), biases 0.1, router 1.0)
so that attention, routing and patch effects are non-trivial. Saved in fp32 with save_pretrained (per-expert keys, as the
real checkpoints) under /opt/dlami/nvme/p6_tiny/<family>/.

Usage: python scripts/ext14_engine_tiny.py [--out /opt/dlami/nvme/p6_tiny]
"""
import argparse, json, os
import torch

COMMON = dict(vocab_size=1000, hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=512, rms_norm_eps=1e-6)


def build(fam: str):
    import transformers as T
    if fam == "mistral":
        cfg = T.MistralConfig(**COMMON, sliding_window=4096, tie_word_embeddings=False, rope_theta=10000.0)
        m = T.MistralForCausalLM(cfg)
    elif fam == "qwen2":
        cfg = T.Qwen2Config(**COMMON, tie_word_embeddings=False, use_sliding_window=False, rope_theta=1e6)
        m = T.Qwen2ForCausalLM(cfg)
    elif fam == "qwen3":
        cfg = T.Qwen3Config(**COMMON, head_dim=32, tie_word_embeddings=True, attention_bias=False, use_sliding_window=False,
                            rope_theta=1e6)
        m = T.Qwen3ForCausalLM(cfg)
    elif fam == "qwen2_moe":
        cfg = T.Qwen2MoeConfig(**COMMON, moe_intermediate_size=48, shared_expert_intermediate_size=160, num_experts=8,
                               num_experts_per_tok=2, norm_topk_prob=False, tie_word_embeddings=False,
                               use_sliding_window=False, rope_theta=1e6)
        m = T.Qwen2MoeForCausalLM(cfg)
    else:
        raise ValueError(fam)
    g = torch.Generator().manual_seed(14)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith(".bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith("mlp.gate.weight"):
                p.copy_(1.0 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
    return m.float().eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/opt/dlami/nvme/p6_tiny")
    args = ap.parse_args()
    for fam in ("mistral", "qwen2", "qwen3", "qwen2_moe"):
        d = os.path.join(args.out, fam)
        m = build(fam)
        m.save_pretrained(d)
        print(fam, d, sum(p.numel() for p in m.parameters()), json.load(open(os.path.join(d, "config.json")))["architectures"])
