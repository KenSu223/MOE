"""Architecture specs: shapes, hyper-parameters and checkpoint key templates for the supported MoE families.

All three families share the same decoder-layer skeleton:
    h = x + Attn(RMSNorm(x));  y = h + MoE(RMSNorm(h))
with RoPE, grouped-query attention, softmax router -> top-k (-> optional renormalisation), SwiGLU experts.
Differences handled here: key names, q/k norm variant, top-k renormalisation, expert weight naming.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ArchSpec:
    family: str                 # 'qwen3_moe' | 'mixtral' | 'olmoe'
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_theta: float
    rms_eps: float
    n_experts: int
    top_k: int
    norm_topk: bool             # renormalise the top-k router probabilities to sum to 1
    inter: int                  # expert intermediate size
    qk_norm: Optional[str]      # 'per_head' (Qwen3), 'full' (OLMoE, over n_heads*head_dim), None (Mixtral)
    vocab: int
    tie_embeddings: bool
    # checkpoint key templates ({l} = layer index, {e} = expert index)
    k_ln1: str = "model.layers.{l}.input_layernorm.weight"
    k_ln2: str = "model.layers.{l}.post_attention_layernorm.weight"
    k_q: str = "model.layers.{l}.self_attn.q_proj.weight"
    k_k: str = "model.layers.{l}.self_attn.k_proj.weight"
    k_v: str = "model.layers.{l}.self_attn.v_proj.weight"
    k_o: str = "model.layers.{l}.self_attn.o_proj.weight"
    k_qn: Optional[str] = "model.layers.{l}.self_attn.q_norm.weight"
    k_kn: Optional[str] = "model.layers.{l}.self_attn.k_norm.weight"
    k_router: str = "model.layers.{l}.mlp.gate.weight"
    k_gate: str = "model.layers.{l}.mlp.experts.{e}.gate_proj.weight"   # W_gate (inter, hidden)
    k_up: str = "model.layers.{l}.mlp.experts.{e}.up_proj.weight"       # W_up   (inter, hidden)
    k_down: str = "model.layers.{l}.mlp.experts.{e}.down_proj.weight"   # W_down (hidden, inter)
    k_embed: str = "model.embed_tokens.weight"
    k_norm: str = "model.norm.weight"
    k_head: str = "lm_head.weight"

    @property
    def n_rep(self) -> int:
        return self.n_heads // self.n_kv_heads


def spec_from_config(cfg: dict) -> ArchSpec:
    arch = cfg["architectures"][0]
    hidden = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim") or hidden // n_heads
    common = dict(
        n_layers=cfg["num_hidden_layers"], hidden=hidden, n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim,
        rope_theta=float(cfg.get("rope_theta", 10000.0)), rms_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        top_k=cfg["num_experts_per_tok"], vocab=cfg["vocab_size"], tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
    )
    if arch == "Qwen3MoeForCausalLM":
        assert not cfg.get("mlp_only_layers"), "dense-only layers not supported"
        assert cfg.get("decoder_sparse_step", 1) == 1
        return ArchSpec(family="qwen3_moe", n_experts=cfg["num_experts"], norm_topk=bool(cfg["norm_topk_prob"]),
                        inter=cfg["moe_intermediate_size"], qk_norm="per_head", **common)
    if arch == "OlmoeForCausalLM":
        return ArchSpec(family="olmoe", n_experts=cfg["num_experts"], norm_topk=bool(cfg.get("norm_topk_prob", False)),
                        inter=cfg["intermediate_size"], qk_norm="full", **common)
    if arch == "MixtralForCausalLM":
        return ArchSpec(family="mixtral", n_experts=cfg["num_local_experts"], norm_topk=True,   # Mixtral always renormalises
                        inter=cfg["intermediate_size"], qk_norm=None, k_qn=None, k_kn=None,
                        k_router="model.layers.{l}.block_sparse_moe.gate.weight",
                        k_gate="model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight",
                        k_up="model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight",
                        k_down="model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight", **common)
    raise ValueError(f"unsupported architecture {arch}")


def snapshot_dir(repo_id: str, hf_home: Optional[str] = None) -> str:
    """Resolve the local snapshot directory of a fully downloaded HF repo (no network)."""
    hf_home = hf_home or os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    base = os.path.join(hf_home, "hub", "models--" + repo_id.replace("/", "--"), "snapshots")
    snaps = sorted(os.listdir(base))
    if not snaps:
        raise FileNotFoundError(base)
    return os.path.join(base, snaps[-1])


def load_spec(repo_or_dir: str) -> tuple[ArchSpec, str]:
    d = repo_or_dir if os.path.isdir(repo_or_dir) else snapshot_dir(repo_or_dir)
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    return spec_from_config(cfg), d
