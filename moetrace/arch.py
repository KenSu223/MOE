"""Architecture specs: shapes, hyper-parameters and checkpoint key templates for the supported MoE families.

All three families share the same decoder-layer skeleton:
    h = x + Attn(RMSNorm(x));  y = h + MoE(RMSNorm(h))
with RoPE, grouped-query attention, softmax router -> top-k (-> optional renormalisation), SwiGLU experts.
Differences handled here: key names, q/k norm variant, top-k renormalisation, expert weight naming.

ext14 (Phase 6, p6-engine) additions, behind new families / fields with defaults (existing families unchanged):
* dense families 'mistral' (MistralForCausalLM), 'qwen2' (Qwen2ForCausalLM: q/k/v biases), 'qwen3' (Qwen3ForCausalLM:
  per-head q/k norm, tied embeddings for the small models): the MLP is represented as ONE expert with weight 1
  (n_experts = 1, top_k = 1, no router; k_router = None). Keys mlp.{gate,up,down}_proj.
* 'qwen2_moe' (Qwen2MoeForCausalLM, Qwen2-57B-A14B): q/k/v biases, routed experts with norm_topk from the config (False
  for 57B), plus an always-on shared expert mlp.shared_expert.{gate,up,down}_proj (intermediate size shared_inter) whose
  output is scaled by sigmoid(mlp.shared_expert_gate.weight . x).
* sliding_window (Mistral 4096): the engine asserts that every prompt is shorter (no sliding-window masking implemented).
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
    # ---- ext14 (Phase 6) additions; defaults = the behaviour of the three original MoE families
    k_qb: Optional[str] = None          # q/k/v projection biases (Qwen2 / Qwen2-MoE)
    k_kb: Optional[str] = None
    k_vb: Optional[str] = None
    shared_inter: int = 0               # Qwen2-MoE shared expert intermediate size (0 = no shared expert)
    k_sh_gate: Optional[str] = None     # shared expert W_gate (shared_inter, hidden)
    k_sh_up: Optional[str] = None       # shared expert W_up   (shared_inter, hidden)
    k_sh_down: Optional[str] = None     # shared expert W_down (hidden, shared_inter)
    k_sh_router: Optional[str] = None   # shared expert gate (1, hidden): weight sigmoid(w . x)
    sliding_window: Optional[int] = None
    router_fp32: bool = False           # ext14 protocol (models.py flag): router logits computed in fp32 (Qwen2-57B)

    @property
    def n_rep(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def is_dense(self) -> bool:
        """ext14: dense family (the MLP is expert 0 with weight 1; no router)."""
        return self.family in DENSE_FAMILIES

    @property
    def has_shared(self) -> bool:
        """ext14: Qwen2-MoE style always-on shared expert (engine index n_experts)."""
        return self.shared_inter > 0


DENSE_FAMILIES = ("mistral", "qwen2", "qwen3")  # ext14


def spec_from_config(cfg: dict) -> ArchSpec:
    arch = cfg["architectures"][0]
    hidden = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim") or hidden // n_heads
    common = dict(
        n_layers=cfg["num_hidden_layers"], hidden=hidden, n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim,
        rope_theta=float(cfg.get("rope_theta", (cfg.get("rope_parameters") or {}).get("rope_theta", 10000.0))), rms_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        top_k=cfg.get("num_experts_per_tok", 1), vocab=cfg["vocab_size"], tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
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
    # ---- ext14 (Phase 6): dense families and Qwen2-MoE
    if arch in ("MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen2MoeForCausalLM"):
        assert not cfg.get("rope_scaling"), f"rope_scaling not supported: {cfg.get('rope_scaling')}"
        assert (cfg.get("rope_parameters") or {}).get("rope_type", "default") == "default", cfg.get("rope_parameters")
    if arch in ("MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"):
        fam = {"MistralForCausalLM": "mistral", "Qwen2ForCausalLM": "qwen2", "Qwen3ForCausalLM": "qwen3"}[arch]
        common_d = dict(common, top_k=1)
        sw = cfg.get("sliding_window")
        if fam != "mistral" and not cfg.get("use_sliding_window", False):
            sw = None  # Qwen2 / Qwen3: sliding window disabled by use_sliding_window = False
        dense = dict(n_experts=1, norm_topk=False, inter=cfg["intermediate_size"], k_router=None,
                     k_gate="model.layers.{l}.mlp.gate_proj.weight", k_up="model.layers.{l}.mlp.up_proj.weight",
                     k_down="model.layers.{l}.mlp.down_proj.weight", sliding_window=sw)
        assert cfg.get("hidden_act", "silu") == "silu", cfg.get("hidden_act")
        if fam == "qwen3":
            assert not cfg.get("attention_bias", False), "Qwen3 attention bias not supported"
            return ArchSpec(family=fam, qk_norm="per_head", **dense, **common_d)
        if fam == "qwen2":
            return ArchSpec(family=fam, qk_norm=None, k_qn=None, k_kn=None, k_qb="model.layers.{l}.self_attn.q_proj.bias",
                            k_kb="model.layers.{l}.self_attn.k_proj.bias", k_vb="model.layers.{l}.self_attn.v_proj.bias",
                            **dense, **common_d)
        return ArchSpec(family=fam, qk_norm=None, k_qn=None, k_kn=None, **dense, **common_d)
    if arch == "Qwen2MoeForCausalLM":
        assert not cfg.get("mlp_only_layers"), "dense-only layers not supported"
        assert cfg.get("decoder_sparse_step", 1) == 1
        assert cfg.get("hidden_act", "silu") == "silu"
        assert cfg.get("qkv_bias", True), "Qwen2-MoE without q/k/v bias not supported"
        sw = cfg.get("sliding_window") if cfg.get("use_sliding_window", False) else None
        return ArchSpec(family="qwen2_moe", n_experts=cfg["num_experts"], norm_topk=bool(cfg.get("norm_topk_prob", False)),
                        inter=cfg["moe_intermediate_size"], qk_norm=None, k_qn=None, k_kn=None,
                        k_qb="model.layers.{l}.self_attn.q_proj.bias", k_kb="model.layers.{l}.self_attn.k_proj.bias",
                        k_vb="model.layers.{l}.self_attn.v_proj.bias",
                        shared_inter=int(cfg["shared_expert_intermediate_size"]),
                        k_sh_gate="model.layers.{l}.mlp.shared_expert.gate_proj.weight",
                        k_sh_up="model.layers.{l}.mlp.shared_expert.up_proj.weight",
                        k_sh_down="model.layers.{l}.mlp.shared_expert.down_proj.weight",
                        k_sh_router="model.layers.{l}.mlp.shared_expert_gate.weight", sliding_window=sw, **common)
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
