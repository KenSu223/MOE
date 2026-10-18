"""Model registry for the reproduction."""
from __future__ import annotations

MODELS = {
    "qwen3": {"repo": "Qwen/Qwen3-30B-A3B-Base", "paper_key": "Qwen3", "paper_layer": 44, "paper_expert": 69,
              "n_controls": 3, "label": "Qwen3-30B-A3B-Base", "family": "qwen3_moe", "base": None, "instruct": False},
    "mixtral": {"repo": "mistralai/Mixtral-8x7B-v0.1", "paper_key": "Mixtral", "paper_layer": 19, "paper_expert": 6,
                "n_controls": 1, "label": "Mixtral-8x7B-v0.1", "family": "mixtral", "base": None, "instruct": False},
    "olmoe": {"repo": "allenai/OLMoE-1B-7B-0125", "paper_key": None, "paper_layer": None, "paper_expert": None,
              "n_controls": 3, "label": "OLMoE-1B-7B-0125", "family": "olmoe", "base": None, "instruct": False},
    # ---- ext2-model-zoo (RESEARCH_PLAN Direction 2). paper_key / paper_layer / paper_expert of the instruct models are
    # the BASE model's values, kept for reference only (the paper did not study these checkpoints); paper_key makes the
    # filter scripts include the base model's paper case set so that results are directly comparable with REPORT.md.
    "olmoe_instruct": {"repo": "allenai/OLMoE-1B-7B-0125-Instruct", "paper_key": None, "paper_layer": None, "paper_expert": None,
                       "n_controls": 3, "label": "OLMoE-1B-7B-0125-Instruct", "family": "olmoe", "base": "olmoe", "instruct": True},
    "qwen3_instruct": {"repo": "Qwen/Qwen3-30B-A3B-Instruct-2507", "paper_key": "Qwen3", "paper_layer": 44, "paper_expert": 69,
                       "n_controls": 3, "label": "Qwen3-30B-A3B-Instruct-2507", "family": "qwen3_moe", "base": "qwen3", "instruct": True},
    "qwen3_coder": {"repo": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "paper_key": "Qwen3", "paper_layer": 44, "paper_expert": 69,
                    "n_controls": 3, "label": "Qwen3-Coder-30B-A3B-Instruct", "family": "qwen3_moe", "base": "qwen3", "instruct": True},
    "mixtral_instruct": {"repo": "mistralai/Mixtral-8x7B-Instruct-v0.1", "paper_key": "Mixtral", "paper_layer": 19, "paper_expert": 6,
                         "n_controls": 1, "label": "Mixtral-8x7B-Instruct-v0.1", "family": "mixtral", "base": "mixtral", "instruct": True},
    # ---- ext14 (Phase 6, p6-engine): dense counterparts, Qwen2 twins, Qwen3 dense capacity series, Mixtral-8x22B.
    # family = arch.spec_from_config family; dense models are run as one expert (the MLP, expert 0) with weight 1, so
    # n_controls = 0 (no other active expert); proto = the Phase-6 protocol name (Mistral family keeps BOS, Qwen has none).
    "mistral7b": {"repo": "mistralai/Mistral-7B-v0.1", "paper_key": None, "paper_layer": None, "paper_expert": None,
                  "n_controls": 0, "label": "Mistral-7B-v0.1", "family": "mistral", "base": None, "instruct": False,
                  "proto": "mistral7b_bos", "dense": True},
    "qwen2_7b": {"repo": "Qwen/Qwen2-7B", "paper_key": None, "paper_layer": None, "paper_expert": None,
                 "n_controls": 0, "label": "Qwen2-7B", "family": "qwen2", "base": None, "instruct": False,
                 "proto": "qwen2_7b", "dense": True,
                 # numerics protocol (coordinator decision 2026-10-17): fp32 activations and weights (upcast from the bf16
                 # checkpoint) for both Qwen2 twins, so that precision is not a confound (Engine(repo) applies it)
                 "act_dtype": "fp32"},
    "qwen2_57b": {"repo": "Qwen/Qwen2-57B-A14B", "paper_key": None, "paper_layer": None, "paper_expert": None,
                  "n_controls": 3, "label": "Qwen2-57B-A14B", "family": "qwen2_moe", "base": None, "instruct": False,
                  "proto": "qwen2_57b", "dense": False,
                  # numerics protocol (2026-10-17): fp32 activations and weights. In bf16 the nearly uniform routers flip
                  # their routed sets almost everywhere relative to fp32 (also in transformers; fp32 router logits alone do
                  # not fix it): logs/ext14_engine_api.md "Qwen2 numerics protocol"
                  "act_dtype": "fp32"},
    "qwen3_1p7b": {"repo": "Qwen/Qwen3-1.7B-Base", "paper_key": None, "paper_layer": None, "paper_expert": None,
                   "n_controls": 0, "label": "Qwen3-1.7B-Base", "family": "qwen3", "base": None, "instruct": False,
                   "proto": "qwen3_1p7b", "dense": True},
    "qwen3_4b": {"repo": "Qwen/Qwen3-4B-Base", "paper_key": None, "paper_layer": None, "paper_expert": None,
                 "n_controls": 0, "label": "Qwen3-4B-Base", "family": "qwen3", "base": None, "instruct": False,
                 "proto": "qwen3_4b", "dense": True},
    "qwen3_8b": {"repo": "Qwen/Qwen3-8B-Base", "paper_key": None, "paper_layer": None, "paper_expert": None,
                 "n_controls": 0, "label": "Qwen3-8B-Base", "family": "qwen3", "base": None, "instruct": False,
                 "proto": "qwen3_8b", "dense": True},
    "qwen3_14b": {"repo": "Qwen/Qwen3-14B-Base", "paper_key": None, "paper_layer": None, "paper_expert": None,
                  "n_controls": 0, "label": "Qwen3-14B-Base", "family": "qwen3", "base": None, "instruct": False,
                  "proto": "qwen3_14b", "dense": True},
    "mixtral22": {"repo": "mistralai/Mixtral-8x22B-v0.1", "paper_key": None, "paper_layer": None, "paper_expert": None,
                  "n_controls": 1, "label": "Mixtral-8x22B-v0.1", "family": "mixtral", "base": None, "instruct": False,
                  "proto": "mixtral22_bos", "dense": False},  # entry only: not downloaded (Stage 3 is gated, DP6)
}

RESULTS = "/home/ubuntu/MOE/results"
