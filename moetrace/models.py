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
}

RESULTS = "/home/ubuntu/MOE/results"
