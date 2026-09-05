"""Model registry for the reproduction."""
from __future__ import annotations

MODELS = {
    "qwen3": {"repo": "Qwen/Qwen3-30B-A3B-Base", "paper_key": "Qwen3", "paper_layer": 44, "paper_expert": 69,
              "n_controls": 3, "label": "Qwen3-30B-A3B-Base"},
    "mixtral": {"repo": "mistralai/Mixtral-8x7B-v0.1", "paper_key": "Mixtral", "paper_layer": 19, "paper_expert": 6,
                "n_controls": 1, "label": "Mixtral-8x7B-v0.1"},
    "olmoe": {"repo": "allenai/OLMoE-1B-7B-0125", "paper_key": None, "paper_layer": None, "paper_expert": None,
              "n_controls": 3, "label": "OLMoE-1B-7B-0125 (pilot)"},
}

RESULTS = "/home/ubuntu/MOE/results"
