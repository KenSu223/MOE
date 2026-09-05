"""Subject-token embedding noise: one Gaussian draw per case, seed 0 + case_id, scale = mult * embedding-matrix std."""
from __future__ import annotations

import torch


def noise_draw(case_id: int, n_subject_tokens: int, hidden: int, sigma: float, seed_base: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed_base + int(case_id))
    return torch.randn(n_subject_tokens, hidden, generator=g) * sigma


def sigma_for(embed_std: float, mult: float = 3.0) -> float:
    return mult * embed_std
