"""Summary statistics: mean, positive fraction, percentile bootstrap CI of the mean, two-sided sign-flip test."""
from __future__ import annotations

import numpy as np

N_BOOT = 5000
N_FLIP = 10000
SEED = 0


def bootstrap_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def signflip_p(x: np.ndarray, n_flip: int = N_FLIP, seed: int = SEED) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(x.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_flip, len(x)))
    flipped = np.abs((signs * x[None, :]).mean(axis=1))
    return float((flipped >= obs - 1e-12).mean())


def summarize(x, with_p: bool = True) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0, "mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "pos_frac": float("nan"), "p": float("nan")}
    lo, hi = bootstrap_ci(x)
    out = {"n": int(len(x)), "mean": float(x.mean()), "ci_lo": lo, "ci_hi": hi, "pos_frac": float((x > 0).mean()),
           "zero_frac": float((x == 0).mean())}
    if with_p:
        out["p"] = signflip_p(x)
    return out


def ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Ratio of means with a paired bootstrap CI (num and den are per-case arrays of equal length)."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def fmt(s: dict, digits: int = 3) -> str:
    if s["n"] == 0:
        return "n/a"
    return f"{s['mean']:+.{digits}f} [{s['ci_lo']:+.{digits}f}, {s['ci_hi']:+.{digits}f}]"
