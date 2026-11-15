"""Phase 8 helpers shared by the GPU runner and the CPU analysis (pandas / numpy only)."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"qwen3": "/opt/dlami/nvme/hf/hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
          "mixtral": "/opt/dlami/nvme/hf/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0"}
TASKS = ("simple", "nounpp", "rc", "within_rc")


def task_dir(model, task, profile="paper"):
    return ROOT / "results/relp_sva_moe" / model / profile / task


def patch_rows(d: Path, split="discovery", direction="denoise", kind=None, columns=None, sub="patch_full"):
    import pyarrow.dataset as ds
    f = (ds.field("split") == split) & (ds.field("direction") == direction)
    if kind is not None:
        f = f & (ds.field("kind") == kind)
    return ds.dataset(sorted(glob.glob(str(d / sub / "part_*.parquet")))).to_table(columns=columns, filter=f).to_pandas()


def exact_means(d: Path, split="discovery", direction="denoise", kind="expert", sub="patch_full") -> pd.DataFrame:
    """Unit means over the split's pairs; a pair in which the unit is absent (expert routed at neither endpoint) counts 0.
    Columns: layer, pos, expert, mean, se, n_present, n_pairs."""
    cols = ["row", "layer", "pos", "expert", "effect"]
    df = patch_rows(d, split, direction, kind, cols, sub)
    n = int(df.row.nunique()) if kind == "position" else int(patch_rows(d, split, direction, "position", ["row"], sub).row.nunique())
    g = df.groupby(["layer", "pos", "expert"]).effect
    s, s2, c = g.sum(), (df.assign(e2=df.effect ** 2).groupby(["layer", "pos", "expert"]).e2.sum()), g.size()
    mean = s / n
    var = (s2 / n - mean ** 2).clip(lower=0) * n / max(n - 1, 1)
    out = pd.DataFrame({"mean": mean, "se": np.sqrt(var / n), "n_present": c}).reset_index()
    out["n_pairs"] = n
    return out


def save_sparse(path: Path, pe: dict, extra: dict | None = None):
    """pe: {variant: float32 [n, L, T, E]} -> coordinates where any variant is non-zero + values per variant."""
    import torch
    stack = torch.stack([v for v in pe.values()])
    nz = torch.nonzero((stack != 0).any(0))
    arrays = {f"v_{k}": v[tuple(nz.t())].numpy().astype(np.float32) for k, v in pe.items()}
    arrays.update(row=nz[:, 0].numpy().astype(np.int16), layer=nz[:, 1].numpy().astype(np.int16),
                  pos=nz[:, 2].numpy().astype(np.int16), expert=nz[:, 3].numpy().astype(np.int16),
                  shape=np.array(stack.shape[1:]), variants=np.array(list(pe)))
    if extra:
        arrays.update(extra)
    np.savez_compressed(path, **arrays)


def load_sparse(path: Path) -> pd.DataFrame:
    z = np.load(path)
    df = pd.DataFrame({k: z[k] for k in ("row", "layer", "pos", "expert")})
    for v in z["variants"]:
        df[str(v)] = z[f"v_{v}"]
    return df


def tau_from_logits(logits: np.ndarray, rows: np.ndarray, first: int, k: int, r: int) -> np.ndarray:
    """RMS over (row, position >= first, top-(k + r) experts of A) of the router-logit difference A - B, per layer.
    logits: fp16 [2, n, L, T, E]."""
    la = logits[0, rows, :, first:].astype(np.float32)
    lb = logits[1, rows, :, first:].astype(np.float32)
    cand = np.argsort(-la, axis=-1)[..., :k + r]
    d = np.take_along_axis(la - lb, cand, -1)
    return np.sqrt((d.astype(np.float64) ** 2).mean(axis=(0, 2, 3)))


def band_of(layer, L):
    return np.select([layer == 0, layer <= 2, layer < L // 2, layer < L - 8], ["L0", "L1-2", "early-mid", "mid-late"], "last8")


def load_json(p):
    return json.loads(Path(p).read_text())


def patch_offsets(nr: pd.DataFrame, info: list) -> np.ndarray:
    """Common shift of the single-neuron patches of one pair spawned at one layer (patched rows run later layers in other
    GEMM shapes than the unpatched row): median effect over the pair's random-set and top-expert-set neurons in that layer
    (true effects ~0 for most of them), if >= 10 such units; else the pair's random-set median; else 0. Returns the offset per
    row of nr (columns row, layer, neuron, effect)."""
    ref = np.array([("random" in x["sets"]) or ("top_expert" in x["sets"]) for x in info])
    rnd = np.array(["random" in x["sets"] for x in info])
    r = nr[ref[nr.neuron.values]]
    g = r.groupby(["row", "layer"]).effect.agg(["median", "size"])
    g = g[g["size"] >= 10]["median"]
    pr = nr[rnd[nr.neuron.values]].groupby("row").effect.median()
    key = pd.MultiIndex.from_arrays([nr.row.values, nr.layer.values])
    off = g.reindex(key).values
    fb = nr.row.map(pr).fillna(0.0).values
    return np.where(np.isnan(off), fb, off)
