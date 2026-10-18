"""ext14 circuit (Phase 6, P6-5, agent p6-circuit): C1 neurons inside the selected experts (forward only).

For a selected expert (l, e) and its own task (STR rows: CounterFact (case, donor) rows or WinoGrande directed cases), at
the final position:

    a_i       = SiLU(W_gate x)_i (W_up x)_i, the expert's intermediate activation of neuron i (engine DiagSpec.expert_neurons)
    w         = the routing weight applied to the expert's output (0 when the expert is not routed in that run; dense 1)
    d_i       = w_clean a_i,clean - w_corrupt a_i,corrupt          (the neuron's weighted activation change)
    dla_i     = d_i (W_down[:, i] . u) / rms(h_final, corrupted run),   u = gamma * (W_U[true] - W_U[foil])
                (direct effect on LD with the final norm frozen at the corrupted run, exactly as the expert DLA of
                Phase 3/4; sum_i dla_i = the expert's DLA up to the bf16 rounding of the expert's down projection)
    mag_i     = |d_i| * ||W_down[:, i]||                            (magnitude baseline, no read-out direction)

Validation is exact: engine kind expert_neurons sets the neurons N to the clean values inside the corrupted run
(v = sum_{i in N} d_i W_down[:, i], everything else as the corrupted run, downstream recomputed), against the expert's own
single-expert patch (kind expert) as 100 %. Rankings per STR row:
    dla_case   the row's own dla_i, descending (signed: neurons writing towards the answer first)
    mag_case   the row's own mag_i, descending
    dla_pop    the discovery-split mean of dla_i (donor-mean per case first), descending: ONE fixed neuron set for all rows
    rand0/1    random neuron order (numpy default_rng(seed = 1000 * s + row_id))
k grid: 1, 2, 5, 10, 20, 50, 100, 200, (400 | 500, 1000, 2000, 5000), all. Plus single-neuron patches of the row's top-8
neurons by |dla_i| (neuron-level DLA vs patch agreement).

Dense analogue (Mistral-7B, MLP = expert 0 with weight 1): the same on the Mixtral CounterFact / WinoGrande STR items
(identical 32k tokenizer, BOS), restricted to rows that pass Mistral's own margin (clean Delta >= 1, corrupted Delta <= -1).

Files per task (results/p6ci_c1_<task>/): c1_rows.parquet (one line per STR row: deltas, rms, per-expert weights /
routed flags), c1_prefill_meta.json, c1_neurons_<LxxEyyy>.npz (compact: per-row top neurons by |dla| / mag, discovery
mean dla, per-neuron W_down norms, row sums), c1_patch_<LxxEyyy>.parquet (one line per spawn), .done_<stage> markers.
Full per-row arrays [rows, inter] (float16) are cached on the NVMe (CACHE) and are only needed for re-ranking.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .models import MODELS, RESULTS

ROOT = "/home/ubuntu/MOE"
CACHE = "/opt/dlami/nvme/p6ci_cache"
SUMMARY = os.path.join(RESULTS, "ext14ci_summary.json")
TOPN_STORE = 2000  # neurons per row kept in the compact npz (all for inter <= TOPN_STORE)
N_SINGLE = 8  # single-neuron patches per row (top by |dla|)
RANKINGS = ("dla_case", "mag_case", "dla_pop", "rand0", "rand1")

WINO_Q = ("data/wino_str/pairs_train_xl_qwen3.parquet", "data/wino_str/case_sets.json")
WINO_M = ("data/wino_str/pairs_train_xl_mixtral_bos.parquet", "data/wino_str/case_sets.json")
F1_Q = ("data/wino_ref/pairs_p5_f1_qwen3.parquet", "data/wino_ref/case_sets_p5_f1_qwen3.json")
DIROPT_Q = ("data/wino_ref/pairs_p5_dir_opt_qwen3.parquet", "data/wino_ref/case_sets_p5_dir_opt_qwen3.json")
DIROPT_M = ("data/wino_ref/pairs_p5_dir_opt_mixtral_bos.parquet", "data/wino_ref/case_sets_p5_dir_opt_mixtral_bos.json")

# task registry: model = engine model key; items = (loader kind, items model, source run, pairs, case sets)
C1_TASKS = {
    "qwen3_cf": dict(model="qwen3", items=("cf", "qwen3", "qwen3_str"), experts=[(44, 69), (42, 115)],
                     label="Qwen3-30B-A3B, CounterFact STR"),
    "qwen3_wino": dict(model="qwen3", items=("wino", "qwen3", "wino_qwen3_str") + WINO_Q, experts=[(41, 117)],
                       label="Qwen3-30B-A3B, WinoGrande option swap (Phase 3)"),
    "qwen3_f1": dict(model="qwen3", items=("wino", "qwen3", "p5_f1_qwen3") + F1_Q, experts=[(37, 32)],
                     label="Qwen3-30B-A3B, WinoGrande attribute -> entity (F1)"),
    "mixtral_cf": dict(model="mixtral", items=("cf", "mixtral", "mixtral_bos_str"), experts=[(19, 2), (21, 1), (18, 1)],
                       label="Mixtral-8x7B (BOS), CounterFact STR"),
    "mixtral_wino": dict(model="mixtral", items=("wino", "mixtral", "wino_mixtral_bos_str") + WINO_M, experts=[(20, 0)],
                         label="Mixtral-8x7B (BOS), WinoGrande option swap (Phase 3)"),
    "mistral7b_cf": dict(model="mistral7b", items=("cf", "mixtral", "mixtral_bos_str"), experts=[(18, 0), (19, 0), (20, 0), (21, 0)],
                         margin=1.0, sweep=True, label="Mistral-7B (BOS), CounterFact STR (Mixtral items)"),
    "mistral7b_wino": dict(model="mistral7b", items=("wino", "mixtral", "wino_mixtral_bos_str") + WINO_M,
                           experts=[(18, 0), (19, 0), (20, 0), (21, 0)], margin=1.0, sweep=True,
                           label="Mistral-7B (BOS), WinoGrande option swap (Mixtral items)"),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ename(l: int, e: int) -> str:
    return f"L{int(l)}E{int(e):03d}"


def parse_ename(s: str) -> tuple[int, int]:
    a, b = s[1:].split("E")
    return int(a), int(b)


DIR_SUFFIX = ""  # "_smoke" for smoke runs (set by the runners)


def task_dir(task: str) -> str:
    d = os.path.join(RESULTS, f"p6ci_c1_{task}{DIR_SUFFIX}")
    os.makedirs(d, exist_ok=True)
    return d


def cache_dir(task: str) -> str:
    d = os.path.join(CACHE, f"c1_{task}{DIR_SUFFIX}")
    os.makedirs(d, exist_ok=True)
    return d


def k_grid(I: int) -> list[int]:
    base = [1, 2, 5, 10, 20, 50, 100, 200]
    base += [400] if I <= 1024 else [500, 1000, 2000, 5000]
    return [k for k in base if k < I] + [I]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    raise TypeError(type(o))


def update_summary(key: str, value) -> None:
    s = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {}
    s[key] = value
    tmp = SUMMARY + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1, default=_json_default)
    os.replace(tmp, SUMMARY)


# ------------------------------------------------------------------------------------------------------------------
# items
# ------------------------------------------------------------------------------------------------------------------
def load_items(task: str):
    """ext8 Task of the task's items (cases: case_id, split, unit, clean_ids, true_id, foil_id; rows: row_id, case_id,
    slot, corrupt_ids, first)."""
    from . import ext8_addback as X
    it = C1_TASKS[task]["items"]
    if it[0] == "cf":
        return X.load_cf(it[1], it[2])
    return X.load_wino(it[1], it[2], os.path.join(ROOT, it[3]), os.path.join(ROOT, it[4]))


@dataclass
class Layout:
    """Prefill layout of one pass: clean rows of the cases first, then one corrupted row per STR row."""
    case_ids: list
    row_ids: list
    clean_row: dict = field(default_factory=dict)  # case_id -> prefill index
    corrupt_row: dict = field(default_factory=dict)  # row_id -> prefill index


def build_prefill(t, row_ids: list):
    from .engine import PrefillSpec
    rows = t.rows.set_index("row_id").loc[row_ids]
    case_ids = list(dict.fromkeys(int(c) for c in rows.case_id))
    cases = t.cases.set_index("case_id")
    pre, lay = [], Layout(case_ids, list(row_ids))
    for c in case_ids:
        cc = cases.loc[c]
        lay.clean_row[c] = len(pre)
        pre.append(PrefillSpec(list(cc.clean_ids), int(cc.true_id), int(cc.foil_id)))
    for r, rr in zip(row_ids, rows.itertuples()):
        cc = cases.loc[int(rr.case_id)]
        lay.corrupt_row[int(r)] = len(pre)
        pre.append(PrefillSpec(list(rr.corrupt_ids), int(cc.true_id), int(cc.foil_id)))
    return pre, lay


# ------------------------------------------------------------------------------------------------------------------
# scoring (CPU / GPU helpers on the engine's globals)
# ------------------------------------------------------------------------------------------------------------------
def readout_dirs(eng, true_ids: np.ndarray, foil_ids: np.ndarray):
    """u = gamma * (W_U[true] - W_U[foil]) fp32 [R, H] (torch, on the engine device)."""
    import torch
    head, gamma = eng.g["head"], eng.g["norm"].float()
    t = torch.as_tensor(true_ids, device=head.device, dtype=torch.long)
    f = torch.as_tensor(foil_ids, device=head.device, dtype=torch.long)
    return (head[t].float() - head[f].float()) * gamma[None]


def neuron_scores(eng, l: int, e: int, act_c: np.ndarray, w_c: np.ndarray, act_o: np.ndarray, w_o: np.ndarray,
                  true_ids: np.ndarray, foil_ids: np.ndarray, rms_o: np.ndarray):
    """(d [R, I], dla [R, I], wnorm [I]) fp32 numpy. d = w_c a_c - w_o a_o; dla = d * (W_down^T u) / rms_o."""
    import torch
    dev = eng.device
    _, _, wd = eng.store.expert_tensors(l, e)  # [H, I]
    wd = wd.to(dev).float()
    wnorm = wd.norm(dim=0).cpu().numpy()
    out_d = np.empty(act_c.shape, dtype=np.float32)
    out_s = np.empty(act_c.shape, dtype=np.float32)
    for c0 in range(0, act_c.shape[0], 512):
        sl = slice(c0, min(act_c.shape[0], c0 + 512))
        u = readout_dirs(eng, true_ids[sl], foil_ids[sl])  # [r, H]
        g = (u @ wd) / torch.as_tensor(rms_o[sl], device=dev)[:, None]  # [r, I]
        d = (torch.as_tensor(w_c[sl], device=dev)[:, None] * torch.as_tensor(act_c[sl], device=dev)
             - torch.as_tensor(w_o[sl], device=dev)[:, None] * torch.as_tensor(act_o[sl], device=dev))
        out_d[sl] = d.cpu().numpy()
        out_s[sl] = (d * g).cpu().numpy()
    del wd
    return out_d, out_s, wnorm


def orders(dla: np.ndarray, d: np.ndarray, wnorm: np.ndarray, pop_order: np.ndarray, row_ids: np.ndarray) -> dict:
    """Per-row neuron orders (int32 [R, I]) for every ranking."""
    R, I = dla.shape
    out = {"dla_case": np.argsort(-dla, axis=1, kind="stable").astype(np.int32),
           "mag_case": np.argsort(-(np.abs(d) * wnorm[None]), axis=1, kind="stable").astype(np.int32),
           "dla_pop": np.broadcast_to(pop_order.astype(np.int32), (R, I))}
    for s in (0, 1):
        o = np.empty((R, I), dtype=np.int32)
        for j, r in enumerate(row_ids):
            o[j] = np.random.default_rng(1000 * s + int(r)).permutation(I).astype(np.int32)
        out[f"rand{s}"] = o
    return out


# ------------------------------------------------------------------------------------------------------------------
# compact storage of the scores
# ------------------------------------------------------------------------------------------------------------------
def save_scores(task: str, name: str, row_ids: np.ndarray, d: np.ndarray, dla: np.ndarray, wnorm: np.ndarray,
                disc_mean: np.ndarray, extra: Optional[dict] = None):
    """Compact npz in the task dir (top neurons per row by |dla| and by mag, row sums, discovery mean) + full float16
    arrays in the NVMe cache."""
    R, I = dla.shape
    n = min(I, TOPN_STORE)
    top_abs = np.argsort(-np.abs(dla), axis=1, kind="stable")[:, :n].astype(np.int16 if I < 32768 else np.int32)
    mag = np.abs(d) * wnorm[None]
    top_mag = np.argsort(-mag, axis=1, kind="stable")[:, :n].astype(top_abs.dtype)
    np.savez_compressed(os.path.join(task_dir(task), f"c1_neurons_{name}.npz"), row_id=row_ids.astype(np.int32),
                        top_abs=top_abs, top_abs_dla=np.take_along_axis(dla, top_abs.astype(np.int64), 1).astype(np.float32),
                        top_abs_d=np.take_along_axis(d, top_abs.astype(np.int64), 1).astype(np.float32),
                        top_mag=top_mag, top_mag_dla=np.take_along_axis(dla, top_mag.astype(np.int64), 1).astype(np.float32),
                        dla_sum=dla.sum(1), dla_pos=np.clip(dla, 0, None).sum(1), dla_neg=np.clip(dla, None, 0).sum(1),
                        dla_abs=np.abs(dla).sum(1), d_norm=np.linalg.norm(d, axis=1), wnorm=wnorm.astype(np.float32),
                        disc_mean=disc_mean.astype(np.float32), inter=np.int32(I), **(extra or {}))
    np.savez(os.path.join(cache_dir(task), f"c1_full_{name}.npz"), row_id=row_ids.astype(np.int32),
             d=d.astype(np.float16), dla=dla.astype(np.float16), wnorm=wnorm.astype(np.float32))


def load_full(task: str, name: str) -> dict:
    p = os.path.join(cache_dir(task), f"c1_full_{name}.npz")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def load_compact(task: str, name: str) -> dict:
    z = np.load(os.path.join(task_dir(task), f"c1_neurons_{name}.npz"))
    return {k: z[k] for k in z.files}


def pop_mean(rows: pd.DataFrame, dla: np.ndarray, disc_cases: set) -> np.ndarray:
    """Discovery mean of dla over cases (donor-mean per case first). rows aligned with dla (case_id column)."""
    m = rows.case_id.isin(disc_cases).to_numpy()
    if not m.any():
        return np.zeros(dla.shape[1], dtype=np.float32)
    sub = pd.DataFrame({"case_id": rows.case_id.to_numpy()[m]})
    acc = np.zeros(dla.shape[1], dtype=np.float64)
    groups = sub.groupby("case_id").indices
    idx = np.nonzero(m)[0]
    for c, gi in groups.items():
        acc += dla[idx[gi]].astype(np.float64).mean(0)
    return (acc / len(groups)).astype(np.float32)
