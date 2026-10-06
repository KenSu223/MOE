"""ext10 (Phase 4, Direction 10): head + expert add-back to FULL repair at the final position.

Phase 3 (ext8) showed that expert outputs alone cap at the all-MoE ceiling (CounterFact STR 0.53 Qwen3 / 0.41 Mixtral,
WinoGrande option swap 0.84 / 0.79 of the drop), while all attention outputs at the final position restore 1.0 by
construction (the final token is shared and the MoE is per-token). With individual attention heads as candidates next to
experts the question "how few components restore the answer" has a real ceiling of 1.

Rows. Exactly the ext8 validation rows (moetrace.ext8_addback.load_cf / load_wino; read-only import): CounterFact STR
(case, donor) rows (Qwen3 432 rows / 108 cases, Mixtral BOS 436 / 106; donor mean primary, first donor sensitivity) and
WinoGrande option-swap directed cases (256 per model, bootstrap over pairs).

Step 1 (existing engine kinds; scripts/ext10_heads_run.py):
  * single-head patches (kind attn_head) at the final position for every (layer, head):
        v_h = W_o[:, h] (H_h_clean - H_h_corrupt)  added to the corrupted run's attention output before the MoE of layer l
    (parent = corrupted row, source = clean row), plus one attn_layer row per layer (same-pass reference: the heads of a
    layer sum to the attention output up to o_proj rounding, so sum_h single-head rescue vs attn_layer rescue measures the
    interaction of heads within a layer);
  * patch-free head DLA from the prefill rows' per-head outputs (DiagSpec.attn_heads_final at every layer):
        dla_h = (W_o[:, h] (H_h_clean - H_h_corrupt) * gamma) . (W_U[r] - W_U[r']) / rms(h_final_corrupt)
    (final RMSNorm frozen at the corrupted run's scale, as ext8's expert DLA), and the expert DLA of the same pass
    (DiagSpec.contrib_dla, ext8's formula) for the mixed DLA ordering of Step 2.
Step 2 (after the ext9 engine: attn_head steps and heads + experts at one layer in `multi`): adaptive greedy over
heads + experts, head-only greedy, static mixed orderings (single-patch oracle, DLA) on a log-k grid.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .models import MODELS, RESULTS
from . import ext8_addback as X

ROOT = os.path.dirname(RESULTS)

RUNS = {
    "cf_qwen3": dict(model="qwen3", task="cf", src="qwen3_str", ext8="qwen3_str_addback", out="qwen3_str_circuit",
                     label="Qwen3-30B-A3B-Base", task_label="CounterFact STR"),
    "cf_mixtral": dict(model="mixtral", task="cf", src="mixtral_bos_str", ext8="mixtral_bos_str_addback",
                       out="mixtral_bos_str_circuit", label="Mixtral-8x7B (BOS)", task_label="CounterFact STR"),
    "wino_qwen3": dict(model="qwen3", task="wino", src="wino_qwen3_str", ext8="wino_qwen3_str_addback",
                       out="wino_qwen3_str_circuit", pairs="data/wino_str/pairs_train_xl_qwen3.parquet",
                       case_sets="data/wino_str/case_sets.json", label="Qwen3-30B-A3B-Base", task_label="WinoGrande STR"),
    "wino_mixtral": dict(model="mixtral", task="wino", src="wino_mixtral_bos_str", ext8="wino_mixtral_bos_str_addback",
                         out="wino_mixtral_bos_str_circuit", pairs="data/wino_str/pairs_train_xl_mixtral_bos.parquet",
                         case_sets="data/wino_str/case_sets.json", label="Mixtral-8x7B (BOS)", task_label="WinoGrande STR"),
}
RUNS.update({
    # IOI (i) S2 -> IO (ext7-controls runs), attention-pole reference: no single-expert patch table exists, so expert
    # orderings / pools use the expert DLA of the step-1 pass instead of single-patch rescue (flagged in the outputs)
    "ioi_qwen3": dict(model="qwen3", task="ioi", src="ioi_qwen3_s2io", ext8=None, out="ioi_qwen3_s2io_circuit",
                      pairs="data/ioi/pairs_s2io_qwen3.parquet", case_sets="data/ioi/case_sets_s2io.json",
                      label="Qwen3-30B-A3B-Base", task_label="IOI STR (S2 -> IO)"),
    "ioi_mixtral": dict(model="mixtral", task="ioi", src="ioi_mixtral_bos_s2io", ext8=None, out="ioi_mixtral_bos_s2io_circuit",
                        pairs="data/ioi/pairs_s2io_mixtral_bos.parquet", case_sets="data/ioi/case_sets_s2io.json",
                        label="Mixtral-8x7B (BOS)", task_label="IOI STR (S2 -> IO)"),
})
RUN_ORDER = ("cf_qwen3", "cf_mixtral", "wino_qwen3", "wino_mixtral", "ioi_qwen3", "ioi_mixtral")
MAIN_RUNS = ("cf_qwen3", "cf_mixtral", "wino_qwen3", "wino_mixtral")

# heads known from earlier directions (for the consistency check)
KNOWN_HEADS = {
    "cf_qwen3": {(40, 13): "F2 GN mover (L40H13)"},
    "cf_mixtral": {(18, 4): "F2 GN mover (L18H4)", (24, 22): "F2 GN (L24H22)", (15, 1): "F2 GN (L15H1)", (15, 3): "F2 GN (L15H3)",
                   (19, 29): "F2 GN (L19H29)", (19, 30): "F2 GN (L19H30)", (19, 31): "F2 GN (L19H31)"},
    "wino_qwen3": {(38, 18): "W5 L38H18", (38, 21): "W5 L38H21", (38, 16): "W5 L38H16 (negative)", (42, 30): "W5 L42H30",
                   (42, 26): "W5 L42H26 (negative)", (39, 5): "W5 L39H5"},
    "ioi_qwen3": {(42, 11): "W5 IOI S2 reader L42H11", (42, 10): "W5 IOI name mover L42H10", (43, 24): "W5 IOI L43H24",
                  (43, 29): "W5 IOI L43H29", (42, 14): "W5 IOI negative mover L42H14"},
    "ioi_mixtral": {(21, 6): "W5 IOI S2 reader L21H6", (19, 8): "W5 IOI name mover L19H8", (28, 0): "IOI attention L28H0"},
    "wino_mixtral": {(25, 9): "W5 L25H9", (19, 13): "W5 L19H13", (13, 18): "W5 L13H18", (13, 11): "W5 L13H11",
                     (19, 12): "W5 L19H12", (13, 4): "W5 L13H4", (25, 11): "W5 L25H11 (negative)"},
}
# GN head runs of F2 (paper case set) for the CounterFact comparison; W5 STR head rows for WinoGrande
F2_RUNS = {"cf_qwen3": "qwen3_heads", "cf_mixtral": "mixtral_bos_heads"}
W5_RUNS = {"wino_qwen3": "wino_qwen3_str", "wino_mixtral": "wino_mixtral_bos_str", "ioi_qwen3": "ioi_qwen3_s2io", "ioi_mixtral": "ioi_mixtral_bos_s2io"}


def cfg(key: str) -> dict:
    return RUNS[key]


def out_dir(key: str) -> str:
    return os.path.join(RESULTS, RUNS[key]["out"])


def load_ioi(model: str, str_run: str, pairs_path: str, case_sets_path: str, family: str = "main") -> X.Task:
    """IOI STR pairs (ext7-controls run) as an ext8 Task without single-expert tables (single / vnorm / pop empty);
    candidates = the clean run's routed (layer, expert) pairs at the final position (str_sweep_routing, slot -1)."""
    from . import ext7_pairs as P
    from .arch import load_spec
    od = os.path.join(RESULTS, str_run)
    fams = P.load_families(str_run)
    disc, val = [int(c) for c in fams[family]["discovery"]], [int(c) for c in fams[family]["validation"]]
    ids = disc + val
    pairs = P.load_pairs(pairs_path, json.load(open(case_sets_path)))
    dc = P.directed_cases(pairs, ids)
    cases = pd.DataFrame(dict(case_id=ids, split=["discovery"] * len(disc) + ["validation"] * len(val),
                              unit=[dc[c].pair_idx for c in ids], clean_ids=[list(dc[c].clean_ids) for c in ids],
                              true_id=[int(dc[c].true_id) for c in ids], foil_id=[int(dc[c].foil_id) for c in ids]))
    rows = pd.DataFrame(dict(row_id=np.arange(len(ids)), case_id=ids, slot=0, corrupt_ids=[list(dc[c].corrupt_ids) for c in ids]))
    rows["first"] = True
    rt = pd.read_parquet(os.path.join(od, "str_sweep_routing.parquet"))
    rt = rt[(rt.slot == -1) & rt.case_id.isin(ids)]
    cand, weight = {}, {}
    for c, g in rt.groupby("case_id"):
        keys = list(zip(g.layer.astype(int), g.expert.astype(int)))
        cand[int(c)] = sorted(keys)
        weight[int(c)] = dict(zip(keys, g.weight.astype(float)))
    spec, _ = load_spec(MODELS[model]["repo"])
    st = bool(json.load(open(os.path.join(od, "run_meta.json"))).get("sweep", {}).get("special_tokens", True)) \
        if isinstance(json.load(open(os.path.join(od, "run_meta.json"))).get("sweep"), dict) else (model != "mixtral" or True)
    return X.Task("ioi", model, str_run, spec.n_layers, spec.top_k, cases, rows, cand, weight, {}, {}, {}, {}, st)


def load_task(key: str) -> X.Task:
    c = RUNS[key]
    if c["task"] == "ioi":
        return load_ioi(c["model"], c["src"], os.path.join(ROOT, c["pairs"]), os.path.join(ROOT, c["case_sets"]))
    if c["task"] == "cf":
        return X.load_cf(c["model"], c["src"])
    return X.load_wino(c["model"], c["src"], os.path.join(ROOT, c["pairs"]), os.path.join(ROOT, c["case_sets"]))


def val_rows(t: X.Task) -> list[int]:
    split = t.rows.case_id.map(dict(zip(t.cases.case_id, t.cases.split)))
    return t.rows.row_id[split == "validation"].astype(int).tolist()


def row_chunks(t: X.Task, rows: list[int], per_pass: int) -> list[list[int]]:
    """Consecutive chunks of rows with whole cases (all donors of a case in one chunk), each <= per_pass rows."""
    case_of = dict(zip(t.rows.row_id.astype(int), t.rows.case_id.astype(int)))
    by_case: dict[int, list[int]] = {}
    order = []
    for r in rows:
        c = case_of[r]
        if c not in by_case:
            by_case[c] = []
            order.append(c)
        by_case[c].append(r)
    n = max(1, -(-len(rows) // per_pass))
    target = -(-len(rows) // n)  # balanced chunks; a case may overshoot the target by its donors - 1
    chunks, cur = [], []
    for c in order:
        g = by_case[c]
        if cur and len(cur) >= target:
            chunks.append(cur)
            cur = []
        cur += g
    if cur:
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------------------------------------------------------------
# loading step-1 outputs
# ------------------------------------------------------------------------------------------------------------------
def _cat(od: str, prefix: str) -> pd.DataFrame:
    parts = sorted(f for f in os.listdir(od) if f.startswith(prefix) and f.endswith(".parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(os.path.join(od, f)) for f in parts], ignore_index=True)


def load_heads(key: str) -> dict:
    od = out_dir(key)
    return {"rows": _cat(od, "head_rows_p"), "prefill": _cat(od, "head_prefill_p"), "dla": _cat(od, "head_dla_p"),
            "edla": _cat(od, "expert_dla_p"), "meta": json.load(open(os.path.join(od, "run_meta.json")))}


# ------------------------------------------------------------------------------------------------------------------
# Step 2: joint add-back over heads + experts (needs the ext9 engine: multi steps attn_head / heads_experts)
# ------------------------------------------------------------------------------------------------------------------
# A candidate is ("h", layer, head) or ("e", layer, expert). A set S is patched as ONE multi spawn (parent = corrupted
# row, source = clean row), one step per layer with the layer's heads and experts:
#   heads only   (l, "attn_head", heads)              heads patched before the MoE of l, own MoE recomputed
#   experts only (l, "coalition_set", experts)        experts' contributions set to the source's
#   both         (l, "heads_experts", (heads, experts))
# so later layers see the effect of earlier patches (an exact joint patch, nothing summed).
from dataclasses import dataclass, field
import pickle

GREEDY_STEPS = 20
HEAD_POOL = 32
EXPERT_POOL = {"qwen3": 32, "mixtral": 64}  # = ext8's greedy pools
K_GRID = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 32, 48, 64, 96, 128)
STATIC_ORDERS = ("mix_oracle", "mix_dla", "head_oracle", "head_dla")
GREEDY_KINDS = ("mix", "head")
SANITY_ROWS = 32  # rows per task with per-layer sanity rows (all heads of a layer vs attn_layer)


def cstr(c) -> str:
    return f"{c[0]}{int(c[1])}.{int(c[2])}"


def set_str(S) -> str:
    if isinstance(S, str):
        return S
    if isinstance(S, tuple) and len(S) == 2 and isinstance(S[0], str) and S[0] in ("layer_heads", "attn_layer"):
        return f"{S[0]}:{S[1]}"
    return ";".join(cstr(c) for c in sorted(S, key=lambda c: (c[1], c[0], c[2])))


def parse_set(s: str) -> tuple:
    if not s or "." not in s or ":" in s:
        return ()
    out = []
    for p in s.split(";"):
        a, b = p[1:].split(".")
        out.append((p[0], int(a), int(b)))
    return tuple(out)


def steps_of(S, L: int, nH: int) -> tuple:
    if isinstance(S, str):
        if S == "all_heads":
            return tuple((l, "attn_head", tuple(range(nH))) for l in range(L))
        if S == "all_moe":
            return tuple((l, "layer", None) for l in range(L))
        if S == "all_attn":
            return tuple((l, "attn_layer", None) for l in range(L))
        raise ValueError(S)
    if isinstance(S, tuple) and len(S) == 2 and isinstance(S[0], str) and S[0] == "layer_heads":
        return ((int(S[1]), "attn_head", tuple(range(nH))),)
    by: dict[int, list[list[int]]] = {}
    for kind, l, x in S:
        by.setdefault(int(l), [[], []])[0 if kind == "h" else 1].append(int(x))
    st = []
    for l in sorted(by):
        hh, ee = sorted(by[l][0]), sorted(by[l][1])
        if hh and ee:
            st.append((l, "heads_experts", (tuple(hh), tuple(ee))))
        elif hh:
            st.append((l, "attn_head", tuple(hh)))
        else:
            st.append((l, "coalition_set", tuple(ee)))
    return tuple(st)


@dataclass
class CTask:
    """One task inside a combined step-2 pass (prefill block: the validation cases' clean rows, then the validation rows'
    corrupted rows, at absolute offset `off`)."""
    key: str
    t: object
    rows: list
    cases: list
    off: int = 0
    ci: dict = field(default_factory=dict)  # case_id -> absolute prefill index of the clean row
    cr: dict = field(default_factory=dict)  # row_id -> absolute prefill index of the corrupted row
    hs: dict = field(default_factory=dict)  # row_id -> single-head rescue [L * nH] (step 1)
    hd: dict = field(default_factory=dict)  # row_id -> head DLA [L * nH] (step 1)
    es: dict = field(default_factory=dict)  # row_id -> {(l, e): single-expert rescue} (ext8 source run)
    ed: dict = field(default_factory=dict)  # row_id -> {(l, e): expert DLA} (step-1 pass)
    L: int = 0
    nH: int = 0

    @property
    def n_prefill(self) -> int:
        return len(self.cases) + len(self.rows)

    def case_of(self, rid: int) -> int:
        return int(self.t.rows.case_id.iloc[rid])


def load_ctask(key: str, off: int = 0, limit: int = 0) -> CTask:
    t = load_task(key)
    rows = val_rows(t)
    if limit:
        rows = rows[:limit]
    cases = list(dict.fromkeys(int(t.rows.case_id.iloc[r]) for r in rows))
    ct = CTask(key, t, rows, cases, off=off, L=t.L)
    ct.ci = {c: off + i for i, c in enumerate(cases)}
    ct.cr = {r: off + len(cases) + j for j, r in enumerate(rows)}
    h = load_heads(key)
    hr = h["rows"]
    hr = hr[hr.kind == "attn_head"]
    nH = int(hr["head"].max()) + 1
    ct.nH = nH
    for r_, g in hr.groupby("row_id"):
        v = np.zeros(t.L * nH)
        v[g.layer.values.astype(int) * nH + g["head"].values.astype(int)] = g.rescue.values
        ct.hs[int(r_)] = v
    for r_, g in h["dla"].groupby("row_id"):
        v = np.zeros(t.L * nH)
        v[g.layer.values.astype(int) * nH + g["head"].values.astype(int)] = g.dla.values
        ct.hd[int(r_)] = v
    for r_, g in h["edla"].groupby("row_id"):
        ct.ed[int(r_)] = dict(zip(zip(g.layer.astype(int), g.expert.astype(int)), g.dla.astype(float)))
    for r in rows:
        ct.es[r] = t.single[r] if t.single else ct.ed[r]  # IOI: no single-expert table -> expert DLA as the proxy
        assert r in ct.hs and r in ct.hd and r in ct.ed, f"{key}: step-1 data missing for row {r}"
    return ct


def head_cands(ct: CTask, rid: int, by: str = "oracle") -> list:
    v = ct.hs[rid] if by == "oracle" else ct.hd[rid]
    o = np.lexsort((np.arange(len(v)), -v))
    return [("h", int(j) // ct.nH, int(j) % ct.nH) for j in o]


def expert_cands(ct: CTask, rid: int, by: str = "oracle") -> list:
    d = ct.es[rid] if by == "oracle" else ct.ed[rid]
    cand = ct.t.cand[ct.case_of(rid)]
    return [("e", l, e) for (l, e) in sorted(cand, key=lambda p: (-d[p], p))]


def value_of(ct: CTask, rid: int, c, by: str = "oracle") -> float:
    if c[0] == "h":
        v = ct.hs[rid] if by == "oracle" else ct.hd[rid]
        return float(v[c[1] * ct.nH + c[2]])
    d = ct.es[rid] if by == "oracle" else ct.ed[rid]
    return float(d[(c[1], c[2])])


def mixed_order(ct: CTask, rid: int, by: str = "oracle") -> list:
    allc = head_cands(ct, rid, by) + expert_cands(ct, rid, by)
    return sorted(allc, key=lambda c: (-value_of(ct, rid, c, by), c[0], c[1], c[2]))


def pools(ct: CTask, rid: int, model: str) -> dict:
    hp = head_cands(ct, rid, "oracle")[:HEAD_POOL]
    ep = expert_cands(ct, rid, "oracle")[: EXPERT_POOL[model]]
    mix = sorted(hp + ep, key=lambda c: (-value_of(ct, rid, c), c[0], c[1], c[2]))
    return {"mix": mix, "head": hp}


@dataclass
class CGreedy:
    task: str
    kind: str  # mix | head
    rid: int
    pool: list
    S: list = field(default_factory=list)
    vals: list = field(default_factory=list)  # Delta after each step (same-pass values)
    resc: list = field(default_factory=list)  # Delta - Delta_corrupt (same pass)
    passes: list = field(default_factory=list)
    done: bool = False

    def items(self) -> list:
        if self.done:
            return []
        base = tuple(self.S)
        out = []
        if not self.vals:
            out.append((self.task, "greedy_" + self.kind, "eval", len(base), self.rid, base))
        for p in self.pool:
            if p not in self.S:
                out.append((self.task, "greedy_" + self.kind, "expand", len(base) + 1, self.rid, base + (p,)))
        return out


def update_cgreedy(g: CGreedy, df: pd.DataFrame, d_corrupt: float, p: int):
    ev = df[df.order == "eval"]
    if len(ev) and not g.vals:
        g.vals.append(float(ev.delta.iloc[0]))
        g.resc.append(float(ev.delta.iloc[0]) - d_corrupt)
        g.passes.append(p)
    ex = df[df.order == "expand"]
    if len(ex):
        j = int(np.argmax(ex.delta.values))  # first maximum in pool order
        new = parse_set(ex.set.iloc[j])
        added = [c for c in new if c not in g.S]
        assert len(added) == 1, (g.S, new)
        g.S.append(added[0])
        g.vals.append(float(ex.delta.iloc[j]))
        g.resc.append(float(ex.delta.iloc[j]) - d_corrupt)
        g.passes.append(p)
    if len(g.S) >= GREEDY_STEPS or len(g.S) >= len(g.pool):
        g.done = True


@dataclass
class CState:
    queue: list
    qpos: int = 0
    passes_done: int = 0
    greedy: dict = field(default_factory=dict)  # (task, kind, rid) -> CGreedy
    notes: list = field(default_factory=list)


def save_cstate(path: str, st: CState):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(st, f)
    os.replace(tmp, path)


def load_cstate(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def static_groups(ct: CTask, model: str) -> list:
    """Static orderings on the k grid (one group per row and ordering) + ceilings and sanity rows (pass-0 groups)."""
    out = []
    for rid in ct.rows:
        out.append([(ct.key, "ceil", kind, 0, rid, kind) for kind in ("all_heads", "all_attn", "all_moe")])
    for rid in ct.rows[:SANITY_ROWS]:
        g = [(ct.key, "sanity", "layer_heads", 0, rid, ("layer_heads", l)) for l in range(ct.L)]
        g += [(ct.key, "sanity", "attn_layer", 0, rid, ("attn_layer", l)) for l in range(ct.L)]
        out.append(g)
    for rid in ct.rows:
        for o in STATIC_ORDERS:
            what, by = o.split("_")
            od = mixed_order(ct, rid, by) if what == "mix" else head_cands(ct, rid, by)
            out.append([(ct.key, "static", o, k, rid, tuple(od[:k])) for k in K_GRID if k <= len(od)])
    return out


def model_dir(model: str) -> str:
    return os.path.join(RESULTS, f"{model}_circuit")
