"""ext8 (Phase 3, A0-A3): expert add-back curves under STR.

Question. How many experts, ranked over (layer, expert) pairs and ignoring layers, must be patched back JOINTLY at the
final position to close the gap between the corrupted and the clean run, against the ceiling "all MoE outputs patched"?
Is greedy enough, are there better strategies, how saturated is the curve?

Units. A *row* is one STR patching unit = (clean prompt, corrupted prompt, r, r'): a (case, donor) pair for CounterFact
(ext6 runs, up to 5 donors per case; donor mean primary, first donor sensitivity) or a directed WinoGrande case (ext7;
one row per directed case, the bootstrap unit is the pair). Candidates of a row = the clean run's routed (layer, expert)
pairs at the final position (Qwen3 48 x 8 = 384, Mixtral 32 x 2 = 64), the same set as the ext6 single-expert rows.

Patches (engine kind `multi`, one step per layer, strictly increasing layers):
  * add-back (denoising, sufficiency): parent = corrupted row, source = clean row; a set S of (layer, expert) pairs is one
    coalition_set step per layer with S_l = {e : (l, e) in S}: c_e := c_e(clean) for e in S_l, everything else is the
    row's own computation (so later layers see the effect of earlier patches: an exact joint patch, not a sum);
  * deletion (noising, necessity): parent = clean row, source = corrupted row: c_e := c_e(corrupt) (0 if not routed);
  * A0 ceilings: `layer` (all MoE outputs), `attn_layer` (all attention outputs), `block` (both) at every layer; the same
    restricted to interior layers (<= L-5).
Metrics per row: Delta_k (logit(r) - logit(r')), rescue_k = Delta_k - Delta_corrupt (same pass), normalised
r(k) = rescue_k / (Delta_clean - Delta_corrupt); deletion: damage d(k) = (Delta_clean - Delta_k) / drop.

Static orderings (A1, A3): pop (discovery all-case single-expert rescue), oracle (the row's own single-expert rescue),
layerwise (layers by discovery MoE-layer rescue, experts within a layer by pop), rand0..rand4 (per-case permutations,
random.Random(8000 + 10 case + i)), weight (clean routing weight), vnorm (|delta_e|), dla (direct logit attribution of
delta_e = c_e(clean) - c_e(corrupt) with the final RMSNorm frozen at the corrupted run's scale:
(delta_e * gamma) . (W_U[r] - W_U[r']) / rms(h_final_corrupt)); deletion adds noise_oracle (the row's own noising
single-expert effect, most damaging first; first donor only).
Adaptive strategies (A2): greedy (pool = the row's top-32 singles for Qwen3, all 64 for Mixtral; 15 steps; the brief's
optional stop at 95 % of the row's all-MoE ceiling is off by default because subsets overshoot the ceiling), beam search (width 4, sizes <= 6; first donor), exact optimum over all 1,023 subsets of
the row's top-10 singles (first donor). Shapley values from the full prefix sweeps of the 5 random permutations (first
donor; Qwen3: a fixed half of the validation cases).
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .models import MODELS, RESULTS

GREEDY_STEPS = 15
GREEDY_STOP = None  # optional early stop at this fraction of the row's all-MoE ceiling (None: always GREEDY_STEPS steps;
#                     subsets can overshoot the ceiling, so stopping would truncate the curves; k at 95 % is reported instead)
BEAM_WIDTH = 4
BEAM_MAX = 6
EXACT_N = 10
N_RAND = 5
A1_ORDERS = ("pop", "oracle", "layerwise", "rand0", "rand1", "rand2", "rand3", "rand4", "weight", "vnorm", "dla")
A3_ORDERS = ("pop", "oracle", "layerwise", "rand0", "rand1", "dla", "noise_oracle")
A0_KINDS = ("all_moe", "all_attn", "all_block", "int_moe", "int_attn")


def k_grid(K: int) -> list[int]:
    """1..10, then roughly log-spaced to K (K included)."""
    g = list(range(1, min(10, K) + 1))
    for v in (12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320):
        if v < K:
            g.append(v)
    if K not in g:
        g.append(K)
    return g


# ------------------------------------------------------------------------------------------------------------------
# task data
# ------------------------------------------------------------------------------------------------------------------
@dataclass
class Task:
    name: str  # 'cf' | 'wino'
    model: str  # MODELS key
    run: str  # source run (ext6 / ext7 STR run dir)
    L: int
    topk: int
    cases: pd.DataFrame  # case_id, split, unit, clean_ids, true_id, foil_id
    rows: pd.DataFrame  # row_id, case_id, slot, corrupt_ids, first (slot == min slot of the case)
    cand: dict  # case_id -> list[(l, e)] clean-active, sorted
    weight: dict  # case_id -> {(l, e): clean routing weight}
    single: dict  # row_id -> {(l, e): single-expert rescue (source run)}
    vnorm: dict  # row_id -> {(l, e): |delta_e|}
    pop: dict  # (l, e) -> discovery all-case rescue
    layer_disc: dict  # l -> discovery mean MoE-layer rescue
    special_tokens: bool = True
    case_index: dict = field(default_factory=dict)

    def __post_init__(self):
        self.case_index = {int(c): i for i, c in enumerate(self.cases.case_id)}

    @property
    def K(self) -> int:
        return self.L * self.topk

    @property
    def nC(self) -> int:
        return len(self.cases)

    def split_of_row(self, rid: int) -> str:
        return self.cases.split.iloc[self.case_index[int(self.rows.case_id.iloc[rid])]]


def _pop_and_layers(single_rows: pd.DataFrame, layer_rows: pd.DataFrame, disc: list[int]):
    """pop[(l, e)] = sum over discovery cases where (l, e) is clean-active of the donor-mean single rescue / n_disc;
    layer_disc[l] = mean over discovery cases of the donor-mean MoE-layer rescue."""
    nd = len(disc)
    s = single_rows[single_rows.case_id.isin(disc)].groupby(["case_id", "layer", "expert"]).rescue.mean().reset_index()
    pop = (s.groupby(["layer", "expert"]).rescue.sum() / nd).to_dict()
    lr = layer_rows[layer_rows.case_id.isin(disc)].groupby(["case_id", "layer"]).rescue.mean().reset_index()
    layer_disc = lr.groupby("layer").rescue.mean().to_dict()
    return {(int(l), int(e)): float(v) for (l, e), v in pop.items()}, {int(l): float(v) for l, v in layer_disc.items()}


def _single_dicts(rows: pd.DataFrame, single_rows: pd.DataFrame):
    rid = {(int(c), int(s)): int(r) for r, c, s in zip(rows.row_id, rows.case_id, rows.slot)}
    single, vnorm = {}, {}
    for (c, s), g in single_rows.groupby(["case_id", "slot"]):
        r = rid.get((int(c), int(s)))
        if r is None:
            continue
        keys = list(zip(g.layer.astype(int), g.expert.astype(int)))
        single[r] = dict(zip(keys, g.rescue.astype(float)))
        vnorm[r] = dict(zip(keys, g.vnorm.astype(float)))
    return single, vnorm


def load_cf(model: str, str_run: str) -> Task:
    """CounterFact STR task from an ext6 run (results/<str_run>)."""
    from .protocol import cases_by_id
    from . import ext6_str as S6
    od = os.path.join(RESULTS, str_run)
    meta = json.load(open(os.path.join(od, "run_meta.json")))
    st = bool(meta["sweep"]["special_tokens"])
    sets = json.load(open(os.path.join(od, "case_sets.json")))["paper"]
    disc, val = [int(c) for c in sets["discovery"]], [int(c) for c in sets["validation"]]
    ids = disc + val
    cs, rej = cases_by_id(model, ids, special_tokens=st)
    assert not rej, rej
    cases = pd.DataFrame(dict(case_id=ids, split=["discovery"] * len(disc) + ["validation"] * len(val), unit=ids,
                              clean_ids=[list(cs[c].ids) for c in ids], true_id=[int(cs[c].true_id) for c in ids],
                              foil_id=[int(cs[c].foil_id) for c in ids]))
    dn = S6.load_donors(str_run)
    dn = dn[dn.case_id.isin(ids)].copy()
    dn["ci"] = dn.case_id.map({c: i for i, c in enumerate(ids)})
    dn = dn.sort_values(["ci", "slot"]).reset_index(drop=True)
    rows = pd.DataFrame(dict(row_id=np.arange(len(dn)), case_id=dn.case_id.astype(int), slot=dn.slot.astype(int),
                             corrupt_ids=dn.ids_list))
    rows["first"] = rows.slot == rows.groupby("case_id").slot.transform("min")
    for r in rows.itertuples():
        assert len(r.corrupt_ids) == len(cs[r.case_id].ids)
    rt = pd.read_parquet(os.path.join(od, "str_sweep_routing.parquet"))
    rt = rt[(rt.slot == -1) & rt.case_id.isin(ids)]
    cand, weight = {}, {}
    for c, g in rt.groupby("case_id"):
        keys = list(zip(g.layer.astype(int), g.expert.astype(int)))
        cand[int(c)] = sorted(keys)
        weight[int(c)] = dict(zip(keys, g.weight.astype(float)))
    er = pd.read_parquet(os.path.join(od, "str_expert_rows.parquet"))
    er = er[(er.kind == "expert") & er.clean_active & er.case_id.isin(ids)][["case_id", "slot", "layer", "expert", "rescue", "vnorm"]]
    sw = pd.read_parquet(os.path.join(od, "str_sweep_rows.parquet"))
    sw = sw[(sw.kind == "layer") & sw.case_id.isin(ids)][["case_id", "slot", "layer", "rescue"]]
    single, vnorm = _single_dicts(rows, er)
    pop, layer_disc = _pop_and_layers(er, sw, disc)
    from .arch import load_spec
    spec, _ = load_spec(MODELS[model]["repo"])
    t = Task("cf", model, str_run, spec.n_layers, spec.top_k, cases, rows, cand, weight, single, vnorm, pop, layer_disc, st)
    _check(t)
    return t


def load_wino(model: str, str_run: str, pairs_path: str, case_sets_path: str, family: str = "main") -> Task:
    """WinoGrande STR task from an ext7 run (results/<str_run>, ext6 schema with slot -1 = clean, slot 0 = corrupted;
    case_id = directed id 2 * pair_idx + d). One row per directed case; the bootstrap unit is the pair."""
    from . import ext7_pairs as P
    from .arch import load_spec
    od = os.path.join(RESULTS, str_run)
    fams = P.load_families(str_run)
    disc, val = [int(c) for c in fams[family]["discovery"]], [int(c) for c in fams[family]["validation"]]
    ids = disc + val
    cs_json = json.load(open(case_sets_path))
    pairs = P.load_pairs(pairs_path, cs_json)
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
    er = pd.read_parquet(os.path.join(od, "str_expert_rows.parquet"))
    er = er[(er.kind == "expert") & er.clean_active & er.case_id.isin(ids)][["case_id", "slot", "layer", "expert", "rescue", "vnorm"]]
    sw = pd.read_parquet(os.path.join(od, "str_sweep_rows.parquet"))
    sw = sw[(sw.kind == "layer") & sw.case_id.isin(ids)][["case_id", "slot", "layer", "rescue"]]
    single, vnorm = _single_dicts(rows, er)
    pop, layer_disc = _pop_and_layers(er, sw, disc)
    spec, _ = load_spec(MODELS[model]["repo"])
    st = bool(json.load(open(os.path.join(od, "run_meta.json"))).get("sweep", {}).get("special_tokens", True))
    t = Task("wino", model, str_run, spec.n_layers, spec.top_k, cases, rows, cand, weight, single, vnorm, pop, layer_disc, st)
    _check(t)
    return t


def _check(t: Task):
    for c in t.cases.case_id:
        assert len(t.cand[int(c)]) == t.K, (c, len(t.cand[int(c)]))
    miss = [r for r in t.rows.row_id if r not in t.single or len(t.single[r]) != t.K]
    assert not miss, f"rows without a full single-expert table: {miss[:5]} ({len(miss)})"


# ------------------------------------------------------------------------------------------------------------------
# orderings
# ------------------------------------------------------------------------------------------------------------------
def ordering(t: Task, rid: int, name: str, dla: Optional[dict] = None, noise_single: Optional[dict] = None) -> list:
    """Full ordering of the row's K candidates (most important first)."""
    c = int(t.rows.case_id.iloc[rid])
    cands = t.cand[c]
    if name == "pop":
        return sorted(cands, key=lambda p: (-t.pop.get(p, 0.0), p))
    if name == "oracle":
        sv = t.single[rid]
        return sorted(cands, key=lambda p: (-sv[p], p))
    if name == "layerwise":
        lrank = {l: i for i, l in enumerate(sorted(range(t.L), key=lambda l: (-t.layer_disc.get(l, 0.0), l)))}
        return sorted(cands, key=lambda p: (lrank[p[0]], -t.pop.get(p, 0.0), p))
    if name.startswith("rand"):
        i = int(name[4:])
        o = list(cands)
        random.Random(8000 + 10 * c + i).shuffle(o)
        return o
    if name == "weight":
        w = t.weight[c]
        return sorted(cands, key=lambda p: (-w[p], p))
    if name == "vnorm":
        v = t.vnorm[rid]
        return sorted(cands, key=lambda p: (-v[p], p))
    if name == "dla":
        v = dla[rid]
        return sorted(cands, key=lambda p: (-v[p], p))
    if name == "noise_oracle":  # most damaging single swap first (most negative noising effect)
        v = noise_single[rid]
        return sorted(cands, key=lambda p: (v[p], p))
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------------------------
# work items: (fam, order, dir, k, row_id, payload); payload = tuple of (l, e) pairs or an A0 kind string
# ------------------------------------------------------------------------------------------------------------------
def steps_of(t: Task, payload) -> tuple:
    L = t.L
    if isinstance(payload, str):
        if payload == "all_moe":
            return tuple((l, "layer", None) for l in range(L))
        if payload == "all_attn":
            return tuple((l, "attn_layer", None) for l in range(L))
        if payload == "all_block":
            return tuple((l, "block", None) for l in range(L))
        if payload == "int_moe":
            return tuple((l, "layer", None) for l in range(L - 4))
        if payload == "int_attn":
            return tuple((l, "attn_layer", None) for l in range(L - 4))
        raise ValueError(payload)
    by: dict[int, list[int]] = {}
    for l, e in payload:
        by.setdefault(int(l), []).append(int(e))
    return tuple((l, "coalition_set", tuple(sorted(by[l]))) for l in sorted(by))


def spawn_of(t: Task, item, SpawnSpec):
    fam, order, dr, k, rid, payload = item
    ci = t.case_index[int(t.rows.case_id.iloc[rid])]
    cr = t.nC + rid
    par, src = (cr, ci) if dr == "d" else (ci, cr)
    st = steps_of(t, payload)
    return SpawnSpec(st[0][0], par, src, "multi", steps=st, kl_ref=ci)


def set_str(payload) -> str:
    if isinstance(payload, str):
        return payload
    return ";".join(f"{l}.{e}" for l, e in sorted(payload))


def parse_set(s: str) -> tuple:
    if not s or "." not in s:
        return ()
    return tuple(sorted(tuple(int(x) for x in p.split(".")) for p in s.split(";")))


def curve_groups(t: Task, rows: list[int], orders, dr: str, fam: str, dla=None, noise_single=None) -> list:
    """One group per (row, ordering): the prefixes of the ordering at the k grid (k = K excluded; it is the shared
    'all_active' item)."""
    K = t.K
    grid = [k for k in k_grid(K) if k < K]
    out = []
    for rid in rows:
        for o in orders:
            od = ordering(t, rid, o, dla, noise_single)
            out.append([(fam, o, dr, k, rid, tuple(od[:k])) for k in grid])
    return out


def all_active_group(t: Task, rows: list[int], dr: str, fam: str) -> list:
    return [[(fam, "all_active", dr, t.K, rid, tuple(t.cand[int(t.rows.case_id.iloc[rid])])) for rid in rows]]


def prefix_groups(t: Task, rows: list[int], orders, fam: str = "shapley") -> list:
    """Full prefix sweeps k = 1..K-1 (denoising) of the given orderings, one group per (row, ordering)."""
    out = []
    for rid in rows:
        for o in orders:
            od = ordering(t, rid, o)
            out.append([(fam, o, "d", k, rid, tuple(od[:k])) for k in range(1, t.K)])
    return out


def exact_groups(t: Task, rows: list[int], n: int = EXACT_N) -> list:
    out = []
    for rid in rows:
        top = ordering(t, rid, "oracle")[:n]
        g = []
        for mask in range(1, 2 ** n):
            s = tuple(top[i] for i in range(n) if mask >> i & 1)
            g.append(("exact", "top10", "d", len(s), rid, s))
        out.append(g)
    return out


def single_noise_groups(t: Task, rows: list[int]) -> list:
    out = []
    for rid in rows:
        c = int(t.rows.case_id.iloc[rid])
        out.append([("single_noise", "single", "n", 1, rid, (p,)) for p in t.cand[c]])
    return out


def a0_groups(t: Task, rows: list[int]) -> list:
    return [[("a0", kind, dr, 0, rid, kind) for kind in A0_KINDS for dr in ("d", "n")] for rid in rows]


# ------------------------------------------------------------------------------------------------------------------
# adaptive state (greedy, beam)
# ------------------------------------------------------------------------------------------------------------------
@dataclass
class Greedy:
    rid: int
    pool: list
    S: list = field(default_factory=list)
    vals: list = field(default_factory=list)  # Delta after each step (same-pass values)
    resc: list = field(default_factory=list)  # rescue after each step
    done: bool = False
    stop_reason: str = ""

    def items(self) -> list:
        if self.done:
            return []
        base = tuple(self.S)
        out = []
        if not self.vals:  # step-1 value of the initial single (in-pass)
            out.append(("greedy", "eval", "d", len(base), self.rid, base))
        for p in self.pool:
            if p not in self.S:
                out.append(("greedy", "expand", "d", len(base) + 1, self.rid, base + (p,)))
        return out


@dataclass
class Beam:
    rid: int
    pool: list
    beams: list = field(default_factory=list)  # [(set tuple sorted, Delta)]
    best: dict = field(default_factory=dict)  # size -> (set, Delta)
    done: bool = False

    def items(self) -> list:
        if self.done:
            return []
        out = []
        seen = set()
        size = len(self.beams[0][0]) if self.beams else 0
        if size == 1 and 1 not in self.best:  # in-pass values of the initial singles
            for s, _ in self.beams:
                out.append(("beam", "eval", "d", 1, self.rid, s))
        for s, _ in self.beams:
            for p in self.pool:
                if p in s:
                    continue
                ns = tuple(sorted(s + (p,)))
                if ns in seen:
                    continue
                seen.add(ns)
                out.append(("beam", "expand", "d", len(ns), self.rid, ns))
        return out


# ------------------------------------------------------------------------------------------------------------------
# run state (persisted after every pass)
# ------------------------------------------------------------------------------------------------------------------
@dataclass
class State:
    queue: list  # list of groups (each a list of items), in order
    qpos: int = 0
    passes_done: int = 0
    greedy: dict = field(default_factory=dict)  # rid -> Greedy
    beam: dict = field(default_factory=dict)  # rid -> Beam
    ceiling: dict = field(default_factory=dict)  # rid -> all_moe rescue (denoising)
    dla_added: bool = False
    noise_oracle_added: bool = False
    notes: list = field(default_factory=list)


def save_state(path: str, st: State):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(st, f)
    os.replace(tmp, path)


def load_state(path: str) -> Optional[State]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def update_greedy(g: Greedy, df: pd.DataFrame, d_corrupt: float, ceiling: Optional[float]):
    """df: this pass's greedy rows of the row (columns kind2 = order, k, set, delta)."""
    ev = df[df.order == "eval"]
    if len(ev) and not g.vals:
        g.vals.append(float(ev.delta.iloc[0]))
        g.resc.append(float(ev.delta.iloc[0]) - d_corrupt)
        if _stop(g, ceiling):
            return
    ex = df[df.order == "expand"]
    if not len(ex):
        return
    j = int(np.argmax(ex.delta.values))  # first maximum in pool order
    new = parse_set(ex.set.iloc[j])
    added = [p for p in new if p not in g.S]
    assert len(added) == 1
    g.S.append(added[0])
    g.vals.append(float(ex.delta.iloc[j]))
    g.resc.append(float(ex.delta.iloc[j]) - d_corrupt)
    _stop(g, ceiling)


def _stop(g: Greedy, ceiling: Optional[float]) -> bool:
    if len(g.S) >= GREEDY_STEPS or len(g.S) >= len(g.pool):
        g.done, g.stop_reason = True, "steps"
    elif GREEDY_STOP is not None and ceiling is not None and ceiling > 0 and g.resc and g.resc[-1] >= GREEDY_STOP * ceiling:
        g.done, g.stop_reason = True, "ceiling"
    return g.done


def update_beam(b: Beam, df: pd.DataFrame):
    ev = df[df.order == "eval"]
    if len(ev):
        sets = [parse_set(s) for s in ev.set]
        vals = ev.delta.values.astype(float)
        b.beams = [(s, float(v)) for s, v in zip(sets, vals)]
        j = int(np.argmax(vals))
        b.best[1] = (sets[j], float(vals[j]))
    ex = df[df.order == "expand"]
    if not len(ex):
        return
    sets = [parse_set(s) for s in ex.set]
    vals = ex.delta.values.astype(float)
    order = sorted(range(len(sets)), key=lambda i: (-vals[i], sets[i]))[:BEAM_WIDTH]
    b.beams = [(sets[i], float(vals[i])) for i in order]
    size = len(b.beams[0][0])
    b.best[size] = b.beams[0]
    if size >= BEAM_MAX:
        b.done = True


# ------------------------------------------------------------------------------------------------------------------
# curve metrics
# ------------------------------------------------------------------------------------------------------------------
def auc_log(ks, r) -> float:
    """Trapezoid area of r(k) over log k from k = 1 to k = max(ks), divided by log(max k)."""
    ks = np.asarray(ks, dtype=float)
    r = np.asarray(r, dtype=float)
    if ks.max() <= 1:
        return float(r[0])
    x = np.log(ks)
    return float(np.trapezoid(r, x) / x.max())


def auc_lin(ks, r, K) -> float:
    """Trapezoid area of r over k/K from 0 (r = 0) to 1."""
    ks = np.concatenate([[0.0], np.asarray(ks, dtype=float)]) / K
    r = np.concatenate([[0.0], np.asarray(r, dtype=float)])
    return float(np.trapezoid(r, ks))


def k_at(ks, r, thr) -> float:
    """Smallest grid k with r(k) >= thr (inf if never)."""
    for k, v in zip(ks, r):
        if v >= thr:
            return float(k)
    return float("inf")


# ------------------------------------------------------------------------------------------------------------------
# analysis helpers
# ------------------------------------------------------------------------------------------------------------------
N_BOOT = 5000


def load_out(out: str) -> dict:
    """All saved rows of an add-back run, with same-pass rescue and the pass-0 reference drop per row."""
    od = out if os.path.isabs(out) else os.path.join(RESULTS, out)
    parts = sorted(f for f in os.listdir(od) if f.startswith("addback_rows_p"))
    rows = pd.concat([pd.read_parquet(os.path.join(od, f)) for f in parts], ignore_index=True)
    pf = pd.concat([pd.read_parquet(os.path.join(od, f)) for f in sorted(os.listdir(od)) if f.startswith("addback_prefill_p")],
                   ignore_index=True)
    p0 = pf[pf.pass_ == pf.pass_.min()]
    dclean0 = p0[p0.kind == "clean"].set_index("case_id").delta
    dcor0 = p0[p0.kind == "corrupt"].set_index("row_id").delta
    rows["drop_ref"] = rows.case_id.map(dclean0).values - rows.row_id.map(dcor0).values
    rows["rescue"] = rows.delta - rows.delta_corrupt
    rows["damage"] = rows.delta_clean - rows.delta
    out_d = {"rows": rows, "prefill": pf, "od": od,
             "state": load_state(os.path.join(od, "addback_state.pkl")),
             "meta": json.load(open(os.path.join(od, "run_meta.json")))}
    for name in ("direct", "dla"):
        p = os.path.join(od, f"addback_{name}.parquet")
        out_d[name] = pd.read_parquet(p) if os.path.exists(p) else None
    # pass-to-pass variation of the prefill rows (batch-composition numerics)
    sd_c = pf[pf.kind == "clean"].groupby("case_id").delta.agg(lambda x: x.max() - x.min())
    sd_j = pf[pf.kind == "corrupt"].groupby("row_id").delta.agg(lambda x: x.max() - x.min())
    sdc = pf[pf.kind == "clean"].groupby("case_id").delta.std()
    sdj = pf[pf.kind == "corrupt"].groupby("row_id").delta.std()
    out_d["prefill_range"] = {"clean_max": float(sd_c.max()), "clean_mean": float(sd_c.mean()),
                              "corrupt_max": float(sd_j.max()), "corrupt_mean": float(sd_j.mean()), "n_passes": int(pf.pass_.nunique()),
                              "clean_sd_median": float(sdc.median()), "corrupt_sd_median": float(sdj.median()),
                              "clean_sd_p99": float(sdc.quantile(0.99)), "corrupt_sd_p99": float(sdj.quantile(0.99))}
    return out_d


def boot_weights(units: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Unit-cluster bootstrap: returns (unique units u, count matrix W [n_boot, len(u)])."""
    u = np.unique(units)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    W = np.zeros((n_boot, len(u)), dtype=np.float32)
    np.add.at(W, (np.repeat(np.arange(n_boot), len(u)), idx.ravel()), 1.0)
    return u, W


def ratio_cols(num: pd.DataFrame, den: pd.Series, unit: pd.Series, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Ratio of means per column (num [cases x cols], den [cases]) with a unit-cluster bootstrap CI.
    Returns DataFrame index = columns, columns ratio, lo, hi, n."""
    units = unit.loc[num.index].values
    u, W = boot_weights(units, n_boot)
    pos = {x: i for i, x in enumerate(u)}
    M = np.zeros((len(u), len(num.index)), dtype=np.float32)
    M[[pos[x] for x in units], np.arange(len(units))] = 1.0  # unit x case membership
    N = num.values.astype(np.float64)
    ok = ~np.isnan(N)
    Nz = np.where(ok, N, 0.0)
    D = den.loc[num.index].values.astype(np.float64)
    out = []
    UN = M @ Nz  # [units, cols]
    UD = M @ (ok * D[:, None])  # drop restricted to cases with a value in the column
    BN, BD = W @ UN, W @ UD
    R = BN / BD
    full = Nz.sum(0) / (ok * D[:, None]).sum(0)
    lo, hi = np.nanpercentile(R, 2.5, axis=0), np.nanpercentile(R, 97.5, axis=0)
    return pd.DataFrame(dict(ratio=full, lo=lo, hi=hi, n=ok.sum(0)), index=num.columns)


def case_matrix(rows: pd.DataFrame, value: str, cols: list, donors: str = "mean", first_rows=None) -> pd.DataFrame:
    """[case x col] matrix of donor-aggregated values; cols = list of column keys present in rows (tuples)."""
    r = rows if donors == "mean" else rows[rows.row_id.isin(first_rows)]
    g = r.groupby(["case_id"] + cols)[value].mean()
    return g.unstack(cols)


def first_k(ks: np.ndarray, M: np.ndarray, thr) -> np.ndarray:
    """Per row of M [cases x len(ks)]: smallest k with M >= thr (thr scalar or per-row array); inf if never."""
    thr = np.broadcast_to(np.asarray(thr, dtype=float).reshape(-1, 1) if np.ndim(thr) else thr, M.shape)
    hit = M >= thr
    out = np.where(hit.any(1), ks[np.argmax(hit, axis=1)], np.inf)
    return out


def med_iqr(x) -> dict:
    x = np.asarray(x, dtype=float)
    fin = x[np.isfinite(x)]
    x = x[~np.isnan(x)]
    pc = (lambda q: float(np.percentile(x, q, method="inverted_cdf"))) if len(x) else (lambda q: float("nan"))
    return {"n": int(len(x)), "median": pc(50), "q25": pc(25), "q75": pc(75),
            "frac_finite": float(len(fin) / len(x)) if len(x) else float("nan")}
