"""ext13ab (Phase 5, agent p5-addback): expert add-back / deletion curves (Phase 3's Direction 8 machinery) for every
Phase-5 condition. No engine change; moetrace/ext8_addback.py (orderings, greedy, loaders, curve metrics) is reused
read-only.

Task registry. One add-back task per (condition, protocol, case-set family) listed in data/wino_ref/p5_conditions.json
(written by agent p5-data) whose base runs are ready (status.base true) and that is not behaviour-only. The task is
ext8's WinoGrande STR task (`ext8_addback.load_wino`) on the condition's run dir results/p5_<cond>_<proto>/ (ext6 /
ext7 schema: str_sweep_routing slot -1 = clean, str_expert_rows all layers, str_sweep_rows MoE layers; case_sets.json
families) with the condition's pair parquet and case-set JSON. Family "main" -> results/p5_<cond>_<proto>_ab/,
family "rep" (replication, where the pool allows) -> results/p5_<cond>_<proto>_rep_ab/. Both are written in exactly
the ext8 file format (addback_rows_pNN.parquet, addback_prefill_pNN.parquet, addback_state.pkl, addback_dla.parquet,
addback_direct.parquet, run_meta.json), so ext8_addback.load_out and scripts/ext8_addback_analyze.analyse read them.

Work per task (the Phase-3 ext8 set without beam / exact top-10 / Shapley):
  A0 ceilings for every row (discovery + validation), add-back and deletion direction: all MoE, all attention, both,
     MoE <= L-5, attention <= L-5;
  A1 static add-back orderings on the validation rows: oracle, pop, layerwise, dla, vnorm, weight, rand0-4 (ext8 k grid)
     + the all-clean-active endpoint;
  A2 adaptive greedy, 15 steps, pools as ext8 (Qwen3 the row's top-32 singles, Mixtral all 64);
  A3 deletion: noising singles (Qwen3 the first 128 validation rows as Phase 3, Mixtral all validation rows), deletion
     curves pop / oracle / layerwise / rand0-1 / dla / noise_oracle + the all-clean-active endpoint.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import ext8_addback as X
from .models import RESULTS

ROOT = "/home/ubuntu/MOE"
COND_JSON = os.path.join(ROOT, "data/wino_ref/p5_conditions.json")
# GPU / plan order of the conditions (RESEARCH_PLAN.md "Phase 5 execution plan")
PLAN_ORDER = ("f1", "dir_f1", "dir_opt", "cp_nat", "cp_vt_orig", "cp_vt_neu", "vt_e2a_orig", "vt_e2a_neu", "vt_a2e_orig",
              "vt_a2e_neu", "f2e")
BEHAVIOUR_ONLY = ("mem", "f2r")
PROTO_MODEL = {"qwen3": "qwen3", "mixtral_bos": "mixtral"}
MODEL_PROTO = {"qwen3": "qwen3", "mixtral": "mixtral_bos"}
MODEL_LABEL = {"qwen3": "Qwen3-30B-A3B-Base", "mixtral": "Mixtral-8x7B (BOS)"}
POOL = {"qwen3": 32, "mixtral": 64}
NOISE_ROWS = {"qwen3": 128, "mixtral": 0}  # 0 = all validation rows
COND_LABEL = {
    "f1": "attr->entity, fill-in-the-blank (F1)",
    "dir_f1": "direction: F1 on the Phase-3 twins",
    "dir_opt": "direction: option swap on the same twins",
    "f2e": "attr->entity, natural sentence (F2E)",
    "vt_e2a_orig": "verb templates ent->attr, orig verb",
    "vt_e2a_neu": "verb templates ent->attr, neutral verb",
    "vt_a2e_orig": "verb templates attr->entity (F1), orig verb",
    "vt_a2e_neu": "verb templates attr->entity (F1), neutral verb",
    "cp_vt_orig": "copy prefix, templates, orig verb",
    "cp_vt_neu": "copy prefix, templates, neutral verb",
    "cp_nat": "copy prefix, Phase-3 twins",
    "F1pre": "F1 pre-neutral (smoke)",
}
# candidate cluster columns of the pair parquets (seed-clustered CIs for template conditions)
CLUSTER_COLS = ("seed_id", "seed", "seed_idx", "template_id", "template", "seed_key")


@dataclass
class TaskSpec:
    key: str  # registry key, e.g. "f1_qwen3" / "f1_qwen3_rep"
    cond: str
    proto: str
    model: str
    family: str  # case-set family of the source run ("main" | "rep")
    src: str  # source run dir name under results/
    pairs: str  # pair parquet (absolute)
    case_sets: str  # case-set JSON (absolute)
    out: str  # add-back run dir name under results/
    pool: int
    noise_rows: int
    limit: Optional[tuple] = None  # (n_discovery, n_validation) directed cases (smoke tests)
    cluster_col: Optional[str] = None  # pair-parquet column for seed / template clustering
    extra: dict = dataclasses.field(default_factory=dict)

    @property
    def label(self) -> str:
        return MODEL_LABEL[self.model]

    @property
    def tlabel(self) -> str:
        return COND_LABEL.get(self.cond, self.cond) + (" [replication]" if self.family == "rep" else "")


# ------------------------------------------------------------------------------------------------------------------
# p5_conditions.json (written by p5-data)
# ------------------------------------------------------------------------------------------------------------------
def _abs(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _get(d: dict, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def cond_entries(path: str = COND_JSON) -> list[dict]:
    """Flat list of {cond, proto, ...entry} from the conditions JSON (nested {cond: {proto: entry}} under an optional
    'conditions' key, or a list of entries with cond / proto fields)."""
    if not os.path.exists(path):
        return []
    js = json.load(open(path))
    body = js.get("conditions", js) if isinstance(js, dict) else js
    out = []
    if isinstance(body, list):
        for e in body:
            out.append(dict(e))
        return out
    for cond, v in body.items():
        if cond.startswith("_") or not isinstance(v, dict):
            continue
        for proto, e in v.items():
            if proto in PROTO_MODEL and isinstance(e, dict):
                out.append({"cond": cond, "proto": proto, **e})
    return out


def _status(e: dict, k: str) -> bool:
    s = e.get("status", {})
    if isinstance(s, dict):
        return bool(s.get(k))
    return False


def _behaviour_only(e: dict) -> bool:
    if e.get("cond") in BEHAVIOUR_ONLY:
        return True
    return bool(_get(e, "behaviour_only", "behavior_only", default=False)) or e.get("kind") == "behaviour"


def _families_of_run(src: str) -> dict:
    p = os.path.join(RESULTS, src, "case_sets.json")
    if not os.path.exists(p):
        return {}
    return {k: v for k, v in json.load(open(p)).items() if not k.startswith("_")}


def _expert_cases(src: str) -> set:
    p = os.path.join(RESULTS, src, "str_expert_rows.parquet")
    if not os.path.exists(p):
        return set()
    return set(pd.read_parquet(p, columns=["case_id"]).case_id.unique().tolist())


def _cluster_col(pairs: str) -> Optional[str]:
    try:
        cols = pd.read_parquet(pairs).columns
    except Exception:
        return None
    for c in CLUSTER_COLS:
        if c in cols:
            return c
    return None


def spec_from_entry(e: dict, family: str = "main") -> TaskSpec:
    cond, proto = e["cond"], e["proto"]
    model = PROTO_MODEL[proto]
    src = _get(e, "run", "run_dir", "out", default=f"p5_{cond}_{proto}")
    src = os.path.basename(str(src).rstrip("/"))
    pairs = _abs(_get(e, "pairs", "pair_file", "pairs_path"))
    cs = _abs(_get(e, "case_sets", "case_set_file", "case_sets_path"))
    key = f"{cond}_{proto}" + ("_rep" if family == "rep" else "")
    out = f"p5_{cond}_{proto}" + ("_rep" if family == "rep" else "") + "_ab"
    cl = _get(e, "cluster_col", "seed_col")
    if cl is None and pairs and os.path.exists(pairs) and (cond.startswith("vt_") or cond.startswith("cp_vt")):
        cl = _cluster_col(pairs)
    return TaskSpec(key, cond, proto, model, family, src, pairs, cs, out, POOL[model], NOISE_ROWS[model], cluster_col=cl,
                    extra={"entry": {k: v for k, v in e.items() if not isinstance(v, (dict, list)) or k in ("status", "families", "splits")}})


def registry(path: str = COND_JSON, ready_only: bool = True, check_rep: bool = True) -> dict[str, TaskSpec]:
    """Add-back tasks of every non-behaviour-only condition (status.base true unless ready_only=False)."""
    out = {}
    for e in cond_entries(path):
        if "cond" not in e or "proto" not in e or e["proto"] not in PROTO_MODEL or _behaviour_only(e):
            continue
        if ready_only and not _status(e, "base"):
            continue
        s = spec_from_entry(e, "main")
        out[s.key] = s
        if check_rep:
            fams = _families_of_run(s.src)
            rep = fams.get("rep", {})
            if rep.get("discovery") and rep.get("validation"):
                have = _expert_cases(s.src)
                need = set(int(c) for c in rep["discovery"]) | set(int(c) for c in rep["validation"])
                if need <= have:
                    r = spec_from_entry(e, "rep")
                    out[r.key] = r
    return out


def smoke_spec(model: str = "qwen3", n_disc: int = 16, n_val: int = 16) -> TaskSpec:
    """Smoke test on the pre-neutral F1 run (results/wino_ref_F1_<proto>)."""
    proto = MODEL_PROTO[model]
    return TaskSpec(f"F1pre_{proto}_smoke", "F1pre", proto, model, "main", f"wino_ref_F1_{proto}",
                    _abs(f"data/wino_ref/pairs_F1_train_xl_{proto}.parquet"), _abs(f"data/wino_ref/case_sets_F1_{proto}.json"),
                    f"p5_F1pre_{proto}_ab_smoke", POOL[model], 8, limit=(n_disc, n_val))


def get_spec(key: str, path: str = COND_JSON) -> TaskSpec:
    if key.endswith("_smoke"):
        proto = key[len("F1pre_"):-len("_smoke")]
        return smoke_spec(PROTO_MODEL[proto])
    reg = registry(path, ready_only=False)
    return reg[key]


# ------------------------------------------------------------------------------------------------------------------
# task loading
# ------------------------------------------------------------------------------------------------------------------
def subset_task(t: X.Task, ids: list[int]) -> X.Task:
    """Restrict a Task to the given case ids (row ids renumbered 0..R-1; pop / layer_disc kept from the full task)."""
    ids = set(int(c) for c in ids)
    cases = t.cases[t.cases.case_id.isin(ids)].reset_index(drop=True)
    rows = t.rows[t.rows.case_id.isin(ids)].reset_index(drop=True).copy()
    old = rows.row_id.astype(int).tolist()
    rows["row_id"] = np.arange(len(rows))
    single = {i: t.single[o] for i, o in enumerate(old)}
    vnorm = {i: t.vnorm[o] for i, o in enumerate(old)}
    cand = {c: v for c, v in t.cand.items() if c in ids}
    weight = {c: v for c, v in t.weight.items() if c in ids}
    return X.Task(t.name, t.model, t.run, t.L, t.topk, cases, rows, cand, weight, single, vnorm, t.pop, t.layer_disc, t.special_tokens)


def load_task(spec: TaskSpec) -> X.Task:
    t = X.load_wino(spec.model, spec.src, spec.pairs, spec.case_sets, family=spec.family)
    if spec.limit:
        nd, nv = spec.limit
        c = t.cases
        ids = c[c.split == "discovery"].case_id.tolist()[:nd] + c[c.split == "validation"].case_id.tolist()[:nv]
        t = subset_task(t, ids)
    return t


def pair_table(spec: TaskSpec, t: X.Task) -> pd.DataFrame:
    """Per evaluation-relevant case: pair_idx, pair_id, d, names flag, cluster (seed) and every scalar pair column."""
    from . import ext7_pairs as P
    cs = json.load(open(spec.case_sets))
    pairs = P.load_pairs(spec.pairs, cs)
    rows = []
    for c, pi, sp in zip(t.cases.case_id, t.cases.unit, t.cases.split):
        r = pairs.loc[int(pi)]
        d = dict(case_id=int(c), pair_idx=int(pi), d=int(c) % 2, split=sp, pair_id=r.pair_id)
        for k in pairs.columns:
            v = r[k]
            if k in ("ids_a", "ids_b", "ids_a_l", "ids_b_l", "str_pos_l", "pair_idx", "pair_id") or isinstance(v, (list, np.ndarray)):
                continue
            d[k] = v
        rows.append(d)
    df = pd.DataFrame(rows)
    if spec.cluster_col and spec.cluster_col in df.columns:
        df["cluster"] = df[spec.cluster_col].astype(str)
    else:
        df["cluster"] = df.pair_idx.astype(str)
    return df


# ------------------------------------------------------------------------------------------------------------------
# work queues (ext8 items: (fam, order, dir, k, row_id, payload))
# ------------------------------------------------------------------------------------------------------------------
def eval_rows(t: X.Task) -> list[int]:
    split = t.rows.case_id.map(dict(zip(t.cases.case_id, t.cases.split)))
    return t.rows.row_id[split == "validation"].tolist()


def noise_rows_of(t: X.Task, n: int) -> list[int]:
    ev = eval_rows(t)
    return ev[:n] if n else ev


def static_queue(t: X.Task, n_noise: int) -> list:
    """ext8's base queue without exact / Shapley / beam: A0 (all rows), A1 static orderings (not dla) + all_active (d, n),
    noising singles, A3 deletion curves (pop, oracle, layerwise, rand0-1)."""
    ev = eval_rows(t)
    q = []
    q += X.a0_groups(t, t.rows.row_id.tolist())
    q += X.curve_groups(t, ev, [o for o in X.A1_ORDERS if o != "dla"], "d", "a1")
    q += X.all_active_group(t, ev, "d", "a1") + X.all_active_group(t, ev, "n", "a3")
    q += X.single_noise_groups(t, noise_rows_of(t, n_noise))
    q += X.curve_groups(t, ev, [o for o in X.A3_ORDERS if o not in ("dla", "noise_oracle")], "n", "a3")
    return q


def dla_queue(t: X.Task, od: str) -> list:
    dla_df = pd.read_parquet(os.path.join(od, "addback_dla.parquet"))
    dla = {int(r_): dict(zip(zip(g.layer.astype(int), g.expert.astype(int)), g.dla)) for r_, g in dla_df.groupby("row_id")}
    ev = eval_rows(t)
    return X.curve_groups(t, ev, ["dla"], "d", "a1", dla=dla) + X.curve_groups(t, ev, ["dla"], "n", "a3", dla=dla)


def load_rows(od: str, fam: Optional[str] = None) -> pd.DataFrame:
    parts = sorted(f for f in os.listdir(od) if f.startswith("addback_rows_p"))
    out = []
    for f in parts:
        x = pd.read_parquet(os.path.join(od, f))
        if fam is not None:
            x = x[x.fam == fam]
        out.append(x)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def noise_oracle_queue(t: X.Task, od: str) -> tuple[list, int]:
    sn = load_rows(od, fam="single_noise")
    ns = {}
    for r_, g in sn.groupby("row_id"):
        ns[int(r_)] = {X.parse_set(s)[0]: float(v) for s, v in zip(g.set, g.delta - g.delta_clean)}
    return X.curve_groups(t, sorted(ns), ["noise_oracle"], "n", "a3", noise_single=ns), len(ns)


def run_complete(out: str) -> bool:
    p = os.path.join(RESULTS, out, "run_meta.json")
    return os.path.exists(p) and bool(json.load(open(p)).get("complete"))
