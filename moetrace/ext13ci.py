"""ext13ci (Phase 5, agent p5-circuit): Phase 4's Direction 10 (attention heads + experts added back together, to full
repair at the final position) for every Phase-5 condition.

Conditions come from data/wino_ref/p5_conditions.json (written by p5-data): per condition and protocol the pair parquet
(ext7 pair contract), the case-set JSON, the base run dir results/p5_<cond>_<proto>/ (ext7 runners: str_sweep_routing,
str_expert_rows, case_sets.json), a status and the selected expert. Rows = the validation directed cases of family "main"
(one STR unit per directed case, d = 0 / 1 of a pair; Delta = logit(r) - logit(r') at the final token). This module
reuses moetrace.ext10_circuit (read-only) for everything that does not depend on the registry: candidate orderings,
pools, the greedy state machine, static groups, multi-step construction.

Differences to ext10 (all additive):
  * registry: condition/protocol keys "<cond>/<proto>", proto in {qwen3, mixtral_bos} (model keys qwen3 / mixtral);
  * tolerant task loader: the single-expert table of the base run may cover only some layers; missing (layer, expert)
    entries of a row are replaced by the row's expert DLA (step-1 pass) in the step-2 oracle ordering / pool, and the
    number of proxied entries is recorded;
  * bootstrap unit = the WinoGrande twin (HIT id of q_a; verb templates: the seed), so both attribute orders of a copy pair
    and all name fills of a template form one cluster; falls back to the pair;
  * step 1 also records the final position's attention distribution (DiagSpec.attn_final) of the clean and corrupted
    prefill rows and reduces it to the mass on position classes from the pair file: every pair of columns <X>_a_pos /
    <X>_b_pos gives <X>_clean (the side of the clean prompt: for d = 0 side a) and <X>_corr; single columns <Y>_pos give
    <Y>; plus str (the STR site), pos0 and final. Classes may overlap (each is the summed attention on its positions).

Outputs: step 1 results/p5_<cond>_<proto>_ci/ (ext10 step-1 file formats + head_attn_pNN.npz), step 2
results/p5_<proto>_ci/<group>/ (ext10 step-2 file formats; several conditions of one model share every pass).
"""
from __future__ import annotations

import json
import os
import re

os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")  # checkpoints (config / tokenizer) live on the NVMe

import numpy as np
import pandas as pd

from .models import MODELS, RESULTS
from . import ext8_addback as X
from . import ext10_circuit as C

ROOT = os.path.dirname(RESULTS)
COND_JSON = os.path.join(ROOT, "data", "wino_ref", "p5_conditions.json")
PROGRESS = os.path.join(ROOT, "logs", "PROGRESS.md")
SUMMARY = os.path.join(RESULTS, "ext13ci_summary.json")
TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")

PROTOS = ("qwen3", "mixtral_bos")
MODEL_OF = {"qwen3": "qwen3", "mixtral_bos": "mixtral"}
MODEL_LABEL = {"qwen3": "Qwen3-30B-A3B-Base", "mixtral_bos": "Mixtral-8x7B (BOS)"}
COND_ORDER = ("f1", "dir_f1", "dir_opt", "cp_nat", "cp_vt_orig", "cp_vt_neu", "vt_e2a_orig", "vt_e2a_neu", "vt_a2e_orig",
              "vt_a2e_neu", "f2e")
# step-2 groups in GPU order (plan "Phase 5 execution plan"); a group runs when all its runnable members have step 1;
# groups larger than the per-pass budget are split (scripts/ext13ci_chain.py)
GROUPS = (("g1", ("f1", "dir_f1", "dir_opt")), ("g2", ("cp_nat", "cp_vt_orig", "cp_vt_neu")),
          ("g3", ("vt_e2a_orig", "vt_e2a_neu", "vt_a2e_orig", "vt_a2e_neu")), ("g4", ("f2e",)))
COND_INFO = {
    "f1": dict(label="F1 attr→entity", direction="a2e", answer="copy (entity mention)", task=1, fmt="fill-in-the-blank"),
    "dir_f1": dict(label="dir: F1 on Phase-3 pairs", direction="a2e", answer="copy (entity mention)", task=2, fmt="fill-in-the-blank"),
    "dir_opt": dict(label="dir: option swap on the same pairs", direction="e2a", answer="generated", task=2, fmt="Phase 3"),
    "f2e": dict(label="F2E natural, blank at end", direction="a2e", answer="copy (entity mention)", task=3, fmt="natural sentence"),
    "vt_e2a_orig": dict(label="verb templates e→a, orig verb", direction="e2a", answer="generated", task=4, fmt="Phase 3"),
    "vt_e2a_neu": dict(label="verb templates e→a, neutral verb", direction="e2a", answer="generated", task=4, fmt="Phase 3"),
    "vt_a2e_orig": dict(label="verb templates a→e, orig verb", direction="a2e", answer="copy (entity mention)", task=5, fmt="fill-in-the-blank"),
    "vt_a2e_neu": dict(label="verb templates a→e, neutral verb", direction="a2e", answer="copy (entity mention)", task=5, fmt="fill-in-the-blank"),
    "cp_vt_orig": dict(label="copy prefix, templates orig", direction="e2a", answer="copyable (prefix attribute)", task=6, fmt="Phase 3 + prefix"),
    "cp_vt_neu": dict(label="copy prefix, templates neutral", direction="e2a", answer="copyable (prefix attribute)", task=6, fmt="Phase 3 + prefix"),
    "cp_nat": dict(label="copy prefix, natural Phase-3 pairs", direction="e2a", answer="copyable (prefix attribute)", task=6, fmt="Phase 3 + prefix"),
}
COPY_CONDS = ("cp_nat", "cp_vt_orig", "cp_vt_neu")
A2E_CONDS = ("f1", "dir_f1", "vt_a2e_orig", "vt_a2e_neu", "f2e")
ATTN_CONDS = COPY_CONDS + A2E_CONDS  # conditions whose top heads get the attention-mass readout in the section

# smoke / pre-neutral record runs (not Phase-5 conditions; outputs carry a suffix)
EXTRA = {
    ("F1pre", "qwen3"): dict(pairs="data/wino_ref/pairs_F1_train_xl_qwen3.parquet", case_sets="data/wino_ref/case_sets_F1_qwen3.json",
                             run="wino_ref_F1_qwen3", family="main", status={"cases": True, "base": True}),
    ("F1pre", "mixtral_bos"): dict(pairs="data/wino_ref/pairs_F1_train_xl_mixtral_bos.parquet",
                                   case_sets="data/wino_ref/case_sets_F1_mixtral_bos.json", run="wino_ref_F1_mixtral_bos",
                                   family="main", status={"cases": True, "base": True}),
}
COND_INFO["F1pre"] = dict(label="F1 pre-neutral (smoke)", direction="a2e", answer="copy (entity mention)", task=0, fmt="fill-in-the-blank")


def key_of(cond: str, proto: str) -> str:
    return f"{cond}/{proto}"


def parse_key(k: str) -> tuple[str, str]:
    c, p = k.split("/")
    return c, p


# ------------------------------------------------------------------------------------------------------------------
# registry
# ------------------------------------------------------------------------------------------------------------------
def _abs(p):
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _norm_entry(cond: str, proto: str, e: dict) -> dict:
    """Normalise one p5_conditions.json entry (tolerant to key spellings)."""
    g = lambda *ks, d=None: next((e[k] for k in ks if k in e and e[k] is not None), d)
    run = g("run", "run_dir", "out", d=f"p5_{cond}_{proto}")
    run = os.path.basename(str(run).rstrip("/")) if os.path.isabs(str(run)) or str(run).startswith("results/") else str(run)
    st = g("status", d={})
    if not isinstance(st, dict):
        st = {"base": bool(st)}
    fams = g("families", d=None)
    return dict(cond=cond, proto=proto, model=MODEL_OF[proto], pairs=_abs(g("pairs", "pair_file", "pairs_path")),
                case_sets=_abs(g("case_sets", "case_set_file", "case_sets_path")), run=run, family=g("family", d="main"),
                families=fams, status=st, behaviour_only=bool(g("behaviour_only", "behavior_only", d=False)) or cond == "mem",
                selected_expert=g("selected_expert"), pop_top10=g("pop_top10"), cluster=g("cluster", "cluster_col", "unit_col"),
                raw=e)


def load_conditions() -> dict:
    """(cond, proto) -> normalised entry, for every entry of p5_conditions.json (plus the EXTRA smoke entries)."""
    out = {k: _norm_entry(k[0], k[1], v) for k, v in EXTRA.items()}
    if not os.path.exists(COND_JSON):
        return out
    js = json.load(open(COND_JSON))
    if isinstance(js, dict) and "conditions" in js:
        js = js["conditions"]
    items = []
    if isinstance(js, list):
        items = [(e.get("cond"), e.get("proto"), e) for e in js]
    elif isinstance(js, dict):
        for k, v in js.items():
            if not isinstance(v, dict):
                continue
            if "/" in k:
                c, p = k.split("/", 1)
                items.append((c, p, v))
            elif any(p in v for p in PROTOS):
                for p in PROTOS:
                    if isinstance(v.get(p), dict):
                        items.append((k, p, v[p]))
            elif "proto" in v:
                items.append((v.get("cond", k), v["proto"], v))
    for c, p, e in items:
        if c and p in PROTOS:
            out[(c, p)] = _norm_entry(c, p, e)
    return out


def entry(cond: str, proto: str) -> dict:
    return load_conditions()[(cond, proto)]


def base_ready_lines() -> set:
    """{(cond, proto)} with an exact '[p5-data] BASE READY <cond>/<proto>' line in logs/PROGRESS.md."""
    out = set()
    if not os.path.exists(PROGRESS):
        return out
    for ln in open(PROGRESS, encoding="utf-8", errors="replace"):
        m = re.search(r"\[p5-data\] BASE READY ([A-Za-z0-9_]+)/([A-Za-z0-9_]+)", ln)
        if m:
            out.add((m.group(1), m.group(2)))
    return out


def step1_dir(cond: str, proto: str, suffix: str = "") -> str:
    return os.path.join(RESULTS, f"p5_{cond}_{proto}_ci{suffix}")


def step2_dir(proto: str, group: str, suffix: str = "") -> str:
    return os.path.join(RESULTS, f"p5_{proto}_ci{suffix}", group)


def step1_complete(cond: str, proto: str, suffix: str = "") -> bool:
    p = os.path.join(step1_dir(cond, proto, suffix), "heads_state.json")
    if not os.path.exists(p):
        return False
    st = json.load(open(p))
    return len(st["done"]) == len(st["chunks"]) and len(st["chunks"]) > 0


# ------------------------------------------------------------------------------------------------------------------
# task loading (ext8 Task, tolerant)
# ------------------------------------------------------------------------------------------------------------------
def _cluster(pairs: pd.DataFrame, col=None) -> tuple[dict, str]:
    """pair_idx -> cluster label (twin HIT id / seed / explicit column / pair)."""
    if col and col in pairs.columns:
        return dict(zip(pairs.pair_idx, pairs[col].astype(str))), col
    if "seed" in pairs.columns:
        return dict(zip(pairs.pair_idx, pairs.seed.astype(str))), "seed"
    if "q_a" in pairs.columns:
        return dict(zip(pairs.pair_idx, pairs.q_a.astype(str).str.rsplit("-", n=1).str[0])), "twin (q_a HIT id)"
    return dict(zip(pairs.pair_idx, pairs.pair_idx.astype(str))), "pair"


def load_task(cond: str, proto: str, family: str | None = None) -> X.Task:
    from . import ext7_pairs as P
    from .arch import load_spec
    e = entry(cond, proto)
    model = e["model"]
    od = os.path.join(RESULTS, e["run"])
    fam = family or e["family"] or "main"
    fams = P.load_families(e["run"])
    disc, val = [int(c) for c in fams[fam].get("discovery", [])], [int(c) for c in fams[fam]["validation"]]
    ids = disc + val
    cs_json = json.load(open(e["case_sets"]))
    pairs = P.load_pairs(e["pairs"], cs_json)
    dc = P.directed_cases(pairs, ids)
    clu, clu_src = _cluster(pairs, e.get("cluster"))
    labs = sorted(set(clu[dc[c].pair_idx] for c in ids))
    code = {l: i for i, l in enumerate(labs)}
    cases = pd.DataFrame(dict(case_id=ids, split=["discovery"] * len(disc) + ["validation"] * len(val),
                              unit=[code[clu[dc[c].pair_idx]] for c in ids], unit_label=[clu[dc[c].pair_idx] for c in ids],
                              pair_idx=[dc[c].pair_idx for c in ids], pair_id=[dc[c].pair_id for c in ids],
                              clean_ids=[list(dc[c].clean_ids) for c in ids], true_id=[int(dc[c].true_id) for c in ids],
                              foil_id=[int(dc[c].foil_id) for c in ids]))
    rows = pd.DataFrame(dict(row_id=np.arange(len(ids)), case_id=ids, slot=0, corrupt_ids=[list(dc[c].corrupt_ids) for c in ids]))
    rows["first"] = True
    rt = pd.read_parquet(os.path.join(od, "str_sweep_routing.parquet"))
    if "slot" in rt.columns:
        rt = rt[rt.slot == -1]
    rt = rt[rt.case_id.isin(ids)]
    cand, weight = {}, {}
    for c, g in rt.groupby("case_id"):
        keys = list(zip(g.layer.astype(int), g.expert.astype(int)))
        cand[int(c)] = sorted(keys)
        weight[int(c)] = dict(zip(keys, g.weight.astype(float)))
    miss = [c for c in ids if c not in cand]
    assert not miss, f"{cond}/{proto}: no clean routing for {len(miss)} cases (e.g. {miss[:3]})"
    single, vnorm = {}, {}
    ep = os.path.join(od, "str_expert_rows.parquet")
    if os.path.exists(ep):
        er = pd.read_parquet(ep, columns=["case_id", "slot", "layer", "kind", "expert", "rescue", "vnorm", "clean_active"])
        er = er[(er.kind == "expert") & er.clean_active & er.case_id.isin(ids)][["case_id", "slot", "layer", "expert", "rescue", "vnorm"]]
        single, vnorm = X._single_dicts(rows, er)
    spec, _ = load_spec(MODELS[model]["repo"])
    meta = json.load(open(os.path.join(od, "run_meta.json"))) if os.path.exists(os.path.join(od, "run_meta.json")) else {}
    stok = bool((meta.get("sweep") or {}).get("special_tokens", proto != "qwen3"))
    t = X.Task("wino", model, e["run"], spec.n_layers, spec.top_k, cases, rows, cand, weight, single, vnorm, {}, {}, stok)
    t.cluster_src = clu_src
    t.pairs = pairs
    t.dc = dc
    t.cond, t.proto = cond, proto
    n_full = sum(1 for r in rows.row_id if r in single and len(single[r]) == t.K)
    t.single_coverage = {"rows": int(len(rows)), "rows_full": int(n_full),
                         "entries": int(sum(len(single.get(r, {})) for r in rows.row_id)), "entries_needed": int(len(rows) * t.K)}
    return t


def val_rows(t: X.Task) -> list[int]:
    return C.val_rows(t)


# ------------------------------------------------------------------------------------------------------------------
# position classes for the attention-mass readout
# ------------------------------------------------------------------------------------------------------------------
def _jl(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        x = json.loads(x) if x.strip() else []
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    return [int(v) for v in x]


def position_columns(pairs: pd.DataFrame) -> tuple[list, list]:
    """(paired bases X with X_a_pos / X_b_pos, single columns Y_pos) present in the pair parquet."""
    cols = set(pairs.columns)
    paired = sorted({m.group(1) for c in cols for m in [re.match(r"^(.*)_a_pos$", c)] if m and f"{m.group(1)}_b_pos" in cols})
    used = {f"{b}_a_pos" for b in paired} | {f"{b}_b_pos" for b in paired}
    single = sorted(c[:-4] for c in cols if c.endswith("_pos") and c not in used and c not in ("str_pos", "opt_pos"))
    return paired, single


def class_names(pairs: pd.DataFrame) -> list[str]:
    paired, single = position_columns(pairs)
    return ["str"] + [f"{b}_{s}" for b in paired for s in ("clean", "corr")] + single + ["pos0", "final"]


def case_positions(t: X.Task, case_id: int) -> dict:
    """class name -> positions for one directed case."""
    x = t.dc[int(case_id)]
    r = t.pairs.loc[x.pair_idx]
    paired, single = position_columns(t.pairs)
    out = {"str": list(x.str_pos)}
    for b in paired:
        pa, pb = _jl(r[f"{b}_a_pos"]), _jl(r[f"{b}_b_pos"])
        out[f"{b}_clean"], out[f"{b}_corr"] = (pa, pb) if x.d == 0 else (pb, pa)
    for s in single:
        out[s] = _jl(r[f"{s}_pos"])
    out["pos0"] = [0]
    out["final"] = [x.T - 1]
    return out


# ------------------------------------------------------------------------------------------------------------------
# step-1 outputs and step-2 tasks
# ------------------------------------------------------------------------------------------------------------------
def load_heads(cond: str, proto: str, suffix: str = "") -> dict:
    od = step1_dir(cond, proto, suffix)
    return {"rows": C._cat(od, "head_rows_p"), "prefill": C._cat(od, "head_prefill_p"), "dla": C._cat(od, "head_dla_p"),
            "edla": C._cat(od, "expert_dla_p"), "meta": json.load(open(os.path.join(od, "run_meta.json")))}


def load_attn_mass(cond: str, proto: str, suffix: str = "") -> dict:
    """Concatenated head_attn_pNN.npz: mass [R, 2 (clean, corrupt), L, nH, C] float32, row_ids [R], classes [C]."""
    od = step1_dir(cond, proto, suffix)
    parts = sorted(f for f in os.listdir(od) if f.startswith("head_attn_p") and f.endswith(".npz"))
    if not parts:
        return {}
    zs = [np.load(os.path.join(od, f), allow_pickle=False) for f in parts]
    cls = [str(c) for c in zs[0]["classes"]]
    for z in zs[1:]:
        assert [str(c) for c in z["classes"]] == cls
    return {"mass": np.concatenate([z["mass"].astype(np.float32) for z in zs]), "row_ids": np.concatenate([z["row_ids"] for z in zs]),
            "classes": cls, "n_pos": np.concatenate([z["n_pos"] for z in zs])}


def load_ctask(cond: str, proto: str, off: int = 0, limit: int = 0, suffix: str = "") -> C.CTask:
    """ext10 CTask for one condition (key '<cond>/<proto>'), single-expert oracle values with expert-DLA fallback."""
    t = load_task(cond, proto)
    rows = val_rows(t)
    if limit:
        rows = rows[:limit]
    cases = list(dict.fromkeys(int(t.rows.case_id.iloc[r]) for r in rows))
    ct = C.CTask(key_of(cond, proto), t, rows, cases, off=off, L=t.L)
    ct.ci = {c: off + i for i, c in enumerate(cases)}
    ct.cr = {r: off + len(cases) + j for j, r in enumerate(rows)}
    h = load_heads(cond, proto, suffix)
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
    n_proxy = 0
    for r in rows:
        assert r in ct.hs and r in ct.hd and r in ct.ed, f"{ct.key}: step-1 data missing for row {r}"
        sv = t.single.get(r, {}) if t.single else {}
        d = {}
        for p in t.cand[ct.case_of(r)]:
            if p in sv:
                d[p] = float(sv[p])
            else:
                d[p] = float(ct.ed[r][p])
                n_proxy += 1
        ct.es[r] = d
    ct.n_proxy = n_proxy
    return ct


def ext10_reference() -> dict:
    """Phase-4 Direction-10 reference values (results/ext10_circuit_summary.json, step 2) for the overview panel."""
    p = os.path.join(RESULTS, "ext10_circuit_summary.json")
    if not os.path.exists(p):
        return {}
    s = json.load(open(p)).get("step2", {})
    out = {}
    for k, v in s.items():
        cv = v["curves"]["greedy_mix"]
        out[k] = {"label": v["label"], "task": v["task"], "r20": cv["r20"]["ratio"], "r20_lo": cv["r20"]["lo"], "r20_hi": cv["r20"]["hi"],
                  "heads_first10": cv["composition"]["10"]["mean_heads"], "k80": cv["k_q"]["0.8"],
                  "all_moe": v["ceilings"]["all_moe"]["ratio"]}
    return out
