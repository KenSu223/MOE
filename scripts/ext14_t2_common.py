"""Phase-6 tier-2 plumbing (agent p6-analysis).

Tier 2 = the full Phase-4/5 set on selected cells: expert singles + two-stage selection, add-back (greedy / oracle /
random), mixed head + expert greedy, knockout with matched controls, writer DLA. The Phase-5 machinery
(moetrace/ext13ab.py, ext13ci.py, ext13ko.py, ext13wr.py and their scripts) is reused UNCHANGED: this module patches the
modules' registry functions and output paths in-process, so they read the Phase-6 cells and write into the cell's tier-2
dir, then the Phase-5 script's main() is called (scripts/ext14_t2_run.py).

Tier-2 dir results/p6_<cell>_<proto>_t2/ ("mirror" of p6-data's base run results/p6_<cell>_<proto>/):
  symlinks   the base run's sweep / routing / cases / direct / joint / DLA / head / grid files (read-only inputs)
  copies     case_sets.json, run_meta.json (the expert runner adds its own "expert" key here)
  own        str_expert_rows.parquet, str_expert_prefill.parquet (expert singles, scripts/ext7_wino_expert.py --out <t2>),
             selection.json (two-stage selection + population top-10, scripts/ext14_t2_select.py),
             ab/ (add-back, ext8 format), ci/ (circuit step 1), ci2/ (circuit step 2, one group per cell), ko/ (knockout),
             wr/ (writer pass), t2_summary.json (scripts/ext14_t2_analyze.py), .done_<stage> markers.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = "/home/ubuntu/MOE"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")

from moetrace.models import RESULTS  # noqa: E402
from moetrace import ext14_analysis as XA  # noqa: E402

# tier-2 cells per protocol (RESEARCH_PLAN Phase 6 cell x model matrix; Stage 2: Qwen2-57B e1_a2e / e3_a2e)
T2_CELLS = {"qwen3": ("e1_a2e", "e1_e2a"), "mixtral_bos": ("e1_a2e", "e1_e2a"), "qwen2_57b": ("e1_a2e", "e3_a2e")}
T2_MODEL = {"qwen3": "qwen3", "mixtral_bos": "mixtral", "qwen2_57b": "qwen2_57b"}
LINK_FILES = ("str_sweep_rows.parquet", "str_sweep_routing.parquet", "sweep_cases.parquet", "str_sweep_summary.json", "direct_split.parquet",
              "joint_rows.parquet", "dla_rows.parquet", "dla_cases.parquet", "head_rows.parquet", "head_positions.parquet", "head_prefill.parquet",
              "head_attn_final.npz", "str_grid_w1_rows.parquet", "str_grid_w1_prefill.parquet", "base_summary.json")
# extra fixed targets for the knockout cross-task controls (Phase-5 attribute -> entity experts)
EXTRA_CF = {"qwen3": [(44, 69), (42, 115), (37, 32)], "mixtral_bos": [(19, 2), (21, 1), (18, 1), (20, 1)]}


SUFFIX = os.environ.get("EXT14_T2_SUFFIX", "")  # smoke tests: "_smoke" -> results/p6_<cell>_<proto>_t2_smoke
LIMIT = os.environ.get("EXT14_T2_LIMIT")  # smoke tests: "n_disc,n_val" directed cases for the add-back tasks


def t2_run(cell: str, proto: str) -> str:
    return f"p6_{cell}_{proto}_t2{SUFFIX}"


def t2_dir(cell: str, proto: str) -> str:
    return os.path.join(RESULTS, t2_run(cell, proto))


def done(cell: str, proto: str, stage: str) -> bool:
    return os.path.exists(os.path.join(t2_dir(cell, proto), f".done_{stage}"))


def mark(cell: str, proto: str, stage: str):
    open(os.path.join(t2_dir(cell, proto), f".done_{stage}"), "w").close()


def base_entry(cell: str, proto: str) -> dict:
    e = XA.entry_of(cell, proto)
    if e is None:
        raise KeyError(f"{cell}/{proto} not in {XA.REG_JSON}")
    return e


POOL_SEED = 2026  # coordinator rule for tier 2 on a pooled cell (2026-10-16)


def pooled_split(e: dict) -> dict:
    """Pooled cell (< 64 validation pairs): all pairs of the case set split 50 / 50 by src_pair_id with
    random.Random(2026); selection half -> 'discovery', evaluation half -> 'validation'."""
    import random
    import pandas as pd
    cs = json.load(open(XA._abs(e["case_sets"])))
    pio = cs["pair_idx_of"]
    allp = sorted({int(i) for v in cs.get("pairs", {}).values() for i in v}) or sorted({int(c) // 2 for v in cs["directed"].values() for c in v})
    pf = pd.read_parquet(XA._abs(e["pair_file"]), columns=["pair_id", "src_pair_id"]).drop_duplicates("pair_id")
    src_of_idx = {int(pio[p]): str(s) for p, s in zip(pf.pair_id, pf.src_pair_id) if p in pio}
    twins = sorted({src_of_idx[i] for i in allp})
    rng = random.Random(POOL_SEED)
    rng.shuffle(twins)
    sel = set(twins[: len(twins) // 2])
    disc = [i for i in allp if src_of_idx[i] in sel]
    val = [i for i in allp if src_of_idx[i] not in sel]
    return {"discovery": disc, "validation": val, "n_twins": len(twins), "seed": POOL_SEED}


def is_pooled(cell: str, proto: str) -> bool:
    try:
        return bool(base_entry(cell, proto).get("pooled"))
    except KeyError:
        return False


def prepare(cell: str, proto: str) -> str:
    """Create / refresh the tier-2 mirror dir (idempotent). Pooled cells get the 50 / 50 pooled split (selection half =
    discovery, evaluation half = validation) in the mirror's case_sets.json and a mirror-local case-set JSON
    (case_sets_t2.json, registry format) used by the knockout / writer item tables."""
    e = base_entry(cell, proto)
    src = os.path.join(RESULTS, XA.run_name(e))
    od = t2_dir(cell, proto)
    os.makedirs(od, exist_ok=True)
    for f in LINK_FILES:
        s, d = os.path.join(src, f), os.path.join(od, f)
        if os.path.exists(s) and not os.path.lexists(d):
            os.symlink(os.path.relpath(s, od), d)
    cs = os.path.join(od, "case_sets.json")
    if e.get("pooled"):
        sp = pooled_split(e)
        dirc = {k: [2 * i + d for i in sp[k] for d in (0, 1)] for k in ("discovery", "validation")}
        if not os.path.exists(cs):
            json.dump({"main": dirc, "_source": {"case_sets": XA._abs(e["case_sets"]), "pairs": XA._abs(e["pair_file"])},
                       "_pooled_split": {"seed": POOL_SEED, "n_twins": sp["n_twins"], "n_selection_pairs": len(sp["discovery"]),
                                         "n_evaluation_pairs": len(sp["validation"]),
                                         "rule": "pooled cell: 50 / 50 split by src_pair_id (random.Random(2026)); selection on 'discovery', evaluation on 'validation'"}},
                      open(cs, "w"), indent=1)
        cst = os.path.join(od, "case_sets_t2.json")
        if not os.path.exists(cst):
            reg_cs = json.load(open(XA._abs(e["case_sets"])))
            reg_cs["pairs"] = {"discovery": sp["discovery"], "validation": sp["validation"]}
            reg_cs["directed"] = dirc
            reg_cs["note"] = (reg_cs.get("note", "") + " | tier-2 pooled split (p6-analysis): 50 / 50 by src_pair_id, random.Random(2026)").strip()
            json.dump(reg_cs, open(cst, "w"), indent=1)
    elif not os.path.exists(cs):
        shutil.copyfile(os.path.join(src, "case_sets.json"), cs)
    mp = os.path.join(od, "run_meta.json")
    bm = json.load(open(os.path.join(src, "run_meta.json"))) if os.path.exists(os.path.join(src, "run_meta.json")) else {}
    cur = json.load(open(mp)) if os.path.exists(mp) else {}
    for k, v in bm.items():  # base keys refreshed, own keys (expert, t2) kept
        if k not in ("expert",) and not k.startswith("t2"):
            cur[k] = v
    cur["t2"] = {"cell": cell, "proto": proto, "base_run": XA.run_name(e), "pair_file": e.get("pair_file"), "case_sets": e.get("case_sets"),
                 "agent": "p6-analysis", "note": "tier-2 mirror of the base run (symlinked inputs, own expert / add-back / circuit / knockout outputs)"}
    json.dump(cur, open(mp, "w"), indent=1, default=str)
    return od


def selection(cell: str, proto: str) -> dict | None:
    p = os.path.join(t2_dir(cell, proto), "selection.json")
    return json.load(open(p)) if os.path.exists(p) else None


def direction_of(cell: str) -> str:
    return "a2e" if cell.endswith("a2e") or "a2e" in cell else "e2a"


def unit_col(pair_file: str | None) -> str | None:
    import pandas as pd
    if not pair_file or not os.path.exists(pair_file):
        return None
    cols = pd.read_parquet(pair_file).columns
    for c in ("src_pair_id", "seed", "seed_id"):
        if c in cols:
            return c
    return None


def t2_entry(cell: str, proto: str) -> dict:
    """Registry entry in the Phase-5 p5_conditions.json layout for the ext13 modules (run = the tier-2 mirror)."""
    e = base_entry(cell, proto)
    sel = selection(cell, proto) or {}
    pf = XA._abs(e.get("pair_file") or e.get("pairs"))
    out = {"pair_file": os.path.relpath(pf, ROOT) if pf else None, "case_sets": e.get("case_sets"), "run_dir": f"results/{t2_run(cell, proto)}",
           "status": {"cases": True, "base": True}, "behaviour_only": False, "families": ["main"], "direction": direction_of(cell),
           "model": T2_MODEL[proto], "proto": proto, "task": cell}
    if "selected_expert" in sel:
        out["selected_expert"] = sel["selected_expert"]
    if sel.get("pop_top10"):
        out["pop_top10"] = sel["pop_top10"]
    uc = unit_col(pf)
    if uc:
        out["unit_col"] = uc
        out["cluster_col"] = uc
    cst = os.path.join(t2_dir(cell, proto), "case_sets_t2.json")
    if e.get("pooled") and os.path.exists(cst):
        out["case_sets"] = os.path.relpath(cst, ROOT)
        out["pooled_split"] = True
    return out


def t2_registry(protos=None, need_selection: bool = False) -> dict:
    """{cell: {proto: entry}} for the tier-2 cells whose mirror dir exists (and selection.json, if asked)."""
    reg = {}
    for proto, cells in T2_CELLS.items():
        if protos and proto not in protos:
            continue
        for c in cells:
            if not os.path.exists(os.path.join(t2_dir(c, proto), "case_sets.json")):
                continue
            if need_selection and selection(c, proto) is None:
                continue
            try:
                reg.setdefault(c, {})[proto] = t2_entry(c, proto)
            except KeyError:
                continue
    return reg


ALL_T2 = tuple(dict.fromkeys(c for cs in T2_CELLS.values() for c in cs))
TABRAW = os.path.join(RESULTS, "tables", "ext14_t2_raw")
FIGRAW = os.path.join(RESULTS, "figures", "ext14_t2_raw")


# ------------------------------------------------------------------------------------------------------------------
# patches
# ------------------------------------------------------------------------------------------------------------------
def patch_ab():
    from moetrace import ext13ab as AB
    if getattr(AB, "_p6", False):
        return AB
    AB.PLAN_ORDER = ALL_T2
    AB.COND_JSON = "(p6 tier-2 registry, scripts/ext14_t2_common.py)"
    for p, m in T2_MODEL.items():
        AB.PROTO_MODEL.setdefault(p, m)
        AB.MODEL_PROTO.setdefault(m, p)
    AB.MODEL_LABEL.setdefault("qwen2_57b", "Qwen2-57B-A14B")
    AB.POOL.setdefault("qwen2_57b", 32)
    AB.NOISE_ROWS.setdefault("qwen2_57b", 128)
    for c in ALL_T2:
        AB.COND_LABEL[c] = XA.cell_label(c)

    def cond_entries(path=None):
        out = []
        for c, v in t2_registry().items():
            for p, e in v.items():
                out.append({"cond": c, "proto": p, **e})
        return out
    orig = AB.spec_from_entry

    def spec_from_entry(e, family="main"):
        s = orig(e, family)
        s.out = f"{t2_run(e['cond'], e['proto'])}/ab" + ("_rep" if family == "rep" else "")
        if LIMIT:
            s.limit = tuple(int(x) for x in LIMIT.split(","))
        return s
    AB.cond_entries = cond_entries
    AB.spec_from_entry = spec_from_entry
    AB._p6 = True
    return AB


def patch_ci():
    from moetrace import ext13ci as CI
    if getattr(CI, "_p6", False):
        return CI
    CI.PROTOS = tuple(T2_CELLS)
    for p, m in T2_MODEL.items():
        CI.MODEL_OF.setdefault(p, m)
        CI.MODEL_LABEL.setdefault(p, XA.LABEL.get(p, p))
    CI.COND_ORDER = ALL_T2
    CI.GROUPS = tuple((c, (c,)) for c in ALL_T2)
    for c in ALL_T2:
        d = direction_of(c)
        CI.COND_INFO[c] = dict(label=XA.cell_label(c), direction=d, answer="copy (entity mention)" if d == "a2e" else "generated / stated attribute",
                               task=7, fmt="fill-in-the-blank" if d == "a2e" else "Phase 3")
    CI.ATTN_CONDS = tuple(c for c in ALL_T2 if direction_of(c) == "a2e")
    CI.EXTRA = {}
    from moetrace import ext10_circuit as C10
    C10.EXPERT_POOL.setdefault("qwen2_57b", 32)
    CI.COND_JSON = "(p6 tier-2 registry)"
    CI.SUMMARY = os.path.join(TABRAW, "ext13ci_summary_p6.json")
    CI.TAB, CI.FIG = TABRAW, FIGRAW
    os.makedirs(TABRAW, exist_ok=True)
    os.makedirs(FIGRAW, exist_ok=True)

    def load_conditions():
        return {(c, p): CI._norm_entry(c, p, e) for c, v in t2_registry().items() for p, e in v.items()}
    CI.load_conditions = load_conditions
    CI.step1_dir = lambda cond, proto, suffix="": os.path.join(t2_dir(cond, proto), "ci" + suffix)
    CI.step2_dir = lambda proto, group, suffix="": os.path.join(t2_dir(group, proto), "ci2" + suffix)
    CI.base_ready_lines = lambda: {(c, p) for c, v in t2_registry().items() for p in v}
    CI._p6 = True
    return CI


def patch_ko():
    from moetrace import ext13ko as K
    if getattr(K, "_p6", False):
        return K
    K.CONDITIONS = ALL_T2
    K.A2E = set(K.A2E) | {c for c in ALL_T2 if direction_of(c) == "a2e"}
    K.CONDS_JSON = "(p6 tier-2 registry)"
    for p, cf in EXTRA_CF.items():
        if p in K.PROTO:
            K.PROTO[p] = dict(K.PROTO[p], cf=list(cf))
    K.PROTO.setdefault("qwen2_57b", dict(model="qwen2_57b", bos=None, n_same=16, rand_near=4, wg=None, cf=[], budget=120_000, max_rows=4_000,
                                         label="Qwen2-57B-A14B"))
    def _reg():
        reg = t2_registry(need_selection=True)
        if not SHARED_MASKABLE:
            for c, v in reg.items():
                for p, e in v.items():
                    if has_shared(T2_MODEL.get(p, p)):
                        if "selected_expert" in e and _drop_shared(e["selected_expert"]) is None:
                            e["selected_expert"] = None
                            e["shared_not_maskable"] = True
                        if e.get("pop_top10"):
                            e["pop_top10"] = _drop_shared(e["pop_top10"])
        return reg
    K._registry = _reg
    K.out_dir = lambda e, suffix="": os.path.join(t2_dir(e["cond"], e["proto"]), "ko" + suffix)
    K._p6 = True
    return K


def patch_wr():
    patch_ko()
    from moetrace import ext13wr as WR
    if getattr(WR, "_p6", False):
        return WR
    WR.RAW = "/opt/dlami/nvme/moe_p6wr" + SUFFIX  # smoke runs never share raw vectors with real runs
    patch_writer_vocab()
    WR.out_dir = lambda e, suffix="": os.path.join(t2_dir(e["cond"], e["proto"]), "wr" + suffix)
    WR._p6 = True
    return WR


def patch_argparse():
    """The Phase-5 drivers restrict their positional model / proto argument to the Stage-1 models (argparse choices);
    drop the restriction for the Stage-2 model (qwen2_57b). Everything else is unchanged."""
    import argparse
    if getattr(argparse.ArgumentParser.add_argument, "_p6", False):
        return
    orig = argparse.ArgumentParser.add_argument

    def add_argument(self, *a, **k):
        if a and a[0] in ("model", "proto") and "choices" in k:
            k = dict(k)
            k["choices"] = list(k["choices"]) + [x for x in (*T2_MODEL.keys(), *T2_MODEL.values()) if x not in k["choices"]]
        return orig(self, *a, **k)
    add_argument._p6 = True
    argparse.ArgumentParser.add_argument = add_argument


# ------------------------------------------------------------------------------------------------------------------
# Qwen2-57B-A14B: the always-on shared expert (engine index E = n_experts = 64) as a component everywhere
# (coordinator decision 2026-10-16): candidate in add-back / greedy / oracle, row in the expert tables with its own DLA
# (contrib_dla_shared), knockout only if the engine can mask it (else reported "not maskable").
# ------------------------------------------------------------------------------------------------------------------
SHARED_MASKABLE = os.environ.get("EXT14_SHARED_MASKABLE", "0") == "1"


def has_shared(model: str) -> bool:
    return model == "qwen2_57b"


def patch_engine_shared():
    """Engine.run results of a model with a shared expert get the shared expert appended as a 9th, always-routed slot:
    route_idx / route_w / route_cnorm [L, B, k + 1] (index E, gate value, |c|), diag contrib_dla [L, B, k + 1]
    (contrib_dla_shared) and contrib_final_vectors[l] [B, k + 1, H]. Every Phase-5 consumer that iterates routed slots
    (add-back DLA ordering, circuit expert DLA, knockout self-repair sums, writer T / D / I) then treats it as an expert."""
    import numpy as np
    import moetrace.engine as EN
    if getattr(EN.Engine.run, "_p6shared", False):
        return
    orig = EN.Engine.run

    def cat(a, b):
        import torch
        if isinstance(a, torch.Tensor):
            return torch.cat([a, b.to(a.dtype).to(a.device)[..., None]], -1)
        return np.concatenate([np.asarray(a), np.asarray(b).astype(np.asarray(a).dtype)[..., None]], -1)

    def run(self, *a, **k):
        res = orig(self, *a, **k)
        if not getattr(self.spec, "shared_inter", 0):
            return res
        E = int(self.spec.n_experts)
        ex = res.extra or {}
        if getattr(res, "route_idx", None) is not None and "shared_w" in ex and res.route_idx.shape[-1] == self.spec.top_k:
            ri = np.asarray(res.route_idx)
            res.route_idx = np.concatenate([ri, np.full(ri.shape[:-1] + (1,), E, dtype=ri.dtype)], -1)
            if getattr(res, "route_w", None) is not None:
                res.route_w = cat(res.route_w, ex["shared_w"])
            if getattr(res, "route_cnorm", None) is not None and "shared_cnorm" in ex:
                res.route_cnorm = cat(res.route_cnorm, ex["shared_cnorm"])
        dg = ex.get("diag")
        if dg:
            if "contrib_dla" in dg and "contrib_dla_shared" in dg and np.asarray(dg["contrib_dla"]).shape[-1] == self.spec.top_k:
                dg["contrib_dla"] = cat(dg["contrib_dla"], dg["contrib_dla_shared"])
            if dg.get("contrib_final_vectors") and dg.get("contrib_final_vectors_shared"):
                import torch
                for l, v in list(dg["contrib_final_vectors"].items()):
                    sv = dg["contrib_final_vectors_shared"].get(l)
                    if sv is not None and v.shape[1] == self.spec.top_k:
                        dg["contrib_final_vectors"][l] = torch.cat([v, sv.to(v.dtype).to(v.device)[:, None, :]], 1)
        ex["p6_shared_slot"] = True
        return res
    run._p6shared = True
    EN.Engine.run = run


def _augment_shared(t):
    """Task of a shared-expert model: (l, E) added to every case's candidates (weight 1.0 = always on), topk + 1; the single
    values / pop come from the expert rows (the shared rows are appended by scripts/ext14_t2_shared.py --singles)."""
    from moetrace.arch import load_spec
    from moetrace.models import MODELS
    spec, _ = load_spec(MODELS[t.model]["repo"])
    E = int(spec.n_experts)
    for c in list(t.cand):
        cs = set(t.cand[c])
        t.cand[c] = sorted(cs | {(l, E) for l in range(t.L)})
        w = t.weight.setdefault(c, {})
        for l in range(t.L):
            w.setdefault((l, E), 1.0)
    t.topk = int(spec.top_k) + 1
    return t


def patch_loaders_shared():
    """moetrace.ext8_addback.load_wino (add-back, knockout population ranking) and moetrace.ext13ci.load_task (circuit)
    return shared-augmented tasks for shared-expert models."""
    from moetrace import ext8_addback as X8
    if getattr(X8.load_wino, "_p6shared", False):
        return
    orig_lw, orig_check = X8.load_wino, X8._check

    def load_wino(model, *a, **k):
        if not has_shared(model):
            return orig_lw(model, *a, **k)
        X8._check = lambda t: None
        try:
            t = orig_lw(model, *a, **k)
        finally:
            X8._check = orig_check
        t = _augment_shared(t)
        orig_check(t)
        return t
    load_wino._p6shared = True
    X8.load_wino = load_wino
    from moetrace import ext13ci as CI
    orig_ci = CI.load_task

    def ci_load_task(cond, proto, family=None):
        t = orig_ci(cond, proto, family)
        if has_shared(t.model):
            t = _augment_shared(t)
            n_full = sum(1 for r in t.rows.row_id if r in t.single and len(t.single[r]) == t.K)
            t.single_coverage = {"rows": int(len(t.rows)), "rows_full": int(n_full),
                                 "entries": int(sum(len(t.single.get(r, {})) for r in t.rows.row_id)), "entries_needed": int(len(t.rows) * t.K)}
        return t
    CI.load_task = ci_load_task


def _drop_shared(x, E=64):
    if x is None:
        return None
    if isinstance(x, dict):
        return None if int(x.get("expert", -1)) == E else x
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
        return [y for y in x if int(y[1]) != E]
    return x


def patch_writer_vocab():
    """ext11_writer.vocab_classes maps every non-qwen3 model to the mixtral_bos trigger ids: rebuild the trigger class with
    the model's own tokenizer (words decoded with the reference tokenizer, re-encoded with the model's). Also assert that the
    decoded readout tokens of the writer prompts equal the pair file's answers (coordinator decision 2026-10-16)."""
    from moetrace import ext11_writer as W11
    from moetrace import ext13wr as WR
    if getattr(W11.vocab_classes, "_p6", False):
        return
    orig_vc = W11.vocab_classes

    def vocab_classes(model, tok):
        out = orig_vc(model, tok)
        if model in ("qwen3", "mixtral"):
            return out
        from transformers import AutoTokenizer
        from moetrace.arch import snapshot_dir
        from moetrace.models import MODELS
        ref = "qwen3" if model.startswith("qwen") else "mixtral"
        rt = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[ref]["repo"]))
        words = {rt.decode([int(i)]).strip() for i in out["wg_trigger"]}
        out["wg_trigger"] = {x for x in (W11.first_cont_token(tok, w) for w in words if w) if x is not None}
        return out
    vocab_classes._p6 = True
    W11.vocab_classes = vocab_classes
    orig_cf = WR.classes_for

    def classes_for(e, it, tok):
        import pandas as pd
        pf = pd.read_parquet(os.path.join(ROOT, e["pairs"])).drop_duplicates("pair_id").set_index("pair_id")
        bad = []
        for r in it.itertuples():
            row = pf.loc[str(r.pair_id)]
            side = str(r.side)
            col = f"answer_{side}" if f"answer_{side}" in pf.columns else (f"ans_{side}" if e["direction"] == "a2e" else f"word_{side}")
            want = str(row[col]).strip().lower()
            got = tok.decode([int(r.true_id)]).strip().lower()
            if got != want:
                bad.append((r.pair_id, side, got, want))
        assert not bad, f"{e['cond']}/{e['proto']}: decoded readout tokens differ from the pair file's answers: {bad[:5]} ({len(bad)} of {len(it)})"
        return orig_cf(e, it, tok)
    WR.classes_for = classes_for


def patch_all():
    patch_engine_shared()
    patch_loaders_shared()
    return patch_ab(), patch_ci(), patch_ko(), patch_wr()
