"""Phase-6 R-causal (agent p6-analysis): a causal handle on the attention share (RESEARCH_PLAN.md Phase 6, "Robustness and
causal checks"; predictions R3 / R4).

Cells e1_a2e and e3_e2a on qwen3 and mixtral_bos. Per cell and model, nine conditions:
  none            no knockout (baseline recomputed in the same code path);
  heads_top3      the cell's top-3 heads by DISCOVERY single-head rescue at the final token (tier-1 head singles,
                  results/p6_<cell>_<proto>/head_rows.parquet), head mask in BOTH prefill rows (clean and corrupted),
                  mode zero;
  heads_r0..r2    matched random heads: one random head in each of the same layers (excluding the cell's top-10 heads);
  experts_top3    the cell's top-3 experts by discovery all-case single-expert rescue (population ranking; e1_a2e: the
                  tier-2 expert singles, e3_e2a: expert singles run here), route mask at all positions, mode reroute;
  experts_r0..r2  matched random experts: per member a random expert of the same layer among the rand_near experts
                  with the nearest final-position routing frequency on the cell's clean prompts (ext13ko rule),
                  excluding the cell's population top-10.
Per condition the tier-1 T2 sweep (single-layer attention / MoE patches at every layer, scripts/ext7_wino_sweep.py) and
T3 (all-MoE joint, scripts/ext7_wino_joint.py) are re-run UNCHANGED (scope cut 2026-10-17: no direct split; zero-mode
knockouts for heads and experts; e3_e2a experts ranked on the Phase-5 p5_dir_opt_<proto> expert rows)
on the validation pairs, with the mask injected into every PrefillSpec the runners build (a PrefillSpec subclass whose
mask fields default to the condition's mask; p6-engine applies the masks inside the spawned wavefront rows as well).

Outputs results/p6_causal_<cell>_<proto>/: plan.json, base/ (expert singles mirror for e3_e2a), <cond>/ (ext7 run dirs).
Analysis: scripts/ext14_causal_analyze.py.

Usage:
  python scripts/ext14_causal_run.py --plan                          print 'NEXT <cell> <proto> <cond> <stage>' (CPU; writes plans)
  python scripts/ext14_causal_run.py <cell> <proto> <cond> <stage>   run one stage (GPU): stage = expert | sweep | joint | direct
"""
import argparse, dataclasses, json, os, random, re, shutil, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd
from moetrace import ext14_analysis as XA
from moetrace.models import RESULTS, MODELS

ROOT = "/home/ubuntu/MOE"
CELLS = ("e1_a2e", "e3_e2a")
PROTOS = ("qwen3", "mixtral_bos")
MODEL = {"qwen3": "qwen3", "mixtral_bos": "mixtral"}
CONDS = ("none", "heads_top3", "heads_r0", "heads_r1", "heads_r2", "experts_top3", "experts_r0", "experts_r1", "experts_r2")
STAGES = ("sweep", "joint")  # scope cut 2026-10-17 (user): re-run only the T2 sweep and the T3 joint under the knockout
RAND_NEAR = {"qwen3": 4, "mixtral_bos": 3}
MAX_ROWS = {"qwen3": 60000, "mixtral_bos": 40000}
HEAD_MASK_MODE = os.environ.get("EXT14_HEAD_MASK_MODE", "zero")
EXPERT_MASK_MODE = os.environ.get("EXT14_EXPERT_MASK_MODE", "zero")  # coordinator 2026-10-17: zero-mode knockouts (Phase 4: zero ~ reroute)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


SUFFIX = os.environ.get("EXT14_CAUSAL_SUFFIX", "")  # smoke tests: "_smoke"


def cdir(cell, proto):
    return os.path.join(RESULTS, f"p6_causal_{cell}_{proto}{SUFFIX}")


def run_name(cell, proto, cond):
    return f"p6_causal_{cell}_{proto}{SUFFIX}/{cond}"


def done(cell, proto, cond, stage):
    return os.path.exists(os.path.join(cdir(cell, proto), cond, f".done_{stage}"))


def mark(cell, proto, cond, stage):
    d = os.path.join(cdir(cell, proto), cond)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, f".done_{stage}"), "w").close()


# ------------------------------------------------------------------------------------------------------------------
# targets
# ------------------------------------------------------------------------------------------------------------------
def base_dir(cell, proto):
    e = XA.entry_of(cell, proto)
    return os.path.join(RESULTS, XA.run_name(e))


def head_ranking(cell, proto):
    hr = pd.read_parquet(os.path.join(base_dir(cell, proto), "head_rows.parquet"))
    cs = json.load(open(os.path.join(base_dir(cell, proto), "case_sets.json")))
    disc = set(int(c) for c in cs["main"]["discovery"])
    val = set(int(c) for c in cs["main"]["validation"])
    h = hr[hr.kind == "attn_head"]
    pm = h.assign(pair=h.case_id // 2)
    dm = pm[pm.case_id.isin(disc)].groupby(["layer", "head", "pair"]).rescue.mean().groupby(["layer", "head"]).mean()
    vm = pm[pm.case_id.isin(val)].groupby(["layer", "head", "pair"]).rescue.mean().groupby(["layer", "head"]).mean()
    nH = int(h["head"].max()) + 1
    order = dm.sort_values(ascending=False)
    return [(int(l), int(x), float(v), float(vm.get((l, x), np.nan))) for (l, x), v in order.items()], nH, sorted(int(l) for l in h.layer.unique())


P5_EXPERT_SOURCE = {"e3_e2a": "p5_dir_opt_{proto}"}  # scope cut 2026-10-17: e3_e2a experts ranked on the Phase-5 dir_opt expert rows


def expert_source(cell, proto):
    """Run dir with str_expert_rows for the population ranking: tier-2 mirror (e1_a2e), the Phase-5 dir_opt run (e3_e2a),
    else this check's own base mirror."""
    t2 = os.path.join(RESULTS, f"p6_{cell}_{proto}_t2")
    if os.path.exists(os.path.join(t2, "str_expert_rows.parquet")) and os.path.exists(os.path.join(t2, ".done_expert")):
        return f"p6_{cell}_{proto}_t2"
    if cell in P5_EXPERT_SOURCE and not SUFFIX:
        r = P5_EXPERT_SOURCE[cell].format(proto=proto)
        if os.path.exists(os.path.join(RESULTS, r, "str_expert_rows.parquet")):
            return r
    b = os.path.join(cdir(cell, proto), "base")
    if os.path.exists(os.path.join(b, ".done_expert")):
        return f"p6_causal_{cell}_{proto}{SUFFIX}/base"
    return None


def prepare_base(cell, proto):
    """Mirror of the base run for the expert singles of a cell without tier 2 (symlinked inputs)."""
    src = base_dir(cell, proto)
    od = os.path.join(cdir(cell, proto), "base")
    os.makedirs(od, exist_ok=True)
    for f in ("str_sweep_rows.parquet", "str_sweep_routing.parquet", "sweep_cases.parquet"):
        d = os.path.join(od, f)
        if os.path.exists(os.path.join(src, f)) and not os.path.lexists(d):
            os.symlink(os.path.relpath(os.path.join(src, f), od), d)
    for f in ("case_sets.json", "run_meta.json"):
        if not os.path.exists(os.path.join(od, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(od, f))
    return od


def pop_ranking(run):
    od = os.path.join(RESULTS, run)
    e = pd.read_parquet(os.path.join(od, "str_expert_rows.parquet"), columns=["case_id", "layer", "kind", "expert", "rescue", "clean_active"])
    cs = json.load(open(os.path.join(od, "case_sets.json")))
    disc = [int(c) for c in cs["main"]["discovery"]]
    s = e[(e.kind == "expert") & e.clean_active & e.case_id.isin(disc)]
    pop = (s.groupby(["layer", "expert"]).rescue.sum() / len(disc)).sort_values(ascending=False)
    return [(int(l), int(x), float(v)) for (l, x), v in pop.head(32).items()]


def final_freq(cell, proto, n_layers, n_experts):
    rt = pd.read_parquet(os.path.join(base_dir(cell, proto), "str_sweep_routing.parquet"), columns=["case_id", "slot", "layer", "expert"])
    rt = rt[rt.slot == -1]
    n = rt.case_id.nunique()
    cnt = np.zeros((n_layers, n_experts))
    np.add.at(cnt, (rt.layer.values.astype(int), rt.expert.values.astype(int)), 1.0)
    return cnt / max(n, 1)


def make_plan(cell, proto):
    p = os.path.join(cdir(cell, proto), "plan.json")
    if os.path.exists(p):
        return json.load(open(p))
    src = expert_source(cell, proto)
    if src is None:
        return None
    from moetrace.arch import load_spec
    spec, _ = load_spec(MODELS[MODEL[proto]]["repo"])
    hrank, nH, hlayers = head_ranking(cell, proto)
    top = [(l, h) for l, h, _, _ in hrank[:3]]
    top10 = {(l, h) for l, h, _, _ in hrank[:10]}
    heads = {"top3": [list(x) for x in top], "top3_disc_val": [[l, h, d, v] for l, h, d, v in hrank[:3]], "layers_scanned": hlayers, "nH": nH}
    for r in range(3):
        rng = random.Random(9100 + r)
        picked = []
        for l, _ in top:
            picked.append([l, rng.choice([x for x in range(nH) if (l, x) not in top10 and [l, x] not in picked])])
        heads[f"r{r}"] = picked
    pop = pop_ranking(src)
    etop = [(l, x) for l, x, _ in pop[:3]]
    excl = {(l, x) for l, x, _ in pop[:10]}
    fq = final_freq(cell, proto, spec.n_layers, spec.n_experts)
    experts = {"top3": [list(x) for x in etop], "pop_top10": [[l, x, v] for l, x, v in pop[:10]], "source": src,
               "freq_top3": [float(fq[l, x]) for l, x in etop]}
    for r in range(3):
        rng = random.Random(9200 + r)
        used, ch = set(), []
        for (l, x) in etop:
            cand = [y for y in range(spec.n_experts) if (l, y) not in excl and (l, y) not in used]
            cand = sorted(cand, key=lambda y: (abs(fq[l, y] - fq[l, x]), y))[: RAND_NEAR[proto]]
            y = rng.choice(cand)
            used.add((l, y))
            ch.append([l, y])
        experts[f"r{r}"] = ch
        experts[f"freq_r{r}"] = [float(fq[l, y]) for l, y in ch]
    masks = {"none": {}}
    for k in ("top3", "r0", "r1", "r2"):
        masks[f"heads_{k}"] = {"head_mask": [list(x) for x in heads[k]], "head_mask_mode": HEAD_MASK_MODE}
        masks[f"experts_{k}"] = {"route_mask": [list(x) for x in experts[k]], "route_mask_mode": EXPERT_MASK_MODE, "route_mask_pos": "all",
                                 "route_mask_inherit": True}
    plan = {"cell": cell, "proto": proto, "heads": heads, "experts": experts, "masks": masks, "splits": "validation",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "heads ranked by discovery single-head rescue (tier-1 W5 layers); experts by discovery all-case single-expert rescue; "
                    "random heads: same layers, excluding the top-10 heads; random experts: same layer, nearest final-position routing frequency "
                    f"({RAND_NEAR[proto]} candidates), excluding the population top-10"}
    os.makedirs(cdir(cell, proto), exist_ok=True)
    json.dump(plan, open(p, "w"), indent=1)
    return plan


# ------------------------------------------------------------------------------------------------------------------
def next_job():
    if not engine_ready():
        return None
    lines = XA.ready_lines()
    for cell in CELLS:
        for proto in PROTOS:
            if (cell, proto) not in lines or XA.entry_of(cell, proto) is None:
                continue
            if not os.path.exists(os.path.join(base_dir(cell, proto), "head_rows.parquet")):
                continue  # head singles of the cell not run yet (p6-data runs heads on e3_e2a for R-causal)
            if expert_source(cell, proto) is None:
                if cell == "e1_a2e":
                    continue  # wait for the tier-2 expert singles
                if not os.path.exists(os.path.join(cdir(cell, proto), "base", ".done_expert")):
                    return (cell, proto, "base", "expert")
            plan = make_plan(cell, proto)
            if plan is None:
                continue
            for st in STAGES:  # one GPU job per (cell, model, stage) with every pending condition (queue waits dominate otherwise)
                if any(not done(cell, proto, cond, st) for cond in CONDS):
                    return (cell, proto, "BATCH", st)
    return None


def engine_ready():
    p = os.path.join(ROOT, "logs", "PROGRESS.md")
    return any(re.match(r"^\s*(-\s*)?\S+\s+\[p6-engine\]\s+HEADMASK READY", ln) for ln in open(p, errors="replace"))


_ORIG_PREFILL = None


def masked_prefill(mask: dict):
    """PrefillSpec subclass whose mask fields default to the condition's mask (the ext7 runners build PrefillSpec(ids, true,
    foil[, clean_ref]) only, so every row they create is masked). Always derived from the engine's ORIGINAL class, so a
    batch of conditions in one process never stacks masks."""
    global _ORIG_PREFILL
    import moetrace.engine as EN
    if _ORIG_PREFILL is None:
        _ORIG_PREFILL = EN.PrefillSpec
    base = _ORIG_PREFILL
    if not mask:
        return base
    fields = {f.name for f in dataclasses.fields(base)}
    need = [k for k in mask if k not in fields]
    if need:
        raise SystemExit(f"engine PrefillSpec has no field(s) {need}; see logs/ext14_engine_api.md")
    vals = {k: (tuple(tuple(x) for x in v) if isinstance(v, list) else v) for k, v in mask.items()}
    ns = {"__annotations__": {k: type(v) for k, v in vals.items()}}
    ns.update(vals)
    cls = dataclasses.dataclass(type("MaskedPrefillSpec", (base,), ns))
    return cls


def run_stage(cell, proto, cond, stage):
    e = XA.entry_of(cell, proto)
    pf, cs = XA._abs(e["pair_file"]), XA._abs(e["case_sets"])
    m = MODEL[proto]
    if cond == "base" and stage == "expert":
        prepare_base(cell, proto)
        sys.argv = ["ext7_wino_expert.py", "--model", m, "--pairs", pf, "--case-sets", cs, "--out", f"p6_causal_{cell}_{proto}{SUFFIX}/base", "--agent", "p6-analysis",
                    "--families", "main", "--layers", "all", "--max-spawn", str(MAX_ROWS[proto])]
        import ext7_wino_expert as S
        S.main()
        mark(cell, proto, "base", "expert")
        return
    plan = make_plan(cell, proto)
    mask = plan["masks"][cond]
    import moetrace.engine as EN
    EN.PrefillSpec = masked_prefill(mask)
    out = run_name(cell, proto, cond)
    common = ["--model", m, "--pairs", pf, "--case-sets", cs, "--out", out, "--agent", "p6-analysis"]
    if stage == "sweep":
        import ext7_wino_sweep as S
        sys.argv = ["ext7_wino_sweep.py"] + common + ["--splits", "validation", "--max-rows", str(MAX_ROWS[proto])]
        S.main()
    elif stage == "joint":
        import ext7_wino_joint as S
        sys.argv = ["ext7_wino_joint.py"] + common + ["--splits", "validation", "--families", "main", "--max-rows", "30000"]
        S.main()
    elif stage == "direct":
        from moetrace import ext7_controls as CTL
        sys.argv = ["ext7_controls", "direct", "--model", m, "--pairs", pf, "--case-sets", cs, "--out", out, "--splits", "validation"]
        CTL._cli()
    else:
        raise SystemExit(stage)
    meta_p = os.path.join(cdir(cell, proto), cond, "run_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta["causal"] = {"cell": cell, "proto": proto, "cond": cond, "mask": mask, "engine_prefill_class": EN.PrefillSpec.__name__}
    json.dump(meta, open(meta_p, "w"), indent=1, default=str)
    mark(cell, proto, cond, stage)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--plan":
        j = next_job()
        if j:
            print("NEXT " + " ".join(j))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", choices=CELLS)
    ap.add_argument("proto", choices=PROTOS)
    ap.add_argument("cond", choices=CONDS + ("base", "BATCH"))
    ap.add_argument("stage", choices=STAGES + ("expert",))
    a = ap.parse_args()
    if a.cond == "BATCH":
        import time as _t
        for cond in CONDS:
            if not done(a.cell, a.proto, cond, a.stage):
                t0 = _t.time()
                log(f"=== {a.cell}/{a.proto} {cond} {a.stage}")
                run_stage(a.cell, a.proto, cond, a.stage)
                log(f"=== {cond} {a.stage} done in {_t.time() - t0:.0f}s")
        return
    run_stage(a.cell, a.proto, a.cond, a.stage)


if __name__ == "__main__":
    main()
