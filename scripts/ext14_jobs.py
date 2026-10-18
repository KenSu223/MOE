"""ext14 Phase-6 tier-1 job list for one cell and proto (agent p6-data; adapted from scripts/ext13_p5_jobs.py).

Prints one line per pending GPU job, "<job-name>\t<stage>\t<command>": tier 1 = sweep -> joint (-> joint_shared for qwen2_57b) -> dla ->
direct (scope cut 2026-10-17: no grid, heads only for HEADS_NEEDED / p6-circuit NEED HEADS cells; Phase-3 generic runners on the cell's
pair / case-set files; expert singles are tier 2, p6-analysis). Jobs whose marker
results/p6_<cell>_<proto>/.done_<stage> exists are skipped (the chain writes it after rc = 0).
Special first lines (the chain acts on them):
  "SKIP\t<reason>"   behaviour-only / insufficient / anchor / VOID by the coordinator: nothing to run
  "WAIT\t<reason>"   a REVIEW is pending (no "[coordinator] REVIEW OK <cell>/<proto>" in logs/PROGRESS.md yet): retry later
Row budgets scale with prompt length: budget = base * min(1, 30 / T_max) (T_max = longest selected prompt), bases as Phase 5
(sweep 60k Qwen-sized / 40k Mixtral-sized rows, joint 30k, heads 50k, grid 60k rows and 200k tokens); --scale multiplies them
(OOM retry with 0.75). Layer counts from the checkpoint config (moetrace.arch), not hard-coded.
Usage: python scripts/ext14_jobs.py <cell> <proto> [--scale 1.0]
"""
import argparse
import json
import os
import re
import sys

ROOT = "/home/ubuntu/MOE"
sys.path.insert(0, ROOT)
REG = f"{ROOT}/data/dial/p6_conditions.json"
BIG_ROWS = {"qwen3"}  # models whose Phase-5 sweep budget was 60k rows (others 40k)
# conservative start for the largest Stage-2 checkpoints (p6-engine runner audit: calibrate on Qwen3-14B and Qwen2-57B); the chain
# still retries an OOM at 3/4
MODEL_SCALE = {"qwen2_57b": 0.5, "qwen3_14b": 0.75}


def n_layers(model: str) -> int:
    from moetrace.models import MODELS
    try:
        from moetrace.arch import load_spec
        return int(load_spec(MODELS[model]["repo"])[0].n_layers)
    except Exception:
        from moetrace.arch import snapshot_dir
        return int(json.load(open(os.path.join(snapshot_dir(MODELS[model]["repo"]), "config.json")))["num_hidden_layers"])


def review_state(cell: str, proto: str) -> str:
    """'ok' | 'void' | 'pending' from the coordinator's PROGRESS lines."""
    try:
        txt = open(f"{ROOT}/logs/PROGRESS.md").read()
    except Exception:
        return "pending"
    # only lines that START with '<UTC> [coordinator] ...' count (quoted mentions in other lines do not)
    if re.search(rf"^(?:- )?\S+ \[coordinator\] VOID {re.escape(cell)}/{re.escape(proto)}\b", txt, re.M):
        return "void"
    if re.search(rf"^(?:- )?\S+ \[coordinator\] REVIEW OK {re.escape(cell)}/{re.escape(proto)}\b", txt, re.M):
        return "ok"
    return "pending"


HEADS_NEEDED = {("e3_e2a", "qwen3"), ("e3_e2a", "mixtral_bos"), ("e1_a2e", "qwen3"), ("e1_a2e", "mixtral_bos")}


def heads_needed(cell: str, proto: str) -> bool:
    if (cell, proto) in HEADS_NEEDED:
        return True
    try:
        txt = open(f"{ROOT}/logs/PROGRESS.md").read()
    except Exception:
        return False
    return bool(re.search(rf"^(?:- )?\S+ \[p6-circuit\] NEED HEADS {re.escape(cell)}/{re.escape(proto)}\b", txt, re.M))


def t_max(e: dict) -> int:
    import pandas as pd
    cs = json.load(open(f"{ROOT}/{e['case_sets']}"))
    ids = set(cs["pairs"].get("discovery", [])) | set(cs["pairs"].get("validation", []))
    p = pd.read_parquet(f"{ROOT}/{e['pair_file']}", columns=["pair_idx", "n_tokens"])
    p = p[p.pair_idx.isin(ids)]
    return int(p.n_tokens.max()) if len(p) else 30


ORDER = ["e1_a2e", "e1_e2a", "e3p3_a2e", "e3_a2e", "e3_e2a", "e1p_a2e", "e1p_e2a", "e2_a2e", "e2_e2a", "l_ind", "l_bind", "l_ant", "l_prop",
         "l_cf", "l_ioi", "e0_a2e", "e0_e2a", "aagr_a2e", "aagr_e2a", "acon_a2e", "acon_e2a", "sc_agree_a2e", "sc_conflict_stmt_a2e",
         "sc_conflict_event_a2e", "rsite_e1_a2e", "e1_a2e_all", "e1_e2a_all", "e1p_a2e_all", "e1p_e2a_all"]
# cells per GPU job (throughput fix 2026-10-17): Stage 1 two cells, dense Stage-2 models four (1-2 GPU-min each), Qwen2-57B one
BUNDLE = {"qwen3": 2, "mixtral_bos": 2, "qwen2_57b": 1}
BUNDLE_DEFAULT = 4


# tier-1 cell lists trimmed per model (coordinator 2026-10-17: Qwen2-57B fp32 costs ~4x; only the twin-comparison cells); behaviour from
# the competence scan is still recorded for every cell
TIER1_ONLY = {"qwen2_57b": {"l_ant", "l_prop", "l_cf", "l_ioi", "l_bind", "l_ind", "e1_a2e", "e1_e2a", "e3p3_a2e", "e3_e2a",
                            "e1_a2e_all", "e1_e2a_all"}}


def status(cell: str, proto: str):
    """('SKIP', reason) | ('WAIT', reason) | ('RUN', entry) | ('NONE', None) when not registered."""
    e = json.load(open(REG))["conditions"].get(cell, {}).get(proto)
    if e is None:
        return "NONE", None
    if proto in TIER1_ONLY and cell not in TIER1_ONLY[proto]:
        return "SKIP", f"tier 1 trimmed for {proto} (behaviour only)"
    if e.get("anchor"):
        return "SKIP", "anchor (existing runs)"
    if e.get("insufficient") or e.get("behaviour_only"):
        return "SKIP", "insufficient / behaviour only"
    if e.get("review"):
        st = review_state(cell, proto)
        if st == "void":
            return "SKIP", "VOID by the coordinator"
        if st == "pending":
            return "WAIT", f"review pending: {e['review']}"
    return "RUN", e


def stage_jobs(cell: str, proto: str, e: dict, scale: float):
    model = e["model"]
    run = f"p6_{cell}_{proto}"
    od = f"{ROOT}/results/{run}"
    P = f"--model {model} --pairs {e['pair_file']} --case-sets {e['case_sets']} --out {run} --agent p6-data"
    fams = ",".join(e["families"])
    f = min(1.0, 30.0 / max(30, t_max(e))) * scale * MODEL_SCALE.get(model, 1.0)
    sweep = int((60000 if model in BIG_ROWS else 40000) * f)
    # scope cut (coordinator 2026-10-17): tier 1 = sweep + joint + DLA + direct split; heads only where a downstream step needs them
    # (HEADS_NEEDED, '[p6-circuit] NEED HEADS <cell>/<proto>' lines); no new grids (the four that ran are kept)
    jobs = [("sweep", f"python scripts/ext7_wino_sweep.py {P} --max-rows {sweep}"),
            ("joint", f"python scripts/ext7_wino_joint.py {P} --families {fams} --max-rows {int(30000 * f)}")]
    if model == "qwen2_57b":  # prediction D6: shared vs routed joint patches next to all-MoE
        jobs.append(("joint_shared", f"python scripts/ext14_scan_joint_shared.py {P} --families {fams} --max-rows {int(30000 * f)}"))
    jobs += [("dla", f"python scripts/ext7_wino_dla.py {P} --families {fams} --chunk {max(128, int(512 * f))}"),
             ("direct", f"python -m moetrace.ext7_controls direct {P.replace(' --agent p6-data', '')} --splits discovery,validation")]
    if heads_needed(cell, proto):
        jobs.append(("heads", f"python scripts/ext7_wino_heads.py {P} --families main --layers auto --max-rows {int(50000 * f)}"))
    return [(f"ext14-{cell}-{proto}-{st}", st, cmd) for st, cmd in jobs if not os.path.exists(os.path.join(od, f".done_{st}"))]


def locked(cell: str, proto: str) -> bool:
    p = f"{ROOT}/results/p6_{cell}_{proto}/.lock_t1"
    try:
        os.kill(int(open(p).read().strip()), 0)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("proto")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--stages", action="store_true", help="per-stage lines (used inside the bundle job)")
    a = ap.parse_args()
    st, e = status(a.cell, a.proto)
    if st in ("SKIP", "WAIT"):
        print(f"{st}\t{e}")
        return
    if st == "NONE":
        raise SystemExit(f"{a.cell}/{a.proto} not registered")
    jobs = stage_jobs(a.cell, a.proto, e, a.scale)
    if a.stages:
        for name, s_, cmd in jobs:
            print(f"{name}\t{s_}\t{cmd}")
        return
    if not jobs:
        return
    # one GPU job per cell (+ the next pending cells of the same model and the same kind (_all or not), up to BUNDLE)
    bundle = [a.cell]
    nmax = BUNDLE.get(a.proto, BUNDLE_DEFAULT)
    if a.cell in ORDER:
        for c in ORDER[ORDER.index(a.cell) + 1:]:
            if len(bundle) >= nmax:
                break
            if c.endswith("_all") != a.cell.endswith("_all") or locked(c, a.proto):
                continue
            st2, e2 = status(c, a.proto)
            if st2 == "RUN" and stage_jobs(c, a.proto, e2, a.scale):
                bundle.append(c)
    print(f"ext14-{a.cell}-{a.proto}-t1\tt1\tbash scripts/ext14_chain_base_cell.sh {a.scale} {a.proto} {' '.join(bundle)}")


if __name__ == "__main__":
    main()
