"""Phase-6 Stage 2 (p6-analysis): Qwen2-57B-A14B shared vs routed experts per layer (prediction D6).

Qwen2-57B-A14B's MoE output = sum of the top-8 routed experts (64) + one always-on shared expert (sigmoid gate). Per Set-A
cell (validation pairs) and layer, three single-layer final-token patches in one pass (parent = corrupted row, source =
clean row; engine index E = n_experts is the shared expert, logs/ext14_engine_api.md):
  layer   the whole MoE output (routed + shared)      = the tier-1 'layer' kind;
  shared  the shared expert only                       (kind expert, expert = E);
  routed  the routed experts active in the clean or the corrupted run (kind coalition_set, routing from the base sweep).
The shared share of the MoE rescue = AUC+(shared) / (AUC+(shared) + AUC+(routed)) over the mean layer curves (as the attention
share), with a pair-cluster bootstrap (unit = the cell's unit column, else the pair; 5,000 resamples); also AUC+(shared) /
AUC+(layer) and the per-layer additivity layer vs shared + routed.
Writes results/p6_<cell>_qwen2_57b_t2/shared_rows.parquet and shared_summary.json.
--singles (tier-2 cells, after the expert singles): all main discovery + validation cases, and the shared rows are appended to
the tier-2 mirror's str_expert_rows.parquet as kind 'expert', expert = E (= 64, clean- and corrupt-active in every case), so the
selection, add-back, circuit and writer treat the shared expert as one more component (coordinator decision 2026-10-16).

Usage: python scripts/ext14_t2_shared.py <cell> [--proto qwen2_57b] [--max-spawn 40000] [--singles] [--analyse-only]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd
from moetrace import ext7_pairs as P
from moetrace import ext14_analysis as XA
from moetrace.models import MODELS, RESULTS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def out_dir(cell, proto):
    d = os.path.join(RESULTS, f"p6_{cell}_{proto}_t2")
    os.makedirs(d, exist_ok=True)
    return d


def run(cell, proto, max_spawn, singles=False):
    e = XA.entry_of(cell, proto)
    base = os.path.join(RESULTS, XA.run_name(e))
    cs = json.load(open(XA._abs(e["case_sets"])))
    pairs = P.load_pairs(XA._abs(e["pair_file"]), cs)
    fam = P.load_families(XA.run_name(e))
    ids = [int(c) for c in fam["main"]["validation"]]
    if singles:
        ids = [int(c) for c in fam["main"]["discovery"]] + ids
    dc = P.directed_cases(pairs, ids)
    rt = pd.read_parquet(os.path.join(base, "str_sweep_routing.parquet"))
    rt = rt[rt.case_id.isin(ids)]
    routed = {}
    for (c, l), g in rt.groupby(["case_id", "layer"]):
        routed[(int(c), int(l))] = tuple(sorted(set(int(x) for x in g.expert)))
    from moetrace.engine import Engine, PrefillSpec, SpawnSpec
    m = MODELS[XA.MODEL_OF[proto]]
    eng = Engine(m["repo"])
    L, E = eng.spec.n_layers, eng.spec.n_experts
    assert getattr(eng.spec, "shared_inter", 0), "model has no shared expert"
    n = len(ids)
    pre = [PrefillSpec(dc[c].clean_ids, dc[c].true_id, dc[c].foil_id) for c in ids]
    pre += [PrefillSpec(dc[c].corrupt_ids, dc[c].true_id, dc[c].foil_id, clean_ref=i) for i, c in enumerate(ids)]
    per_layer = 3 * n
    chunks, cur = [], []
    for l in range(L):
        if cur and (len(cur) + 1) * per_layer > max_spawn:
            chunks.append(cur)
            cur = []
        cur.append(l)
    chunks.append(cur)
    rows = []
    for ci, ch in enumerate(chunks):
        sp, tags = [], []
        for i, c in enumerate(ids):
            for l in ch:
                sp.append(SpawnSpec(l, n + i, i, "layer")); tags.append((i, l, "layer"))
                sp.append(SpawnSpec(l, n + i, i, "expert", expert=E)); tags.append((i, l, "shared"))
                sp.append(SpawnSpec(l, n + i, i, "coalition_set", experts=routed[(c, l)])); tags.append((i, l, "routed"))
        log(f"pass {ci + 1}/{len(chunks)}: layers {ch[0]}..{ch[-1]}, {len(pre)} prefill, {len(sp)} spawns")
        res = eng.run(pre, sp, record_routing=False, log=log if ci == 0 else None)
        d = res.delta
        for jj, (i, l, k) in enumerate(tags):
            rows.append(dict(case_id=ids[i], layer=l, kind=k, delta=float(res.sp_delta[jj]), delta_clean=float(d[i]), delta_corrupt=float(d[n + i]),
                             rescue=float(res.sp_delta[jj] - d[n + i]), vnorm=float(res.sp_vnorm[jj])))
    od = out_dir(cell, proto)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(od, "shared_rows.parquet"), index=False)
    if singles:
        append_expert_rows(od, df, E, routed)
    json.dump({"cell": cell, "proto": proto, "n_cases": n, "chunks": chunks, "agent": "p6-analysis", "command": "python " + " ".join(sys.argv),
               "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(os.path.join(od, "shared_meta.json"), "w"), indent=1)


def append_expert_rows(od, df, E, routed):
    """Shared rows -> the tier-2 mirror's str_expert_rows.parquet (kind 'expert', expert E; earlier E rows replaced)."""
    p = os.path.join(od, "str_expert_rows.parquet")
    assert os.path.exists(p), f"{p}: run the expert singles first (scripts/ext7_wino_expert.py --out <t2 run>)"
    old = pd.read_parquet(p)
    sh = df[df.kind == "shared"]
    n_act = {(c, l): len(v) for (c, l), v in routed.items()}
    new = pd.DataFrame(dict(case_id=sh.case_id.astype(old.case_id.dtype), slot=0, layer=sh.layer.astype(old.layer.dtype), kind="expert", expert=E, partner=-1,
                            alpha=1.0, norm_e=np.nan, norm_partner=np.nan, vnorm=sh.vnorm, delta=sh.delta, delta_clean=sh.delta_clean,
                            delta_corrupt=sh.delta_corrupt, rescue=sh.rescue, clean_active=True, corrupt_active=True, clean_weight=np.nan,
                            corrupt_weight=np.nan, n_clean_active=[n_act.get((int(c), int(l)), np.nan) for c, l in zip(sh.case_id, sh.layer)]))
    new = new[[c for c in old.columns if c in new.columns]]
    keep = old[~((old.kind == "expert") & (old.expert == E))]
    pd.concat([keep, new], ignore_index=True).to_parquet(p, index=False)
    log(f"appended {len(new)} shared-expert rows (expert {E}) to {p}")


def analyse(cell, proto, nb=5000):
    od = out_dir(cell, proto)
    df = pd.read_parquet(os.path.join(od, "shared_rows.parquet"))
    e = XA.entry_of(cell, proto)
    d = XA.CellData(cell, proto, entry=e)  # units / pairs of the validation set
    df = df[df.case_id.isin(d.ids)]
    M = {k: P.matrix_pairs(df[df.kind == k].pivot(index="case_id", columns="layer", values="rescue")).reindex(d.pidx).to_numpy(np.float64)
         for k in ("layer", "shared", "routed")}
    u, Wu = XA.unit_weights(d.unit, nb, 0)
    W = XA.pair_weights(d, u, Wu)
    ws = W.sum(1)
    auc = {k: np.clip((W @ v) / ws[:, None], 0, None).sum(1) for k, v in M.items()}
    drop = (W @ d.drop) / ws
    out = {"cell": cell, "proto": proto, "n_pairs": int(d.n), "n_units": int(len(u)),
           "shared_share_of_moe_auc": XA.ci(auc["shared"] / (auc["shared"] + auc["routed"])),
           "shared_over_layer_auc": XA.ci(auc["shared"] / auc["layer"]),
           "auc_layer_norm": XA.ci(auc["layer"] / drop), "auc_shared_norm": XA.ci(auc["shared"] / drop), "auc_routed_norm": XA.ci(auc["routed"] / drop),
           "curves": {k: [round(float(x), 5) for x in v.mean(0) / d.drop.mean()] for k, v in M.items()},
           "additivity_max_abs_gap_norm": float(np.max(np.abs((M["layer"] - M["shared"] - M["routed"]).mean(0))) / d.drop.mean())}
    json.dump(out, open(os.path.join(od, "shared_summary.json"), "w"), indent=1)
    log(f"{cell}/{proto}: shared share of the MoE rescue {XA.fr(out['shared_share_of_moe_auc'])}, shared / layer {XA.fr(out['shared_over_layer_auc'])}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--proto", default="qwen2_57b")
    ap.add_argument("--max-spawn", type=int, default=40000)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--singles", action="store_true", help="tier-2 cell: all main cases, rows appended to the mirror's expert table")
    a = ap.parse_args()
    if not a.analyse_only:
        run(a.cell, a.proto, a.max_spawn, a.singles)
    analyse(a.cell, a.proto)


if __name__ == "__main__":
    main()
