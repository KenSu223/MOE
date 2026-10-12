"""p5-atp engine stages (GPU; run through scripts/gpu_queue.sh).

  scan-ioi            prefill-only competence scan of the 1,600 OLMoE IOI pairs (results/p5_atp_data/scan_ioi_olmoe.parquet),
                      then the IOI case sets (scripts/ext13atp_build.py ioi-sets logic)
  truth --task T      activation-patching ground truth at the final position, one row per (directed case, component):
                      kinds expert (every expert routed in the clean OR corrupted run at that layer), attn_head (every head),
                      attn_layer, layer (MoE output). Pass 0 = prefill routing. -> results/p5_atp_<T>/truth_rows.parquet,
                      truth_cases.parquet
  addback --task T    add-back curves on the validation rows with static orderings (oracle = the row's own engine single-
                      patch rescue, atp, atpstar, ig8, dla, pop (discovery mean truth), rand0..2), pools experts
                      (clean-active, ext8 convention) and heads + experts (Direction 10): one `multi` spawn per (row, order,
                      k); ceilings all-MoE and all clean-active experts. -> results/p5_atp_<T>/addback_rows.parquet

Usage: python scripts/ext13atp_engine.py scan-ioi | truth --task wino|ioi [--max-spawn 45000] [--dry-run] |
       addback --task wino|ioi [--max-spawn 45000]
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext13atp as A

log = A.log


def _meta(task, key, upd):
    p = os.path.join(A.run_dir(task), "run_meta.json")
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.setdefault(key, {}).update(upd)
    json.dump(cur, open(p, "w"), indent=1, default=str)


# ------------------------------------------------------------------------------------------------------------------
def scan_ioi(args):
    from moetrace.engine import Engine
    from moetrace import ext7_controls as C
    p = pd.read_parquet(os.path.join(A.DATA, "pairs_ioi_s2io_olmoe.parquet"))
    rows = C.scan_rows_to_engine(p)
    eng = Engine(A.REPO)
    t0 = time.time()
    sc = C.run_scan(eng, rows, log=log)
    a = sc[sc.kind == "a"].set_index("pair").delta
    b = sc[sc.kind == "b"].set_index("pair").delta
    out = pd.DataFrame({"pair_id": p.pair_id, "d_clean_a": a.reindex(range(len(p))).to_numpy(),
                        "d_clean_b": b.reindex(range(len(p))).to_numpy()})
    out["margin"] = (out.d_clean_a >= A.MARGIN) & (out.d_clean_b <= -A.MARGIN)
    out.to_parquet(os.path.join(A.DATA, "scan_ioi_olmoe.parquet"), index=False)
    info = {"n_pairs": len(out), "margin": int(out.margin.sum()), "mean_d_a": float(out.d_clean_a.mean()),
            "mean_d_b": float(out.d_clean_b.mean()), "gpu_s": time.time() - t0, "pass_times_s": sc.attrs.get("pass_times_s")}
    json.dump(info, open(os.path.join(A.DATA, "scan_ioi_olmoe.json"), "w"), indent=1)
    log("IOI scan", info)
    cs3 = json.load(open(os.path.join(A.ROOT, "data", "ioi", "case_sets_s2io.json")))
    cs = A.make_case_sets(cs3, set(out[out.margin].pair_id),
                          note="p5-atp: 64/64 pairs of the Phase-3 IOI (s2io) discovery / validation pairs that pass the OLMoE "
                               "margin (results/p5_atp_data/scan_ioi_olmoe.parquet), random.Random(13)")
    json.dump(cs, open(A.TASKS["ioi"]["case_sets"], "w"), indent=1)
    log("IOI case sets", {k: len(v) for k, v in cs["directed"].items()})


# ------------------------------------------------------------------------------------------------------------------
def truth(args):
    pairs, cs, fam, ids, dc, split = A.load_task(args.task)
    from moetrace.engine import Engine, PrefillSpec, SpawnSpec
    n = len(ids)
    eng = Engine(A.REPO)
    L, nH, k = eng.spec.n_layers, eng.spec.n_heads, eng.spec.top_k
    t_start = time.time()
    pre_all = [PrefillSpec(dc[c].clean_ids, dc[c].true_id, dc[c].foil_id) for c in ids] + \
              [PrefillSpec(dc[c].corrupt_ids, dc[c].true_id, dc[c].foil_id) for c in ids]
    r0 = eng.run(pre_all, [], record_routing=True, log=log)
    ri = r0.route_idx  # [L, 2n, k]
    cases = pd.DataFrame({"case_id": ids, "split": [split[c] for c in ids], "pair_idx": [dc[c].pair_idx for c in ids],
                          "d": [dc[c].d for c in ids], "T": [dc[c].T for c in ids],
                          "delta_clean": r0.delta[:n], "delta_corrupt": r0.delta[n:],
                          "route_clean": [json.dumps(ri[:, i].tolist()) for i in range(n)],
                          "route_corrupt": [json.dumps(ri[:, n + i].tolist()) for i in range(n)]})
    cases["drop"] = cases.delta_clean - cases.delta_corrupt
    od = A.run_dir(args.task)
    cases.to_parquet(os.path.join(od, "truth_cases.parquet"), index=False)
    log(f"{args.task}: {n} directed cases, mean drop {cases["drop"].mean():.3f}, min clean margin {cases.delta_clean.min():.3f}")

    def spawns_of(i, j, nc):
        """spawns of case ids[i] at chunk position j (clean row j, corrupt row nc + j)."""
        sp, tg = [], []
        c = ids[i]
        for l in range(L):
            for h in range(nH):
                sp.append(SpawnSpec(l, nc + j, j, "attn_head", expert=h)); tg.append((c, "head", l, h))
            sp.append(SpawnSpec(l, nc + j, j, "attn_layer")); tg.append((c, "attn", l, -1))
            for e in sorted(set(ri[l, i].tolist()) | set(ri[l, n + i].tolist())):
                sp.append(SpawnSpec(l, nc + j, j, "expert", expert=int(e))); tg.append((c, "expert", l, int(e)))
            sp.append(SpawnSpec(l, nc + j, j, "layer")); tg.append((c, "moe", l, -1))
        return sp, tg

    per = [len(spawns_of(i, 0, 1)[0]) for i in range(n)]
    chunks, cur, tot = [], [], 0
    for i in range(n):
        if cur and tot + per[i] > args.max_spawn:
            chunks.append(cur); cur, tot = [], 0
        cur.append(i); tot += per[i]
    chunks.append(cur)
    log(f"{sum(per)} spawn rows in {len(chunks)} passes")
    _meta(args.task, "truth", {"n_cases": n, "n_spawns": int(sum(per)), "chunks": len(chunks), "max_spawn": args.max_spawn,
                               "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "complete": False})
    if args.dry_run:
        return
    rows, times = [], [r0.extra.get("total_s")]
    for ci, ch in enumerate(chunks):
        nc = len(ch)
        pre = [pre_all[i] for i in ch] + [pre_all[n + i] for i in ch]
        sp, tg = [], []
        for j, i in enumerate(ch):
            a, b = spawns_of(i, j, nc)
            sp += a; tg += b
        res = eng.run(pre, sp, record_routing=False, log=log if ci == 0 else None)
        times.append(res.extra.get("total_s"))
        d, sd = res.delta, res.sp_delta
        jpos = {ids[i]: j for j, i in enumerate(ch)}
        for s, (c, ct, l, e) in enumerate(tg):
            j = jpos[c]
            rows.append((c, ct, l, e, float(sd[s]), float(sd[s] - d[nc + j]), float(res.sp_vnorm[s]), float(d[j]), float(d[nc + j])))
        log(f"pass {ci + 1}/{len(chunks)}: {len(sp)} spawns, {res.extra.get('total_s', 0):.1f}s")
    tr = pd.DataFrame(rows, columns=["case_id", "ctype", "layer", "idx", "delta", "rescue", "vnorm", "delta_clean_pass", "delta_corrupt_pass"])
    tr.to_parquet(os.path.join(od, "truth_rows.parquet"), index=False)
    _meta(args.task, "truth", {"complete": True, "pass_times_s": times, "gpu_s": time.time() - t_start,
                               "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    log(f"truth done: {len(tr)} rows, {time.time() - t_start:.0f}s")


# ------------------------------------------------------------------------------------------------------------------
ADD_ORDERS = ("oracle", "atp", "atpstar", "ig8", "dla", "pop", "rand0", "rand1", "rand2")
K_GRID = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)


def addback_plan(task):
    """Per validation row and pool: ordered component lists for every ordering (signed scores, descending)."""
    from moetrace.ext13atp_analysis import load_joined
    J, cases = load_joined(task)
    val = cases[cases.split == "validation"].case_id.tolist()
    disc = cases[cases.split == "discovery"].case_id.tolist()
    J = J[J.ctype.isin(["expert", "head"])]
    pop = J[J.case_id.isin(disc)].groupby(["ctype", "layer", "idx"]).truth.sum() / len(disc)
    plans = {}
    for c in val:
        g = J[J.case_id == c]
        ex = g[(g.ctype == "expert") & (g.clean_active_eng == 1)]
        hd = g[g.ctype == "head"]
        for pool, P in (("experts", ex), ("heads_experts", pd.concat([ex, hd]))):
            comps = list(zip(P.ctype, P.layer.astype(int), P.idx.astype(int)))
            sc = {"oracle": P.truth.to_numpy(), "atp": P.atp.to_numpy(), "atpstar": P.atpstar.to_numpy(), "ig8": P.ig8.to_numpy(),
                  "dla": P.dla.to_numpy(), "pop": np.array([pop.get(x, 0.0) for x in comps])}
            for o in ADD_ORDERS:
                if o.startswith("rand"):
                    perm = list(range(len(comps)))
                    random.Random(9000 + 10 * int(c) + int(o[4:]) + (0 if pool == "experts" else 5)).shuffle(perm)
                    order = perm
                else:
                    tie = np.random.default_rng(int(c)).random(len(comps))
                    order = list(np.lexsort((tie, -np.nan_to_num(sc[o], nan=-1e9))))
                plans[(c, pool, o)] = [comps[t] for t in order]
    return plans, val


def steps_of(comps, pool):
    by = {}
    for ct, l, i in comps:
        by.setdefault(l, ([], []))[0 if ct == "head" else 1].append(i)
    out = []
    for l in sorted(by):
        hs, es = by[l]
        if pool == "experts":
            out.append((l, "coalition_set", tuple(sorted(es))))
        else:
            out.append((l, "heads_experts", (tuple(sorted(hs)), tuple(sorted(es)))))
    return tuple(out)


def addback(args):
    pairs, cs, fam, ids, dc, split = A.load_task(args.task)
    plans, val = addback_plan(args.task)
    from moetrace.engine import Engine, PrefillSpec, SpawnSpec
    L = 16  # OLMoE-1B-7B
    t_start = time.time()
    tasks = []  # (case, pool, order, k, steps)
    for (c, pool, o), comps in plans.items():
        for kk in K_GRID:
            if kk > len(comps):
                continue
            tasks.append((c, pool, o, kk, steps_of(comps[:kk], pool)))
    for c in val:  # ceilings
        tasks.append((c, "ceiling", "all_moe", 0, tuple((l, "layer", None) for l in range(L))))
        tasks.append((c, "ceiling", "all_attn", 0, tuple((l, "attn_layer", None) for l in range(L))))
    by_case = {}
    for t in tasks:
        by_case.setdefault(t[0], []).append(t)
    chunks, cur, tot = [], [], 0
    for c in val:
        if cur and tot + len(by_case[c]) > args.max_spawn:
            chunks.append(cur); cur, tot = [], 0
        cur.append(c); tot += len(by_case[c])
    chunks.append(cur)
    log(f"{args.task} add-back: {len(val)} validation rows, {len(tasks)} multi spawns in {len(chunks)} passes")
    _meta(args.task, "addback", {"n_rows": len(val), "n_spawns": len(tasks), "orders": ADD_ORDERS, "k_grid": K_GRID,
                                 "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "complete": False})
    if args.dry_run:
        return
    eng = Engine(A.REPO)
    assert eng.spec.n_layers == L
    rows, times = [], []
    for ci, ch in enumerate(chunks):
        nc = len(ch)
        pre = [PrefillSpec(dc[c].clean_ids, dc[c].true_id, dc[c].foil_id) for c in ch] + \
              [PrefillSpec(dc[c].corrupt_ids, dc[c].true_id, dc[c].foil_id) for c in ch]
        sp, tg = [], []
        for j, c in enumerate(ch):
            for (c_, pool, o, kk, st) in by_case[c]:
                sp.append(SpawnSpec(st[0][0], nc + j, j, "multi", steps=st)); tg.append((c, j, pool, o, kk))
        res = eng.run(pre, sp, record_routing=False, log=log if ci == 0 else None)
        times.append(res.extra.get("total_s"))
        d, sd = res.delta, res.sp_delta
        for s, (c, j, pool, o, kk) in enumerate(tg):
            drop = d[j] - d[nc + j]
            rows.append((c, pool, o, kk, float(sd[s]), float(sd[s] - d[nc + j]), float((sd[s] - d[nc + j]) / drop), float(d[j]), float(d[nc + j])))
        log(f"pass {ci + 1}/{len(chunks)}: {len(sp)} spawns, {res.extra.get('total_s', 0):.1f}s")
    ab = pd.DataFrame(rows, columns=["case_id", "pool", "order", "k", "delta", "rescue", "r", "delta_clean", "delta_corrupt"])
    ab.to_parquet(os.path.join(A.run_dir(args.task), "addback_rows.parquet"), index=False)
    _meta(args.task, "addback", {"complete": True, "pass_times_s": times, "gpu_s": time.time() - t_start,
                                 "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    log(f"add-back done: {len(ab)} rows, {time.time() - t_start:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scan-ioi", "truth", "addback"])
    ap.add_argument("--task", choices=list(A.TASKS))
    ap.add_argument("--max-spawn", type=int, default=45000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    {"scan-ioi": scan_ioi, "truth": truth, "addback": addback}[args.stage](args)


if __name__ == "__main__":
    main()
