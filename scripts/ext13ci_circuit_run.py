"""ext13ci step 2 (Phase 5, p5-circuit): joint add-back over attention heads + experts at the final position for a group
of Phase-5 conditions of one model (resumable passes) = scripts/ext10_circuit_run.py with the Phase-5 registry.

The conditions of the group share every pass (their prefill blocks are concatenated; prompts are right-padded, so each
row's computation is unchanged). Every pass carries all prefill rows plus
  * the adaptive rows of the current greedy step: mixed greedy (pool = the row's top-32 heads by single-head rescue ∪ the
    row's expert pool by single-expert rescue of the base run, Qwen3 top 32, Mixtral all 64; missing single-expert
    entries -> expert DLA of the step-1 pass) and head-only greedy (pool = the 32 heads), 20 steps;
  * whole groups of the static queue up to --budget spawn rows: in-pass ceilings (all heads at all layers, all attention,
    all MoE), per-layer sanity rows for the first 32 rows of each condition (all heads of a layer vs attn_layer), the
    static orderings mix_oracle / mix_dla / head_oracle / head_dla on the k grid.
Rows saved after every pass (circuit_rows_pNN.parquet, circuit_prefill_pNN.parquet) with the pickled state
(circuit_state.pkl) in results/p5_<proto>_ci<suffix>/<group>/.

Usage: python scripts/ext13ci_circuit_run.py <proto> --group g1 --conds f1,dir_f1,dir_opt [--budget N] [--max-passes 99]
       [--limit N] [--out-suffix _smoke] [--wf-chunk 2048] [--dry-run]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS
from moetrace import ext10_circuit as C
from moetrace import ext13ci as E13


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_meta(od, upd):
    p = os.path.join(od, "run_meta.json")
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.update(upd)
    with open(p, "w") as f:
        json.dump(cur, f, indent=1, default=str)


def spawn_of(cts, item, SpawnSpec):
    task, fam, order, k, rid, payload = item
    ct = cts[task]
    ci, cr = ct.ci[ct.case_of(rid)], ct.cr[rid]
    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] == "attn_layer":
        return SpawnSpec(int(payload[1]), cr, ci, "attn_layer")
    st = C.steps_of(payload, ct.L, ct.nH)
    return SpawnSpec(st[0][0], cr, ci, "multi", steps=st, kl_ref=ci)


def build_pass(st, queue, budget):
    items = []
    for g in st.greedy.values():
        items += g.items()
    n_dyn = len(items)
    q0 = st.qpos
    while st.qpos < len(queue):
        g = queue[st.qpos]
        first = st.passes_done == 0 and g[0][1] in ("ceil", "sanity")
        if not first and items and len(items) + len(g) > budget:
            break
        items += g
        st.qpos += 1
    return items, n_dyn, st.qpos - q0


def adaptive_per_pass(model: str, n_rows: int) -> int:
    """Upper bound of adaptive spawn rows per pass (mixed pool + head pool per row)."""
    return n_rows * (C.HEAD_POOL + C.EXPERT_POOL[model] + C.HEAD_POOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto", choices=list(E13.PROTOS))
    ap.add_argument("--group", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--budget", type=int, default=0, help="spawn rows per pass (0 = Qwen3 100000, Mixtral 95000)")
    ap.add_argument("--max-passes", type=int, default=99)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--wf-chunk", type=int, default=2048)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    model = E13.MODEL_OF[args.proto]
    od = E13.step2_dir(args.proto, args.group, args.out_suffix)
    os.makedirs(od, exist_ok=True)
    conds = args.conds.split(",")
    keys = [E13.key_of(c, args.proto) for c in conds]
    cts, off = {}, 0
    for c, k in zip(conds, keys):
        cts[k] = E13.load_ctask(c, args.proto, off=off, limit=args.limit, suffix=args.out_suffix)
        off += cts[k].n_prefill
        log(f"{k}: {len(cts[k].rows)} rows, {len(cts[k].cases)} cases, prefill offset {cts[k].off}, "
            f"single-expert entries proxied by DLA {cts[k].n_proxy}")
    budget = args.budget or (100000 if model == "qwen3" else 95000)
    sp = os.path.join(od, "circuit_state.pkl")
    st = C.load_cstate(sp)
    queue = []
    for k in keys:
        queue += C.static_groups(cts[k], model)
    if st is None:
        st = C.CState(queue=[])
        for k in keys:
            ct = cts[k]
            for rid in ct.rows:
                pl = C.pools(ct, rid, model)
                for kind in C.GREEDY_KINDS:
                    st.greedy[(k, kind, rid)] = C.CGreedy(k, kind, rid, pl[kind], S=[pl[kind][0]])
        init = {"tasks": keys, "rows": {k: len(cts[k].rows) for k in keys}, "budget": budget,
                "queue_groups": len(queue), "queue_items": int(sum(len(g) for g in queue)),
                "pool_sizes": {k: {kind: len(st.greedy[(k, kind, cts[k].rows[0])].pool) for kind in C.GREEDY_KINDS} for k in keys},
                "n_proxy": {k: cts[k].n_proxy for k in keys}}
        st.notes.append(init)
        C.save_cstate(sp, st)
        write_meta(od, {"proto": args.proto, "model": model, "repo": MODELS[model]["repo"], "group": args.group, "tasks": keys,
                        "budget": budget, "greedy_steps": C.GREEDY_STEPS, "head_pool": C.HEAD_POOL, "expert_pool": C.EXPERT_POOL[model],
                        "k_grid": list(C.K_GRID), "static_orders": list(C.STATIC_ORDERS), "init": init, "agent": "p5-circuit",
                        "step1_dirs": {k: E13.step1_dir(c, args.proto, args.out_suffix) for c, k in zip(conds, keys)},
                        "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "complete": False})
        log(f"initialised: {init}")
    else:
        assert st.notes[0]["tasks"] == keys, f"state was created for tasks {st.notes[0]['tasks']}, not {keys}"
        log(f"resuming after {st.passes_done} passes (queue {st.qpos}/{len(queue)})")
    if args.dry_run:
        dyn = sum(len(g.items()) for g in st.greedy.values())
        log(f"[dry-run] static items {sum(len(g) for g in queue)}, first adaptive step {dyn}")
        return
    from moetrace import engine as E
    eng = E.Engine(MODELS[model]["repo"])
    pre, kinds, ckeys, ccase, crow = [], [], [], [], []
    for k in keys:
        ct = cts[k]
        cases = ct.t.cases.set_index("case_id")
        for c in ct.cases:
            r_ = cases.loc[c]
            pre.append(E.PrefillSpec(list(r_.clean_ids), int(r_.true_id), int(r_.foil_id), clean_ref=ct.ci[c]))
            kinds.append("clean"); ckeys.append(k); ccase.append(c); crow.append(-1)
        for rid in ct.rows:
            c = ct.case_of(rid)
            r_ = cases.loc[c]
            cid = list(ct.t.rows.corrupt_ids.iloc[rid])
            assert len(cid) == len(r_.clean_ids)
            pre.append(E.PrefillSpec(cid, int(r_.true_id), int(r_.foil_id), clean_ref=ct.ci[c]))
            kinds.append("corrupt"); ckeys.append(k); ccase.append(c); crow.append(rid)
    assert len(pre) == off
    kinds, ckeys, ccase, crow = map(np.array, (kinds, ckeys, ccase, crow))
    n_run = 0
    while n_run < args.max_passes:
        p = st.passes_done
        items, n_dyn, n_groups = build_pass(st, queue, budget)
        if not items:
            log("nothing left to run")
            if not json.load(open(os.path.join(od, "run_meta.json"))).get("complete"):
                write_meta(od, {"complete": True, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "passes": st.passes_done, "pass_notes": st.notes})
            break
        spawns = [spawn_of(cts, it, E.SpawnSpec) for it in items]
        log(f"pass {p}: {len(pre)} prefill rows, {len(spawns)} spawn rows ({n_dyn} adaptive, {n_groups} static groups; queue {st.qpos}/{len(queue)})")
        import gc; gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            res = eng.run(pre, spawns, record_routing=False, log=log, wf_chunk=args.wf_chunk, metrics=True)
        except torch.OutOfMemoryError:
            if n_groups == 0:
                raise
            st.qpos -= n_groups
            items = items[:n_dyn]
            spawns = spawns[:n_dyn]
            st.notes.append({"pass": p, "oom_retry_without_static": n_groups})
            log(f"OOM in pass {p}: retrying with the {n_dyn} adaptive rows only")
            gc.collect(); torch.cuda.empty_cache()
            res = eng.run(pre, spawns, record_routing=False, log=log, wf_chunk=args.wf_chunk, metrics=True)
            budget = max(n_dyn, int(budget * 0.85))
            log(f"budget lowered to {budget}")
        peak = torch.cuda.max_memory_allocated() / 2**30
        d = res.delta
        mp, ms = res.metrics_prefill, res.metrics_spawn
        pf = pd.DataFrame(dict(pass_=p, idx=np.arange(len(pre)), task=ckeys, kind=kinds, case_id=ccase, row_id=crow, delta=d,
                               top1=res.top1, rank_true=mp["rank_true"], p_true=mp["p_true"]))
        pf.to_parquet(os.path.join(od, f"circuit_prefill_p{p:02d}.parquet"), index=False)
        task = np.array([it[0] for it in items])
        rid = np.array([it[4] for it in items])
        dcl = np.array([d[cts[tk].ci[cts[tk].case_of(int(r_))]] for tk, r_ in zip(task, rid)])
        dco = np.array([d[cts[tk].cr[int(r_)]] for tk, r_ in zip(task, rid)])
        df = pd.DataFrame(dict(pass_=p, task=task, fam=[it[1] for it in items], order=[it[2] for it in items],
                               k=np.array([it[3] for it in items], dtype=np.int32), row_id=rid,
                               case_id=[cts[tk].case_of(int(r_)) for tk, r_ in zip(task, rid)],
                               delta=res.sp_logit_true - res.sp_logit_foil, delta_clean=dcl, delta_corrupt=dco,
                               set=[C.set_str(it[5]) if it[1].startswith("greedy") or it[1] in ("ceil", "sanity") else "" for it in items],
                               rank_true=ms["rank_true"], p_true=ms["p_true"], kl_to_clean=ms["kl_to_clean"]))
        df.to_parquet(os.path.join(od, f"circuit_rows_p{p:02d}.parquet"), index=False)
        gd = df[df.fam.str.startswith("greedy")]
        for (tk, fam, r_), g in gd.groupby(["task", "fam", "row_id"]):
            C.update_cgreedy(st.greedy[(tk, fam.split("_")[1], int(r_))], g, float(d[cts[tk].cr[int(r_)]]), p)
        st.passes_done += 1
        n_run += 1
        left = sum(not g.done for g in st.greedy.values())
        ch = df[(df.fam == "ceil") & (df.order == "all_heads")]
        chk = float(np.abs(ch.delta - ch.delta_clean).max()) if len(ch) else float("nan")
        gm = {}
        for kind in C.GREEDY_KINDS:
            v = [g.resc[-1] / (float(d[cts[g.task].ci[cts[g.task].case_of(g.rid)]]) - float(d[cts[g.task].cr[g.rid]]))
                 for g in st.greedy.values() if g.kind == kind and g.resc]
            gm[kind] = float(np.median(v)) if v else float("nan")
        st.notes.append({"pass": p, "spawn_rows": len(spawns), "adaptive": n_dyn, "static_groups": n_groups, "pass_s": res.extra["total_s"],
                         "peak_GB": peak, "greedy_left": left, "queue": f"{st.qpos}/{len(queue)}", "all_heads_max_abs_dev": chk,
                         "median_r_greedy": gm, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        C.save_cstate(sp, st)
        log(f"pass {p} done in {res.extra['total_s']:.0f}s, peak {peak:.1f} GB; greedy left {left}; median r: {gm}; "
            f"all-heads |Δ - Δ_clean| max {chk:.3f}; queue {st.qpos}/{len(queue)}")
        del res
        torch.cuda.empty_cache()
        if st.qpos >= len(queue) and left == 0:
            write_meta(od, {"complete": True, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "passes": st.passes_done,
                            "pass_notes": st.notes})
            log("ALL PASSES COMPLETE")
            break
    else:
        write_meta(od, {"passes": st.passes_done, "pass_notes": st.notes})


if __name__ == "__main__":
    main()
