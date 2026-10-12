"""ext13ko driver (Phase 5, p5-expert): expert knockout per condition (resumable pass by pass, frozen plans).

    python scripts/ext13ko_run.py <proto> --conds f1,dir_f1 [--phase main|cross] [--time-budget S] [--suffix _smoke]
           [--limit N] [--budget N] [--max-rows N] [--with-wr] [--max-passes N]

proto = qwen3 | mixtral_bos. For each condition (moetrace/ext13ko.py) the plan is frozen in
results/p5_<proto>_ko<suffix>/<cond>/plan_<phase>.json on the first call; passes already written are skipped. The job stops
starting new passes once --time-budget seconds have elapsed (queue jobs <= ~25 min); rerun to continue.
--with-wr: after a condition's knockout passes are complete, run its writer prefill pass (moetrace/ext13wr.py) in the same
job (same engine) unless it exists.
--phase cross: knock out the other ready conditions' selected experts on this condition (sub scope).

Per pass, in <od>:
  ko_rows_<phase>_<pass>.parquet  sig, item, delta, logit_true, logit_foil, top1, p_true, rank_true, logp_true, logp_foil,
                                  route_final (JSON: masked layer -> final-position experts, weights), moe_dla_masked (JSON:
                                  masked layer -> frozen-norm DLA of the whole final-position MoE output of that layer),
                                  final_rms
  ko_basefinal_<phase>_<pass>.npz baseline rows: final-position routing (route_idx, route_w) and per-slot DLA (contrib_dla,
                                  numerator) at every layer, final_rms
"""
import argparse, json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace import ext13ko as K
from moetrace.models import MODELS

T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def done_rows(od: str) -> set:
    out = set()
    for f in sorted(os.listdir(od)):
        if f.startswith("ko_rows_") and f.endswith(".parquet"):
            d = pd.read_parquet(os.path.join(od, f), columns=["sig", "item"])
            out |= set(zip(d.sig, d.item.astype(int)))
    return out


def update_meta(od, e, phase, note, argv):
    p = os.path.join(od, "run_meta.json")
    m = json.load(open(p)) if os.path.exists(p) else {}
    m.setdefault("cond", e["cond"])
    m.setdefault("proto", e["proto"])
    m.setdefault("model", e["model"])
    m.setdefault("repo", MODELS[e["model"]]["repo"])
    m.setdefault("agent", "p5-expert")
    m.setdefault("experiment", "ext13ko expert knockout (route masks on clean prompts), Phase 5 / Direction 9 per condition")
    m.setdefault("source", {"pairs": e["pairs"], "case_sets": e["case_sets"], "run": e["run"]})
    m.setdefault("special_tokens", True)
    m.setdefault("dtype", "bf16")
    ph = m.setdefault("phases", {}).setdefault(phase, {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "passes": []})
    ph["command"] = "python " + " ".join(argv)
    if note.get("pass"):
        ph["passes"] = [x for x in ph["passes"] if x.get("pass") != note["pass"]] + [note]
    for k_, v_ in note.items():
        if k_ in ("complete", "completed_utc", "n_passes", "n_conditions", "n_rows"):
            ph[k_] = v_
    json.dump(m, open(p, "w"), indent=1)


def get_plan(e, od, it, phase, args, cross=None):
    plan_path = os.path.join(od, f"plan_{phase}.json")
    if os.path.exists(plan_path):
        return json.load(open(plan_path))
    conds = K.conditions(e, od, it, phase, cross=cross)
    cfg = K.PROTO[e["proto"]]
    passes = K.plan_passes(it, conds, args.budget or cfg["budget"], args.max_rows or cfg["max_rows"], done=done_rows(od))
    plan = {"phase": phase, "cond": e["cond"], "proto": e["proto"],
            "targets": {k: (K.le(*v) if isinstance(v, tuple) else [K.le(*x) for x in v] if isinstance(v, list) else None)
                        for k, v in K.targets(e, od).items()},
            "conditions": [dict(c, sig=K.cond_sig(c), mask=[list(m) for m in c["mask"]]) for c in conds], "passes": passes}
    json.dump(plan, open(plan_path, "w"), indent=0)
    log(f"{e['cond']}: plan {phase} written: {len(conds)} conditions, {len(passes)} passes, {sum(len(p['rows']) for p in passes)} rows")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto", choices=list(K.PROTO))
    ap.add_argument("--conds", required=True)
    ap.add_argument("--phase", default="main", choices=["main", "cross"])
    ap.add_argument("--time-budget", type=float, default=1320.0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--limit", type=int, default=None, help="pairs per split (smoke)")
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-passes", type=int, default=0)
    ap.add_argument("--with-wr", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    proto = args.proto
    entries = [K.entry(c, proto) for c in args.conds.split(",") if c]
    for e in entries:  # the registry must carry the selection before a plan is frozen
        if e["cond"] in K.SMOKE:
            continue
        if not e.get("pop_top10") or "selected_expert" not in e["raw"] or not e.get("run"):
            raise SystemExit(f"registry entry {e['cond']}/{proto} incomplete (selected_expert / pop_top10 / run): {sorted(e['raw'])}")
        for f in ("str_sweep_routing.parquet", "str_expert_rows.parquet"):
            if not os.path.exists(os.path.join(K.RESULTS, e["run"], f)):
                raise SystemExit(f"{e['cond']}/{proto}: {e['run']}/{f} missing")
    eng = None
    E = None
    n_run = 0
    last_dt = 0.0
    out_of_time = False
    for e in entries:
        od = K.out_dir(e, args.suffix)
        os.makedirs(od, exist_ok=True)
        it = K.build_items(e, od, limit=args.limit)
        cross = None
        if args.phase == "cross":
            cross = []
            for c in K.ready_conditions(proto):
                if c != e["cond"]:
                    s_ = K.entry(c, proto)["selected"]
                    if s_:
                        cross.append((c, s_))
        plan = get_plan(e, od, it, args.phase, args, cross=cross)
        if args.plan_only:
            continue
        passes = plan["passes"]
        itx = it.set_index("item")

        def pass_done(name):
            return os.path.exists(os.path.join(od, f"ko_rows_{args.phase}_{name}.parquet")) or (
                os.path.exists(os.path.join(od, f"ko_split_{args.phase}_{name}.json")) and pass_done(name + "a") and pass_done(name + "b"))

        todo = [p for p in passes if not pass_done(p["name"])]
        log(f"{e['cond']}/{proto} {args.phase}: {len(todo)} of {len(passes)} passes to run")
        for p in todo:
            el = time.time() - T0
            if el + max(last_dt, 60.0) > args.time_budget or (args.max_passes and n_run >= args.max_passes):
                out_of_time = True
                break
            if eng is None:
                import importlib
                E = importlib.import_module("moetrace.engine")
                eng = E.Engine(MODELS[e["model"]]["repo"])
            t1 = time.time()
            run_pass(eng, E, e, od, itx, args.phase, p["name"], [(s_, int(x)) for s_, x in p["rows"]], n_run == 0, sys.argv)
            last_dt = time.time() - t1
            n_run += 1
        if out_of_time:
            break
        left = [p for p in passes if not pass_done(p["name"])]
        if not left:
            update_meta(od, e, args.phase, {"complete": True, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                            "n_passes": len(passes), "n_conditions": len(plan["conditions"]),
                                            "n_rows": sum(len(p["rows"]) for p in passes)}, sys.argv)
            log(f"{e['cond']}/{proto} {args.phase} complete")
            if args.with_wr and args.phase == "main":
                from moetrace import ext13wr as WR
                if not WR.pass_done(e, args.suffix):
                    if time.time() - T0 + max(last_dt, 90.0) > args.time_budget:
                        out_of_time = True
                        break
                    if eng is None:
                        import importlib
                        E = importlib.import_module("moetrace.engine")
                        eng = E.Engine(MODELS[e["model"]]["repo"])
                    WR.run_pass(eng, E, e, suffix=args.suffix, limit=args.limit, log=log)
    log(f"exit: {n_run} passes run, {time.time() - T0:.0f}s, out_of_time={out_of_time}")


def run_pass(eng, E, e, od, itx, phase, name, rows, verbose, argv):
    """One pass; on CUDA OOM split it into halves <name>a / <name>b (recorded in ko_split_<phase>_<name>.json)."""
    if os.path.exists(os.path.join(od, f"ko_rows_{phase}_{name}.parquet")):
        return
    pre = []
    for s_, x in rows:
        mask, pos, mode = K.parse_sig(s_)
        r = itx.loc[x]
        pre.append(E.PrefillSpec(list(r.ids), int(r.true_id), int(r.foil_id), route_mask=tuple(mask), route_mask_pos=pos,
                                 route_mask_mode=mode))
    diag = E.DiagSpec(contrib_dla=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    oom = False
    try:
        res = eng.run(pre, [], record_routing=True, diag=diag, metrics=True, log=log if verbose else None)
    except torch.cuda.OutOfMemoryError:
        oom = True
    if oom:
        del pre
        torch.cuda.empty_cache()
        h = len(rows) // 2
        log(f"pass {name}: CUDA OOM with {len(rows)} rows -> split {h} + {len(rows) - h}")
        json.dump({"rows": len(rows), "split": [h, len(rows) - h]}, open(os.path.join(od, f"ko_split_{phase}_{name}.json"), "w"))
        run_pass(eng, E, e, od, itx, phase, name + "a", rows[:h], False, argv)
        run_pass(eng, E, e, od, itx, phase, name + "b", rows[h:], False, argv)
        return
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    dg = res.extra["diag"]
    cd = dg["contrib_dla"]  # [L, B, k]
    fr = dg["final_rms"]  # [B]
    out = pd.DataFrame({"sig": [s_ for s_, _ in rows], "item": [x for _, x in rows]})
    out["delta"] = res.delta.astype(np.float32)
    out["logit_true"] = res.logit_true
    out["logit_foil"] = res.logit_foil
    out["top1"] = res.top1.astype(np.int64)
    if res.metrics_prefill is not None:
        for k_ in ("p_true", "rank_true", "logp_true", "logp_foil"):
            out[k_] = res.metrics_prefill[k_]
    rf, md = [], []
    for b, (s_, x) in enumerate(rows):
        mask, _, _ = K.parse_sig(s_)
        ls = sorted({l for l, _ in mask})
        rf.append(json.dumps({str(l): [res.route_idx[l, b].tolist(), np.round(res.route_w[l, b], 5).tolist()] for l in ls}))
        md.append(json.dumps({str(l): round(float(cd[l, b].sum() / fr[b]), 5) for l in ls}))
    out["route_final"] = rf
    out["moe_dla_masked"] = md
    out["final_rms"] = fr.astype(np.float32)
    bsel = np.array([b for b, (s_, _) in enumerate(rows) if s_ == "none|all|reroute"], dtype=np.int64)
    if len(bsel):
        np.savez_compressed(os.path.join(od, f"ko_basefinal_{phase}_{name}.npz"), items=np.array([rows[b][1] for b in bsel]),
                            route_idx=res.route_idx[:, bsel].astype(np.int16), route_w=res.route_w[:, bsel].astype(np.float16),
                            contrib_dla=cd[:, bsel].astype(np.float32), final_rms=fr[bsel].astype(np.float32),
                            n_experts=eng.spec.n_experts)
    out.to_parquet(os.path.join(od, f"ko_rows_{phase}_{name}.parquet"), index=False)
    Tmax = int(res.extra["T"])
    log(f"{e['cond']} pass {name}: {len(rows)} rows, T={Tmax}, padded {Tmax * len(rows)}, {dt:.1f}s, peak {peak:.1f} GB")
    update_meta(od, e, phase, {"pass": name, "rows": len(rows), "T": Tmax, "pass_s": round(dt, 1), "peak_GB": round(peak, 2)}, argv)
    del res, dg


if __name__ == "__main__":
    main()
