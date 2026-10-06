"""ext9 knockout driver (one GPU job per model and phase; resumable pass by pass).

    python scripts/ext9_knockout_run.py qwen3|mixtral --phase full|random|samelayer|final [--engine main|dev]
           [--max-passes N] [--budget N] [--max-rows N] [--wiki-budget N]

Phases (moetrace/ext9_knockout.py): full = baseline + targets + population sets on the full scope; random = frequency-matched
random sets (sub scope); samelayer = same-layer controls (sub scope); final = final-position-only masks (targets, sets) and
zero mode (targets). random / samelayer need the baseline routing of the full phase (controls.json is built from it).

Per pass, in results/<model>_knockout/:
  ko_rows_<phase>_<pass>.parquet   sig, item, task, delta, logit_true, logit_foil, top1, p_true, rank_true, logp_true,
                                   logp_foil (short prompts), nll_mean (wiki), route_final (JSON: masked layer -> final-
                                   position experts and weights)
  ko_wiki_<phase>_<pass>.npz       per-token NLL [rows, 126] of the wiki windows (sig, item)
  ko_basefinal_<pass>.npz          baseline rows: final-position routing at every layer (route_idx [L, n, k], route_w)
  ko_baseallpos_<pass>.npz         baseline rows: all-position routing counts per (task, watch layer, expert) and per-row
                                   position counts of the watched experts
The pass plan is frozen in plan_<phase>.json on the first call (rows already computed in earlier phases are skipped).
"""
import argparse, importlib, json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # fewer fragmentation OOMs (allocation only)
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace import ext9_knockout as K
from moetrace.models import MODELS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def done_rows(od: str) -> set:
    out = set()
    for f in sorted(os.listdir(od)):
        if f.startswith("ko_rows_") and f.endswith(".parquet"):
            d = pd.read_parquet(os.path.join(od, f), columns=["sig", "item"])
            out |= set(zip(d.sig, d.item.astype(int)))
    return out


def update_meta(od: str, key: str, phase: str, note: dict, args):
    p = os.path.join(od, "run_meta.json")
    m = json.load(open(p)) if os.path.exists(p) else {}
    m.setdefault("model", K.KO[key]["model"])
    m.setdefault("repo", MODELS[K.KO[key]["model"]]["repo"])
    m.setdefault("agent", "ext9-knockout")
    m.setdefault("experiment", "ext9 expert knockout (route masks on clean prompts), Phase 4 item 1 / Direction 9")
    m.setdefault("special_tokens", True)
    m.setdefault("dtype", "bf16")
    ph = m.setdefault("phases", {}).setdefault(phase, {"command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "passes": []})
    ph["command"] = "python " + " ".join(sys.argv)
    if note.get("pass"):
        ph["passes"] = [x for x in ph["passes"] if x.get("pass") != note["pass"]] + [note]
    for k_, v_ in note.items():
        if k_ in ("complete", "completed_utc", "n_passes", "n_conditions", "n_rows"):
            ph[k_] = v_
    json.dump(m, open(p, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(K.KO))
    ap.add_argument("--phase", required=True, choices=K.PHASES)
    ap.add_argument("--engine", default="main", choices=["main", "dev"])
    ap.add_argument("--max-passes", type=int, default=0)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--wiki-budget", type=int, default=0)
    args = ap.parse_args()
    key, phase = args.model, args.phase
    cfg = K.KO[key]
    od = K.run_dir(key)
    os.makedirs(od, exist_ok=True)
    it = K.build_items(key)
    conds = K.conditions(key, phase)
    plan_path = os.path.join(od, f"plan_{phase}.json")
    if os.path.exists(plan_path):
        plan = json.load(open(plan_path))
        passes = plan["passes"]
        log(f"plan loaded: {len(passes)} passes")
    else:
        done = done_rows(od)
        passes = K.plan_passes(key, it, conds, done=done, budget=args.budget or None, max_rows=args.max_rows or None,
                               wiki_budget=args.wiki_budget or None)
        for p in passes:
            p["rows"] = [[s_, int(x)] for s_, x in p["rows"]]
        json.dump({"phase": phase, "conditions": [dict(c, sig=K.cond_sig(c), mask=[list(m) for m in c["mask"]]) for c in conds],
                   "skipped_done_rows": len(done), "passes": passes}, open(plan_path, "w"), indent=0)
        log(f"plan written: {len(conds)} conditions, {len(passes)} passes, {sum(len(p['rows']) for p in passes)} rows")
    E = importlib.import_module("moetrace.engine_ext9_dev" if args.engine == "dev" else "moetrace.engine")

    def pass_done(name):
        return os.path.exists(os.path.join(od, f"ko_rows_{phase}_{name}.parquet")) or (
            os.path.exists(os.path.join(od, f"ko_split_{phase}_{name}.json")) and pass_done(name + "a") and pass_done(name + "b"))

    todo = [p for p in passes if not pass_done(p["name"])]
    log(f"{len(todo)} of {len(passes)} passes to run")
    if not todo:
        update_meta(od, key, phase, {"complete": True, "n_passes": len(passes), "n_conditions": len(conds),
                                     "n_rows": sum(len(p["rows"]) for p in passes)}, args)
        return
    eng = E.Engine(MODELS[cfg["model"]]["repo"])
    L, NE, Kk = eng.spec.n_layers, eng.spec.n_experts, eng.spec.top_k
    itx = it.set_index("item")
    pr = K.population_ranking(key)
    watch_pairs = sorted(set(K.all_targets(key)) | {(l, e) for t in ("wg", "cf") for l, e, _ in pr[t][:10]})
    watch_layers = tuple(sorted({l for l, _ in watch_pairs}))
    tasks = ("wg", "cf", "ioi", "wiki")
    nll_sl = K.wiki_nll_slice(key)
    def do_pass(name, rows, wiki, verbose):
        """Run one pass; on CUDA OOM split it into halves <name>a / <name>b (recorded in ko_split_<phase>_<name>.json)."""
        if pass_done(name):
            return
        pre = []
        for s_, x in rows:
            mask, pos, mode = K.parse_sig(s_)
            r = itx.loc[x]
            pre.append(E.PrefillSpec(list(r.ids), int(r.true_id), int(r.foil_id), route_mask=tuple(mask), route_mask_pos=pos,
                                     route_mask_mode=mode))
        has_base = any(s_ == "none|all|reroute" for s_, _ in rows)
        diag = E.DiagSpec(token_logprobs=bool(wiki), route_all_layers=watch_layers if (has_base and phase == "full") else ())
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        oom = False
        try:
            res = eng.run(pre, [], record_routing=True, diag=diag, metrics=not wiki, log=log if verbose else None)
        except torch.cuda.OutOfMemoryError:
            oom = True  # (handled outside the except block so that the failed pass's tensors are released first)
        if oom:
            del pre
            torch.cuda.empty_cache()
            h = len(rows) // 2
            log(f"pass {name}: CUDA OOM with {len(rows)} rows -> split {h} + {len(rows) - h}")
            json.dump({"rows": len(rows), "split": [h, len(rows) - h]}, open(os.path.join(od, f"ko_split_{phase}_{name}.json"), "w"))
            do_pass(name + "a", rows[:h], wiki, False)
            do_pass(name + "b", rows[h:], wiki, False)
            return
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        dg = res.extra["diag"]
        out = pd.DataFrame({"sig": [s_ for s_, _ in rows], "item": [x for _, x in rows]})
        out["task"] = itx.task.loc[out.item].values
        out["delta"] = res.delta.astype(np.float32)
        out["logit_true"] = res.logit_true
        out["logit_foil"] = res.logit_foil
        out["top1"] = res.top1.astype(np.int64)
        if res.metrics_prefill is not None:
            for k_ in ("p_true", "rank_true", "logp_true", "logp_foil"):
                out[k_] = res.metrics_prefill[k_]
        rf = []
        for b, (s_, x) in enumerate(rows):
            mask, _, _ = K.parse_sig(s_)
            ls = sorted({l for l, _ in mask})
            rf.append(json.dumps({str(l): [res.route_idx[l, b].tolist(), np.round(res.route_w[l, b], 5).tolist()] for l in ls}))
        out["route_final"] = rf
        if wiki:
            lp = dg["token_logprobs"][:, nll_sl].astype(np.float32)  # [B, 126]
            out["nll_mean"] = -lp.mean(1)
            np.savez_compressed(os.path.join(od, f"ko_wiki_{phase}_{name}.npz"), sig=np.array([s_ for s_, _ in rows]),
                                item=np.array([x for _, x in rows]), nll=-lp)
        if has_base:
            bsel = np.array([b for b, (s_, _) in enumerate(rows) if s_ == "none|all|reroute"])
            np.savez_compressed(os.path.join(od, f"ko_basefinal_{phase}_{name}.npz"), items=np.array([rows[b][1] for b in bsel]),
                                route_idx=res.route_idx[:, bsel].astype(np.int16), route_w=res.route_w[:, bsel].astype(np.float16),
                                n_experts=NE)
            if diag.route_all_layers:
                lens = np.array([len(pre[b].ids) for b in bsel])
                cnt = np.zeros((len(tasks), len(watch_layers), NE), dtype=np.int64)
                tot = np.zeros(len(tasks), dtype=np.int64)
                row_cnt = np.zeros((len(bsel), len(watch_pairs)), dtype=np.int32)
                tk = np.array([tasks.index(itx.task.loc[rows[b][1]]) for b in bsel])
                valid = np.arange(dg["route_all"][watch_layers[0]][0].shape[1])[None, :] < lens[:, None]  # [n, T]
                for ti in range(len(tasks)):
                    tot[ti] = valid[tk == ti].sum()
                for wi, l in enumerate(watch_layers):
                    topi = dg["route_all"][l][0][bsel].astype(np.int64)  # [n, T, k] (top-k experts are distinct per token)
                    for ti in range(len(tasks)):
                        m_ = valid & (tk == ti)[:, None]
                        cnt[ti, wi] = np.bincount(topi[m_].reshape(-1), minlength=NE)
                    for pj, (l2, e) in enumerate(watch_pairs):
                        if l2 == l:
                            row_cnt[:, pj] = ((topi == e).any(-1) & valid).sum(1)
                np.savez_compressed(os.path.join(od, f"ko_baseallpos_{phase}_{name}.npz"), tasks=np.array(tasks),
                                    watch_layers=np.array(watch_layers), counts=cnt, totals=tot, items=np.array([rows[b][1] for b in bsel]),
                                    lens=lens, watch_pairs=np.array(watch_pairs), row_counts=row_cnt)
        out.to_parquet(os.path.join(od, f"ko_rows_{phase}_{name}.parquet"), index=False)
        Tmax = int(res.extra["T"])
        log(f"pass {name}: {len(rows)} rows, T={Tmax}, padded {Tmax * len(rows)}, {dt:.1f}s, peak {peak:.1f} GB")
        update_meta(od, key, phase, {"pass": name, "rows": len(rows), "T": Tmax, "pass_s": round(dt, 1), "peak_GB": round(peak, 2)}, args)
        del res, dg

    n_run = 0
    for p in todo:
        if args.max_passes and n_run >= args.max_passes:
            break
        do_pass(p["name"], [(s_, int(x)) for s_, x in p["rows"]], bool(p["wiki"]), n_run == 0)
        n_run += 1
    left = [p for p in passes if not pass_done(p["name"])]
    if not left:
        update_meta(od, key, phase, {"complete": True, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     "n_passes": len(passes), "n_conditions": len(conds), "n_rows": sum(len(p["rows"]) for p in passes)}, args)
        log(f"phase {phase} complete")


if __name__ == "__main__":
    main()
