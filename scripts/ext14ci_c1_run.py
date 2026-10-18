"""ext14 C1 (p6-circuit): neurons inside selected experts. GPU stages (one engine pass = one checkpoint read):

  score   prefill clean + corrupted rows of every STR row (discovery + validation) with DiagSpec.expert_neurons for the
          task's experts and contrib_dla (final rms); per-neuron scores d / dla / mag; dense models also get the MLP-layer
          patch (kind layer) at every layer for every row (where does the MLP matter on these items).
  patch   one expert: validation rows where the expert is routed in the clean or the corrupted run (dense: rows passing
          the model's own margin): expert patch (100 % reference), all-neuron patch, top-k neuron patches for the
          rankings dla_case / mag_case / dla_pop / rand0 / rand1 over the k grid, single-neuron patches of the top-8 by
          |dla|. Rows are packed by case into passes of <= --max-spawn wavefront rows.

Usage: python scripts/ext14ci_c1_run.py <task> score [--limit N]
       python scripts/ext14ci_c1_run.py <task> patch --expert L44E069 [--max-spawn 30000] [--limit N]
Outputs results/p6ci_c1_<task>/ (see moetrace/ext14_circuit.py).
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext14_circuit as C
from moetrace.models import MODELS

log = C.log


def stage_score(task, limit=None):
    from moetrace.engine import Engine, DiagSpec, SpawnSpec
    cfg = C.C1_TASKS[task]
    t = C.load_items(task)
    od = C.task_dir(task)
    rows = t.rows.copy()
    if limit:  # smoke: first `limit` cases of each split
        keep = set(t.cases.groupby("split").head(limit).case_id)
        rows = rows[rows.case_id.isin(keep)]
    row_ids = [int(r) for r in rows.row_id]
    pre, lay = C.build_prefill(t, row_ids)
    experts = [tuple(x) for x in cfg["experts"]]
    eng = Engine(MODELS[cfg["model"]]["repo"])
    L = eng.spec.n_layers
    spawns, tags = [], []
    if cfg.get("sweep"):
        case_of = dict(zip(rows.row_id.astype(int), rows.case_id.astype(int)))
        for r in row_ids:
            c = case_of[r]
            for l in range(L):
                spawns.append(SpawnSpec(l, lay.corrupt_row[r], lay.clean_row[c], "layer"))
                tags.append((r, l))
    log(f"score {task}: {len(pre)} prefill rows, {len(spawns)} spawns, experts {experts}")
    t0 = time.time()
    res = eng.run(pre, spawns, record_routing=True, diag=DiagSpec(expert_neurons=tuple(experts), contrib_dla=True))
    log(f"pass {time.time() - t0:.0f}s")
    dg = res.extra["diag"]
    delta = res.delta
    cases = t.cases.set_index("case_id")
    rr = rows.set_index("row_id")
    ci = np.array([lay.clean_row[int(rr.case_id.loc[r])] for r in row_ids])
    oi = np.array([lay.corrupt_row[r] for r in row_ids])
    out = pd.DataFrame(dict(row_id=row_ids, case_id=[int(rr.case_id.loc[r]) for r in row_ids],
                            slot=[int(rr.slot.loc[r]) for r in row_ids], first=[bool(rr["first"].loc[r]) for r in row_ids]))
    out["split"] = out.case_id.map(cases.split).values
    out["unit"] = out.case_id.map(cases.unit).values
    out["true_id"] = out.case_id.map(cases.true_id).values.astype(int)
    out["foil_id"] = out.case_id.map(cases.foil_id).values.astype(int)
    out["delta_clean"] = delta[ci]
    out["delta_corrupt"] = delta[oi]
    out["rms_corrupt"] = dg["final_rms"][oi]
    out["rms_clean"] = dg["final_rms"][ci]
    disc = set(int(c) for c in t.cases.case_id[t.cases.split == "discovery"])
    for (l, e) in experts:
        nm = C.ename(l, e)
        dn = dg["expert_neurons"][(l, e)]
        out[f"w_clean_{nm}"] = dn["w"][ci]
        out[f"w_corrupt_{nm}"] = dn["w"][oi]
        out[f"routed_clean_{nm}"] = dn["routed"][ci]
        out[f"routed_corrupt_{nm}"] = dn["routed"][oi]
        # expert-level DLA from the routing slots (cross-check of sum_i dla_i)
        ri = res.route_idx[l]
        cd = dg["contrib_dla"][l]
        dla_e = np.array([(cd[a][ri[a] == e].sum() - cd[b][ri[b] == e].sum()) / dg["final_rms"][b] for a, b in zip(ci, oi)])
        out[f"dla_expert_{nm}"] = dla_e
        d, dla, wnorm = C.neuron_scores(eng, l, e, dn["act"][ci], dn["w"][ci], dn["act"][oi], dn["w"][oi],
                                        out.true_id.to_numpy(), out.foil_id.to_numpy(), out.rms_corrupt.to_numpy())
        out[f"dla_sum_{nm}"] = dla.sum(1)
        pm = C.pop_mean(out, dla, disc)
        C.save_scores(task, nm, np.array(row_ids), d, dla, wnorm, pm)
        log(f"{nm}: active clean {int((dn['w'][ci] > 0).sum())}/{len(ci)}, corrupt {int((dn['w'][oi] > 0).sum())}; "
            f"sum dla vs slot dla r={np.corrcoef(out[f'dla_sum_{nm}'], dla_e)[0, 1]:.4f}, max |diff| "
            f"{np.abs(out[f'dla_sum_{nm}'] - dla_e).max():.4f}")
    out.to_parquet(os.path.join(od, "c1_rows.parquet"))
    if spawns:
        sw = pd.DataFrame(tags, columns=["row_id", "layer"])
        sw["rescue"] = res.sp_delta - delta[[s.parent for s in spawns]]
        sw["vnorm"] = res.sp_vnorm
        sw.to_parquet(os.path.join(od, "c1_sweep_rows.parquet"))
    meta = dict(task=task, model=cfg["model"], items=cfg["items"], experts=[C.ename(*x) for x in experts],
                n_rows=len(row_ids), n_prefill=len(pre), limit=limit, pass_s=round(time.time() - t0, 1),
                command=" ".join(sys.argv), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(meta, open(os.path.join(od, "c1_score_meta.json"), "w"), indent=1)
    open(os.path.join(od, ".done_score"), "w").write(meta["utc"] + "\n")
    log("score done")


def patch_rows(task, nm, cfg):
    od = C.task_dir(task)
    R = pd.read_parquet(os.path.join(od, "c1_rows.parquet"))
    v = R[R.split == "validation"]
    act = (v[f"w_clean_{nm}"] > 0) | (v[f"w_corrupt_{nm}"] > 0)
    v = v[act]
    if cfg.get("margin") is not None:
        m = cfg["margin"]
        ok = (v.delta_clean >= m) & (v.delta_corrupt <= -m)
        if C.C1_TASKS[task]["items"][0] == "wino":  # both directions of a pair must pass (equivalent per row; keep pairs whole)
            okp = ok.groupby(v.unit).transform("all")
            ok = ok & okp
        v = v[ok]
    return v


def stage_patch(task, nm, max_spawn=30000, limit=None, n_single=C.N_SINGLE):
    from moetrace.engine import Engine, SpawnSpec
    cfg = C.C1_TASKS[task]
    l, e = C.parse_ename(nm)
    od = C.task_dir(task)
    t = C.load_items(task)
    v = patch_rows(task, nm, cfg)
    if limit:
        v = v[v.case_id.isin(list(dict.fromkeys(v.case_id))[:limit])]
    full = C.load_full(task, nm)
    pos = {int(r): j for j, r in enumerate(full["row_id"])}
    sel = np.array([pos[int(r)] for r in v.row_id])
    dla = full["dla"][sel].astype(np.float32)
    d = full["d"][sel].astype(np.float32)
    wnorm = full["wnorm"]
    comp = C.load_compact(task, nm)
    pop_order = np.argsort(-comp["disc_mean"], kind="stable")
    I = dla.shape[1]
    ks = C.k_grid(I)
    rid = v.row_id.to_numpy()
    ords = C.orders(dla, d, wnorm, pop_order, rid)
    top_abs = np.argsort(-np.abs(dla), axis=1, kind="stable")[:, :n_single]
    # spawn plan per row: (kind, ranking, k, neuron-array)
    plan = []
    for j, r in enumerate(rid):
        items = [("expert", "ref", 0, None), ("expert_neurons", "all", I, ords["dla_case"][j])]
        for rk in C.RANKINGS:
            for k in ks[:-1]:
                items.append(("expert_neurons", rk, k, ords[rk][j, :k]))
        for s in range(n_single):
            items.append(("expert_neurons", f"single{s}", 1, top_abs[j, s:s + 1].astype(np.int32)))
        plan.append(items)
    per_row = len(plan[0])
    # pack rows by case
    case_of = v.case_id.to_numpy()
    order_cases = list(dict.fromkeys(case_of))
    passes, cur, n_cur = [], [], 0
    for c in order_cases:
        js = list(np.nonzero(case_of == c)[0])
        if cur and n_cur + per_row * len(js) > max_spawn:
            passes.append(cur)
            cur, n_cur = [], 0
        cur += js
        n_cur += per_row * len(js)
    if cur:
        passes.append(cur)
    log(f"patch {task} {nm}: {len(rid)} rows ({len(order_cases)} cases), {per_row} spawns/row, {len(passes)} passes, k grid {ks}")
    eng = Engine(MODELS[cfg["model"]]["repo"])
    parts = []
    for pi, js in enumerate(passes):
        pfile = os.path.join(od, f"c1_patch_{nm}_p{pi:02d}.parquet")
        if os.path.exists(pfile):
            parts.append(pd.read_parquet(pfile))
            continue
        pre, lay = C.build_prefill(t, [int(rid[j]) for j in js])
        spawns, tags = [], []
        for j in js:
            r = int(rid[j])
            c = int(case_of[j])
            for kind, rk, k, nn in plan[j]:
                if kind == "expert":
                    spawns.append(SpawnSpec(l, lay.corrupt_row[r], lay.clean_row[c], "expert", expert=e))
                    pred = float(dla[j].sum())
                else:
                    spawns.append(SpawnSpec(l, lay.corrupt_row[r], lay.clean_row[c], "expert_neurons", expert=e, neurons=nn))
                    pred = float(dla[j, nn].sum())
                tags.append((r, c, rk, k, pred, int(nn[0]) if (nn is not None and k == 1) else -1))
        t0 = time.time()
        res = eng.run(pre, spawns, record_routing=False)
        par = np.array([s.parent for s in spawns])
        df = pd.DataFrame(tags, columns=["row_id", "case_id", "ranking", "k", "pred_dla", "neuron"])
        df["rescue"] = res.sp_delta - res.delta[par]
        df["vnorm"] = res.sp_vnorm
        df["delta_corrupt_pass"] = res.delta[par]
        df.to_parquet(pfile)
        parts.append(df)
        log(f"pass {pi + 1}/{len(passes)}: {len(pre)} prefill, {len(spawns)} spawns, {time.time() - t0:.0f}s")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(os.path.join(od, f"c1_patch_{nm}.parquet"))
    for pi in range(len(passes)):
        p = os.path.join(od, f"c1_patch_{nm}_p{pi:02d}.parquet")
        if os.path.exists(p):
            os.remove(p)
    meta = dict(task=task, expert=nm, n_rows=int(len(rid)), n_cases=len(order_cases), per_row=per_row, k_grid=ks,
                passes=len(passes), max_spawn=max_spawn, limit=limit, command=" ".join(sys.argv),
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(meta, open(os.path.join(od, f"c1_patch_{nm}_meta.json"), "w"), indent=1)
    open(os.path.join(od, f".done_patch_{nm}"), "w").write(meta["utc"] + "\n")
    log("patch done")


def stage_oracle(task, nm, n_rows=128, max_spawn=30000):
    """Exact single-neuron patches of EVERY neuron of the expert on the first n_rows patch rows (whole cases, case order),
    then top-k patches ordered by the row's own exact single-neuron effects (ranking 'oracle'). Qwen3 experts (768 neurons)."""
    from moetrace.engine import Engine, SpawnSpec
    cfg = C.C1_TASKS[task]
    l, e = C.parse_ename(nm)
    od = C.task_dir(task)
    t = C.load_items(task)
    v = patch_rows(task, nm, cfg)
    keep, n = [], 0
    for c in dict.fromkeys(v.case_id):
        if n >= n_rows:
            break
        keep.append(c)
        n += int((v.case_id == c).sum())
    v = v[v.case_id.isin(keep)]
    full = C.load_full(task, nm)
    pos = {int(r): j for j, r in enumerate(full["row_id"])}
    dla = full["dla"][[pos[int(r)] for r in v.row_id]].astype(np.float32)
    I = dla.shape[1]
    rid, case_of = v.row_id.to_numpy(), v.case_id.to_numpy()
    eng = Engine(MODELS[cfg["model"]]["repo"])

    def run_plan(plan_of_row, tag):
        per_row = len(plan_of_row(0))
        passes, cur = [], []
        for c in dict.fromkeys(case_of):
            js = list(np.nonzero(case_of == c)[0])
            if cur and per_row * (len(cur) + len(js)) > max_spawn:
                passes.append(cur)
                cur = []
            cur += js
        if cur:
            passes.append(cur)
        parts = []
        for pi, js in enumerate(passes):
            pre, lay = C.build_prefill(t, [int(rid[j]) for j in js])
            spawns, tags = [], []
            for j in js:
                r, c = int(rid[j]), int(case_of[j])
                for rk, k, nn in plan_of_row(j):
                    if nn is None:
                        spawns.append(SpawnSpec(l, lay.corrupt_row[r], lay.clean_row[c], "expert", expert=e))
                        pred = float(dla[j].sum())
                    else:
                        spawns.append(SpawnSpec(l, lay.corrupt_row[r], lay.clean_row[c], "expert_neurons", expert=e, neurons=nn))
                        pred = float(dla[j, nn].sum())
                    tags.append((r, c, rk, k, pred, int(nn[0]) if (nn is not None and k == 1) else -1))
            t0 = time.time()
            res = eng.run(pre, spawns, record_routing=False)
            par = np.array([s.parent for s in spawns])
            df = pd.DataFrame(tags, columns=["row_id", "case_id", "ranking", "k", "pred_dla", "neuron"])
            df["rescue"] = res.sp_delta - res.delta[par]
            parts.append(df)
            log(f"{tag} pass {pi + 1}/{len(passes)}: {len(spawns)} spawns, {time.time() - t0:.0f}s")
        return pd.concat(parts, ignore_index=True)

    neur = [np.array([i], dtype=np.int32) for i in range(I)]
    sing = run_plan(lambda j: [("ref", 0, None)] + [("oracle_single", 1, neur[i]) for i in range(I)], "singles")
    sing.to_parquet(os.path.join(od, f"c1_oracle_singles_{nm}.parquet"))
    S = sing[sing.ranking == "oracle_single"].pivot(index="row_id", columns="neuron", values="rescue").reindex(rid)
    order = np.argsort(-S.to_numpy(), axis=1, kind="stable").astype(np.int32)
    ks = C.k_grid(I)
    cur = run_plan(lambda j: [("ref", 0, None)] + [("oracle", k, order[j, :k]) for k in ks[:-1]], "oracle top-k")
    cur.to_parquet(os.path.join(od, f"c1_oracle_{nm}.parquet"))
    open(os.path.join(od, f".done_oracle_{nm}"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    log("oracle done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(C.C1_TASKS))
    ap.add_argument("stage", choices=["score", "patch", "oracle"])
    ap.add_argument("--n-rows", type=int, default=128)
    ap.add_argument("--expert", default=None)
    ap.add_argument("--max-spawn", type=int, default=30000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="write to results/p6ci_c1_<task>_smoke")
    a = ap.parse_args()
    if a.smoke:
        C.DIR_SUFFIX = "_smoke"
    if a.stage == "score":
        stage_score(a.task, a.limit)
    elif a.stage == "oracle":
        assert a.expert, "--expert required"
        stage_oracle(a.task, a.expert, a.n_rows, a.max_spawn)
    else:
        assert a.expert, "--expert required"
        stage_patch(a.task, a.expert, a.max_spawn, a.limit)


if __name__ == "__main__":
    main()
