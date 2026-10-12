"""ext13ab (Phase 5): multi-task add-back / deletion driver (ext8 machinery; several Phase-5 conditions of ONE model share
every engine pass). Resumable: every task keeps its own run dir, state and files in exactly the ext8 format
(results/p5_<cond>_<proto>[_rep]_ab/: addback_rows_pNN.parquet, addback_prefill_pNN.parquet, addback_state.pkl,
addback_dla.parquet, addback_direct.parquet, run_meta.json), so moetrace.ext8_addback.load_out and
scripts/ext8_addback_analyze.analyse read them unchanged.

Per task (moetrace/ext13ab.py): A0 ceilings (all rows, add-back and deletion), A1 static orderings + all-clean-active
endpoint, A2 greedy (15 steps), noising singles, A3 deletion curves incl. dla and noise_oracle.

A pass carries, for every active task, the prefill rows it needs (a task's first pass: every case and row, in the
canonical ext8 order, because A0 runs on discovery rows too and the diagnostics for the DLA ordering / direct split are
sliced from it; later passes: only the validation cases and rows, the only ones later work refers to), all of the
task's adaptive greedy rows, and static groups taken round-robin from the active tasks' queues up to --budget spawn rows.
At most --max-tasks tasks are active per pass; tasks already started have priority, then the plan order.

Usage: python scripts/ext13ab_run.py <qwen3|mixtral> [--tasks key,key] [--smoke] [--max-tasks 3] [--budget 45000]
       [--max-passes 99] [--dry-run]
"""
import argparse, dataclasses, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, pandas as pd
from moetrace.models import MODELS, RESULTS
from moetrace import ext8_addback as X
from moetrace import ext13ab as AB

METRICS = ("logp_true", "logp_foil", "p_true", "p_foil", "rank_true", "kl_to_clean")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_meta(od, upd):
    p = os.path.join(od, "run_meta.json")
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.update(upd)
    with open(p, "w") as f:
        json.dump(cur, f, indent=1, default=str)


class TaskRun:
    def __init__(self, spec: AB.TaskSpec, args):
        self.spec = spec
        self.key = spec.key
        self.od = os.path.join(RESULTS, spec.out)
        if not args.dry_run:
            os.makedirs(self.od, exist_ok=True)
        self.sp = os.path.join(self.od, "addback_state.pkl")
        self.t = AB.load_task(spec)
        t = self.t
        self.n_noise = spec.noise_rows
        self.st = X.load_state(self.sp)
        if self.st is None:
            st = X.State(queue=[])
            pool = min(spec.pool, t.K)
            ev = AB.eval_rows(t)
            for rid in ev:
                o = X.ordering(t, rid, "oracle")[:pool]
                st.greedy[rid] = X.Greedy(rid, o, S=[o[0]])
            q = AB.static_queue(t, self.n_noise)
            st.notes.append({"eval_rows": len(ev), "noise_rows": len(AB.noise_rows_of(t, self.n_noise)), "pool": pool,
                             "queue_groups": len(q), "queue_items": int(sum(len(g) for g in q))})
            self.st = st
            meta = {"model": spec.model, "repo": MODELS[spec.model]["repo"], "task": "wino", "p5_task": spec.key, "cond": spec.cond,
                    "proto": spec.proto, "family": spec.family, "src_run": spec.src, "pairs": spec.pairs, "case_sets": spec.case_sets,
                    "special_tokens": t.special_tokens, "pool": pool, "noise_rows": len(AB.noise_rows_of(t, self.n_noise)),
                    "limit": spec.limit, "greedy_steps": X.GREEDY_STEPS, "cluster_col": spec.cluster_col, "budget": args.budget, "n_cases": t.nC, "n_rows": len(t.rows),
                    "n_eval_rows": len(ev), "init": st.notes[0], "agent": "p5-addback",
                    "driver": "scripts/ext13ab_run.py (multi-task; ext8 machinery)",
                    "work": "A0 (both directions, all rows), A1 static orderings + all_active, greedy 15 steps, noising singles, "
                            "A3 deletion curves (pop/oracle/layerwise/rand0-1/dla/noise_oracle) + all_active; no beam / exact / Shapley",
                    "note": f"family {spec.family}: population ranking from its discovery split, curves on its validation split; "
                            "prefill rows of later passes = validation cases only (first pass: all)",
                    "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "complete": False}
            if not args.dry_run:
                X.save_state(self.sp, st)
                write_meta(self.od, meta)
            log(f"{self.key}: initialised {st.notes[0]}")
        self.queue = AB.static_queue(t, self.n_noise)
        if self.st.dla_added:
            self.queue += AB.dla_queue(t, self.od)
        if self.st.noise_oracle_added:
            self.queue += AB.noise_oracle_queue(t, self.od)[0]
        ev = set(AB.eval_rows(t))
        self.val_rows = [int(r) for r in t.rows.row_id if int(r) in ev]
        vc = set(int(t.rows.case_id.iloc[r]) for r in self.val_rows)
        self.val_cases = [i for i, c in enumerate(t.cases.case_id) if int(c) in vc]

    @property
    def done(self):
        return self.st.qpos >= len(self.queue) and all(g.done for g in self.st.greedy.values())

    @property
    def first(self):
        return self.st.passes_done == 0

    def needed(self):
        """(case indices, row ids) whose prefill rows this task needs in the next pass."""
        if self.first:
            return list(range(self.t.nC)), [int(r) for r in self.t.rows.row_id]
        return self.val_cases, self.val_rows

    def adaptive_items(self):
        its = []
        for g in self.st.greedy.values():
            its += g.items()
        return its

    def a0_items(self):
        n, q = 0, self.st.qpos
        while q < len(self.queue) and self.queue[q][0][0] == "a0":
            n += len(self.queue[q])
            q += 1
        return n


def select(runs, args):
    order = {c: i for i, c in enumerate(AB.PLAN_ORDER)}
    cand = [r for r in runs if not r.done]
    cand.sort(key=lambda r: (r.first, order.get(r.spec.cond, 99), r.spec.family != "main", r.key))
    act, tot = [], 0
    for r in cand:
        if len(act) >= args.max_tasks:
            break
        n = len(r.adaptive_items()) + (r.a0_items() if r.first else 0)
        if act and tot + n > args.budget:
            continue
        act.append(r)
        tot += n
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["qwen3", "mixtral"])
    ap.add_argument("--tasks", default=None, help="comma list of registry keys (default: every ready, incomplete task of the model)")
    ap.add_argument("--smoke", action="store_true", help="smoke task on results/wino_ref_F1_<proto> (16 + 16 directed cases)")
    ap.add_argument("--smoke-n", type=int, default=16)
    ap.add_argument("--cond-json", default=AB.COND_JSON)
    ap.add_argument("--max-tasks", type=int, default=3)
    ap.add_argument("--budget", type=int, default=45000)
    ap.add_argument("--max-passes", type=int, default=99)
    ap.add_argument("--greedy-steps", type=int, default=X.GREEDY_STEPS, help="greedy steps (15 as ext8; smaller only for smoke tests)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    X.GREEDY_STEPS = args.greedy_steps
    from ext8_addback_run import dla_table, direct_split
    if args.smoke:
        specs = [AB.smoke_spec(args.model, args.smoke_n, args.smoke_n)]
    else:
        reg = AB.registry(args.cond_json, ready_only=True)
        keys = args.tasks.split(",") if args.tasks else [k for k, s in reg.items() if s.model == args.model]
        specs = [reg[k] for k in keys]
        assert all(s.model == args.model for s in specs)
        specs = [s for s in specs if not AB.run_complete(s.out)]
    if not specs:
        log("no ready, incomplete task for this model")
        return
    runs = [TaskRun(s, args) for s in specs]
    for r in runs:
        log(f"{r.key}: {r.t.nC} cases, {len(r.t.rows)} rows ({len(r.val_rows)} validation), K={r.t.K}; passes done {r.st.passes_done}, "
            f"queue {r.st.qpos}/{len(r.queue)}, greedy left {sum(not g.done for g in r.st.greedy.values())}")
    if args.dry_run:
        for r in runs:
            log(f"[dry-run] {r.key}: static items {sum(len(g) for g in r.queue)}, first dynamic step {len(r.adaptive_items())}")
        log(f"[dry-run] first selection: {[r.key for r in select(runs, args)]}")
        return
    import torch
    import moetrace.engine as E
    eng = E.Engine(MODELS[args.model]["repo"])
    n_run = 0
    while n_run < args.max_passes:
        act = select(runs, args)
        if not act:
            log("nothing left to run")
            break
        # ---- prefill rows
        pre, cpos, rpos, span = [], [], [], []
        for r in act:
            t = r.t
            cis, rids = r.needed()
            a = len(pre)
            cp = {}
            for ci in cis:
                c = t.cases.iloc[ci]
                cp[ci] = len(pre)
                pre.append(E.PrefillSpec(list(c.clean_ids), int(c.true_id), int(c.foil_id), clean_ref=len(pre)))
            rp = {}
            for rid in rids:
                rr = t.rows.iloc[rid]
                ci = t.case_index[int(rr.case_id)]
                c = t.cases.iloc[ci]
                assert len(rr.corrupt_ids) == len(c.clean_ids)
                rp[rid] = len(pre)
                pre.append(E.PrefillSpec(list(rr.corrupt_ids), int(c.true_id), int(c.foil_id), clean_ref=cp[ci]))
            cpos.append(cp)
            rpos.append(rp)
            span.append((a, len(pre)))
        # ---- spawn items
        items = [r.adaptive_items() for r in act]
        n_dyn = [len(x) for x in items]
        total = sum(n_dyn)
        q0 = [r.st.qpos for r in act]
        first = [r.first for r in act]
        for j, r in enumerate(act):  # A0 groups of a task's first pass are forced (ceilings; discovery rows only then)
            if first[j]:
                while r.st.qpos < len(r.queue) and r.queue[r.st.qpos][0][0] == "a0":
                    items[j] += r.queue[r.st.qpos]
                    total += len(r.queue[r.st.qpos])
                    r.st.qpos += 1
        progress = True
        while progress:  # round-robin static groups up to the budget
            progress = False
            for j, r in enumerate(act):
                if r.st.qpos >= len(r.queue):
                    continue
                g = r.queue[r.st.qpos]
                if g[0][0] == "a0":  # never outside the first pass (prefill of discovery rows absent)
                    raise RuntimeError(f"{r.key}: A0 group outside the first pass")
                if total + len(g) > args.budget and total > 0:
                    continue
                items[j] += g
                r.st.qpos += 1
                total += len(g)
                progress = True
        spawns, sp_off = [], []
        for j, r in enumerate(act):
            sp_off.append(len(spawns))
            t = r.t
            for (fam, order, dr, k, rid, payload) in items[j]:
                ci = t.case_index[int(t.rows.case_id.iloc[rid])]
                c_i, r_i = cpos[j][ci], rpos[j][rid]
                par, src = (r_i, c_i) if dr == "d" else (c_i, r_i)
                steps = X.steps_of(t, payload)
                spawns.append(E.SpawnSpec(steps[0][0], par, src, "multi", steps=steps, kl_ref=c_i))
        sp_off.append(len(spawns))
        any_first = any(first)
        diag = E.DiagSpec(contrib_dla=True, attn_out_final=True, resid_final=True) if any_first else None
        log(f"pass {n_run}: tasks {[r.key for r in act]}, {len(pre)} prefill rows (T max {max(len(p.ids) for p in pre)}), {len(spawns)} spawn rows "
            f"(adaptive {n_dyn}, static groups {[r.st.qpos - q for r, q in zip(act, q0)]})")
        torch.cuda.reset_peak_memory_stats()
        res = eng.run(pre, spawns, record_routing=any_first, log=log if n_run == 0 else None, diag=diag, metrics=True)
        peak = torch.cuda.max_memory_allocated() / 2**30
        mp, ms = res.metrics_prefill, res.metrics_spawn
        for j, r in enumerate(act):
            t, st, od = r.t, r.st, r.od
            p = st.passes_done
            nC = t.nC
            cis = sorted(cpos[j], key=lambda ci: cpos[j][ci])
            rids = sorted(rpos[j], key=lambda rid: rpos[j][rid])
            gi = np.array([cpos[j][ci] for ci in cis] + [rpos[j][rid] for rid in rids], dtype=np.int64)
            row_case_all = t.rows.case_id.values
            row_slot_all = t.rows.slot.values
            pf = pd.DataFrame(dict(pass_=p, idx=np.array(list(cis) + [nC + rid for rid in rids], dtype=np.int64),
                                   kind=["clean"] * len(cis) + ["corrupt"] * len(rids),
                                   case_id=np.concatenate([t.cases.case_id.values[cis], row_case_all[rids]]).astype(np.int64),
                                   row_id=np.concatenate([np.full(len(cis), -1), np.array(rids, dtype=np.int64)]),
                                   slot=np.concatenate([np.full(len(cis), -1), row_slot_all[rids]]), delta=res.delta[gi], top1=res.top1[gi],
                                   **{k: np.asarray(mp[k])[gi] for k in METRICS}))
            pf.to_parquet(os.path.join(od, f"addback_prefill_p{p:02d}.parquet"), index=False)
            s0, s1 = sp_off[j], sp_off[j + 1]
            its = items[j]
            fam = [it[0] for it in its]
            rid = np.array([it[4] for it in its], dtype=np.int64)
            dcl = np.array([res.delta[cpos[j][t.case_index[int(row_case_all[x])]]] for x in rid]) if len(rid) else np.zeros(0)
            dco = np.array([res.delta[rpos[j][int(x)]] for x in rid]) if len(rid) else np.zeros(0)
            keep_set = {"greedy", "beam", "exact", "single_noise"}
            df = pd.DataFrame(dict(pass_=p, fam=fam, order=[it[1] for it in its], dir=[it[2] for it in its],
                                   k=np.array([it[3] for it in its], dtype=np.int32), row_id=rid, case_id=row_case_all[rid] if len(rid) else rid,
                                   slot=row_slot_all[rid] if len(rid) else rid,
                                   delta=res.sp_delta[s0:s1], delta_clean=dcl, delta_corrupt=dco,
                                   set=[X.set_str(it[5]) if it[0] in keep_set else "" for it in its],
                                   **{k: np.asarray(ms[k])[s0:s1] for k in METRICS}))
            df.to_parquet(os.path.join(od, f"addback_rows_p{p:02d}.parquet"), index=False)
            d_cor = {int(x): float(res.delta[rpos[j][x]]) for x in rpos[j]}
            if first[j]:
                a, b = span[j]
                assert b - a == nC + len(t.rows) and list(gi) == list(range(a, b)), "first pass must hold every row in canonical order"
                a0 = df[(df.fam == "a0") & (df.order == "all_moe") & (df.dir == "d")]
                st.ceiling = {int(r_): float(v) for r_, v in zip(a0.row_id, a0.delta - a0.delta_corrupt)}
                dg = res.extra["diag"]
                dgs = {k: dg[k][:, a:b] for k in ("contrib_dla", "attn_out_final", "resid_final")}
                dgs["final_rms"] = dg["final_rms"][a:b]
                dla_df, n_flip = dla_table(t, eng, dgs, res.route_idx[:, a:b])
                dla_df.to_parquet(os.path.join(od, "addback_dla.parquet"), index=False)
                direct_split(t, eng, dgs).to_parquet(os.path.join(od, "addback_direct.parquet"), index=False)
                r.queue += AB.dla_queue(t, od)
                st.dla_added = True
                st.notes.append({"pass0_candidates_not_routed_in_pass0_clean": int(n_flip)})
                log(f"{r.key} first pass: ceilings for {len(st.ceiling)} rows (mean all-MoE rescue {np.mean(list(st.ceiling.values())):+.3f}); "
                    f"DLA rows {len(dla_df)} ({n_flip} candidate pairs not routed in this pass's clean run)")
            gd = df[df.fam == "greedy"]
            for r_, g in gd.groupby("row_id"):
                X.update_greedy(st.greedy[int(r_)], g, d_cor[int(r_)], st.ceiling.get(int(r_)))
            # noise_oracle groups once every noising single has run (and been saved)
            if not st.noise_oracle_added and not any(g and g[0][0] == "single_noise" for g in r.queue[st.qpos:]):
                gq, nrows = AB.noise_oracle_queue(t, od)
                if nrows:
                    r.queue += gq
                    st.noise_oracle_added = True
                    log(f"{r.key}: noise_oracle groups added for {nrows} rows")
            st.passes_done += 1
            g_left = sum(not g.done for g in st.greedy.values())
            st.notes.append({"pass": p, "group_pass": n_run, "spawn_rows": len(its), "adaptive": n_dyn[j], "static_groups": st.qpos - q0[j],
                             "prefill_rows": len(gi), "pass_s": res.extra["total_s"], "peak_GB": peak, "pass_spawn_rows_total": len(spawns),
                             "pass_prefill_rows_total": len(pre), "greedy_left": g_left, "queue": f"{st.qpos}/{len(r.queue)}",
                             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            X.save_state(r.sp, st)
            gm = [g.resc[-1] / st.ceiling[g.rid] for g in st.greedy.values() if g.resc and st.ceiling.get(g.rid, 0) > 0]
            log(f"{r.key} pass {p} done: greedy left {g_left} (median step {np.median([len(g.S) for g in st.greedy.values()]):.0f}, "
                f"median rescue/ceiling {np.median(gm) if gm else float('nan'):.2f}); queue {st.qpos}/{len(r.queue)}")
            if r.done:
                write_meta(od, {"complete": True, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "passes": st.passes_done,
                                "pass_notes": st.notes})
                log(f"{r.key}: ALL PASSES COMPLETE")
            else:
                write_meta(od, {"passes": st.passes_done, "pass_notes": st.notes})
        log(f"group pass {n_run} done in {res.extra['total_s']:.0f}s, peak {peak:.1f} GB")
        del res
        torch.cuda.empty_cache()
        n_run += 1


if __name__ == "__main__":
    main()
