"""ext12 4a / 4c: multi-task add-back driver (ext8 machinery, several tasks of ONE model share every engine pass).

Each task (moetrace/ext12_complete.TASKS) keeps its own run dir, state and files in exactly the ext8 format
(addback_rows_pNN.parquet, addback_prefill_pNN.parquet, addback_state.pkl, addback_dla.parquet, addback_direct.parquet,
run_meta.json), so moetrace.ext8_addback.load_out and scripts/ext8_addback_analyze.analyse read them unchanged. A pass
carries the prefill rows of every task that still has work (indices offset per task), the adaptive greedy rows of every
task (always), and static groups taken round-robin from the tasks' queues up to --budget spawn rows. A task's first pass
also records the diagnostics (contrib_dla, attn_out_final, resid_final) for its DLA ordering and the direct-path split.

Work per task (4a brief): A0 ceilings (all rows, add-back and deletion direction), A1 static orderings oracle / pop /
layerwise / dla / vnorm / weight / rand0-4 on the evaluation rows (k grid of ext8), the all-clean-active endpoint, and
adaptive greedy (ext8 pools: Qwen3 the row's top-32 singles, Mixtral all 64; 15 steps, no ceiling stop). Not run: beam,
exact top-10, Shapley, noising singles, deletion curves.

Usage: python scripts/ext12_addback_run.py <qwen3|mixtral> [--tasks key,key] [--budget 45000] [--max-passes 99] [--dry-run]
"""
import argparse, dataclasses, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, pandas as pd
from moetrace.models import MODELS, RESULTS
from moetrace import ext8_addback as X
from moetrace import ext12_complete as C

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
    def __init__(self, key, args):
        self.key = key
        self.cfg = C.TASKS[key]
        self.od = os.path.join(RESULTS, self.cfg["out"])
        if not args.dry_run:
            os.makedirs(self.od, exist_ok=True)
        self.sp = os.path.join(self.od, "addback_state.pkl")
        self.t = C.load_task(key)
        t = self.t
        self.st = X.load_state(self.sp)
        if self.st is None:
            st = X.State(queue=[])
            pool = min(self.cfg["pool"], t.K)
            ev = C.eval_rows(t)
            for rid in ev:
                o = X.ordering(t, rid, "oracle")[:pool]
                st.greedy[rid] = X.Greedy(rid, o, S=[o[0]])
            q = C.static_queue(t)
            st.notes.append({"eval_rows": len(ev), "pool": pool, "queue_groups": len(q), "queue_items": int(sum(len(g) for g in q))})
            self.st = st
            meta = {"model": self.cfg["model"], "repo": MODELS[self.cfg["model"]]["repo"], "task": "cf" if self.cfg["kind"] == "cf_swap" else "wino",
                    "ext12_task": key, "kind": self.cfg["kind"], "src_run": self.cfg["src"], "family": self.cfg.get("family"),
                    "special_tokens": t.special_tokens, "pool": pool, "budget": args.budget, "n_cases": t.nC, "n_rows": len(t.rows),
                    "init": st.notes[0], "agent": "ext12-complete", "driver": "scripts/ext12_addback_run.py (multi-task; ext8 machinery)",
                    "fold_swap": self.cfg["kind"] == "cf_swap",
                    "note": ("CounterFact fold swap: split 'discovery' = paper validation (ranking source), 'validation' = paper discovery "
                             "(curves); orig_split in the task's case table") if self.cfg["kind"] == "cf_swap" else
                            f"WinoGrande family {self.cfg.get('family')}: ranking from its discovery split, curves on its validation split",
                    "work": "A0 (both directions, all rows), A1 static orderings + all_active on evaluation rows, greedy 15 steps; no beam / exact / Shapley / deletion",
                    "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
            if not args.dry_run:
                X.save_state(self.sp, st)
                write_meta(self.od, meta)
            log(f"{key}: initialised {st.notes[0]}")
        self.queue = C.static_queue(t)
        if self.st.dla_added:
            self.queue += C.dla_queue(t, self.od)
        self.pre = None

    @property
    def done(self):
        return self.st.qpos >= len(self.queue) and all(g.done for g in self.st.greedy.values())

    def prefill_specs(self, E, off):
        t = self.t
        pre = [E.PrefillSpec(list(c.clean_ids), int(c.true_id), int(c.foil_id), clean_ref=off + i) for i, c in enumerate(t.cases.itertuples())]
        for r in t.rows.itertuples():
            c = t.cases.iloc[t.case_index[int(r.case_id)]]
            assert len(r.corrupt_ids) == len(c.clean_ids)
            pre.append(E.PrefillSpec(list(r.corrupt_ids), int(c.true_id), int(c.foil_id), clean_ref=off + t.case_index[int(r.case_id)]))
        return pre


def slice_diag(dg, a, b):
    out = {}
    for k in ("contrib_dla", "attn_out_final", "resid_final"):
        out[k] = dg[k][:, a:b]
    out["final_rms"] = dg["final_rms"][a:b]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=sorted(C.GROUPS))  # groups of moetrace.ext12_complete
    ap.add_argument("--tasks", default=None, help="comma list of task keys (default: the group's tasks)")
    ap.add_argument("--budget", type=int, default=45000)
    ap.add_argument("--max-passes", type=int, default=99)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from ext8_addback_run import dla_table, direct_split
    keys = args.tasks.split(",") if args.tasks else list(C.GROUPS[args.group])
    assert all(C.TASKS[k]["model"] == args.group for k in keys)
    runs = [TaskRun(k, args) for k in keys]
    for r in runs:
        log(f"{r.key}: {r.t.nC} cases, {len(r.t.rows)} rows, K={r.t.K}; passes done {r.st.passes_done}, queue {r.st.qpos}/{len(r.queue)}, "
            f"greedy left {sum(not g.done for g in r.st.greedy.values())}")
    if args.dry_run:
        for r in runs:
            log(f"[dry-run] {r.key}: static items {sum(len(g) for g in r.queue)}, first dynamic step {sum(len(g.items()) for g in r.st.greedy.values())}")
        return
    import torch
    import moetrace.engine as E
    eng = E.Engine(MODELS[args.group]["repo"])
    n_run = 0
    while n_run < args.max_passes:
        act = [r for r in runs if not r.done]
        if not act:
            log("nothing left to run")
            break
        # ---- assemble the pass
        pre, offs = [], []
        for r in act:
            offs.append(len(pre))
            pre += r.prefill_specs(E, len(pre))
        items = []  # per task lists
        for r in act:
            its = []
            for g in r.st.greedy.values():
                its += g.items()
            items.append(its)
        n_dyn = [len(x) for x in items]
        total = sum(n_dyn)
        q0 = [r.st.qpos for r in act]
        first = [r.st.passes_done == 0 for r in act]
        for j, r in enumerate(act):  # A0 groups of a task's first pass are forced (needed for the ceilings)
            if first[j]:
                while r.st.qpos < len(r.queue) and r.queue[r.st.qpos][0][0] == "a0":
                    items[j] += r.queue[r.st.qpos]
                    r.st.qpos += 1
                    total += len(r.queue[r.st.qpos - 1])
        progress = True
        while progress:  # round-robin static groups up to the budget
            progress = False
            for j, r in enumerate(act):
                if r.st.qpos >= len(r.queue):
                    continue
                g = r.queue[r.st.qpos]
                if total + len(g) > args.budget and total > 0:
                    continue
                items[j] += g
                r.st.qpos += 1
                total += len(g)
                progress = True
        spawns, sp_off = [], []
        for j, r in enumerate(act):
            sp_off.append(len(spawns))
            o = offs[j]
            for it in items[j]:
                s = X.spawn_of(r.t, it, E.SpawnSpec)
                spawns.append(dataclasses.replace(s, parent=s.parent + o, clean=s.clean + o, kl_ref=s.kl_ref + o))
        sp_off.append(len(spawns))
        any_first = any(first)
        diag = E.DiagSpec(contrib_dla=True, attn_out_final=True, resid_final=True) if any_first else None
        log(f"pass {n_run}: tasks {[r.key for r in act]}, {len(pre)} prefill rows, {len(spawns)} spawn rows "
            f"(adaptive {n_dyn}, static groups {[r.st.qpos - q for r, q in zip(act, q0)]})")
        torch.cuda.reset_peak_memory_stats()
        res = eng.run(pre, spawns, record_routing=any_first, log=log if n_run == 0 else None, diag=diag, metrics=True)
        peak = torch.cuda.max_memory_allocated() / 2**30
        mp, ms = res.metrics_prefill, res.metrics_spawn
        for j, r in enumerate(act):
            t, st, od = r.t, r.st, r.od
            p = st.passes_done
            a, b = offs[j], offs[j] + t.nC + len(t.rows)
            nC, R = t.nC, len(t.rows)
            d = res.delta[a:b]
            row_case = t.rows.case_id.values
            row_slot = t.rows.slot.values
            pf = pd.DataFrame(dict(pass_=p, idx=np.arange(nC + R), kind=["clean"] * nC + ["corrupt"] * R,
                                   case_id=np.concatenate([t.cases.case_id.values, row_case]),
                                   row_id=np.concatenate([np.full(nC, -1), t.rows.row_id.values]),
                                   slot=np.concatenate([np.full(nC, -1), row_slot]), delta=d, top1=res.top1[a:b],
                                   **{k: mp[k][a:b] for k in METRICS}))
            pf.to_parquet(os.path.join(od, f"addback_prefill_p{p:02d}.parquet"), index=False)
            d_clean_row = d[[t.case_index[int(c)] for c in row_case]]
            d_cor_row = d[nC:]
            s0, s1 = sp_off[j], sp_off[j + 1]
            its = items[j]
            fam = [it[0] for it in its]
            rid = np.array([it[4] for it in its], dtype=np.int64)
            keep_set = {"greedy", "beam", "exact", "single_noise"}
            df = pd.DataFrame(dict(pass_=p, fam=fam, order=[it[1] for it in its], dir=[it[2] for it in its],
                                   k=np.array([it[3] for it in its], dtype=np.int32), row_id=rid, case_id=row_case[rid], slot=row_slot[rid],
                                   delta=res.sp_delta[s0:s1], delta_clean=d_clean_row[rid], delta_corrupt=d_cor_row[rid],
                                   set=[X.set_str(it[5]) if it[0] in keep_set else "" for it in its],
                                   **{k: ms[k][s0:s1] for k in METRICS}))
            df.to_parquet(os.path.join(od, f"addback_rows_p{p:02d}.parquet"), index=False)
            if first[j]:
                a0 = df[(df.fam == "a0") & (df.order == "all_moe") & (df.dir == "d")]
                st.ceiling = {int(r_): float(v) for r_, v in zip(a0.row_id, a0.delta - a0.delta_corrupt)}
                dg = slice_diag(res.extra["diag"], a, b)
                dla_df, n_flip = dla_table(t, eng, dg, res.route_idx[:, a:b])
                dla_df.to_parquet(os.path.join(od, "addback_dla.parquet"), index=False)
                direct_split(t, eng, dg).to_parquet(os.path.join(od, "addback_direct.parquet"), index=False)
                r.queue += C.dla_queue(t, od)
                st.dla_added = True
                st.notes.append({"pass0_candidates_not_routed_in_pass0_clean": int(n_flip)})
                log(f"{r.key} first pass: ceilings for {len(st.ceiling)} rows (mean all-MoE rescue {np.mean(list(st.ceiling.values())):+.3f}); "
                    f"DLA rows {len(dla_df)} ({n_flip} candidate pairs not routed in this pass's clean run)")
            gd = df[df.fam == "greedy"]
            for r_, g in gd.groupby("row_id"):
                X.update_greedy(st.greedy[int(r_)], g, float(d_cor_row[int(r_)]), st.ceiling.get(int(r_)))
            st.passes_done += 1
            g_left = sum(not g.done for g in st.greedy.values())
            st.notes.append({"pass": p, "group_pass": n_run, "spawn_rows": len(its), "adaptive": n_dyn[j], "static_groups": st.qpos - q0[j],
                             "pass_s": res.extra["total_s"], "peak_GB": peak, "pass_spawn_rows_total": len(spawns), "greedy_left": g_left,
                             "queue": f"{st.qpos}/{len(r.queue)}"})
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
