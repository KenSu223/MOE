"""ext8 A0-A3 driver: resumable sequence of engine passes for the add-back study (see moetrace/ext8_addback.py).

Every pass carries all prefill rows (clean rows of every case, corrupted rows of every patching row) plus
  * the adaptive rows of the current greedy / beam step (they need the previous pass's result), and
  * whole groups of the static work queue (A0 ceilings, A1 curves, A3 deletion curves, noising singles, exact top-10
    subsets, Shapley prefix sweeps) up to --budget spawn rows.
Pass 0 also records diagnostics (DiagSpec contrib_dla, attn_out_final, resid_final) for the DLA ordering and the
direct-path split of the A0 decomposition. Rows are saved after every pass (addback_rows_pNN.parquet,
addback_prefill_pNN.parquet) together with the pickled state (addback_state.pkl), so an interrupted run resumes at the
next pass without redoing GPU work.

Usage: python scripts/ext8_addback_run.py <model> --task cf --src-run qwen3_str --out qwen3_str_addback
       [--pool 32] [--beam-cases 0] [--shap-cases 0] [--budget 40000] [--max-passes 99] [--engine main|dev] [--dry-run]
"""
import argparse, importlib, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS, RESULTS
from moetrace import ext8_addback as X

METRICS = ("logp_true", "logp_foil", "p_true", "p_foil", "rank_true", "kl_to_clean")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_meta(od, upd):
    p = os.path.join(od, "run_meta.json")
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.update(upd)
    with open(p, "w") as f:
        json.dump(cur, f, indent=1, default=str)


def load_task(args):
    if args.task == "cf":
        return X.load_cf(args.model, args.src_run)
    if args.task == "wino":
        return X.load_wino(args.model, args.src_run, args.pairs, args.case_sets)
    raise ValueError(args.task)


def row_sets(t, args):
    rows = t.rows
    split = rows.case_id.map(dict(zip(t.cases.case_id, t.cases.split)))
    val = rows.row_id[split == "validation"].tolist()
    first_val = rows.row_id[(split == "validation") & rows["first"]].tolist()
    beam_rows = first_val[: args.beam_cases] if args.beam_cases else first_val
    shap_rows = first_val[: args.shap_cases] if args.shap_cases else first_val
    exact_rows = first_val[: args.exact_cases] if args.exact_cases else first_val
    noise_rows = first_val[: args.noise_cases] if getattr(args, "noise_cases", 0) else first_val
    return val, first_val, beam_rows, shap_rows, exact_rows, noise_rows


def base_queue(t, args):
    """The deterministic static work queue (groups in priority order)."""
    val, first_val, beam_rows, shap_rows, exact_rows, noise_rows = row_sets(t, args)
    allr = t.rows.row_id.tolist()
    q = []
    q += X.a0_groups(t, allr)  # pass 0 (needed for the greedy stop rule)
    q += X.curve_groups(t, val, [o for o in X.A1_ORDERS if o != "dla"], "d", "a1")
    q += X.all_active_group(t, val, "d", "a1") + X.all_active_group(t, val, "n", "a3")
    q += X.single_noise_groups(t, noise_rows)
    q += X.curve_groups(t, val, [o for o in X.A3_ORDERS if o not in ("dla", "noise_oracle")], "n", "a3")
    q += X.exact_groups(t, exact_rows)
    q += X.prefix_groups(t, shap_rows, [f"rand{i}" for i in range(X.N_RAND)])
    return q


def dla_groups(t, args, od):
    val = row_sets(t, args)[0]
    dla_df = pd.read_parquet(os.path.join(od, "addback_dla.parquet"))
    dla = {int(r_): dict(zip(zip(g.layer.astype(int), g.expert.astype(int)), g.dla)) for r_, g in dla_df.groupby("row_id")}
    return X.curve_groups(t, val, ["dla"], "d", "a1", dla=dla) + X.curve_groups(t, val, ["dla"], "n", "a3", dla=dla)


def noise_oracle_groups(t, od):
    sn = load_rows(od, fam="single_noise")
    ns = {}
    for r_, g in sn.groupby("row_id"):
        ns[int(r_)] = {X.parse_set(s)[0]: float(v) for s, v in zip(g.set, g.delta - g.delta_clean)}
    return X.curve_groups(t, sorted(ns), ["noise_oracle"], "n", "a3", noise_single=ns), len(ns)


def full_queue(t, args, st, od):
    q = base_queue(t, args)
    if st.dla_added:
        q += dla_groups(t, args, od)
    if st.noise_oracle_added:
        q += noise_oracle_groups(t, od)[0]
    return q


def init_state(t, args):
    val, first_val, beam_rows, shap_rows, exact_rows, noise_rows = row_sets(t, args)
    q = base_queue(t, args)
    st = X.State(queue=[])
    pool = min(args.pool, t.K)
    for rid in val:
        o = X.ordering(t, rid, "oracle")[:pool]
        st.greedy[rid] = X.Greedy(rid, o, S=[o[0]])
    for rid in beam_rows:
        o = X.ordering(t, rid, "oracle")[:pool]
        st.beam[rid] = X.Beam(rid, o, beams=[((p,), float("nan")) for p in o[: X.BEAM_WIDTH]])
    st.notes.append({"val_rows": len(val), "first_val_rows": len(first_val), "beam_rows": len(beam_rows), "shap_rows": len(shap_rows),
                     "exact_rows": len(exact_rows), "noise_rows": len(noise_rows),
                     "pool": pool, "queue_groups": len(q), "queue_items": int(sum(len(g) for g in q))})
    return st


def build_pass(st, queue, budget, first_pass):
    """Adaptive rows first, then whole static groups up to the budget (all A0 groups go into pass 0)."""
    items = []
    for rid, g in st.greedy.items():
        items += g.items()
    for rid, b in st.beam.items():
        items += b.items()
    n_dyn = len(items)
    q0 = st.qpos
    while st.qpos < len(queue):
        g = queue[st.qpos]
        if not (first_pass and g[0][0] == "a0") and items and len(items) + len(g) > budget:
            break
        items += g
        st.qpos += 1
    return items, n_dyn, st.qpos - q0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--task", default="cf", choices=["cf", "wino"])
    ap.add_argument("--src-run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--case-sets", default=None)
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--beam-cases", type=int, default=0)
    ap.add_argument("--shap-cases", type=int, default=0)
    ap.add_argument("--exact-cases", type=int, default=0)
    ap.add_argument("--noise-cases", type=int, default=0)
    ap.add_argument("--budget", type=int, default=40000)
    ap.add_argument("--max-passes", type=int, default=99)
    ap.add_argument("--engine", default="main", choices=["main", "dev"])
    ap.add_argument("--greedy-stop", type=float, default=0.0, help="stop greedy at this fraction of the row's all-MoE ceiling (0 = off)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    X.GREEDY_STOP = args.greedy_stop if args.greedy_stop > 0 else None
    od = os.path.join(RESULTS, args.out)
    os.makedirs(od, exist_ok=True)
    sp = os.path.join(od, "addback_state.pkl")
    t = load_task(args)
    log(f"task {t.name} {t.model}: {t.nC} cases, {len(t.rows)} rows, L={t.L} k={t.topk} K={t.K}")
    st = X.load_state(sp)
    if st is None:
        st = init_state(t, args)
        queue = base_queue(t, args)
        X.save_state(sp, st)
        write_meta(od, {"model": args.model, "repo": MODELS[args.model]["repo"], "task": args.task, "src_run": args.src_run,
                        "special_tokens": t.special_tokens, "pool": args.pool, "beam_cases": args.beam_cases,
                        "shap_cases": args.shap_cases, "exact_cases": args.exact_cases, "noise_cases": args.noise_cases, "budget": args.budget, "greedy_stop": args.greedy_stop, "n_cases": t.nC, "n_rows": len(t.rows),
                        "init": st.notes[0], "command": "python " + " ".join(sys.argv),
                        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False})
        log(f"initialised: {st.notes[0]}")
    else:
        queue = full_queue(t, args, st, od)
        log(f"resuming after {st.passes_done} passes (queue {st.qpos}/{len(queue)} groups)")
    if args.dry_run:
        tot = sum(len(g) for g in queue)
        dyn = sum(len(g.items()) for g in st.greedy.values()) + sum(len(b.items()) for b in st.beam.values())
        log(f"[dry-run] static items {tot}, first dynamic step {dyn}")
        return
    E = importlib.import_module("moetrace.engine_ext8_dev" if args.engine == "dev" else "moetrace.engine")
    eng = E.Engine(MODELS[args.model]["repo"])
    nC, R = t.nC, len(t.rows)
    pre = [E.PrefillSpec(list(c.clean_ids), int(c.true_id), int(c.foil_id), clean_ref=i) for i, c in enumerate(t.cases.itertuples())]
    for r in t.rows.itertuples():
        c = t.cases.iloc[t.case_index[int(r.case_id)]]
        assert len(r.corrupt_ids) == len(c.clean_ids)
        pre.append(E.PrefillSpec(list(r.corrupt_ids), int(c.true_id), int(c.foil_id), clean_ref=t.case_index[int(r.case_id)]))
    row_case = t.rows.case_id.values
    row_slot = t.rows.slot.values
    n_run = 0
    while n_run < args.max_passes:
        p = st.passes_done
        first = p == 0
        items, n_dyn, n_groups = build_pass(st, queue, args.budget, first)
        if not items:
            log("nothing left to run")
            break
        spawns = [X.spawn_of(t, it, E.SpawnSpec) for it in items]
        diag = E.DiagSpec(contrib_dla=True, attn_out_final=True, resid_final=True) if first else None
        log(f"pass {p}: {len(pre)} prefill rows, {len(spawns)} spawn rows ({n_dyn} adaptive, {n_groups} static groups; queue {st.qpos}/{len(queue)})")
        torch.cuda.reset_peak_memory_stats()
        res = eng.run(pre, spawns, record_routing=first, log=log, diag=diag, metrics=True)
        peak = torch.cuda.max_memory_allocated() / 2**30
        d = res.delta
        mp, ms = res.metrics_prefill, res.metrics_spawn
        # ---- prefill rows
        pf = pd.DataFrame(dict(pass_=p, idx=np.arange(nC + R), kind=["clean"] * nC + ["corrupt"] * R,
                               case_id=np.concatenate([t.cases.case_id.values, row_case]),
                               row_id=np.concatenate([np.full(nC, -1), t.rows.row_id.values]),
                               slot=np.concatenate([np.full(nC, -1), row_slot]), delta=d, top1=res.top1,
                               **{k: mp[k] for k in METRICS}))
        pf.to_parquet(os.path.join(od, f"addback_prefill_p{p:02d}.parquet"), index=False)
        d_clean_row = d[[t.case_index[int(c)] for c in row_case]]
        d_cor_row = d[nC:]
        # ---- spawn rows
        fam = [it[0] for it in items]
        rid = np.array([it[4] for it in items])
        keep_set = {"greedy", "beam", "exact", "single_noise"}
        df = pd.DataFrame(dict(pass_=p, fam=fam, order=[it[1] for it in items], dir=[it[2] for it in items],
                               k=np.array([it[3] for it in items], dtype=np.int32), row_id=rid, case_id=row_case[rid], slot=row_slot[rid],
                               delta=res.sp_delta, delta_clean=d_clean_row[rid], delta_corrupt=d_cor_row[rid],
                               set=[X.set_str(it[5]) if it[0] in keep_set else "" for it in items],
                               **{k: ms[k] for k in METRICS}))
        df.to_parquet(os.path.join(od, f"addback_rows_p{p:02d}.parquet"), index=False)
        # ---- pass 0: ceilings, DLA, direct-path split
        if first:
            a0 = df[(df.fam == "a0") & (df.order == "all_moe") & (df.dir == "d")]
            st.ceiling = {int(r_): float(v) for r_, v in zip(a0.row_id, a0.delta - a0.delta_corrupt)}
            dg = res.extra["diag"]
            dla_df, n_flip = dla_table(t, eng, dg, res.route_idx)
            dla_df.to_parquet(os.path.join(od, "addback_dla.parquet"), index=False)
            direct_split(t, eng, dg).to_parquet(os.path.join(od, "addback_direct.parquet"), index=False)
            queue += dla_groups(t, args, od)
            st.dla_added = True
            st.notes.append({"pass0_candidates_not_routed_in_pass0_clean": int(n_flip)})
            log(f"pass 0: ceilings for {len(st.ceiling)} rows (mean all-MoE rescue {np.mean(list(st.ceiling.values())):+.3f}); "
                f"DLA rows {len(dla_df)} ({n_flip} candidate pairs not routed in this pass's clean run)")
            del dg
        # ---- adaptive updates
        gd = df[df.fam == "greedy"]
        for r_, g in gd.groupby("row_id"):
            X.update_greedy(st.greedy[int(r_)], g, float(d_cor_row[int(r_)]), st.ceiling.get(int(r_)))
        bd = df[df.fam == "beam"]
        for r_, g in bd.groupby("row_id"):
            X.update_beam(st.beam[int(r_)], g)
        # ---- noise_oracle groups once every noising single is done
        if not st.noise_oracle_added and not any(g and g[0][0] == "single_noise" for g in queue[st.qpos:]):
            gq, nrows = noise_oracle_groups(t, od)
            if nrows:
                queue += gq
                st.noise_oracle_added = True
                log(f"noise_oracle groups added for {nrows} rows")
        st.passes_done += 1
        n_run += 1
        g_left = sum(not g.done for g in st.greedy.values())
        b_left = sum(not b.done for b in st.beam.values())
        st.notes.append({"pass": p, "spawn_rows": len(spawns), "adaptive": n_dyn, "static_groups": n_groups, "pass_s": res.extra["total_s"],
                         "peak_GB": peak, "greedy_left": g_left, "beam_left": b_left, "queue": f"{st.qpos}/{len(queue)}"})
        X.save_state(sp, st)
        gm = [g.resc[-1] / st.ceiling[g.rid] for g in st.greedy.values() if g.resc and st.ceiling.get(g.rid, 0) > 0]
        log(f"pass {p} done in {res.extra['total_s']:.0f}s, peak {peak:.1f} GB; greedy left {g_left} (median step {np.median([len(g.S) for g in st.greedy.values()]):.0f}, "
            f"median rescue/ceiling {np.median(gm) if gm else float('nan'):.2f}), beam left {b_left}; queue {st.qpos}/{len(queue)}")
        del res
        torch.cuda.empty_cache()
        done = st.qpos >= len(queue) and g_left == 0 and b_left == 0
        if done:
            write_meta(od, {"complete": True, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "passes": st.passes_done,
                            "pass_notes": st.notes})
            log("ALL PASSES COMPLETE")
            break
    else:
        write_meta(od, {"passes": st.passes_done, "pass_notes": st.notes})


def load_rows(od, fam=None):
    parts = sorted(f for f in os.listdir(od) if f.startswith("addback_rows_p"))
    out = []
    for f in parts:
        x = pd.read_parquet(os.path.join(od, f))
        if fam is not None:
            x = x[x.fam == fam]
        out.append(x)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def dla_table(t, eng, dg, route_idx):
    """DLA of delta_e = c_e(clean) - c_e(corrupt) per (row, candidate) with the final norm frozen at the corrupted run."""
    cd, rms = dg["contrib_dla"], dg["final_rms"]
    nC = t.nC
    out = []
    n_flip = 0
    for r in t.rows.itertuples():
        ci = t.case_index[int(r.case_id)]
        cr = nC + int(r.row_id)
        for (l, e) in t.cand[int(r.case_id)]:
            rc = route_idx[l, ci]
            rj = route_idx[l, cr]
            pc = cd[l, ci, int(np.nonzero(rc == e)[0][0])] if (rc == e).any() else 0.0
            if not (rc == e).any():
                n_flip += 1
            pj = cd[l, cr, int(np.nonzero(rj == e)[0][0])] if (rj == e).any() else 0.0
            out.append((int(r.row_id), l, e, float(pc - pj) / float(rms[cr])))
    return pd.DataFrame(out, columns=["row_id", "layer", "expert", "dla"]), n_flip


def direct_split(t, eng, dg):
    """Direct-path split of the final-position residual difference (exact final norm, fp32, offline):
    A_direct: h_corrupt + sum_l (Attn_clean - Attn_corrupt); M_direct: h_corrupt + sum_l (MoE_clean - MoE_corrupt)
    = h_clean - sum_l dAttn (the final token is shared, so the embeddings cancel). Also the linear DLA split with the
    norm frozen at the corrupted run's scale. Delta values from the same fp32 path for clean / corrupt."""
    att, res_f = dg["attn_out_final"], dg["resid_final"]  # bf16 [L, B, H] CPU
    L = att.shape[0]
    gamma = eng.g["norm"].float().cpu()
    head = eng.g["head"]
    eps = eng.spec.rms_eps
    out = []
    for r in t.rows.itertuples():
        c = t.cases.iloc[t.case_index[int(r.case_id)]]
        ci, cr = t.case_index[int(r.case_id)], t.nC + int(r.row_id)
        u = (head[int(c.true_id)].float() - head[int(c.foil_id)].float()).cpu() * gamma
        hc, hj = res_f[L - 1, ci].float(), res_f[L - 1, cr].float()
        dA = (att[:, ci].float() - att[:, cr].float()).sum(0)

        def D(h):
            return float((h * torch.rsqrt(h.pow(2).mean() + eps)) @ u)
        dc, dj = D(hc), D(hj)
        rms_j = float(torch.sqrt(hj.pow(2).mean() + eps))
        out.append(dict(row_id=int(r.row_id), case_id=int(r.case_id), slot=int(r.slot), d_clean_fp32=dc, d_corrupt_fp32=dj,
                        d_attn_direct=D(hj + dA), d_moe_direct=D(hc - dA),
                        dla_attn=float(dA @ u) / rms_j, dla_moe=float((hc - hj - dA) @ u) / rms_j, dla_total=float((hc - hj) @ u) / rms_j))
    return pd.DataFrame(out)


if __name__ == "__main__":
    main()
