"""ext14 C2 RelP pilot (p6-circuit): neuron attributions at every position and layer (RelP, plain AtP, DLA) on STR pairs,
validated by exact patching in the same functional forward (moetrace/ext14_circuit_relp.py), with mean-ablation
faithfulness / completeness of per-case neuron circuits (Arora et al. 2026) and STR denoising / noising of the same circuits.

Per case (clean prompt c, corrupted prompt o, answer t / contrast f, swap position s):
  attributions per node n = (layer, [expert,] neuron, position):
     relp_o = (q_c - q_o) . dDelta/dq at o (RelP rules, frozen routing)    relp_c = same gradient at c
     atp_o  = (q_c - q_o) . plain gradient at o (frozen routing weights)    mean_c = (q_c - qbar) . RelP gradient at c
     dla    = direct path only, final position: (q_c - q_o)_i (W_down[:, i] . u) / rms(h_final o)
  exact checks (corrupted input, everything else recomputed with live routing):
     V3 sublayer grid: MLP / MoE output at (l, t), t >= s, set to the clean value -> rescue vs the attribution sums;
        routing switch = some later layer's final-position top-k set differs from the unpatched corrupted run
     V1 focal layer (OLMoE: the task's top expert; Mistral: L18-L21 MLP), final position: top-k neurons by relp_o / relp_c /
        atp_o / dla / mag / rand set to clean, k grid -> curves; single-neuron patches of the top-8 by |relp_o|
     V5 multi-layer final-position sets: top-k nodes over ALL layers at the final position by relp_o (joint patch)
  circuits (per case; nodes at positions >= s for STR, >= 1 for mean ablation; top-N by |score|):
     STR denoise   corrupted input, circuit nodes := clean values                     -> R = (m - m_o) / (m_c - m_o)
     STR noise     clean input, circuit nodes := corrupted values                     -> remaining = (m - m_o) / (m_c - m_o)
     mean faithful clean input, every non-circuit node (positions >= 1) := qbar        -> F = (m(C) - m(0)) / (m(M) - m(0))
     mean complete clean input, circuit nodes := qbar                                  -> (m(M\\C) - m(0)) / (m(M) - m(0))
     node pools: all positions / final position only / final + swap position.

Usage: python scripts/ext14ci_relp_run.py <model: olmoe|mistral7b> <task> [--n-cases 48] [--smoke]
Outputs results/p6ci_relp_<model>_<task>/ (relp_cases.parquet, relp_grid.parquet, relp_focal.parquet, relp_multi.parquet,
relp_circuits.parquet, relp_nodes_top.parquet, run_meta.json).
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import RESULTS
from moetrace import ext14_circuit as C
from moetrace.ext14_circuit_relp import FModel, log

N_GRID = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
FOCAL = {("olmoe", "wino"): [(12, 11), (13, 40)], ("olmoe", "ioi"): [(12, 5)], ("olmoe", "cf"): [(13, 56)],
         ("mistral7b", "cf"): [(18, 0), (19, 0), (20, 0), (21, 0)], ("mistral7b", "wino"): [(18, 0), (19, 0), (20, 0), (21, 0)]}
BS = {"olmoe": 12, "mistral7b": 24}


def load_cases(model, task, n_cases):
    """list of dict(case_id, clean, corrupt, true, foil, swap, unit) from validation pairs."""
    out = []
    if model == "olmoe" and task == "cf":  # merged CounterFact STR pairs built by scripts/ext14ci_c3_cf_build.py olmoe --merge-as mix
        from moetrace import ext7_pairs as P
        from moetrace.ext14_circuit_c3 import DATA
        cs = json.load(open(os.path.join(DATA, "case_sets_cf_mix_olmoe.json")))
        pairs = P.load_pairs(os.path.join(DATA, "cf_mix_olmoe.parquet"), cs)
        val = [int(c) for c in cs["directed"]["validation"]]
        dc = P.directed_cases(pairs, val)
        for c in val:
            x = dc[c]
            out.append(dict(case_id=int(c), clean=list(x.clean_ids), corrupt=list(x.corrupt_ids), true=int(x.true_id),
                            foil=int(x.foil_id), swap=int(min(x.str_pos)), unit=int(x.pair_idx)))
    elif model == "olmoe":
        from moetrace import ext13atp as A
        pairs, cs, fam, ids, dc, split = A.load_task(task)
        val = [c for c in cs["directed"]["validation"]]
        for c in val:
            x = dc[c]
            diff = [i for i, (a, b) in enumerate(zip(x.clean_ids, x.corrupt_ids)) if a != b]
            out.append(dict(case_id=int(c), clean=list(x.clean_ids), corrupt=list(x.corrupt_ids), true=int(x.true_id),
                            foil=int(x.foil_id), swap=min(diff), unit=int(x.pair_idx)))
    else:
        ctask = f"mistral7b_{task}"
        rows = pd.read_parquet(os.path.join(C.task_dir(ctask), "c1_rows.parquet"))
        v = rows[rows.split == "validation"]
        ok = (v.delta_clean >= 1) & (v.delta_corrupt <= -1)
        v = v[ok]
        v = v.sort_values(["case_id", "slot"]).groupby("case_id").head(1)  # one donor per case (CounterFact)
        t = C.load_items(ctask)
        cases = t.cases.set_index("case_id")
        rr = t.rows.set_index("row_id")
        for r in v.itertuples():
            cl, co = list(cases.clean_ids.loc[r.case_id]), list(rr.corrupt_ids.loc[r.row_id])
            diff = [i for i, (a, b) in enumerate(zip(cl, co)) if a != b]
            out.append(dict(case_id=int(r.case_id), clean=cl, corrupt=co, true=int(r.true_id), foil=int(r.foil_id),
                            swap=min(diff), unit=int(r.unit)))
    # interleave so that both directions of a pair are not both taken when truncating: keep order, take first n
    seen, picked = set(), []
    for x in out:
        if x["unit"] in seen:
            continue
        seen.add(x["unit"])
        picked.append(x)
        if len(picked) >= n_cases:
            break
    return picked


def tens(ids, dev):
    return torch.tensor(ids, device=dev)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["olmoe", "mistral7b"])
    ap.add_argument("task")
    ap.add_argument("--n-cases", type=int, default=48)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(RESULTS, f"p6ci_relp_{a.model}_{a.task}" + ("_smoke" if a.smoke else ""))
    os.makedirs(out_dir, exist_ok=True)
    cases = load_cases(a.model, a.task, 3 if a.smoke else a.n_cases)
    log(f"{a.model} {a.task}: {len(cases)} cases")
    fm = FModel(a.model)
    dev = fm.dev
    L, I, E, moe = fm.L, fm.I, fm.E, fm.moe
    bs = BS[a.model]
    focal = FOCAL[(a.model, a.task)]
    t_all = time.time()
    # ---- pass 0: per-neuron means over positions >= 1 of all clean + corrupt prompts --------------------------------
    # dense: plain per-neuron mean over all tokens; MoE: per-neuron mean over the tokens where the expert is routed
    # (conditional mean; mean ablation then replaces only experts routed in the live run, non-routed experts stay 0)
    qsum, qcnt = [None] * L, [None] * L
    ntok = 0
    with torch.no_grad():
        for x in cases:
            for ids in (x["clean"], x["corrupt"]):
                r = fm.forward(tens(ids, dev), [x["true"]], [x["foil"]], record=True)
                for l in range(L):
                    q_ = r["q"][l][0, 1:].float()
                    s = q_.sum(0)
                    cnt = (q_.abs().sum(-1) > 0).float().sum(0) if moe else torch.tensor(float(q_.shape[0]), device=dev)
                    qsum[l] = s if qsum[l] is None else qsum[l] + s
                    qcnt[l] = cnt if qcnt[l] is None else qcnt[l] + cnt
                ntok += len(ids) - 1
    qbar = [(qsum[l] / (qcnt[l].clamp_min(1.0)[:, None] if moe else ntok)).to(fm.dtype) for l in range(L)]
    log(f"means over {ntok} tokens")
    rec_cases, rec_grid, rec_focal, rec_multi, rec_circ, rec_top = [], [], [], [], [], []
    for ci, x in enumerate(cases):
        T, sw = len(x["clean"]), x["swap"]
        tc, to = tens(x["clean"], dev), tens(x["corrupt"], dev)
        tr, fo = [x["true"]], [x["foil"]]
        with torch.no_grad():
            rc = fm.forward(tc, tr, fo, record=True)
            ro = fm.forward(to, tr, fo, record=True)
        m_c, m_o = float(rc["delta"][0]), float(ro["delta"][0])
        drop = m_c - m_o
        qc = [q[0] for q in rc["q"]]
        qo = [q[0] for q in ro["q"]]
        d_o, g_ro = fm.node_grads(to, tr, fo, relp=True, route=[(ti, tw) for ti, tw in ro["route"]] if moe else None)
        d_c, g_rc = fm.node_grads(tc, tr, fo, relp=True, route=[(ti, tw) for ti, tw in rc["route"]] if moe else None)
        _, g_ao = fm.node_grads(to, tr, fo, relp=False, route=[(ti, tw) for ti, tw in ro["route"]] if moe else None)
        dq = [qc[l].float() - qo[l].float() for l in range(L)]
        A = {"relp_o": [dq[l] * g_ro[l][0].float() for l in range(L)],
             "relp_c": [dq[l] * g_rc[l][0].float() for l in range(L)],
             "atp_o": [dq[l] * g_ao[l][0].float() for l in range(L)],
             "mean_c": [(qc[l].float() - qbar[l].float()) * g_rc[l][0].float()
                        * ((qc[l].abs().sum(-1, keepdim=True) > 0).float() if moe else 1.0) for l in range(L)]}
        # DLA at the final position (direct path, final norm frozen at o)
        u = fm.dla_dir(x["true"], x["foil"], ro["h_final"][0])
        dla_fin = []
        for l in range(L):
            if moe:
                wd = fm.layers[l]["down"]  # [E, H, I]
                gdir = torch.einsum("h,ehi->ei", u, wd.float())
            else:
                gdir = u @ fm.layers[l]["wd"].float()
            dla_fin.append(dq[l][T - 1] * gdir)
        # routing switch reference: corrupted run's final-position top-k sets
        ref_route = [set(ro["route"][l][0][T - 1].tolist()) for l in range(L)] if moe else None  # (topi [T, k], topw)
        sums = {k: torch.stack([v[l].reshape(T, -1).sum(-1) for l in range(L)]).cpu().numpy() for k, v in A.items()}  # [L, T]
        dla_sum = np.array([float(dla_fin[l].sum()) for l in range(L)])
        rec_cases.append(dict(case_id=x["case_id"], unit=x["unit"], T=T, swap=sw, m_clean=m_c, m_corrupt=m_o,
                              grad_delta_o=float(d_o[0]), grad_delta_c=float(d_c[0]),
                              relp_o_total=float(sums["relp_o"].sum()), relp_c_total=float(sums["relp_c"].sum()),
                              atp_o_total=float(sums["atp_o"].sum()), mean_c_total=float(sums["mean_c"].sum()),
                              relp_o_final=float(sums["relp_o"][:, T - 1].sum()),
                              relp_o_abs_final_share=float(np.abs(sums["relp_o"][:, T - 1]).sum() / max(np.abs(sums["relp_o"][:, sw:]).sum(), 1e-9)),
                              dla_final_total=float(dla_sum.sum())))
        # ---- V3: sublayer grid (MLP / MoE output at (l, t), t >= swap) -----------------------------------------------
        cells = [(l, t) for l in range(L) for t in range(sw, T)]
        for b0 in range(0, len(cells), bs):
            ch = cells[b0:b0 + bs]
            B = len(ch)
            op = {}
            for j, (l, t) in enumerate(ch):
                op.setdefault(l, torch.zeros(B, T, dtype=torch.bool, device=dev))[j, t] = True
            op = {l: (m_, rc["out"][l].expand(B, -1, -1)) for l, m_ in op.items()}
            with torch.no_grad():
                r = fm.forward(to.expand(B, -1), tr * B, fo * B, out_patch=op)
            dl = r["delta"].cpu().numpy() - m_o
            for j, (l, t) in enumerate(ch):
                swn = None
                if moe:
                    swn = int(sum(set(r["route"][l2][0][j * T + T - 1].tolist()) != ref_route[l2] for l2 in range(l + 1, L)))
                rec_grid.append(dict(case_id=x["case_id"], layer=l, pos=t, rel=t - (T - 1), is_final=t == T - 1, is_swap=t == sw,
                                     rescue=float(dl[j]), relp_o=float(sums["relp_o"][l, t]), relp_c=float(sums["relp_c"][l, t]),
                                     atp_o=float(sums["atp_o"][l, t]), dla=float(dla_sum[l]) if t == T - 1 else np.nan,
                                     n_switch=swn))
        # ---- V1: focal layer / expert, final position ---------------------------------------------------------------
        for (fl, fe) in focal:
            sl = (lambda v: v[fl][T - 1, fe]) if moe else (lambda v: v[fl][T - 1])
            sc = {"relp_o": sl(A["relp_o"]), "relp_c": sl(A["relp_c"]), "atp_o": sl(A["atp_o"]),
                  "dla": dla_fin[fl][fe] if moe else dla_fin[fl]}
            wdn = fm.W_down(fl, fe).float().norm(dim=0)
            dqf = dq[fl][T - 1, fe] if moe else dq[fl][T - 1]
            sc["mag"] = dqf.abs() * wdn
            gen = torch.Generator(device="cpu").manual_seed(1000 + x["case_id"])
            orders = {k: torch.argsort(v, descending=True).cpu() for k, v in sc.items() if k != "mag"}
            orders["mag"] = torch.argsort(sc["mag"], descending=True).cpu()
            orders["rand"] = torch.randperm(I, generator=gen)
            ks = C.k_grid(I)
            items = [("ref_all", I, torch.arange(I))]
            for rk, o in orders.items():
                for k in ks[:-1]:
                    items.append((rk, k, o[:k]))
            top_single = torch.argsort(sc["relp_o"].abs(), descending=True)[:8].cpu()
            for s_ in range(8):
                items.append((f"single{s_}", 1, top_single[s_:s_ + 1]))
            for b0 in range(0, len(items), bs):
                ch = items[b0:b0 + bs]
                B = len(ch)
                shape = (B, T, E, I) if moe else (B, T, I)
                mk = torch.zeros(shape, dtype=torch.bool, device=dev)
                for j, (_, _, nn) in enumerate(ch):
                    if moe:
                        mk[j, T - 1, fe, nn.to(dev)] = True
                    else:
                        mk[j, T - 1, nn.to(dev)] = True
                with torch.no_grad():
                    r = fm.forward(to.expand(B, -1), tr * B, fo * B, interv={fl: (mk, qc[fl][None])})
                dl = r["delta"].cpu().numpy() - m_o
                for j, (rk, k, nn) in enumerate(ch):
                    nd = nn.to(dev)
                    rec_focal.append(dict(case_id=x["case_id"], layer=fl, expert=fe, ranking=rk, k=k, rescue=float(dl[j]),
                                          pred_relp_o=float(sc["relp_o"][nd].sum()), pred_relp_c=float(sc["relp_c"][nd].sum()),
                                          pred_atp_o=float(sc["atp_o"][nd].sum()), pred_dla=float(sc["dla"][nd].sum()),
                                          neuron=int(nn[0]) if k == 1 else -1))
        # ---- V5: multi-layer final-position sets (top-k nodes over all layers at T-1 by relp_o / dla) ----------------
        flat = {k: torch.cat([v[l][T - 1].reshape(-1) for l in range(L)]) for k, v in (("relp_o", A["relp_o"]),)}
        flat["dla"] = torch.cat([dla_fin[l].reshape(-1) for l in range(L)])
        per_layer = E * I if moe else I
        items = []
        for rk, v in flat.items():
            o = torch.argsort(v, descending=True)
            for k in N_GRID:
                if k <= len(o):
                    items.append((rk, k, o[:k]))
        for b0 in range(0, len(items), bs):
            ch = items[b0:b0 + bs]
            B = len(ch)
            masks = {}
            for j, (_, _, nn) in enumerate(ch):
                lay = (nn // per_layer).cpu().numpy()
                rem = nn % per_layer
                for l in np.unique(lay):
                    if l not in masks:
                        masks[l] = torch.zeros((B, T, E * I) if moe else (B, T, I), dtype=torch.bool, device=dev)
                    sel = torch.as_tensor(lay == l, device=nn.device)
                    masks[l][j, T - 1, rem[sel].to(dev)] = True
            interv = {l: (m_.view((B, T, E, I) if moe else (B, T, I)), qc[l][None]) for l, m_ in masks.items()}
            with torch.no_grad():
                r = fm.forward(to.expand(B, -1), tr * B, fo * B, interv=interv)
            dl = r["delta"].cpu().numpy() - m_o
            for j, (rk, k, nn) in enumerate(ch):
                lay = (nn // per_layer).cpu().numpy()
                rec_multi.append(dict(case_id=x["case_id"], ranking=rk, k=k, rescue=float(dl[j]),
                                      pred=float(flat[rk][nn].sum()), n_layers=int(len(np.unique(lay))),
                                      mean_layer=float(lay.mean())))
        # ---- circuits ------------------------------------------------------------------------------------------------
        def node_view(scores, pos_lo):
            """flat |score| over nodes (l, t, [e,] i) with t >= pos_lo -> (flat abs scores, shape)."""
            st = torch.stack([s for s in scores])  # [L, T, (E,) I]
            m = torch.zeros(T, dtype=torch.bool, device=dev)
            m[pos_lo:] = True
            st = st * m.view(1, T, *([1] * (st.dim() - 2)))
            return st
        pools = {"all": None, "final": [T - 1], "final_swap": sorted({T - 1, sw})}
        for mode, key, pos_lo in (("str", "relp_o", sw), ("mean", "mean_c", 1)):
            st = node_view(A[key], pos_lo)  # [L, T, ...]
            for pool, plist in pools.items():
                if plist is not None:
                    pm = torch.zeros(T, dtype=torch.bool, device=dev)
                    pm[plist] = True
                    st_p = st * pm.view(1, T, *([1] * (st.dim() - 2)))
                else:
                    st_p = st
                flat_abs = st_p.abs().reshape(-1)
                n_nonzero = int((flat_abs > 0).sum())
                o = torch.argsort(flat_abs, descending=True)
                if pool == "all":
                    # where do the top nodes sit (position class, layer)
                    for N in (10, 100, 1000):
                        top = o[:N]
                        tt = (top // (flat_abs.numel() // (L * T))) % T
                        ll = top // (flat_abs.numel() // L)
                        rec_top.append(dict(case_id=x["case_id"], mode=mode, N=N, frac_final=float((tt == T - 1).float().mean()),
                                            frac_swap=float((tt == sw).float().mean()),
                                            frac_between=float(((tt > sw) & (tt < T - 1)).float().mean()),
                                            frac_before=float((tt < sw).float().mean()), mean_layer=float(ll.float().mean()),
                                            mass_share=float(flat_abs[top].sum() / flat_abs.sum())))
                Ns = [N for N in N_GRID if N <= max(n_nonzero, 1)]
                rows_spec = []
                for N in Ns:
                    for kind in (("denoise", "noise") if mode == "str" else ("faithful", "complete")):
                        rows_spec.append((kind, N))
                if mode == "mean":
                    rows_spec += [("empty", 0), ("full", 0)]
                for b0 in range(0, len(rows_spec), bs):
                    ch = rows_spec[b0:b0 + bs]
                    B = len(ch)
                    shp = tuple(st.shape)  # [L, T, ...]
                    cm = torch.zeros((B,) + shp, dtype=torch.bool, device=dev)  # circuit membership per row
                    for j, (kind, N) in enumerate(ch):
                        if N > 0:
                            cm[j].view(-1)[o[:N]] = True
                    base_in = []
                    interv = {}
                    for l in range(L):
                        cl = cm[:, l]  # [B, T, ...]
                        if mode == "str":
                            # denoise rows: corrupted input, circuit := clean; noise rows: clean input, circuit := corrupt
                            val = torch.stack([qc[l] if kind == "denoise" else qo[l] for kind, _ in ch])
                            interv[l] = (cl, val)
                        else:
                            pos_ok = torch.zeros(T, dtype=torch.bool, device=dev)
                            pos_ok[1:] = True
                            pos_ok = pos_ok.view(1, T, *([1] * (cl.dim() - 2)))
                            sel = []
                            for kind, _ in ch:
                                sel.append(kind)
                            msk = torch.stack([(~cl[j] if kind == "faithful" else cl[j]) if kind in ("faithful", "complete")
                                               else (torch.ones_like(cl[j]) if kind == "empty" else torch.zeros_like(cl[j]))
                                               for j, kind in enumerate(sel)]) & pos_ok
                            interv[l] = (msk, qbar[l].view(1, 1, *qbar[l].shape), "routed")
                    inp = torch.cat([to if (mode == "str" and kind == "denoise") else tc for kind, _ in ch], 0)
                    with torch.no_grad():
                        r = fm.forward(inp, tr * B, fo * B, interv=interv)
                    dl = r["delta"].cpu().numpy()
                    for j, (kind, N) in enumerate(ch):
                        rec_circ.append(dict(case_id=x["case_id"], mode=mode, pool=pool, kind=kind, N=N, m=float(dl[j]),
                                             n_nonzero=n_nonzero))
        if (ci + 1) % 4 == 0 or ci == len(cases) - 1:
            log(f"case {ci + 1}/{len(cases)} ({time.time() - t_all:.0f}s)")
            for nm, recs in (("relp_cases", rec_cases), ("relp_grid", rec_grid), ("relp_focal", rec_focal),
                             ("relp_multi", rec_multi), ("relp_circuits", rec_circ), ("relp_nodes_top", rec_top)):
                pd.DataFrame(recs).to_parquet(os.path.join(out_dir, f"{nm}.parquet"))
    meta = dict(model=a.model, task=a.task, n_cases=len(cases), focal=focal, n_grid=N_GRID, seconds=round(time.time() - t_all),
                rules="RMSNorm denominators (incl. OLMoE q/k norms) detached; softmax pattern detached; SiLU sigmoid detached; "
                      "half rule on SiLU(g)*u; MoE top-k set and weights frozen (routing weights detached in every backward)",
                command=" ".join(sys.argv), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=1)
    open(os.path.join(out_dir, ".done_relp"), "w").write(meta["utc"] + "\n")
    log("done")


if __name__ == "__main__":
    main()
