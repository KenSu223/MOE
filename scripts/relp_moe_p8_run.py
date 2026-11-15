#!/usr/bin/env python3
"""Phase 8 stage-2 runner (docs/relp_moe_phase8_plan.md, section 7) on the Phase-7 SVA manifests (paper profile), with
the verified streamed engine (unmodified) and moetrace/relp_moe_p8.py. Same verification gate as the Phase-7 runners.

  probe    S2.2  every pair (discovery + validation + test): A-vs-B differences of routing, SiLU gates, norm denominators,
                 attention patterns, residual; router logits.
  rules    S2.4 / S2.6  discovery: per-pair expert-position scores at endpoints B and A for every backward-rule variant
                 (RULES), one reverse stream per endpoint; D1 check against Phase 7's per-row scores.
  path     S2.6  discovery: multi-step scores on the straight line between the B and A embeddings.
  freeze   S2.3  first --freeze-rows discovery pairs, denoise: position units + top exact / top RelP / random expert units,
                 each in the 8 FREEZE variants.
  neuron   S2.7  discovery, denoise: single-neuron patches (RelP's top 1,000, 1,000 matched random, the top
                 expert-position's neurons) + per-pair RelP scores of the same neurons.
  addback  S2.1  validation + test: add-back / removal curves (denoise, noise, keep, remove with mean_q_routed) for the
                 rankings relp, relp_B, exact, exact_noL0-2, lu, grad, the selected router-aware variant, random x 3.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

EP_BUDGETS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
WE_BUDGETS = [1, 2, 5, 10, 20, 50, 100]


def load_rows(S, manifest_dir, task, splits, max_rows=0):
    rows, split_of = [], []
    for sp in splits:
        _, rs = S.read_split(manifest_dir, task, sp)
        if max_rows:
            rs = rs[:max_rows]
        rows += rs
        split_of += [sp] * len(rs)
    return rows, split_of


def run_units(m, A, Bb, tg, fo, units, variants, buf_gb, sub_rows, out, tag):
    """patch_p8 over row-chunked passes (resumable part files <tag>_part_XXX.npz). Returns dict of concatenated arrays
    in the order of `units`."""
    import numpy as np
    import torch
    from moetrace.relp_moe_p8 import patch_p8
    n, T = A.shape
    cap = int(buf_gb * 1e9 / (T * m.H * 2 * len(variants)))
    per_row = np.bincount(units[:, 0], minlength=n)
    passes, cur, cnt = [], [], 0
    for i in range(n):
        if cur and cnt + per_row[i] > cap:
            passes.append(cur)
            cur, cnt = [], 0
        cur.append(i)
        cnt += per_row[i]
    if cur:
        passes.append(cur)
    print(f"{tag}: {n} rows, {len(units)} units x {len(variants)} variants, {len(passes)} passes", flush=True)
    order = np.argsort(units[:, 0], kind="stable")
    parts = []
    for pi, rows_i in enumerate(passes):
        f = out / f"{tag}_part_{pi:03d}.npz"
        idx = np.asarray(rows_i)
        sel = order[np.isin(units[order, 0], idx)]
        if not f.exists():
            tp = time.time()
            uu = units[sel].copy()
            remap = np.full(n, -1)
            remap[idx] = np.arange(len(idx))
            uu[:, 0] = remap[uu[:, 0]]
            r = patch_p8(m, A[idx], Bb[idx], [tg[i] for i in idx], [fo[i] for i in idx], uu, variants=variants,
                         sub_rows=sub_rows)
            np.savez(f, sel=sel, margin=r["margin"], route_changes=r["route_changes"], direct=r["direct"],
                     margin_A=r["margin_A"][uu[:, 0]], margin_B=r["margin_B"][uu[:, 0]])
            print(f"{tag} pass {pi + 1}/{len(passes)}: {len(uu)} units in {time.time() - tp:.0f}s", flush=True)
            del r
            torch.cuda.empty_cache()
        parts.append(np.load(f))
    U = len(units)
    res = {"margin": np.zeros((U, len(variants)), np.float32), "route_changes": np.zeros(U, np.int32),
           "direct": np.zeros(U, np.float32), "margin_A": np.zeros(U, np.float32), "margin_B": np.zeros(U, np.float32)}
    for z in parts:
        for k_ in res:
            res[k_][z["sel"]] = z[k_]
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("probe", "rules", "path", "freeze", "neuron", "addback"))
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--verification", type=Path, required=True)
    p.add_argument("--phase7-dir", type=Path, required=True, help="results/relp_sva_moe/<model>/paper/<task>")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--variants-B", nargs="+", default=None, help="rules: variants at endpoint B (default: all)")
    p.add_argument("--variants-A", nargs="+", default=None, help="rules: variants at endpoint A (default: all)")
    p.add_argument("--path-variants", nargs="+", default=["relp", "relp_rw", "grad_rw"])
    p.add_argument("--path-steps", type=int, default=8)
    p.add_argument("--chunk", type=int, default=100, help="path: pairs per chunk")
    p.add_argument("--freeze-rows", type=int, default=100)
    p.add_argument("--neuron-top", type=int, default=1000)
    p.add_argument("--buf-gb", type=float, default=6.0)
    p.add_argument("--sub-rows", type=int, default=3000)
    p.add_argument("--rw-variant", default=None, help="addback: the selected router-aware variant")
    p.add_argument("--eval-chunk", type=int, default=1040)
    p.add_argument("--max-rows", type=int, default=0, help="first N rows per split (smoke only)")
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from moetrace import relp_sva as S
    from moetrace.relp_moe_p8 import FREEZE, RULES, RuleRelP, endpoint_triples, gate_probe, multi_backward, path_forward, path_triples
    from moetrace.relp_moe_stream import Group, forward_pass
    import relp_sva_moe_run as R
    import relp_moe_p8_data as D
    meta, _ = S.read_split(a.manifest_dir, a.task, "calibration")
    S.check_tokenizer_files(a.model_dir, meta)
    m = RuleRelP(str(a.model_dir.resolve()), device=a.device, dtype=torch.bfloat16)
    ident = R.identity(m, a.model_dir, S)
    v = json.loads(a.verification.read_text())
    if not v.get("passed") or v["identity"] != ident or v["task"] != a.task or v["manifest_sha256"] != S.sha(a.manifest_dir / "manifest.json"):
        p.error("Verification is failed, stale, or for another model/task/manifest/runtime")
    out = a.output_dir
    if (out / "run_meta.json").exists() and json.loads((out / "run_meta.json").read_text()).get("status") == "complete":
        print("complete; nothing to do")
        return
    out.mkdir(parents=True, exist_ok=True)
    common = {"utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "task": a.task,
              "manifest_sha256": S.sha(a.manifest_dir / "manifest.json"), "identity": ident,
              "verification_sha256": S.sha(a.verification), "script_sha256": S.sha(__file__),
              "p8_module_sha256": S.sha(ROOT / "moetrace/relp_moe_p8.py"),
              "patch_module_sha256": S.sha(ROOT / "moetrace/relp_moe_patch.py"), "status": "running"}
    S.write_json(out / "run_meta.json", common)
    t0 = time.time()
    p7 = a.phase7_dir
    done = lambda **kw: S.write_json(out / "run_meta.json", {**common, "status": "complete", "seconds": time.time() - t0, **kw})

    if a.command == "probe":
        rows, split_of = load_rows(S, a.manifest_dir, a.task, ["discovery", "validation", "test"], a.max_rows)
        A, Bb = R.ids_tensor(rows), R.ids_tensor(rows, True)
        first = min(min(r["changed_positions"]) for r in rows)
        res = gate_probe(m, A, Bb, first)
        np.save(out / "router_logits.npy", res.pop("logits"))
        np.savez_compressed(out / "probe.npz", **res, split=np.array(split_of), row_id=np.array([r["row_id"] for r in rows]),
                            first_changed=first)
        done(rows=len(rows), first_changed=first, k=m.k, soft_r=m.soft_r)
        print(f"probe done in {time.time() - t0:.0f}s", flush=True)
        return

    if a.command in ("rules", "path"):
        rows, _ = load_rows(S, a.manifest_dir, a.task, ["discovery"], a.max_rows)
        n = len(rows)
        A, Bb = R.ids_tensor(rows), R.ids_tensor(rows, True)
        tg, fo = [r["target_id"] for r in rows], [r["foil_id"] for r in rows]
        pm = R.pool_mask(rows, "non_special")
        first = min(min(r["changed_positions"]) for r in rows)
        probe_dir = out.parent / "p8_probe"
        logits = np.load(probe_dir / "router_logits.npy", mmap_mode="r")
        pz = np.load(probe_dir / "probe.npz")
        disc = np.where(pz["split"] == "discovery")[0]
        assert list(pz["row_id"][disc][:n]) == [r["row_id"] for r in rows]
        tau = D.tau_from_logits(np.asarray(logits[:, disc]), np.arange(len(disc)), first, m.k, m.soft_r)
        tau_t = torch.tensor(tau, dtype=torch.float32)
        if a.command == "rules":
            g = Group("score", torch.cat([A, Bb]), tg + tg, fo + fo, record_h_in=True)
            forward_pass(m, [g])
            info = {"tau_per_layer": tau.tolist()}
            for ep, vs in (("B", a.variants_B or list(RULES)), ("A", a.variants_A or list(RULES))):
                tb = time.time()
                res = multi_backward(m, g, endpoint_triples(n, ep), pm, vs, tau=tau_t)
                pe = {k_: res[k_]["pair_expert"] for k_ in vs}
                D.save_sparse(out / f"scores_{ep}.npz", pe)
                info[f"seconds_{ep}"] = time.time() - tb
                if not a.max_rows:   # D1: the 'relp' variant must reproduce Phase 7's per-row expert-position scores
                    ref = torch.load(p7 / "score_endpointB/per_row_expert_scores.pt", map_location="cpu", weights_only=True)
                    dif = (pe["relp"] - ref[ep]).abs()
                    info[f"D1_{ep}"] = {"bit_identical": bool(torch.equal(pe["relp"], ref[ep])), "max_abs": float(dif.max()),
                                        "ref_max_abs": float(ref[ep].abs().max())}
                    print(f"D1 endpoint {ep}:", info[f"D1_{ep}"], flush=True)
                print(f"endpoint {ep}: {len(vs)} variants in {info[f'seconds_{ep}']:.0f}s", flush=True)
                del res, pe
            d = g.delta.cpu()
            pd.DataFrame({"row_id": [r["row_id"] for r in rows], "margin_A": d[:n].numpy(), "margin_B": d[n:].numpy()}
                         ).to_parquet(out / "cases.parquet", index=False)
            done(rows=n, **info)
        else:
            vs = a.path_variants
            K = a.path_steps
            pe = {k_: [] for k_ in vs}
            deltas = []
            for c0 in range(0, n, a.chunk):
                c1 = min(n, c0 + a.chunk)
                gp = path_forward(m, A[c0:c1], Bb[c0:c1], tg[c0:c1], fo[c0:c1], K)
                res = multi_backward(m, gp, path_triples(c1 - c0, K), pm[c0:c1], vs, tau=tau_t)
                for k_ in vs:
                    pe[k_].append(res[k_]["pair_expert"])
                deltas.append(gp.delta.view(K + 1, c1 - c0).cpu())
                print(f"path pairs {c1}/{n} ({time.time() - t0:.0f}s)", flush=True)
                del gp, res
                torch.cuda.empty_cache()
            pe = {k_: torch.cat(x_) for k_, x_ in pe.items()}
            D.save_sparse(out / "scores_path.npz", pe, extra={"deltas": torch.cat(deltas, 1).numpy()})
            done(rows=n, steps=K, variants=vs, tau_per_layer=tau.tolist(),
                 score="sum_j (q_{j+1} - q_j) . (beta_j + beta_{j+1}) / 2 on the straight line from the B to the A embeddings")
        print(f"{a.command} done in {time.time() - t0:.0f}s", flush=True)
        return

    if a.command in ("freeze", "neuron"):
        rows, _ = load_rows(S, a.manifest_dir, a.task, ["discovery"])
        if a.command == "freeze":
            rows = rows[:a.freeze_rows]
        if a.max_rows:
            rows = rows[:a.max_rows]
        n = len(rows)
        A, Bb = R.ids_tensor(rows), R.ids_tensor(rows, True)
        T = A.shape[1]
        tg, fo = [r["target_id"] for r in rows], [r["foil_id"] for r in rows]
        ta, tb = torch.load(p7 / "patch_full/routes.pt", weights_only=True)      # [L, 380, T, k], discovery rows first
        ta, tb = ta[:, :n].numpy(), tb[:, :n].numpy()
        pool = R.pool_mask(rows, "non_special")[0]
        changed = min(min(r["changed_positions"]) for r in rows)
        positions = [t for t in range(changed, T) if bool(pool[t])]
        L = m.L
        if a.command == "freeze":
            ex = D.exact_means(p7)
            ex = ex[ex.pos.isin(positions)]
            top_exact = ex.reindex(ex["mean"].abs().sort_values(ascending=False).index).head(200)
            rk = D.load_json(p7 / "score_endpointB/expert_position_ranking.json")
            top_relp = pd.DataFrame([c["coordinate"] for c in rk[:200]], columns=["layer", "pos", "expert"])
            rng = np.random.default_rng(101)
            ex["band"] = D.band_of(ex.layer.values, L)
            rnd = pd.concat([g_.iloc[rng.permutation(len(g_))[:20]] for _, g_ in ex.groupby("band")])
            units_e = {}
            for src, df in (("exact", top_exact), ("relp", top_relp), ("random", rnd)):
                for l, t, e in df[["layer", "pos", "expert"]].itertuples(index=False):
                    units_e.setdefault((int(l), int(t), int(e)), []).append(src)
            spec = [dict(kind="position", layer=l, pos=t, expert=-1, sources=["grid"]) for l in range(L) for t in positions]
            spec += [dict(kind="expert", layer=l, pos=t, expert=e, sources=s_) for (l, t, e), s_ in sorted(units_e.items())]
            S.write_json(out / "units.json", spec)
            units = []
            for i in range(n):
                for si, u in enumerate(spec):
                    l, t, e = u["layer"], u["pos"], u["expert"]
                    if u["kind"] == "position":
                        units.append((i, 0, 0, l, t, -1, 0, 0, si))
                    else:
                        aa, bb = e in ta[l, i, t], e in tb[l, i, t]
                        if aa or bb:
                            units.append((i, 0, 1, l, t, e, int(aa), int(bb), si))
            units = np.asarray(units, dtype=np.int64)
            spec_idx = units[:, 8].copy()
            units[:, 8] = -1
            variants = tuple(FREEZE)
            res = run_units(m, A, Bb, tg, fo, units, variants, a.buf_gb, a.sub_rows, out, "freeze")
            cols = {"row": units[:, 0], "row_id": [rows[i]["row_id"] for i in units[:, 0]], "unit": spec_idx,
                    "kind": np.where(units[:, 2] == 0, "position", "expert"), "layer": units[:, 3], "pos": units[:, 4],
                    "expert": units[:, 5], "active_A": units[:, 6].astype(bool), "active_B": units[:, 7].astype(bool),
                    "margin_A": res["margin_A"], "margin_B": res["margin_B"], "route_changes": res["route_changes"],
                    "direct": res["direct"]}
            for vi, vn in enumerate(variants):
                cols[f"effect_{vn}"] = res["margin"][:, vi] - res["margin_B"]
            pd.DataFrame(cols).to_parquet(out / "freeze.parquet", index=False)
            done(rows=n, units=len(spec), patched=int(len(units)), variants=list(variants), positions=positions,
                 effect="denoise F(B <- A) - F_B per freeze variant (later layers keep the B run's quantities)")
        else:
            agg = torch.load(p7 / "score_zero/aggregate.pt", map_location="cpu", weights_only=True)
            mean, support = agg["mean"], agg["support"]
            nr = D.load_json(p7 / "score_zero/neuron_ranking.json")[:a.neuron_top]
            top = [tuple(c["coordinate"]) for c in nr]
            chosen = torch.zeros_like(support)
            for c in top:
                chosen[c] = True
            rnd = S.random_matched(chosen, support, torch.Generator().manual_seed(101 + a.neuron_top))
            rand = [tuple(x) for x in torch.nonzero(rnd).tolist()]
            l0, t0_, e0 = D.load_json(p7 / "score_zero/expert_position_ranking.json")[0]["coordinate"]
            mv = mean[l0, t0_, e0].abs()
            js = torch.argsort(mv, descending=True)[:768].tolist() if m.I > 768 else list(range(m.I))
            topexp = [(l0, t0_, e0, j) for j in js]
            sets = {}
            for name, lst in (("top", top), ("random", rand), ("top_expert", topexp)):
                for c in lst:
                    sets.setdefault(tuple(int(x) for x in c), []).append(name)
            coords = sorted(sets)
            track = torch.tensor(coords, dtype=torch.long)
            S.write_json(out / "neurons.json", [{"coordinate": list(c), "sets": sets[c]} for c in coords])
            units = []
            for i in range(n):
                for ci, (l, t, e, j) in enumerate(coords):
                    aa, bb = e in ta[l, i, t], e in tb[l, i, t]
                    if aa or bb:
                        units.append((i, 0, 2, l, t, e, int(aa), int(bb), j, ci))
            units = np.asarray(units, dtype=np.int64)
            cidx = units[:, 9]
            units = units[:, :9]
            res = run_units(m, A, Bb, tg, fo, units, ("natural",), a.buf_gb, a.sub_rows, out, "neuron")
            pd.DataFrame({"row": units[:, 0], "neuron": cidx, "layer": units[:, 3], "pos": units[:, 4], "expert": units[:, 5],
                          "j": units[:, 8], "active_A": units[:, 6].astype(bool), "active_B": units[:, 7].astype(bool),
                          "margin_A": res["margin_A"], "margin_B": res["margin_B"],
                          "effect": res["margin"][:, 0] - res["margin_B"], "route_changes": res["route_changes"]}
                         ).to_parquet(out / "neuron_patch.parquet", index=False)
            # per-pair RelP of the same neurons at endpoint B (denoise prediction)
            g = Group("score", torch.cat([A, Bb]), tg + tg, fo + fo, record_h_in=True)
            forward_pass(m, [g])
            pm = R.pool_mask(rows, "non_special")
            vs = ["relp", "relp_nohalf", "grad_rw"]
            rr = multi_backward(m, g, endpoint_triples(n, "B"), pm, vs, track=track)
            np.savez_compressed(out / "relp_neuron_rows.npz", coords=track.numpy(),
                                **{f"v_{k_}": rr[k_]["tracked"].numpy() for k_ in vs})
            done(rows=n, neurons=len(coords), patched=int(len(units)), sets={k_: sum(k_ in s_ for s_ in sets.values())
                 for k_ in ("top", "random", "top_expert")}, top_expert=[l0, t0_, e0],
                 unit_semantics="this neuron's q := the clean run's q (0 if the expert is not routed); MoE output changes by dq * W_down column")
        print(f"{a.command} done in {time.time() - t0:.0f}s", flush=True)
        return

    # ---- addback (S2.1)
    rows, split_of = load_rows(S, a.manifest_dir, a.task, ["validation", "test"], a.max_rows)
    n_disc = len(load_rows(S, a.manifest_dir, a.task, ["discovery"])[0])
    agg = torch.load(p7 / "score_zero/aggregate.pt", map_location="cpu", weights_only=True)
    support_ep = agg["support"].any(-1)                                         # [L, T, E]
    L, T, E = support_ep.shape
    vals = {"relp": agg["mean"].sum(-1)}
    del agg
    vals["relp_B"] = torch.load(p7 / "score_endpointB/aggregate.pt", map_location="cpu", weights_only=True)["mean"].sum(-1)
    ex = D.exact_means(p7)
    xv = torch.zeros(L, T, E, dtype=torch.float64)
    xv[torch.tensor(ex.layer.values), torch.tensor(ex.pos.values), torch.tensor(ex.expert.values)] = torch.tensor(ex["mean"].values, dtype=torch.float64)
    vals["exact"] = xv
    nol = xv.clone()
    nol[:3] = 0
    vals["exact_noL0-2"] = nol
    sc = D.load_sparse(out.parent / "p8_rules/scores_A.npz")
    for name, col in (("grad", "grad"), ("rw", a.rw_variant)):
        if col is None:
            continue
        x_ = torch.zeros(L, T, E, dtype=torch.float64)
        x_.index_put_((torch.tensor(sc.layer.values, dtype=torch.long), torch.tensor(sc.pos.values, dtype=torch.long),
                       torch.tensor(sc.expert.values, dtype=torch.long)), torch.tensor(sc[col].values, dtype=torch.float64),
                      accumulate=True)
        vals[name] = x_ / n_disc
    pth = out.parent / "p8_path/scores_path.npz"           # multi-step scores (simple, nounpp): extra rankings
    if pth.exists():
        sp = D.load_sparse(pth)
        for name, col in (("path_grad_rw", "grad_rw"), ("path_relp", "relp")):
            if col in sp.columns:
                x_ = torch.zeros(L, T, E, dtype=torch.float64)
                x_.index_put_((torch.tensor(sp.layer.values, dtype=torch.long), torch.tensor(sp.pos.values, dtype=torch.long),
                               torch.tensor(sp.expert.values, dtype=torch.long)), torch.tensor(sp[col].values, dtype=torch.float64),
                              accumulate=True)
                vals[name] = x_ / n_disc
    # Lu's order (final position only): layers by exact whole-MoE rescue at the final position, inside a layer the
    # recurrent experts (routed at A in >= half of the discovery pairs) by exact rescue, then the others
    px = D.exact_means(p7, kind="position")
    lay_order = px[px.pos == T - 1].sort_values("mean", ascending=False).layer.tolist()
    ta = torch.load(p7 / "patch_full/routes.pt", weights_only=True)[0][:, :n_disc, T - 1]   # [L, n_disc, k]
    lu_list = []
    for l in lay_order:
        cnt = torch.bincount(ta[l].reshape(-1), minlength=E)
        cand = [e for e in range(E) if bool(support_ep[l, T - 1, e])]
        rec = sorted([e for e in cand if cnt[e] >= (n_disc + 1) // 2], key=lambda e: -float(xv[l, T - 1, e]))
        rest = sorted([e for e in cand if cnt[e] < (n_disc + 1) // 2], key=lambda e: -float(xv[l, T - 1, e]))
        lu_list += [(l, T - 1, e) for e in rec + rest]
    Label = namedtuple("Label", "ranking seed budget")
    plan = json.loads((p7 / "score_mean/selection_plan.json").read_text())
    qmean = torch.load(p7 / "score_mean/aggregate.pt", map_location="cpu", weights_only=True)["qmean"]
    jobs, rk_out = [], {}
    for unit in ("expert_position", "expert_all_positions"):
        if unit == "expert_position":
            elig = support_ep
            V = {k_: v_ for k_, v_ in vals.items()}
            lu_order = [l * T * E + t * E + e for l, t, e in lu_list]
            budgets = EP_BUDGETS
        else:
            elig = support_ep.any(1)
            V = {k_: v_.sum(1) for k_, v_ in vals.items()}
            seen, lu_order = set(), []
            for l, _, e in lu_list:
                if (l, e) not in seen:
                    seen.add((l, e))
                    lu_order.append(l * E + e)
            budgets = WE_BUDGETS
        n_el = int(elig.sum())
        masks = {}
        orders = {k_: S.rank_indices(v_, elig, n_el) for k_, v_ in V.items()}
        orders["lu"] = torch.tensor(lu_order, dtype=torch.long)
        rk_out[unit] = {k_: o.tolist()[:500] for k_, o in orders.items()}
        for name, o in orders.items():
            for b in budgets:
                if b > len(o) or b >= n_el:
                    continue
                c_ = torch.zeros_like(elig)
                c_.reshape(-1)[o[:b]] = True
                masks[Label(name, 0, b)] = c_
                if name == "relp":
                    for seed in plan["random_seeds"]:
                        if unit == "expert_all_positions":
                            rnd = S.random_matched(c_[:, None, :], elig[:, None, :], torch.Generator().manual_seed(seed + b))[:, 0, :]
                        else:
                            rnd = S.random_matched(c_, elig, torch.Generator().manual_seed(seed + b))
                        masks[Label("random", seed, b)] = rnd
        masks[Label("all", 0, n_el)] = elig.clone()
        jobs.append({"masks": masks, "unit": unit, "baseline": "mean_q_routed", "qmean": qmean})
        print(f"{unit}: {len(masks)} labels, {n_el} eligible units", flush=True)
    S.write_json(out / "rankings.json", rk_out)
    # memory: evaluate() keeps every group's per-layer pair records; split the labels into passes that fit
    T_ = len(rows[0]["input_ids"])
    per_label = 4 * len(rows) * T_ * (m.k * m.I + m.k * m.H) * 2
    lpp = max(1, int(4e9 // per_label))
    recs = []
    for job in jobs:
        labs = list(job["masks"])
        parts = []
        for c0 in range(0, len(labs), lpp):
            sub = {**job, "masks": {k_: job["masks"][k_] for k_ in labs[c0:c0 + lpp]}}
            parts.append(R.evaluate(m, rows, [sub], "non_special", a.eval_chunk)[0])
            print(f"{job['unit']}: labels {min(len(labs), c0 + lpp)}/{len(labs)} ({time.time() - t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()
        recs.append([r_ for part in parts for r_ in part if r_["ranking"] is not None])
    allrec, summ = [], []
    for job, rr in zip(jobs, recs):
        for r_ in rr:
            r_["unit"] = job["unit"]
            r_["split"] = split_of[[x["row_id"] for x in rows].index(r_["row_id"])]
        allrec += rr
        for sp in ("validation", "test", "all"):
            sub = [r_ for r_ in rr if sp == "all" or r_["split"] == sp]
            for s_ in S.summarize(sub):
                summ.append({**s_, "unit": job["unit"], "split": sp})
    pd.DataFrame(allrec).to_parquet(out / "interventions.parquet", index=False)
    S.write_json(out / "summary.json", summ)
    done(rows=len(rows), rw_variant=a.rw_variant, baseline="mean_q_routed (keep / remove); noise / denoise use the other run",
         budgets={"expert_position": EP_BUDGETS, "expert_all_positions": WE_BUDGETS},
         exact_whole_expert="sum over positions of single expert-position exact means (additive approximation)")
    print(f"addback done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
