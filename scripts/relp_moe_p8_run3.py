#!/usr/bin/env python3
"""Phase 8 stage 3 runner (docs/relp_moe_phase8_plan.md section 8d): tasks whose prompts vary in length (CounterFact, two-hop
capitals), manifests of scripts/relp_moe_p8_prepare.py. Rows are bucketed by token length; every per-bucket computation is one
of the stage-2 functions, and moetrace.relp_moe_p8.lockstep serves all buckets from shared passes over the checkpoint.
Units are read at each row's own role positions: role 0 = subject (last token of the changed word), role 1 = final token.

  scan     two-hop competence scan (no verification needed): logit(own capital) - max logit(other capitals) per city.
  verify   the Phase-7 verify() on the first 4 calibration pairs (one length) -> verification.json (gate of all other commands).
  probe    A-vs-B differences of the quantities RelP holds constant, at the roles and averaged from the changed token on.
  rules    discovery: per-pair scores at endpoints B and A for the backward-rule variants.
  path     discovery: 8-step path scores.
  patch    discovery + validation + test: exact patches, every position from the first changed token x every layer x
           (whole MoE output + every expert routed at A or B).
  noise    the role-position patches of the first 40 discovery pairs again, in other batches.
  freeze   first 100 discovery pairs, denoise: role-level units in the 8 FREEZE variants.
  addback  validation + test: noise / denoise curves for role units and whole experts under several rankings.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CLS = ("subject_other", "subject", "between", "final")
ROLE_BUDGETS = [1, 2, 5, 10, 20, 50, 100, 200]
WHOLE_BUDGETS = [1, 2, 5, 10, 20, 50, 100]
PROBE_Q = ("jaccard", "tv_w", "dlogit_rms", "dsig_mean", "lognorm2", "attn_tv", "resid_rel")


def meta_of(r):
    """(first changed position, subject role position, final position, class per position (-1 before the first change))."""
    import numpy as np
    T = len(r["input_ids"])
    s, f = r["roles"]["subject"], r["roles"]["final"]
    first = min(r["changed_positions"])
    cls = np.full(T, -1)
    for t in range(first, T):
        cls[t] = 3 if t == f else 1 if t == s else 0 if t in r["subject_positions"] else 2
    return first, s, f, cls


def buckets_of(rows):
    b = {}
    for i, r in enumerate(rows):
        b.setdefault(len(r["input_ids"]), []).append(i)
    return dict(sorted(b.items()))


def reduce_pe(pe, metas):
    """pair_expert [n, L, T, E] -> role [n, L, 2, E], all positions [n, L, E], class totals over experts [n, L, 4]."""
    import torch
    n, L, T, E = pe.shape
    role = torch.stack([pe[i][:, [metas[i][1], metas[i][2]]] for i in range(n)])
    tot = pe.sum(-1)                                             # [n, L, T]
    cls = torch.zeros(n, L, 4)
    for i in range(n):
        c = torch.from_numpy(metas[i][3])
        for k in range(4):
            if bool((c == k).any()):
                cls[i, :, k] = tot[i][:, c == k].sum(-1)
    return role, pe.sum(2), cls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("scan", "verify", "probe", "rules", "path", "patch", "noise", "freeze", "addback"))
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path)
    p.add_argument("--task", required=True)
    p.add_argument("--base-dir", type=Path, required=True, help="results/relp_p8/<model>/<task>")
    p.add_argument("--candidates", type=Path, help="scan: data/relp_p8/twohop_candidates_<model>.json")
    p.add_argument("--variants", nargs="+", default=["relp", "grad", "grad_rw", "relp_nohalf", "relp_rw"])
    p.add_argument("--path-variants", nargs="+", default=["relp", "relp_rw", "grad_rw"])
    p.add_argument("--path-steps", type=int, default=8)
    p.add_argument("--chunk", type=int, default=100, help="path: pairs per chunk inside a length bucket")
    p.add_argument("--directions", nargs="+", default=["denoise", "noise"], choices=("denoise", "noise"))
    p.add_argument("--freeze-rows", type=int, default=100)
    p.add_argument("--noise-rows", type=int, default=40)
    p.add_argument("--buf-gb", type=float, default=6.0)
    p.add_argument("--sub-rows", type=int, default=3000)
    p.add_argument("--max-rows", type=int, default=0, help="first N rows per split (smoke only)")
    a = p.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from moetrace import relp_sva as S
    from moetrace.relp_moe_p8 import (FREEZE, RuleRelP, endpoint_triples, evaluate_roles, gate_probe, lockstep, multi_backward,
                                      path_forward, path_triples)
    from moetrace.relp_moe_patch import routes_pass
    from moetrace.relp_moe_stream import Group, StreamRelP, forward_pass
    import relp_sva_moe_run as R
    import relp_moe_p8_data as D
    import relp_moe_p8_run as P2
    out = a.base_dir / a.command
    if (out / "run_meta.json").exists() and json.loads((out / "run_meta.json").read_text()).get("status") == "complete":
        print("complete; nothing to do")
        return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cls_m = StreamRelP if a.command == "verify" else RuleRelP
    m = cls_m(str(a.model_dir.resolve()), device="cuda", dtype=torch.bfloat16)
    common = {"utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "task": a.task, "script_sha256": S.sha(__file__),
              "p8_module_sha256": S.sha(ROOT / "moetrace/relp_moe_p8.py"), "status": "running"}

    def done(**kw):
        S.write_json(out / "run_meta.json", {**common, "status": "complete", "seconds": time.time() - t0, **kw})
        print(f"{a.command} done in {time.time() - t0:.0f}s", flush=True)

    # ---- scan (two-hop competence; a data-selection step, stated in the manifest)
    if a.command == "scan":
        cand = json.loads(a.candidates.read_text())
        items = cand["items"]
        states = sorted(cand["capital_tokens"])
        cap_ids = torch.tensor([cand["capital_tokens"][s] for s in states])
        bk = {}
        for i, it in enumerate(items):
            bk.setdefault(len(it["ids"]), []).append(i)

        def job(mp, idx):
            ids = torch.tensor([items[i]["ids"] for i in idx])
            g = Group("s", ids, [0] * len(idx), [1] * len(idx))
            forward_pass(mp, [g])
            normed = mp.rms(g.h[:, -1:], mp.final_norm)[:, 0].float()
            return (normed @ mp.W_U[cap_ids.to(mp.dev)].float().t()).cpu()
        res = lockstep(m, [lambda mp, idx=idx: job(mp, idx) for idx in bk.values()])
        recs = []
        for idx, lg in zip(bk.values(), res):
            for j, i in enumerate(idx):
                it = items[i]
                own = states.index(it["state"])
                same = cap_ids == cap_ids[own]
                other = lg[j].masked_fill(same, float("-inf"))
                recs.append({"city": it["city"], "state": it["state"], "capital": it["capital"],
                             "n_city_tokens": len(it["city_positions"]), "n_tokens": len(it["ids"]),
                             "logit_own": float(lg[j, own]), "best_other": float(other.max()),
                             "best_other_state": states[int(other.argmax())], "margin": float(lg[j, own] - other.max()),
                             "rank_own": int((lg[j] > lg[j, own]).sum()) + 1})
        df = pd.DataFrame(recs)
        df.to_parquet(out / "scan.parquet", index=False)
        S.write_json(out / "run_meta.json", {**common, "status": "complete", "seconds": time.time() - t0, "cities": len(df),
                     "pass_margin_1": int((df.margin >= 1).sum()), "pass_margin_0": int((df.margin >= 0).sum()),
                     "rule": "logit(first token of the own capital) - max over the other capitals' first tokens >= 1"})
        print(f"scan: {len(df)} cities, {(df.margin >= 1).sum()} with margin >= 1, {(df.margin >= 0).sum()} with margin >= 0", flush=True)
        return

    meta, calibration = S.read_split(a.manifest_dir, a.task, "calibration")
    S.check_tokenizer_files(a.model_dir, meta)
    ident = R.identity(m, a.model_dir, S)
    common.update(manifest_sha256=S.sha(a.manifest_dir / "manifest.json"), identity=ident)
    vfile = a.base_dir / "verify/verification.json"
    if a.command == "verify":
        rows = calibration[:4]
        assert len({len(r["input_ids"]) for r in rows}) == 1
        report = R.verify(m, rows, 1e-4, 1e-3, 1e-3)
        report.update(common)
        report["passed"] = all(c["passed"] for c in report["checks"])
        report["status"] = "complete" if report["passed"] else "failed"
        S.write_json(vfile, report)
        S.write_json(out / "run_meta.json", {**common, "status": report["status"], "seconds": time.time() - t0})
        print(json.dumps({"passed": report["passed"], "n_checks": len(report["checks"]),
                          "failed_checks": [c for c in report["checks"] if not c["passed"]][:20]}))
        if not report["passed"]:
            raise SystemExit(1)
        return
    v = json.loads(vfile.read_text())
    if not v.get("passed") or v["identity"] != ident or v["task"] != a.task or v["manifest_sha256"] != common["manifest_sha256"]:
        p.error("Verification is failed, stale, or for another model/task/manifest/runtime")
    common["verification_sha256"] = S.sha(vfile)
    S.write_json(out / "run_meta.json", common)

    def tensors(rs):
        return (R.ids_tensor(rs), R.ids_tensor(rs, True), [r["target_id"] for r in rs], [r["foil_id"] for r in rs])

    def run_buckets(rows, job):
        """job(mp, rows_of_bucket, global indices) per length bucket, in lockstep; returns {T: (indices, result)}."""
        bk = buckets_of(rows)
        res = lockstep(m, [lambda mp, idx=idx: job(mp, [rows[i] for i in idx], idx) for idx in bk.values()])
        return {T: (idx, r) for (T, idx), r in zip(bk.items(), res)}

    def scatter(n, parts, shape_of):
        """Assemble per-bucket arrays (first axis = bucket rows) into arrays over all rows."""
        full = None
        for idx, arr in parts:
            arr = np.asarray(arr)
            if full is None:
                full = np.zeros((n,) + arr.shape[1:], dtype=arr.dtype)
            full[np.asarray(idx)] = arr
        return full

    disc, _ = P2.load_rows(S, a.manifest_dir, a.task, ["discovery"], a.max_rows)
    nd = len(disc)

    if a.command == "probe":
        rows, split_of = P2.load_rows(S, a.manifest_dir, a.task, ["discovery", "validation", "test"], a.max_rows)

        def job(mp, rs, idx):
            A, Bb, _, _ = tensors(rs)
            ms = [meta_of(r) for r in rs]
            pr = gate_probe(mp, A, Bb, min(x[0] for x in ms))
            o = {}
            for q in PROBE_Q:
                x = np.abs(pr[q]) if q.startswith("lognorm") else pr[q]            # [n, L, T]
                o[q + "_role"] = np.stack([x[i][:, [ms[i][1], ms[i][2]]] for i in range(len(rs))])
                o[q + "_mean"] = np.stack([np.nanmean(x[i][:, ms[i][0]:], axis=1) for i in range(len(rs))])
            return o
        res = run_buckets(rows, job)
        arrays = {k_: scatter(len(rows), [(idx, r[k_]) for idx, r in res.values()], None) for k_ in next(iter(res.values()))[1]}
        np.savez_compressed(out / "probe.npz", **arrays, split=np.array(split_of), row_id=np.array([r["row_id"] for r in rows]))
        done(rows=len(rows), buckets={int(T): len(idx) for T, (idx, _) in res.items()})
        return

    if a.command in ("rules", "path"):
        vs = a.variants if a.command == "rules" else a.path_variants
        K = a.path_steps

        def job(mp, rs, idx):
            A, Bb, tg, fo = tensors(rs)
            n = len(rs)
            ms = [meta_of(r) for r in rs]
            pm = R.pool_mask(rs, "non_special")
            o = {}
            if a.command == "rules":
                g = Group("score", torch.cat([A, Bb]), tg + tg, fo + fo, record_h_in=True)
                forward_pass(mp, [g])
                for ep in "BA":
                    res = multi_backward(mp, g, endpoint_triples(n, ep), pm, vs)
                    for v_ in vs:
                        o[(ep, v_)] = reduce_pe(res[v_]["pair_expert"], ms)
                d = g.delta.float().cpu()
                o["margin_A"], o["margin_B"] = d[:n], d[n:]
            else:   # pairs of a bucket in chunks (memory: (K + 1) rows per pair, all buckets alive in lockstep)
                parts, dl = {v_: [] for v_ in vs}, []
                for c0 in range(0, n, a.chunk):
                    c1 = min(n, c0 + a.chunk)
                    g = path_forward(mp, A[c0:c1], Bb[c0:c1], tg[c0:c1], fo[c0:c1], K)
                    res = multi_backward(mp, g, path_triples(c1 - c0, K), pm[c0:c1], vs)
                    for v_ in vs:
                        parts[v_].append(res[v_]["pair_expert"])
                    dl.append(g.delta.float().view(K + 1, c1 - c0).t().cpu())            # [pairs, K + 1]
                    del g, res
                    torch.cuda.empty_cache()
                for v_ in vs:
                    o[("P", v_)] = reduce_pe(torch.cat(parts[v_]), ms)
                o["deltas"] = torch.cat(dl)
                g = None
            del g
            torch.cuda.empty_cache()
            return o
        res = run_buckets(disc, job)
        for ep in ("B", "A") if a.command == "rules" else ("P",):
            arrays = {}
            for v_ in vs:
                for j, part in enumerate(("role", "all", "cls")):
                    arrays[f"v_{v_}_{part}"] = scatter(nd, [(idx, r[(ep, v_)][j].numpy()) for idx, r in res.values()], None).astype(np.float32)
            if ep == "P":
                arrays["deltas"] = scatter(nd, [(idx, r["deltas"].numpy()) for idx, r in res.values()], None)
            np.savez_compressed(out / ("scores_path.npz" if ep == "P" else f"scores_{ep}.npz"), variants=np.array(vs),
                                row_id=np.array([r["row_id"] for r in disc]), **arrays)
        if a.command == "rules":
            pd.DataFrame({"row_id": [r["row_id"] for r in disc], "group_id": [r["group_id"] for r in disc],
                          "margin_A": scatter(nd, [(idx, r["margin_A"].numpy()) for idx, r in res.values()], None),
                          "margin_B": scatter(nd, [(idx, r["margin_B"].numpy()) for idx, r in res.values()], None)}
                         ).to_parquet(out / "cases.parquet", index=False)
        done(rows=nd, variants=vs, buckets={int(T): len(idx) for T, (idx, _) in res.items()},
             arrays="v_<variant>_role [pair, layer, role(subject, final), expert]; _all [pair, layer, expert] (all positions); _cls [pair, layer, class] (sum over experts; classes " + ", ".join(CLS) + ")")
        return

    def share_buffers(unit_sets, variants):
        """Per-bucket buffer budget (GB) such that all buckets need about the same number of passes within --buf-gb in total."""
        need = {T: len(u) * T * m.H * 2 * len(variants) for T, u in unit_sets.items()}
        npass = max(1, math.ceil(sum(need.values()) / (a.buf_gb * 1e9)))
        return {T: max(need[T] / npass / 1e9 * 1.02, 1e-3) for T in need}, npass

    def patch_buckets(rows, unit_sets, variants, tag):
        """unit_sets {T: int64 [U, 9] with LOCAL row indices of that bucket}; returns {T: (indices, units, result)}."""
        bk = buckets_of(rows)
        bufs, npass = share_buffers(unit_sets, variants)
        print(f"{tag}: {sum(len(u) for u in unit_sets.values())} units x {len(variants)} variants in {len(unit_sets)} buckets, ~{npass} passes", flush=True)

        def job(mp, T):
            rs = [rows[i] for i in bk[T]]
            A, Bb, tg, fo = tensors(rs)
            return P2.run_units(mp, A, Bb, tg, fo, unit_sets[T], variants, bufs[T], a.sub_rows, out, f"{tag}_T{T}")
        Ts = [T for T in bk if T in unit_sets and len(unit_sets[T])]
        res = lockstep(m, [lambda mp, T=T: job(mp, T) for T in Ts])
        return {T: (bk[T], unit_sets[T], r) for T, r in zip(Ts, res)}

    def load_routes(pdir, rows):
        """{T: (topi_A, topi_B [L, n, T, k] numpy)} saved by the patch command for the same row list."""
        o = {}
        for T, idx in buckets_of(rows).items():
            z = torch.load(pdir / f"routes_T{T}.pt", weights_only=True)
            assert z["row_ids"][: len(idx)] == [rows[i]["row_id"] for i in idx] or a.max_rows
            o[T] = (z["ta"].numpy(), z["tb"].numpy())
        return o

    if a.command == "patch":
        rows, split_of = P2.load_rows(S, a.manifest_dir, a.task, ["discovery", "validation", "test"], a.max_rows)
        bk = buckets_of(rows)
        rres = lockstep(m, [lambda mp, idx=idx: routes_pass(mp, *tensors([rows[i] for i in idx])[:2]) for idx in bk.values()])
        unit_sets = {}
        for (T, idx), (ta, tb) in zip(bk.items(), rres):
            torch.save({"ta": ta, "tb": tb, "row_ids": [rows[i]["row_id"] for i in idx]}, out / f"routes_T{T}.pt")
            ta, tb = ta.numpy(), tb.numpy()
            pool = R.pool_mask([rows[idx[0]]], "non_special")[0]
            us = []
            for j, i in enumerate(idx):
                first = meta_of(rows[i])[0]
                for t in range(first, T):
                    if not bool(pool[t]):
                        continue
                    for l in range(m.L):
                        sa, sb = set(ta[l, j, t].tolist()), set(tb[l, j, t].tolist())
                        un = sorted(sa | sb)
                        for di, dname in enumerate(("denoise", "noise")):
                            if dname not in a.directions:
                                continue
                            us.append((j, di, 0, l, t, -1, 0, 0, -1))
                            us += [(j, di, 1, l, t, e, int(e in sa), int(e in sb), -1) for e in un]
            unit_sets[T] = np.asarray(us, dtype=np.int64).reshape(-1, 9)
        res = patch_buckets(rows, unit_sets, ("natural",), "patch")
        frames = []
        for T, (idx, u, r) in res.items():
            gi = np.asarray(idx)[u[:, 0]]
            ms = [meta_of(rows[i]) for i in idx]
            role = np.array([0 if t == ms[j][1] else 1 if t == ms[j][2] else -1 for j, t in zip(u[:, 0], u[:, 4])])
            cl = np.array([ms[j][3][t] for j, t in zip(u[:, 0], u[:, 4])])
            mA, mB = r["margin_A"], r["margin_B"]
            frames.append(pd.DataFrame({
                "row": gi, "direction": np.where(u[:, 1] == 0, "denoise", "noise"), "kind": np.where(u[:, 2] == 0, "position", "expert"),
                "layer": u[:, 3].astype(np.int16), "pos": u[:, 4].astype(np.int16), "role": role.astype(np.int8), "cls": cl.astype(np.int8),
                "expert": u[:, 5].astype(np.int16), "active_A": u[:, 6].astype(bool), "active_B": u[:, 7].astype(bool),
                "margin_A": mA, "margin_B": mB, "effect": np.where(u[:, 1] == 0, r["margin"][:, 0] - mB, mA - r["margin"][:, 0]),
                "route_changes": r["route_changes"]}))
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(out / "patch.parquet", index=False)
        pd.DataFrame({"row": np.arange(len(rows)), "row_id": [r["row_id"] for r in rows], "group_id": [r["group_id"] for r in rows],
                      "split": split_of, "case": [r.get("case") for r in rows], "T": [len(r["input_ids"]) for r in rows],
                      "subject_pos": [r["roles"]["subject"] for r in rows], "first_changed": [min(r["changed_positions"]) for r in rows]}
                     ).to_parquet(out / "rows.parquet", index=False)
        done(rows=len(rows), units=int(len(df)), directions=a.directions,
             unit_semantics="complete replacement at one layer (position: MoE output; expert: w_e E_e(x)); later layers natural",
             columns="role: 0 subject, 1 final, -1 other; cls: " + ", ".join(f"{i} {c}" for i, c in enumerate(CLS)))
        return

    pdir = a.base_dir / "patch"

    def role_exact(kind="expert"):
        """Discovery denoise means at the roles (absent = 0): expert -> [L, 2, E], position -> [L, 2]; and counts."""
        import pyarrow.dataset as ds
        f = (ds.field("direction") == "denoise") & (ds.field("kind") == kind) & (ds.field("role") >= 0) & (ds.field("row") < nd)
        t = ds.dataset(pdir / "patch.parquet").to_table(filter=f, columns=["layer", "role", "expert", "effect"]).to_pandas()
        if kind == "position":
            x = torch.zeros(m.L, 2, dtype=torch.float64)
            g = t.groupby(["layer", "role"]).effect.sum()
            x[torch.tensor(g.index.get_level_values(0).values, dtype=torch.long), torch.tensor(g.index.get_level_values(1).values, dtype=torch.long)] = torch.tensor(g.values, dtype=torch.float64)
            return x / nd, None
        g = t.groupby(["layer", "role", "expert"]).effect.agg(["sum", "size"])
        ix = [torch.tensor(g.index.get_level_values(i).values, dtype=torch.long) for i in range(3)]
        x = torch.zeros(m.L, 2, m.E, dtype=torch.float64)
        c = torch.zeros(m.L, 2, m.E, dtype=torch.long)
        x[ix[0], ix[1], ix[2]] = torch.tensor(g["sum"].values, dtype=torch.float64)
        c[ix[0], ix[1], ix[2]] = torch.tensor(g["size"].values, dtype=torch.long)
        return x / nd, c

    def units_for(rows, routes, spec, with_noise=False):
        """Per bucket: every spec unit {kind, layer, role, expert} of every row at the row's role position (expert units only
        where routed at A or B). Returns {T: int64 [U, 9]} (local rows) and {T: spec index per unit}."""
        us, si = {}, {}
        for T, idx in buckets_of(rows).items():
            ta, tb = routes[T]
            u, s_ = [], []
            for j, i in enumerate(idx):
                ms = meta_of(rows[i])
                for k_, sp in enumerate(spec):
                    l, t, e = sp["layer"], (ms[1], ms[2])[sp["role"]], sp["expert"]
                    if sp["kind"] == "position":
                        u.append((j, 0, 0, l, t, -1, 0, 0, -1))
                        s_.append(k_)
                    else:
                        aa, bb = e in ta[l, j, t], e in tb[l, j, t]
                        if aa or bb:
                            u.append((j, 0, 1, l, t, e, int(aa), int(bb), -1))
                            s_.append(k_)
            us[T], si[T] = np.asarray(u, dtype=np.int64).reshape(-1, 9), np.asarray(s_, dtype=np.int64)
        return us, si

    def unit_frame(res, si, rows, variants):
        frames = []
        for T, (idx, u, r) in res.items():
            d = {"row": np.asarray(idx)[u[:, 0]], "unit": si[T], "kind": np.where(u[:, 2] == 0, "position", "expert"),
                 "layer": u[:, 3], "pos": u[:, 4], "expert": u[:, 5], "active_A": u[:, 6].astype(bool), "active_B": u[:, 7].astype(bool),
                 "margin_A": r["margin_A"], "margin_B": r["margin_B"], "route_changes": r["route_changes"], "direct": r["direct"]}
            for vi, vn in enumerate(variants):
                d[f"effect_{vn}"] = r["margin"][:, vi] - r["margin_B"]
            frames.append(pd.DataFrame(d))
        return pd.concat(frames, ignore_index=True)

    if a.command in ("noise", "freeze"):
        rows = disc[: (a.noise_rows if a.command == "noise" else a.freeze_rows)]
        routes = load_routes(pdir, rows)
        xe, cnt = role_exact()
        present = torch.nonzero(cnt > 0)
        if a.command == "noise":
            spec = [dict(kind="position", layer=l, role=r_, expert=-1) for l in range(m.L) for r_ in (0, 1)]
            spec += [dict(kind="expert", layer=int(l), role=int(r_), expert=int(e)) for l, r_, e in present.tolist()]
            variants = ("natural",)
        else:
            sb = np.load(a.base_dir / "rules/scores_B.npz")["v_relp_role"].mean(0)             # [L, 2, E]
            flat = lambda x, n_: [tuple(int(v_) for v_ in np.unravel_index(i, x.shape)) for i in np.argsort(-np.abs(x).ravel())[:n_]]
            src = {}
            for name, lst in (("exact", flat(xe.numpy(), 150)), ("relp", flat(sb, 150))):
                for c in lst:
                    src.setdefault(c, []).append(name)
            rng = np.random.default_rng(101)
            pres = present.numpy()
            band = D.band_of(pres[:, 0], m.L)
            for b in np.unique(band):
                sel = pres[band == b]
                for c in sel[rng.permutation(len(sel))[:20]]:
                    src.setdefault(tuple(int(v_) for v_ in c), []).append("random")
            spec = [dict(kind="position", layer=l, role=r_, expert=-1, sources=["grid"]) for l in range(m.L) for r_ in (0, 1)]
            spec += [dict(kind="expert", layer=l, role=r_, expert=e, sources=s_) for (l, r_, e), s_ in sorted(src.items())]
            variants = tuple(FREEZE)
        S.write_json(out / "units.json", spec)
        us, si = units_for(rows, routes, spec)
        res = patch_buckets(rows, us, variants, a.command)
        df = unit_frame(res, si, rows, variants)
        df["role"] = [spec[k_]["role"] for k_ in df.unit]
        df.to_parquet(out / f"{a.command}.parquet", index=False)
        done(rows=len(rows), units=len(spec), patched=int(len(df)), variants=list(variants))
        return

    # ---- addback
    rows, split_of = P2.load_rows(S, a.manifest_dir, a.task, ["validation", "test"], a.max_rows)
    patch_rows, _ = P2.load_rows(S, a.manifest_dir, a.task, ["discovery", "validation", "test"], a.max_rows)
    routes = load_routes(pdir, patch_rows)
    L, E = m.L, m.E
    sA, sB = np.load(a.base_dir / "rules/scores_A.npz"), np.load(a.base_dir / "rules/scores_B.npz")
    xe, cnt = role_exact()
    xp, _ = role_exact("position")
    vals_r = {"relp": torch.tensor(sA["v_relp_role"].mean(0)), "relp_B": torch.tensor(sB["v_relp_role"].mean(0)),
              "grad": torch.tensor(sA["v_grad_role"].mean(0)), "exact": xe.float()}
    vals_w = {"relp": torch.tensor(sA["v_relp_all"].mean(0)), "relp_B": torch.tensor(sB["v_relp_all"].mean(0)),
              "grad": torch.tensor(sA["v_grad_all"].mean(0))}
    nol = xe.float().clone()
    nol[:3] = 0
    vals_r["exact_noL0-2"] = nol
    import pyarrow.dataset as ds
    tw = ds.dataset(pdir / "patch.parquet").to_table(filter=(ds.field("direction") == "denoise") & (ds.field("kind") == "expert") & (ds.field("row") < nd),
                                                      columns=["layer", "expert", "effect"]).to_pandas().groupby(["layer", "expert"]).effect.sum() / nd
    xw = torch.zeros(L, E)
    xw[torch.tensor(tw.index.get_level_values(0).values, dtype=torch.long), torch.tensor(tw.index.get_level_values(1).values, dtype=torch.long)] = torch.tensor(tw.values, dtype=torch.float32)
    vals_w["exact"] = xw
    nw = xw.clone()
    nw[:3] = 0
    vals_w["exact_noL0-2"] = nw
    pth = a.base_dir / "path/scores_path.npz"
    if pth.exists():
        z = np.load(pth)
        for name, col in (("path_grad_rw", "grad_rw"), ("path_relp", "relp")):
            if f"v_{col}_role" in z.files:
                vals_r[name], vals_w[name] = torch.tensor(z[f"v_{col}_role"].mean(0)), torch.tensor(z[f"v_{col}_all"].mean(0))
    elig_r = (cnt > 0) | torch.tensor(np.abs(sB["v_relp_role"]).sum(0) > 0)
    elig_w = (xw != 0) | torch.tensor(np.abs(sB["v_relp_all"]).sum(0) > 0)
    # Lu's order at the final role: layers by exact whole-MoE rescue, inside a layer recurrent experts (routed at A in >= half of the
    # discovery pairs at the final position) by exact rescue, then the others
    rec_cnt = torch.zeros(L, E, dtype=torch.long)
    for T, idx in buckets_of(patch_rows).items():
        ta = routes[T][0]
        for j, i in enumerate(idx):
            if i < nd:
                for l in range(L):
                    rec_cnt[l, torch.tensor(ta[l, j, T - 1], dtype=torch.long)] += 1
    lu = []
    for l in torch.argsort(xp[:, 1], descending=True).tolist():
        cand = [e for e in range(E) if bool(elig_r[l, 1, e])]
        recur = sorted([e for e in cand if rec_cnt[l, e] >= (nd + 1) // 2], key=lambda e: -float(xe[l, 1, e]))
        rest = sorted([e for e in cand if rec_cnt[l, e] < (nd + 1) // 2], key=lambda e: -float(xe[l, 1, e]))
        lu += [(l, e) for e in recur + rest]
    Label = namedtuple("Label", "ranking seed budget")
    jobs, rk_out = {}, {}
    for unit, vals, elig, budgets in (("role", vals_r, elig_r, ROLE_BUDGETS), ("all", vals_w, elig_w, WHOLE_BUDGETS)):
        n_el = int(elig.sum())
        orders = {k_: S.rank_indices(v_, elig, n_el) for k_, v_ in vals.items()}
        if unit == "role":
            orders["lu"] = torch.tensor([l * 2 * E + E + e for l, e in lu], dtype=torch.long)
        else:
            seen, o = set(), []
            for l, e in lu:
                if (l, e) not in seen and bool(elig[l, e]):
                    seen.add((l, e))
                    o.append(l * E + e)
            orders["lu"] = torch.tensor(o, dtype=torch.long)
        rk_out[unit] = {k_: o.tolist()[:300] for k_, o in orders.items()}
        masks = {}
        for name, o in orders.items():
            for b in budgets:
                if b > len(o) or b >= n_el:
                    continue
                c_ = torch.zeros_like(elig)
                c_.reshape(-1)[o[:b]] = True
                masks[Label(name, 0, b)] = c_
                if name == "relp":
                    for seed in (101, 102, 103):
                        gen = torch.Generator().manual_seed(seed + b)
                        masks[Label("random", seed, b)] = (S.random_matched(c_, elig, gen) if unit == "role" else
                                                           S.random_matched(c_[:, None, :], elig[:, None, :], gen)[:, 0, :])
        masks[Label("all", 0, n_el)] = elig.clone()
        jobs[unit] = masks
        print(f"{unit}: {len(masks)} labels, {n_el} eligible units", flush=True)
    S.write_json(out / "rankings.json", rk_out)
    bk = buckets_of(rows)
    tok = sum(len(idx) * T for T, idx in bk.items())
    per_label = 2 * tok * m.k * (m.I + m.H) * 2
    lpp = max(1, int(5e9 // per_label))
    recs = []
    for unit, masks in jobs.items():
        labs = list(masks)
        for c0 in range(0, len(labs), lpp):
            sub = {k_: masks[k_] for k_ in labs[c0:c0 + lpp]}

            def job(mp, rs, idx, sub=sub, unit=unit):
                A, Bb, tg, fo = tensors(rs)
                pool = R.pool_mask(rs, "non_special")[0]
                rolepos = torch.tensor([[r["roles"]["subject"], r["roles"]["final"]] for r in rs])
                return evaluate_roles(mp, A, Bb, tg, fo, pool, rolepos, sub, unit)
            res = run_buckets(rows, job)
            for T, (idx, r) in res.items():
                for lab, d in r["margin"].items():
                    for kind, mg in d.items():
                        for j, i in enumerate(idx):
                            recs.append({"row_id": rows[i]["row_id"], "group_id": rows[i]["group_id"], "split": split_of[i], "unit": unit,
                                         "ranking": lab.ranking, "seed": lab.seed, "budget": lab.budget, "intervention": kind,
                                         "margin": float(mg[j]), "margin_A": float(r["margin_A"][j]), "margin_B": float(r["margin_B"][j])})
            print(f"{unit}: labels {min(len(labs), c0 + lpp)}/{len(labs)} ({time.time() - t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()
    pd.DataFrame(recs).to_parquet(out / "interventions.parquet", index=False)
    done(rows=len(rows), budgets={"role": ROLE_BUDGETS, "all": WHOLE_BUDGETS},
         interventions="denoise: input B, the units' q := A's; noise: input A, the units' q := B's (no mean-baseline keep / remove in stage 3)",
         exact_whole_expert="sum over the grid positions of single expert-position exact means (additive approximation)")


if __name__ == "__main__":
    main()
