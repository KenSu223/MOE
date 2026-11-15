#!/usr/bin/env python3
"""Verify, discover, and evaluate fixed SVA neuron/expert lists on layer-streamed MoE checkpoints (Qwen3-30B-A3B-Base,
Mixtral-8x7B-v0.1; OLMoE-1B-7B for cross-checks). No training of model weights.

Same commands, flags, verification gate, output files and definitions as scripts/relp_sva_run.py
(docs/relp_sva_experiment_protocol.md sections 5-7); the engine is moetrace/relp_moe_stream.StreamRelP (layer streaming,
reverse-streamed RelP backward, HF-exact sparse experts) instead of FModel, which needs the whole model on the GPU.
Rows of a split are processed as batches (rows are independent); the size of the bf16 batch-composition effect is
reported by `verify` (diagnostics) and does not enter any gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KINDS = ("remove", "keep", "noise", "denoise")


def identity(m, model_dir, S):
    import torch
    import transformers
    model_dir = Path(model_dir).resolve()
    files = [p for p in sorted(model_dir.glob("*")) if p.is_file() and p.suffix in (".json", ".safetensors", ".bin", ".model")]
    return {"model_dir": str(model_dir), "dtype": str(m.dtype), "device": str(m.dev), "torch": torch.__version__,
            "transformers": transformers.__version__, "engine": "relp_moe_stream.StreamRelP (layer streaming)",
            "native_reference": "transformers eager attention + eager experts (HF reference loop)",
            "engine_sha256": S.sha(ROOT / "moetrace/relp_moe_stream.py"), "sva_sha256": S.sha(ROOT / "moetrace/relp_sva.py"),
            "weights_sha256": S.sha(ROOT / "moetrace/weights.py"), "arch_sha256": S.sha(ROOT / "moetrace/arch.py"),
            "files": {p.name: ({"sha256": S.sha(p)} if p.suffix == ".json" else
                               {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}) for p in files}}


def ids_tensor(rows, reference=False):
    import torch
    return torch.tensor([r["reference_ids" if reference else "input_ids"] for r in rows], dtype=torch.long)


def pool_mask(rows, pool):
    import torch
    from moetrace import relp_sva as S
    return torch.stack([S.position_mask(r, pool) for r in rows])


# ---------------------------------------------------------------------------------------------------------------------
def verify(m, rows, atol, rtol, max_relative):
    """relp_sva.verify, streamed: same checks and names (per sequence, B = 1), plus the same checks on one batched group
    of all sequences ('batch/...'), plus non-gating diagnostics of the bf16 batch-composition effect."""
    import torch
    from moetrace.relp_moe_stream import Group, Override, forward_pass, relp_backward
    checks = []

    def compare(name, a, b):
        a, b = a.detach().float().cpu(), b.detach().float().cpu()
        diff = a - b
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        rel = float(diff.norm() / b.norm().clamp_min(atol)) if finite else None
        checks.append({"name": name, "max_abs": float(diff.abs().max()) if finite else None, "relative_l2": rel,
                       "passed": finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)) and rel <= max_relative})

    def topk_checks(tag, l, rp, rh):
        (pi, pw), (hi, hw) = rp, rh
        ps, po = pi.reshape(-1, m.k).sort(-1)
        hs, ho = hi.reshape(-1, m.k).sort(-1)
        checks.append({"name": f"{tag}/HF/L{l}/topk", "passed": bool(torch.equal(ps, hs))})
        compare(f"{tag}/HF/L{l}/weights", pw.reshape(-1, m.k).gather(-1, po), hw.reshape(-1, m.k).gather(-1, ho))

    seqs = []
    for row in rows:
        for reference in (False, True):
            seqs.append((row, reference, row["row_id"] + ("/B" if reference else "/A")))
    groups = []
    for row, reference, tag in seqs:
        ids = ids_tensor([row], reference)
        t, f = [row["target_id"]], [row["foil_id"]]
        groups += [Group(tag + "|hf", ids, t, f, native=True, record_out=True, record_route=True, record_dense_q=True),
                   Group(tag + "|plain", ids, t, f, pairs=True, keep=set(), record_out=True, record_route=True,
                         record_dense_q=True),
                   Group(tag + "|relp", ids, t, f, relp=True, record_out=True, record_dense_q=True)]
    # the same comparisons on one batched group of all sequences (the composition used for scoring)
    allids = torch.cat([ids_tensor([r], ref) for r, ref, _ in seqs])
    at = [r["target_id"] for r, _, _ in seqs]
    af = [r["foil_id"] for r, _, _ in seqs]
    groups += [Group("batch|hf", allids, at, af, native=True, record_out=True, record_route=True, record_dense_q=True),
               Group("batch|plain", allids, at, af, record_out=True, record_route=True, record_dense_q=True),
               Group("betaA", ids_tensor(rows), [r["target_id"] for r in rows], [r["foil_id"] for r in rows],
                     record_h_in=True)]
    # patch identities on the first row's B input with the first row's A run (layers 0, L/2, L-1; final position)
    row0 = rows[0]
    a0 = row0["row_id"] + "/A"
    idsB = ids_tensor([row0], True)
    T = idsB.shape[1]
    t0, f0 = [row0["target_id"]], [row0["foil_id"]]
    bt = torch.zeros(1, T, dtype=torch.bool, device=m.dev)
    bt[0, -1] = True
    Sfull = torch.zeros(T, m.E, m.I, dtype=torch.bool, device=m.dev)
    Sfull[-1] = True
    one = torch.tensor([True], device=m.dev)
    zero_row = torch.tensor([0], device=m.dev)
    patch_layers = sorted({0, m.L // 2, m.L - 1})
    for lp in patch_layers:
        groups += [
            Group(f"patch{lp}|q", idsB, t0, f0, override_fn=lambda l, w, st, lp=lp: Override(
                rows=one, S=Sfull, value="source", src=st["layer_out"][a0 + "|plain"], src_rows=zero_row) if l == lp else None),
            Group(f"patch{lp}|out", idsB, t0, f0, out_patch_fn=lambda l, st, lp=lp: (
                bt, st["groups"][a0 + "|plain"].outs[l]) if l == lp else None),
            Group(f"patch{lp}|hf", idsB, t0, f0, native=True, out_patch_fn=lambda l, st, lp=lp: (
                bt, st["groups"][a0 + "|hf"].outs[l]) if l == lp else None)]
    forward_pass(m, groups)
    G = {g.name: g for g in groups}
    for row, reference, tag in seqs:
        hf, plain, relp = G[tag + "|hf"], G[tag + "|plain"], G[tag + "|relp"]
        compare(tag + "/HF/metric", plain.delta, hf.delta)
        compare(tag + "/HF/final_residual", plain.residual_final, hf.residual_final)
        compare(tag + "/RelP/metric", relp.delta, plain.delta)
        for l in range(m.L):
            compare(f"{tag}/HF/L{l}/out", plain.outs[l], hf.outs[l])
            compare(f"{tag}/HF/L{l}/q", plain.dense_q[l], hf.dense_q[l])
            compare(f"{tag}/RelP/L{l}/q", relp.dense_q[l], plain.dense_q[l])
            topk_checks(tag, l, plain.routes[l], hf.routes[l])
    hf, plain = G["batch|hf"], G["batch|plain"]
    compare("batch/HF/metric", plain.delta, hf.delta)
    compare("batch/HF/final_residual", plain.residual_final, hf.residual_final)
    for l in range(m.L):
        compare(f"batch/HF/L{l}/out", plain.outs[l], hf.outs[l])
        compare(f"batch/HF/L{l}/q", plain.dense_q[l], hf.dense_q[l])
        topk_checks("batch", l, plain.routes[l], hf.routes[l])
    for lp in patch_layers:
        compare(f"patch/L{lp}/all_q_vs_out", G[f"patch{lp}|q"].delta, G[f"patch{lp}|out"].delta)
        compare(f"patch/L{lp}/functional_vs_native", G[f"patch{lp}|q"].delta, G[f"patch{lp}|hf"].delta)
    # natural-A RelP coefficients exist and are finite (beta for every expert, layer, position, neuron; natural routing
    # of each A input), one reverse stream for the batched A rows
    _, _, finite = relp_backward(m, G["betaA"], len(rows), torch.ones(len(rows), T, dtype=torch.bool), "unpaired",
                                 beta_finite=True)
    for i, row in enumerate(rows):
        checks.append({"name": row["row_id"] + "/natural_A/finite_beta", "passed": bool(finite[i])})
    # diagnostics (not gating): bf16 batch-composition effect, native vs native and functional vs functional
    single = torch.cat([G[tag + "|hf"].delta for _, _, tag in seqs]).cpu()
    single_f = torch.cat([G[tag + "|plain"].delta for _, _, tag in seqs]).cpu()
    diag = {"metric_native_batched_vs_single_max_abs": float((G["batch|hf"].delta.cpu() - single).abs().max()),
            "metric_functional_batched_vs_single_max_abs": float((G["batch|plain"].delta.cpu() - single_f).abs().max()),
            "metric_abs_mean": float(single.abs().mean()),
            "route_sets_native_batched_vs_single_changed": int(sum(
                int((G["batch|hf"].routes[l][0][i].sort(-1)[0] != G[tag + "|hf"].routes[l][0][0].sort(-1)[0]).any(-1).sum())
                for l in range(m.L) for i, (_, _, tag) in enumerate(seqs)))}
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "tolerances": {"atol": atol, "rtol": rtol, "max_relative_l2": max_relative},
            "rows": [r["row_id"] for r in rows], "diagnostics": diag,
            "scope": "short calibration pairs (B = 1 per sequence, plus one batched group); native q/out/routes, RelP "
                     "on/off forward, finite natural-A beta, complete-position q patch = output patch = native output hook"}


# ---------------------------------------------------------------------------------------------------------------------
def score(m, rows, mode, pool, chunk):
    import torch
    from moetrace.relp_moe_stream import Group, forward_pass, relp_backward
    total = support = None
    cases = []
    for s0 in range(0, len(rows), chunk):
        part = rows[s0:s0 + chunk]
        n = len(part)
        A = ids_tensor(part)
        t = [r["target_id"] for r in part]
        f = [r["foil_id"] for r in part]
        if mode == "paired":
            g = Group("score", torch.cat([A, ids_tensor(part, True)]), t + t, f + f, record_h_in=True)
        else:
            g = Group("score", A, t, f, record_h_in=True)
        forward_pass(m, [g])
        tot, sup, _ = relp_backward(m, g, n, pool_mask(part, pool), mode)
        total = tot.cpu() if total is None else total + tot.cpu()
        support = sup.cpu() if support is None else support | sup.cpu()
        d = g.delta.cpu()
        for i, r in enumerate(part):
            cases.append({"row_id": r["row_id"], "group_id": r["group_id"], "margin_A": float(d[i]),
                          "margin_B": float(d[n + i]) if mode == "paired" else None, "endpoint": "A", "mode": mode})
        print(f"scored {s0 + n}/{len(rows)}", flush=True)
        del g
        torch.cuda.empty_cache()
    return total, support, cases


def calibrate_q(m, rows):
    """relp_sva.calibrate_q: E[q | expert selected] per (layer, position, expert, neuron) over calibration A rows; counts
    from the true top-k sets (not from nonzero activations)."""
    import torch
    from moetrace.relp_moe_stream import Group, forward_pass
    A = ids_tensor(rows)
    n, T = A.shape
    sums = torch.zeros(m.L, T, m.E, m.I, dtype=torch.float64)
    counts = torch.zeros(m.L, T, m.E, dtype=torch.float64)
    g = Group("cal", A, [r["target_id"] for r in rows], [r["foil_id"] for r in rows], pairs=True, keep=set())

    def on_layer(l, w, st):
        rec = st["layer_out"]["cal"]
        q = m.dense_q(rec, n, T).double().sum(0)           # [T, E, I]
        c = torch.zeros(n * T, m.E, dtype=torch.float64, device=m.dev).scatter_(1, rec.topi, 1.0).view(n, T, m.E).sum(0)
        sums[l] += q.cpu()
        counts[l] += c.cpu()
    forward_pass(m, [g], on_layer=on_layer)
    return (sums / counts.clamp_min(1)[..., None]).float(), counts


def evaluate(m, rows, jobs, pool, chunk_rows):
    """relp_sva.evaluate_case for every held-out row and every job, batched in one streamed pass.

    jobs: list of dicts {masks: {Label: unit-level bool mask}, unit, baseline, qmean}; unit-level shapes: neuron [L, T, E, I],
    expert_position [L, T, E], expert_all_positions [L, E]. Returns one record list per job (protocol order)."""
    import torch
    from moetrace import relp_sva as S
    from moetrace.relp_moe_stream import Group, Override, forward_pass
    n = len(rows)
    A, Bb = ids_tensor(rows), ids_tensor(rows, True)
    T = A.shape[1]
    t = [r["target_id"] for r in rows]
    f = [r["foil_id"] for r in rows]
    pm = pool_mask(rows, pool)
    if not bool((pm == pm[0]).all()):
        raise ValueError("pool positions differ across rows; the batched evaluation assumes one template")
    p = pm[0].to(m.dev)[:, None, None]                      # [T, 1, 1]
    labels = [list(job["masks"]) for job in jobs]
    # intervened rows in (job, label, kind) blocks of all held-out rows, so that one Override handles a whole block;
    # records are re-assembled in the protocol's order (per held-out row: empty, then label x kind) at the end
    spec = []                                               # (job, held-out row, label index or -1, kind)
    for ji in range(len(jobs)):
        spec += [(ji, i, -1, "empty") for i in range(n)]
        for li in range(len(labels[ji])):
            for kind in KINDS:
                spec += [(ji, i, li, kind) for i in range(n)]
    where = {key: j for j, key in enumerate(spec)}
    chunk_rows = max(1, chunk_rows // n) * n              # whole blocks per chunk
    base = Group("base", torch.cat([A, Bb]), t + t, f + f, pairs=True, keep=set(), record_route=True)
    dev_masks = {}

    def unit_mask(ji, li, l):
        key = (ji, li, l)
        if key not in dev_masks:
            x = jobs[ji]["masks"][labels[ji][li]]
            unit = jobs[ji]["unit"]
            if unit == "neuron":
                mk = x[l]
            elif unit == "expert_position":
                mk = x[l][..., None].expand(T, m.E, m.I)
            else:
                mk = x[l][None, :, None].expand(T, m.E, m.I)
            dev_masks[key] = mk.to(m.dev)
        return dev_masks[key]

    groups = []
    for c0 in range(0, len(spec), chunk_rows):
        sub = spec[c0:c0 + chunk_rows]
        ids = torch.stack([(Bb if kind == "denoise" else A)[i] for _, i, _, kind in sub])
        tt = [t[i] for _, i, _, _ in sub]
        ff = [f[i] for _, i, _, _ in sub]
        keys = {}                                           # rows of this chunk per (job, label, kind)
        for j, (ji, i, li, kind) in enumerate(sub):
            keys.setdefault((ji, li, kind), []).append((j, i))

        def ofn(l, w, st, keys=keys, nsub=len(sub)):
            for key in [k_ for k_ in dev_masks if k_[2] != l]:  # device masks of earlier layers are no longer needed
                del dev_masks[key]
            out = []
            src = st["layer_out"]["base"]
            for (ji, li, kind), js in keys.items():
                baseline = jobs[ji]["baseline"]
                rows_mask = torch.zeros(nsub, dtype=torch.bool, device=m.dev)
                rows_mask[torch.tensor([j for j, _ in js], device=m.dev)] = True
                if kind == "empty":
                    Sl = p.expand(T, m.E, m.I)
                else:
                    M = unit_mask(ji, li, l)
                    Sl = (~M & p) if kind == "keep" else (M & p)
                if kind in ("empty", "remove", "keep"):
                    mq = jobs[ji]["qmean"][l].to(m.dev, m.dtype) if baseline == "mean_q_routed" else None
                    out.append(Override(rows=rows_mask, S=Sl, value="mean" if baseline == "mean_q_routed" else "zero",
                                        mean_q=mq))
                else:
                    # noise: on A, values = B's q; denoise: on B, values = A's q (base rows: A = i, B = n + i)
                    srows = torch.tensor([(n + i) if kind == "noise" else i for _, i in js], device=m.dev)
                    out.append(Override(rows=rows_mask, S=Sl, value="source", src=src, src_rows=srows))
            return out
        groups.append(Group(f"iv{c0}", ids, tt, ff, override_fn=ofn, record_route=True))
    forward_pass(m, [base] + groups)
    margins = torch.cat([g.delta.cpu() for g in groups])
    # route changes vs the unpatched run of the same input (B for denoise, A otherwise), all records at once
    before = torch.tensor([(n + i) if kind == "denoise" else i for _, i, _, kind in spec])
    after_routes = [(torch.cat([g.routes[l][0] for g in groups]), torch.cat([g.routes[l][1] for g in groups]))
                    for l in range(m.L)]
    changes = route_changes([(base.routes[l][0][before], base.routes[l][1][before]) for l in range(m.L)], after_routes, m.E)
    bd = base.delta.cpu()
    out = []
    for ji in range(len(jobs)):
        records = []
        for i, row in enumerate(rows):
            common = {"row_id": row["row_id"], "group_id": row["group_id"], "margin_A": float(bd[i]),
                      "margin_B": float(bd[n + i]), "margin_empty": float(margins[where[(ji, i, -1, "empty")]])}
            for li, label in enumerate(labels[ji]):
                for kind in KINDS:
                    j = where[(ji, i, li, kind)]
                    records.append({**common, **label._asdict(), "intervention": kind, "margin": float(margins[j]),
                                    "topk_changed_layer_positions": int(changes[0][j]), "routing_weight_l1": float(changes[1][j])})
        out.append(records)
    return out


def route_changes(before, after, E):
    """relp_sva.route_change for many rows at once. before / after: per layer (topi [R, T, k], topw [R, T, k]).
    Returns (changed top-k sets [R] int, routing-weight L1 [R] float64), with the protocol's arithmetic: per layer the
    L1 of the scattered weight vectors is summed in the weights' dtype, then accumulated across layers in float64."""
    import torch
    R = before[0][0].shape[0]
    sets = torch.zeros(R, dtype=torch.long)
    l1 = torch.zeros(R, dtype=torch.float64)
    for (ai, aw), (bi, bw) in zip(before, after):
        sets += (ai.sort(-1)[0] != bi.sort(-1)[0]).any(-1).sum(-1)
        ga = aw.new_zeros(*ai.shape[:2], E).scatter(2, ai, aw)
        gb = bw.new_zeros(*bi.shape[:2], E).scatter(2, bi, bw)
        l1 += (ga - gb).abs().sum(dim=(1, 2)).double()
    return sets, l1


# ---------------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("verify", "score", "evaluate", "evaluate-multi"))
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--task", choices=("simple", "rc", "within_rc", "nounpp"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--verification", type=Path, help="Passed verify/verification.json from this installation")
    p.add_argument("--score-dir", type=Path)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--mode", choices=("paired", "unpaired"), default="paired")
    p.add_argument("--pool", choices=("non_special", "all", "final"), default="non_special")
    p.add_argument("--unit", choices=("neuron", "expert_position", "expert_all_positions"), default="neuron")
    p.add_argument("--baseline", choices=("zero", "mean_q_routed"), default="zero", help="Lock at score time")
    p.add_argument("--neuron-budgets", nargs="+", type=int, default=[1, 5, 10, 50, 100, 200, 500, 1000])
    p.add_argument("--expert-budgets", nargs="+", type=int, default=[1, 2, 5, 10, 20, 50])
    p.add_argument("--random-seeds", nargs="+", type=int, default=[101, 102, 103])
    p.add_argument("--verify-cases", type=int, default=4)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--max-relative", type=float, default=1e-3)
    p.add_argument("--score-chunk", type=int, default=300, help="discovery rows per streamed forward+backward")
    p.add_argument("--eval-chunk", type=int, default=1040, help="intervened rows per functional group")
    p.add_argument("--reuse-scores", type=Path, help="score --baseline mean_q_routed: copy the (baseline-independent) "
                   "discovery scores of this complete zero-baseline score run; only the calibration mean is computed")
    p.add_argument("--score-dirs", type=Path, nargs="+", help="evaluate-multi: locked score runs")
    p.add_argument("--units", nargs="+", choices=("neuron", "expert_position", "expert_all_positions"),
                   help="evaluate-multi: units; outputs <output-dir>/<split>_<unit>_<baseline>/")
    a = p.parse_args()
    if a.output_dir.exists():
        p.error("Output directory exists; do not overwrite runs")
    if any(n <= 0 for n in a.neuron_budgets + a.expert_budgets) or a.verify_cases < 1:
        p.error("Budgets and verify-cases must be positive")
    import torch
    import pandas as pd
    from moetrace import relp_sva as S
    from moetrace.relp_moe_stream import StreamRelP
    meta, calibration = S.read_split(a.manifest_dir, a.task, "calibration")
    S.check_tokenizer_files(a.model_dir, meta)
    config = json.loads((a.model_dir / "config.json").read_text())
    if config.get("model_type") not in ("qwen3_moe", "mixtral", "olmoe") or config.get("hidden_act", "silu") != "silu":
        p.error("Unsupported model/activation")
    if config.get("clip_qkv") is not None or any(config.get(k) for k in (
            "attention_bias", "mlp_bias", "num_shared_experts", "shared_expert_intermediate_size")):
        p.error("Unsupported clipping/bias/shared-expert configuration")
    length = len(calibration[0]["input_ids"])
    for k in ("sliding_window", "max_position_embeddings"):
        bound = config.get(k)
        if isinstance(bound, int) and bound > 0 and length > bound:
            p.error("Sequence exceeds " + k)
    m = StreamRelP(str(a.model_dir.resolve()), device=a.device, dtype=getattr(torch, a.dtype))
    ident = identity(m, a.model_dir, S)
    common = {"utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "task": a.task,
              "manifest_sha256": S.sha(a.manifest_dir / "manifest.json"), "identity": ident,
              "model_revision": meta["model"]["revision"], "source_revision": meta["source_revision"],
              "script_sha256": S.sha(__file__), "status": "running"}
    if a.command != "verify":
        if a.verification is None:
            p.error("--verification is required; run native checks first")
        v = json.loads(a.verification.read_text())
        if not v.get("passed") or v["identity"] != ident or v["task"] != a.task or v["manifest_sha256"] != common["manifest_sha256"]:
            p.error("Verification is failed, stale, or for another model/task/manifest/runtime")
        common["verification_sha256"] = S.sha(a.verification)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    S.write_json(a.output_dir / "run_meta.json", common)
    if a.command == "verify":
        report = verify(m, calibration[:a.verify_cases], a.atol, a.rtol, a.max_relative)
        report.update(common)
        report["passed"] = all(c["passed"] for c in report["checks"])
        report["status"] = "complete" if report["passed"] else "failed"
        S.write_json(a.output_dir / "verification.json", report)
        S.write_json(a.output_dir / "run_meta.json", {**common, "status": report["status"]})
        print(json.dumps({"passed": report["passed"], "n_checks": len(report["checks"]),
                          "failed_checks": [c for c in report["checks"] if not c["passed"]][:20],
                          "diagnostics": report["diagnostics"]}))
        if not report["passed"]:
            raise SystemExit(1)
        return
    if a.command == "score":
        _, rows = S.read_split(a.manifest_dir, a.task, "discovery")
        if a.reuse_scores is not None:
            if a.baseline != "mean_q_routed":
                p.error("--reuse-scores is only for the mean_q_routed score run")
            rmeta = json.loads((a.reuse_scores / "run_meta.json").read_text())
            rplan = json.loads((a.reuse_scores / "selection_plan.json").read_text())
            if rmeta["status"] != "complete" or any(rmeta[k] != common[k] for k in ("identity", "task", "manifest_sha256")) \
                    or (rplan["mode"], rplan["pool"]) != (a.mode, a.pool) or S.sha(a.reuse_scores / "aggregate.pt") != rplan["aggregate_sha256"]:
                p.error("--reuse-scores run is incomplete or for another model/task/manifest/runtime/mode/pool")
            prev = torch.load(a.reuse_scores / "aggregate.pt", map_location="cpu", weights_only=True)
            total, support = prev["mean"] * prev["n"], prev["support"]
            cases = [json.loads(x) for x in (a.reuse_scores / "cases.jsonl").read_text().splitlines()]
            common["scores_reused_from"] = {"dir": str(a.reuse_scores), "aggregate_sha256": rplan["aggregate_sha256"]}
        else:
            total, support, cases = score(m, rows, a.mode, a.pool, a.score_chunk)
        with (a.output_dir / "cases.jsonl").open("x") as stream:
            for record in cases:
                stream.write(json.dumps(record) + "\n")
        mean = prev["mean"] if a.reuse_scores is not None else total / len(rows)
        payload = {"mean": mean, "support": support, "n": len(rows)}
        if a.baseline == "mean_q_routed":
            payload["qmean"], payload["calibration_counts"] = calibrate_q(m, calibration)
        torch.save(payload, a.output_dir / "aggregate.pt")
        lists = S.rankings(mean, support, max(a.neuron_budgets + [1000]))
        for unit, values in lists.items():
            S.write_json(a.output_dir / f"{unit}_ranking.json", values)
        pd.DataFrame(cases).to_parquet(a.output_dir / "cases.parquet", index=False)
        plan = {"mode": a.mode, "endpoint": "A", "pool": a.pool, "baseline": a.baseline,
                "neuron_budgets": sorted(set(a.neuron_budgets)), "expert_budgets": sorted(set(a.expert_budgets)),
                "random_seeds": a.random_seeds, "ranking": "abs(mean(signed score))",
                "expert_aggregation": "sum over all neurons before ranking",
                "mean_baseline": "E[q | expert selected], per layer-position-expert on calibration A only; applied to live-routed experts",
                "discovery_sha256": meta["tasks"][a.task]["files_sha256"]["discovery"],
                "n_discovery": len(rows), "aggregate_sha256": S.sha(a.output_dir / "aggregate.pt")}
        S.write_json(a.output_dir / "selection_plan.json", plan)
        S.write_json(a.output_dir / "run_meta.json", {**common, "status": "complete",
                     "selection_plan_sha256": S.sha(a.output_dir / "selection_plan.json")})
        print("Saved complete signed means and independent neuron/expert rankings.", flush=True)
        return
    # ---- evaluate / evaluate-multi: fixed lists locked at score time, held-out split, one streamed pass
    if a.command == "evaluate":
        if a.score_dir is None:
            p.error("--score-dir required for evaluation")
        targets = [(a.score_dir, a.unit, a.output_dir)]
    else:
        if not a.score_dirs or not a.units:
            p.error("evaluate-multi needs --score-dirs and --units")
        targets = []
        for sd in a.score_dirs:
            base_plan = json.loads((sd / "selection_plan.json").read_text())
            for unit in a.units:
                targets.append((sd, unit, a.output_dir / f"{a.split}_{unit}_{base_plan['baseline']}"))
    jobs, metas = [], []
    for sd, unit, out_dir in targets:
        job, plan = load_job(S, torch, common, sd, unit)
        jobs.append(job)
        metas.append((sd, unit, out_dir, plan))
    pools = {pl["pool"] for *_, pl in metas}
    if len(pools) != 1:
        raise ValueError("all evaluations of one pass must share the score-time position pool")
    _, rows = S.read_split(a.manifest_dir, a.task, a.split)
    results = evaluate(m, rows, jobs, pools.pop(), a.eval_chunk)
    for (sd, unit, out_dir, plan), records in zip(metas, results):
        out_dir.mkdir(parents=True, exist_ok=out_dir == a.output_dir)  # evaluate: created (empty) above; multi: new
        with (out_dir / "interventions.jsonl").open("x") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        pd.DataFrame(records).to_parquet(out_dir / "interventions.parquet", index=False)
        S.write_json(out_dir / "summary.json", S.summarize(records))
        S.write_json(out_dir / "run_meta.json", {**common, "status": "complete", "split": a.split, "unit": unit,
                     "score_dir": str(sd), "selection_plan": plan, "n": len(rows),
                     "batched_with": [str(o) for *_, o, _ in metas],
                     "evaluation": "streamed HF-exact functional forward after native acceptance; downstream natural routing; "
                                   "no rerouting/renormalization at patched q",
                     "normalization": "ratio of means; keep=faithfulness, remove=remaining(completeness), noise=lost paired difference, denoise=recovery"})
    if a.command == "evaluate-multi":
        S.write_json(a.output_dir / "run_meta.json", {**common, "status": "complete", "split": a.split,
                     "outputs": [str(o) for *_, o, _ in metas]})


def load_job(S, torch, common, score_dir, unit):
    """The selection of relp_sva_run.py evaluate: verify the locked discovery run, rank, budgets, matched random sets."""
    scoremeta = json.loads((score_dir / "run_meta.json").read_text())
    for key in ("identity", "task", "manifest_sha256"):
        if scoremeta[key] != common[key]:
            raise ValueError("Score run mismatch: " + key)
    if scoremeta["status"] != "complete":
        raise ValueError("Incomplete discovery run")
    plan = json.loads((score_dir / "selection_plan.json").read_text())
    if S.sha(score_dir / "selection_plan.json") != scoremeta["selection_plan_sha256"] or \
            S.sha(score_dir / "aggregate.pt") != plan["aggregate_sha256"]:
        raise ValueError("Discovery output changed after selection was locked")
    payload = torch.load(score_dir / "aggregate.pt", map_location="cpu", weights_only=True)
    mean, support = payload["mean"], payload["support"]
    if unit == "neuron":
        values, eligible = mean, support
    elif unit == "expert_position":
        values, eligible = mean.sum(-1), support.any(-1)
    else:
        values, eligible = mean.sum(-1).sum(1), support.any(-1).any(1)
    budgets = plan["neuron_budgets" if unit == "neuron" else "expert_budgets"]
    Label = namedtuple("Label", "ranking seed budget")
    masks = {}
    order = S.rank_indices(values, eligible, max(budgets))
    if len(order) < max(budgets):
        raise ValueError("Budget exceeds eligible units; revise discovery plan on development data")
    for n in budgets:  # identical to relp_sva_run.py; masks stay at the unit level (expanded per layer on the GPU)
        chosen = torch.zeros_like(eligible)
        chosen.reshape(-1)[order[:n]] = True
        masks[Label("relp", 0, n)] = chosen
        for seed in plan["random_seeds"]:
            if unit == "expert_all_positions":
                random = S.random_matched(chosen[:, None, :], eligible[:, None, :], torch.Generator().manual_seed(seed + n))[:, 0, :]
            else:
                random = S.random_matched(chosen, eligible, torch.Generator().manual_seed(seed + n))
            masks[Label("random", seed, n)] = random
    return {"masks": masks, "unit": unit, "baseline": plan["baseline"], "qmean": payload.get("qmean")}, plan


if __name__ == "__main__":
    main()
