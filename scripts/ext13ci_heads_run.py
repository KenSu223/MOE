"""ext13ci step 1 (Phase 5, p5-circuit): single-head patches at every (layer, head) + attn_layer references + patch-free
head / expert DLA + final-position attention mass on position classes, on one Phase-5 condition's validation rows.

= scripts/ext10_heads_run.py with the Phase-5 registry (moetrace/ext13ci.py) and one addition: the prefill rows also
record DiagSpec.attn_final (final position's attention distribution over positions, every layer and head), reduced per
row to the mass on the position classes of moetrace.ext13ci.case_positions for the clean and the corrupted run
(head_attn_pNN.npz: mass [rows, 2, L, nH, classes] float16, row_ids, classes, n_pos [rows, classes]).

One engine pass per chunk of validation rows. Per row: one attn_head spawn per (layer, head) (parent = corrupted row,
source = clean row) and one attn_layer spawn per layer. Outputs in results/p5_<cond>_<proto>_ci<suffix>/ with ext10's
step-1 formats: head_rows_pNN, head_prefill_pNN, head_dla_pNN, expert_dla_pNN (.parquet), head_attn_pNN.npz,
heads_state.json, run_meta.json. Resumable (chunks listed as done are skipped; a raw_pNN.pt left by a crash after the GPU
pass is post-processed on the CPU).

Usage: python scripts/ext13ci_heads_run.py <cond> <proto> [--rows-per-pass N] [--max-passes 99] [--limit N]
       [--wf-chunk 2048] [--out-suffix _smoke] [--dry-run]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS
from moetrace import ext10_circuit as C
from moetrace import ext13ci as E13


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_meta(od, upd):
    p = os.path.join(od, "run_meta.json")
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.update(upd)
    with open(p, "w") as f:
        json.dump(cur, f, indent=1, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond")
    ap.add_argument("proto", choices=list(E13.PROTOS))
    ap.add_argument("--rows-per-pass", type=int, default=0, help="0 = model default (Qwen3 160, Mixtral 110)")
    ap.add_argument("--max-passes", type=int, default=99)
    ap.add_argument("--limit", type=int, default=0, help="only the first N validation rows (smoke)")
    ap.add_argument("--wf-chunk", type=int, default=2048)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    e = E13.entry(args.cond, args.proto)
    model = e["model"]
    od = E13.step1_dir(args.cond, args.proto, args.out_suffix)
    os.makedirs(od, exist_ok=True)
    t = E13.load_task(args.cond, args.proto)
    vr = E13.val_rows(t)
    if args.limit:
        vr = vr[: args.limit]
    rpp = args.rows_per_pass or (160 if model == "qwen3" else 110)
    classes = E13.class_names(t.pairs)
    sp_path = os.path.join(od, "heads_state.json")
    if os.path.exists(sp_path):
        st = json.load(open(sp_path))
        log(f"resuming: {len(st['done'])}/{len(st['chunks'])} chunks done")
    else:
        st = {"chunks": C.row_chunks(t, vr, rpp), "done": [], "notes": []}
        json.dump(st, open(sp_path, "w"))
        Ts = [len(x) for x in t.cases.clean_ids]
        write_meta(od, {"cond": args.cond, "proto": args.proto, "model": model, "repo": MODELS[model]["repo"], "src_run": e["run"],
                        "pairs": e["pairs"], "case_sets": e["case_sets"], "family": e["family"], "cluster": t.cluster_src,
                        "special_tokens": t.special_tokens, "n_val_rows": len(vr), "n_val_units": int(t.cases.set_index("case_id").unit.loc[[int(t.rows.case_id.iloc[r]) for r in vr]].nunique()),
                        "prompt_tokens_median": float(np.median(Ts)), "single_expert_coverage": t.single_coverage,
                        "attn_classes": classes, "rows_per_pass": rpp, "n_chunks": len(st["chunks"]), "wf_chunk": args.wf_chunk,
                        "step": "step1 single heads + DLA + attention mass", "agent": "p5-circuit",
                        "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "complete": False})
    from moetrace.arch import load_spec
    spec, _ = load_spec(MODELS[model]["repo"])
    L, nH, D = spec.n_layers, spec.n_heads, spec.head_dim
    todo = [i for i in range(len(st["chunks"])) if i not in st["done"]]
    log(f"{args.cond}/{args.proto}: {len(vr)} validation rows, {len(st['chunks'])} chunks ({len(todo)} to do), L={L} nH={nH}; "
        f"spawn rows per row {L * nH + L}; attention classes {classes}")
    if args.dry_run:
        for i in todo:
            n = len(st["chunks"][i])
            log(f"  chunk {i}: {n} rows, {n * (L * nH + L)} spawn rows")
        return
    case_rows = t.cases.set_index("case_id")

    def chunk_layout(i):
        rows = st["chunks"][i]
        rdf = t.rows.set_index("row_id").loc[rows]
        cases = list(dict.fromkeys(int(x) for x in rdf.case_id))
        return rows, rdf, cases, {cc: j for j, cc in enumerate(cases)}

    def finish(i, raw, Wacc):
        rows, rdf, cases, ci_of = chunk_layout(i)
        post(i, raw, Wacc, rows, rdf, cases, ci_of)
        rp = os.path.join(od, f"raw_p{i:02d}.pt")
        if os.path.exists(rp):
            os.remove(rp)

    def post(i, raw, Wacc, rows, rdf, cases, ci_of):
        nCc, nR = len(cases), len(rows)
        d = raw["delta"]
        rms = raw["final_rms"]
        row_case = np.array([int(rdf.loc[r].case_id) for r in rows])
        row_slot = np.array([int(rdf.loc[r].slot) for r in rows])
        cidx = np.array([ci_of[cc] for cc in row_case])
        pf = pd.DataFrame(dict(chunk=i, idx=np.arange(nCc + nR), kind=["clean"] * nCc + ["corrupt"] * nR,
                               case_id=np.concatenate([np.array(cases), row_case]), row_id=np.concatenate([np.full(nCc, -1), rows]),
                               slot=np.concatenate([np.full(nCc, -1), row_slot]), delta=d, top1=raw["top1"], final_rms=rms))
        pf.to_parquet(os.path.join(od, f"head_prefill_p{i:02d}.parquet"), index=False)
        mr = np.repeat(np.arange(nR), L * nH + L)
        ml = np.tile(np.concatenate([np.repeat(np.arange(L), nH), np.arange(L)]), nR)
        mh = np.tile(np.concatenate([np.tile(np.arange(nH), L), np.full(L, -1)]), nR)
        assert len(mr) == len(raw["sp_delta"])
        hd = pd.DataFrame(dict(row_id=np.array(rows)[mr].astype(np.int32), case_id=row_case[mr].astype(np.int32),
                               slot=row_slot[mr].astype(np.int16), layer=ml.astype(np.int16), head=mh.astype(np.int16),
                               delta=raw["sp_delta"], delta_clean=d[cidx][mr], delta_corrupt=d[nCc:][mr], vnorm=raw["sp_vnorm"]))
        hd["kind"] = np.where(hd["head"].values >= 0, "attn_head", "attn_layer")
        hd["rescue"] = (hd.delta - hd.delta_corrupt).astype(np.float32)
        hd.to_parquet(os.path.join(od, f"head_rows_p{i:02d}.parquet"), index=False)
        # ---- head DLA (final norm frozen at the corrupted run)
        dev = Wacc["device"]
        tid = torch.tensor([int(case_rows.loc[cc].true_id) for cc in row_case], device=dev)
        fid = torch.tensor([int(case_rows.loc[cc].foil_id) for cc in row_case], device=dev)
        U = Wacc["gamma"][None, :] * (Wacc["head"][tid].float() - Wacc["head"][fid].float())  # [nR, H]
        rms_j = torch.tensor(rms[nCc:], device=dev, dtype=torch.float32)
        ci_t = torch.tensor(cidx, device=dev)
        cr_t = torch.arange(nCc, nCc + nR, device=dev)
        out = np.zeros((nR, L, nH), dtype=np.float32)
        for l in range(L):
            Wo = Wacc["wo"](l)  # fp32 [H, nH*D]
            A = (U @ Wo).view(nR, nH, D)
            Hl = raw["heads"][l].to(dev)
            dH = Hl[ci_t].float() - Hl[cr_t].float()
            out[:, l] = ((A * dH).sum(-1) / rms_j[:, None]).cpu().numpy()
            del Wo, A, Hl, dH
        rr, ll, hh = np.meshgrid(np.arange(nR), np.arange(L), np.arange(nH), indexing="ij")
        hdla = pd.DataFrame(dict(row_id=np.array(rows)[rr.ravel()].astype(np.int32), layer=ll.ravel().astype(np.int16),
                                 head=hh.ravel().astype(np.int16), dla=out.ravel()))
        hdla.to_parquet(os.path.join(od, f"head_dla_p{i:02d}.parquet"), index=False)
        # ---- expert DLA (ext8 formula), candidates = the base run's clean-active pairs
        cd = raw["contrib_dla"]
        ri = raw["route_idx"]
        er = []
        n_flip = 0
        for j, r in enumerate(rows):
            ci, cr = cidx[j], nCc + j
            for (l, e_) in t.cand[int(row_case[j])]:
                mc = ri[l, ci] == e_
                mj = ri[l, cr] == e_
                pc = float(cd[l, ci][mc][0]) if mc.any() else 0.0
                pj = float(cd[l, cr][mj][0]) if mj.any() else 0.0
                n_flip += int(not mc.any())
                er.append((int(r), int(l), int(e_), (pc - pj) / float(rms[cr]), bool(mc.any())))
        edla = pd.DataFrame(er, columns=["row_id", "layer", "expert", "dla", "routed_clean"])
        edla.to_parquet(os.path.join(od, f"expert_dla_p{i:02d}.parquet"), index=False)
        # ---- attention mass of the final position on the position classes (clean and corrupted prefill rows)
        attn = raw["attn"]  # fp16 [L, B, nH, T]
        T = attn.shape[-1]
        Cn = len(classes)
        ind = np.zeros((nR, Cn, T), dtype=np.float32)
        npos = np.zeros((nR, Cn), dtype=np.int16)
        for j, r in enumerate(rows):
            pos = E13.case_positions(t, int(row_case[j]))
            Tr = len(case_rows.loc[int(row_case[j])].clean_ids)
            for k, cl in enumerate(classes):
                ps = sorted({p for p in pos.get(cl, []) if 0 <= p < Tr})
                ind[j, k, ps] = 1.0
                npos[j, k] = len(ps)
        mass = np.zeros((nR, 2, L, nH, Cn), dtype=np.float16)
        It = torch.tensor(ind)
        for src, idx in ((0, cidx), (1, np.arange(nCc, nCc + nR))):
            A_ = torch.tensor(attn[:, idx].astype(np.float32))  # [L, nR, nH, T]
            mass[:, src] = torch.einsum("lrht,rct->rlhc", A_, It).numpy().astype(np.float16)
            del A_
        np.savez_compressed(os.path.join(od, f"head_attn_p{i:02d}.npz"), mass=mass, row_ids=np.array(rows, dtype=np.int32),
                            classes=np.array(classes), n_pos=npos)
        # ---- checks and state
        H = hd[hd.kind == "attn_head"]
        A_l = hd[hd.kind == "attn_layer"].set_index(["row_id", "layer"]).rescue
        sumh = H.groupby(["row_id", "layer"]).rescue.sum()
        r_sum = float(np.corrcoef(sumh.loc[A_l.index].values, A_l.values)[0, 1])
        tot = attn[:, :, :, :].astype(np.float32).sum(-1)
        note = {"chunk": i, "rows": nR, "prefill": nCc + nR, "spawn_rows": len(hd), "pass_s": raw.get("total_s"),
                "peak_GB": raw.get("peak_GB"), "mean_drop": float(np.mean(d[cidx] - d[nCc:])), "r_sum_heads_vs_attn_layer": r_sum,
                "head_dla_sum_mean": float(out.sum((1, 2)).mean()), "candidates_not_routed_clean": n_flip,
                "attn_row_sum_minmax": [float(tot.min()), float(tot.max())],
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        st["done"].append(i)
        st["notes"].append(note)
        json.dump(st, open(sp_path, "w"))
        log(f"chunk {i} done: {note}")

    raw_todo = [i for i in todo if os.path.exists(os.path.join(od, f"raw_p{i:02d}.pt"))]
    if raw_todo:
        from moetrace.weights import CheckpointStore
        from moetrace.arch import load_spec as _ls
        _, snap = _ls(MODELS[model]["repo"])
        store = CheckpointStore(snap, spec, device="cpu")
        g = store.load_globals()
        Wc = {"device": "cpu", "gamma": g["norm"].float(), "head": g["head"],
              "wo": lambda l: store.gpu_tensor(spec.k_o.format(l=l)).float()}
        for i in raw_todo:
            log(f"post-processing chunk {i} from raw (CPU)")
            finish(i, torch.load(os.path.join(od, f"raw_p{i:02d}.pt"), weights_only=False), Wc)
        todo = [i for i in todo if i not in raw_todo]
    from moetrace import engine as EN
    eng = EN.Engine(MODELS[model]["repo"]) if todo else None
    n_run = 0
    todo = list(todo)
    while todo:
        i = todo.pop(0)
        if n_run >= args.max_passes:
            break
        rows, rdf, cases, ci_of = chunk_layout(i)
        nCc = len(cases)
        pre = []
        for cc in cases:
            cr_ = case_rows.loc[cc]
            pre.append(EN.PrefillSpec(list(cr_.clean_ids), int(cr_.true_id), int(cr_.foil_id), clean_ref=ci_of[cc]))
        for r in rows:
            cc = int(rdf.loc[r].case_id)
            cr_ = case_rows.loc[cc]
            assert len(rdf.loc[r].corrupt_ids) == len(cr_.clean_ids)
            pre.append(EN.PrefillSpec(list(rdf.loc[r].corrupt_ids), int(cr_.true_id), int(cr_.foil_id), clean_ref=ci_of[cc]))
        sp = []
        for j, r in enumerate(rows):
            ci, cr = ci_of[int(rdf.loc[r].case_id)], nCc + j
            for l in range(L):
                for h in range(nH):
                    sp.append(EN.SpawnSpec(l, cr, ci, "attn_head", expert=h))
            for l in range(L):
                sp.append(EN.SpawnSpec(l, cr, ci, "attn_layer"))
        diag = EN.DiagSpec(attn_heads_final=tuple(range(L)), contrib_dla=True, attn_final=True)
        log(f"pass (chunk {i}): {len(pre)} prefill rows ({nCc} clean), {len(sp)} spawn rows")
        import gc; gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            res = eng.run(pre, sp, record_routing=True, log=log, wf_chunk=args.wf_chunk, diag=diag)
        except torch.OutOfMemoryError:
            del pre, sp
            torch.cuda.empty_cache()
            half = C.row_chunks(t, rows, max(1, (len(rows) + 1) // 2))
            if len(half) < 2:
                raise
            st["chunks"][i] = half[0]
            st["chunks"] += half[1:]
            st["notes"].append({"oom_split": i, "sizes": [len(h) for h in half]})
            json.dump(st, open(sp_path, "w"))
            log(f"OOM on chunk {i} ({len(rows)} rows): split into {[len(h) for h in half]}")
            todo = [i] + list(range(len(st["chunks"]) - len(half) + 1, len(st["chunks"]))) + todo
            continue
        dg = res.extra["diag"]
        raw = {"delta": res.delta, "top1": res.top1, "sp_delta": res.sp_logit_true - res.sp_logit_foil, "sp_vnorm": res.sp_vnorm,
               "route_idx": res.route_idx, "heads": dg["attn_heads_final"], "contrib_dla": dg["contrib_dla"],
               "final_rms": dg["final_rms"], "attn": dg["attn_final"], "total_s": res.extra.get("total_s"),
               "peak_GB": torch.cuda.max_memory_allocated() / 2**30}
        torch.save(raw, os.path.join(od, f"raw_p{i:02d}.pt"))
        del res, dg
        torch.cuda.empty_cache()
        Wg = {"device": eng.device, "gamma": eng.g["norm"].float(), "head": eng.g["head"],
              "wo": lambda l: eng.store.gpu_tensor(spec.k_o.format(l=l)).float()}
        finish(i, raw, Wg)
        del raw
        n_run += 1
    if len(st["done"]) == len(st["chunks"]):
        write_meta(od, {"complete": True, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pass_notes": st["notes"]})
        log("ALL CHUNKS COMPLETE")


if __name__ == "__main__":
    main()
