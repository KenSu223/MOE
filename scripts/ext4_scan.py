"""Direction 4, calibration / filter + layer-sweep scan over CodeFact items (one GPU job, several engine passes).

For every item that resolves with the model tokenizer under the chosen protocol: clean and subject-noised prefill rows
(sigma = 3 x embed std, noise seed 0 + case_id as in the paper protocol) AND a MoE-block output patch at every layer
(the layer sweep), with the final-position routing recorded and the per-position residual norms (sink diagnostic).
Items are sorted by prefix length and processed in chunks of --chunk items (2 x chunk prefill rows + chunk x L spawn
rows per pass); Parquet files are (re)written after every chunk.

Usage: python scripts/ext4_scan.py <model_key> --out codefact_<model>_<protocol> [--protocol raw|nobos|chat]
                                   [--items data/codefact/items.jsonl] [--chunk 1024] [--max-tokens 160]
Outputs results/<out>/: scan_rows.parquet (case_id, kind clean|noised|layer, layer, logit_true, logit_foil, delta,
delta_full, top1, rescue), scan_routing.parquet (case_id, run, layer, slot, expert, weight, cnorm), scan_cases.parquet
(one row per item with tokenisation, category, delta_clean/noised, drop, strict/relaxed/relative flags, top-1 flags,
sink diagnostics), scan_rejects.parquet, scan_meta.json.
"""
import argparse, json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS, RESULTS
from moetrace.arch import snapshot_dir
from moetrace.data import passes, STRICT, RELAXED
from moetrace.ext4_data import load_items, resolve_items, chat_prefix_ids, CATEGORIES
from moetrace.noise import noise_draw

OK_FLAG = {"qwen3": "ok_qwen3", "qwen3_coder": "ok_qwen3", "qwen3_instruct": "ok_qwen3", "mixtral": "ok_mixtral", "mixtral_instruct": "ok_mixtral"}
DIAG_LAYERS = (1, 5)  # + L//2 and L-4 at run time


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def relative_pass(dc: float, dn: float, frac: float = 0.25, margin: float = 1.0) -> bool:
    return dc >= margin and (dc - dn) >= frac * dc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default="raw", choices=["raw", "nobos", "chat"])
    ap.add_argument("--items", default="/home/ubuntu/MOE/data/codefact/items.jsonl")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--chunk-tokens", type=int, default=80000, help="max items x T per chunk")
    ap.add_argument("--wf-tokens", type=int, default=160000, help="wavefront attention chunk = wf_tokens // T (bounds the [w, nkv, rep, D, T] score operand)")
    ap.add_argument("--max-tokens", type=int, default=160, help="max code-prefix tokens (chat template tokens excluded)")
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N resolved items")
    ap.add_argument("--per-category-max", type=int, default=None, help="scan at most N resolved items per category (items.jsonl order, which is random)")
    args = ap.parse_args()
    m = MODELS[args.model]
    od = os.path.join(RESULTS, args.out)
    os.makedirs(od, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_dir(m["repo"]))
    items = load_items(args.items)
    cats = args.categories.split(",")
    flag = OK_FLAG.get(args.model)
    items = [it for it in items if it["category"] in cats and (flag is None or it.get(flag, True))]
    special = args.protocol != "nobos"
    pre_ids = chat_prefix_ids(tok) if args.protocol == "chat" else None
    cases, rej = resolve_items(items, tok, special_tokens=special, prefix_ids=pre_ids, max_tokens=None)
    # length cap on the code prefix only
    k0 = len(pre_ids) if pre_ids else 0
    too_long = [c for c, cs in cases.items() if len(cs.ids) - k0 > args.max_tokens]
    for c in too_long:
        rej.append({"case_id": c, "item_id": cases[c].item_id, "category": cases[c].category, "reason": "too_long"})
        del cases[c]
    if args.per_category_max:
        keep, cnt = [], {}
        for it in items:
            c = it["case_id"]
            if c in cases and cnt.get(it["category"], 0) < args.per_category_max:
                keep.append(c)
                cnt[it["category"]] = cnt.get(it["category"], 0) + 1
        cases = {c: cases[c] for c in keep}
    ids = sorted(cases, key=lambda c: len(cases[c].ids))
    if args.limit:
        ids = ids[: args.limit]
    log(f"{args.model} protocol={args.protocol}: {len(items)} items with tokenizer flag, {len(cases)} resolved, {len(rej)} rejected "
        f"({pd.Series([r['reason'] for r in rej]).value_counts().to_dict() if rej else {}}); chat prefix {k0} tokens")
    per_cat = pd.Series([cases[c].category for c in ids]).value_counts().to_dict()
    log("resolved per category:", per_cat)
    pd.DataFrame(rej).to_parquet(os.path.join(od, "scan_rejects.parquet"), index=False)

    eng = __import__("moetrace.engine", fromlist=["Engine"]).Engine(m["repo"])
    from moetrace.engine import PrefillSpec, SpawnSpec, DiagSpec
    sigma = args.sigma_mult * eng.embed_std
    Hd, L, k = eng.hidden, eng.spec.n_layers, eng.spec.top_k
    diag_layers = sorted(set(list(DIAG_LAYERS) + [L // 2, L - 4]))
    log(f"embed_std={eng.embed_std:.6f} sigma={sigma:.6f} hidden={Hd} layers={L} top_k={k}; diag layers {diag_layers}")
    # length-sorted chunks bounded by --chunk items AND by --chunk-tokens item-tokens (prefill memory grows with items x T)
    chunks, cur = [], []
    for c in ids:
        T_if = len(cases[c].ids)
        if cur and (len(cur) >= args.chunk or (len(cur) + 1) * T_if > args.chunk_tokens):
            chunks.append(cur)
            cur = []
        cur.append(c)
    if cur:
        chunks.append(cur)
    rows_all, route_all, case_rows, pass_times = [], [], [], []
    meta = {"model": args.model, "repo": m["repo"], "protocol": args.protocol, "special_tokens": special, "chat_prefix_len": k0,
            "chat_prefix_text": tok.decode(pre_ids) if pre_ids else None, "items_file": args.items, "n_items_flagged": len(items),
            "n_resolved": len(cases), "n_scanned": len(ids), "rejects": pd.Series([r["reason"] for r in rej]).value_counts().to_dict() if rej else {},
            "sigma_mult": args.sigma_mult, "sigma": sigma, "embed_std": eng.embed_std, "max_tokens": args.max_tokens, "chunk": args.chunk, "chunk_tokens": args.chunk_tokens, "wf_tokens": args.wf_tokens, "per_category_max": args.per_category_max,
            "n_chunks": len(chunks), "diag_layers": diag_layers, "command": " ".join(sys.argv), "agent": "ext4-codefact",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tokenizer_vocab": len(tok), "pass_times_s": pass_times}

    def write():
        pd.DataFrame(rows_all).to_parquet(os.path.join(od, "scan_rows.parquet"), index=False)
        pd.concat(route_all, ignore_index=True).to_parquet(os.path.join(od, "scan_routing.parquet"), index=False)
        pd.DataFrame(case_rows).to_parquet(os.path.join(od, "scan_cases.parquet"), index=False)
        with open(os.path.join(od, "scan_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)

    for ci, ch in enumerate(chunks):
        n = len(ch)
        pre = [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ch]
        pre += [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id, cases[c].subject_pos,
                            noise_draw(c, len(cases[c].subject_pos), Hd, sigma)) for c in ch]
        spawns, tags = [], []
        for i, c in enumerate(ch):
            for l in range(L):
                spawns.append(SpawnSpec(l, n + i, i, "layer"))
                tags.append((c, l))
        T = max(len(cases[c].ids) for c in ch)
        log(f"chunk {ci + 1}/{len(chunks)}: {n} items, T={T}, {len(pre)} prefill rows, {len(spawns)} spawn rows, wf_chunk={int(max(512, min(8192, args.wf_tokens // T)))}")
        t0 = time.time()
        wf_chunk = int(max(512, min(8192, args.wf_tokens // T)))
        res = eng.run(pre, spawns, record_routing=True, log=log if ci == 0 else None, diag=DiagSpec(resid_norms=True), wf_chunk=wf_chunk)
        pass_times.append(round(time.time() - t0, 1))
        log(f"pass time {pass_times[-1]}s; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        meta.setdefault("peak_mem_gb", []).append(round(torch.cuda.max_memory_allocated() / 2**30, 2))
        torch.cuda.reset_peak_memory_stats()
        d = res.delta
        dfull = res.logit_true_full - res.logit_foil_full
        sd = res.sp_delta
        rn = res.extra["diag"]["resid_norms"]  # [L, B, T]
        for i, c in enumerate(ch):
            cs = cases[c]
            for kind, j in (("clean", i), ("noised", n + i)):
                rows_all.append(dict(case_id=c, kind=kind, layer=-1, sigma_mult=args.sigma_mult, logit_true=float(res.logit_true[j]),
                                     logit_foil=float(res.logit_foil[j]), delta=float(d[j]), delta_full=float(dfull[j]), top1=int(res.top1[j]), rescue=np.nan))
            dc, dn = float(d[i]), float(d[n + i])
            r = cs.to_row()
            r.update(dict(delta_clean=dc, delta_noised=dn, drop=dc - dn, strict=bool(passes(dc, dn, STRICT)), relaxed=bool(passes(dc, dn, RELAXED)),
                          relative25=bool(relative_pass(dc, dn, 0.25)), relative50=bool(relative_pass(dc, dn, 0.50)),
                          logit_true_clean=float(res.logit_true[i]), logit_foil_clean=float(res.logit_foil[i]),
                          logit_true_noised=float(res.logit_true[n + i]), logit_foil_noised=float(res.logit_foil[n + i]),
                          delta_clean_full=float(dfull[i]), delta_noised_full=float(dfull[n + i]),
                          top1_clean=int(res.top1[i]), top1_noised=int(res.top1[n + i]), top1_clean_is_true=bool(res.top1[i] == cs.true_id),
                          top1_noised_is_true=bool(res.top1[n + i] == cs.true_id), top1_clean_is_foil=bool(res.top1[i] == cs.foil_id),
                          top1_clean_tok=str(tok.convert_ids_to_tokens(int(res.top1[i]))), top1_noised_tok=str(tok.convert_ids_to_tokens(int(res.top1[n + i]))),
                          top1_clean_startswith_true=bool(tok.decode([int(res.top1[i])]).startswith(cs.true_str) or res.top1[i] == cs.true_id),
                          top1_clean_startswith_foil=bool(tok.decode([int(res.top1[i])]).startswith(cs.foil_str) or res.top1[i] == cs.foil_id),
                          top1_noised_startswith_true=bool(tok.decode([int(res.top1[n + i])]).startswith(cs.true_str) or res.top1[n + i] == cs.true_id),
                          chunk=ci, T_chunk=T, protocol=args.protocol))
            Tc = len(cs.ids)
            for l in diag_layers:
                v = rn[l, i, :Tc]
                mx = int(np.argmax(v))
                r[f"final_norm_ratio_L{l}"] = float(v[-1] / (v.max() + 1e-6))
                r[f"final_is_max_L{l}"] = bool(mx == Tc - 1)
                r[f"maxnorm_pos_L{l}"] = mx
                r[f"pos0_norm_ratio_L{l}"] = float(v[0] / (np.median(v[1:]) + 1e-6)) if Tc > 1 else np.nan
            case_rows.append(r)
        idx_of = {c: i for i, c in enumerate(ch)}
        for j, (c, l) in enumerate(tags):
            i = idx_of[c]
            rows_all.append(dict(case_id=c, kind="layer", layer=l, sigma_mult=args.sigma_mult, logit_true=float(res.sp_logit_true[j]),
                                 logit_foil=float(res.sp_logit_foil[j]), delta=float(sd[j]), delta_full=np.nan, top1=-1, rescue=float(sd[j] - d[n + i])))
        for run, off in (("clean", 0), ("noised", n)):
            ri = res.route_idx[:, off: off + n]
            rw = res.route_w[:, off: off + n]
            rc = res.route_cnorm[:, off: off + n]
            Ls, Ns, Ks = np.meshgrid(np.arange(L), np.arange(n), np.arange(k), indexing="ij")
            route_all.append(pd.DataFrame(dict(case_id=np.array(ch)[Ns.ravel()], run=run, layer=Ls.ravel().astype(np.int16),
                                               slot=Ks.ravel().astype(np.int8), expert=ri.ravel().astype(np.int16), weight=rw.ravel().astype(np.float32),
                                               cnorm=rc.ravel().astype(np.float32))))
        write()
        ct = pd.DataFrame(case_rows)
        summ = ct.groupby("category").agg(n=("case_id", "size"), strict=("strict", "sum"), relaxed=("relaxed", "sum"), rel25=("relative25", "sum"),
                                          top1=("top1_clean_is_true", "mean"), dclean=("delta_clean", "median"), drop=("drop", "median"))
        log(f"chunk {ci + 1} written; cumulative per category:\n{summ.round(3).to_string()}")
        del res
        torch.cuda.empty_cache()
    meta["total_gpu_s"] = float(sum(pass_times))
    write()
    log("done:", json.dumps({k: v for k, v in meta.items() if k not in ("chat_prefix_text",)}))


if __name__ == "__main__":
    main()
