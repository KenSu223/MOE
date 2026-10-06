"""ext11 Part B (GPU, one prefill pass per model; needs the ext9 engine, DiagSpec.contrib_final_vectors):
vocabulary projections of the target experts' writes at the final position.

Prompts: CounterFact STR (Direction-6 clean prompts + every selected donor), WinoGrande STR main family (the 512 clean
prompts of the directed cases; the corrupted prompt of case c is the clean prompt of its twin c ^ 1), the context-free
local prompts of the main pairs (c_e only) and the first 400 IOI clean prompts (c_e only).
Experts: targets of both tasks + both tasks' ext8 population top-10 (Part A).

Vectors (fp32, final position, from DiagSpec.contrib_final_vectors; slot s <-> route_idx[l, b, s]):
  delta  = c_e(clean) - c_e(corrupt)  (c_e(corrupt) = 0 when e is not routed in the corrupted run), per STR row where e is
           clean-active; projected with the final norm frozen at the CORRUPTED run's scale (= ext8's DLA convention):
           logits_v = ((v * gamma) @ W_U^T) / rms(h_final_corrupt)
  clean  = c_e(clean) of the clean prompt (STR clean prompts, local prompts, IOI), frozen at the prompt's own scale.
Per vector: value and rank of r and r' (rank 1 = most promoted; 'rank_low' 1 = most suppressed), top-20 promoted and
suppressed token ids, class means (logits_v over the class tokens minus the vocabulary mean, and in SD units) and the
share of the top-50 promoted tokens in each class (classes: WinoGrande trigger vocabulary, CounterFact object first tokens
per relation group, IOI names; moetrace.ext11_writer.vocab_classes), rank of r within its class (trigger vocabulary /
CounterFact objects of the case's relation group).
Exact-norm direct effect: for delta, Delta(norm(h_corrupt + delta)) - Delta(norm(h_corrupt)) through the real final RMSNorm
(fp32, offline from DiagSpec.resid_final) vs the frozen-norm DLA delta . u / rms(h_corrupt), u = gamma * (W_U[r] - W_U[r']);
for clean, Delta(norm(h_clean)) - Delta(norm(h_clean - c_e)) vs c_e . u / rms(h_clean).

Outputs: raw vectors /opt/dlami/nvme/moe_ext11/<out>/vectors.pt; results/<out>/{prompts.parquet, vec_rows.parquet
(one row per vector), vec_top.parquet (top-20 promoted / suppressed ids per vector), run_meta.json}.

Usage: python scripts/ext11_writer_vocab.py qwen3|mixtral --out <run> [--engine main|dev] [--from-raw] [--ioi 400]
"""
import argparse, importlib, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, pandas as pd, torch

from moetrace.models import MODELS, RESULTS
from moetrace import ext11_writer as W

RAW = "/opt/dlami/nvme/moe_ext11"
TOPN = 20


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def interest(model):
    A = json.load(open(W.SUMMARY))["A"]
    ex = [W.ename(x) for x in W.model_targets(model)]
    for k, r in A.items():
        if r["model"] == model:
            ex += r["pop_top10"]
    ex = list(dict.fromkeys(ex))
    return ex, sorted({W.parse_ename(e)[0] for e in ex})


def build(model, n_ioi):
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    from moetrace import ext7_pairs as P
    from ext7_wino_scan import local_prompt
    st = True
    tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[model]["repo"]))
    rows = []

    def add(set_, ids, t, f, **meta):
        rows.append(dict(set=set_, ids=[int(x) for x in ids], true_id=int(t), foil_id=int(f), **meta))
    ck = "cf_qwen3" if model == "qwen3" else "cf_mixtral"
    tc = W.load_task(ck)
    sc = pd.read_parquet(os.path.join(RESULTS, W.RUNS[ck]["src"], "sweep_cases.parquet")).set_index("case_id")
    for c in tc.cases.itertuples():
        add("cf_clean", c.clean_ids, c.true_id, c.foil_id, case_id=int(c.case_id), slot=-1, split=c.split, unit=int(c.unit),
            relation=str(sc.relation.get(int(c.case_id), "")))
    cm = tc.cases.set_index("case_id")
    for r in tc.rows.itertuples():
        c = cm.loc[int(r.case_id)]
        add("cf_donor", r.corrupt_ids, c.true_id, c.foil_id, case_id=int(r.case_id), slot=int(r.slot), split=c.split, unit=int(c.unit),
            relation=str(sc.relation.get(int(r.case_id), "")), row_id=int(r.row_id))
    wk = "wino_qwen3" if model == "qwen3" else "wino_mixtral"
    tw = W.load_task(wk)
    pairs = P.load_pairs(os.path.join(W.ROOT, W.RUNS[wk]["pairs"]), json.load(open(os.path.join(W.ROOT, W.RUNS[wk]["case_sets"]))))
    for c in tw.cases.itertuples():
        pi, d = divmod(int(c.case_id), 2)
        r = pairs.loc[pi]
        add("wino", c.clean_ids, c.true_id, c.foil_id, case_id=int(c.case_id), split=c.split, unit=int(c.unit), pair_id=r.pair_id,
            final_word=r.context.split()[-1])
    for pi in sorted({int(c) // 2 for c in tw.cases.case_id}):
        r = pairs.loc[pi]
        for d, (ans, tt, ff) in enumerate(((r.ans_a, r.trig_a, r.trig_b), (r.ans_b, r.trig_b, r.trig_a))):
            add("wino_local", tok(local_prompt(r.context, ans), add_special_tokens=st)["input_ids"], tt, ff, case_id=2 * pi + d,
                unit=int(pi), pair_id=r.pair_id, final_word=r.context.split()[-1])
    ip = "qwen3" if model == "qwen3" else "mixtral_bos"
    it = pd.read_parquet(os.path.join(W.ROOT, "data/ioi/items.parquet")).head(n_ioi)
    for r in it.itertuples():
        add("ioi", json.loads(getattr(r, f"{ip}_ids_clean")), int(getattr(r, f"{ip}_trig_io")), int(getattr(r, f"{ip}_trig_s")),
            pair_id=r.pair_id)
    df = pd.DataFrame(rows)
    df.insert(0, "idx", np.arange(len(df)))
    df["n_tok"] = df.ids.map(len)
    return df, tok


def vector_table(df, ex_le, route, cvec, h_final):
    """List of vectors: (kind, prompt idx, ref idx (norm / corrupted run), layer, expert, true, foil, vector)."""
    pos = {}
    for r in df.itertuples():
        if r.set in ("cf_clean", "wino"):
            pos[(r.set, int(r.case_id))] = int(r.idx)
    out = []

    def c_of(i, l, e):
        hit = np.nonzero(route[l, i] == e)[0]
        return (cvec[l][i, int(hit[0])], True) if len(hit) else (None, False)
    for r in df.itertuples():
        i = int(r.idx)
        for (l, e) in ex_le:
            if r.set in ("cf_clean", "wino_local", "ioi", "wino"):
                v, act = c_of(i, l, e)
                if act:
                    out.append(("clean", r.set, i, i, l, e, int(r.true_id), int(r.foil_id), v))
            if r.set == "cf_donor":
                ci = pos[("cf_clean", int(r.case_id))]
                vc, act = c_of(ci, l, e)
                if not act:
                    continue
                vj, actj = c_of(i, l, e)
                out.append(("delta", "cf", ci, i, l, e, int(r.true_id), int(r.foil_id), vc - (vj if actj else 0.0)))
            if r.set == "wino":
                tw = pos[("wino", int(r.case_id) ^ 1)]  # the corrupted prompt = the twin's clean prompt
                vc, act = c_of(i, l, e)
                if not act:
                    continue
                vj, actj = c_of(tw, l, e)
                out.append(("delta", "wino", i, tw, l, e, int(r.true_id), int(r.foil_id), vc - (vj if actj else 0.0)))
    return out


def project(vecs, df, h_final, head, gamma, eps, classes, dev, rel_cls):
    """Projection statistics per vector (see module docstring)."""
    V = head.shape[0]
    Wf = head.float()
    g = gamma.float().to(dev)
    cls_idx = {k: torch.tensor(sorted(v), device=dev, dtype=torch.long) for k, v in classes.items() if len(v)}
    hf = h_final.float().to(dev)  # [N, H]
    rms = torch.sqrt(hf.pow(2).mean(-1) + eps)  # [N]
    rows, tops = [], []
    B = 256
    for c0 in range(0, len(vecs), B):
        chunk = vecs[c0:c0 + B]
        v = torch.stack([torch.as_tensor(x[8], dtype=torch.float32) for x in chunk]).to(dev)  # [b, H]
        ref = torch.tensor([x[3] for x in chunk], device=dev)
        own = torch.tensor([x[2] for x in chunk], device=dev)
        tr = torch.tensor([x[6] for x in chunk], device=dev)
        fo = torch.tensor([x[7] for x in chunk], device=dev)
        kind_delta = torch.tensor([x[0] == "delta" for x in chunk], device=dev)
        nref = torch.where(kind_delta, ref, own)  # norm reference row: corrupted run (delta) / own prompt (clean)
        lg = ((v * g) @ Wf.T) / rms[nref][:, None]  # [b, V]
        ar = torch.arange(len(chunk), device=dev)
        vt, vf = lg[ar, tr], lg[ar, fo]
        rank_t = (lg > vt[:, None]).sum(1) + 1
        rank_f = (lg > vf[:, None]).sum(1) + 1
        rlow_f = (lg < vf[:, None]).sum(1) + 1
        rlow_t = (lg < vt[:, None]).sum(1) + 1
        mu, sd = lg.mean(1), lg.std(1)
        tp_v, tp_i = lg.topk(TOPN, dim=1)
        bt_v, bt_i = (-lg).topk(TOPN, dim=1)
        top50 = lg.topk(50, dim=1).indices
        cl = {}
        for k, idx in cls_idx.items():
            m = lg[:, idx].mean(1)
            cl[f"cls_{k}"] = (m - mu).cpu().numpy()
            cl[f"clsz_{k}"] = ((m - mu) / sd).cpu().numpy()
            cl[f"top50_{k}"] = torch.isin(top50, idx).float().mean(1).cpu().numpy()
        # rank of r within the trigger vocabulary (wino) / the CF objects of the case's relation group (cf)
        rk_in = np.full(len(chunk), np.nan)
        rk_f_in = np.full(len(chunk), np.nan)
        n_in = np.full(len(chunk), np.nan)
        for j, x in enumerate(chunk):
            r = df.iloc[x[2]]
            key = "wg_trigger" if r.set in ("wino", "wino_local") else (rel_cls.get(r.get("relation", ""), None) if r.set.startswith("cf") else None)
            if key and key in cls_idx and (x[6] in classes[key]):
                idx = cls_idx[key]
                vals = lg[j, idx]
                rk_in[j] = float((vals > vt[j]).sum() + 1)
                n_in[j] = float(len(idx))
                if x[7] in classes[key]:
                    rk_f_in[j] = float((vals > vf[j]).sum() + 1)
        # exact final norm vs frozen
        u = g * (Wf[tr] - Wf[fo])  # [b, H]
        def Dfun(h):
            return ((h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)) * u).sum(-1)
        hr = hf[nref]
        exact = torch.where(kind_delta, Dfun(hr + v) - Dfun(hr), Dfun(hr) - Dfun(hr - v))
        frozen = (v * u).sum(-1) / rms[nref]
        for j, x in enumerate(chunk):
            rows.append(dict(kind=x[0], task=x[1], prompt=x[2], ref=x[3], layer=x[4], expert=x[5], true_id=x[6], foil_id=x[7],
                             val_true=float(vt[j]), val_foil=float(vf[j]), rank_true=int(rank_t[j]), rank_foil=int(rank_f[j]),
                             rank_low_foil=int(rlow_f[j]), rank_low_true=int(rlow_t[j]), mean=float(mu[j]), sd=float(sd[j]),
                             vnorm=float(v[j].norm()), exact=float(exact[j]), frozen=float(frozen[j]), rank_true_in_class=rk_in[j],
                             rank_foil_in_class=rk_f_in[j], n_class=n_in[j], **{k: float(a[j]) for k, a in cl.items()}))
            tops.append(dict(vec=c0 + j, top_ids=tp_i[j].cpu().numpy().astype(np.int32).tolist(), top_vals=tp_v[j].cpu().numpy().round(4).tolist(),
                             bot_ids=bt_i[j].cpu().numpy().astype(np.int32).tolist(), bot_vals=(-bt_v[j]).cpu().numpy().round(4).tolist()))
        del lg
    return pd.DataFrame(rows), pd.DataFrame(tops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["qwen3", "mixtral"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="main", choices=["main", "dev"])
    ap.add_argument("--ioi", type=int, default=400)
    ap.add_argument("--from-raw", action="store_true", help="skip the model pass, reuse the saved vectors")
    args = ap.parse_args()
    od = os.path.join(RESULTS, args.out)
    rd = os.path.join(RAW, args.out)
    os.makedirs(od, exist_ok=True)
    os.makedirs(rd, exist_ok=True)
    ex, layers = interest(args.model)
    ex_le = [W.parse_ename(e) for e in ex]
    df, tok = build(args.model, args.ioi)
    df.assign(ids=df.ids.map(json.dumps)).to_parquet(os.path.join(od, "prompts.parquet"), index=False)
    E = importlib.import_module("moetrace.engine_ext9_dev" if args.engine == "dev" else "moetrace.engine")
    meta = {"model": args.model, "repo": MODELS[args.model]["repo"], "experts_of_interest": ex, "layers": layers,
            "sets": df.set.value_counts().to_dict(), "n_prompts": int(len(df)), "engine": args.engine,
            "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    eng = E.Engine(MODELS[args.model]["repo"])
    rawp = os.path.join(rd, "vectors.pt")
    if args.from_raw and os.path.exists(rawp):
        raw = torch.load(rawp)
    else:
        pre = [E.PrefillSpec(list(r.ids), int(r.true_id), int(r.foil_id)) for r in df.itertuples()]
        diag = E.DiagSpec(contrib_final_vectors=tuple(layers), resid_final=True, contrib_dla=True)
        log(f"{args.model}: {len(pre)} prefill rows, T <= {df.n_tok.max()}, layers {layers}")
        torch.cuda.reset_peak_memory_stats()
        res = eng.run(pre, [], record_routing=True, log=log, diag=diag)
        dg = res.extra["diag"]
        L = res.route_idx.shape[0]
        raw = {"route_idx": torch.from_numpy(res.route_idx.astype(np.int16)), "delta": torch.from_numpy(np.asarray(res.delta)),
               "h_final": dg["resid_final"][L - 1].clone(), "final_rms": torch.from_numpy(dg["final_rms"]),
               "cvec": {int(l): dg["contrib_final_vectors"][int(l)].clone() for l in layers},
               "contrib_dla": torch.from_numpy(dg["contrib_dla"]), "pass_s": res.extra["total_s"],
               "peak_GB": torch.cuda.max_memory_allocated() / 2**30}
        torch.save(raw, rawp)
        log(f"pass done in {res.extra['total_s']:.0f}s, peak {raw['peak_GB']:.1f} GB; vectors saved")
        del res, dg
    route = raw["route_idx"].numpy()
    cvec = {l: raw["cvec"][l].numpy() for l in layers}
    vecs = vector_table(df, ex_le, route, cvec, raw["h_final"])
    log(f"{len(vecs)} vectors ({sum(v[0] == 'delta' for v in vecs)} delta, {sum(v[0] == 'clean' for v in vecs)} clean)")
    classes = W.vocab_classes(args.model, tok)
    rel_cls = {}
    for g, rels in W.REL_GROUPS.items():
        for r in rels:
            rel_cls[r] = f"cf_{g}"
    dev = "cuda"
    vr, vt = project(vecs, df, raw["h_final"], eng.g["head"], eng.g["norm"], eng.spec.rms_eps, classes, dev, rel_cls)
    # same-pass check: the frozen DLA of clean c_e equals the engine's contrib_dla / final_rms
    vr.to_parquet(os.path.join(od, "vec_rows.parquet"), index=False)
    vt.to_parquet(os.path.join(od, "vec_top.parquet"), index=False)
    cd = raw["contrib_dla"].numpy()
    fr = raw["final_rms"].numpy()
    chk = []
    for j, x in enumerate(vecs):
        if x[0] == "clean":
            s = int(np.nonzero(route[x[4], x[2]] == x[5])[0][0])
            chk.append((vr.frozen.iloc[j], cd[x[4], x[2], s] / fr[x[2]]))
    chk = np.array(chk)
    meta.update({"complete": True, "n_vectors": int(len(vr)), "pass_s": raw.get("pass_s"), "peak_GB": raw.get("peak_GB"),
                 "check_frozen_vs_contrib_dla_max_abs": float(np.abs(chk[:, 0] - chk[:, 1]).max()) if len(chk) else None,
                 "class_sizes": {k: len(v) for k, v in classes.items()},
                 "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    log("done", json.dumps({k: meta[k] for k in ("n_vectors", "check_frozen_vs_contrib_dla_max_abs")}))


if __name__ == "__main__":
    main()
