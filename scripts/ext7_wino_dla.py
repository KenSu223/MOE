"""ext7 W4 complement: direct logit attribution (DLA) of the final-position sublayer outputs on STR pairs.

Why. At the final position the MoE block is a per-token function, so patching the attention output at EVERY layer
restores the clean final residual exactly whenever the final token is shared (STR); the joint "all attention" patch of
W4 is therefore 1 by construction and "all MoE" is the informative joint patch (see scripts/ext7_wino_joint.py). The
residual stream is additive, so the clean - corrupted difference of the final residual splits EXACTLY into per-layer
attention and MoE output differences; their projections on the logit-difference direction give the direct effect of
each sublayer (no downstream recomputation), which sum to the drop up to the final RMSNorm's scale.

Pass: prefill rows only (clean and corrupted prompt of every directed case), DiagSpec(attn_out_final, resid_final).
With u = w_norm * (W_U[true] - W_U[foil]) and s = 1 / rms(h_final) of each run, the direct logit contribution of a
component x is DLA(x) = s * (u . x); per directed case and layer: DLA difference clean - corrupted of the attention output
and of the MoE output (MoE_l = h_l - h_{l-1} - Attn_l), plus the embedding term (same token, different s).
Sum over components = Delta_clean - Delta_corrupt up to bf16 rounding of the head (checked: column 'dla_total_vs_drop').
Direct-path split with the EXACT final norm (ext8's suggestion): with dA = sum_l (Attn_l clean - corrupted) and dM likewise
for the MoE outputs (h_clean = h_corrupt + dA + dM), den_attn = Delta(h_corrupt + dA) - Delta(h_corrupt), den_moe likewise
(sufficiency of the direct paths), noi_attn = Delta(h_clean) - Delta(h_clean - dA), noi_moe likewise (necessity); recon =
Delta(h_corrupt + dA + dM) - Delta(h_clean) (= 0 up to bf16). Delta(.) = rmsnorm + head on the final residual (bf16 as the engine).

Usage: python scripts/ext7_wino_dla.py --model qwen3 --pairs <parquet> --case-sets <json> --out wino_qwen3_str
           [--families main,rep] [--chunk 512]
Outputs results/<out>/dla_rows.parquet (case_id, layer, attn_c, attn_k, moe_c, moe_k, ...), dla_cases.parquet, run_meta ("dla")
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import MODELS
from moetrace import ext7_pairs as P


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    P.add_common_args(ap)
    ap.add_argument("--families", default="main,rep")
    ap.add_argument("--chunk", type=int, default=512, help="directed cases per pass (memory of the CPU diag tensors)")
    args = ap.parse_args()
    m = MODELS[args.model]
    cs, pairs, _, _, _ = P.setup_run(args, "dla")
    fam_all = P.load_families(args.out)
    fam = {k: v for k, v in fam_all.items() if k in args.families.split(",")}
    ids = P.all_ids(fam)
    dc = P.directed_cases(pairs, ids)
    chunks = [ids[i:i + args.chunk] for i in range(0, len(ids), args.chunk)]
    log(f"{args.model} -> {args.out}: {len(ids)} directed cases, families {list(fam)}, {len(chunks)} prefill-only passes")
    if args.dry_run:
        return
    import torch
    from moetrace.engine import Engine, PrefillSpec, DiagSpec, rmsnorm
    meta = {"model": args.model, "repo": m["repo"], "families": list(fam), "n_cases": len(ids), "agent": args.agent,
            "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    P.write_meta(args.out, "dla", meta)
    eng = Engine(m["repo"])
    s = eng.spec
    head, wn = eng.g["head"], eng.g["norm"]
    rows, crow, times = [], [], []
    for ci, ch in enumerate(chunks):
        n = len(ch)
        pre = [PrefillSpec(dc[c].clean_ids, dc[c].true_id, dc[c].foil_id) for c in ch]
        pre += [PrefillSpec(dc[c].corrupt_ids, dc[c].true_id, dc[c].foil_id) for c in ch]
        res = eng.run(pre, [], record_routing=False, log=log if ci == 0 else None, diag=DiagSpec(attn_out_final=True, resid_final=True))
        times.append(res.extra["total_s"])
        dg = res.extra["diag"]
        Rf = dg["resid_final"].float()  # [L, B, H] residual after each layer (final position)
        Af = dg["attn_out_final"].float()  # [L, B, H]
        L, B, H = Rf.shape
        fin = torch.tensor([p.ids[-1] for p in pre])
        emb = eng.g["embed"][fin.to(eng.device)].float().cpu()  # [B, H]
        prev = torch.cat([emb[None], Rf[:-1]], 0)
        Mf = Rf - prev - Af  # MoE output (incl. bf16 rounding of the residual adds)
        tid = torch.tensor([p.true_id for p in pre], device=eng.device)
        fid = torch.tensor([p.foil_id for p in pre], device=eng.device)
        u = (wn.float()[None] * (head[tid].float() - head[fid].float())).cpu()  # [B, H]
        hfin = Rf[-1]
        sc = torch.rsqrt(hfin.pow(2).mean(-1) + s.rms_eps)  # [B]
        dla_a = (Af * u[None]).sum(-1) * sc[None]  # [L, B]
        dla_m = (Mf * u[None]).sum(-1) * sc[None]
        dla_e = (emb * u).sum(-1) * sc
        # direct-path split with the exact final norm: Delta(h) of modified final residuals
        dA = (Af[:, :n] - Af[:, n:]).sum(0)  # [n, H] sum over layers of the attention-output differences (clean - corrupted)
        dM = (Mf[:, :n] - Mf[:, n:]).sum(0)
        hc, hk = hfin[:n], hfin[n:]
        tt, ff = tid[:n], fid[:n]

        def dlt(h):
            hn = rmsnorm(h.to(eng.device).to(torch.bfloat16), wn, s.rms_eps)
            return ((hn.float() * (head[tt].float() - head[ff].float())).sum(-1)).cpu()
        base_k, base_c = dlt(hk), dlt(hc)
        dir_rows = dict(den_attn=dlt(hk + dA) - base_k, den_moe=dlt(hk + dM) - base_k, noi_attn=base_c - dlt(hc - dA), noi_moe=base_c - dlt(hc - dM),
                        recon=dlt(hk + dA + dM) - base_c)
        d = res.delta
        for i, c in enumerate(ch):
            for l in range(L):
                rows.append(dict(case_id=c, layer=l, attn_c=float(dla_a[l, i]), attn_k=float(dla_a[l, n + i]), moe_c=float(dla_m[l, i]),
                                 moe_k=float(dla_m[l, n + i])))
            tot_c = float(dla_a[:, i].sum() + dla_m[:, i].sum() + dla_e[i])
            tot_k = float(dla_a[:, n + i].sum() + dla_m[:, n + i].sum() + dla_e[n + i])
            crow.append(dict(case_id=c, delta_clean=float(d[i]), delta_corrupt=float(d[n + i]), emb_c=float(dla_e[i]), emb_k=float(dla_e[n + i]),
                             dla_total_c=tot_c, dla_total_k=tot_k, base_c=float(base_c[i]), base_k=float(base_k[i]),
                             **{k: float(v[i]) for k, v in dir_rows.items()}))
        pd.DataFrame(rows).to_parquet(os.path.join(P.run_dir(args.out), "dla_rows.parquet"), index=False)
        cdf = pd.DataFrame(crow)
        cdf["drop"] = cdf.delta_clean - cdf.delta_corrupt
        cdf["dla_total_vs_drop"] = (cdf.dla_total_c - cdf.dla_total_k) - cdf["drop"]
        cdf.to_parquet(os.path.join(P.run_dir(args.out), "dla_cases.parquet"), index=False)
        log(f"pass {ci + 1}/{len(chunks)}: {res.extra['total_s']:.0f}s; DLA total - drop: max |.| {cdf.dla_total_vs_drop.abs().max():.3f}")
        del res, dg, Rf, Af, Mf, prev
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    a = (df.attn_c - df.attn_k).groupby(df.case_id).sum()
    mo = (df.moe_c - df.moe_k).groupby(df.case_id).sum()
    dr = cdf.set_index("case_id")["drop"]
    log(f"direct effect / drop: attention {a.mean() / dr.mean():+.3f}, MoE {mo.mean() / dr.mean():+.3f}, "
        f"embedding {(cdf.emb_c - cdf.emb_k).mean() / dr.mean():+.3f}")
    meta.update({"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": True, "pass_times_s": times})
    P.write_meta(args.out, "dla", meta)


if __name__ == "__main__":
    main()
