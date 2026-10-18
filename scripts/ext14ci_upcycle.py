"""ext14 C1 upcycling cross-check (CPU only; p6-circuit): Mistral-7B-v0.1 vs Mixtral-8x7B-v0.1 weights.

1. Attention / norm weights per layer: cosine of the flattened matrices and the relative difference |A - B| / |A|
   (q, k, v, o, input / post-attention RMSNorm), plus embeddings, LM head, final norm.
2. Expert vs dense FFN, same neuron index: cosine of gate rows (w1 vs gate_proj), up rows (w3 vs up_proj) and down
   columns (w2 vs down_proj), mean / median over the 14,336 neurons; for every expert at the layers LAYERS.
3. Best match at layer 20 (and the other --match-layers): for every Mixtral expert neuron, the Mistral neuron with the
   highest cosine of the concatenated, separately normalised [gate row | up row | down column] vectors; identity rate
   (best match = same index), mutual best matches, matched-cosine distribution, and the same for a shuffled control.
   Saved for E000 (and every expert) as results/p6ci_upcycle/match_L{l}.npz (best index, cosine, second-best cosine).

Usage: python scripts/ext14ci_upcycle.py [--threads 8] [--layers 0,4,...] [--match-layers 20]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, torch
from moetrace.arch import load_spec
from moetrace.models import MODELS, RESULTS
from moetrace.weights import CheckpointStore

OUT = os.path.join(RESULTS, "p6ci_upcycle")
LAYERS = [0, 4, 8, 12, 16, 18, 19, 20, 21, 24, 28, 31]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()))


def reld(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / a.norm())


def rowcos(A, B):
    """cosine of matching rows (A, B [n, d]) -> [n]."""
    A, B = A.float(), B.float()
    return ((A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1)).clamp_min(1e-12)).numpy()


def neuron_vecs(gate, up, down):
    """[I, 3H] concatenation of separately normalised gate row, up row and down column per neuron."""
    g = torch.nn.functional.normalize(gate.float(), dim=1)
    u = torch.nn.functional.normalize(up.float(), dim=1)
    d = torch.nn.functional.normalize(down.float().t(), dim=1)
    return torch.cat([g, u, d], 1) / np.sqrt(3.0)


def best_match(X, Y, chunk=2048):
    """for every row of X the best and second-best cosine row of Y (rows unit-norm). -> idx, c1, c2"""
    n = X.shape[0]
    idx = np.empty(n, dtype=np.int64)
    c1 = np.empty(n, dtype=np.float32)
    c2 = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        S = X[s:s + chunk] @ Y.t()
        v, i = S.topk(2, dim=1)
        idx[s:s + chunk] = i[:, 0].numpy()
        c1[s:s + chunk] = v[:, 0].numpy()
        c2[s:s + chunk] = v[:, 1].numpy()
    return idx, c1, c2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--match-layers", default="20")
    ap.add_argument("--match-experts", default="all", help="'all' or comma list of experts for the best-match step")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    sd, snd = load_spec(MODELS["mistral7b"]["repo"])
    sm, snm = load_spec(MODELS["mixtral"]["repo"])
    D = CheckpointStore(snd, sd, device="cpu")
    M = CheckpointStore(snm, sm, device="cpu")
    res = {"attn": [], "globals": {}, "ffn": [], "match": {}}
    t0 = time.time()
    for nm, kd, km in (("embed", sd.k_embed, sm.k_embed), ("head", sd.k_head, sm.k_head), ("final_norm", sd.k_norm, sm.k_norm)):
        A, B = D.cpu_tensor(kd), M.cpu_tensor(km)
        res["globals"][nm] = {"cos": cos(A, B), "reldiff": reld(A, B)}
    for l in range(sd.n_layers):
        row = {"layer": l}
        for nm, k in (("q", "k_q"), ("k", "k_k"), ("v", "k_v"), ("o", "k_o"), ("ln1", "k_ln1"), ("ln2", "k_ln2")):
            A = D.cpu_tensor(getattr(sd, k).format(l=l))
            B = M.cpu_tensor(getattr(sm, k).format(l=l))
            row[f"{nm}_cos"] = cos(A, B)
            row[f"{nm}_reldiff"] = reld(A, B)
        res["attn"].append(row)
    log(f"attention done ({time.time() - t0:.0f}s)")
    layers = [int(x) for x in a.layers.split(",") if x != ""]
    mlayers = [int(x) for x in a.match_layers.split(",") if x != ""]
    for l in sorted(set(layers) | set(mlayers)):
        g0 = D.cpu_tensor(sd.k_gate.format(l=l))
        u0 = D.cpu_tensor(sd.k_up.format(l=l))
        d0 = D.cpu_tensor(sd.k_down.format(l=l))
        Y = neuron_vecs(g0, u0, d0) if l in mlayers else None
        for e in range(sm.n_experts):
            g = M.cpu_tensor(sm.k_gate.format(l=l, e=e))
            u = M.cpu_tensor(sm.k_up.format(l=l, e=e))
            d = M.cpu_tensor(sm.k_down.format(l=l, e=e))
            cg, cu, cd = rowcos(g, g0), rowcos(u, u0), rowcos(d.t(), d0.t())
            rec = {"layer": l, "expert": e, "gate_mean": float(cg.mean()), "gate_median": float(np.median(cg)),
                   "up_mean": float(cu.mean()), "up_median": float(np.median(cu)), "down_mean": float(cd.mean()),
                   "down_median": float(np.median(cd)), "gate_cos_matrix": cos(g, g0), "up_cos_matrix": cos(u, u0),
                   "down_cos_matrix": cos(d, d0), "frac_gate_gt05": float((cg > 0.5).mean())}
            if l in mlayers and (a.match_experts == "all" or str(e) in a.match_experts.split(",")):
                X = neuron_vecs(g, u, d)
                idx, c1, c2 = best_match(X, Y)
                diag = (X * Y).sum(1).numpy()
                ridx, rc1, _ = best_match(Y, X)
                mutual = ridx[idx] == np.arange(len(idx))
                # control: Mistral neurons matched against a random rotation-free shuffle of Mixtral's gate/up/down parts
                perm = np.random.default_rng(l * 100 + e).permutation(len(idx))
                Xs = neuron_vecs(g, u[perm], d[:, np.random.default_rng(l * 100 + e + 1).permutation(len(idx))])
                _, cs1, _ = best_match(Xs[:4096], Y)
                rec.update({"identity_rate": float((idx == np.arange(len(idx))).mean()), "mutual_rate": float(mutual.mean()),
                            "match_cos_median": float(np.median(c1)), "match_cos_q10": float(np.percentile(c1, 10)),
                            "second_cos_median": float(np.median(c2)), "diag_cos_median": float(np.median(diag)),
                            "shuffled_match_cos_median": float(np.median(cs1))})
                np.savez_compressed(os.path.join(OUT, f"match_L{l}_E{e}.npz"), best=idx.astype(np.int32), cos=c1, cos2=c2,
                                    diag=diag.astype(np.float32), rev_best=ridx.astype(np.int32), rev_cos=rc1,
                                    gate_diag=cg.astype(np.float32), up_diag=cu.astype(np.float32), down_diag=cd.astype(np.float32))
                log(f"L{l}E{e}: identity {rec['identity_rate']:.3f}, mutual {rec['mutual_rate']:.3f}, match cos median "
                    f"{rec['match_cos_median']:.3f} (shuffled {rec['shuffled_match_cos_median']:.3f}), diag {rec['diag_cos_median']:.3f}")
            res["ffn"].append(rec)
        log(f"layer {l} done ({time.time() - t0:.0f}s)")
    res["meta"] = {"layers": layers, "match_layers": mlayers, "seconds": round(time.time() - t0),
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(res, open(os.path.join(OUT, "upcycle.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
