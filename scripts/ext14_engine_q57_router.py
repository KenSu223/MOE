"""ext14 (p6-engine) diagnostic: is Qwen2-57B-A14B's bf16 fragility caused by the bf16 router logits?

transformers decoder layers chained one at a time (as the chain mode of scripts/ext14_engine_verify.py) on the same
prompts in three numerics: fp32 (reference), bf16, and bf16 with the router logits computed in fp32 (router input and
weight upcast; softmax / top-k as usual; routing weights cast back to bf16). Optionally also "bf16 + fp32 router + fp32
shared-expert gate". Reports per variant: Delta vs fp32, sanity ranks, routed-set agreement with fp32 per layer.

Usage: python scripts/ext14_engine_q57_router.py [--units 3] -> results/verify_ext14_qwen2_57b_router.json
"""
import argparse, gc, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, torch
import torch.nn.functional as F
import ext14_engine_verify as V
from moetrace.arch import load_spec
from moetrace.weights import CheckpointStore
from transformers.models.qwen2_moe import modeling_qwen2_moe as M

ORIG = M.Qwen2MoeTopKRouter.forward


def router_fp32(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states.float(), self.weight.float())
    router_probs = torch.softmax(router_logits, dtype=torch.float, dim=-1)
    top_v, top_i = torch.topk(router_probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        top_v = top_v / top_v.sum(dim=-1, keepdim=True)
    return router_logits, top_v.to(hidden_states.dtype), top_i


@torch.no_grad()
def chain(path, prompts, dt_, device="cuda"):
    spec, d = load_spec(path)
    cfg_dict = json.load(open(os.path.join(d, "config.json")))
    st = CheckpointStore(d, spec, device="cpu", dtype=dt_)
    emb = st.cpu_tensor(spec.k_embed).to(dt_)
    hs = [emb[torch.tensor(ids)][None].to(device) for (ids, _, _) in prompts]
    routes = {}
    for l in range(spec.n_layers):
        layer, rot = V._hf_layer(path, cfg_dict, l, dt_, device)
        rl_ = []
        for i in range(len(hs)):
            out, idx = V._apply_layer(layer, rot, hs[i], dt_, True, device)
            hs[i] = out[None].to(device, dt_)
            rl_.append(idx)
        routes[l] = rl_
        del layer, rot
        if device == "cuda":
            torch.cuda.empty_cache()
    norm = M.Qwen2MoeRMSNorm(spec.hidden, eps=spec.rms_eps).to(device, dt_)
    norm.weight.data.copy_(st.cpu_tensor(spec.k_norm).to(dt_))
    head = st.cpu_tensor(spec.k_head).to(dt_).to(device)
    rows = []
    for i, (ids, t, f) in enumerate(prompts):
        lg = F.linear(norm(hs[i])[0, -1], head).float()
        lp = torch.log_softmax(lg, -1)
        rows.append({"delta": float(lg[t] - lg[f]), "rank_true": int((lp > lp[t]).sum()) + 1, "logp_true": float(lp[t])})
    del hs, head, emb
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows, routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=3)
    ap.add_argument("--out", default="results/verify_ext14_qwen2_57b_router.json")
    ap.add_argument("--path", default="Qwen/Qwen2-57B-A14B", help="checkpoint (a tiny directory for testing)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    path = args.path
    if os.path.isdir(path):
        units = V.tiny_units(load_spec(path)[0], 2 * args.units)
    else:
        units, _ = V.wino_units(path, "qwen", args.units)
    san = V.sanity_rows(path)
    prompts = [(u["clean"], u["true"], u["foil"]) for u in units] + [(ids, t, t) for (_, _, ids, t) in san]
    nu = len(units)
    runs = {}
    for tag, dt_, fwd in (("fp32", torch.float32, ORIG), ("bf16", torch.bfloat16, ORIG), ("bf16_router32", torch.bfloat16, router_fp32)):
        t0 = time.time()
        M.Qwen2MoeTopKRouter.forward = fwd
        runs[tag] = chain(path, prompts, dt_, args.device)
        M.Qwen2MoeTopKRouter.forward = ORIG
        V.log(f"{tag}: {time.time() - t0:.0f}s")
    ref_rows, ref_routes = runs["fp32"]
    rep = {"units": nu, "sanity": [p for (p, _, _, _) in san], "variants": {}}
    for tag, (rows, routes) in runs.items():
        per_l = [float(np.mean([np.mean([set(a[t]) == set(b[t]) for t in range(len(a))]) for a, b in zip(routes[l], ref_routes[l])]))
                 for l in sorted(routes)]
        ent = {"delta_units": [r["delta"] for r in rows[:nu]], "sanity_rank": [r["rank_true"] for r in rows[nu:]],
               "delta_absdiff_vs_fp32": V._stats([a["delta"] - b["delta"] for a, b in zip(rows[:nu], ref_rows[:nu])]),
               "logp_true_absdiff_vs_fp32": V._stats([a["logp_true"] - b["logp_true"] for a, b in zip(rows, ref_rows)]),
               "routing_agreement_vs_fp32_by_layer": per_l, "routing_agreement_vs_fp32_mean": float(np.mean(per_l))}
        rep["variants"][tag] = ent
        V.log(f"{tag}: |dDelta| vs fp32 mean {ent['delta_absdiff_vs_fp32']['meanabs']:.3g} max {ent['delta_absdiff_vs_fp32']['maxabs']:.3g}, "
              f"sanity ranks {ent['sanity_rank']}, routing agreement vs fp32 mean {ent['routing_agreement_vs_fp32_mean']:.3f}")
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    V.log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
