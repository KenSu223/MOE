"""ext9 engine verification (E4) on OLMoE-1B-7B-0125 against transformers hooks (model fully on the GPU, eager attention).

Units = the ext8 STR units (CounterFact first-donor pairs + WinoGrande OLMoE margin twins, both directions).

E4a route masks (prefill rows, clean prompts): per unit six mask configurations x pos {all, final} x mode {reroute, zero}
    single_l8     the final position's top-1 expert at layer 8
    multi_layer   two final-position experts at layer 3 + one at layer 10
    random5       five random (layer, expert) pairs routed somewhere in the prompt (engine and HF routing)
    allfinal_l12  all 8 experts routed at the final position at layer 12
    unrouted_l6   one expert routed at some earlier position but NOT at the final position, layer 6
    many_l5       20 experts at layer 5 (reroute chooses from the remaining 44)
  HF side: a forward hook on mlp.gate (OlmoeTopKRouter) that returns, at the masked positions, reroute = softmax over the
  logits with the masked experts at -inf, top-k, the model's renormalisation rule (none for OLMoE), cast to bf16;
  zero = the router's own (scores, indices) with the masked slots' scores set to 0. Compared: Delta at the final position,
  token log-probs at every position, final-position routing at the masked layers; invariants (masked experts absent,
  zero mode keeps the routing, final-only masks leave every earlier position bit-identical, zero mode with an expert
  that is not routed at the final position = baseline bit for bit, OLMoE reroute weight inflation 1 / (1 - p_masked)).
E4b head steps in `multi` (both directions): one head step == attn_head kind (exact), heads_experts with no experts ==
  attn_head kind, heads_experts with no heads ~ coalition_set, all heads of a layer == attn_layer (first and later step,
  to o_proj bf16 rounding), all heads at every layer vs the source Delta, identity (source = parent = clean), and random
  mixed head / expert / layer / attention configurations vs HF (o_proj forward pre-hook replacing the listed heads'
  slices of the final position's o_proj input by the source run's; MoE forward hook for expert sets: own + sum_S
  (c_e(source) - c_e(own)) with own contributions from the hook input).
E4c DiagSpec.contrib_final_vectors: sum over slots vs the fp32 MoE output, norms vs route_cnorm, vs HF contributions,
  zero-mode masked slots exactly 0.
Plus: spawn on a masked parent raises; batch independence of unmasked rows; a stress pass of head-step multi rows.

Usage: python scripts/ext9_engine_verify.py [--engine dev|main] [--stress 10000] -> results/verify_ext9_engine_olmoe.json
"""
import argparse, importlib, json, os, random, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, torch
import torch.nn.functional as F
from moetrace.verify import _hf_load, _moe_module
import ext8_engine_verify as V8  # read-only reuse: STR units, HF contribution helper

REPO = "allenai/OLMoE-1B-7B-0125"
CFV_LAYERS = (0, 5, 10, 15)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _stats(z):
    z = np.asarray(z, dtype=np.float64)
    if len(z) == 0:
        return {"n": 0}
    return {"n": int(len(z)), "maxabs": float(np.abs(z).max()), "meanabs": float(np.abs(z).mean()),
            "frac_exact": float((z == 0).mean()), "frac_within_0.1": float((np.abs(z) < 0.1).mean()),
            "frac_within_0.25": float((np.abs(z) < 0.25).mean())}


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None


# ------------------------------------------------------------------------------------------------------------------
# HF side
# ------------------------------------------------------------------------------------------------------------------
class HF:
    def __init__(self):
        self.model = _hf_load(REPO)
        cfg = self.model.config
        self.L, self.nH = cfg.num_hidden_layers, cfg.num_attention_heads
        self.D = cfg.hidden_size // self.nH
        self.k = cfg.num_experts_per_tok

    def fwd(self, ids, hooks):
        hs = []
        try:
            for mod, kind, fn in hooks:
                hs.append(mod.register_forward_pre_hook(fn) if kind == "pre" else mod.register_forward_hook(fn))
            return self.model(input_ids=torch.tensor([ids], device="cuda")).logits[0].float()  # [T, V]
        finally:
            for h in hs:
                h.remove()

    def record(self, ids):
        """final-position attention output, MoE output, expert contributions, o_proj input; routing at every position."""
        st, hooks = {}, []
        for l in range(self.L):
            lay = self.model.model.layers[l]
            moe = _moe_module(self.model, l)

            def ra(mod, args, out, l=l):
                st[("attn", l)] = V8._o(out)[0, -1].detach().clone()

            def rm(mod, args, out, l=l, moe=moe):
                st[("moe", l)] = V8._o(out)[0, -1].detach().clone()
                st[("c", l)] = V8._contribs(moe, args[0][0, -1:].detach())

            def ro(mod, args, l=l):
                st[("oin", l)] = args[0][0, -1].detach().clone()

            def rg(mod, args, out, l=l):  # first call = the full forward (_contribs calls the gate again on the final token)
                st.setdefault(("route", l), out[2].detach().cpu().numpy())  # [T, k]

            hooks += [(lay.self_attn, "fwd", ra), (moe, "fwd", rm), (lay.self_attn.o_proj, "pre", ro), (moe.gate, "fwd", rg)]
        lg = self.fwd(ids, hooks)
        return lg, st

    def gate_hooks(self, mask, pos, mode, T, rec):
        """forward hooks on mlp.gate implementing the route mask (see module docstring); rec[l] = final-position indices."""
        by_l = {}
        for l, e in mask:
            by_l.setdefault(int(l), set()).add(int(e))
        hooks = []
        for l, es in by_l.items():
            gate = _moe_module(self.model, l).gate
            ex = torch.tensor(sorted(es), device="cuda")

            def hk(mod, args, out, ex=ex, l=l):
                logits, scores, idx = out
                rows = torch.arange(T, device="cuda") if pos == "all" else torch.tensor([T - 1], device="cuda")
                if mode == "reroute":
                    lf = logits.float().clone()
                    lf[rows[:, None], ex[None, :]] = float("-inf")
                    probs = torch.softmax(lf, dim=-1)
                    tv, ti = torch.topk(probs, mod.top_k, dim=-1)
                    if mod.norm_topk_prob:
                        tv = tv / tv.sum(dim=-1, keepdim=True)
                    tv = tv.to(logits.dtype)
                    sc, ix = scores.clone(), idx.clone()
                    sc[rows], ix[rows] = tv[rows], ti[rows]
                else:
                    sc, ix = scores.clone(), idx
                    hit = torch.isin(idx, ex)
                    pm = torch.zeros(T, dtype=torch.bool, device="cuda")
                    pm[rows] = True
                    sc[hit & pm[:, None]] = 0
                rec[l] = ix[T - 1].cpu().numpy()
                return logits, sc, ix
            hooks.append((gate, "fwd", hk))
        return hooks

    def patch_hooks(self, steps, src):
        """steps: list of (layer, kind, heads, experts), kind in attn / moe / heads / coal / he; src = recorded store."""
        hooks = []
        D = self.D
        for (l, kind, heads, experts) in steps:
            lay = self.model.model.layers[l]
            moe = _moe_module(self.model, l)
            if kind == "attn":
                hooks.append((lay.self_attn, "fwd", lambda m, a, o, k=("attn", l): V8._rep(o, src[k])))
            if kind == "moe":
                hooks.append((moe, "fwd", lambda m, a, o, k=("moe", l): V8._rep(o, src[k])))
            if kind in ("heads", "he") and heads:
                def hp(mod, args, l=l, heads=tuple(heads)):
                    x = args[0].clone()
                    s_ = src[("oin", l)]
                    for h in heads:
                        x[0, -1, h * D : (h + 1) * D] = s_[h * D : (h + 1) * D]
                    return (x,) + tuple(args[1:])
                hooks.append((lay.self_attn.o_proj, "pre", hp))
            if kind in ("coal", "he") and experts:
                def hc(mod, args, out, l=l, ex=tuple(experts), moe=moe):
                    own = V8._contribs(moe, args[0][0, -1:])
                    srcc = src[("c", l)]
                    v = sum((srcc.get(e, 0.0) - own.get(e, 0.0)) for e in ex)
                    o = V8._o(out)
                    return V8._rep(out, (o[0, -1].float() + v).to(o.dtype))
                hooks.append((moe, "fwd", hc))
        return hooks


def eng_steps(steps):
    out = []
    for (l, kind, heads, experts) in steps:
        if kind == "attn":
            out.append((l, "attn_layer", None))
        elif kind == "moe":
            out.append((l, "layer", None))
        elif kind == "heads":
            out.append((l, "attn_head", tuple(heads)))
        elif kind == "coal":
            out.append((l, "coalition_set", tuple(experts)))
        elif kind == "he":
            out.append((l, "heads_experts", (tuple(heads), tuple(experts))))
        else:
            raise ValueError(kind)
    return tuple(out)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="dev", choices=["dev", "main"])
    ap.add_argument("--n-cf", type=int, default=6)
    ap.add_argument("--n-wino-pairs", type=int, default=3)
    ap.add_argument("--n-mixed", type=int, default=6)
    ap.add_argument("--stress", type=int, default=10000)
    ap.add_argument("--out", default="results/verify_ext9_engine_olmoe.json")
    args = ap.parse_args()
    E = importlib.import_module("moetrace.engine_ext9_dev" if args.engine == "dev" else "moetrace.engine")
    Engine, PrefillSpec, SpawnSpec, DiagSpec = E.Engine, E.PrefillSpec, E.SpawnSpec, E.DiagSpec
    units = V8.cf_units(args.n_cf) + V8.wino_units(args.n_wino_pairs)
    n = len(units)
    log(f"{n} units; engine module {E.__name__}")
    eng = Engine(REPO)
    L, NE, K, nH = eng.spec.n_layers, eng.spec.n_experts, eng.spec.top_k, eng.spec.n_heads
    ALLH = tuple(range(nH))
    rep = {"engine_module": E.__name__, "n_units": n, "units": [{"src": u["src"], "name": u["name"], "T": len(u["clean"])} for u in units]}

    # ---------------- engine pass A0: baselines (clean rows 0..n-1, corrupted rows n..2n-1)
    pre0 = [PrefillSpec(u["clean"], u["true"], u["foil"]) for u in units] + [PrefillSpec(u["corrupt"], u["true"], u["foil"]) for u in units]
    diag0 = DiagSpec(token_logprobs=True, route_all_layers=tuple(range(L)), router_logits_final=True, contrib_final_vectors=CFV_LAYERS)
    r0 = eng.run(pre0, [], record_routing=True, diag=diag0)
    dg0 = r0.extra["diag"]

    # ---------------- HF recording of every clean / corrupted run
    t0 = time.time()
    hf = HF()
    hf_lg, hf_st = {}, {}
    for i, u in enumerate(units):
        for run in ("clean", "corrupt"):
            lg, st = hf.record(u[run])
            hf_lg[(i, run)] = lg
            hf_st[(i, run)] = st
    route_agree = float(np.mean([sorted(hf_st[(i, run)][("route", l)][-1].tolist()) == sorted(r0.route_idx[l, i + (0 if run == "clean" else n)].tolist())
                                 for i in range(n) for run in ("clean", "corrupt") for l in range(L)]))
    rep["routing_agreement_final_engine_vs_hf"] = route_agree

    def hf_delta(lg, u):
        return float(lg[-1, u["true"]] - lg[-1, u["foil"]])

    def hf_logprobs(lg, ids):
        lp = torch.log_softmax(lg, dim=-1)
        nxt = torch.tensor(ids[1:], device=lg.device)
        return lp[:-1].gather(-1, nxt[:, None])[:, 0].cpu().numpy()

    # ---------------- E4a mask configurations (experts routed in both engine and HF clean runs)
    rng = random.Random(9)
    mask_cfgs = {}  # (unit, name) -> list of (l, e)
    for i, u in enumerate(units):
        T = len(u["clean"])
        fin = lambda l: sorted(set(int(e) for e in r0.route_idx[l, i]) & set(int(e) for e in hf_st[(i, "clean")][("route", l)][-1]))
        anyp = lambda l: sorted(set(int(e) for e in np.unique(dg0["route_all"][l][0][i, :T])) & set(int(e) for e in np.unique(hf_st[(i, "clean")][("route", l)])))
        top1 = int(r0.route_idx[8, i, 0])
        mask_cfgs[(i, "single_l8")] = [(8, top1)]
        f3, f10 = fin(3), fin(10)
        mask_cfgs[(i, "multi_layer")] = [(3, e) for e in rng.sample(f3, 2)] + [(10, rng.choice(f10))]
        pool = [(l, e) for l in range(L) for e in anyp(l)]
        mask_cfgs[(i, "random5")] = sorted(rng.sample(pool, 5))
        mask_cfgs[(i, "allfinal_l12")] = [(12, e) for e in fin(12)]
        cand = [e for e in anyp(6) if e not in set(int(x) for x in r0.route_idx[6, i]) and e not in set(int(x) for x in hf_st[(i, "clean")][("route", 6)][-1])]
        mask_cfgs[(i, "unrouted_l6")] = [(6, rng.choice(cand))] if cand else []
        mask_cfgs[(i, "many_l5")] = [(5, e) for e in sorted(rng.sample(range(NE), 20))]
    mnames = ["single_l8", "multi_layer", "random5", "allfinal_l12", "unrouted_l6", "many_l5"]
    rows_a = []  # (unit, name, pos, mode)
    pre1 = [PrefillSpec(u["clean"], u["true"], u["foil"]) for u in units]  # unmasked copies (batch independence)
    for i, u in enumerate(units):
        for nm in mnames:
            if not mask_cfgs[(i, nm)]:
                continue
            for pos in ("all", "final"):
                for mode in ("reroute", "zero"):
                    rows_a.append((i, nm, pos, mode))
                    pre1.append(PrefillSpec(u["clean"], u["true"], u["foil"], route_mask=tuple(mask_cfgs[(i, nm)]),
                                            route_mask_pos=pos, route_mask_mode=mode))
    diag1 = DiagSpec(token_logprobs=True, router_logits_final=True, contrib_final_vectors=CFV_LAYERS)
    r1 = eng.run(pre1, [], record_routing=True, diag=diag1)
    dg1 = r1.extra["diag"]
    log(f"engine route-mask pass: {len(pre1)} rows")
    # HF masked forwards
    hf_mask = {}
    for j, (i, nm, pos, mode) in enumerate(rows_a):
        u = units[i]
        rec = {}
        lg = hf.fwd(u["clean"], hf.gate_hooks(mask_cfgs[(i, nm)], pos, mode, len(u["clean"]), rec))
        hf_mask[j] = (hf_delta(lg, u), hf_logprobs(lg, u["clean"]), rec)
    # comparisons
    am = {}
    d1 = r1.delta
    base_e = d1[:n]
    base_h = np.array([hf_delta(hf_lg[(i, "clean")], units[i]) for i in range(n)])
    am["baseline_delta_engine_vs_hf"] = _stats(base_e - base_h)
    am["batch_independence_unmasked_rows_vs_pass0"] = _stats(d1[:n] - r0.delta[:n])
    lp_base_diff = []
    for i in range(n):
        T = len(units[i]["clean"])
        lp_base_diff.append(dg1["token_logprobs"][i, : T - 1] - hf_logprobs(hf_lg[(i, "clean")], units[i]["clean"]))
    am["baseline_token_logprob_engine_vs_hf"] = _stats(np.concatenate(lp_base_diff))
    per = {}
    inv = {"reroute_masked_absent_final": [], "zero_first_layer_routing_unchanged": [], "zero_masked_slots_cnorm_zero": [],
           "zero_masked_slot_vectors_zero": [], "final_only_earlier_positions_identical": [], "routing_agree_masked_layers_vs_hf": []}
    unrouted_zero_identity, unrouted_reroute_ratio_err = [], []
    for pos in ("all", "final"):
        for mode in ("reroute", "zero"):
            js = [j for j, r in enumerate(rows_a) if r[2] == pos and r[3] == mode]
            de = np.array([d1[n + j] for j in js])
            dh = np.array([hf_mask[j][0] for j in js])
            be = np.array([base_e[rows_a[j][0]] for j in js])
            bh = np.array([base_h[rows_a[j][0]] for j in js])
            lpd = np.concatenate([dg1["token_logprobs"][n + j, : len(units[rows_a[j][0]]["clean"]) - 1] - hf_mask[j][1] for j in js])
            st = _stats(de - dh)
            st["effect_corr"] = _corr(de - be, dh - bh)
            st["mean_effect_engine"] = float((de - be).mean())
            st["mean_effect_hf"] = float((dh - bh).mean())
            st["token_logprob_engine_vs_hf"] = _stats(lpd)
            per[f"{pos}:{mode}"] = st
    for j, (i, nm, pos, mode) in enumerate(rows_a):
        row = n + j
        msk = mask_cfgs[(i, nm)]
        by_l = {}
        for l, e in msk:
            by_l.setdefault(l, set()).add(e)
        for l, es in by_l.items():
            ridx = set(int(e) for e in r1.route_idx[l, row])
            inv["routing_agree_masked_layers_vs_hf"].append(ridx == set(int(e) for e in hf_mask[j][2][l]))
            if mode == "reroute":
                inv["reroute_masked_absent_final"].append(len(ridx & es) == 0)
            else:
                slots = [s_ for s_ in range(K) if int(r1.route_idx[l, row, s_]) in es]
                inv["zero_masked_slots_cnorm_zero"].append(all(r1.route_cnorm[l, row, s_] == 0 for s_ in slots))
                if l in CFV_LAYERS:
                    inv["zero_masked_slot_vectors_zero"].append(all(float(dg1["contrib_final_vectors"][l][row, s_].abs().max()) == 0 for s_ in slots))
        if mode == "zero":
            l0 = min(by_l)
            inv["zero_first_layer_routing_unchanged"].append(bool((r1.route_idx[l0, row] == r1.route_idx[l0, i]).all()
                                                                   and (r1.route_w[l0, row] == r1.route_w[l0, i]).all()))
        if pos == "final":
            T = len(units[i]["clean"])
            a, b = dg1["token_logprobs"][row, : T - 1], dg1["token_logprobs"][i, : T - 1]
            inv["final_only_earlier_positions_identical"].append(bool(np.array_equal(a, b)))
        if nm == "unrouted_l6" and pos == "final":
            if mode == "zero":
                unrouted_zero_identity.append(float(d1[row] - d1[i]))
            else:  # OLMoE: indices unchanged, weights inflated by 1 / (1 - p_masked) (before the bf16 cast)
                (l, e), = msk
                if (r1.route_idx[l, row] == r1.route_idx[l, i]).all():
                    pm = float(torch.softmax(torch.tensor(dg1["router_logits_final"][l, i]), -1)[e])
                    ratio = r1.route_w[l, row] / r1.route_w[l, i]
                    unrouted_reroute_ratio_err.append(float(np.abs(ratio * (1 - pm) - 1).max()))
    am["per_pos_mode"] = per
    am["all_vs_hf"] = _stats(np.array([d1[n + j] - hf_mask[j][0] for j in range(len(rows_a))]))
    am["invariants"] = {k: {"n": len(v), "frac_true": float(np.mean(v)) if v else None} for k, v in inv.items()}
    am["invariants"]["final_unrouted_zero_identity_delta"] = _stats(unrouted_zero_identity)
    am["invariants"]["final_unrouted_reroute_weight_ratio_relerr_max"] = float(max(unrouted_reroute_ratio_err)) if unrouted_reroute_ratio_err else None
    am["invariants"]["final_unrouted_reroute_n"] = len(unrouted_reroute_ratio_err)
    am["n_rows"] = len(rows_a)
    # spawn on a masked parent must raise
    try:
        eng.run([PrefillSpec(units[0]["clean"], units[0]["true"], units[0]["foil"], route_mask=((3, 0),)),
                 PrefillSpec(units[0]["corrupt"], units[0]["true"], units[0]["foil"])],
                [SpawnSpec(4, 0, 1, "layer")], record_routing=False)
        am["spawn_on_masked_parent_raises"] = False
    except ValueError:
        am["spawn_on_masked_parent_raises"] = True
    rep["route_mask"] = am
    log("route mask: " + json.dumps({k: v for k, v in am.items() if k != "per_pos_mode"})[:1500])

    # ---------------- E4c contribution vectors
    cf = {}
    sum_d, rel_d, norm_d, hf_rel = [], [], [], []
    for r_, dg_ in ((r0, dg0), (r1, dg1)):
        for l in CFV_LAYERS:
            cv = dg_["contrib_final_vectors"][l]  # [B, k, H]
            mo = dg_["moe_out_final_fp32"][l]
            d_ = (cv.sum(1) - mo).abs()
            sum_d.append(float(d_.max()))
            rel_d.append(float((d_.norm(dim=-1) / mo.norm(dim=-1).clamp_min(1e-12)).max()))
            norm_d.append(float(np.abs(cv.norm(dim=-1).numpy() - r_.route_cnorm[l]).max()))
    for i in range(n):  # clean rows of pass 0 vs HF contributions (experts routed in both)
        for l in CFV_LAYERS:
            hc = hf_st[(i, "clean")][("c", l)]
            for s_ in range(K):
                e = int(r0.route_idx[l, i, s_])
                if e in hc:
                    a = dg0["contrib_final_vectors"][l][i, s_]
                    b = hc[e].float().cpu()
                    hf_rel.append(float((a - b).norm() / b.norm().clamp_min(1e-12)))
    cf["sum_vs_moe_out_maxabs"] = float(max(sum_d))
    cf["sum_vs_moe_out_max_relnorm"] = float(max(rel_d))
    cf["norm_vs_route_cnorm_maxabs"] = float(max(norm_d))
    cf["vs_hf_contrib_relnorm"] = {"n": len(hf_rel), "max": float(max(hf_rel)), "median": float(np.median(hf_rel))}
    cf["shape"] = list(dg0["contrib_final_vectors"][CFV_LAYERS[0]].shape)
    rep["contrib_final_vectors"] = cf
    log("contrib vectors: " + json.dumps(cf))

    # ---------------- E4b head steps
    spawns, tags = [], []

    def add(sp, tag):
        spawns.append(sp)
        tags.append(tag)

    hf_cfgs = {}  # (unit, direction, name) -> steps (neutral format) evaluated with HF
    one_heads = [(4, 3), (9, 11), (13, 0)]
    for i, u in enumerate(units):
        for direction in ("denoise", "noise"):
            par, src = (n + i, i) if direction == "denoise" else (i, n + i)
            src_run = "clean" if direction == "denoise" else "corrupt"
            common = lambda l: sorted(set(int(e) for e in r0.route_idx[l, src]) & set(hf_st[(i, src_run)][("c", l)].keys()))
            for (l, h) in one_heads:
                hf_cfgs[(i, direction, f"head_{l}_{h}")] = [(l, "heads", (h,), ())]
                add(SpawnSpec(l, par, src, "multi", steps=((l, "attn_head", (h,)),)), ("m", i, direction, f"head_{l}_{h}"))
                add(SpawnSpec(l, par, src, "attn_head", expert=h), ("k", i, direction, f"head_{l}_{h}"))
            l, h = one_heads[0]
            add(SpawnSpec(l, par, src, "multi", steps=((l, "heads_experts", ((h,), ())),)), ("he_noexp", i, direction, None))
            ex2 = tuple(common(7)[:2])
            add(SpawnSpec(7, par, src, "multi", steps=((7, "heads_experts", ((), ex2)),)), ("he_noheads", i, direction, None))
            add(SpawnSpec(7, par, src, "multi", steps=((7, "coalition_set", ex2),)), ("coal_ref", i, direction, None))
            for l in (5, 12):
                hf_cfgs[(i, direction, f"allheads_first_{l}")] = [(l, "heads", ALLH, ())]
                add(SpawnSpec(l, par, src, "multi", steps=((l, "attn_head", ALLH),)), ("m", i, direction, f"allheads_first_{l}"))
                add(SpawnSpec(l, par, src, "attn_layer"), ("k", i, direction, f"allheads_first_{l}"))
            for l in (6, 11):
                hf_cfgs[(i, direction, f"allheads_later_{l}")] = [(2, "moe", (), ()), (l, "heads", ALLH, ())]
                add(SpawnSpec(2, par, src, "multi", steps=((2, "layer", None), (l, "attn_head", ALLH))), ("m", i, direction, f"allheads_later_{l}"))
                add(SpawnSpec(2, par, src, "multi", steps=((2, "layer", None), (l, "attn_layer", None))), ("k", i, direction, f"allheads_later_{l}"))
            hf_cfgs[(i, direction, "allheads_all_layers")] = [(l, "heads", ALLH, ()) for l in range(L)]
            add(SpawnSpec(0, par, src, "multi", steps=tuple((l, "attn_head", ALLH) for l in range(L))), ("m", i, direction, "allheads_all_layers"))
            add(SpawnSpec(0, par, src, "multi", steps=tuple((l, "attn_layer", None) for l in range(L))), ("k", i, direction, "allheads_all_layers"))
            # random mixed configurations (first-step kind cycles through heads / he / coal / moe / attn)
            r_ = random.Random(1000 * i + (0 if direction == "denoise" else 1))
            firsts = ["heads", "he", "coal", "heads", "he", "attn", "moe", "he"]
            for c in range(args.n_mixed):
                ns = r_.choice([2, 3, 4, 5])
                layers = sorted(r_.sample(range(L), ns))
                steps = []
                for jj, l in enumerate(layers):
                    kind = firsts[c % len(firsts)] if jj == 0 else r_.choice(["heads", "he", "he", "coal", "moe", "attn"])
                    heads = tuple(sorted(r_.sample(range(nH), r_.randint(1, 4)))) if kind in ("heads", "he") else ()
                    cm = common(l)
                    experts = tuple(sorted(r_.sample(cm, min(len(cm), r_.randint(1, 3))))) if kind in ("coal", "he") else ()
                    if kind == "coal" and not experts:
                        kind = "moe"
                    steps.append((l, kind, heads, experts))
                hf_cfgs[(i, direction, f"mixed{c}")] = steps
                add(SpawnSpec(steps[0][0], par, src, "multi", steps=eng_steps(steps)), ("m", i, direction, f"mixed{c}"))
        # identity: parent = source = clean, all heads at every layer
        add(SpawnSpec(0, i, i, "multi", steps=tuple((l, "attn_head", ALLH) for l in range(L))), ("identity_heads", i, None, None))
        add(SpawnSpec(0, i, i, "multi", steps=tuple((l, "attn_layer", None) for l in range(L))), ("identity_attn", i, None, None))
    torch.cuda.reset_peak_memory_stats()
    rb = eng.run(pre0, spawns, record_routing=True, diag=DiagSpec(spawn_vectors=True))
    rep["heads_pass_peak_mem_GB"] = float(torch.cuda.max_memory_allocated() / 2**30)
    sd, d0 = rb.sp_delta, rb.delta
    sv = rb.extra["diag"]["spawn_v"]
    idx = {t: j for j, t in enumerate(tags)}
    log(f"engine head pass: {len(spawns)} spawn rows")
    # HF patched forwards
    hfp = {}
    for (i, direction, name), steps in hf_cfgs.items():
        u = units[i]
        base, srcr = ("corrupt", "clean") if direction == "denoise" else ("clean", "corrupt")
        lg = hf.fwd(u[base], hf.patch_hooks(steps, hf_st[(i, srcr)]))
        hfp[(i, direction, name)] = hf_delta(lg, u)
    hb = {}

    def v0(j):
        v = sv.get(j)
        return v[0] if isinstance(v, list) else v

    # exact invariants
    ex = {}
    z, vz = [], []
    for i in range(n):
        for direction in ("denoise", "noise"):
            for (l, h) in one_heads:
                a, b = idx[("m", i, direction, f"head_{l}_{h}")], idx[("k", i, direction, f"head_{l}_{h}")]
                z.append(float(sd[a] - sd[b]))
                vz.append(float((v0(a) - v0(b)).abs().max()))
    ex["one_head_step_vs_attn_head_kind_delta"] = _stats(z)
    ex["one_head_step_vs_attn_head_kind_vec_maxabs"] = float(max(vz))
    z, vz = [], []
    for i in range(n):
        for direction in ("denoise", "noise"):
            l, h = one_heads[0]
            a, b = idx[("he_noexp", i, direction, None)], idx[("k", i, direction, f"head_{l}_{h}")]
            z.append(float(sd[a] - sd[b]))
            vz.append(float((v0(a) - v0(b)).abs().max()))
    ex["heads_experts_noexperts_vs_attn_head_kind_delta"] = _stats(z)
    ex["heads_experts_noexperts_vs_attn_head_kind_vec_maxabs"] = float(max(vz))
    z, vz = [], []
    for i in range(n):
        for direction in ("denoise", "noise"):
            a, b = idx[("he_noheads", i, direction, None)], idx[("coal_ref", i, direction, None)]
            z.append(float(sd[a] - sd[b]))
            vz.append(float((v0(a) - v0(b)).abs().max()))
    ex["heads_experts_noheads_vs_coalition_set_delta"] = _stats(z)
    ex["heads_experts_noheads_vs_coalition_set_vec_maxabs"] = float(max(vz))
    for name in ("allheads_first_5", "allheads_first_12", "allheads_later_6", "allheads_later_11", "allheads_all_layers"):
        z, vz = [], []
        for i in range(n):
            for direction in ("denoise", "noise"):
                a, b = idx[("m", i, direction, name)], idx[("k", i, direction, name)]
                z.append(float(sd[a] - sd[b]))
                va, vb = sv.get(a), sv.get(b)
                va = va if isinstance(va, list) or va is None else [va]
                vb = vb if isinstance(vb, list) or vb is None else [vb]
                if isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb):
                    vz.append(max(float((x - y).abs().max() / max(float(y.norm()), 1e-6)) for x, y in zip(va, vb)))
        ex[f"{name}_heads_vs_attn_layer_delta"] = _stats(z)
        if vz:
            ex[f"{name}_heads_vs_attn_layer_vec_max_rel"] = float(max(vz))
    z = []
    for i in range(n):
        for direction, ref in (("denoise", i), ("noise", n + i)):
            z.append(float(sd[idx[("m", i, direction, "allheads_all_layers")]] - d0[ref]))
    ex["allheads_all_layers_vs_source_delta"] = _stats(z)
    ex["identity_heads_minus_clean_delta"] = _stats([float(sd[idx[("identity_heads", i, None, None)]] - d0[i]) for i in range(n)])
    ex["identity_attn_minus_clean_delta"] = _stats([float(sd[idx[("identity_attn", i, None, None)]] - d0[i]) for i in range(n)])
    ok = []
    for (i, direction, name), steps in hf_cfgs.items():
        j = idx[("m", i, direction, name)]
        v = sv.get(j)
        ok.append(isinstance(v, list) and len(v) == len(steps) and len(rb.extra["multi_vnorm"].get(j, [])) == len(steps) - 1)
    ex["spawn_vectors_one_per_step_frac"] = float(np.mean(ok))
    rep["head_steps_invariants"] = ex
    # vs HF
    vsh = {}
    groups = {"one_head": [k for k in hf_cfgs if k[2].startswith("head_")],
              "allheads_first": [k for k in hf_cfgs if k[2].startswith("allheads_first")],
              "allheads_later": [k for k in hf_cfgs if k[2].startswith("allheads_later")],
              "allheads_all_layers": [k for k in hf_cfgs if k[2] == "allheads_all_layers"],
              "mixed": [k for k in hf_cfgs if k[2].startswith("mixed")]}
    for direction in ("denoise", "noise"):
        for gname, keys in groups.items():
            ks = [k for k in keys if k[1] == direction]
            en = np.array([sd[idx[("m",) + k]] for k in ks])
            hv = np.array([hfp[k] for k in ks])
            be = np.array([d0[n + k[0]] if direction == "denoise" else d0[k[0]] for k in ks])
            bh = np.array([hf_delta(hf_lg[(k[0], "corrupt" if direction == "denoise" else "clean")], units[k[0]]) for k in ks])
            st = _stats(en - hv)
            st["effect_corr"] = _corr(en - be, hv - bh)
            st["mean_effect_engine"] = float((en - be).mean())
            st["mean_effect_hf"] = float((hv - bh).mean())
            vsh[f"{direction}:{gname}"] = st
    allk = list(hf_cfgs)
    vsh["all"] = _stats([sd[idx[("m",) + k]] - hfp[k] for k in allk])
    ee = [sd[idx[("m",) + k]] - (d0[n + k[0]] if k[1] == "denoise" else d0[k[0]]) for k in allk]
    hh = [hfp[k] - hf_delta(hf_lg[(k[0], "corrupt" if k[1] == "denoise" else "clean")], units[k[0]]) for k in allk]
    vsh["all_effect_corr"] = _corr(ee, hh)
    # calibration: the HF-vs-engine spread of the established attn_layer kind on the same units (allheads_first rows)
    cal = [sd[idx[("k",) + k]] - hfp[k] for k in groups["allheads_first"]]
    vsh["calibration_attn_layer_kind_vs_hf_heads_all"] = _stats(cal)
    rep["head_steps_vs_hf"] = vsh
    rep["head_pass_spawn_rows"] = len(spawns)
    rep["mixed_configs_example"] = {f"{k[0]}:{k[1]}:{k[2]}": [list(map(lambda x: list(x) if isinstance(x, tuple) else x, st)) for st in v]
                                    for k, v in list(hf_cfgs.items()) if k[2].startswith("mixed") and k[0] == 0}
    log("head steps: " + json.dumps(ex)[:1500])
    log("vs HF: " + json.dumps({k: (v["maxabs"], v["meanabs"], v.get("effect_corr")) if isinstance(v, dict) and "maxabs" in v else v for k, v in vsh.items()}))
    del hf
    torch.cuda.empty_cache()

    # ---------------- stress: many multi rows with head steps
    if args.stress:
        rs_ = random.Random(5)
        st_sp, st_par = [], []
        for s_ in range(args.stress):
            i = s_ % n
            direction = "denoise" if (s_ // n) % 2 == 0 else "noise"
            par, src = (n + i, i) if direction == "denoise" else (i, n + i)
            steps = []
            for l in range(L):
                kind = rs_.choice(["attn_head", "heads_experts", "coalition_set", "attn_head"])
                hs_ = tuple(sorted(rs_.sample(range(nH), rs_.randint(1, 3))))
                es_ = tuple(sorted(rs_.sample([int(e) for e in r0.route_idx[l, src]], rs_.randint(1, 3))))
                steps.append((l, kind, hs_ if kind == "attn_head" else (hs_, es_) if kind == "heads_experts" else es_))
            st_sp.append(SpawnSpec(0, par, src, "multi", steps=tuple(steps)))
            st_par.append(par)
        torch.cuda.reset_peak_memory_stats()
        t2 = time.time()
        rs = eng.run(pre0, st_sp, record_routing=False)
        t_big = time.time() - t2
        peak = torch.cuda.max_memory_allocated() / 2**30
        small = list(range(0, args.stress, max(1, args.stress // 200)))
        rs2 = eng.run(pre0, [st_sp[j] for j in small], record_routing=False)
        rep["stress"] = {"rows": args.stress, "steps_per_row": L, "pass_s": t_big, "peak_mem_GB": float(peak),
                         "batch_independence_delta": _stats(rs.sp_delta[small] - rs2.sp_delta)}
        log(f"stress: {rep['stress']}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
