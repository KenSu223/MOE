"""ext5 engine verification on OLMoE-1B-7B-0125: metrics (F5), attn_head (F2), coalition_set and multi (F1).

Two engine passes on the same cases (the second needs the routing recorded by the first) and one transformers
reference (small model fully on the GPU, eager attention):

HF side
  * clean / noised full logits at the final position (metrics reference: log-softmax, rank, KL(noised || clean));
  * attn_head: forward pre-hook on self_attn.o_proj that replaces, at the final position, the head-h slice of the
    o_proj INPUT by the clean run's (= zero all but head h of the difference; o_proj is linear so this equals
    Attn_noised + W_o[:, h](H_h_clean - H_h_noised) up to bf16 rounding);
  * multi: MoE-output replace hooks at several layers simultaneously (the `multi` semantics: at every step layer the
    patched component is set to its clean value, everything else comes from the row's own computation).
Engine side
  pass 1 (metrics=True, spawn_vectors): layer at every layer, attn_layer + all attn_head at HEAD_LAYERS (+ identity
    heads on the clean run), zero rows, multi (layer, layer) at MULTI_PAIRS and MULTI_TRIPLE, multi with
    coalition_set S = all experts at the same pairs, single-step multi.
  pass 2 (spawn_vectors): coalition_set with S = clean set / clean ∪ noised / all / {e} against coalition_clean /
    layer / expert, single-step multi coalition_set(S = clean) against coalition_clean, and mixed multi steps.

Usage: python scripts/ext5_engine_verify.py [n_cases=20]  -> results/verify_ext5_engine_olmoe.json
"""
import json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, torch
import torch.nn.functional as F
from moetrace.verify import pick_cases, _hf_load, _moe_module
from moetrace.engine import Engine, PrefillSpec, SpawnSpec, DiagSpec
from moetrace.noise import noise_draw

REPO = "allenai/OLMoE-1B-7B-0125"
HEAD_LAYERS = (4, 10)
COAL_LAYERS = (4, 10)
MULTI_PAIRS = ((4, 10), (2, 8), (6, 12))
MULTI_TRIPLE = (4, 8, 12)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _with_replaced(out, new_final):
    o = _out_tensor(out).clone()
    o[0, -1] = new_final
    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o


@torch.no_grad()
def hf_reference(cases, sigma_mult=3.0):
    model = _hf_load(REPO)
    dev = "cuda"
    emb = model.get_input_embeddings()
    sigma = sigma_mult * float(emb.weight.float().std().item())
    cfg = model.config
    L, nH = cfg.num_hidden_layers, cfg.num_attention_heads
    D = cfg.hidden_size // nH

    def embeds(c, noised):
        e = emb(torch.tensor([c.ids], device=dev))
        if noised:
            eps = noise_draw(c.case_id, len(c.subject_pos), e.shape[-1], sigma).to(dev)
            e = e.clone()
            pos = torch.tensor(c.subject_pos, device=dev)
            e[0, pos] = (e[0, pos].float() + eps).to(e.dtype)
        return e

    def fwd(e):
        return model(inputs_embeds=e).logits[0, -1].float()

    store = {}

    def rec(key):
        def hook(mod, args, out):
            store[key] = _out_tensor(out)[0, -1].detach().clone()
        return hook

    def rec_pre(key):
        def hook(mod, args):
            store[key] = args[0][0, -1].detach().clone()
        return hook

    def rep(key):
        def hook(mod, args, out):
            return _with_replaced(out, store[key])
        return hook

    def rep_head_pre(key, h):
        def hook(mod, args):
            x = args[0].clone()
            x[0, -1, h * D : (h + 1) * D] = store[key][h * D : (h + 1) * D]
            return (x,) + tuple(args[1:])
        return hook

    out = {"logits_clean": [], "logits_noised": [], "head_patch": {}, "oin_patch": {}, "multi_patch": {}, "nH": nH, "D": D}
    for i, c in enumerate(cases):
        ti, fi = c.true_id, c.foil_id
        hs = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(rec_pre(("oin", l))) for l in HEAD_LAYERS]
        hs += [_moe_module(model, l).register_forward_hook(rec(("moe", l))) for l in range(L)]
        lg_c = fwd(embeds(c, False))
        for h in hs:
            h.remove()
        lg_n = fwd(embeds(c, True))
        out["logits_clean"].append(lg_c.cpu().numpy())
        out["logits_noised"].append(lg_n.cpu().numpy())
        for l in HEAD_LAYERS:
            for h in range(nH):
                hh = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(rep_head_pre(("oin", l), h))
                lg = fwd(embeds(c, True))
                hh.remove()
                out["head_patch"][(i, l, h)] = float(lg[ti] - lg[fi])
            # whole o_proj input replaced (= attn_layer up to bf16 rounding of o_proj)
            def rep_all(mod, args, key=("oin", l)):
                x = args[0].clone()
                x[0, -1] = store[key]
                return (x,) + tuple(args[1:])
            hh = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(rep_all)
            lg = fwd(embeds(c, True))
            hh.remove()
            out["oin_patch"][(i, l)] = float(lg[ti] - lg[fi])
        for layers in MULTI_PAIRS + (MULTI_TRIPLE,):
            hs = [_moe_module(model, l).register_forward_hook(rep(("moe", l))) for l in layers]
            lg = fwd(embeds(c, True))
            for h in hs:
                h.remove()
            out["multi_patch"][(i, layers)] = float(lg[ti] - lg[fi])
        if i % 5 == 4:
            log(f"HF: {i + 1}/{len(cases)} cases done")
    del model
    torch.cuda.empty_cache()
    return out


def build_prefill(eng, cases):
    Hd = eng.hidden
    sigma = 3.0 * eng.embed_std
    pre = [PrefillSpec(c.ids, c.true_id, c.foil_id) for c in cases]
    pre += [PrefillSpec(c.ids, c.true_id, c.foil_id, list(c.subject_pos), noise_draw(c.case_id, len(c.subject_pos), Hd, sigma))
            for c in cases]
    return pre


def pass1(eng, cases):
    L, nH, E = eng.spec.n_layers, eng.spec.n_heads, eng.spec.n_experts
    n = len(cases)
    ALL = tuple(range(E))
    spawns, tags = [], []

    def add(sp, tag):
        spawns.append(sp)
        tags.append(tag)

    for i in range(n):
        for l in range(L):
            add(SpawnSpec(l, n + i, i, "layer"), ("layer", i, l, None))
        for l in HEAD_LAYERS:
            add(SpawnSpec(l, n + i, i, "attn_layer"), ("attn_layer", i, l, None))
            for h in range(nH):
                add(SpawnSpec(l, n + i, i, "attn_head", expert=h), ("attn_head", i, l, h))
            for h in (0, nH // 2):
                add(SpawnSpec(l, i, i, "attn_head", expert=h), ("id_attn_head", i, l, h))
        for l in (0, L // 2, L - 1):
            add(SpawnSpec(l, n + i, i, "zero"), ("zero_noised", i, l, None))
            add(SpawnSpec(l, i, i, "zero"), ("zero_clean", i, l, None))
        for (l1, l2) in MULTI_PAIRS:
            add(SpawnSpec(l1, n + i, i, "multi", steps=((l1, "layer", None), (l2, "layer", None))), ("multi_layer", i, (l1, l2), None))
            add(SpawnSpec(l1, n + i, i, "multi", steps=((l1, "coalition_set", ALL), (l2, "coalition_set", ALL))), ("multi_setall", i, (l1, l2), None))
        l1, l2, l3 = MULTI_TRIPLE
        add(SpawnSpec(l1, n + i, i, "multi", steps=((l1, "layer", None), (l2, "layer", None), (l3, "layer", None))), ("multi_layer", i, MULTI_TRIPLE, None))
        add(SpawnSpec(4, n + i, i, "multi", steps=((4, "layer", None),)), ("multi1_layer", i, 4, None))
    diag = DiagSpec(attn_heads_final=HEAD_LAYERS, attn_out_final=True, spawn_vectors=True)
    t0 = time.time()
    res = eng.run(build_prefill(eng, cases), spawns, record_routing=True, log=log, diag=diag, metrics=True)
    return res, tags, time.time() - t0


def pass2(eng, cases, route_idx):
    """Coalition-set checks; route_idx [L, 2n, k] from pass 1 (clean rows 0..n-1, noised n..2n-1)."""
    n = len(cases)
    E = eng.spec.n_experts
    ALL = tuple(range(E))
    spawns, tags = [], []

    def add(sp, tag):
        spawns.append(sp)
        tags.append(tag)

    for i in range(n):
        for l in COAL_LAYERS:
            cs = tuple(sorted(int(e) for e in route_idx[l, i]))
            ns = tuple(sorted(int(e) for e in route_idx[l, n + i]))
            un = tuple(sorted(set(cs) | set(ns)))
            add(SpawnSpec(l, n + i, i, "layer"), ("layer", i, l, None))
            add(SpawnSpec(l, n + i, i, "coalition_clean"), ("coalition_clean", i, l, None))
            add(SpawnSpec(l, n + i, i, "coalition_union"), ("coalition_union", i, l, None))
            add(SpawnSpec(l, n + i, i, "coalition_set", experts=cs), ("set_clean", i, l, None))
            add(SpawnSpec(l, n + i, i, "coalition_set", experts=un), ("set_union", i, l, None))
            add(SpawnSpec(l, n + i, i, "coalition_set", experts=ALL), ("set_all", i, l, None))
            for e in cs:
                add(SpawnSpec(l, n + i, i, "expert", expert=e), ("expert", i, l, e))
                add(SpawnSpec(l, n + i, i, "coalition_set", experts=(e,)), ("set_single", i, l, e))
            add(SpawnSpec(l, n + i, i, "multi", steps=((l, "coalition_set", cs),)), ("multi1_set_clean", i, l, None))
            add(SpawnSpec(l, n + i, i, "multi", steps=((l, "expert", cs[0]),)), ("multi1_expert", i, l, cs[0]))
            add(SpawnSpec(l, n + i, i, "multi", steps=((l, "coalition_clean", None),)), ("multi1_coal_clean", i, l, None))
        l1, l2 = 4, 10
        add(SpawnSpec(l1, n + i, i, "multi", steps=((l1, "layer", None), (l2, "layer", None))), ("m_layer_layer", i, (l1, l2), None))
        add(SpawnSpec(l1, n + i, i, "multi", steps=((l1, "layer", None), (l2, "coalition_set", ALL))), ("m_layer_all", i, (l1, l2), None))
        add(SpawnSpec(l1, n + i, i, "multi", steps=((l1, "coalition_set", ALL), (l2, "layer", None))), ("m_all_layer", i, (l1, l2), None))
        cs2 = tuple(sorted(int(e) for e in route_idx[l2, i]))
        add(SpawnSpec(l1, n + i, i, "multi", steps=((l1, "layer", None), (l2, "coalition_set", cs2))), ("m_layer_setclean", i, (l1, l2), None))
        add(SpawnSpec(l1, n + i, i, "multi", steps=((l1, "layer", None), (l2, "coalition_clean", None))), ("m_layer_coalclean", i, (l1, l2), None))
    diag = DiagSpec(spawn_vectors=True)
    t0 = time.time()
    res = eng.run(build_prefill(eng, cases), spawns, record_routing=True, log=log, diag=diag)
    return res, tags, time.time() - t0


def _idx(tags, kind):
    return [j for j, t in enumerate(tags) if t[0] == kind]


def _by_key(tags, kind, keyf):
    return {keyf(tags[j]): j for j in _idx(tags, kind)}


def _vec(dg, j):
    v = dg["spawn_v"][j]
    return v[0] if isinstance(v, list) else v


def _stats(z: np.ndarray) -> dict:
    z = np.asarray(z, dtype=np.float64)
    return {"n": int(len(z)), "maxabs": float(np.abs(z).max()), "meanabs": float(np.abs(z).mean()),
            "frac_equal": float((z == 0).mean()), "frac_within_0.1": float((np.abs(z) < 0.1).mean())}


def main():
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    cases = pick_cases(REPO, n_cases)
    n = len(cases)
    log(f"{n} cases; first: {cases[0].prompt!r} -> {cases[0].true_str} / {cases[0].foil_str}")
    ref = hf_reference(cases)
    log("HF reference done")
    eng = Engine(REPO)
    L, nH, E = eng.spec.n_layers, eng.spec.n_heads, eng.spec.n_experts
    res, tags, t1 = pass1(eng, cases)
    rep = {"n_cases": n, "head_layers": list(HEAD_LAYERS), "multi_pairs": [list(p) for p in MULTI_PAIRS], "multi_triple": list(MULTI_TRIPLE),
           "pass1_s": t1, "pass1_metrics_s": res.extra["metrics_s"], "pass1_spawn_rows": len(tags)}
    d, sd = res.delta, res.sp_delta
    dc, dn = d[:n], d[n:]
    ti = np.array([c.true_id for c in cases])
    fi = np.array([c.foil_id for c in cases])
    lc, ln = np.stack(ref["logits_clean"]), np.stack(ref["logits_noised"])
    ar = np.arange(n)
    rep["delta_clean_maxdiff_vs_hf"] = float(np.abs(dc - (lc[ar, ti] - lc[ar, fi])).max())
    rep["delta_noised_maxdiff_vs_hf"] = float(np.abs(dn - (ln[ar, ti] - ln[ar, fi])).max())

    # ---------------- F5 metrics
    mp, ms = res.metrics_prefill, res.metrics_spawn
    lp_c = lc - np.log(np.exp(lc - lc.max(1, keepdims=True)).sum(1, keepdims=True)) - lc.max(1, keepdims=True)
    lp_n = ln - np.log(np.exp(ln - ln.max(1, keepdims=True)).sum(1, keepdims=True)) - ln.max(1, keepdims=True)
    m = {}
    m["logp_true_clean_maxdiff_vs_hf"] = float(np.abs(mp["logp_true"][:n] - lp_c[ar, ti]).max())
    m["logp_true_noised_maxdiff_vs_hf"] = float(np.abs(mp["logp_true"][n:] - lp_n[ar, ti]).max())
    m["logp_foil_clean_maxdiff_vs_hf"] = float(np.abs(mp["logp_foil"][:n] - lp_c[ar, fi]).max())
    m["p_true_clean_maxdiff_vs_hf"] = float(np.abs(mp["p_true"][:n] - np.exp(lp_c[ar, ti])).max())
    rank_hf = 1 + (lc > lc[ar, ti][:, None]).sum(1)
    m["rank_true_clean_agree_frac_vs_hf"] = float((mp["rank_true"][:n] == rank_hf).mean())
    m["rank_true_clean_maxabsdiff_vs_hf"] = int(np.abs(mp["rank_true"][:n].astype(int) - rank_hf).max())
    top = rank_hf <= 10
    m["rank_true_clean_agree_frac_vs_hf_hfrank_le10"] = float((mp["rank_true"][:n][top] == rank_hf[top]).mean()) if top.any() else None
    m["rank_true_clean_n_hfrank_le10"] = int(top.sum())
    m["rank_true_clean_maxabsdiff_vs_hf_hfrank_le10"] = int(np.abs(mp["rank_true"][:n][top].astype(int) - rank_hf[top]).max()) if top.any() else None
    m["rank_true_clean_hf_ranks"] = [int(r) for r in rank_hf]
    kl_hf = (np.exp(lp_n) * (lp_n - lp_c)).sum(1)
    m["kl_noised_to_clean_maxdiff_vs_hf"] = float(np.abs(mp["kl_to_clean"][n:] - kl_hf).max())
    m["kl_noised_to_clean_mean_engine"] = float(mp["kl_to_clean"][n:].mean())
    m["kl_noised_to_clean_mean_hf"] = float(kl_hf.mean())
    m["kl_clean_rows_max"] = float(np.abs(mp["kl_to_clean"][:n]).max())
    m["kl_min_all_rows"] = float(min(mp["kl_to_clean"].min(), ms["kl_to_clean"].min()))
    psum = np.concatenate([mp["p_true"] + mp["p_foil"], ms["p_true"] + ms["p_foil"]])
    m["p_true_plus_p_foil_max"] = float(psum.max())
    m["p_true_plus_p_foil_gt1_count"] = int((psum > 1.0).sum())
    lo_pre = mp["logp_true"] - mp["logp_foil"]
    lo_sp = ms["logp_true"] - ms["logp_foil"]
    m["delta_vs_logodds_prefill_maxdiff"] = float(np.abs(d - lo_pre).max())
    m["delta_vs_logodds_spawn_maxdiff"] = float(np.abs(sd - lo_sp).max())
    m["delta_vs_logodds_spawn_meanabsdiff"] = float(np.abs(sd - lo_sp).mean())
    r1 = mp["rank_true"] == 1
    t1_ = res.top1 == np.concatenate([ti, ti])
    m["rank1_iff_top1_frac"] = float((r1 == t1_).mean())
    jz = _idx(tags, "zero_clean")
    m["zero_on_clean_kl_max"] = float(ms["kl_to_clean"][jz].max())
    m["zero_on_clean_kl_mean"] = float(ms["kl_to_clean"][jz].mean())
    jz = _idx(tags, "zero_noised")
    m["zero_on_noised_kl_vs_noised_prefill_maxdiff"] = float(np.abs(ms["kl_to_clean"][jz] - mp["kl_to_clean"][n:][[tags[j][1] for j in jz]]).max())
    m["p_true_clean_mean"] = float(mp["p_true"][:n].mean())
    m["p_true_noised_mean"] = float(mp["p_true"][n:].mean())
    rep["metrics"] = m

    # ---------------- F2 attn_head
    dg = res.extra["diag"]
    a = {}
    lin_max, lin_rel, al_max, al_rel = [], [], [], []
    for l in HEAD_LAYERS:
        wo = eng.store.load_layer(l).wo.float().cpu()  # [H, nH*D]
        heads = dg["attn_heads_final"][l].float()  # [2n, nH, D]
        attn_out = dg["attn_out_final"][l].float()  # [2n, H]
        jh = _by_key(tags, "attn_head", lambda t: (t[1], t[2], t[3]))
        ja = _by_key(tags, "attn_layer", lambda t: (t[1], t[2]))
        for i in range(n):
            vsum = sum(_vec(dg, jh[(i, l, h)]) for h in range(nH))
            dh = (heads[i] - heads[n + i]).reshape(-1)
            lin = wo @ dh
            lin_max.append(float((vsum - lin).abs().max()))
            lin_rel.append(float((vsum - lin).norm() / lin.norm()))
            va = _vec(dg, ja[(i, l)])
            al_max.append(float((vsum - va).abs().max()))
            al_rel.append(float((vsum - va).norm() / va.norm()))
    a["headsum_vs_fp32_linear_maxabs"] = float(max(lin_max))
    a["headsum_vs_fp32_linear_max_relnorm"] = float(max(lin_rel))
    a["headsum_vs_attn_layer_vec_maxabs"] = float(max(al_max))
    a["headsum_vs_attn_layer_vec_mean_relnorm"] = float(np.mean(al_rel))
    # rescue additivity and HF comparison
    jh = _by_key(tags, "attn_head", lambda t: (t[1], t[2], t[3]))
    ja = _by_key(tags, "attn_layer", lambda t: (t[1], t[2]))
    sums, attn_r, diffs, re_, rh_ = [], [], [], [], []
    for l in HEAD_LAYERS:
        for i in range(n):
            rs = [sd[jh[(i, l, h)]] - dn[i] for h in range(nH)]
            sums.append(sum(rs))
            attn_r.append(sd[ja[(i, l)]] - dn[i])
            for h in range(nH):
                diffs.append(sd[jh[(i, l, h)]] - ref["head_patch"][(i, l, h)])
                re_.append(sd[jh[(i, l, h)]] - dn[i])
                rh_.append(ref["head_patch"][(i, l, h)] - (ln[i, ti[i]] - ln[i, fi[i]]))
    sums, attn_r = np.array(sums), np.array(attn_r)
    a["headsum_rescue_vs_attn_layer_rescue"] = _stats(sums - attn_r)
    a["headsum_rescue_vs_attn_layer_rescue_corr"] = float(np.corrcoef(sums, attn_r)[0, 1])
    a["headsum_rescue_mean"] = float(sums.mean())
    a["attn_layer_rescue_mean"] = float(attn_r.mean())
    a["attn_head_vs_hf"] = _stats(np.array(diffs))
    a["attn_head_vs_hf_frac_within_0.25"] = float((np.abs(diffs) < 0.25).mean())
    a["attn_head_rescue_corr_vs_hf"] = float(np.corrcoef(re_, rh_)[0, 1])
    a["attn_head_rescue_mean_engine"] = float(np.mean(re_))
    a["attn_head_rescue_mean_hf"] = float(np.mean(rh_))
    # calibration: layer kind vs HF on the same cases is in verify_olmoe.json; attn_layer vs HF whole-o_proj-input replace
    a["attn_layer_vs_hf_oin_replace"] = _stats(np.array([sd[ja[(i, l)]] - ref["oin_patch"][(i, l)] for l in HEAD_LAYERS for i in range(n)]))
    jid = _idx(tags, "id_attn_head")
    a["identity_attn_head_on_clean"] = _stats(np.array([sd[j] - dc[tags[j][1]] for j in jid]))
    rep["attn_head"] = a

    # ---------------- F1 multi (pass 1)
    mu = {}
    jm = _by_key(tags, "multi_layer", lambda t: (t[1], t[2]))
    diffs = np.array([sd[j] - ref["multi_patch"][(i, layers)] for (i, layers), j in jm.items()])
    mu["multi_layer_vs_hf"] = _stats(diffs)
    mu["multi_layer_vs_hf_frac_within_0.25"] = float((np.abs(diffs) < 0.25).mean())
    re_ = np.array([sd[j] - dn[i] for (i, layers), j in jm.items()])
    rh_ = np.array([ref["multi_patch"][(i, layers)] - (ln[i, ti[i]] - ln[i, fi[i]]) for (i, layers), j in jm.items()])
    mu["multi_layer_rescue_corr_vs_hf"] = float(np.corrcoef(re_, rh_)[0, 1])
    mu["multi_layer_rescue_mean_engine"] = float(re_.mean())
    mu["multi_layer_rescue_mean_hf"] = float(rh_.mean())
    jl = _by_key(tags, "layer", lambda t: (t[1], t[2]))
    # two-layer multi rescue vs single-layer rescues (information only)
    mu["multi_pair_rescue_minus_max_single_mean"] = float(np.mean([sd[j] - dn[i] - max(sd[jl[(i, layers[0])]] - dn[i], sd[jl[(i, layers[1])]] - dn[i])
                                                                   for (i, layers), j in jm.items() if len(layers) == 2]))
    js = _by_key(tags, "multi_setall", lambda t: (t[1], t[2]))
    z = np.array([sd[js[k]] - sd[jm[k]] for k in js])
    mu["multi_setall_vs_multi_layer_delta"] = _stats(z)
    vz = []
    for k in js:
        va, vb = dg["spawn_v"][js[k]], dg["spawn_v"][jm[k]]
        vz.append(max(float((x - y).abs().max()) for x, y in zip(va, vb)))
    mu["multi_setall_vs_multi_layer_vec_maxabs"] = float(max(vz))
    j1 = _by_key(tags, "multi1_layer", lambda t: t[1])
    z = np.array([sd[j1[i]] - sd[jl[(i, 4)]] for i in j1])
    mu["multi1_layer_vs_layer_delta"] = _stats(z)
    mu["multi1_layer_vs_layer_vec_maxabs"] = float(max(float((_vec(dg, j1[i]) - _vec(dg, jl[(i, 4)])).abs().max()) for i in j1))
    mu["multi_step_vnorm_example"] = res.extra["multi_vnorm"][list(jm.values())[0]]
    rep["multi_pass1"] = mu

    # ---------------- F1 coalition_set (pass 2)
    res2, tags2, t2 = pass2(eng, cases, res.route_idx)
    rep["pass2_s"] = t2
    rep["pass2_spawn_rows"] = len(tags2)
    dg2 = res2.extra["diag"]
    sd2 = res2.sp_delta
    cs = {}
    # the sets S handed to pass 2 come from pass 1's routing; where pass 2's in-pass clean/noised sets differ (cross-pass
    # bf16 noise flips near-tie routings) the invariants are not expected to hold, so they are also reported on the
    # matched (case, layer) pairs only
    same = {}
    for l in COAL_LAYERS:
        for i in range(n):
            same[(i, l)] = (sorted(res2.route_idx[l, i].tolist()) == sorted(res.route_idx[l, i].tolist())
                            and sorted(res2.route_idx[l, n + i].tolist()) == sorted(res.route_idx[l, n + i].tolist()))
    cs["routing_pass2_equals_pass1_pairs"] = f"{sum(same.values())}/{len(same)}"

    def cmp(kind_a, kind_b, keyf, name):
        ja_, jb_ = _by_key(tags2, kind_a, keyf), _by_key(tags2, kind_b, keyf)
        keys = [k for k in ja_ if k in jb_]
        z = np.array([sd2[ja_[k]] - sd2[jb_[k]] for k in keys])
        vz = np.array([float((_vec(dg2, ja_[k]) - _vec(dg2, jb_[k])).abs().max()) for k in keys])
        cs[name + "_delta"] = _stats(z)
        cs[name + "_vec_maxabs"] = float(vz.max())
        cs[name + "_vec_frac_exact"] = float((vz == 0).mean())
        ok = np.array([same[(k[0], k[1])] for k in keys])
        if ok.any():
            cs[name + "_matched_routing_delta"] = _stats(z[ok])
            cs[name + "_matched_routing_vec_maxabs"] = float(vz[ok].max())

    k2 = lambda t: (t[1], t[2])
    k3 = lambda t: (t[1], t[2], t[3])
    cmp("set_clean", "coalition_clean", k2, "set_clean_vs_coalition_clean")
    cmp("set_union", "coalition_union", k2, "set_union_vs_coalition_union")
    cmp("set_union", "layer", k2, "set_union_vs_layer")
    cmp("set_all", "layer", k2, "set_all_vs_layer")
    cmp("set_single", "expert", k3, "set_single_vs_expert")
    cmp("multi1_set_clean", "coalition_clean", k2, "multi1_set_clean_vs_coalition_clean")
    cmp("multi1_expert", "expert", k3, "multi1_expert_vs_expert")
    cmp("multi1_coal_clean", "coalition_clean", k2, "multi1_coalition_clean_vs_coalition_clean")
    # mixed multi steps: second-step vectors and deltas
    jm_ll = _by_key(tags2, "m_layer_layer", lambda t: t[1])
    for other in ("m_layer_all", "m_all_layer", "m_layer_setclean", "m_layer_coalclean"):
        jo = _by_key(tags2, other, lambda t: t[1])
        z = np.array([sd2[jo[i]] - sd2[jm_ll[i]] for i in jo])
        cs[f"{other}_vs_m_layer_layer_delta"] = _stats(z)
        if other in ("m_layer_all", "m_all_layer"):
            vz = [max(float((x - y).abs().max()) for x, y in zip(dg2["spawn_v"][jo[i]], dg2["spawn_v"][jm_ll[i]])) for i in jo]
            cs[f"{other}_vs_m_layer_layer_vec_maxabs"] = float(max(vz))
    # multi (layer, coalition_set clean) vs (layer, coalition_clean): identical definitions -> exact where the clean set
    # at layer 10 is the in-pass one
    ja_, jb_ = _by_key(tags2, "m_layer_setclean", lambda t: t[1]), _by_key(tags2, "m_layer_coalclean", lambda t: t[1])
    vz = np.array([max(float((x - y).abs().max()) for x, y in zip(dg2["spawn_v"][ja_[i]], dg2["spawn_v"][jb_[i]])) for i in ja_])
    z = np.array([sd2[ja_[i]] - sd2[jb_[i]] for i in ja_])
    okc = np.array([sorted(res2.route_idx[10, i].tolist()) == sorted(res.route_idx[10, i].tolist()) for i in ja_])
    cs["m_layer_setclean_vs_m_layer_coalclean_vec_maxabs"] = float(vz.max())
    cs["m_layer_setclean_vs_m_layer_coalclean_delta"] = _stats(z)
    cs["m_layer_setclean_vs_m_layer_coalclean_matched_n"] = int(okc.sum())
    cs["m_layer_setclean_vs_m_layer_coalclean_matched_vec_maxabs"] = float(vz[okc].max()) if okc.any() else None
    cs["m_layer_setclean_vs_m_layer_coalclean_matched_delta_maxabs"] = float(np.abs(z[okc]).max()) if okc.any() else None
    rep["coalition_set_pass2"] = cs
    rep["delta_prefill_pass1_vs_pass2_maxdiff"] = float(np.abs(res.delta - res2.delta).max())

    os.makedirs("results", exist_ok=True)
    with open("results/verify_ext5_engine_olmoe.json", "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep.items():
        log(f"{k}: {json.dumps(v) if isinstance(v, dict) else v}")


if __name__ == "__main__":
    main()
