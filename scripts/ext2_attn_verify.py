"""ext2 verification on OLMoE-1B-7B-0125: the new sublayer kinds (attn_layer, block, resid) against transformers hooks
and the engine's internal invariants.

HF side (small model fully on GPU, eager attention): clean forward with recording hooks on every decoder layer's
self_attn output (after o_proj, the vector added to the residual), MoE output and layer output at the final position;
then, per (case, layer), noised forwards with replacement hooks:
    attn_layer  self_attn output[final] := clean            (MoE of that layer recomputes on the patched residual)
    block       self_attn output[final] := clean  AND  MoE output[final] := clean
    resid       decoder-layer output[final] := clean        (classic hidden-state restoration)
    layer       MoE output[final] := clean                  (already verified by verify_olmoe.py; calibration)
Engine side: the same cases in one pass with spawns of those kinds at every layer, plus
    (a) identity: attn_layer / block / resid spawned on the CLEAN run as parent (donor = itself) must reproduce the
        clean logits (rescue 0 within bf16 noise) -- attn_layer exercises the pre-MoE spawn path;
    (b) block vs block_diff (the two sublayer differences added to the noised residual) must agree to bf16 rounding.

Usage: python scripts/ext2_attn_verify.py [n_cases=20]   -> results/verify_ext2_attn_olmoe.json
"""
import json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, torch
from moetrace.verify import pick_cases, _hf_load, _moe_module
from moetrace.engine import Engine, PrefillSpec, SpawnSpec
from moetrace.noise import noise_draw

REPO = "allenai/OLMoE-1B-7B-0125"
NEW_KINDS = ("attn_layer", "block", "resid")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _with_replaced(out, new_final):
    o = _out_tensor(out).clone()
    o[0, -1] = new_final
    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o


@torch.no_grad()
def hf_reference(cases, layers, sigma_mult=3.0):
    model = _hf_load(REPO)
    dev = "cuda"
    emb = model.get_input_embeddings()
    sigma = sigma_mult * float(emb.weight.float().std().item())
    L = model.config.num_hidden_layers

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

    def mods(l):
        layer = model.model.layers[l]
        return layer.self_attn, _moe_module(model, l), layer

    store = {}

    def rec(key):
        def hook(mod, args, out):
            store[key] = _out_tensor(out)[0, -1].detach().clone()
        return hook

    def rep(key):
        def hook(mod, args, out):
            return _with_replaced(out, store[key])
        return hook

    out = {"delta_clean": [], "delta_noised": [], "patch": {}, "norms": {}}
    for i, c in enumerate(cases):
        ti, fi = c.true_id, c.foil_id
        # clean + noised forwards with recording hooks at every layer
        for run, noised in (("c", False), ("n", True)):
            hs = []
            for l in range(L):
                a, m, d = mods(l)
                hs += [a.register_forward_hook(rec((run, "attn", l))), m.register_forward_hook(rec((run, "moe", l))),
                       d.register_forward_hook(rec((run, "out", l)))]
            lg = fwd(embeds(c, noised))
            for h in hs:
                h.remove()
            out["delta_clean" if not noised else "delta_noised"].append(float(lg[ti] - lg[fi]))
        for l in range(L):
            out["norms"][(i, l)] = {k: float((store[("c", k, l)].float() - store[("n", k, l)].float()).norm())
                                    for k in ("attn", "moe", "out")}
        # patched noised forwards
        for l in layers:
            a, m, d = mods(l)
            variants = {"attn_layer": [(a, ("c", "attn", l))], "block": [(a, ("c", "attn", l)), (m, ("c", "moe", l))],
                        "resid": [(d, ("c", "out", l))], "layer": [(m, ("c", "moe", l))]}
            for kind, hs in variants.items():  # one variant at a time (registering all would combine them)
                handles = [mod.register_forward_hook(rep(key)) for mod, key in hs]
                lg = fwd(embeds(c, True))
                for h in handles:
                    h.remove()
                out["patch"][(i, l, kind)] = float(lg[ti] - lg[fi])
        if i % 5 == 4:
            log(f"HF: {i + 1}/{len(cases)} cases done")
    del model
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def engine_pass(cases, layers):
    eng = Engine(REPO)
    L, Hd = eng.spec.n_layers, eng.hidden
    sigma = 3.0 * eng.embed_std
    n = len(cases)
    prefill = [PrefillSpec(c.ids, c.true_id, c.foil_id) for c in cases]
    prefill += [PrefillSpec(c.ids, c.true_id, c.foil_id, list(c.subject_pos), noise_draw(c.case_id, len(c.subject_pos), Hd, sigma))
                for c in cases]
    spawns, tags = [], []
    for i in range(n):
        for l in range(L):
            for kind in NEW_KINDS + ("layer", "block_diff"):
                spawns.append(SpawnSpec(l, n + i, i, kind))
                tags.append((kind, i, l))
        for l in sorted({0, 1, L // 2, L - 2, L - 1}):
            for kind in NEW_KINDS:  # identity: clean parent, clean donor
                spawns.append(SpawnSpec(l, i, i, kind))
                tags.append(("id_" + kind, i, l))
    t0 = time.time()
    res = eng.run(prefill, spawns, record_routing=True, log=log)
    return res, tags, n, time.time() - t0


def main():
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    cases = pick_cases(REPO, n_cases)
    log(f"{len(cases)} cases; first: {cases[0].prompt!r} -> {cases[0].true_str} / {cases[0].foil_str}")
    L = 16
    layers = list(range(L))
    ref = hf_reference(cases, layers)
    log("HF reference done")
    res, tags, n, t_eng = engine_pass(cases, layers)
    rep = {"n_cases": len(cases), "layers": layers, "engine_pass_s": t_eng}
    d = res.delta
    dc, dn = d[:n], d[n:]
    rep["delta_clean_maxdiff_vs_hf"] = float(np.abs(dc - np.array(ref["delta_clean"])).max())
    rep["delta_noised_maxdiff_vs_hf"] = float(np.abs(dn - np.array(ref["delta_noised"])).max())
    sd = res.sp_delta
    tag_kind = np.array([t[0] for t in tags])
    # --- (c) new kinds vs HF hooks (layer as calibration of the bf16 noise floor)
    curves = {}
    for kind in NEW_KINDS + ("layer",):
        js = np.nonzero(tag_kind == kind)[0]
        diffs = np.array([sd[j] - ref["patch"][(tags[j][1], tags[j][2], kind)] for j in js])
        resc_e = np.array([sd[j] - dn[tags[j][1]] for j in js])
        resc_h = np.array([ref["patch"][(tags[j][1], tags[j][2], kind)] - ref["delta_noised"][tags[j][1]] for j in js])
        rep[f"{kind}_vs_hf_n"] = int(len(js))
        rep[f"{kind}_vs_hf_maxdiff"] = float(np.abs(diffs).max())
        rep[f"{kind}_vs_hf_meanabsdiff"] = float(np.abs(diffs).mean())
        rep[f"{kind}_vs_hf_frac_within_0.1"] = float((np.abs(diffs) < 0.1).mean())
        rep[f"{kind}_vs_hf_frac_within_0.25"] = float((np.abs(diffs) < 0.25).mean())
        rep[f"{kind}_rescue_corr_vs_hf"] = float(np.corrcoef(resc_e, resc_h)[0, 1])
        cur_e = np.zeros(L); cur_h = np.zeros(L)
        for j, re_, rh in zip(js, resc_e, resc_h):
            cur_e[tags[j][2]] += re_ / n
            cur_h[tags[j][2]] += rh / n
        curves[kind] = {"engine": [round(float(x), 3) for x in cur_e], "hf": [round(float(x), 3) for x in cur_h]}
        rep[f"{kind}_mean_curve_maxdiff_vs_hf"] = float(np.abs(cur_e - cur_h).max())
    rep["mean_rescue_curves"] = curves
    # --- (a) identity on the clean run
    for kind in NEW_KINDS:
        js = np.nonzero(tag_kind == "id_" + kind)[0]
        z = np.array([abs(sd[j] - dc[tags[j][1]]) for j in js])
        rep[f"identity_{kind}_n"] = int(len(js))
        rep[f"identity_{kind}_maxdiff"] = float(z.max())
        rep[f"identity_{kind}_meanabsdiff"] = float(z.mean())
    # --- (b) block vs block_diff
    jb = {(tags[j][1], tags[j][2]): j for j in np.nonzero(tag_kind == "block")[0]}
    jd = {(tags[j][1], tags[j][2]): j for j in np.nonzero(tag_kind == "block_diff")[0]}
    z = np.array([sd[jb[k]] - sd[jd[k]] for k in jb])
    rep["block_vs_block_diff_n"] = int(len(z))
    rep["block_vs_block_diff_maxdiff"] = float(np.abs(z).max())
    rep["block_vs_block_diff_meanabsdiff"] = float(np.abs(z).mean())
    rep["block_vs_block_diff_frac_within_0.1"] = float((np.abs(z) < 0.1).mean())
    rep["block_vs_block_diff_frac_equal"] = float((z == 0).mean())
    # --- vnorm bookkeeping vs HF norms of the final-position differences
    vn = {}
    for kind, hk in (("attn_layer", "attn"), ("layer", "moe"), ("resid", "out")):
        js = np.nonzero(tag_kind == kind)[0]
        e = np.array([res.sp_vnorm[j] for j in js])
        h = np.array([ref["norms"][(tags[j][1], tags[j][2])][hk] for j in js])
        vn[kind] = {"max_reldiff": float((np.abs(e - h) / np.maximum(h, 1e-6)).max()), "mean_engine": float(e.mean()), "mean_hf": float(h.mean())}
    rep["vnorm_vs_hf"] = vn
    rep["pass_total_s"] = res.extra["total_s"]
    os.makedirs("results", exist_ok=True)
    with open("results/verify_ext2_attn_olmoe.json", "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep.items():
        if k not in ("mean_rescue_curves",):
            log(f"{k}: {v}")
    for kind, cv in curves.items():
        log(f"curve {kind} engine: {cv['engine']}")
        log(f"curve {kind} hf:     {cv['hf']}")


if __name__ == "__main__":
    main()
