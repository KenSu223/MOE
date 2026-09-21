"""ext5 F4 verification on OLMoE-1B-7B-0125: suffix wavefront rows (moetrace/ext5_subject.py) patched at the LAST
SUBJECT TOKEN p against transformers hooks, plus the engine's internal invariants.

HF side (small model on the GPU, eager attention): clean and noised forwards with recording hooks on every decoder
layer's self_attn output, MoE output and layer output AT POSITION p; then per (case, layer) noised forwards with the
replacement at position p:
    attn_layer  self_attn output[p] := clean      layer  MoE output[p] := clean      resid  layer output[p] := clean
    block       both sublayer outputs at p := clean
Delta is read at the final position. Engine side, one pass with
    (a) the same kinds at every layer at p (vs HF; `layer` calibrates the bf16 floor as in ext2),
    (b) null invariant: `zero` at p on the noised parent must reproduce the noised run,
    (c) identity: every kind spawned on the CLEAN run with the clean donor must reproduce the clean run,
    (d) consistency: rows spawned at p = T-1 (rec_pos = -1) must reproduce the final-token machinery: compared with the
        stored rows of results/olmoe_attnsweep (same cases, kinds attn_layer / layer / resid) within bf16 noise.
Cases: the first N of results/olmoe_attnsweep's strict set (so that (d) has stored reference rows).

Usage: python scripts/ext5_subject_verify.py [n_cases=20]   -> results/verify_ext5_subject_olmoe.json
"""
import json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.verify import _hf_load, _moe_module
from moetrace.protocol import cases_by_id
from moetrace.noise import noise_draw
from moetrace.ext5_subject import SubjectEngine, SubjectPrefill, SubjectSpawn, subject_prefill_rows, last_subject_pos

REPO = "allenai/OLMoE-1B-7B-0125"
REF_RUN = "/home/ubuntu/MOE/results/olmoe_attnsweep"
KINDS = ("attn_layer", "layer", "resid", "block")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _out_tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _with_replaced(out, pos, new):
    o = _out_tensor(out).clone()
    o[0, pos] = new
    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o


@torch.no_grad()
def hf_reference(cases, layers, sigma_mult=3.0):
    model = _hf_load(REPO)
    dev = "cuda"
    emb = model.get_input_embeddings()
    sigma = sigma_mult * float(emb.weight.float().std().item())
    L = model.config.num_hidden_layers

    def embeds(c, noise_pos):
        e = emb(torch.tensor([c.ids], device=dev))
        if noise_pos:
            eps = noise_draw(c.case_id, len(c.subject_pos), e.shape[-1], sigma).to(dev)
            sel = [c.subject_pos.index(p) for p in noise_pos]
            e = e.clone()
            pos = torch.tensor(noise_pos, device=dev)
            e[0, pos] = (e[0, pos].float() + eps[sel]).to(e.dtype)
        return e

    def fwd(e):
        return model(inputs_embeds=e).logits[0, -1].float()

    def mods(l):
        layer = model.model.layers[l]
        return layer.self_attn, _moe_module(model, l), layer

    store = {}

    def rec(key, pos):
        def hook(mod, args, out):
            store[key] = _out_tensor(out)[0, pos].detach().clone()
        return hook

    def rep(key, pos):
        def hook(mod, args, out):
            return _with_replaced(out, pos, store[key])
        return hook

    out = {"delta_clean": [], "delta_noised": [], "delta_noised_lastonly": [], "delta_noised_exceptlast": [], "patch": {}, "norms": {}}
    for i, c in enumerate(cases):
        ti, fi = c.true_id, c.foil_id
        p = last_subject_pos(c)
        for run, npos in (("c", []), ("n", list(c.subject_pos))):
            hs = []
            for l in range(L):
                a, m, d = mods(l)
                hs += [a.register_forward_hook(rec((run, "attn", l), p)), m.register_forward_hook(rec((run, "moe", l), p)),
                       d.register_forward_hook(rec((run, "out", l), p))]
            lg = fwd(embeds(c, npos))
            for h in hs:
                h.remove()
            out["delta_clean" if run == "c" else "delta_noised"].append(float(lg[ti] - lg[fi]))
        lg = fwd(embeds(c, [c.subject_pos[-1]]))
        out["delta_noised_lastonly"].append(float(lg[ti] - lg[fi]))
        lg = fwd(embeds(c, list(c.subject_pos[:-1])))
        out["delta_noised_exceptlast"].append(float(lg[ti] - lg[fi]))
        for l in range(L):
            out["norms"][(i, l)] = {k: float((store[("c", k, l)].float() - store[("n", k, l)].float()).norm()) for k in ("attn", "moe", "out")}
        for l in layers:
            a, m, d = mods(l)
            variants = {"attn_layer": [(a, ("c", "attn", l))], "layer": [(m, ("c", "moe", l))], "resid": [(d, ("c", "out", l))],
                        "block": [(a, ("c", "attn", l)), (m, ("c", "moe", l))]}
            for kind, hs in variants.items():
                handles = [mod.register_forward_hook(rep(key, p)) for mod, key in hs]
                lg = fwd(embeds(c, list(c.subject_pos)))
                for h in handles:
                    h.remove()
                out["patch"][(i, l, kind)] = float(lg[ti] - lg[fi])
        if i % 5 == 4:
            log(f"HF: {i + 1}/{len(cases)} cases done")
    del model
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def engine_pass(cases_l, layers):
    eng = SubjectEngine(REPO)
    L, Hd = eng.spec.n_layers, eng.hidden
    sigma = 3.0 * eng.embed_std
    ids = [c.case_id for c in cases_l]
    cases = {c.case_id: c for c in cases_l}
    n = len(ids)
    pre, offs = subject_prefill_rows(cases, ids, sigma, Hd, last_subject_pos, noise_draw)
    # final-position rows (rec_pos = -1) for the consistency check (d)
    offs["clean_fin"] = len(pre)
    pre += [SubjectPrefill(cases[c].ids, cases[c].true_id, cases[c].foil_id) for c in ids]
    offs["noised_fin"] = len(pre)
    pre += [SubjectPrefill(cases[c].ids, cases[c].true_id, cases[c].foil_id, cases[c].subject_pos,
                           noise_draw(c, len(cases[c].subject_pos), Hd, sigma)) for c in ids]
    spawns, tags = [], []
    for i, c in enumerate(ids):
        p = last_subject_pos(cases[c])
        T = len(cases[c].ids)
        for l in range(L):
            for kind in KINDS:
                spawns.append(SubjectSpawn(l, offs["noised"] + i, offs["clean"] + i, kind, p)); tags.append((kind, i, l))
            spawns.append(SubjectSpawn(l, offs["noised"] + i, offs["clean"] + i, "zero", p)); tags.append(("zero", i, l))
            for kind in ("attn_layer", "layer", "resid"):
                spawns.append(SubjectSpawn(l, offs["noised_fin"] + i, offs["clean_fin"] + i, kind, T - 1)); tags.append(("fin_" + kind, i, l))
        for l in sorted({0, 1, L // 2, L - 2, L - 1}):
            for kind in KINDS:
                spawns.append(SubjectSpawn(l, offs["clean"] + i, offs["clean"] + i, kind, p)); tags.append(("id_" + kind, i, l))
    t0 = time.time()
    res = eng.run_subject(pre, spawns, log=log)
    t_eng = time.time() - t0
    # (e) in-process consistency: identical prefill batch (clean_fin, noised_fin only) through run_subject at p = T-1 and
    #     through the original Engine.run with the final-token kinds; prefill shapes are identical so near bit-identity
    #     is expected (the only difference is the wavefront row layout [W, 1, H] vs [W, H])
    from moetrace.engine import SpawnSpec
    pre_fin = pre[offs["clean_fin"] :]
    sp_s, sp_o, tags_e = [], [], []
    for i, c in enumerate(ids):
        T = len(cases[c].ids)
        for l in range(L):
            for kind in ("attn_layer", "layer", "resid"):
                sp_s.append(SubjectSpawn(l, n + i, i, kind, T - 1))
                sp_o.append(SpawnSpec(l, n + i, i, kind))
                tags_e.append((kind, i, l))
    r_s = eng.run_subject(pre_fin, sp_s, log=None)
    r_o = eng.run(pre_fin, sp_o, record_routing=False, log=None)
    same = {"prefill_delta_maxdiff": float(np.abs(r_s.delta - r_o.delta).max()), "n": len(tags_e)}
    z = r_s.sp_delta - r_o.sp_delta
    same.update({"delta_maxdiff": float(np.abs(z).max()), "delta_meanabsdiff": float(np.abs(z).mean()), "delta_frac_equal": float((z == 0).mean()),
                 "delta_frac_within_0.1": float((np.abs(z) < 0.1).mean()), "vnorm_maxdiff": float(np.abs(r_s.sp_vnorm - r_o.sp_vnorm).max())})
    for kind in ("attn_layer", "layer", "resid"):
        js = [j for j, t in enumerate(tags_e) if t[0] == kind]
        same[kind + "_delta_maxdiff"] = float(np.abs(z[js]).max())
    return res, tags, offs, n, t_eng, same


def main():
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sets = json.load(open(os.path.join(REF_RUN, "case_sets.json")))
    ids = (sets["strict"]["discovery"] + sets["strict"]["validation"])[:n_cases]
    cases, rej = cases_by_id("olmoe", ids)
    assert not rej, rej
    cases_l = [cases[c] for c in ids]
    log(f"{len(cases_l)} cases; first: {cases_l[0].prompt!r} p={last_subject_pos(cases_l[0])} T={len(cases_l[0].ids)}")
    L = 16
    layers = list(range(L))
    ref = hf_reference(cases_l, layers)
    log("HF reference done")
    res, tags, offs, n, t_eng, same = engine_pass(cases_l, layers)
    rep = {"n_cases": n, "layers": layers, "engine_pass_s": t_eng, "pos": [last_subject_pos(c) for c in cases_l],
           "T": [len(c.ids) for c in cases_l]}
    d = res.delta
    dc, dn = d[offs["clean"] : offs["clean"] + n], d[offs["noised"] : offs["noised"] + n]
    rep["delta_clean_maxdiff_vs_hf"] = float(np.abs(dc - np.array(ref["delta_clean"])).max())
    rep["delta_noised_maxdiff_vs_hf"] = float(np.abs(dn - np.array(ref["delta_noised"])).max())
    for key in ("noised_lastonly", "noised_exceptlast"):
        dd = d[offs[key] : offs[key] + n]
        rep[f"delta_{key}_maxdiff_vs_hf"] = float(np.abs(dd - np.array(ref["delta_" + key])).max())
    rep["prefill_percase_absdiff_vs_hf"] = {key: [round(float(x), 3) for x in np.abs(d[offs[key] : offs[key] + n] - np.array(ref["delta_" + key]))]
                                          for key in ("clean", "noised", "noised_lastonly", "noised_exceptlast")}
    # final-position prefill rows must equal the subject-recorded ones (recording does not change the computation)
    rep["prefill_rec_pos_vs_final_maxdiff"] = float(max(np.abs(dc - d[offs["clean_fin"] : offs["clean_fin"] + n]).max(),
                                                        np.abs(dn - d[offs["noised_fin"] : offs["noised_fin"] + n]).max()))
    sd = res.sp_delta
    tk = np.array([t[0] for t in tags])
    curves = {}
    for kind in KINDS:
        js = np.nonzero(tk == kind)[0]
        diffs = np.array([sd[j] - ref["patch"][(tags[j][1], tags[j][2], kind)] for j in js])
        re_ = np.array([sd[j] - dn[tags[j][1]] for j in js])
        rh = np.array([ref["patch"][(tags[j][1], tags[j][2], kind)] - ref["delta_noised"][tags[j][1]] for j in js])
        rep[f"{kind}_vs_hf_n"] = int(len(js))
        rep[f"{kind}_vs_hf_maxdiff"] = float(np.abs(diffs).max())
        rep[f"{kind}_vs_hf_meanabsdiff"] = float(np.abs(diffs).mean())
        rep[f"{kind}_vs_hf_frac_within_0.1"] = float((np.abs(diffs) < 0.1).mean())
        rep[f"{kind}_vs_hf_frac_within_0.25"] = float((np.abs(diffs) < 0.25).mean())
        rep[f"{kind}_rescue_corr_vs_hf"] = float(np.corrcoef(re_, rh)[0, 1])
        rep[f"{kind}_hf_rescue_std"] = float(rh.std())
        rep[f"{kind}_rescue_diff_std"] = float((re_ - rh).std())
        ce, ch = np.zeros(L), np.zeros(L)
        for j, a, b in zip(js, re_, rh):
            ce[tags[j][2]] += a / n
            ch[tags[j][2]] += b / n
        curves[kind] = {"engine": [round(float(x), 3) for x in ce], "hf": [round(float(x), 3) for x in ch]}
        rep[f"{kind}_mean_curve_maxdiff_vs_hf"] = float(np.abs(ce - ch).max())
    rep["mean_rescue_curves_subject"] = curves
    # (b) null invariant
    js = np.nonzero(tk == "zero")[0]
    z = np.array([abs(sd[j] - dn[tags[j][1]]) for j in js])
    rep["zero_on_noised_n"], rep["zero_on_noised_maxdiff"], rep["zero_on_noised_meanabsdiff"] = int(len(z)), float(z.max()), float(z.mean())
    # (c) identity
    for kind in KINDS:
        js = np.nonzero(tk == "id_" + kind)[0]
        z = np.array([abs(sd[j] - dc[tags[j][1]]) for j in js])
        rep[f"identity_{kind}_n"], rep[f"identity_{kind}_maxdiff"], rep[f"identity_{kind}_meanabsdiff"] = int(len(z)), float(z.max()), float(z.mean())
    # (d) consistency with the final-token machinery (stored ext2 rows of the same cases)
    sr = pd.read_parquet(os.path.join(REF_RUN, "sweep_rows.parquet"))
    sr = sr[sr.case_id.isin(ids)]
    dn_ref = sr[sr.kind == "noised"].set_index("case_id").delta
    dn_fin = d[offs["noised_fin"] : offs["noised_fin"] + n]
    rep["final_noised_vs_stored_maxdiff"] = float(np.abs(dn_fin - np.array([dn_ref[c] for c in ids])).max())
    cons = {}
    for kind in ("attn_layer", "layer", "resid"):
        js = np.nonzero(tk == "fin_" + kind)[0]
        ref_d = sr[sr.kind == kind].set_index(["case_id", "layer"]).delta
        z = np.array([sd[j] - ref_d[(ids[tags[j][1]], tags[j][2])] for j in js])
        zr = np.array([(sd[j] - dn_fin[tags[j][1]]) - (ref_d[(ids[tags[j][1]], tags[j][2])] - dn_ref[ids[tags[j][1]]]) for j in js])
        r_new = np.array([sd[j] - dn_fin[tags[j][1]] for j in js])
        r_old = np.array([ref_d[(ids[tags[j][1]], tags[j][2])] - dn_ref[ids[tags[j][1]]] for j in js])
        ce, ch = np.zeros(L), np.zeros(L)
        for j, a, b in zip(js, r_new, r_old):
            ce[tags[j][2]] += a / n
            ch[tags[j][2]] += b / n
        cons[kind] = {"n": int(len(z)), "delta_maxdiff": float(np.abs(z).max()), "delta_meanabsdiff": float(np.abs(z).mean()),
                      "delta_frac_equal": float((z == 0).mean()), "delta_frac_within_0.1": float((np.abs(z) < 0.1).mean()),
                      "rescue_maxdiff": float(np.abs(zr).max()), "rescue_meanabsdiff": float(np.abs(zr).mean()),
                      "rescue_corr": float(np.corrcoef(r_new, r_old)[0, 1]), "mean_curve_maxdiff": float(np.abs(ce - ch).max())}
    rep["final_token_consistency_vs_olmoe_attnsweep"] = cons
    rep["final_token_consistency_in_process"] = same
    vn = {}
    for kind, hk in (("attn_layer", "attn"), ("layer", "moe"), ("resid", "out")):
        js = np.nonzero(tk == kind)[0]
        e = np.array([res.sp_vnorm[j] for j in js])
        h = np.array([ref["norms"][(tags[j][1], tags[j][2])][hk] for j in js])
        vn[kind] = {"max_reldiff": float((np.abs(e - h) / np.maximum(h, 1e-6)).max()), "mean_engine": float(e.mean()), "mean_hf": float(h.mean())}
    rep["vnorm_vs_hf"] = vn
    rep["pass_total_s"] = res.extra["total_s"]
    rep["Smax"] = res.extra["Smax"]
    rep["drop_full_mean"] = float((dc - dn).mean())
    rep["drop_lastonly_mean"] = float((dc - d[offs["noised_lastonly"] : offs["noised_lastonly"] + n]).mean())
    rep["drop_exceptlast_mean"] = float((dc - d[offs["noised_exceptlast"] : offs["noised_exceptlast"] + n]).mean())
    with open("/home/ubuntu/MOE/results/verify_ext5_subject_olmoe.json", "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep.items():
        if k not in ("mean_rescue_curves_subject", "pos", "T", "prefill_percase_absdiff_vs_hf"):
            log(f"{k}: {v}")
    for kind, cv in curves.items():
        log(f"curve {kind} engine: {cv['engine']}")
        log(f"curve {kind} hf:     {cv['hf']}")


if __name__ == "__main__":
    main()
