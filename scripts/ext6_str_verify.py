"""ext6 verification on OLMoE-1B-7B-0125: STR corrupted runs (a donor prompt as a plain prefill row) against
transformers hooks.

HF side (model fully on GPU, eager attention): per case, clean and donor forwards with recording hooks on every MoE
block (final position); layer patch = donor forward with the MoE output at the final position replaced by the clean
one, at every layer; expert patch (first n_expert cases, three layers) = literal ablation difference under the
original routing (verify.py's method) with delta_e = c_e(clean) - c_e(donor) added to the donor MoE output.
Engine side: the same cases in one pass (parent = donor row, clean = clean row), plus an identity check (layer patch
spawned on the clean row as parent reproduces the clean delta). Calibration: the GN layer-patch agreement of
results/verify_ext2_attn_olmoe.json (same model, same kind).

Usage: python scripts/ext6_str_verify.py [n_cases=12]   -> results/verify_ext6_str_olmoe.json
"""
import json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, torch
from transformers import AutoTokenizer
from moetrace.arch import snapshot_dir
from moetrace.data import load_records, prepare_case, shuffled_order
from moetrace.verify import _hf_load, _moe_module
from moetrace.engine import Engine, PrefillSpec, SpawnSpec
from moetrace import ext6_str as S

REPO = "allenai/OLMoE-1B-7B-0125"
EXPERT_LAYERS = (3, 8, 13)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pick(n):
    tok = AutoTokenizer.from_pretrained(snapshot_dir(REPO))
    recs = load_records()
    index = S.donor_index(recs)
    out = []
    for ri in shuffled_order(len(recs), 0):
        c, _ = prepare_case(recs[ri], tok)
        if c is None:
            continue
        ds = S.candidates(c, recs[ri], tok, index, True)
        if ds:
            out.append((c, ds[0]))
        if len(out) >= n:
            break
    return out


def _o(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def hf_reference(pairs, n_expert):
    model = _hf_load(REPO)
    L = model.config.num_hidden_layers
    store, routing = {}, {}

    def fwd(ids):
        return model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float()

    def rec(key):
        def h(mod, args, out):
            store[key] = _o(out)[0, -1].detach().clone()
        return h

    def rep(key):
        def h(mod, args, out):
            o = _o(out).clone()
            o[0, -1] = store[key]
            return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        return h

    ref = {"clean": [], "donor": [], "layer": {}, "expert": {}}
    for i, (c, d) in enumerate(pairs):
        ti, fi = c.true_id, c.foil_id
        for run, ids in (("c", c.ids), ("d", d.ids)):
            hs = [_moe_module(model, l).register_forward_hook(rec((run, l))) for l in range(L)]
            lg = fwd(ids)
            for h in hs:
                h.remove()
            ref["clean" if run == "c" else "donor"].append(float(lg[ti] - lg[fi]))
        for l in range(L):
            h = _moe_module(model, l).register_forward_hook(rep(("c", l)))
            lg = fwd(d.ids)
            h.remove()
            ref["layer"][(i, l)] = float(lg[ti] - lg[fi])
        if i < n_expert:
            for l in EXPERT_LAYERS:
                moe = _moe_module(model, l)

                def route_hook(key):
                    def h(mod, args, out):
                        routing[key] = out[2][-1].tolist()
                    return h
                for run, ids in (("c", c.ids), ("d", d.ids)):
                    h = moe.gate.register_forward_hook(route_hook((i, l, run)))
                    fwd(ids)
                    h.remove()

                def block_out(ids, suppress):
                    def pre(mod, args):
                        hs_, idx, wts = args
                        wts = wts.clone()
                        wts[-1][idx[-1] == suppress] = 0
                        return (hs_, idx, wts)
                    hp = moe.experts.register_forward_pre_hook(pre) if suppress >= 0 else None
                    hr = moe.register_forward_hook(rec(("tmp", l)))
                    fwd(ids)
                    hr.remove()
                    if hp is not None:
                        hp.remove()
                    return store[("tmp", l)].float().clone()
                oc, od_ = block_out(c.ids, -1), block_out(d.ids, -1)
                for e in sorted(set(routing[(i, l, "c")]) | set(routing[(i, l, "d")])):
                    de = (oc - block_out(c.ids, e)) - (od_ - block_out(d.ids, e))

                    def add(mod, args, out, de=de):
                        o = _o(out).clone()
                        o[0, -1] = (o[0, -1].float() + de).to(o.dtype)
                        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
                    h = moe.register_forward_hook(add)
                    lg = fwd(d.ids)
                    h.remove()
                    ref["expert"][(i, l, int(e))] = float(lg[ti] - lg[fi])
        if i % 4 == 3:
            log(f"HF: {i + 1}/{len(pairs)}")
    del model
    torch.cuda.empty_cache()
    return ref, L


def main():
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_expert = 4
    pairs = pick(n_cases)
    log(f"{len(pairs)} (case, donor) pairs; first: {pairs[0][0].prompt!r} -> {pairs[0][1].prompt!r} "
        f"({pairs[0][0].true_str} / {pairs[0][0].foil_str})")
    ref, L = hf_reference(pairs, n_expert)
    n = len(pairs)
    eng = Engine(REPO)
    pre = [PrefillSpec(c.ids, c.true_id, c.foil_id) for c, _ in pairs] + [PrefillSpec(d.ids, c.true_id, c.foil_id) for c, d in pairs]
    spawns, tags = [], []
    for i in range(n):
        for l in range(L):
            spawns.append(SpawnSpec(l, n + i, i, "layer")); tags.append(("layer", i, l, -1))
        for l in (0, L // 2, L - 1):
            spawns.append(SpawnSpec(l, i, i, "layer")); tags.append(("identity", i, l, -1))
    for (i, l, e) in ref["expert"]:
        spawns.append(SpawnSpec(l, n + i, i, "expert", expert=e)); tags.append(("expert", i, l, e))
    res = eng.run(pre, spawns, record_routing=True, log=log)
    d, sd = res.delta, res.sp_delta
    rep = {"n_pairs": n, "n_expert_cases": n_expert, "expert_layers": list(EXPERT_LAYERS),
           "pairs": [{"case_id": c.case_id, "prompt": c.prompt, "donor": dd.prompt, "true": c.true_str, "foil": c.foil_str} for c, dd in pairs],
           "delta_clean_maxdiff_vs_hf": float(np.abs(d[:n] - np.array(ref["clean"])).max()),
           "delta_donor_maxdiff_vs_hf": float(np.abs(d[n:] - np.array(ref["donor"])).max()),
           "mean_delta_clean_hf": float(np.mean(ref["clean"])), "mean_delta_donor_hf": float(np.mean(ref["donor"]))}
    kinds = np.array([t[0] for t in tags])
    for kind in ("layer", "expert"):
        js = np.nonzero(kinds == kind)[0]
        key = (lambda t: (t[1], t[2])) if kind == "layer" else (lambda t: (t[1], t[2], t[3]))
        hf = np.array([ref[kind][key(tags[j])] for j in js])
        en = sd[js]
        dn_hf = np.array([ref["donor"][tags[j][1]] for j in js])
        dn_en = np.array([d[n + tags[j][1]] for j in js])
        diff = en - hf
        rep[f"{kind}_n"] = int(len(js))
        rep[f"{kind}_maxdiff"] = float(np.abs(diff).max())
        rep[f"{kind}_meanabsdiff"] = float(np.abs(diff).mean())
        rep[f"{kind}_frac_within_0.1"] = float((np.abs(diff) < 0.1).mean())
        rep[f"{kind}_frac_within_0.25"] = float((np.abs(diff) < 0.25).mean())
        rep[f"{kind}_rescue_corr_vs_hf"] = float(np.corrcoef(en - dn_en, hf - dn_hf)[0, 1])
        if kind == "layer":
            ce = np.zeros(L); ch = np.zeros(L)
            for j, a, b in zip(js, en - dn_en, hf - dn_hf):
                ce[tags[j][2]] += a / n
                ch[tags[j][2]] += b / n
            rep["layer_mean_curve_engine"] = [round(float(x), 3) for x in ce]
            rep["layer_mean_curve_hf"] = [round(float(x), 3) for x in ch]
            rep["layer_mean_curve_maxdiff"] = float(np.abs(ce - ch).max())
    js = np.nonzero(kinds == "identity")[0]
    rep["identity_maxdiff"] = float(max(abs(sd[j] - d[tags[j][1]]) for j in js))
    cal = "results/verify_ext2_attn_olmoe.json"
    if os.path.exists(cal):
        c = json.load(open(cal))
        rep["calibration_gn_layer"] = {k: c.get(f"layer_vs_hf_{k}") for k in ("maxdiff", "meanabsdiff", "frac_within_0.1", "frac_within_0.25")}
        rep["calibration_gn_layer"]["rescue_corr"] = c.get("layer_rescue_corr_vs_hf")
    rep["pass_total_s"] = res.extra["total_s"]
    with open("results/verify_ext6_str_olmoe.json", "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep.items():
        if k not in ("pairs",):
            log(f"{k}: {v}")


if __name__ == "__main__":
    main()
