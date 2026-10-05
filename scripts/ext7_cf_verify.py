"""ext7-controls verification on OLMoE-1B-7B-0125: the sublayer kinds attn_layer and block (and layer, calibration) with
an STR corrupted run (a CounterFact donor prompt as a plain prefill row) as parent, against transformers hooks.

The engine path is the one verified for GN in scripts/ext2_attn_verify.py (results/verify_ext2_attn_olmoe.json); STR
only changes the parent row (no noise injection). HF side (model fully on GPU, eager attention): per (case, donor) pair,
clean forward with recording hooks on every layer's self_attn output and MoE output at the final position; donor
forwards with replacement hooks at each layer:
    attn_layer  self_attn output[final] := clean      (MoE of that layer recomputes on the patched residual)
    block       self_attn output[final] := clean  AND  MoE output[final] := clean
    layer       MoE output[final] := clean
Engine side: one pass with the same spawns (parent = donor row, clean = clean row), plus an identity check (each kind
spawned with the clean row as parent reproduces the clean delta).

Usage: python scripts/ext7_cf_verify.py [n_pairs=12]   -> results/verify_ext7_controls_cf_olmoe.json
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
KINDS = ("attn_layer", "block", "layer")
OUT = "/home/ubuntu/MOE/results/verify_ext7_controls_cf_olmoe.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pick(n):
    """First n paper-order CounterFact cases with a symmetric donor (as scripts/ext6_str_verify.py)."""
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


def _replaced(out, new_final):
    o = _o(out).clone()
    o[0, -1] = new_final
    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o


@torch.no_grad()
def hf_reference(pairs):
    model = _hf_load(REPO)
    L = model.config.num_hidden_layers
    store = {}

    def fwd(ids):
        return model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float()

    def rec(key):
        def h(mod, args, out):
            store[key] = _o(out)[0, -1].detach().clone()
        return h

    def rep(key):
        def h(mod, args, out):
            return _replaced(out, store[key])
        return h

    ref = {"clean": [], "donor": [], "patch": {}}
    for i, (c, d) in enumerate(pairs):
        ti, fi = c.true_id, c.foil_id
        for run, ids in (("c", c.ids), ("d", d.ids)):
            hs = []
            for l in range(L):
                hs.append(model.model.layers[l].self_attn.register_forward_hook(rec((run, "attn", l))))
                hs.append(_moe_module(model, l).register_forward_hook(rec((run, "moe", l))))
            lg = fwd(ids)
            for h in hs:
                h.remove()
            ref["clean" if run == "c" else "donor"].append(float(lg[ti] - lg[fi]))
        for l in range(L):
            a, m = model.model.layers[l].self_attn, _moe_module(model, l)
            variants = {"attn_layer": [(a, ("c", "attn", l))], "block": [(a, ("c", "attn", l)), (m, ("c", "moe", l))],
                        "layer": [(m, ("c", "moe", l))]}
            for kind, hs in variants.items():  # one variant at a time
                handles = [mod.register_forward_hook(rep(key)) for mod, key in hs]
                lg = fwd(d.ids)
                for h in handles:
                    h.remove()
                ref["patch"][(i, l, kind)] = float(lg[ti] - lg[fi])
        if i % 4 == 3:
            log(f"HF: {i + 1}/{len(pairs)}")
    del model
    torch.cuda.empty_cache()
    return ref, L


def main():
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    pairs = pick(n_pairs)
    log(f"{len(pairs)} (case, donor) pairs; first: {pairs[0][0].prompt!r} -> {pairs[0][1].prompt!r}")
    ref, L = hf_reference(pairs)
    n = len(pairs)
    eng = Engine(REPO)
    pre = [PrefillSpec(c.ids, c.true_id, c.foil_id) for c, _ in pairs] + [PrefillSpec(d.ids, c.true_id, c.foil_id) for c, d in pairs]
    spawns, tags = [], []
    for i in range(n):
        for l in range(L):
            for kind in KINDS:
                spawns.append(SpawnSpec(l, n + i, i, kind)); tags.append((kind, i, l))
        for l in (0, L // 2, L - 1):
            for kind in KINDS:
                spawns.append(SpawnSpec(l, i, i, kind)); tags.append(("id_" + kind, i, l))
    res = eng.run(pre, spawns, record_routing=False, log=log)
    d, sd = res.delta, res.sp_delta
    rep = {"n_pairs": n, "kinds": list(KINDS), "layers": L,
           "pairs": [{"case_id": c.case_id, "prompt": c.prompt, "donor": dd.prompt, "true": c.true_str, "foil": c.foil_str} for c, dd in pairs],
           "delta_clean_maxdiff_vs_hf": float(np.abs(d[:n] - np.array(ref["clean"])).max()),
           "delta_donor_maxdiff_vs_hf": float(np.abs(d[n:] - np.array(ref["donor"])).max())}
    kinds_arr = np.array([t[0] for t in tags])
    for kind in KINDS:
        js = np.nonzero(kinds_arr == kind)[0]
        hf = np.array([ref["patch"][(tags[j][1], tags[j][2], kind)] for j in js])
        en = sd[js]
        dn_hf = np.array([ref["donor"][tags[j][1]] for j in js])
        dn_en = np.array([d[n + tags[j][1]] for j in js])
        diff = en - hf
        rep[f"{kind}_n"] = int(len(js))
        rep[f"{kind}_maxdiff"] = float(np.abs(diff).max())
        rep[f"{kind}_meanabsdiff"] = float(np.abs(diff).mean())
        rep[f"{kind}_frac_within_0.25"] = float((np.abs(diff) < 0.25).mean())
        rep[f"{kind}_rescue_corr_vs_hf"] = float(np.corrcoef(en - dn_en, hf - dn_hf)[0, 1])
        ce, ch = np.zeros(L), np.zeros(L)
        for j, a, b in zip(js, en - dn_en, hf - dn_hf):
            ce[tags[j][2]] += a / n
            ch[tags[j][2]] += b / n
        rep[f"{kind}_mean_curve_engine"] = [round(float(x), 3) for x in ce]
        rep[f"{kind}_mean_curve_hf"] = [round(float(x), 3) for x in ch]
        rep[f"{kind}_mean_curve_maxdiff"] = float(np.abs(ce - ch).max())
        ji = np.nonzero(kinds_arr == "id_" + kind)[0]
        rep[f"{kind}_identity_maxdiff"] = float(max(abs(sd[j] - d[tags[j][1]]) for j in ji))
    cal = "/home/ubuntu/MOE/results/verify_ext2_attn_olmoe.json"
    if os.path.exists(cal):
        c = json.load(open(cal))
        rep["calibration_gn_ext2"] = {k: {"maxdiff": c.get(f"{k}_vs_hf_maxdiff"), "meanabsdiff": c.get(f"{k}_vs_hf_meanabsdiff"),
                                          "rescue_corr": c.get(f"{k}_rescue_corr_vs_hf")} for k in KINDS}
    rep["pass_total_s"] = res.extra["total_s"]
    rep["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep.items():
        if k != "pairs" and "curve" not in k:
            log(f"{k}: {v}")


if __name__ == "__main__":
    main()
