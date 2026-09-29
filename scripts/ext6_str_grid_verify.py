"""ext6 grid verification on OLMoE-1B-7B-0125: STR layer x position MoE-output patching with the suffix executor
(moetrace/ext5_subject.py, SubjectSpawn.window and run_subject(metrics=True)) against transformers hooks.

HF side (model on the GPU, eager attention): per (case, donor) pair, the clean forward records every MoE block's full
output; then, for every position p from the first subject token to the final token and every layer l, the donor forward
with the MoE output at p replaced by the clean one at layer l (window 1) and at layers max(0, l-2)..min(L-1, l+2)
(window 5, centred as in Meng et al. / Zhang & Nanda). Delta and the full-softmax p(true) are read at the final position.
Engine side: one SubjectEngine pass with the same rows (clean and donor prefill rows recording at p), metrics on, plus
the null invariant (kind 'zero' at p on the donor row reproduces the donor run).

Usage: python scripts/ext6_str_grid_verify.py [n_pairs=8]   -> results/verify_ext6_str_grid_olmoe.json
"""
import json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, torch
from transformers import AutoTokenizer
from moetrace.arch import snapshot_dir
from moetrace.data import load_records, prepare_case, shuffled_order
from moetrace.verify import _hf_load, _moe_module
from moetrace.ext5_subject import SubjectEngine, SubjectPrefill, SubjectSpawn
from moetrace import ext6_str as S

REPO = "allenai/OLMoE-1B-7B-0125"
HALF = 2  # window 5 = l-2 .. l+2


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


def win(l, L):
    return max(0, l - HALF), min(L - 1, l + HALF)


def _o(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def hf_reference(pairs):
    model = _hf_load(REPO)
    L = model.config.num_hidden_layers
    store = {}

    def fwd(ids):
        return model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float()

    def rec(l):
        def h(mod, args, out):
            store[l] = _o(out)[0].detach().clone()  # [T, H]
        return h

    def rep(l, p):
        def h(mod, args, out):
            o = _o(out).clone()
            o[0, p] = store[l][p]
            return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        return h

    ref = {"clean": [], "donor": [], "p_donor": [], "w1": {}, "w5": {}}
    for i, (c, d) in enumerate(pairs):
        hs = [_moe_module(model, l).register_forward_hook(rec(l)) for l in range(L)]
        lg = fwd(c.ids)
        for h in hs:
            h.remove()
        ref["clean"].append(float(lg[c.true_id] - lg[c.foil_id]))
        lg = fwd(d.ids)
        ref["donor"].append(float(lg[c.true_id] - lg[c.foil_id]))
        ref["p_donor"].append(float(torch.softmax(lg, -1)[c.true_id]))
        for p in range(c.subject_pos[0], len(c.ids)):
            for l in range(L):
                for key, (a, b) in (("w1", (l, l)), ("w5", win(l, L))):
                    hs = [_moe_module(model, x).register_forward_hook(rep(x, p)) for x in range(a, b + 1)]
                    lg = fwd(d.ids)
                    for h in hs:
                        h.remove()
                    ref[key][(i, p, l)] = (float(lg[c.true_id] - lg[c.foil_id]), float(torch.softmax(lg, -1)[c.true_id]))
        log(f"HF: pair {i + 1}/{len(pairs)} done")
    del model
    torch.cuda.empty_cache()
    return ref, L


def main():
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    pairs = pick(n_pairs)
    log(f"{len(pairs)} pairs; first {pairs[0][0].prompt!r} -> {pairs[0][1].prompt!r}")
    ref, L = hf_reference(pairs)
    eng = SubjectEngine(REPO)
    pre, spawns, tags, unit_rows = [], [], [], {}
    for i, (c, d) in enumerate(pairs):
        for p in range(c.subject_pos[0], len(c.ids)):
            ci = len(pre); pre.append(SubjectPrefill(c.ids, c.true_id, c.foil_id, rec_pos=p))
            di = len(pre); pre.append(SubjectPrefill(d.ids, c.true_id, c.foil_id, rec_pos=p))
            unit_rows[(i, p)] = (ci, di)
            for l in range(L):
                spawns.append(SubjectSpawn(l, di, ci, "layer", p)); tags.append(("w1", i, p, l))
                a, b = win(l, L)
                spawns.append(SubjectSpawn(a, di, ci, "layer", p, window=b - a + 1)); tags.append(("w5", i, p, l))
            spawns.append(SubjectSpawn(0, di, ci, "zero", p)); tags.append(("zero", i, p, 0))
    res = eng.run_subject(pre, spawns, log=log, metrics=True)
    d_, sd = res.delta, res.sp_delta
    mp, ms = res.extra["metrics_prefill"], res.extra["metrics_spawn"]
    rep = {"n_pairs": len(pairs), "n_units": len(unit_rows), "n_spawns": len(spawns), "window_half": HALF}
    ci0 = np.array([unit_rows[k][0] for k in unit_rows]); di0 = np.array([unit_rows[k][1] for k in unit_rows])
    ui = np.array([k[0] for k in unit_rows])
    rep["delta_clean_maxdiff_vs_hf"] = float(np.abs(d_[ci0] - np.array(ref["clean"])[ui]).max())
    rep["delta_donor_maxdiff_vs_hf"] = float(np.abs(d_[di0] - np.array(ref["donor"])[ui]).max())
    rep["p_true_donor_maxdiff_vs_hf"] = float(np.abs(mp["p_true"][di0] - np.array(ref["p_donor"])[ui]).max())
    kinds = np.array([t[0] for t in tags])
    for key in ("w1", "w5"):
        js = np.nonzero(kinds == key)[0]
        hf_d = np.array([ref[key][tags[j][1:]][0] for j in js])
        hf_p = np.array([ref[key][tags[j][1:]][1] for j in js])
        en_d = sd[js]
        en_p = ms["p_true"][js]
        base_e = np.array([d_[unit_rows[tags[j][1:3]][1]] for j in js])
        base_h = np.array([ref["donor"][tags[j][1]] for j in js])
        diff = en_d - hf_d
        rep[f"{key}_n"] = int(len(js))
        rep[f"{key}_delta_maxdiff"] = float(np.abs(diff).max())
        rep[f"{key}_delta_meanabsdiff"] = float(np.abs(diff).mean())
        rep[f"{key}_delta_frac_within_0.25"] = float((np.abs(diff) < 0.25).mean())
        rep[f"{key}_rescue_corr_vs_hf"] = float(np.corrcoef(en_d - base_e, hf_d - base_h)[0, 1])
        rep[f"{key}_hf_rescue_std"] = float(np.std(hf_d - base_h))
        rep[f"{key}_p_true_maxdiff"] = float(np.abs(en_p - hf_p).max())
        rep[f"{key}_p_true_meanabsdiff"] = float(np.abs(en_p - hf_p).mean())
    js = np.nonzero(kinds == "zero")[0]
    rep["zero_on_donor_maxdiff"] = float(max(abs(sd[j] - d_[unit_rows[tags[j][1:3]][1]]) for j in js))
    cal = "results/verify_ext5_subject_olmoe.json"
    if os.path.exists(cal):
        c = json.load(open(cal))
        rep["calibration_ext5_layer_at_p_gn"] = {k: c.get(f"layer_vs_hf_{k}") for k in ("maxdiff", "meanabsdiff", "frac_within_0.25")}
        rep["calibration_ext5_layer_at_p_gn"]["rescue_corr"] = c.get("layer_rescue_corr_vs_hf")
    rep["pass_total_s"] = res.extra["total_s"]
    with open("results/verify_ext6_str_grid_olmoe.json", "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep.items():
        log(f"{k}: {v}")


if __name__ == "__main__":
    main()
