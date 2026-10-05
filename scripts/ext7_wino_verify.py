"""ext7 W1 verification on OLMoE-1B-7B-0125: WinoGrande STR pairs (both directions) against transformers hooks.

Pairs: the first N margin pairs of results/wino_olmoe/scan_pairs.parquet (seed-0 shuffle), directed cases
d = 0 (clean A, corrupted B, true = trig_a) and d = 1 (clean B, corrupted A, true = trig_b); ids from
data/wino_str/pairs_train_xl_olmoe.parquet.

HF side (model on the GPU, eager attention): per directed case, the clean forward records every decoder layer's
self_attn output, MoE output and layer output at ALL positions and the per-head attention outputs before o_proj at the
final position (head layers); then corrupted forwards with replacement hooks:
  final position (all directed cases): layer (MoE), attn_layer (attention), block (both), resid (layer output) at every
      layer; attn_head (one head's pre-o_proj output) at three layers x all heads (first --n-head cases);
  STR positions (first --n-grid cases): every position p from the first STR token to the final token, kinds layer /
      attn_layer / block / resid at every layer, and the window-5 MoE restoration (layers l-2..l+2) at p.
Engine side: Engine.run (final position; plus identity rows: every kind spawned on the clean row as parent) and
SubjectEngine.run_subject (STR positions; plus the null invariant: kind 'zero' at p on the corrupted row).

Usage: python scripts/ext7_wino_verify.py [--n-pairs 20] [--n-grid 8] [--n-head 10]  -> results/verify_ext7_wino_olmoe.json
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.verify import _hf_load, _moe_module
from moetrace.engine import Engine, PrefillSpec, SpawnSpec
from moetrace.ext5_subject import SubjectEngine, SubjectPrefill, SubjectSpawn
from moetrace.ext7_wino import load_twins
from moetrace import ext7_pairs as P

REPO = "allenai/OLMoE-1B-7B-0125"
FINAL_KINDS = ("layer", "attn_layer", "block", "resid")
GRID_KINDS = ("layer", "attn_layer", "block", "resid")
HALF = 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _o(out):
    return out[0] if isinstance(out, tuple) else out


def pick(n_pairs):
    twins, _ = load_twins("train_xl")
    pio = {t.pair_id: i for i, t in enumerate(twins)}
    sc = pd.read_parquet("/home/ubuntu/MOE/results/wino_olmoe/scan_pairs.parquet")
    pool = sorted(pio[x] for x in sc[sc.margin].pair_id)
    random.Random(0).shuffle(pool)
    sel = pool[:n_pairs]
    cs = {"pair_idx_of": {t.pair_id: i for i, t in enumerate(twins) if i in set(sel)},
          "directed": {"validation": [2 * i + d for i in sel for d in (0, 1)]}}
    pairs = P.load_pairs("/home/ubuntu/MOE/data/wino_str/pairs_train_xl_olmoe.parquet", cs)
    ids = cs["directed"]["validation"]
    return P.directed_cases(pairs, ids), ids, pairs


@torch.no_grad()
def hf_reference(dc, ids, n_grid, n_head, head_layers):
    model = _hf_load(REPO)
    L = model.config.num_hidden_layers
    nH = model.config.num_attention_heads
    D = model.config.hidden_size // nH
    store = {}

    def fwd(x):
        return model(input_ids=torch.tensor([x], device="cuda")).logits[0, -1].float()

    def mods(l):
        lay = model.model.layers[l]
        return {"attn": lay.self_attn, "moe": _moe_module(model, l), "out": lay}

    def rec(key):
        def h(mod, args, out):
            store[key] = _o(out)[0].detach().clone()  # [T, H]
        return h

    def rec_heads(key):
        def h(mod, args):
            store[key] = args[0][0, -1].detach().clone()  # [nH * D] pre-o_proj at the final position
        return h

    def rep(key, p):
        def h(mod, args, out):
            o = _o(out).clone()
            o[0, p] = store[key][p]
            return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        return h

    def rep_head(key, hh):
        def h(mod, args):
            x = args[0].clone()
            x[0, -1, hh * D : (hh + 1) * D] = store[key][hh * D : (hh + 1) * D]
            return (x,) + tuple(args[1:])
        return h

    KMODS = {"layer": ("moe",), "attn_layer": ("attn",), "block": ("attn", "moe"), "resid": ("out",)}
    ref = {"clean": {}, "corrupt": {}, "p_corrupt": {}, "final": {}, "grid": {}, "w5": {}, "head": {}}
    for i, c in enumerate(ids):
        x = dc[c]
        ti, fi = x.true_id, x.foil_id
        hs = []
        for l in range(L):
            for nm, md in mods(l).items():
                hs.append(md.register_forward_hook(rec(("c", nm, l))))
        if i < n_head:
            for l in head_layers:
                hs.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(rec_heads(("c", "heads", l))))
        lg = fwd(x.clean_ids)
        for h in hs:
            h.remove()
        ref["clean"][c] = float(lg[ti] - lg[fi])
        lg = fwd(x.corrupt_ids)
        ref["corrupt"][c] = float(lg[ti] - lg[fi])
        ref["p_corrupt"][c] = float(torch.softmax(lg, -1)[ti])
        T = x.T
        positions = range(x.first_str, T) if i < n_grid else [T - 1]
        for p in positions:
            for l in range(L):
                for k in (FINAL_KINDS if p == T - 1 else GRID_KINDS):
                    hs = [mods(l)[nm].register_forward_hook(rep(("c", nm, l), p)) for nm in KMODS[k]]
                    lg = fwd(x.corrupt_ids)
                    for h in hs:
                        h.remove()
                    val = float(lg[ti] - lg[fi])
                    if p == T - 1:
                        ref["final"][(c, l, k)] = val
                    if i < n_grid:
                        ref["grid"][(c, p, l, k)] = val
                if i < n_grid:
                    a, b = max(0, l - HALF), min(L - 1, l + HALF)
                    hs = [mods(z)["moe"].register_forward_hook(rep(("c", "moe", z), p)) for z in range(a, b + 1)]
                    lg = fwd(x.corrupt_ids)
                    for h in hs:
                        h.remove()
                    ref["w5"][(c, p, l)] = (float(lg[ti] - lg[fi]), float(torch.softmax(lg, -1)[ti]))
        if i < n_head:
            for l in head_layers:
                for hh in range(nH):
                    h = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(rep_head(("c", "heads", l), hh))
                    lg = fwd(x.corrupt_ids)
                    h.remove()
                    ref["head"][(c, l, hh)] = float(lg[ti] - lg[fi])
        if i % 8 == 7:
            log(f"HF: {i + 1}/{len(ids)} directed cases")
    del model
    torch.cuda.empty_cache()
    return ref, L, nH


def stats(en, hf, base_en, base_hf):
    diff = np.asarray(en) - np.asarray(hf)
    re, rh = np.asarray(en) - np.asarray(base_en), np.asarray(hf) - np.asarray(base_hf)
    return {"n": int(len(diff)), "maxdiff": float(np.abs(diff).max()), "meanabsdiff": float(np.abs(diff).mean()),
            "frac_within_0.25": float((np.abs(diff) < 0.25).mean()), "rescue_corr_vs_hf": float(np.corrcoef(re, rh)[0, 1]),
            "hf_rescue_sd": float(rh.std()), "hf_rescue_mean": float(rh.mean()), "engine_rescue_mean": float(re.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--n-grid", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=10)
    args = ap.parse_args()
    dc, ids, pairs = pick(args.n_pairs)
    x0 = pairs.iloc[0]
    log(f"{len(ids)} directed cases from {args.n_pairs} OLMoE margin pairs; first: {x0.prompt_a!r} -> {x0.word_a!r} / {x0.prompt_b!r} -> {x0.word_b!r}")
    head_layers = (4, 9, 14)
    import pickle
    cache = f"/tmp/claude-1000/-home-ubuntu-MOE/488f7156-7059-40cb-9629-447467831e6b/scratchpad/ext7w/hf_ref_{args.n_pairs}_{args.n_grid}_{args.n_head}.pkl"
    if os.path.exists(cache):
        ref, L, nH = pickle.load(open(cache, "rb"))
        log(f"HF reference loaded from {cache}")
    else:
        ref, L, nH = hf_reference(dc, ids, args.n_grid, args.n_head, head_layers)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pickle.dump((ref, L, nH), open(cache, "wb"))
    n = len(ids)
    rep = {"n_pairs": args.n_pairs, "n_directed": n, "n_grid_cases": args.n_grid, "n_head_cases": args.n_head, "head_layers": list(head_layers),
           "window_half": HALF, "pairs": [{"pair_id": r.pair_id, "prompt_a": r.prompt_a, "prompt_b": r.prompt_b, "word_a": r.word_a, "word_b": r.word_b,
                                           "str_pos": r.str_pos_l} for r in pairs.itertuples()]}
    # ---- final position: Engine.run
    eng = Engine(REPO)
    pre = [PrefillSpec(dc[c].clean_ids, dc[c].true_id, dc[c].foil_id) for c in ids]
    pre += [PrefillSpec(dc[c].corrupt_ids, dc[c].true_id, dc[c].foil_id) for c in ids]
    spawns, tags = [], []
    for i, c in enumerate(ids):
        for l in range(L):
            for k in FINAL_KINDS:
                spawns.append(SpawnSpec(l, n + i, i, k)); tags.append(("final", c, l, k))
        for l in (0, L // 2, L - 1):
            for k in FINAL_KINDS:
                spawns.append(SpawnSpec(l, i, i, k)); tags.append(("identity", c, l, k))
        if i < args.n_head:
            for l in head_layers:
                for hh in range(nH):
                    spawns.append(SpawnSpec(l, n + i, i, "attn_head", expert=hh)); tags.append(("head", c, l, hh))
    res = eng.run(pre, spawns, record_routing=False, log=log)
    d, sd = res.delta, res.sp_delta
    pos = {c: i for i, c in enumerate(ids)}
    rep["delta_clean_maxdiff_vs_hf"] = float(max(abs(d[pos[c]] - ref["clean"][c]) for c in ids))
    rep["delta_corrupt_maxdiff_vs_hf"] = float(max(abs(d[n + pos[c]] - ref["corrupt"][c]) for c in ids))
    rep["mean_delta_clean_hf"] = float(np.mean(list(ref["clean"].values())))
    rep["mean_delta_corrupt_hf"] = float(np.mean(list(ref["corrupt"].values())))
    # direction antisymmetry: clean delta of d=1 equals minus the corrupted delta of d=0 (same prompt, swapped true/foil)
    rep["antisymmetry_maxdiff"] = float(max(abs(d[pos[2 * (c // 2) + 1]] + d[n + pos[2 * (c // 2)]]) for c in ids if c % 2 == 0))
    tk = np.array([t[0] for t in tags])
    for k in FINAL_KINDS:
        js = [j for j in np.nonzero(tk == "final")[0] if tags[j][3] == k]
        rep[f"final_{k}"] = stats(sd[js], [ref["final"][tags[j][1:]] for j in js], [d[n + pos[tags[j][1]]] for j in js],
                                  [ref["corrupt"][tags[j][1]] for j in js])
        ji = [j for j in np.nonzero(tk == "identity")[0] if tags[j][3] == k]
        rep[f"identity_{k}_maxdiff"] = float(max(abs(sd[j] - d[pos[tags[j][1]]]) for j in ji))
        # mean curves
        ce = np.zeros(L); ch = np.zeros(L)
        for j in js:
            c, l = tags[j][1], tags[j][2]
            ce[l] += (sd[j] - d[n + pos[c]]) / n
            ch[l] += (ref["final"][(c, l, k)] - ref["corrupt"][c]) / n
        rep[f"final_{k}"]["mean_curve_maxdiff"] = float(np.abs(ce - ch).max())
        rep[f"final_{k}"]["mean_curve_engine"] = [round(float(v), 3) for v in ce]
        rep[f"final_{k}"]["mean_curve_hf"] = [round(float(v), 3) for v in ch]
    js = np.nonzero(tk == "head")[0]
    rep["final_attn_head"] = stats(sd[js], [ref["head"][tags[j][1:]] for j in js], [d[n + pos[tags[j][1]]] for j in js],
                                   [ref["corrupt"][tags[j][1]] for j in js])
    rep["engine_final_pass_s"] = res.extra["total_s"]
    del res, eng
    torch.cuda.empty_cache()
    # ---- STR positions: SubjectEngine.run_subject
    seng = SubjectEngine(REPO)
    gids = ids[: args.n_grid]
    pre, spawns, tags, unit = [], [], [], {}
    for c in gids:
        x = dc[c]
        for p in range(x.first_str, x.T):
            ci = len(pre); pre.append(SubjectPrefill(x.clean_ids, x.true_id, x.foil_id, rec_pos=p))
            di = len(pre); pre.append(SubjectPrefill(x.corrupt_ids, x.true_id, x.foil_id, rec_pos=p))
            unit[(c, p)] = (ci, di)
            for l in range(L):
                for k in GRID_KINDS:
                    spawns.append(SubjectSpawn(l, di, ci, k, p)); tags.append(("grid", c, p, l, k))
            spawns.append(SubjectSpawn(0, di, ci, "zero", p)); tags.append(("zero", c, p, 0, "zero"))
            spawns.append(SubjectSpawn(L // 2, ci, ci, "resid", p)); tags.append(("identity", c, p, L // 2, "resid"))
    # window-5 rows in a SEPARATE pass (as in production): in SubjectEngine.run_subject a pass that contains any
    # window > 1 row also overwrites the spawn-layer MoE output of pre-MoE kinds (attn_layer) with the clean one
    spawns5, tags5 = [], []
    for (c, p), (ci, di) in unit.items():
        for l in range(L):
            a, b = max(0, l - HALF), min(L - 1, l + HALF)
            spawns5.append(SubjectSpawn(a, di, ci, "layer", p, window=b - a + 1)); tags5.append(("w5", c, p, l, "layer"))
    res5 = seng.run_subject(pre, spawns5, log=log, metrics=True)
    res = seng.run_subject(pre, spawns, log=log, metrics=True)
    # prefill rows are batched with different suffix tokens in the two passes (different GEMM shapes): bf16-level
    # differences; every spawn is compared against the parent row of ITS OWN pass
    rep["grid_prefill_two_pass_maxdiff"] = float(np.abs(res.delta - res5.delta).max())
    d = res.delta
    sd = np.concatenate([res.sp_delta, res5.sp_delta - res5.delta[[unit[t[1:3]][1] for t in tags5]] + res.delta[[unit[t[1:3]][1] for t in tags5]]])
    ms = {k: np.concatenate([res.extra["metrics_spawn"][k], res5.extra["metrics_spawn"][k]]) for k in res.extra["metrics_spawn"]}
    tags = tags + tags5
    tk = np.array([t[0] for t in tags])
    rep["grid_units"] = len(unit)
    rep["grid_positions_by_category"] = pd.Series([P.position_category(p, dc[c].str_pos, dc[c].T) for c, p in unit]).value_counts().to_dict()
    for k in GRID_KINDS:
        js = [j for j in np.nonzero(tk == "grid")[0] if tags[j][4] == k]
        rep[f"grid_{k}"] = stats(sd[js], [ref["grid"][tags[j][1:]] for j in js], [d[unit[tags[j][1:3]][1]] for j in js],
                                 [ref["corrupt"][tags[j][1]] for j in js])
        jn = [j for j in js if tags[j][2] < dc[tags[j][1]].T - 1]  # positions before the final token only
        rep[f"grid_{k}_nonfinal"] = stats(sd[jn], [ref["grid"][tags[j][1:]] for j in jn], [d[unit[tags[j][1:3]][1]] for j in jn],
                                          [ref["corrupt"][tags[j][1]] for j in jn])
    js = np.nonzero(tk == "w5")[0]
    rep["grid_w5_layer"] = stats(sd[js], [ref["w5"][tags[j][1:4]][0] for j in js], [d[unit[tags[j][1:3]][1]] for j in js],
                                 [ref["corrupt"][tags[j][1]] for j in js])
    rep["grid_w5_p_true_maxdiff"] = float(max(abs(ms["p_true"][j] - ref["w5"][tags[j][1:4]][1]) for j in js))
    js = np.nonzero(tk == "zero")[0]
    rep["grid_zero_null_maxdiff"] = float(max(abs(sd[j] - d[unit[tags[j][1:3]][1]]) for j in js))
    js = np.nonzero(tk == "identity")[0]
    rep["grid_identity_resid_maxdiff"] = float(max(abs(sd[j] - d[unit[tags[j][1:3]][0]]) for j in js))
    rep["engine_grid_pass_s"] = res.extra["total_s"] + res5.extra["total_s"]
    for cal in ("verify_ext6_str_olmoe.json", "verify_ext6_str_grid_olmoe.json", "verify_ext2_attn_olmoe.json"):
        pth = os.path.join("/home/ubuntu/MOE/results", cal)
        if os.path.exists(pth):
            c = json.load(open(pth))
            rep.setdefault("calibration", {})[cal] = {k: v for k, v in c.items() if isinstance(v, (int, float)) and ("maxdiff" in k or "corr" in k or "within" in k)}
    # verdict (thresholds of the earlier verifications: bf16 noise floor, rescue r >= 0.95)
    # criterion: bf16 noise floor of the earlier verifications (mean |diff| <= 0.1 logit, >= 90 % of rows within 0.25, as the
    # ext6 grid check: 93 %) and
    # rescue r >= 0.9 (attn_head effects are small, SD ~0.15, so r is bounded by the ~0.06 noise floor)
    groups = [f"{g}_{k}" for g in ("final", "grid") for k in FINAL_KINDS] + ["final_attn_head", "grid_w5_layer"]
    rep["min_rescue_corr"] = float(min(rep[g]["rescue_corr_vs_hf"] for g in groups))
    rep["max_meanabsdiff"] = float(max(rep[g]["meanabsdiff"] for g in groups))
    rep["min_frac_within_0.25"] = float(min(rep[g]["frac_within_0.25"] for g in groups))
    rep["pass"] = bool(rep["min_rescue_corr"] >= 0.9 and rep["max_meanabsdiff"] <= 0.1 and rep["min_frac_within_0.25"] >= 0.9
                       and rep["grid_zero_null_maxdiff"] <= 0.25 and rep["delta_clean_maxdiff_vs_hf"] <= 0.5 and rep["antisymmetry_maxdiff"] <= 0.125)
    with open("/home/ubuntu/MOE/results/verify_ext7_wino_olmoe.json", "w") as f:
        json.dump(rep, f, indent=1, default=str)
    for k, v in rep.items():
        if k not in ("pairs", "calibration") and not (isinstance(v, dict) and "mean_curve_engine" in v):
            log(f"{k}: {v}")
        elif isinstance(v, dict) and "mean_curve_engine" in v:
            log(f"{k}: " + json.dumps({kk: vv for kk, vv in v.items() if not kk.startswith("mean_curve_e") and kk != "mean_curve_hf"}))


if __name__ == "__main__":
    main()
