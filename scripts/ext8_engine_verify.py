"""ext8 engine verification on OLMoE-1B-7B-0125: attention steps in `multi`, all-layer `multi`, and the noising direction,
against transformers hooks (model fully on the GPU, eager attention).

Units = STR pairs (clean prompt, corrupted prompt, r, r'): CounterFact STR pairs built like ext6 (first symmetric donor)
and WinoGrande STR pairs (OLMoE margin pairs of results/wino_olmoe, both directions of each pair).

HF side, per unit: clean and corrupted forwards with recording hooks at every layer (final position: attention-sublayer
output after o_proj, MoE output, and the per-expert contributions c_e = w_e * E_e(x) of the routed experts computed
from the MoE input), then patched forwards of a BASE run with replacement hooks taken from a SOURCE run:
    attn       self_attn output[final] := source            (the MoE of that layer recomputes on the patched residual)
    moe        MoE output[final] := source
    block      both
    coal(S)    MoE output[final] := own + sum_{e in S} (c_e(source) - c_e(own))   (own contributions from the hook input)
Denoising: base = corrupted, source = clean. Noising: base = clean, source = corrupted.
Engine side: the same patches as `multi` spawns (parent = base row, clean = source row), single-layer kinds for
reference, and internal invariants:
    * all-layer block reproduces the source run's Delta exactly (denoising: clean; noising: corrupted);
    * single-step multi attn_layer / block == the single-layer kinds (vectors and Delta);
    * all-layer coalition_set(S = all experts) == all-layer `layer` (fp32 summation order);
    * identity: all-layer attn on the clean row with source = itself;
    * metrics with kl_ref (noising all-block vs the corrupted row: KL ~ 0);
    * stress pass: many all-layer coalition_set multi rows in one pass (time, peak memory) and the same rows' Delta in
      a small pass (batch independence).

Usage: python scripts/ext8_engine_verify.py [--engine dev|main] [--n-cf 6] [--n-wino-pairs 3] [--stress 20000]
       -> results/verify_ext8_engine_olmoe.json
"""
import argparse, importlib, json, os, random, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from moetrace.arch import snapshot_dir
from moetrace.data import load_records, prepare_case, shuffled_order
from moetrace.verify import _hf_load, _moe_module
from moetrace import ext6_str as S6

REPO = "allenai/OLMoE-1B-7B-0125"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------------------------
# units
# ------------------------------------------------------------------------------------------------------------------
def cf_units(n):
    tok = AutoTokenizer.from_pretrained(snapshot_dir(REPO))
    recs = load_records()
    index = S6.donor_index(recs)
    out = []
    for ri in shuffled_order(len(recs), 0):
        c, _ = prepare_case(recs[ri], tok)
        if c is None:
            continue
        ds = S6.candidates(c, recs[ri], tok, index, True)
        if ds:
            out.append(dict(src="cf", name=f"cf{c.case_id}", clean=list(c.ids), corrupt=list(ds[0].ids), true=int(c.true_id), foil=int(c.foil_id)))
        if len(out) >= n:
            break
    return out


def wino_units(n_pairs):
    p = pd.read_parquet("/home/ubuntu/MOE/data/wino_str/pairs_train_xl_olmoe.parquet")
    s = pd.read_parquet("/home/ubuntu/MOE/results/wino_olmoe/scan_pairs.parquet")
    keep = set(s[s.margin].pair_id)
    p = p[p.pair_id.isin(keep)].sort_values("pair_id").reset_index(drop=True)
    idx = list(range(len(p)))
    random.Random(8).shuffle(idx)
    out = []
    for i in idx[:n_pairs]:
        r = p.iloc[i]
        a, b = json.loads(r.ids_a), json.loads(r.ids_b)
        out.append(dict(src="wino", name=f"w{i}a", clean=a, corrupt=b, true=int(r.trig_a), foil=int(r.trig_b)))
        out.append(dict(src="wino", name=f"w{i}b", clean=b, corrupt=a, true=int(r.trig_b), foil=int(r.trig_a)))
    return out


# ------------------------------------------------------------------------------------------------------------------
# patch configurations (layer, kind, experts); kinds here: attn / moe / block / coal
# ------------------------------------------------------------------------------------------------------------------
def configs(L):
    A = list(range(L))
    return {
        "attn2": [(4, "attn"), (10, "attn")],
        "attn3": [(2, "attn"), (7, "attn"), (12, "attn")],
        "moe_attn": [(3, "moe"), (8, "attn")],
        "attn_moe": [(3, "attn"), (8, "moe")],
        "block_attn": [(4, "block"), (10, "attn")],
        "attn_block": [(4, "attn"), (10, "block")],
        "block3": [(2, "block"), (7, "block"), (12, "block")],
        "attn_coal": [(3, "attn"), (6, "coal"), (11, "attn"), (13, "coal")],
        "all_moe": [(l, "moe") for l in A],
        "all_attn": [(l, "attn") for l in A],
        "all_block": [(l, "block") for l in A],
        "coal_all": [(l, "coal") for l in A],
        "coal_sparse": "sparse",
        "single_moe_0": [(0, "moe")], "single_moe_5": [(5, "moe")], "single_moe_10": [(10, "moe")], "single_moe_15": [(15, "moe")],
        "single_attn_1": [(1, "attn")], "single_attn_8": [(8, "attn")], "single_attn_14": [(14, "attn")],
        "single_block_6": [(6, "block")],
    }


def coal_sets(unit_i, direction, cfg, L, src_routes, rng_seed=0):
    """Per-layer expert sets S for the coal steps: 3 random experts routed in the source run (both engine and HF),
    or for 'sparse' 10 random (layer, expert) source-active pairs grouped by layer."""
    rng = random.Random(1000 * unit_i + (0 if direction == "denoise" else 1) + rng_seed)
    if cfg == "sparse":
        pool = [(l, e) for l in range(L) for e in src_routes[l]]
        pick = sorted(rng.sample(pool, min(10, len(pool))))
        by = {}
        for l, e in pick:
            by.setdefault(l, []).append(e)
        return [(l, "coal", tuple(sorted(es))) for l, es in sorted(by.items())]
    out = []
    for (l, kind) in cfg:
        if kind == "coal":
            out.append((l, kind, tuple(sorted(rng.sample(sorted(src_routes[l]), min(3, len(src_routes[l])))))))
        else:
            out.append((l, kind, ()))
    return out


ENG_KIND = {"attn": "attn_layer", "moe": "layer", "block": "block", "coal": "coalition_set"}


# ------------------------------------------------------------------------------------------------------------------
# HF reference
# ------------------------------------------------------------------------------------------------------------------
def _o(out):
    return out[0] if isinstance(out, tuple) else out


def _rep(out, new_final):
    o = _o(out).clone()
    o[0, -1] = new_final
    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o


def _contribs(moe, x):
    """{expert: fp32 [H]} for the final-position token x [1, H] bf16 (engine numerics: bf16 expert MLP, fp32 weight)."""
    _, wts, idx = moe.gate(x)
    ex = moe.experts
    out = {}
    for s in range(idx.shape[1]):
        e = int(idx[0, s])
        g, u = F.linear(x, ex.gate_up_proj[e]).chunk(2, dim=-1)
        y = F.linear(ex.act_fn(g) * u, ex.down_proj[e])
        out[e] = y[0].float() * wts[0, s].float()
    return out


@torch.no_grad()
def hf_reference(units, plan, L):
    """plan: unit index -> list of (tag, direction, steps); returns rec (routing per run) and tag -> Delta."""
    model = _hf_load(REPO)

    def fwd(ids):
        return model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float()

    res, routes = {}, {}
    for ui, u in enumerate(units):
        st = {}
        for run, ids in (("clean", u["clean"]), ("corrupt", u["corrupt"])):
            hs = []
            for l in range(L):
                lay = model.model.layers[l]
                moe = _moe_module(model, l)

                def ra(mod, args, out, l=l, run=run):
                    st[(run, "attn", l)] = _o(out)[0, -1].detach().clone()

                def rm(mod, args, out, l=l, run=run, moe=moe):
                    st[(run, "moe", l)] = _o(out)[0, -1].detach().clone()
                    st[(run, "c", l)] = _contribs(moe, args[0][0, -1:].detach())
                hs += [lay.self_attn.register_forward_hook(ra), moe.register_forward_hook(rm)]
            lg = fwd(ids)
            for h in hs:
                h.remove()
            res[(ui, run)] = float(lg[u["true"]] - lg[u["foil"]])
            routes[(ui, run)] = [sorted(st[(run, "c", l)].keys()) for l in range(L)]
        for tag, direction, steps in plan(ui, routes):
            base, src = ("corrupt", "clean") if direction == "denoise" else ("clean", "corrupt")
            hs = []
            for (l, kind, ex) in steps:
                lay = model.model.layers[l]
                moe = _moe_module(model, l)
                if kind in ("attn", "block"):
                    hs.append(lay.self_attn.register_forward_hook(lambda m, a, o, k=(src, "attn", l): _rep(o, st[k])))
                if kind in ("moe", "block"):
                    hs.append(moe.register_forward_hook(lambda m, a, o, k=(src, "moe", l): _rep(o, st[k])))
                if kind == "coal":
                    def hc(mod, args, out, l=l, ex=ex, moe=moe):
                        own = _contribs(moe, args[0][0, -1:])
                        srcc = st[(src, "c", l)]
                        v = sum((srcc.get(e, 0.0) - own.get(e, 0.0)) for e in ex)
                        o = _o(out)
                        return _rep(out, (o[0, -1].float() + v).to(o.dtype))
                    hs.append(moe.register_forward_hook(hc))
            lg = fwd(u["corrupt"] if base == "corrupt" else u["clean"])
            for h in hs:
                h.remove()
            res[(ui, tag)] = float(lg[u["true"]] - lg[u["foil"]])
        if ui % 4 == 3:
            log(f"HF: {ui + 1}/{len(units)} units")
    del model
    torch.cuda.empty_cache()
    return res, routes


def _stats(z):
    z = np.asarray(z, dtype=np.float64)
    return {"n": int(len(z)), "maxabs": float(np.abs(z).max()), "meanabs": float(np.abs(z).mean()),
            "frac_exact": float((z == 0).mean()), "frac_within_0.1": float((np.abs(z) < 0.1).mean()),
            "frac_within_0.25": float((np.abs(z) < 0.25).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="dev", choices=["dev", "main"])
    ap.add_argument("--n-cf", type=int, default=6)
    ap.add_argument("--n-wino-pairs", type=int, default=3)
    ap.add_argument("--stress", type=int, default=20000)
    ap.add_argument("--out", default="results/verify_ext8_engine_olmoe.json")
    args = ap.parse_args()
    E = importlib.import_module("moetrace.engine_ext8_dev" if args.engine == "dev" else "moetrace.engine")
    Engine, PrefillSpec, SpawnSpec, DiagSpec = E.Engine, E.PrefillSpec, E.SpawnSpec, E.DiagSpec
    units = cf_units(args.n_cf) + wino_units(args.n_wino_pairs)
    n = len(units)
    log(f"{n} units ({args.n_cf} CounterFact, {2 * args.n_wino_pairs} WinoGrande directed); engine module {E.__name__}")
    eng = Engine(REPO)
    L, NE = eng.spec.n_layers, eng.spec.n_experts
    pre = [PrefillSpec(u["clean"], u["true"], u["foil"], clean_ref=i) for i, u in enumerate(units)] + \
          [PrefillSpec(u["corrupt"], u["true"], u["foil"], clean_ref=i) for i, u in enumerate(units)]
    # pass 0: engine routing of the prefill rows (coalition sets are drawn from experts routed in engine AND HF source runs)
    r0 = eng.run(pre, [], record_routing=True)
    eng_routes = {(i, run): [sorted(int(e) for e in r0.route_idx[l, i + (0 if run == "clean" else n)]) for l in range(L)]
                  for i in range(n) for run in ("clean", "corrupt")}
    CF = configs(L)
    plans = {}

    def plan(ui, hf_routes):
        out = []
        for direction in ("denoise", "noise"):
            src = "clean" if direction == "denoise" else "corrupt"
            common = [sorted(set(hf_routes[(ui, src)][l]) & set(eng_routes[(ui, src)][l])) for l in range(L)]
            for name, cfg in CF.items():
                steps = coal_sets(ui, direction, cfg, L, common) if (cfg == "sparse" or any(k == "coal" for _, k in cfg)) else \
                    [(l, k, ()) for (l, k) in cfg]
                out.append((f"{direction}:{name}", direction, steps))
        plans[ui] = out
        return out

    t0 = time.time()
    hf, hf_routes = hf_reference(units, plan, L)
    t_hf = time.time() - t0
    log(f"HF reference done ({t_hf:.0f}s)")
    route_agree = float(np.mean([hf_routes[k][l] == eng_routes[k][l] for k in eng_routes for l in range(L)]))

    # ---- engine pass 1: every planned configuration as multi + single-layer reference kinds + invariants
    spawns, tags = [], []

    def add(sp, tag):
        spawns.append(sp)
        tags.append(tag)

    ALL = tuple(range(NE))
    for ui in range(n):
        for tag, direction, steps in plans[ui]:
            par, src = (n + ui, ui) if direction == "denoise" else (ui, n + ui)
            st = tuple((l, ENG_KIND[k], ex if k == "coal" else None) for (l, k, ex) in steps)
            add(SpawnSpec(st[0][0], par, src, "multi", steps=st, kl_ref=ui), ("multi", ui, tag))
            if len(steps) == 1:  # single-layer kind reference
                l, k, _ = steps[0]
                add(SpawnSpec(l, par, src, ENG_KIND[k]), ("single", ui, tag))
        for direction in ("denoise", "noise"):
            par, src = (n + ui, ui) if direction == "denoise" else (ui, n + ui)
            add(SpawnSpec(0, par, src, "multi", steps=tuple((l, "coalition_set", ALL) for l in range(L))), ("setall", ui, direction))
            add(SpawnSpec(0, par, src, "multi", steps=tuple((l, "layer", None) for l in range(L))), ("layerall", ui, direction))
            # all-block with KL reference = the BASE row (kl_ref != clean): the row equals the source run, so its KL must
            # equal the source prefill row's KL to the base row (noising: KL(corrupt || clean) = the corrupted row's kl_to_clean)
            add(SpawnSpec(0, par, src, "multi", steps=tuple((l, "block", None) for l in range(L)), kl_ref=par), ("blockall_klbase", ui, direction))
        # DLA diagnostic check: single-expert patches (spawn vectors = delta_e) at three layers, every clean-routed expert
        for l in (3, 9, 15):
            for e in eng_routes[(ui, "clean")][l]:
                add(SpawnSpec(l, n + ui, ui, "expert", expert=e), ("dla_expert", ui, (l, e)))
        # identity: clean row with itself as source
        add(SpawnSpec(0, ui, ui, "multi", steps=tuple((l, "attn_layer", None) for l in range(L))), ("identity_attn", ui, None))
        add(SpawnSpec(0, ui, ui, "multi", steps=tuple((l, "block", None) for l in range(L))), ("identity_block", ui, None))
    diag = DiagSpec(spawn_vectors=True, contrib_dla=True)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    res = eng.run(pre, spawns, record_routing=True, log=log, diag=diag, metrics=True)
    t_pass1 = time.time() - t1
    d, sd = res.delta, res.sp_delta
    dg = res.extra["diag"]
    rep = {"engine_module": E.__name__, "n_units": n, "units": [{k: u[k] for k in ("src", "name")} | {"T": len(u["clean"])} for u in units],
           "hf_s": t_hf, "pass1_s": t_pass1, "pass1_spawn_rows": len(spawns), "routing_agreement_engine_vs_hf_layers": route_agree,
           "delta_clean_maxdiff_vs_hf": float(max(abs(d[i] - hf[(i, "clean")]) for i in range(n))),
           "delta_corrupt_maxdiff_vs_hf": float(max(abs(d[n + i] - hf[(i, "corrupt")]) for i in range(n))),
           "mean_delta_clean": float(d[:n].mean()), "mean_delta_corrupt": float(d[n:].mean())}
    idx = {t: j for j, t in enumerate(tags)}
    # ---- vs HF, per configuration and direction
    per = {}
    for direction in ("denoise", "noise"):
        for name in CF:
            tag = f"{direction}:{name}"
            js = [idx[("multi", ui, tag)] for ui in range(n)]
            en = np.array([sd[j] for j in js])
            hfv = np.array([hf[(ui, tag)] for ui in range(n)])
            base_en = d[n:] if direction == "denoise" else d[:n]
            base_hf = np.array([hf[(ui, "corrupt" if direction == "denoise" else "clean")] for ui in range(n)])
            eff_en, eff_hf = en - base_en, hfv - base_hf
            st = _stats(en - hfv)
            st["effect_corr"] = float(np.corrcoef(eff_en, eff_hf)[0, 1]) if np.std(eff_hf) > 0 and np.std(eff_en) > 0 else None
            st["mean_effect_engine"] = float(eff_en.mean())
            st["mean_effect_hf"] = float(eff_hf.mean())
            per[tag] = st
    rep["vs_hf"] = per
    allz = np.concatenate([[sd[idx[("multi", ui, f"{dr}:{nm}")]] - hf[(ui, f"{dr}:{nm}")] for ui in range(n)] for dr in ("denoise", "noise") for nm in CF])
    rep["vs_hf_all"] = _stats(allz)
    ee = np.concatenate([[sd[idx[("multi", ui, f"{dr}:{nm}")]] - (d[n + ui] if dr == "denoise" else d[ui]) for ui in range(n)] for dr in ("denoise", "noise") for nm in CF])
    hh = np.concatenate([[hf[(ui, f"{dr}:{nm}")] - hf[(ui, "corrupt" if dr == "denoise" else "clean")] for ui in range(n)] for dr in ("denoise", "noise") for nm in CF])
    rep["vs_hf_all_effect_corr"] = float(np.corrcoef(ee, hh)[0, 1])
    # calibration: the same numbers for the established single-layer `layer` kind (HF single_moe configs)
    # ---- invariants
    inv = {}
    for direction, ref_rows in (("denoise", np.arange(n)), ("noise", n + np.arange(n))):
        js = [idx[("multi", ui, f"{direction}:all_block")] for ui in range(n)]
        inv[f"{direction}_all_block_minus_source_delta"] = _stats(np.array([sd[j] for j in js]) - d[ref_rows])
        hfz = np.array([hf[(ui, f"{direction}:all_block")] - hf[(ui, "clean" if direction == "denoise" else "corrupt")] for ui in range(n)])
        inv[f"{direction}_all_block_minus_source_delta_HF"] = _stats(hfz)
        z, vz = [], []
        for ui in range(n):
            ja, jb = idx[("setall", ui, direction)], idx[("layerall", ui, direction)]
            z.append(sd[ja] - sd[jb])
            vz.append(max(float((x - y).abs().max()) for x, y in zip(dg["spawn_v"][ja], dg["spawn_v"][jb])))
        inv[f"{direction}_all_setall_vs_all_layer_delta"] = _stats(np.array(z))
        inv[f"{direction}_all_setall_vs_all_layer_vec_maxabs"] = float(max(vz))
        j = [idx[("blockall_klbase", ui, direction)] for ui in range(n)]
        kl_sp = res.metrics_spawn["kl_to_clean"][j]
        if direction == "noise":  # KL(row = corrupt || clean) vs the corrupted prefill row's KL to its clean_ref (the clean row)
            kl_ref_v = res.metrics_prefill["kl_to_clean"][n:]
            inv["noise_all_block_kl_ref_base_vs_prefill_kl"] = {"maxabs": float(np.abs(kl_sp - kl_ref_v).max()),
                                                                "mean_spawn": float(kl_sp.mean()), "mean_prefill": float(kl_ref_v.mean())}
        else:  # KL(row = clean || corrupt): positive, not referenced to the clean row (kl_ref honoured)
            inv["denoise_all_block_kl_ref_base_mean"] = float(kl_sp.mean())
        inv[f"{direction}_all_block_delta_vs_source_exact_frac"] = float(np.mean([sd[jj] == d[r] for jj, r in zip(j, ref_rows)]))
    for name in [k for k in CF if k.startswith("single_")]:
        for direction in ("denoise", "noise"):
            tag = f"{direction}:{name}"
            z, vz = [], []
            for ui in range(n):
                ja, jb = idx[("multi", ui, tag)], idx[("single", ui, tag)]
                z.append(sd[ja] - sd[jb])
                va = dg["spawn_v"].get(ja)
                vb = dg["spawn_v"].get(jb)
                if va is not None and vb is not None:
                    va = va[0] if isinstance(va, list) else va
                    vz.append(float((va - vb).abs().max()))
            inv[f"multi1_vs_single_{tag}_delta_maxabs"] = float(np.abs(z).max())
            if vz:
                inv[f"multi1_vs_single_{tag}_vec_maxabs"] = float(max(vz))
    for t in ("identity_attn", "identity_block"):
        inv[f"{t}_minus_clean_delta"] = _stats(np.array([sd[idx[(t, ui, None)]] - d[ui] for ui in range(n)]))
    # spawn-vector bookkeeping: one entry per step for multi rows with attention steps
    ok = []
    for ui in range(n):
        for tag, direction, steps in plans[ui]:
            j = idx[("multi", ui, tag)]
            v = dg["spawn_v"].get(j)
            ok.append(isinstance(v, list) and len(v) == len(steps) and len(res.extra["multi_vnorm"].get(j, [])) == len(steps) - 1)
    inv["spawn_vectors_one_per_step_frac"] = float(np.mean(ok))
    rep["invariants"] = inv
    # ---- DLA diagnostic vs the spawn vectors (same quantity two ways) and vs the actual single-expert rescue
    gamma = eng.g["norm"].float().cpu()
    head = eng.g["head"]
    cd, rms = dg["contrib_dla"], dg["final_rms"]
    a_vec, a_diag, resc = [], [], []
    for ui in range(n):
        u = units[ui]
        wd = (head[u["true"]].float() - head[u["foil"]].float()).cpu()
        for l in (3, 9, 15):
            for e in eng_routes[(ui, "clean")][l]:
                j = idx[("dla_expert", ui, (l, e))]
                v = dg["spawn_v"][j]
                a_vec.append(float((v * gamma) @ wd) / float(rms[n + ui]))
                ci = list(r0.route_idx[l, ui]).index(e) if e in r0.route_idx[l, ui] else None
                cc = list(res.route_idx[l, ui])
                cn = list(res.route_idx[l, n + ui])
                pc = cd[l, ui, cc.index(e)] if e in cc else 0.0
                pn = cd[l, n + ui, cn.index(e)] if e in cn else 0.0
                a_diag.append(float(pc - pn) / float(rms[n + ui]))
                resc.append(float(sd[j] - d[n + ui]))
    a_vec, a_diag, resc = map(np.array, (a_vec, a_diag, resc))
    rep["dla"] = {"n": int(len(a_vec)), "diag_vs_vector_maxabs": float(np.abs(a_vec - a_diag).max()),
                  "diag_vs_vector_max_rel": float((np.abs(a_vec - a_diag) / np.maximum(np.abs(a_vec), 1e-3)).max()),
                  "corr_dla_vs_rescue": float(np.corrcoef(a_diag, resc)[0, 1]),
                  "mean_dla": float(a_diag.mean()), "mean_rescue": float(resc.mean())}
    rep["pass1_peak_mem_GB"] = float(torch.cuda.max_memory_allocated() / 2**30)
    del res
    torch.cuda.empty_cache()

    # ---- stress: many all-layer multi rows (coalition_set at every layer, random clean-active subsets) in one pass
    if args.stress:
        rng = random.Random(5)
        st_sp, st_tag = [], []
        for s_ in range(args.stress):
            ui = s_ % n
            ks = rng.choice([1, 2, 4, 8])
            steps = []
            for l in range(L):
                ce = eng_routes[(ui, "clean")][l]
                steps.append((l, "coalition_set", tuple(sorted(rng.sample(ce, min(ks, len(ce)))))))
            if s_ % 7 == 0:  # some rows with attention / block steps mixed in
                steps = [(l, ("attn_layer" if l % 3 == 0 else "block" if l % 3 == 1 else k), (None if l % 3 < 2 else ex)) for (l, k, ex) in steps]
            st_sp.append(SpawnSpec(0, n + ui, ui, "multi", steps=tuple(steps)))
            st_tag.append(ui)
        torch.cuda.reset_peak_memory_stats()
        t2 = time.time()
        rs = eng.run(pre, st_sp, record_routing=False, log=log)
        t_big = time.time() - t2
        peak = torch.cuda.max_memory_allocated() / 2**30
        small = list(range(0, args.stress, max(1, args.stress // 200)))
        rs2 = eng.run(pre, [st_sp[i] for i in small], record_routing=False)
        z = rs.sp_delta[small] - rs2.sp_delta
        rep["stress"] = {"rows": args.stress, "steps_per_row": L, "pass_s": t_big, "peak_mem_GB": float(peak),
                         "batch_independence_delta": _stats(z), "mean_rescue": float(np.mean(rs.sp_delta - rs.delta[n + np.array(st_tag)]))}
        del rs, rs2
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep.items():
        if k not in ("units", "vs_hf"):
            log(f"{k}: {json.dumps(v) if isinstance(v, dict) else v}")
    for k, v in rep["vs_hf"].items():
        log(f"vs_hf {k}: maxabs {v['maxabs']:.3f} meanabs {v['meanabs']:.3f} within0.25 {v['frac_within_0.25']:.2f} "
            f"effect r {v['effect_corr']} engine {v['mean_effect_engine']:+.3f} hf {v['mean_effect_hf']:+.3f}")


if __name__ == "__main__":
    main()
