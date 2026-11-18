"""GPU tests of moetrace/relp_moe_p8.py on tiny random HF checkpoints (helpers from the Phase-7 tests).
Run through the queue: bash scripts/gpu_queue.sh relp-test-p8 -- .venv-relp/bin/python -m pytest tests/test_relp_moe_p8.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_relp_moe_stream import DEV, FAMS, tiny  # noqa: E402
from test_relp_moe_patch import pair_batch  # noqa: E402
from moetrace.relp_moe_p8 import (FREEZE, RULES, RuleRelP, endpoint_triples, gate_probe, multi_backward,  # noqa: E402
                                  patch_p8, path_forward, path_triples)
from moetrace.relp_moe_patch import (diagnose_pass, enumerate_units, patch_pass, relp_backward_endpoint,  # noqa: E402
                                     routes_pass)
from moetrace.relp_moe_stream import Group, Override, StreamRelP, forward_pass  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
ALL = list(RULES)


def u9(u8, neuron=-1):
    return np.concatenate([u8, np.full((len(u8), 1), neuron, dtype=np.int64)], 1)


def scored(m, A, Bb, tg, fo):
    g = Group("s", torch.cat([A, Bb]), tg * 2, fo * 2, record_h_in=True)
    forward_pass(m, [g])
    return g


@pytest.mark.parametrize("family", FAMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_relp_variant_is_the_engine(family, dtype, tmp_path):
    """T1 / T3: rule 'relp' = StreamRelP bit for bit (forward, both endpoints), alone or next to other variants."""
    d = tiny(family, tmp_path)
    m0, m = StreamRelP(str(d), DEV, dtype), RuleRelP(str(d), DEV, dtype)
    n, T = 3, 5
    A, Bb = pair_batch(n, T, 51)
    tg, fo = [5, 6, 7], [9, 10, 11]
    pool = torch.ones(n, T, dtype=torch.bool)
    pool[:, 0] = False
    g0, g = scored(m0, A, Bb, tg, fo), scored(m, A, Bb, tg, fo)
    assert torch.equal(g0.delta, g.delta)
    tau = torch.ones(m.L)
    for ep in ("A", "B"):
        t0, s0, r0 = relp_backward_endpoint(m0, g0, n, pool, ep, per_row=True)
        one = multi_backward(m, g, endpoint_triples(n, ep), pool, ["relp"], neuron_agg=("relp",))
        many = multi_backward(m, g, endpoint_triples(n, ep), pool, ALL, neuron_agg=("relp", "relp_rw"), tau=tau)
        for res in (one, many):
            assert torch.equal(res["relp"]["total"], t0.cpu())
            assert torch.equal(res["support"], s0.cpu())
            assert torch.equal(res["relp"]["pair_expert"], r0)
        # tracked neurons: all neurons of two expert-positions sum to their per-pair expert scores
        cand = torch.nonzero(r0.abs().sum(0) > 0)[:2]                    # (layer, position, expert)
        track = torch.cat([torch.cat([c.repeat(m.I, 1)[:, :2], c[2].repeat(m.I, 1), torch.arange(m.I)[:, None]], 1)
                           for c in cand])
        tr = multi_backward(m, g, endpoint_triples(n, ep), pool, ["relp", "grad"], track=track)
        for ci, c in enumerate(cand):
            got = tr["relp"]["tracked"][:, ci * m.I:(ci + 1) * m.I].sum(1)
            torch.testing.assert_close(got, r0[:, c[0], c[1], c[2]], rtol=1e-4, atol=1e-5)
        # per-pair expert scores sum to the neuron-level totals for a second variant too
        torch.testing.assert_close(many["relp_rw"]["pair_expert"].double().sum(0), many["relp_rw"]["total"].sum(-1),
                                   rtol=1e-3, atol=1e-4)
        for v in ALL:
            assert torch.isfinite(many[v]["pair_expert"]).all()
    # the variants differ from each other (the switches do something)
    assert not torch.equal(many["relp"]["pair_expert"], many["grad"]["pair_expert"])
    assert not torch.equal(many["relp"]["pair_expert"], many["relp_rw"]["pair_expert"])
    assert not torch.equal(many["relp_rw"]["pair_expert"], many["relp_soft1"]["pair_expert"])


@pytest.mark.parametrize("family", FAMS)
def test_router_variants_keep_forward_values(family, tmp_path):
    """T9: under every rule the recomputed layer output has the values of the 'relp' recomputation (bf16)."""
    d = tiny(family, tmp_path)
    m = RuleRelP(str(d), DEV, torch.bfloat16)
    A, Bb = pair_batch(3, 5, 52)
    g = scored(m, A, Bb, [5, 6, 7], [9, 10, 11])
    cos, sin = m.rope(5, g.h)
    mask = m.causal_mask(6, 5, g.h.dtype)
    for l, w in m.stream():
        outs = {}
        for v in ("relp", "relp_rw", "relp_rw3", "relp_soft1", "relp_nonorm", "relp_noattn"):
            m.set_rule(v, 1.0)
            with torch.enable_grad():
                hin = g.h_in[l].detach().requires_grad_(True)
                outs[v], _ = m.layer(hin, w, cos, sin, mask, relp=True)
        for v in outs:
            assert torch.equal(outs[v], outs["relp"]), (l, v)
    m.set_rule("relp")


@pytest.mark.parametrize("family", FAMS)
def test_patch_p8_equals_phase7_patches(family, tmp_path):
    """T5: natural = patch_pass; R / A / RA = diagnose_pass's frozen routing / attention / both; direct effect."""
    d = tiny(family, tmp_path)
    m = StreamRelP(str(d), DEV, torch.float32)
    n, T = 2, 5
    A, Bb = pair_batch(n, T, 61)
    tg, fo = [5, 6], [9, 10]
    ta, tb = routes_pass(m, A, Bb)
    units = enumerate_units(ta.numpy(), tb.numpy(), positions=[2, 3, 4])
    ref = patch_pass(m, A, Bb, tg, fo, units)
    dg = diagnose_pass(m, A, Bb, tg, fo, units, sub_rows=5)
    out = patch_p8(m, A, Bb, tg, fo, u9(units), variants=("natural", "R", "A", "RA", "Rset", "N", "RAN", "RANG"), sub_rows=7)
    np.testing.assert_allclose(out["margin"][:, 0], ref.margin, rtol=1e-5, atol=1e-5)
    assert (out["route_changes"] == ref.route_changes).all()
    np.testing.assert_allclose(out["margin_A"], ref.margin_A, rtol=1e-6, atol=1e-6)
    for i, v in ((1, "frozen_routing"), (2, "frozen_attention"), (3, "frozen_both")):
        np.testing.assert_allclose(out["margin"][:, i], dg[v], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out["direct"], dg["direct"], rtol=1e-5, atol=1e-5, equal_nan=True)
    # set-only freeze = natural where the patch switches no later top-k set
    same = ref.route_changes == 0
    assert same.any() and (~same).any()
    np.testing.assert_allclose(out["margin"][same, 4], out["margin"][same, 0], rtol=1e-5, atol=1e-5)
    # the freeze variants do something
    assert np.abs(out["margin"][:, 7] - out["margin"][:, 0]).max() > 1e-3
    # null patch (scale 0): every variant returns the unpatched margin (forced values = the run's own values)
    z = patch_p8(m, A, Bb, tg, fo, u9(units), variants=tuple(FREEZE), sub_rows=7, scale=0.0)
    base = np.where(units[:, 1] == 0, z["margin_B"][units[:, 0]], z["margin_A"][units[:, 0]])
    for i in range(len(FREEZE)):
        np.testing.assert_allclose(z["margin"][:, i], base, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("family", FAMS)
def test_neuron_unit_equals_engine_override(family, tmp_path):
    """T6: neuron unit (rank-1 update of the MoE output) = the engine's source override with a one-neuron mask."""
    d = tiny(family, tmp_path)
    m = StreamRelP(str(d), DEV, torch.float32)
    n, T = 2, 5
    A, Bb = pair_batch(n, T, 71)
    tg, fo = [5, 6], [9, 10]
    ta, tb = routes_pass(m, A, Bb)
    u8 = enumerate_units(ta.numpy(), tb.numpy(), positions=[2, 4], layer_units=False)
    units = np.concatenate([u9(u8, 0), u9(u8, 7)])
    units[:, 2] = 2
    res = patch_p8(m, A, Bb, tg, fo, units, sub_rows=9)
    base = Group("base", torch.cat([A, Bb]), tg * 2, fo * 2, pairs=True, keep=set())
    groups = [base]
    for j, (i, di, _, l, t, e, _, _, nj) in enumerate(units.tolist()):
        rsrc = i if di == 0 else n + i
        ids = (Bb if di == 0 else A)[i:i + 1]
        S = torch.zeros(T, m.E, m.I, dtype=torch.bool, device=DEV)
        S[t, e, nj] = True
        groups.append(Group(f"u{j}", ids, [tg[i]], [fo[i]], override_fn=lambda l_, w, st, l=l, S=S, rsrc=rsrc: Override(
            rows=torch.tensor([True], device=DEV), S=S, value="source", src=st["layer_out"]["base"],
            src_rows=torch.tensor([rsrc], device=DEV)) if l_ == l else None))
    forward_pass(m, groups)
    bf = np.array([float(g.delta[0]) for g in groups[1:]])
    np.testing.assert_allclose(res["margin"][:, 0], bf, rtol=1e-4, atol=1e-4)
    inp = np.where(units[:, 1] == 0, res["margin_B"][units[:, 0]], res["margin_A"][units[:, 0]])
    assert np.abs(res["margin"][:, 0] - inp).max() > 1e-4


@pytest.mark.parametrize("family", FAMS)
def test_rules_are_derivatives_of_the_frozen_models(family, tmp_path):
    """T2 / T7: finite differences of a scaled whole-output patch (denoise, B <- A) against the endpoint-B scores summed
    over experts: 'grad' = model with later routing frozen (R); 'relp_nohalf' = model with everything RelP holds fixed
    frozen (RANG); 'grad_rw' = natural model (where the patch switches no top-k set)."""
    d = tiny(family, tmp_path)
    m = RuleRelP(str(d), DEV, torch.float32)
    n, T = 3, 5
    A, Bb = pair_batch(n, T, 81)
    tg, fo = [5, 6, 7], [9, 10, 11]
    pool = torch.ones(n, T, dtype=torch.bool)
    g = scored(m, A, Bb, tg, fo)
    res = multi_backward(m, g, endpoint_triples(n, "B"), pool, ["relp", "grad", "grad_rw", "relp_nohalf"])
    units = np.array([(i, 0, 0, l, t, -1, 0, 0, -1) for i in range(n) for l in range(m.L) for t in (2, 3, 4)], dtype=np.int64)
    eps = 0.02
    var = ("R", "natural", "RANG")
    up = patch_p8(m, A, Bb, tg, fo, units, variants=var, scale=eps)
    dn = patch_p8(m, A, Bb, tg, fo, units, variants=var, scale=-eps)
    fd = (up["margin"] - dn["margin"]) / (2 * eps)
    i, l, t = units[:, 0], units[:, 3], units[:, 4]
    for vi, rule, frac in ((0, "grad", 1.0), (1, "grad_rw", 0.85), (2, "relp_nohalf", 1.0)):
        sc = res[rule]["pair_expert"][i, l, t].sum(-1).numpy()
        ok = np.abs(fd[:, vi] - sc) <= 3e-3 + 3e-2 * np.abs(sc)
        if rule == "grad_rw":
            ok |= (up["route_changes"] + dn["route_changes"]) > 0
        assert ok.mean() >= frac, (rule, ok.mean(), np.abs(fd[:, vi] - sc).max())
    # RelP itself is NOT the derivative of the frozen model (half rule) unless nothing lies downstream
    sc = res["relp"]["pair_expert"][i, l, t].sum(-1).numpy()
    last = l == m.L - 1
    np.testing.assert_allclose(fd[last, 2], sc[last], rtol=3e-2, atol=3e-3)
    assert np.abs(fd[~last, 2] - sc[~last]).max() > 1e-2


@pytest.mark.parametrize("family", FAMS)
def test_path_and_probe(family, tmp_path):
    """T10 / T11: one path segment = mean of the endpoint scores; probe distances vanish for identical runs."""
    d = tiny(family, tmp_path)
    m = RuleRelP(str(d), DEV, torch.float32)
    n, T = 3, 5
    A, Bb = pair_batch(n, T, 91)
    tg, fo = [5, 6, 7], [9, 10, 11]
    pool = torch.ones(n, T, dtype=torch.bool)
    g = scored(m, A, Bb, tg, fo)
    ea = multi_backward(m, g, endpoint_triples(n, "A"), pool, ["relp"])["relp"]["pair_expert"]
    eb = multi_backward(m, g, endpoint_triples(n, "B"), pool, ["relp"])["relp"]["pair_expert"]
    gp = path_forward(m, A, Bb, tg, fo, 1)
    torch.testing.assert_close(gp.delta, torch.cat([g.delta[n:], g.delta[:n]]), rtol=1e-5, atol=1e-5)
    pe = multi_backward(m, gp, path_triples(n, 1), pool, ["relp"])["relp"]["pair_expert"]
    torch.testing.assert_close(pe, 0.5 * (ea + eb), rtol=1e-4, atol=1e-5)
    g4 = path_forward(m, A, Bb, tg, fo, 4)
    p4 = multi_backward(m, g4, path_triples(n, 4), pool, ["relp", "relp_rw"])
    assert torch.isfinite(p4["relp"]["pair_expert"]).all() and not torch.equal(p4["relp"]["pair_expert"], pe)
    pr = gate_probe(m, A, Bb, 2)
    ta, tb = routes_pass(m, A, Bb)                                   # [L, n, T, k]
    assert (np.sort(pr["topi"][0], -1) == np.sort(ta.permute(1, 0, 2, 3).numpy(), -1)).all()
    assert (np.sort(pr["topi"][1], -1) == np.sort(tb.permute(1, 0, 2, 3).numpy(), -1)).all()
    assert (pr["jaccard"][:, :, :2] == 1).all() and (pr["tv_w"][:, :, :2] == 0).all() and (pr["resid_rel"][:, :, :2] == 0).all()
    assert pr["jaccard"].min() < 1 and pr["attn_tv"][:, :, 2:].max() > 0 and (pr["tau"] > 0).all()
    same = gate_probe(m, A, A, 2)
    for key in ("tv_w", "dlogit_rms", "dsig_mean", "lognorm1", "lognorm2", "attn_tv", "resid_rel"):
        assert np.nanmax(np.abs(same[key])) == 0, key
    assert (same["jaccard"] == 1).all()


@pytest.mark.parametrize("family", FAMS)
def test_lockstep_equals_separate_runs(family, tmp_path):
    """Stage 3: buckets of different length served by shared passes = each bucket run alone (bit for bit), including
    clients that need different numbers of passes."""
    from moetrace.relp_moe_p8 import lockstep
    d = tiny(family, tmp_path)
    m = RuleRelP(str(d), DEV, torch.bfloat16)
    buckets = []
    for bi, (n, T) in enumerate(((3, 5), (2, 7), (4, 4))):
        A, Bb = pair_batch(n, T, 100 + bi)
        buckets.append((A, Bb, [5 + i for i in range(n)], [20 + i for i in range(n)]))

    def job(mp, b, extra_pass):
        A, Bb, tg, fo = b
        n, T = A.shape
        pool = torch.ones(n, T, dtype=torch.bool)
        g = scored(mp, A, Bb, tg, fo)
        res = multi_backward(mp, g, endpoint_triples(n, "B"), pool, ["relp", "grad_rw", "relp_rw"])
        ta, tb = routes_pass(mp, A, Bb)
        units = u9(enumerate_units(ta.numpy(), tb.numpy(), positions=[2, T - 1]))
        out = [g.delta.cpu(), res["relp"]["pair_expert"], res["grad_rw"]["pair_expert"], res["relp_rw"]["pair_expert"],
               torch.from_numpy(patch_p8(mp, A, Bb, tg, fo, units, variants=("natural", "R"), sub_rows=7)["margin"])]
        if extra_pass:
            out.append(scored(mp, A, Bb, tg, fo).delta.cpu())
        return out

    alone = [job(m, b, i == 1) for i, b in enumerate(buckets)]
    together = lockstep(m, [lambda mp, b=b, i=i: job(mp, b, i == 1) for i, b in enumerate(buckets)])
    for a, t in zip(alone, together):
        assert len(a) == len(t)
        for x, y in zip(a, t):
            assert torch.equal(x, y)
    # an error in one client is re-raised
    with pytest.raises(ZeroDivisionError):
        lockstep(m, [lambda mp: scored(mp, *buckets[0]).delta, lambda mp: 1 / 0])
    # the model object itself is still usable afterwards
    assert torch.equal(scored(m, *buckets[0]).delta.cpu(), alone[0][0])


@pytest.mark.parametrize("family", FAMS)
def test_evaluate_roles_equals_protocol_evaluate(family, tmp_path):
    """Stage 3: role-unit noise / denoise with per-row positions = the Phase-7 evaluate() when the roles sit at fixed positions."""
    from collections import namedtuple
    from moetrace.relp_moe_p8 import evaluate_roles
    sys.path.insert(0, str(ROOT / "scripts"))
    import relp_sva_moe_run as R
    d = tiny(family, tmp_path)
    m = StreamRelP(str(d), DEV, torch.float32)
    n, T = 3, 6
    A, Bb = pair_batch(n, T, 111)
    tg, fo = [5, 6, 7], [9, 10, 11]
    rows = [{"row_id": f"r{i}", "group_id": f"g{i}", "input_ids": A[i].tolist(), "reference_ids": Bb[i].tolist(),
             "target_id": tg[i], "foil_id": fo[i], "special_tokens_mask": [1] + [0] * (T - 1)} for i in range(n)]
    Label = namedtuple("Label", "ranking seed budget")
    g = torch.Generator().manual_seed(5)
    role_masks = {Label("a", 0, k_): torch.rand(m.L, 2, m.E, generator=g) < p_ for k_, p_ in ((1, 0.15), (2, 0.5), (3, 1.0))}
    role_masks[Label("empty_layers", 0, 4)] = torch.zeros(m.L, 2, m.E, dtype=torch.bool)
    role_masks[Label("empty_layers", 0, 4)][1, 0, 2] = True
    all_masks = {Label("w", 0, k_): torch.rand(m.L, m.E, generator=g) < p_ for k_, p_ in ((1, 0.2), (2, 0.6))}
    pos = [2, T - 1]
    ep = {}
    for lab, mk in role_masks.items():
        x = torch.zeros(m.L, T, m.E, dtype=torch.bool)
        x[:, pos[0]], x[:, pos[1]] = mk[:, 0], mk[:, 1]
        ep[lab] = x
    ref = R.evaluate(m, rows, [{"masks": ep, "unit": "expert_position", "baseline": "zero", "qmean": None},
                               {"masks": all_masks, "unit": "expert_all_positions", "baseline": "zero", "qmean": None}],
                     "non_special", 1040)
    pool = R.pool_mask(rows, "non_special")[0]
    rolepos = torch.tensor([pos] * n)
    for masks, unit, recs in ((role_masks, "role", ref[0]), (all_masks, "all", ref[1])):
        out = evaluate_roles(m, A, Bb, tg, fo, pool, rolepos, masks, unit, chunk_rows=7)
        for r in recs:
            if r["intervention"] not in ("noise", "denoise"):
                continue
            lab = Label(r["ranking"], r["seed"], r["budget"])
            i = int(r["row_id"][1:])
            assert abs(float(out["margin"][lab][r["intervention"]][i]) - r["margin"]) < 1e-4, (unit, lab, r["intervention"], i)
            assert abs(float(out["margin_A"][i]) - r["margin_A"]) < 1e-5
    # the interventions do something, and the full role mask restores more than the sparse one on average
    full, sparse = Label("a", 0, 3), Label("a", 0, 1)
    assert (out_r := evaluate_roles(m, A, Bb, tg, fo, pool, rolepos, role_masks, "role"))["margin"][full]["denoise"].ne(out_r["margin_B"]).any()
    assert not torch.equal(out_r["margin"][full]["denoise"], out_r["margin"][sparse]["denoise"])


def _mix_reference(m, A, Bb, tg, fo, layer, pos, Cset, mode, kind):
    """Slow reference for one single-layer, single-position patch: the MoE output at (layer, pos) of every row is built by
    hand from the unpatched runs (expert by expert, fp32) and injected with the engine's whole-output patch (out_patch_fn)."""
    import torch.nn.functional as F
    n, T = A.shape
    live_ids, src_ids = (Bb, A) if kind == "denoise" else (A, Bb)
    cap = {}
    g0 = Group("ref", torch.cat([live_ids, src_ids]), list(tg) * 2, list(fo) * 2, record_h_in=True)

    def on_layer(l, w, st):
        if l != layer:
            return
        h = g0.h_in[l]
        cos, sin = m.rope(T, h)
        hm = h + m.attention(m.rms(h, w.ln1), w, cos, sin, m.causal_mask(2 * n, T, h.dtype), False)
        x = m.rms(hm, w.ln2)[:, pos].float()
        probs = torch.softmax(F.linear(x, w.router.float()), -1)
        topv, topi = probs.topk(m.k, -1)
        if m.norm_topk:
            topv = topv / topv.sum(-1, keepdim=True)

        def u(e, xx):
            gg, uu = F.linear(xx, w.gate_up[e].float()).chunk(2, -1)
            return F.linear(F.silu(gg) * uu, w.down[e].float())
        outs = []
        for i in range(n):
            xl, xs = x[i], x[n + i]
            Rl = {int(e): float(v) for e, v in zip(topi[i], topv[i])}
            Rs = {int(e): float(v) for e, v in zip(topi[n + i], topv[n + i])}
            o = torch.zeros_like(xl)
            if mode == "full":
                for e, gl in Rl.items():
                    if e not in Cset:
                        o += gl * u(e, xl)
                for e, gs in Rs.items():
                    if e in Cset:
                        o += gs * u(e, xs)
            elif mode in ("content_both", "content_all"):
                for e, gl in Rl.items():
                    swap = e in Cset and (mode == "content_all" or e in Rs)
                    o += gl * u(e, xs if swap else xl)
            elif mode == "renorm":
                kept = sum(gl for e, gl in Rl.items() if e not in Cset) + sum(gs for e, gs in Rs.items() if e in Cset)
                s = sum(Rl.values()) / kept if kept > 0 else 0.0
                for e, gl in Rl.items():
                    if e not in Cset:
                        o += s * gl * u(e, xl)
                for e, gs in Rs.items():
                    if e in Cset:
                        o += s * gs * u(e, xs)
            else:  # route
                for e, gs in Rs.items():
                    o += gs * u(e, xs if e in Cset else xl)
            outs.append(o)
        cap["o"] = torch.stack(outs)
    forward_pass(m, [g0], on_layer=on_layer)
    mk = torch.zeros(n, T, dtype=torch.bool, device=DEV)
    mk[:, pos] = True
    vals = torch.zeros(n, T, cap["o"].shape[-1], device=DEV)
    vals[:, pos] = cap["o"]
    gp = Group("pat", live_ids, list(tg), list(fo), out_patch_fn=lambda l, st: (mk, vals) if l == layer else None)
    forward_pass(m, [gp])
    return gp.delta.float().cpu()


@pytest.mark.parametrize("family", FAMS)
def test_evaluate_mix(family, tmp_path):
    """Patch semantics at the patched MoE sites: 'full' = the engine (evaluate_roles); every mode = a hand-built reference on
    single-layer, single-position patches; with every expert of every changed position selected, route = full (and
    renorm = full where the routing weights sum to 1)."""
    from collections import namedtuple
    from moetrace.relp_moe_p8 import MIX_MODES, evaluate_mix, evaluate_roles
    d = tiny(family, tmp_path)
    m = StreamRelP(str(d), DEV, torch.float32)
    n, T = 4, 6
    A, Bb = pair_batch(n, T, 222)
    tg, fo = [5, 6, 7, 8], [9, 10, 11, 12]
    pool = torch.ones(T, dtype=torch.bool)
    rolepos = torch.arange(T)[None].expand(n, T).contiguous()
    Label = namedtuple("Label", "name")
    gen = torch.Generator().manual_seed(3)
    masks = {Label("sparse"): torch.rand(m.L, T, m.E, generator=gen) < 0.2,
             Label("dense"): torch.rand(m.L, T, m.E, generator=gen) < 0.6}
    allm = torch.zeros(m.L, T, m.E, dtype=torch.bool)
    allm[:, 2:] = True
    masks[Label("all")] = allm
    sites = {Label("site1"): (1, 3, {0, 1, 2, 3}), Label("site2"): (0, 2, {2, 5}), Label("site3"): (2, 5, {1, 4})}
    for lab, (l, t, cs) in sites.items():
        x = torch.zeros(m.L, T, m.E, dtype=torch.bool)
        x[l, t, sorted(cs)] = True
        masks[lab] = x
    out = evaluate_mix(m, A, Bb, tg, fo, pool, rolepos, masks, "role", chunk_rows=9)
    ref = evaluate_roles(m, A, Bb, tg, fo, pool, rolepos, masks, "role", chunk_rows=7)
    assert torch.allclose(out["margin_A"], ref["margin_A"], atol=1e-5) and torch.allclose(out["margin_B"], ref["margin_B"], atol=1e-5)
    for lab in masks:
        for kind in ("noise", "denoise"):
            assert torch.allclose(out["margin"][lab]["full"][kind], ref["margin"][lab][kind], atol=1e-4), (lab, kind)
    for lab, (l, t, cs) in sites.items():
        for mode in MIX_MODES:
            for kind in ("noise", "denoise"):
                r = _mix_reference(m, A, Bb, tg, fo, l, t, cs, mode, kind)
                assert torch.allclose(out["margin"][lab][mode][kind], r, atol=1e-4), (lab, mode, kind, out["margin"][lab][mode][kind], r)
    allx = Label("all")
    for kind in ("noise", "denoise"):
        assert torch.allclose(out["margin"][allx]["route"][kind], out["margin"][allx]["full"][kind], atol=1e-5)
        if family != "olmoe":   # OLMoE keeps the softmax weights of the top-k (sum < 1); renorm rescales to the live total
            assert torch.allclose(out["margin"][allx]["renorm"][kind], out["margin"][allx]["full"][kind], atol=1e-4)
    # the modes differ where routing differs between the runs, and content_* never moves an expert routed in neither run
    d_ = {mode: out["margin"][Label("dense")][mode]["denoise"] for mode in MIX_MODES}
    assert not torch.allclose(d_["full"], d_["content_both"], atol=1e-4)
    assert not torch.allclose(d_["full"], d_["route"], atol=1e-4)
