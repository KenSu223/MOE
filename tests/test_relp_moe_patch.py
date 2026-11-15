"""GPU tests of moetrace/relp_moe_patch.py on tiny random HF checkpoints (helpers from test_relp_moe_stream).

1. patch_pass (units joining at their layer from the base state) = brute force: the input run with the engine's verified
   Override (complete (t, e) replacement from the source run) / out_patch (MoE output at t), every unit, both directions.
2. Route-change counts = brute force (later layers' top-k sets vs the unpatched input run).
3. relp_backward_endpoint('A') reproduces relp_moe_stream.relp_backward; endpoint 'B' equals the explicit all-expert
   reference with the coefficients taken at B.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_relp_moe_stream import DEV, FAMS, Ref, hf_model, make_ids, tiny  # noqa: E402
from moetrace.relp_moe_patch import enumerate_units, patch_pass, relp_backward_endpoint, routes_pass  # noqa: E402
from moetrace.relp_moe_stream import Group, Override, StreamRelP, forward_pass, relp_backward  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def pair_batch(n, T, seed):
    A = make_ids(n, T, seed)
    Bb = A.clone()
    Bb[:, 2] = make_ids(n, 1, seed + 1)[:, 0]
    return A, Bb


@pytest.mark.parametrize("family", FAMS)
def test_patch_pass_equals_brute_force(family, tmp_path):
    d = tiny(family, tmp_path)
    m = StreamRelP(str(d), DEV, torch.float32)
    n, T = 2, 5
    A, Bb = pair_batch(n, T, 41)
    tg, fo = [5, 6], [9, 10]
    ta, tb = routes_pass(m, A, Bb)
    units = enumerate_units(ta.numpy(), tb.numpy(), positions=[2, 3, 4])
    res = patch_pass(m, A, Bb, tg, fo, units, sub_rows=7)
    # brute force: every unit as its own group in one forward pass, with the verified engine paths
    base = Group("base", torch.cat([A, Bb]), tg * 2, fo * 2, pairs=True, keep=set(), record_out=True, record_route=True)
    groups = [base]
    for j, (i, di, kind, l, t, e, _, _) in enumerate(units.tolist()):
        rin, rsrc = (n + i, i) if di == 0 else (i, n + i)
        ids = (Bb if di == 0 else A)[i:i + 1]
        if kind == 0:
            bt = torch.zeros(1, T, dtype=torch.bool, device=DEV)
            bt[0, t] = True
            groups.append(Group(f"u{j}", ids, [tg[i]], [fo[i]], record_route=True, out_patch_fn=lambda l_, st, l=l, bt=bt, rsrc=rsrc: (
                bt, st["groups"]["base"].outs[l_][rsrc:rsrc + 1]) if l_ == l else None))
        else:
            S = torch.zeros(T, m.E, m.I, dtype=torch.bool, device=DEV)
            S[t, e] = True
            groups.append(Group(f"u{j}", ids, [tg[i]], [fo[i]], record_route=True, override_fn=lambda l_, w, st, l=l, S=S, rsrc=rsrc: Override(
                rows=torch.tensor([True], device=DEV), S=S, value="source", src=st["layer_out"]["base"],
                src_rows=torch.tensor([rsrc], device=DEV)) if l_ == l else None))
    forward_pass(m, groups)
    bf = np.array([float(g.delta[0]) for g in groups[1:]])
    np.testing.assert_allclose(res.margin, bf, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res.margin_A, base.delta[:n].cpu().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res.margin_B, base.delta[n:].cpu().numpy(), rtol=1e-5, atol=1e-5)
    # route changes after the patched layer vs the unpatched input run
    for j, (i, di, kind, l, t, e, _, _) in enumerate(units.tolist()):
        rin = n + i if di == 0 else i
        g = groups[1 + j]
        cnt = sum(int((g.routes[l2][0][0].sort(-1)[0] != base.routes[l2][0][rin].sort(-1)[0]).any(-1).sum())
                  for l2 in range(l + 1, m.L))
        assert cnt == res.route_changes[j], (j, cnt, res.route_changes[j])
    # sanity: the grid is not trivial (some units move the margin, some expert units inject unrouted experts)
    assert np.abs(res.margin - np.where(units[:, 1] == 0, res.margin_B[units[:, 0]], res.margin_A[units[:, 0]])).max() > 1e-3
    assert ((units[:, 2] == 1) & (units[:, 6] != units[:, 7])).any()


@pytest.mark.parametrize("family", FAMS)
def test_relp_endpoints(family, tmp_path):
    d = tiny(family, tmp_path)
    m = StreamRelP(str(d), DEV, torch.float32)
    n, T = 3, 5
    A, Bb = pair_batch(n, T, 51)
    tg, fo = [5, 6, 7], [9, 10, 11]
    pool = torch.ones(n, T, dtype=torch.bool)
    pool[:, 0] = False
    g = Group("s", torch.cat([A, Bb]), tg * 2, fo * 2, record_h_in=True)
    forward_pass(m, [g])
    ta, sa, _ = relp_backward(m, g, n, pool, "paired")
    tA, sA = relp_backward_endpoint(m, g, n, pool, "A")
    assert torch.equal(sa, sA)
    torch.testing.assert_close(tA, ta, rtol=1e-6, atol=1e-9)
    tB, sB, rowsB = relp_backward_endpoint(m, g, n, pool, "B", per_row=True)
    assert torch.equal(sB, sa)
    # per-row expert-position scores sum to the aggregate's expert-position totals
    torch.testing.assert_close(rowsB.double().sum(0), tB.sum(-1).cpu(), rtol=1e-4, atol=1e-6)
    R = Ref(hf_model(d, torch.float32), family)
    exp = torch.zeros_like(tB)
    for i in range(n):
        with torch.no_grad():
            _, qa = R.forward(A[i:i + 1].to(DEV), tg[i:i + 1], fo[i:i + 1])
        with torch.enable_grad():
            db, qb = R.forward(Bb[i:i + 1].to(DEV), tg[i:i + 1], fo[i:i + 1], relp=True, grad=True)
            beta = torch.autograd.grad(db.sum(), qb)
        for l in range(R.L):
            sc = (qa[l][0].float() - qb[l][0].detach().float()) * beta[l][0].float()
            exp[l] += torch.where(pool[i].to(DEV)[:, None, None], sc, 0).double()
    torch.testing.assert_close(tB, exp, rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("family", FAMS)
def test_diagnose_pass(family, tmp_path):
    from moetrace.relp_moe_patch import diagnose_pass
    d = tiny(family, tmp_path)
    m = StreamRelP(str(d), DEV, torch.float32)
    n, T = 2, 5
    A, Bb = pair_batch(n, T, 61)
    tg, fo = [5, 6], [9, 10]
    ta, tb = routes_pass(m, A, Bb)
    units = enumerate_units(ta.numpy(), tb.numpy(), positions=[2, 4])
    ref = patch_pass(m, A, Bb, tg, fo, units)
    out = diagnose_pass(m, A, Bb, tg, fo, units, sub_rows=5)
    # natural variant = patch_pass; last-layer units have nothing downstream -> all variants equal
    np.testing.assert_allclose(out["natural"], ref.margin, rtol=1e-4, atol=1e-4)
    last = units[:, 3] == m.L - 1
    for v in ("frozen_routing", "frozen_attention", "frozen_both"):
        np.testing.assert_allclose(out[v][last], out["natural"][last], rtol=1e-5, atol=1e-5)
    # (frozen routing also pins the routing WEIGHTS, so it differs from natural even without top-k set switches)
    # direct effect is defined only at the final position
    assert np.isnan(out["direct"][units[:, 4] != T - 1]).all() and np.isfinite(out["direct"][units[:, 4] == T - 1]).all()
