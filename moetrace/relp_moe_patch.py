"""Phase 7 (S3, S4): Lu et al.-style activation patching on the SVA pairs, and RelP at the B endpoint, with the streamed
engine of moetrace/relp_moe_stream.py (imported, never modified: its sha256 is part of the verification identity).

Patching = the STR version of the paper's protocol (Direction 6): the SVA pair is the corruption (A = clean prefix,
B = patch prefix), metric F = logit(clean answer) - logit(patch answer) (the RelP F). Units are COMPLETE replacements at
one layer l (the q-level semantics of the protocol's evaluate, exact):
  position unit (l, t):    the MoE output at position t := the source run's (all experts, all neurons at (l, t))
  expert unit   (l, t, e): expert e's contribution w_e E_e(x) at (l, t) := the source run's (0 if e is not routed there),
                            i.e. all q of (l, t, e) := the source's; other experts keep their live contributions
Directions: denoise (Lu's rescue): input B, source A, effect = F(B <- A) - F_B; noise: input A, source B,
effect (drop) = F_A - F(A <- B). Patched rows join the pass at their layer from the base run's exact state (residual after
attention + recomposed MoE output; layers < l and the attention of layer l are unchanged), then run naturally: later
attention, routing and experts are recomputed; the number of later (layer, position) top-k sets that differ from the
unpatched input run is counted (protocol's route-change definition).

RelP endpoint B (guide 2.3: "(q_clean - q_corrupt) * beta_corrupt, labelled"): the same score as the protocol but with the
coefficients taken at the B run: (q_A - q_B) . beta_B. It is the first-order prediction of the denoise effect (Lu's
rescue); the protocol's endpoint-A score predicts the noise drop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .relp_moe_stream import Group, StreamRelP, forward_pass


# ----------------------------------------------------------------------------------------------------------------------
# RelP with the coefficients at either endpoint (endpoint A = relp_moe_stream.relp_backward, re-implemented here so that
# the verified engine file stays untouched; the test checks that endpoint A reproduces relp_backward exactly)
# ----------------------------------------------------------------------------------------------------------------------
def relp_backward_endpoint(m: StreamRelP, g: Group, n: int, pool: torch.Tensor, endpoint: str = "B",
                           per_row: bool = False):
    """g: forward-passed group with rows [A_1..A_n, B_1..B_n] and record_h_in. Score (q_A - q_B) . beta_X, X = endpoint,
    summed over rows (float64) [L, T, E, I], and support ((q_A != 0) | (q_B != 0)) & pool. per_row=True also returns
    each row's expert-position score (sum over the expert's neurons, fp32) [n, L, T, E] on the CPU."""
    B, T = g.ids.shape
    assert B == 2 * n and endpoint in ("A", "B")
    E, I, H, k = m.E, m.I, m.H, m.k
    total = torch.zeros(m.L, T, E, I, dtype=torch.float64, device=m.dev)
    support = torch.zeros(m.L, T, E, I, dtype=torch.bool, device=m.dev)
    pool = pool.to(m.dev)
    sl = slice(0, n) if endpoint == "A" else slice(n, 2 * n)
    with torch.enable_grad():
        hL = g.h.detach().requires_grad_(True)
        d = m.readout(hL, g.target, g.foil, relp=True)
        lam, = torch.autograd.grad(d[sl].sum(), hL)
    cos, sin = m.rope(T, g.h)
    mask = m.causal_mask(B, T, g.h.dtype)
    pos_of = torch.arange(n * T, device=m.dev) % T
    keep_pos = pool.reshape(-1)
    rows_e = torch.zeros(n * T, m.L, E, dtype=torch.float32, device=m.dev) if per_row else None
    for l, w in m.stream(reverse=True):
        hin = g.h_in[l].detach().requires_grad_(True)
        with torch.enable_grad():
            hout, rec = m.layer(hin, w, cos, sin, mask, relp=True, pairs=True)
            lam_in, = torch.autograd.grad(hout, hin, grad_outputs=lam)
        with torch.no_grad():
            lam_o = lam[sl].reshape(n * T, H)
            topi = rec.topi.view(B * T, k)
            qp = rec.q.view(B * T, k, I)
            ia, ib = topi[: n * T], topi[n * T:]
            qa, qb = qp[: n * T], qp[n * T:]
            for e in torch.unique(torch.cat([ia, ib], 1)).tolist():
                in_a, in_b = ia == e, ib == e
                tix = torch.where((in_a.any(1) | in_b.any(1)) & keep_pos)[0]
                if tix.numel() == 0:
                    continue
                beta = F.linear(lam_o[tix], w.down[e].t())
                q_a = torch.zeros(tix.numel(), I, dtype=qa.dtype, device=m.dev)
                q_b = torch.zeros_like(q_a)
                ra, sa = torch.where(in_a[tix])
                q_a[ra] = qa[tix[ra], sa]
                rb, sb = torch.where(in_b[tix])
                q_b[rb] = qb[tix[rb], sb]
                sc = (q_a.float() - q_b.float()) * beta.float()
                if not torch.isfinite(sc).all():
                    raise ValueError("Nonfinite attribution")
                tp = pos_of[tix]
                total[l, :, e].index_add_(0, tp, sc.double())
                sup = torch.zeros(T, I, dtype=torch.int32, device=m.dev)
                sup.index_add_(0, tp, ((q_a != 0) | (q_b != 0)).int())
                support[l, :, e] |= sup > 0
                if per_row:
                    rows_e[tix, l, e] = sc.sum(-1)
        lam = lam_in.detach()
        del w, rec, hout, hin
    if per_row:
        return total, support, rows_e.view(n, T, m.L, E).permute(0, 2, 1, 3).contiguous().cpu()
    return total, support


# ----------------------------------------------------------------------------------------------------------------------
# activation patching
# ----------------------------------------------------------------------------------------------------------------------
def routes_pass(m: StreamRelP, A: torch.Tensor, Bb: torch.Tensor):
    """Top-k sets of A and B rows at every layer: (topi_A [L, n, T, k], topi_B [L, n, T, k]) on the CPU."""
    n = A.shape[0]
    g = Group("r", torch.cat([A, Bb]), [0] * (2 * n), [1] * (2 * n), record_route=True)
    forward_pass(m, [g])
    ti = torch.stack([r[0] for r in g.routes])          # [L, 2n, T, k]
    return ti[:, :n], ti[:, n:]


def enumerate_units(topi_A, topi_B, positions, directions=("denoise", "noise"), experts=True, layer_units=True):
    """Units per base row: list of tuples (row, direction, kind, layer, pos, expert, active_A, active_B), kind 0 = position,
    1 = expert; experts = union of the experts routed at A or B at (layer, pos). Returns an int64 array [U, 8]."""
    L, n, T, k = topi_A.shape
    out = []
    for i in range(n):
        for di, _ in enumerate(directions):
            for l in range(L):
                for t in positions:
                    if layer_units:
                        out.append((i, di, 0, l, t, -1, 0, 0))
                    if experts:
                        sa = set(topi_A[l, i, t].tolist())
                        sb = set(topi_B[l, i, t].tolist())
                        for e in sorted(sa | sb):
                            out.append((i, di, 1, l, t, e, int(e in sa), int(e in sb)))
    return np.asarray(out, dtype=np.int64).reshape(-1, 8)


def _recompose(topi_live, c_live, topi_src, c_src, e, E):
    """MoE output at one token after replacing expert e's contribution by the source's (0 if not routed there).
    topi_* [P, k], c_* [P, k, H], e [P]. Accumulation as the HF loop: ascending expert id, bf16 index_add_ from zeros."""
    P, k, H = c_live.shape
    in_live = topi_live == e[:, None]
    in_src = topi_src == e[:, None]
    slot = in_src.float().argmax(1)
    c_e = c_src[torch.arange(P, device=c_src.device), slot]
    c_e = torch.where(in_src.any(1)[:, None], c_e, torch.zeros_like(c_e))
    contrib_live = torch.where(in_live[..., None], c_e[:, None, :], c_live)
    extra_e = torch.where(in_live.any(1), torch.full_like(e, E + 1), e)
    extra_c = torch.where(in_live.any(1)[:, None], torch.zeros_like(c_e), c_e)
    ex = torch.cat([topi_live, extra_e[:, None]], 1)
    cc = torch.cat([contrib_live, extra_c[:, None]], 1)
    order = torch.argsort(ex, dim=1, stable=True)
    cc = torch.gather(cc, 1, order[..., None].expand(-1, -1, H))
    acc = torch.zeros(P, H, dtype=c_live.dtype, device=c_live.device)
    ar = torch.arange(P, device=c_live.device)
    for s in range(k + 1):
        acc.index_add_(0, ar, cc[:, s])
    return acc


@dataclass
class PatchResult:
    units: np.ndarray        # [U, 8] as enumerate_units, rows relative to the pass's base rows
    margin: np.ndarray       # [U] F of the patched run
    route_changes: np.ndarray  # [U] later (layer, position) top-k sets that differ from the unpatched input run
    margin_A: np.ndarray     # [n] base margins
    margin_B: np.ndarray


def patch_pass(m: StreamRelP, A: torch.Tensor, Bb: torch.Tensor, target: list, foil: list, units: np.ndarray,
               sub_rows: int = 6000) -> PatchResult:
    """One streamed pass: base rows A_i, B_i (i < n) plus every unit's patched row (joining at its layer).
    units rows index into A / B; directions: 0 = denoise (input B, source A), 1 = noise (input A, source B).
    Patched rows live in one preallocated buffer in spawn (layer) order, updated in place."""
    n, T = A.shape
    dev, H, k, E = m.dev, m.H, m.k, m.E
    ids_b = torch.cat([A, Bb]).to(dev)
    tg = torch.tensor(list(target) * 2, device=dev)
    fo = torch.tensor(list(foil) * 2, device=dev)
    U = units.shape[0]
    u = torch.from_numpy(units).to(dev)
    order = torch.argsort(u[:, 3], stable=True)               # buffer slot p holds unit order[p]
    uo = u[order]
    u_dir, u_kind, u_layer, u_pos, u_exp = (uo[:, j].contiguous() for j in (1, 2, 3, 4, 5))
    u_in = torch.where(u_dir == 0, uo[:, 0] + n, uo[:, 0])    # input row in the base group
    u_src = torch.where(u_dir == 0, uo[:, 0], uo[:, 0] + n)   # source row
    first = torch.searchsorted(u_layer, torch.arange(m.L + 1, device=dev))  # slots [first[l], first[l+1]) spawn at l
    buf = torch.empty(U, T, H, dtype=m.dtype, device=dev)
    changes = torch.zeros(U, dtype=torch.int32, device=dev)
    with torch.no_grad():
        hb = F.embedding(ids_b, m.embed)
        cos, sin = m.rope(T, hb)
        mask_b = m.causal_mask(2 * n, T, hb.dtype)
        for l, w in m.stream():
            # base rows: the layer as StreamRelP.layer computes it, keeping the post-attention residual and pair contributions
            a = m.attention(m.rms(hb, w.ln1), w, cos, sin, mask_b, False)
            hmid = hb + a
            xf = m.rms(hmid, w.ln2).view(-1, H)
            topi, topw = m.route(xf, w)
            o, _, cp = m.experts(xf, topi, topw, w, False, True)
            o = o.view(2 * n, T, H)
            topi_s = topi.view(2 * n, T, k).sort(-1)[0]
            # alive patched rows (spawned at earlier layers): natural forward, in place; count routing differences
            n_alive = int(first[l])
            for s0 in range(0, n_alive, sub_rows):
                s1 = min(n_alive, s0 + sub_rows)
                hs = buf[s0:s1]
                hn, rec = m.layer(hs, w, cos, sin, m.causal_mask(s1 - s0, T, hs.dtype))
                diff = (rec.topi.view(-1, T, k).sort(-1)[0] != topi_s[u_in[s0:s1]]).any(-1).sum(-1)
                changes[s0:s1] += diff.int()
                buf[s0:s1] = hn
                del rec, hn
            # spawn the units of this layer from the base state (layers < l and attention of l unchanged)
            s0, s1 = int(first[l]), int(first[l + 1])
            if s1 > s0:
                sl = slice(s0, s1)
                rin, rsrc, tt = u_in[sl], u_src[sl], u_pos[sl]
                o_new = o[rin].clone()
                pos_u = u_kind[sl] == 0
                if pos_u.any():
                    j = torch.where(pos_u)[0]
                    o_new[j, tt[j]] = o[rsrc[j], tt[j]]
                if (~pos_u).any():
                    j = torch.where(~pos_u)[0]
                    tl = rin[j] * T + tt[j]
                    ts = rsrc[j] * T + tt[j]
                    c3 = cp.view(2 * n * T, k, H)
                    o_new[j, tt[j]] = _recompose(topi[tl], c3[tl], topi[ts], c3[ts], u_exp[sl][j], E)
                buf[sl] = hmid[rin] + o_new
                del o_new
            hb = hmid + o
            del w, cp, o, hmid, a
        dbase = m.readout(hb, tg, fo)
        margin = torch.empty(U, dtype=torch.float32, device=dev)
        for s0 in range(0, U, sub_rows):
            s1 = min(U, s0 + sub_rows)
            margin[s0:s1] = m.readout(buf[s0:s1], tg[u_in[s0:s1]], fo[u_in[s0:s1]])
    inv = torch.empty_like(order)
    inv[order] = torch.arange(U, device=dev)                  # back to the caller's unit order
    return PatchResult(units=units, margin=margin[inv].cpu().numpy(), route_changes=changes[inv].cpu().numpy(),
                       margin_A=dbase[:n].cpu().numpy(), margin_B=dbase[n:].cpu().numpy())


# ----------------------------------------------------------------------------------------------------------------------
# S4f diagnostics: why does a unit's exact patch effect differ from its RelP score?
# ----------------------------------------------------------------------------------------------------------------------
VARIANTS = ("natural", "frozen_routing", "frozen_attention", "frozen_both")


def _attention_p(m: StreamRelP, x, w, cos, sin, mask, p_forced=None):
    """StreamRelP.attention with the pattern returned, optionally replaced by a forced pattern (same operations)."""
    from .relp_moe_stream import repeat_kv, rotate_half
    B, T, _ = x.shape
    hs = (B, T, -1, m.D)
    q, k, v = F.linear(x, w.wq), F.linear(x, w.wk), F.linear(x, w.wv)
    if m.family == "qwen3_moe":
        q = m.rms(q.view(hs), w.q_norm).transpose(1, 2)
        k = m.rms(k.view(hs), w.k_norm).transpose(1, 2)
    elif m.family == "olmoe":
        q = m.rms(q, w.q_norm).view(hs).transpose(1, 2)
        k = m.rms(k, w.k_norm).view(hs).transpose(1, 2)
    else:
        q, k = q.view(hs).transpose(1, 2), k.view(hs).transpose(1, 2)
    v = v.view(hs).transpose(1, 2)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    q = (q * c) + (rotate_half(q) * s)
    k = (k * c) + (rotate_half(k) * s)
    rep = m.nH // m.nKV
    k, v = repeat_kv(k, rep), repeat_kv(v, rep)
    if p_forced is None:
        aw = torch.matmul(q, k.transpose(2, 3)) * m.scaling + mask
        p = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    else:
        p = p_forced
    a = torch.matmul(p, v).transpose(1, 2).contiguous().reshape(B, T, -1).contiguous()
    return F.linear(a, w.wo), p


def diagnose_pass(m: StreamRelP, A: torch.Tensor, Bb: torch.Tensor, target: list, foil: list, units: np.ndarray,
                  sub_rows: int = 4000) -> dict:
    """Every unit (enumerate_units format) in the four VARIANTS: after the patched layer, later layers run naturally, with
    the routing (top-k set and weights) of the unpatched input run, with its attention patterns, or with both.
    Also the direct effect of final-position units: the change of the patched layer's MoE output at the final position
    read out through the final norm with the input run's denominator (DLA; positive = towards the clean answer).
    Returns {variant: margins [U]} and 'direct' [U] (nan for non-final positions), 'margin_A', 'margin_B'."""
    n, T = A.shape
    dev, H, k, E = m.dev, m.H, m.k, m.E
    U = units.shape[0]
    V = len(VARIANTS)
    ids_b = torch.cat([A, Bb]).to(dev)
    tg = torch.tensor(list(target) * 2, device=dev)
    fo = torch.tensor(list(foil) * 2, device=dev)
    u = torch.from_numpy(units).to(dev).repeat_interleave(V, 0)          # row r = unit r // V, variant r % V
    var = torch.arange(V, device=dev).repeat(U)
    order = torch.argsort(u[:, 3], stable=True)
    uo, vo = u[order], var[order]
    u_dir, u_kind, u_layer, u_pos, u_exp = (uo[:, j].contiguous() for j in (1, 2, 3, 4, 5))
    u_in = torch.where(u_dir == 0, uo[:, 0] + n, uo[:, 0])
    u_src = torch.where(u_dir == 0, uo[:, 0], uo[:, 0] + n)
    first = torch.searchsorted(u_layer, torch.arange(m.L + 1, device=dev))
    buf = torch.empty(U * V, T, H, dtype=m.dtype, device=dev)
    dfinal = torch.full((U * V, H), float("nan"), dtype=torch.float32, device=dev)
    with torch.no_grad():
        hb = F.embedding(ids_b, m.embed)
        cos, sin = m.rope(T, hb)
        mask_b = m.causal_mask(2 * n, T, hb.dtype)
        for l, w in m.stream():
            a, p_b = _attention_p(m, m.rms(hb, w.ln1), w, cos, sin, mask_b)
            hmid = hb + a
            xf = m.rms(hmid, w.ln2).view(-1, H)
            topi, topw = m.route(xf, w)
            o, _, cp = m.experts(xf, topi, topw, w, False, True)
            o = o.view(2 * n, T, H)
            ti3, tw3 = topi.view(2 * n, T, k), topw.view(2 * n, T, k)
            n_alive = int(first[l])
            for s0 in range(0, n_alive, sub_rows):
                s1 = min(n_alive, s0 + sub_rows)
                hs, rin, vv = buf[s0:s1], u_in[s0:s1], vo[s0:s1]
                fr = (vv == 1) | (vv == 3)
                fa = (vv == 2) | (vv == 3)
                mask = m.causal_mask(s1 - s0, T, hs.dtype)
                a_n, _ = _attention_p(m, m.rms(hs, w.ln1), w, cos, sin, mask)
                if fa.any():
                    j = torch.where(fa)[0]
                    a_f, _ = _attention_p(m, m.rms(hs[j], w.ln1), w, cos, sin, mask[: j.numel()], p_b[rin[j]])
                    a_n[j] = a_f
                hm = hs + a_n
                xs = m.rms(hm, w.ln2).view(-1, H)
                ti, tw = m.route(xs, w)
                ti, tw = ti.view(-1, T, k), tw.view(-1, T, k)
                if fr.any():
                    j = torch.where(fr)[0]
                    ti[j], tw[j] = ti3[rin[j]], tw3[rin[j]]
                os_, _, _ = m.experts(xs, ti.reshape(-1, k), tw.reshape(-1, k), w, False, False)
                buf[s0:s1] = hm + os_.view(-1, T, H)
            s0, s1 = int(first[l]), int(first[l + 1])
            if s1 > s0:
                sl = slice(s0, s1)
                rin, rsrc, tt = u_in[sl], u_src[sl], u_pos[sl]
                o_new = o[rin].clone()
                pos_u = u_kind[sl] == 0
                if pos_u.any():
                    j = torch.where(pos_u)[0]
                    o_new[j, tt[j]] = o[rsrc[j], tt[j]]
                if (~pos_u).any():
                    j = torch.where(~pos_u)[0]
                    tl, ts = rin[j] * T + tt[j], rsrc[j] * T + tt[j]
                    c3 = cp.view(2 * n * T, k, H)
                    o_new[j, tt[j]] = _recompose(topi[tl], c3[tl], topi[ts], c3[ts], u_exp[sl][j], E)
                fin = tt == T - 1
                if fin.any():
                    j = torch.where(fin)[0]
                    dfinal[s0 + j] = (o_new[j, T - 1].float() - o[rin[j], T - 1].float())
                buf[sl] = hmid[rin] + o_new
            hb = hmid + o
            del w, cp, o, hmid, a
        dbase = m.readout(hb, tg, fo)
        # DLA of the final-position change through the final norm with the input run's (frozen) denominator
        hf = hb[:, -1].float()
        r = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + m.eps)                       # [2n, 1]
        udir = (m.W_U[tg].float() - m.W_U[fo].float()) * m.final_norm.float()           # [2n, H]
        direct = (dfinal * udir[u_in] * r[u_in]).sum(-1)
        margin = torch.empty(U * V, dtype=torch.float32, device=dev)
        for s0 in range(0, U * V, sub_rows):
            s1 = min(U * V, s0 + sub_rows)
            margin[s0:s1] = m.readout(buf[s0:s1], tg[u_in[s0:s1]], fo[u_in[s0:s1]])
    inv = torch.empty_like(order)
    inv[order] = torch.arange(U * V, device=dev)
    mg = margin[inv].view(U, V).cpu().numpy()
    dr = direct[inv].view(U, V)[:, 0].cpu().numpy()
    out = {v: mg[:, i] for i, v in enumerate(VARIANTS)}
    out.update(direct=dr, margin_A=dbase[:n].cpu().numpy(), margin_B=dbase[n:].cpu().numpy())
    return out
