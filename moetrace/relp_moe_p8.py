"""Phase 8, stage 2 (docs/relp_moe_phase8_plan.md): tools around the verified streamed RelP engine
(moetrace/relp_moe_stream.py and moetrace/relp_moe_patch.py are imported, never modified).

* RuleRelP: StreamRelP whose BACKWARD rules can be switched one by one (the forward values stay the natural ones):
  norm (RMSNorm denominators constant), attn (attention pattern constant), gate (SiLU sigmoid gate constant), half
  (half rule on SiLU(g) * u), router: 'frozen' (the guide), 'weights' (gradient through the routing weights, top-k set
  fixed), 'weights3' (the same with the relevance of w * s * u split in three), 'soft' ('weights' plus a straight-through
  soft top-k boundary over the top-(k + r) experts, temperature tau per layer).
  RULES names the variants; 'relp' = the guide's rules (must reproduce StreamRelP exactly).
* multi_backward: one reverse stream for several rule variants; scores (q_hi - q_lo) . beta_coef for arbitrary row
  triples (endpoint A / B of a pair are special cases; a path between the two runs uses several triples per pair).
* path_forward: rows on the straight line between the B and the A embeddings.
* gate_probe: how much the quantities RelP holds constant differ between the A and the B run.
* patch_p8: exact patches as relp_moe_patch.patch_pass / diagnose_pass, plus neuron units and "freeze" variants in which
  later layers keep the unpatched run's routing / attention patterns / norm denominators / SiLU gates.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .relp_moe_patch import _attention_p, _recompose
from .relp_moe_stream import Group, StreamRelP, repeat_kv, rotate_half

_R = dict(norm=True, attn=True, gate=True, half=True, router="frozen", tau=None)
RULES = {
    "relp": dict(_R),
    "grad": dict(_R, norm=False, attn=False, gate=False, half=False),
    "grad_rw": dict(_R, norm=False, attn=False, gate=False, half=False, router="weights"),
    "relp_nohalf": dict(_R, half=False),
    "relp_nonorm": dict(_R, norm=False),
    "relp_noattn": dict(_R, attn=False),
    "relp_nogate": dict(_R, gate=False),
    "relp_rw": dict(_R, router="weights"),
    "relp_rw3": dict(_R, router="weights3"),
    "relp_soft05": dict(_R, router="soft", tau=0.5),
    "relp_soft1": dict(_R, router="soft", tau=1.0),
    "relp_soft2": dict(_R, router="soft", tau=2.0),
}


class RuleRelP(StreamRelP):
    """StreamRelP with switchable backward rules (self.rule; forward values are those of StreamRelP)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rule = dict(RULES["relp"])
        self.soft_r = 2 if self.k >= 4 else 1
        self._tau = None            # temperature of the current layer (router='soft')
        self._ctx = None

    def set_rule(self, name: str, tau_layer: float | None = None):
        self.rule = dict(RULES[name])
        self._tau = None if self.rule["tau"] is None else float(self.rule["tau"]) * float(tau_layer)

    # ---- rules ---------------------------------------------------------------------------------------------------
    def rms(self, x, w, relp=False):
        return super().rms(x, w, relp and self.rule["norm"])

    def act(self, g, u, relp):
        if not relp:
            return F.silu(g) * u
        gate, half = self.rule["gate"], self.rule["half"]
        if gate and half:
            return StreamRelP.act(g, u, True)
        if gate:
            sg = torch.sigmoid(g).detach()
            s = g * sg
            s = s + (F.silu(g) - s).detach()
        else:
            s = F.silu(g)
        if half:
            return 0.5 * (s * u.detach() + s.detach() * u)
        return s * u

    def attention(self, x, w, cos, sin, mask, relp):  # StreamRelP.attention; the pattern is constant only under rule attn
        B, T, _ = x.shape
        hs = (B, T, -1, self.D)
        q = F.linear(x, w.wq)
        k = F.linear(x, w.wk)
        v = F.linear(x, w.wv)
        if self.family == "qwen3_moe":
            q = self.rms(q.view(hs), w.q_norm, relp).transpose(1, 2)
            k = self.rms(k.view(hs), w.k_norm, relp).transpose(1, 2)
        elif self.family == "olmoe":
            q = self.rms(q, w.q_norm, relp).view(hs).transpose(1, 2)
            k = self.rms(k, w.k_norm, relp).view(hs).transpose(1, 2)
        else:
            q = q.view(hs).transpose(1, 2)
            k = k.view(hs).transpose(1, 2)
        v = v.view(hs).transpose(1, 2)
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        q = (q * c) + (rotate_half(q) * s)
        k = (k * c) + (rotate_half(k) * s)
        rep = self.nH // self.nKV
        k = repeat_kv(k, rep)
        v = repeat_kv(v, rep)
        aw = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        aw = aw + mask
        p = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
        if relp and self.rule["attn"]:
            p = p.detach()
        a = torch.matmul(p, v).transpose(1, 2).contiguous()
        a = a.reshape(B, T, -1).contiguous()
        return F.linear(a, w.wo)

    def route(self, xf, w):
        logits = F.linear(xf, w.router)
        if self.family == "mixtral":
            probs = torch.softmax(logits.float(), dim=-1)
        else:
            probs = F.softmax(logits, dtype=torch.float, dim=-1)
        topv, topi = torch.topk(probs, self.k, dim=-1)
        if self.norm_topk:
            topv = topv / topv.sum(dim=-1, keepdim=True)
        if self.cast_route:
            topv = topv.to(logits.dtype)
        if self.rule["router"] == "frozen" or not torch.is_grad_enabled():
            return topi, topv.detach()
        self._ctx = (logits, probs)
        return topi, topv

    def experts(self, xf, topi, topw, w, relp, pairs: bool):
        mode = self.rule["router"] if (relp and torch.is_grad_enabled() and topw.requires_grad) else "frozen"
        if mode in ("frozen", "weights"):
            return super().experts(xf, topi, topw, w, relp, pairs)
        if pairs:
            raise ValueError("router variants do not record pairs; take q from the 'relp' variant")
        N, k, E = xf.shape[0], self.k, self.E
        if mode == "soft":
            logits, probs = self._ctx
            r = self.soft_r
            with torch.no_grad():   # the r best experts outside the top-k set
                extra_i = torch.topk(probs.detach().scatter(1, topi, -1.0), r, dim=-1)[1]
                ci = torch.cat([topi, extra_i], 1)
            cv = probs.gather(1, ci)
            lg = logits.float().gather(1, ci)
            theta = 0.5 * (lg[:, :k].min(1)[0] + lg[:, k:].max(1)[0])   # boundary between the k-th and (k+1)-th logit
            s = torch.sigmoid((lg - theta[:, None]) / self._tau)
            ps = cv.float() * s
            wsoft = ps / ps.sum(-1, keepdim=True) if self.norm_topk else ps
            whard = torch.cat([topw.detach().float(), torch.zeros(N, r, device=xf.device)], 1)
            wst = whard + (wsoft - wsoft.detach())          # value: the hard weights; gradient: the soft surrogate
            if self.cast_route:
                wst = wst.to(logits.dtype)
            cand = ci
        elif mode == "weights3":
            wst = topw.detach() + (topw - topw.detach()) / 3
            cand = topi
        else:
            raise ValueError(mode)
        final = torch.zeros_like(xf)
        with torch.no_grad():
            em = F.one_hot(cand, num_classes=E + 1).permute(2, 1, 0)
            hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit[:, 0].tolist():
            if e == E:
                continue
            pos, tok = torch.where(em[e])
            cur = xf[tok]
            g, u = F.linear(cur, w.gate_up[e]).chunk(2, dim=-1)
            a = self.act(g, u, relp)
            if mode == "weights3":
                a = a.detach() + (a - a.detach()) * (2.0 / 3.0)
            y = F.linear(a, w.down[e])
            extra = pos >= k
            if bool(extra.any()):   # not routed: value 0, only the boundary term of the router gets a gradient
                y = torch.where(extra[:, None], y.detach(), y)
            y = y * wst[tok, pos, None]
            final.index_add_(0, tok, y.to(final.dtype))
        return final, None, None


# ----------------------------------------------------------------------------------------------------------------------
# backward for several rule variants and arbitrary (hi, lo, coef) row triples
# ----------------------------------------------------------------------------------------------------------------------
def endpoint_triples(n: int, endpoint: str = "B") -> dict:
    """Rows [A_1..A_n, B_1..B_n]: score (q_A - q_B) . beta_X of pair i."""
    ar = torch.arange(n)
    return dict(hi=ar, lo=ar + n, coef=ar if endpoint == "A" else ar + n, weight=torch.ones(n), pair=ar, n_pairs=n)


def path_triples(n: int, steps: int) -> dict:
    """Rows of path_forward (row = j * n + i, j = 0 (B) .. steps (A)): trapezoid rule over the segments,
    sum_j (q_{j+1} - q_j) . (beta_j + beta_{j+1}) / 2."""
    hi, lo, coef, pair = [], [], [], []
    ar = torch.arange(n)
    for j in range(steps):
        for c in (j, j + 1):
            hi.append((j + 1) * n + ar)
            lo.append(j * n + ar)
            coef.append(c * n + ar)
            pair.append(ar)
    hi, lo, coef, pair = (torch.cat(x) for x in (hi, lo, coef, pair))
    return dict(hi=hi, lo=lo, coef=coef, weight=torch.full((hi.numel(),), 0.5), pair=pair, n_pairs=n)


def multi_backward(m: RuleRelP, g: Group, triples: dict, pool: torch.Tensor, variants: list,
                   neuron_agg: tuple = (), tau: torch.Tensor | None = None, track: torch.Tensor | None = None) -> dict:
    """g: forward-passed group with record_h_in (any rows). triples (see endpoint_triples): for triple r the score is
    weight_r * (q[hi_r] - q[lo_r]) . beta[coef_r], accumulated into pair pair_r. pool: bool [n_pairs, T].
    One reverse stream; for every variant its own coefficients (q and routing are those of the 'relp' recomputation of
    each layer, identical for all variants). tau [L]: temperature scale per layer for router='soft'.
    Returns {variant: {'pair_expert': float32 [n_pairs, L, T, E] (cpu), 'total': float64 [L, T, E, I] (cpu, only for
    variants in neuron_agg)}, 'support': bool [L, T, E, I] (cpu)}.
    track: int64 [M, 4] neuron coordinates (layer, position, expert, neuron) whose per-pair scores are returned as
    'tracked' float32 [n_pairs, M] per variant.
    With variants = ['relp'] and endpoint_triples this is relp_moe_patch.relp_backward_endpoint."""
    if variants[0] != "relp":
        raise ValueError("variants must start with 'relp' (its recomputation records q)")
    B, T = g.ids.shape
    E, I, H, k, dev = m.E, m.I, m.H, m.k, m.dev
    P = int(triples["n_pairs"])
    hi, lo, co, pr = (triples[x].to(dev) for x in ("hi", "lo", "coef", "pair"))
    wt = triples["weight"].to(dev)
    R = hi.numel()
    unit_w = bool((wt == 1).all())
    pool = pool.to(dev)
    keep_pos = pool[pr].reshape(-1)                                   # [R * T]
    pos_of = torch.arange(R * T, device=dev) % T
    pair_of = pr.repeat_interleave(T)
    wt_tok = wt.repeat_interleave(T)
    lam = {}
    for v in variants:
        m.set_rule(v, 1.0)
        with torch.enable_grad():
            hL = g.h.detach().requires_grad_(True)
            d = m.readout(hL, g.target, g.foil, relp=True)
            lam[v], = torch.autograd.grad(d.sum(), hL)
    cos, sin = m.rope(T, g.h)
    mask = m.causal_mask(B, T, g.h.dtype)
    out = {v: {"pair_expert": torch.zeros(P * T, m.L, E, dtype=torch.float32, device=dev)} for v in variants}
    for v in neuron_agg:
        out[v]["total"] = torch.zeros(m.L, T, E, I, dtype=torch.float64, device=dev)
    if track is not None:
        track = track.to(dev)
        for v in variants:
            out[v]["tracked"] = torch.zeros(P, track.shape[0], dtype=torch.float32, device=dev)
    support = torch.zeros(m.L, T, E, I, dtype=torch.bool, device=dev)
    for l, w in m.stream(reverse=True):
        lam_in, rec0 = {}, None
        for vi, v in enumerate(variants):
            m.set_rule(v, None if tau is None else float(tau[l]))
            hin = g.h_in[l].detach().requires_grad_(True)
            with torch.enable_grad():
                hout, rec = m.layer(hin, w, cos, sin, mask, relp=True, pairs=vi == 0)
                lam_in[v], = torch.autograd.grad(hout, hin, grad_outputs=lam[v])
            if vi == 0:
                rec0 = rec
            del hout, hin, rec
        with torch.no_grad():
            topi = rec0.topi.view(B, T, k)
            qp = rec0.q.view(B, T, k, I)
            ia, ib = topi[hi].reshape(R * T, k), topi[lo].reshape(R * T, k)
            qa, qb = qp[hi].reshape(R * T, k, I), qp[lo].reshape(R * T, k, I)
            lam_o = {v: lam[v][co].reshape(R * T, H) for v in variants}
            for e in torch.unique(torch.cat([ia, ib], 1)).tolist():
                in_a, in_b = ia == e, ib == e
                tix = torch.where((in_a.any(1) | in_b.any(1)) & keep_pos)[0]
                if tix.numel() == 0:
                    continue
                q_a = torch.zeros(tix.numel(), I, dtype=qa.dtype, device=dev)
                q_b = torch.zeros_like(q_a)
                ra, sa = torch.where(in_a[tix])
                q_a[ra] = qa[tix[ra], sa]
                rb, sb = torch.where(in_b[tix])
                q_b[rb] = qb[tix[rb], sb]
                dq = q_a.float() - q_b.float()
                tp = pos_of[tix]
                sup = torch.zeros(T, I, dtype=torch.int32, device=dev)
                sup.index_add_(0, tp, ((q_a != 0) | (q_b != 0)).int())
                support[l, :, e] |= sup > 0
                flat = pair_of[tix] * T + tp
                if track is not None:
                    tsel = torch.where((track[:, 0] == l) & (track[:, 2] == e))[0]
                    tmatch = (tp[:, None] == track[tsel, 1][None, :]).float() if tsel.numel() else None
                for v in variants:
                    beta = F.linear(lam_o[v][tix], w.down[e].t())
                    sc = dq * beta.float()
                    if not unit_w:
                        sc = sc * wt_tok[tix, None]
                    if not torch.isfinite(sc).all():
                        raise ValueError(f"Nonfinite attribution ({v}, layer {l}, expert {e})")
                    if "total" in out[v]:
                        out[v]["total"][l, :, e].index_add_(0, tp, sc.double())
                    out[v]["pair_expert"][:, l, e].index_add_(0, flat, sc.sum(-1))
                    if track is not None and tsel.numel():
                        vals = sc[:, track[tsel, 3]] * tmatch                       # [m, M_le]
                        acc = torch.zeros(P, tsel.numel(), dtype=torch.float32, device=dev)
                        acc.index_add_(0, pair_of[tix], vals)
                        out[v]["tracked"][:, tsel] += acc                            # (advanced indexing copies)
        for v in variants:
            lam[v] = lam_in[v].detach()
        del w, rec0, lam_in
    res = {"support": support.cpu()}
    for v in variants:
        res[v] = {"pair_expert": out[v]["pair_expert"].view(P, T, m.L, E).permute(0, 2, 1, 3).contiguous().cpu()}
        if "total" in out[v]:
            res[v]["total"] = out[v]["total"].cpu()
        if "tracked" in out[v]:
            res[v]["tracked"] = out[v]["tracked"].cpu()
    m.set_rule("relp", 1.0)
    return res


def path_forward(m: StreamRelP, A: torch.Tensor, Bb: torch.Tensor, target: list, foil: list, steps: int) -> Group:
    """Rows j * n + i: the input embeddings of pair i at fraction j / steps of the way from B (j = 0) to A (j = steps);
    natural forward of every point (its own routing), residual entering each layer recorded for the backward."""
    n, T = A.shape
    with torch.no_grad():
        eA = F.embedding(A.to(m.dev), m.embed)
        eB = F.embedding(Bb.to(m.dev), m.embed)
        pts = [eB]
        for j in range(1, steps):
            pts.append((eB.float() + (j / steps) * (eA.float() - eB.float())).to(eB.dtype))
        pts.append(eA)
        h = torch.cat(pts)
        g = Group("path", torch.cat([Bb] + [A] * steps), list(target) * (steps + 1), list(foil) * (steps + 1),
                  record_h_in=True)
        cos, sin = m.rope(T, h)
        mask = m.causal_mask(h.shape[0], T, h.dtype)
        for l, w in m.stream():
            g.h_in.append(h)
            h, _ = m.layer(h, w, cos, sin, mask)
            del w
        g.h = h
        g.delta = m.readout(h, g.target, g.foil)
    return g


# ----------------------------------------------------------------------------------------------------------------------
# forward helpers shared by the probe and the freeze variants
# ----------------------------------------------------------------------------------------------------------------------
def _rsq(m, x):
    return torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + m.eps)


def _rms_forced(m, x, wt, r=None):
    """StreamRelP.rms with an optional forced denominator r (same operations)."""
    dt = x.dtype
    xf = x.to(torch.float32)
    if r is None:
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + m.eps)
    return wt * (xf * r).to(dt)


def _route_logits(m, xf, w):
    logits = F.linear(xf, w.router)
    if m.family == "mixtral":
        probs = torch.softmax(logits.float(), dim=-1)
    else:
        probs = F.softmax(logits, dtype=torch.float, dim=-1)
    return logits, probs


def _weights_on_set(m, xf, w, ti):
    """Routing weights of the given expert set ti [N, k] from this run's own router (as StreamRelP.route)."""
    logits, probs = _route_logits(m, xf, w)
    tv = probs.gather(1, ti)
    if m.norm_topk:
        tv = tv / tv.sum(dim=-1, keepdim=True)
    if m.cast_route:
        tv = tv.to(logits.dtype)
    return tv


def _experts_gate(m, xf, topi, topw, w, capture=False, forced=None, want_q=False):
    """StreamRelP.experts (no RelP rules). capture: also return (g, silu(g), sigmoid(g)) per routed pair [N, k, I].
    forced = (g_b, silu_b, sig_b) [N, k, I]: the SiLU gate is held at another run's value,
    act = (silu_b + (g - g_b) * sig_b) * u  (= g * sig_b * u; equal to the natural value when g = g_b).
    want_q: also return q [N, k, I] and c [N, k, H]."""
    N, k, E = xf.shape[0], m.k, m.E
    final = torch.zeros_like(xf)
    cap = [xf.new_zeros(N, k, m.I) for _ in range(3)] if capture else None
    qp = xf.new_zeros(N, k, m.I) if want_q else None
    cp = xf.new_zeros(N, k, m.H) if want_q else None
    em = F.one_hot(topi, num_classes=E + 1).permute(2, 1, 0)
    hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()
    for e in hit[:, 0].tolist():
        if e == E:
            continue
        pos, tok = torch.where(em[e])
        g, u = F.linear(xf[tok], w.gate_up[e]).chunk(2, dim=-1)
        if forced is None:
            sv = F.silu(g)
        else:
            sv = forced[1][tok, pos] + (g - forced[0][tok, pos]) * forced[2][tok, pos]
        a = sv * u
        if capture:
            cap[0][tok, pos], cap[1][tok, pos], cap[2][tok, pos] = g, sv, torch.sigmoid(g)
        y = F.linear(a, w.down[e])
        y = y * topw[tok, pos, None]
        yc = y.to(final.dtype)
        final.index_add_(0, tok, yc)
        if want_q:
            qp[tok, pos] = (a * topw[tok, pos, None]).to(m.dtype)
            cp[tok, pos] = yc
    return final, cap, qp, cp


# ----------------------------------------------------------------------------------------------------------------------
# gate probe
# ----------------------------------------------------------------------------------------------------------------------
def gate_probe(m: StreamRelP, A: torch.Tensor, Bb: torch.Tensor, first_changed: int) -> dict:
    """Per (pair, layer, position): how much the quantities RelP holds constant differ between the A and the B run.
    Returns float32 arrays [n, L, T]: jaccard (top-k sets), tv_w (total variation of the routing weight vectors),
    margin_A / margin_B (logit of the k-th minus the (k+1)-th expert), dlogit_rms (all experts), dsig_mean / dsig_frac
    (experts routed in both runs: mean |sigma(g_A) - sigma(g_B)| and share of neurons with a change > 0.1; nan if no common
    expert), lognorm1 / lognorm2 (log RMS_A / RMS_B before attention / before the MoE), attn_tv (mean over heads),
    resid_rel (||h_A - h_B|| / ||h_B|| after the layer); 'logits' fp16 [2, n, L, T, E]; 'topi' int16 [2, n, L, T, k];
    'tau' [L]: RMS logit difference over the top-(k + r) experts of A at positions >= first_changed."""
    n, T = A.shape
    dev, H, k, E = m.dev, m.H, m.k, m.E
    r_extra = 2 if k >= 4 else 1
    names = ("jaccard", "tv_w", "margin_A", "margin_B", "dlogit_rms", "dsig_mean", "dsig_frac", "lognorm1", "lognorm2",
             "attn_tv", "resid_rel")
    res = {x: np.zeros((n, m.L, T), dtype=np.float32) for x in names}
    logits_all = np.zeros((2, n, m.L, T, E), dtype=np.float16)
    topi_all = np.zeros((2, n, m.L, T, k), dtype=np.int16)
    tau = np.zeros(m.L, dtype=np.float64)
    with torch.no_grad():
        h = F.embedding(torch.cat([A, Bb]).to(dev), m.embed)
        cos, sin = m.rope(T, h)
        mask = m.causal_mask(2 * n, T, h.dtype)
        for l, w in m.stream():
            r1 = _rsq(m, h)
            a, p = _attention_p(m, m.rms(h, w.ln1), w, cos, sin, mask)
            hmid = h + a
            r2 = _rsq(m, hmid)
            xf = m.rms(hmid, w.ln2).view(-1, H)
            logits, _ = _route_logits(m, xf, w)
            topi, topw = m.route(xf, w)
            o, cap, _, _ = _experts_gate(m, xf, topi, topw, w, capture=True)
            h = hmid + o.view(2 * n, T, H)
            lg = logits.float().view(2 * n, T, E)
            ti = topi.view(2 * n, T, k)
            tw = topw.float().view(2 * n, T, k)
            sig = cap[2].view(2 * n, T, k, -1)
            wa = torch.zeros(n, T, E, device=dev).scatter_(2, ti[:n], tw[:n])
            wb = torch.zeros(n, T, E, device=dev).scatter_(2, ti[n:], tw[n:])
            ina = torch.zeros(n, T, E, dtype=torch.bool, device=dev).scatter_(2, ti[:n], True)
            inb = torch.zeros(n, T, E, dtype=torch.bool, device=dev).scatter_(2, ti[n:], True)
            inter = (ina & inb).sum(-1).float()
            res["jaccard"][:, l] = (inter / (ina | inb).sum(-1).float()).cpu().numpy()
            res["tv_w"][:, l] = (0.5 * (wa - wb).abs().sum(-1)).cpu().numpy()
            srt = lg.sort(-1, descending=True)[0]
            mg = srt[..., k - 1] - srt[..., k]
            res["margin_A"][:, l], res["margin_B"][:, l] = mg[:n].cpu().numpy(), mg[n:].cpu().numpy()
            dl = lg[:n] - lg[n:]
            res["dlogit_rms"][:, l] = dl.pow(2).mean(-1).sqrt().cpu().numpy()
            cand = lg[:n].topk(k + r_extra, dim=-1)[1]
            dc = dl.gather(2, cand)[:, first_changed:]
            tau[l] = float(dc.double().pow(2).mean().sqrt())
            # SiLU gates of experts routed in both runs (match B's slot to A's slot by expert id)
            match = ti[:n][..., :, None] == ti[n:][..., None, :]                 # [n, T, k(A), k(B)]
            both = match.any(-1)
            slot_b = match.float().argmax(-1)
            sig_b = torch.gather(sig[n:], 2, slot_b[..., None].expand(-1, -1, -1, sig.shape[-1]))
            ds = (sig[:n].float() - sig_b.float()).abs()
            cnt = both.sum(-1).float()
            res["dsig_mean"][:, l] = ((ds.mean(-1) * both).sum(-1) / cnt).cpu().numpy()
            res["dsig_frac"][:, l] = (((ds > 0.1).float().mean(-1) * both).sum(-1) / cnt).cpu().numpy()
            res["lognorm1"][:, l] = torch.log(r1[n:] / r1[:n]).squeeze(-1).cpu().numpy()   # r = 1 / RMS
            res["lognorm2"][:, l] = torch.log(r2[n:] / r2[:n]).squeeze(-1).cpu().numpy()
            res["attn_tv"][:, l] = (0.5 * (p[:n].float() - p[n:].float()).abs().sum(-1)).mean(1).cpu().numpy()
            hf = h.float()
            res["resid_rel"][:, l] = ((hf[:n] - hf[n:]).norm(dim=-1) / hf[n:].norm(dim=-1)).cpu().numpy()
            logits_all[0, :, l], logits_all[1, :, l] = lg[:n].half().cpu().numpy(), lg[n:].half().cpu().numpy()
            topi_all[0, :, l], topi_all[1, :, l] = ti[:n].cpu().numpy(), ti[n:].cpu().numpy()
            del w, cap, o, a, p
    res.update(logits=logits_all, topi=topi_all, tau=tau)
    return res


# ----------------------------------------------------------------------------------------------------------------------
# exact patches: expert / position / neuron units, natural or with later layers partly frozen
# ----------------------------------------------------------------------------------------------------------------------
#                 routing(set+weights) set-only attention norm  SiLU-gate
FREEZE = {
    "natural": (False, False, False, False, False),
    "R": (True, False, False, False, False),
    "Rset": (False, True, False, False, False),
    "A": (False, False, True, False, False),
    "N": (False, False, False, True, False),
    "RA": (True, False, True, False, False),
    "RAN": (True, False, True, True, False),
    "RANG": (True, False, True, True, True),
}


def patch_p8(m: StreamRelP, A: torch.Tensor, Bb: torch.Tensor, target: list, foil: list, units: np.ndarray,
             variants: tuple = ("natural",), sub_rows: int = 3000, scale: float = 1.0) -> dict:
    """Every unit in every freeze variant, in one streamed pass (generalises relp_moe_patch.patch_pass / diagnose_pass).

    units int64 [U, 9]: (row, direction, kind, layer, pos, expert, active_A, active_B, neuron); direction 0 = denoise
    (input B, source A), 1 = noise (input A, source B); kind 0 = position (whole MoE output), 1 = expert (complete
    replacement of its contribution), 2 = neuron (the MoE output changes by (q_src - q_live)[neuron] * W_down,e[:, neuron],
    added in fp32; q = 0 where the expert is not routed).
    Freeze variants (FREEZE): after the patched layer, later layers keep the unpatched INPUT run's routing (top-k set and
    weights) / top-k set only (weights recomputed) / attention patterns / RMSNorm denominators of the residual stream
    (incl. the final norm) / SiLU gates (act = g * sigmoid(g_base) * u; only together with frozen routing).
    scale: the patch is input + scale * (patched - input) at the patched layer (1 = the patch; small values for the
    linearity test).
    Returns {'margin': float32 [U, V], 'route_changes': int32 [U] (natural variant; -1 if absent), 'direct': float32 [U]
    (final-position units: change of the MoE output through the final norm with the input run's denominator; nan
    elsewhere), 'margin_A', 'margin_B' [n]}."""
    n, T = A.shape
    dev, H, k, E = m.dev, m.H, m.k, m.E
    U, V = units.shape[0], len(variants)
    flags = torch.tensor([FREEZE[v] for v in variants], dtype=torch.bool, device=dev)       # [V, 5]
    if bool((flags[:, 4] & ~flags[:, 0]).any()):
        raise ValueError("the SiLU gate can only be frozen together with the routing")
    need_gate = bool(flags[:, 4].any())
    ids_b = torch.cat([A, Bb]).to(dev)
    tg = torch.tensor(list(target) * 2, device=dev)
    fo = torch.tensor(list(foil) * 2, device=dev)
    u = torch.from_numpy(units).to(dev).repeat_interleave(V, 0)
    var = torch.arange(V, device=dev).repeat(U)
    order = torch.argsort(u[:, 3], stable=True)
    uo, vo = u[order], var[order]
    u_dir, u_kind, u_layer, u_pos, u_exp, u_neu = (uo[:, j].contiguous() for j in (1, 2, 3, 4, 5, 8))
    u_in = torch.where(u_dir == 0, uo[:, 0] + n, uo[:, 0])
    u_src = torch.where(u_dir == 0, uo[:, 0], uo[:, 0] + n)
    first = torch.searchsorted(u_layer, torch.arange(m.L + 1, device=dev))
    buf = torch.empty(U * V, T, H, dtype=m.dtype, device=dev)
    dfinal = torch.full((U * V, H), float("nan"), dtype=torch.float32, device=dev)
    changes = torch.zeros(U * V, dtype=torch.int32, device=dev)
    nat = [i for i, v in enumerate(variants) if v == "natural"]
    with torch.no_grad():
        hb = F.embedding(ids_b, m.embed)
        cos, sin = m.rope(T, hb)
        mask_b = m.causal_mask(2 * n, T, hb.dtype)
        for l, w in m.stream():
            r1_b = _rsq(m, hb)
            a, p_b = _attention_p(m, m.rms(hb, w.ln1), w, cos, sin, mask_b)
            hmid = hb + a
            r2_b = _rsq(m, hmid)
            xf = m.rms(hmid, w.ln2).view(-1, H)
            topi, topw = m.route(xf, w)
            o, cap_b, qp, cp = _experts_gate(m, xf, topi, topw, w, capture=need_gate, want_q=True)
            o = o.view(2 * n, T, H)
            ti3, tw3 = topi.view(2 * n, T, k), topw.view(2 * n, T, k)
            ti3_sorted = ti3.sort(-1)[0]
            if need_gate:
                cap_b = [c.view(2 * n, T, k, -1) for c in cap_b]
            n_alive = int(first[l])
            for vi in range(V):
                f_r, f_set, f_a, f_n, f_g = (bool(x) for x in flags[vi])
                sel = torch.where(vo[:n_alive] == vi)[0]
                step = max(1, sub_rows // 3) if f_g else sub_rows
                for s0 in range(0, sel.numel(), step):
                    j = sel[s0:s0 + step]
                    hs, rin = buf[j], u_in[j]
                    msk = m.causal_mask(j.numel(), T, hs.dtype)
                    x1 = _rms_forced(m, hs, w.ln1, r1_b[rin] if f_n else None)
                    a_n, _ = _attention_p(m, x1, w, cos, sin, msk, p_b[rin] if f_a else None)
                    hm = hs + a_n
                    xs = _rms_forced(m, hm, w.ln2, r2_b[rin] if f_n else None).view(-1, H)
                    if f_r:
                        ti, tw = ti3[rin].reshape(-1, k), tw3[rin].reshape(-1, k)
                    elif f_set:
                        ti = ti3[rin].reshape(-1, k)
                        tw = _weights_on_set(m, xs, w, ti)
                    else:
                        ti, tw = m.route(xs, w)
                    if vi in nat:
                        changes[j] += (ti.view(-1, T, k).sort(-1)[0] != ti3_sorted[rin]).any(-1).sum(-1).int()
                    forced = [c[rin].reshape(-1, k, c.shape[-1]) for c in cap_b] if f_g else None
                    os_, _, _, _ = _experts_gate(m, xs, ti, tw, w, forced=forced)
                    buf[j] = hm + os_.view(-1, T, H)
                    del hs, hm, os_, forced
            s0, s1 = int(first[l]), int(first[l + 1])
            for c0 in range(s0, s1, 8000):
                sl = slice(c0, min(s1, c0 + 8000))
                rin, rsrc, tt, kind = u_in[sl], u_src[sl], u_pos[sl], u_kind[sl]
                o_in = o[rin, tt]                                      # [P, H] MoE output of the input run at the unit
                o_new = o_in.clone()
                j = torch.where(kind == 0)[0]
                if j.numel():
                    o_new[j] = o[rsrc[j], tt[j]]
                j = torch.where(kind == 1)[0]
                if j.numel():
                    tl, ts = rin[j] * T + tt[j], rsrc[j] * T + tt[j]
                    c3 = cp.view(2 * n * T, k, H)
                    o_new[j] = _recompose(topi[tl], c3[tl], topi[ts], c3[ts], u_exp[sl][j], E)
                j = torch.where(kind == 2)[0]
                if j.numel():
                    tl, ts = rin[j] * T + tt[j], rsrc[j] * T + tt[j]
                    e, nj = u_exp[sl][j], u_neu[sl][j]
                    q3 = qp.view(2 * n * T, k, -1)

                    def qval(tok):
                        hit = topi[tok] == e[:, None]
                        val = q3[tok, hit.float().argmax(1), nj].float()
                        return torch.where(hit.any(1), val, torch.zeros_like(val))
                    dq = qval(ts) - qval(tl)
                    col = torch.empty(j.numel(), H, dtype=torch.float32, device=dev)
                    for ee in torch.unique(e).tolist():
                        mm = e == ee
                        col[mm] = w.down[ee][:, nj[mm]].t().float()
                    o_new[j] = (o_in[j].float() + dq[:, None] * col).to(o_in.dtype)
                if scale != 1.0:
                    o_new = (o_in.float() + scale * (o_new.float() - o_in.float())).to(o_in.dtype)
                fin = torch.where(tt == T - 1)[0]
                if fin.numel():
                    dfinal[c0 + fin] = o_new[fin].float() - o_in[fin].float()
                hnew = hmid[rin] + o[rin]
                hnew[torch.arange(rin.numel(), device=dev), tt] = hmid[rin, tt] + o_new
                buf[sl] = hnew
                del hnew, o_new, o_in
            hb = hmid + o
            del w, cp, qp, o, hmid, a, cap_b
        dbase = m.readout(hb, tg, fo)
        hf = hb[:, -1].float()
        r_fin = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + m.eps)                       # [2n, 1]
        udir = (m.W_U[tg].float() - m.W_U[fo].float()) * m.final_norm.float()
        direct = (dfinal * udir[u_in] * r_fin[u_in]).sum(-1)
        margin = torch.empty(U * V, dtype=torch.float32, device=dev)
        wu = m.W_U
        for s0 in range(0, U * V, sub_rows):
            s1 = min(U * V, s0 + sub_rows)
            rin = u_in[s0:s1]
            x = buf[s0:s1, -1]
            frz = flags[vo[s0:s1], 3]
            normed = torch.where(frz[:, None], _rms_forced(m, x, m.final_norm, r_fin[rin]), _rms_forced(m, x, m.final_norm))
            margin[s0:s1] = (normed.float() * (wu[tg[rin]].float() - wu[fo[rin]].float())).sum(-1)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(U * V, device=dev)
    mg = margin[inv].view(U, V).cpu().numpy()
    ch = changes[inv].view(U, V)
    return {"margin": mg, "route_changes": (ch[:, nat[0]] if nat else torch.full((U,), -1, device=dev)).cpu().numpy(),
            "direct": direct[inv].view(U, V)[:, 0].cpu().numpy(),
            "margin_A": dbase[:n].cpu().numpy(), "margin_B": dbase[n:].cpu().numpy()}


# ----------------------------------------------------------------------------------------------------------------------
# lockstep: several per-bucket computations served by ONE streamed pass (prompt lengths vary across rows in stage 3)
# ----------------------------------------------------------------------------------------------------------------------
_STOP = object()


class _Client:
    def __init__(self, idx):
        self.idx, self.state, self.go, self.item = idx, "new", False, None
        self.it, self.result, self.error, self.thread = None, None, None, None


class _ClientIter:
    def __init__(self, ls, c, reverse):
        self.ls, self.c, self.reverse = ls, c, reverse

    def __iter__(self):
        return self

    def __next__(self):
        ls, c = self.ls, self.c
        with ls.cv:
            c.state, c.go = "wait", False
            ls.cv.notify_all()
            while not c.go:
                ls.cv.wait()
            c.state = "run"
            item, c.item = c.item, None
        if item is _STOP:
            raise StopIteration
        return item


class _Lockstep:
    def __init__(self):
        import threading
        self.cv = threading.Condition()

    def resume(self, c, item):
        """Let client c run until it blocks again (next layer / a new stream) or finishes."""
        with self.cv:
            c.item, c.go = item, True
            self.cv.notify_all()
            while c.go or c.state == "run":
                self.cv.wait()


def lockstep(m: StreamRelP, fns: list) -> list:
    """Run fns[i](m_i) for all i, where every m_i is a shallow copy of m whose .stream() is served by shared passes over the
    checkpoint: whenever all unfinished clients wait for a stream, one real pass (forward or reverse) is run and each layer's
    weights are handed to the waiting clients one after the other. Clients never run concurrently, so every client computes
    exactly what it would compute alone (same batches, same operations). Clients may need different numbers of passes.
    Returns the results in order; the first client error is re-raised."""
    import copy
    import threading
    ls = _Lockstep()
    clients = [_Client(i) for i in range(len(fns))]

    def body(c, f):
        with ls.cv:
            while not c.go:
                ls.cv.wait()
            c.state = "run"
        try:
            mp = copy.copy(m)

            def stream(reverse=False, c=c):
                c.it = _ClientIter(ls, c, reverse)
                return c.it
            mp.stream = stream
            c.result = f(mp)
        except BaseException as e:   # noqa: BLE001 (re-raised in the caller's thread)
            c.error = e
        with ls.cv:
            c.state, c.go = "done", False
            ls.cv.notify_all()

    for c, f in zip(clients, fns):
        c.thread = threading.Thread(target=body, args=(c, f), daemon=True)
        c.thread.start()
        ls.resume(c, None)
    while True:
        err = next((c.error for c in clients if c.error is not None), None)
        waiting = [c for c in clients if c.state == "wait"]
        if err is not None or not waiting:
            break
        rev = waiting[0].it.reverse
        served = [c for c in waiting if c.it.reverse == rev]
        for l, w in m.stream(reverse=rev):
            for c in served:
                if c.state == "wait":
                    ls.resume(c, (l, w))
            del w
        for c in served:
            if c.state == "wait":
                ls.resume(c, _STOP)
    for c in clients:
        if c.error is not None:
            raise c.error
    return [c.result for c in clients]


# ----------------------------------------------------------------------------------------------------------------------
# add-back with role units (stage 3): each row has its own positions for the roles
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_roles(m: StreamRelP, A: torch.Tensor, Bb: torch.Tensor, target: list, foil: list, pool: torch.Tensor,
                   rolepos: torch.Tensor, masks: dict, unit: str, chunk_rows: int = 1040) -> dict:
    """Held-out interventions with the other run as baseline, for rows of one length (the protocol's noise / denoise of
    relp_sva_moe_run.evaluate, with per-row positions):
      denoise: input B, the selected units' q := the A run's q;   noise: input A, the selected units' q := the B run's q.
    masks: {label: bool [L, R, E]} for unit 'role' (role r of row i sits at position rolepos[i, r]) or {label: bool [L, E]} for
    unit 'all' (the expert at every pool position). pool: bool [T]. Layers in which a label selects nothing run untouched.
    Returns {'margin_A' [n], 'margin_B' [n], 'margin': {label: {'noise': [n], 'denoise': [n]}}} (float32, cpu)."""
    from .relp_moe_stream import Override, forward_pass
    n, T = A.shape
    dev = m.dev
    labels = list(masks)
    spec = [(li, kind) for li in range(len(labels)) for kind in ("noise", "denoise")]
    pool_d = pool.to(dev)
    rp = rolepos.to(dev)
    ar = torch.arange(n, device=dev)
    base = Group("base", torch.cat([A, Bb]), list(target) * 2, list(foil) * 2, pairs=True, keep=set())
    per_group = max(1, chunk_rows // n)
    groups = []
    for c0 in range(0, len(spec), per_group):
        sub = spec[c0:c0 + per_group]
        ids = torch.cat([(Bb if kind == "denoise" else A) for _, kind in sub])

        def ofn(l, w, st, sub=sub):
            out = []
            src = st["layer_out"]["base"]
            for bi, (li, kind) in enumerate(sub):
                mk = masks[labels[li]][l]
                if not bool(mk.any()):
                    continue
                rows_mask = torch.zeros(len(sub) * n, dtype=torch.bool, device=dev)
                rows_mask[bi * n:(bi + 1) * n] = True
                if unit == "all":
                    S = (mk.to(dev)[None, :, None] & pool_d[:, None, None]).expand(T, m.E, m.I)
                else:
                    s3 = torch.zeros(n, T, m.E, dtype=torch.bool, device=dev)
                    for r in range(rp.shape[1]):
                        s3[ar, rp[:, r]] |= mk[r].to(dev)[None, :]
                    S = s3[..., None].expand(n, T, m.E, m.I)
                srows = ar + n if kind == "noise" else ar
                out.append(Override(rows=rows_mask, S=S, value="source", src=src, src_rows=srows))
            return out or None
        groups.append(Group(f"iv{c0}", ids, list(target) * len(sub), list(foil) * len(sub), override_fn=ofn))
    forward_pass(m, [base] + groups)
    mg = torch.cat([g.delta.float().cpu() for g in groups]).view(len(spec), n)
    res = {"margin_A": base.delta[:n].float().cpu(), "margin_B": base.delta[n:].float().cpu(), "margin": {}}
    for si, (li, kind) in enumerate(spec):
        res["margin"].setdefault(labels[li], {})[kind] = mg[si]
    return res


# ----------------------------------------------------------------------------------------------------------------------
# patch semantics at the patched MoE sites (the routing-weight sum question; guide section 8)
# ----------------------------------------------------------------------------------------------------------------------
# At a patched site (row, position, layer) with selected experts C, live routing (R_l, g_l), the source run's routing
# (R_s, g_s), unweighted expert outputs u_e(x) = W_down,e (SiLU(W_g,e x) * W_u,e x), live MoE input x_l and the source
# run's MoE input x_s, the MoE output becomes
#   full          sum_{e in R_l \ C} g_l,e u_e(x_l) + sum_{e in C} g_s,e u_e(x_s)        (the engine's complete replacement:
#                 weights of the two runs mixed, sum 1 - sum_C g_l + sum_C g_s, an expert routed in one run only is injected
#                 or removed)
#   content_both  sum_{e in R_l} g_l,e * (u_e(x_s) if e in C and e in R_s else u_e(x_l))  (guide section 8: the live routing is
#                 kept, only the content of experts routed in both runs is replaced)
#   content_all   sum_{e in R_l} g_l,e * (u_e(x_s) if e in C else u_e(x_l))               (as content_both; experts routed only
#                 live get the content they compute on the source input, which the source run did not use)
#   renorm        the full output with every weight multiplied by s = sum_{R_l} g_l / (sum_{R_l \ C} g_l + sum_{C n R_s} g_s)
#                 (the natural total of the live run, 1 for Qwen3-MoE and Mixtral; s = 0 if nothing is left)
#   route         sum_{e in R_s} g_s,e * (u_e(x_s) if e in C else u_e(x_l))               (the source run's routing decision
#                 at the site is moved as well; with C containing R_s it equals the source MoE output)
# Downstream routing is recomputed as usual. Sites without a selected expert are untouched.
MIX_MODES = ("full", "content_both", "content_all", "renorm", "route")


class MixOverride:
    """Expert-level intervention with one of the MIX_MODES semantics (not 'full': that one is the engine's Override).
    rows [B] bool (intervened rows of the group); S [n, T, E] bool selected experts per intervened row and position;
    src: the source group's LayerOut of this layer (with .xf, set by _mix_layer); src_rows [n] rows in the source group."""

    def __init__(self, rows, S, mode, src, src_rows):
        if mode not in MIX_MODES[1:]:
            raise ValueError(mode)
        self.rows, self.S, self.mode, self.src, self.src_rows = rows, S, mode, src, src_rows


def _mix_layer(self, h, w, cos, sin, mask, relp=False, pairs=False, override=None, out_patch=None):
    """StreamRelP.layer (same operations) that also keeps the MoE input on the record (rec.xf) and applies MixOverride
    objects. Records of groups without pairs are released after the layer (only the source group's are read)."""
    from .relp_moe_stream import LayerOut
    B, T, H = h.shape
    a = self.attention(self.rms(h, w.ln1, relp), w, cos, sin, mask, relp)
    h = h + a
    x2 = self.rms(h, w.ln2, relp)
    xf = x2.view(-1, H)
    topi, topw = self.route(xf, w)
    ovs = [] if override is None else (override if isinstance(override, (list, tuple)) else [override])
    o, qp, cp = self.experts(xf, topi, topw, w, relp, pairs or bool(ovs))
    rec = LayerOut(topi=topi, topw=topw, q=qp, c=cp)
    if ovs:
        o = o.view(B, T, H)
        for ov in ovs:
            o = _apply_mix(self, o, rec, xf, w, ov) if isinstance(ov, MixOverride) else self._apply_override(o, rec, w, ov)
        o = o.reshape(-1, H)
    o = o.reshape(B, T, H)
    if out_patch is not None:
        mk, val = out_patch
        o = torch.where(mk[..., None], val.to(o.dtype), o)
    rec.out = o.detach()
    if pairs:
        rec.xf = xf
    else:
        rec.q = rec.c = None
    return h + o, rec


def _expert_out(m, w, e, x, wt, dtype):
    """HF eager expert e on inputs x [n, H] with routing weights wt [n]: (W_down (SiLU(g) * u)) * w, cast to dtype."""
    g, u = F.linear(x, w.gate_up[e]).chunk(2, dim=-1)
    y = F.linear(m.act(g, u, False), w.down[e])
    return (y * wt[:, None]).to(dtype)


def _apply_mix(m, o, rec, xf, w, ov: MixOverride):
    B, T, H = o.shape
    E = m.E
    rows = torch.where(ov.rows)[0]
    if rows.numel() == 0:
        return o
    sites = ov.S.any(-1)                                            # [n, T]
    if not bool(sites.any()):
        return o
    pi, ti = torch.nonzero(sites, as_tuple=True)
    ri = rows[pi]
    lf = ri * T + ti                                                # live (row, position) in the group's flat N
    sf = ov.src_rows[pi] * T + ti                                   # source (row, position)
    C = ov.S[pi, ti]                                                # [P, E]
    l_ti, l_tw, l_c = rec.topi[lf], rec.topw[lf], rec.c[lf]
    s_ti, s_tw, s_c = ov.src.topi[sf], ov.src.topw[sf], ov.src.c[sf]
    inC_l, inC_s = C.gather(1, l_ti), C.gather(1, s_ti)
    l_in_s = (l_ti[:, :, None] == s_ti[:, None, :]).any(-1)
    dt = o.dtype

    def recompute(contrib, exp, need, x, wts):
        idx = torch.nonzero(need)
        if idx.numel() == 0:
            return contrib
        ee = exp[need]
        for e in torch.unique(ee).tolist():
            sel = ee == e
            p_, j_ = idx[sel, 0], idx[sel, 1]
            contrib[p_, j_] = _expert_out(m, w, e, x[p_], wts[p_, j_], dt)
        return contrib

    if ov.mode in ("content_both", "content_all"):
        exp, valid = l_ti, torch.ones_like(inC_l)
        need = inC_l & l_in_s if ov.mode == "content_both" else inC_l
        contrib = recompute(l_c.clone(), l_ti, need, ov.src.xf[sf], l_tw)
    elif ov.mode == "renorm":
        exp = torch.cat([l_ti, s_ti], 1)
        valid = torch.cat([~inC_l, inC_s], 1)
        kept = torch.where(~inC_l, l_tw.float(), torch.zeros_like(l_tw, dtype=torch.float32)).sum(1) + \
            torch.where(inC_s, s_tw.float(), torch.zeros_like(s_tw, dtype=torch.float32)).sum(1)
        scale = torch.where(kept > 0, l_tw.float().sum(1) / kept.clamp_min(1e-30), torch.zeros_like(kept))
        contrib = (torch.cat([l_c, s_c], 1).float() * scale[:, None, None]).to(dt)
    else:  # route
        exp, valid = s_ti, torch.ones_like(inC_s)
        contrib = recompute(s_c.clone(), s_ti, ~inC_s, xf[lf], s_tw)
    # accumulate in ascending expert order from zeros (as the HF loop's index_add_)
    order = torch.argsort(torch.where(valid, exp, torch.full_like(exp, E + 1)), dim=1, stable=True)
    contrib = torch.gather(contrib, 1, order[..., None].expand(-1, -1, H))
    vs = torch.gather(valid, 1, order)
    acc = torch.zeros(contrib.shape[0], H, dtype=dt, device=o.device)
    for s in range(contrib.shape[1]):
        acc = acc + torch.where(vs[:, s, None], contrib[:, s], torch.zeros_like(contrib[:, s]))
    o = o.clone()
    o[ri, ti] = acc
    return o


def mix_view(m: StreamRelP) -> StreamRelP:
    """A shallow copy of the model whose layer() also applies MixOverride (the engine object itself is not changed)."""
    import copy
    import types
    mv = copy.copy(m)
    mv.layer = types.MethodType(_mix_layer, mv)
    return mv


def evaluate_mix(m: StreamRelP, A: torch.Tensor, Bb: torch.Tensor, target: list, foil: list, pool: torch.Tensor,
                 rolepos: torch.Tensor, masks: dict, unit: str, modes=MIX_MODES, chunk_rows: int = 1040) -> dict:
    """evaluate_roles with the patch semantics of MIX_MODES at every patched site (expert units only).
    'full' uses the engine's Override exactly as evaluate_roles. Returns {'margin_A', 'margin_B',
    'margin': {label: {mode: {'noise': [n], 'denoise': [n]}}}} (float32, cpu)."""
    from .relp_moe_stream import Override, forward_pass
    n, T = A.shape
    dev = m.dev
    mv = mix_view(m)
    labels = list(masks)
    spec = [(li, kind, mode) for li in range(len(labels)) for kind in ("noise", "denoise") for mode in modes]
    pool_d = pool.to(dev)
    rp = rolepos.to(dev)
    ar = torch.arange(n, device=dev)
    base = Group("base", torch.cat([A, Bb]), list(target) * 2, list(foil) * 2, pairs=True, keep=set())
    per_group = max(1, chunk_rows // n)
    groups = []
    for c0 in range(0, len(spec), per_group):
        sub = spec[c0:c0 + per_group]
        ids = torch.cat([(Bb if kind == "denoise" else A) for _, kind, _ in sub])

        def ofn(l, w, st, sub=sub):
            out = []
            src = st["layer_out"]["base"]
            for bi, (li, kind, mode) in enumerate(sub):
                mk = masks[labels[li]][l]
                if not bool(mk.any()):
                    continue
                rows_mask = torch.zeros(len(sub) * n, dtype=torch.bool, device=dev)
                rows_mask[bi * n:(bi + 1) * n] = True
                if unit == "all":
                    s3 = (mk.to(dev)[None, :] & pool_d[:, None])[None].expand(n, T, m.E)
                else:
                    s3 = torch.zeros(n, T, m.E, dtype=torch.bool, device=dev)
                    for r in range(rp.shape[1]):
                        s3[ar, rp[:, r]] |= mk[r].to(dev)[None, :]
                srows = ar + n if kind == "noise" else ar
                if mode == "full":
                    out.append(Override(rows=rows_mask, S=s3[..., None].expand(n, T, m.E, m.I), value="source", src=src,
                                        src_rows=srows))
                else:
                    out.append(MixOverride(rows_mask, s3, mode, src, srows))
            return out or None
        groups.append(Group(f"mix{c0}", ids, list(target) * len(sub), list(foil) * len(sub), override_fn=ofn))
    forward_pass(mv, [base] + groups)
    mg = torch.cat([g.delta.float().cpu() for g in groups]).view(len(spec), n)
    res = {"margin_A": base.delta[:n].float().cpu(), "margin_B": base.delta[n:].float().cpu(), "margin": {}}
    for si, (li, kind, mode) in enumerate(spec):
        res["margin"].setdefault(labels[li], {}).setdefault(mode, {})[kind] = mg[si]
    return res
