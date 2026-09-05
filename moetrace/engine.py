"""Layer-streaming executor for expert-aware causal tracing.

One decoder layer is resident on the GPU at a time (see weights.LayerStreamer). Two row types advance together:

* prefill rows: full short sequences (clean prompt, noised prompt), right-padded, causal attention. At every layer the
  final position's MoE routing, per-expert contributions c_e = w_e * E_e(x) (fp32) and the fp32 block output
  (sum of c_e) are recorded so that intervention vectors can be built in-pass.
* wavefront rows: single-token continuations of a patched run. A patched run equals its parent (noised) run except at
  the final position from the patch layer upward, so a row spawned at layer l starts as
      h = resid_pre_moe_noised[final] + bf16(MoEOut_noised_fp32 + v)
  and then runs layers l+1..L-1 attending to the parent's K/V (positions 0..len-2) plus its own K/V (position len-1).

Numerics follow transformers 5.16 modeling files (RMSNorm in fp32 then cast; RoPE tables fp32 -> bf16; router logits
bf16, softmax fp32, top-k, optional renormalisation; routing weights cast to bf16 for Qwen3/OLMoE and kept fp32 for
Mixtral; expert MLP in bf16; residual stream bf16).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .arch import ArchSpec, load_spec
from .weights import CheckpointStore, LayerStreamer, LayerWeights

BF16 = torch.bfloat16
KINDS = ("zero", "layer", "expert", "expert_scaled", "coalition_clean", "coalition_union")


# ----------------------------------------------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------------------------------------------
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def rope_tables(T: int, head_dim: int, theta: float, device, dtype=BF16):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(T, dtype=torch.float32, device=device)
    freqs = pos[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)  # [T, D]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [B, nH, T, D]; cos/sin broadcastable to [B, 1, T, D]
    return (x * cos) + (rotate_half(x) * sin)


def attn_prefill(q, k, v, causal: torch.Tensor, scale: float) -> torch.Tensor:
    """q [B, nH, T, D], k/v [B, nKV, T, D] -> [B, nH, T, D] (eager-style, softmax in fp32)."""
    B, nH, T, D = q.shape
    nkv = k.shape[1]
    rep = nH // nkv
    qg = q.view(B, nkv, rep, T, D)
    scores = torch.matmul(qg, k[:, :, None].transpose(-1, -2)) * scale  # [B, nkv, rep, T, T]
    scores = scores.masked_fill(~causal, float("-inf"))
    p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(p, v[:, :, None])  # [B, nkv, rep, T, D]
    return out.view(B, nH, T, D)


def attn_wavefront(qw, kw, vw, K, V, parent, valid_len, scale: float, chunk: int = 8192) -> torch.Tensor:
    """Single-token queries attending to a parent prefill row's K/V plus their own K/V.

    qw [W, nH, 1, D]; kw/vw [W, nKV, 1, D]; K/V [P, nKV, T, D]; parent [W] long; valid_len [W] long (= parent len - 1:
    the parent's own final-position key is excluded and replaced by the row's own key).
    """
    W, nH, _, D = qw.shape
    nkv = kw.shape[1]
    rep = nH // nkv
    T = K.shape[2]
    out = torch.empty(W, nH, 1, D, dtype=qw.dtype, device=qw.device)
    ar_T = torch.arange(T, device=qw.device)
    for s in range(0, W, chunk):
        e = min(W, s + chunk)
        pk = K[parent[s:e]]  # [w, nkv, T, D]
        pv = V[parent[s:e]]
        kk = torch.cat([pk, kw[s:e]], dim=2)  # [w, nkv, T+1, D]
        vv = torch.cat([pv, vw[s:e]], dim=2)
        qg = qw[s:e].view(e - s, nkv, rep, 1, D)
        scores = torch.matmul(qg, kk[:, :, None].transpose(-1, -2)) * scale  # [w, nkv, rep, 1, T+1]
        valid = ar_T[None, :] < valid_len[s:e, None]  # [w, T]
        valid = torch.cat([valid, torch.ones(e - s, 1, dtype=torch.bool, device=qw.device)], dim=1)
        scores = scores.masked_fill(~valid[:, None, None, None, :], float("-inf"))
        p = torch.softmax(scores.float(), dim=-1).to(qw.dtype)
        out[s:e] = torch.matmul(p, vv[:, :, None]).view(e - s, nH, 1, D)
    return out


def moe_forward(x: torch.Tensor, w: LayerWeights, spec: ArchSpec, final_map: Optional[torch.Tensor], n_final: int,
                token_chunk: int = 8192):
    """Sparse MoE block on flattened tokens x [N, H] (bf16).

    Returns (out_bf16 [N, H], acc_fp32 [N, H], topi [N, k], topv [N, k] fp32, contrib [n_final, k, H] fp32 or None).
    contrib[j, s] is the contribution of the expert in slot s of the token with final_map[token] == j.
    """
    N, Hd = x.shape
    k, E = spec.top_k, spec.n_experts
    logits = F.linear(x, w.router)  # bf16 [N, E]
    probs = torch.softmax(logits.float(), dim=-1)
    topv, topi = torch.topk(probs, k, dim=-1)
    if spec.norm_topk:
        topv = topv / topv.sum(dim=-1, keepdim=True)
    if spec.family != "mixtral":
        topv = topv.to(x.dtype).float()  # HF casts routing weights back to the activation dtype (not for Mixtral)
    flat_e = topi.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    counts = torch.bincount(flat_e, minlength=E).tolist()
    acc = torch.zeros(N, Hd, dtype=torch.float32, device=x.device)
    contrib = None
    if final_map is not None:
        contrib = torch.zeros(n_final + 1, k, Hd, dtype=torch.float32, device=x.device)  # last row = dump
    start = 0
    for e in range(E):
        c = counts[e]
        if c == 0:
            continue
        sel = order[start : start + c]
        start += c
        tok = sel // k
        slot = sel - tok * k
        for s0 in range(0, c, token_chunk):
            tk = tok[s0 : s0 + token_chunk]
            sl = slot[s0 : s0 + token_chunk]
            xe = x[tk]
            gu = F.linear(xe, w.gate_up[e])
            g, u = gu.chunk(2, dim=-1)
            y = F.linear(F.silu(g) * u, w.down[e])  # bf16 [n, H]
            ce = y.float() * topv[tk, sl, None]  # fp32 contribution of expert e to each token
            acc.index_add_(0, tk, ce)
            if contrib is not None:
                fm = final_map[tk]
                fm = torch.where(fm >= 0, fm, torch.full_like(fm, n_final))
                contrib[fm, sl] = ce
    if contrib is not None:
        contrib = contrib[:n_final]
    return acc.to(x.dtype), acc, topi, topv, contrib


# ----------------------------------------------------------------------------------------------------------------
# job specification
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class PrefillSpec:
    ids: list[int]
    true_id: int
    foil_id: int
    noise_pos: Optional[list[int]] = None
    noise_eps: Optional[torch.Tensor] = None  # fp32 [len(noise_pos), hidden] on CPU


@dataclass
class SpawnSpec:
    layer: int
    parent: int  # prefill row index of the run whose residual/K/V is continued (the noised run)
    clean: int  # prefill row index of the clean run of the same case
    kind: str  # one of KINDS
    expert: int = -1
    partner: int = -1  # equal-norm partner expert (kind == 'expert_scaled')


@dataclass
class PassResult:
    lens: np.ndarray
    logit_true: np.ndarray
    logit_foil: np.ndarray
    logit_true_full: np.ndarray
    logit_foil_full: np.ndarray
    top1: np.ndarray
    route_idx: Optional[np.ndarray]  # [L, B, k]
    route_w: Optional[np.ndarray]  # [L, B, k]
    route_cnorm: Optional[np.ndarray]  # [L, B, k]
    sp_logit_true: np.ndarray
    sp_logit_foil: np.ndarray
    sp_alpha: np.ndarray
    sp_norm_e: np.ndarray
    sp_norm_partner: np.ndarray
    sp_vnorm: np.ndarray
    layer_times: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def delta(self):
        return self.logit_true - self.logit_foil

    @property
    def sp_delta(self):
        return self.sp_logit_true - self.sp_logit_foil


# ----------------------------------------------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, repo_or_dir: str, device: str = "cuda"):
        self.spec, self.snapshot = load_spec(repo_or_dir)
        self.device = device
        self.store = CheckpointStore(self.snapshot, self.spec, device=device)
        self.g = self.store.load_globals()
        self._embed_std: Optional[float] = None
        self.hidden = self.spec.hidden

    @property
    def embed_std(self) -> float:
        if self._embed_std is None:
            self._embed_std = float(self.g["embed"].float().std().item())
        return self._embed_std

    # -- helpers --------------------------------------------------------------------------------------------------
    def _qk(self, x: torch.Tensor, w: LayerWeights, B: int, T: int):
        """Project + q/k norm + reshape. Returns q [B, nH, T, D], k [B, nKV, T, D], v [B, nKV, T, D]."""
        s = self.spec
        q = F.linear(x, w.wq)
        k = F.linear(x, w.wk)
        v = F.linear(x, w.wv)
        if s.qk_norm == "full":  # OLMoE: RMSNorm over the full projection before the head split
            q = rmsnorm(q, w.q_norm, s.rms_eps)
            k = rmsnorm(k, w.k_norm, s.rms_eps)
        q = q.view(B, T, s.n_heads, s.head_dim)
        k = k.view(B, T, s.n_kv_heads, s.head_dim)
        v = v.view(B, T, s.n_kv_heads, s.head_dim)
        if s.qk_norm == "per_head":  # Qwen3: RMSNorm over head_dim
            q = rmsnorm(q, w.q_norm, s.rms_eps)
            k = rmsnorm(k, w.k_norm, s.rms_eps)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def _build_v(self, spawns, idx, acc_f, topi_f, contrib, out_alpha, out_ne, out_np):
        """Intervention vectors for the spawn indices idx (all at the current layer). Returns [n, H] fp32."""
        dev = self.device
        n = len(idx)
        Hd = self.hidden
        v = torch.zeros(n, Hd, dtype=torch.float32, device=dev)
        kinds = np.array([spawns[i].kind for i in idx])
        parent = torch.tensor([spawns[i].parent for i in idx], device=dev)
        clean = torch.tensor([spawns[i].clean for i in idx], device=dev)
        expert = torch.tensor([spawns[i].expert for i in idx], device=dev)
        partner = torch.tensor([spawns[i].partner for i in idx], device=dev)

        def c_of(rows, ex):  # contribution of expert ex in the run of rows (zero if not routed) -> [m, H]
            m = (topi_f[rows] == ex[:, None]).float()
            return (contrib[rows] * m[..., None]).sum(1)

        def delta(rows_c, rows_n, ex):
            return c_of(rows_c, ex) - c_of(rows_n, ex)

        alpha = torch.ones(n, device=dev)
        ne = torch.zeros(n, device=dev)
        npn = torch.zeros(n, device=dev)
        for kind in np.unique(kinds):
            sel_np = np.nonzero(kinds == kind)[0]
            sel = torch.tensor(sel_np, device=dev)
            p, c, ex, pa = parent[sel], clean[sel], expert[sel], partner[sel]
            if kind == "zero":
                continue
            elif kind == "layer":
                v[sel] = acc_f[c] - acc_f[p]
            elif kind == "expert":
                d = delta(c, p, ex)
                v[sel] = d
                ne[sel] = d.norm(dim=-1)
            elif kind == "expert_scaled":
                d = delta(c, p, ex)
                dp = delta(c, p, pa)
                n_e = d.norm(dim=-1)
                n_p = dp.norm(dim=-1)
                a = torch.where(n_e > 0, torch.minimum(n_e, n_p) / n_e.clamp_min(1e-30), torch.zeros_like(n_e))
                v[sel] = d * a[:, None]
                alpha[sel] = a
                ne[sel] = n_e
                npn[sel] = n_p
            elif kind in ("coalition_clean", "coalition_union"):
                # sum over e in S_clean of delta_e = sum_c c_e^clean - sum_{e in S_clean ∩ S_noised} c_e^noised
                tc, tn = topi_f[c], topi_f[p]
                in_clean = (tn[:, :, None] == tc[:, None, :]).any(-1).float()  # [m, k] noised slots whose expert is clean-routed
                vv = contrib[c].sum(1) - (contrib[p] * in_clean[..., None]).sum(1)
                if kind == "coalition_union":
                    # experts routed only in the noised run contribute delta_e = -c_e^noised
                    vv = vv - (contrib[p] * (1.0 - in_clean)[..., None]).sum(1)
                v[sel] = vv
            else:
                raise ValueError(kind)
        out_alpha[idx] = alpha.cpu().numpy()
        out_ne[idx] = ne.cpu().numpy()
        out_np[idx] = npn.cpu().numpy()
        return v

    # -- main pass ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, prefill: list[PrefillSpec], spawns: list[SpawnSpec], record_routing: bool = True,
            log=None, wf_chunk: int = 8192) -> PassResult:
        s, dev = self.spec, self.device
        Hd = self.hidden
        B = len(prefill)
        lens = np.array([len(p.ids) for p in prefill], dtype=np.int64)
        T = int(lens.max())
        ids = torch.zeros(B, T, dtype=torch.long)
        for b, p in enumerate(prefill):
            ids[b, : len(p.ids)] = torch.tensor(p.ids, dtype=torch.long)
        ids = ids.to(dev)
        lens_t = torch.tensor(lens, device=dev)
        final_t = lens_t - 1
        ar = torch.arange(B, device=dev)
        Hs = self.g["embed"][ids]  # [B, T, H] bf16
        rows, poss, eps = [], [], []
        for b, p in enumerate(prefill):
            if p.noise_pos:
                rows += [b] * len(p.noise_pos)
                poss += list(p.noise_pos)
                eps.append(p.noise_eps)
        if rows:
            r = torch.tensor(rows, device=dev)
            c = torch.tensor(poss, device=dev)
            e_ = torch.cat(eps, 0).to(dev)
            Hs[r, c] = (Hs[r, c].float() + e_).to(BF16)

        # spawns grouped by layer
        S = len(spawns)
        by_layer: dict[int, list[int]] = {}
        for i, sp in enumerate(spawns):
            assert sp.kind in KINDS, sp.kind
            by_layer.setdefault(sp.layer, []).append(i)
        wf_H = torch.empty(S, Hd, dtype=BF16, device=dev)
        wf_parent = torch.empty(S, dtype=torch.long, device=dev)
        wf_row_of_spawn = np.full(S, -1, dtype=np.int64)
        n_wf = 0
        sp_alpha = np.ones(S, dtype=np.float32)
        sp_ne = np.zeros(S, dtype=np.float32)
        sp_np = np.zeros(S, dtype=np.float32)
        sp_vnorm = np.zeros(S, dtype=np.float32)

        cos_all, sin_all = rope_tables(T, s.head_dim, s.rope_theta, dev)
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        scale = s.head_dim ** -0.5
        final_idx = ar * T + final_t
        final_map = torch.full((B * T + S,), -1, dtype=torch.long, device=dev)
        final_map[final_idx] = ar

        L = s.n_layers
        if record_routing:
            route_idx = np.zeros((L, B, s.top_k), dtype=np.int32)
            route_w = np.zeros((L, B, s.top_k), dtype=np.float32)
            route_cn = np.zeros((L, B, s.top_k), dtype=np.float32)
        else:
            route_idx = route_w = route_cn = None
        layer_times = []
        t_start = time.time()
        t_prev = t_start
        for l, w in LayerStreamer(self.store):
            t_loaded = time.time()
            # ---- attention: prefill rows
            x = rmsnorm(Hs, w.ln1, s.rms_eps)
            q, k, v = self._qk(x, w, B, T)
            q = apply_rope(q, cos_all, sin_all)
            k = apply_rope(k, cos_all, sin_all)
            o = attn_prefill(q, k, v, causal, scale)
            o = F.linear(o.transpose(1, 2).reshape(B, T, s.n_heads * s.head_dim), w.wo)
            Hs = Hs + o
            # ---- attention: wavefront rows
            if n_wf > 0:
                hw = wf_H[:n_wf]
                xw = rmsnorm(hw, w.ln1, s.rms_eps)
                qw, kw, vw = self._qk(xw, w, n_wf, 1)
                pos = final_t[wf_parent[:n_wf]]
                cw = cos_all[pos][:, None, None, :]
                sw = sin_all[pos][:, None, None, :]
                qw = apply_rope(qw, cw, sw)
                kw = apply_rope(kw, cw, sw)
                ow = attn_wavefront(qw, kw, vw, k, v, wf_parent[:n_wf], pos, scale, chunk=wf_chunk)
                ow = F.linear(ow.reshape(n_wf, s.n_heads * s.head_dim), w.wo)
                wf_H[:n_wf] = hw + ow
            del q, k, v, o, x
            # ---- MoE
            x2 = rmsnorm(Hs, w.ln2, s.rms_eps).view(B * T, Hd)
            if n_wf > 0:
                x2 = torch.cat([x2, rmsnorm(wf_H[:n_wf], w.ln2, s.rms_eps)], 0)
            need_contrib = record_routing or (l in by_layer)
            out_bf, acc, topi, topv, contrib = moe_forward(x2, w, s, final_map[: B * T + n_wf] if need_contrib else None, B)
            resid_final = Hs[ar, final_t]  # pre-MoE residual at the final position (bf16)
            Hs = Hs + out_bf[: B * T].view(B, T, Hd)
            if n_wf > 0:
                wf_H[:n_wf] += out_bf[B * T :]
            acc_f = acc[final_idx]
            topi_f = topi[final_idx]
            topv_f = topv[final_idx]
            if record_routing:
                route_idx[l] = topi_f.cpu().numpy()
                route_w[l] = topv_f.cpu().numpy()
                route_cn[l] = contrib.norm(dim=-1).cpu().numpy()
            # ---- spawns at this layer
            if l in by_layer:
                idx = by_layer[l]
                vvec = self._build_v(spawns, idx, acc_f, topi_f, contrib, sp_alpha, sp_ne, sp_np)
                sp_vnorm[idx] = vvec.norm(dim=-1).cpu().numpy()
                parents = torch.tensor([spawns[i].parent for i in idx], device=dev)
                h_new = resid_final[parents] + (acc_f[parents] + vvec).to(BF16)
                n = len(idx)
                wf_H[n_wf : n_wf + n] = h_new
                wf_parent[n_wf : n_wf + n] = parents
                wf_row_of_spawn[np.array(idx)] = np.arange(n_wf, n_wf + n)
                n_wf += n
            del x2, out_bf, acc, topi, topv, contrib
            torch.cuda.synchronize()
            t_done = time.time()
            layer_times.append((l, t_loaded - t_prev, t_done - t_loaded))
            t_prev = t_done
            if log is not None and (l % 8 == 0 or l == L - 1):
                log(f"  layer {l:3d}: load-wait {layer_times[-1][1]:.2f}s compute {layer_times[-1][2]:.2f}s  wf_rows={n_wf}")
        del w
        # ---- head
        head, norm = self.g["head"], self.g["norm"]
        true_ids = torch.tensor([p.true_id for p in prefill], device=dev)
        foil_ids = torch.tensor([p.foil_id for p in prefill], device=dev)
        hN = rmsnorm(Hs[ar, final_t], norm, s.rms_eps)  # [B, H]
        top1 = torch.empty(B, dtype=torch.long, device=dev)
        lt_full = torch.empty(B, dtype=torch.float32, device=dev)
        lf_full = torch.empty(B, dtype=torch.float32, device=dev)
        for c0 in range(0, B, 512):
            lg = F.linear(hN[c0 : c0 + 512], head)  # bf16 [b, V]
            top1[c0 : c0 + 512] = lg.argmax(-1)
            arb = torch.arange(lg.shape[0], device=dev)
            lt_full[c0 : c0 + 512] = lg[arb, true_ids[c0 : c0 + 512]].float()
            lf_full[c0 : c0 + 512] = lg[arb, foil_ids[c0 : c0 + 512]].float()
        lt = self._pair_logits(hN, head, true_ids)
        lf = self._pair_logits(hN, head, foil_ids)
        # wavefront rows
        sp_lt = np.zeros(S, dtype=np.float32)
        sp_lf = np.zeros(S, dtype=np.float32)
        if n_wf > 0:
            hw = rmsnorm(wf_H[:n_wf], norm, s.rms_eps)
            par = wf_parent[:n_wf]
            wlt = self._pair_logits(hw, head, true_ids[par])
            wlf = self._pair_logits(hw, head, foil_ids[par])
            wlt = wlt.cpu().numpy()
            wlf = wlf.cpu().numpy()
            sp_lt[:] = wlt[wf_row_of_spawn]
            sp_lf[:] = wlf[wf_row_of_spawn]
        total = time.time() - t_start
        if log is not None:
            log(f"  pass done: {B} prefill rows (T={T}), {S} spawn rows, {total:.1f}s")
        return PassResult(
            lens=lens, logit_true=lt.cpu().numpy(), logit_foil=lf.cpu().numpy(),
            logit_true_full=lt_full.cpu().numpy(), logit_foil_full=lf_full.cpu().numpy(), top1=top1.cpu().numpy(),
            route_idx=route_idx, route_w=route_w, route_cnorm=route_cn,
            sp_logit_true=sp_lt, sp_logit_foil=sp_lf, sp_alpha=sp_alpha, sp_norm_e=sp_ne, sp_norm_partner=sp_np,
            sp_vnorm=sp_vnorm, layer_times=layer_times, extra={"total_s": total, "T": T},
        )

    @staticmethod
    def _pair_logits(h: torch.Tensor, head: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """bf16 dot products h[i] . head[ids[i]] with fp32 accumulation (bmm), returned as fp32."""
        out = torch.empty(h.shape[0], dtype=torch.float32, device=h.device)
        for c0 in range(0, h.shape[0], 16384):
            hh = h[c0 : c0 + 16384]
            ww = head[ids[c0 : c0 + 16384]]
            out[c0 : c0 + 16384] = torch.bmm(hh[:, None, :], ww[:, :, None]).view(-1).float()
        return out
