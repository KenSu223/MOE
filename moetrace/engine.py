"""Layer-streaming executor for expert-aware causal tracing.

One decoder layer is resident on the GPU at a time (see weights.LayerStreamer). Two row types advance together:

* prefill rows: full short sequences (clean prompt, noised prompt), right-padded, causal attention. At every layer the
  final position's MoE routing, per-expert contributions c_e = w_e * E_e(x) (fp32) and the fp32 block output
  (sum of c_e) are recorded so that intervention vectors can be built in-pass.
* wavefront rows: single-token continuations of a patched run. A patched run equals its parent (noised) run except at
  the final position from the patch layer upward, so a row spawned at layer l starts as
      h = resid_pre_moe_noised[final] + bf16(MoEOut_noised_fp32 + v)
  and then runs layers l+1..L-1 attending to the parent's K/V (positions 0..len-2) plus its own K/V (position len-1).

  Sublayer kinds (ext2, attention-vs-MoE attribution). Writing the decoder layer as
      h_attn = h_pre + Attn_l(h_pre);   h_out = h_attn + MoE_l(h_attn)
  with all quantities at the final position of the noised (parent) or clean run:
    * attn_layer  spawns BEFORE the MoE of layer l:  h = h_pre_noised + Attn_l_clean  (bf16, the same op order as the
                  clean run), the MoE of layer l then runs on the patched residual, and the row continues from l+1.
    * block       h = (h_pre_noised + Attn_l_clean) + MoE_l_clean: both sublayer outputs of layer l replaced by the
                  clean run's, i.e. attention + MoE patched together (equals `resid` only if h_pre agreed).
    * resid       h = h_out_clean: the clean final-position residual after layer l (classic hidden-state restoration);
                  differs from `block` by the upstream difference h_pre_clean - h_pre_noised.
    * block_diff  numerics check for `block`: h = h_pre_moe_noised + bf16(MoE_noised + (MoE_clean - MoE_noised) +
                  (Attn_clean - Attn_noised)), the two sublayer differences added to the noised residual.
  `layer` (MoE output patch) is unchanged. PassResult.sp_vnorm holds |Attn_clean - Attn_noised| (attn_layer),
  |dAttn + dMoE| (block, block_diff) and |h_out_clean - h_out_noised| (resid) in fp32.

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
KINDS = ("zero", "layer", "expert", "expert_scaled", "coalition_clean", "coalition_union",
         "attn_layer", "block", "resid", "block_diff")
PRE_MOE_KINDS = ("attn_layer",)  # spawned after the attention sublayer, before the MoE of the spawn layer
DIRECT_KINDS = ("block", "resid")  # spawned after the MoE with a directly constructed residual (no _build_v vector)


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


def attn_prefill(q, k, v, causal: torch.Tensor, scale: float, final_t: Optional[torch.Tensor] = None,
                 row_chunk: Optional[int] = None):
    """q [B, nH, T, D], k/v [B, nKV, T, D] -> [B, nH, T, D] (eager-style, softmax in fp32).

    If final_t [B] is given, also returns the fp32 attention distribution of each row's final position over all
    positions, [B, nH, T] (diagnostics). row_chunk processes the batch in row chunks (rows are independent, so the
    result is identical; it only bounds the [B, nH, T, T] score tensor).
    """
    B, nH, T, D = q.shape
    Tk = k.shape[2]  # = T, or T + 1 when slot 0 is an extra (sink) key/value
    if row_chunk is not None and row_chunk < B:
        outs, pfs = [], []
        for s0 in range(0, B, row_chunk):
            r = attn_prefill(q[s0 : s0 + row_chunk], k[s0 : s0 + row_chunk], v[s0 : s0 + row_chunk],
                             causal[s0 : s0 + row_chunk] if causal.dim() == 5 else causal, scale,
                             None if final_t is None else final_t[s0 : s0 + row_chunk])
            if final_t is None:
                outs.append(r)
            else:
                outs.append(r[0])
                pfs.append(r[1])
        out = torch.cat(outs, 0)
        return out if final_t is None else (out, torch.cat(pfs, 0))
    nkv = k.shape[1]
    rep = nH // nkv
    qg = q.view(B, nkv, rep, T, D)
    scores = torch.matmul(qg, k[:, :, None].transpose(-1, -2)) * scale  # [B, nkv, rep, T, Tk]
    scores = scores.masked_fill(~causal, float("-inf"))
    p32 = torch.softmax(scores.float(), dim=-1)
    p = p32.to(q.dtype)
    out = torch.matmul(p, v[:, :, None])  # [B, nkv, rep, T, D]
    out = out.view(B, nH, T, D)
    if final_t is None:
        return out
    ar = torch.arange(B, device=q.device)
    p_final = p32.view(B, nH, T, Tk)[ar, :, final_t, :]  # [B, nH, Tk]
    return out, p_final


def attn_wavefront(qw, kw, vw, K, V, parent, valid_len, scale: float, chunk: int = 8192,
                   sink_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Single-token queries attending to a parent prefill row's K/V plus their own K/V.

    qw [W, nH, 1, D]; kw/vw [W, nKV, 1, D]; K/V [P, nKV, T, D]; parent [W] long; valid_len [W] long (= parent len - 1:
    the parent's own final-position key is excluded and replaced by the row's own key).
    sink_valid (bool [P], ext3): K/V then carry an extra slot 0 (an attendable sink key/value that is not a token of
    the row); it is visible to the rows whose parent has sink_valid set.
    """
    W, nH, _, D = qw.shape
    nkv = kw.shape[1]
    rep = nH // nkv
    T = K.shape[2] - (1 if sink_valid is not None else 0)  # token positions of the parent
    out = torch.empty(W, nH, 1, D, dtype=qw.dtype, device=qw.device)
    ar_T = torch.arange(T, device=qw.device)
    for s in range(0, W, chunk):
        e = min(W, s + chunk)
        pk = K[parent[s:e]]  # [w, nkv, T(+1), D]
        pv = V[parent[s:e]]
        kk = torch.cat([pk, kw[s:e]], dim=2)  # [w, nkv, T(+1)+1, D]
        vv = torch.cat([pv, vw[s:e]], dim=2)
        qg = qw[s:e].view(e - s, nkv, rep, 1, D)
        scores = torch.matmul(qg, kk[:, :, None].transpose(-1, -2)) * scale  # [w, nkv, rep, 1, T(+1)+1]
        valid = ar_T[None, :] < valid_len[s:e, None]  # [w, T]
        if sink_valid is not None:
            valid = torch.cat([sink_valid[parent[s:e]][:, None], valid], dim=1)
        valid = torch.cat([valid, torch.ones(e - s, 1, dtype=torch.bool, device=qw.device)], dim=1)
        scores = scores.masked_fill(~valid[:, None, None, None, :], float("-inf"))
        p = torch.softmax(scores.float(), dim=-1).to(qw.dtype)
        out[s:e] = torch.matmul(p, vv[:, :, None]).view(e - s, nH, 1, D)
    return out


def moe_forward(x: torch.Tensor, w: LayerWeights, spec: ArchSpec, final_map: Optional[torch.Tensor], n_final: int,
                token_chunk: int = 8192, return_logits: bool = False):
    """Sparse MoE block on flattened tokens x [N, H] (bf16).

    Returns (out_bf16 [N, H], acc_fp32 [N, H], topi [N, k], topv [N, k] fp32, contrib [n_final, k, H] fp32 or None).
    contrib[j, s] is the contribution of the expert in slot s of the token with final_map[token] == j.
    With return_logits=True the raw router logits (bf16 [N, E]) are appended as a sixth element.
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
    if return_logits:
        return acc.to(x.dtype), acc, topi, topv, contrib, logits
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
    pos_offset: int = 0  # RoPE position of ids[0] (default 0); lets a row keep the positions of a longer prompt
    sink_donor: int = -1  # ext3: prefill row whose position-0 key/value (every layer) is added as an extra attendable
    #                       slot for this row (a "transplanted sink" that is not a token of the row); -1 = none
    sink_vscale: float = 1.0  # scale of the transplanted VALUE (1 = donor's value; 0 = key-only sink, pure absorber)


@dataclass
class DiagSpec:
    """Optional per-layer diagnostics of the prefill rows (ext3). All off by default; results in PassResult.extra['diag'].

    attn_final          final-position attention distribution over all positions, per head: fp16 [L, B, nH, T]
    resid_norms         L2 norm of the residual stream at every position after each layer: fp32 [L, B, T]
    router_logits_final raw router logits at the final position: fp32 [L, B, E]
    resid_final         final-position residual after each layer: bf16 [L, B, H] (as a torch tensor on CPU)
    token_logprobs      log p(ids[t+1] | ids[..t]) at every prefill position: fp32 [B, T] (NaN where undefined), plus
                        logprob_true / logprob_foil [B] at the final position
    route_all_layers    layers at which the routing of EVERY prefill token is recorded: dict layer -> (topi int16
                        [B, T, k], topv fp32 [B, T, k], router logits fp32 [B, T, E])
    attn_sink           (with attn_final, only when some row has a sink_donor) final-position attention mass on the
                        transplanted sink slot: fp16 [L, B, nH]
    attn_out_final      (ext2) final-position attention-sublayer output (after o_proj, the vector added to the
                        residual) at each layer: bf16 [L, B, H] (torch tensor on CPU)
    """
    attn_final: bool = False
    resid_norms: bool = False
    router_logits_final: bool = False
    resid_final: bool = False
    token_logprobs: bool = False
    route_all_layers: tuple = ()
    attn_out_final: bool = False

    @property
    def any_layer(self) -> bool:
        return (self.attn_final or self.resid_norms or self.router_logits_final or self.resid_final
                or bool(self.route_all_layers) or self.attn_out_final)


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

    def _build_v(self, spawns, idx, acc_f, topi_f, contrib, out_alpha, out_ne, out_np, attn_f=None):
        """Intervention vectors for the spawn indices idx (all at the current layer). Returns [n, H] fp32.
        attn_f (bf16 [B, H], ext2): final-position attention-sublayer output of the prefill rows (kind block_diff)."""
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
            elif kind == "block_diff":  # ext2 numerics check: both sublayer differences of this layer
                assert attn_f is not None
                v[sel] = (acc_f[c] - acc_f[p]) + (attn_f[c].float() - attn_f[p].float())
            else:
                raise ValueError(kind)
        out_alpha[idx] = alpha.cpu().numpy()
        out_ne[idx] = ne.cpu().numpy()
        out_np[idx] = npn.cpu().numpy()
        return v

    # -- main pass ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, prefill: list[PrefillSpec], spawns: list[SpawnSpec], record_routing: bool = True,
            log=None, wf_chunk: int = 8192, diag: Optional[DiagSpec] = None,
            attn_score_budget: int = 768 * 2**20, bv_chunk: int = 8192) -> PassResult:
        s, dev = self.spec, self.device
        Hd = self.hidden
        B = len(prefill)
        lens = np.array([len(p.ids) for p in prefill], dtype=np.int64)
        T = int(lens.max())
        offsets = np.array([int(getattr(p, "pos_offset", 0)) for p in prefill], dtype=np.int64)
        use_offsets = bool((offsets != 0).any())
        sink_donor = np.array([int(getattr(p, "sink_donor", -1)) for p in prefill], dtype=np.int64)
        use_sink = bool((sink_donor >= 0).any())
        diag = diag or DiagSpec()
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
        by_layer: dict[int, list[int]] = {}  # vector kinds, spawned after the MoE: h = resid_pre_moe + bf16(acc + v)
        by_layer_pre: dict[int, list[int]] = {}  # ext2 attn_layer: spawned after attention, before this layer's MoE
        by_layer_direct: dict[int, list[int]] = {}  # ext2 block / resid: spawned after the MoE, residual built directly
        need_attn_f: set[int] = set()  # layers whose final-position attention output is needed for spawns
        for i, sp in enumerate(spawns):
            assert sp.kind in KINDS, sp.kind
            if sp.kind in PRE_MOE_KINDS:
                by_layer_pre.setdefault(sp.layer, []).append(i)
                need_attn_f.add(sp.layer)
            elif sp.kind in DIRECT_KINDS:
                by_layer_direct.setdefault(sp.layer, []).append(i)
                need_attn_f.add(sp.layer)
            else:
                by_layer.setdefault(sp.layer, []).append(i)
                if sp.kind == "block_diff":
                    need_attn_f.add(sp.layer)
        wf_H = torch.empty(S, Hd, dtype=BF16, device=dev)
        wf_parent = torch.empty(S, dtype=torch.long, device=dev)
        wf_row_of_spawn = np.full(S, -1, dtype=np.int64)
        n_wf = 0
        sp_alpha = np.ones(S, dtype=np.float32)
        sp_ne = np.zeros(S, dtype=np.float32)
        sp_np = np.zeros(S, dtype=np.float32)
        sp_vnorm = np.zeros(S, dtype=np.float32)

        cos_all, sin_all = rope_tables(T + int(offsets.max()), s.head_dim, s.rope_theta, dev)
        offsets_t = torch.tensor(offsets, device=dev)
        if use_offsets:  # per-row RoPE positions t + offset; gathered tables [B, 1, T, D]
            pos_ids = torch.arange(T, device=dev)[None, :] + offsets_t[:, None]
            cos_pre, sin_pre = cos_all[pos_ids][:, None], sin_all[pos_ids][:, None]
        else:  # identical numerics to the original shared-table path
            cos_pre, sin_pre = cos_all[:T], sin_all[:T]
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        scale = s.head_dim ** -0.5
        if use_sink:
            assert (sink_donor < B).all() and (sink_donor[sink_donor >= 0] != np.arange(B)[sink_donor >= 0]).all()
            has_sink_t = torch.tensor(sink_donor >= 0, device=dev)
            donor_t = torch.tensor(np.where(sink_donor >= 0, sink_donor, 0), device=dev)
            vscale_t = torch.tensor([float(getattr(p, "sink_vscale", 1.0)) for p in prefill], device=dev, dtype=BF16)[:, None, None, None]
            mask_ext = torch.cat([has_sink_t[:, None, None, None, None].expand(B, 1, 1, T, 1),
                                  causal[None, None, None].expand(B, 1, 1, T, T)], dim=-1)  # [B, 1, 1, T, T+1]
        # attention score tensor [B, nH, T, T] fp32 is bounded by attn_score_budget bytes via row chunking
        score_bytes = B * s.n_heads * T * T * 4
        row_chunk = None if score_bytes <= attn_score_budget else max(1, attn_score_budget // (s.n_heads * T * T * 4))
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
        dg: dict = {}
        if diag.attn_final:
            dg["attn_final"] = np.zeros((L, B, s.n_heads, T), dtype=np.float16)
            if use_sink:
                dg["attn_sink"] = np.zeros((L, B, s.n_heads), dtype=np.float16)
        if diag.resid_norms:
            dg["resid_norms"] = np.zeros((L, B, T), dtype=np.float32)
        if diag.router_logits_final:
            dg["router_logits_final"] = np.zeros((L, B, s.n_experts), dtype=np.float32)
        if diag.resid_final:
            dg["resid_final"] = torch.empty((L, B, Hd), dtype=BF16)
        if diag.attn_out_final:
            dg["attn_out_final"] = torch.empty((L, B, Hd), dtype=BF16)
        if diag.route_all_layers:
            dg["route_all"] = {}
        valid_pos = (torch.arange(T, device=dev)[None, :] < lens_t[:, None])  # [B, T]
        layer_times = []
        t_start = time.time()
        t_prev = t_start
        for l, w in LayerStreamer(self.store):
            t_loaded = time.time()
            # ---- attention: prefill rows
            x = rmsnorm(Hs, w.ln1, s.rms_eps)
            q, k, v = self._qk(x, w, B, T)
            q = apply_rope(q, cos_pre, sin_pre)
            k = apply_rope(k, cos_pre, sin_pre)
            if use_sink:  # extra slot 0 = donor row's position-0 key/value (zero and masked for rows without a donor)
                hs4 = has_sink_t[:, None, None, None].to(BF16)
                k_att = torch.cat([k[donor_t, :, 0:1, :] * hs4, k], dim=2)  # [B, nkv, T+1, D]
                v_att = torch.cat([v[donor_t, :, 0:1, :] * hs4 * vscale_t, v], dim=2)
                mask_att = mask_ext
            else:
                k_att, v_att, mask_att = k, v, causal
            if diag.attn_final:
                o, p_final = attn_prefill(q, k_att, v_att, mask_att, scale, final_t, row_chunk)
                if use_sink:
                    dg["attn_sink"][l] = p_final[:, :, 0].to(torch.float16).cpu().numpy()
                    p_final = p_final[:, :, 1:]
                dg["attn_final"][l] = p_final.to(torch.float16).cpu().numpy()
                del p_final
            else:
                o = attn_prefill(q, k_att, v_att, mask_att, scale, None, row_chunk)
            o = F.linear(o.transpose(1, 2).reshape(B, T, s.n_heads * s.head_dim), w.wo)
            attn_f = pre_attn_f = None
            if l in need_attn_f or diag.attn_out_final:
                attn_f = o[ar, final_t]  # final-position attention-sublayer output (added to the residual), bf16 [B, H]
                if diag.attn_out_final:
                    dg["attn_out_final"][l] = attn_f.cpu()
            if l in need_attn_f:
                pre_attn_f = Hs[ar, final_t]  # residual entering layer l at the final position, bf16 [B, H]
            Hs = Hs + o
            # ---- attention: wavefront rows
            if n_wf > 0:
                hw = wf_H[:n_wf]
                xw = rmsnorm(hw, w.ln1, s.rms_eps)
                qw, kw, vw = self._qk(xw, w, n_wf, 1)
                pos = final_t[wf_parent[:n_wf]]
                rpos = pos + offsets_t[wf_parent[:n_wf]]  # RoPE position of the wavefront token
                cw = cos_all[rpos][:, None, None, :]
                sw = sin_all[rpos][:, None, None, :]
                qw = apply_rope(qw, cw, sw)
                kw = apply_rope(kw, cw, sw)
                ow = attn_wavefront(qw, kw, vw, k_att, v_att, wf_parent[:n_wf], pos, scale, chunk=wf_chunk,
                                    sink_valid=has_sink_t if use_sink else None)
                ow = F.linear(ow.reshape(n_wf, s.n_heads * s.head_dim), w.wo)
                wf_H[:n_wf] = hw + ow
            # ---- ext2 spawns before the MoE: attn_layer rows start as h_pre_noised + Attn_l_clean (same op order as
            #      the clean run's residual add) and go through this layer's MoE with the other wavefront rows
            if l in by_layer_pre:
                idx = by_layer_pre[l]
                n = len(idx)
                parents = torch.tensor([spawns[i].parent for i in idx], device=dev)
                cleans = torch.tensor([spawns[i].clean for i in idx], device=dev)
                wf_H[n_wf : n_wf + n] = pre_attn_f[parents] + attn_f[cleans]
                sp_vnorm[idx] = (attn_f[cleans].float() - attn_f[parents].float()).norm(dim=-1).cpu().numpy()
                wf_parent[n_wf : n_wf + n] = parents
                wf_row_of_spawn[np.array(idx)] = np.arange(n_wf, n_wf + n)
                n_wf += n
            del q, k, v, o, x, k_att, v_att
            # ---- MoE
            x2 = rmsnorm(Hs, w.ln2, s.rms_eps).view(B * T, Hd)
            if n_wf > 0:
                x2 = torch.cat([x2, rmsnorm(wf_H[:n_wf], w.ln2, s.rms_eps)], 0)
            need_contrib = record_routing or (l in by_layer)
            want_logits = diag.router_logits_final or (l in diag.route_all_layers)
            mo = moe_forward(x2, w, s, final_map[: B * T + n_wf] if need_contrib else None, B, return_logits=want_logits)
            out_bf, acc, topi, topv, contrib = mo[:5]
            if want_logits:
                rlog = mo[5]
                if diag.router_logits_final:
                    dg["router_logits_final"][l] = rlog[final_idx].float().cpu().numpy()
                if l in diag.route_all_layers:
                    dg["route_all"][int(l)] = (topi[: B * T].view(B, T, -1).to(torch.int16).cpu().numpy(),
                                               topv[: B * T].view(B, T, -1).float().cpu().numpy(),
                                               rlog[: B * T].view(B, T, -1).float().cpu().numpy())
                del rlog
            resid_final = Hs[ar, final_t]  # pre-MoE residual at the final position (bf16)
            Hs = Hs + out_bf[: B * T].view(B, T, Hd)
            if n_wf > 0:
                wf_H[:n_wf] += out_bf[B * T :]
            if diag.resid_norms:
                dg["resid_norms"][l] = (Hs.float().norm(dim=-1) * valid_pos).cpu().numpy()
            if diag.resid_final:
                dg["resid_final"][l] = Hs[ar, final_t].cpu()
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
                if len(idx) <= bv_chunk:
                    vvec = self._build_v(spawns, idx, acc_f, topi_f, contrib, sp_alpha, sp_ne, sp_np, attn_f=attn_f)
                else:  # bound the [n, k, H] fp32 temporaries of _build_v (identical result)
                    vvec = torch.cat([self._build_v(spawns, idx[c0 : c0 + bv_chunk], acc_f, topi_f, contrib, sp_alpha, sp_ne, sp_np,
                                                    attn_f=attn_f) for c0 in range(0, len(idx), bv_chunk)], 0)
                sp_vnorm[idx] = vvec.norm(dim=-1).cpu().numpy()
                parents = torch.tensor([spawns[i].parent for i in idx], device=dev)
                h_new = resid_final[parents] + (acc_f[parents] + vvec).to(BF16)
                n = len(idx)
                wf_H[n_wf : n_wf + n] = h_new
                wf_parent[n_wf : n_wf + n] = parents
                wf_row_of_spawn[np.array(idx)] = np.arange(n_wf, n_wf + n)
                n_wf += n
            # ---- ext2 direct kinds (after the MoE): block = (h_pre_noised + Attn_clean) + MoE_clean; resid = h_out_clean
            if l in by_layer_direct:
                idx = by_layer_direct[l]
                n = len(idx)
                kinds_d = np.array([spawns[i].kind for i in idx])
                parents = torch.tensor([spawns[i].parent for i in idx], device=dev)
                cleans = torch.tensor([spawns[i].clean for i in idx], device=dev)
                h_out_f = Hs[ar, final_t]  # residual after layer l at the final position, bf16 [B, H]
                out_f = out_bf[final_idx]  # bf16 MoE output of the prefill rows at the final position
                h_new = torch.empty(n, Hd, dtype=BF16, device=dev)
                vn = torch.zeros(n, dtype=torch.float32, device=dev)
                sel_np = np.nonzero(kinds_d == "block")[0]
                if len(sel_np):
                    sel = torch.tensor(sel_np, device=dev)
                    p, c = parents[sel], cleans[sel]
                    h_new[sel] = (pre_attn_f[p] + attn_f[c]) + out_f[c]
                    vn[sel] = ((attn_f[c].float() - attn_f[p].float()) + (acc_f[c] - acc_f[p])).norm(dim=-1)
                sel_np = np.nonzero(kinds_d == "resid")[0]
                if len(sel_np):
                    sel = torch.tensor(sel_np, device=dev)
                    p, c = parents[sel], cleans[sel]
                    h_new[sel] = h_out_f[c]
                    vn[sel] = (h_out_f[c].float() - h_out_f[p].float()).norm(dim=-1)
                wf_H[n_wf : n_wf + n] = h_new
                sp_vnorm[idx] = vn.cpu().numpy()
                wf_parent[n_wf : n_wf + n] = parents
                wf_row_of_spawn[np.array(idx)] = np.arange(n_wf, n_wf + n)
                n_wf += n
                del h_out_f, out_f, h_new, vn
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
        if diag.token_logprobs:
            dg.update(self._token_logprobs(Hs, ids, lens_t, norm, head, true_ids, foil_ids))
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
            sp_vnorm=sp_vnorm, layer_times=layer_times,
            extra={"total_s": total, "T": T, "diag": dg, "row_chunk": row_chunk, "use_offsets": use_offsets, "use_sink": use_sink},
        )

    def _token_logprobs(self, Hs, ids, lens_t, norm, head, true_ids, foil_ids) -> dict:
        """Next-token log-probabilities at every prefill position (fp32 log-softmax of the bf16 logits)."""
        s = self.spec
        B, T, Hd = Hs.shape
        dev = Hs.device
        lp = torch.full((B, T), float("nan"), dtype=torch.float32, device=dev)
        lp_true = torch.zeros(B, dtype=torch.float32, device=dev)
        lp_foil = torch.zeros(B, dtype=torch.float32, device=dev)
        top1_all = torch.full((B, T), -1, dtype=torch.long, device=dev)
        rows_per_chunk = max(1, int(2**28 // (T * s.vocab)))  # ~1 GB of fp32 log-probs per chunk
        arT = torch.arange(T, device=dev)
        for c0 in range(0, B, rows_per_chunk):
            c1 = min(B, c0 + rows_per_chunk)
            hN = rmsnorm(Hs[c0:c1], norm, s.rms_eps).view(-1, Hd)
            lg = F.linear(hN, head).float().view(c1 - c0, T, -1)
            logp = torch.log_softmax(lg, dim=-1)
            top1_all[c0:c1] = lg.argmax(-1)
            nxt = torch.cat([ids[c0:c1, 1:], torch.zeros(c1 - c0, 1, dtype=torch.long, device=dev)], dim=1)
            g = logp.gather(-1, nxt[..., None])[..., 0]  # [b, T]
            valid = arT[None, :] < (lens_t[c0:c1, None] - 1)
            lp[c0:c1] = torch.where(valid, g, torch.full_like(g, float("nan")))
            fin = lens_t[c0:c1] - 1
            arb = torch.arange(c1 - c0, device=dev)
            lp_true[c0:c1] = logp[arb, fin, true_ids[c0:c1]]
            lp_foil[c0:c1] = logp[arb, fin, foil_ids[c0:c1]]
            del lg, logp
        return {"token_logprobs": lp.cpu().numpy(), "logprob_true": lp_true.cpu().numpy(), "logprob_foil": lp_foil.cpu().numpy(),
                "top1_all": top1_all.cpu().numpy()}

    @staticmethod
    def _pair_logits(h: torch.Tensor, head: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """bf16 dot products h[i] . head[ids[i]] with fp32 accumulation (bmm), returned as fp32."""
        out = torch.empty(h.shape[0], dtype=torch.float32, device=h.device)
        for c0 in range(0, h.shape[0], 16384):
            hh = h[c0 : c0 + 16384]
            ww = head[ids[c0 : c0 + 16384]]
            out[c0 : c0 + 16384] = torch.bmm(hh[:, None, :], ww[:, :, None]).view(-1).float()
        return out
