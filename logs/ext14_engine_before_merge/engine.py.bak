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

  ext5 (Phase 2) additions, all behind new kinds / arguments; existing kinds and outputs are unchanged:
    * attn_head     per-head attention patch (F2). With H_h the head-h output of the final position BEFORE o_proj
                    (bf16 [head_dim]) and W_o[:, h] the matching column block of o_proj,
                        v_h = W_o[:, h] . (H_h_clean - H_h_noised)     (fp32; SpawnSpec.expert = head index)
                    and the row starts before the MoE of layer l as  h = h_pre_noised + bf16(Attn_l_noised + v_h).
                    Because o_proj is linear, sum_h v_h = W_o (H_clean - H_noised) = Attn_clean - Attn_noised up to
                    bf16 rounding of the o_proj outputs; sp_vnorm = |v_h|.
    * coalition_set explicit expert list S (SpawnSpec.experts):  v = sum_{e in S} (c_e_clean - c_e_noised), c_e = 0 when
                    e is not routed in that run. S = clean top-k reproduces coalition_clean exactly, S = every expert
                    (or clean ∪ noised) reproduces `layer` to fp32 summation order.
    * multi         one wavefront row carrying interventions at several layers (SpawnSpec.steps = ((layer, kind,
                    experts), ...), strictly increasing layers; step kinds layer / expert / coalition_set /
                    coalition_clean / zero). The first step spawns the row exactly like the single-layer kind; at every
                    later step layer l' the live row's own MoE output is replaced component-wise by the clean run's:
                        h = h_pre_moe_own + bf16(MoE_own + v),   v = sum_{e in S} (c_e_clean - c_e_own)
                    (kind layer: v = MoE_clean - MoE_own), i.e. "set the patched component to its clean value" with
                    everything else taken from the row's own (already patched upstream) computation.
    ext8 (Phase 3) additions to `multi`, all behind new step kinds / fields; existing kinds and outputs are unchanged:
    * step kinds attn_layer and block. As the FIRST step they spawn the row exactly like the single-layer kinds
      (attn_layer before the MoE of that layer: h = h_pre_parent + Attn_clean; block after it:
      h = (h_pre_parent + Attn_clean) + MoE_clean). As a LATER step at layer l' (row already live):
        attn_layer  the live row's own attention-sublayer output at the final position is replaced by the clean run's,
                    h_mid = h_in_own + Attn_l'_clean (bf16, the clean run's op order); the row's own MoE of l' then runs
                    on h_mid and is added as usual;
        block       attention as attn_layer, and the MoE output is set to the clean run's bf16 MoE output as well:
                    h_out = h_mid + MoE_l'_clean (the row's own MoE of l' is computed and discarded).
      Both sublayers of one layer = one `block` step (layers stay strictly increasing). With block steps at EVERY layer
      the row's final-position residual is the clean one exactly whenever its layer-0 input equals the clean row's
      (STR: the final token is shared), so its Delta equals the clean Delta bit for bit (sanity check).
      multi_vnorm gets one entry per later step: |Attn_clean - Attn_own| (attn_layer), |dAttn + (MoE_clean - MoE_own)|
      (block, fp32). spawn_vectors (diag) stores the same vectors, one list entry per step.
    * noising direction: every kind only uses SpawnSpec.parent (the run that is continued) and SpawnSpec.clean (the
      run whose component values are written in), so "set a component to its corrupted value inside the clean run"
      is parent = clean prefill row, clean = corrupted prefill row (layer, attn_layer, block, coalition_set, multi).
      For metrics in that direction set PrefillSpec.clean_ref on the prefill rows (otherwise the clean row's KL
      reference is inferred from the spawns as the corrupted row) and SpawnSpec.kl_ref (new, -1 = SpawnSpec.clean) on
      the spawns whose KL reference should be another row (e.g. the clean run = the parent).
    * DiagSpec.contrib_dla: per-expert direct logit attribution of the prefill rows' final-position contributions (see
      DiagSpec). Note (structural, not a numerical property): at the final position the MoE is a per-token function, so
      attn_layer steps at EVERY layer already restore the source run's final residual exactly (the MoE inputs become the
      source's); all-attention == all-block whenever the final token is shared.

    ext9 (Phase 4, E4) additions, all behind new fields / step kinds; existing kinds and outputs are unchanged:
    * route masks on PREFILL rows (expert knockout): PrefillSpec.route_mask = ((layer, expert), ...),
      route_mask_pos = "all" (every position of the row, default) | "final" (the final position only),
      route_mask_mode = "reroute" (default) | "zero".
        reroute  the masked experts' router logits are set to -inf before the fp32 softmax, then top-k and the model's
                 own renormalisation rule apply as usual, so the token still uses k experts (the next-best expert takes
                 the slot). For Qwen3-MoE and Mixtral (top-k renormalised) this is exactly "remove the expert from the
                 menu": the weights of the k chosen experts are their original softmax probabilities renormalised over
                 the k (the softmax normaliser cancels). For OLMoE (norm_topk_prob = False) the -inf also renormalises
                 the FULL softmax over the remaining experts, so the surviving weights grow by 1 / (1 - p_masked); it
                 is not the same as "zero the probability after the softmax" there.
        zero     routing unchanged (indices AND weights as in the unmasked computation); the masked experts'
                 contributions c_e are set to 0 (their slot is dropped from the MoE output, no replacement).
      At most E - k experts may be masked per (row, layer). Recorded routing (route_idx / route_w / route_cnorm,
      DiagSpec.route_all_layers) is the masked routing: reroute -> the replacement expert in the slot; zero -> the
      original indices and weights with route_cnorm = 0 (and zero contribution vectors) in the masked slots. Router
      logits in the diagnostics (router_logits_final, route_all_layers) are the RAW router outputs (before the -inf).
      Wavefront rows are never masked, so a spawn whose PARENT row carries a route mask raises ValueError; a masked
      row may be a spawn's source (`clean`), whose recorded components are then those of the masked run.
    * `multi` step kinds attn_head and heads_experts (per-head attention patches inside multi; first and later steps):
        (layer, "attn_head", heads)                  heads = int or tuple of head indices
        (layer, "heads_experts", (heads, experts))   both at one layer (layers stay strictly increasing)
      With H_h the final-position head-h output BEFORE o_proj (bf16 [head_dim]) and W_o[:, h] its o_proj column block,
          v_heads = sum_{h in heads} W_o[:, h] (H_h_source - H_h_own)        (fp32; heads in increasing order)
          h_mid   = h_in_own + bf16(Attn_own + v_heads)
      where "own" is the parent prefill row for a FIRST step (spawned before the MoE of that layer, like the attn_head
      kind; a one-head first step equals the attn_head kind bit for bit) and the live wavefront row itself for a LATER
      step. The row's own MoE of that layer then runs on h_mid. heads_experts additionally sets the listed experts'
      contributions to the source's, exactly like a later coalition_set step on the MoE side:
          h_out = h_mid + bf16(MoE_own + sum_{e in experts} (c_e_source - c_e_own)).
      All heads of a layer = an attn_layer step up to the bf16 rounding of the o_proj outputs. Works in both directions
      (denoising: parent = corrupted, clean = clean; noising: parent = clean, clean = corrupted). sp_vnorm (first step)
      and multi_vnorm (one entry per later step) hold |v_heads| (attn_head) or |v_heads + v_experts| (heads_experts);
      spawn_vectors (diag) store the same vectors, one list entry per step.
    * DiagSpec.contrib_final_vectors = tuple of layers: per-slot expert contribution vectors c_e = w_e E_e(x) (fp32
      [B, k, H], CPU) at the final position of the prefill rows, aligned with route_idx[layer], in
      extra["diag"]["contrib_final_vectors"][layer]; extra["diag"]["moe_out_final_fp32"][layer] is the fp32 MoE output
      at the final position (= sum over slots up to fp32 summation order).

    * metrics=True  (F5) full-vocabulary softmax statistics for every prefill and wavefront row at the head:
                    logp_true, logp_foil (log-softmax of the bf16 logits in fp32), p_true, p_foil, rank_true (1 +
                    number of vocabulary logits strictly greater than the true logit) and kl_to_clean =
                    KL(softmax(row) || softmax(clean prefill row of the same case)), computed in row chunks. The
                    clean row of a prefill row is PrefillSpec.clean_ref, or inferred from the spawns (parent -> clean),
                    else itself (KL = 0 exactly). Note delta = logit_true - logit_foil = logp_true - logp_foil (the
                    log-odds; the normaliser cancels).

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
         "attn_layer", "block", "resid", "block_diff",
         "attn_head", "coalition_set", "multi")  # ext5
PRE_MOE_KINDS = ("attn_layer", "attn_head")  # spawned after the attention sublayer, before the MoE of the spawn layer
DIRECT_KINDS = ("block", "resid")  # spawned after the MoE with a directly constructed residual (no _build_v vector)
MULTI_STEP_KINDS = ("layer", "expert", "coalition_set", "coalition_clean", "zero",  # ext5: kinds allowed in `multi` steps
                    "attn_layer", "block")  # ext8: attention-sublayer steps (block = attention + MoE of one layer)
MULTI_ATTN_STEP_KINDS = ("attn_layer", "block")  # ext8: later steps that replace the live row's attention output
MULTI_STEP_KINDS = MULTI_STEP_KINDS + ("attn_head", "heads_experts")  # ext9: per-head steps (+ experts at one layer)
MULTI_HEAD_STEP_KINDS = ("attn_head", "heads_experts")  # ext9: steps that patch heads before the MoE of their layer
ROUTE_MASK_POS = ("all", "final")  # ext9: PrefillSpec.route_mask_pos
ROUTE_MASK_MODES = ("reroute", "zero")  # ext9: PrefillSpec.route_mask_mode


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
                token_chunk: int = 8192, return_logits: bool = False, logit_mask: Optional[torch.Tensor] = None,
                zero_mask: Optional[torch.Tensor] = None):
    """Sparse MoE block on flattened tokens x [N, H] (bf16).

    Returns (out_bf16 [N, H], acc_fp32 [N, H], topi [N, k], topv [N, k] fp32, contrib [n_final, k, H] fp32 or None).
    contrib[j, s] is the contribution of the expert in slot s of the token with final_map[token] == j.
    With return_logits=True the raw router logits (bf16 [N, E]) are appended as a sixth element.
    ext9 route masks (bool [N, E] or None): logit_mask = experts removed from the router's menu (logit -inf before the
    softmax, then top-k / renormalisation as usual); zero_mask = experts whose contribution is set to 0 with the routing
    (indices and returned weights) unchanged.
    """
    N, Hd = x.shape
    k, E = spec.top_k, spec.n_experts
    logits = F.linear(x, w.router)  # bf16 [N, E]
    if logit_mask is not None:  # ext9 reroute
        probs = torch.softmax(logits.float().masked_fill(logit_mask, float("-inf")), dim=-1)
    else:
        probs = torch.softmax(logits.float(), dim=-1)
    topv, topi = torch.topk(probs, k, dim=-1)
    if spec.norm_topk:
        topv = topv / topv.sum(dim=-1, keepdim=True)
    if spec.family != "mixtral":
        topv = topv.to(x.dtype).float()  # HF casts routing weights back to the activation dtype (not for Mixtral)
    topv_c = topv  # weights applied to the expert outputs
    if zero_mask is not None:  # ext9 zero: masked slots contribute nothing (routing unchanged)
        topv_c = topv * (~torch.gather(zero_mask, 1, topi)).float()
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
            ce = y.float() * topv_c[tk, sl, None]  # fp32 contribution of expert e to each token
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
    clean_ref: int = -1  # ext5 metrics: prefill row whose softmax is the KL reference (-1: inferred from spawns, else self)
    route_mask: tuple = ()  # ext9: ((layer, expert), ...) experts knocked out in this row (see module docstring)
    route_mask_pos: str = "all"  # ext9: "all" positions of the row | "final" position only
    route_mask_mode: str = "reroute"  # ext9: "reroute" (logit -inf, next-best expert takes the slot) | "zero" (c_e := 0)


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
    attn_heads_final    (ext5) layers at which the final position's per-head attention outputs BEFORE o_proj are
                        recorded: dict layer -> bf16 [B, n_heads, head_dim] (torch tensor on CPU)
    spawn_vectors       (ext5, verification only) the fp32 intervention vector of every vector-kind spawn (attn_head,
                        attn_layer difference, _build_v kinds, multi steps): dict spawn index -> fp32 [H] CPU tensor
                        (multi: list per step). Memory-heavy; use on small passes.
    contrib_dla         (ext8) direct logit attribution of every routed expert of the prefill rows at the final position:
                        contrib_dla fp32 [L, B, k] = (c_e * gamma) . (W_U[true] - W_U[foil]) in fp32 (c_e = the fp32
                        contribution w_e E_e(x) of the expert in routing slot k, gamma = the final RMSNorm weight; the
                        normaliser is NOT applied) and final_rms fp32 [B] = sqrt(mean(h_final^2) + eps) of the row's final
                        residual. DLA of a vector v added to row b's final residual with the final norm frozen at row b's
                        scale = (v * gamma) . (W_U[true] - W_U[foil]) / final_rms[b]; e.g. delta_e = c_e(clean) -
                        c_e(corrupt) -> (contrib_dla[clean] - contrib_dla[corrupt]) / final_rms[corrupt].
    contrib_final_vectors (ext9) layers at which the prefill rows' per-slot expert contribution vectors at the final
                        position are recorded: contrib_final_vectors dict layer -> fp32 [B, k, H] CPU tensor (slot s =
                        route_idx[layer, b, s]; c_e = w_e E_e(x), 0 for a zero-masked slot) and moe_out_final_fp32 dict
                        layer -> fp32 [B, H] (the fp32 MoE output = sum over slots up to fp32 summation order).
    """
    attn_final: bool = False
    resid_norms: bool = False
    router_logits_final: bool = False
    resid_final: bool = False
    token_logprobs: bool = False
    route_all_layers: tuple = ()
    attn_out_final: bool = False
    attn_heads_final: tuple = ()
    spawn_vectors: bool = False
    contrib_dla: bool = False
    contrib_final_vectors: tuple = ()  # ext9

    @property
    def any_layer(self) -> bool:
        return (self.attn_final or self.resid_norms or self.router_logits_final or self.resid_final
                or bool(self.route_all_layers) or self.attn_out_final or bool(self.attn_heads_final))


@dataclass
class SpawnSpec:
    layer: int
    parent: int  # prefill row index of the run whose residual/K/V is continued (the noised run)
    clean: int  # prefill row index of the clean run of the same case
    kind: str  # one of KINDS
    expert: int = -1  # expert index (expert, expert_scaled) or head index (attn_head)
    partner: int = -1  # equal-norm partner expert (kind == 'expert_scaled')
    experts: tuple = ()  # ext5 coalition_set: explicit expert list S
    steps: tuple = ()  # ext5 multi: ((layer, kind, experts), ...) with strictly increasing layers; layer == steps[0][0]
    #                    ext9: (layer, "attn_head", heads), (layer, "heads_experts", (heads, experts))
    kl_ref: int = -1  # ext8 metrics: prefill row whose softmax is this row's KL reference (-1 = clean)


def _int_set(x) -> tuple:
    """ext9: None / int / iterable of ints -> sorted tuple of distinct ints."""
    if x is None:
        return ()
    if isinstance(x, (tuple, list, set, frozenset, np.ndarray)):
        return tuple(sorted({int(e) for e in x}))
    return (int(x),)


def _norm_step(step) -> tuple[int, str, tuple]:
    """(layer, kind, experts) of a multi step with experts normalised to a sorted tuple (expert kind: one element).
    ext9: attn_head -> (layer, 'attn_head', heads); heads_experts -> (layer, 'heads_experts', (heads, experts))."""
    layer, kind, ex = step
    assert kind in MULTI_STEP_KINDS, kind
    if kind == "expert":
        ex = (int(ex),) if not isinstance(ex, (tuple, list)) else tuple(int(e) for e in ex)
        assert len(ex) == 1, "expert step takes one expert"
    elif kind == "coalition_set":
        ex = tuple(sorted({int(e) for e in (ex if isinstance(ex, (tuple, list)) else [ex])}))
        assert len(ex) > 0, "empty coalition_set"
    elif kind == "attn_head":  # ext9
        ex = _int_set(ex)
        assert len(ex) > 0, "attn_head step without heads"
    elif kind == "heads_experts":  # ext9
        assert isinstance(ex, (tuple, list)) and len(ex) == 2, "heads_experts step takes (heads, experts)"
        ex = (_int_set(ex[0]), _int_set(ex[1]))
        assert ex[0] or ex[1], "heads_experts step with neither heads nor experts"
    else:
        ex = ()
    return int(layer), kind, ex


def _step_heads(nstep) -> tuple:
    """ext9: head set of a normalised multi step (empty for non-head kinds)."""
    if nstep[1] == "attn_head":
        return nstep[2]
    if nstep[1] == "heads_experts":
        return nstep[2][0]
    return ()


def _first_step_spec(sp: "SpawnSpec") -> "SpawnSpec":
    """The single-layer SpawnSpec equivalent to the first step of a multi spawn (ext9 head kinds: kind attn_head /
    heads_experts with the head set in `experts`; only the kind is used for those)."""
    layer, kind, ex = _norm_step(sp.steps[0])
    assert layer == sp.layer, "multi spawn: layer must equal steps[0][0]"
    if kind in MULTI_HEAD_STEP_KINDS:  # ext9
        return SpawnSpec(layer, sp.parent, sp.clean, kind, experts=_step_heads((layer, kind, ex)))
    return SpawnSpec(layer, sp.parent, sp.clean, kind, expert=ex[0] if kind == "expert" else -1,
                     experts=ex if kind == "coalition_set" else ())


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
    # ext5 metrics (None unless run(metrics=True)): dicts with keys logp_true, logp_foil, p_true, p_foil (fp32),
    # rank_true (int32), kl_to_clean (fp32); arrays [B] (prefill rows) and [S] (spawn rows)
    metrics_prefill: Optional[dict] = None
    metrics_spawn: Optional[dict] = None

    @property
    def delta(self):
        return self.logit_true - self.logit_foil

    @property
    def sp_delta(self):
        return self.sp_logit_true - self.sp_logit_foil


METRIC_NAMES = ("logp_true", "logp_foil", "p_true", "p_foil", "rank_true", "kl_to_clean")


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

    def _set_mask(self, expert_lists, n_experts: int) -> torch.Tensor:
        """bool [m, E] membership mask from a list of m expert tuples."""
        m = len(expert_lists)
        mask = np.zeros((m, n_experts), dtype=bool)
        lens = np.array([len(s) for s in expert_lists], dtype=np.int64)
        if lens.sum():
            rows = np.repeat(np.arange(m), lens)
            cols = np.concatenate([np.asarray(s, dtype=np.int64) for s in expert_lists if len(s)])
            mask[rows, cols] = True
        return torch.from_numpy(mask).to(self.device)

    @staticmethod
    def _set_delta(mask, contrib_a, topi_a, contrib_b, topi_b) -> torch.Tensor:
        """sum_{e in S} c_e(a) - c_e(b) for per-row sets S (mask bool [m, E]); contrib [m, k, H] fp32, topi [m, k]."""
        in_a = torch.gather(mask, 1, topi_a).float()  # [m, k]: slot's expert in S
        in_b = torch.gather(mask, 1, topi_b).float()
        return (contrib_a * in_a[..., None]).sum(1) - (contrib_b * in_b[..., None]).sum(1)

    def _heads_v(self, w: LayerWeights, sets, src_rows: torch.Tensor, src_tab: torch.Tensor, own_rows: torch.Tensor,
                 own_tab: torch.Tensor) -> torch.Tensor:
        """ext9 (generalises the ext5 attn_head vector): v[j] = sum_{h in sets[j]} W_o[:, h] (src_tab[src_rows[j], h] -
        own_tab[own_rows[j], h]) in fp32, heads in increasing order, rows sharing a head in one matmul (one head per row
        reproduces the ext5 attn_head computation exactly). tabs: bf16 [*, n_heads, head_dim]. Returns fp32 [m, H]."""
        s = self.spec
        m, D = len(sets), s.head_dim
        v = torch.zeros(m, self.hidden, dtype=torch.float32, device=self.device)
        member = np.zeros((m, s.n_heads), dtype=bool)
        for j, hs in enumerate(sets):
            if len(hs):
                member[j, np.asarray(hs, dtype=np.int64)] = True
        for h in np.nonzero(member.any(0))[0]:
            sh = torch.tensor(np.nonzero(member[:, h])[0], device=self.device)
            dh = src_tab[src_rows[sh], int(h)].float() - own_tab[own_rows[sh], int(h)].float()  # [m_h, D]
            wo_h = w.wo[:, int(h) * D : (int(h) + 1) * D].float()  # [H, D]
            v[sh] += dh @ wo_h.t()
        return v

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
            elif kind == "coalition_set":  # ext5: explicit expert list S per spawn
                mask = self._set_mask([spawns[idx[j]].experts for j in sel_np], self.spec.n_experts)
                v[sel] = self._set_delta(mask, contrib[c], topi_f[c], contrib[p], topi_f[p])
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
            attn_score_budget: int = 768 * 2**20, bv_chunk: int = 8192,
            metrics: bool = False, metrics_chunk: int = 512) -> PassResult:
        """metrics / metrics_chunk (ext5): full-vocabulary softmax statistics for every row (see module docstring),
        computed in chunks of metrics_chunk rows at the head; fills PassResult.metrics_prefill / metrics_spawn."""
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
        # ext9 route masks: layer -> mode -> (flat token indices b*T + t, expert indices), numpy int64 (built on CPU)
        route_masked = np.zeros(B, dtype=bool)
        rm_by_layer: dict[int, dict[str, tuple]] = {}
        rm_parts: dict[tuple, list] = {}
        for b, p in enumerate(prefill):
            rmask = tuple(getattr(p, "route_mask", ()) or ())
            if not rmask:
                continue
            pos_, mode_ = getattr(p, "route_mask_pos", "all"), getattr(p, "route_mask_mode", "reroute")
            assert pos_ in ROUTE_MASK_POS, f"route_mask_pos {pos_!r}"
            assert mode_ in ROUTE_MASK_MODES, f"route_mask_mode {mode_!r}"
            route_masked[b] = True
            per_l: dict[int, set] = {}
            for (l_, e_) in rmask:
                l_, e_ = int(l_), int(e_)
                assert 0 <= l_ < s.n_layers and 0 <= e_ < s.n_experts, f"route_mask entry {(l_, e_)} out of range"
                per_l.setdefault(l_, set()).add(e_)
            toks = np.arange(lens[b] - 1, lens[b]) if pos_ == "final" else np.arange(lens[b])
            toks = b * T + toks
            for l_, es in per_l.items():
                assert len(es) <= s.n_experts - s.top_k, f"row {b} layer {l_}: at most E - k experts can be masked"
                es_ = np.array(sorted(es), dtype=np.int64)
                rm_parts.setdefault((l_, mode_), []).append((np.repeat(toks, len(es_)), np.tile(es_, len(toks))))
        for (l_, mode_), parts in rm_parts.items():
            rm_by_layer.setdefault(l_, {})[mode_] = (np.concatenate([a for a, _ in parts]), np.concatenate([e for _, e in parts]))
        del rm_parts
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
        need_heads: set[int] = set(int(l) for l in diag.attn_heads_final)  # ext5: per-head pre-o_proj outputs
        multi_later: dict[int, list[tuple[int, int]]] = {}  # ext5: layer -> [(spawn idx, step idx)] for steps >= 1
        multi_attn_later: dict[int, list[tuple[int, int]]] = {}  # ext8: later attn_layer / block steps (attention side)
        multi_kinds: dict[int, list[str]] = {}  # ext8: spawn idx -> normalised step kinds (multi spawns only)
        multi_head_later: dict[int, list[tuple[int, int]]] = {}  # ext9: later attn_head / heads_experts steps (heads side)
        multi_steps_n: dict[int, list] = {}  # ext9: spawn idx -> normalised steps (multi spawns with head steps)
        head_set_first: dict[int, tuple] = {}  # ext9: spawn idx -> heads patched at the spawn (attn_head kind / first step)
        spawn_view = list(spawns)  # multi spawns replaced by their first-step single-layer spec (for _build_v)
        for i, sp in enumerate(spawns):
            assert sp.kind in KINDS, sp.kind
            if route_masked[sp.parent]:  # ext9
                raise ValueError(f"spawn {i}: its parent prefill row {sp.parent} carries a route_mask; wavefront rows are "
                                 "never masked, so patching a knocked-out run is not supported")
            if sp.kind == "attn_head":  # ext9 bookkeeping (unchanged semantics)
                head_set_first[i] = (int(sp.expert),)
            if sp.kind in PRE_MOE_KINDS:
                by_layer_pre.setdefault(sp.layer, []).append(i)
                need_attn_f.add(sp.layer)
                if sp.kind == "attn_head":
                    assert 0 <= sp.expert < s.n_heads, f"attn_head: head index {sp.expert} out of range"
                    need_heads.add(sp.layer)
            elif sp.kind in DIRECT_KINDS:
                by_layer_direct.setdefault(sp.layer, []).append(i)
                need_attn_f.add(sp.layer)
            elif sp.kind == "multi":
                steps = [_norm_step(st) for st in sp.steps]
                assert steps, "multi spawn without steps"
                assert all(steps[j][0] < steps[j + 1][0] for j in range(len(steps) - 1)), "multi: layers must increase"
                spawn_view[i] = _first_step_spec(sp)
                multi_kinds[i] = [st[1] for st in steps]
                if any(st[1] in MULTI_HEAD_STEP_KINDS for st in steps):  # ext9
                    multi_steps_n[i] = steps
                    for st in steps:
                        hs_ = _step_heads(st)
                        assert all(0 <= h < s.n_heads for h in hs_), f"multi head step: head index out of range {hs_}"
                        assert st[1] != "heads_experts" or all(0 <= e < s.n_experts for e in st[2][1]), "expert out of range"
                        if hs_:
                            need_heads.add(st[0])
                if steps[0][1] in PRE_MOE_KINDS or steps[0][1] == "heads_experts":  # ext8: first step attn_layer (ext9:
                    by_layer_pre.setdefault(sp.layer, []).append(i)  # or attn_head / heads_experts), spawned before
                    need_attn_f.add(sp.layer)  # the MoE of its layer
                    if steps[0][1] in MULTI_HEAD_STEP_KINDS:  # ext9
                        head_set_first[i] = _step_heads(steps[0])
                    if steps[0][1] == "heads_experts":  # ext9: the experts part = MoE side of the spawn layer
                        multi_later.setdefault(steps[0][0], []).append((i, 0))
                elif steps[0][1] in DIRECT_KINDS:  # ext8: first step block, spawned after the MoE (direct residual)
                    by_layer_direct.setdefault(sp.layer, []).append(i)
                    need_attn_f.add(sp.layer)
                else:
                    by_layer.setdefault(sp.layer, []).append(i)
                for j in range(1, len(steps)):
                    if steps[j][1] in MULTI_ATTN_STEP_KINDS:  # ext8
                        multi_attn_later.setdefault(steps[j][0], []).append((i, j))
                        need_attn_f.add(steps[j][0])
                    if steps[j][1] in MULTI_HEAD_STEP_KINDS:  # ext9: heads side of a later head step
                        multi_head_later.setdefault(steps[j][0], []).append((i, j))
                    if steps[j][1] not in ("attn_layer", "attn_head"):  # MoE side (block: the MoE output is set to the
                        multi_later.setdefault(steps[j][0], []).append((i, j))  # clean one; heads_experts: its experts)
            else:
                if sp.kind == "coalition_set":
                    assert len(sp.experts) > 0, "coalition_set with an empty expert list"
                by_layer.setdefault(sp.layer, []).append(i)
                if sp.kind == "block_diff":
                    need_attn_f.add(sp.layer)
        multi_vnorm: dict[int, list] = {i: [] for l_ in multi_later for (i, _) in multi_later[l_]}
        for l_ in multi_attn_later:  # ext8: rows whose later steps are all attn_layer
            for (i, _) in multi_attn_later[l_]:
                multi_vnorm.setdefault(i, [])
        for l_ in multi_head_later:  # ext9: rows whose later steps are all attn_head
            for (i, _) in multi_head_later[l_]:
                multi_vnorm.setdefault(i, [])
        mh_v: dict[int, torch.Tensor] = {}  # ext9: spawn idx -> fp32 head vector of its head step at the current layer
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
        if diag.attn_heads_final:
            dg["attn_heads_final"] = {}
        if diag.spawn_vectors:
            dg["spawn_v"] = {}
        if diag.contrib_dla:  # ext8: u_b = gamma * (W_U[true_b] - W_U[foil_b]) in fp32 [B, H]
            tid_ = torch.tensor([p.true_id for p in prefill], device=dev)
            fid_ = torch.tensor([p.foil_id for p in prefill], device=dev)
            u_dla = self.g["norm"].float()[None, :] * (self.g["head"][tid_].float() - self.g["head"][fid_].float())
            dg["contrib_dla"] = np.zeros((L, B, s.top_k), dtype=np.float32)
        cfv_layers = set(int(l_) for l_ in diag.contrib_final_vectors)  # ext9
        if cfv_layers:
            dg["contrib_final_vectors"] = {}
            dg["moe_out_final_fp32"] = {}
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
            heads_f = None
            if l in need_heads:  # ext5: final-position per-head outputs before o_proj, bf16 [B, nH, D]
                heads_f = o[ar, :, final_t, :]
                if l in diag.attn_heads_final:
                    dg["attn_heads_final"][int(l)] = heads_f.cpu()
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
                ow_h = attn_wavefront(qw, kw, vw, k_att, v_att, wf_parent[:n_wf], pos, scale, chunk=wf_chunk,
                                      sink_valid=has_sink_t if use_sink else None)  # per-head, before o_proj
                ow = F.linear(ow_h.reshape(n_wf, s.n_heads * s.head_dim), w.wo)
                if l not in multi_head_later:  # ext9: the per-head outputs are only kept for later head steps
                    del ow_h
                if l in multi_attn_later:  # ext8: the rows' own pre-attention residual (hw is a view of wf_H)
                    ma = multi_attn_later[l]
                    ma_rows = wf_row_of_spawn[np.array([i for i, _ in ma])]
                    assert (ma_rows >= 0).all(), "multi attention step on a row that has not been spawned yet"
                    ma_rows_t = torch.tensor(ma_rows, device=dev)
                    ma_pre = wf_H[ma_rows_t]  # copy
                if l in multi_head_later:  # ext9: same for later head steps
                    mh = multi_head_later[l]
                    mh_rows = wf_row_of_spawn[np.array([i for i, _ in mh])]
                    assert (mh_rows >= 0).all(), "multi head step on a row that has not been spawned yet"
                    mh_rows_t = torch.tensor(mh_rows, device=dev)
                    mh_pre = wf_H[mh_rows_t]  # copy
                wf_H[:n_wf] = hw + ow
                if l in multi_attn_later:  # ext8: h_mid = h_in_own + Attn_clean for later attn_layer / block steps
                    ma_cleans = torch.tensor([spawns[i].clean for i, _ in ma], device=dev)
                    wf_H[ma_rows_t] = ma_pre + attn_f[ma_cleans]
                    ma_da = attn_f[ma_cleans].float() - ow[ma_rows_t].float()  # fp32 [n_ma, H]
                    ma_pos = {int(i): jj for jj, (i, _) in enumerate(ma)}  # block rows: the MoE side adds its difference
                    ma_vn = ma_da.norm(dim=-1).cpu().numpy()
                    for jj, (i, j) in enumerate(ma):
                        if multi_kinds[i][j] == "attn_layer":
                            multi_vnorm[i].append(float(ma_vn[jj]))
                            if diag.spawn_vectors:
                                dg["spawn_v"].setdefault(int(i), []).append(ma_da[jj].cpu())
                    del ma_pre
                if l in multi_head_later:  # ext9: h_mid = h_in_own + bf16(Attn_own + sum_h W_o[:, h](H_h_src - H_h_own))
                    mh_cleans = torch.tensor([spawns[i].clean for i, _ in mh], device=dev)
                    mh_sets = [_step_heads(multi_steps_n[i][j]) for i, j in mh]
                    mh_vh = self._heads_v(w, mh_sets, mh_cleans, heads_f, mh_rows_t, ow_h[:, :, 0, :])
                    wf_H[mh_rows_t] = mh_pre + (ow[mh_rows_t].float() + mh_vh).to(BF16)
                    mh_vn = mh_vh.norm(dim=-1).cpu().numpy()
                    for jj, (i, j) in enumerate(mh):
                        if multi_kinds[i][j] == "attn_head":
                            multi_vnorm[i].append(float(mh_vn[jj]))
                            if diag.spawn_vectors:
                                dg["spawn_v"].setdefault(int(i), []).append(mh_vh[jj].cpu())
                        else:  # heads_experts: combined with the experts' vector on the MoE side of this layer
                            mh_v[int(i)] = mh_vh[jj]
                    del mh_pre, mh_vh, ow_h
            elif l in multi_attn_later or l in multi_head_later:
                raise AssertionError("multi attention step on a row that has not been spawned yet")
            # ---- ext2 spawns before the MoE: attn_layer rows start as h_pre_noised + Attn_l_clean (same op order as
            #      the clean run's residual add) and go through this layer's MoE with the other wavefront rows
            if l in by_layer_pre:
                idx = by_layer_pre[l]
                n = len(idx)
                kinds_p = np.array([spawn_view[i].kind for i in idx])  # ext8: multi rows by their first-step kind
                parents = torch.tensor([spawns[i].parent for i in idx], device=dev)
                cleans = torch.tensor([spawns[i].clean for i in idx], device=dev)
                h_new = torch.empty(n, Hd, dtype=BF16, device=dev)
                vn = torch.zeros(n, dtype=torch.float32, device=dev)
                sel_np = np.nonzero(kinds_p == "attn_layer")[0]
                if len(sel_np):
                    sel = torch.tensor(sel_np, device=dev)
                    p, c = parents[sel], cleans[sel]
                    h_new[sel] = pre_attn_f[p] + attn_f[c]
                    dv = attn_f[c].float() - attn_f[p].float()
                    vn[sel] = dv.norm(dim=-1)
                    if diag.spawn_vectors:
                        for jj, i in enumerate(sel_np):
                            if spawns[idx[i]].kind == "multi":  # ext8: one list entry per step
                                dg["spawn_v"].setdefault(int(idx[i]), []).insert(0, dv[jj].cpu())
                            else:
                                dg["spawn_v"][int(idx[i])] = dv[jj].cpu()
                sel_np = np.nonzero(np.isin(kinds_p, MULTI_HEAD_STEP_KINDS))[0]  # ext9: + multi first head steps
                if len(sel_np):  # ext5: v_h = W_o[:, h] (H_h_clean - H_h_noised), h = h_pre_noised + bf16(Attn_noised + v_h)
                    sel = torch.tensor(sel_np, device=dev)  # (ext9: v = sum over the step's heads, ascending)
                    p, c = parents[sel], cleans[sel]
                    vh = self._heads_v(w, [head_set_first.get(idx[i], ()) for i in sel_np], c, heads_f, p, heads_f)
                    h_new[sel] = pre_attn_f[p] + (attn_f[p].float() + vh).to(BF16)
                    vn[sel] = vh.norm(dim=-1)
                    for jj, i in enumerate(sel_np):
                        if spawns[idx[i]].kind == "multi":  # ext9: one list entry per step
                            if diag.spawn_vectors:
                                dg["spawn_v"].setdefault(int(idx[i]), []).insert(0, vh[jj].cpu())
                            if spawn_view[idx[i]].kind == "heads_experts":  # experts part on the MoE side below
                                mh_v[int(idx[i])] = vh[jj]
                        elif diag.spawn_vectors:
                            dg["spawn_v"][int(idx[i])] = vh[jj].cpu()
                    del vh
                wf_H[n_wf : n_wf + n] = h_new
                sp_vnorm[idx] = vn.cpu().numpy()
                wf_parent[n_wf : n_wf + n] = parents
                wf_row_of_spawn[np.array(idx)] = np.arange(n_wf, n_wf + n)
                n_wf += n
                del h_new, vn
            del q, k, v, o, x, k_att, v_att, heads_f
            # ---- MoE
            x2 = rmsnorm(Hs, w.ln2, s.rms_eps).view(B * T, Hd)
            if n_wf > 0:
                x2 = torch.cat([x2, rmsnorm(wf_H[:n_wf], w.ln2, s.rms_eps)], 0)
            need_contrib = record_routing or (l in by_layer) or (l in multi_later) or diag.contrib_dla or (l in cfv_layers)
            want_logits = diag.router_logits_final or (l in diag.route_all_layers)
            n_multi = 0
            if l in multi_later:  # ext5: live rows receiving a later step here also get their contributions recorded
                ml = multi_later[l]
                m_rows = wf_row_of_spawn[np.array([i for i, _ in ml])]
                assert (m_rows >= 0).all(), "multi step on a row that has not been spawned yet"
                m_rows_t = torch.tensor(m_rows, device=dev)
                n_multi = len(ml)
                final_map[B * T + m_rows_t] = B + torch.arange(n_multi, device=dev)
                wf_pre_multi = wf_H[m_rows_t].clone()  # the rows' own pre-MoE residual at this layer
            lmask = zmask = None
            if l in rm_by_layer:  # ext9 route masks (prefill tokens only; wavefront rows are never masked)
                for mode_, (tk_, ex_) in rm_by_layer[l].items():
                    m_ = torch.zeros(x2.shape[0], s.n_experts, dtype=torch.bool, device=dev)
                    m_[torch.from_numpy(tk_).to(dev), torch.from_numpy(ex_).to(dev)] = True
                    if mode_ == "reroute":
                        lmask = m_
                    else:
                        zmask = m_
                    del m_
            mo = moe_forward(x2, w, s, final_map[: B * T + n_wf] if need_contrib else None, B + n_multi, return_logits=want_logits,
                             logit_mask=lmask, zero_mask=zmask)
            del lmask, zmask
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
                route_cn[l] = contrib[:B].norm(dim=-1).cpu().numpy()
            if diag.contrib_dla:  # ext8
                dg["contrib_dla"][l] = (contrib[:B] * u_dla[:, None, :]).sum(-1).cpu().numpy()
            if l in cfv_layers:  # ext9
                dg["contrib_final_vectors"][int(l)] = contrib[:B].cpu()
                dg["moe_out_final_fp32"][int(l)] = acc_f.cpu()
            # ---- ext5 multi: later steps on live rows, h = h_pre_moe_own + bf16(MoE_own + v), v = clean - own
            if n_multi:
                own_contrib = contrib[B : B + n_multi]  # [n_multi, k, H] fp32
                own_topi = topi[B * T + m_rows_t]
                own_acc = acc[B * T + m_rows_t]
                cleans_m = torch.tensor([spawns[i].clean for i, _ in ml], device=dev)
                steps_m = [_norm_step(spawns[i].steps[j]) for i, j in ml]
                kinds_m = np.array([st[1] for st in steps_m])
                vm = torch.zeros(n_multi, Hd, dtype=torch.float32, device=dev)
                for kind in np.unique(kinds_m):
                    sel_all = np.nonzero(kinds_m == kind)[0]
                    for c0_ in range(0, len(sel_all), bv_chunk):  # ext8: row chunks bound the [n, k, H] fp32 temporaries
                        sel_np = sel_all[c0_ : c0_ + bv_chunk]  # (rows are independent: identical result)
                        sel = torch.tensor(sel_np, device=dev)
                        c = cleans_m[sel]
                        if kind == "zero":
                            continue
                        elif kind == "layer":
                            vm[sel] = acc_f[c] - own_acc[sel]
                        elif kind in ("expert", "coalition_set"):
                            mask = self._set_mask([steps_m[j][2] for j in sel_np], s.n_experts)
                            vm[sel] = self._set_delta(mask, contrib[c], topi_f[c], own_contrib[sel], own_topi[sel])
                        elif kind == "coalition_clean":  # S = the clean run's top-k set
                            mask = torch.zeros(len(sel_np), s.n_experts, dtype=torch.bool, device=dev)
                            mask.scatter_(1, topi_f[c], True)
                            vm[sel] = self._set_delta(mask, contrib[c], topi_f[c], own_contrib[sel], own_topi[sel])
                        elif kind == "block":  # ext8: MoE side of a later block step (residual rebuilt below)
                            vm[sel] = acc_f[c] - own_acc[sel]
                        elif kind == "heads_experts":  # ext9: experts part (heads already patched into h_mid)
                            mask = self._set_mask([steps_m[j][2][1] for j in sel_np], s.n_experts)
                            vm[sel] = self._set_delta(mask, contrib[c], topi_f[c], own_contrib[sel], own_topi[sel])
                        else:
                            raise ValueError(kind)
                h_m = wf_pre_multi + (own_acc + vm).to(BF16)
                blk_np = np.nonzero(kinds_m == "block")[0]
                if len(blk_np):  # ext8: h_out = h_mid + MoE_clean (bf16, the clean run's op), h_mid set on the attention side
                    bsel = torch.tensor(blk_np, device=dev)
                    h_m[bsel] = wf_pre_multi[bsel] + out_bf[final_idx[cleans_m[bsel]]]
                    bpos = torch.tensor([ma_pos[int(ml[jb][0])] for jb in blk_np], device=dev)
                    vm[bsel] = vm[bsel] + ma_da[bpos]  # vnorm / spawn vector of a block step: dAttn + dMoE
                wf_H[m_rows_t] = h_m
                he_np = np.nonzero(kinds_m == "heads_experts")[0]
                if len(he_np):  # ext9: the step's vector = v_heads (attention side) + v_experts
                    vm[torch.tensor(he_np, device=dev)] += torch.stack([mh_v.pop(int(ml[jb][0])) for jb in he_np])
                vn_m = vm.norm(dim=-1).cpu().numpy()
                for jj, (i, j) in enumerate(ml):
                    if j == 0:  # ext9: experts part of a FIRST heads_experts step -> the spawn's own vector
                        sp_vnorm[i] = vn_m[jj]
                        if diag.spawn_vectors:
                            dg["spawn_v"][int(i)][0] = vm[jj].cpu()
                        continue
                    multi_vnorm[i].append(float(vn_m[jj]))
                    if diag.spawn_vectors:
                        dg["spawn_v"].setdefault(int(i), []).append(vm[jj].cpu())
                final_map[B * T + m_rows_t] = -1
                del own_contrib, own_topi, own_acc, vm, wf_pre_multi, h_m
            # ---- spawns at this layer
            if l in by_layer:
                idx = by_layer[l]
                if len(idx) <= bv_chunk:
                    vvec = self._build_v(spawn_view, idx, acc_f, topi_f, contrib, sp_alpha, sp_ne, sp_np, attn_f=attn_f)
                else:  # bound the [n, k, H] fp32 temporaries of _build_v (identical result)
                    vvec = torch.cat([self._build_v(spawn_view, idx[c0 : c0 + bv_chunk], acc_f, topi_f, contrib, sp_alpha, sp_ne, sp_np,
                                                    attn_f=attn_f) for c0 in range(0, len(idx), bv_chunk)], 0)
                sp_vnorm[idx] = vvec.norm(dim=-1).cpu().numpy()
                if diag.spawn_vectors:
                    for jj, i in enumerate(idx):
                        if spawns[i].kind == "multi":
                            dg["spawn_v"].setdefault(int(i), []).insert(0, vvec[jj].cpu())
                        else:
                            dg["spawn_v"][int(i)] = vvec[jj].cpu()
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
                kinds_d = np.array([spawn_view[i].kind for i in idx])  # ext8: multi rows by their first-step kind
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
                    if diag.spawn_vectors:  # ext8: multi rows only (one list entry per step)
                        for jj, i in enumerate(sel_np):
                            if spawns[idx[i]].kind == "multi":
                                dg["spawn_v"].setdefault(int(idx[i]), []).insert(
                                    0, ((attn_f[c[jj]].float() - attn_f[p[jj]].float()) + (acc_f[c[jj]] - acc_f[p[jj]])).cpu())
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
            if l in multi_attn_later:  # ext8 (n_wf > 0 asserted on the attention side)
                del ma_da
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
        if diag.contrib_dla:  # ext8
            dg["final_rms"] = torch.sqrt(Hs[ar, final_t].float().pow(2).mean(-1) + s.rms_eps).cpu().numpy()
        # wavefront rows
        sp_lt = np.zeros(S, dtype=np.float32)
        sp_lf = np.zeros(S, dtype=np.float32)
        hw = None
        if n_wf > 0:
            hw = rmsnorm(wf_H[:n_wf], norm, s.rms_eps)
            par = wf_parent[:n_wf]
            wlt = self._pair_logits(hw, head, true_ids[par])
            wlf = self._pair_logits(hw, head, foil_ids[par])
            wlt = wlt.cpu().numpy()
            wlf = wlf.cpu().numpy()
            sp_lt[:] = wlt[wf_row_of_spawn]
            sp_lf[:] = wlf[wf_row_of_spawn]
        # ext5: full-vocabulary metrics
        metrics_prefill = metrics_spawn = None
        metrics_s = 0.0
        if metrics:
            t_m = time.time()
            clean_of = np.arange(B)
            inferred: dict[int, int] = {}
            for sp in spawns:
                inferred.setdefault(sp.parent, sp.clean)
            for b, p in enumerate(prefill):
                cr = int(getattr(p, "clean_ref", -1))
                clean_of[b] = cr if cr >= 0 else inferred.get(b, b)
            sp_clean = np.array([sp.clean if getattr(sp, "kl_ref", -1) < 0 else sp.kl_ref for sp in spawns], dtype=np.int64)  # ext8 kl_ref
            # log-softmax of every prefill row once (fp32 [B, V]); it is both the prefill rows' own distribution and the
            # KL reference of every row, so a row referencing itself gets KL = 0 exactly
            lp_all = self._logp_cache(hN, head, metrics_chunk)
            metrics_prefill = self._metrics(hN, head, true_ids, foil_ids, lp_all, torch.tensor(clean_of, device=dev),
                                            metrics_chunk, lp_rows=lp_all)
            if n_wf > 0:
                spawn_of_row = np.empty(n_wf, dtype=np.int64)
                spawn_of_row[wf_row_of_spawn] = np.arange(S)
                cmap = torch.tensor(sp_clean[spawn_of_row], device=dev)
                mrow = self._metrics(hw, head, true_ids[par], foil_ids[par], lp_all, cmap, metrics_chunk)
                metrics_spawn = {k_: v_[wf_row_of_spawn] for k_, v_ in mrow.items()}
            else:
                metrics_spawn = {k_: np.zeros(0, dtype=np.int32 if k_ == "rank_true" else np.float32) for k_ in METRIC_NAMES}
            del lp_all
            metrics_s = time.time() - t_m
        total = time.time() - t_start
        if log is not None:
            log(f"  pass done: {B} prefill rows (T={T}), {S} spawn rows, {total:.1f}s" + (f" (metrics {metrics_s:.1f}s)" if metrics else ""))
        return PassResult(
            lens=lens, logit_true=lt.cpu().numpy(), logit_foil=lf.cpu().numpy(),
            logit_true_full=lt_full.cpu().numpy(), logit_foil_full=lf_full.cpu().numpy(), top1=top1.cpu().numpy(),
            route_idx=route_idx, route_w=route_w, route_cnorm=route_cn,
            sp_logit_true=sp_lt, sp_logit_foil=sp_lf, sp_alpha=sp_alpha, sp_norm_e=sp_ne, sp_norm_partner=sp_np,
            sp_vnorm=sp_vnorm, layer_times=layer_times,
            extra={"total_s": total, "T": T, "diag": dg, "row_chunk": row_chunk, "use_offsets": use_offsets, "use_sink": use_sink,
                   "metrics_s": metrics_s, "multi_vnorm": multi_vnorm, "route_masked_rows": int(route_masked.sum())},
            metrics_prefill=metrics_prefill, metrics_spawn=metrics_spawn,
        )

    # -- ext5 metrics helpers ------------------------------------------------------------------------------------
    @staticmethod
    def _logp_cache(hN: torch.Tensor, head: torch.Tensor, chunk: int) -> torch.Tensor:
        """fp32 log-softmax [n, V] of the bf16 logits of the normed rows hN [n, H] (row chunks)."""
        n = hN.shape[0]
        out = torch.empty(n, head.shape[0], dtype=torch.float32, device=hN.device)
        for c0 in range(0, n, chunk):
            lg = F.linear(hN[c0 : c0 + chunk], head).float()
            out[c0 : c0 + chunk] = torch.log_softmax(lg, dim=-1)
            del lg
        return out

    @staticmethod
    def _metrics(hN: torch.Tensor, head: torch.Tensor, true_ids: torch.Tensor, foil_ids: torch.Tensor,
                 cache: torch.Tensor, cache_map: torch.Tensor, chunk: int, lp_rows: Optional[torch.Tensor] = None) -> dict:
        """Full-vocabulary metrics of the rows hN [n, H] (bf16, final-normed): log-softmax in fp32 of the bf16 logits,
        rank of the true token (1 + number of strictly larger logits), KL(row || cache[cache_map[row]]).
        lp_rows (fp32 [n, V], optional): the rows' log-softmax if already computed (prefill rows: the same tensor as
        the cache, so a row referencing itself gets KL = 0 exactly)."""
        n = hN.shape[0]
        dev = hN.device
        lp_t = torch.empty(n, dtype=torch.float32, device=dev)
        lp_f = torch.empty(n, dtype=torch.float32, device=dev)
        rank = torch.empty(n, dtype=torch.int32, device=dev)
        kl = torch.empty(n, dtype=torch.float32, device=dev)
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            arb = torch.arange(c1 - c0, device=dev)
            if lp_rows is not None:
                lp = lp_rows[c0:c1]
            else:
                lp = torch.log_softmax(F.linear(hN[c0:c1], head).float(), dim=-1)  # [b, V]
            t_ = lp[arb, true_ids[c0:c1]]
            lp_t[c0:c1] = t_
            lp_f[c0:c1] = lp[arb, foil_ids[c0:c1]]
            rank[c0:c1] = (lp > t_[:, None]).sum(-1).to(torch.int32) + 1
            ref = cache[cache_map[c0:c1]]
            kl[c0:c1] = (lp.exp() * (lp - ref)).sum(-1)
            del lp, ref
        return {"logp_true": lp_t.cpu().numpy(), "logp_foil": lp_f.cpu().numpy(),
                "p_true": lp_t.exp().cpu().numpy(), "p_foil": lp_f.exp().cpu().numpy(),
                "rank_true": rank.cpu().numpy(), "kl_to_clean": kl.cpu().numpy()}

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
