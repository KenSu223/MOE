"""RelP -> MoE for checkpoints that do not fit on the GPU: Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1 (OLMoE-1B-7B is
supported as well, for node-by-node cross-checks against the guide's reference implementation, ext14_circuit_relp.FModel).

The attribution is exactly the one of docs/relp_moe_implementation_guide.md (sections 2-4) and the SVA protocol of
docs/relp_sva_experiment_protocol.md (sections 5-6); only the execution differs from FModel:

* Layer streaming. One decoder layer's weights are on the GPU at a time (moetrace.weights.LayerStreamer); all rows of a
  pass advance layer by layer (rows are independent).
* RelP backward by reverse streaming. The forward stores the residual entering every layer. The backward reloads layer l,
  recomputes it from that stored residual with the RelP rules (the forward VALUES are identical, so the backward runs at
  the recorded A-endpoint quantities) and back-propagates the coefficient lambda_{l+1} of the layer output to lambda_l
  (one autograd call per layer; cross-position terms through the attention V path are included). The MoE output enters
  the residual additively, so its coefficient is lambda_{l+1} and, for every expert e (routed or not),
  beta_e = W_down,e^T lambda_{l+1} (guide 2.3 / 9.2). beta is evaluated on the union of experts routed at A or B
  (paired) or at A (unpaired); the score is zero elsewhere because q = 0 at both endpoints.
* Sparse experts with native numerics. The forward reproduces transformers 5.22.2's eager modules operation by operation
  (Qwen3MoeDecoderLayer / MixtralDecoderLayer / OlmoeDecoderLayer with eager attention and the eager expert loop: routing
  weights exactly as HF passes them (bf16 for Qwen3 / OLMoE, fp32 for Mixtral), expert output = (W_down act) * w,
  bf16 index_add_ accumulation in ascending expert order), so that for the SAME batch composition it equals the native
  forward bit for bit. bf16 GEMM results depend on the number of rows in a call, so equality is only defined per batch
  composition; the size of that effect is measured separately (verify diagnostics).
* q (the node) = (SiLU(g) * u) * w in the activation dtype, exactly as relp_sva.native_capture defines it; w = 0 for
  experts that are not routed. Interventions replace q per (position, expert, neuron) and rebuild the MoE output:
  a pair whose q is replaced completely by another run's q takes that run's contribution vector (so a complete-position
  patch equals an output patch bit for bit), a pair zeroed completely contributes 0, partially replaced pairs contribute
  W_down q_final; downstream routing is recomputed (no rerouting / renormalisation at the patched layer).

RelP rules (guide 3.1, identical to FModel): RMSNorm denominators detached (incl. Qwen3's per-head and OLMoE's
full-projection q/k norms), attention pattern detached, SiLU's sigmoid gate detached, half rule on SiLU(g) * u, router
top-k set and weights detached (no relevance through the router).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .arch import spec_from_config
from .weights import CheckpointStore, LayerStreamer

FAMILIES = ("qwen3_moe", "mixtral", "olmoe")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, nkv, t, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, nkv, n_rep, t, d).reshape(b, nkv * n_rep, t, d)


@dataclass
class LayerOut:
    """Per-layer record of one functional group (tensors on the GPU, detached)."""
    topi: torch.Tensor                 # [N, k] expert ids (N = B * T)
    topw: torch.Tensor                 # [N, k] routing weights as HF passes them to the experts
    q: Optional[torch.Tensor] = None   # [N, k, I] q of each routed (token, slot) pair, activation dtype
    c: Optional[torch.Tensor] = None   # [N, k, H] contribution of each pair to the MoE output (what index_add_ adds)
    out: Optional[torch.Tensor] = None  # [B, T, H] MoE output


@dataclass
class Override:
    """q intervention of one functional group at one layer (see module docstring).

    rows: [B] bool, rows of the group that are intervened (others untouched).
    S:    callable(row_index_tensor) -> bool [n, T, E, I] selection per intervened row, or a fixed [T, E, I] tensor shared
          by all intervened rows; True = replace q by the value.
    value: 'zero' | 'mean' | 'source'
    mean_q: [T, E, I] (value='mean'): replacement for live-routed pairs only ('routed' restriction of the protocol).
    src:  LayerOut of the source run and src_rows [n] (row index in the source group) for value='source'.
    """
    rows: torch.Tensor
    S: torch.Tensor
    value: str
    mean_q: Optional[torch.Tensor] = None
    src: Optional[LayerOut] = None
    src_rows: Optional[torch.Tensor] = None


class StreamRelP:
    def __init__(self, model_dir: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.model_dir = os.path.abspath(model_dir)
        with open(os.path.join(self.model_dir, "config.json")) as f:
            self.config_json = json.load(f)
        cj = dict(self.config_json)
        if "num_experts" not in cj and "num_local_experts" in cj and cj["architectures"][0] != "MixtralForCausalLM":
            cj["num_experts"] = cj["num_local_experts"]   # transformers 5.x writes this alias for Qwen3-MoE / OLMoE
        self.spec = spec_from_config(cj)
        if self.spec.family not in FAMILIES:
            raise ValueError(f"unsupported family {self.spec.family}")
        if self.config_json.get("hidden_act", "silu") != "silu":
            raise ValueError("SiLU experts only")
        for kk in ("attention_bias", "mlp_bias", "num_shared_experts", "shared_expert_intermediate_size"):
            if self.config_json.get(kk):
                raise ValueError(f"unsupported configuration: {kk}")
        if self.config_json.get("clip_qkv") is not None:
            raise ValueError("clip_qkv would need its own RelP rule audit")
        s = self.spec
        self.family = s.family
        self.dev, self.dtype = device, dtype
        self.store = CheckpointStore(self.model_dir, s, device=device, dtype=dtype)
        g = self.store.load_globals()
        self.embed, self.final_norm, self.W_U = g["embed"], g["norm"], g["head"]
        self.L, self.H, self.nH, self.nKV, self.D = s.n_layers, s.hidden, s.n_heads, s.n_kv_heads, s.head_dim
        self.E, self.k, self.I = s.n_experts, s.top_k, s.inter
        self.eps = s.rms_eps
        self.norm_topk = s.norm_topk
        self.cast_route = self.family != "mixtral"   # HF: Qwen3 / OLMoE cast routing weights to the logits dtype
        self.scaling = self.D ** -0.5
        self.moe = True
        from transformers import AutoConfig
        hf = AutoConfig.from_pretrained(self.model_dir)
        hf._attn_implementation = "eager"
        hf._experts_implementation = "eager"
        self.hfcfg = hf
        self._mods = self._hf_modules()
        self.rotary = self._mods["rotary"](config=hf).to(device)
        self.streamer = LayerStreamer(self.store)
        self._native = None

    # ---- HF classes (native reference and rotary tables) ---------------------------------------------------------
    def _hf_modules(self) -> dict:
        if self.family == "qwen3_moe":
            from transformers.models.qwen3_moe import modeling_qwen3_moe as M
            return dict(layer=M.Qwen3MoeDecoderLayer, norm=M.Qwen3MoeRMSNorm, rotary=M.Qwen3MoeRotaryEmbedding)
        if self.family == "mixtral":
            from transformers.models.mixtral import modeling_mixtral as M
            return dict(layer=M.MixtralDecoderLayer, norm=M.MixtralRMSNorm, rotary=M.MixtralRotaryEmbedding)
        from transformers.models.olmoe import modeling_olmoe as M
        return dict(layer=M.OlmoeDecoderLayer, norm=M.OlmoeRMSNorm, rotary=M.OlmoeRotaryEmbedding)

    def stream(self, reverse: bool = False):
        self.streamer.layers = list(range(self.L))[::-1] if reverse else list(range(self.L))
        return iter(self.streamer)

    # ---- building blocks (HF-exact; relp switches only change the backward) --------------------------------------
    def rms(self, x, w, relp=False):
        dt = x.dtype
        xf = x.to(torch.float32)
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if relp:
            r = r.detach()
        return w * (xf * r).to(dt)

    @staticmethod
    def act(g, u, relp):
        if relp:  # FModel._act: frozen sigmoid gate (identity rule) + half rule; forward value = F.silu(g) * u exactly
            sg = torch.sigmoid(g).detach()
            s = g * sg
            s = s + (F.silu(g) - s).detach()
            return 0.5 * (s * u.detach() + s.detach() * u)
        return F.silu(g) * u

    def rope(self, T: int, like: torch.Tensor):
        pos = torch.arange(T, device=self.dev)[None]
        return self.rotary(like, position_ids=pos)

    def causal_mask(self, B: int, T: int, dtype) -> torch.Tensor:
        m = torch.full((T, T), torch.finfo(dtype).min, dtype=dtype, device=self.dev).triu(1)
        return m[None, None].expand(B, 1, T, T)

    def attention(self, x, w, cos, sin, mask, relp):
        B, T, _ = x.shape
        hs = (B, T, -1, self.D)
        q = F.linear(x, w.wq)
        k = F.linear(x, w.wk)
        v = F.linear(x, w.wv)
        if self.family == "qwen3_moe":   # per-head RMSNorm on [B, T, nH, D]
            q = self.rms(q.view(hs), w.q_norm, relp).transpose(1, 2)
            k = self.rms(k.view(hs), w.k_norm, relp).transpose(1, 2)
        elif self.family == "olmoe":     # RMSNorm over the full projection, then the head split
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
        if relp:
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
        return topi, topv.detach()   # frozen routing (selection and weights) in every backward

    def experts(self, xf, topi, topw, w, relp, pairs: bool):
        """HF eager expert loop. Returns (out [N, H], q [N, k, I] | None, c [N, k, H] | None)."""
        N = xf.shape[0]
        final = torch.zeros_like(xf)
        qp = xf.new_zeros(N, self.k, self.I) if pairs else None
        cp = xf.new_zeros(N, self.k, self.H) if pairs else None
        with torch.no_grad():
            em = F.one_hot(topi, num_classes=self.E + 1).permute(2, 1, 0)
            hit = torch.greater(em.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit[:, 0].tolist():
            if e == self.E:
                continue
            pos, tok = torch.where(em[e])
            cur = xf[tok]
            g, u = F.linear(cur, w.gate_up[e]).chunk(2, dim=-1)
            a = self.act(g, u, relp)
            y = F.linear(a, w.down[e])
            y = y * topw[tok, pos, None]
            yc = y.to(final.dtype)
            final.index_add_(0, tok, yc)
            if pairs:
                qp[tok, pos] = (a * topw[tok, pos, None]).to(self.dtype).detach()
                cp[tok, pos] = yc.detach()
        return final, qp, cp

    # ---- one functional decoder layer ----------------------------------------------------------------------------
    def layer(self, h, w, cos, sin, mask, relp=False, pairs=False, override=None, out_patch=None):
        """override: None, an Override, or a list of Overrides on disjoint rows (applied in order)."""
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
                o = self._apply_override(o, rec, w, ov)
            o = o.reshape(-1, H)
        o = o.reshape(B, T, H)
        if out_patch is not None:  # exact sublayer patch: MoE output at masked (row, position) := values
            mk, val = out_patch
            o = torch.where(mk[..., None], val.to(o.dtype), o)
        rec.out = o.detach()
        return h + o, rec

    def readout(self, h, target_ids, foil_ids, relp=False):
        """Metric F = (W_U[target] - W_U[foil]) . final_norm(h)[last position], fp32 readout (as relp_sva / FModel)."""
        normed = self.rms(h, self.final_norm, relp)
        t = torch.as_tensor(target_ids, device=self.dev)
        f = torch.as_tensor(foil_ids, device=self.dev)
        u = self.W_U[t].float() - self.W_U[f].float()
        return (normed[:, -1].float() * u).sum(-1)

    # ---- interventions -------------------------------------------------------------------------------------------
    def _apply_override(self, o, rec: LayerOut, w, ov: Override):
        """Rebuild the MoE output of the intervened rows from (possibly replaced) pair contributions."""
        B, T, H = o.shape
        k, E = self.k, self.E
        rows = torch.where(ov.rows)[0]
        if rows.numel() == 0:
            return o
        n = rows.numel()
        topi = rec.topi.view(B, T, k)[rows]          # [n, T, k]
        q_l = rec.q.view(B, T, k, -1)[rows]          # [n, T, k, I]
        c_l = rec.c.view(B, T, k, H)[rows]           # [n, T, k, H]
        S = ov.S if ov.S.dim() == 4 else ov.S[None].expand(n, *ov.S.shape)  # [n, T, E, I]
        ar_n = torch.arange(n, device=o.device)[:, None, None]
        ar_t = torch.arange(T, device=o.device)[None, :, None]
        if ov.value == "source":
            s_rows = ov.src_rows
            Ts = ov.src.topi.shape[0] // T
            s_topi = ov.src.topi.view(Ts, T, k)[s_rows]
            s_q = ov.src.q.view(Ts, T, k, -1)[s_rows]
            s_c = ov.src.c.view(Ts, T, k, H)[s_rows]
            # live entries: is the same expert routed in the source run (and in which slot)?
            match = topi[..., :, None] == s_topi[..., None, :]          # [n, T, k(live), k(src)]
            in_src = match.any(-1)
            s_slot = match.float().argmax(-1)
            q_src_live = torch.gather(s_q, 2, s_slot[..., None].expand(-1, -1, -1, s_q.shape[-1]))
            c_src_live = torch.gather(s_c, 2, s_slot[..., None].expand(-1, -1, -1, H))
            q_src_live = torch.where(in_src[..., None], q_src_live, torch.zeros_like(q_src_live))
            c_src_live = torch.where(in_src[..., None], c_src_live, torch.zeros_like(c_src_live))
            # source-only entries (expert not routed live): q_live = 0
            s_only = ~(s_topi[..., :, None] == topi[..., None, :]).any(-1)  # [n, T, k(src)]
            exp_all = torch.cat([topi, torch.where(s_only, s_topi, torch.full_like(s_topi, E))], -1)  # [n, T, 2k]
            q_live_all = torch.cat([q_l, torch.zeros_like(s_q)], 2)
            q_val_all = torch.cat([q_src_live, s_q], 2)
            c_live_all = torch.cat([c_l, torch.zeros_like(s_c)], 2)
            c_full_all = torch.cat([c_src_live, s_c], 2)
            valid = torch.cat([torch.ones_like(s_only), s_only], -1)
        else:
            exp_all = topi
            q_live_all, c_live_all = q_l, c_l
            valid = torch.ones_like(topi, dtype=torch.bool)
            if ov.value == "zero":
                q_val_all = torch.zeros_like(q_l)
                c_full_all = torch.zeros_like(c_l)
            elif ov.value == "mean":
                q_val_all = ov.mean_q[ar_t, topi]                   # [n, T, k, I]
                c_full_all = None                                   # computed per pair below (q-style)
            else:
                raise ValueError(ov.value)
        e_safe = exp_all.clamp_max(E - 1)
        sel = S[ar_n, ar_t, e_safe]                                  # [n, T, m, I]
        s_all = sel.all(-1) & valid
        s_any = sel.any(-1) & valid
        partial = s_any & ~s_all
        if c_full_all is None:  # mean: complete replacement also needs W_down q_mean
            part_or_full = s_any
        else:
            part_or_full = partial
        contrib = torch.where(s_all[..., None], c_full_all if c_full_all is not None else c_live_all, c_live_all)
        contrib = torch.where(valid[..., None], contrib, torch.zeros_like(contrib))
        idx = torch.nonzero(part_or_full)  # [p, 3]
        if idx.numel():
            qf = torch.where(sel[part_or_full], q_val_all[part_or_full], q_live_all[part_or_full])
            ee = exp_all[part_or_full]
            outp = torch.empty(qf.shape[0], H, dtype=o.dtype, device=o.device)
            for e in torch.unique(ee).tolist():
                m = ee == e
                outp[m] = F.linear(qf[m].to(o.dtype), w.down[e])
            contrib[idx[:, 0], idx[:, 1], idx[:, 2]] = outp
        # accumulate in ascending expert order, starting from zeros (as the HF loop's index_add_)
        order = torch.argsort(torch.where(valid, exp_all, torch.full_like(exp_all, E + 1)), dim=-1, stable=True)
        contrib = torch.gather(contrib, 2, order[..., None].expand(-1, -1, -1, H))
        flat = contrib.reshape(n * T, -1, H)
        acc = torch.zeros(n * T, H, dtype=o.dtype, device=o.device)
        tok = torch.arange(n * T, device=o.device)
        for s in range(flat.shape[1]):
            acc.index_add_(0, tok, flat[:, s])
        o = o.clone()
        o[rows] = acc.view(n, T, H)
        return o

    # ---- native HF reference (verification only) -----------------------------------------------------------------
    def native(self):
        if self._native is None:
            with torch.device("meta"):
                lay = self._mods["layer"](self.hfcfg, layer_idx=0)
                norm = self._mods["norm"](self.H, eps=self.hfcfg.rms_norm_eps)
            lay = lay.to_empty(device=self.dev).eval()
            norm = norm.to_empty(device=self.dev).eval()
            norm.load_state_dict({"weight": self.final_norm}, strict=True, assign=True)
            self._native = dict(layer=lay, norm=norm)
        return self._native

    def native_load(self, w):
        lay = self.native()["layer"]
        sd = {"input_layernorm.weight": w.ln1, "post_attention_layernorm.weight": w.ln2,
              "self_attn.q_proj.weight": w.wq, "self_attn.k_proj.weight": w.wk, "self_attn.v_proj.weight": w.wv,
              "self_attn.o_proj.weight": w.wo, "mlp.gate.weight": w.router,
              "mlp.experts.gate_up_proj": w.gate_up, "mlp.experts.down_proj": w.down}
        if w.q_norm is not None:
            sd.update({"self_attn.q_norm.weight": w.q_norm, "self_attn.k_norm.weight": w.k_norm})
        lay.load_state_dict(sd, strict=True, assign=True)
        return lay

    def native_mask(self, h):
        from transformers.masking_utils import create_causal_mask
        pos = torch.arange(h.shape[1], device=self.dev)[None]
        return create_causal_mask(config=self.hfcfg, inputs_embeds=h, attention_mask=None, past_key_values=None,
                                  position_ids=pos)

    def native_layer(self, lay, h, cos, sin, mask, capture: dict, out_patch=None):
        """One native HF decoder layer with the hooks of relp_sva.native_capture (q rebuilt from the experts' inputs)."""
        handles = []

        def experts_pre(mod, args):
            x, indices, weights = args
            n = x.shape[0]
            q = x.new_zeros(n, self.E, self.I)
            for e in range(self.E):
                row, slot = torch.where(indices == e)
                if row.numel():
                    g, u = F.linear(x[row], mod.gate_up_proj[e]).chunk(2, -1)
                    # = relp_sva.native_capture; explicit cast because Mixtral's routing weights are fp32
                    q[row, e] = (F.silu(g) * u * weights[row, slot, None]).to(q.dtype)
            capture["q"] = q.view(h.shape[0], h.shape[1], self.E, self.I)
            capture["route"] = (indices.detach(), weights.detach())

        def out_hook(mod, args, out):
            capture["out_raw"] = out.detach()
            if out_patch is not None:
                mk, val = out_patch
                out = torch.where(mk[..., None], val.to(out), out)
            capture["out"] = out.detach()
            return out

        handles.append(lay.mlp.experts.register_forward_pre_hook(experts_pre))
        handles.append(lay.mlp.register_forward_hook(out_hook))
        try:
            pos = torch.arange(h.shape[1], device=self.dev)[None]
            return lay(h, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin))
        finally:
            for hd in handles:
                hd.remove()

    # ---- dense helpers -------------------------------------------------------------------------------------------
    def dense_q(self, rec: LayerOut, B: int, T: int) -> torch.Tensor:
        """[B, T, E, I] q (zeros for unrouted experts) from the routed pairs."""
        q = torch.zeros(B * T, self.E, self.I, dtype=self.dtype, device=self.dev)
        q.scatter_(1, rec.topi[..., None].expand(-1, -1, self.I), rec.q)
        return q.view(B, T, self.E, self.I)


# ======================================================================================================================
# passes
# ======================================================================================================================
@dataclass
class Group:
    """Rows that go through every layer together (one batch; GEMM shapes = this group's)."""
    name: str
    ids: torch.Tensor                       # [B, T] long
    target: list
    foil: list
    native: bool = False
    relp: bool = False
    pairs: bool = False                     # keep LayerOut.q / .c per layer (in `records`)
    keep: Optional[set] = None              # layers whose LayerOut is kept (None = all, if pairs or record_route)
    record_route: bool = False
    record_out: bool = False
    record_h_in: bool = False               # residual entering each layer (for the RelP backward)
    record_dense_q: bool = False            # dense q per layer on the CPU (verification; small groups only)
    override_fn: Optional[Callable] = None  # (layer, w, state) -> Override | [Override, ...] | None
    out_patch_fn: Optional[Callable] = None  # (layer, state) -> (mask [B, T], values [B, T, H]) | None
    # results
    h: Optional[torch.Tensor] = None
    records: dict = field(default_factory=dict)
    routes: list = field(default_factory=list)
    outs: dict = field(default_factory=dict)
    h_in: list = field(default_factory=list)
    dense_q: dict = field(default_factory=dict)
    delta: Optional[torch.Tensor] = None
    residual_final: Optional[torch.Tensor] = None


def forward_pass(m: StreamRelP, groups: list[Group], on_layer: Optional[Callable] = None):
    """One streamed forward over all groups (in list order within each layer). No autograd."""
    state = {"groups": {g.name: g for g in groups}, "layer_out": {}}
    with torch.no_grad():
        for g in groups:
            ids = g.ids.to(m.dev)
            g.h = F.embedding(ids, m.embed)
            B, T = ids.shape
            g._cos, g._sin = m.rope(T, g.h)
            g._mask = m.native_mask(g.h) if g.native else m.causal_mask(B, T, g.h.dtype)
        for l, w in m.stream():
            state["layer_out"] = {}
            lay = m.native_load(w) if any(g.native for g in groups) else None
            for g in groups:
                B, T = g.ids.shape
                if g.record_h_in:
                    g.h_in.append(g.h)
                if g.native:
                    cap = {}
                    op = g.out_patch_fn(l, state) if g.out_patch_fn else None
                    g.h = m.native_layer(lay, g.h, g._cos, g._sin, g._mask, cap, op)
                    rec = LayerOut(topi=cap["route"][0], topw=cap["route"][1], out=cap["out"].view(B, T, -1))
                    if g.record_dense_q:
                        g.dense_q[l] = cap["q"].cpu()
                else:
                    ov = g.override_fn(l, w, state) if g.override_fn else None
                    op = g.out_patch_fn(l, state) if g.out_patch_fn else None
                    g.h, rec = m.layer(g.h, w, g._cos, g._sin, g._mask, relp=g.relp,
                                       pairs=g.pairs or g.record_dense_q, override=ov, out_patch=op)
                    if g.record_dense_q:
                        g.dense_q[l] = m.dense_q(rec, B, T).cpu()
                state["layer_out"][g.name] = rec
                if g.record_route:
                    g.routes.append((rec.topi.view(B, T, -1).cpu(), rec.topw.view(B, T, -1).cpu()))
                if g.record_out:
                    g.outs[l] = rec.out
                if g.pairs and (g.keep is None or l in g.keep):
                    g.records[l] = rec
            if on_layer is not None:
                on_layer(l, w, state)
            del w, lay
        for g in groups:
            g.residual_final = g.h[:, -1]
            if g.native:
                normed = m.native()["norm"](g.h)
                t = torch.as_tensor(g.target, device=m.dev)
                f = torch.as_tensor(g.foil, device=m.dev)
                g.delta = (normed[:, -1].float() * (m.W_U[t].float() - m.W_U[f].float())).sum(-1)
            else:
                g.delta = m.readout(g.h, g.target, g.foil)
    return state


def relp_backward(m: StreamRelP, g: Group, n_a: int, pool: torch.Tensor, mode: str = "paired",
                  dense_beta: bool = False, beta_finite: bool = False):
    """RelP backward for a group whose forward recorded h_in. Rows [0, n_a) = A endpoints (coefficients taken there);
    in paired mode rows [n_a, 2 n_a) are the aligned B endpoints. pool: bool [n_a, T] (scored positions).

    Returns (score_sum float64 [L, T, E, I] summed over the A rows, support bool [L, T, E, I], betas | None):
      paired:   sum_i (q_A,i - q_B,i) * beta_A,i     unpaired: sum_i q_A,i * beta_A,i
    in fp32 per row (as relp_sva.score_pair), accumulated in float64. betas (dense_beta=True, small groups only): the full
    beta [n_a, T, E, I] per layer on the CPU. beta_finite=True: betas = bool [n_a], every beta of the row (all layers,
    positions, experts, neurons) finite.
    """
    B, T = g.ids.shape
    paired = mode == "paired"
    assert B == (2 * n_a if paired else n_a)
    E, I, H, k = m.E, m.I, m.H, m.k
    total = torch.zeros(m.L, T, E, I, dtype=torch.float64, device=m.dev)
    support = torch.zeros(m.L, T, E, I, dtype=torch.bool, device=m.dev)
    betas = {} if dense_beta else None
    finite = torch.ones(n_a, dtype=torch.bool, device=m.dev) if beta_finite else None
    pool = pool.to(m.dev)
    # lambda at the final residual (final norm with frozen denominator, fp32 readout), A rows only
    with torch.enable_grad():
        hL = g.h.detach().requires_grad_(True)
        d = m.readout(hL, g.target, g.foil, relp=True)
        lam, = torch.autograd.grad(d[:n_a].sum(), hL)
    ids = g.ids.to(m.dev)
    cos, sin = m.rope(T, g.h)
    mask = m.causal_mask(B, T, g.h.dtype)
    for l, w in m.stream(reverse=True):
        hin = g.h_in[l].detach().requires_grad_(True)
        with torch.enable_grad():
            hout, rec = m.layer(hin, w, cos, sin, mask, relp=True, pairs=True)
            lam_in, = torch.autograd.grad(hout, hin, grad_outputs=lam)
        with torch.no_grad():
            lam_o = lam[:n_a].reshape(n_a * T, H)          # coefficient of this layer's MoE output (A rows)
            topi = rec.topi.view(B * T, k)
            qp = rec.q.view(B * T, k, I)
            ia = topi[: n_a * T]
            qa = qp[: n_a * T]
            if paired:
                ib = topi[n_a * T:]
                qb = qp[n_a * T:]
            pos_of = torch.arange(n_a * T, device=m.dev) % T
            keep_pos = pool.reshape(-1)
            if dense_beta or beta_finite:
                bfull = torch.einsum("nh,ehi->nei", lam_o, w.down)  # [N, E, I] activation dtype
                if dense_beta:
                    betas[l] = bfull.view(n_a, T, E, I).float().cpu()
                if beta_finite:
                    finite &= torch.isfinite(bfull.view(n_a, -1)).all(-1)
                del bfull
            # union of experts routed at A or B, per token
            if paired:
                cand = torch.cat([ia, ib], 1)
            else:
                cand = ia
            for e in torch.unique(cand).tolist():
                in_a = ia == e
                tok_a = in_a.any(1)
                if paired:
                    in_b = ib == e
                    tok = (tok_a | in_b.any(1)) & keep_pos
                else:
                    tok = tok_a & keep_pos
                tix = torch.where(tok)[0]
                if tix.numel() == 0:
                    continue
                beta = F.linear(lam_o[tix], w.down[e].t())   # [m, I] = (W_down,e^T lambda) per token
                q_a = torch.zeros(tix.numel(), I, dtype=qa.dtype, device=m.dev)
                ra, sa = torch.where(in_a[tix])
                q_a[ra] = qa[tix[ra], sa]
                if paired:
                    q_b = torch.zeros_like(q_a)
                    rb, sb = torch.where(in_b[tix])
                    q_b[rb] = qb[tix[rb], sb]
                    sc = (q_a.float() - q_b.float()) * beta.float()
                    act = (q_a != 0) | (q_b != 0)
                else:
                    sc = q_a.float() * beta.float()
                    act = q_a != 0
                if not torch.isfinite(sc).all():
                    raise ValueError("Nonfinite attribution")
                tp = pos_of[tix]
                total[l, :, e].index_add_(0, tp, sc.double())
                sup = torch.zeros(T, I, dtype=torch.int32, device=m.dev)
                sup.index_add_(0, tp, act.int())
                support[l, :, e] |= sup > 0
        lam = lam_in.detach()
        del w, rec, hout, hin
    if beta_finite:
        betas = finite.cpu()
    return total, support, betas
