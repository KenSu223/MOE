"""ext14 circuit C2 (p6-circuit): RelP (Arora, Wu, Steinhardt, Schwettmann 2026; Rezaei Jafari et al. 2025) on the neuron
basis, re-implemented with autograd for a dense model (Mistral-7B) and ported to an MoE (OLMoE-1B-7B).

A functional forward over the HF model's own parameter tensors (one copy of the weights) with three switches:
  relp=True    replacement-model backward (the forward VALUES are unchanged):
               RMSNorm: denominator rsqrt(mean(x^2) + eps) detached (also OLMoE's q / k norms);
               attention: softmax pattern detached (gradient only through V);
               SiLU: SiLU(g) = g * sigmoid(g) with sigmoid(g) detached ("frozen SiLU gate", identity rule);
               gated product: a = 0.5 * (s * u.detach() + s.detach() * u) (half rule);
               MoE: top-k set AND routing weights detached (frozen routing), so q_e = w_e * a_e is the gated product of
               a constant weight and the half-rule product (the generalised half rule; w_e carries no relevance).
  relp=False   plain autograd (AtP gradients).
  route        None = live routing; or per layer (topi [N, k], topw [N, k]) to freeze the routing to a reference run.
Nodes = neurons per position: q[l] = the weighted intermediate activation that the down projection reads,
  dense  q[l] [B, T, I]          (a = SiLU(W_gate x) * (W_up x))
  MoE    q[l] [B, T, E, I]       (q_e = w_e a_e; w_e = 0 for experts not routed at that token, so a node of an expert
                                  that is routed in only one run of a pair has q = 0 in the other run; the gradient of
                                  m w.r.t. q_e exists for every expert because out = sum_e q_e W_down_e^T)
Interventions per layer: q := where(mask, values, q_live) (mask / values broadcastable to q[l]; MoE: a third element
"routed" restricts the mask to experts routed in the live run, used for conditional mean ablation); this expresses exact
patches of any neuron set at any position (values = the source run's q), MLP / MoE-output patches (all neurons of a
position), and mean ablation (values = per-neuron means).
Metric m = Delta = (W_U[true] - W_U[foil]) . final_norm(h_T-1), read out in fp32 from the bf16 normed residual.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .models import MODELS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class FModel:
    """Functional forward of a Mistral (dense) or OLMoE (MoE) checkpoint loaded with transformers."""

    def __init__(self, key: str, device: str = "cuda", dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM
        from .arch import snapshot_dir
        import os
        self.key = key
        src = key if os.path.isdir(key) else snapshot_dir(MODELS[key]["repo"])  # a directory = local test checkpoint
        self.model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, device_map=device, attn_implementation="eager")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.cfg = cfg
        self.dev, self.dtype = device, dtype
        self.L = cfg.num_hidden_layers
        self.H = cfg.hidden_size
        self.nH = cfg.num_attention_heads
        self.nKV = cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or self.H // self.nH
        self.eps = float(cfg.rms_norm_eps)
        self.moe = hasattr(cfg, "num_experts") and getattr(cfg, "num_experts", 0) and cfg.num_experts > 1
        self.E = int(cfg.num_experts) if self.moe else 1
        self.k = int(cfg.num_experts_per_tok) if self.moe else 1
        self.norm_topk = bool(getattr(cfg, "norm_topk_prob", False))
        self.I = int(cfg.intermediate_size)
        m = self.model.model
        self.embed = m.embed_tokens.weight
        self.final_norm = m.norm.weight
        self.W_U = self.model.lm_head.weight
        self.rotary = m.rotary_emb
        self.layers = []
        for lay in m.layers:
            at = lay.self_attn
            d = dict(ln1=lay.input_layernorm.weight, ln2=lay.post_attention_layernorm.weight, wq=at.q_proj.weight,
                     wk=at.k_proj.weight, wv=at.v_proj.weight, wo=at.o_proj.weight)
            if hasattr(at, "q_norm"):
                d.update(qn=at.q_norm.weight, kn=at.k_norm.weight)
            if self.moe:
                d.update(router=lay.mlp.gate.weight, gate_up=lay.mlp.experts.gate_up_proj, down=lay.mlp.experts.down_proj)
            else:
                d.update(wg=lay.mlp.gate_proj.weight, wu=lay.mlp.up_proj.weight, wd=lay.mlp.down_proj.weight)
            self.layers.append(d)
        self.clip_qkv = getattr(cfg, "clip_qkv", None)

    # ---- building blocks ----------------------------------------------------------------------------------------
    def rms(self, x, w, relp):
        xf = x if x.dtype == torch.float64 else x.float()  # fp64 only in the CPU gradient checks
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if relp:
            r = r.detach()
        return w * (xf * r).to(x.dtype)

    def W_down(self, l: int, e: int = 0) -> torch.Tensor:
        """[H, I] down projection of expert e (dense: the MLP)."""
        d = self.layers[l]
        return d["down"][e] if self.moe else d["wd"]

    def dla_dir(self, true_id: int, foil_id: int, h_final: torch.Tensor) -> torch.Tensor:
        u = self.final_norm.float() * (self.W_U[true_id].float() - self.W_U[foil_id].float())
        return u / torch.sqrt(h_final.float().pow(2).mean() + self.eps)

    # ---- forward ------------------------------------------------------------------------------------------------
    def forward(self, ids: torch.Tensor, true_ids, foil_ids, relp: bool = False, record: bool = False,
                interv: Optional[dict] = None, route: Optional[list] = None, grad: bool = False,
                out_patch: Optional[dict] = None) -> dict:
        """ids [B, T] (equal lengths). interv: {layer: (mask, values)} on q[l]. out_patch: {layer: (mask_bt [B, T] bool,
        values [B, T, H])} replaces the MLP / MoE output vector at the masked (row, position) (exact sublayer patch).
        Returns dict(delta fp32 [B], hN, q (list per layer, if record or grad), route (list of (topi, topw)),
        h_final [B, H], out (MLP outputs per layer if record))."""
        B, T = ids.shape
        dev = self.dev
        h = self.embed[ids]
        if grad:
            h = h.detach().requires_grad_(True)
        pos = torch.arange(T, device=dev)[None].expand(B, -1)
        cos, sin = self.rotary(h, pos)
        cos, sin = cos[:, None], sin[:, None]
        causal = torch.full((T, T), float("-inf"), device=dev).triu(1)
        qs, routes, outs, attns = [], [], [], []
        for l, d in enumerate(self.layers):
            x = self.rms(h, d["ln1"], relp)
            q = F.linear(x, d["wq"])
            k = F.linear(x, d["wk"])
            v = F.linear(x, d["wv"])
            if "qn" in d:  # OLMoE: RMSNorm over the full projection
                q = self.rms(q, d["qn"], relp)
                k = self.rms(k, d["kn"], relp)
            if self.clip_qkv is not None:
                q, k, v = (t.clamp(-self.clip_qkv, self.clip_qkv) for t in (q, k, v))
            q = q.view(B, T, self.nH, self.D).transpose(1, 2)
            k = k.view(B, T, self.nKV, self.D).transpose(1, 2)
            v = v.view(B, T, self.nKV, self.D).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            rep = self.nH // self.nKV
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            s = torch.matmul(q, k.transpose(2, 3)) * (self.D ** -0.5) + causal
            p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
            if relp:
                p = p.detach()
            a = torch.matmul(p, v).transpose(1, 2).reshape(B, T, self.nH * self.D)
            a = F.linear(a, d["wo"])
            h = h + a
            x2 = self.rms(h, d["ln2"], relp)
            if self.moe:
                xf = x2.reshape(B * T, self.H)
                if route is None:
                    lg = F.linear(xf, d["router"])
                    pr = torch.softmax(lg, dim=-1, dtype=torch.float32)
                    tv, ti = torch.topk(pr, self.k, dim=-1)
                    if self.norm_topk:
                        tv = tv / tv.sum(-1, keepdim=True)
                    tw = tv.to(lg.dtype)
                else:
                    ti, tw = route[l]
                routes.append((ti.detach(), tw.detach()))
                w = torch.zeros(B * T, self.E, dtype=x2.dtype, device=dev).scatter(1, ti, tw)
                w = w.detach()  # frozen routing weights in every backward (AtP / RelP alike); the router gets no gradient
                gu = torch.einsum("nh,eih->nei", xf, d["gate_up"])
                g, u = gu.chunk(2, dim=-1)
                act = self._act(g, u, relp)
                qn = (w[..., None] * act).view(B, T, self.E, self.I)
            else:
                g = F.linear(x2, d["wg"])
                u = F.linear(x2, d["wu"])
                qn = self._act(g, u, relp)
            if interv is not None and l in interv:
                mk, val = interv[l][:2]
                if self.moe and len(interv[l]) > 2 and interv[l][2] == "routed":
                    mk = mk & (w.view(B, T, self.E, 1) > 0)  # only experts routed in THIS (live) run are replaced
                qn = torch.where(mk, val.to(qn.dtype), qn)
            qs.append(qn)
            if self.moe:
                o = torch.einsum("btei,ehi->bth", qn, d["down"])
            else:
                o = F.linear(qn, d["wd"])
            if out_patch is not None and l in out_patch:
                mk, val = out_patch[l]
                o = torch.where(mk[..., None], val.to(o.dtype), o)
            if record:
                outs.append(o.detach())
                attns.append(a.detach())
            h = h + o
        hf = h[:, -1]
        hN = self.rms(hf, self.final_norm, relp)
        tid = torch.as_tensor(true_ids, device=dev)
        fid = torch.as_tensor(foil_ids, device=dev)
        ct = torch.float64 if hN.dtype == torch.float64 else torch.float32
        u = (self.W_U[tid].to(ct) - self.W_U[fid].to(ct))  # [B, H]
        delta = (hN.to(ct) * u).sum(-1)
        return {"delta": delta, "q": qs if (record or grad) else None, "route": routes, "h_final": hf,
                "out": outs if record else None, "attn": attns if record else None}

    def _act(self, g, u, relp):
        if relp:
            sg = torch.sigmoid(g).detach()
            s = g * sg
            s = s + (F.silu(g) - s).detach()  # forward value exactly F.silu(g) (as HF), gradient sigmoid(g) (frozen gate)
            return 0.5 * (s * u.detach() + s.detach() * u)
        return F.silu(g) * u

    # ---- gradients ----------------------------------------------------------------------------------------------
    def node_grads(self, ids: torch.Tensor, true_ids, foil_ids, relp: bool, route=None) -> tuple[torch.Tensor, list]:
        """d delta / d q[l] for every layer (list of tensors shaped like q[l]) and delta [B]; one backward."""
        with torch.enable_grad():
            r = self.forward(ids, true_ids, foil_ids, relp=relp, grad=True, route=route)
            gs = torch.autograd.grad(r["delta"].sum(), r["q"])
        return r["delta"].detach(), [g.detach() for g in gs]
