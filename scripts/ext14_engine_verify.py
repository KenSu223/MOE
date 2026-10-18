"""ext14 (Phase 6, p6-engine) engine verification against transformers hooks.

Modes
-----
full (default): the transformers model is loaded completely (on the GPU, or with --offload on GPU + CPU RAM) with eager
    attention, every configuration is computed with forward hooks, the model is freed, then the engine runs the same
    configurations in one pass. Units: STR pairs (WinoGrande train_xl twins re-tokenised with the model's own tokenizer,
    both directions) for checkpoints, random token prompts differing at one position for the tiny random models.
    Configurations per unit (denoising: parent = corrupted run, source = clean run; a subset also noising):
      layer / attn_layer / block / resid at several layers, attn_head (single heads), expert (dense: expert 0; Qwen2-MoE:
      a clean-routed expert and the shared expert E), coalition_set (Qwen2-MoE), expert_neurons (random half, all),
      multi (attention + MLP steps, head steps, heads_experts, all-layer MLP), zero; head masks (zero and mean mode) in
      the clean and corrupted rows followed by layer / attn_layer / attn_head spawns.
    HF side: forward hooks replace the final position's self_attn / mlp / decoder-layer outputs by the source run's,
    o_proj pre-hooks replace per-head slices (head steps) or set heads at every position (head masks), and mlp hooks add
    sum_e (c_e(source) - c_e(own)) for expert sets or sum_{i in N} (w_s a_i,s - w_o a_i,o) W_down[:, i] for neurons.
    Compared: Delta of every row (max / mean |dDelta|, rescue correlation), metrics (log p, p, rank, KL), diagnostics
    (attn_heads_mean, expert_neurons, resid_layers), engine identities (expert_neurons all == expert, linearity of the
    neuron vectors, expert 0 == layer for dense, coalition_set(all active + shared) == layer for Qwen2-MoE, attn_head on a
    masked head == parent, zero == parent).
per-layer (--per-layer l1,l2,...): only the engine runs the model; one transformers decoder layer at a time is built
    from the checkpoint and applied to the engine's own residual stream entering that layer (DiagSpec.resid_layers);
    compared with the engine's residual after the layer (all positions) and the routing at every position. For models
    that do not fit on the GPU (Qwen2-57B-A14B, Qwen3-14B) without any disk offload.

Usage:
  python scripts/ext14_engine_verify.py --model mistral7b [--units 6] [--out results/verify_ext14_mistral7b.json]
  python scripts/ext14_engine_verify.py --model /opt/dlami/nvme/p6_tiny/qwen2_moe --dtype fp32
  python scripts/ext14_engine_verify.py --model qwen2_57b --per-layer 1,13,27 --units 3
"""
import argparse, gc, json, os, random, sys, time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "moetrace" not in sys.modules:
    sys.path.insert(0, TREE)
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from moetrace.arch import load_spec, snapshot_dir
from moetrace.engine import Engine, PrefillSpec, SpawnSpec, DiagSpec

DT = {"bf16": torch.bfloat16, "fp32": torch.float32}
ROUTER32 = None  # --router-fp32: transformers Qwen2-MoE router patched to fp32 logits, engines built with router_fp32=True


def patch_router_fp32():
    """transformers Qwen2MoeTopKRouter with fp32 logits (input and weight upcast), as the engine's router_fp32 protocol."""
    from transformers.models.qwen2_moe import modeling_qwen2_moe as QM

    def fwd(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        router_probs = torch.softmax(router_logits, dtype=torch.float, dim=-1)
        top_v, top_i = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_v = top_v / top_v.sum(dim=-1, keepdim=True)
        return router_logits, top_v.to(hidden_states.dtype), top_i
    QM.Qwen2MoeTopKRouter.forward = fwd


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _stats(z):
    z = np.abs(np.asarray(z, dtype=np.float64))
    if len(z) == 0:
        return {"n": 0}
    return {"n": int(len(z)), "maxabs": float(z.max()), "meanabs": float(z.mean()), "frac_exact": float((z == 0).mean())}


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None


def resolve(model):
    if os.path.isdir(model):
        return model, os.path.basename(model.rstrip("/")), True
    from moetrace.models import MODELS
    return MODELS[model]["repo"], model, False


# ------------------------------------------------------------------------------------------------------------------
# units
# ------------------------------------------------------------------------------------------------------------------
def tiny_units(spec, n, seed=14):
    rng = random.Random(seed)
    out = []
    for u in range(n):
        T = rng.randint(9, 20)
        ids = [rng.randrange(1, spec.vocab) for _ in range(T)]
        cor = list(ids)
        p = rng.randrange(1, T - 1)
        cor[p] = (ids[p] + rng.randrange(1, spec.vocab - 1)) % spec.vocab
        t, f = rng.sample(range(spec.vocab), 2)
        out.append(dict(name=f"t{u}", clean=ids, corrupt=cor, true=t, foil=f, pos=p))
    return out


def wino_units(repo, family, n_pairs, seed=14, pairs_file=None):
    """WinoGrande train_xl STR twins of the tokenizer family (or any STR pair parquet with prompt_a / prompt_b / ids_a /
    ids_b / trig_a / trig_b), kept only if the model's own tokenizer reproduces the ids."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_dir(repo))
    f = {"mistral": "pairs_train_xl_mixtral_bos.parquet", "olmoe": "pairs_train_xl_olmoe.parquet"}.get(family, "pairs_train_xl_qwen3.parquet")
    f = pairs_file or os.path.join("/home/ubuntu/MOE/data/wino_str", f)
    p = pd.read_parquet(f).sort_values("pair_id").reset_index(drop=True)
    idx = list(range(len(p)))
    random.Random(seed).shuffle(idx)
    out, n_checked, n_same = [], 0, 0
    for i in idx:
        r = p.iloc[i]
        a, b = json.loads(r.ids_a), json.loads(r.ids_b)
        n_checked += 1
        ta, tb = tok(r.prompt_a)["input_ids"], tok(r.prompt_b)["input_ids"]
        if ta != a or tb != b:
            continue
        n_same += 1
        out.append(dict(name=f"w{i}a", clean=a, corrupt=b, true=int(r.trig_a), foil=int(r.trig_b)))
        out.append(dict(name=f"w{i}b", clean=b, corrupt=a, true=int(r.trig_b), foil=int(r.trig_a)))
        if len(out) >= 2 * n_pairs:
            break
    return out, {"pairs_checked": n_checked, "pairs_tokenizer_identical": n_same, "file": f}


# ------------------------------------------------------------------------------------------------------------------
# configuration plan (shared by HF and engine)
# ------------------------------------------------------------------------------------------------------------------
def pick_layers(L):
    return sorted(set([0, max(1, L // 4), L // 2, (3 * L) // 4, L - 1]))


def make_plan(spec, units, light=False, seed=14):
    """Returns rows (prefill row specs) and configs. Row: (unit, which) with which in C, X, CMz, XMz, CMm, XMm.
    Config: dict(tag, unit, parent, src, eng=(kind, layer, expert, experts, steps, neurons), hf=[steps])."""
    L, nH, E = spec.n_layers, spec.n_heads, spec.n_experts
    I = spec.inter
    rng = random.Random(seed)
    Ls = pick_layers(L) if (not light or L < 8) else sorted(set([1, L // 3, (2 * L) // 3, L - 2]))
    configs = []

    def add(tag, u, parent, src, eng, hf):
        configs.append(dict(tag=tag, unit=u, parent=parent, src=src, eng=eng, hf=hf))

    for u in range(len(units)):
        for d, (par, src) in (("den", ("X", "C")), ("noi", ("C", "X"))):
            if d == "noi" and light:
                continue
            lay = Ls if d == "den" else Ls[1:3]
            for l in lay:
                add(f"{d}_layer_{l}", u, par, src, ("layer", l), [("mlp", l)])
                add(f"{d}_attn_{l}", u, par, src, ("attn_layer", l), [("attn", l)])
                if not light or d == "den":
                    add(f"{d}_block_{l}", u, par, src, ("block", l), [("attn", l), ("mlp", l)])
                    add(f"{d}_resid_{l}", u, par, src, ("resid", l), [("resid", l)])
            if d == "noi":
                continue
            for l in Ls[1:4]:
                h = rng.randrange(nH)
                add(f"{d}_head_{l}_{h}", u, par, src, ("attn_head", l, h), [("heads", l, (h,))])
            # experts / neurons (dense: expert 0 = the MLP)
            le = Ls[len(Ls) // 2]
            if spec.is_dense:
                add(f"{d}_expert0_{le}", u, par, src, ("expert", le, 0), [("experts", le, (0,))])
                exps = [0]
            else:
                exps = ["active", E] if spec.shared_inter else ["active"]
                for e in exps:
                    tag_e = "shared" if e == E else "act"
                    add(f"{d}_expert_{tag_e}_{le}", u, par, src, ("expert", le, e), [("experts", le, (e,))])
                add(f"{d}_coalall_{le}", u, par, src, ("coalition_set", le, "all"), [("experts", le, "all")])
                add(f"{d}_coal2_{le}", u, par, src, ("coalition_set", le, "rand2"), [("experts", le, "rand2")])
            for e in (exps if not light else []):  # light (offloaded HF): no neuron configs (offloaded weights)
                tag_e = "shared" if e == E else ("act" if e == "active" else "0")
                Ie = spec.shared_inter if e == E else I
                half = tuple(sorted(rng.sample(range(Ie), Ie // 2)))
                add(f"{d}_neur_half_{tag_e}_{le}", u, par, src, ("expert_neurons", le, e, half), [("neurons", le, e, half)])
                add(f"{d}_neur_all_{tag_e}_{le}", u, par, src, ("expert_neurons", le, e, tuple(range(Ie))),
                    [("neurons", le, e, tuple(range(Ie)))])
            # multi
            l1, l2, l3 = Ls[1], Ls[2], Ls[-1]
            add(f"{d}_multi_attn_mlp", u, par, src, ("multi", [(l1, "attn_layer", ()), (l2, "layer", ())]), [("attn", l1), ("mlp", l2)])
            add(f"{d}_multi_mlp_block_attn", u, par, src, ("multi", [(l1, "layer", ()), (l2, "block", ()), (l3, "attn_layer", ())]),
                [("mlp", l1), ("attn", l2), ("mlp", l2), ("attn", l3)])
            h1, h2, h3 = rng.sample(range(nH), 3)
            e_he = 0 if spec.is_dense else "active"
            add(f"{d}_multi_heads", u, par, src,
                ("multi", [(l1, "attn_head", (h1, h2)), (l3, "heads_experts", ((h3,), (e_he,)))]),
                [("heads", l1, (h1, h2)), ("heads", l3, (h3,)), ("experts", l3, (e_he,))])
            if not light:
                add(f"{d}_multi_allmlp", u, par, src, ("multi", [(l, "layer", ()) for l in range(L)]), [("mlp", l) for l in range(L)])
            if spec.shared_inter:  # ext8-style add-back steps with the shared expert (coalition_set / expert steps containing E)
                add(f"{d}_multi_shared", u, par, src,
                    ("multi", [(l1, "coalition_set", ("active", E)), (l2, "coalition_set", (E,)), (l3, "expert", E)]),
                    [("experts", l1, ("active", E)), ("experts", l2, (E,)), ("experts", l3, (E,))])
            add(f"{d}_zero_{l1}", u, par, src, ("zero", l1), [])
        # head masks: zero and mean mode, rows CM*/XM*
        for mode in ("z", "m"):
            par, src = "XM" + mode, "CM" + mode
            lm = Ls[1]
            add(f"hm{mode}_none", u, par, src, ("zero", lm), [])
            add(f"hm{mode}_layer_{lm}", u, par, src, ("layer", lm), [("mlp", lm)])
            add(f"hm{mode}_attn_{Ls[2]}", u, par, src, ("attn_layer", Ls[2]), [("attn", Ls[2])])
            add(f"hm{mode}_headmasked", u, par, src, ("attn_head", "mask0"), [("heads", "mask0")])
            add(f"hm{mode}_headfree", u, par, src, ("attn_head", "free"), [("heads", "free")])
            add(f"hm{mode}_multi", u, par, src, ("multi", [(Ls[1], "attn_layer", ()), (Ls[2], "layer", ()), (Ls[-1], "attn_layer", ())]),
                [("attn", Ls[1]), ("mlp", Ls[2]), ("attn", Ls[-1])])
        # route masks (MoE only) with route_mask_inherit: rows CR* / XR*, reroute (r) and zero (z) mode, all positions
        if not spec.is_dense:
            for mode in ("r", "z"):
                par, src = "XR" + mode, "CR" + mode
                add(f"rm{mode}_none", u, par, src, ("zero", Ls[1]), [])
                add(f"rm{mode}_layer_{Ls[1]}", u, par, src, ("layer", Ls[1]), [("mlp", Ls[1])])
                add(f"rm{mode}_layer_{Ls[2]}", u, par, src, ("layer", Ls[2]), [("mlp", Ls[2])])
                add(f"rm{mode}_attn_{Ls[1]}", u, par, src, ("attn_layer", Ls[1]), [("attn", Ls[1])])
                add(f"rm{mode}_attn_{Ls[0]}", u, par, src, ("attn_layer", Ls[0]), [("attn", Ls[0])])
                add(f"rm{mode}_multi", u, par, src, ("multi", [(Ls[0], "layer", ()), (Ls[1], "attn_layer", ()), (Ls[2], "block", ())]),
                    [("mlp", Ls[0]), ("attn", Ls[1]), ("attn", Ls[2]), ("mlp", Ls[2])])
                add(f"rm{mode}_allmlp", u, par, src, ("multi", [(l, "layer", ()) for l in range(L)]), [("mlp", l) for l in range(L)])
        # Qwen2-MoE: shared expert knocked out (route_mask entry E, zero mode) at Ls[1] and Ls[2] + one routed expert at Ls[1]
        if spec.shared_inter:
            par, src = "XRs", "CRs"
            add("rms_none", u, par, src, ("zero", Ls[1]), [])
            add(f"rms_layer_{Ls[1]}", u, par, src, ("layer", Ls[1]), [("mlp", Ls[1])])
            add(f"rms_layer_{Ls[2]}", u, par, src, ("layer", Ls[2]), [("mlp", Ls[2])])
            add(f"rms_attn_{Ls[1]}", u, par, src, ("attn_layer", Ls[1]), [("attn", Ls[1])])
            add(f"rms_attn_{Ls[0]}", u, par, src, ("attn_layer", Ls[0]), [("attn", Ls[0])])
            add("rms_multi", u, par, src, ("multi", [(Ls[0], "layer", ()), (Ls[1], "attn_layer", ()), (Ls[2], "block", ())]),
                [("mlp", Ls[0]), ("attn", Ls[1]), ("attn", Ls[2]), ("mlp", Ls[2])])
            add("rms_allmlp", u, par, src, ("multi", [(l, "layer", ()) for l in range(L)]), [("mlp", l) for l in range(L)])
            add(f"rms_sharedmasked_{Ls[1]}", u, par, src, ("expert", Ls[1], E), [("experts", Ls[1], (E,))])
            add(f"rms_sharedfree_{Ls[-1]}", u, par, src, ("expert", Ls[-1], E), [("experts", Ls[-1], (E,))])
    return configs


def head_mask_of(spec, u, seed=14):
    """per unit: 3 (layer, head) pairs at two layers"""
    rng = random.Random(seed * 1000 + u)
    Ls = pick_layers(spec.n_layers)
    la, lb = Ls[1], Ls[2] if len(Ls) > 2 else Ls[1]
    ha = rng.randrange(spec.n_heads)
    hb = rng.sample(range(spec.n_heads), 2)
    return [(la, ha), (lb, hb[0]), (lb, hb[1])]


# ------------------------------------------------------------------------------------------------------------------
# HF side
# ------------------------------------------------------------------------------------------------------------------
def _o(out):
    return out[0] if isinstance(out, tuple) else out


def _rep(out, new_final):
    o = _o(out).clone()
    o[0, -1] = new_final.to(o.dtype)
    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o


class HFRef:
    def __init__(self, path, dtype, offload=False, device="cuda"):
        from transformers import AutoModelForCausalLM
        kw = dict(dtype=dtype, attn_implementation="eager")
        if offload:
            kw.update(device_map="auto", max_memory={0: os.environ.get("EXT14_GPU_MEM", "17GiB"), "cpu": "48GiB"})
        else:
            kw.update(device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
        cfg = self.model.config
        self.L, self.nH = cfg.num_hidden_layers, cfg.num_attention_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nH
        self.dense = not hasattr(cfg, "num_experts")
        self.E = getattr(cfg, "num_experts", 1)
        self.dtype = dtype
        self.offload = offload
        self._shz = set()  # layers whose shared expert is zeroed in the current forward (route mask entry E)
        assert not (offload and not self.dense), "offloaded reference only for dense models"

    def layer(self, l):
        return self.model.model.layers[l]

    def contribs(self, l, x):
        """{expert: fp32 [H]} for one token x [1, H] (engine numerics: activation-dtype expert MLP, fp32 weight);
        also {expert: (a [I] fp32 of activation-dtype values, w)} activations. Shared expert = index E."""
        mlp = self.layer(l).mlp
        out, acts = {}, {}
        if self.dense:
            g, u = mlp.gate_proj(x), mlp.up_proj(x)
            a = mlp.act_fn(g) * u
            y = mlp.down_proj(a)
            out[0] = y[0].float()
            acts[0] = (a[0].float(), 1.0, True)
            return out, acts
        _, wts, idx = mlp.gate(x)
        ex = mlp.experts
        for s_ in range(idx.shape[1]):
            e = int(idx[0, s_])
            g, u = F.linear(x, ex.gate_up_proj[e]).chunk(2, dim=-1)
            a = ex.act_fn(g) * u
            y = F.linear(a, ex.down_proj[e])
            out[e] = y[0].float() * wts[0, s_].float()
            acts[e] = (a[0].float(), float(wts[0, s_].float()), True)
        if hasattr(mlp, "shared_expert"):
            se = mlp.shared_expert
            a = se.act_fn(se.gate_proj(x)) * se.up_proj(x)
            y = se.down_proj(a)
            gw = torch.sigmoid(mlp.shared_expert_gate(x))
            wz = 0.0 if l in self._shz else 1.0
            out[self.E] = y[0].float() * gw[0, 0].float() * wz
            acts[self.E] = (a[0].float(), float(gw[0, 0].float()) * wz, True)
        return out, acts

    def expert_act(self, l, e, x):
        """activations of expert e for x [1, H] (hypothetical if not routed)."""
        mlp = self.layer(l).mlp
        if self.dense:
            return (mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))[0].float()
        if e == self.E:
            se = mlp.shared_expert
            return (se.act_fn(se.gate_proj(x)) * se.up_proj(x))[0].float()
        g, u = F.linear(x, mlp.experts.gate_up_proj[e]).chunk(2, dim=-1)
        return (mlp.experts.act_fn(g) * u)[0].float()

    def down(self, l, e):
        mlp = self.layer(l).mlp
        if self.dense:
            return mlp.down_proj.weight.float()
        if e == self.E:
            return mlp.shared_expert.down_proj.weight.float()
        return mlp.experts.down_proj[e].float()

    def mask_hooks(self, mask):
        """mask: list of (layer, head, vec or None); o_proj pre-hooks setting the heads at every position."""
        by = {}
        for l, h, vec in mask:
            by.setdefault(l, []).append((h, vec))
        hooks = []
        for l, hv in by.items():
            def pre(mod, args, hv=hv):
                x = args[0].clone()
                for h, vec in hv:
                    x[0, :, h * self.D:(h + 1) * self.D] = 0 if vec is None else vec.to(x.device, x.dtype)
                return (x,) + tuple(args[1:])
            hooks.append((self.layer(l).self_attn.o_proj, "pre", pre))
        return hooks

    def route_hooks(self, rmask, mode):
        """route masks at every position: reroute = softmax with the masked experts at -inf, top-k, the router's own
        renormalisation rule, cast to the logits dtype; zero = the router's own (scores, indices) with the masked slots' scores
        set to 0. Applied to every call of the gate (also the one-token calls inside contribs)."""
        by = {}
        for l, e in rmask or []:
            by.setdefault(int(l), set()).add(int(e))
        hooks = []
        for l, es in by.items():
            if self.E in es and hasattr(self.layer(l).mlp, "shared_expert"):  # shared expert: zero its output
                assert mode == "zero"

                def hz(mod, args, out, l=l):
                    self._shz.add(l)
                    return torch.zeros_like(out)
                hooks.append((self.layer(l).mlp.shared_expert, "fwd", hz))
                es = es - {self.E}
                if not es:
                    continue
            gate = self.layer(l).mlp.gate
            ex = sorted(es)

            def hk(mod, args, out, ex=ex):
                logits, scores, idx = out
                exl = torch.tensor(ex, device=logits.device)
                if mode == "reroute":
                    lf = logits.float().clone()
                    lf[:, exl] = float("-inf")
                    probs = torch.softmax(lf, dim=-1)
                    tv, ti = torch.topk(probs, mod.top_k, dim=-1)
                    if mod.norm_topk_prob:
                        tv = tv / tv.sum(dim=-1, keepdim=True)
                    return logits, tv.to(logits.dtype), ti
                sc = scores.clone()
                sc[torch.isin(idx, exl)] = 0
                return logits, sc, idx
            hooks.append((gate, "fwd", hk))
        return hooks

    def fwd(self, ids, hooks, hidden=False):
        hs = []
        try:
            for mod, kind, fn in hooks:
                hs.append(mod.register_forward_pre_hook(fn) if kind == "pre" else mod.register_forward_hook(fn))
            out = self.model(input_ids=torch.tensor([ids], device=self.model.device), output_hidden_states=hidden)
            lg = out.logits[0, -1].float()
            return (lg, out.hidden_states) if hidden else lg
        finally:
            for h in hs:
                h.remove()
            self._shz.clear()

    @torch.no_grad()
    def record(self, ids, mask, want_hidden=False, act_experts=None, extra_hooks=()):
        """final-position attention out, o_proj input, mlp in / out, layer out; contributions; o_proj input mean."""
        st, hooks = {}, list(self.mask_hooks(mask)) + list(extra_hooks)
        for l in range(self.L):
            lay = self.layer(l)

            def ra(mod, args, out, l=l):
                st[("attn", l)] = _o(out)[0, -1].detach().clone()

            def ro(mod, args, l=l):
                st[("oin", l)] = args[0][0, -1].detach().clone()
                st[("oin_mean", l)] = args[0][0].float().mean(0).detach().clone()

            def rm(mod, args, out, l=l):
                x = args[0][0, -1:].detach().clone()
                st[("mlp", l)] = _o(out)[0, -1].detach().clone()
                st[("x", l)] = x
                if self.offload:  # dense + offloaded weights: contribution = the MLP output; no activations
                    st[("c", l)], st[("a", l)] = {0: st[("mlp", l)].float()}, {}
                else:
                    st[("c", l)], st[("a", l)] = self.contribs(l, x)

            def rl(mod, args, out, l=l):
                st[("resid", l)] = _o(out)[0, -1].detach().clone()
            hooks += [(lay.self_attn, "fwd", ra), (lay.self_attn.o_proj, "pre", ro), (lay.mlp, "fwd", rm), (lay, "fwd", rl)]
        if want_hidden:
            lg, hid = self.fwd(ids, hooks, hidden=True)
            st["hidden"] = [h[0].detach().cpu() for h in hid]
        else:
            lg = self.fwd(ids, hooks)
        st["logits"] = lg
        return st

    def patch_hooks(self, steps, src, mask_heads_spec):
        if self.offload:  # dense: expert 0 = the MLP output (no submodule calls on offloaded weights)
            steps = [("mlp", st[1]) if st[0] == "experts" else st for st in steps]
            assert all(st[0] != "neurons" for st in steps)
        hooks = []
        by_l = {}
        for st in steps:
            by_l.setdefault(st[1], []).append(st)
        for l, sts in by_l.items():
            lay = self.layer(l)
            kinds = [s_[0] for s_ in sts]
            if "attn" in kinds:
                def ha(mod, args, out, l=l):
                    return _rep(out, src[("attn", l)])
                hooks.append((lay.self_attn, "fwd", ha))
            heads = sorted(set(h for s_ in sts if s_[0] == "heads" for h in s_[2]))
            if heads:
                def hh(mod, args, l=l, heads=heads):
                    x = args[0].clone()
                    for h in heads:
                        x[0, -1, h * self.D:(h + 1) * self.D] = src[("oin", l)][h * self.D:(h + 1) * self.D]
                    return (x,) + tuple(args[1:])
                hooks.append((lay.self_attn.o_proj, "pre", hh))
            if "mlp" in kinds:
                def hm(mod, args, out, l=l):
                    return _rep(out, src[("mlp", l)])
                hooks.append((lay.mlp, "fwd", hm))
            ex_steps = [s_ for s_ in sts if s_[0] in ("experts", "neurons")]
            if ex_steps:
                def he(mod, args, out, l=l, ex_steps=ex_steps):
                    x = args[0][0, -1:]
                    own_c, own_a = self.contribs(l, x)
                    v = torch.zeros(x.shape[-1], dtype=torch.float32, device=x.device)
                    for s_ in ex_steps:
                        if s_[0] == "experts":
                            for e in s_[2]:
                                v += src[("c", l)].get(e, torch.zeros_like(v)) - own_c.get(e, torch.zeros_like(v))
                        else:
                            e, N = s_[2], s_[3]
                            m = torch.zeros(self._inter(l, e), device=x.device)
                            m[torch.tensor(N, device=x.device)] = 1.0
                            sa = src[("a", l)].get(e)
                            oa = own_a.get(e)
                            d = torch.zeros_like(m)
                            if sa is not None:
                                d += sa[1] * sa[0]
                            if oa is not None:
                                d -= oa[1] * oa[0]
                            v += (d * m) @ self.down(l, e).t()
                    o = _o(out)
                    return _rep(out, (o[0, -1].float() + v))
                hooks.append((lay.mlp, "fwd", he))
            if "resid" in kinds:
                def hr(mod, args, out, l=l):
                    return _rep(out, src[("resid", l)])
                hooks.append((lay, "fwd", hr))
        return hooks

    def _inter(self, l, e):
        mlp = self.layer(l).mlp
        if self.dense:
            return mlp.down_proj.weight.shape[1]
        if e == self.E:
            return mlp.shared_expert.down_proj.weight.shape[1]
        return mlp.experts.down_proj.shape[2]


def resolve_steps(hf_steps, cfg_rt):
    """replace symbolic expert / head arguments ('active', 'all', 'rand2', 'mask0', 'free') by the resolved ones."""
    out = []
    for st in hf_steps:
        st = list(st)
        if st[0] == "experts":
            ea = cfg_rt.get("e_by_layer", {}).get(st[1], cfg_rt.get("e"))
            st[2] = cfg_rt["experts"] if isinstance(st[2], str) else tuple(ea if e == "active" else e for e in st[2])
        elif st[0] == "neurons":
            st[2] = cfg_rt.get("e", st[2]) if st[2] == "active" else st[2]
        elif st[0] == "heads" and isinstance(st[1], str):
            st = ["heads", cfg_rt["head"][0], (cfg_rt["head"][1],)]
        out.append(tuple(st))
    return out


def metrics_of(lg, t, f, lg_ref):
    lp = torch.log_softmax(lg.float(), -1)
    lr = torch.log_softmax(lg_ref.float(), -1)
    return {"delta": float(lg[t] - lg[f]), "logp_true": float(lp[t]), "logp_foil": float(lp[f]), "p_true": float(lp[t].exp()),
            "rank_true": int((lp > lp[t]).sum()) + 1, "kl_to_clean": float((lp.exp() * (lp - lr)).sum())}


@torch.no_grad()
def hf_full(path, dtype, units, configs, masks, offload, device="cuda"):
    hf = HFRef(path, dtype, offload, device)
    E = hf.E
    res, recs, rt = {}, {}, {}
    rng = random.Random(7)
    t0 = time.time()
    for u, un in enumerate(units):
        mz = [(l, h, None) for (l, h) in masks[u]]
        rows = {"C": (un["clean"], []), "X": (un["corrupt"], []), "CMz": (un["clean"], mz), "XMz": (un["corrupt"], mz)}
        base = {k: hf.record(ids, m, want_hidden=(k == "C"), act_experts=None) for k, (ids, m) in rows.items()}
        rrows = {}  # route-masked rows: k -> (route mask, mode)
        # mean vectors for the mean-mode mask: the clean run's final-position head outputs (any fixed vector works)
        mm = [(l, h, base["C"][("oin", l)][h * hf.D:(h + 1) * hf.D].clone()) for (l, h) in masks[u]]
        masks_mean_vec = [m[2].float().cpu() for m in mm]
        rows.update({"CMm": (un["clean"], mm), "XMm": (un["corrupt"], mm)})
        for k in ("CMm", "XMm"):
            base[k] = hf.record(rows[k][0], rows[k][1])
        recs[u] = {"mean_vecs": masks_mean_vec,
                   "oin_mean": {l: base["C"][("oin_mean", l)].cpu() for l in range(hf.L)},
                   "hidden": base["C"]["hidden"],
                   "acts": {l: {e: (a.cpu(), w) for e, (a, w, _) in base["C"][("a", l)].items()} for l in range(hf.L)},
                   "x_C": {l: base["C"][("x", l)].cpu() for l in range(hf.L)},
                   "x_X": {l: base["X"][("x", l)].cpu() for l in range(hf.L)},
                   "route_C": {l: sorted(k_ for k_ in base["C"][("c", l)] if k_ != E or hf.dense) for l in range(hf.L)},
                   "route_X": {l: sorted(k_ for k_ in base["X"][("c", l)] if k_ != E or hf.dense) for l in range(hf.L)}}
        if not hf.dense:  # route masks: the clean and corrupted top experts at Ls[1], the clean top expert at Ls[2]
            Lp = pick_layers(hf.L)
            rC1, rX1, rC2 = recs[u]["route_C"][Lp[1]], recs[u]["route_X"][Lp[1]], recs[u]["route_C"][Lp[2]]
            e_x = next((e for e in rX1 if e != rC1[0]), (rC1[0] + 1) % E)
            rmask_u = [(Lp[1], rC1[0]), (Lp[1], e_x), (Lp[2], rC2[0])]
            recs[u]["route_mask"] = rmask_u
            for mode, tag in (("reroute", "r"), ("zero", "z")):
                for k, ids in (("CR" + tag, un["clean"]), ("XR" + tag, un["corrupt"])):
                    rows[k] = (ids, [])
                    rrows[k] = (rmask_u, mode)
                    base[k] = hf.record(ids, [], extra_hooks=hf.route_hooks(rmask_u, mode))
            if hasattr(hf.layer(Lp[1]).mlp, "shared_expert"):  # shared expert knocked out (zero mode)
                rmask_s = [(Lp[1], E), (Lp[1], rC1[0]), (Lp[2], E)]
                recs[u]["route_mask_shared"] = rmask_s
                for k, ids in (("CRs", un["clean"]), ("XRs", un["corrupt"])):
                    rows[k] = (ids, [])
                    rrows[k] = (rmask_s, "zero")
                    base[k] = hf.record(ids, [], extra_hooks=hf.route_hooks(rmask_s, "zero"))
        for k in rows:
            ref_k = "C" if k in ("C", "X") else "C" + k[1:]
            res[(u, "row", k)] = metrics_of(base[k]["logits"], un["true"], un["foil"], base[ref_k]["logits"])
        # hypothetical activations of every expert (+ shared) at the middle layer of the clean run (diag check)
        le = pick_layers(hf.L)[len(pick_layers(hf.L)) // 2]
        xC = base["C"][("x", le)]
        if not hf.offload:
            recs[u]["hyp_acts"] = {e: hf.expert_act(le, e, xC).cpu() for e in range(E + (1 if hasattr(hf.layer(le).mlp, "shared_expert") else 0))}
        for ci, c in enumerate(configs):
            if c["unit"] != u:
                continue
            par_ids, par_mask = rows[c["parent"]]
            src = base[c["src"]]
            crt = {}
            kind = c["eng"][0]
            # resolve symbolic arguments with the HF routing of the source / parent run
            if kind in ("expert", "expert_neurons") and c["eng"][2] == "active":
                l = c["eng"][1]
                crt["e"] = recs[u]["route_C"][l][0] if not hf.dense else 0
            if kind == "coalition_set":
                l = c["eng"][1]
                act_s = sorted(set(recs[u]["route_C"][l]) | set(recs[u]["route_X"][l]))
                if c["eng"][2] == "all":
                    crt["experts"] = tuple(act_s) + ((E,) if hasattr(hf.layer(l).mlp, "shared_expert") else ())
                else:
                    pool = sorted(set(range(E)) - set(recs[u]["route_C"][l]))[:1] + recs[u]["route_C"][l][:1]
                    crt["experts"] = tuple(sorted(pool))
            if kind == "multi":
                l3 = c["eng"][1][-1][0]
                crt["e"] = recs[u]["route_C"][l3][0] if not hf.dense else 0
                crt["e_by_layer"] = {st_[0]: (recs[u]["route_C"][st_[0]][0] if not hf.dense else 0) for st_ in c["eng"][1]}
            if kind == "attn_head" and isinstance(c["eng"][1], str):
                if c["eng"][1] == "mask0":
                    crt["head"] = masks[u][0]
                else:
                    lm = masks[u][0][0]
                    free = [h for h in range(hf.nH) if (lm, h) not in masks[u]]
                    crt["head"] = (lm, free[0])
            rt[ci] = crt
            steps = resolve_steps(c["hf"], crt)
            hooks = list(hf.mask_hooks(par_mask)) + hf.patch_hooks(steps, src, None)
            if c["parent"] in rrows:
                hooks += hf.route_hooks(*rrows[c["parent"]])
            lg = hf.fwd(par_ids, hooks)
            res[(u, "cfg", ci)] = metrics_of(lg, un["true"], un["foil"], base[c["src"]]["logits"])
        log(f"  HF unit {u + 1}/{len(units)} done ({time.time() - t0:.0f}s)")
    cfg_hf = hf.model.config
    del hf
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return res, recs, rt, cfg_hf


# ------------------------------------------------------------------------------------------------------------------
# engine side
# ------------------------------------------------------------------------------------------------------------------
def engine_full(eng, units, configs, masks, recs, rt):
    spec = eng.spec
    E = spec.n_experts
    pre, row_of = [], {}
    for u, un in enumerate(units):
        mz = tuple(masks[u])
        mv = recs[u]["mean_vecs"]
        for k, ids, hm, mode, mean in (("C", un["clean"], (), "zero", None), ("X", un["corrupt"], (), "zero", None),
                                        ("CMz", un["clean"], mz, "zero", None), ("XMz", un["corrupt"], mz, "zero", None),
                                        ("CMm", un["clean"], mz, "mean", mv), ("XMm", un["corrupt"], mz, "mean", mv)):
            row_of[(u, k)] = len(pre)
            pre.append(PrefillSpec(ids, un["true"], un["foil"], head_mask=hm, head_mask_mode=mode, head_mask_mean=mean,
                                   clean_ref=-1))
        if not spec.is_dense:  # route-masked rows whose spawns inherit the mask
            rm_ = tuple(recs[u]["route_mask"])
            for mode, tag in (("reroute", "r"), ("zero", "z")):
                for k, ids in (("CR" + tag, un["clean"]), ("XR" + tag, un["corrupt"])):
                    row_of[(u, k)] = len(pre)
                    pre.append(PrefillSpec(ids, un["true"], un["foil"], route_mask=rm_, route_mask_pos="all",
                                           route_mask_mode=mode, route_mask_inherit=True))
            if spec.shared_inter:
                for k, ids in (("CRs", un["clean"]), ("XRs", un["corrupt"])):
                    row_of[(u, k)] = len(pre)
                    pre.append(PrefillSpec(ids, un["true"], un["foil"], route_mask=tuple(recs[u]["route_mask_shared"]),
                                           route_mask_pos="all", route_mask_mode="zero", route_mask_inherit=True))
    # KL reference of prefill rows: their clean row (unmasked rows -> C, masked -> CM*)
    for u in range(len(units)):
        for k in ("X",):
            pre[row_of[(u, k)]].clean_ref = row_of[(u, "C")]
        for m in ("Mz", "Mm", "Rr", "Rz", "Rs"):
            if (u, "X" + m) in row_of:
                pre[row_of[(u, "X" + m)]].clean_ref = row_of[(u, "C" + m)]
                pre[row_of[(u, "C" + m)]].clean_ref = row_of[(u, "C" + m)]
        pre[row_of[(u, "C")]].clean_ref = row_of[(u, "C")]
    spawns, sp_of = [], {}
    for ci, c in enumerate(configs):
        u = c["unit"]
        par, src = row_of[(u, c["parent"])], row_of[(u, c["src"])]
        crt = rt[ci]
        kind = c["eng"][0]
        if kind in ("layer", "attn_layer", "block", "resid", "zero"):
            sp = SpawnSpec(c["eng"][1], par, src, kind)
        elif kind == "attn_head":
            l, h = (c["eng"][1], c["eng"][2]) if not isinstance(c["eng"][1], str) else crt["head"]
            sp = SpawnSpec(l, par, src, "attn_head", expert=h)
        elif kind == "expert":
            e = crt.get("e", c["eng"][2]) if c["eng"][2] == "active" else c["eng"][2]
            sp = SpawnSpec(c["eng"][1], par, src, "expert", expert=int(e))
        elif kind == "coalition_set":
            sp = SpawnSpec(c["eng"][1], par, src, "coalition_set", experts=tuple(int(e) for e in crt["experts"]))
        elif kind == "expert_neurons":
            e = crt.get("e", c["eng"][2]) if c["eng"][2] == "active" else c["eng"][2]
            sp = SpawnSpec(c["eng"][1], par, src, "expert_neurons", expert=int(e), neurons=c["eng"][3])
        elif kind == "multi":
            steps = []
            for (l, k_, ex) in c["eng"][1]:
                if k_ == "heads_experts":
                    ex = (ex[0], tuple(crt["e"] if e == "active" else e for e in ex[1]))
                elif k_ == "coalition_set":
                    ex = tuple(crt["e_by_layer"][l] if e == "active" else e for e in ex)
                steps.append((l, k_, ex))
            sp = SpawnSpec(steps[0][0], par, src, "multi", steps=tuple(steps))
        else:
            raise ValueError(kind)
        sp_of[ci] = len(spawns)
        spawns.append(sp)
    # extra identity spawns
    extra = {}
    for u in range(len(units)):
        Ls = pick_layers(spec.n_layers)
        le = Ls[len(Ls) // 2]
        X, C = row_of[(u, "X")], row_of[(u, "C")]
        if spec.is_dense:
            extra[(u, "coal_clean")] = len(spawns); spawns.append(SpawnSpec(le, X, C, "coalition_clean"))
            extra[(u, "layer_le")] = len(spawns); spawns.append(SpawnSpec(le, X, C, "layer"))
        else:
            extra[(u, "coal_clean")] = len(spawns); spawns.append(SpawnSpec(le, X, C, "coalition_clean"))
            extra[(u, "coal_union")] = len(spawns); spawns.append(SpawnSpec(le, X, C, "coalition_union"))
            extra[(u, "layer_le")] = len(spawns); spawns.append(SpawnSpec(le, X, C, "layer"))
        # neuron linearity: two disjoint halves and their union (vectors compared via spawn_vectors)
        e0 = 0 if spec.is_dense else int(recs[u]["route_C"][le][0])
        Ie = spec.inter
        perm = list(range(Ie))
        random.Random(u).shuffle(perm)
        A_, B_ = tuple(sorted(perm[: Ie // 3])), tuple(sorted(perm[Ie // 3:]))
        for nm, N in (("lin_a", A_), ("lin_b", B_), ("lin_ab", tuple(range(Ie)))):
            extra[(u, nm)] = len(spawns); spawns.append(SpawnSpec(le, X, C, "expert_neurons", expert=e0, neurons=N))
        extra[(u, "lin_expert")] = len(spawns); spawns.append(SpawnSpec(le, X, C, "expert", expert=e0))
    Ls = pick_layers(spec.n_layers)
    le = Ls[len(Ls) // 2]
    neur_diag = [(le, 0)] if spec.is_dense else [(le, int(recs[0]["route_C"][le][0])), (le, (E + 1) % E)] + ([(le, E)] if spec.shared_inter else [])
    diag = DiagSpec(attn_heads_mean=tuple(range(spec.n_layers)), expert_neurons=tuple(neur_diag),
                    resid_layers=tuple(range(-1, spec.n_layers - 1)), spawn_vectors=True, attn_heads_final=tuple(Ls))
    t0 = time.time()
    res = eng.run(pre, spawns, diag=diag, metrics=True)
    log(f"  engine pass: {len(pre)} prefill rows, {len(spawns)} spawns, {time.time() - t0:.1f}s")
    return res, pre, spawns, row_of, sp_of, extra, neur_diag


def compare_full(spec, units, configs, masks, hfres, recs, rt, eres, row_of, sp_of, extra, neur_diag, eng_dtype):
    rep = {}
    E = spec.n_experts
    mp, ms = eres.metrics_prefill, eres.metrics_spawn
    # prefill rows
    d_rows = {k: [] for (_, k) in row_of}
    met_rows = {m: [] for m in ("logp_true", "logp_foil", "p_true", "kl_to_clean")}
    rank_eq = []
    for (u, k), b in row_of.items():
        h = hfres[(u, "row", k)]
        d_rows[k].append(float(eres.delta[b]) - h["delta"])
        for m in met_rows:
            met_rows[m].append(float(mp[m][b]) - h[m])
        rank_eq.append(int(mp["rank_true"][b]) == h["rank_true"])
    rep["prefill_delta_vs_hf"] = {k: _stats(v) for k, v in d_rows.items()}
    rep["prefill_metrics_vs_hf"] = {m: _stats(v) for m, v in met_rows.items()}
    rep["prefill_rank_equal_frac"] = float(np.mean(rank_eq))
    # the head masks must change the runs (else the mask checks are vacuous)
    rep["head_mask_effect_on_delta_hf"] = {m: float(np.mean([abs(hfres[(u, "row", k + m)]["delta"] - hfres[(u, "row", k)]["delta"])
                                                             for u in range(len(units)) for k in ("C", "X")])) for m in ("Mz", "Mm")}
    if (0, "CRr") in row_of:
        rep["route_mask_effect_on_delta_hf"] = {m: float(np.mean([abs(hfres[(u, "row", k + m)]["delta"] - hfres[(u, "row", k)]["delta"])
                                                                  for u in range(len(units)) for k in ("C", "X")])) for m in ("Rr", "Rz")}
    if (0, "CRs") in row_of:
        rep["route_mask_effect_on_delta_hf"]["Rs"] = float(np.mean([abs(hfres[(u, "row", k + "Rs")]["delta"] - hfres[(u, "row", k)]["delta"])
                                                                     for u in range(len(units)) for k in ("C", "X")]))
    # configs by family of tag
    groups = {}
    for ci, c in enumerate(configs):
        parts = c["tag"].split("_")
        while parts and parts[-1].isdigit():
            parts.pop()
        groups.setdefault("_".join(parts), []).append(ci)
    out = {}
    all_e, all_h = [], []
    met_sp = {m: [] for m in ("logp_true", "logp_foil", "p_true", "kl_to_clean")}
    rank_sp = []
    for key, cis in sorted(groups.items()):
        de, dh, re_, rh = [], [], [], []
        for ci in cis:
            c = configs[ci]
            u = c["unit"]
            j = sp_of[ci]
            h = hfres[(u, "cfg", ci)]
            par_b = row_of[(u, c["parent"])]
            de.append(float(eres.sp_delta[j]))
            dh.append(h["delta"])
            re_.append(float(eres.sp_delta[j]) - float(eres.delta[par_b]))
            rh.append(h["delta"] - hfres[(u, "row", c["parent"])]["delta"])
            for m in met_sp:
                met_sp[m].append(float(ms[m][j]) - h[m])
            rank_sp.append(int(ms["rank_true"][j]) == h["rank_true"])
        z = np.array(de) - np.array(dh)
        all_e += re_; all_h += rh
        out[key] = dict(_stats(z), rescue_corr=_corr(re_, rh), rescue_hf_meanabs=float(np.mean(np.abs(rh))))
    rep["spawn_delta_vs_hf"] = out
    rep["spawn_rescue_corr_all"] = _corr(all_e, all_h)
    rep["rows"] = [{"tag": c["tag"], "unit": c["unit"], "eng_delta": float(eres.sp_delta[sp_of[ci]]),
                    "hf_delta": hfres[(c["unit"], "cfg", ci)]["delta"],
                    "eng_parent": float(eres.delta[row_of[(c["unit"], c["parent"])]]),
                    "hf_parent": hfres[(c["unit"], "row", c["parent"])]["delta"]} for ci, c in enumerate(configs)]
    # units the model is competent on (transformers clean Delta >= 1): rescue correlation over their spawns
    comp = {u for u in range(len(units)) if hfres[(u, "row", "C")]["delta"] >= 1.0}
    ce, ch = [], []
    for ci, c in enumerate(configs):
        if c["unit"] in comp:
            j, h = sp_of[ci], hfres[(c["unit"], "cfg", ci)]
            ce.append(float(eres.sp_delta[j]) - float(eres.delta[row_of[(c["unit"], c["parent"])]]))
            ch.append(h["delta"] - hfres[(c["unit"], "row", c["parent"])]["delta"])
    rep["competent_units"] = sorted(comp)
    rep["spawn_rescue_corr_competent"] = _corr(ce, ch)
    rep["spawn_delta_maxabs_all"] = float(max(v["maxabs"] for v in out.values()))
    rep["spawn_metrics_vs_hf"] = {m: _stats(v) for m, v in met_sp.items()}
    rep["spawn_rank_equal_frac"] = float(np.mean(rank_sp))
    # identities
    ide = {}
    sv = eres.extra["diag"]["spawn_v"]

    def dd(a, b):
        return abs(float(eres.sp_delta[a]) - float(eres.sp_delta[b]))
    lin_err, all_vs_exp_d, all_vs_exp_v, coal_layer = [], [], [], []
    for u in range(len(units)):
        va, vb, vab, vex = (sv[extra[(u, n)]] for n in ("lin_a", "lin_b", "lin_ab", "lin_expert"))
        lin_err.append(float((va + vb - vab).norm() / max(vab.norm(), 1e-30)))
        all_vs_exp_v.append(float((vab - vex).norm() / max(vex.norm(), 1e-30)))
        all_vs_exp_d.append(dd(extra[(u, "lin_ab")], extra[(u, "lin_expert")]))
        # dense: coalition_clean == layer; MoE: coalition_union == layer (coalition_clean omits noised-only experts)
        coal_layer.append(dd(extra[(u, "coal_union" if (u, "coal_union") in extra else "coal_clean")], extra[(u, "layer_le")]))
    ide["neurons_linearity_relerr_max"] = float(max(lin_err))
    ide["neurons_all_vs_expert_vec_relerr_max"] = float(max(all_vs_exp_v))
    ide["neurons_all_vs_expert_ddelta"] = _stats(all_vs_exp_d)
    ide["coalition_union_or_dense_clean_vs_layer_ddelta"] = _stats(coal_layer)
    # zero == parent, masked head == parent
    zero_d, mh_d = [], []
    for ci, c in enumerate(configs):
        j = sp_of[ci]
        pb = row_of[(c["unit"], c["parent"])]
        if c["eng"][0] == "zero":
            zero_d.append(float(eres.sp_delta[j]) - float(eres.delta[pb]))
        if c["tag"].endswith("headmasked"):
            mh_d.append(float(eres.sp_delta[j]) - float(eres.delta[pb]))
    ide["zero_vs_parent"] = _stats(zero_d)
    ide["attn_head_on_masked_head_vs_parent"] = _stats(mh_d)
    rep["identities"] = ide
    # diagnostics vs HF
    dg = eres.extra["diag"]
    hm_err, rl_err, na_err, nw_err, nh_err, routed_mismatch = [], [], [], [], [], []
    for u in range(len(units)):
        bC = row_of[(u, "C")]
        for l in range(spec.n_layers):
            a = dg["attn_heads_mean"][l][bC].reshape(-1)
            b = recs[u]["oin_mean"][l].float().reshape(-1)
            hm_err.append(float((a - b).norm() / max(b.norm(), 1e-30)))
        for l in range(-1, spec.n_layers - 1):
            T_ = len(units[u]["clean"])
            a = dg["resid_layers"][l][bC, :T_].float()
            b = recs[u]["hidden"][l + 1].float()
            rl_err.append(float((a - b).norm() / max(b.norm(), 1e-30)))
        for (l, e) in (neur_diag if "hyp_acts" in recs[u] else []):
            dn = dg["expert_neurons"][(l, e)]
            acts = recs[u]["acts"][l]
            w_hf = acts[e][1] if e in acts else 0.0
            nw_err.append(abs(float(dn["w"][bC]) - w_hf))
            routed_mismatch.append(bool(dn["routed"][bC]) != (e in acts))
            a_hf = recs[u]["hyp_acts"][e].numpy()  # = the routed activations when routed
            (na_err if e in acts else nh_err).append(float(np.linalg.norm(dn["act"][bC] - a_hf) / max(np.linalg.norm(a_hf), 1e-30)))
    rep["diag_attn_heads_mean_relerr"] = _stats(hm_err)
    rep["diag_resid_layers_relerr"] = _stats(rl_err)
    rep["diag_expert_neurons_act_relerr"] = _stats(na_err)
    rep["diag_expert_neurons_w_abs"] = _stats(nw_err)
    rep["diag_expert_neurons_hypothetical_act_relerr"] = _stats(nh_err)
    rep["diag_expert_neurons_routed_mismatch"] = int(sum(routed_mismatch))
    # head mask visible in attn_heads_final
    vis = []
    for u in range(len(units)):
        for (l, h) in masks[u]:
            if l in dg["attn_heads_final"]:
                vis.append(float(dg["attn_heads_final"][l][row_of[(u, "XMz")], h].float().abs().max()))
    rep["head_mask_zero_heads_final_maxabs"] = float(max(vis)) if vis else None
    if spec.shared_inter:
        rep["shared_w_final_range"] = [float(eres.extra["shared_w"].min()), float(eres.extra["shared_w"].max())]
    return rep


# ------------------------------------------------------------------------------------------------------------------
# per-layer mode
# ------------------------------------------------------------------------------------------------------------------
def _hf_layer(path, cfg_dict, l, dtype, device="cuda"):
    """transformers decoder layer l of the checkpoint (weights read from the safetensors, experts fused as HF does)."""
    import transformers as T
    from moetrace.weights import CheckpointStore
    spec, d = load_spec(path)
    arch = cfg_dict["architectures"][0]
    mod = {"Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2Moe"), "Qwen3ForCausalLM": ("qwen3", "Qwen3"),
           "Qwen2ForCausalLM": ("qwen2", "Qwen2"), "MistralForCausalLM": ("mistral", "Mistral")}[arch]
    M = __import__(f"transformers.models.{mod[0]}.modeling_{mod[0]}", fromlist=["x"])
    C = getattr(T, f"{mod[1]}Config")
    cfg = C.from_pretrained(d)
    cfg._attn_implementation = "eager"
    with torch.device("meta"):  # no random init of the (large) layer; weights are assigned from the checkpoint below
        layer = getattr(M, f"{mod[1]}DecoderLayer")(cfg, l)
    store = CheckpointStore(d, spec, device="cpu", dtype=dtype)
    sd = {}
    pre = f"model.layers.{l}."
    keys = [k for k in (store.weight_map or {}) if k.startswith(pre)] if store.weight_map is not None else \
        [k for k in store._handle(os.path.basename(store._single)).keys() if k.startswith(pre)]
    for k in keys:
        sd[k[len(pre):]] = store.cpu_tensor(k).to(dtype)
    if spec.family == "qwen2_moe":
        E = spec.n_experts
        gu = torch.stack([torch.cat([sd.pop(f"mlp.experts.{e}.gate_proj.weight"), sd.pop(f"mlp.experts.{e}.up_proj.weight")], 0) for e in range(E)])
        dn = torch.stack([sd.pop(f"mlp.experts.{e}.down_proj.weight") for e in range(E)])
        sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"] = gu, dn
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if "rotary" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    assert not any(t.is_meta for t in list(layer.parameters()) + list(layer.buffers())), "meta tensors left in the layer"
    layer = layer.to(device, dtype).eval()
    rot = getattr(M, f"{mod[1]}RotaryEmbedding")(cfg).to(device)
    return layer, rot


SANITY = [("The capital of France is", " Paris"), ("The Eiffel Tower is located in the city of", " Paris"),
          ("Water is made of hydrogen and", " oxygen"), ("The opposite of hot is", " cold"),
          ("Monday, Tuesday, Wednesday,", " Thursday")]


def sanity_rows(path):
    """factual prompts with single-token answers in the model's tokenizer (BOS as the tokenizer adds it)."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(load_spec(path)[1])
    except Exception:  # tiny random checkpoints have no tokenizer
        return []
    out = []
    for p_, a_ in SANITY:
        ids = tok(p_)["input_ids"]
        t = tok(a_, add_special_tokens=False)["input_ids"]
        if len(t) == 2 and t[0] == tok.convert_tokens_to_ids("\u2581"):  # SentencePiece: "▁" + word
            t = t[1:]
        if len(t) == 1:
            out.append((p_, a_, ids, t[0]))
    return out


def engine_identities(eng, units):
    """engine-only identities on a real checkpoint (no HF): zero == parent, coalition_union / coalition_set(all active
    [+ shared]) == layer, expert_neurons(all) == expert, neuron linearity, shared expert kinds (Qwen2-MoE)."""
    from moetrace.engine import DiagSpec as DS
    spec = eng.spec
    E = spec.n_experts
    n = len(units)
    pre = [PrefillSpec(un["clean"], un["true"], un["foil"]) for un in units]
    pre += [PrefillSpec(un["corrupt"], un["true"], un["foil"], clean_ref=i) for i, un in enumerate(units)]
    r1 = eng.run(pre, [], record_routing=True)
    Ls = pick_layers(spec.n_layers)
    le = Ls[len(Ls) // 2]
    spawns, tag = [], []

    def add(t, sp):
        tag.append(t); spawns.append(sp)
    for u in range(n):
        C, X = u, n + u
        rc, rx = [int(e) for e in r1.route_idx[le, C]], [int(e) for e in r1.route_idx[le, X]]
        e_r = rc[0]
        act = sorted(set(rc) | set(rx)) + ([E] if spec.shared_inter else [])
        Ie = spec.inter
        perm = list(range(Ie)); random.Random(u).shuffle(perm)
        add((u, "zero"), SpawnSpec(le, X, C, "zero"))
        add((u, "layer"), SpawnSpec(le, X, C, "layer"))
        add((u, "union"), SpawnSpec(le, X, C, "coalition_union"))
        add((u, "setall"), SpawnSpec(le, X, C, "coalition_set", experts=tuple(act)))
        add((u, "expert"), SpawnSpec(le, X, C, "expert", expert=e_r))
        add((u, "n_all"), SpawnSpec(le, X, C, "expert_neurons", expert=e_r, neurons=tuple(range(Ie))))
        add((u, "n_a"), SpawnSpec(le, X, C, "expert_neurons", expert=e_r, neurons=tuple(sorted(perm[: Ie // 3]))))
        add((u, "n_b"), SpawnSpec(le, X, C, "expert_neurons", expert=e_r, neurons=tuple(sorted(perm[Ie // 3:]))))
        if spec.shared_inter:
            add((u, "sh_expert"), SpawnSpec(le, X, C, "expert", expert=E))
            add((u, "sh_set"), SpawnSpec(le, X, C, "coalition_set", experts=(E,)))
            add((u, "sh_n_all"), SpawnSpec(le, X, C, "expert_neurons", expert=E, neurons=tuple(range(spec.shared_inter))))
    r2 = eng.run(pre, spawns, diag=DS(spawn_vectors=True))
    j = {t: i for i, t in enumerate(tag)}
    sd, sv, d = r2.sp_delta, r2.extra["diag"]["spawn_v"], r2.delta
    out = {"layer": le}
    out["zero_vs_parent"] = _stats([sd[j[(u, "zero")]] - d[n + u] for u in range(n)])
    out["union_vs_layer"] = _stats([sd[j[(u, "union")]] - sd[j[(u, "layer")]] for u in range(n)])
    out["setall_vs_layer"] = _stats([sd[j[(u, "setall")]] - sd[j[(u, "layer")]] for u in range(n)])
    out["neurons_all_vs_expert"] = _stats([sd[j[(u, "n_all")]] - sd[j[(u, "expert")]] for u in range(n)])
    out["neurons_linearity_relerr_max"] = float(max(float((sv[j[(u, "n_a")]] + sv[j[(u, "n_b")]] - sv[j[(u, "n_all")]]).norm()
                                                          / max(sv[j[(u, "n_all")]].norm(), 1e-30)) for u in range(n)))
    out["expert_rescue_mean"] = float(np.mean([sd[j[(u, "expert")]] - d[n + u] for u in range(n)]))
    out["layer_rescue_mean"] = float(np.mean([sd[j[(u, "layer")]] - d[n + u] for u in range(n)]))
    if spec.shared_inter:
        out["shared_expert_vs_set"] = _stats([sd[j[(u, "sh_expert")]] - sd[j[(u, "sh_set")]] for u in range(n)])
        out["shared_neurons_all_vs_expert"] = _stats([sd[j[(u, "sh_n_all")]] - sd[j[(u, "sh_expert")]] for u in range(n)])
        out["shared_rescue_mean"] = float(np.mean([sd[j[(u, "sh_expert")]] - d[n + u] for u in range(n)]))
        out["shared_w_final_mean_by_layer"] = [float(x) for x in r1.extra["shared_w"].mean(1)]
    out["clean_delta_pos_frac"] = float(np.mean(r1.delta[:n] > 0))
    out["drop_mean"] = float(np.mean(r1.delta[:n] - r1.delta[n:]))
    return out


@torch.no_grad()
def per_layer(path, units, layers, dtype, device="cuda"):
    spec, d = load_spec(path)
    cfg_dict = json.load(open(os.path.join(d, "config.json")))
    eng = Engine(path, dtype=dtype, device=device, router_fp32=ROUTER32)
    san = sanity_rows(path)
    t0 = time.time()
    ident = engine_identities(eng, units)
    log(f"  engine identities {time.time() - t0:.1f}s: " + json.dumps(ident)[:800])
    pre = [PrefillSpec(un["clean"], un["true"], un["foil"]) for un in units]
    pre += [PrefillSpec(ids, t, t) for (_, _, ids, t) in san]
    rl = sorted(set([l - 1 for l in layers] + list(layers)))
    diag = DiagSpec(resid_layers=tuple(rl), route_all_layers=tuple(layers) if not spec.is_dense else ())
    t0 = time.time()
    res = eng.run(pre, [], diag=diag, metrics=True)
    log(f"  engine pass {time.time() - t0:.1f}s")
    dg = res.extra["diag"]
    nu = len(units)
    beh = {"delta": [float(x) for x in res.delta[:nu]], "rank_true": [int(x) for x in res.metrics_prefill["rank_true"][:nu]],
           "top1_is_true": [bool(int(res.top1[b]) == units[b]["true"]) for b in range(nu)],
           "sanity": [{"prompt": p_, "answer": a_, "rank": int(res.metrics_prefill["rank_true"][nu + i]),
                       "p": float(res.metrics_prefill["p_true"][nu + i])} for i, (p_, a_, _, _) in enumerate(san)]}
    log("  sanity: " + json.dumps(beh["sanity"]))
    del eng
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    rep = {"behaviour": beh, "identities": ident, "layers": {}}
    prompts = units + [dict(clean=ids) for (_, _, ids, _) in san]
    for l in layers:
        # transformers layer in the run dtype and an fp32 reference of the same layer, both on the engine's input
        outs = {}
        for tag, dt_ in (("hf", dtype), ("hf32", torch.float32)):
            if tag == "hf32" and dtype == torch.float32:
                continue
            layer, rot = _hf_layer(path, cfg_dict, l, dt_, device)
            res_t = []
            for b, un in enumerate(prompts):
                T_ = len(un["clean"])
                hin = dg["resid_layers"][l - 1][b:b + 1, :T_].to(device, dt_)
                res_t.append(_apply_layer(layer, rot, hin, dt_, spec.family == "qwen2_moe", device))
            outs[tag] = res_t
            del layer, rot
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        errs, errs_out, route_agree, route_overlap, errs_req = [], [], [], [], []
        e32, h32, ra_e32, ra_h32 = [], [], [], []
        for b, un in enumerate(prompts):
            T_ = len(un["clean"])
            h_in = dg["resid_layers"][l - 1][b, :T_].float()
            de = dg["resid_layers"][l][b, :T_].float() - h_in
            hout_h, ti_h = outs["hf"][b]
            dh = hout_h - h_in
            errs.append(float((de - dh).norm() / max(dh.norm(), 1e-30)))  # relative error of the layer's update
            errs_out.append(float((de - dh).abs().max()))
            ti_e = dg["route_all"][l][0][b, :T_].astype(np.int64) if ti_h is not None else None
            if ti_h is not None:
                eq = np.array([set(ti_e[t]) == set(ti_h[t]) for t in range(T_)])
                route_agree.append(float(np.mean(eq)))
                route_overlap.append(float(np.mean([len(set(ti_e[t]) & set(ti_h[t])) / ti_e.shape[1] for t in range(T_)])))
                if eq.any():  # positions with the same routed set (a bf16 near-tie flip changes that token's update)
                    eqt = torch.from_numpy(eq)
                    errs_req.append(float((de[eqt] - dh[eqt]).norm() / max(dh[eqt].norm(), 1e-30)))
            if "hf32" in outs:  # noise floor: transformers in the run dtype vs fp32, and the engine vs fp32
                h32o, ti_32 = outs["hf32"][b]
                d32 = h32o - h_in
                e32.append(float((de - d32).norm() / max(d32.norm(), 1e-30)))
                h32.append(float((dh - d32).norm() / max(d32.norm(), 1e-30)))
                if ti_32 is not None:
                    ra_e32.append(float(np.mean([set(ti_e[t]) == set(ti_32[t]) for t in range(T_)])))
                    ra_h32.append(float(np.mean([set(ti_h[t]) == set(ti_32[t]) for t in range(T_)])))
        ent = {"update_relerr": _stats(errs), "out_maxabs": _stats(errs_out),
               "update_relerr_routing_equal": _stats(errs_req) if errs_req else None,
               "routing_set_agreement": float(np.mean(route_agree)) if route_agree else None,
               "routing_overlap": float(np.mean(route_overlap)) if route_overlap else None}
        if e32:
            ent.update({"engine_vs_fp32_update_relerr": _stats(e32), "hf_vs_fp32_update_relerr": _stats(h32),
                        "engine_vs_fp32_routing_agreement": float(np.mean(ra_e32)) if ra_e32 else None,
                        "hf_vs_fp32_routing_agreement": float(np.mean(ra_h32)) if ra_h32 else None})
        rep["layers"][str(l)] = ent
        log(f"  layer {l}: update relerr max {max(errs):.4g} mean {np.mean(errs):.4g}"
            + (f", routing sets equal {np.mean(route_agree):.3f}, overlap {np.mean(route_overlap):.3f}" if route_agree else "")
            + (f"; vs fp32: engine {max(e32):.4g} / transformers {max(h32):.4g}" if e32 else "")
            + (f", routing vs fp32: engine {np.mean(ra_e32):.3f} / transformers {np.mean(ra_h32):.3f}" if ra_e32 else ""))
    return rep


def _apply_layer(layer, rot, hin, dt_, moe, device):
    """one transformers decoder layer on hin [1, T, H] (eager causal mask); returns (output fp32 CPU [T, H], routed
    indices [T, k] or None)."""
    T_ = hin.shape[1]
    pos = torch.arange(T_, device=device)[None]
    cos, sin = rot(hin, pos)
    mask = torch.full((T_, T_), float("-inf"), device=device).triu(1)[None, None].to(dt_)
    rec = {}
    hk = None
    if moe:
        def rec_idx(m, a, o):
            rec.setdefault("idx", o[2].detach().cpu().numpy())
        hk = layer.mlp.gate.register_forward_hook(rec_idx)
    with torch.no_grad():
        out = layer(hin, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin))
    if hk is not None:
        hk.remove()
    return _o(out)[0].float().cpu(), rec.get("idx")


@torch.no_grad()
def chain_mode(path, units, device="cuda", engine_fp32=True):
    """transformers decoder layers chained one at a time (no full-model load, no disk offload) in bf16 and fp32 on the
    units' clean prompts and the sanity prompts, vs the engine in bf16 (and fp32 if it fits): Delta, log p of the answer,
    rank, routing at every layer and position. transformers bf16 vs fp32 = transformers' own bf16 noise floor."""
    import transformers as T
    spec, d = load_spec(path)
    cfg_dict = json.load(open(os.path.join(d, "config.json")))
    san = sanity_rows(path)
    prompts = [(un["clean"], un["true"], un["foil"]) for un in units] + [(ids, t, t) for (_, _, ids, t) in san]
    moe = spec.family == "qwen2_moe"
    L = spec.n_layers
    runs = {}
    for tag, dt_ in (("hf_bf16", torch.bfloat16), ("hf_fp32", torch.float32)):
        t0 = time.time()
        from moetrace.weights import CheckpointStore
        st = CheckpointStore(d, spec, device="cpu", dtype=dt_)
        emb = st.cpu_tensor(spec.k_embed).to(dt_)
        hs = [emb[torch.tensor(ids)][None].to(device) for (ids, _, _) in prompts]
        routes = {}
        for l in range(L):
            layer, rot = _hf_layer(path, cfg_dict, l, dt_, device)
            rl_ = []
            for i in range(len(hs)):
                out, idx = _apply_layer(layer, rot, hs[i], dt_, moe, device)
                hs[i] = out[None].to(device, dt_)
                rl_.append(idx)
            routes[l] = rl_
            del layer, rot
            if device == "cuda":
                torch.cuda.empty_cache()
        arch = cfg_dict["architectures"][0]
        mod = {"Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2Moe"), "Qwen3ForCausalLM": ("qwen3", "Qwen3"),
               "Qwen2ForCausalLM": ("qwen2", "Qwen2"), "MistralForCausalLM": ("mistral", "Mistral")}[arch]
        M = __import__(f"transformers.models.{mod[0]}.modeling_{mod[0]}", fromlist=["x"])
        norm = getattr(M, f"{mod[1]}RMSNorm")(spec.hidden, eps=spec.rms_eps).to(device, dt_)
        norm.weight.data.copy_(st.cpu_tensor(spec.k_norm).to(dt_))
        head = (emb if (spec.tie_embeddings or not st.has(spec.k_head)) else st.cpu_tensor(spec.k_head).to(dt_)).to(device)
        rows = []
        for i, (ids, t, f) in enumerate(prompts):
            lg = F.linear(norm(hs[i])[0, -1], head).float()
            lp = torch.log_softmax(lg, -1)
            rows.append({"delta": float(lg[t] - lg[f]), "logp_true": float(lp[t]), "rank_true": int((lp > lp[t]).sum()) + 1})
        runs[tag] = {"rows": rows, "routes": routes}
        del hs, head, emb, norm
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        log(f"  {tag} chain {time.time() - t0:.0f}s")
    for tag, dt_ in (("eng_bf16", torch.bfloat16), ("eng_fp32", torch.float32)):
        if tag == "eng_fp32" and not engine_fp32:
            continue
        t0 = time.time()
        try:
            eng = Engine(path, dtype=dt_, device=device, router_fp32=ROUTER32)
            pre = [PrefillSpec(ids, t, f) for (ids, t, f) in prompts]
            res = eng.run(pre, [], diag=DiagSpec(route_all_layers=tuple(range(L)) if moe else ()), metrics=True)
        except torch.cuda.OutOfMemoryError as ex:
            runs[tag] = {"error": f"OOM: {str(ex)[:200]}"}
            log(f"  {tag}: OOM")
            eng = res = None
            gc.collect(); torch.cuda.empty_cache()
            continue
        rows = [{"delta": float(res.delta[i]), "logp_true": float(res.metrics_prefill["logp_true"][i]),
                 "rank_true": int(res.metrics_prefill["rank_true"][i])} for i in range(len(prompts))]
        routes = {l: [res.extra["diag"]["route_all"][l][0][i, :len(prompts[i][0])].astype(np.int64) for i in range(len(prompts))]
                  for l in range(L)} if moe else {}
        runs[tag] = {"rows": rows, "routes": routes}
        del eng, res
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        log(f"  {tag} pass {time.time() - t0:.0f}s")
    nu = len(units)
    rep = {"n_unit_prompts": nu, "sanity_prompts": [p_ for (p_, _, _, _) in san], "runs": {}}
    for tag, r in runs.items():
        if "error" in r:
            rep["runs"][tag] = r
            continue
        rep["runs"][tag] = {"delta_units": [x["delta"] for x in r["rows"][:nu]],
                            "sanity_rank": [x["rank_true"] for x in r["rows"][nu:]]}
    pairs = [("eng_bf16", "hf_bf16"), ("hf_bf16", "hf_fp32"), ("eng_bf16", "hf_fp32"), ("eng_fp32", "hf_fp32")]
    rep["compare"] = {}
    for a, b in pairs:
        if a not in runs or b not in runs or "error" in runs[a] or "error" in runs[b]:
            continue
        ra, rb = runs[a]["rows"], runs[b]["rows"]
        ent = {"delta_absdiff": _stats([x["delta"] - y["delta"] for x, y in zip(ra[:nu], rb[:nu])]),
               "logp_true_absdiff": _stats([x["logp_true"] - y["logp_true"] for x, y in zip(ra, rb)]),
               "rank_equal_frac": float(np.mean([x["rank_true"] == y["rank_true"] for x, y in zip(ra, rb)]))}
        if moe:
            per_l = []
            for l in range(L):
                ag = [np.mean([set(ia[t]) == set(ib[t]) for t in range(len(ia))]) for ia, ib in zip(runs[a]["routes"][l], runs[b]["routes"][l])]
                per_l.append(float(np.mean(ag)))
            ent["routing_agreement_by_layer"] = per_l
            ent["routing_agreement_mean"] = float(np.mean(per_l))
            ent["routing_agreement_min"] = float(np.min(per_l))
        rep["compare"][f"{a}__{b}"] = ent
        log(f"  {a} vs {b}: |dDelta| mean {ent['delta_absdiff']['meanabs']:.4g} max {ent['delta_absdiff']['maxabs']:.4g}, "
            f"|d logp| max {ent['logp_true_absdiff']['maxabs']:.4g}"
            + (f", routing agreement mean {ent['routing_agreement_mean']:.3f} min {ent['routing_agreement_min']:.3f}" if moe else ""))
    return rep


@torch.no_grad()
def engine_vs_fp32(path, units, device="cuda"):
    """engine in the model's protocol (bf16, + fp32 router if flagged / --router-fp32) vs the engine in fp32 (= transformers
    fp32, chain mode) on a T2-like plan: single-layer MoE (layer) and attention (attn_layer) patches at every layer, block at
    three layers, the shared expert at every layer (Qwen2-MoE). Rescue correlation, |dDelta|, final-position routing
    agreement, and the attention share AUC+ attn / (AUC+ attn + AUC+ MoE) per unit and on the mean curves."""
    spec, _ = load_spec(path)
    L, n = spec.n_layers, len(units)
    pre = [PrefillSpec(un["clean"], un["true"], un["foil"]) for un in units]
    pre += [PrefillSpec(un["corrupt"], un["true"], un["foil"], clean_ref=i) for i, un in enumerate(units)]
    spawns, tags = [], []
    for u in range(n):
        for l in range(L):
            for k in ("layer", "attn_layer"):
                spawns.append(SpawnSpec(l, n + u, u, k)); tags.append((u, k, l))
            if spec.shared_inter:
                spawns.append(SpawnSpec(l, n + u, u, "expert", expert=spec.n_experts)); tags.append((u, "shared", l))
        for l in pick_layers(L)[1:4]:
            spawns.append(SpawnSpec(l, n + u, u, "block")); tags.append((u, "block", l))
    out = {}
    # "proto" = the bf16 engine (+ fp32 router logits with --router-fp32), explicitly bf16 even if models.py sets an fp32
    # protocol, so that the measurement is always bf16 vs fp32
    for tag, kw in (("proto", dict(dtype=torch.bfloat16, router_fp32=ROUTER32)), ("fp32", dict(dtype=torch.float32))):
        t0 = time.time()
        eng = Engine(path, device=device, **kw)
        res = eng.run(pre, spawns, record_routing=True, metrics=True)
        out[tag] = {"delta": res.delta.copy(), "sp": res.sp_delta.copy(), "route": res.route_idx.copy(), "protocol": eng.protocol,
                    "s": time.time() - t0, "mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}
        log(f"  engine {tag} {eng.protocol}: {time.time() - t0:.0f}s")
        del eng, res
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    A, B = out["proto"], out["fp32"]
    ra = np.array([A["sp"][j] - A["delta"][n + tags[j][0]] for j in range(len(tags))])
    rb = np.array([B["sp"][j] - B["delta"][n + tags[j][0]] for j in range(len(tags))])
    kinds = sorted(set(t[1] for t in tags))
    rep = {"protocol": A["protocol"], "pass_s": {"proto": A["s"], "fp32": B["s"]}, "peak_mem_gb": {"proto": A["mem_gb"], "fp32": B["mem_gb"]},
           "rescue_corr_all": _corr(ra, rb), "rescue_absdiff": _stats(ra - rb),
           "prefill_delta_absdiff": _stats(A["delta"] - B["delta"]),
           "by_kind": {k: dict(_stats(ra[[i for i, t in enumerate(tags) if t[1] == k]] - rb[[i for i, t in enumerate(tags) if t[1] == k]]),
                               rescue_corr=_corr(ra[[i for i, t in enumerate(tags) if t[1] == k]], rb[[i for i, t in enumerate(tags) if t[1] == k]]),
                               rescue_fp32_meanabs=float(np.abs(rb[[i for i, t in enumerate(tags) if t[1] == k]]).mean()))
                       for k in kinds}}
    if not spec.is_dense:
        agree = [float(np.mean([set(A["route"][l, b]) == set(B["route"][l, b]) for b in range(2 * n)])) for l in range(L)]
        rep["final_routing_agreement_by_layer"] = agree
        rep["final_routing_agreement_mean"] = float(np.mean(agree))

    def share(r, units_):
        att = np.array([[r[j] for j, t in enumerate(tags) if t[0] == u and t[1] == "attn_layer"] for u in units_]).mean(0)
        moe = np.array([[r[j] for j, t in enumerate(tags) if t[0] == u and t[1] == "layer"] for u in units_]).mean(0)
        a, m = np.clip(att, 0, None).sum(), np.clip(moe, 0, None).sum()
        return float(a / (a + m)) if a + m > 0 else None
    rep["attn_share_mean_curve"] = {"proto": share(ra, range(n)), "fp32": share(rb, range(n))}
    rep["attn_share_per_unit"] = {"proto": [share(ra, [u]) for u in range(n)], "fp32": [share(rb, [u]) for u in range(n)]}
    log(f"  engine proto vs fp32: rescue r {rep['rescue_corr_all']:.4f}, |d rescue| mean {rep['rescue_absdiff']['meanabs']:.3g}, "
        f"attention share {rep['attn_share_mean_curve']['proto']:.3f} vs {rep['attn_share_mean_curve']['fp32']:.3f}"
        + (f", final routing agreement {rep['final_routing_agreement_mean']:.3f}" if "final_routing_agreement_mean" in rep else ""))
    return rep


# ------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="MODELS key or checkpoint directory (tiny models)")
    ap.add_argument("--dtype", default="bf16", choices=list(DT))
    ap.add_argument("--units", type=int, default=6, help="pairs (checkpoints: 2 directed units each) or tiny units")
    ap.add_argument("--offload", action="store_true", help="HF model on GPU + CPU RAM (device_map auto, no disk)")
    ap.add_argument("--light", action="store_true", help="fewer configurations (offloaded HF reference)")
    ap.add_argument("--per-layer", default="", help="comma-separated layers: per-layer mode instead of the full mode")
    ap.add_argument("--chain", action="store_true", help="chain mode: transformers layers chained (bf16 + fp32) vs the engine")
    ap.add_argument("--no-engine-fp32", action="store_true", help="chain mode: skip the fp32 engine run")
    ap.add_argument("--router-fp32", action="store_true", help="Qwen2-MoE: fp32 router logits in the engine and in transformers")
    ap.add_argument("--vs-fp32", action="store_true", help="engine (model protocol) vs engine fp32 on a T2-like plan")
    ap.add_argument("--pairs", default=None, help="STR pair parquet for the units (default: WinoGrande train_xl twins)")
    ap.add_argument("--tag", default="", help="suffix of the output name")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda", help="cpu = debugging only (needs a CUDA-free monkeypatch harness)")
    args = ap.parse_args()
    global ROUTER32
    if args.router_fp32:
        ROUTER32 = True
        patch_router_fp32()
    path, name, tiny = resolve(args.model)
    dtype = DT[args.dtype]
    spec, d = load_spec(path)
    sfx = "_perlayer" if args.per_layer else ("_chain" if args.chain else ("_vsfp32" if args.vs_fp32 else
                                                                            ("_fp32" if args.dtype == "fp32" and not tiny else "")))
    if args.router_fp32:
        sfx = "_r32" + sfx
    sfx += args.tag
    out = args.out or f"results/verify_ext14_{'tiny_' if tiny else ''}{name}{sfx}.json"
    rep = {"model": args.model, "path": d, "family": spec.family, "dtype": args.dtype, "n_layers": spec.n_layers,
           "n_experts": spec.n_experts, "top_k": spec.top_k, "shared_inter": spec.shared_inter,
           "engine_file": sys.modules["moetrace.engine"].__file__, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "hf_offload": bool(args.offload), "light_plan": bool(args.light)}
    if tiny:
        units = tiny_units(spec, 2 * args.units)
        rep["units"] = {"kind": "tiny_random", "n": len(units)}
    else:
        tf = "mistral" if spec.family in ("mistral", "mixtral") else ("olmoe" if spec.family == "olmoe" else "qwen")
        units, info = wino_units(path, tf, args.units, pairs_file=args.pairs)
        rep["units"] = dict(info, kind="wino_train_xl_str", n=len(units))
    log(f"{name}: {spec.family}, {len(units)} units, dtype {args.dtype}")
    t0 = time.time()
    rep["router_fp32_flag"] = bool(args.router_fp32)
    if args.vs_fp32:
        rep["vs_fp32"] = engine_vs_fp32(path, units, args.device)
    elif args.chain:
        rep["chain"] = chain_mode(path, units, args.device, engine_fp32=not args.no_engine_fp32)
    elif args.per_layer:
        layers = [int(x) for x in args.per_layer.split(",")]
        rep["per_layer"] = per_layer(path, units, layers, dtype, args.device)
    else:
        masks = [head_mask_of(spec, u) for u in range(len(units))]
        configs = make_plan(spec, units, light=args.light)
        log(f"  {len(configs)} configurations")
        hfres, recs, rt, _ = hf_full(path, dtype, units, configs, masks, args.offload, args.device)
        rep["hf_s"] = time.time() - t0
        eng = Engine(path, dtype=dtype, device=args.device, router_fp32=ROUTER32)
        eres, pre, spawns, row_of, sp_of, extra, neur_diag = engine_full(eng, units, configs, masks, recs, rt)
        rep["n_configs"] = len(configs)
        rep["n_spawns"] = len(spawns)
        rep.update(compare_full(spec, units, configs, masks, hfres, recs, rt, eres, row_of, sp_of, extra, neur_diag, dtype))
        if spec.shared_inter:  # reroute is undefined for the always-on shared expert -> must raise
            try:
                eng.run([PrefillSpec(units[0]["clean"], units[0]["true"], units[0]["foil"], route_mask=((1, spec.n_experts),),
                                     route_mask_mode="reroute")], [], record_routing=False)
                rep["shared_reroute_raises"] = False
            except ValueError:
                rep["shared_reroute_raises"] = True
        del eng
    rep["total_s"] = time.time() - t0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    log(f"wrote {out}")
    for k, v in rep.items():
        if k not in ("per_layer", "chain", "rows", "vs_fp32"):
            log(f"{k}: {json.dumps(v)[:600]}")


if __name__ == "__main__":
    main()
