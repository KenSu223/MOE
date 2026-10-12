"""ext13atp (Phase 5, P5-5): attribution patching (AtP, AtP*, EAP-IG) against activation patching on OLMoE-1B-7B-0125.

Question (team meeting 2026-10-11): can gradient-based attribution replace the activation-patching sweeps, so that a
reverse-layer-streaming engine for Qwen3 / Mixtral would be worth building? Pilot on OLMoE, where a normal autograd
forward fits on the GPU.

Tasks (STR pairs, ext7_pairs contract; directed id = 2 * pair_idx + d; 64 discovery + 64 validation pairs = 128 + 128
directed cases):
  wino  WinoGrande option swap in the Phase-3 format (data/wino_str/pairs_train_xl_olmoe.parquet, OLMoE margin pairs of
        results/wino_olmoe), drawn from the Phase-3 discovery / validation pairs (data/wino_str/case_sets.json).
  ioi   IOI with the S2 -> IO corruption (Phase-3 items data/ioi/items.parquet re-tokenised for OLMoE with the rules of
        scripts/ext7_ioi_build.py), drawn from the Phase-3 IOI discovery / validation pairs, margin scanned here.

Components (all at the final position; the patch replaces the component's corrupted value by its clean value in the
corrupted run = denoising, as every engine kind used in Phases 3-4):
  expert (l, e)   c_e = w_e E_e(x), every expert routed in the clean OR the corrupted run at layer l (others: delta = 0)
  head   (l, h)   W_o[:, h] H_h (H_h = head output before o_proj)
  attn   (l)      attention-sublayer output;   moe (l)  MoE-sublayer output
Truth = engine activation patching (kinds expert, attn_head, attn_layer, layer): rescue = Delta_patched - Delta_corrupt,
Delta = logit(true) - logit(foil) (= LD of Zhang & Nanda).

Estimates (transformers, eager attention, bf16 weights and activations, gradients w.r.t. activations only):
  delta_n  = a_n(clean) - a_n(corrupt) (fp32 vector in the residual stream; the node's write)
  g_mid(l) = dDelta / d h_mid(l)[final], h_mid = residual after the attention sublayer (head / attn nodes write there)
  g_post(l)= dDelta / d h_post(l)[final], residual after the MoE sublayer (expert / moe nodes write there)
  AtP      delta_n . g(alpha = 0)                          (gradient at the corrupted run; Nanda 2023, Kramar et al. 2024)
  IG       delta_n . mean_k g(alpha_k), alpha on the straight line between the corrupted and the clean INPUT EMBEDDINGS
           (EAP-IG of Hanna et al. 2024: right Riemann sum alpha_k = k / m, m = 8; also m = 16 and the midpoint rule)
  DLA      delta_n . (gamma * (W_U[true] - W_U[foil])) / rms(h_final_corrupt)  (direct path, final norm frozen)
  AtP* (cut estimates)  Kramar et al.'s QK fix computes the softmax that a query / key node feeds EXACTLY and linearises
           only after it. Our nodes are sublayer outputs at the final position, so the softmaxes they feed are those of
           the NEXT attention sublayer at the final position (its query and its own key / value) and, for MoE models,
           the next ROUTER (softmax + top-k). We therefore compute the node's patched final-position residual exactly up to
           a cut c after the node and linearise from there: est_c = (h'_c - h_c) . dDelta/dh_c (corrupted-run gradient).
           Cut 0 = AtP; "router fix" = the next MoE exact (attention nodes); "QK fix" = the next attention exact (MoE
           nodes); AtP* = both (one attention and one router exact, i.e. the next full sublayer pair); the last cut = exact
           activation patching (HF numerics). The cut estimates come from the HF patched forwards below, which a reverse-
           streaming engine would replace by one single-token, two-sublayer forward per node.
Routing (the MoE caveat, RESEARCH_PLAN Phase 5 review issue 8): every node is also patched in HF (free routing) with the
final position's top-k recorded at every layer; a node "switches" routing when any downstream layer's top-k SET differs
from the unpatched corrupted row of the same batch. Switching nodes are patched again with the downstream top-k frozen to
the corrupted run's (weights still recomputed from the router softmax), so that
    exact - AtP = (exact - frozen) [routing switch, invisible to a gradient] + (frozen - AtP) [smooth non-linearity].
"""
from __future__ import annotations

import json
import os
import random
import time
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .models import MODELS, RESULTS

ROOT = "/home/ubuntu/MOE"
MODEL = "olmoe"
REPO = MODELS[MODEL]["repo"]
DATA = os.path.join(RESULTS, "p5_atp_data")
N_PAIRS = 64  # per split
SEED = 13
MARGIN = 1.0
IG_M = 16  # gradient points alpha = k / IG_M, k = 0..IG_M (k = 0: AtP; even k: EAP-IG m = 8; odd k: midpoint m = 8)

TASKS = {
    "wino": {"label": "WinoGrande option swap", "pairs": os.path.join(ROOT, "data", "wino_str", "pairs_train_xl_olmoe.parquet"),
             "case_sets": os.path.join(DATA, "case_sets_wino.json"), "run": "p5_atp_wino"},
    "ioi": {"label": "IOI (S2 -> IO)", "pairs": os.path.join(DATA, "pairs_ioi_s2io_olmoe.parquet"),
            "case_sets": os.path.join(DATA, "case_sets_ioi.json"), "run": "p5_atp_ioi"},
}
CTYPES = ("expert", "head", "attn", "moe")
ATTN_SIDE = ("head", "attn")  # nodes written into h_mid(l); expert / moe nodes are written into h_post(l)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run_dir(task: str) -> str:
    d = os.path.join(RESULTS, TASKS[task]["run"])
    os.makedirs(d, exist_ok=True)
    return d


def node_cut(ctype: str, layer: int) -> int:
    """Index of the residual point a node writes into: 2 l = h_mid(l), 2 l + 1 = h_post(l)."""
    return 2 * int(layer) + (0 if ctype in ATTN_SIDE else 1)


# ===================================================================================================================
# data: IOI pairs for OLMoE, case sets
# ===================================================================================================================
def build_ioi_pairs(tok) -> tuple[pd.DataFrame, dict]:
    """Re-tokenise the Phase-3 IOI items (data/ioi/items.parquet) for OLMoE with the rules of scripts/ext7_ioi_build.py:
    every name one token in context, the S2 -> IO prompt differs from the clean one only at S2 (same length), ' IO' and
    ' S' single-token continuations with the same ids after both prompts. Pair = (A = clean, B = S2 -> IO),
    trig_a = IO, trig_b = S. pair_id as in the Phase-3 files."""
    from . import ext7_controls as C
    from .data import _single_token_continuation
    it = pd.read_parquet(os.path.join(ROOT, "data", "ioi", "items.parquet"))
    rows, why = [], {}
    for r in it.itertuples():
        pr = C.ioi_prompts(r.template, r.io, r.s, r.place, r.obj, r.c, r.d)
        ok, ids, pos = True, {}, {}
        for k in ("clean", "s2io"):
            txt, spans = pr[k]
            e = C._enc(tok, txt, True)
            ids[k] = list(e["input_ids"])
            pos[k] = {lab: C._span_positions(e, a, b) for lab, (a, b) in spans.items()}
            if any(len(v) != 1 for v in pos[k].values()):
                ok = False
                why["name_not_one_token"] = why.get("name_not_one_token", 0) + 1
                break
        if not ok:
            continue
        if len(ids["clean"]) != len(ids["s2io"]) or pos["clean"] != pos["s2io"]:
            why["length_or_position_mismatch"] = why.get("length_or_position_mismatch", 0) + 1
            continue
        P = {lab: v[0] for lab, v in pos["clean"].items()}
        diff = [t for t in range(len(ids["clean"])) if ids["clean"][t] != ids["s2io"][t]]
        if diff != [P["S2"]]:
            why["diff_outside_s2"] = why.get("diff_outside_s2", 0) + 1
            continue
        trig = {}
        for nm, w in (("io", r.io), ("s", r.s)):
            t = {k: _single_token_continuation(tok, ids[k], pr[k][0], w, True) for k in ("clean", "s2io")}
            if None in t.values() or len(set(t.values())) != 1:
                ok = False
                break
            trig[nm] = t["clean"]
        if not ok or trig["io"] == trig["s"]:
            why["answer_not_single_token"] = why.get("answer_not_single_token", 0) + 1
            continue
        rows.append(dict(pair_id=r.pair_id, ids_a=json.dumps(ids["clean"]), ids_b=json.dumps(ids["s2io"]),
                         str_pos=json.dumps([P["S2"]]), trig_a=int(trig["io"]), trig_b=int(trig["s"]),
                         prompt_a=pr["clean"][0], prompt_b=pr["s2io"][0], word_a=r.io, word_b=r.s,
                         template_type=r.template_type, template_idx=int(r.template_idx), io=r.io, s=r.s,
                         s1_pos=P["S1"], io_pos=P["IO"], s2_pos=P["S2"], n_tokens=len(ids["clean"]), corruption="s2io",
                         mention_a_pos=json.dumps([P["IO"]]), mention_b_pos=json.dumps([P["S1"]])))
    return pd.DataFrame(rows), {"n_items": len(it), "n_pairs": len(rows), "rejections": why}


def make_case_sets(phase3_cs: dict, eligible: set, seed: int = SEED, n: int = N_PAIRS, note: str = "") -> dict:
    """64 / 64 pairs drawn (random.Random(seed)) from the Phase-3 discovery / validation pairs that pass the OLMoE
    margin (eligible pair ids); the Phase-3 split assignment is kept. pair_idx 0..n-1 discovery, n..2n-1 validation."""
    inv = {int(i): p for p, i in phase3_cs["pair_idx_of"].items()}
    out_pairs, sets = [], {}
    rng = random.Random(seed)
    for sp in ("discovery", "validation"):
        pool = sorted(inv[int(i)] for i in phase3_cs["pairs"][sp] if inv[int(i)] in eligible)
        if len(pool) < n:
            raise SystemExit(f"only {len(pool)} eligible {sp} pairs (< {n})")
        rng.shuffle(pool)
        pick = pool[:n]
        sets[sp] = list(range(len(out_pairs), len(out_pairs) + n))
        out_pairs += pick
    return {"note": note, "seed": seed, "pair_idx_of": {p: i for i, p in enumerate(out_pairs)},
            "pairs": sets, "directed": {k: [2 * i + d for i in v for d in (0, 1)] for k, v in sets.items()},
            "both_directions": True}


def load_task(task: str):
    """(pairs, case-set JSON, families, ordered directed ids, {id: DCase}, {id: split})."""
    from . import ext7_pairs as P
    cs = json.load(open(TASKS[task]["case_sets"]))
    pairs = P.load_pairs(TASKS[task]["pairs"], cs)
    fam = P.resolve_sets(cs)
    ids = cs["directed"]["discovery"] + cs["directed"]["validation"]
    dc = P.directed_cases(pairs, ids)
    split = {c: s for s in ("discovery", "validation") for c in cs["directed"][s]}
    return pairs, cs, fam, ids, dc, split


# ===================================================================================================================
# HF model with hooks: records, gradients, patched forwards
# ===================================================================================================================
class HFAtP:
    """transformers OLMoE (eager attention, eager experts) with forward hooks on every decoder layer. All rows of one
    forward share one length T (no padding); every quantity is taken at the final position T - 1."""

    def __init__(self, device: str = "cuda", dtype=torch.bfloat16, experts_impl: str = "eager"):
        from transformers import AutoModelForCausalLM
        from .arch import snapshot_dir
        self.device, self.dtype = device, dtype
        self.model = AutoModelForCausalLM.from_pretrained(snapshot_dir(REPO), dtype=dtype, device_map=device,
                                                          attn_implementation="eager", experts_implementation=experts_impl)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.L, self.nH, self.E, self.k = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_experts, cfg.num_experts_per_tok
        self.H = cfg.hidden_size
        self.D = self.H // self.nH
        self.norm_topk = bool(cfg.norm_topk_prob)
        self.eps = float(cfg.rms_norm_eps)
        self.layers = self.model.model.layers
        self.embed = self.model.model.embed_tokens.weight
        self.W_U = self.model.lm_head.weight
        self.gamma = self.model.model.norm.weight
        self.st: Optional[dict] = None
        self._hooks = []
        for l, lay in enumerate(self.layers):
            self._hooks += [lay.register_forward_pre_hook(partial(self._pre_layer, l)),
                            lay.self_attn.register_forward_hook(partial(self._post_attn, l)),
                            lay.self_attn.o_proj.register_forward_pre_hook(partial(self._pre_oproj, l)),
                            lay.mlp.register_forward_pre_hook(partial(self._pre_mlp, l)),
                            lay.mlp.gate.register_forward_hook(partial(self._post_gate, l)),
                            lay.mlp.register_forward_hook(partial(self._post_mlp, l)),
                            lay.register_forward_hook(partial(self._post_layer, l))]

    # ---- hooks -----------------------------------------------------------------------------------------------------
    def _rec(self, name, l, x):
        self.st["rec"].setdefault(name, [None] * self.L)[l] = x.detach().clone()

    def _pre_layer(self, l, mod, args):
        st = self.st
        if st is None:
            return
        h = args[0]
        if st.get("record"):
            self._rec("h_pre", l, h[:, -1])
        if st.get("cut") is not None:
            st["_hpre"] = h[:, -1].detach()

    @staticmethod
    def _apply_patch(t, ps):
        """t [B, T, H] bf16; ps = (rows long [n], vec fp32 [n, H], setmask bool [n]) -> patched clone."""
        rows, vec, setm = ps
        t = t.clone()
        fin = t[rows, -1].float()
        t[rows, -1] = torch.where(setm[:, None], vec, fin + vec).to(t.dtype)
        return t

    def _post_attn(self, l, mod, args, out):
        st = self.st
        if st is None:
            return
        o = out[0]
        changed = False
        ps = st.get("patch_attn", {}).get(l)
        if ps is not None:
            o = self._apply_patch(o, ps)
            changed = True
        if st.get("record"):
            self._rec("attn", l, o[:, -1])
        if st.get("grad"):
            st["gt_attn"][l] = o
        if st.get("cut") is not None:
            hm = (st["_hpre"] + o[:, -1].detach())  # bf16, the residual add of the decoder layer
            st["est"][2 * l] = ((hm - hm[0:1]).float() @ st["cut"]["g_mid"][l]).cpu()
        if changed:
            return (o,) + tuple(out[1:])

    def _pre_oproj(self, l, mod, args):
        st = self.st
        if st is not None and st.get("record"):
            x = args[0]
            self._rec("heads", l, x[:, -1].reshape(x.shape[0], self.nH, self.D))

    def _pre_mlp(self, l, mod, args):
        st = self.st
        if st is not None and st.get("record"):
            self._rec("moe_in", l, args[0][:, -1])

    def _post_gate(self, l, mod, args, out):
        st = self.st
        if st is None:
            return
        logits, scores, idx = out
        T = st["T"]
        B = logits.shape[0] // T
        fin = torch.arange(B, device=logits.device) * T + (T - 1)
        changed = False
        fz = st.get("freeze", {}).get(l)
        if fz is not None:
            probs = torch.softmax(logits[fin].float(), dim=-1)
            nidx = fz[None, :].expand(B, -1)
            w = probs.gather(1, nidx)
            if self.norm_topk:
                w = w / w.sum(-1, keepdim=True)
            scores = scores.clone()
            idx = idx.clone()
            scores[fin] = w.to(scores.dtype)
            idx[fin] = nidx
            changed = True
        if st.get("record") or st.get("route"):
            st.setdefault("route_idx", [None] * self.L)[l] = idx[fin].detach().clone()
            st.setdefault("route_w", [None] * self.L)[l] = scores[fin].detach().float().clone()
        if changed:
            return logits, scores, idx

    def _post_mlp(self, l, mod, args, out):
        st = self.st
        if st is None:
            return
        o = out
        changed = False
        ps = st.get("patch_moe", {}).get(l)
        if ps is not None:
            o = self._apply_patch(o, ps)
            changed = True
        if st.get("record"):
            self._rec("moe", l, o[:, -1])
        if st.get("grad"):
            st["gt_moe"][l] = o
        if changed:
            return o

    def _post_layer(self, l, mod, args, out):
        st = self.st
        if st is None:
            return
        h = out[0] if isinstance(out, tuple) else out
        if st.get("record"):
            self._rec("h_post", l, h[:, -1])
        if st.get("cut") is not None:
            hp = h[:, -1].detach()
            st["est"][2 * l + 1] = ((hp - hp[0:1]).float() @ st["cut"]["g_post"][l]).cpu()

    # ---- helpers -----------------------------------------------------------------------------------------------------
    def contribs(self, l: int, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> dict:
        """{expert: fp32 [H]} = w_e * E_e(x) for the routed experts of one final-position token (x bf16 [H])."""
        ex = self.layers[l].mlp.experts
        out = {}
        for s in range(idx.shape[0]):
            e = int(idx[s])
            g, u = F.linear(x[None], ex.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(F.silu(g) * u, ex.down_proj[e])
            out[e] = y[0].float() * float(w[s])
        return out

    def readout(self, hN: torch.Tensor, true_id: int, foil_id: int) -> torch.Tensor:
        """Delta in fp32 from the final-norm output hN (bf16 [B, H]): (W_U[t] - W_U[f]) . hN in fp32. The engine's (and
        HF's) bf16 logits round Delta to the bf16 spacing of the logits (0.03-0.06 at |logit| 8-16); this readout keeps
        only the residual stream's own bf16 noise."""
        u = self.W_U[true_id].float() - self.W_U[foil_id].float()
        return hN.float() @ u

    def readout_bf16(self, hN: torch.Tensor, true_id: int, foil_id: int) -> torch.Tensor:
        """Delta as the engine computes it: bf16 logits (fp32 accumulation, bf16 output) of true and foil."""
        w = self.W_U[[true_id, foil_id]]  # [2, H] bf16
        lg = (hN @ w.t()).float()  # [B, 2]
        return lg[:, 0] - lg[:, 1]

    def dla_vector(self, true_id: int, foil_id: int, h_final: torch.Tensor) -> torch.Tensor:
        """gamma * (W_U[t] - W_U[f]) / rms(h_final) in fp32 [H] (direct path, final norm frozen)."""
        u = self.gamma.float() * (self.W_U[true_id].float() - self.W_U[foil_id].float())
        rms = torch.sqrt(h_final.float().pow(2).mean() + self.eps)
        return u / rms

    # ---- passes ------------------------------------------------------------------------------------------------------
    def grad_pass(self, clean_ids: list, corrupt_ids: list, true_id: int, foil_id: int, m: int = IG_M) -> dict:
        """One batch of m + 1 rows on the straight line between the corrupted (alpha = 0) and the clean (alpha = 1) input
        embeddings; returns Delta per row, g_mid / g_post fp32 [m + 1, L, H] and the records of rows 0 (corrupt) and
        m (clean)."""
        dev = self.embed.device
        T = len(clean_ids)
        ec = self.embed[torch.tensor(clean_ids, device=dev)].float()
        eo = self.embed[torch.tensor(corrupt_ids, device=dev)].float()
        al = torch.arange(m + 1, device=dev, dtype=torch.float32) / m
        x = (eo[None] + al[:, None, None] * (ec - eo)[None]).to(self.dtype)
        x[0] = eo.to(self.dtype)
        x[m] = ec.to(self.dtype)
        x = x.detach().requires_grad_(True)
        self.st = {"T": T, "record": True, "grad": True, "rec": {}, "gt_attn": [None] * self.L, "gt_moe": [None] * self.L}
        try:
            with torch.enable_grad():
                hN = self.model.model(inputs_embeds=x, use_cache=False).last_hidden_state[:, -1]
                d = self.readout(hN, true_id, foil_id)
                gs = torch.autograd.grad(d.sum(), self.st["gt_attn"] + self.st["gt_moe"])
        finally:
            st, self.st = self.st, None
        g_mid = torch.stack([g[:, -1].float() for g in gs[: self.L]], 1)  # [m + 1, L, H]
        g_post = torch.stack([g[:, -1].float() for g in gs[self.L:]], 1)
        rec = {k: [t for t in v] for k, v in st["rec"].items()}
        return {"delta": d.detach().float().cpu().numpy(), "g_mid": g_mid.detach(), "g_post": g_post.detach(), "rec": rec,
                "route_idx": [t.cpu() for t in st["route_idx"]], "route_w": [t.cpu() for t in st["route_w"]], "T": T, "m": m}

    def nodes(self, gp: dict, m: int) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Node table of one directed case from a grad pass (row 0 = corrupt, row m = clean).
        Returns (table [ctype, layer, idx, cut, clean_active, corrupt_active, setmode], delta fp32 [N, H] (the node's write
        clean - corrupt), setval fp32 [N, H] (the clean value for set-mode nodes, else 0), and the DLA vector)."""
        rec = gp["rec"]
        ri, rw = gp["route_idx"], gp["route_w"]
        rows, vec, setv = [], [], []
        Wo = [self.layers[l].self_attn.o_proj.weight for l in range(self.L)]
        for l in range(self.L):
            hc, ho = rec["heads"][l][m], rec["heads"][l][0]  # [nH, D]
            for h in range(self.nH):
                v = (hc[h].float() - ho[h].float()) @ Wo[l][:, h * self.D:(h + 1) * self.D].float().t()
                rows.append(("head", l, h, 0, 0, False))
                vec.append(v)
                setv.append(None)
            ac, ao = rec["attn"][l][m], rec["attn"][l][0]
            rows.append(("attn", l, -1, 0, 0, True))
            vec.append(ac.float() - ao.float())
            setv.append(ac.float())
            cc = self.contribs(l, rec["moe_in"][l][m], ri[l][m], rw[l][m])
            co = self.contribs(l, rec["moe_in"][l][0], ri[l][0], rw[l][0])
            for e in sorted(set(cc) | set(co)):
                z = torch.zeros(self.H, device=ac.device)
                rows.append(("expert", l, e, int(e in cc), int(e in co), False))
                vec.append(cc.get(e, z) - co.get(e, z))
                setv.append(None)
            mc, mo = rec["moe"][l][m], rec["moe"][l][0]
            rows.append(("moe", l, -1, 0, 0, True))
            vec.append(mc.float() - mo.float())
            setv.append(mc.float())
        tab = pd.DataFrame(rows, columns=["ctype", "layer", "idx", "clean_active", "corrupt_active", "setmode"])
        tab["cut"] = [node_cut(c, l) for c, l in zip(tab.ctype, tab.layer)]
        V = torch.stack(vec)
        S = torch.stack([s if s is not None else torch.zeros(self.H, device=V.device) for s in setv])
        return tab, V, S

    @torch.no_grad()
    def patched_pass(self, corrupt_ids: list, tab: pd.DataFrame, V: torch.Tensor, S: torch.Tensor, sel: np.ndarray,
                     g_mid0: torch.Tensor, g_post0: torch.Tensor, freeze: Optional[list] = None) -> dict:
        """Rows = [null row] + nodes sel of the case, each the corrupted prompt with ONE node set to its clean value at the
        final position (add mode: + delta in fp32 then bf16, as the engine; set mode (attn / moe layer nodes): the clean
        bf16 value). Returns Delta [1 + n], cut estimates est [2L, 1 + n] ((h'_c - h_c) . g_c vs the null row), final-
        position top-k per layer [L, 1 + n, k]. freeze = per-layer top-k index tensors applied at the final position of
        every row (routing switches suppressed)."""
        dev = self.embed.device
        T = len(corrupt_ids)
        n = len(sel)
        B = n + 1
        ids = torch.tensor(corrupt_ids, device=dev)[None].expand(B, -1).contiguous()
        sub = tab.iloc[sel]
        pa, pm = {}, {}
        rows_all = torch.arange(1, B, device=dev)
        Vs, Ss = V[torch.as_tensor(sel, device=V.device)], S[torch.as_tensor(sel, device=V.device)]
        setm_all = torch.tensor(sub.setmode.to_numpy(), device=dev)
        vals = torch.where(setm_all[:, None], Ss, Vs)
        side = np.where(sub.ctype.isin(ATTN_SIDE).to_numpy(), "a", "m")
        lay = sub.layer.to_numpy()
        for sd in ("a", "m"):
            for l in np.unique(lay[side == sd]):
                j = torch.tensor(np.nonzero((side == sd) & (lay == l))[0], device=dev)
                (pa if sd == "a" else pm)[int(l)] = (rows_all[j], vals[j], setm_all[j])
        self.st = {"T": T, "route": True, "patch_attn": pa, "patch_moe": pm,
                   "cut": {"g_mid": g_mid0, "g_post": g_post0}, "est": [None] * (2 * self.L),
                   "freeze": {l: freeze[l].to(dev) for l in range(self.L)} if freeze is not None else {}}
        try:
            hN = self.model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
        finally:
            st, self.st = self.st, None
        return {"hN": hN, "est": torch.stack(st["est"]).numpy(),
                "route_idx": torch.stack([t.cpu() for t in st["route_idx"]]).numpy(),
                "route_w": torch.stack([t.cpu() for t in st["route_w"]]).numpy()}


def estimates(tab: pd.DataFrame, V: torch.Tensor, gp: dict, dla: torch.Tensor, m: int = IG_M) -> pd.DataFrame:
    """Gradient estimates per node from one grad pass: atp (alpha = 0), atp_clean (alpha = 1), ig8 (EAP-IG, right Riemann,
    alpha = k/8), ig8mid (midpoint, alpha = (2k-1)/16), ig16 (right, alpha = k/16), dla, vnorm."""
    cut = torch.tensor(tab.cut.to_numpy(), device=V.device)
    gm, gpost = gp["g_mid"], gp["g_post"]  # [m + 1, L, H]
    lay = cut // 2
    is_mid = (cut % 2 == 0)
    G = torch.where(is_mid[None, :, None], gm[:, lay], gpost[:, lay])  # [m + 1, N, H]
    S = torch.einsum("anh,nh->an", G, V).cpu().numpy()  # [m + 1, N]
    out = tab.copy()
    out["atp"] = S[0]
    out["atp_clean"] = S[m]
    out["ig8"] = S[2:m + 1:2].mean(0)
    out["ig8mid"] = S[1:m:2].mean(0)
    out["ig16"] = S[1:].mean(0)
    out["dla"] = (V @ dla).cpu().numpy()
    out["vnorm"] = V.norm(dim=-1).cpu().numpy()
    return out


def run_case_hf(hf: HFAtP, x, bs: int = 256, patch: bool = True, m: int = IG_M) -> tuple[pd.DataFrame, dict, dict]:
    """All HF quantities of one directed case x (ext7_pairs.DCase): gradient estimates for every node and, with
    patch=True, the HF activation patches (free routing; frozen routing for the nodes whose patch switches downstream
    top-k), cut estimates c00..c{2L-1} and routing-switch statistics. Returns (node table, case info, grad pass)."""
    L, k = hf.L, hf.k
    gp = hf.grad_pass(x.clean_ids, x.corrupt_ids, x.true_id, x.foil_id, m)
    tab, V, S = hf.nodes(gp, m)
    dla = hf.dla_vector(x.true_id, x.foil_id, gp["rec"]["h_post"][L - 1][0])
    out = estimates(tab, V, gp, dla, m)
    info = {"case_id": int(x.case_id), "T": int(x.T), "delta_corrupt_hf": float(gp["delta"][0]), "delta_clean_hf": float(gp["delta"][m]),
            "ig_path_delta": json.dumps([round(float(v), 4) for v in gp["delta"]]), "n_nodes": int(len(tab))}
    if not patch:
        return out, info, gp
    N = len(tab)
    cutcols = np.full((N, 2 * L), np.nan, dtype=np.float32)
    dfree = np.full(N, np.nan)
    dfree_b = np.full(N, np.nan)
    nsw_l = np.zeros(N, dtype=np.int16)
    nsw_s = np.zeros(N, dtype=np.int16)
    sw_mass = np.zeros(N, dtype=np.float32)
    first_sw = np.full(N, -1, dtype=np.int16)
    null_route, null_ok, null_d = None, True, []
    first_aff = np.where(tab.ctype.isin(ATTN_SIDE).to_numpy(), tab.layer.to_numpy(), tab.layer.to_numpy() + 1)
    for s0 in range(0, N, bs - 1):
        sel = np.arange(s0, min(N, s0 + bs - 1))
        r = hf.patched_pass(x.corrupt_ids, tab, V, S, sel, gp["g_mid"][0], gp["g_post"][0])
        d = hf.readout(r["hN"], x.true_id, x.foil_id).cpu().numpy()
        db = hf.readout_bf16(r["hN"], x.true_id, x.foil_id).cpu().numpy()
        null_d.append(float(d[0]))
        dfree[sel] = d[1:] - d[0]
        dfree_b[sel] = db[1:] - db[0]
        est = r["est"][:, 1:].T  # [n, 2L]
        cc = tab.cut.to_numpy()[sel]
        valid = np.arange(2 * L)[None, :] >= cc[:, None]
        cutcols[sel] = np.where(valid, est, np.nan)
        ri, rw = r["route_idx"], r["route_w"]  # [L, B, k]
        if null_route is None:
            null_route, null_w = ri[:, 0].copy(), rw[:, 0].copy()
        elif not np.array_equal(np.sort(null_route, -1), np.sort(ri[:, 0], -1)):
            null_ok = False
        for l in range(L):
            M0 = np.zeros(hf.E, dtype=bool)
            M0[ri[l, 0]] = True
            ov = M0[ri[l, 1:]].sum(-1)  # [n] experts kept
            dropped_mass = np.array([rw[l, 0][~np.isin(ri[l, 0], ri[l, 1 + j])].sum() for j in range(len(sel))]) if (ov < k).any() else np.zeros(len(sel))
            aff = l >= first_aff[sel]
            sw = (ov < k) & aff
            nsw_l[sel] += sw.astype(np.int16)
            nsw_s[sel] += np.where(aff, k - ov, 0).astype(np.int16)
            sw_mass[sel] += np.where(aff, dropped_mass, 0).astype(np.float32)
            fs = first_sw[sel]
            first_sw[sel] = np.where((fs < 0) & sw, l, fs)
    out["rescue_hf"] = dfree  # fp32 readout of Delta (see HFAtP.readout)
    out["rescue_hf_bf16"] = dfree_b  # bf16 logits as in the engine
    out["n_switch_layers"] = nsw_l
    out["n_switch_slots"] = nsw_s
    out["switch_mass"] = sw_mass
    out["first_switch"] = first_sw
    # frozen-routing re-run of the switching nodes
    rfz = np.full(N, np.nan)
    swn = np.nonzero(nsw_l > 0)[0]
    freeze = [torch.as_tensor(null_route[l], dtype=torch.long) for l in range(L)]
    for s0 in range(0, len(swn), bs - 1):
        sel = swn[s0:s0 + bs - 1]
        r = hf.patched_pass(x.corrupt_ids, tab, V, S, sel, gp["g_mid"][0], gp["g_post"][0], freeze=freeze)
        d = hf.readout(r["hN"], x.true_id, x.foil_id).cpu().numpy()
        rfz[sel] = d[1:] - d[0]
    out["rescue_frozen"] = rfz
    for c in range(2 * L):
        out[f"c{c:02d}"] = cutcols[:, c]
    route_gp = np.stack([gp["route_idx"][l][0].numpy() for l in range(L)])
    info.update({"null_delta_free": null_d[0], "null_delta_spread": float(np.ptp(null_d)), "null_route_consistent": bool(null_ok),
                 "null_route_eq_grad_route": bool(np.array_equal(np.sort(route_gp, -1), np.sort(null_route, -1))),
                 "n_switch_nodes": int(len(swn))})
    return out, info, gp
