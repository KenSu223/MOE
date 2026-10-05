"""ext5 F4 (RESEARCH_PLAN Phase 2): causal tracing at the LAST SUBJECT TOKEN with suffix wavefront rows.

Executor kept separate from moetrace.engine (owned by another agent this wave). It REUSES the engine's building blocks
(rmsnorm, rope_tables, apply_rope, attn_prefill, moe_forward, Engine.__init__/_qk/_pair_logits, weights.LayerStreamer)
and re-implements the pass loop for rows that carry a SUFFIX of the prompt instead of a single final token. Copied
from engine.py (to be consolidated later): the vector-building branches of Engine._build_v (zero, layer, expert,
coalition_clean, coalition_union) in _build_v_at, and the overall structure of Engine.run.

Mechanics. A patch at position p and layer l changes positions p..T-1 for every layer >= l. A suffix wavefront row
spawned at (l, p) starts with the parent (noised) run's residuals at positions p..T-1 after layer l, the intervention
applied at p only, and then runs layers l+1..L-1 for those T-p tokens, attending to the parent's K/V at positions < p
and to its own K/V at positions >= p (causal). Delta is read at the row's last token (= the prompt's final position).
Rows are right-padded to the longest suffix; padded tokens are masked out of the MoE (packed) and never read.

Kinds at p (mirroring engine.KINDS at the final position; quantities at position p of the noised (parent) or clean run):
    zero             h[p] = h_pre_moe_noised + bf16(MoE_noised + 0)        null invariant: must equal the noised run
    layer            h[p] = h_pre_moe_noised + bf16(MoE_noised + (MoE_clean - MoE_noised))   the paper's MoE-output patch
    expert           v = c_e^clean(p) - c_e^noised(p), the contribution difference of expert e at p (0 if not routed)
    coalition_clean / coalition_union     as in engine._build_v, at p
    attn_layer       spawned BEFORE the MoE of layer l: h[p] = h_pre_noised + Attn_l_clean; positions p+1..T-1 of the row
                     enter the MoE of layer l with the parent's post-attention residuals (the MoE is per token, so
                     they reproduce the parent's values) and the row continues from l+1
    block            h[p] = (h_pre_noised + Attn_l_clean) + MoE_l_clean
    resid            h[p] = h_out_clean: the clean residual after layer l at p (classic hidden-state restoration)
With p = T-1 (rec_pos = -1) every kind reduces to the final-token machinery of engine.Engine.run, which is used as a
consistency check (scripts/ext5_subject_verify.py). Not supported here (default protocol only): pos_offset, sink_donor.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .engine import BF16, Engine, PrefillSpec, apply_rope, attn_prefill, moe_forward, rmsnorm, rope_tables
from .weights import LayerStreamer

VEC_KINDS = ("zero", "layer", "expert", "coalition_clean", "coalition_union")  # after the MoE: h = pre_moe + bf16(acc + v)
PRE_KINDS = ("attn_layer",)  # after the attention, before the MoE of the spawn layer
DIRECT_KINDS = ("block", "resid")  # after the MoE, residual at p built directly
KINDS = VEC_KINDS + PRE_KINDS + DIRECT_KINDS


# ----------------------------------------------------------------------------------------------------------------
# job specification
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class SubjectPrefill(PrefillSpec):
    rec_pos: int = -1  # position whose per-layer quantities (attention out, MoE out, residuals, routing) are recorded;
    #                    -1 = the final position (then the pass reproduces the final-token machinery)


@dataclass
class SubjectSpawn:
    layer: int
    parent: int  # prefill row whose residuals / K/V are continued (the noised run)
    clean: int  # prefill row of the clean run of the same case (the donor)
    kind: str  # one of KINDS
    pos: int  # patch position p (absolute token index); must equal rec_pos of both prefill rows
    expert: int = -1
    window: int = 1  # ext6 (kind 'layer' only): the MoE output at p is set to the clean one at layers layer..layer+window-1
    #                  (sliding-window restoration, Meng et al. 2022 / Zhang & Nanda 2024); 1 = the single-layer patch


@dataclass
class SubjectResult:
    lens: np.ndarray
    rec_pos: np.ndarray
    logit_true: np.ndarray
    logit_foil: np.ndarray
    logit_true_full: np.ndarray
    logit_foil_full: np.ndarray
    top1: np.ndarray
    route_idx: np.ndarray  # [L, B, k] routing at rec_pos
    route_w: np.ndarray
    route_cnorm: np.ndarray
    sp_logit_true: np.ndarray
    sp_logit_foil: np.ndarray
    sp_vnorm: np.ndarray
    sp_norm_e: np.ndarray
    layer_times: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def delta(self):
        return self.logit_true - self.logit_foil

    @property
    def sp_delta(self):
        return self.sp_logit_true - self.sp_logit_foil


# ----------------------------------------------------------------------------------------------------------------
# suffix attention
# ----------------------------------------------------------------------------------------------------------------
def attn_suffix(qw, kw, vw, K, V, parent, pos, scale: float, chunk: int = 4096) -> torch.Tensor:
    """Suffix rows attend to their parent prefill row's K/V at positions < pos and to their own K/V (causal).

    qw [W, nH, S, D]; kw/vw [W, nKV, S, D] (S = padded suffix length, token j sits at position pos + j); K/V [B, nKV, T, D]
    of the current layer; parent [W] long; pos [W] long. Returns [W, nH, S, D] (bf16; softmax in fp32 as in engine.attn_*).
    Padded suffix tokens always see their own key, so no query row is fully masked.
    """
    W, nH, S, D = qw.shape
    nkv = kw.shape[1]
    rep = nH // nkv
    T = K.shape[2]
    dev = qw.device
    ar_T = torch.arange(T, device=dev)
    own = torch.ones(S, S, dtype=torch.bool, device=dev).tril()  # [S(query), S(key)]
    out = torch.empty_like(qw)
    for s0 in range(0, W, chunk):
        e = min(W, s0 + chunk)
        w = e - s0
        par = parent[s0:e]
        kk = torch.cat([K[par], kw[s0:e]], dim=2)  # [w, nkv, T+S, D]
        vv = torch.cat([V[par], vw[s0:e]], dim=2)
        qg = qw[s0:e].view(w, nkv, rep, S, D)
        scores = torch.matmul(qg, kk[:, :, None].transpose(-1, -2)) * scale  # [w, nkv, rep, S, T+S]
        vp = (ar_T[None, :] < pos[s0:e, None])[:, None, :].expand(w, S, T)  # parent keys before the patch position
        valid = torch.cat([vp, own[None].expand(w, S, S)], dim=-1)  # [w, S, T+S]
        scores = scores.masked_fill(~valid[:, None, None], float("-inf"))
        p = torch.softmax(scores.float(), dim=-1).to(qw.dtype)
        out[s0:e] = torch.matmul(p, vv[:, :, None]).view(w, nH, S, D)
        del kk, vv, scores, p
    return out


# ----------------------------------------------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------------------------------------------
class SubjectEngine(Engine):
    """engine.Engine with a pass that spawns SUFFIX wavefront rows at an arbitrary position of each prompt."""

    def _build_v_at(self, spawns, idx, acc_r, topi_r, contrib, out_ne):
        """Intervention vectors at the recorded position for spawn indices idx (all VEC_KINDS, current layer) -> [n, H] fp32.
        Copied from engine.Engine._build_v (branches zero / layer / expert / coalition_clean / coalition_union)."""
        dev = self.device
        n = len(idx)
        v = torch.zeros(n, self.hidden, dtype=torch.float32, device=dev)
        kinds = np.array([spawns[i].kind for i in idx])
        parent = torch.tensor([spawns[i].parent for i in idx], device=dev)
        clean = torch.tensor([spawns[i].clean for i in idx], device=dev)
        expert = torch.tensor([spawns[i].expert for i in idx], device=dev)
        ne = torch.zeros(n, device=dev)

        def c_of(rows, ex):  # contribution of expert ex in the run of rows (zero if not routed) -> [m, H]
            m = (topi_r[rows] == ex[:, None]).float()
            return (contrib[rows] * m[..., None]).sum(1)

        for kind in np.unique(kinds):
            sel_np = np.nonzero(kinds == kind)[0]
            sel = torch.tensor(sel_np, device=dev)
            p, c, ex = parent[sel], clean[sel], expert[sel]
            if kind == "zero":
                continue
            elif kind == "layer":
                v[sel] = acc_r[c] - acc_r[p]
            elif kind == "expert":
                d = c_of(c, ex) - c_of(p, ex)
                v[sel] = d
                ne[sel] = d.norm(dim=-1)
            elif kind in ("coalition_clean", "coalition_union"):
                tc, tn = topi_r[c], topi_r[p]
                in_clean = (tn[:, :, None] == tc[:, None, :]).any(-1).float()  # noised slots whose expert is clean-routed
                vv = contrib[c].sum(1) - (contrib[p] * in_clean[..., None]).sum(1)
                if kind == "coalition_union":
                    vv = vv - (contrib[p] * (1.0 - in_clean)[..., None]).sum(1)
                v[sel] = vv
            else:
                raise ValueError(kind)
        out_ne[idx] = ne.cpu().numpy()
        return v

    @torch.no_grad()
    def run_subject(self, prefill: list[SubjectPrefill], spawns: list[SubjectSpawn], log=None, wf_chunk: int = 4096,
                    attn_score_budget: int = 768 * 2**20, metrics: bool = False, metrics_chunk: int = 512) -> SubjectResult:
        """ext6 additions (defaults reproduce the ext5 behaviour bit for bit):
        SubjectSpawn.window > 1 (kind 'layer'): at every later layer l' <= layer + window - 1 the row's own MoE output at
            its first token (position p) is replaced by the clean row's MoE output at p (h = h_pre_moe_own + MoE_clean),
            i.e. a joint restoration of `window` consecutive MoE outputs at p.
        metrics=True: full-vocabulary log-softmax (fp32 of the bf16 logits) at the final position for every prefill and
            suffix row -> extra['metrics_prefill'] / extra['metrics_spawn'] with logp_true, logp_foil, p_true, p_foil,
            rank_true (no KL)."""
        s, dev = self.spec, self.device
        Hd = self.hidden
        B = len(prefill)
        lens = np.array([len(p.ids) for p in prefill], dtype=np.int64)
        T = int(lens.max())
        rec = np.array([p.rec_pos if p.rec_pos >= 0 else len(p.ids) - 1 for p in prefill], dtype=np.int64)
        assert (rec < lens).all() and (rec >= 0).all()
        for p in prefill:
            assert getattr(p, "pos_offset", 0) == 0 and getattr(p, "sink_donor", -1) < 0, "not supported in SubjectEngine"
        ids = torch.zeros(B, T, dtype=torch.long)
        for b, p in enumerate(prefill):
            ids[b, : len(p.ids)] = torch.tensor(p.ids, dtype=torch.long)
        ids = ids.to(dev)
        lens_t = torch.tensor(lens, device=dev)
        final_t = lens_t - 1
        rec_t = torch.tensor(rec, device=dev)
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

        # ---- spawns: grouping, suffix geometry
        S = len(spawns)
        by_vec: dict[int, list[int]] = {}
        by_pre: dict[int, list[int]] = {}
        by_direct: dict[int, list[int]] = {}
        sp_pos = np.array([sp.pos for sp in spawns], dtype=np.int64)
        sp_parent = np.array([sp.parent for sp in spawns], dtype=np.int64)
        sp_len = lens[sp_parent] - sp_pos if S else np.zeros(0, dtype=np.int64)  # suffix lengths
        for i, sp in enumerate(spawns):
            assert sp.kind in KINDS, sp.kind
            assert rec[sp.parent] == sp.pos == rec[sp.clean], (i, sp)
            assert lens[sp.parent] == lens[sp.clean]
            assert sp.window >= 1 and (sp.window == 1 or sp.kind == "layer"), (i, sp)
            (by_pre if sp.kind in PRE_KINDS else by_direct if sp.kind in DIRECT_KINDS else by_vec).setdefault(sp.layer, []).append(i)
        has_window = any(sp.window > 1 for sp in spawns)
        Smax = int(sp_len.max()) if S else 1
        Nw_max = int(sp_len.sum())
        wf_H = torch.empty(S, Smax, Hd, dtype=BF16, device=dev)
        wf_parent = torch.empty(S, dtype=torch.long, device=dev)
        wf_pos = torch.empty(S, dtype=torch.long, device=dev)
        wf_len = torch.empty(S, dtype=torch.long, device=dev)
        wf_clean = torch.empty(S, dtype=torch.long, device=dev)  # ext6 window: clean prefill row of each suffix row
        wf_winend = torch.full((S,), -1, dtype=torch.long, device=dev)  # ext6 window: last layer whose MoE output at p is clean
        wf_row_of_spawn = np.full(S, -1, dtype=np.int64)
        n_wf = 0
        sp_ne = np.zeros(S, dtype=np.float32)
        sp_vnorm = np.zeros(S, dtype=np.float32)
        ar_S = torch.arange(Smax, device=dev)

        cos_all, sin_all = rope_tables(T, s.head_dim, s.rope_theta, dev)
        causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        scale = s.head_dim ** -0.5
        score_bytes = B * s.n_heads * T * T * 4
        row_chunk = None if score_bytes <= attn_score_budget else max(1, attn_score_budget // (s.n_heads * T * T * 4))
        rec_idx = ar * T + rec_t
        rec_map = torch.full((B * T + Nw_max,), -1, dtype=torch.long, device=dev)
        rec_map[rec_idx] = ar

        L = s.n_layers
        route_idx = np.zeros((L, B, s.top_k), dtype=np.int32)
        route_w = np.zeros((L, B, s.top_k), dtype=np.float32)
        route_cn = np.zeros((L, B, s.top_k), dtype=np.float32)

        def spawn_rows(idx: list[int]):
            """Parent's suffix residuals (current Hs) for the spawn indices -> ([n, Smax, H] bf16, parents, cleans)."""
            parents = torch.tensor([spawns[i].parent for i in idx], device=dev)
            cleans = torch.tensor([spawns[i].clean for i in idx], device=dev)
            pos = torch.tensor([spawns[i].pos for i in idx], device=dev)
            gidx = (pos[:, None] + ar_S[None, :]).clamp(max=T - 1)  # [n, Smax] (pads clamped; never read)
            return Hs[parents[:, None], gidx], parents, cleans, pos

        def commit(idx: list[int], h_new, parents, pos):
            nonlocal n_wf
            n = len(idx)
            wf_H[n_wf : n_wf + n] = h_new
            wf_parent[n_wf : n_wf + n] = parents
            wf_pos[n_wf : n_wf + n] = pos
            wf_len[n_wf : n_wf + n] = lens_t[parents] - pos
            if has_window:
                wf_clean[n_wf : n_wf + n] = torch.tensor([spawns[i].clean for i in idx], device=dev)
                # window-1 rows get no window end (-1): otherwise a pre-MoE row (attn_layer) spawned at layer l would have
                # its MoE output at p replaced by the clean one at l, i.e. silently become `block` (found by ext7-wino)
                wf_winend[n_wf : n_wf + n] = torch.tensor([spawns[i].layer + spawns[i].window - 1 if spawns[i].window > 1 else -1
                                                           for i in idx], device=dev)
            wf_row_of_spawn[np.array(idx)] = np.arange(n_wf, n_wf + n)
            n_wf += n

        layer_times = []
        t_start = t_prev = time.time()
        for l, w in LayerStreamer(self.store):
            t_loaded = time.time()
            # ---- attention: prefill rows
            x = rmsnorm(Hs, w.ln1, s.rms_eps)
            q, k, v = self._qk(x, w, B, T)
            q = apply_rope(q, cos_all[:T], sin_all[:T])
            k = apply_rope(k, cos_all[:T], sin_all[:T])
            o = attn_prefill(q, k, v, causal, scale, None, row_chunk)
            o = F.linear(o.transpose(1, 2).reshape(B, T, s.n_heads * s.head_dim), w.wo)
            attn_r = o[ar, rec_t]  # attention-sublayer output at the recorded position, bf16 [B, H]
            pre_attn_r = Hs[ar, rec_t]  # residual entering layer l at the recorded position
            Hs = Hs + o
            # ---- attention: suffix wavefront rows (parent K/V of this layer for positions < pos, own K/V for >= pos)
            if n_wf > 0:
                hw = wf_H[:n_wf]
                xw = rmsnorm(hw, w.ln1, s.rms_eps)
                qw, kw, vw = self._qk(xw, w, n_wf, Smax)
                rpos = (wf_pos[:n_wf, None] + ar_S[None, :]).clamp(max=T - 1)  # [W, Smax]
                cw, sw = cos_all[rpos][:, None], sin_all[rpos][:, None]  # [W, 1, Smax, D]
                qw = apply_rope(qw, cw, sw)
                kw = apply_rope(kw, cw, sw)
                ow = attn_suffix(qw, kw, vw, k, v, wf_parent[:n_wf], wf_pos[:n_wf], scale, chunk=wf_chunk)
                ow = F.linear(ow.transpose(1, 2).reshape(n_wf, Smax, s.n_heads * s.head_dim), w.wo)
                wf_H[:n_wf] = hw + ow
                del hw, xw, qw, kw, vw, ow, cw, sw, rpos
            # ---- attn_layer spawns (before the MoE): h[p] = h_pre_noised + Attn_l_clean, rest of the suffix = parent's
            if l in by_pre:
                idx = by_pre[l]
                h_new, parents, cleans, pos = spawn_rows(idx)
                h_new[:, 0] = pre_attn_r[parents] + attn_r[cleans]
                sp_vnorm[idx] = (attn_r[cleans].float() - attn_r[parents].float()).norm(dim=-1).cpu().numpy()
                commit(idx, h_new, parents, pos)
                del h_new
            del q, k, v, o, x
            # ---- MoE (prefill tokens + packed valid suffix tokens)
            x2 = rmsnorm(Hs, w.ln2, s.rms_eps).view(B * T, Hd)
            if n_wf > 0:
                vm = ar_S[None, :] < wf_len[:n_wf, None]  # [W, Smax] valid suffix tokens
                x2 = torch.cat([x2, rmsnorm(wf_H[:n_wf], w.ln2, s.rms_eps)[vm]], 0)
            N = x2.shape[0]
            out_bf, acc, topi, topv, contrib = moe_forward(x2, w, s, rec_map[:N], B)[:5]
            pre_moe_r = Hs[ar, rec_t]  # pre-MoE residual at the recorded position (bf16)
            Hs = Hs + out_bf[: B * T].view(B, T, Hd)
            if n_wf > 0:
                win = torch.nonzero(wf_winend[:n_wf] >= l).view(-1) if has_window else None  # rows spawned below l
                if win is not None and len(win):
                    pre0 = wf_H[win, 0].clone()  # pre-MoE residual at the row's first token (position p)
                wf_H[:n_wf][vm] += out_bf[B * T :]
                if win is not None and len(win):  # ext6 window: MoE output at p := the clean row's at this layer
                    wf_H[win, 0] = pre0 + out_bf[rec_idx][wf_clean[win]]
                    del pre0
            acc_r, topi_r, topv_r = acc[rec_idx], topi[rec_idx], topv[rec_idx]
            out_r = out_bf[rec_idx]
            h_out_r = Hs[ar, rec_t]  # residual after layer l at the recorded position
            route_idx[l] = topi_r.cpu().numpy()
            route_w[l] = topv_r.cpu().numpy()
            route_cn[l] = contrib.norm(dim=-1).cpu().numpy()
            # ---- vector kinds (after the MoE): h[p] = pre_moe_noised + bf16(acc_noised + v)
            if l in by_vec:
                idx = by_vec[l]
                vvec = self._build_v_at(spawns, idx, acc_r, topi_r, contrib, sp_ne)
                sp_vnorm[idx] = vvec.norm(dim=-1).cpu().numpy()
                h_new, parents, cleans, pos = spawn_rows(idx)
                h_new[:, 0] = pre_moe_r[parents] + (acc_r[parents] + vvec).to(BF16)
                commit(idx, h_new, parents, pos)
                del h_new, vvec
            # ---- direct kinds: block = (h_pre_noised + Attn_clean) + MoE_clean; resid = h_out_clean
            if l in by_direct:
                idx = by_direct[l]
                kinds_d = np.array([spawns[i].kind for i in idx])
                h_new, parents, cleans, pos = spawn_rows(idx)
                vn = torch.zeros(len(idx), dtype=torch.float32, device=dev)
                sel_np = np.nonzero(kinds_d == "block")[0]
                if len(sel_np):
                    sel = torch.tensor(sel_np, device=dev)
                    p_, c_ = parents[sel], cleans[sel]
                    h_new[sel, 0] = (pre_attn_r[p_] + attn_r[c_]) + out_r[c_]
                    vn[sel] = ((attn_r[c_].float() - attn_r[p_].float()) + (acc_r[c_] - acc_r[p_])).norm(dim=-1)
                sel_np = np.nonzero(kinds_d == "resid")[0]
                if len(sel_np):
                    sel = torch.tensor(sel_np, device=dev)
                    p_, c_ = parents[sel], cleans[sel]
                    h_new[sel, 0] = h_out_r[c_]
                    vn[sel] = (h_out_r[c_].float() - h_out_r[p_].float()).norm(dim=-1)
                sp_vnorm[idx] = vn.cpu().numpy()
                commit(idx, h_new, parents, pos)
                del h_new, vn
            del x2, out_bf, acc, topi, topv, contrib, acc_r, topi_r, topv_r, out_r, h_out_r, pre_moe_r, attn_r, pre_attn_r
            torch.cuda.synchronize()
            t_done = time.time()
            layer_times.append((l, t_loaded - t_prev, t_done - t_loaded))
            t_prev = t_done
            if log is not None and (l % 8 == 0 or l == L - 1):
                log(f"  layer {l:3d}: load-wait {layer_times[-1][1]:.2f}s compute {layer_times[-1][2]:.2f}s  wf_rows={n_wf} wf_tokens={N - B * T}")
        del w
        # ---- head
        head, norm = self.g["head"], self.g["norm"]
        true_ids = torch.tensor([p.true_id for p in prefill], device=dev)
        foil_ids = torch.tensor([p.foil_id for p in prefill], device=dev)
        hN = rmsnorm(Hs[ar, final_t], norm, s.rms_eps)
        top1 = torch.empty(B, dtype=torch.long, device=dev)
        lt_full = torch.empty(B, dtype=torch.float32, device=dev)
        lf_full = torch.empty(B, dtype=torch.float32, device=dev)
        for c0 in range(0, B, 512):
            lg = F.linear(hN[c0 : c0 + 512], head)
            top1[c0 : c0 + 512] = lg.argmax(-1)
            arb = torch.arange(lg.shape[0], device=dev)
            lt_full[c0 : c0 + 512] = lg[arb, true_ids[c0 : c0 + 512]].float()
            lf_full[c0 : c0 + 512] = lg[arb, foil_ids[c0 : c0 + 512]].float()
        lt = self._pair_logits(hN, head, true_ids)
        lf = self._pair_logits(hN, head, foil_ids)
        sp_lt = np.zeros(S, dtype=np.float32)
        sp_lf = np.zeros(S, dtype=np.float32)
        if n_wf > 0:
            arw = torch.arange(n_wf, device=dev)
            hw = rmsnorm(wf_H[arw, wf_len[:n_wf] - 1], norm, s.rms_eps)  # each row's last token = the prompt's final position
            par = wf_parent[:n_wf]
            wlt = self._pair_logits(hw, head, true_ids[par]).cpu().numpy()
            wlf = self._pair_logits(hw, head, foil_ids[par]).cpu().numpy()
            sp_lt[:] = wlt[wf_row_of_spawn]
            sp_lf[:] = wlf[wf_row_of_spawn]
        extra_m = {}
        if metrics:  # ext6: full-vocabulary probabilities of the two objects at the final position
            t_m = time.time()
            extra_m["metrics_prefill"] = self._logp_pair(hN, head, true_ids, foil_ids, metrics_chunk)
            if n_wf > 0:
                mw = self._logp_pair(hw, head, true_ids[par], foil_ids[par], metrics_chunk)
                extra_m["metrics_spawn"] = {k_: v_[wf_row_of_spawn] for k_, v_ in mw.items()}
            extra_m["metrics_s"] = time.time() - t_m
        total = time.time() - t_start
        if log is not None:
            log(f"  pass done: {B} prefill rows (T={T}), {S} suffix rows (Smax={Smax}, {Nw_max} tokens), {total:.1f}s")
        return SubjectResult(
            lens=lens, rec_pos=rec, logit_true=lt.cpu().numpy(), logit_foil=lf.cpu().numpy(),
            logit_true_full=lt_full.cpu().numpy(), logit_foil_full=lf_full.cpu().numpy(), top1=top1.cpu().numpy(),
            route_idx=route_idx, route_w=route_w, route_cnorm=route_cn,
            sp_logit_true=sp_lt, sp_logit_foil=sp_lf, sp_vnorm=sp_vnorm, sp_norm_e=sp_ne, layer_times=layer_times,
            extra={"total_s": total, "T": T, "Smax": Smax, "n_wf_tokens": Nw_max, "row_chunk": row_chunk, **extra_m},
        )

    @staticmethod
    def _logp_pair(h: torch.Tensor, head: torch.Tensor, true_ids: torch.Tensor, foil_ids: torch.Tensor, chunk: int) -> dict:
        """ext6: log-softmax (fp32 of the bf16 logits, as Engine._metrics) of the rows h [n, H] (final-normed) at the true
        and foil ids, their probabilities and the rank of the true token (1 + number of strictly larger logits)."""
        n = h.shape[0]
        dev = h.device
        lp_t = torch.empty(n, dtype=torch.float32, device=dev)
        lp_f = torch.empty(n, dtype=torch.float32, device=dev)
        rank = torch.empty(n, dtype=torch.int32, device=dev)
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            arb = torch.arange(c1 - c0, device=dev)
            lp = torch.log_softmax(F.linear(h[c0:c1], head).float(), dim=-1)
            t_ = lp[arb, true_ids[c0:c1]]
            lp_t[c0:c1] = t_
            lp_f[c0:c1] = lp[arb, foil_ids[c0:c1]]
            rank[c0:c1] = (lp > t_[:, None]).sum(-1).to(torch.int32) + 1
            del lp
        return {"logp_true": lp_t.cpu().numpy(), "logp_foil": lp_f.cpu().numpy(), "p_true": lp_t.exp().cpu().numpy(),
                "p_foil": lp_f.exp().cpu().numpy(), "rank_true": rank.cpu().numpy()}


# ----------------------------------------------------------------------------------------------------------------
# job helpers shared by the scripts
# ----------------------------------------------------------------------------------------------------------------
def last_subject_pos(case) -> int:
    return int(case.subject_pos[-1])


def subject_prefill_rows(cases: dict, ids: list[int], sigma: float, hidden: int, pos_of, noise_draw,
                         extra_noise_rows: bool = True) -> tuple[list[SubjectPrefill], dict[str, int]]:
    """Prefill rows for the subject-site protocol, in blocks of len(ids):
        clean | noised (whole subject span) | [noised_lastonly (same draw, last subject token only) | noised_exceptlast]
    All rows of a case record at pos_of(case). Returns (rows, block offsets)."""
    n = len(ids)
    rows = [SubjectPrefill(cases[c].ids, cases[c].true_id, cases[c].foil_id, rec_pos=pos_of(cases[c])) for c in ids]
    offs = {"clean": 0, "noised": n}
    for c in ids:
        cs = cases[c]
        rows.append(SubjectPrefill(cs.ids, cs.true_id, cs.foil_id, cs.subject_pos,
                                   noise_draw(c, len(cs.subject_pos), hidden, sigma), rec_pos=pos_of(cs)))
    if extra_noise_rows:
        offs["noised_lastonly"] = 2 * n
        for c in ids:
            cs = cases[c]
            eps = noise_draw(c, len(cs.subject_pos), hidden, sigma)
            rows.append(SubjectPrefill(cs.ids, cs.true_id, cs.foil_id, [cs.subject_pos[-1]], eps[-1:], rec_pos=pos_of(cs)))
        offs["noised_exceptlast"] = 3 * n
        for c in ids:
            cs = cases[c]
            eps = noise_draw(c, len(cs.subject_pos), hidden, sigma)
            npos = list(cs.subject_pos[:-1])
            rows.append(SubjectPrefill(cs.ids, cs.true_id, cs.foil_id, npos if npos else None, eps[:-1] if npos else None,
                                       rec_pos=pos_of(cs)))
    return rows, offs
