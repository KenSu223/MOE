"""ext13wr (Phase 5, agent p5-expert): Direction-11 writer analysis for every Phase-5 condition.

Writer vs computer (Phase 4, moetrace/ext11_writer.py) carried to the Phase-5 conditions. For a single-expert STR patch at
layer l on directed case c (clean prompt X, corrupted prompt = the twin Y; final position):
    total     T = rescue of the single-expert patch (p5-data's expert rows, results/<run>/str_expert_rows.parquet)
    direct    D = DLA of delta_e = c_e(X) - c_e(Y) onto W_U[r] - W_U[r'], final norm frozen at the corrupted run Y
              (ext8 convention). From ONE prefill pass over both twins of every pair (DiagSpec.contrib_dla): with
              cdla_P[l, e] = (c_e(P) * gamma) . (W_U[true_P] - W_U[foil_P]) (0 if e is not routed at P's final position)
              and the twins' readouts swapped (true_Y = foil_X), D = (cdla_X[l, e] + cdla_Y[l, e]) / rms_Y.
    indirect  I = T - D.
Part B (vocabulary projections, same pass, DiagSpec.contrib_final_vectors at the layers of interest): clean vectors c_e
(final norm frozen at the prompt's own scale) and delta vectors delta_e (frozen at the corrupted twin) of the targets and the
condition's population top-10 projected on the unembedding: rank of the right answer token, top promoted tokens, class
means / top-50 shares for the condition's own attribute vocabulary (trigger words) and entity vocabulary (answers), Phase
3's WinoGrande trigger vocabulary, IOI names and CounterFact objects; exact final-RMSNorm direct effect vs frozen DLA.

Raw vectors of the experts of interest: /opt/dlami/nvme/moe_p5wr/<proto><suffix>/<cond>/vectors.pt.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from .models import MODELS, RESULTS
from . import ext13ko as K

ROOT = "/home/ubuntu/MOE"
RAW = "/opt/dlami/nvme/moe_p5wr"
TOPN = 20


def out_dir(e: dict, suffix: str = "") -> str:
    return os.path.join(RESULTS, f"p5_{e['proto']}_wr{suffix}", e["cond"])


def raw_dir(e: dict, suffix: str = "") -> str:
    return os.path.join(RAW, f"{e['proto']}{suffix}", e["cond"])


def pass_done(e: dict, suffix: str = "") -> bool:
    od = out_dir(e, suffix)
    return os.path.exists(os.path.join(od, "wr_prefill.npz")) and os.path.exists(os.path.join(od, "vec_rows.parquet"))


def interest(e: dict, suffix: str = "") -> list:
    """Targets (selected, WinoGrande, CounterFact experts) + the condition's population top-10."""
    kod = K.out_dir(e, suffix)
    tg = K.targets(e, kod)
    pop = [(l, x) for l, x, _ in K.pop_ranking(e, kod)[:10]]
    return list(dict.fromkeys(tg["all"] + pop))


# ------------------------------------------------------------------------------------------------------------------
# token classes
# ------------------------------------------------------------------------------------------------------------------
def first_cont_token(tok, word: str) -> Optional[int]:
    from .ext11_writer import first_cont_token as f
    return f(tok, word)


def classes_for(e: dict, it: pd.DataFrame, tok) -> dict:
    """Token-id sets: attr_own (the condition's trigger words; + the readout ids if the readout is the attribute),
    ent_own (the condition's answers; + the readout ids if the readout is the entity), wg_trigger (Phase-3 trigger
    vocabulary), names (IOI names), cf_any (CounterFact objects)."""
    from .ext11_writer import vocab_classes
    vc = vocab_classes(e["model"], tok)
    words = set()
    ents = set()
    for c in ("word_a", "word_b"):
        if c in it.columns:
            words |= set(str(w) for w in it[c].dropna())
    for c in ("ans_a", "ans_b"):
        if c in it.columns:
            ents |= set(str(w) for w in it[c].dropna())
    attr = {first_cont_token(tok, w) for w in words}
    ent = {first_cont_token(tok, w) for w in ents}
    attr.discard(None)
    ent.discard(None)
    ro = set(int(x) for x in it.true_id) | set(int(x) for x in it.foil_id)
    if e["direction"] == "a2e":
        ent |= ro
    else:
        attr |= ro
    return {"attr_own": attr, "ent_own": ent, "wg_trigger": vc["wg_trigger"], "names": vc["ioi_names"], "cf_any": vc["cf_any"]}


# ------------------------------------------------------------------------------------------------------------------
# GPU pass
# ------------------------------------------------------------------------------------------------------------------
def run_pass(eng, E, e: dict, suffix: str = "", limit: Optional[int] = None, log=print) -> None:
    """One prefill pass over every clean prompt of the condition (both twins of all pairs) with contrib_dla, contrib final
    vectors at the layers of interest and the final residual; then the vocabulary projections on the GPU."""
    import torch
    od, rd = out_dir(e, suffix), raw_dir(e, suffix)
    os.makedirs(od, exist_ok=True)
    os.makedirs(rd, exist_ok=True)
    it = K.build_items(e, od, limit=limit)
    ex = interest(e, suffix)
    layers = tuple(sorted({l for l, _ in ex}))
    meta = {"cond": e["cond"], "proto": e["proto"], "model": e["model"], "repo": MODELS[e["model"]]["repo"], "agent": "p5-expert",
            "experiment": "ext13wr writer pass (contrib_dla, contrib_final_vectors, resid_final), Phase 5 / Direction 11",
            "source": {"pairs": e["pairs"], "case_sets": e["case_sets"], "run": e["run"]},
            "experts_of_interest": [K.le(*x) for x in ex], "layers": list(layers), "n_prompts": int(len(it)),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False, "dtype": "bf16"}
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    rawp = os.path.join(rd, "vectors.pt")
    if os.path.exists(rawp):
        raw = torch.load(rawp)
    else:
        pre = [E.PrefillSpec(list(r.ids), int(r.true_id), int(r.foil_id)) for r in it.itertuples()]
        diag = E.DiagSpec(contrib_dla=True, contrib_final_vectors=layers, resid_final=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log(f"{e['cond']}/{e['proto']} writer pass: {len(pre)} prompts, T <= {it['T'].max()}, layers {list(layers)}")
        res = eng.run(pre, [], record_routing=True, diag=diag, metrics=True)
        dg = res.extra["diag"]
        L = res.route_idx.shape[0]
        np.savez_compressed(os.path.join(od, "wr_prefill.npz"), items=it.item.values, route_idx=res.route_idx.astype(np.int16),
                            route_w=res.route_w.astype(np.float16), contrib_dla=dg["contrib_dla"].astype(np.float32),
                            final_rms=dg["final_rms"].astype(np.float32), delta=np.asarray(res.delta, dtype=np.float32),
                            logit_true=np.asarray(res.logit_true, dtype=np.float32),
                            logit_foil=np.asarray(res.logit_foil, dtype=np.float32), top1=res.top1.astype(np.int64),
                            rank_true=np.asarray(res.metrics_prefill["rank_true"]) if res.metrics_prefill is not None else np.zeros(0))
        vec = {}
        for (l, x) in ex:
            hit = res.route_idx[l] == x  # [B, k]
            b_idx, s_idx = np.nonzero(hit)
            cv = dg["contrib_final_vectors"][int(l)]
            vec[(int(l), int(x))] = (torch.from_numpy(b_idx.astype(np.int64)), cv[torch.from_numpy(b_idx), torch.from_numpy(s_idx)].clone())
        raw = {"vec": vec, "h_final": dg["resid_final"][L - 1].clone(), "final_rms": torch.from_numpy(dg["final_rms"]),
               "route_idx": torch.from_numpy(res.route_idx.astype(np.int16)), "pass_s": res.extra.get("total_s"),
               "peak_GB": torch.cuda.max_memory_allocated() / 2**30}
        torch.save(raw, rawp)
        log(f"{e['cond']} writer pass done in {raw['pass_s']:.0f}s, peak {raw['peak_GB']:.1f} GB")
        del res, dg
    from transformers import AutoTokenizer
    from .arch import snapshot_dir
    tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[e["model"]]["repo"]))
    classes = classes_for(e, it, tok)
    vr, vt = project(raw, it, ex, eng.g["head"], eng.g["norm"], eng.spec.rms_eps, classes, e["direction"])
    vr.to_parquet(os.path.join(od, "vec_rows.parquet"), index=False)
    vt.to_parquet(os.path.join(od, "vec_top.parquet"), index=False)
    o = it.drop(columns=["ids"]).copy()
    o["ids_json"] = it.ids.map(json.dumps)
    o.to_parquet(os.path.join(od, "prompts.parquet"), index=False)
    meta.update({"complete": True, "n_vectors": int(len(vr)), "pass_s": raw.get("pass_s"), "peak_GB": raw.get("peak_GB"),
                 "class_sizes": {k: len(v) for k, v in classes.items()},
                 "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    json.dump(meta, open(os.path.join(od, "run_meta.json"), "w"), indent=1)
    json.dump({k: sorted(int(x) for x in v) for k, v in classes.items()}, open(os.path.join(od, "classes.json"), "w"))
    log(f"{e['cond']} writer projections: {len(vr)} vectors")


def vector_list(raw: dict, it: pd.DataFrame, ex: list) -> list:
    """(kind, prompt, ref, layer, expert, true, foil, vector): clean c_e of every prompt routing e at the final position;
    delta = c_e(X) - c_e(Y) for every directed case whose clean prompt X routes e (Y = the twin)."""
    twin = {}
    by_pair = it.groupby("pair_idx").item.apply(list).to_dict()
    for pi, items in by_pair.items():
        if len(items) == 2:
            twin[items[0]], twin[items[1]] = items[1], items[0]
    tr = it.true_id.values
    fo = it.foil_id.values
    out = []
    for (l, x) in ex:
        b_idx, V = raw["vec"][(int(l), int(x))]
        pos = {int(b): j for j, b in enumerate(b_idx.tolist())}
        for b, j in pos.items():
            out.append(("clean", b, b, l, x, int(tr[b]), int(fo[b]), V[j]))
            y = twin.get(b)
            if y is None:
                continue
            vy = V[pos[y]] if y in pos else 0.0
            out.append(("delta", b, y, l, x, int(tr[b]), int(fo[b]), V[j] - vy))
    return out


def project(raw: dict, it: pd.DataFrame, ex: list, head, gamma, eps: float, classes: dict, direction: str,
            dev: str = "cuda"):
    import torch
    vecs = vector_list(raw, it, ex)
    Wf = head.float()
    g = gamma.float().to(dev)
    cls_idx = {k: torch.tensor(sorted(v), device=dev, dtype=torch.long) for k, v in classes.items() if len(v)}
    own_cls = "ent_own" if direction == "a2e" else "attr_own"
    hf = raw["h_final"].float().to(dev)
    rms = torch.sqrt(hf.pow(2).mean(-1) + eps)
    rows, tops = [], []
    B = 256
    for c0 in range(0, len(vecs), B):
        chunk = vecs[c0:c0 + B]
        v = torch.stack([torch.as_tensor(x[7], dtype=torch.float32) for x in chunk]).to(dev)
        ref = torch.tensor([x[2] for x in chunk], device=dev)
        own = torch.tensor([x[1] for x in chunk], device=dev)
        trv = torch.tensor([x[5] for x in chunk], device=dev)
        fov = torch.tensor([x[6] for x in chunk], device=dev)
        is_delta = torch.tensor([x[0] == "delta" for x in chunk], device=dev)
        nref = torch.where(is_delta, ref, own)
        lg = ((v * g) @ Wf.T) / rms[nref][:, None]
        ar = torch.arange(len(chunk), device=dev)
        vt_, vf_ = lg[ar, trv], lg[ar, fov]
        rank_t = (lg > vt_[:, None]).sum(1) + 1
        rank_f = (lg > vf_[:, None]).sum(1) + 1
        rlow_f = (lg < vf_[:, None]).sum(1) + 1
        mu, sd = lg.mean(1), lg.std(1)
        tp_v, tp_i = lg.topk(TOPN, dim=1)
        bt_v, bt_i = (-lg).topk(TOPN, dim=1)
        top50 = lg.topk(50, dim=1).indices
        cl = {}
        for k, idx in cls_idx.items():
            m = lg[:, idx].mean(1)
            cl[f"cls_{k}"] = (m - mu).cpu().numpy()
            cl[f"clsz_{k}"] = ((m - mu) / sd).cpu().numpy()
            cl[f"top50_{k}"] = torch.isin(top50, idx).float().mean(1).cpu().numpy()
        rk_in = np.full(len(chunk), np.nan)
        n_in = np.full(len(chunk), np.nan)
        if own_cls in cls_idx:
            idx = cls_idx[own_cls]
            vals = lg[:, idx]
            rk_in = ((vals > vt_[:, None]).sum(1) + 1).float().cpu().numpy()
            n_in[:] = len(idx)
        u = g * (Wf[trv] - Wf[fov])

        def Dfun(h):
            return ((h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)) * u).sum(-1)
        hr = hf[nref]
        exact = torch.where(is_delta, Dfun(hr + v) - Dfun(hr), Dfun(hr) - Dfun(hr - v))
        frozen = (v * u).sum(-1) / rms[nref]
        for j, x in enumerate(chunk):
            rows.append(dict(kind=x[0], prompt=x[1], ref=x[2], layer=x[3], expert=x[4], true_id=x[5], foil_id=x[6],
                             val_true=float(vt_[j]), val_foil=float(vf_[j]), rank_true=int(rank_t[j]), rank_foil=int(rank_f[j]),
                             rank_low_foil=int(rlow_f[j]), mean=float(mu[j]), sd=float(sd[j]), vnorm=float(v[j].norm()),
                             exact=float(exact[j]), frozen=float(frozen[j]), rank_true_in_class=float(rk_in[j]),
                             n_class=float(n_in[j]), **{k: float(a[j]) for k, a in cl.items()}))
            tops.append(dict(vec=c0 + j, top_ids=tp_i[j].cpu().numpy().astype(np.int32).tolist(),
                             top_vals=tp_v[j].cpu().numpy().round(4).tolist(),
                             bot_ids=bt_i[j].cpu().numpy().astype(np.int32).tolist(),
                             bot_vals=(-bt_v[j]).cpu().numpy().round(4).tolist()))
        del lg
    return pd.DataFrame(rows), pd.DataFrame(tops)


# ------------------------------------------------------------------------------------------------------------------
# Part A (CPU): total vs direct per (directed case, layer, expert)
# ------------------------------------------------------------------------------------------------------------------
def pair_table(e: dict, suffix: str = "") -> pd.DataFrame:
    """One row per (directed case, clean-active expert) of p5-data's expert rows that is also routed at the clean
    prompt's final position in the writer pass: T, D, I, drop, split (discovery / validation / rep_discovery /
    rep_validation), unit, case_id, row_id (= case_id), first (True). Columns match ext11_writer.pair_table, so the ext11
    statistics (layer_profile, expert_stats, set_stats, band_agreement, ranking_agreement) apply unchanged."""
    od = out_dir(e, suffix)
    z = np.load(os.path.join(od, "wr_prefill.npz"))
    it = pd.read_parquet(os.path.join(od, "prompts.parquet"))
    ri, cd, fr = z["route_idx"], z["contrib_dla"], z["final_rms"]
    item_of_case = dict(zip(it.case_id.astype(int), it.item.astype(int)))
    er = pd.read_parquet(os.path.join(RESULTS, e["run"], "str_expert_rows.parquet"),
                         columns=["case_id", "slot", "layer", "kind", "expert", "rescue", "delta_clean", "delta_corrupt", "clean_active"])
    er = er[(er.kind == "expert") & er.clean_active & er.case_id.isin(item_of_case)]
    er = er.groupby(["case_id", "layer", "expert"], as_index=False).agg(T=("rescue", "mean"), dc=("delta_clean", "mean"),
                                                                         dk=("delta_corrupt", "mean"))
    X = er.case_id.map(item_of_case).values.astype(int)
    Y = (er.case_id.values ^ 1)
    Y = np.array([item_of_case.get(int(c), -1) for c in Y])
    keep = Y >= 0
    er, X, Y = er[keep].reset_index(drop=True), X[keep], Y[keep]
    l = er.layer.values.astype(int)
    x = er.expert.values.astype(int)
    hitX = ri[l, X] == x[:, None]
    hitY = ri[l, Y] == x[:, None]
    cX = np.where(hitX.any(1), (cd[l, X] * hitX).sum(1), np.nan)
    cY = (cd[l, Y] * hitY).sum(1)
    er["D"] = (cX + cY) / fr[Y]
    er["routed_wr"] = hitX.any(1)
    er["I"] = er["T"] - er["D"]
    er["drop"] = er["dc"] - er["dk"]
    sp = it.set_index("case_id")
    er["split"] = er.case_id.map(sp.split).values
    er["unit"] = er.case_id.map(sp.unit).values
    er["row_id"] = er.case_id.values
    er["first"] = True
    return er


def match_rate(m: pd.DataFrame) -> float:
    return float(m.routed_wr.mean()) if len(m) else float("nan")
