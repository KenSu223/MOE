"""ext11 Part C analysis (CPU): routing of the target experts outside their own STR rows.

Reads results/<run>/prompts_input.parquet and the raw chunks in /opt/dlami/nvme/moe_ext11/<run>/ written by
scripts/ext11_writer_routing.py; writes results/<run>/{final_routing.parquet, token_events.parquet},
results/tables/ext11_C_*.md|csv and key "C" of results/ext11_writer_summary.json.

Usage: python scripts/ext11_writer_routing_analyze.py [--runs qwen3:qwen3_writer_routing,mixtral:mixtral_bos_writer_routing]
"""
import argparse, glob, json, os, re, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
import numpy as np, pandas as pd
from moetrace.models import MODELS, RESULTS
from moetrace import ext11_writer as W

TAB = os.path.join(RESULTS, "tables")
RAW = "/opt/dlami/nvme/moe_ext11"
MLAB = {"qwen3": "Qwen3-30B-A3B-Base", "mixtral": "Mixtral-8x7B (BOS)"}
SET_LABEL = {"wino_main": "WinoGrande main (512 prompts)", "wino_local": "WinoGrande local prompts (main pairs)",
             "wino_pool": "WinoGrande margin pool (other pairs)", "wino_pool_local": "local prompts (pool pairs)",
             "cf_str": "CounterFact STR (clean + donors)", "cf_scan": "CounterFact scan (1,024 clean)", "ioi": "IOI clean",
             "wiki": "wikitext-103 (final token of windows)"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pieces_of(tok, model, ids):
    toks = tok.convert_ids_to_tokens([int(i) for i in ids])
    if model == "mixtral":
        out = []
        for t in toks:
            if t.startswith("<0x") and t.endswith(">"):
                try:
                    out.append(bytes([int(t[3:-1], 16)]).decode("latin-1"))
                except Exception:
                    out.append(" ")
            else:
                out.append(t.replace("▁", " "))
        return out
    return [tok.convert_tokens_to_string([t]) for t in toks]


def load(run):
    od = os.path.join(RESULTS, run)
    meta = json.load(open(os.path.join(od, "run_meta.json")))
    P = pd.read_parquet(os.path.join(od, "prompts_input.parquet"))
    P["ids"] = P.ids.map(json.loads)
    files = sorted(glob.glob(os.path.join(RAW, run, "chunk_*.npz")))
    assert len(files) == meta["n_chunks"], (len(files), meta["n_chunks"])
    ex = [W.parse_ename(e) for e in meta["experts_of_interest"]]
    N = len(P)
    first = np.load(files[0])
    L, _, k = first["route_idx"].shape
    R = np.full((N, L, k), -1, np.int16)
    RW = np.zeros((N, L, k), np.float32)
    DLA = np.zeros((N, L, k), np.float32)
    delta = np.full(N, np.nan)
    top1 = np.full(N, -1)
    rms = np.full(N, np.nan)
    tok_rows = []  # per chunk: dict of flat arrays
    for f in files:
        z = np.load(f)
        idx = z["idx"]
        lens = z["lens"]
        R[idx] = z["route_idx"].transpose(1, 0, 2)
        RW[idx] = z["route_w"].transpose(1, 0, 2)
        DLA[idx] = (z["contrib_dla"] / z["final_rms"][None, :, None]).transpose(1, 0, 2)
        delta[idx], top1[idx], rms[idx] = z["delta"], z["top1"], z["final_rms"]
        B = len(idx)
        Tm = z[f"topi_{meta['route_all_layers'][0]}"].shape[1]
        mask = np.arange(Tm)[None, :] < lens[:, None]
        bb, tt = np.nonzero(mask)
        d = {"prompt": idx[bb], "pos": tt}
        for (l, e) in ex:
            topi = z[f"topi_{l}"]
            cols = list(z[f"cols_{l}"])
            j = cols.index(e)
            d[f"r_{l}_{e}"] = (topi[bb, tt] == e).any(-1)
            d[f"p_{l}_{e}"] = z[f"prob_{l}"][bb, tt, j].astype(np.float32)
            d[f"k_{l}_{e}"] = z[f"rank_{l}"][bb, tt, j]
        tok_rows.append(d)
    TK = {c: np.concatenate([d[c] for d in tok_rows]) for c in tok_rows[0]}
    order = np.lexsort((TK["pos"], TK["prompt"]))
    TK = {c: v[order] for c, v in TK.items()}
    ids_flat = np.concatenate([np.asarray(P.ids.iloc[i], dtype=np.int64) for i in range(N)])
    pr = np.repeat(np.arange(N), P.n_tok.values)
    assert (pr == TK["prompt"]).all(), "token table does not align with the prompt ids"
    TK["tok"] = ids_flat
    P["delta"], P["top1"], P["final_rms"] = delta, top1, rms
    return meta, P, ex, R, RW, DLA, TK


def final_active(R, DLA, RW, ex):
    """[N, n_ex] active at the final position, DLA and weight of the expert there (NaN if not active)."""
    N = R.shape[0]
    A = np.zeros((N, len(ex)), bool)
    D = np.full((N, len(ex)), np.nan)
    Wt = np.full((N, len(ex)), np.nan)
    for j, (l, e) in enumerate(ex):
        hit = R[:, l, :] == e
        A[:, j] = hit.any(-1)
        D[A[:, j], j] = (DLA[:, l, :] * hit).sum(-1)[A[:, j]]
        Wt[A[:, j], j] = (RW[:, l, :] * hit).sum(-1)[A[:, j]]
    return A, D, Wt


def rate_ci(x, units, n_boot=1000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"v": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    u, Wm = W.boot_matrix(units, n_boot)
    df = pd.DataFrame(dict(unit=units, x=x, one=1.0))
    S = W.unit_sums(df, ["x", "one"], u)
    r = W.ratio_ci(S[:, 0], S[:, 1], Wm)
    r["n"] = int(len(x))
    return r


def fmt_r(r, pct=True, d=1):
    if r.get("n", 1) == 0 or not np.isfinite(r["v"]):
        return "n/a"
    if pct:
        return f"{100 * r['v']:.{d}f} [{100 * r['lo']:.{d}f}, {100 * r['hi']:.{d}f}]"
    return f"{r['v']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"


def units_of(P):
    u = np.where(P.pair_id.notna() & P.set.str.startswith("wino"), "wp:" + P.pair_id.astype(str), "p:" + P.idx.astype(str))
    u = np.where(P.set == "cf_str", "cf:" + P.case_id.astype("Int64").astype(str), u)
    u = np.where(P.set == "wiki", "w:" + P.window.astype("Int64").astype(str), u)
    return u


def analyse(model, run):
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    t0 = time.time()
    meta, P, ex, R, RW, DLA, TK = load(run)
    names = [W.ename(x) for x in ex]
    log(f"{run}: {len(P)} prompts, {len(TK['tok'])} tokens loaded in {time.time() - t0:.0f}s")
    tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[model]["repo"]))
    A, D, Wt = final_active(R, DLA, RW, ex)
    units = units_of(P)
    targets = [W.ename(x) for x in W.model_targets(model)]
    own = {"wino": [W.ename(x) for x in W.TARGETS[model]["wino"]], "cf": [W.ename(x) for x in W.TARGETS[model]["cf"]]}
    out = {"model": model, "run": run, "experts": names, "targets": targets, "own": own, "sets": P.set.value_counts().to_dict()}
    # ---- compact tables into the run dir
    fr = []
    for j, (l, e) in enumerate(ex):
        a = A[:, j]
        fr.append(pd.DataFrame(dict(idx=P.idx.values[a], set=P.set.values[a], layer=l, expert=e, weight=Wt[a, j], dla=D[a, j])))
    pd.concat(fr).to_parquet(os.path.join(RESULTS, run, "final_routing.parquet"), index=False)
    ev = []
    for (l, e) in ex:
        m = TK[f"r_{l}_{e}"]
        ev.append(pd.DataFrame(dict(prompt=TK["prompt"][m], pos=TK["pos"][m].astype(np.int16), layer=np.int16(l), expert=np.int16(e),
                                    prob=TK[f"p_{l}_{e}"][m])))
    pd.concat(ev).to_parquet(os.path.join(RESULTS, run, "token_events.parquet"), index=False)
    # ---- sanity: wino_main / cf_str clean final routing vs the ext7 / ext6 source runs
    san = {}
    for set_, src, key in (("wino_main", "wino_qwen3_str" if model == "qwen3" else "wino_mixtral_bos_str", "case_id"),):
        rt = pd.read_parquet(os.path.join(RESULTS, src, "str_sweep_routing.parquet"))
        rt = rt[rt.slot == -1]
        s = P[P.set == set_]
        agree, tot = 0, 0
        for j, (l, e) in enumerate(ex):
            act_src = set(rt[(rt.layer == l) & (rt.expert == e)].case_id.astype(int))
            mine = set(s.case_id.astype(int)[A[s.index.values, j]])
            allc = set(s.case_id.astype(int))
            agree += sum((c in act_src) == (c in mine) for c in allc)
            tot += len(allc)
        san[set_] = {"agreement_with_source_routing": agree / tot, "n": tot}
    out["sanity"] = san
    # ---- C1: final-position rates and DLA by set
    rows, c1 = [], {}
    for set_ in ["wino_main", "wino_local", "wino_pool", "wino_pool_local", "cf_str", "cf_scan", "ioi", "wiki"]:
        sm = (P.set == set_).values
        if set_ == "cf_str":
            sm = sm & (P.kind == "clean").values
        c1[set_] = {"n": int(sm.sum()), "mean_delta": float(np.nanmean(P.delta.values[sm]))}
        for j, nm in enumerate(names):
            rr = rate_ci(A[sm, j], units[sm])
            act = sm & A[:, j]
            dl = rate_ci(D[act, j], units[act]) if act.sum() >= 3 else {"v": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(act.sum())}
            c1[set_][nm] = {"rate": rr, "dla_active": dl, "dla_pos_frac": float(np.mean(D[act, j] > 0)) if act.any() else float("nan"),
                            "weight_active": float(np.nanmean(Wt[act, j])) if act.any() else float("nan"),
                            "r_dla_delta": float(np.corrcoef(D[act, j], P.delta.values[act])[0, 1]) if act.sum() > 10 else float("nan"),
                            "mean_delta_active": float(np.nanmean(P.delta.values[act])) if act.any() else float("nan")}
        for nm in targets:
            j = names.index(nm)
            x = c1[set_][nm]
            rows.append({"model": MLAB[model], "expert": nm, "set": SET_LABEL[set_] + ("" if set_ != "cf_str" else ", clean only"),
                         "prompts": int(sm.sum()), "routed at final position (%)": fmt_r(x["rate"]),
                         "DLA when routed (logits, true - foil)": fmt_r(x["dla_active"], pct=False) if set_ != "wiki" else "",
                         "DLA > 0 (%)": f"{100 * x['dla_pos_frac']:.0f}" if set_ != "wiki" and np.isfinite(x["dla_pos_frac"]) else "",
                         "r(DLA, Delta) when routed": f"{x['r_dla_delta']:.2f}" if set_ != "wiki" and np.isfinite(x["r_dla_delta"]) else "",
                         "mean Delta of the set": f"{c1[set_]['mean_delta']:+.2f}" if set_ != "wiki" else ""})
    out["C1"] = c1
    T1 = pd.DataFrame(rows)
    # ---- C2: WinoGrande by final word, full vs local prompts
    c2 = {}
    wg = P.set.isin(["wino_main", "wino_pool"]).values
    lo = P.set.isin(["wino_local", "wino_pool_local"]).values
    fw_counts = P[wg].final_word.value_counts()
    top_fw = [w for w in fw_counts.index if fw_counts[w] >= 40][:12]
    rows2 = []
    for nm in own["wino"] + [n for n in names if n not in targets][:0]:
        j = names.index(nm)
        c2[nm] = {}
        for w in top_fw:
            sm = wg & (P.final_word == w).values
            sl = lo & (P.final_word == w).values
            c2[nm][w] = {"full": rate_ci(A[sm, j], units[sm]), "local": rate_ci(A[sl, j], units[sl])}
            rows2.append({"model": MLAB[model], "expert": nm, "final word": w, "prompts": int(sm.sum()),
                          "routed, full prompt (%)": fmt_r(c2[nm][w]["full"]), "routed, local prompt (%)": fmt_r(c2[nm][w]["local"])})
    # paired full vs local for the main pairs (same case_id): 2x2 and DLA
    mm = P[P.set == "wino_main"].set_index("case_id")
    ml = P[P.set == "wino_local"].set_index("case_id")
    common = mm.index.intersection(ml.index)
    pair_full_local = {}
    for nm in names:
        j = names.index(nm)
        af = A[mm.loc[common].idx.values, j]
        al = A[ml.loc[common].idx.values, j]
        df_ = D[mm.loc[common].idx.values, j]
        dl_ = D[ml.loc[common].idx.values, j]
        both = af & al
        pair_full_local[nm] = {"full": float(af.mean()), "local": float(al.mean()), "both": float(both.mean()),
                               "local_given_full": float(al[af].mean()) if af.any() else float("nan"),
                               "dla_full_when_both": float(np.nanmean(df_[both])) if both.any() else float("nan"),
                               "dla_local_when_both": float(np.nanmean(dl_[both])) if both.any() else float("nan"),
                               "r_dla_full_local": float(np.corrcoef(df_[both], dl_[both])[0, 1]) if both.sum() > 5 else float("nan"),
                               "n": int(len(common))}
    c2["paired_main_full_vs_local"] = pair_full_local
    c2["local_delta"] = {"main_local_mean_delta": float(np.nanmean(P.delta[P.set == "wino_local"])),
                         "main_mean_delta": float(np.nanmean(P.delta[P.set == "wino_main"])),
                         "local_correct_frac": float((P.delta[P.set == "wino_local"] > 0).mean())}
    out["C2"] = c2
    T2 = pd.DataFrame(rows2)
    # ---- C3: CounterFact scan by relation group; STR clean vs donor
    c3 = {}
    rows3 = []
    cs = P[P.set == "cf_scan"].copy()
    cs["group"] = cs.relation.map(W.rel_group)
    for nm in names:
        j = names.index(nm)
        c3[nm] = {}
        for g in W.REL_GROUPS:
            sm = (P.set == "cf_scan").values & P.relation.map(lambda r: W.rel_group(r) if isinstance(r, str) else "").eq(g).values
            act = sm & A[:, j]
            c3[nm][g] = {"rate": rate_ci(A[sm, j], units[sm]), "n": int(sm.sum()),
                         "dla_active": rate_ci(D[act, j], units[act]) if act.sum() >= 3 else None}
        for kind in ("clean", "donor"):
            sm = ((P.set == "cf_str") & (P.kind == kind)).values
            c3[nm][f"str_{kind}"] = {"rate": rate_ci(A[sm, j], units[sm]), "n": int(sm.sum())}
    for nm in targets:
        for g in W.REL_GROUPS:
            x = c3[nm][g]
            rows3.append({"model": MLAB[model], "expert": nm, "relation group": g, "prompts": x["n"],
                          "routed at final position (%)": fmt_r(x["rate"]),
                          "DLA when routed (logits)": fmt_r(x["dla_active"], pct=False) if x["dla_active"] else "n/a"})
        rows3.append({"model": MLAB[model], "expert": nm, "relation group": "STR clean / donor prompts",
                      "prompts": c3[nm]["str_clean"]["n"],
                      "routed at final position (%)": fmt_r(c3[nm]["str_clean"]["rate"]) + " / " + fmt_r(c3[nm]["str_donor"]["rate"]),
                      "DLA when routed (logits)": ""})
    out["C3"] = c3
    T3 = pd.DataFrame(rows3)
    # ---- token pieces and classes for the all-token analyses
    uniq = np.unique(TK["tok"])
    pcs = dict(zip(uniq.tolist(), pieces_of(tok, model, uniq)))
    cur_cls = {i: W.cur_class(p) for i, p in pcs.items()}
    tokp = np.array([pcs[int(i)] for i in TK["tok"]], dtype=object)
    tokc = np.array([cur_cls[int(i)] for i in TK["tok"]], dtype=object)
    # ---- C4: within WinoGrande / CounterFact / IOI prompts: final position vs other positions, by current-token class
    is_final = np.zeros(len(TK["tok"]), bool)
    ends = np.cumsum(P.n_tok.values) - 1
    is_final[ends] = True
    tset = P.set.values[TK["prompt"]]
    c4 = {}
    bos_pos0 = (model == "mixtral")
    keep = ~(bos_pos0 & (TK["pos"] == 0))
    for nm, (l, e) in zip(names, ex):
        r_ = TK[f"r_{l}_{e}"]
        c4[nm] = {}
        for set_ in ("wino_main", "cf_str", "ioi", "wiki"):
            s = (tset == set_) & keep
            c4[nm][set_] = {"final": float(r_[s & is_final].mean()), "non_final": float(r_[s & ~is_final].mean()),
                            "all": float(r_[s].mean())}
        # same token, different role: the WinoGrande final words at non-final positions of WinoGrande prompts
        s = (tset == "wino_main") & keep
        fwords = {" was", " too", " is", " very", " were"}
        m_tok = np.isin(tokp, list(fwords))
        c4[nm]["wino_main_finalword_tokens_nonfinal"] = float(r_[s & m_tok & ~is_final].mean()) if (s & m_tok & ~is_final).any() else float("nan")
        c4[nm]["wino_main_finalword_tokens_final"] = float(r_[s & m_tok & is_final].mean()) if (s & m_tok & is_final).any() else float("nan")
        c4[nm]["n_wino_main_finalword_nonfinal"] = int((s & m_tok & ~is_final).sum())
    out["C4"] = c4
    # ---- C5: wikitext contexts
    wk = (tset == "wiki") & keep
    wpos = np.nonzero(wk)[0]
    # next token / next word text (within the stream: the last position of a window continues into the next window)
    nxt_idx = np.full(len(TK["tok"]), -1)
    nxt_idx[:-1] = np.arange(1, len(TK["tok"]))
    nxt_idx[ends] = -1
    nxt_tok = np.where(nxt_idx >= 0, TK["tok"][np.clip(nxt_idx, 0, None)], -1)
    wiki_rows = P[P.set == "wiki"]
    for r in wiki_rows.itertuples():
        nxt_tok[ends[r.idx]] = int(r.next_after)
    all_nt = np.unique(nxt_tok[nxt_tok >= 0])
    missing = [int(i) for i in all_nt if int(i) not in pcs]
    if missing:
        pcs.update(dict(zip(missing, pieces_of(tok, model, missing))))
    # next-word text: concatenate up to 6 following pieces
    flat_p = tokp

    def next_text(i):
        out_, j, n = [], i + 1, 0
        if nxt_idx[i] < 0:
            return pcs.get(int(nxt_tok[i]), "")
        while j < len(flat_p) and n < 6 and TK["prompt"][j] == TK["prompt"][i]:
            out_.append(flat_p[j])
            j += 1
            n += 1
        return "".join(out_)
    ntext = np.array([next_text(i) for i in wpos], dtype=object)
    nword = np.array([(re.match(r"\s*([A-Za-z][A-Za-z\-']*)", t).group(1) if re.match(r"\s*([A-Za-z][A-Za-z\-']*)", t) else "") for t in ntext], dtype=object)
    nstart = np.array([(t[:1] in (" ", "\n")) for t in ntext])  # next token starts a new word
    cfv = W.load_cf_vocab()
    pm_place = W.PhraseMatcher(sorted(cfv["place"]))
    pm_lang = W.PhraseMatcher(sorted(cfv["language"]))
    pm_obj = W.PhraseMatcher(sorted(set().union(*cfv.values())))
    trig = set()
    for f in (f"data/wino_str/pairs_train_xl_{'qwen3' if model == 'qwen3' else 'mixtral_bos'}.parquet",):
        pp = pd.read_parquet(os.path.join(W.ROOT, f), columns=["word_a", "word_b"])
        trig |= {w.strip().lower() for w in pp.word_a} | {w.strip().lower() for w in pp.word_b}
    is_place = np.array([nstart[i] and pm_place.match(ntext[i]) is not None for i in range(len(wpos))])
    is_lang = np.array([nstart[i] and pm_lang.match(ntext[i]) is not None for i in range(len(wpos))])
    is_obj = np.array([nstart[i] and pm_obj.match(ntext[i]) is not None for i in range(len(wpos))])
    is_trig = np.array([nstart[i] and nword[i].lower() in trig for i in range(len(wpos))])
    is_capnext = np.array([nstart[i] and nword[i][:1].isupper() for i in range(len(wpos))])
    cur = tokc[wpos]
    curp = tokp[wpos]
    curl = np.array([p.strip().lower() for p in curp], dtype=object)
    wunits = units[TK["prompt"][wpos]]
    ctx = {
        "all tokens": np.ones(len(wpos), bool),
        "current = ' too'": curl == "too",
        "current = ' very' / ' so' / ' quite'": np.isin(curl, ["very", "so", "quite"]) & (cur == "degree adverb"),
        "current = degree adverb": cur == "degree adverb",
        "current = copula (was / is / were / ...)": cur == "copula",
        "copula or degree, next word in WG trigger vocabulary": np.isin(cur, ["copula", "degree adverb"]) & is_trig,
        "copula or degree, next word not in trigger vocabulary": np.isin(cur, ["copula", "degree adverb"]) & ~is_trig,
        "next word in WG trigger vocabulary": is_trig,
        "next word = CF place name": is_place,
        "next word = CF language name": is_lang,
        "next word = any CF object": is_obj,
        "next word capitalised, not a CF object": is_capnext & ~is_obj,
        "current = preposition, next = CF place": (cur == "preposition") & is_place,
        "current = preposition, next not a CF object": (cur == "preposition") & ~is_obj,
        "current = ' in', next = CF place": (curl == "in") & is_place,
        "current = ' in', next = other": (curl == "in") & ~is_place,
    }
    for c in sorted(set(cur)):
        ctx[f"class: {c}"] = cur == c
    c5 = {"n_tokens": int(len(wpos)), "contexts": {}, "top_current": {}, "top_next": {}}
    rows5 = []
    for nm, (l, e) in zip(names, ex):
        r_ = TK[f"r_{l}_{e}"][wpos]
        base = float(r_.mean())
        c5["contexts"][nm] = {"base": base}
        for cname, msk in ctx.items():
            if msk.sum() < 10:
                continue
            rr = rate_ci(r_[msk], wunits[msk], n_boot=500)
            c5["contexts"][nm][cname] = {"rate": rr, "lift": rr["v"] / base if base > 0 else float("nan"), "n": int(msk.sum())}
        # top current tokens / next words by rate (min 30 occurrences), and by share of the expert's routings
        dfc = pd.DataFrame(dict(p=curp, r=r_))
        g = dfc.groupby("p").r.agg(["mean", "size", "sum"])
        g = g[g["size"] >= 30].sort_values("mean", ascending=False)
        c5["top_current"][nm] = [(k_, round(float(v["mean"]), 3), int(v["size"])) for k_, v in g.head(15).iterrows()]
        g2 = dfc.groupby("p").r.agg(["sum"]).sort_values("sum", ascending=False)
        c5.setdefault("top_current_by_count", {})[nm] = [(k_, int(v["sum"]), round(float(v["sum"] / max(r_.sum(), 1)), 3)) for k_, v in g2.head(10).iterrows()]
        dfn = pd.DataFrame(dict(w=[w.lower() for w in nword], r=r_, s=nstart))
        gn = dfn[(dfn.w != "") & dfn.s].groupby("w").r.agg(["mean", "size"])  # whole next words only
        gn = gn[gn["size"] >= 20].sort_values("mean", ascending=False)
        c5["top_next"][nm] = [(k_, round(float(v["mean"]), 3), int(v["size"])) for k_, v in gn.head(15).iterrows()]
    for nm in targets:
        x = c5["contexts"][nm]
        row = {"model": MLAB[model], "expert": nm, "base rate (%)": f"{100 * x['base']:.2f}"}
        for cname in ["current = ' too'", "current = degree adverb", "current = copula (was / is / were / ...)",
                      "copula or degree, next word in WG trigger vocabulary", "next word in WG trigger vocabulary",
                      "next word = CF place name", "next word capitalised, not a CF object", "current = ' in', next = CF place",
                      "current = ' in', next = other"]:
            if cname in x:
                row[cname] = f"{100 * x[cname]['rate']['v']:.1f} ({x[cname]['lift']:.1f}x, n={x[cname]['n']})"
        rows5.append(row)
    out["C5"] = c5
    T5 = pd.DataFrame(rows5)
    # top-token table for the targets
    rows6 = []
    for nm in targets:
        rows6.append({"model": MLAB[model], "expert": nm,
                      "highest-rate current tokens (rate, n)": "; ".join(f"{repr(a)} {100 * b:.0f}% ({c})" for a, b, c in c5["top_current"][nm][:8]),
                      "most frequent current tokens among its routings (share)": "; ".join(f"{repr(a)} {100 * c:.1f}%" for a, b, c in c5["top_current_by_count"][nm][:6]),
                      "highest-rate next words (rate, n)": "; ".join(f"{a} {100 * b:.0f}% ({c})" for a, b, c in c5["top_next"][nm][:8])})
    T6 = pd.DataFrame(rows6)
    # C4 table
    rows4 = []
    for nm in targets:
        x = c4[nm]
        rows4.append({"model": MLAB[model], "expert": nm,
                      **{f"{SET_LABEL[s].split(' (')[0]}: final / other positions (%)": f"{100 * x[s]['final']:.1f} / {100 * x[s]['non_final']:.1f}"
                         for s in ("wino_main", "cf_str", "ioi", "wiki")},
                      "WG prompts, tokens ' was' ' too' ' is' ' very' ' were': final / earlier (%)":
                          f"{100 * x['wino_main_finalword_tokens_final']:.1f} / {100 * x['wino_main_finalword_tokens_nonfinal']:.1f} (n={x['n_wino_main_finalword_nonfinal']})"})
    T4 = pd.DataFrame(rows4)
    log(f"  analysed in {time.time() - t0:.0f}s")
    return out, dict(T1=T1, T2=T2, T3=T3, T4=T4, T5=T5, T6=T6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="qwen3:qwen3_writer_routing,mixtral:mixtral_bos_writer_routing")
    args = ap.parse_args()
    res, tabs = {}, {}
    for spec in args.runs.split(","):
        model, run = spec.split(":")
        if not json.load(open(os.path.join(RESULTS, run, "run_meta.json"))).get("complete"):
            log(f"{run} not complete, skipped")
            continue
        res[model], tabs[model] = analyse(model, run)
    if not res:
        return
    notes = {
        "T1": "Final-position routing of the target experts in each prompt set (CounterFact STR: clean prompts) and the expert's own DLA "
              "there, (c_e . gamma) . (W_U[true] - W_U[foil]) / rms(h_final) with true / foil = own trigger / twin trigger (WinoGrande), "
              "true object / counterfactual object (CounterFact), IO / S (IOI). CIs: bootstrap over pairs (WinoGrande), cases or prompts.",
        "T2": "WinoGrande prompts (main + margin pool) by the word before the trigger; local = the context-free prompt from the blank on.",
        "T3": "CounterFact base-scan prompts by relation group (place = P17, P19, P20, P27, P30, P36, P131, P159, P190, P276, P495, P740, P937).",
        "T4": "All-token routing (route_all_layers): rate at the final position vs every other position of the same prompts "
              "(Mixtral: BOS position excluded); last column: the same tokens as the WinoGrande final words when they occur earlier.",
        "T5": "wikitext-103 test windows (127 content tokens, consecutive): routing rate of the expert at the current token, in % "
              "(lift over the expert's base rate, n). Word classes are heuristic word lists; 'WG trigger vocabulary' = all sentence-final "
              "trigger words of the model's W1-W6 pairs; CF names = target_true / target_new strings of CounterFact relations.",
        "T6": "wikitext-103: current tokens with the highest routing rate (≥ 30 occurrences), the most frequent current tokens among the "
              "expert's routings, and next words with the highest rate (≥ 20 occurrences).",
    }
    names = {"T1": "ext11_C_final_rates", "T2": "ext11_C_wino_finalword", "T3": "ext11_C_cf_relations", "T4": "ext11_C_positions",
             "T5": "ext11_C_wiki_contexts", "T6": "ext11_C_wiki_tokens"}
    os.makedirs(TAB, exist_ok=True)
    for t, nm in names.items():
        df = pd.concat([tabs[m][t] for m in tabs], ignore_index=True)
        df.to_csv(os.path.join(TAB, nm + ".csv"), index=False)
        with open(os.path.join(TAB, nm + ".md"), "w") as f:
            f.write(W.md_table(df) + "\n\n" + notes[t] + "\n")
    W.update_summary("C", res)
    log("written key C, tables ext11_C_*")


if __name__ == "__main__":
    main()
