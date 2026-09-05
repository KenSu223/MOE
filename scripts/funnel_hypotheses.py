"""Test alternative protocol readings for the filter funnel against the paper's Table 8 IDs (Qwen3).
Variants: BOS prepended or not; object token with or without a leading space (where single-token)."""
import sys, json, time, os
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer
from moetrace.arch import snapshot_dir
from moetrace.models import MODELS
from moetrace.data import load_records, shuffled_order, load_paper_ids, prepare_case
from moetrace.engine import Engine, PrefillSpec
from moetrace.noise import noise_draw
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
model = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
max_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 390
m = MODELS[model]
tok = AutoTokenizer.from_pretrained(snapshot_dir(m["repo"]))
recs = load_records(); order = shuffled_order(len(recs), 0)
pids = load_paper_ids(m["paper_key"]); allp = set(pids["discovery"] + pids["validation"])
bos = tok.bos_token_id if tok.bos_token_id is not None else 151643
cases = []
for r, ri in enumerate(order[:max_rank]):
    c, why = prepare_case(recs[ri], tok)
    if c is None: continue
    c._rank = r; cases.append(c)
eng = Engine(m["repo"]); sigma = 3.0 * eng.embed_std; Hd = eng.hidden
def nospace_id(s):
    ids = tok(s).input_ids
    return ids[0] if len(ids) == 1 else None
variants = {}
pre = []
for c in cases:
    ns_t, ns_f = nospace_id(c.true_str), nospace_id(c.foil_str)
    for use_bos in (False, True):
        ids = ([bos] if use_bos and (not c.ids or c.ids[0] != bos) else []) + c.ids
        sp = [p + (1 if use_bos and ids[0] == bos and (not c.ids or c.ids[0] != bos) else 0) for p in c.subject_pos]
        for space in ("space", "nospace"):
            if space == "nospace" and (ns_t is None or ns_f is None): continue
            t, f = (c.true_id, c.foil_id) if space == "space" else (ns_t, ns_f)
            key = (c.case_id, use_bos, space)
            variants[key] = (len(pre), len(pre) + 1)
            pre.append(PrefillSpec(ids, t, f))
            pre.append(PrefillSpec(ids, t, f, sp, noise_draw(c.case_id, len(sp), Hd, sigma)))
log(f"{len(cases)} cases, {len(pre)} rows")
res = eng.run(pre, [], record_routing=False, log=log)
d = res.delta
rows = []
for (cid, use_bos, space), (i, j) in variants.items():
    rows.append(dict(case_id=cid, bos=use_bos, space=space, delta_clean=float(d[i]), delta_noised=float(d[j]), paper=cid in allp))
df = pd.DataFrame(rows)
df["drop"] = df.delta_clean - df.delta_noised
df["strict"] = (df.delta_clean >= 1.0) & (df["drop"] >= 0.5)
df.to_parquet(f"results/{model}/funnel_hypotheses.parquet", index=False)
out = {}
for (use_bos, space), g in df.groupby(["bos", "space"]):
    p = g[g.paper]; q = g[~g.paper]
    out[f"bos={use_bos},{space}"] = dict(n_paper=len(p), paper_strict_pass=int(p.strict.sum()), n_nonpaper=len(q), nonpaper_strict_pass=int(q.strict.sum()),
                                          paper_pass_rate=round(p.strict.mean(), 3), nonpaper_pass_rate=round(q.strict.mean(), 3))
    log(f"bos={use_bos} {space}: paper {int(p.strict.sum())}/{len(p)} pass strict; non-paper {int(q.strict.sum())}/{len(q)} pass")
# ideal: paper all pass, non-paper all fail. Also test whether "first 256 strict-passing" under a variant equals the paper set
for (use_bos, space), g in df.groupby(["bos", "space"]):
    g = g.copy(); g["rank"] = g.case_id.map({c.case_id: c._rank for c in cases}); g = g.sort_values("rank")
    first = g[g.strict].case_id.tolist()[:256]
    out[f"bos={use_bos},{space}"]["overlap_first256_with_paper"] = len(set(first) & allp)
    log(f"bos={use_bos} {space}: overlap of first-256 strict with paper set: {len(set(first) & allp)}")
json.dump(out, open(f"results/{model}/funnel_hypotheses.json", "w"), indent=1)
