"""Second round of funnel hypotheses: object-token variants over ALL tokenizable records in the paper prefix.
Variants (no BOS): space_single (ours), space_first (first token of ' '+obj), nospace_first (first token of obj),
nospace_single_else_space. Agreement = (paper & pass) + (non-paper & fail) over the prefix."""
import sys, json, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer
from moetrace.arch import snapshot_dir
from moetrace.models import MODELS
from moetrace.data import load_records, shuffled_order, load_paper_ids
from moetrace.engine import Engine, PrefillSpec
from moetrace.noise import noise_draw
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
model = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
max_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 390
m = MODELS[model]
tok = AutoTokenizer.from_pretrained(snapshot_dir(m["repo"]))
recs = load_records(); order = shuffled_order(len(recs), 0)
pids = load_paper_ids(m["paper_key"]); allp = set(pids["discovery"] + pids["validation"])
eng = Engine(m["repo"]); sigma = 3.0 * eng.embed_std; Hd = eng.hidden
def ids_of(s): return tok(s, add_special_tokens=False).input_ids
pre, keys = [], []
info = []
for r, ri in enumerate(order[:max_rank]):
    rec = recs[ri]; rw = rec["requested_rewrite"]
    tpl, subj = rw["prompt"], rw["subject"]; prompt = tpl.format(subj)
    start = tpl.index("{}"); end = start + len(subj)
    enc = tok(prompt, return_offsets_mapping=True, return_special_tokens_mask=True)
    ids = list(enc["input_ids"])
    spos = [i for i, ((a, b), sp) in enumerate(zip(enc["offset_mapping"], enc["special_tokens_mask"])) if not sp and b > a and a < end and b > start]
    t, f = rw["target_true"]["str"], rw["target_new"]["str"]
    sp_t, sp_f = ids_of(" " + t), ids_of(" " + f)
    ns_t, ns_f = ids_of(t), ids_of(f)
    cont_t = tok(prompt + " " + t).input_ids; cont_f = tok(prompt + " " + f).input_ids
    single_space = len(cont_t) == len(ids) + 1 and len(cont_f) == len(ids) + 1 and cont_t[:len(ids)] == ids and cont_f[:len(ids)] == ids
    variants = {
        "space_single": (cont_t[-1], cont_f[-1]) if single_space else None,
        "space_first": (sp_t[0], sp_f[0]),
        "nospace_first": (ns_t[0], ns_f[0]),
        "nospace_single_else_space": (ns_t[0] if len(ns_t) == 1 else sp_t[0], ns_f[0] if len(ns_f) == 1 else sp_f[0]),
        "nospace_single_only": (ns_t[0], ns_f[0]) if (len(ns_t) == 1 and len(ns_f) == 1) else None,
    }
    info.append(dict(case_id=rec["case_id"], rank=r, paper=rec["case_id"] in allp, single_space=single_space,
                     single_nospace=(len(ns_t) == 1 and len(ns_f) == 1), n_sp_t=len(sp_t), n_sp_f=len(sp_f), n_ns_t=len(ns_t), n_ns_f=len(ns_f)))
    eps = noise_draw(rec["case_id"], len(spos), Hd, sigma)
    for v, tf in variants.items():
        if tf is None or tf[0] == tf[1]: continue
        keys.append((rec["case_id"], v)); pre.append(PrefillSpec(ids, tf[0], tf[1]))
        keys.append((rec["case_id"], v + "#n")); pre.append(PrefillSpec(ids, tf[0], tf[1], spos, eps))
log(f"{len(info)} records, {len(pre)} rows")
res = eng.run(pre, [], record_routing=False, log=log)
d = dict(zip(keys, res.delta.tolist()))
info = pd.DataFrame(info)
out = {}
for v in ["space_single", "space_first", "nospace_first", "nospace_single_else_space", "nospace_single_only"]:
    dc = info.case_id.map(lambda c: d.get((c, v), np.nan)); dn = info.case_id.map(lambda c: d.get((c, v + "#n"), np.nan))
    ok = (dc >= 1.0) & ((dc - dn) >= 0.5)
    defined = dc.notna()
    agree = ((ok & info.paper) | (~ok & ~info.paper))[defined].mean()
    first256 = info[defined & ok].sort_values("rank").case_id.tolist()[:256]
    out[v] = dict(defined=int(defined.sum()), paper_pass=int((ok & info.paper).sum()), paper_defined=int((defined & info.paper).sum()),
                  nonpaper_pass=int((ok & ~info.paper).sum()), nonpaper_defined=int((defined & ~info.paper).sum()),
                  agreement=round(float(agree), 3), overlap_first256=len(set(first256) & allp))
    log(v, json.dumps(out[v]))
    info[f"dc_{v}"] = dc; info[f"dn_{v}"] = dn; info[f"pass_{v}"] = ok
info.to_parquet(f"results/{model}/funnel_hypotheses2.parquet", index=False)
json.dump(out, open(f"results/{model}/funnel_hypotheses2.json", "w"), indent=1)
