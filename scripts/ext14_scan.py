"""ext14 Phase 6 competence scans (GPU, prefill rows only; agent p6-data).

  --part cells   clean rows of both prompts of every candidate pair of data/dial/build/<cell>_<fam>.parquet (fam = the proto's
                 tokenizer family): d_clean_a = Delta(prompt A), d_clean_b = Delta(prompt B), Delta = logit(trig_a) - logit(trig_b);
                 top-1 ids; margin = d_a >= 1 and d_b <= -1; reverse_margin = d_a <= -1 and d_b >= 1 (Set C event-following).
                 --subset core (core-776 items / all non-WinoGrande items) | ext (extension twins, first --max-ext-items items in
                 the twin priority order) | all
                 -> results/dial_scan_<proto>/scan_<cell>_<subset>.parquet
  --part probes  association probes (Phase-5 v1 / v2 prompts) of data/dial/build/probe_specs_<fam>.parquet
                 -> results/dial_scan_<proto>/probe_rows.parquet, probe_values.parquet (key, v1_a, v1_b, v2_a, v2_b)
Usage: python scripts/ext14_scan.py <proto> --part cells --cells e1_a2e,... [--subset core] | --part probes
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace import ext14_data as X
from moetrace.models import MODELS, RESULTS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chunks_by_tokens(lens, max_tokens):
    order = sorted(range(len(lens)), key=lambda j: lens[j])
    out, cur, cur_max = [], [], 0
    for j in order:
        T = lens[j]
        if cur and max(cur_max, T) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, cur_max = [], 0
        cur.append(j)
        cur_max = max(cur_max, T)
    if cur:
        out.append(cur)
    return out


def run_rows(eng, specs, max_tokens):
    ch = chunks_by_tokens([len(s.ids) for s in specs], max_tokens)
    delta = np.full(len(specs), np.nan)
    top1 = np.full(len(specs), -1, dtype=np.int64)
    for ci, c in enumerate(ch):
        res = eng.run([specs[j] for j in c], [], record_routing=False, log=log if ci == 0 else None)
        delta[c], top1[c] = res.delta, res.top1
        log(f"  pass {ci + 1}/{len(ch)} ({len(c)} rows, {res.extra['total_s']:.0f}s)")
    return delta, top1, len(ch)


def write_meta(od, key, meta):
    rm = os.path.join(od, "run_meta.json")
    allm = json.load(open(rm)) if os.path.exists(rm) else {}
    allm[key] = meta
    json.dump(allm, open(rm, "w"), indent=1, default=str)


def select_subset(df, subset, max_ext_items, scanned=frozenset()):
    """core = core-776 / non-WinoGrande items; ext = extension items NOT scanned yet, discovery and validation interleaved by their rank
    (the twin rank is split-ordered, so a plain rank cut would take discovery items only), first max_ext_items items."""
    if subset == "all":
        return df
    core = (df.source != "wino") | df.core776 if "core776" in df else pd.Series(True, index=df.index)
    if subset == "core":
        return df[core]
    ext = df[~core & ~df.src_pair_id.isin(scanned)]
    it = ext.drop_duplicates("src_pair_id")[["src_pair_id", "split", "rank"]].sort_values("rank")
    it["k"] = it.groupby("split").cumcount()
    keep = it.sort_values(["k", "split"]).src_pair_id.head(max_ext_items)
    return ext[ext.src_pair_id.isin(set(keep))]


def scanned_items(od, cell):
    out = set()
    for f in os.listdir(od):
        if f.startswith(f"scan_{cell}_") and f.endswith(".parquet") and not f.startswith("VOID"):
            out |= set(pd.read_parquet(os.path.join(od, f), columns=["src_pair_id"]).src_pair_id)
    return frozenset(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto", choices=list(X.PROTOS))
    ap.add_argument("--part", required=True, choices=["cells", "probes"])
    ap.add_argument("--cells", default="")
    ap.add_argument("--subset", default="core", choices=["core", "ext", "all"])
    ap.add_argument("--max-ext-items", type=int, default=1200)
    ap.add_argument("--specs", default=None, help="probe-spec parquet (default data/dial/build/probe_specs_<fam>.parquet)")
    ap.add_argument("--max-tokens", type=int, default=140000)
    ap.add_argument("--margin", type=float, default=1.0)
    a = ap.parse_args()
    key, st, fam = X.PROTOS[a.proto]
    m = MODELS[key]
    if key in ("qwen2_57b",):  # fp32 protocol, 64 + shared experts: smaller prefill batches (memory)
        a.max_tokens = min(a.max_tokens, 50000)
    elif m.get("act_dtype") == "fp32":
        a.max_tokens = min(a.max_tokens, 90000)
    od = os.path.join(RESULTS, f"dial_scan_{a.proto}")
    os.makedirs(od, exist_ok=True)
    mkey = f"scan_{a.part}_{a.subset}_{a.cells.replace(',', '+')}"[:120]
    meta = {"model": key, "repo": m["repo"], "proto": a.proto, "family": fam, "special_tokens": st, "part": a.part, "agent": "p6-data",
            "command": "python " + " ".join(sys.argv), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    write_meta(od, mkey, meta)
    from moetrace.engine import Engine, PrefillSpec
    eng = Engine(m["repo"])
    meta["engine_protocol"] = dict(getattr(eng, "protocol", {}))  # e.g. {'dtype': 'float32'} for Qwen2-7B / Qwen2-57B (models.py flags)
    write_meta(od, mkey, meta)
    if a.part == "probes":
        sp = pd.read_parquet(a.specs or f"{X.BUILD}/probe_specs_{fam}.parquet")
        specs = [PrefillSpec(json.loads(r.ids), int(r.ea), int(r.eb)) for r in sp.itertuples()]
        log(f"probes: {len(sp)} rows, {sp.key.nunique()} keys")
        d, t1, n = run_rows(eng, specs, a.max_tokens)
        sp = sp.drop(columns=["ids"]).assign(delta=d, top1=t1)
        sp.to_parquet(os.path.join(od, "probe_rows.parquet"), index=False)
        sp["trig"] = sp.slot.str[0]
        g = sp.groupby(["key", "version", "trig"]).delta.mean().unstack(["version", "trig"])
        g.columns = [f"{v}_{t}" for v, t in g.columns]
        g.reset_index().to_parquet(os.path.join(od, "probe_values.parquet"), index=False)
        meta.update(n_rows=len(sp), n_keys=int(len(g)), n_passes=n)
    else:
        tot = {}
        for cell in a.cells.split(","):
            bp = f"{X.BUILD}/{cell}_{a.proto}.parquet"  # per-model builds (l_cf) take precedence over the family build
            p = pd.read_parquet(bp if os.path.exists(bp) else f"{X.BUILD}/{cell}_{fam}.parquet")
            p = select_subset(p, a.subset, a.max_ext_items, scanned_items(od, cell) if a.subset == "ext" else frozenset())
            tag = a.subset
            if tag == "ext":  # never overwrite an earlier extension scan of the cell
                n = 1
                while os.path.exists(os.path.join(od, f"scan_{cell}_{tag}.parquet")):
                    n += 1
                    tag = f"ext{n}"
            if p.empty:
                log(cell, "no pairs in subset", a.subset)
                continue
            specs = []
            for r in p.itertuples():
                specs.append(PrefillSpec(json.loads(r.ids_a), int(r.trig_a), int(r.trig_b)))
                specs.append(PrefillSpec(json.loads(r.ids_b), int(r.trig_a), int(r.trig_b)))
            log(f"{cell} [{a.subset}]: {len(p)} pairs, {len(specs)} rows")
            d, t1, n = run_rows(eng, specs, a.max_tokens)
            q = p.drop(columns=[c for c in ("ids_a", "ids_b") if c in p.columns]).copy()
            q["d_clean_a"], q["d_clean_b"] = d[0::2], d[1::2]
            q["top1_a"], q["top1_b"] = t1[0::2], t1[1::2]
            q["drop_str"] = q.d_clean_a - q.d_clean_b
            q["margin"] = (q.d_clean_a >= a.margin) & (q.d_clean_b <= -a.margin)
            q["reverse_margin"] = (q.d_clean_a <= -a.margin) & (q.d_clean_b >= a.margin)
            q["correct_both"] = (q.d_clean_a > 0) & (q.d_clean_b < 0)
            q["top1_both"] = (q.top1_a == q.trig_a) & (q.top1_b == q.trig_b)
            q["scan_subset"] = tag
            q.to_parquet(os.path.join(od, f"scan_{cell}_{tag}.parquet"), index=False)
            byo = q.groupby("order").margin.mean().round(3).to_dict() if "order" in q else {}
            tot[cell] = {"pairs": int(len(q)), "margin": int(q.margin.sum()), "reverse_margin": int(q.reverse_margin.sum()),
                         "margin_rate": round(float(q.margin.mean()), 3), "by_order": byo, "top1_both": round(float(q.top1_both.mean()), 3),
                         "passes": n}
            log(cell, json.dumps(tot[cell]))
        meta.update(cells=tot)
    meta.update(complete=True, completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_meta(od, mkey, meta)


if __name__ == "__main__":
    main()
