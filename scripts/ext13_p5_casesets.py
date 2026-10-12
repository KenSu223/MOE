"""ext13 Phase 5 case sets and condition registry (CPU, agent p5-data; after scripts/ext13_p5_scan.py).

1. Probe outcome per model (pre-registered rule, moetrace/ext13_p5.py docstring): fraction of trigger probes with
   |probe| >= 1 on the margin pairs of the valid attribute -> entity formats (F1, F2E, F2R; unique twins) for v1 and v2;
   R = responding probes; association stratum of every probe key -> results/wino_ref_<proto>/p5_probe_outcome.json
2. Per condition (RESEARCH_PLAN.md "Phase 5 execution plan", condition table) and model: pair file
   data/wino_ref/pairs_p5_<cond>_<proto>.parquet (pair contract: pair_id, ids_a, ids_b, str_pos, trig_a, trig_b; strata
   columns names, stratum_assoc, stratum_neutral, ...; token positions mention_a_pos / mention_b_pos (+ blank_pos, copy
   prefix positions attr_a_pos / attr_b_pos / prefix_ment_a_pos / prefix_ment_b_pos); scan values d_clean_a / d_clean_b /
   margin) and case sets data/wino_ref/case_sets_p5_<cond>_<proto>.json (generic format: pair_idx_of, pairs, directed,
   strata, pools).
   pair_idx: natural twins = index of the HIT id in the sorted train_xl twins (rule W1, as data/wino_str/case_sets.json);
   cp_nat = 2 * Phase-3 pair_idx + (order - 1); templates = 1000 * rank of the seed (sorted ids) + 2 * name-pair index j +
   (order - 1) (vt_* use order 1, so the same template item has the same pair_idx in vt_e2a, vt_a2e and cp order 1).
   Splits: f1 seed-0 shuffle 128 / 128 + replication 128 / 128 where the pool allows (>= 32 each, else none); pools < 256 pairs
   half / half;
   dir_f1 = dir_opt = f2e: seed-0 shuffle, half / half; templates: seed split S (55 ent2attr seeds sorted, random.Random(0)
   shuffle, 27 discovery / 28 validation seeds), <= 128 pairs per split drawn round-robin over the split's seeds in S
   order (within a seed by name pair j, then order); cp_nat: the Phase-3 split of the twin (discovery / validation /
   rep_discovery / rep_validation; shared_unused not used), both orders, restricted to margin pairs.
   Copy conditions (cp_vt_*, cp_nat): an item (template item / twin) enters only if BOTH attribute orders pass the margin
   (exact counterbalancing). Prefix (coordinator decision 2026-10-11): "Options: t1, t2." + newline (moetrace/ext13_p5.py
   options_prefix); the earlier "Of X and Y, one .. and the other .." prefix (respectively-congruent order 1, column congruent)
   is void and kept as behaviour only.
3. data/wino_ref/p5_conditions.json (registry; status.cases = true), PROGRESS line "[p5-data] CASES READY <cond>/<proto>".
Usage: python scripts/ext13_p5_casesets.py [--protos qwen3,mixtral_bos] [--conds ...] [--no-progress]
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
from moetrace import ext13_p5 as X
from moetrace.ext7_wino import load_twins
from moetrace.models import RESULTS

D = "/home/ubuntu/MOE/data/wino_ref"
PROTOS = {"qwen3": ("qwen3", True), "mixtral_bos": ("mixtral", True)}
CONDS = ["f1", "dir_f1", "dir_opt", "cp_nat", "cp_vt_orig", "cp_vt_neu", "vt_e2a_orig", "vt_e2a_neu", "vt_a2e_orig", "vt_a2e_neu", "f2e"]
TASK = {"f1": 1, "dir_f1": 2, "dir_opt": 2, "f2e": 3, "vt_e2a_orig": 4, "vt_e2a_neu": 4, "vt_a2e_orig": 5, "vt_a2e_neu": 5,
        "cp_vt_orig": 6, "cp_vt_neu": 6, "cp_nat": 6}
FORMAT = {"f1": "fill-in-the-blank (entity readout)", "dir_f1": "fill-in-the-blank (entity readout)",
          "dir_opt": "Phase-3 option swap (fill the blank, predict the trigger)", "f2e": "natural truncation, blank at the sentence end",
          "vt_e2a_orig": "Phase-3 option swap on verb templates", "vt_e2a_neu": "Phase-3 option swap on verb templates (neutral verb)",
          "vt_a2e_orig": "fill-in-the-blank on verb templates", "vt_a2e_neu": "fill-in-the-blank on verb templates (neutral verb)",
          "cp_vt_orig": "Phase-3 + copy prefix on verb templates", "cp_vt_neu": "Phase-3 + copy prefix on verb templates (neutral verb)",
          "cp_nat": "Phase-3 + copy prefix on the 776 Phase-3 twins"}
DIRECTION = {c: ("attr2ent" if c in ("f1", "dir_f1", "f2e", "vt_a2e_orig", "vt_a2e_neu") else "ent2attr") for c in CONDS}
MIN_SPLIT = 16


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jl(x):
    return json.loads(x) if isinstance(x, str) else list(x)


def tokenizer(proto):
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    from moetrace.models import MODELS
    key, st = PROTOS[proto]
    return AutoTokenizer.from_pretrained(snapshot_dir(MODELS[key]["repo"])), st


def blank_at_end(sent: str) -> bool:
    return not re.search(r"[A-Za-z]", sent[sent.index("_") + 1:])


# ---------------------------------------------------------------------------------------------------------------
# probe outcome
# ---------------------------------------------------------------------------------------------------------------
def probe_outcome(proto):
    od = f"{RESULTS}/wino_ref_{proto}"
    pv = pd.read_parquet(f"{od}/p5_probe_values.parquet").set_index("key")
    margin = []
    per_fmt = {}
    for fmt, fname, filt in (("F1", "F1", None), ("F2E", "F2", "blank_at_end"), ("F2R", "F2R", None)):
        s = pd.read_parquet(f"{od}/scan_pairs_{fname}.parquet")
        if filt:
            s = s[[blank_at_end(x) for x in s.sent_a]]
        s = s[s.margin]
        s = s.assign(key=[X.probe_key(r.names, r.word_a, r.word_b, r.ans_a, r.ans_b) for r in s.itertuples()], fmt=fmt)
        margin.append(s[["pair_id", "key", "fmt", "names"]])
        per_fmt[fmt] = s
    mg = pd.concat(margin).drop_duplicates("pair_id")
    vals = pv.loc[mg.key]
    out = {"n_margin_pairs_unique_twins": int(len(mg)), "by_format": {k: int(len(v)) for k, v in per_fmt.items()}, "frac_ge1": {}, "frac_ge1_by_format": {},
           "frac_ge1_names": {}, "frac_ge1_objects": {}, "sd": {}}
    for v in X.PROBE_VERSIONS:
        x = np.concatenate([vals[f"{v}_a"].to_numpy(float), vals[f"{v}_b"].to_numpy(float)])
        x = x[~np.isnan(x)]
        out["frac_ge1"][v] = float((np.abs(x) >= X.THR).mean())
        out["sd"][v] = float(x.std())
        for nm, sel in (("names", mg.names.to_numpy()), ("objects", ~mg.names.to_numpy())):
            y = np.concatenate([vals[f"{v}_a"].to_numpy(float)[sel], vals[f"{v}_b"].to_numpy(float)[sel]])
            y = y[~np.isnan(y)]
            out[f"frac_ge1_{nm}"][v] = float((np.abs(y) >= X.THR).mean()) if len(y) else None
        out["frac_ge1_by_format"][v] = {}
        for fmt, s in per_fmt.items():
            vv = pv.loc[s.key]
            y = np.concatenate([vv[f"{v}_a"].to_numpy(float), vv[f"{v}_b"].to_numpy(float)])
            y = y[~np.isnan(y)]
            out["frac_ge1_by_format"][v][fmt] = float((np.abs(y) >= X.THR).mean()) if len(y) else None
    a = vals[["v1_a", "v1_b"]].to_numpy(float).ravel()
    b = vals[["v2_a", "v2_b"]].to_numpy(float).ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    out["r_v1_v2"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    resp = [v for v in X.PROBE_VERSIONS if out["frac_ge1"][v] >= X.RESPOND_FRAC]
    out["responding"] = resp
    out["flag_none_responds"] = not resp
    R = resp if resp else list(X.PROBE_VERSIONS)
    out["R"] = R
    # templates as a check (random IOI names; all template keys)
    tk = pv[pv.src.str.contains("templates")]
    out["templates_frac_ge1"] = {v: float((np.abs(np.concatenate([tk[f"{v}_a"].to_numpy(float), tk[f"{v}_b"].to_numpy(float)])) >= X.THR).mean())
                                 for v in X.PROBE_VERSIONS}
    out["templates_n_keys"] = int(len(tk))
    # strata of every key
    st = {}
    for k, r in pv.iterrows():
        vv = {v: (r[f"{v}_a"], r[f"{v}_b"]) for v in X.PROBE_VERSIONS}
        st[k] = (X.assoc_stratum(vv, R), X.stratum_conflict(vv, R))
    pv["stratum_assoc"] = [st[k][0] for k in pv.index]
    pv["stratum_conflict"] = [st[k][1] for k in pv.index]
    mgs = pv.loc[mg.key]
    out["margin_strata"] = mgs.stratum_assoc.value_counts().to_dict()
    out["margin_strata_names"] = mgs[mg.names.to_numpy()].stratum_assoc.value_counts().to_dict()
    out["margin_strata_objects"] = mgs[~mg.names.to_numpy()].stratum_assoc.value_counts().to_dict()
    out["margin_conflicts"] = int(mgs.stratum_conflict.sum())
    json.dump(out, open(f"{od}/p5_probe_outcome.json", "w"), indent=1)
    return out, pv


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------
def hit_index():
    twins, _ = load_twins("train_xl")
    return {t.pair_id: i for i, t in enumerate(twins)}


def seed_split():
    seeds = sorted(json.load(open(f"{D}/verb_templates.json"))["seeds"])
    vt = json.load(open(f"{D}/verb_templates.json"))["seeds"]
    seeds = sorted(s for s in seeds if vt[s]["dir"] == "ent2attr")
    assert len(seeds) == 55
    rank = {s: i for i, s in enumerate(seeds)}
    order = list(seeds)
    random.Random(0).shuffle(order)
    return rank, order[:27], order[27:]


def shuffle_split(pool, n=None, rep=False):
    order = sorted(int(i) for i in pool)
    random.Random(0).shuffle(order)
    if n is None:
        h = len(order) // 2
        return {"discovery": order[:h], "validation": order[h:2 * h]}
    sp = {"discovery": order[:n], "validation": order[n:2 * n]}
    rest = order[2 * n:]
    if rep:
        r = min(n, len(rest) // 2)
        if r >= 32:
            sp["rep_discovery"], sp["rep_validation"] = rest[:r], rest[r:2 * r]
    return sp


def round_robin(df, seeds, cap=128):
    """df: margin pairs with columns seed, name_j, order, pair_idx; seeds in S order."""
    lists = {s: list(g.sort_values(["name_j", "order"]).pair_idx) for s, g in df.groupby("seed")}
    lists = [lists.get(s, []) for s in seeds]
    out, r = [], 0
    while len(out) < cap and any(r < len(x) for x in lists):
        for x in lists:
            if r < len(x) and len(out) < cap:
                out.append(int(x[r]))
        r += 1
    return out


def add_strata(df, pv, key_col="probe_key"):
    df["stratum_assoc"] = df[key_col].map(pv.stratum_assoc).fillna("no_probe")
    df["stratum_neutral"] = df.stratum_assoc == "neutral"
    for v in X.PROBE_VERSIONS:
        for t in ("a", "b"):
            df[f"probe_{v}_{t}"] = df[key_col].map(pv[f"{v}_{t}"])
    return df


def strata_counts(df_idx, ids):
    sub = df_idx.loc[ids]
    out = {"pairs": int(len(sub)), "names": int(sub.names.sum()), "objects": int((~sub.names.astype(bool)).sum())}
    out.update({f"assoc_{k}": int(v) for k, v in sub.stratum_assoc.value_counts().items()})
    if "stratum_local_assoc" in sub:
        out["local_assoc"] = int(sub.stratum_local_assoc.fillna(False).astype(bool).sum())
    if "seed" in sub:
        out["seeds"] = int(sub.seed.nunique())
    if "order" in sub:
        out["order1"] = int((sub.order == 1).sum())
    if "congruent" in sub:
        out["congruent"] = int(sub.congruent.sum())
    return out


# ---------------------------------------------------------------------------------------------------------------
# condition builders: return (pair dataframe with pair_idx and all columns, splits {name: [pair_idx]}, pools, note)
# ---------------------------------------------------------------------------------------------------------------
def cond_f1(proto, pv, tok, st, ctx):
    pairs = pd.read_parquet(f"{D}/pairs_F1_train_xl_{proto}.parquet")
    sc = pd.read_parquet(f"{RESULTS}/wino_ref_{proto}/scan_pairs_F1.parquet").set_index("pair_id")
    df = pairs.join(sc[["d_clean_a", "d_clean_b", "margin", "top1_both", "correct_both", "drop_str"]], on="pair_id")
    df["pair_idx"] = df.pair_id.map(ctx["hit"])
    df["probe_key"] = [X.probe_key(r.names, r.word_a, r.word_b, r.ans_a, r.ans_b) for r in df.itertuples()]
    df = add_strata(df, pv)
    pool = df[df.margin & df.main & df.stratum_neutral]
    # 128 / 128 (+ replication) when the pool allows, else half / half (a 128 / rest split would be lopsided)
    sp = shuffle_split(pool.pair_idx, 128, rep=True) if len(pool) >= 256 else shuffle_split(pool.pair_idx)
    pools = {"margin_main_neutral": [int(i) for i in pool.pair_idx],
             "margin_main_by_stratum": df[df.margin & df.main].stratum_assoc.value_counts().to_dict(),
             "margin_connective": int((df.margin & ~df.main).sum())}
    return df, sp, pools, "pool = margin & content swap & neutral (both probes in R, |probe| < 1); seed-0 shuffle 128/128 + rep"


def _dir_pairs(proto, ctx):
    cs = json.load(open("/home/ubuntu/MOE/data/wino_str/case_sets.json"))
    inv = {v: k for k, v in cs["pair_idx_of"].items()}
    twins = {inv[i]: (int(i), s) for s, v in cs["pairs"].items() for i in v}
    f1 = pd.read_parquet(f"{RESULTS}/wino_ref_{proto}/scan_pairs_F1.parquet")
    f1 = f1[f1.pair_id.isin(twins)]
    passing = sorted(ctx["hit"][p] for p in f1[f1.margin].pair_id)
    return twins, f1, passing


def cond_dir_f1(proto, pv, tok, st, ctx):
    twins, f1, passing = _dir_pairs(proto, ctx)
    pairs = pd.read_parquet(f"{D}/pairs_F1_train_xl_{proto}.parquet")
    pairs = pairs[pairs.pair_id.isin(twins)]
    df = pairs.join(f1.set_index("pair_id")[["d_clean_a", "d_clean_b", "margin", "top1_both", "correct_both", "drop_str"]], on="pair_id")
    df["pair_idx"] = df.pair_id.map(ctx["hit"])
    df["phase3_split"] = df.pair_id.map(lambda p: twins[p][1])
    df["probe_key"] = [X.probe_key(r.names, r.word_a, r.word_b, r.ans_a, r.ans_b) for r in df.itertuples()]
    df = add_strata(df, pv)
    df = add_local_assoc(df, proto)
    sp = shuffle_split(passing)
    pools = {"twins_776_with_f1_pair": int(len(df)), "f1_margin": len(passing)}
    return df, sp, pools, "the 776 Phase-3 twins whose fill-in-the-blank pair passes the margin in this model; seed-0 shuffle half / half"


def add_local_assoc(df, proto):
    p3 = pd.read_parquet(f"{RESULTS}/wino_{proto}/scan_pairs.parquet").set_index("pair_id")
    df["stratum_local_assoc"] = df.pair_id.map(p3["assoc"]) if "base_pair_id" not in df else df.base_pair_id.map(p3["assoc"])
    return df


def cond_dir_opt(proto, pv, tok, st, ctx):
    twins, f1, passing = _dir_pairs(proto, ctx)
    p = pd.read_parquet(f"/home/ubuntu/MOE/data/wino_str/pairs_train_xl_{proto}.parquet")
    p = p[p.pair_id.isin(twins)].copy()
    p["str_pos"] = p.opt_pos
    p["pair_idx"] = p.pair_id.map(ctx["hit"])
    p["phase3_split"] = p.pair_id.map(lambda q: twins[q][1])
    pos = [X.positions_phase3(tok, st, dict(r._asdict())) for r in p.itertuples(index=False)]
    for k in ("mention_a_pos", "mention_b_pos"):
        p[k] = [x[k] for x in pos]
    # probe key of the twin (as the dir_f1 pair: word_diff of the sentences = Phase-3 trigger words)
    f1k = f1.set_index("pair_id")
    p["probe_key"] = [X.probe_key(f1k.at[q, "names"], f1k.at[q, "word_a"], f1k.at[q, "word_b"], f1k.at[q, "ans_a"], f1k.at[q, "ans_b"])
                      if q in f1k.index else None for q in p.pair_id]
    p = add_strata(p, pv)
    p = add_local_assoc(p, proto)
    sc = pd.read_parquet(f"{RESULTS}/wino_{proto}/scan_pairs.parquet").set_index("pair_id")
    for c in ("d_clean_a", "d_clean_b", "margin", "top1_both", "correct_both", "drop_str"):
        p[c] = p.pair_id.map(sc[c])
    sp = shuffle_split(passing)
    return p, sp, {"f1_margin": len(passing)}, "same twins and split as dir_f1 (Phase-3 option-swap format; all 776 pass the Phase-3 margin)"


def cond_f2e(proto, pv, tok, st, ctx):
    pairs = pd.read_parquet(f"{D}/pairs_F2_train_xl_{proto}.parquet")
    pairs = pairs[[blank_at_end(s) for s in pairs.sent_a]].copy()
    sc = pd.read_parquet(f"{RESULTS}/wino_ref_{proto}/scan_pairs_F2.parquet").set_index("pair_id")
    df = pairs.join(sc[["d_clean_a", "d_clean_b", "margin", "top1_both", "correct_both", "drop_str"]], on="pair_id")
    df["pair_idx"] = df.pair_id.map(ctx["hit"])
    df["probe_key"] = [X.probe_key(r.names, r.word_a, r.word_b, r.ans_a, r.ans_b) for r in df.itertuples()]
    df = add_strata(df, pv)
    pos = [X.positions_f2(tok, st, dict(r._asdict())) for r in df.itertuples(index=False)]
    for k in ("mention_a_pos", "mention_b_pos"):
        df[k] = [x[k] for x in pos]
    pool = df[df.margin & df.main & df.stratum_neutral]
    sp = shuffle_split(pool.pair_idx)
    pools = {"margin_main_neutral": [int(i) for i in pool.pair_idx], "margin_main_by_stratum": df[df.margin & df.main].stratum_assoc.value_counts().to_dict()}
    return df, sp, pools, "F2 pairs with no word after the blank; pool = margin & content & neutral; seed-0 shuffle half / half"


def respectively(df) -> bool:
    """True for the void 'Of X and Y, one .. and the other ..' copy builds (no prefix_style column), False for the options prefix."""
    return "prefix_style" not in df.columns or (df.prefix_style == "respectively").all()


def congruent(df, x_name):
    """Copy prefix 'Of X and Y, one .. t1 and the other .. t2': the pair is 'respectively'-congruent when the entity listed
    first (X = option1) is the one whose attribute is listed first (t1); then the X <-> t1 / Y <-> t2 mapping answers both
    twins without the sentence."""
    a_is_x = (df.ans_a == x_name).to_numpy()
    return pd.Series(np.where(df.order.to_numpy() == 1, a_is_x, ~a_is_x), index=df.index)


def _templates(df, pv, ctx, variant):
    rank, disc_seeds, val_seeds = ctx["seed_split"]
    df = df[df.variant == variant].copy()
    if "order" not in df:
        df["order"] = 1
    df["pair_idx"] = [1000 * rank[s] + 2 * int(j) + (int(o) - 1) for s, j, o in zip(df.seed, df.name_j, df.order)]
    assert df.pair_idx.is_unique
    df = add_strata(df, pv)
    df["stratum_seed"] = df.seed
    m = df[df.margin]
    if "prefix" in df.columns:  # copy conditions: both attribute orders of an item must pass (exact counterbalancing)
        if respectively(df):
            df["congruent"] = congruent(df, df.name_x)
            df["stratum_congruent"] = df.congruent
        ok = m.groupby(["seed", "name_j"]).order.nunique()
        ok = set(ok[ok == 2].index)
        df["both_orders_margin"] = [(s_, int(j)) in ok for s_, j in zip(df.seed, df.name_j)]
        m = df[df.margin & df.both_orders_margin]
    sp = {"discovery": round_robin(m[m.seed.isin(disc_seeds)], disc_seeds), "validation": round_robin(m[m.seed.isin(val_seeds)], val_seeds)}
    if "prefix" in df.columns:
        mm = df[df.margin]
        pools_cp = {"margin_pairs": int(len(mm)), "margin_order1": int((mm.order == 1).sum()), "margin_order2": int((mm.order == 2).sum()),
                    "both_orders_pairs": int(len(m)), "prefix_style": "respectively" if respectively(df) else "options"}
    else:
        pools_cp = {}
    pools = pools_cp | {"margin": int(len(m)), "margin_seeds": int(m.seed.nunique()), "margin_disc_seeds": int(m[m.seed.isin(disc_seeds)].seed.nunique()),
             "margin_val_seeds": int(m[m.seed.isin(val_seeds)].seed.nunique()), "pairs": int(len(df))}
    return df, sp, pools


def _name_j(df, ctx):
    """name-pair index j of the verbswap e2a pairs (pair_id = seed|variant|x|y) from the vt_a2e build (same name pairs)."""
    jj = ctx["name_j"]
    return [jj[(s, x, y)] for s, x, y in zip(df.seed, df.name_x, df.name_y)]


def cond_vt_e2a(variant):
    def f(proto, pv, tok, st, ctx):
        p = pd.read_parquet(f"{D}/verbswap_ent2attr_{proto}.parquet")
        sc = pd.read_parquet(f"{RESULTS}/wino_ref_{proto}/scan_pairs_verbs_ent2attr.parquet").set_index("pair_id")
        df = p.join(sc[["d_clean_a", "d_clean_b", "margin", "correct_both", "drop_str"]], on="pair_id")
        df["name_j"] = _name_j(df, ctx)
        df["str_pos"] = df.opt_pos
        df["probe_key"] = [X.probe_key(True, a.lower(), b.lower(), x, y) for a, b, x, y in zip(df.word_a, df.word_b, df.ans_a, df.ans_b)]
        df, sp, pools = _templates(df, pv, ctx, variant)
        pos = [X.positions_phase3(tok, st, dict(r._asdict())) for r in df.itertuples(index=False)]
        for k in ("mention_a_pos", "mention_b_pos"):
            df[k] = [x[k] for x in pos]
        return df, sp, pools, f"verb templates, variant {variant}, Phase-3 format; seed split S, <= 128 pairs per split round-robin over seeds"
    return f


def cond_vt_a2e(variant):
    def f(proto, pv, tok, st, ctx):
        df = pd.read_parquet(f"{RESULTS}/wino_ref_{proto}/scan_pairs_p5_vt_a2e.parquet")
        b = pd.read_parquet(f"{D}/p5build_vt_a2e_{proto}.parquet", columns=["pair_id", "ids_a", "ids_b"]).set_index("pair_id")
        df = df.join(b, on="pair_id")
        df, sp, pools = _templates(df, pv, ctx, variant)
        return df, sp, pools, f"verb templates, variant {variant}, fill-in-the-blank (entity readout); seed split S"
    return f


def cond_cp_vt(variant):
    def f(proto, pv, tok, st, ctx):
        df = pd.read_parquet(f"{RESULTS}/wino_ref_{proto}/scan_pairs_p5_cp_vt.parquet")
        b = pd.read_parquet(f"{D}/p5build_cp_vt_{proto}.parquet", columns=["pair_id", "ids_a", "ids_b"]).set_index("pair_id")
        df = df.join(b, on="pair_id")
        df, sp, pools = _templates(df, pv, ctx, variant)
        return df, sp, pools, f"verb templates + copy prefix, variant {variant}, both attribute orders; seed split S"
    return f


def cond_cp_nat(proto, pv, tok, st, ctx):
    df = pd.read_parquet(f"{RESULTS}/wino_ref_{proto}/scan_pairs_p5_cp_nat.parquet")
    b = pd.read_parquet(f"{D}/p5build_cp_nat_{proto}.parquet", columns=["pair_id", "ids_a", "ids_b"]).set_index("pair_id")
    df = df.join(b, on="pair_id")
    df["pair_idx"] = 2 * df.phase3_pair_idx.astype(int) + df.order.astype(int) - 1
    df = add_strata(df, pv)
    df = add_local_assoc(df, proto)
    from moetrace.ext7_wino import load_twins as _lt
    opt1 = {t.pair_id: t.opt1 for t in _lt("train_xl")[0]}
    if respectively(df):
        df["congruent"] = congruent(df, df.base_pair_id.map(opt1))
        df["stratum_congruent"] = df.congruent
    mm = df[df.margin]
    ok = mm.groupby("base_pair_id").order.nunique()
    ok = set(ok[ok == 2].index)
    df["both_orders_margin"] = df.base_pair_id.isin(ok)
    m = df[df.margin & df.both_orders_margin]  # both attribute orders of a twin must pass (exact counterbalancing)
    sp = {s: sorted(int(i) for i in m[m.phase3_split == s].pair_idx) for s in ("discovery", "validation", "rep_discovery", "rep_validation")}
    pools = {"pairs": int(len(df)), "twins": int(df.base_pair_id.nunique()), "margin_pairs": int(len(mm)), "margin_order1": int((mm.order == 1).sum()),
             "margin_order2": int((mm.order == 2).sum()), "prefix_style": "respectively" if respectively(df) else "options",
             "both_orders_pairs": int(len(m)), "both_orders_twins": int(m.base_pair_id.nunique()),
             "both_orders_shared_unused": int((m.phase3_split == "shared_unused").sum())}
    return df, sp, pools, "776 Phase-3 twins + copy prefix, both orders; Phase-3 split of the twin restricted to margin pairs (+ replication)"


BUILDERS = {"f1": cond_f1, "dir_f1": cond_dir_f1, "dir_opt": cond_dir_opt, "f2e": cond_f2e, "cp_nat": cond_cp_nat,
            "vt_e2a_orig": cond_vt_e2a("orig"), "vt_e2a_neu": cond_vt_e2a("neutral"), "vt_a2e_orig": cond_vt_a2e("orig"),
            "vt_a2e_neu": cond_vt_a2e("neutral"), "cp_vt_orig": cond_cp_vt("orig"), "cp_vt_neu": cond_cp_vt("neutral")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protos", default="qwen3,mixtral_bos")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--progress-suffix", default="", help="appended to the CASES READY line, e.g. ' (options prefix)'")
    a = ap.parse_args()
    reg_p = f"{D}/p5_conditions.json"
    reg = json.load(open(reg_p)) if os.path.exists(reg_p) else {"note": "Phase-5 condition registry (agent p5-data; scripts/ext13_p5_casesets.py). "
                                                                          "Per cond and proto: pair file, case-set file, run dir results/p5_<cond>_<proto>, families, split sizes, strata, status {cases, base}, selected_expert, pop_top10.",
                                                                  "conditions": {}}
    ctx = {"hit": hit_index(), "seed_split": seed_split()}
    for proto in a.protos.split(","):
        key, st = PROTOS[proto]
        tok, st = tokenizer(proto)
        out, pv = probe_outcome(proto)
        log(proto, "probe outcome", json.dumps({k: out[k] for k in ("frac_ge1", "responding", "R", "margin_strata", "templates_frac_ge1")}))
        vb = pd.read_parquet(f"{D}/p5build_vt_a2e_{proto}.parquet", columns=["seed", "name_x", "name_y", "name_j"])
        ctx["name_j"] = {(s, x, y): int(j) for s, x, y, j in zip(vb.seed, vb.name_x, vb.name_y, vb.name_j)}
        for cond in a.conds.split(","):
            df, sp, pools, note = BUILDERS[cond](proto, pv, tok, st, ctx)
            df = df[df.pair_idx.notna()].copy()
            df["pair_idx"] = df.pair_idx.astype(int)
            assert df.pair_idx.is_unique, cond
            for c in ("ids_a", "ids_b", "str_pos"):
                df[c] = [json.dumps([int(v) for v in jl(x)]) for x in df[c]]
            df["names"] = df.names.astype(bool)
            pf = f"{D}/pairs_p5_{cond}_{proto}.parquet"
            df.to_parquet(pf, index=False)
            di = df.set_index("pair_idx")
            sizes = {k: len(v) for k, v in sp.items()}
            behaviour_only = any(sizes.get(k, 0) < MIN_SPLIT for k in ("discovery", "validation"))
            fam = ["main"] + (["rep"] if "rep_discovery" in sp and sizes["rep_discovery"] >= MIN_SPLIT and sizes.get("rep_validation", 0) >= MIN_SPLIT else [])
            if "rep" not in fam:
                sp = {k: v for k, v in sp.items() if not k.startswith("rep_")}
            cs = {"note": f"ext13 Phase 5 condition {cond} ({FORMAT[cond]}), {proto}: {note}. pair_idx rule and splits: scripts/ext13_p5_casesets.py docstring. "
                          f"Association strata with R = {out['R']} (pre-registered probe rule).",
                  "pair_idx_of": {p: int(i) for p, i in zip(df.pair_id, df.pair_idx)},
                  "pairs": {k: [int(i) for i in v] for k, v in sp.items()},
                  "directed": {k: [2 * int(i) + d for i in v for d in (0, 1)] for k, v in sp.items()},
                  "strata": {k: strata_counts(di, v) for k, v in sp.items()},
                  "pools": pools}
            cf = f"{D}/case_sets_p5_{cond}_{proto}.json"
            json.dump(cs, open(cf, "w"), indent=1, default=int)
            reg = json.load(open(reg_p)) if os.path.exists(reg_p) else reg  # re-read: the base chain writes status.base meanwhile
            ent = reg["conditions"].setdefault(cond, {}).get(proto, {})
            ent.pop("rebuild_pending", None)
            if cond.startswith("cp_"):
                ent["note"] = (f"copy prefix style: {pools.get('prefix_style')} ('Options: t1, t2.' + newline before the Phase-3 prompt; coordinator decision "
                               "2026-10-11; the 'Of X and Y, one .. and the other ..' sets are void, kept as behaviour in data/wino_ref/void_cp_respectively/)")
            ent.update({"task": TASK[cond], "format": FORMAT[cond], "direction": DIRECTION[cond], "model": key, "proto": proto,
                        "pair_file": os.path.relpath(pf, "/home/ubuntu/MOE"), "case_sets": os.path.relpath(cf, "/home/ubuntu/MOE"),
                        "run_dir": f"results/p5_{cond}_{proto}", "families": fam, "split_sizes": {k: len(v) for k, v in sp.items()},
                        "strata_counts": cs["strata"], "behaviour_only": behaviour_only, "probe_R": out["R"],
                        "probe_flag_none_responds": out["flag_none_responds"]})
            ent.setdefault("status", {"cases": False, "base": False})
            ent["status"]["cases"] = not behaviour_only
            ent.setdefault("selected_expert", None)
            ent.setdefault("pop_top10", None)
            reg["conditions"].setdefault(cond, {})[proto] = ent
            json.dump(reg, open(reg_p, "w"), indent=1, default=int)
            log(proto, cond, "splits", ent["split_sizes"], "families", fam, "behaviour_only" if behaviour_only else "", json.dumps(cs["strata"].get("validation", {})))
            if not a.no_progress and not behaviour_only:
                with open("/home/ubuntu/MOE/logs/PROGRESS.md", "a") as fpr:
                    fpr.write(f"- {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())} [p5-data] CASES READY {cond}/{proto}{a.progress_suffix}\n")


if __name__ == "__main__":
    main()
