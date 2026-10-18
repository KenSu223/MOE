"""ext14 Phase 6 case sets and condition registry (CPU, agent p6-data; after scripts/ext14_scan.py).

Per cell and proto:
  data/dial/pairs_<cell>_<proto>.parquet     every scanned pair (pair contract + level / direction / order / src_pair_id / n_tokens,
                                             scan values d_clean_a / d_clean_b / margin, strata, token positions)
  data/dial/case_sets_<cell>_<proto>.json    generic case-set format (pair_idx_of, pairs, directed, strata, pools)
  data/dial/p6_conditions.json               registry (schema of data/wino_ref/p5_conditions.json + n_pool, n_margin,
                                             order_pass_rates, top1_rate, review, insufficient)
  PROGRESS lines "[p6-data] CASES READY <cell>/<proto> n_val=<k>" or "[p6-data] REVIEW <cell>/<proto>: <reason>"

Rules (RESEARCH_PLAN "Common rules"): an ITEM (WinoGrande twin / synthetic item) counts only if every order built for it passes
the margin (>= 1 both directions); splits = the item's fixed split (data/dial/twin_split.json for WinoGrande items; builder
splits for the ladder); WinoGrande cells keep association-neutral items only (Phase-5 probe rule; strong-prior items are Dial A);
cap 128 pairs per split, filled in priority order:
  WinoGrande cells: core-776 items first (extension items only when the cell has < 256 core margin pairs: --plan writes
  data/dial/build/need_ext_<proto>.txt for the extension scan), then the number of Dial-E cells (e1 / e1p / e2 / e3 x a2e / e2a)
  the item passes in this model (maximises the paired intersections across levels and directions), then items built in both
  tokenizer families, then the fixed twin rank;
  ladder cells: pairs built in both families first, then the builder's rank.
Set C conflict: two cells, sc_conflict_stmt_a2e (clean prefers the statement's answer, margin >= 1 both ways) and
sc_conflict_event_a2e (prefers the event's answer; trig_a / trig_b flipped so that the clean answer is 'true'); behaviour =
fraction of scanned items following the statement / the event / neither.
REVIEW triggers: order-pass-rate gap > 10 pp (two-order cells), < 64 validation pairs ("insufficient"), completeness failures.
Usage: python scripts/ext14_casesets.py <proto> [--cells ...] [--plan] [--no-progress]
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
from moetrace.models import RESULTS

REG = f"{X.DIAL}/p6_conditions.json"
DIALE = ["e1_a2e", "e1_e2a", "e1p_a2e", "e1p_e2a", "e2_a2e", "e2_e2a", "e3_a2e", "e3_e2a", "e3p3_a2e"]
WINO = DIALE + ["sc_agree_a2e", "sc_conflict_a2e", "rsite_e1_a2e"]
LADDER = ["l_ant", "l_prop", "l_bind", "l_ind"]
DIALA = ["e0_a2e", "e0_e2a", "aagr_a2e", "aagr_e2a", "acon_a2e", "acon_e2a"]
CAP, MIN_VAL, NEED = 128, 64, 256
FORMAT = {
    "e1": "E1 stated verbatim ('<E1> was X and <E2> was Y, so <blank clause>')", "e1p": "E1p paraphrased statements",
    "e2": "E2 negated statements ('<E1> was not Y and <E2> was not X')", "e3": "E3 original WinoGrande sentence",
    "e3p3": "E3 content in the P3 frame ('<event clause>. Fill in the blank: The _ was X.')", "e0": "E0 prior only ('Of the <E1> and the <E2>, <blank clause>')", "aagr": "Dial A agree (E1 frame, statements = prior)",
    "acon": "Dial A conflict (E1 frame, statements reversed vs prior)", "sc": "Set C (statement + original sentence)",
    "rsite": "R-site (E1-a2e, corruption = the two entity mentions in the evidence swapped)",
    "l_ant": "ladder: antonym ('The opposite of X is')", "l_prop": "ladder: object property (LRE commonsense + curated)",
    "l_bind": "ladder: explicit binding look-up ('The X is in the red box ... The red box contains the')",
    "l_ind": "ladder: induction on word pairs ('Pairs: a = b, c = d. a =')"}
TWO_ORDER = lambda cell: not (cell.startswith("e3_") or cell.startswith("e3p3_") or cell in ("l_ant", "l_prop"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def progress(line, enabled):
    if enabled:
        with open(f"{X.ROOT}/logs/PROGRESS.md", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [p6-data] {line}\n")


def load_scan(cell, proto, scan_cell=None):
    """Scanned pairs of a cell (all subsets), merged with the build (token ids)."""
    sc = scan_cell or cell
    fam = X.PROTOS[proto][2]
    od = f"{RESULTS}/dial_scan_{proto}"
    parts = [pd.read_parquet(f"{od}/scan_{sc}_{s}.parquet") for s in ("core", "ext", "ext2", "ext3", "ext4", "all")
             if os.path.exists(f"{od}/scan_{sc}_{s}.parquet")]
    if not parts:
        return None
    s = pd.concat(parts).drop_duplicates("pair_id")
    bp = f"{X.BUILD}/{sc}_{proto}.parquet"
    b = pd.read_parquet(bp if os.path.exists(bp) else f"{X.BUILD}/{sc}_{fam}.parquet",
                        columns=["pair_id", "ids_a", "ids_b", "prompt_a", "prompt_b"]).set_index("pair_id")
    s = s.join(b.rename(columns={"prompt_a": "_pa", "prompt_b": "_pb"}), on="pair_id")
    # a rebuild (e.g. a new frame) keeps pair ids: keep only scan rows whose prompts equal the current build
    stale = (s._pa != s.prompt_a) | (s._pb != s.prompt_b)
    if stale.any():
        log(cell, proto, f"dropping {int(stale.sum())} scanned pairs whose prompts differ from the current build")
    s = s[~stale].drop(columns=["_pa", "_pb"])
    return s


def item_pass(df, col="margin"):
    """src_pair_id -> True if every built order passes (two-order frames need both orders scanned)."""
    g = df.groupby("src_pair_id")
    ok = g[col].all()
    if "order" in df and df.order.nunique() > 1:
        ok &= g.order.nunique() == df.groupby("src_pair_id").order.nunique().max()
    return ok


def other_family_ids(cell, proto):
    fam = X.PROTOS[proto][2]
    other = "mistral" if fam == "qwen" else "qwen"
    p = f"{X.BUILD}/{cell.replace('_stmt', '').replace('_event', '')}_{other}.parquet"
    return set(pd.read_parquet(p, columns=["pair_id"]).pair_id) if os.path.exists(p) else set()


def strata_counts(df):
    out = {"pairs": int(len(df)), "items": int(df.src_pair_id.nunique()), "names": int(df.names.astype(bool).sum()) if "names" in df else 0}
    out["objects"] = out["pairs"] - out["names"]
    if "order" in df:
        out.update({f"order{int(k)}": int(v) for k, v in df.order.value_counts().items()})
    for c in ("core776", "antonym_pair", "both_fams", "degree_adv"):
        if c in df:
            out[c] = int(df[c].fillna(False).astype(bool).sum())
    for c in ("relation", "source", "para_src", "stratum_assoc", "antonym_type", "group", "frame"):
        if c in df and df[c].notna().any():
            out[c] = {str(k): int(v) for k, v in df[c].value_counts().items()}
    return out


def dialE_npass(proto):
    """Number of Dial-E cells each twin passes in this model (priority for the cap)."""
    n = {}
    for c in DIALE:
        s = load_scan(c, proto)
        if s is None:
            continue
        for k, v in item_pass(s).items():
            n[k] = n.get(k, 0) + int(v)
    return n


def select(df, ok_items, prio_cols, ascending):
    """Items in priority order per split until CAP pairs."""
    sp = {}
    for split in ("discovery", "validation"):
        sub = df[(df.split == split) & df.src_pair_id.isin(ok_items)]
        items = sub.drop_duplicates("src_pair_id").sort_values(prio_cols, ascending=ascending).src_pair_id.tolist()
        per = sub.groupby("src_pair_id").size()
        take, n = [], 0
        for k in items:
            if n + per[k] > CAP:
                break
            take.append(k)
            n += per[k]
        sp[split] = sorted(int(i) for i in sub[sub.src_pair_id.isin(take)].pair_idx)
    return sp


PHASE5_R = {"qwen3": ["v1", "v2"], "mixtral_bos": ["v2"]}


def probe_R(proto, vals):
    """Responding probes: Phase-5 R for qwen3 / mixtral_bos; Stage-2 models: the pre-registered 10 % rule (a probe responds if >= 10 %
    of the trigger probes of the model's passing attribute -> entity items (e3_a2e / e3p3_a2e margin items; all scanned items if none)
    reach |probe| >= 1); none responds -> both (flagged)."""
    if proto in PHASE5_R:
        return PHASE5_R[proto]
    cache = f"{RESULTS}/dial_scan_{proto}/probe_R.json"
    if os.path.exists(cache):
        return json.load(open(cache))["R"]
    keys = set()
    for c in ("e3_a2e", "e3p3_a2e"):
        sc = load_scan(c, proto)
        if sc is not None:
            ok = item_pass(sc)
            sub = sc[sc.src_pair_id.isin(ok[ok].index)]
            keys |= {X.probe_key(bool(n), str(a).lower(), str(b).lower(), x, y) for n, a, b, x, y in zip(sub.names, sub.word_a, sub.word_b, sub.ans_a, sub.ans_b)}
    if not keys:
        keys = set(vals)
    frac = {}
    for i, v in ((0, "v1"), (2, "v2")):
        x = np.array([vals[k][i:i + 2] for k in keys if k in vals], dtype=float).ravel()
        x = x[np.isfinite(x)]
        frac[v] = float((np.abs(x) >= 1).mean()) if len(x) else 0.0
    R = [v for v in ("v1", "v2") if frac[v] >= 0.10] or ["v1", "v2"]
    json.dump({"R": R, "frac_ge1": frac, "n_keys": len(keys), "flag_none_responds": not any(frac[v] >= 0.10 for v in frac)}, open(cache, "w"), indent=1)
    return R


def add_strata_cols(df, proto):
    """Association-probe values (Phase-5 v1 / v2 + Phase-6 probes) and the Phase-5 stratum rule for WinoGrande items."""
    from moetrace import ext13_p5 as P5
    vals = {}
    p5 = f"{RESULTS}/wino_ref_{proto}/p5_probe_values.parquet"
    if os.path.exists(p5):
        v = pd.read_parquet(p5)
        vals.update({r.key: (r.v1_a, r.v1_b, r.v2_a, r.v2_b) for r in v.itertuples()})
    p6 = f"{RESULTS}/dial_scan_{proto}/probe_values.parquet"
    if os.path.exists(p6):
        v = pd.read_parquet(p6)
        for r in v.itertuples():
            old = vals.get(r.key, (np.nan,) * 4)
            new = tuple(getattr(r, c, np.nan) for c in ("v1_a", "v1_b", "v2_a", "v2_b"))
            vals[r.key] = tuple(o if (o == o and o is not None) else n for o, n in zip(old, new))
    R = probe_R(proto, vals)
    keys = [X.probe_key(bool(n), str(a).lower(), str(b).lower(), x, y) for n, a, b, x, y in zip(df.names, df.word_a, df.word_b, df.ans_a, df.ans_b)]
    arr = np.array([vals.get(k, (np.nan,) * 4) for k in keys], dtype=float).reshape(len(df), 4)
    df["probe_key"] = keys
    df["probe_v1_a"], df["probe_v1_b"], df["probe_v2_a"], df["probe_v2_b"] = arr.T
    df["stratum_assoc"] = [P5.assoc_stratum({"v1": (r[0], r[1]), "v2": (r[2], r[3])}, R) for r in arr]
    return df


def flip(s, rows=None, tag="|rev"):
    """Swap the answer orientation of the selected rows (clean answer := the other token): trig / answers / Delta / margins / top-1."""
    m = np.ones(len(s), bool) if rows is None else np.asarray(rows, bool)
    s = s.copy()
    for a_, b_ in (("trig_a", "trig_b"), ("answer_a", "answer_b")):
        x, y = s.loc[m, a_].copy(), s.loc[m, b_].copy()
        s.loc[m, a_], s.loc[m, b_] = y.values, x.values
    s.loc[m, "d_clean_a"], s.loc[m, "d_clean_b"] = -s.loc[m, "d_clean_a"].values, -s.loc[m, "d_clean_b"].values
    mg, rv = s.loc[m, "margin"].copy(), s.loc[m, "reverse_margin"].copy()
    s.loc[m, "margin"], s.loc[m, "reverse_margin"] = rv.values, mg.values
    s.loc[m, "pair_id"] = s.loc[m, "pair_id"] + tag
    s["top1_both"] = (s.top1_a == s.trig_a) & (s.top1_b == s.trig_b)
    s["correct_both"] = (s.d_clean_a > 0) & (s.d_clean_b < 0)
    return s


def dialA_frame(cell, proto, min_strength):
    """aagr_* / acon_* / e0_* from the Dial-A builds: each item's prior in this model (scripts/ext14_build_dialA.prior_of) labels the
    statement variant agree / conflict; E0 answers follow the prior (reverse-prior items flipped)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bda", f"{X.ROOT}/scripts/ext14_build_dialA.py")
    bda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bda)
    lev, direction = cell.split("_")
    src = f"dA_e1_{direction}" if lev in ("aagr", "acon") else f"e0_{direction}"
    s = load_scan(src, proto)
    if s is None:
        return None
    vals = bda.prior_table(proto)
    pri = {}
    for r in s.drop_duplicates("src_pair_id").itertuples():
        k = X.probe_key(bool(r.names), str(r.word_a).lower(), str(r.word_b).lower(), r.ans_a, r.ans_b)
        pri[r.src_pair_id] = bda.prior_of(vals.get(k), proto)
    s["prior_sign"] = s.src_pair_id.map(lambda k: pri[k][0])
    s["prior_strength"] = s.src_pair_id.map(lambda k: pri[k][1])
    s["prior_s_a"] = s.src_pair_id.map(lambda k: pri[k][2])
    s["prior_s_b"] = s.src_pair_id.map(lambda k: pri[k][3])
    s = s[(s.prior_sign != 0) & (s.prior_strength >= min_strength)].copy()
    if lev in ("aagr", "acon"):
        agree = ((s.stmt == "bind") & (s.prior_sign == 1)) | ((s.stmt == "swap") & (s.prior_sign == -1))
        s["stmt_agrees_prior"] = agree
        s = s[agree if lev == "aagr" else ~agree].copy()
    else:
        s = flip(s, (s.prior_sign == -1).to_numpy())
    # prior support toward the stated / answered binding of prompt A (for the dose-response)
    s["prior_support"] = np.where((lev == "acon"), -s.prior_strength, s.prior_strength)
    return s


# all-strata variants (coordinator 2026-10-17): dense vs MoE / capacity comparisons only; the neutral cells stay primary for the dial
ALL_STRATA = ("e1_a2e", "e1_e2a", "e1p_a2e", "e1p_e2a")
# counterpart whose probes give the second association label, and whose passing items are preferred for the Stage-1 _all cells
TWINS = {"mixtral_bos": ["mistral7b_bos"], "qwen3": ["qwen3_4b", "qwen3_8b", "qwen3_14b"]}


def build_cell(cell, proto, npass, reg, a):
    all_strata = cell.endswith("_all") and cell[:-4] in ALL_STRATA
    base = (cell[:-4] if all_strata else cell).replace("_stmt", "").replace("_event", "")
    if cell.split("_")[0] in ("aagr", "acon", "e0"):
        s = dialA_frame(cell, proto, a.min_strength)
    else:
        s = load_scan(base, proto)
    if s is None:
        log(cell, proto, "not scanned")
        return None
    ent = {"cell": cell, "level": base.split("_")[0] if not cell.startswith("l_") else cell, "direction": base.split("_")[-1] if not cell.startswith("l_") else "",
           "format": FORMAT.get(base.split("_")[0] if not cell.startswith("l_") else cell, cell) + (" [all association strata]" if all_strata else ""),
           "model": X.PROTOS[proto][0], "proto": proto}
    col = "margin"
    if cell == "sc_conflict_event_a2e":
        col = "reverse_margin"
    s = s.copy()
    if cell == "sc_conflict_event_a2e":  # event orientation: the clean answer is the event's answer
        s = flip(s, None, "|event")
        s["orientation"] = "event"
    elif cell == "sc_conflict_stmt_a2e":
        s["orientation"] = "statement"
    ok = item_pass(s, "margin")
    ok_items = set(ok[ok].index)
    wino = base in WINO
    n_pass_all_strata = len(ok_items)
    if wino:
        s = add_strata_cols(s, proto)
        # Dial-E / Set C / R-site main sets are association-neutral (coordinator 2026-10-16, as the Phase-5 main sets): both triggers
        # |probe| < 1 under the model's responding probes; strong-prior items are used in Dial A only
        neutral = set(s[s.stratum_assoc == "neutral"].src_pair_id)
        if not all_strata:
            ok_items &= neutral
        else:  # association class by the model's own probes and by the counterpart's probes (strata for the analysis)
            cps = [X.PREFER[proto]] if proto in X.PREFER else [t for t in TWINS.get(proto, []) if os.path.exists(f"{RESULTS}/dial_scan_{t}/probe_values.parquet")]
            s["assoc_own"] = s.stratum_assoc
            s["stratum_neutral_own"] = s.stratum_assoc == "neutral"
            if cps:
                c2 = add_strata_cols(s[["names", "word_a", "word_b", "ans_a", "ans_b"]].copy(), cps[0])
                s["assoc_counterpart"] = c2.stratum_assoc.values
                s["assoc_counterpart_model"] = cps[0]
                s["stratum_neutral_counterpart"] = (c2.stratum_assoc == "neutral").values
        s["n_pass_dialE"] = s.src_pair_id.map(npass).fillna(0).astype(int)
        s["is_core"] = s.core776.astype(bool)
    oth = other_family_ids(base, proto)
    s["both_fams"] = s.pair_id.str.replace("|event", "", regex=False).isin(oth)
    for c in ("order",):
        s[f"stratum_{c}"] = s[c]
    if "names" in s:
        s["stratum_names"] = s.names.astype(bool)
    pref = X.PREFER.get(proto)
    pref_items = set()
    if pref and os.path.exists(f"{X.DIAL}/case_sets_{cell}_{pref}.json"):
        pcs = json.load(open(f"{X.DIAL}/case_sets_{cell}_{pref}.json"))
        sel_idx = set(pcs["pairs"].get("discovery", [])) | set(pcs["pairs"].get("validation", []))
        pdf = pd.read_parquet(f"{X.DIAL}/pairs_{cell}_{pref}.parquet", columns=["pair_idx", "src_pair_id"])
        pref_items = set(pdf[pdf.pair_idx.isin(sel_idx)].src_pair_id)
    if all_strata and proto in TWINS:  # Stage-1 _all cells: prefer items that also pass in the dense twin / series (identical items)
        for t in TWINS[proto]:
            st_ = load_scan(base, t)
            if st_ is not None:
                okt = item_pass(st_)
                pref_items |= set(okt[okt].index)
        pref = ",".join(TWINS[proto])
    s["preferred"] = s.src_pair_id.isin(pref_items)
    if wino:
        s["stratum_core"] = s.core776.astype(bool)
        s["stratum_antonym"] = s.antonym_pair.astype(bool) if "antonym_pair" in s else False
        # E2 semantics (coordinator QA 2026-10-16): with a degree adverb or a gradable pair, 'not too big' does not entail 'too small'
        s["degree_adv"] = s.adv.fillna("").astype(str).str.len() > 0
        s["antonym_type"] = [X.antonym_type(str(a), str(b)) for a, b in zip(s.word_a, s.word_b)]
        s["stratum_degree_adv"] = s.degree_adv
        s["stratum_antonym_type"] = s.antonym_type
        prio, asc = ["is_core", "preferred", "n_pass_dialE", "both_fams", "rank"], [False, False, False, False, True]
        if all_strata or proto in X.PREFER:  # twin / series comparisons: identical items first (from 2026-10-17 03:25Z; mistral7b_bos
            # Set-A cells registered at 02:08Z keep core-first)
            prio, asc = ["preferred", "is_core", "n_pass_dialE", "both_fams", "rank"], [False, False, False, False, True]
    else:
        prio, asc = ["preferred", "both_fams", "rank"], [False, False, True]
    # extension need (WinoGrande cells): core margin pairs of passing items
    core_margin_pairs = int(s[s.src_pair_id.isin(ok_items) & s.core776.astype(bool)].shape[0]) if wino else None  # neutral items
    ext_scanned = bool(wino and (~s.core776.astype(bool)).any())
    sp = select(s, ok_items, prio, asc)
    pidx = sorted(set(sp["discovery"]) | set(sp["validation"]))
    sel = s[s.pair_idx.isin(pidx)]
    # behaviour / pools
    two = TWO_ORDER(base)
    by_order = {}
    if two and "order" in s:
        full = s.groupby("src_pair_id").order.nunique()
        sub = s[s.src_pair_id.isin(full[full == 2].index)]
        by_order = {int(k): round(float(v), 4) for k, v in sub.groupby("order").margin.mean().items()}
    gap = abs(by_order.get(1, 0) - by_order.get(2, 0)) if len(by_order) == 2 else 0.0
    pools = {"n_pool_pairs": int(len(s)), "n_pool_items": int(s.src_pair_id.nunique()), "n_margin_pairs": int(s.margin.sum()),
             "n_items_pass_all_strata": n_pass_all_strata, "n_items_pass": len(ok_items),
             "assoc_strata_pool_items": ({str(k): int(v) for k, v in s.drop_duplicates("src_pair_id").stratum_assoc.value_counts().items()} if wino else None), "n_pairs_of_passing_items": int(s.src_pair_id.isin(ok_items).sum()),
             "core_margin_pairs_of_passing_items": core_margin_pairs, "extension_scanned": ext_scanned,
             "order_pass_rates": by_order, "order_gap": round(gap, 4),
             "top1_rate_pool": round(float(s.top1_both.mean()), 4), "top1_rate_selected": round(float(sel.top1_both.mean()), 4) if len(sel) else None,
             "margin_rate_pool": round(float(s.margin.mean()), 4)}
    if base == "sc_conflict_a2e":
        st = item_pass(s if cell != "sc_conflict_event_a2e" else s.assign(margin=s.margin), "margin")
        raw = load_scan(base, proto)
        pools["behaviour_items"] = {"statement_following": int(item_pass(raw, "margin").sum()), "event_following": int(item_pass(raw, "reverse_margin").sum()),
                                    "items": int(raw.src_pair_id.nunique())}
        pools["behaviour_pairs"] = {"statement_margin": float(raw.margin.mean()), "event_margin": float(raw.reverse_margin.mean()),
                                    "statement_prefers": float(((raw.d_clean_a > 0) & (raw.d_clean_b < 0)).mean()),
                                    "event_prefers": float(((raw.d_clean_a < 0) & (raw.d_clean_b > 0)).mean())}
    n_val = len(sp["validation"])
    n_all = len(sp["discovery"]) + n_val
    # coordinator rule (2026-10-16): < 64 validation but >= 64 pairs in discovery + validation -> 'pooled' (tier 1 on all pairs; layers /
    # heads chosen on the same items); < 64 pooled -> behaviour only ('insufficient'). Pooled sets keep family 'main' with discovery =
    # validation = all pairs, so the runners and analyses work unchanged (registry flag pooled: true).
    pooled = n_val < MIN_VAL and n_all >= MIN_VAL
    insufficient = n_all < MIN_VAL if n_val < MIN_VAL else False
    if pooled:
        allp = sorted(set(sp["discovery"]) | set(sp["validation"]))
        sp = {"discovery": allp, "validation": allp}
    # which answer wins: items whose clean prompts prefer the cell's answer in every order (target) / the other answer in every order
    # (other = prior for acon, the event for sc_conflict_stmt, the copied word for e2, the other entity / attribute otherwise)
    okr = item_pass(s, "reverse_margin")
    other_name = {"acon": "prior", "sc": "event" if cell == "sc_conflict_stmt_a2e" else ("statement" if cell == "sc_conflict_event_a2e" else "other"),
                  "e2": "copied word (copy trap)"}.get(cell.split("_")[0], "other answer")
    pools["behaviour"] = {"items": int(s.src_pair_id.nunique()), "target_wins_all_orders": int(item_pass(s, "margin").sum()),
                          "other_wins_all_orders": int(okr.sum()), "other_is": other_name,
                          "pair_rate_target_margin": round(float(s.margin.mean()), 4), "pair_rate_other_margin": round(float(s.reverse_margin.mean()), 4),
                          "pair_rate_target_prefers": round(float(((s.d_clean_a > 0) & (s.d_clean_b < 0)).mean()), 4),
                          "pair_rate_other_prefers": round(float(((s.d_clean_a < 0) & (s.d_clean_b > 0)).mean()), 4)}
    reasons = []
    if gap > 0.10:
        reasons.append(f"order gap {gap:.2f} > 0.10 (pass rates by order {by_order})")
    # files
    out = s.copy()
    for c in ("ids_a", "ids_b", "str_pos"):
        out[c] = [json.dumps([int(v) for v in (json.loads(x) if isinstance(x, str) else x)]) for x in out[c]]
    out["names"] = out.names.astype(bool) if "names" in out else False
    pf = f"{X.DIAL}/pairs_{cell}_{proto}.parquet"
    out.to_parquet(pf, index=False)
    di = out.set_index("pair_idx")
    cs = {"note": f"ext14 Phase 6 cell {cell}, {proto}: items passing the margin in every order; fixed item split; cap {CAP} pairs per split "
                  f"(priority {prio}). scripts/ext14_casesets.py docstring.",
          "pair_idx_of": {p: int(i) for p, i in zip(out.pair_id, out.pair_idx)},
          "pairs": sp, "directed": {k: [2 * int(i) + d for i in v for d in (0, 1)] for k, v in sp.items()},
          "strata": {k: strata_counts(di.loc[v].reset_index()) for k, v in sp.items() if v}, "pools": pools}
    cf = f"{X.DIAL}/case_sets_{cell}_{proto}.json"
    json.dump(cs, open(cf, "w"), indent=1, default=int)
    old = reg["conditions"].get(cell, {}).get(proto, {})
    ent.update({"pair_file": os.path.relpath(pf, X.ROOT), "case_sets": os.path.relpath(cf, X.ROOT), "run_dir": f"results/p6_{cell}_{proto}",
                "families": ["main"], "split_sizes": {k: len(v) for k, v in sp.items()}, "strata_counts": cs["strata"],
                "behaviour_only": bool(insufficient), "probe_R": (probe_R(proto, {}) if wino else None), "preferred_from": pref,
                "all_strata": bool(all_strata), "use": ("dense vs MoE / capacity comparisons only (all association strata); not for the "
                                                        "pre-registered dial predictions" if all_strata else None),
                "pooled": bool(pooled), "family_note": ("pooled: discovery = validation = all passing pairs (selection and evaluation on the same "
                                                        "items)" if pooled else None),
                "behaviour": pools["behaviour"],
                "n_pool": pools["n_pool_pairs"], "n_margin": pools["n_margin_pairs"], "order_pass_rates": by_order,
                "top1_rate": pools["top1_rate_selected"], "insufficient": bool(insufficient), "review": "; ".join(reasons) or None,
                "unit": "src_pair_id"})
    ent["status"] = old.get("status", {"cases": False, "base": False})
    ent["status"]["cases"] = not insufficient
    if "review_ok" in old:
        ent["review_ok"] = old["review_ok"]
    reg["conditions"].setdefault(cell, {})[proto] = ent
    log(cell, proto, "splits", ent["split_sizes"], "pool", pools["n_pool_items"], "items pass", len(ok_items), "order", by_order,
        "top1", ent["top1_rate"], "REVIEW: " + ent["review"] if ent["review"] else "")
    return ent


def plan_ext(proto, cells):
    """Cells whose core margin pairs < NEED and whose extension items are not scanned yet -> need_ext_<proto>.txt."""
    need = []
    for c in cells:
        base = c.replace("_stmt", "").replace("_event", "")
        if base not in WINO:
            continue
        s = load_scan(base, proto)
        if s is None:
            continue
        ok = item_pass(s)
        s = add_strata_cols(s, proto)
        okc = s[s.src_pair_id.isin(ok[ok].index) & s.core776.astype(bool) & (s.stratum_assoc == "neutral")]
        if len(okc) < NEED and not (~s.core776.astype(bool)).any() and base not in need:
            need.append(base)
        log(proto, base, "core margin pairs of passing items", len(okc), "-> extension" if base in need else "")
    if proto == "qwen2_57b":  # tier 1 trimmed (coordinator 2026-10-17): extension scans only for the cells that run tier 1
        need = [c for c in need if c in ("e1_a2e", "e1_e2a", "e3p3_a2e", "e3_e2a")]
    with open(f"{X.BUILD}/need_ext_{proto}.txt", "w") as f:
        f.write("\n".join(need) + ("\n" if need else ""))
    return need


def anchors(reg, proto):
    """l_cf / l_ioi anchors for qwen3 / mixtral_bos: existing Phase-3/4/6 runs (no new GPU)."""
    m = {"qwen3": ("qwen3", "qwen3"), "mixtral_bos": ("mixtral_bos", "mixtral_bos")}[proto]
    a = {"l_cf": {"format": "ladder anchor: CounterFact STR (Direction 6 / Phase 3)",
                  "anchor_runs": {"sweep_moe_and_experts": f"results/{m[0]}_str", "sweep_attention": f"results/{m[0]}_str_attnsweep",
                                  "heads": f"results/{m[0]}_str_circuit", "grid_moe": f"results/{m[0]}_str (str_grid_w1, MoE output only)"},
                  "known_share": {"qwen3": 0.54, "mixtral_bos": 0.58}[proto]},
         "l_ioi": {"format": "ladder anchor: IOI, S2 -> IO swap (Phase 3)",
                   "anchor_runs": {"run": f"results/ioi_{m[1]}_s2io", "grid": f"results/ioi_{m[1]}_s2io_grid",
                                   "heads": f"results/ioi_{m[1]}_s2io_circuit" if proto == "qwen3" else None,
                                   "pairs": f"data/ioi/pairs_s2io_{proto}.parquet", "case_sets": "data/ioi/case_sets_s2io.json"},
                   "known_share": {"qwen3": 0.92, "mixtral_bos": 0.80}[proto]}}
    for cell, e in a.items():
        ent = reg["conditions"].setdefault(cell, {}).get(proto, {})
        ent.update({"cell": cell, "level": cell, "direction": "", "model": X.PROTOS[proto][0], "proto": proto, "anchor": True,
                    "run_dir": e["anchor_runs"].get("run") or e["anchor_runs"]["sweep_moe_and_experts"], "families": ["main"],
                    "behaviour_only": False, "status": {"cases": True, "base": True}, "insufficient": False, "review": None, **e})
        reg["conditions"][cell][proto] = ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto", choices=list(X.PROTOS))
    ap.add_argument("--cells", default=",".join(WINO + LADDER))
    ap.add_argument("--plan", action="store_true", help="only write need_ext_<proto>.txt")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--min-strength", type=float, default=1.0, help="Dial A / E0: minimal prior strength min(|s_a|, |s_b|)")
    ap.add_argument("--behaviour-only", action="store_true", help="register as behaviour only by design (Stage-2 Set C)")
    a = ap.parse_args()
    cells = []
    for c in a.cells.split(","):
        cells += ["sc_conflict_stmt_a2e", "sc_conflict_event_a2e"] if c == "sc_conflict_a2e" else [c]
    if a.plan:
        log("need extension:", plan_ext(a.proto, cells))
        return
    reg = json.load(open(REG)) if os.path.exists(REG) else {
        "note": "Phase-6 condition registry (agent p6-data; scripts/ext14_casesets.py). Schema of data/wino_ref/p5_conditions.json: per cell "
                "and proto pair_file, case_sets, run_dir results/p6_<cell>_<proto>, families, split_sizes, strata_counts, behaviour_only, "
                "probe_R, status {cases, base}; plus n_pool, n_margin, order_pass_rates, top1_rate, insufficient, review (reason or null), "
                "review_ok (set when the coordinator answers REVIEW OK), anchor entries for l_cf / l_ioi.", "conditions": {}}
    if a.anchors:
        anchors(reg, a.proto)
    npass = dialE_npass(a.proto)
    for cell in cells:
        reg = json.load(open(REG)) if os.path.exists(REG) else reg  # re-read (the base chain writes status.base meanwhile)
        if a.anchors:
            anchors(reg, a.proto)
        ent = build_cell(cell, a.proto, npass, reg, a)
        if ent is None:
            continue
        trimmed = a.proto == "qwen2_57b" and cell not in ("l_ant", "l_prop", "l_cf", "l_ioi", "l_bind", "l_ind", "e1_a2e", "e1_e2a", "e3p3_a2e",
                                                         "e3_e2a", "e1_a2e_all", "e1_e2a_all")
        if trimmed and not ent.get("behaviour_only"):  # coordinator 2026-10-17: Qwen2-57B tier 1 only on the twin-comparison cells
            ent.update(behaviour_only=True, behaviour_only_by_design=True, tier1_trimmed=True, status={"cases": False, "base": False})
            json.dump(reg, open(REG, "w"), indent=1, default=int)
            progress(f"BEHAVIOUR ONLY {cell}/{a.proto} (tier 1 trimmed for Qwen2-57B; scan-level behaviour recorded)", not a.no_progress)
            continue
        if a.behaviour_only:
            ent.update(behaviour_only=True, behaviour_only_by_design=True, status={"cases": False, "base": False})
            json.dump(reg, open(REG, "w"), indent=1, default=int)
            b = ent["behaviour"]
            progress(f"BEHAVIOUR ONLY {cell}/{a.proto} (by design, Stage-2 Set C): items {b['items']}: target answer wins in every order "
                     f"{b['target_wins_all_orders']}, {b['other_is']} wins {b['other_wins_all_orders']}; pair rates margin "
                     f"{b['pair_rate_target_margin']:.2f} vs {b['pair_rate_other_margin']:.2f}", not a.no_progress)
            continue
        json.dump(reg, open(REG, "w"), indent=1, default=int)
        if ent["review"]:
            progress(f"REVIEW {cell}/{a.proto}: {ent['review']}", not a.no_progress)
        if ent["insufficient"]:
            b = ent["behaviour"]
            progress(f"BEHAVIOUR ONLY {cell}/{a.proto}: {len(set(json.load(open(f'{X.ROOT}/' + ent['case_sets']))['pairs'].get('discovery', [])) | set(json.load(open(f'{X.ROOT}/' + ent['case_sets']))['pairs'].get('validation', [])))} "
                     f"pairs < 64 pooled (insufficient); items {b['items']}: target answer wins in every order {b['target_wins_all_orders']}, "
                     f"{b['other_is']} wins {b['other_wins_all_orders']}; pair rates margin {b['pair_rate_target_margin']:.2f} vs {b['pair_rate_other_margin']:.2f}",
                     not a.no_progress)
        else:
            progress(f"CASES READY {cell}/{a.proto} n_val={ent['split_sizes']['validation']}" + (" (pooled)" if ent["pooled"] else ""), not a.no_progress)


if __name__ == "__main__":
    main()
