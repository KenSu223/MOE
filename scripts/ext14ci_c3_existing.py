"""ext14 C3 concept circuits from EXISTING rows (p6-circuit; scope cut 2026-10-17: no new sweeps / singles).

Tasks per model (selection rows = existing single-component patches on the SOURCE validation split, where the Phase-4
all-layer head runs live; held-out evaluation = the source discovery split, used only by the GPU validation):
  SIZE_STATE   ss_e3   Phase-3 WinoGrande option swap (entity -> attribute; ~ Dial-E E3), answers in the size / state lexicon
                       heads: wino_<p>_str_circuit (all layers); experts: wino_<p>_str (all layers)
               ss_e1   Phase-6 E1 entity -> attribute (stated verbatim), same lexicon
                       heads: p6_e1_e2a_<p>[_t2]/head_rows (5 layers); experts: p6_e1_e2a_<p>_t2 (if run)
  COUNTRY      co_p17  CounterFact P17 (country of a place);  co_nat  P27 + P495 (citizenship, country of origin)
                       heads: <p>_str_circuit (all layers, validation cases); experts: <p>_str (all layers)
  PERSON_ROLE  pr_ioi  IOI S2 -> IO; heads: ioi_qwen3_s2io_circuit (all layers) / ioi_mixtral_bos_s2io/head_rows (5 layers);
                       no expert singles exist for IOI
               pr_f1n  Phase-5 F1 attribute -> entity, name items; heads p5_f1_<p>/head_rows (5 layers), experts all layers
               pr_e1n  Phase-6 E1 attribute -> entity, name items; heads + experts p6_e1_a2e_<p>_t2
  control      ct_lang CounterFact language relations (P103, P1412, P364, P37): CounterFact without countries
effect = single-component rescue / drop of the case (CounterFact: donor mean). Top set of a task = top-10 heads + top-10
experts by mean effect on the selection cases. Concept circuit = in the top set of >= 2 tasks of the concept and in no top set
of another concept's task (control tasks do not exclude; overlaps reported). A component can only enter a task's top set where
it was measured (head layers differ between runs; IOI has no expert rows).

Writes results/p6ci_c3_data/pairs_<task>_<proto>.parquet + case_sets_<task>_<proto>.json (the evaluation pairs, directed ids
under "validation" for scripts/ext14ci_c3_validate.py), circuits_<proto>.json, effects_<proto>.parquet; with --report (after the
validation run) the tables results/tables/ext14ci_c3_<proto>.md and the 'c3' summary key.

Usage: python scripts/ext14ci_c3_existing.py --proto qwen3|mixtral_bos [--report]
"""
import argparse, glob, json, os, random, sys
sys.path.insert(0, "/home/ubuntu/MOE")
sys.path.insert(0, "/home/ubuntu/MOE/scripts")
import numpy as np, pandas as pd
from moetrace import ext14_circuit_c3 as C3
from moetrace import ext14_circuit as C
from moetrace.models import RESULTS, MODELS

R = RESULTS
CAP = 48  # evaluation pairs per task
K = C3.K_TOP
CF_REL = {"co_p17": ("P17",), "co_nat": ("P27", "P495"), "ct_lang": ("P103", "P1412", "P364", "P37")}
CONCEPT = {"ss_e3": "SIZE_STATE", "ss_e1": "SIZE_STATE", "co_p17": "COUNTRY", "co_nat": "COUNTRY", "pr_ioi": "PERSON_ROLE",
           "pr_f1n": "PERSON_ROLE", "pr_e1n": "PERSON_ROLE", "ct_lang": "CONTROL"}
ORDER = ("ss_e3", "ss_e1", "co_p17", "co_nat", "pr_ioi", "pr_f1n", "pr_e1n", "ct_lang")


def spec(task, proto):
    m = "qwen3" if proto == "qwen3" else "mixtral_bos"
    first = lambda *ps: next((p for p in ps if os.path.exists(os.path.join(R, p))), None)
    if task == "ss_e3":
        return dict(kind="pairs", pairs=f"data/wino_str/pairs_train_xl_{m}.parquet", cs="data/wino_str/case_sets.json", sel="ss_words",
                    heads=[f"wino_{proto}_str_circuit"], experts=f"wino_{proto}_str")
    if task == "ss_e1":
        d = dict(kind="pairs", pairs=f"data/dial/pairs_e1_e2a_{proto}.parquet", cs=f"data/dial/case_sets_e1_e2a_{proto}.json",
                 sel="ss_answers", heads=[first(f"p6_e1_e2a_{proto}_t2/head_rows.parquet", f"p6_e1_e2a_{proto}/head_rows.parquet")],
                 experts=f"p6_e1_e2a_{proto}_t2")
        if not os.path.exists(os.path.join(R, d["experts"], "str_expert_rows.parquet")) and \
                os.path.exists(os.path.join(R, f"p6ci_c3_dla_ss_e1_{proto}", ".done_dla")):
            d.update(experts=None, expert_dla=f"p6ci_c3_dla_ss_e1_{proto}/expert_dla.parquet")  # tier 2 cancelled: expert DLA
        return d
    if task in CF_REL and os.path.exists(os.path.join(C3.DATA, f"pairs_cfc_{proto}.parquet")) \
            and os.path.exists(os.path.join(R, f"p6ci_c3_cfc_{proto}", ".done_dla")):
        # COUNTRY CounterFact pair sets of scripts/ext14ci_c3_country.py (NEED HEADS cf_country): heads + expert DLA
        return dict(kind="cfc", heads=[f"p6ci_c3_cfc_{proto}/head_rows.parquet"], experts=None,
                    expert_dla=f"p6ci_c3_cfc_{proto}/expert_dla.parquet")
    if task in CF_REL:
        return dict(kind="cf", run=f"{proto}_str", rels=CF_REL[task], heads=[f"{proto}_str_circuit"], experts=f"{proto}_str")
    if task == "pr_ioi":
        return dict(kind="pairs", pairs=f"data/ioi/pairs_s2io_{m}.parquet", cs="data/ioi/case_sets_s2io.json", sel="all",
                    heads=[first(f"ioi_{proto}_s2io_circuit", f"ioi_{proto}_s2io/head_rows.parquet")], experts=None)
    if task == "pr_f1n":
        return dict(kind="pairs", pairs=f"data/wino_ref/pairs_p5_f1_{m}.parquet", cs=f"data/wino_ref/case_sets_p5_f1_{m}.json",
                    sel="names", heads=[f"p5_f1_{proto}/head_rows.parquet"], experts=f"p5_f1_{proto}")
    if task == "pr_e1n":
        return dict(kind="pairs", pairs=f"data/dial/pairs_e1_a2e_{proto}.parquet", cs=f"data/dial/case_sets_e1_a2e_{proto}.json",
                    sel="names", heads=[f"p6_e1_a2e_{proto}_t2/head_rows.parquet"], experts=f"p6_e1_a2e_{proto}_t2")
    raise KeyError(task)


def ab(p):
    return p if os.path.isabs(p) else os.path.join("/home/ubuntu/MOE", p)


def item_ok(pairs, sel):
    if sel == "ss_words":
        return {p for p, a, b in zip(pairs.pair_id, pairs.word_a, pairs.word_b) if str(a).strip().lower() in C3.SS and str(b).strip().lower() in C3.SS}
    if sel == "ss_answers":
        ca, cb = ("answer_a", "answer_b") if "answer_a" in pairs.columns else ("word_a", "word_b")
        return {p for p, a, b in zip(pairs.pair_id, pairs[ca], pairs[cb]) if str(a).strip().lower() in C3.SS and str(b).strip().lower() in C3.SS}
    if sel == "names":
        return set(pairs.pair_id[pairs.names.astype(bool)])
    return set(pairs.pair_id)


# ------------------------------------------------------------------------------------------------------------------
# selection cases + evaluation pair files
# ------------------------------------------------------------------------------------------------------------------
def build_task(task, proto):
    """-> (selection unit ids in the source id space, info) and writes the evaluation pair files."""
    sp = spec(task, proto)
    os.makedirs(C3.DATA, exist_ok=True)
    if sp["kind"] == "pairs":
        pairs = pd.read_parquet(ab(sp["pairs"]))
        cs = json.load(open(ab(sp["cs"])))
        inv = {int(v): k for k, v in cs["pair_idx_of"].items()}
        ok = item_ok(pairs, sp["sel"])
        src = cs["pairs"]
        val_p = [i for i in src["validation"] if inv.get(int(i)) in ok]
        dis_p = [i for i in src["discovery"] if inv.get(int(i)) in ok]
        sel_ids = [2 * int(i) + d for i in val_p for d in (0, 1)]
        rng = random.Random(31)
        dis_p = list(dis_p)
        rng.shuffle(dis_p)
        dis_p = sorted(dis_p[:CAP])
        keep = [inv[int(i)] for i in dis_p]
        sub = pairs.set_index("pair_id").loc[keep].reset_index()
    elif sp["kind"] == "cfc":
        cs = json.load(open(os.path.join(C3.DATA, f"case_sets_cfc_{proto}.json")))
        pairs = pd.read_parquet(os.path.join(C3.DATA, f"pairs_cfc_{proto}.parquet"))
        idx = cs["pair_idx_of"]
        mine = pairs[pairs.c3_task == task]
        sel_ids = [2 * int(idx[p]) + d for p in mine.pair_id[mine.c3_split == "discovery"] for d in (0, 1)]
        sub = mine[mine.c3_split == "validation"].reset_index(drop=True).iloc[:CAP]
    else:
        from moetrace import ext6_str as S6
        from moetrace.protocol import cases_by_id
        from moetrace.data import load_records
        recs = {r["case_id"]: r for r in load_records()}
        cs = json.load(open(os.path.join(R, sp["run"], "case_sets.json")))["paper"]
        rel = lambda c: recs[int(c)]["requested_rewrite"]["relation_id"]
        sel_ids = [int(c) for c in cs["validation"] if rel(c) in sp["rels"]]
        dis = [int(c) for c in cs["discovery"] if rel(c) in sp["rels"]]
        model = C3.PROTO_MODEL[proto]
        st = bool(json.load(open(os.path.join(R, sp["run"], "run_meta.json")))["sweep"]["special_tokens"])
        cases, _ = cases_by_id(model, dis, special_tokens=st)
        dn = S6.load_donors(sp["run"])
        rows = []
        for c in dis:
            d = dn[dn.case_id == c].sort_values("slot")
            if c not in cases or not len(d):
                continue
            a, b = list(cases[c].ids), list(d.ids_list.iloc[0])
            diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            rows.append(dict(pair_id=f"cf|{c}|{int(d.donor_record.iloc[0])}", ids_a=json.dumps(a), ids_b=json.dumps(b), str_pos=json.dumps(diff),
                             trig_a=int(cases[c].true_id), trig_b=int(cases[c].foil_id), relation=rel(c), case_id=int(c), names=False))
        sub = pd.DataFrame(rows)
    n = len(sub)
    js = {"note": f"p6-circuit C3 (existing rows) task {task} for {proto}: EVALUATION pairs (source discovery split, <= {CAP}); "
                  "selection used the source validation split. Directed ids under 'validation' are the evaluation cases.",
          "pair_idx_of": {p: i for i, p in enumerate(sub.pair_id)},
          "pairs": {"discovery": [], "validation": list(range(n))},
          "directed": {"discovery": [], "validation": [2 * i + d for i in range(n) for d in (0, 1)]}}
    sub = sub.assign(concept=CONCEPT[task], c3_task=task)
    sub.to_parquet(os.path.join(C3.DATA, f"pairs_{task}_{proto}.parquet"), index=False)
    json.dump(js, open(os.path.join(C3.DATA, f"case_sets_{task}_{proto}.json"), "w"), indent=1)
    return sel_ids, {"task": task, "concept": CONCEPT[task], "n_sel_units": len(sel_ids), "n_eval_pairs": n}


# ------------------------------------------------------------------------------------------------------------------
# effects on the selection cases
# ------------------------------------------------------------------------------------------------------------------
def read_heads(src):
    p = os.path.join(R, src)
    files = [p] if p.endswith(".parquet") else sorted(glob.glob(os.path.join(p, "head_rows*.parquet")))
    cols = ["case_id", "layer", "head", "kind", "rescue", "delta_clean", "delta_corrupt"]
    out = []
    for f in files:
        h = pd.read_parquet(f)
        h = h[h.kind == "attn_head"]
        if "slot" not in h.columns:
            h = h.assign(slot=0)
        out.append(h[cols + ["slot"]])
    return pd.concat(out, ignore_index=True) if out else None


def effects(task, proto, sel_ids):
    sp = spec(task, proto)
    frames = []
    for src in sp["heads"]:
        if src is None:
            continue
        h = read_heads(src)
        if h is None:
            continue
        h = h[h.case_id.isin(sel_ids)]
        frames.append(pd.DataFrame(dict(unit=h.case_id, slot=h.slot, kind="head", layer=h.layer, idx=h["head"], rescue=h.rescue,
                                        drp=h.delta_clean - h.delta_corrupt)))
    measured_head_layers = sorted(set(frames[0].layer)) if frames else []
    has_experts = False
    if sp.get("expert_dla") and os.path.exists(os.path.join(R, sp["expert_dla"])):
        e = pd.read_parquet(os.path.join(R, sp["expert_dla"]))
        e = e[e.case_id.isin(sel_ids)]
        if len(e):
            has_experts = True
            frames.append(pd.DataFrame(dict(unit=e.case_id, slot=0, kind="expert", layer=e.layer, idx=e.expert, rescue=e.dla,
                                            drp=e.delta_clean - e.delta_corrupt)))
    if sp["experts"] and os.path.exists(os.path.join(R, sp["experts"], "str_expert_rows.parquet")):
        e = pd.read_parquet(os.path.join(R, sp["experts"], "str_expert_rows.parquet"),
                            columns=["case_id", "slot", "layer", "kind", "expert", "rescue", "delta_clean", "delta_corrupt"])
        e = e[e.kind.isin(["expert", "expert_corrupt_only"]) & e.case_id.isin(sel_ids)]
        if len(e):
            has_experts = True
            frames.append(pd.DataFrame(dict(unit=e.case_id, slot=e.slot, kind="expert", layer=e.layer, idx=e.expert, rescue=e.rescue,
                                            drp=e.delta_clean - e.delta_corrupt)))
    if not frames:
        return None, measured_head_layers, has_experts
    df = pd.concat(frames, ignore_index=True)
    df = df[df.drp.abs() > 1e-6]
    df["effect"] = df.rescue / df.drp
    df["comp"] = [C3.comp_name(k, l, i) for k, l, i in zip(df.kind, df.layer, df.idx)]
    # CounterFact: mean over the donors (slots) of a case; components missing in a row (inactive expert) count 0
    g = df.groupby(["unit", "comp"]).effect.sum() / df.groupby("unit").slot.nunique().reindex(df.groupby(["unit", "comp"]).effect.sum().index.get_level_values(0)).to_numpy()
    out = g.rename("effect").reset_index()
    out["task"] = task
    return out, measured_head_layers, has_experts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", required=True)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    proto = a.proto
    if a.report:
        return report(proto)
    infos, eff, measured = [], {}, {}
    for t in ORDER:
        try:
            sel, info = build_task(t, proto)
        except Exception as ex:
            print("skip", t, repr(ex)[:200])
            continue
        e, hl, he = effects(t, proto, set(sel))
        info.update(head_layers=hl, has_experts=he, n_effect_units=int(e.unit.nunique()) if e is not None else 0)
        infos.append(info)
        print(info)
        if e is None or info["n_effect_units"] < 8 or info["n_eval_pairs"] < 8:
            print("  -> excluded (too few units)")
            continue
        eff[t] = e
        measured[t] = (set(hl), he)
    tlist = [t for t in ORDER if t in eff]
    concept_of = {t: CONCEPT[t] for t in tlist}
    top = {}
    for t in tlist:
        e = eff[t]
        units = e.unit.unique()
        M = e.pivot_table(index="unit", columns="comp", values="effect", aggfunc="mean").reindex(units).fillna(0.0)
        mean = M.mean(0)
        is_h = pd.Series([("H" in c) for c in mean.index], index=mean.index)
        th = mean[is_h].sort_values(ascending=False).index[:K].tolist()
        te = mean[~is_h].sort_values(ascending=False).index[:K].tolist()
        top[t] = {"heads": th, "experts": te, "head_vals": mean.loc[th].round(4).tolist(), "expert_vals": mean.loc[te].round(4).tolist(),
                  "n_units": int(len(units))}
    circuits = {}
    for conc in C3.CONCEPTS:
        own = [t for t in tlist if concept_of[t] == conc]
        other = [t for t in tlist if concept_of[t] not in (conc, "CONTROL")]
        ctl = [t for t in tlist if concept_of[t] == "CONTROL"]
        if len(own) < 2:
            circuits[conc] = {"tasks": own, "insufficient": True, "heads": [], "experts": []}
            continue
        cnt = pd.Series([c for t in own for c in top[t]["heads"] + top[t]["experts"]]).value_counts()
        cand = [c for c, n in cnt.items() if n >= 2]
        oth = set(c for t in other for c in top[t]["heads"] + top[t]["experts"])
        cset = set(c for t in ctl for c in top[t]["heads"] + top[t]["experts"])
        keep = [c for c in cand if c not in oth]
        circuits[conc] = {"tasks": own, "insufficient": False, "heads": sorted(c for c in keep if "H" in c),
                          "experts": sorted(c for c in keep if "E" in c), "shared_with_other_concepts": sorted(set(cand) & oth),
                          "in_control_top": sorted(set(keep) & cset), "n_candidates": len(cand)}
    sys.argv = [sys.argv[0]]
    from ext14ci_c3_analyze import random_sets
    layers = {}
    for conc, cc in circuits.items():
        cc["random"] = random_sets(cc, layers, C3.PROTO_MODEL[proto], seed=sum(map(ord, conc))) if not cc.get("insufficient") else []
    allE = pd.concat(eff.values(), ignore_index=True)
    allE.to_parquet(os.path.join(C3.DATA, f"effects_{proto}.parquet"))
    js = {"proto": proto, "tasks": tlist, "concept_of": concept_of, "top": top, "circuits": circuits, "task_info": infos,
          "measured": {t: {"head_layers": sorted(v[0]), "experts": v[1]} for t, v in measured.items()},
          "note": "existing-rows C3 (scope cut 2026-10-17); selection = source validation split, evaluation = source discovery split"}
    json.dump(js, open(os.path.join(C3.DATA, f"circuits_{proto}.json"), "w"), indent=1, default=C._json_default)
    for conc, cc in circuits.items():
        print(conc, cc.get("tasks"), "heads", cc["heads"], "experts", cc["experts"], "shared", cc.get("shared_with_other_concepts"))


def auroc_rows(proto, js):
    """AUROC of each top component for each concept on the selection cases of the tasks where it was measured."""
    E = pd.read_parquet(os.path.join(C3.DATA, f"effects_{proto}.parquet"))
    meas = js["measured"]
    concept_of = js["concept_of"]
    comps = sorted(set(c for t in js["top"] for c in js["top"][t]["heads"] + js["top"][t]["experts"]))
    units = E[["task", "unit"]].drop_duplicates()
    piv = E[E.comp.isin(comps)].pivot_table(index=["task", "unit"], columns="comp", values="effect", aggfunc="mean")
    piv = piv.reindex(pd.MultiIndex.from_frame(units)).fillna(0.0)
    tasks_of = units.task.to_numpy()
    conc = np.array([concept_of[t] for t in tasks_of])
    rows = []
    for c in comps:
        is_h = "H" in c
        lay = int(c[1:].split("H" if is_h else "E")[0])
        ok = np.array([(lay in meas[t]["head_layers"]) if is_h else meas[t]["experts"] for t in tasks_of])
        x = piv[c].to_numpy() if c in piv.columns else np.zeros(len(piv))
        rec = {"comp": c, "kind": "head" if is_h else "expert",
               "home": pd.Series([concept_of[t] for t in js["top"] if c in js["top"][t]["heads"] + js["top"][t]["experts"]]).value_counts().index[0]}
        for k in C3.CONCEPTS:
            pos, neg = x[ok & (conc == k)], x[ok & (conc != k)]
            rec[f"auroc_{k}"] = C3.auroc(pos, neg) if len(pos) >= 4 and len(neg) >= 4 else np.nan
        rec["n_tasks_measured"] = int(len(set(tasks_of[ok])))
        rows.append(rec)
    A = pd.DataFrame(rows)
    A["auroc_home"] = [r.get(f"auroc_{r['home']}", np.nan) if r["home"] in C3.CONCEPTS else np.nan for r in A.to_dict("records")]
    return A


def report(proto):
    from ext14ci_c3_analyze import validation_tables
    js = json.load(open(os.path.join(C3.DATA, f"circuits_{proto}.json")))
    concept_of, circuits, top = js["concept_of"], js["circuits"], js["top"]
    A = auroc_rows(proto, js)
    hh, ee = A[A.kind == "head"].auroc_home.dropna(), A[A.kind == "expert"].auroc_home.dropna()
    L = [f"## {proto}: concept circuits from existing rows (selection: source validation split; evaluation: source discovery split)", "",
         "| task | concept | selection units | evaluation pairs | head layers measured | expert rows |", "|---|---|---|---|---|---|"]
    for inf in js["task_info"]:
        L.append(f"| {inf['task']} | {inf['concept']} | {inf.get('n_effect_units', 0)} | {inf['n_eval_pairs']} | "
                 f"{'all' if len(inf.get('head_layers', [])) > 20 else ','.join(map(str, inf.get('head_layers', [])))} | {'yes' if inf.get('has_experts') else 'no'} |")
    L += ["", "| concept | tasks | circuit heads | circuit experts | in top sets of other concepts (excluded) | also in the control's top set |",
          "|---|---|---|---|---|---|"]
    for conc, cc in circuits.items():
        L.append(f"| {conc} | {', '.join(cc['tasks'])} | {', '.join(cc['heads']) or '–'} | {', '.join(cc['experts']) or '–'} | "
                 f"{', '.join(cc.get('shared_with_other_concepts', [])) or '–'} | {', '.join(cc.get('in_control_top', [])) or '–'} |")
    L += ["", "| task | top heads (mean effect = rescue / drop) | top experts |", "|---|---|---|"]
    for t, tt in top.items():
        L.append(f"| {t} | " + ", ".join(f"{c} {v:+.3f}" for c, v in zip(tt["heads"][:5], tt["head_vals"][:5])) + " | "
                 + (", ".join(f"{c} {v:+.3f}" for c, v in zip(tt["experts"][:5], tt["expert_vals"][:5])) or "– (no expert rows)") + " |")
    from scipy.stats import mannwhitneyu
    mw = mannwhitneyu(ee, hh, alternative="greater") if len(hh) > 2 and len(ee) > 2 else None
    L += ["", f"AUROC of each top component for its home concept (selection cases of the tasks where it was measured): heads median "
              f"{hh.median():.3f} (n {len(hh)}), experts median {ee.median():.3f} (n {len(ee)})"
          + (f"; Mann-Whitney experts > heads p = {mw.pvalue:.3g}" if mw else "")]
    summ = {"proto": proto, "tasks": js["tasks"], "task_info": js["task_info"], "circuits": {k: {kk: v for kk, v in c.items() if kk != "random"}
                                                                                          for k, c in circuits.items()},
            "top": top, "auroc_home": {"heads_median": float(hh.median()) if len(hh) else None, "experts_median": float(ee.median()) if len(ee) else None,
                                       "n_heads": int(len(hh)), "n_experts": int(len(ee)), "mw_p_experts_gt_heads": float(mw.pvalue) if mw else None},
            "auroc_rows": A.round(4).to_dict("records")}
    vt = validation_tables(proto, circuits, concept_of)
    if vt is not None:
        Kt = vt["ko"]
        real = Kt[Kt.draw == 0]
        rnd = Kt[Kt.draw > 0].groupby(["concept", "part", "task"]).F.mean().rename("F_rand")
        M = real.merge(rnd.reset_index(), on=["concept", "part", "task"], how="left")
        M["F_net"] = M.F - M.F_rand
        spec_ = []
        for (conc, part), g in M.groupby(["concept", "part"]):
            own = g[g.task_concept == conc].F_net.mean()
            oth = g[(g.task_concept != conc)].F_net.mean()
            oth_abs = g[(g.task_concept != conc)].F_net.abs().mean()
            ctl = g[g.task_concept == "CONTROL"].F_net.mean() if (g.task_concept == "CONTROL").any() else np.nan
            spec_.append(dict(concept=conc, part=part, F_own=own, F_other=oth, F_other_abs=oth_abs, F_control=ctl, specificity=own - oth,
                              selectivity=own / (abs(own) + oth_abs) if (abs(own) + oth_abs) > 0 else np.nan))
        S = pd.DataFrame(spec_)
        summ["knockout_specificity"] = S.round(4).to_dict("records")
        summ["knockout_matrix"] = M.round(4).to_dict("records")
        tasks_order = [t for t in ORDER if t in set(M.task)]
        L += ["", "### Knockout matrix: fraction of the clean margin lost F (circuit) minus the mean of 2 matched random sets; evaluation cases", "",
              "| circuit (part) | " + " | ".join(f"{t} ({concept_of[t][:4]})" for t in tasks_order) + " | own − other |",
              "|---|" + "---|" * (len(tasks_order) + 1)]
        for (conc, part), g in M.groupby(["concept", "part"]):
            gi = g.set_index("task")
            sp_ = S[(S.concept == conc) & (S.part == part)].specificity
            L.append(f"| {conc} ({part}) | " + " | ".join(f"{gi.F_net.get(t, np.nan):+.3f}" for t in tasks_order)
                     + f" | {float(sp_.iloc[0]) if len(sp_) else float('nan'):+.3f} |")
        L += ["", "Raw F of the circuits (draw 0) and of the random sets (mean of draws 1-2):", "",
              "| circuit (part) | " + " | ".join(tasks_order) + " |", "|---|" + "---|" * len(tasks_order)]
        for (conc, part), g in M.groupby(["concept", "part"]):
            gi = g.set_index("task")
            L.append(f"| {conc} ({part}) | " + " | ".join(f"{gi.F.get(t, np.nan):.3f} ({gi.F_rand.get(t, np.nan):.3f})" for t in tasks_order) + " |")
        L += ["", "Specificity per circuit part (N3): F_own = mean net F on the concept's own tasks, F_other = mean |net F| on every other "
              "task (other concepts and the control), selectivity = F_own / (|F_own| + F_other):", "",
              "| circuit (part) | F_own | F_other (mean abs) | F on the control task | selectivity |", "|---|---|---|---|---|"]
        for r in S.itertuples():
            L.append(f"| {r.concept} ({r.part}) | {r.F_own:+.3f} | {r.F_other_abs:.3f} | {r.F_control:+.3f} | {r.selectivity:.2f} |")
        if "addback" in vt:
            Ad = vt["addback"]
            Ar = Ad[Ad.draw == 0].merge(Ad[Ad.draw > 0].groupby(["concept", "part", "task"]).R.mean().rename("R_rand").reset_index(),
                                        on=["concept", "part", "task"], how="left")
            summ["addback"] = Ar.round(4).to_dict("records")
            L += ["", "### Add-back of the concept set (fraction of the drop restored; matched random sets in parentheses)", "",
                  "| circuit (part) | " + " | ".join(tasks_order) + " |", "|---|" + "---|" * len(tasks_order)]
            for (conc, part), g in Ar.groupby(["concept", "part"]):
                gi = g.set_index("task")
                L.append(f"| {conc} ({part}) | " + " | ".join(f"{gi.R.get(t, np.nan):.3f} ({gi.R_rand.get(t, np.nan):.3f})" for t in tasks_order) + " |")
        if "edges" in vt:
            Ed = vt["edges"].sort_values("edge_over_drop", ascending=False)
            summ["edges"] = Ed.round(4).to_dict("records")
            L += ["", "### Edges A -> B among circuit components (mediated effect rescue(A) + rescue(B) - rescue(A u B) / drop; top 12)", "",
                  "| concept | A | B | edge / drop | share of A's rescue through B |", "|---|---|---|---|---|"]
            for r in Ed.head(12).itertuples():
                L.append(f"| {r.concept} | {r.a} | {r.b} | {r.edge_over_drop:+.3f} | "
                         + ("–" if r.share_of_a is None or not np.isfinite(r.share_of_a) else f"{r.share_of_a:.2f}") + " |")
    os.makedirs(os.path.join(R, "tables"), exist_ok=True)
    open(os.path.join(R, "tables", f"ext14ci_c3_{proto}.md"), "w").write("\n".join(L) + "\n")
    s = json.load(open(C.SUMMARY)) if os.path.exists(C.SUMMARY) else {}
    c3 = s.get("c3", {})
    c3[proto] = summ
    C.update_summary("c3", c3)
    print("report written", proto)


if __name__ == "__main__":
    main()
