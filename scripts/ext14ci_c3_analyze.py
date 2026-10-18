"""ext14 C3 analysis (CPU; p6-circuit): concept circuits of heads + experts.

--layers-only  choose the common head / expert layers of a proto from the tasks' sweeps (discovery mean single-layer rescue
               of attn_layer / layer; per task the top-3 layers of each kind; union, at most 10 each, most frequent first) ->
               results/p6ci_c3_data/layers_<proto>.json (frozen once written, so tasks added later use the same layers).
default        per-task component effects (single head / expert patch rescue / drop, validation and discovery), task top sets,
               concept circuits, AUROC specificity, matched random sets -> results/p6ci_c3_data/circuits_<proto>.json;
               and, when the validation runs exist (results/p6ci_c3_data/validate_<proto>.parquet), the concept x task
               knockout matrix, add-back and edges -> results/tables/ext14ci_c3_*.md, results/figures/ext14ci_c3_*.png,
               key 'c3' of results/ext14ci_summary.json.

Usage: python scripts/ext14ci_c3_analyze.py --proto qwen3|mixtral_bos [--layers-only]
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext14_circuit_c3 as C3
from moetrace import ext14_circuit as C
from moetrace.models import RESULTS, MODELS

TAB = os.path.join(RESULTS, "tables")
FIG = os.path.join(RESULTS, "figures")
N_BOOT = 2000
N_RAND = 2


def task_list(proto):
    out = []
    for t in C3.TASK_ORDER:
        r = os.path.join(RESULTS, C3.run_name(t, proto))
        if os.path.exists(os.path.join(r, ".done_sweep")):
            out.append(t)
    return out


def splits(t, proto):
    cs = json.load(open(C3.pair_paths(t, proto)[1]))
    return {s: [int(c) for c in cs["directed"][s]] for s in ("discovery", "validation")}


def choose_layers(proto):
    path = os.path.join(C3.DATA, f"layers_{proto}.json")
    if os.path.exists(path):
        return json.load(open(path))
    per = {}
    he, ex = [], []
    for t in task_list(proto):
        sw = pd.read_parquet(os.path.join(RESULTS, C3.run_name(t, proto), "str_sweep_rows.parquet"),
                             columns=["case_id", "kind", "layer", "rescue", "delta_clean", "delta_corrupt"])
        d = splits(t, proto)["discovery"]
        s = sw[sw.case_id.isin(d)]
        drop = (s[s.kind == "layer"].groupby("case_id").first().eval("delta_clean - delta_corrupt")).mean()
        a = s[s.kind == "attn_layer"].groupby("layer").rescue.mean() / drop
        m = s[s.kind == "layer"].groupby("layer").rescue.mean() / drop
        ta = [int(l) for l in a.sort_values(ascending=False).index[:3]]
        tm = [int(l) for l in m.sort_values(ascending=False).index[:3]]
        per[t] = {"attn_top": ta, "moe_top": tm, "attn_top_vals": a.loc[ta].round(3).tolist(), "moe_top_vals": m.loc[tm].round(3).tolist()}
        he += ta
        ex += tm
    # union ranked by frequency, at most 12
    def top_union(xs, n=10):
        vc = pd.Series(xs).value_counts()
        return sorted(int(l) for l in vc.index[:n])
    res = {"heads": top_union(he), "experts": top_union(ex), "per_task": per, "tasks": list(per)}
    json.dump(res, open(path, "w"), indent=1)
    return res


def load_effects(t, proto, layers):
    """long frame: case_id, split, kind (head|expert), layer, idx, effect (rescue / drop); drop per case."""
    r = os.path.join(RESULTS, C3.run_name(t, proto))
    sp = splits(t, proto)
    split_of = {c: s for s in sp for c in sp[s]}
    out = []
    hp = os.path.join(r, "head_rows.parquet")
    if os.path.exists(hp):
        h = pd.read_parquet(hp, columns=["case_id", "layer", "head", "kind", "rescue", "delta_clean", "delta_corrupt"])
        h = h[(h.kind == "attn_head") & h.layer.isin(layers["heads"])]
        out.append(pd.DataFrame(dict(case_id=h.case_id, kind="head", layer=h.layer, idx=h["head"], rescue=h.rescue,
                                     drop=h.delta_clean - h.delta_corrupt)))
    ep = os.path.join(r, "str_expert_rows.parquet")
    if os.path.exists(ep):
        e = pd.read_parquet(ep, columns=["case_id", "layer", "kind", "expert", "rescue", "delta_clean", "delta_corrupt"])
        e = e[e.kind.isin(["expert", "expert_corrupt_only"]) & e.layer.isin(layers["experts"])]
        out.append(pd.DataFrame(dict(case_id=e.case_id, kind="expert", layer=e.layer, idx=e.expert, rescue=e.rescue,
                                     drop=e.delta_clean - e.delta_corrupt)))
    if not out:
        return None
    df = pd.concat(out, ignore_index=True)
    df["split"] = df.case_id.map(split_of)
    df["effect"] = df.rescue / df.drop
    df["task"] = t
    df["comp"] = [C3.comp_name(k, l, i) for k, l, i in zip(df.kind, df.layer, df.idx)]
    return df


def effect_matrix(df, cases):
    """cases x components matrix of effects (missing = 0: expert not routed in either run)."""
    M = df.pivot_table(index="case_id", columns="comp", values="effect", aggfunc="mean")
    return M.reindex(cases).fillna(0.0)


def build_circuits(proto, layers, eff):
    tlist = list(eff)
    tasks = C3.tasks(proto)
    concept_of = {t: tasks[t]["concept"] for t in tlist}
    top = {}
    for t in tlist:
        d = eff[t][eff[t].split == "discovery"]
        cases = sorted(d.case_id.unique())
        M = effect_matrix(d, cases)
        mean = M.mean(0)
        kinds = pd.Series({c: ("head" if "H" in c else "expert") for c in M.columns})
        th = mean[kinds == "head"].sort_values(ascending=False).index[:C3.K_TOP].tolist()
        te = mean[kinds == "expert"].sort_values(ascending=False).index[:C3.K_TOP].tolist()
        top[t] = {"heads": th, "experts": te, "head_vals": mean.loc[th].round(4).tolist(), "expert_vals": mean.loc[te].round(4).tolist()}
    circuits = {}
    for conc in C3.CONCEPTS:
        own = [t for t in tlist if concept_of[t] == conc]
        other = [t for t in tlist if concept_of[t] not in (conc, "CONTROL")]
        if len(own) < 2:
            circuits[conc] = {"tasks": own, "insufficient": True, "heads": [], "experts": []}
            continue
        cnt = pd.Series([c for t in own for c in top[t]["heads"] + top[t]["experts"]]).value_counts()
        cand = [c for c, n in cnt.items() if n >= 2]
        oth = set(c for t in other for c in top[t]["heads"] + top[t]["experts"])
        ctl = set(c for t in tlist if concept_of[t] == "CONTROL" for c in top[t]["heads"] + top[t]["experts"])
        keep = [c for c in cand if c not in oth]
        circuits[conc] = {"tasks": own, "insufficient": False, "heads": sorted(c for c in keep if "H" in c),
                          "experts": sorted(c for c in keep if "E" in c), "shared_with_other_concepts": sorted(set(cand) & oth),
                          "in_control_top": sorted(set(keep) & ctl), "n_candidates": len(cand)}
    return top, circuits, concept_of


def auroc_table(eff, concept_of, comps):
    """AUROC per component and concept on validation cases of all tasks (negatives = cases of every other task, controls incl.)."""
    rows = []
    allv = pd.concat([e[e.split == "validation"] for e in eff.values()], ignore_index=True)
    cases = allv[["task", "case_id"]].drop_duplicates()
    cases["concept"] = cases.task.map(concept_of)
    key = list(zip(cases.task, cases.case_id))
    piv = allv[allv.comp.isin(comps)].pivot_table(index=["task", "case_id"], columns="comp", values="effect", aggfunc="mean")
    piv = piv.reindex(pd.MultiIndex.from_tuples(key, names=["task", "case_id"])).fillna(0.0)
    conc = cases.concept.to_numpy()
    for c in comps:
        x = piv[c].to_numpy() if c in piv.columns else np.zeros(len(piv))
        rec = {"comp": c, "kind": "head" if "H" in c else "expert"}
        for k in C3.CONCEPTS:
            if (conc == k).sum() and (conc != k).sum():
                rec[f"auroc_{k}"] = C3.auroc(x[conc == k], x[conc != k])
                rec[f"mean_{k}"] = float(x[conc == k].mean())
        rec["mean_other"] = float(x.mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def random_sets(circ, layers, model, seed=0):
    """matched random sets: same number of heads per layer (heads not in the circuit), same number of experts per layer."""
    from moetrace.arch import load_spec
    spec, _ = load_spec(MODELS[model]["repo"])
    rng = np.random.default_rng(seed)
    out = []
    for r in range(N_RAND):
        hs, es = [], []
        for c in circ["heads"]:
            l = int(c[1:].split("H")[0])
            used = set(int(x[1:].split("H")[1]) for x in circ["heads"] if int(x[1:].split("H")[0]) == l) | set(h for (ll, h) in hs if ll == l)
            pool = [h for h in range(spec.n_heads) if h not in used]
            hs.append((l, int(rng.choice(pool))))
        for c in circ["experts"]:
            l, e = C.parse_ename(c)
            used = set(C.parse_ename(x)[1] for x in circ["experts"] if C.parse_ename(x)[0] == l) | set(x for (ll, x) in es if ll == l)
            pool = [x for x in range(spec.n_experts) if x not in used]
            es.append((l, int(rng.choice(pool))))
        out.append({"heads": [C3.comp_name("head", l, h) for l, h in hs], "experts": [C3.comp_name("expert", l, e) for l, e in es]})
    return out


def validation_tables(proto, circuits, concept_of):
    import glob as _glob
    fs = sorted(_glob.glob(os.path.join(C3.DATA, f"validate_{proto}__*.parquet")))
    p = os.path.join(C3.DATA, f"validate_{proto}.parquet")
    if not fs and not os.path.exists(p):
        return None
    v = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.read_parquet(p)
    out = {}
    # knockout: F = sum(delta_clean - delta_ko) / sum(delta_clean) over validation cases of a task; pair bootstrap
    ko = v[v.kind == "ko"]
    base = v[v.kind == "base"].set_index(["task", "case_id"]).delta
    ko = ko.assign(d0=[base.get((t, c), np.nan) for t, c in zip(ko.task, ko.case_id)])
    rows = []
    for (conc, part, draw, task), g in ko.groupby(["concept", "part", "draw", "task"]):
        F = float((g.d0 - g.delta).sum() / g.d0.sum())
        rows.append(dict(concept=conc, part=part, draw=draw, task=task, task_concept=concept_of.get(task), F=F, n=len(g)))
    K = pd.DataFrame(rows)
    out["ko"] = K
    ab = v[v.kind == "addback"]
    if len(ab):
        ab = ab.assign(frac=(ab.delta - ab.delta_corrupt) / (ab.delta_clean - ab.delta_corrupt))
        A = ab.groupby(["concept", "part", "draw", "task"]).apply(
            lambda g: (g.delta - g.delta_corrupt).sum() / (g.delta_clean - g.delta_corrupt).sum(), include_groups=False).rename("R").reset_index()
        A["task_concept"] = A.task.map(concept_of)
        out["addback"] = A
    ed = v[v.kind.isin(["edge_single", "edge_pair"])]
    if len(ed):
        sg = ed[ed.kind == "edge_single"].groupby(["concept", "task", "case_id", "a"]).rescue.mean()
        pr = ed[ed.kind == "edge_pair"]
        recs = []
        for (conc, a_, b_), g in pr.groupby(["concept", "a", "b"]):
            ra = np.array([sg.get((conc, t, c, a_), np.nan) for t, c in zip(g.task, g.case_id)])
            rb = np.array([sg.get((conc, t, c, b_), np.nan) for t, c in zip(g.task, g.case_id)])
            m = ra + rb - g.rescue.to_numpy()
            dr = (g.delta_clean - g.delta_corrupt).to_numpy()
            recs.append(dict(concept=conc, a=a_, b=b_, edge=float(np.nanmean(m)), edge_over_drop=float(np.nansum(m) / dr.sum()),
                             rescue_a=float(np.nanmean(ra)), share_of_a=float(np.nanmean(m) / np.nanmean(ra)) if np.nanmean(ra) else None,
                             n=len(g)))
        out["edges"] = pd.DataFrame(recs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", required=True)
    ap.add_argument("--layers-only", action="store_true")
    a = ap.parse_args()
    proto = a.proto
    model = C3.PROTO_MODEL[proto]
    layers = choose_layers(proto)
    print("layers", layers["heads"], layers["experts"])
    if a.layers_only:
        return
    eff = {}
    for t in task_list(proto):
        r = os.path.join(RESULTS, C3.run_name(t, proto))
        if not (os.path.exists(os.path.join(r, ".done_expert")) and os.path.exists(os.path.join(r, ".done_heads"))):
            continue
        e = load_effects(t, proto, layers)
        if e is not None:
            eff[t] = e
    if not eff:
        print("no task with expert + head singles yet")
        return
    top, circuits, concept_of = build_circuits(proto, layers, eff)
    comps = sorted(set(c for t in top for c in top[t]["heads"] + top[t]["experts"]))
    au = auroc_table(eff, concept_of, comps)
    # home concept of a component = the concept of the tasks whose top set contains it (most often)
    home = {}
    for c in comps:
        cs = [concept_of[t] for t in top if c in top[t]["heads"] + top[t]["experts"]]
        home[c] = pd.Series(cs).value_counts().index[0]
    au["home"] = au.comp.map(home)
    au["auroc_home"] = [r.get(f"auroc_{r['home']}", np.nan) if r["home"] in C3.CONCEPTS else np.nan for r in au.to_dict("records")]
    for conc, cc in circuits.items():
        cc["random"] = random_sets(cc, layers, model, seed=hash(conc) % 1000) if not cc.get("insufficient") else []
    js = {"proto": proto, "layers": layers, "tasks": list(eff), "concept_of": concept_of, "top": top, "circuits": circuits}
    json.dump(js, open(os.path.join(C3.DATA, f"circuits_{proto}.json"), "w"), indent=1, default=C._json_default)
    os.makedirs(TAB, exist_ok=True)
    L = [f"## {proto}: concept circuits (top-{C3.K_TOP} heads + top-{C3.K_TOP} experts per task, discovery mean effect)", "",
         "| concept | tasks | heads | experts | shared with other concepts | in a control task's top set |", "|---|---|---|---|---|---|"]
    for conc, cc in circuits.items():
        L.append(f"| {conc} | {', '.join(cc['tasks'])} | {', '.join(cc['heads']) or '–'} | {', '.join(cc['experts']) or '–'} | "
                 f"{', '.join(cc.get('shared_with_other_concepts', [])) or '–'} | {', '.join(cc.get('in_control_top', [])) or '–'} |")
    L += ["", "| task | concept | top heads (effect) | top experts (effect) |", "|---|---|---|---|"]
    for t, tt in top.items():
        L.append(f"| {t} | {concept_of[t]} | " + ", ".join(f"{c} {v:+.3f}" for c, v in zip(tt["heads"][:5], tt["head_vals"][:5])) + " | "
                 + ", ".join(f"{c} {v:+.3f}" for c, v in zip(tt["experts"][:5], tt["expert_vals"][:5])) + " |")
    hh = au[au.kind == "head"].auroc_home.dropna()
    ee = au[au.kind == "expert"].auroc_home.dropna()
    from scipy.stats import mannwhitneyu
    mw = mannwhitneyu(ee, hh, alternative="greater") if len(hh) > 2 and len(ee) > 2 else None
    L += ["", f"AUROC of each top component for its home concept (validation cases of all tasks): heads median {hh.median():.3f} "
              f"(n {len(hh)}), experts median {ee.median():.3f} (n {len(ee)})"
          + (f"; Mann-Whitney experts > heads p = {mw.pvalue:.3g}" if mw else "")]
    summ = {"proto": proto, "layers": layers, "tasks": list(eff), "circuits": circuits, "top": top,
            "auroc_home": {"heads_median": float(hh.median()) if len(hh) else None, "experts_median": float(ee.median()) if len(ee) else None,
                           "n_heads": int(len(hh)), "n_experts": int(len(ee)), "mw_p_experts_gt_heads": float(mw.pvalue) if mw else None},
            "auroc_rows": au.round(4).to_dict("records")}
    vt = validation_tables(proto, circuits, concept_of)
    if vt is not None:
        K = vt["ko"]
        real = K[K.draw == 0]
        rnd = K[K.draw > 0].groupby(["concept", "part", "task"]).F.mean().rename("F_rand")
        M = real.merge(rnd.reset_index(), on=["concept", "part", "task"], how="left")
        M["F_net"] = M.F - M.F_rand
        spec = []
        for (conc, part), g in M.groupby(["concept", "part"]):
            own = g[g.task_concept == conc].F_net.mean()
            oth = g[(g.task_concept != conc)].F_net.mean()
            spec.append(dict(concept=conc, part=part, F_own=own, F_other=oth, specificity=own - oth))
        S = pd.DataFrame(spec)
        summ["knockout_specificity"] = S.round(4).to_dict("records")
        summ["knockout_matrix"] = M.round(4).to_dict("records")
        tasks_order = [t for t in C3.TASK_ORDER if t in set(M.task)]
        L += ["", f"### Knockout matrix (fraction of the clean margin lost F, minus the mean of {N_RAND} matched random sets; validation cases)", "",
              "| circuit (part) | " + " | ".join(f"{t} ({concept_of[t][:4]})" for t in tasks_order) + " | specificity (own - other) |",
              "|---|" + "---|" * (len(tasks_order) + 1)]
        for (conc, part), g in M.groupby(["concept", "part"]):
            gi = g.set_index("task")
            sp = S[(S.concept == conc) & (S.part == part)].specificity
            L.append(f"| {conc} ({part}) | " + " | ".join(f"{gi.F_net.get(t, np.nan):+.3f}" for t in tasks_order)
                     + f" | {float(sp.iloc[0]) if len(sp) else float('nan'):+.3f} |")
        if "addback" in vt:
            A = vt["addback"]
            Ar = A[A.draw == 0].merge(A[A.draw > 0].groupby(["concept", "part", "task"]).R.mean().rename("R_rand").reset_index(),
                                      on=["concept", "part", "task"], how="left")
            summ["addback"] = Ar.round(4).to_dict("records")
            L += ["", "### Add-back of the concept set (fraction of the drop restored; random matched in parentheses)", "",
                  "| circuit (part) | " + " | ".join(tasks_order) + " |", "|---|" + "---|" * len(tasks_order)]
            for (conc, part), g in Ar.groupby(["concept", "part"]):
                gi = g.set_index("task")
                L.append(f"| {conc} ({part}) | " + " | ".join(f"{gi.R.get(t, np.nan):.3f} ({gi.R_rand.get(t, np.nan):.3f})" for t in tasks_order) + " |")
        if "edges" in vt:
            E = vt["edges"].sort_values("edge_over_drop", ascending=False)
            summ["edges"] = E.round(4).to_dict("records")
            L += ["", "### Strongest edges A -> B (mediated effect rescue(A) + rescue(B) - rescue(A u B), fraction of the drop)", "",
                  "| concept | A | B | edge / drop | share of A's rescue |", "|---|---|---|---|---|"]
            for r in E.head(15).itertuples():
                L.append(f"| {r.concept} | {r.a} | {r.b} | {r.edge_over_drop:+.3f} | {r.share_of_a if r.share_of_a is None else round(r.share_of_a, 2)} |")
    open(os.path.join(TAB, f"ext14ci_c3_{proto}.md"), "w").write("\n".join(L) + "\n")
    s = json.load(open(C.SUMMARY)) if os.path.exists(C.SUMMARY) else {}
    c3 = s.get("c3", {})
    c3[proto] = summ
    C.update_summary("c3", c3)
    print("c3 analysis done", proto, {k: (len(v["heads"]), len(v["experts"])) for k, v in circuits.items()})


if __name__ == "__main__":
    main()
