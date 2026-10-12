"""ext13ab (Phase 5): section results/sections/ext13_addback.md from results/ext13ab_summary.json and the ext13ab tables.

Everything numeric is templated from the summary JSON (re-run after scripts/ext13ab_analyze.py); the READINGS block holds
the hand-written interpretation, checked against the numbers printed in the same section.

Usage: python scripts/ext13ab_text.py [--summary PATH] [--out PATH]
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
from moetrace import ext13ab as AB

R = "/home/ubuntu/MOE/results"
CEIL_MIN = 0.1
MODELS = ("qwen3", "mixtral")
SHORT = {"qwen3": "Qwen3", "mixtral": "Mixtral"}
READINGS = []  # extra hand-written readings (optional); the templated ones come from readings(S)


def _r(S, cond, model, fam="main"):
    for r in S["runs"].values():
        if r["cond"] == cond and r["model"] == model and r["family"] == fam:
            return r
    return None


def _c(S, a, b, model):
    for c in S.get("contrasts", []):
        if c["model"] == model and c["a"].startswith(a + "_") and (c["b"].startswith(b + "_") or c["b"] == f"wino_{model}" and b == "P3wino"):
            return c
    return None


def _excl0(x):
    return x is not None and np.isfinite(num(x[1])) and (num(x[1]) > 0 or num(x[2]) < 0)


def readings(S):
    """Data-driven readings (templated; only for conditions / contrasts present in the summary)."""
    out = []
    # 1. direction on identical twins
    parts = []
    for m in MODELS:
        c = _c(S, "dir_f1", "dir_opt", m)
        if not c:
            continue
        ra, rb = _r(S, "dir_f1", m), _r(S, "dir_opt", m)
        da, db = (ra.get("direct") or {}), (rb.get("direct") or {})
        amin = min(amin, num(da.get("a_dir", [np.nan])[0])) if "amin" in dir() else num(da.get("a_dir", [np.nan])[0])
        parts.append(f"{SHORT[m]} M {f3(rb['boot']['ceiling'][0])} → {f3(ra['boot']['ceiling'][0])} (Δ {ci(c['diff']['ceiling'], sign=True)}), "
                     f"direct-path attention share {f3(db.get('a_dir', [None])[0], 2)} → {f3(da.get('a_dir', [None])[0], 2)}, oracle overshoot "
                     f"{f3(rb['boot']['oracle']['overshoot'][0], sign=True)} → {f3(ra['boot']['oracle']['overshoot'][0], sign=True)}")
    if parts:
        out.append("**Direction decides the expert ceiling.** On identical WinoGrande twins (paired over twins), turning the option swap "
                   "(entity → attribute) into fill-in-the-blank (attribute → entity) collapses what the final-position MoE outputs can "
                   "repair: " + "; ".join(parts) + ". In attribute → entity the final position copies a context entity, which the "
                   f"attention outputs write directly (direct share ≥ {f3(amin, 2)}); the summed MoE outputs add little or even oppose the answer, "
                   "while individual experts still help, so subsets overshoot the all-MoE ceiling far more than in the option swap "
                   "(where overshoot ≈ 0, as in Phase 3).")
    # 2. F1 main and replication
    parts = []
    for m in MODELS:
        r = _r(S, "f1", m)
        if r:
            rr = _r(S, "f1", m, "rep")
            parts.append(f"{SHORT[m]} M {ci(r['boot']['ceiling'])}" + (f" (replication {ci(rr['boot']['ceiling'])})" if rr else "")
                         + f", greedy r(15) {f3(r['boot']['greedy']['r15'][0])}, overshoot {f3(r['boot']['oracle']['overshoot'][0], sign=True)}, "
                         f"{k80s(r)}, oracle maximum {f3(r['ext8']['a1']['oracle']['max_r'])} at k = {r['ext8']['a1']['oracle']['k_max_r']} of {r['K']}")
    if parts:
        out.append("**F1 (neutral main set) behaves like the F1 twins**: " + "; ".join(parts) + ". A few experts reach 80 % of the small "
                   "ceiling, but the oracle curve keeps rising well past k = 15 before the last, answer-opposing experts pull it back to M: "
                   "the useful expert writes are spread over many weak contributors and partly cancelled, unlike the option swap or "
                   "CounterFact.")
    # 3. copy contrasts
    for kind, title in (("copy", "Copy availability (C2 − C1)"), ("copy_dir", "Direction with copy fixed (C3 − C2)")):
        parts = []
        for c in [c for c in S.get("contrasts", []) if c["kind"] == kind]:
            d = c["diff"].get("ceiling")
            verdict = ("lower" if num(d[0]) < 0 else "higher") if _excl0(d) else "no detectable change in"
            parts.append(f"{SHORT[c['model']]} {c['a'].rsplit('_', 1)[0] if c['a'].endswith(('qwen3', 'bos')) else c['a']} vs {c['b']}: "
                         f"M {f3(c['a_vals']['ceiling'][0])} vs {f3(c['b_vals']['ceiling'][0])} ({verdict} M; Δ {ci(d, sign=True)})")
        if parts:
            out.append(f"**{title}.** " + "; ".join(parts) + ".")
    # 4. sanity
    parts = []
    for m in MODELS:
        c = _c(S, "dir_opt", "P3wino", m)
        if c:
            parts.append(f"{SHORT[m]} M {f3(c['a_vals']['ceiling'][0])} vs Phase 3 {f3(c['b_vals']['ceiling'][0])}, greedy k80 "
                         f"{fk(c['a_vals']['greedy']['k80'][0])} vs {fk(c['b_vals']['greedy']['k80'][0])}")
    if parts:
        out.append("**The option swap on the F1-margin subset reproduces Phase 3** (independent samples; the F1-margin twins are a "
                   "subset of the 776): " + "; ".join(parts) + ", so the direction contrast is not a sample artefact.")
    return out + READINGS


def num(x):
    if x is None:
        return float("nan")
    if isinstance(x, str):
        return float("inf") if x in ("inf", "Infinity") else (float("-inf") if x == "-inf" else float(x))
    return float(x)


def f3(v, d=3, sign=False):
    v = num(v)
    if not np.isfinite(v):
        return "n/a" if np.isnan(v) else "never"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def ci(x, d=3, sign=False):
    if not x:
        return "n/a"
    return f"{f3(x[0], d, sign)} [{f3(x[1], d, sign)}, {f3(x[2], d, sign)}]"


def fk(v):
    v = num(v)
    return "never" if np.isinf(v) else ("n/a" if np.isnan(v) else str(int(v)))


def kci(x):
    if not x:
        return "n/a"
    return f"{fk(x[0])} [{fk(x[1])}, {fk(x[2])}]"


def k80s(r):
    b = r["boot"]
    if num(b["ceiling"][0]) > CEIL_MIN:
        return f"k80 greedy / oracle / random {fk(b['greedy']['k80'][0])} / {fk(b['oracle']['k80'][0])} / {fk(b['rand']['k80'][0])}"
    return (f"ceiling ≤ {CEIL_MIN}: k for 80 % of each curve's maximum greedy / oracle / random {fk(b['greedy'].get('k80max', [None])[0])} / "
            f"{fk(b['oracle'].get('k80max', [None])[0])} / {fk(b['rand'].get('k80max', [None])[0])}")


def headline(r):
    b = r["boot"]
    dr = r.get("direct") or {}
    s = (f"{SHORT[r['model']]} M = {ci(b['ceiling'])} ({r['n_units']} pairs), {k80s(r)}, oracle overshoot "
         f"{ci(b['oracle']['overshoot'], sign=True)}, greedy r(15) {f3(b['greedy']['r15'][0])}")
    if dr.get("a_dir"):
        s += f", direct attention / MoE {f3(dr['a_dir'][0], 2)} / {f3(dr['m_dir'][0], 2)}"
    return s


def table_md(name):
    p = os.path.join(R, "tables", name + ".md")
    return open(p).read().strip() if os.path.exists(p) else f"_(table {name} missing)_"


def contrast_line(c):
    d = c["diff"]
    a, b = c["a_vals"], c["b_vals"]
    s = (f"{SHORT[c['model']]}, {c['label']} ({'paired, ' + str(c['n_common_units']) + ' common ' + c['cluster'] + ' units' if c['paired'] else 'independent samples'}): "
         f"ceiling {f3(a['ceiling'][0])} vs {f3(b['ceiling'][0])}, Δ {ci(d.get('ceiling'), sign=True)}; "
         f"oracle overshoot Δ {ci(d.get('oracle', {}).get('overshoot'), sign=True)}; greedy AUC k ≤ 15 Δ {ci(d.get('greedy', {}).get('auc_log_k15'), sign=True)}")
    if min(num(a["ceiling"][0]), num(b["ceiling"][0])) > CEIL_MIN:
        s += f"; greedy k80 {fk(a['greedy']['k80'][0])} vs {fk(b['greedy']['k80'][0])}"
    else:
        s += f"; greedy k80 of own max {fk(a['greedy'].get('k80max', [None])[0])} vs {fk(b['greedy'].get('k80max', [None])[0])}"
    s += f"; answer restored by all MoE Δ {ci(d.get('restored_all_moe'), 2, sign=True)}"
    return s


def runs_table(S):
    rows = ["| Run dir | Condition | Model | Family | Eval. directed cases (pairs) | Passes | Greedy rows (pool) | Noising-single rows | Prefill Δ SD across passes (median / 99th pct, clean) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, r in sorted(S["runs"].items(), key=lambda kv: (AB.PLAN_ORDER.index(kv[1]["cond"]) if kv[1]["cond"] in AB.PLAN_ORDER else 99, kv[1]["family"], kv[1]["model"])):
        e8 = r.get("ext8", {})
        pr = e8.get("prefill_range", {})
        meta = json.load(open(os.path.join(R, r["out"], "run_meta.json")))
        g = e8.get("greedy", {})
        rows.append(f"| `results/{r['out']}` | {r['tlabel']} | {SHORT[r['model']]} | {r['family']} | {r['n_eval_cases']} ({r['n_units']}) | "
                    f"{meta.get('passes', '?')} | {g.get('n_rows', '?')} ({g.get('pool', '?')}) | {meta.get('noise_rows', '?')} | "
                    f"{f3(pr.get('clean_sd_median'), 2)} / {f3(pr.get('clean_sd_p99'), 2)} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(R, "ext13ab_summary.json"))
    ap.add_argument("--out", default=os.path.join(R, "sections", "ext13_addback.md"))
    args = ap.parse_args()
    S = json.load(open(args.summary))
    runs = S["runs"]
    out = []
    # ---------------- summary
    out.append("**Summary.** Expert add-back and deletion curves (Phase 3's Direction 8 machinery, `moetrace/ext8_addback.py` "
               "unchanged) for every Phase-5 condition with a ready base run, Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS, STR pairs, "
               "validation directed cases, final position. M = all MoE outputs patched back (fraction of the drop); k80 = experts "
               "for 80 % of M (n/a when M ≤ 0.1, then k for 80 % of each curve's own maximum); overshoot = max r(k) − M of the "
               "per-case oracle ordering.")
    out.append("")
    conds = [c for c in AB.PLAN_ORDER if any(r["cond"] == c and r["family"] == "main" for r in runs.values())]
    for c in conds:
        rs = sorted([r for r in runs.values() if r["cond"] == c and r["family"] == "main"], key=lambda r: r["model"] != "qwen3")
        out.append(f"- **{AB.COND_LABEL.get(c, c)}** (`{c}`): " + "; ".join(headline(r) for r in rs) + ".")
    for k, r in S.get("phase3", {}).items():
        pass
    p3 = S.get("phase3", {})
    if p3:
        out.append("- Phase-3 references (same code): " + "; ".join(
            f"{r['tlabel']} {SHORT[r['model']]} M = {f3(r['boot']['ceiling'][0])}, greedy / oracle k80 {fk(r['boot']['greedy']['k80'][0])} / "
            f"{fk(r['boot']['oracle']['k80'][0])}, overshoot {f3(r['boot']['oracle']['overshoot'][0], sign=True)}" for r in p3.values()) + ".")
    kinds = {"copy": "Copy availability (C2 − C1)", "copy_dir": "Direction with the answer copyable in both cells (C3 − C2)",
             "direction": "Direction on identical items (attribute → entity minus entity → attribute)", "verb": "Verb control",
             "sanity": "Sanity check", "other": "Other comparisons"}
    for kd, title in kinds.items():
        cs = [c for c in S.get("contrasts", []) if c["kind"] == kd]
        if cs:
            out.append(f"- **{title}.** " + " ".join(contrast_line(c) + "." for c in cs))
    rd = readings(S)
    if rd:
        out.append("")
        out.append("**Readings** (templated from the numbers above; conditions still running are absent).")
        out += [f"{i + 1}. {t}" for i, t in enumerate(rd)]
    # ---------------- what was run
    out += ["", "### What was run", "",
            "**Tasks.** One add-back task per condition × model × case-set family listed in `data/wino_ref/p5_conditions.json` "
            "(agent p5-data; behaviour-only conditions excluded): ext8's WinoGrande STR task (`load_wino`) on the condition's base run "
            "`results/p5_<cond>_<proto>/` (clean-run routing at the final position, all-layer single-expert rows of discovery and validation "
            "cases, MoE-layer sweep rows), its pair parquet and case-set JSON. Candidates = the clean run's routed (layer, expert) pairs at "
            "the final position (Qwen3 48 × 8 = 384, Mixtral 32 × 2 = 64). Family `main` → `results/p5_<cond>_<proto>_ab/`, the replication "
            "family → `results/p5_<cond>_<proto>_rep_ab/` (ranking from its discovery split, curves on its validation split).",
            "",
            "**Work per task** (`moetrace/ext13ab.py`, driver `scripts/ext13ab_run.py`): A0 ceilings for every row in both directions "
            "(all MoE, all attention, both, MoE ≤ L−5, attention ≤ L−5); A1 static add-back orderings oracle, pop, layerwise, dla, vnorm, "
            "weight, rand0–4 on ext8's k grid plus the all-clean-active endpoint; A2 adaptive greedy, 15 steps (Qwen3 pool = the row's "
            "top-32 singles, Mixtral all 64); A3 noising singles (Qwen3 the first 128 validation rows as in Phase 3, Mixtral all validation "
            "rows) and deletion curves pop, oracle, layerwise, rand0–1, dla, noise_oracle plus the endpoint. Not run (as in ext12): beam, "
            "exact top-10, Shapley. Several conditions of one model share every engine pass (ext12's multi-task scheme); each run dir is "
            "in exactly the ext8 file format, so `ext8_addback.load_out` and `scripts/ext8_addback_analyze.analyse` read it unchanged.",
            "",
            "**Statistics** (`scripts/ext13ab_analyze.py`). ext8's `analyse` plus ext12's unit-cluster bootstrap of the curve summaries "
            "(5,000 resamples, seed 0, units = pairs: both directed cases of a pair together): M, answer restored by all MoE, AUC over "
            "log k (1..K and 1..15), k80, max r, overshoot, greedy − best static r(10), DLA − oracle AUC; the same for the deletion "
            "curves. Template conditions (verb templates × IOI name pairs; many pairs share a seed template) also get seed-clustered CIs "
            "(same point estimates). Contrasts on identical items are paired (restricted to items in both conditions' validation sets, "
            "same resamples over the common units) when ≥ 20 units are shared, otherwise independent (seed 0 vs seed 1). The "
            "attention-vs-MoE bars take the direct-path split from p5-data's `direct_split.parquet` (validation cases of the family); "
            "this run's own `addback_direct.parquet` (same quantity, first pass) is the fallback and a cross-check.",
            "", runs_table(S), ""]
    # ---------------- results
    out += ["### Per-condition results", "", table_md("ext13ab_conditions"), "",
            "![Task-ladder overview](figures/ext13ab_overview.png)", "",
            "![All-MoE ceilings and direct-path split, Qwen3](figures/ext13ab_split_all_qwen3.png)", "",
            "![All-MoE ceilings and direct-path split, Mixtral](figures/ext13ab_split_all_mixtral.png)", ""]
    for c in conds:
        for fam in ("main", "rep"):
            if any(r["cond"] == c and r["family"] == fam for r in runs.values()):
                nm = f"ext13ab_{c}" + ("_rep" if fam == "rep" else "")
                lab = AB.COND_LABEL.get(c, c) + (" (replication)" if fam == "rep" else "")
                out.append(f"**{lab}.** ![add-back](figures/{nm}_addback.png) ![deletion](figures/{nm}_deletion.png) ![split](figures/{nm}_split.png)")
                out.append("")
    out += ["### Contrasts", "", table_md("ext13ab_contrasts"), "",
            "### Orderings (add-back)", "", table_md("ext13ab_orderings"), "",
            "### Deletion", "", table_md("ext13ab_deletion"), "",
            "### Strata (names / objects, association probe)", "", table_md("ext13ab_strata"), ""]
    if any("boot_seed" in r for r in runs.values()):
        out += ["### Seed-clustered CIs (template conditions)", "", table_md("ext13ab_seed_ci"), ""]
    out += ["### Deviations and caveats", "",
            "- Prefill rows: a task's first pass holds every case (A0 runs on discovery rows too; the DLA and direct-split diagnostics "
            "are sliced from it); later passes hold only the validation cases, the only ones later work refers to (rows unchanged; "
            "bf16 batch-composition noise as in Phase 3, see the prefill SD column).",
            "- Deletion k80 is computed on each curve's own cases (noise_oracle: its noising-single rows), so its ceiling can differ "
            "slightly from the all-validation deletion ceiling shown in the figures.",
            "- k80 of the ceiling is undefined when M ≤ 0.1 (a negative or near-zero ceiling is reached by any k); those cells report "
            "k for 80 % of each curve's own maximum instead (a different quantity: it says how fast a curve saturates, not how much).",
            "- Noising singles: Qwen3 the first 128 validation rows (Phase 3), Mixtral all validation rows (Phase 3: 128; 64 candidates "
            "per row make the full set cheap).",
            "- No beam / exact / Shapley (ext12 scope); greedy ≈ optimal was shown in Phase 3.",
            "", "### Files", "",
            "Code `moetrace/ext13ab.py`, `scripts/ext13ab_{run,analyze,plot,text,ready}.py`, chains `scripts/ext13ab_chain.sh`, smoke "
            "`scripts/ext13ab_smoke_chain.sh` (`results/p5_F1pre_*_ab_smoke`, testing only). Runs `results/p5_*_ab/`. Numbers "
            "`results/ext13ab_summary.json`; tables `results/tables/ext13ab_*`; figures `results/figures/ext13ab_*`."]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
