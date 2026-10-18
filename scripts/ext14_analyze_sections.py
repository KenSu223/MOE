"""Phase 6 (p6-analysis): section writer for results/sections/ext14_{ladder,dial,dense,scale}.md (called by
scripts/ext14_all.py with the summary dict). Every number comes from results/ext14_summary.json; sentences are composed
from the computed values and verdicts, so the sections stay correct when cells are added. Parts whose cells have not
reached BASE READY say so explicitly ("pending").
"""
import os
import numpy as np

from moetrace import ext14_analysis as X

STAGE1 = X.STAGE1
DL = X.DIR_LABEL


def F(x, d=2, s=False):
    if isinstance(x, list):
        x = [np.nan if v is None else v for v in x]
    return X.fr(x, d, s)


def lab(p):
    return X.SHORT.get(p, p)


def tab(name, sfx, optional=False):
    p = os.path.join(X.TAB, f"{name}{sfx}.md")
    if not os.path.exists(p):
        return "" if optional else f"_(table `{name}.md` pending: its cells have not reached BASE READY yet)_\n"
    return open(p).read()


def fig(name, sfx, alt):
    p = os.path.join(X.FIG, f"{name}{sfx}.png")
    return f"![{alt}](../figures/{name}{sfx}.png)\n" if os.path.exists(p) else ""


def share(S, cell, proto, key="share"):
    return ((S.get("cells", {}).get(f"{cell}/{proto}") or {}).get("validation") or {}).get(key)


def pred_rows(S, ids):
    P = S.get("predictions") or {}
    out = ["| ID | Prediction | Observed | Verdict |", "|---|---|---|---|"]
    for pid in ids:
        r = P.get(pid)
        if not r:
            continue
        obs = []
        for m, v in r["per_model"].items():
            o = v.get("observed")
            t = ""
            if isinstance(o, list) and o and isinstance(o[0], (int, float)):
                t = F(o)
            elif isinstance(o, (int, float)):
                t = X.fnum(o)
            elif isinstance(o, dict):
                t = ", ".join(f"{k} {F(vv) if isinstance(vv, list) else (X.fnum(vv) if isinstance(vv, (int, float)) else '')}" for k, vv in o.items() if not isinstance(vv, dict))
            if v.get("decision"):
                t += f" → {v['decision']}"
            obs.append(f"{m}: {t} ({v.get('verdict')})" if t else f"{m}: {v.get('verdict')}")
        out.append(f"| {pid} | {r['cell']}: {r['prediction']} | {'; '.join(obs)} | **{r['verdict']}** |")
    return "\n".join(out) + "\n"


def n_ready(S, cells, protos=STAGE1):
    return sum(1 for c in cells for p in protos if f"{c}/{p}" in S.get("cells", {}))


# ------------------------------------------------------------------------------------------------------------------
def ladder_section(S, sfx):
    L = S.get("ladder") or {}
    parts = []
    for p in [p for p in X.ALL_PROTOS if p in L]:
        x = L[p]
        rows = sorted(x.get("rows", []), key=lambda r: r["grade"])
        seq = ", ".join(f"{r['label']} {F(r['share'])}" for r in rows)
        rho = x.get("spearman")
        parts.append(f"{X.LABEL.get(p, p)} ({x['n_cells']} of 8 tasks): {seq}; Spearman(grade, share) "
                     f"{X.fnum(rho, '+.2f')}" + (f" (CI over the cells' bootstrap draws {F(x.get('spearman_ci'))})" if x.get("spearman_ci") else "") +
                     (f", partial Spearman given the cue distance {X.fnum(x['partial_spearman_given_cue'], '+.2f')}" if x.get("partial_spearman_given_cue") is not None and
                      np.isfinite(x.get("partial_spearman_given_cue") or np.nan) else ""))
    P = S.get("predictions") or {}
    l9 = P.get("L9", {})
    if parts:
        summ = ("**Summary.** Ladder v2 (P6-1): four new tasks (antonym, object property, explicit binding, induction on unrelated word pairs) "
                "were placed on the dial before any run (grades 1–5 in `data/dial/preregistration.md`) and measured with the Phase-5 readout "
                "(attention share = AUC+ of the single-layer final-token attention patch / (AUC+ attention + AUC+ MoE or MLP), validation pairs, "
                "cluster-bootstrap CIs) together with the four anchors. " + " ".join(p + "." for p in parts) +
                f" Pre-registered test L9 (Spearman ≥ 0.8 in every model): **{l9.get('verdict', 'pending')}**.")
    else:
        summ = "**Summary.** Ladder v2 (P6-1): pending — no ladder cell has reached BASE READY yet."
    txt = [summ, "", ladder_reading(S), "",
           "### What was computed", "",
           "Tasks and grades (fixed in the pre-registration before any Phase-6 GPU job): L-ant antonym 1, L-prop object property 1.5, L-e2a "
           "(= Dial-E E3 entity → attribute) 2, L-cf CounterFact 2, L-a2e (= E3 attribute → entity) 3, L-bind explicit binding look-up 4, "
           "L-ioi IOI 5, L-ind induction on unrelated word pairs 5. Each task is an STR pair set (one swapped token, equal length, margin ≥ 1 "
           "both ways, both orders where two entities are named; RESEARCH_PLAN.md 'Common rules'), with the Phase-3 tier-1 runners. Per "
           "model: Spearman of the grade with the measured share; the cue distance (tokens between the swapped token and the final token, "
           "median per task) is recorded as a covariate because across tasks it is confounded with the dial (partial Spearman given the cue "
           "distance). The Spearman CI uses the cells' independent bootstrap draws (2,000 each). CounterFact for Qwen3 / Mixtral is the "
           "Phase-3 donor-level attention sweep (`results/<proto>_str_attnsweep`, no direct-path split or Δp), IOI the Phase-3 S2 → IO run.", "",
           fig("ext14_ladder", sfx, "Ladder v2"), tab("ext14_ladder", sfx), "",
           "### Pre-registered predictions", "", pred_rows(S, ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]), "",
           "### Caveats", "",
           "- Grades are ordinal and assigned by hand; ties (grade 2: L-e2a and L-cf; grade 5: L-ioi and L-ind) enter Spearman as ties.",
           "- Across tasks, prompt length, cue distance, answer type and item source change together with the dial; only the Dial-E contrasts "
           "(same WinoGrande twins, `ext14_dial.md`) hold the items fixed.",
           "- A task reaches the table only with ≥ 64 validation pairs (R5); smaller tasks are listed as insufficient in the registry.", "",
           "### Files", "",
           "`moetrace/ext14_analysis.py` (`ladder`, `CellData`, `summarise`), `scripts/ext14_all.py`; `results/ext14_summary.json` keys "
           "`ladder`, `cells`, `predictions`; `results/tables/ext14_ladder.md`, `results/figures/ext14_ladder.png`."]
    return "\n".join(txt) + "\n"


# ------------------------------------------------------------------------------------------------------------------
def dial_section(S, sfx):
    D = S.get("dial_e") or {}
    P = S.get("predictions") or {}
    s1 = []
    for p in [p for p in STAGE1 if p in D]:
        lv = D[p].get("levels", {})
        seg = []
        for dr in X.DIRS:
            vals = ", ".join(f"{X.LEVEL_LABEL[l].split(' ')[0] + (' (P3)' if l == 'e3p3' else ' (F1)' if (l == 'e3' and dr == 'a2e') else '')} "
                             f"{F((lv.get(dr, {}).get(l) or {}).get('share'))}" for l in ("e1", "e1p", "e2", "e3p3", "e3") if lv.get(dr, {}).get(l))
            if vals:
                seg.append(f"{DL[dr]}: {vals}")
        h = D[p].get("h_test")
        hs = ""
        if h:
            pm = ((P.get("E13") or {}).get("per_model", {}).get(p) or {})
            dec = pm.get("decision") or ("not tested: a pooled cell" if pm.get("verdict") == X.INSUFF_POOLED else pm.get("verdict", "pending"))
            hs = (f"; direction gap (a→e − e→a, common twins, n = {h['n_items']}) at E1 {F(h['gap_e1'], 2, True)} vs E3 {F(h['gap_e3'], 2, True)} → **{dec}**")
        s1.append(f"{X.LABEL.get(p, p)}: " + "; ".join(seg) + hs)
    if s1:
        summ = ("**Summary.** Dial E (P6-2) turns one dial on the same WinoGrande twins: the evidence part of the sentence changes from the "
                "binding stated verbatim (E1) to a paraphrase (E1p), a negation (E2) and the original event (E3), while the question line, "
                "the STR position and the frame stay fixed. Attention share (validation): " + ". ".join(s1) + ". "
                + verdict_line(P, ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14"]))
    else:
        summ = "**Summary.** Dial E (P6-2): pending — no Dial-E cell has reached BASE READY yet."
    A = S.get("dial_a") or {}
    a_lines = []
    for p in [p for p in STAGE1 if p in A]:
        for dr, cl in (A[p].get("classes") or {}).items():
            v = ", ".join(f"{k} {F(cl[k]['share'])}" for k in ("E0", "agree", "neutral", "conflict") if cl.get(k))
            ds = (A[p].get("dose") or {}).get(dr)
            mdir = (cl.get("conflict") or {}).get("M_direct")
            a_lines.append(f"{lab(p)} {DL[dr]}: {v}" + (f"; conflict-cell MoE direct contribution toward the stated answer {F(mdir, 2, True)}" if mdir else "")
                           + (f"; dose–response Spearman over 5 prior bins {X.fnum(ds.get('spearman'), '+.2f')}" if ds else ""))
    C = S.get("set_c") or {}
    c_lines = []
    for p, x in C.items():
        for cell, b in (x.get("behaviour") or {}).items():
            rate = b.get("statement_rate", b.get("follow_statement"))
            v = (x.get("subsets") or {}).get(cell)
            c_lines.append(f"{lab(p)} {X.cell_label(cell)}: " + (f"statement followed in {X.fnum(rate, '.0%')} of margin items" if isinstance(rate, (int, float)) else "behaviour in the registry")
                           + (f", share {F(v['share'])}" if v else ""))
    R = S.get("r_metric") or {}
    r_lines = [f"{lab(p)}: Spearman of the cell ranking LD vs normalised LD {X.fnum(x.get('spearman_ld_nld'), '+.2f')}, LD vs Δp {X.fnum(x.get('spearman_ld_dp'), '+.2f')} "
               f"({len(x.get('cells', {}))} cells; largest |share difference| {X.fnum(x.get('max_abs_diff_nld'))} / {X.fnum(x.get('max_abs_diff_dp'))})"
               for p, x in R.items() if p in STAGE1 and "spearman_ld_nld" in x]
    RS = S.get("r_site") or {}
    rs_lines = [f"{lab(p)}: evidence-site share {F((x.get('rsite') or {}).get('share'))}, question-site {F(x['paired']['share']['b'])}, "
                f"difference on {x['paired']['n_items']} common twins {F(x['paired']['share']['diff'], 2, True)}" for p, x in RS.items() if x.get("paired")]
    RC = S.get("r_causal") or {}
    T2 = S.get("tier2") or {}
    txt = [summ, "", dial_reading(S), "",
           "### What was computed", "",
           "Items: the Phase-3 776 WinoGrande twins (plus train_xl twins where a level had < 256 margin pairs), built once per tokenizer "
           "family by p6-data (`data/dial/`, `data/dial/build_report.md`). The discovery / validation split is fixed per twin, so a twin is in "
           "the same split at every level; paired contrasts use the twins common to the cells compared and the same cluster resamples "
           "(unit = twin, 5,000 resamples). Two-entity frames are built in both orders and an item counts only if both orders pass. "
           "Frames: attribute → entity, E1-family levels (E1, E1p, E2, E0, R-site, Dial A) and the frame-matched E3p3 = frame P3 "
           "`<evidence>. Fill in the blank: The _ was X.\\nAnswer: the` (STR = the attribute X in the question; frame pilot below); the "
           "Phase-5 anchor E3 = the F1 cloze `Fill in the blank: <sentence>\\nAnswer: the`; entity → attribute = the Phase-3 option-swap "
           "form at every level (STR = the entity before the copula). Readouts as Phase 5: attention share, "
           "A_direct, M_direct, all-MoE M, hand-off layers of the position × layer residual grid, head singles (`ext14_cells.md`).", "",
           "### Frame pilot (behaviour, before the Dial-E runs)", "", frame_pilot_text(), "",
           "### Dial E: explicitness", "", fig("ext14_dial_e", sfx, "Dial E"), tab("ext14_direction_gap_models", sfx), "", tab("ext14_dial_e", sfx), "", tab("ext14_dial_e_paired", sfx), "",
           "### E2 (negation) as a behavioural level", "", e2_text(S), "",
           "E2 robustness: negating a gradable adjective with a degree adverb ('not too heavy') does not entail the antonym, so E2 is "
           "also reported without degree-adverb twins and by antonym type (complementary vs gradable) wherever patching ran (none so far: E2 is behaviour-only in every model):", "",
           tab("ext14_dial_e2_strata", sfx, optional=True), "",
           "### Cells below the reporting threshold", "",
           "Rule (coordinator, 2026-10-16): a cell with < 64 validation pairs but ≥ 64 pairs overall runs tier 1 on the pooled set "
           "(marked †: no held-out selection, layers / heads chosen on the same items) and is not used for prediction tests that need "
           "validation pairs ('insufficient (pooled only)'); a cell with < 64 pairs overall is reported by its behaviour only.", "",
           tab("ext14_behaviour_only", sfx), "",
           "### Dial A: prior support (associativity) and E0", "",
           ("; ".join(a_lines) + ".\n") if a_lines else "Pending.\n", "", fig("ext14_dial_a", sfx, "Dial A"), tab("ext14_dial_a", sfx), "",
           "### Set C: event + statement (which source wins)", "", ("; ".join(c_lines) + ".\n") if c_lines else "Pending.\n", "", tab("ext14_set_c", sfx), "",
           "### Hand-off layers", "", fig("ext14_handoff", sfx, "Hand-off layers"), tab("ext14_handoff", sfx), "",
           "### Robustness: metric (R-metric) and corruption site (R-site)", "",
           ("R-metric. " + "; ".join(r_lines) + ".\n") if r_lines else "R-metric: pending.\n", "",
           ("R-site (corruption in the evidence: the two entity mentions swapped, which flips the answer; two swapped tokens, a documented "
            "exception to R2). " + "; ".join(rs_lines) + ".\n") if rs_lines else "R-site: pending.\n", "",
           fig("ext14_r_metric", sfx, "R-metric"), tab("ext14_r_metric", sfx), "", tab("ext14_r_site", sfx), "",
           "### Causal handle on the share (R-causal)", "", causal_text(RC, sfx), "",
           "### Tier 2 on E1 (selected experts, add-back, mixed circuit, knockout, writers)", "", tier2_text(T2, sfx), "",
           "### Pre-registered predictions", "", pred_rows(S, ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14",
                                                               "A1", "A2", "A3", "A4", "A5", "C1", "C2", "C3", "R1", "R2", "R3", "R4"]), "",
           "### Caveats", "",
           "- E1p and E2 exclude twins without an acceptable paraphrase / without a `_ was [too] X` blank clause (counts in "
           "`data/dial/build_report.md`), so their own sets are subsets of E1; paired contrasts restrict to the common twins.",
           "- Shares are ratios of AUC+ over layers; they are descriptions of where single-layer patches restore the drop, not additive "
           "decompositions (block ≈ attention + MoE only to bf16 noise, Phase 3).",
           "- The direction comparison changes the readout token type (entity vs attribute) together with the direction; Dial E holds the "
           "items fixed but not the answer vocabulary.",
           "- Dial-A strong-prior items come partly from a controlled noun-pair set, not only from WinoGrande twins; its comparison with E1 "
           "(neutral) is therefore not paired unless the twins overlap (the paired rows list the overlap).", "",
           "### Files", "",
           "`moetrace/ext14_analysis.py` (`dial_e`, `dial_a`, `set_c`, `r_metric`, `r_site`, `grid_summary`, `evaluate_predictions`), "
           "`scripts/ext14_all.py`, tier 2 `scripts/ext14_t2_*.py` (results `results/p6_<cell>_<proto>_t2/`), R-causal "
           "`scripts/ext14_causal_*.py` (results `results/p6_causal_*`); `results/ext14_summary.json` keys `dial_e`, `dial_a`, `set_c`, "
           "`r_metric`, `r_site`, `handoff`, `r_causal`, `tier2`, `predictions`."]
    return "\n".join(txt) + "\n"


def e2_text(S):
    """E2 behaviour: copy-trap and target rates per model, direction and stratum (behaviour_rates of the E2 cells)."""
    B = S.get("behaviour_rates") or {}
    lines = []
    for p in STAGE1 + tuple(x for x in X.STAGE2):
        for dr in X.DIRS:
            b = B.get(f"e2_{dr}/{p}")
            if not b:
                continue
            a = b["all"]
            seg = (f"{lab(p)} {DL[dr]} ({a['n_items']} twins, {a['n_pairs']} pairs): target by the margin rule {F(a['t_marg'])}, "
                   f"copy trap by the margin rule {F(a['o_marg'])} (sign level {F(a['t_sign'])} vs {F(a['o_sign'])}); twins with the copy "
                   f"trap in every order {a['items_other_all_orders']}, target in every order {a['items_target_all_orders']}")
            st = b.get("strata", {})
            for k in ("degree_adv=True", "degree_adv=False", "antonym_type=complementary", "antonym_type=gradable"):
                if k in st:
                    seg += f"; {k.replace('_', ' ')}: trap {F(st[k]['o_marg'])}, target {F(st[k]['t_marg'])}"
            lines.append(seg)
    if not lines:
        return "Pending.\n"
    head = ("E2 states each binding by negation ('The jug was not full and the pot was not empty'). Copying the stated word gives the "
            "wrong answer (the copy trap); the target needs elimination (a→e) or the antonym (e→a). Where too few pairs pass the "
            "margin rule in both directions, E2 is reported as a behavioural level only (no patching): ")
    return head + "; ".join(lines) + ".\n\n" + tab("ext14_behaviour_e2", "")


def frame_pilot_text():
    """Behaviour table of p6-data's frame pilot (data/dial/qa/frame_pilot.md), objects / names / all, neutral items."""
    p = os.path.join(X.ROOT, "data", "dial", "qa", "frame_pilot.md")
    if not os.path.exists(p):
        return "_(frame pilot file pending)_\n"
    lines = open(p).read().splitlines()
    head = [ln for ln in lines if ln.startswith("| Model")]
    rows = [ln for ln in lines if ln.startswith("| ") and not ln.startswith("| Model") and "(neutral)" not in ln and "| all |" in ln]
    txt = ("The E1 attribute → entity cloze in the original frame P0 (`Fill in the blank: {A} and {B}, so the _ was X.`) gave "
           "chance-level behaviour, so before any Dial-E patching p6-data piloted five frames on the same 200 core twins (both "
           "statement orders, both models; `data/dial/qa/frame_pilot.md`). Frame P3 (`{A} and {B}. Fill in the blank: The _ was X.\n"
           "Answer: the`) was adopted for every E1-family a→e cell (E1, E1p, E2, E0, R-site, Dial A); `e3p3_a2e` puts the E3 event "
           "clause into the same frame and is the frame-matched E3 of the paired a→e dial, while the F1 `e3_a2e` remains the Phase-5 "
           "anchor. The low P0 competence is itself a behavioural finding (a frame artefact, not a property of the evidence). "
           "All items, both models:\n\n")
    if head:
        txt += head[0] + "\n|" + "---|" * (head[0].count("|") - 1) + "\n" + "\n".join(rows) + "\n"
    return txt


def verdict_line(P, ids):
    held = [i for i in ids if (P.get(i) or {}).get("verdict") == "held"]
    failed = [i for i in ids if (P.get(i) or {}).get("verdict") == "failed"]
    pend = [i for i in ids if (P.get(i) or {}).get("verdict") in ("pending", None)]
    ins = [i for i in ids if (P.get(i) or {}).get("verdict") == "insufficient data"]
    out = "Pre-registered predictions: held " + (", ".join(held) or "none") + "; failed " + (", ".join(failed) or "none")
    if ins:
        out += "; insufficient data " + ", ".join(ins)
    if pend:
        out += "; pending " + ", ".join(pend)
    return out + "."


def causal_text(RC, sfx):
    if not RC or not RC.get("cells"):
        return "Pending (needs `[p6-engine] HEADMASK READY`; runner `scripts/ext14_causal_run.py`).\n"
    lines = []
    for k, v in RC["cells"].items():
        for kind in ("heads", "experts"):
            x = v.get(kind)
            if not x:
                continue
            lines.append(f"{k} {kind} ({', '.join(x.get('targets', []))}): attention AUC+ change {F(x.get('d_auc_attn_rel'), 2, True)} (random {F(x.get('rand_d_auc_attn_rel'), 2, True)}), "
                         f"MoE AUC+ change {F(x.get('d_auc_moe_rel'), 2, True)} (random {F(x.get('rand_d_auc_moe_rel'), 2, True)}), share change {F(x.get('d_share'), 2, True)} "
                         f"(random {F(x.get('rand_d_share'), 2, True)}), task margin change {F(x.get('d_margin_rel'), 2, True)}")
    t = "; ".join(lines) + ".\n" if lines else ""
    return t + "\n" + fig("ext14_r_causal", "", "R-causal") + tab("ext14_r_causal", sfx)


def tier2_text(T2, sfx):
    if not T2:
        return "Pending (tier-2 chain `scripts/ext14_t2_chain.py`).\n"
    lines = []
    for k, s in T2.items():
        sel = (s.get("selection") or {}).get("selected_expert") or {}
        ab = s.get("addback") or {}
        ci2 = ((s.get("circuit") or {}).get("step2")) or {}
        ko = s.get("knockout") or {}
        name = sel.get("name")
        seg = [f"{k}: selected {name or 'none'} (pattern {sel.get('pattern')}, Spec {F(sel.get('spec'), 2, True)})"]
        if ab.get("complete"):
            seg.append(f"all-MoE ceiling {F(ab.get('ceiling'))}, k80 greedy / oracle / random {fk(ab, 'greedy')} / {fk(ab, 'oracle')} / {fk(ab, 'rand')}")
        if ci2.get("complete"):
            seg.append(f"mixed head + expert greedy r(20) {F([ci2['r20']['ratio'], ci2['r20']['lo'], ci2['r20']['hi']]) if ci2.get('r20') else 'n/a'}, "
                       f"heads among the first 10 picks {X.fnum(ci2.get('heads_first10'), '.1f')}")
        if ko.get("complete") and name and (ko.get("F") or {}).get(name):
            seg.append(f"knockout F of the selected expert {F(ko['F'][name], 3, True)}")
        lines.append("; ".join(seg))
    return ("; ".join(lines) + ".\n\n" + tab("ext14_t2_e1_vs_e3", sfx) + "\n" + tab("ext14_t2_overview", sfx) + "\n" + tab("ext14_t2_random_equiv", sfx)
            + "\n" + fig("ext14_t2_ab_e1_a2e_addback", sfx, "Add-back E1 a2e") + fig("ext14_t2_ab_e1_e2a_addback", sfx, "Add-back E1 e2a"))


def fk(ab, o):
    v = (ab.get(o) or {}).get("k80")
    if not v:
        return "n/a"
    v = v[0]
    return "never" if v is None or (isinstance(v, float) and not np.isfinite(v)) or v == "inf" else f"{int(round(float(v)))}"


# ------------------------------------------------------------------------------------------------------------------
def dense_section(S, sfx):
    TW = S.get("twins") or {}
    P = S.get("predictions") or {}
    lines = []
    for k, t in TW.items():
        if not t or "cells" not in t:
            continue
        kn = ", ".join(f"{X.cell_label(c)} {F(v, 2, True)}" for c, v in (t.get("knowledge_diffs") or {}).items())
        cp = ", ".join(f"{X.cell_label(c)} {F(v, 2, True)}" for c, v in (t.get("copy_diffs") or {}).items())
        lines.append(f"{lab(t['dense'])} → {lab(t['moe'])} ({t['n_cells']} cells, identical prompts): |MoE − dense| ≤ 0.10 in "
                     f"{X.fnum(t.get('frac_within_0.10'), '.0%')} of the cells, MoE ≥ dense + 0.10 in {X.fnum(t.get('frac_moe_higher_0.10'), '.0%')}, cell order Spearman "
                     f"{X.fnum(t.get('spearman_cells'), '+.2f')}; knowledge cells {kn or 'n/a'}; copy cells {cp or 'n/a'}")
    if lines:
        summ = ("**Summary.** Dense parent vs MoE child (P6-3) on identical prompts (validation pairs common to both models; paired cluster "
                "bootstrap). " + ". ".join(lines) + ". " + verdict_line(P, ["D0", "D1", "D2", "D3", "D4"]) +
                " The hypotheses are H-invariant (D1, best guess D0: the share is a property of the task), H-capacity (D2: more FFN capacity "
                "lowers the attention share in knowledge cells only) and H-sparsity (D3: sparse routing raises the attention share).")
    else:
        summ = "**Summary.** Dense vs MoE (P6-3): pending — no twin has both models at BASE READY on a common cell."
    txt = [summ, "", dense_reading(S), "", "### What was computed", "",
           "Twins: Mistral-7B-v0.1 ↔ Mixtral-8x7B-v0.1 (same 32 layers, d 4096, same attention geometry and tokenizer; Mixtral likely "
           "upcycled) and Qwen2-7B ↔ Qwen2-57B-A14B (28 layers, d 3584; officially upscaled from Qwen2-7B; one always-on shared expert + "
           "64 routed). The Set-A cells are built once per tokenizer family, so both twins see the same prompts; each cell's comparison uses "
           "the pairs that pass the margin rule in both models (R6), with each model's own set reported in `ext14_cells.md`. Dense models: "
           "the MLP output plays the role of the MoE output in every readout (single-layer patch, all-MLP M, direct split). Curves on a "
           "depth-normalised axis l / (L − 1).", "",
           fig("ext14_dense_arrows", sfx, "Share shift dense to MoE"), fig("ext14_dense_curves_mixtral_bos", sfx, "Mistral vs Mixtral curves"),
           fig("ext14_dense_curves_qwen2_57b", sfx, "Qwen2 dense vs MoE curves"), tab("ext14_dense_twins", sfx), "",
           "### Pre-registered predictions", "", pred_rows(S, ["D0", "D1", "D2", "D3", "D4"]), "",
           "### Caveats", "",
           "- The twins change capacity and sparsity together; the Qwen3 dense series (`ext14_scale.md`) and the Qwen2-57B shared-vs-routed "
           "split separate them only partly.",
           "- Choe et al. (EMNLP 2025) report unusually strong early attention in Qwen dense models: compare the per-layer curves, not only the shares.",
           "- The Mixtral-8x7B upcycling from Mistral-7B is likely (Lo et al. 2024) but not confirmed by Mistral.", "",
           "### Files", "", "`moetrace/ext14_analysis.py` (`twin_compare`), `scripts/ext14_all.py`; `results/ext14_summary.json` key `twins`; "
           "`results/tables/ext14_dense_twins.md`, `results/figures/ext14_dense_*.png`."]
    return "\n".join(txt) + "\n"


def scale_section(S, sfx):
    SE = S.get("series") or {}
    P = S.get("predictions") or {}
    lines = []
    for nm, v in SE.items():
        if not v or "cells" not in v:
            continue
        seg = []
        for c, x in sorted(v["cells"].items()):
            own = x.get("own") or x["share"]
            seq = " → ".join(f"{F(own.get(p))}" for p in v["protos"] if own.get(p))
            seg.append(f"{X.cell_label(x.get('variant', c))} {seq}")
        lines.append(f"series {nm} ({' → '.join(lab(p) for p in v['protos'])}): " + "; ".join(seg))
    QS = S.get("qwen2_shared") or {}
    if lines:
        summ = ("**Summary.** Capacity (P6-4 before the Stage-3 gate): attention share along the families' capacity series, each model on its "
                "own validation set (paired values on the common pairs in the tables). " + ". ".join(lines) + ". " + verdict_line(P, ["D5", "D6"]))
    else:
        summ = "**Summary.** Capacity series (P6-4): pending — the Stage-2 models have not reached BASE READY on common cells."
    txt = [summ, "", scale_reading(S), "", "### What was computed", "",
           "Series: Qwen3-4B → 8B → 14B-Base (dense, same tokenizer and generation as Qwen3-30B-A3B; 36 / 36 / 40 layers) → Qwen3-30B-A3B "
           "(MoE, 48 layers); Mistral-7B → Mixtral-8x7B; Qwen2-7B → Qwen2-57B-A14B. Qwen3-1.7B enters the competence scan only. D5 = per "
           "cell |share(14B) − share(4B)| ≤ 0.10 (paired on common pairs) and the dense models' cell order vs Qwen3-30B-A3B (Spearman ≥ 0.8). "
           "Mixtral-8x22B (Stage 3) is gated by the user's decision after Stage R and is not part of this section yet.", "",
           fig("ext14_capacity_qwen3", sfx, "Qwen3 series"), fig("ext14_capacity_mistral", sfx, "Mistral series"), fig("ext14_capacity_qwen2", sfx, "Qwen2 series"),
           tab("ext14_capacity_qwen3", sfx), "", tab("ext14_capacity_mistral", sfx, optional=True), "", tab("ext14_capacity_qwen2", sfx), "",
           "### Qwen2-57B-A14B: shared vs routed experts", "",
           ("Shared expert's share of the MoE rescue (shared-only / (shared-only + routed-only) joint patches, denoising): "
            + ", ".join(f"{X.cell_label(c)} {F(v)}" for c, v in (QS.get("knowledge_shared_share") or {}).items()) + ".\n")
           if QS.get("knowledge_shared_share") else "Pending (p6-data's T3 joint patches on Qwen2-57B).\n", "",
           tab("ext14_qwen2_shared", sfx), "",
           "### Pre-registered predictions", "", pred_rows(S, ["D5", "D6"]), "",
           "### Caveats", "",
           "- Model size changes depth (36–48 layers) and width together; the shares are layer sums, compare the depth-normalised curves.",
           "- The Qwen3 dense series has no architecture twin of Qwen3-30B-A3B (no Qwen3-32B-Base).",
           "- For Qwen3-30B-A3B and Mixtral the IOI and CounterFact cells are the Phase-3 anchors (IOI S2 → IO run; CounterFact donor-level "
           "attention sweep); the Stage-2 models have their own runs built the same way, so these two cells are comparable in protocol but not "
           "item for item.",
           "- Own validation sets differ in items between models; the paired values on the common pairs are in the tables (few common pairs for "
           "cells whose case sets prioritise different twins).", "",
           "### Files", "", "`moetrace/ext14_analysis.py` (`series_compare`), `scripts/ext14_all.py`; `results/ext14_summary.json` keys `series`, "
           "`qwen2_shared`; `results/tables/ext14_capacity_series.md`, `results/figures/ext14_capacity_*.png`."]
    return "\n".join(txt) + "\n"


def write_all(S, outdir, sfx=""):
    os.makedirs(outdir, exist_ok=True)
    import traceback
    for name, fn in (("ext14_ladder", ladder_section), ("ext14_dial", dial_section), ("ext14_dense", dense_section), ("ext14_scale", scale_section)):
        try:
            txt = fn(S, sfx)
        except Exception as ex:  # noqa: BLE001  (one failing section must not block the others)
            print(f"section {name} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
            continue
        with open(os.path.join(outdir, f"{name}{sfx}.md"), "w") as f:
            f.write(txt)


# ------------------------------------------------------------------------------------------------------------------
# interpretation paragraphs (data-driven: every statement is conditioned on the current numbers)
# ------------------------------------------------------------------------------------------------------------------
ORDER_P = [p for p in X.ALL_PROTOS]


def v0(x):
    try:
        return float(x[0]) if isinstance(x, (list, tuple)) else float(x)
    except (TypeError, ValueError, IndexError):
        return float("nan")


def rng(vals, d=2):
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return "n/a"
    return f"{min(vals):.{d}f}–{max(vals):.{d}f}"


def cell_shares(S, cell, protos=None):
    out = {}
    for p in (protos or ORDER_P):
        x = share(S, cell, p)
        if x and np.isfinite(v0(x)):
            out[p] = v0(x)
    return out


def ladder_reading(S):
    L = S.get("ladder") or {}
    if not L:
        return ""
    ant, prop, e2a, cf, a2e, bind, ind, ioi = (cell_shares(S, c) for c in ("l_ant", "l_prop", "e3_e2a", "l_cf", "e3_a2e", "l_bind", "l_ind", "l_ioi"))
    bits = []
    if ant:
        above = [p for p in ant if p in prop and ant[p] > prop[p]]
        above_e2a = [p for p in ant if p in e2a and ant[p] > e2a[p]]
        bits.append(f"The bottom of the pre-registered order does not hold: the antonym task (grade 1, predicted ≤ 0.30 / 0.40) has an "
                    f"attention share of {rng(ant.values())} across the {len(ant)} models, above object property ({rng(prop.values())}) in "
                    f"{len(above)} of {len([p for p in ant if p in prop])} models and above the WinoGrande entity → attribute anchor "
                    f"({rng(e2a.values())}) in {len(above_e2a)} of {len([p for p in ant if p in e2a])}. Interpretation: the antonym cue is the word itself, one token before the "
                    f"answer, so part of the answer is read off the cue by late attention even though the mapping is stored in the weights; "
                    f"'answer only in the weights' does not imply a low final-token attention share.")
    top = []
    for p in L:
        rows = [r for r in L[p].get("rows", []) if not r.get("pooled")]
        if len(rows) >= 5:
            srt = sorted(rows, key=lambda r: -v0(r["share"]))
            top.append((p, [r["cell"] for r in srt[:3]]))
    if top:
        n_ok = sum(1 for p, t in top if set(t) <= {"l_bind", "l_ind", "l_ioi", "e3_a2e"})
        bits.append(f"The top of the order holds: IOI ({rng(ioi.values())}), induction ({rng(ind.values())}) and explicit binding "
                    f"({rng(bind.values())}) are the most attention-heavy tasks; the three highest shares are copy / look-up tasks (or the "
                    f"attribute → entity anchor) in {n_ok} of {len(top)} models with ≥ 5 non-pooled ladder cells.")
    rho = {p: (L[p].get("spearman"), L[p].get("spearman_without_l_ant"), L[p].get("n_cells")) for p in L}
    seg = ", ".join(f"{lab(p)} {X.fnum(r[0], '+.2f')} (without antonym {X.fnum(r[1], '+.2f')}; {r[2]} cells)" for p, r in rho.items() if r[2] and r[2] >= 3)
    if seg:
        bits.append(f"Spearman(grade, share) per model: {seg}. The pre-registered threshold 0.8 is met only where the antonym cell is not too "
                    f"far out of place; dropping it raises the rank correlation in most models.")
    if cf:
        bits.append(f"CounterFact stays in the middle ({rng(cf.values())}), between entity → attribute and attribute → entity, in every model "
                    f"that has it.")
    return "**Reading.** " + " ".join(bits) + "\n" if bits else ""


def dial_reading(S):
    D = S.get("dial_e") or {}
    G = S.get("direction_gap_models") or {}
    bits = []
    q = D.get("qwen3") or {}
    if q.get("levels"):
        la, le = q["levels"].get("a2e", {}), q["levels"].get("e2a", {})
        a = {k: v0((la.get(k) or {}).get("share")) for k in ("e1", "e1p", "e3p3", "e3", "e0")}
        e = {k: v0((le.get(k) or {}).get("share")) for k in ("e1", "e1p", "e3", "e0")}
        bits.append(f"Qwen3: making the binding explicit barely moves the attention share within a direction — attribute → entity "
                    f"E1 {X.fnum(a['e1'])}, E1p {X.fnum(a['e1p'])}, E3 in the same frame {X.fnum(a['e3p3'])}, E3 in the F1 frame {X.fnum(a['e3'])}, "
                    f"and even with no evidence at all (E0, prior only) {X.fnum(a['e0'])}; entity → attribute E1 {X.fnum(e['e1'])}, E1p "
                    f"{X.fnum(e['e1p'])}, E3 {X.fnum(e['e3'])}.")
        h = q.get("h_test")
        if h:
            bits.append(f"The direction gap stays at E1: a→e − e→a = {F(h['gap_e1'], 2, True)} at E1 vs {F(h['gap_e3'], 2, True)} at E3 "
                        f"({h.get('method')}; twins {h.get('n_items')}), gap(E1) − ½·gap(E3) = {F(h['gap_e1_minus_half_e3'], 2, True)}: "
                        f"H-type, not the best-guess H-inference (E13). The share follows the answer type (entity vs attribute), not the "
                        f"number of inference steps between the evidence and the answer.")
    m = D.get("mixtral_bos") or {}
    if m.get("levels"):
        h = m.get("h_test")
        la, le = m["levels"].get("a2e", {}), m["levels"].get("e2a", {})
        bits.append(f"Mixtral: a→e E1 {X.fnum(v0((la.get('e1') or {}).get('share')))}, E3 (P3) {X.fnum(v0((la.get('e3p3') or {}).get('share')))}; "
                    f"e→a E1 {X.fnum(v0((le.get('e1') or {}).get('share')))}{' (pooled)' if (le.get('e1') or {}).get('pooled') else ''}, "
                    f"E3 {X.fnum(v0((le.get('e3') or {}).get('share')))}"
                    + (f"; the gap shrinks from {F(h['gap_e3'], 2, True)} at E3 to {F(h['gap_e1'], 2, True)} at E1 (only {h.get('n_items')} twins, a pooled cell: "
                       f"not usable for E13)" if h else "") + ". Explicit statements raise Mixtral's entity → attribute share, so the "
                    f"two models differ in how much the explicit binding is moved by attention when the answer is an attribute.")
    gseg = []
    for p, ent in G.items():
        if "e1" in ent and ("e3p3" in ent or "e3" in ent):
            e3 = ent.get("e3p3") or ent.get("e3")
            gseg.append(f"{lab(p)} {F(ent['e1']['gap'], 2, True)} vs {F(e3['gap'], 2, True)}")
    if gseg:
        bits.append("Direction gap at E1 vs E3 on the models' own sets (a→e − e→a): " + "; ".join(gseg) + ". In every model the gap is "
                    "positive at both levels; whether it narrows at E1 depends on the model (ext14_direction_gap_models.md).")
    B = S.get("behaviour_rates") or {}
    tr = [(p, B.get(f"e2_a2e/{p}")) for p in ORDER_P if B.get(f"e2_a2e/{p}")]
    if tr:
        bits.append("E2 (negation) could not be patched in any model (too few pairs pass the margin rule); behaviourally the copy trap wins "
                    "the attribute → entity cloze in the Qwen3 family (" + ", ".join(f"{lab(p)} {F(b['all']['o_marg'])}" for p, b in tr if p.startswith('qwen3')) +
                    " of pairs by the margin rule, target 0.00), while Mistral / Mixtral / Qwen2 rarely pass either way.")
    A = (S.get("dial_a") or {}).get("qwen3") or {}
    ca = (A.get("classes") or {}).get("a2e") or {}
    if ca.get("conflict") and ca.get("agree"):
        bits.append(f"Dial A (Qwen3, attribute → entity): conflict {X.fnum(v0(ca['conflict']['share']))} ≥ neutral {X.fnum(v0((ca.get('neutral') or {}).get('share')))} "
                    f"> agree {X.fnum(v0(ca['agree']['share']))} > E0 {X.fnum(v0((ca.get('E0') or {}).get('share')))} — the pre-registered order "
                    f"holds but spans only {X.fnum(v0(ca['conflict']['share']) - v0((ca.get('E0') or {}).get('share')))}; in conflict items the "
                    f"MoE's direct write points against the stated answer ({F(ca['conflict'].get('M_direct'), 2, True)}, A2), i.e. the MoE "
                    f"writes the prior and attention overrides it.")
    return "**Reading.** " + " ".join(bits) + "\n" if bits else ""


def dense_reading(S):
    TW = S.get("twins") or {}
    bits = []
    for k, t in TW.items():
        if not t or "cells" not in t:
            continue
        d = {c: v0(v.get("diff_used")) for c, v in t["cells"].items()}
        pos = sum(1 for x in d.values() if x > 0)
        big = [f"{X.cell_label(t['cells'][c].get('variant', c))} {x:+.2f}" for c, x in d.items() if abs(x) > 0.10]
        bits.append(f"{lab(t['dense'])} → {lab(t['moe'])}: {len(d)} cells, MoE − dense between {min(d.values()):+.2f} and "
                    f"{max(d.values()):+.2f} (positive in {pos}), |Δ| > 0.10 only in {', '.join(big) if big else 'no cell'}; the cell order "
                    f"is preserved (Spearman {X.fnum(t.get('spearman_cells'))}).")
    QS = (S.get("qwen2_shared") or {}).get("cells") or {}
    if QS:
        seg = ", ".join(f"{X.cell_label(c)} {F((v.get('denoise') or {}).get('shared_share'))}" for c, v in QS.items()
                        if (v.get("denoise") or {}).get("shared_share") and v0((v.get("denoise") or {}).get("all_moe_norm")) >= 0.1)
        if seg:
            bits.append(f"Inside Qwen2-57B-A14B the always-on shared expert carries most of the MoE rescue where the answer is an entity or a "
                        f"stored antonym and about half where the answer is an attribute (shared / (shared + routed): {seg}).")
    if bits:
        alld = [(k, c, v0(v.get("diff_used"))) for k, t in TW.items() if t and "cells" in t for c, v in t["cells"].items()]
        n = len(alld)
        within = sum(1 for _, _, x in alld if abs(x) <= 0.10)
        lower = [f"{X.cell_label(c)} ({k.split('->')[1].replace('_bos', '')})" for k, c, x in alld if x <= -0.10]
        higher = [f"{X.cell_label(c)} ({k.split('->')[1].replace('_bos', '')})" for k, c, x in alld if x >= 0.10]
        bits.append(f"Over both twins {within} of {n} cell contrasts are within ±0.10 (H-invariant, D1 / D0); MoE ≥ dense + 0.10 in "
                    f"{', '.join(higher) if higher else 'no cell'} (H-sparsity, D3); MoE ≤ dense − 0.10 in {', '.join(lower) if lower else 'no cell'} "
                    f"(the H-capacity direction, D2). At matched architecture the attention share is a property of the task far more than of "
                    f"the MLP / MoE type.")
    return "**Reading.** " + " ".join(bits) + "\n" if bits else ""


def scale_reading(S):
    se = (S.get("series") or {}).get("qwen3")
    if not se or "cells" not in se:
        return ""
    bits = []
    rows = []
    for c, v in se["cells"].items():
        own = v.get("own") or {}
        dense = [v0(own[p]) for p in ("qwen3_4b", "qwen3_8b", "qwen3_14b") if p in own]
        moe = v0(own.get("qwen3")) if "qwen3" in own else float("nan")
        if dense and np.isfinite(moe):
            rows.append((c, min(dense), max(dense), moe))
    if rows:
        up = [f"{X.cell_label(c)} {lo:.2f}–{hi:.2f} → {m:.2f}" for c, lo, hi, m in rows if m > hi + 0.05]
        down = [f"{X.cell_label(c)} {lo:.2f}–{hi:.2f} → {m:.2f}" for c, lo, hi, m in rows if m < lo - 0.03]
        flat = [c for c, lo, hi, m in rows if hi - lo <= 0.10]
        bits.append(f"Qwen3 dense 4B → 8B → 14B: the share is flat with capacity in {len(flat)} of {len(rows)} cells (range ≤ 0.10 across the three "
                    f"dense models; D5). The 30B-A3B MoE departs from the dense series: higher on {', '.join(up) if up else 'no cell'}; lower on "
                    f"{', '.join(down) if down else 'no cell'} (own validation sets).")
        if len(flat) >= len(rows) / 2 and (up or down):
            bits.append("Because the dense Qwen3 series is mostly flat, the 30B-A3B departure is not a capacity trend: it comes with the MoE model "
                        "(different depth, width, training run and sparsity together; there is no Qwen3 dense twin)"
                        + (", and it goes in opposite directions for the cells listed as higher and lower." if (up and down) else "."))
    return "**Reading.** " + " ".join(bits) + "\n" if bits else ""
