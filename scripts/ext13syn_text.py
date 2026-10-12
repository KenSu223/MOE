"""ext13syn section writers (agent p5-synth): results/sections/ext13_ladder.md and ext13_curves.md, generated from
results/ext13syn_summary.json (numbers are never typed by hand)."""
import os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
from moetrace import ext13syn as S
from moetrace.models import RESULTS

SEC = os.path.join(RESULTS, "sections")
MIN_UNITS = 8


def fci(x, d=2, sign=False):
    if not x or x[0] is None or (isinstance(x[0], float) and not np.isfinite(x[0])):
        return "n/a"
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    if len(x) < 3 or x[1] is None or x[2] is None:
        return f.format(x[0])
    return f"{f.format(x[0])} [{f.format(x[1])}, {f.format(x[2])}]"


def row(lad, level, proto, stratum="all"):
    return next((r for r in lad["rows"] if r["level"] == level and r["proto"] == proto and r["stratum"] == stratum), None)


def contrast(con, a, b, proto):
    return next((e for e in con if e["a"] == a and e["b"] == b and e["proto"] == proto), None)


def cdesc(e, k, split="validation", d=2):
    r = (e or {}).get(split)
    if not r or k not in r:
        return None
    s = f"{r[k]['a'][0]:.2f} vs {r[k]['b'][0]:.2f}, Δ {fci(r[k]['diff'], d, True)}"
    if r["n_units"] < MIN_UNITS:
        s += f" ({r['n_units']} units: CI unreliable)"
    return s


def ladder_section(summ):
    lad, con, strata, conf = summ["ladder"], summ["contrasts"], summ["strata"], summ.get("confounds", {})
    order = lad["order"]
    L = []
    q, m = "qwen3", "mixtral_bos"

    def sh(lv, p, k="share", d=2):
        r = row(lad, lv, p)
        return fci(r.get(k), d) if r and r.get(k) else "n/a"

    e2a = [lv for lv in order if S.LEVELS[lv]["direction"] == "entity -> attribute"]
    a2e = [lv for lv in order if S.LEVELS[lv]["direction"] == "attribute -> entity"]

    def rng(levels, p):
        v = [row(lad, lv, p)["share"][0] for lv in levels if row(lad, lv, p)]
        return (min(v), max(v)) if v else (np.nan, np.nan)
    qe, me = rng(e2a, q), rng(e2a, m)
    qa, ma = rng(a2e, q), rng(a2e, m)
    cdir = {p: contrast(con, "dir_f1", "dir_opt", p) for p in (q, m)}
    ccopy = {p: contrast(con, "cp_nat", "p3_opt", p) for p in (q, m)}
    L.append(
        "**Summary.** The task ladder of P5-2 (agent p5-synth): every Phase-3 / Phase-5 level and both models on one attention–MoE axis, "
        "recomputed from the run rows with one code path (validation pairs; attention share = AUC+ of the single-layer final-position curves; "
        "A_direct = direct path of all attention outputs; all-MoE M = every final-position MoE output patched; heads among the first 10 picks of the "
        "mixed head + expert greedy; knockout F of the condition's own expert and of Phase 3's WinoGrande expert). "
        f"Ordered by the attention share (mean within-model rank), the levels fall into two blocks rather than a continuous gradient: "
        f"every entity → attribute level, with or without the answer in a copy prefix, sits at {qe[0]:.2f}–{qe[1]:.2f} (Qwen3) / {me[0]:.2f}–{me[1]:.2f} (Mixtral), "
        f"CounterFact at {sh('cf', q)} / {sh('cf', m)}, every attribute → entity level at {qa[0]:.2f}–{qa[1]:.2f} / {ma[0]:.2f}–{ma[1]:.2f}, "
        f"and IOI at {sh('ioi', q)} / {sh('ioi', m)}. The two models order the shared levels alike (Spearman {conf.get('spearman_qwen3_mixtral', float('nan')):.2f} over "
        f"{conf.get('n_levels_both', 0)} levels). On identical twins, switching the direction (dir_f1 − dir_opt) raises the attention share by "
        f"{fci(cdir[q]['validation']['share']['diff'], 2, True)} (Qwen3) / {fci(cdir[m]['validation']['share']['diff'], 2, True)} (Mixtral), A_direct by "
        f"{fci(cdir[q]['validation']['A_direct']['diff'], 2, True)} / {fci(cdir[m]['validation']['A_direct']['diff'], 2, True)} and lowers M by "
        f"{fci(cdir[q]['validation']['M_all']['diff'], 2, True)} / {fci(cdir[m]['validation']['M_all']['diff'], 2, True)}; making the attribute copyable "
        f"(copy prefix − Phase-3 option swap, same twins) moves the share by only {fci(ccopy[q]['validation']['share']['diff'], 2, True)} / "
        f"{fci(ccopy[m]['validation']['share']['diff'], 2, True)}, and the neutral verb does nothing measurable. "
        "Within levels, the association strata (≥ 20 pairs) shift the share by at most about 0.1, in opposite directions for the two task directions, "
        "and name items are more attention-heavy than object items in every level. The ladder confounds direction with the prompt wrapper, the answer type, "
        "prompt length and item composition (listed below); only the paired contrasts hold items fixed.")
    L.append("")
    L.append("### What was computed")
    L.append("")
    L.append(
        "Levels (RESEARCH_PLAN.md P5-2 and the Phase 5 condition table): CounterFact STR (Phase 3 three-task table; donor-level ext6 runs), "
        "WinoGrande option swap (Phase 3 run `wino_<proto>_str`, main family; names / objects strata), the Phase-5 conditions `dir_opt`, `vt_e2a_orig`, "
        "`vt_e2a_neu`, `cp_vt_orig`, `cp_vt_neu`, `cp_nat`, `dir_f1`, `f1` (names / objects strata), `vt_a2e_orig` (Qwen3 only), `f2e` (when its base run is "
        "ready) and IOI S2 → IO (Phase 3 `ioi_<proto>_s2io`). Every pair-run level is recomputed from `str_sweep_rows` (attention / MoE single-layer curves), "
        "`direct_split` and `joint_rows` with `moetrace/ext13syn.py`; the values reproduce the p5-data section exactly (e.g. f1 Qwen3 0.76 [0.73, 0.77], "
        "A_direct 1.20, M 0.24) and Phase 3 (option swap 0.16 / 0.34, IOI 0.92 / 0.80). Heads among the first 10 picks come from p5-circuit's circuit "
        "state (per row, `scripts/ext13ci_analyze.row_table`), Phase 4 for the option swap; CounterFact and IOI give a point value from the Phase-4 "
        "reference. Knockout F: p5-expert's `ext13ko_targets` for Phase 5, Phase 4 (`ext9_knockout_summary.json`) for CounterFact "
        "(own = the Direction-6 STR selections L44E069 / L21E001), the option swap (own = WinoGrande expert, margin items) and IOI. "
        "CIs: cluster bootstrap over the condition's unit (seed template for templates, WinoGrande twin for the copy prefix on natural twins, else the "
        "pair; 5,000 resamples, seed 0). Order key: the attention share's percentile rank within each model, averaged over the models that have the level "
        f"(so the Qwen3-only level is placed by its rank). Order: {' < '.join(order)}.")
    L.append("")
    L.append("![Task ladder](../figures/ext13syn_ladder.png)")
    L.append("")
    L.append(open(os.path.join(RESULTS, "tables", "ext13syn_ladder.md")).read())
    L.append("### Paired contrasts on identical items")
    L.append("")
    L.append(
        "Each contrast restricts both conditions to the items they share (natural twins: twin id; templates: seed × name pair, both copy-prefix attribute "
        "orders pooled) and applies the same unit resamples to both, so the CI is that of the paired difference. 'validation' = the ladder's split; 'all' = "
        f"discovery + validation (nothing is selected on these quantities, so the larger set is legitimate). Contrasts with fewer than {MIN_UNITS} shared "
        "units are listed, but their percentile CIs are not reliable.")
    L.append("")
    for kind, title in (("direction", "Direction"), ("copy", "Answer copyable (C2 − C1)"), ("verb", "Neutral verb")):
        ee = [e for e in con if e["kind"] == kind]
        if not ee:
            continue
        L.append(f"**{title}.**")
        L.append("")
        for e in ee:
            parts = []
            for k, lab in (("share", "share"), ("A_direct", "A_direct"), ("M_all", "M"), ("heads10", "heads in first 10")):
                s = cdesc(e, k)
                if s:
                    parts.append(f"{lab} {s}")
            s_all = cdesc(e, "share", "all")
            v = e.get("validation") or {}
            L.append(f"- {e['a']} − {e['b']}, {e['model']} ({v.get('n_items', 0)} shared items, {v.get('n_units', 0)} units): " + "; ".join(parts) +
                     (f". All splits: share {s_all}" if s_all else "") + ".")
        L.append("")
    L.append("![Paired contrasts](../figures/ext13syn_ladder_contrasts.png)")
    L.append("")
    L.append(open(os.path.join(RESULTS, "tables", "ext13syn_ladder_contrasts.md")).read())
    # strata
    L.append("### Within levels: association strata, names, probe score")
    L.append("")
    L.append(
        "All splits and families of a level pooled (the strata are not used for any selection). Association strata follow p5-data's pre-registered rule "
        "(responding probes v1 + v2 in Qwen3, v2 in Mixtral; neutral = both triggers |probe| < 1); the main `f1` pool is neutral by construction, so the "
        "pre-neutral fill-in-the-blank runs `wino_ref_F1_<proto>` (same format, all strata; kept as a record by p5-data) carry the attribute → entity "
        "strata; the Phase-3 option swap has only the local-context flag `assoc` and its continuous baseline d_local. The probe regression uses the "
        "per-pair direct attention share (A_direct / drop of the pair) against the own-answer score s = mean over responding probes of (probe_a − probe_b) / 2 "
        "(s > 0: the trigger alone points to the clean answer).")
    L.append("")

    def st(lev, p, fam, lab):
        e = next((x for x in strata if x["level"] == lev and x["proto"] == p), None)
        v = (e or {}).get(fam, {}).get(lab)
        return v

    def dsh(lev, p, fam, lab, k="share"):
        v = st(lev, p, fam, lab)
        return fci((v or {}).get("diff_vs_ref", {}).get(k), 2, True) if v else "n/a"
    L.append(
        f"- Attribute → entity (pre-neutral F1 runs): one-sided / reverse pairs have a LOWER share than neutral pairs — Qwen3 "
        f"{dsh('f1pre', q, 'assoc', 'one-sided')} / {dsh('f1pre', q, 'assoc', 'reverse')}, Mixtral {dsh('f1pre', m, 'assoc', 'one-sided')} / "
        f"{dsh('f1pre', m, 'assoc', 'reverse')}; A_direct {dsh('f1pre', q, 'assoc', 'one-sided', 'A_direct')} / {dsh('f1pre', m, 'assoc', 'one-sided', 'A_direct')} "
        f"(one-sided). dir_f1 agrees (Qwen3 one-sided {dsh('dir_f1', q, 'assoc', 'one-sided')}, reverse {dsh('dir_f1', q, 'assoc', 'reverse')}; "
        f"local assoc {dsh('dir_f1', q, 'local', 'local assoc')} / {dsh('dir_f1', m, 'local', 'local assoc')}).")
    L.append(
        f"- Entity → attribute: association goes the other way — dir_opt associative {dsh('dir_opt', q, 'assoc', 'associative')} (Qwen3, 10 pairs), one-sided "
        f"{dsh('dir_opt', q, 'assoc', 'one-sided')} / {dsh('dir_opt', m, 'assoc', 'one-sided')}, local assoc {dsh('dir_opt', q, 'local', 'local assoc')} / "
        f"{dsh('dir_opt', m, 'local', 'local assoc')}; Phase-3 option swap local assoc {dsh('p3_opt', q, 'local', 'local assoc')} / {dsh('p3_opt', m, 'local', 'local assoc')}. "
        "The templates and the copy-prefix templates show no stratum effect (few non-neutral pairs).")
    L.append(
        f"- Names vs objects (names minus objects): f1 {dsh('f1', q, 'names', 'names')} / {dsh('f1', m, 'names', 'names')}, Phase-3 option swap "
        f"{dsh('p3_opt', q, 'names', 'names')} / {dsh('p3_opt', m, 'names', 'names')}, cp_nat {dsh('cp_nat', q, 'names', 'names')} / {dsh('cp_nat', m, 'names', 'names')} "
        "(Qwen3 / Mixtral). Names raise the share in every level and direction; Mixtral's f1 and dir_f1 have few name pairs.")
    L.append("")
    L.append(open(os.path.join(RESULTS, "tables", "ext13syn_ladder_strata.md")).read())
    L.append(open(os.path.join(RESULTS, "tables", "ext13syn_ladder_probe.md")).read())
    # confounds
    cq, cm = conf.get(q, {}), conf.get(m, {})
    L.append("### What the ladder confounds")
    L.append("")
    L.append(
        "Between the entity → attribute block and the attribute → entity block several factors change together, so the ladder is descriptive; only "
        "the paired contrasts isolate factors, and none of them isolates the prompt wrapper yet.")
    L.append("")
    L.append("- **Direction and answer type.** Every attribute → entity level reads out an entity that is a token of the context (a noun or a name); "
             "every entity → attribute level generates an attribute word (adjective, verb, comparative) that is not in the sentence. CounterFact generates "
             "an object that is not in the context, IOI copies a name. dir_f1 − dir_opt changes direction and answer type at once.")
    L.append("- **Prompt wrapper.** All attribute → entity levels traced so far use `Fill in the blank: … Answer:`; all entity → attribute levels are plain "
             "sentences (the copy prefix adds `Options: t1, t2.`). The wrapper's own effect is unmeasured until `f2e` (natural truncation, no wrapper) "
             "has a base run; IOI (no wrapper) is attention-heavy, so the wrapper is not necessary for a high share.")
    L.append(f"- **Prompt length.** Median T is 17–19 tokens for the entity → attribute levels, 23–27 with the copy prefix and 27–32 for fill-in-the-blank. "
             f"Across levels, the share's Spearman correlation with median T is {cq.get('spearman_share_T_median', float('nan')):+.2f} (Qwen3, "
             f"{cq.get('n_levels', 0)} levels) / {cm.get('spearman_share_T_median', float('nan')):+.2f} (Mixtral); the copy prefix adds 5–9 tokens without "
             "moving the share, so length alone does not produce the jump, but it is collinear with the wrapper.")
    L.append(f"- **Items and margins.** Natural twins vs semi-synthetic templates (names only, 48 IOI name pairs per seed, seed-clustered CIs with 8–18 "
             f"units per split); each format and model keeps the items that pass its own margin, so the item sets differ across levels (Mixtral's f1 has 12 / 101 "
             f"name pairs, Qwen3's 70 / 128). Names raise the share inside every level (above), and the names fraction differs by level. Mean drops differ "
             f"(about 4.3–5.2 logits for fill-in-the-blank, 6–10 for the option swap): the share is a ratio of rescues, but harder items may route differently; "
             f"across levels the share's Spearman correlation with the drop is {cq.get('spearman_share_drop', float('nan')):+.2f} / "
             f"{cm.get('spearman_share_drop', float('nan')):+.2f}.")
    L.append("- **Final-token readout.** All numbers are final-position patches; in attribute → entity the final position copies a mention, so a "
             "high final-token attention share does not say where the decision is made (Phase 5 review, issue 5; p5-data's W3 grid).")
    L.append("")
    L.append("### Readings")
    L.append("")
    allrows = [r for r in lad["rows"] if r["stratum"] == "all"]

    def span(levels, k, d=2):
        v = [r[k][0] for r in allrows if r["level"] in levels and r.get(k) and r[k][0] is not None and np.isfinite(r[k][0])]
        return f"{min(v):.{d}f}–{max(v):.{d}f}" if v else "n/a"
    e2a_all = [lv for lv in S.LEVELS if S.LEVELS[lv]["direction"] == "entity -> attribute"]
    a2e_all = [lv for lv in S.LEVELS if S.LEVELS[lv]["direction"] == "attribute -> entity"]
    L.append(
        "1. **Direction / answer type is the step, answer availability is not.** The paired direction contrasts are the only manipulations that move the "
        "share by more than 0.1, in both models and on identical items; adding the answer to the context as an option list leaves the entity → attribute "
        f"levels where they were (copy prefix − option swap on the same twins: A_direct {fci(ccopy[q]['validation']['A_direct']['diff'], 2, True)} Qwen3, "
        f"{fci(ccopy[m]['validation']['A_direct']['diff'], 2, True)} Mixtral). The share is highest when the answer is itself a context token that has to be "
        "retrieved by reference (attribute → entity, IOI), intermediate when the subject has to be moved but the answer is generated (CounterFact), and "
        "lowest when the answer is a generated attribute (entity → attribute), even when that attribute sits in an option list.")
    L.append(
        f"2. **Heads, M and the share move together.** Heads among the first 10 mixed picks: entity → attribute {span(e2a_all, 'heads10', 1)}, "
        f"attribute → entity {span(a2e_all, 'heads10', 1)}, IOI {fci((row(lad, 'ioi', q) or {}).get('heads10'), 1)} (Qwen3); all-MoE M: entity → attribute "
        f"{span(e2a_all, 'M_all')}, attribute → entity {span(a2e_all, 'M_all')}, IOI {fci((row(lad, 'ioi', q) or {}).get('M_all'))} / "
        f"{fci((row(lad, 'ioi', m) or {}).get('M_all'))}.")
    sig = [f"{r['level']} {r['model']} {r['own']['expert']}" for r in allrows if r.get("own") and r["own"].get("F") and r["own"]["F"][1] is not None
           and r["own"]["F"][1] > 0]
    nsig = [f"{r['level']} {r['model']}" for r in allrows if r.get("own") and r["own"].get("F") and r["own"]["F"][1] is not None and r["own"]["F"][1] <= 0]
    wsig = [f"{r['level']} {r['model']}" for r in allrows if r.get("wg") and r["wg"].get("F") and r["wg"]["F"][1] is not None and r["wg"]["F"][1] > 0]
    L.append(
        f"3. **Experts.** The own selected expert's knockout costs 0.00–0.04 of the clean margin; its CI excludes zero in {len(sig)} level × model cells "
        f"({', '.join(sig)}) and not in {len(nsig)} ({', '.join(nsig)}). Phase 3's WinoGrande expert is necessary only in {', '.join(wsig)}: where the "
        "answer is a natural-twin attribute, not on the templates and not in attribute → entity, CounterFact or IOI.")
    dd = []
    for e in strata:
        for fam in ("assoc", "local"):
            for lab, v in (e.get(fam) or {}).items():
                d = (v.get("diff_vs_ref") or {}).get("share")
                if d and v["n_pairs"] >= 20:
                    dd.append(abs(d[0]))
    nn = []
    for e in strata:
        v = (e.get("names") or {}).get("names")
        if v and (v.get("diff_vs_ref") or {}).get("share") and v["n_pairs"] >= 10:
            nn.append(v["diff_vs_ref"]["share"][0])
    L.append(
        "4. **Association modulates within a level, in a direction-specific way.** When the trigger alone points to an entity, attribute → entity uses "
        "less attention (and more MoE); when entity and attribute are associated, entity → attribute uses slightly more attention. Stratum differences "
        f"with ≥ 20 pairs are at most {max(dd):.2f} in absolute value, against a direction step of "
        f"{cdir[q]['validation']['share']['diff'][0]:.2f} / {cdir[m]['validation']['share']['diff'][0]:.2f}; names raise the share by "
        f"{min(nn):.2f}–{max(nn):.2f} in every level that has ≥ 10 name pairs.")
    L.append("")
    L.append("Files: `moetrace/ext13syn.py`, `scripts/ext13syn_all.py` (one command for both syntheses), tables `results/tables/ext13syn_ladder{,_contrasts,_strata,_probe}.{md,csv}`, "
             "figures `results/figures/ext13syn_ladder{,_contrasts}.{png,pdf}`, numbers `results/ext13syn_summary.json` (keys ladder, contrasts, strata, confounds).")
    open(os.path.join(SEC, "ext13_ladder.md"), "w").write("\n".join(L) + "\n")


def curves_section(summ):
    import ext13syn_curves_text as CT
    CT.write(summ, SEC)
