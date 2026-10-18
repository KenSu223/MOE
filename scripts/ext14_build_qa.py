"""ext14 Phase 6 build QA (CPU, agent p6-data): completeness audit table and yields per level, from the build funnels.

Writes data/dial/qa/completeness.md (every check, every cell x tokenizer family; an item failing any check is dropped before the
scan; a cell is void if its checks fail systematically) and data/dial/qa/yields.md (blank-clause yield recount on the 749 dir_f1
twins and the 776 core twins; items / pairs per level and family; reasons). Both are included in data/dial/build_report.md by
scripts/ext14_casesets.py.
Usage: python scripts/ext14_build_qa.py
"""
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace import ext14_data as X
from moetrace.ext7_wino import load_twins, split_final_word

QA = f"{X.DIAL}/qa"
STRICT = re.compile(r"(?:^|\s)the _ (was|is) (too )?[A-Za-z\-]+$")


def recount():
    """Yield of the blank-clause forms on the 776 core twins and on the 749 with a Phase-5 F1 pair (Qwen3)."""
    twins, _ = load_twins("train_xl")
    byid = {t.pair_id: t for t in twins}
    core = X.core_776()
    f1 = set(pd.read_parquet(f"{X.ROOT}/data/wino_ref/pairs_p5_dir_f1_qwen3.parquet", columns=["pair_id"]).pair_id)
    out = {}
    for name, ids in (("core 776", sorted(core)), ("749 dir_f1 twins", sorted(f1))):
        n = len(ids)
        strict = bc = 0
        for k in ids:
            t = byid[k]
            ha, wa = split_final_word(t.sent_a)
            s = re.sub(r"\s+", " ", ha) + " " + wa
            strict += bool(STRICT.search(s))
            it, _ = X.parse_twin(t, 0, True)
            bc += it is not None
        out[name] = {"n": n, "strict_the_was_too_X": strict, "copular_blank_clause_BC_RE": bc}
    return out


NOTES = """## Design notes and caveats

- **Item pool (Dial E, Set C, R-site).** WinoGrande train_xl twins in Phase-3 geometry (the two sentences differ only in their final
  word) with a copular blank clause `(det) _ (was|is|were|are) [degree adverbs] TRIGGER`; core = the 776 Phase-3 shared twins;
  extension twins are scanned only for cells with < 256 core margin pairs. One discovery / validation assignment per twin
  (data/dial/twin_split.json) for every cell; cap 128 pairs per split filled in priority order (core, number of Dial-E cells the twin
  passes in the model, built in both tokenizer families, twin rank), so that the paired intersections across levels are large.
- **Attribute -> entity frame P3** (FRAME PILOT, data/dial/qa/frame_pilot.md; coordinator rule, confirmed 2026-10-16): the E1-family
  a2e cells (e1, e1p, e2, rsite, e0, Dial A) use `{statements}. Fill in the blank: The _ was X.\\nAnswer: the` (P0 ', so the _' was at
  chance on objects in both models). New cell e3p3_a2e = E3 content in the same frame (`{event clause}. Fill in the blank: The _ was X.`;
  event clause = the sentence before the blank clause minus the connective; audit in data/dial/qa/completeness.md), so that
  E1 -> E1p -> E2 -> E3p3 is frame-matched; e3_a2e (F1 sentence) stays as the Phase-5 anchor. Set C uses the P3 layout
  (`{statement}. Fill in the blank: <original sentence>`). e2a cells keep the Phase-3 option-swap frame. E0 in P3 reads
  `Of the X and the Y. Fill in the blank: The _ was X.` (a fragment; frame-matched prior-only control).
- **PP blanks excluded** from the whole pool (D5_blank_in_pp: the word before the blank NP is a preposition, e.g. 'the sauce on the _ was',
  'the job of _ was'): the blank is not the subject of the blank clause. e3p3 additionally excludes complement clauses ('I thought the _
  was') and items where anything other than connective words / punctuation precedes the determiner + blank.
- **Association-neutral main sets** (coordinator 2026-10-16): WinoGrande cells keep only items whose two triggers have |probe| < 1
  under the model's responding Phase-5 probes (Qwen3 v1 and v2, Mixtral v2). Strong-prior items are used only in Dial A / E0.
- **Small cells (coordinator rule 2026-10-16).** < 64 validation pairs but >= 64 pairs in discovery + validation -> `pooled`
  (registry flag; family 'main' with discovery = validation = all passing pairs, so layers / heads for T5 are chosen on the same items);
  < 64 pooled -> behaviour only (`insufficient`, no tier 1), with the behaviour record (items whose clean prompts follow the target
  answer in every order vs the competing answer: the prior for acon, the event / statement for Set C, the copied word for E2).
- **All-strata E1-family cells (`<cell>_all`; coordinator decision 2026-10-17 03:15Z).** For the dense vs MoE and capacity
  comparisons only: e1_a2e / e1_e2a / e1p_* of the Stage-2 models whose association-neutral set is insufficient run on all association
  strata (e.g. Mistral-7B e1_a2e: 47 items pass both orders, 19 of them neutral); matching Stage-1 cells e1_a2e_all / e1_e2a_all for
  mixtral_bos (twin of Mistral-7B) and qwen3 (anchor of the Qwen3 dense series), minimal tier 1. Items that also pass in the twin /
  series (Stage 1) or that the counterpart selected (Stage 2) go first, so the comparisons are on identical items where both pass
  (Mixtral-Mistral: 27 shared items in e1_a2e and e1_e2a). Each pair carries the association class by the model's own probes
  (assoc_own, stratum_neutral_own) and by the counterpart's probes (assoc_counterpart, stratum_neutral_counterpart). The neutral cells
  stay primary for the pre-registered dial predictions.
- **Mixtral Dial A / E0 are behaviour only (why).** (1) 135 of the 378 controlled noun-pair items have a multi-token noun in the 32k
  vocabulary (elephant, whale, feather, ...), leaving 243 (batch 2 was added for this); (2) Mixtral expresses weak and often incoherent
  lexical priors for them: under the responding probe v2 only 38 % of the items are coherent (big -> elephant AND small -> mouse), the
  median |v2| is 0.4-0.6, and 12 items reach strength >= 1 on both triggers (Qwen3: 165 of 328); the in-frame prior (E0 margin, no
  evidence) agrees: 20 items with E0 strength >= 1, 8 pass E0 in both orders (Qwen3: 227 / 179); (3) WinoGrande twins add < 20 items
  (adversarially filtered against such priors). A lower threshold (strength >= 0.5: 33 controlled items) would stop being a strong-
  prior set. So Mixtral Dial A / E0 stay behaviour only; the Qwen3 cells carry Dial A.
- **Dedup.** E3 / Set C: Phase-3 placeholder key (same sentence template with other options = duplicate). E0-E2 / R-site / Dial A:
  exact normalised text (the event is gone, so the same trigger pair with other entities is a different item; `trig_key` keeps the
  trigger pair for clustering).
- **E1p paraphrases**: hand-checked list (data/dial/paraphrase_hand.json, 372 triggers, 34 without an acceptable paraphrase);
  WordNet candidates were often wrong in a 40-item look (rusty -> chromatic, retiring -> past), so they are not used. The paraphrase
  must differ from both triggers and share no 4-letter stem with either.
- **E2 caveat (coordinator QA).** With gradable adjectives and degree adverbs, negation does not entail the antonym ("the bowl is not
  too big" does not entail "too small"; "not very young" does not entail "very old"). E2-a2e still works by elimination ("the pot was
  not empty" excludes the pot); E2-e2a relies on pragmatic contrast only. Per pair: `degree_adv` (a degree adverb in the blank
  clause) and `antonym_type` (complementary for the hand list moetrace/ext14_data.COMPLEMENTARY: open / closed, alive / dead,
  present / absent, ...; full / empty and every other pair count as gradable); both are strata in the registry, so E2 can be reported
  overall and on the no-degree-adverb / complementary strata.
- **R-site.** Corrupted prompt = E1-a2e with the two entity mentions in the evidence swapped (two differing tokens, a documented
  exception to R2), same question trigger; the clean prompt is E1-a2e's prompt A. (The plan's alternative, swapping the two
  attributes, states the same binding; the brief's entity swap was built.)
- **Set C.** The statement is placed before the whole original sentence; Delta is signed toward the statement's answer.
  sc_conflict is split into sc_conflict_stmt_a2e (clean prefers the statement, margin both ways) and sc_conflict_event_a2e
  (prefers the event; trig_a / trig_b flipped so that the clean answer is 'true'); both orders of an item must fall in the same subset.
- **Ladder.** l_ant: LRE adjective_antonym (100 samples) + curated antonyms (data/dial/curated_antonyms.json); 3 templates ('The
  opposite of X is', 'The antonym of X is', 'Q: What is the opposite of X?\\nA:'); split by word pair. l_prop: LRE commonsense
  relations with single-word subjects (fruit colours, substance phase, object superclass, work location) + curated object colour /
  taste / habitat / material (data/dial/curated_props.json); split by subject halves per relation. l_bind / l_ind: synthetic, word
  lists restricted to words that are single tokens in both families (identical texts in both); 400 items x 2 orders.
- **Dial A / E0.** Controlled noun pairs with a designed prior (data/dial/curated_dialA_nouns.json, batch 1 + batch 2 with nouns
  that are single tokens in both families) plus WinoGrande twins with a coherent prior (strength >= 1); the agree / conflict label and
  the E0 answer orientation come from each model's own probe values.
"""


def report():
    """data/dial/build_report.md: yields, completeness, notes, frame pilot, per-cell registry summary."""
    reg_p = f"{X.DIAL}/p6_conditions.json"
    parts = [open(f"{QA}/yields.md").read() if os.path.exists(f"{QA}/yields.md") else "", NOTES]
    if os.path.exists(f"{QA}/frame_pilot.md"):
        parts.append(open(f"{QA}/frame_pilot.md").read().replace("# FRAME PILOT", "## FRAME PILOT"))
    if os.path.exists(reg_p):
        reg = json.load(open(reg_p))["conditions"]
        rows = ["## Cells (registry data/dial/p6_conditions.json)", "",
                "| Cell | Model | Pool items | Items passing (all strata) | Items passing (main set) | Order pass rates | Top-1 (selected) | Disc / val pairs | "
                "Names / objects (val) | Status |", "|---|---|---|---|---|---|---|---|---|---|"]
        for cell, d in reg.items():
            for proto, e in d.items():
                if e.get("anchor"):
                    rows.append(f"| {cell} | {proto} | anchor | | | | | | | known share {e.get('known_share')}, runs {e.get('run_dir')} |")
                    continue
                cs = json.load(open(f"{X.ROOT}/{e['case_sets']}")) if os.path.exists(f"{X.ROOT}/{e['case_sets']}") else {}
                pools = cs.get("pools", {})
                val = (e.get("strata_counts") or {}).get("validation", {})
                st = ("insufficient" if e.get("insufficient") else ("BASE READY" if e["status"].get("base") else "cases ready")) + \
                     (f"; REVIEW: {e['review']}" if e.get("review") else "")
                rows.append(f"| {cell} | {proto} | {pools.get('n_pool_items')} | {pools.get('n_items_pass_all_strata', pools.get('n_items_pass'))} | "
                            f"{pools.get('n_items_pass')} | {e.get('order_pass_rates')} | {e.get('top1_rate')} | "
                            f"{e['split_sizes'].get('discovery', 0)} / {e['split_sizes'].get('validation', 0)} | {val.get('names', 0)} / {val.get('objects', 0)} | {st} |")
        parts.insert(0, status_matrix(reg))
        parts.append("\n".join(rows))
    parts.append(PROBLEMS)
    for extra in ("ind_pilot.md", "tokenizer_identity.md"):
        if os.path.exists(f"{QA}/{extra}"):
            parts.append(open(f"{QA}/{extra}").read().replace("\n# ", "\n## ").replace("# IND PILOT", "## IND PILOT").replace("# Stage-2", "## Stage-2"))
    open(f"{X.DIAL}/build_report.md", "w").write("# Phase 6 build report (p6-data)\n\n" + "\n\n".join(p for p in parts if p) + "\n")


MODELS_ORDER = ["qwen3", "mixtral_bos", "mistral7b_bos", "qwen2_7b", "qwen2_57b", "qwen3_1p7b", "qwen3_4b", "qwen3_8b", "qwen3_14b"]
CELLS_ORDER = ["e1_a2e", "e1_e2a", "e1p_a2e", "e1p_e2a", "e2_a2e", "e2_e2a", "e3p3_a2e", "e3_a2e", "e3_e2a", "e1_a2e_all", "e1_e2a_all",
               "e1p_a2e_all", "e1p_e2a_all", "e0_a2e", "e0_e2a", "aagr_a2e", "aagr_e2a", "acon_a2e", "acon_e2a", "sc_agree_a2e",
               "sc_conflict_stmt_a2e", "sc_conflict_event_a2e", "rsite_e1_a2e", "l_ant", "l_prop", "l_cf", "l_ioi", "l_bind", "l_ind"]


def code(e):
    """Short status code with n (validation pairs; pooled = all pairs used for both splits)."""
    if e.get("anchor"):
        return "anchor"
    n_val = (e.get("split_sizes") or {}).get("validation", 0)
    if e.get("tier1_trimmed"):
        return "beh (trimmed)"
    if e.get("behaviour_only_by_design"):
        return "beh (design)"
    if e.get("insufficient") or e.get("behaviour_only"):
        cs = json.load(open(f"{X.ROOT}/{e['case_sets']}")) if e.get("case_sets") and os.path.exists(f"{X.ROOT}/{e['case_sets']}") else {}
        n = len(set(cs.get("pairs", {}).get("discovery", [])) | set(cs.get("pairs", {}).get("validation", [])))
        return f"beh ({n} < 64)"
    tag = f"{'pooled ' if e.get('pooled') else ''}{n_val}"
    return ("BASE " if e["status"].get("base") else "cases ") + tag


def status_matrix(reg):
    lines = ["## Status matrix (cell x model)", "",
             "BASE n = tier 1 done (BASE READY), n validation pairs; 'pooled n' = discovery = validation = all n passing pairs (< 64 validation, >= 64 "
             "together); beh (n < 64) = behaviour only, too few pairs (scan-level behaviour recorded); beh (design) = Stage-2 Set C / Qwen3-1.7B "
             "competence only; beh (trimmed) = Qwen2-57B tier 1 trimmed to the twin-comparison cells; anchor = existing Phase-3/6 runs; '-' = "
             "not built for that model.", "", "| Cell | " + " | ".join(MODELS_ORDER) + " |", "|---" * (len(MODELS_ORDER) + 1) + "|"]
    for c in CELLS_ORDER:
        d = reg.get(c, {})
        if not d:
            continue
        lines.append(f"| {c} | " + " | ".join(code(d[p]) if p in d else "-" for p in MODELS_ORDER) + " |")
    return "\n".join(lines)


PROBLEMS = """## Problems met and how they were handled

- **Qwen3 at chance on objects in the E1 a2e cloze** (P0 ', so the _ was X'; first-mention bias ~1 logit) -> FRAME PILOT (P0-P4, 200 core
  twins, both models, pre-set rule) -> P3 '{statements}. Fill in the blank: The _ was X.' for every E1-family a2e cell (Qwen3 neutral objects
  correct-both 0.39 -> 0.99, Mixtral 0.05 -> 0.80); new cell e3p3_a2e (E3 content in P3) for a frame-matched dial; Set C moved to the P3
  layout; old-frame scans renamed VOID_oldframe_*.
- **e3p3 truncation bug** (blank NP longer than a determiner: 'because the one on. Fill in the blank: The _ ...') -> PP-blank items excluded
  from the whole pool (D5_blank_in_pp), e3p3 eligibility = only connective word(s) / punctuation before a determiner + blank, complement
  clauses excluded, string-wise reconstruction checks on 100 % of e3p3 / Set C / R-site items (0 failures).
- **E2 semantics** (gradable adjectives + degree adverbs: negation does not entail the antonym) -> strata degree_adv / antonym_type.
- **Mixtral at chance on 'Pairs: a = b, c = d. a ='** -> IND PILOT (I0-I4, pre-set rule) -> I2 (newline 'a: b' lines + two distractor
  pairs); I3 (flat sequence) behaviour only because its STR is the final token (R2).
- **l_bind / l_ind too few passing items** (Qwen3 first-mention bias) -> more items (l_bind 1,600, l_ind 800); 'orange' removed from the
  objects (also a colour).
- **Extension scans were split-unbalanced** (the twin rank is split-ordered, so 'first 1,200 extension items by rank' took discovery twins
  only) -> interleaved selection that skips scanned items (scan_<cell>_ext<k>); affected Mixtral cells withdrawn and rescanned.
- **Small cells** -> coordinator rule: pooled (< 64 validation, >= 64 together) / behaviour only (< 64 pooled), REVIEW lines only for other
  problems.
- **Mixtral Dial A / E0 tiny pools** -> diagnosed (multi-token nouns, weak / incoherent v2 priors, in-frame E0 prior agrees); behaviour only.
- **Neutral rule leaves the dense twins too few E1 items** -> `_all` cells (all association strata, own + counterpart association labels,
  twin-preferred items first) for the dense vs MoE / capacity comparisons only.
- **Scope cut** (only what the attention vs MoE / dense question needs) -> tier 1 = sweep + joint + DLA + direct; heads only for e1_a2e,
  e3_e2a and NEED HEADS cells; no new grids; BASE READY = W2 + W4.
- **Queue throughput** (~15 chains, ~25-60 min wait per job) -> one GPU job per cell (stages back-to-back, .done markers kept), bundles of
  2 (Stage 1) / 4 (dense) cells.
- **Qwen2-57B / Qwen2-7B bf16 fragility** -> fp32 protocol applied by Engine(repo) from moetrace/models.py; Qwen2-57B held until
  p6-engine's PROTOCOL line (a watcher pattern that matched a quoted mention started it once; cancelled before any GPU work; watchers now
  only count lines that start with the engine tag); Qwen2-57B scan batches capped at 50k tokens, tier-1 budgets at 0.5x (two OOMs retried
  at 0.75x automatically); its tier 1 trimmed to the 10 twin-comparison cells.
- **Missed NEED HEADS e1_e2a/mixtral_bos** (stage list fetched before the request; done cells are not revisited) -> heads run as one job
  afterwards and the base analysis re-run.
- **Three agent stops (session limits)** -> every chain nohup setsid; resumed from disk, no GPU work redone.
- **Accidental smoke-test write** into results/p5_dir_f1_qwen3/base_summary.json (19:32Z on 10-16) -> restored with git checkout; test mode
  writes to the scratch dir only.
"""


def main():
    if "--report" in sys.argv:
        report()
        return
    rc = recount()
    lines = ["# Dial-E yields (p6-data)", "",
             "Blank-clause recount (before any split). 'strict' = `the _ (was|is) (too) X` at the end of the sentence (the other session's "
             "pattern, 82 % = 615 / 749); 'BC_RE' = the builder's pattern `(det) _ (was|is|were|are) [degree adverbs] TRIGGER` "
             "(moetrace/ext14_data.py), the Dial-E item rule.", "",
             "| Twin set | Twins | strict pattern | BC_RE (item rule) |", "|---|---|---|---|"]
    for k, v in rc.items():
        lines.append(f"| {k} | {v['n']} | {v['strict_the_was_too_X']} ({v['strict_the_was_too_X'] / v['n']:.0%}) | "
                     f"{v['copular_blank_clause_BC_RE']} ({v['copular_blank_clause_BC_RE'] / v['n']:.0%}) |")
    comp = ["# Completeness audit (p6-data; Phase-5 F2-cut guard)", "",
            "Checks are string-level on 100 % of built items, before any scan (moetrace/ext14_data.completeness). E3 a2e / Set C: the whole "
            "WinoGrande sentence (with the blank) is in the prompt. E3 e2a: prompt + ' ' + trigger = the filled sentence (only punctuation "
            "after it). E1 / E1p / E2 / R-site / E0 (WinoGrande items): the whole blank clause is in the prompt (a2e: `<det> _ <cop> [adv] "
            "TRIGGER.`; e2a: the prompt ends with the filled clause up to the trigger). E1: both filled blank clauses are in the evidence. "
            "E1p: neither trigger appears in the evidence, both paraphrases do. E2: each entity is stated as `not [adv] <other trigger>`. "
            "Whitespace is normalised (some WinoGrande sentences have double spaces). Items failing any check are dropped (column "
            "'dropped').", "",
            "| Cell | Family | Items built (both orders) | Check | Checked pairs | Failed | Items dropped |", "|---|---|---|---|---|---|---|"]
    ylines = ["", "## Items and pairs per cell (after R1 / R2 / R4 dedup / completeness; before the scan)", "",
              "| Cell | Family | Items | of which core 776 | names | Pairs | Main losses (items or pairs) |", "|---|---|---|---|---|---|---|"]
    for fam in ("qwen", "mistral"):
        fp = f"{X.BUILD}/funnel_{fam}.json"
        if not os.path.exists(fp):
            continue
        f = json.load(open(fp))
        for cell, d in f.items():
            if cell.startswith("items") or not isinstance(d, dict) or "reasons" not in d:
                continue
            r = d["reasons"]
            checks = sorted({k.split(":", 1)[1] for k in r if k.startswith("check_n:")})
            for i, c in enumerate(checks):
                comp.append(f"| {cell} | {fam} | {d['items'] if i == 0 else ''} | {c} | {r.get('check_n:' + c, 0)} | {r.get('check_fail:' + c, 0)} | "
                            f"{r.get('item:incomplete_dropped', 0) if i == 0 else ''} |")
            loss = sorted(((k, v) for k, v in r.items() if not k.startswith("check") and k != "ok" and v), key=lambda x: -x[1])[:4]
            ylines.append(f"| {cell} | {fam} | {d['items']} | {d['items_core']} | {d['items_names']} | {d['pairs']} | "
                          + "; ".join(f"{k} {v}" for k, v in loss) + " |")
    # e3p3_a2e audit (coordinator 2026-10-16): event clause = sentence before the blank clause minus the connective
    comp += ["", "## e3p3_a2e (E3 content in the P3 frame): event clause audit", "",
             "Prompt `{event clause}. Fill in the blank: The _ was X.\\nAnswer: the`. Event clause = the WinoGrande sentence before the blank "
             "clause with the connective word(s) introducing it dropped (moetrace/ext14_data.e3p3_parts), final punctuation normalised to '.'. "
             "Checks: both entities in the event clause; the whole blank clause in the question (capitalised); the event clause equals the raw "
             "text before the blank clause minus exactly the dropped connective words (punctuation ignored). Excluded: blank inside a PP "
             "('the sauce on the _ was', all cells: D5_blank_in_pp), blank clause = object of a verb of saying / thinking ('I thought the _ "
             "was', 'realized that the _ was': E3P3_complement_clause), no event clause. Blank clauses are sentence-final by construction "
             "(Phase-3 geometry: the trigger is the last word).", ""]
    for fam in ("qwen", "mistral"):
        bp = f"{X.BUILD}/e3p3_a2e_{fam}.parquet"
        fp = f"{X.BUILD}/funnel_{fam}.json"
        if not os.path.exists(bp):
            continue
        b = pd.read_parquet(bp, columns=["dropped_connective", "core776"])
        r = json.load(open(fp)).get("e3p3_a2e", {}).get("reasons", {})
        comp.append(f"**{fam}**: {len(b)} items (core {int(b.core776.sum())}); excluded: " +
                    ", ".join(f"{k} {v}" for k, v in sorted(r.items()) if k.startswith(("E3P3", "item:", "check_fail"))))
        comp.append("")
        comp.append("| Dropped connective | Items |")
        comp.append("|---|---|")
        for k, v in b.dropped_connective.value_counts().items():
            comp.append(f"| {k} | {v} |")
        comp.append("")
    with open(f"{QA}/completeness.md", "w") as fo:
        fo.write("\n".join(comp) + "\n")
    with open(f"{QA}/yields.md", "w") as fo:
        fo.write("\n".join(lines + ylines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
