"""ext13 Phase-5 base section (CPU, agent p5-data): results/sections/ext13_base.md from results/ext13base_summary.json,
results/ext13base_behaviour.json, results/wino_ref_<proto>/p5_probe_outcome.json, data/wino_ref/p5_conditions.json and the
ext13base_* tables. Conditions without a finished base analysis are listed as pending. Per-condition paragraphs, the
headline list and the contrasts are generated from the summaries (numbers-driven, so every condition appears as soon as its
base analysis exists; the chain re-runs this script after every analysis). Static text (setup, probe, behaviour, deviations,
summary intro / findings) comes from scripts/ext13_p5_notes.json.
Usage: python scripts/ext13_p5_text.py
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import pandas as pd
from moetrace.models import RESULTS

ROOT = "/home/ubuntu/MOE"
TAB = os.path.join(RESULTS, "tables")
OUT = os.path.join(RESULTS, "sections", "ext13_base.md")
CONDS = ["f1", "dir_f1", "dir_opt", "cp_nat", "cp_vt_orig", "cp_vt_neu", "vt_e2a_orig", "vt_e2a_neu", "vt_a2e_orig", "vt_a2e_neu", "f2e"]
PROTOS = ("qwen3", "mixtral_bos")
LABEL = {"qwen3": "Qwen3", "mixtral_bos": "Mixtral BOS"}
NOTES_PATH = os.path.join(ROOT, "scripts", "ext13_p5_notes.json")


def tab(name):
    p = os.path.join(TAB, name + ".md")
    return open(p).read() if os.path.exists(p) else f"*(table {name} pending)*\n"


def fr(x, d=2):
    return f"{x[0]:.{d}f} [{x[1]:.{d}f}, {x[2]:.{d}f}]" if x else "n/a"


def summary_table(S, reg):
    head = ["Condition", "Direction / format", "Model", "Val. pairs (units)", "Attention share (AUC+)", "A_direct / drop", "M_direct / drop", "All-MoE M / drop",
            "MoE peak (val)", "Attention peak (val)", "Selected expert: Spec [95% CI], pattern", "Top heads (val rescue)"]
    rows = []
    for c in CONDS:
        for p in PROTOS:
            e = reg.get(c, {}).get(p)
            if not e:
                continue
            s = S.get(f"{c}/{p}")
            fmt = f"{e['direction']}, {e['format']}"
            if e.get("behaviour_only"):
                rows.append([c, fmt, LABEL[p], "behaviour only", "", "", "", "", "", "", "", ""])
                continue
            if not s:
                rows.append([c, fmt, LABEL[p], f"{e['split_sizes'].get('validation')}", "pending", "", "", "", "", "", "", ""])
                continue
            x = s["cluster"]["main_validation"]
            pk = x["peaks_val_argmax"]
            sel = e.get("selected_expert") or {}
            st = (f"**{sel['name']}** {sel['spec'][0]:+.2f} [{sel['spec'][1]:+.2f}, {sel['spec'][2]:+.2f}], {sel['pattern']}" if sel.get("name") else
                  (f"none at L{sel['layer']} ({sel.get('pattern')})" if sel else "pending"))
            if sel.get("rep"):
                st += f"; rep L{sel['rep']['layer']}E{sel['rep']['expert']:03d}"
            heads = ""
            if s.get("w5"):
                heads = ", ".join(f"L{int(r['layer'])}H{int(r['head'])} {r['val_mean']:+.2f}" for r in s["w5"]["table"][:3])
            rows.append([c, fmt, LABEL[p], f"{x['n_pairs']} ({x['n_units']} {s['unit']}s)", fr(x["share"]), fr(x.get("A_direct")), fr(x.get("M_direct")),
                         fr(x.get("M_all_moe")), f"L{pk['layer']['layer']} {pk['layer']['norm'][0]:.3f}" if "layer" in pk else "",
                         f"L{pk['attn_layer']['layer']} {pk['attn_layer']['norm'][0]:.3f}" if "attn_layer" in pk else "", st, heads])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    txt = "\n".join(lines) + "\n"
    open(os.path.join(TAB, "ext13base_summary.md"), "w").write("**Phase-5 base summary (validation pairs; cluster-bootstrap 95% CIs, unit = seed for templates, Phase-3 twin for cp_nat, else pair)**\n\n" + txt)
    pd.DataFrame(rows, columns=head).to_csv(os.path.join(TAB, "ext13base_summary.csv"), index=False)
    return txt


# ---------------------------------------------------------------------------------------------------------------
# numbers-driven narrative (so that every condition appears in the section as soon as its base analysis exists)
# ---------------------------------------------------------------------------------------------------------------
PH3 = {"qwen3": (41, 117), "mixtral_bos": (20, 0)}
TITLE = {"f1": "attribute → entity, fill-in-the-blank (task 1)", "dir_f1": "direction on identical items: fill-in-the-blank (task 2)",
         "dir_opt": "direction on identical items: Phase-3 option swap (task 2)", "cp_nat": "copy prefix on the natural Phase-3 twins (task 6)",
         "cp_vt_orig": "copy prefix on the verb templates, original verb (task 6)", "cp_vt_neu": "copy prefix on the verb templates, neutral verb (task 6)",
         "vt_e2a_orig": "verb templates, entity → attribute, original verb (task 4)", "vt_e2a_neu": "verb templates, entity → attribute, neutral verb (task 4)",
         "vt_a2e_orig": "verb templates, attribute → entity fill-in-the-blank, original verb (task 5)", "vt_a2e_neu": "verb templates, attribute → entity, neutral verb (task 5)",
         "f2e": "attribute → entity, natural sentence cut at a final blank (task 3)"}


def ci3(x, d=2):
    return f"{x[0]:.{d}f} [{x[1]:.{d}f}, {x[2]:.{d}f}]" if x else "n/a"


def sel_txt(e):
    sel = (e or {}).get("selected_expert") or {}
    if not sel:
        return "no selection"
    if not sel.get("name"):
        return f"no recurrent expert at L{sel.get('layer')}"
    t = f"{sel['name']} (Spec {sel['spec'][0]:+.2f} [{sel['spec'][1]:+.2f}, {sel['spec'][2]:+.2f}], pattern {sel['pattern']}"
    if sel.get("rep"):
        r = sel["rep"]
        t += f"; replication re-selects L{r['layer']}E{r['expert']:03d}" if (r["layer"], r["expert"]) == (sel["layer"], sel["expert"]) else f"; replication selects L{r['layer']}E{r['expert']:03d}"
    return t + ")"


def model_sentence(c, p, s, e):
    x = s["cluster"]["main_validation"]
    pk = x["peaks_val_argmax"]
    d = s["descriptors"]["main"]
    parts = [f"{LABEL[p]} ({x['n_pairs']} validation pairs, {x['n_units']} {s['unit']}s; drop {d['drop']['mean']:.2f}): attention share {ci3(x['share'])}, "
             f"A_direct {ci3(x.get('A_direct'))}, M_direct {ci3(x.get('M_direct'))}, all-MoE M {ci3(x.get('M_all_moe'))}; peaks MoE "
             f"L{pk['layer']['layer']} {pk['layer']['norm'][0]:.3f}, attention L{pk['attn_layer']['layer']} {pk['attn_layer']['norm'][0]:.3f} of the drop"]
    parts.append("expert " + sel_txt(e))
    fx = [f for f in (s.get("w6") or {}).get("main", {}).get("cf_experts", []) if (f["layer"], f["expert"]) == PH3[p]]
    if fx:
        f = fx[0]
        parts.append(f"Phase-3 WinoGrande expert L{f['layer']}E{f['expert']:03d}: rescue {f['rescue']['mean']:+.2f}, Spec {f['spec']['mean']:+.2f} "
                     f"(discovery-active {f['disc_active']})")
    if s.get("w5"):
        parts.append("top heads " + ", ".join(f"L{int(r['layer'])}H{int(r['head'])} {r['val_mean']:+.2f}" for r in s["w5"]["table"][:3]))
    for nm, lab in (("names", "names"), ("congruent", "congruent order")):
        a, b = s["cluster"].get(f"{nm}_True"), s["cluster"].get(f"{nm}_False")
        if a and b:
            parts.append(f"share {lab} {a['share'][0]:.2f} ({a['n_pairs']} pairs) vs " + ("objects" if nm == "names" else "incongruent") + f" {b['share'][0]:.2f} ({b['n_pairs']})")
    g = s.get("w3_sites") or {}
    if "layer" in g and "STR" in g["layer"]:
        parts.append(f"grid: MoE at the STR site peaks L{g['layer']['STR']['peak_layer']} {g['layer']['STR']['peak'][0]:.2f}, attention at the final "
                     f"position L{g['attn_layer']['final']['peak_layer']} {g['attn_layer']['final']['peak'][0]:.2f}" if "attn_layer" in g and "final" in g["attn_layer"] else "")
    cp = (s.get("w5_copy") or {}).get("top_heads")
    if cp:
        h = cp[0]
        parts.append(f"copy attention of the top head L{h['layer']}H{h['head']}: in-context answer {h['clean_attr_true']:.2f} vs other attribute "
                     f"{h['clean_attr_other']:.2f} (clean), {h['corrupt_attr_true']:.2f} vs {h['corrupt_attr_other']:.2f} (corrupted)")
    return "; ".join(q for q in parts if q) + "."


def contrast(S, reg, a, b, lab):
    out = []
    for p in PROTOS:
        sa, sb = S.get(f"{a}/{p}"), S.get(f"{b}/{p}")
        if not (sa and sb):
            continue
        xa, xb = sa["cluster"]["main_validation"], sb["cluster"]["main_validation"]
        out.append(f"{LABEL[p]} share {xa['share'][0]:.2f} vs {xb['share'][0]:.2f}, A_direct {xa['A_direct'][0]:.2f} vs {xb['A_direct'][0]:.2f}, "
                   f"all-MoE M {xa['M_all_moe'][0]:.2f} vs {xb['M_all_moe'][0]:.2f}")
    return f"- {lab} (`{a}` vs `{b}`): " + "; ".join(out) + "." if out else ""


def auto_summary(S, reg):
    lines = []
    for c in CONDS:
        ms = [(p, S[f"{c}/{p}"]) for p in PROTOS if f"{c}/{p}" in S]
        if not ms:
            continue
        lines.append(f"- `{c}` ({TITLE[c]}): " + "; ".join(
            f"{LABEL[p]} share {s['cluster']['main_validation']['share'][0]:.2f}, A_direct {s['cluster']['main_validation']['A_direct'][0]:.2f}, "
            f"expert {((reg[c][p].get('selected_expert') or {}).get('name') or 'none')} ({(reg[c][p].get('selected_expert') or {}).get('pattern', '–')})"
            for p, s in ms) + ".")
    con = [contrast(S, reg, "dir_f1", "dir_opt", "Direction on identical pairs, attribute → entity vs entity → attribute"),
           contrast(S, reg, "cp_nat", "dir_opt", "Copy vs generate on natural twins (different item filters)"),
           contrast(S, reg, "cp_vt_orig", "vt_e2a_orig", "Copy vs generate on templates, original verb"),
           contrast(S, reg, "cp_vt_neu", "vt_e2a_neu", "Copy vs generate on templates, neutral verb"),
           contrast(S, reg, "vt_e2a_orig", "vt_e2a_neu", "Original vs neutral verb, entity → attribute"),
           contrast(S, reg, "vt_a2e_orig", "vt_e2a_orig", "Direction on the templates (Qwen3 only)"),
           contrast(S, reg, "f2e", "f1", "Natural sentence vs fill-in-the-blank, attribute → entity")]
    con = [x for x in con if x]
    return lines, con


def strata_table(S):
    head = ["Condition", "Model", "Stratum", "Pairs (main, all)", "Attention share", "A_direct / drop", "M_direct / drop", "All-MoE M / drop"]
    rows = []
    for c in CONDS:
        for p in PROTOS:
            s = S.get(f"{c}/{p}")
            if not s:
                continue
            for k, lab in (("names_True", "names"), ("names_False", "objects"), ("s_neutral_True", "probe-neutral"), ("s_neutral_False", "not neutral"),
                           ("congruent_True", "copy order congruent"), ("congruent_False", "copy order incongruent")):
                x = s["cluster"].get(k)
                if x:
                    rows.append([c, LABEL[p], lab, x["n_pairs"], fr(x["share"]), fr(x.get("A_direct")), fr(x.get("M_direct")), fr(x.get("M_all_moe"))])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    txt = "\n".join(lines) + "\n"
    open(os.path.join(TAB, "ext13base_strata.md"), "w").write("**Phase-5 base strata (main discovery + validation pairs; cluster bootstrap)**\n\n" + txt)
    pd.DataFrame(rows, columns=head).to_csv(os.path.join(TAB, "ext13base_strata.csv"), index=False)
    return txt


def main():
    S = json.load(open(os.path.join(RESULTS, "ext13base_summary.json"))) if os.path.exists(os.path.join(RESULTS, "ext13base_summary.json")) else {}
    reg = json.load(open(os.path.join(ROOT, "data/wino_ref/p5_conditions.json")))["conditions"]
    notes = json.load(open(NOTES_PATH)) if os.path.exists(NOTES_PATH) else {}
    ready = [f"{c}/{p}" for c in CONDS for p in PROTOS if reg.get(c, {}).get(p, {}).get("status", {}).get("base")]
    pend = [f"{c}/{p}" for c in CONDS for p in PROTOS if reg.get(c, {}).get(p) and not reg[c][p].get("behaviour_only") and not reg[c][p].get("status", {}).get("base")]
    bo = [f"{c}/{p}" for c in CONDS for p in PROTOS if reg.get(c, {}).get(p, {}).get("behaviour_only")]
    out = []
    lines, con = auto_summary(S, reg)
    out.append(notes.get("summary_intro", "**Summary.**") + "\n")
    out.append("Headlines per condition (validation pairs; generated from `results/ext13base_summary.json`):\n\n" + "\n".join(lines) + "\n")
    if con:
        out.append("Contrasts (validation means; the CIs are in the summary table; only `dir_f1` / `dir_opt` share their pairs):\n\n" + "\n".join(con) + "\n")
    out.append(notes.get("summary_findings", "") + "\n")
    out.append(f"Status: base analysis complete for {len(ready)} condition × model runs ({', '.join(ready) or 'none'}); pending: {', '.join(pend) or 'none'}; "
               f"behaviour only: {', '.join(bo) or 'none'}.\n")
    out.append("### What was run\n")
    out.append(notes.get("setup", "") + "\n")
    out.append(tab("ext13base_conditions"))
    out.append("\n### Association probe (pre-registered rule)\n")
    out.append(notes.get("probe", "") + "\n")
    out.append(tab("ext13base_probe"))
    out.append("\n" + tab("ext13base_probe_strata_overall"))
    out.append("\n### Summary across conditions and models\n")
    out.append(notes.get("results", "") + "\n")
    out.append(summary_table(S, reg))
    out.append("\n" + strata_table(S))
    for c in CONDS:
        if not any(f"{c}/{p}" in S for p in PROTOS):
            continue
        out.append(f"\n#### {c}: {TITLE[c]}\n")
        for p in PROTOS:
            if f"{c}/{p}" in S:
                out.append("- " + model_sentence(c, p, S[f"{c}/{p}"], reg[c][p]))
        out.append("")
        if notes.get(c):
            out.append(notes[c] + "\n")
        out.append(f"Tables: `results/tables/ext13base_{c}_{{descriptors,headline,w2_peaks,w2_strata,w4,dla,w6_experts,w6_joint,w6_fixed,w5_heads,w3_sites}}.md`"
                   + (f", `ext13base_{c}_w5_copy.md`" if c.startswith("cp_") else "") + f"; figures `results/figures/ext13base_{c}_{{w2_curves,w3_grid_<proto>,w5_heads,w4_dla}}.png`.\n")
        out.append(tab(f"ext13base_{c}_headline"))
        out.append("\n" + tab(f"ext13base_{c}_w6_experts"))
    out.append("\n### Behaviour: restated format, memorisation, fill-in-the-blank on templates, copy prefixes\n")
    out.append(notes.get("behaviour", "") + "\n")
    out.append(tab("ext13base_f2r_behaviour"))
    out.append("\n" + tab("ext13base_mem_behaviour"))
    out.append("\n" + tab("ext13base_cp_options"))
    out.append("\n" + tab("ext13base_cp_respectively"))
    out.append("\n" + tab("ext13base_scan_behaviour"))
    out.append("\n### Deviations and caveats\n")
    out.append(notes.get("deviations", "") + "\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(out))
    print("wrote", OUT, f"({len(ready)} ready, {len(pend)} pending)")


if __name__ == "__main__":
    main()
