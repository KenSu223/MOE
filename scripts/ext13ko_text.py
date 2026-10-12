"""Section writer for Phase 5 / Direction 9 per condition: results/sections/ext13_knockout.md (from
results/ext13ko_summary<suffix>.json and the ext13ko_* tables).

    python scripts/ext13ko_text.py [--suffix _smoke]
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
from moetrace import ext13ko as K
from moetrace.models import RESULTS

SEC = os.path.join(RESULTS, "sections")
TAB = os.path.join(RESULTS, "tables")
CLAB = {"f1": "F1 attribute → entity (fill-in-the-blank)", "dir_f1": "776 set, attribute → entity (fill-in-the-blank)",
        "dir_opt": "776 set, entity → attribute (Phase-3 option swap)", "f2e": "attribute → entity, natural sentence",
        "vt_e2a_orig": "templates entity → attribute, original verb", "vt_e2a_neu": "templates entity → attribute, neutral verb",
        "vt_a2e_orig": "templates attribute → entity, original verb", "vt_a2e_neu": "templates attribute → entity, neutral verb",
        "cp_vt_orig": "copy: templates e → a + prefix, original verb", "cp_vt_neu": "copy: templates e → a + prefix, neutral verb",
        "cp_nat": "copy: 776 set e → a + prefix", "f1pre": "F1 pre-neutral (smoke)"}
GROUP = {"e2a": ["dir_opt", "vt_e2a_orig", "vt_e2a_neu"], "copy": ["cp_nat", "cp_vt_orig", "cp_vt_neu"],
         "a2e": ["f1", "dir_f1", "f2e", "vt_a2e_orig", "vt_a2e_neu"]}


def embed(name, sfx):
    p = os.path.join(TAB, f"{name}{sfx}.md")
    if not os.path.exists(p):
        return "_(table not available)_\n"
    lines = open(p).read().strip().splitlines()
    if lines and lines[0].startswith("**"):
        lines = lines[2:]
    return "\n".join(lines) + "\n"


def f3(x, sign=True):
    return "n/a" if x is None else (f"{x:+.3f}" if sign else f"{x:.3f}")


def fci(e):
    if not e or e.get("F") is None:
        return "n/a"
    return f"{e['F']:+.3f} [{e['F_lo']:+.3f}, {e['F_hi']:+.3f}]"


def tgtF(d, t, scope="full"):
    return d["effects"].get(f"tgt:{t}", {}).get(scope) if t else None


def necessary(e, null):
    """Lower CI bound above max(0.005, 3 |null F|)."""
    if not e:
        return None
    thr = max(0.005, 3 * abs(null["F"]) if null else 0.005)
    return e["F_lo"] > thr


def reading(src, crit):
    """Verbal reading of a contrast a - b (a = copy / a->e condition, b = reference e->a condition)."""
    lo_a = src.get("F_a_lo")
    d, lo, hi = src["diff"], src["diff_lo"], src["diff_hi"]
    if lo <= 0 <= hi:
        return "unchanged within the CI"
    if d < 0:
        return "reduced to about zero" if abs(src["F_a"]) < max(crit, 0.25 * abs(src["F_b"])) else "reduced"
    return "increased"


def readings(C, M):
    out = {"copy": [], "direction": []}
    for k, v in C.items():
        if not k.endswith(":wg"):
            continue
        pv = v.get("paired")
        src = pv if pv else v
        null = M[v["proto"]][v["b"]]["effects"].get("null", {}).get("full")
        crit = max(0.005, 3 * abs(null["F"])) if null else 0.005
        txt = (f"{K.PROTO[v['proto']]['label']} {v['expert']}: {v['a']} F {src['F_a']:+.3f} vs {v['b']} {src['F_b']:+.3f}, "
               f"difference {src['diff']:+.3f} [{src['diff_lo']:+.3f}, {src['diff_hi']:+.3f}]" + (" (paired, shared source items)" if pv else " (unpaired)")
               + f" -> {reading(src, crit)}")
        if v["b"] in GROUP["e2a"] and v["a"] in GROUP["copy"]:
            out["copy"].append(txt)
        elif v["a"] in GROUP["a2e"] and (v["b"] in GROUP["e2a"] or v["b"] in GROUP["copy"]):
            out["direction"].append(txt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    sfx = a.suffix
    S = json.load(open(os.path.join(RESULTS, f"ext13ko_summary{sfx}.json")))
    M = S["models"]
    C = S.get("contrasts", {})
    out = []
    # ---------------- summary
    summ = ["**Summary.** Phase 4's knockout (engine route mask at every position, reroute: the expert leaves the router's menu "
            "and the next-best expert takes the slot; F = fraction of the clean margin lost) applied to every Phase-5 condition "
            "on clean prompts of the validation and replication pairs (both twins). Targets: each condition's own selected expert, "
            "Phase 3's WinoGrande expert (Qwen3 L41E117, Mixtral L20E000) on every condition, the CounterFact experts as "
            "cross-task controls, the condition's population top-1/3/5/10 sets, frequency-matched random sets and same-layer controls."]
    for proto, cd in M.items():
        lab = K.PROTO[proto]["label"]
        wg = K.le(*K.PROTO[proto]["wg"])
        parts = []
        for gname, glab in (("e2a", "entity → attribute"), ("copy", "copyable answer (prefix)"), ("a2e", "attribute → entity")):
            xs = [f"{c} {fci(tgtF(cd[c], wg))}" for c in GROUP[gname] if c in cd]
            if xs:
                parts.append(f"{glab}: " + "; ".join(xs))
        if parts:
            summ.append(f"{lab}, WinoGrande expert {wg} knocked out — " + " | ".join(parts) + ".")
        sel = []
        for c, d in cd.items():
            t = d["targets"].get("selected")
            null = d["effects"].get("null", {}).get("full")
            s10 = d["sets"].get("k10")
            x = f"{c}: "
            if t:
                e = tgtF(d, t)
                sl = d["same_layer"].get(t)
                x += f"{t} {fci(e)}" + (f" (rank {sl['rank']}/{sl['n']} in its layer)" if sl else "")
                nec = necessary(e, null)
                x += "" if nec is None else (", necessary" if nec else ", below the necessity criterion")
            else:
                x += "no specific expert"
            if s10 and s10.get("pop_full") is not None:
                x += f"; top-10 {s10['pop_full']:+.3f} (random 10-sets {s10['rand_mean']:+.3f})"
            sel.append(x)
        if sel:
            summ.append(f"{lab}, own selections — " + " | ".join(sel) + ".")
    p58 = readings(C, M)
    if p58["copy"]:
        summ.append("**P5-8 (is the WinoGrande expert necessary when the answer can be copied?)** " + "; ".join(p58["copy"]) + ".")
    elif any(c in cd for cd in M.values() for c in GROUP["copy"]) is False:
        summ.append("P5-8 (copy conditions cp_*) not yet available.")
    if p58["direction"]:
        summ.append("**Direction (same items, answer in context vs generated):** " + "; ".join(p58["direction"]) + ".")
    out.append(" ".join(summ) + "\n")
    # ---------------- design
    out.append("### What was run\n")
    out.append("**Design.** Models Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1 with BOS (Phases 3-4 default), bf16, engine E4 route "
               "masks (`PrefillSpec.route_mask`, reroute; Phase 4 verification `results/verify_ext9_engine_olmoe.json`). Per condition "
               "(registry `data/wino_ref/p5_conditions.json`, agent p5-data): items = clean prompts of both twins of every pair "
               "except the main discovery pairs (validation + replication splits; the discovery pairs selected the experts), Δ toward "
               "each prompt's own answer. Full scope (baseline, targets, population sets, null) = these eval items; sub scope (same-layer "
               "controls, 5 frequency-matched random sets per set size, final-position-only masks, cross-condition knockouts) = the main "
               "validation pairs. Same-layer controls for the main targets (own selected expert, WinoGrande expert): Mixtral the 7 other "
               "experts of the layer, Qwen3 the 16 experts of the layer most often routed at the final position on the condition's eval "
               "prompts (targets excluded). Random sets: each member of the population top-k replaced by one of the 4 (Qwen3) / 3 "
               "(Mixtral) experts of its layer with the nearest final-position routing frequency (Phase 4 rule). Null = the unmasked "
               "baseline recomputed in passes of another composition (noise floor). F = (Σ Δ_base − Σ Δ_ko) / Σ Δ_base; bootstrap "
               "(5,000) over the condition's units: template seeds for the semi-synthetic template conditions (vt_*, cp_vt_*; 48 name "
               "pairs per seed are not independent), source pairs for cp_nat (both prefix orders together), pairs otherwise. "
               "Cross-condition differences resample the conditions independently; paired differences use the source items present in "
               "both conditions with a joint bootstrap over shared units. Not run: zero mode (Phase 4: zero ≈ reroute for every target, "
               "`results/tables/ext9_knockout_final_zero.md`) and the wikitext NLL control (generic damage of the same experts measured "
               "in Phase 4, ≤ 0.008 nats/token for single experts).\n")
    rows = ["| Model | Condition | Eval pairs (units) | Baseline Δ | Pair acc | Selected expert | Population top-3 | GPU (min) |",
            "|---|---|---|---|---|---|---|---|"]
    for proto, cd in M.items():
        for c, d in cd.items():
            b = d["baseline"]["full"]
            rows.append(f"| {K.PROTO[proto]['label']} | {c} ({CLAB.get(c, c)}) | {b['n_pairs']} ({b['n_units']} {d['unit_col']}) | "
                        f"{b['mean_delta']:.2f} | {b['pair_acc']:.3f} | {d['targets'].get('selected') or '—'} | "
                        f"{' '.join(d['pop_top10'][:3])} | {d['gpu_s'] / 60:.1f} |")
    out.append("\n".join(rows) + "\n")
    out.append("\n### 1. Targets and population sets (eval items)\n")
    out.append(embed("ext13ko_targets", sfx))
    out.append("\n### 2. Main targets against same-layer controls (validation pairs)\n")
    out.append(embed("ext13ko_controls", sfx))
    out.append("\n### 3. Population sets against frequency-matched random sets (validation pairs)\n")
    out.append(embed("ext13ko_sets", sfx))
    out.append("\n### 4. P5-8: copy vs generate, and direction\n")
    out.append("Contrasts of F between conditions (a − b). C1 = entity → attribute, answer generated (dir_opt, vt_e2a_*); C2 = the "
               "same items with a prefix that names both attributes (cp_*), answer copyable; C3 = attribute → entity (fill-in-the-blank; "
               "the entity is copyable from the sentence). C2 − C1 isolates copy availability at fixed direction and format; C3 − C2 "
               "isolates direction at fixed copy availability (the fill-in wrapper still differs).\n")
    out.append(embed("ext13ko_p58", sfx))
    rd = readings(C, M)
    if rd["copy"]:
        out.append("\n**Reading (P5-8, WinoGrande expert).** " + "; ".join(rd["copy"]) + ". If F stays at the e → a level the "
                   "expert is still necessary although the answer token is in the prefix (the attribute is still computed by the "
                   "MoE); if it falls to about zero the copy route replaces it.\n")
    else:
        out.append("\n_P5-8 copy conditions (cp_nat, cp_vt_orig, cp_vt_neu) not yet run; this section is regenerated after every "
                   "finished condition._\n")
    if rd["direction"]:
        out.append("\n**Reading (direction).** " + "; ".join(rd["direction"]) + ".\n")
    out.append("\n### 5. Final position only, and which expert takes the slot\n")
    out.append(embed("ext13ko_final", sfx))
    out.append("\n" + embed("ext13ko_routing", sfx))
    out.append("\n### 6. Self-repair: knockout loss vs the expert's own direct write\n")
    out.append("Only meaningful for experts with a sizeable direct write (W); for experts that act through later layers (W ≈ 0) the "
               "ratios are undefined (see the writer section).\n")
    out.append(embed("ext13ko_selfrepair", sfx))
    if os.path.exists(os.path.join(TAB, f"ext13ko_cross{sfx}.md")):
        out.append("\n### 7. Cross-condition knockout (other conditions' selected experts)\n")
        out.append(embed("ext13ko_cross", sfx))
    if os.path.exists(os.path.join(TAB, f"ext13ko_strata{sfx}.md")):
        out.append("\n### 8. Strata\n")
        out.append(embed("ext13ko_strata", sfx))
    out.append("\n### Caveats\n")
    out.append("- Item sets differ between conditions (each has its own margin filter and split); only the paired contrasts compare the "
               "same source items. Template conditions have about 27-28 validation seeds: their CIs are wide by design.\n"
               "- F is a ratio of means over items; small baselines (low-margin conditions) inflate F's variance.\n"
               "- bf16: rows of different pass compositions differ at the 0.1-logit level (null row of every table).\n"
               "- The selected expert and population ranking come from p5-data's discovery runs (STR single-expert patches); "
               "knockout tests necessity on held-out pairs only.\n"
               f"- Figures: `results/figures/ext13ko_<proto>_<cond>{sfx}.png` (targets vs same-layer controls; sets vs random), "
               f"`results/figures/ext13ko_overview{sfx}.png` (all conditions).\n")
    os.makedirs(SEC, exist_ok=True)
    p = os.path.join(SEC, f"ext13_knockout{sfx}.md")
    open(p, "w").write("\n".join(out))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
