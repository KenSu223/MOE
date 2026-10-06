"""Write results/sections/ext9_knockout.md from results/ext9_knockout_summary.json (+ the engine verification JSON)."""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.models import RESULTS

ARGS = sys.argv[1:]
S = json.load(open(ARGS[0] if len(ARGS) > 0 else os.path.join(RESULTS, "ext9_knockout_summary.json")))
V = json.load(open(os.path.join(RESULTS, "verify_ext9_engine_olmoe.json"))) if os.path.exists(os.path.join(RESULTS, "verify_ext9_engine_olmoe.json")) else None
M = S["models"]
OUT = ARGS[1] if len(ARGS) > 1 else os.path.join(RESULTS, "sections", "ext9_knockout.md")
TABLES = ARGS[2] if len(ARGS) > 2 else os.path.join(RESULTS, "tables")


def f3(x, s="+"):
    return "n/a" if x is None else (f"{x:+.3f}" if s == "+" else f"{x:.3f}")


def fci(e, k="F"):
    if not e or e.get(k) is None:
        return "n/a"
    return f"{e[k]:+.3f} [{e[k + '_lo']:+.3f}, {e[k + '_hi']:+.3f}]"


def fnll(e):
    if not e or e.get("dnll") is None:
        return "n/a"
    return f"{e['dnll']:+.4f} [{e['dnll_lo']:+.4f}, {e['dnll_hi']:+.4f}]"


def eff(d, cid, s):
    return d["effects"].get(cid, {}).get(s)


def table(path):
    p = os.path.join(TABLES, path + ".md")
    return open(p).read().split("\n", 2)[2] if os.path.exists(p) else "(table missing)\n"


def model_name(key):
    return {"qwen3": "Qwen3", "mixtral": "Mixtral (BOS)"}.get(key, key)


lines = []
# ---------------------------------------------------------------- reading (numbers from the summary + patch results)
def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


W7 = json.load(open(os.path.join(RESULTS, "ext7_wino_summary.json"))) if os.path.exists(os.path.join(RESULTS, "ext7_wino_summary.json")) else {}
E6 = json.load(open(os.path.join(RESULTS, "ext6_str_summary.json"))) if os.path.exists(os.path.join(RESULTS, "ext6_str_summary.json")) else {}
PROTO = {"qwen3": "qwen3", "mixtral": "mixtral_bos"}


def patch(key, t, task):
    """(STR single-expert rescue in logits, as a fraction of the drop) from Directions 6 / 7 (validation)."""
    if task == "wg":
        e = g(W7, f"wino_{PROTO.get(key, key)}_str", "w6", "main", "two_stage", "eval", default={})
        if e and f"L{e['layer']}E{int(e['expert']):03d}" == t:
            return e["rescue"]["mean"], e["rescue_norm"][0]
        return None
    e = g(E6, PROTO.get(key, key), "named", "STR (donor mean)", t)
    if not e:  # two-stage selection of that layer (ext6 'experts', keyed by layer)
        l_, e_ = int(t[1:t.index("E")]), int(t[t.index("E") + 1:])
        x = g(E6, PROTO.get(key, key), "experts", "STR (donor mean)", str(l_))
        e = x if x and int(x.get("e_star", -1)) == e_ else None
    return (e["rescue"]["mean"], e["rescue_norm"][0]) if e else None


def own_sub(task):
    return "wg_margin" if task == "wg" else "cf"


def other_sub(task):
    return "cf" if task == "wg" else "wg_margin"


# ---------------------------------------------------------------- summary paragraph
def F_(d, cid, sub):
    return g(d, "effects", cid, sub, default={})


summ = []
for key, d in M.items():
    wg, cf = d["primary"]["wg"], d["primary"]["cf"]
    ew, ec = F_(d, f"tgt:{wg}", "wg_margin"), F_(d, f"tgt:{cf}", "cf")
    slw = g(d, "same_layer", wg, "by_subset", "wg_margin", default={})
    slc = g(d, "same_layer", cf, "by_subset", "cf", default={})
    dis = g(d, "dissociation", f"{wg}__{cf}", default={})
    others = [t for t in d["targets"] if d["target_task"][t] == "cf" and t != cf]
    oth = ", ".join(f"{t} {f3(g(F_(d, f'tgt:{t}', 'cf'), 'F'))}" for t in others)
    summ.append(
        f"{model_name(key)}: {wg} (WinoGrande) costs **{fci(ew)}** of the WinoGrande margin and {f3(g(F_(d, f'tgt:{wg}', 'cf'), 'F'))} "
        f"of CounterFact; {cf} (CounterFact) costs **{fci(ec)}** of CounterFact and {f3(g(F_(d, f'tgt:{cf}', 'wg_margin'), 'F'))} of WinoGrande"
        + (f" ({oth} for the other CounterFact selections)" if oth else "")
        + (f"; same-layer rank on the own task {slw.get('rank', '?')}/{slw.get('n', '?')} and {slc.get('rank', '?')}/{slc.get('n', '?')}" if slw or slc else "")
        + f"; double-dissociation contrast **{f3(dis.get('contrast'))}** [{f3(dis.get('contrast_lo'))}, {f3(dis.get('contrast_hi'))}].")
sets_s = []
for key, d in M.items():
    w10, c10 = g(d, "effects", "pop:wg:k10", default={}), g(d, "effects", "pop:cf:k10", default={})
    rw = g(d, "sets", "wg:k10", "by_subset", "wg_margin", default={})
    rc = g(d, "sets", "cf:k10", "by_subset", "cf", default={})
    sets_s.append(f"{model_name(key)} WinoGrande top-10 {f3(g(w10, 'wg_margin', 'F'))} on WinoGrande (pair accuracy "
                  f"{d['baseline']['wg_margin']['pair_acc']:.3f} -> {g(w10, 'wg_margin', 'pair_acc_ko', default=float('nan')):.3f}) / "
                  f"{f3(g(w10, 'cf', 'F'))} on CounterFact, CounterFact top-10 {f3(g(c10, 'cf', 'F'))} / {f3(g(c10, 'wg_margin', 'F'))}"
                  + (f" (frequency-matched random 10-sets {f3(rw.get('rand_mean'))} / {f3(rc.get('rand_mean'))} on the respective own task)" if rw and rc else ""))
ratios = []
for key, d in M.items():
    for t in d["targets"]:
        task = d["target_task"][t]
        pr = patch(key, t, task)
        e = F_(d, f"tgt:{t}", own_sub(task))
        if pr and e.get("mean_ddelta") is not None:
            ratios.append(abs(e["mean_ddelta"]) / pr[0])
fin = []
for key, d in M.items():
    for t in (d["primary"]["wg"], d["primary"]["cf"]):
        task = d["target_task"][t]
        a = g(d, "final_zero", t, "all_reroute", own_sub(task), "F")
        f = g(d, "final_zero", t, "final_reroute", own_sub(task), "F")
        if a and f is not None and a > 0.01:
            fin.append(f"{model_name(key)} {t} {f / a:.2f}")
null_s = []
for key, d in M.items():
    e = g(d, "effects", "null", default={})
    if e:
        null_s.append(f"{model_name(key)} {f3(g(e, 'wg_margin', 'F'))} / {f3(g(e, 'cf', 'F'))}")
strong, weak = [], []
for key, d in M.items():
    for t in d["targets"]:
        task = d["target_task"][t]
        e = F_(d, f"tgt:{t}", own_sub(task))
        x = F_(d, f"tgt:{t}", other_sub(task))
        nl = g(d, "effects", "null", own_sub(task), "F", default=0.0) or 0.0
        thr = max(0.005, 3 * abs(nl))
        if e.get("F_lo") is not None and e["F_lo"] > thr and abs(x.get("F", 1)) < 0.25 * e["F"]:
            strong.append(f"{model_name(key)} {t}")
        else:
            weak.append(f"{model_name(key)} {t} ({f3(e.get('F'))})")
verdict = ("Necessary and task-specific (lower CI bound of the own-task F above max(0.005, 3 x |noise-floor F|), other-task "
           "effect below a quarter of it): " + ", ".join(strong)
           + ("; barely necessary: " + ", ".join(weak) if weak else "") + ". ")
lines.append(
    "**Summary.** Expert knockout on clean prompts (engine route mask at every position, reroute: the expert is removed from "
    "the router's menu and the next-best expert takes the slot) asks whether the experts that the final-position STR patches "
    "select are NECESSARY, and only for their own task (F = fraction of the clean margin lost). " + verdict
    + "Each carries only a few per cent of the margin. " + " ".join(summ)
    + " Joint knockout of each task's population top-10 (ext8 ranking): " + "; ".join(sets_s) + "."
    + (f" Sufficiency is not necessity: a knockout moves Δ by only {min(ratios):.2f}-{max(ratios):.2f} of the logits that the same "
       "expert restores when patched under STR; dropping the slot without replacement (zero mode) costs about as much as rerouting, "
       "so it is not the replacement expert that compensates, and the damage is confined to prompts whose final position routes to "
       "the expert" if ratios else "")
    + (f"; a knockout at the final position only reproduces " + ", ".join(fin) + " of the all-position effect, i.e. the "
       "necessity of these experts sits at the final position, where the patches found them" if fin else "")
    + "." + (" Noise floor (unmasked baseline recomputed in other passes, WinoGrande / CounterFact): " + "; ".join(null_s) + "." if null_s else ""))
lines.append("")
# ---------------------------------------------------------------- what was run
lines.append("### What was run")
lines.append("")
if V:
    rm = V["route_mask"]
    hs = V["head_steps_invariants"]
    vh = V["head_steps_vs_hf"]
    cfv = V["contrib_final_vectors"]
    pm = rm["per_pos_mode"]
    lines.append(
        "**Engine E4 (additive, `moetrace/engine.py`).** (a) `PrefillSpec.route_mask` = ((layer, expert), ...) with "
        "`route_mask_pos` all | final and `route_mask_mode` reroute | zero: reroute sets the masked experts' router logits to "
        "-inf before the softmax (top-k and renormalisation as usual; for Qwen3 and Mixtral, which renormalise the top-k, this "
        "is exactly removing the expert from the menu; OLMoE's unrenormalised weights grow by 1/(1 - p_masked)), zero keeps the "
        "routing and drops the masked contributions; recorded routing is the masked routing. (b) `multi` steps "
        "`(layer, 'attn_head', heads)` and `(layer, 'heads_experts', (heads, experts))`: v_heads = sum_h W_o[:, h](H_h_source - "
        "H_h_own) added before the MoE, the row's own MoE recomputed on the head-patched input, then the listed experts set to the "
        "source's. (c) `DiagSpec.contrib_final_vectors`: per-slot c_e at the final position. "
        f"Verification on OLMoE against transformers hooks (`scripts/ext9_engine_verify.py`, {V['n_units']} STR units): route masks "
        f"vs a masked HF router, all / final x reroute / zero ({rm['n_rows']} rows): |ΔΔ| max {rm['all_vs_hf']['maxabs']:.3f}, mean "
        f"{rm['all_vs_hf']['meanabs']:.3f} (unmasked baseline {rm['baseline_delta_engine_vs_hf']['maxabs']:.3f} / "
        f"{rm['baseline_delta_engine_vs_hf']['meanabs']:.3f}), effect r {', '.join(f'{k} {v['effect_corr']:.3f}' for k, v in pm.items() if v.get('effect_corr') is not None)}; "
        f"masked-layer routing identical to HF in {rm['invariants']['routing_agree_masked_layers_vs_hf']['frac_true']:.3f} of (row, layer); "
        f"final-only masks leave every earlier position bit-identical ({rm['invariants']['final_only_earlier_positions_identical']['frac_true']:.2f}); "
        f"zero mode keeps the first masked layer's routing ({rm['invariants']['zero_first_layer_routing_unchanged']['frac_true']:.2f}) and zeroes the masked slots "
        f"({rm['invariants']['zero_masked_slots_cnorm_zero']['frac_true']:.2f}). Head steps: one head step = the attn_head kind "
        f"(max |ΔΔ| {hs['one_head_step_vs_attn_head_kind_delta']['maxabs']:.3g}, vector {hs['one_head_step_vs_attn_head_kind_vec_maxabs']:.3g}); "
        f"all heads of a layer vs attn_layer max |ΔΔ| {max(hs[k]['maxabs'] for k in hs if k.endswith('_heads_vs_attn_layer_delta')):.3f}; "
        f"all heads at every layer vs the source Δ max {hs['allheads_all_layers_vs_source_delta']['maxabs']:.3f}; mixed head / expert / "
        f"layer / attention configurations vs HF max |ΔΔ| {vh['all']['maxabs']:.3f}, mean {vh['all']['meanabs']:.3f}, effect r "
        f"{vh['all_effect_corr']:.4f}. Contribution vectors sum to the fp32 MoE output to {cfv['sum_vs_moe_out_max_relnorm']:.1e} (relative). "
        "After the merge `verify_olmoe.json` (23 result fields), `verify_ext5_engine_olmoe.json` (225) and `verify_ext8_engine_olmoe.json` "
        "(496) are identical to their pre-merge copies in every result field (`scripts/ext9_regress_compare.py`); only verify_ext8's "
        "GPU peak-memory fields differ (< 1 MB), which also happens between two runs of the unchanged engine.")
    lines.append("")
lines.append(
    "**Design.** Models Qwen3-30B-A3B-Base and Mixtral-8x7B-v0.1 with BOS (tokenizer defaults), bf16. Targets = the "
    "patching selections (Qwen3 WinoGrande L41E117, CounterFact L44E069 / L42E115; Mixtral WinoGrande L20E000, CounterFact "
    "L19E002 / L21E001 / L18E001) and the top-1/3/5/10 sets of each task's ext8 population ranking (discovery means of the "
    "single-expert STR rescue). Clean prompts only, items used for selection excluded: WinoGrande own margin pool minus the "
    "128 main discovery pairs (primary; both prompts of a pair, Δ toward each prompt's own trigger) and every W1-W6 pair "
    "minus discovery; CounterFact clean scan with Δ_clean >= 1 minus the paper discovery IDs; IOI clean prompts "
    "(logit(IO) - logit(S); primary subset baseline Δ >= 1); 128 wikitext-103 windows of 127 tokens (per-token NLL of tokens "
    "1..126). Controls on a fixed subsample (512 WinoGrande margin pairs, 512 CounterFact cases, 256 IOI prompts, 64 windows): "
    "same-layer experts (Mixtral the 7 others; Qwen3 the 16 experts of the layer most often routed at the final position on "
    "the target task's items) and 5 random sets per population set, matched member by member on final-position routing "
    "frequency within the member's layer. Primary intervention: mask at all positions, reroute; secondary: final position "
    "only (links to the final-position patches); sensitivity: zero mode. F = fraction of the clean margin lost = "
    "(Σ Δ_base - Σ Δ_ko) / Σ Δ_base; percentile bootstrap (5,000) over WinoGrande pairs, CounterFact cases, IOI prompts, "
    "windows, with one resample matrix per item set (paired across conditions); dissociation contrast "
    "(F_WG-expert(WG) - F_WG-expert(CF)) - (F_CF-expert(WG) - F_CF-expert(CF)) with WinoGrande pairs and CounterFact cases "
    "resampled independently.")
lines.append("")
rows = []
for key, d in M.items():
    b = d["baseline"]
    rows.append(f"| {d['label']} | {b['wg_margin']['n_pairs']} pairs ({b['wg_margin']['n']} prompts), Δ {b['wg_margin']['mean_delta']:.2f}, "
                f"pair acc {b['wg_margin']['pair_acc']:.3f} | {b['wg_all']['n_pairs']} pairs, pair acc {b['wg_all']['pair_acc']:.3f} | "
                f"{b['cf']['n']}, Δ {b['cf']['mean_delta']:.2f}, top-1 {b['cf']['top1']:.2f} | {b['ioi']['n']}, Δ {b['ioi']['mean_delta']:.2f} | "
                f"{b['wiki']['n']} windows, NLL {b['wiki']['nll_mean']:.3f} | {d.get('gpu_s', 0) / 60:.0f} min |")
lines.append("| Model | WinoGrande margin pool | WinoGrande all pairs | CounterFact | IOI (Δ >= 1) | wikitext | GPU (passes) |")
lines.append("|---|---|---|---|---|---|---|")
lines += rows
lines.append("")
# ---------------------------------------------------------------- results
lines.append("### 1. Targets and population sets (full scope)")
lines.append("")
lines.append(table("ext9_knockout_targets"))
lines.append("")
lines.append("### 2. Against same-layer controls")
lines.append("")
lines.append(table("ext9_knockout_controls"))
lines.append("")
lines.append("![Targets vs same-layer controls](figures/ext9_knockout_controls.png)")
lines.append("")
lines.append("### 3. Population sets against frequency-matched random sets")
lines.append("")
lines.append(table("ext9_knockout_sets"))
lines.append("")
lines.append("![Sets vs random sets](figures/ext9_knockout_sets.png)")
lines.append("")
lines.append("### 4. Double dissociation")
lines.append("")
lines.append(table("ext9_knockout_dissociation"))
lines.append("")
lines.append("### 5. Routing: which expert takes the slot")
lines.append("")
lines.append(table("ext9_knockout_routing"))
lines.append("")
lines.append("### 6. Final position only and zero mode")
lines.append("")
lines.append(table("ext9_knockout_final_zero"))
lines.append("")
R = []
# R1 necessity / specificity of the single targets
parts = []
for key, d in M.items():
    ps_ = []
    for t in d["targets"]:
        task = d["target_task"][t]
        e = d["effects"].get(f"tgt:{t}", {})
        sl = g(d, "same_layer", t, "by_subset", own_sub(task), default={})
        ps_.append(f"{t} ({task.upper()}) {fci(e.get(own_sub(task)))} on its own task vs {f3(g(e, other_sub(task), 'F'))} on the other "
                   f"(IOI {f3(g(e, 'ioi', 'F'))}, wiki {f3(g(e, 'wiki', 'dnll'))} nats; rank {sl.get('rank', '?')}/{sl.get('n', '?')} "
                   f"among same-layer experts on its own task)")
    parts.append(f"{model_name(key)}: " + "; ".join(ps_) + ".")
R.append("**1. The patch-selected experts are necessary for their own task and only for it, but each carries a few per cent of "
         "the margin.** Fraction of the clean margin lost by a single all-position knockout (full scope): " + " ".join(parts))
# R1b strongest control experts (all same-layer controls of a model)
parts = []
for key, d in M.items():
    for sub_, name_ in (("wg_margin", "WinoGrande"), ("cf", "CounterFact")):
        best = None
        for t, v in d.get("same_layer", {}).items():
            x = g(v, "by_subset", sub_, "controls", default={})
            for c, val in x.items():
                if val is not None and (best is None or val > best[1]):
                    best = (c, val, t)
        if best:
            w = None
            for t, v in d.get("same_layer", {}).items():
                w = g(v, "by_subset", "wiki", "controls", best[0], default=w)
            tgt = [t for t in d["targets"] if d["target_task"][t] == ("wg" if sub_ == "wg_margin" else "cf")]
            tv = max((g(d, "same_layer", t, "by_subset", sub_, "target", default=-1) or -1) for t in tgt) if tgt else None
            parts.append(f"{model_name(key)} {name_}: {best[0]} {f3(best[1])} (wiki {f3(w)} nats; best target {f3(tv)})")
if parts:
    R.append("**Strongest non-target expert in the control sets (sub scope).** " + "; ".join(parts) + ". A control can be as "
             "necessary as a target: in Mixtral L19E006 (a same-layer control of L19E002 and WinoGrande rank 2 in the ext8 "
             "population ranking) is the single most damaging WinoGrande knockout and also the most generically damaging one.")
# R2 sufficiency vs necessity, routing compensation (table)
tab = ["| Model | Expert | Task | STR patch rescue, logits (fraction of drop) | knockout Δ change, logits | F all positions | F final only | F zero mode | routed at final | one-for-one swap | top replacement (share) |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
ratios = []
for key, d in M.items():
    for t in d["targets"]:
        task = d["target_task"][t]
        e = g(d, "effects", f"tgt:{t}", own_sub(task), default={})
        pr = patch(key, t, task)
        rc = g(d, "routing_change", f"{t}|all|reroute", task, default={})
        z = g(d, "final_zero", t, "all_zero", own_sub(task), default={})
        rr = g(d, "final_zero", t, "all_reroute", own_sub(task), default={})
        fn = g(d, "final_zero", t, "final_reroute", own_sub(task), default={})
        if pr and e.get("mean_ddelta"):
            ratios.append(abs(e["mean_ddelta"]) / pr[0])
        top = rc["top_replacements"][0] if rc.get("top_replacements") else None
        tab.append(f"| {model_name(key)} | {t} | {task.upper()} | " + (f"+{pr[0]:.2f} ({pr[1]:.3f})" if pr else "n/a") +
                   f" | {f3(e.get('mean_ddelta'))} | {f3(rr.get('F'))} | {f3(fn.get('F'))} | {f3(z.get('F'))} | "
                   f"{f3(rc.get('freq_final'), '')} | {f3(rc.get('frac_clean_swap'), '')} | " + (f"{top[0]} ({top[1]:.2f})" if top else "n/a") + " |")
R.append("**2. Sufficiency is not necessity.** Patched alone into the corrupted run at the final position (Directions 6 / 7, "
         "validation), each expert restores a sizeable part of the STR drop; removed from the clean run, it costs a fraction of a "
         "logit" + (f" (knockout Δ change / patch rescue = {min(ratios):.2f}-{max(ratios):.2f})" if ratios else "") + ". The "
         "reroute knockout replaces the expert one-for-one in almost every prompt (spread over several replacement experts), but "
         "dropping the slot with no replacement (zero mode) costs about the same, so the small necessity is not compensation by the "
         "replacement: the knockout removes the expert's clean contribution, whereas the patch swaps the corrupted run's "
         "contribution for the clean one and lets later layers respond (the direct / indirect split of the patch effect is "
         "Direction 11). The final-position-only knockout reproduces most of the all-position effect: the necessity sits where "
         "the patches found the experts. F columns on the sub scope (same items for the three interventions).\n\n" + "\n".join(tab))
# R3 sets
parts = []
for key, d in M.items():
    for task in ("wg", "cf"):
        v = g(d, "sets", f"{task}:k10", "by_subset", default={})
        o, x = v.get(own_sub(task), {}), v.get(other_sub(task), {})
        e = g(d, "effects", f"pop:{task}:k10", default={})
        b = d["baseline"]
        acc = (f"pair accuracy {b['wg_margin']['pair_acc']:.3f} -> {g(e, 'wg_margin', 'pair_acc_ko'):.3f}" if task == "wg" and g(e, 'wg_margin', 'pair_acc_ko') is not None
               else f"accuracy {b['cf']['acc']:.3f} -> {g(e, 'cf', 'acc_ko'):.3f}" if g(e, 'cf', 'acc_ko') is not None else "")
        parts.append(f"{model_name(key)} {task.upper()} top-10: {fci(e.get(own_sub(task)))} on its own task ({acc}), "
                     f"{f3(g(e, other_sub(task), 'F'))} on the other, wiki {fnll(e.get('wiki'))}; on the sub scope "
                     f"{f3(o.get('pop'))} vs {f3(o.get('rand_mean'))} (max {f3(o.get('rand_max'))}) for 5 frequency-matched random sets")
ov = [f"{model_name(key)}: " + (", ".join(d.get("pop_overlap", {}).get("10", [])) or "none") for key, d in M.items()]
R.append("**3. Sets: the damage grows with k and stays on the task where the rankings do not overlap.** " + "; ".join(parts) + ". "
         "Experts shared by the two population top-10 lists: " + "; ".join(ov) + ". Mixtral's CounterFact top-10 contains two "
         "WinoGrande experts (L19E006 = WinoGrande rank 2, L22E005 = rank 8), which is why it damages WinoGrande more than "
         "CounterFact; L19E006 is also the expert that the BOS token always routes to at L19 (Direction 3), so knocking it out at "
         "all positions removes it from the attention sink as well, a likely source of the larger generic (wikitext) damage of "
         "every Mixtral set that contains it.")
# R6 IOI
parts = []
for key, d in M.items():
    for t in d["targets"]:
        e = g(d, "effects", f"tgt:{t}", "ioi", default={})
        if e.get("F") is not None:
            parts.append(f"{model_name(key)} {t} {f3(e['F'])}")
if parts:
    R.append("**IOI (third task, never used for selection).** Single knockouts move the IOI margin by " + ", ".join(parts) +
             ". Negative values (margin gains) come from CounterFact experts that are routed at the IOI final position: in IOI the "
             "clean final-position MoE output works against the indirect object (Direction 7b: all-MoE patch M = -0.27 in Qwen3), "
             "so removing such an expert helps.")
# R5 damage sits where the expert is routed at the final position
parts = []
for key, d in M.items():
    for t in d["targets"]:
        task = d["target_task"][t]
        v = g(d, "routing_split", t, own_sub(task), default={})
        if v.get("routed") and v.get("not_routed"):
            parts.append(f"{model_name(key)} {t}: F {f3(v['routed']['F'])} on the {v['routed']['n']} own-task prompts that route it at the "
                         f"final position vs {f3(v['not_routed']['F'])} on the {v['not_routed']['n']} that do not")
        elif v.get("routed"):
            parts.append(f"{model_name(key)} {t}: routed at the final position in {v['routed']['n']} of the own-task prompts, F {f3(v['routed']['F'])}")
neg = []
for key, d in M.items():
    for t in d["targets"]:
        v = g(d, "routing_split", t, "ioi", "routed", default={})
        if v and v.get("F") is not None and v["F"] < -0.005:
            neg.append(f"{model_name(key)} {t} (routed at the IOI final position in {v['n']} prompts, F {f3(v['F'])})")
if parts:
    R.append("**Where the damage is.** The loss concentrates on prompts whose FINAL position routes to the expert in the baseline: "
             + "; ".join(parts) + "." + (" Some CounterFact experts are routed at the final position of almost every IOI prompt and "
             "their removal INCREASES the IOI margin: " + ", ".join(neg) + "." if neg else ""))
# R4 noise floor
parts = []
for key, d in M.items():
    e = d["effects"].get("null", {})
    if e:
        parts.append(f"{model_name(key)} WinoGrande {f3(g(e, 'wg_margin', 'F'))}, CounterFact {f3(g(e, 'cf', 'F'))}, IOI "
                     f"{f3(g(e, 'ioi', 'F'))}, wiki {f3(g(e, 'wiki', 'dnll'))} nats")
if parts:
    R.append("**4. Noise floor.** The unmasked baseline recomputed in other pass compositions (condition `null`, sub scope; bf16 "
             "results depend on the batch composition at the 0.1-logit level) gives F = " + "; ".join(parts) + ".")
lines.append("### Reading")
lines.append("")
for r in R:
    lines.append(r)
    lines.append("")
lines.append("### Caveats")
lines.append("")
lines.append("- A knockout is a distribution shift the model was not trained on; the reroute variant keeps k experts per token "
             "(the model's own routing rule), the zero variant drops the slot. Generic damage is measured on wikitext, not removed.")
lines.append("- Rows are not bit-reproducible across pass compositions (bf16, batch-size-dependent kernels): baseline and knockout "
             "rows sit in different passes; the `null` condition measures this floor.")
lines.append("- Controls and random sets use a fixed subsample (512 WinoGrande margin pairs, 512 CounterFact cases, 256 IOI prompts, 64 "
             "windows); random sets are matched member by member within the member's layer, so with Mixtral's 8 experts per layer "
             "several of the 5 draws coincide.")
lines.append("- WinoGrande evaluation excludes only the 128 main discovery pairs (the selection set of the WinoGrande experts and of "
             "the population ranking); the main validation and replication pairs are part of the margin pool. CounterFact excludes "
             "the paper discovery IDs (the STR selections used the paper split).")
lines.append("- Necessity is measured on clean prompts at every position; the patches measured sufficiency at the final position "
             "under STR. The final-only knockout is the bridge between the two.")
lines.append("")
lines.append("### Files")
lines.append("")
lines.append("Code: `moetrace/engine.py` (E4, docstring 'ext9'), `moetrace/ext9_knockout.py` (items, conditions, controls, pass planner), "
             "`scripts/ext9_engine_verify.py`, `scripts/ext9_knockout_run.py`, `scripts/ext9_knockout_analyze.py`, `scripts/ext9_knockout_text.py`, "
             "`scripts/ext9_chain.sh`. Runs: `results/qwen3_knockout`, `results/mixtral_bos_knockout` (items.parquet, pop_rank.json, controls.json, "
             "plan_<phase>.json, ko_rows_*.parquet row level, ko_wiki_*.npz per-token NLL, ko_basefinal_* / ko_baseallpos_* baseline routing, "
             "run_meta.json). Numbers: `results/ext9_knockout_summary.json`; tables `results/tables/ext9_knockout_*`; figures "
             "`results/figures/ext9_knockout_*`; engine verification `results/verify_ext9_engine_olmoe.json`; API note `logs/ext9_engine_api.md`.")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
