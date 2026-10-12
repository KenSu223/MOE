"""ext13ci section writer (Phase 5, p5-circuit): results/sections/ext13_circuit.md from results/ext13ci_summary.json and the
ext13ci tables. Numbers are filled in from the summary; the readings block (READINGS) is written by hand after the runs.

Usage: python scripts/ext13ci_text.py [--tag _smoke]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
from moetrace.models import RESULTS
from moetrace import ext13ci as E13

SEC = os.path.join(RESULTS, "sections", "ext13_circuit.md")
READINGS = ""  # filled after the runs (hand-written interpretation)
DEVIATIONS = """- Σ of single-head effects (step-1 summary, additive k for q) carries the wavefront-vs-prefill bf16 offset of every
  single patch (about −0.0015 of the drop per head; visible as a constant negative Σ-heads curve at early layers in the
  layer figures); the joint patches of step 2 are exact and do not have it.
- Expert pool / oracle order of step 2 = the base run's single-expert rescue (W6 rows); entries missing there are replaced by
  the step-1 expert DLA (count per condition in the step-1 summary JSON, experts.n_single_proxied_by_dla).
- Attention masses are shown for the clean prompts only: with both directions of every pair as rows, the corrupted prompts
  are the same set of prompts with the sides exchanged.
- Head labels are a descriptive rule on mean attention (see the attention table caption), not a causal test of copying.
- Not run (registry behaviour-only, too few pairs): vt_a2e_neu (both models), vt_a2e_orig for Mixtral. f2e Mixtral has
  18 validation pairs (36 rows): wide CIs.
- Phase-4 reference values (CounterFact, WinoGrande option swap, IOI) come from ext10 on different items; the paired
  contrasts use only shared clusters (twins / template seeds), the f1 − P3wino comparison is unpaired (2–3 shared twins).
- k50/80/90 = 'never' when the mixed greedy does not reach that fraction within its 20 steps."""


def attn_block(s1):
    """Data-driven paragraph on what the top heads attend to (copy heads vs entity movers)."""
    lines = []
    for k, r in s1.items():
        a = r.get("attn")
        if not a or r["cond"] not in E13.ATTN_CONDS:
            continue
        w = a["rescue_weighted_mass_top10"]
        parts = []
        for b in ("attr", "prefix_ment", "mention"):
            if f"{b}_clean" in w:
                parts.append(f"{b} {w[b + '_clean']:.2f} / {w[b + '_corr']:.2f}")
        for c in ("str", "pos0", "final"):
            if c in w:
                parts.append(f"{c} {w[c]:.2f}")
        labs = ", ".join(f"{x} {n}" for x, n in sorted(a["top10_labels"].items(), key=lambda kv: -kv[1]))
        lines.append(f"- {short(r)}: top-10 heads' rescue-weighted attention (answer side / other) {'; '.join(parts)}; labels {labs}.")
    return "\n".join(lines)


def tab(name, tag=""):
    p = os.path.join(E13.TAB, f"{name}{tag}.md")
    return open(p).read() if os.path.exists(p) else f"(table {name} not yet available)\n"


def fk(x):
    return "never" if x is None or not np.isfinite(x) else f"{int(x)}"


def short(r):
    return f"{'Qwen3' if r['proto'] == 'qwen3' else 'Mixtral'} {r['cond']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag
    sp = E13.SUMMARY if not tag else os.path.join(RESULTS, f"ext13ci_summary{tag}.json")
    S = json.load(open(sp))
    s1, s2, ref = S.get("step1", {}), S.get("step2", {}), S.get("reference", {})
    out = []
    # ---------------- summary
    order = [E13.key_of(c, p) for c in list(E13.COND_ORDER) + ["F1pre"] for p in E13.PROTOS]
    k2 = [k for k in order if k in s2]
    k1 = [k for k in order if k in s1]
    if k2:
        parts = []
        for k in k2:
            r = s2[k]
            g = r["curves"]["greedy_mix"]
            rk = g.get("r20") or g["r_last"]
            parts.append(f"{short(r)} r(20) {rk['ratio']:.2f} [{rk['lo']:.2f}, {rk['hi']:.2f}], heads among the first 10 picks "
                         f"{g['composition']['10']['mean_heads']:.1f}, k80 {fk(g['k_q']['0.8'])}")
        refs = "; ".join(f"{v['label'].split('-')[0].split(' ')[0]} {v['task']} r(20) {v['r20']:.2f}, heads in first 10 {v['heads_first10']:.1f}"
                         for v in ref.values())
        out.append("**Summary.** Mixed head + expert greedy add-back at the final position (validation directed cases, 20 exact "
                   "joint steps, ceiling 1 = all heads): " + "; ".join(parts) + f". Phase 4 references: {refs}.")
    if k1:
        parts = []
        for k in k1:
            r = s1[k]
            th = ", ".join(f"{h['head']} {h['rescue_over_drop']['ratio']:+.3f}" + (f" ({h['attn_label']})" if h.get("attn_label") else "")
                           for h in r["top"][:3])
            parts.append(f"{short(r)}: {th}")
        out.append("Best single heads (rescue / drop; attention label from the clean run's final-position attention): " + "; ".join(parts) + ".")
    if READINGS:
        out.append(READINGS)
    out.append("")
    # ---------------- what was run
    out.append("### What was run\n")
    out.append("Rows = the validation directed cases (family main) of every Phase-5 condition with a finished base run "
               "(data/wino_ref/p5_conditions.json, '[p5-data] BASE READY' lines), Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS, "
               "bf16, STR corruption, denoising at the final position. Machinery = Phase 4 Direction 10 (moetrace/ext10_circuit.py, "
               "reused read-only through moetrace/ext13ci.py). Step 1 (scripts/ext13ci_heads_run.py): one attn_head patch per "
               "(layer, head) and one attn_layer patch per layer per row, patch-free head DLA and expert DLA, and the final "
               "position's attention distribution of the clean and corrupted prompts reduced to position classes from the pair "
               "files (mention_* = candidate mentions, attr_* = the in-context attributes of the copy prefix, prefix_ment_* = "
               "the names in the copy prefix, blank, str = STR site, pos0, final; _clean = the side of the clean prompt). Step 2 "
               "(scripts/ext13ci_circuit_run.py): mixed greedy over the row's top-32 heads ∪ its expert pool (Qwen3 top-32, "
               "Mixtral all 64 clean-active experts by single-expert rescue of the base run), head-only greedy, static "
               "orderings (single-patch oracle and DLA, mixed and heads only), in-pass ceilings (all heads, all attention, all "
               "MoE) and sanity rows; several conditions of one model share every pass. Bootstrap clusters = WinoGrande twins "
               "(HIT id) and, for the verb templates, the seed (template).\n")
    rr = []
    for k in order:
        if k not in s1 and k not in s2:
            continue
        a, b = s1.get(k, {}), s2.get(k, {})
        run2 = next((v for v in S.get("step2_runs", {}).values() if k in v.get("tasks", [])), None)
        rr.append(f"| {E13.MODEL_LABEL[E13.parse_key(k)[1]]} | {E13.parse_key(k)[0]} | {E13.COND_INFO.get(E13.parse_key(k)[0], {}).get('label', '')} | "
                  f"{a.get('n_rows', b.get('n_rows', ''))} / {a.get('n_units', b.get('n_units', ''))} | {a.get('cluster', '')} | "
                  f"{a.get('mean_drop', float('nan')):.2f} | {a.get('passes', '')} | {a.get('gpu_s', 0) / 60:.1f} | {a.get('peak_GB', 0):.1f} | "
                  + (f"{run2['group']} ({len(run2['tasks'])} conditions, {run2['passes']} passes, {run2['pass_s_sum'] / 60:.1f} min, peak {run2['peak_GB_max']:.1f} GB)" if run2 else "")
                  + " |")
    out.append("| Model | Condition | Description | Rows / clusters | Cluster | Mean drop | Step-1 passes | Step-1 GPU min | Step-1 peak GB | Step-2 group |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    out += rr
    out.append("")
    # ---------------- step 1
    out.append("### Step 1. Single heads at every layer, head DLA, attention of the top heads\n")
    out.append(tab("ext13ci_heads_summary", tag))
    out.append("**Top and bottom heads (validation).**\n")
    out.append(tab("ext13ci_heads_top", tag))
    out.append("**Layers with the largest attention-output patch.**\n")
    out.append(tab("ext13ci_heads_layers", tag))
    out.append("**Final-position attention of the top heads (clean run, mean over validation rows).** Classes as in 'What was "
               "run'; a pair 'answer side / other' = the mass on the clean prompt's own candidate (the true answer's mention, or "
               "the true attribute in the copy prefix) vs on the other candidate; rest = all other tokens. Label = the largest of "
               "these. The corrupted prompts are the same set of prompts with the sides exchanged (both directions of every "
               "pair are rows), so their masses mirror the clean ones and are not shown.\n")
    out.append(tab("ext13ci_attn_heads", tag))
    out.append("**Population.** all heads = mean over every head; rescue-weighted = heads weighted by their positive single-head "
               "effect; top-10 = the ten best heads; row top-1 = the best head of each row, read in that row.\n")
    out.append(tab("ext13ci_attn_population", tag))
    ab = attn_block(s1)
    if ab:
        out.append("**Copy heads vs entity movers (attribute → entity and copy conditions).**\n")
        out.append(ab + "\n")
    # ---------------- step 2
    out.append("### Step 2. Heads + experts added back jointly\n")
    out.append(tab("ext13ci_circuit_curves", tag))
    out.append(tab("ext13ci_circuit_greedy", tag))
    out.append("**In-pass ceilings and sanity.**\n")
    out.append(tab("ext13ci_circuit_sanity", tag))
    out.append("### Cross-condition overview (key panel of the task ladder)\n")
    out.append(f"![overview](../figures/ext13ci_overview{tag}.png)\n")
    out.append(tab("ext13ci_overview", tag))
    out.append("### Contrasts (paired over shared clusters)\n")
    out.append("Per cluster (twin or template seed): r = Σ rescue / Σ drop over its rows, heads = mean count among the first 10 "
               "mixed-greedy picks; difference A − B averaged over the clusters present in both conditions, 95 % percentile "
               "bootstrap over clusters (5,000 resamples). 'P3wino' = Phase 4's WinoGrande option-swap rows (Direction 10).\n")
    out.append(tab("ext13ci_contrasts", tag))
    if DEVIATIONS:
        out.append("### Deviations and caveats\n")
        out.append(DEVIATIONS)
    out.append("### Files\n")
    out.append("Code moetrace/ext13ci.py, scripts/ext13ci_{heads_run,circuit_run,analyze,chain,text}.py; runs results/p5_<cond>_<proto>_ci/ "
               "(step 1), results/p5_<proto>_ci/<group>/ (step 2); numbers results/ext13ci_summary.json; tables "
               "results/tables/ext13ci_*; figures results/figures/ext13ci_{heads_heatmap,heads_dla_layers,heads_attn,circuit_curves,"
               "circuit_composition}_<cond>.png and ext13ci_overview.png.\n")
    out.append(f"_Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by scripts/ext13ci_text.py._\n")
    path = SEC if not tag else os.path.join(RESULTS, f"p5_qwen3_ci{tag}", f"ext13_circuit{tag}.md")  # smoke: not in sections/
    with open(path, "w") as f:
        f.write("\n".join(out))
    print("wrote", path)


if __name__ == "__main__":
    main()
