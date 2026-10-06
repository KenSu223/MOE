"""ext12: write results/sections/ext12_complete.md from results/ext12_complete_summary.json and the ext12 tables
(scripts/ext12_analyze.py must have run).

Usage: python scripts/ext12_complete_text.py
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
from moetrace.models import RESULTS
from moetrace import ext12_complete as C

SEC = os.path.join(RESULTS, "sections", "ext12_complete.md")
TAB = os.path.join(RESULTS, "tables")
S = json.load(open(os.path.join(RESULTS, "ext12_complete_summary.json")))
E8 = json.load(open(os.path.join(RESULTS, "ext8_addback_summary.json")))["runs"]


def tab(name):
    p = os.path.join(TAB, f"{name}.md")
    return open(p).read() if os.path.exists(p) else f"(table {name} not available)\n"


def num(x):
    if x is None:
        return float("nan")
    if isinstance(x, str):
        return float("inf") if x == "inf" else (float("-inf") if x == "-inf" else float(x))
    return float(x)


def f(x, d=3, sign=False):
    v = num(x)
    if not np.isfinite(v):
        return "never" if v > 0 else "n/a"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def ci(x, d=3, sign=False):
    if x is None:
        return "n/a"
    return f"{f(x[0], d, sign)} [{f(x[1], d, sign)}, {f(x[2], d, sign)}]"


def kci(x):
    def k(v):
        v = num(v)
        return "never" if not np.isfinite(v) else str(int(v))
    return f"{k(x[0])} [{k(x[1])}, {k(x[2])}]" if x else "n/a"


def contains0(x):
    return x is not None and num(x[1]) <= 0 <= num(x[2])


def short(key):
    c = C.TASKS[key]
    m = "Qwen3" if c["model"] == "qwen3" else ("Mixtral no BOS" if "nobos" in key else "Mixtral")
    t = "CounterFact" if c["kind"] == "cf_swap" else "WinoGrande"
    return f"{m} {t}"


A = S.get("addback", {})
RUNS = A.get("runs", {})
COMP = A.get("comparisons", {})
MAIN = RUNS.get("_main", {})
REP_KEYS = [k for k in ("wino_qwen3_rep", "cf_qwen3_fold", "wino_mixtral_rep", "cf_mixtral_fold") if k in RUNS]


def b(key):
    return RUNS[key]["boot"]


def mb(key):
    return MAIN[C.TASKS[key]["main_key"]]["boot"]


def d(key):
    return COMP.get(key, {}).get("diff") or {}


# ---------------------------------------------------------------------------------------------------------------
def sec_4a():
    if not REP_KEYS:
        return "### 4a. Add-back replication\n\nNot available yet.\n"
    L = []
    L.append("### 4a. Add-back replication: WinoGrande replication split and CounterFact fold swap\n")
    L.append("**What was run.** The Phase-3 add-back protocol (ext8, `moetrace/ext8_addback.py`, unchanged) on case sets that played "
             "no role in Phase 3's add-back curves. WinoGrande: the disjoint replication split of the 776-pair case set (population ranking "
             "and layer-wise baseline from rep_discovery, curves on the 128 rep_validation pairs = 256 directed cases; all-layer single-expert "
             "rows from the ext7 runs, which already covered the replication split). CounterFact has no replication set: the two paper folds "
             "were exchanged (ranking from the paper's validation cases, curves on its discovery cases: Qwen3 "
             f"{RUNS.get('cf_qwen3_fold', {}).get('n_eval_cases', '?')}, Mixtral {RUNS.get('cf_mixtral_fold', {}).get('n_eval_cases', '?')} cases, "
             "every known donor, donor mean). Per run: A0 ceilings (all MoE / all attention / both, interior layers; add-back and deletion "
             "directions), the static orderings oracle / pop / layer-wise / DLA / |δ_e| / routing weight / random (5 permutations) on ext8's k grid "
             "with the all-clean-active endpoint, and adaptive greedy with ext8's pools (Qwen3 the row's top-32 singles, Mixtral all 64) for 15 steps. "
             "Not run (brief): beam, exact top-10, Shapley, deletion curves. The runs of one model shared every engine pass "
             "(`scripts/ext12_addback_run.py`, a multi-task driver over the ext8 functions: each task keeps its own prefill rows, state and files in "
             "the ext8 format and is analysed with ext8's own `analyse`). CIs: 5,000-resample unit bootstrap (CounterFact cases, WinoGrande pairs); "
             "differences to Phase 3 are between independent samples (difference of independent resamples).\n")
    L.append(tab("ext12_4a_addback"))
    L.append(tab("ext12_4a_differences"))
    L.append("![Add-back curves, new runs (solid) vs Phase 3 (dashed)](../figures/ext12_4a_curves.png)\n")
    # reading
    rd = []
    for key in REP_KEYS:
        n, o = b(key), mb(key)
        g, go = n.get("greedy", {}), o.get("greedy", {})
        dd = d(key)
        rd.append(f"- **{short(key)}** ({C.TASKS[key]['tlabel']}): ceiling {ci(n['ceiling'])} vs {ci(o['ceiling'])} in Phase 3 "
                  f"(difference {ci(dd.get('ceiling'), sign=True)}); answer restored by all MoE {ci(n['restored_all_moe'], 2)} vs {f(o['restored_all_moe'][0], 2)}; "
                  f"k80 of the ceiling oracle / greedy / random {kci(n['oracle']['k80'])} / {kci(g.get('k80'))} / {kci(n['rand']['k80'])} "
                  f"(Phase 3 {kci(o['oracle']['k80'])} / {kci(go.get('k80'))} / {kci(o['rand']['k80'])}); AUC over log k oracle {ci(n['oracle']['auc_log'])} "
                  f"(Phase 3 {f(o['oracle']['auc_log'][0])}), DLA {ci(n['dla']['auc_log'])} ({f(o['dla']['auc_log'][0])}); k ≤ 15 greedy "
                  f"{ci(g.get('auc_log_k15'))} ({f(go.get('auc_log_k15', [None])[0])}); greedy − static oracle at k = 10 {ci(n.get('greedy_minus_oracle_r10'), sign=True)} "
                  f"({f(o.get('greedy_minus_oracle_r10', [None])[0], sign=True)}); DLA − oracle AUC {ci(n.get('dla_minus_oracle_auc'), sign=True)} "
                  f"({f(o.get('dla_minus_oracle_auc', [None])[0], sign=True)}); overshoot of the oracle curve over the ceiling {ci(n['oracle']['overshoot'], sign=True)} "
                  f"({f(o['oracle']['overshoot'][0], sign=True)}).")
    L.append("**Reading.**\n")
    L.append("\n".join(rd) + "\n")
    L.append(READ_4A)
    return "\n".join(L)


def sec_4b():
    R = S.get("role_vs_option")
    if not R:
        return "### 4b. Role swap vs option swap on identical items\n\nNot available yet.\n"
    L = ["### 4b. Role swap vs option swap on identical items\n"]
    q = R.get("qwen3", {}).get("all", {})
    m = R.get("mixtral_bos", {}).get("all", {})
    L.append("**What was run.** Every W7 role-swap pair was built inside its model's option-swap margin pool, so each role-swap item (a "
             "WinoGrande twin) also has an option-swap pair (A, B) (`scripts/ext12_roleitems_casesets.py` → `results/wino_roleitems_inputs/`; "
             "check: the option pair passes the margin both ways, and the role pair's clean prompt and (r, r′) equal those of one option-swap "
             f"direction). Items: Qwen3 {q.get('n_twins', '?')} twins (all four role splits: {q.get('n_role_cases', '?')} role and {q.get('n_opt_cases', '?')} "
             f"option directed cases), Mixtral BOS {m.get('n_twins', '?')} twins ({m.get('n_role_cases', '?')} / {m.get('n_opt_cases', '?')}). On exactly "
             "these items the option swap was re-run with the Phase-3 runners: W2 final-position sweep (`layer`, `attn_layer`, `block` at every layer), "
             "W4 joint patch (all-MoE M, both directions) and the direct-path split (exact final norm); the role-swap side is the Phase-3 run "
             "(`results/wino_role_*_str`). Comparison unit = twin: the role-swap value of a twin pools its one or two role pairs (both directions), "
             "the option-swap value its option pair (both directions); the attention share is the Direction-2b AUC+ share of the twin-weighted mean "
             "single-layer curves; ratios (A_dir, M) are ratios of twin means; 5,000-resample twin bootstrap with the SAME resamples for both "
             "corruptions (paired). Sensitivity: the clean-prompt-matched variant (role direction d = 0 against the option-swap direction with the "
             "same clean prompt and the same r, r′; only the corruption differs) and the validation split alone (the Phase-3 comparison set).\n")
    L.append(tab("ext12_4b_role_vs_option"))
    L.append(tab("ext12_4b_peaks"))
    L.append("![Role swap vs option swap on identical items](../figures/ext12_4b_role_vs_option.png)\n")
    rd = []
    for proto, lab in (("qwen3", "Qwen3"), ("mixtral_bos", "Mixtral BOS")):
        e = R.get(proto)
        if not e:
            continue
        x, v, mm = e["all"], e["validation"], e["all_matched"]
        p3 = e.get("phase3_option_main_validation_share") or {}
        rd.append(f"- **{lab}**: attention share role {ci(x['share_role'])} vs option {ci(x['share_option'])} on the same {x['n_twins']} twins, "
                  f"paired difference {ci(x['share_diff'], sign=True)} (validation split only {ci(v['share_diff'], sign=True)}; clean-prompt matched "
                  f"{ci(mm['share_diff'], sign=True)}). Phase 3 compared the role-swap validation pairs with the option swap's own 776-pair validation set "
                  f"(option share {f(p3.get('share'))}). Direct-path attention share A_dir role {ci(x.get('A_direct_role'))} vs option {ci(x.get('A_direct_option'))}, "
                  f"difference {ci(x.get('A_direct_diff'), sign=True)}; all-MoE M role {ci(x.get('M_all_moe_role'))} vs option {ci(x.get('M_all_moe_option'))}, "
                  f"difference {ci(x.get('M_all_moe_diff'), sign=True)}. Attention AUC+ / drop {ci(x['auc_attn_norm_diff'], sign=True)}, MoE AUC+ / drop "
                  f"{ci(x['auc_moe_norm_diff'], sign=True)} (role − option).")
    L.append("**Reading.**\n")
    L.append("\n".join(rd) + "\n")
    L.append(READ_4B)
    return "\n".join(L)


def sec_4c():
    N = S.get("nobos")
    if not N:
        return "### 4c. Mixtral without BOS on WinoGrande\n\nNot available yet.\n"
    L = ["### 4c. Mixtral without BOS on WinoGrande (the base reproduction's paper protocol)\n"]
    L.append("**What was run.** The same 776-pair case set (main 128/128 and replication 128/128 pairs, both directions) with the pairs "
             "tokenised without `<s>` (`data/wino_str/pairs_train_xl_mixtral_nobos.parquet`; identical text and token positions, no BOS), run dir "
             "`results/wino_mixtral_nobos_str`: W2 final-position sweep, W6 all-layer expert pass with equal-norm rows at the MoE argmax layer and "
             "the paper's two-stage selection (recurrence gate 128 of 256 discovery directed cases), joint search, fixed hypotheses; W4 all-MoE / "
             "all-attention / all-block joint patches and the direct-path split; add-back (static orderings + greedy, main validation, "
             "`results/wino_mixtral_nobos_str_addback`); sink-carrying final-token flags (Direction 3: the final position holds the maximal residual "
             "norm at layer 5) from one prefill pass over both prompts of every pair under both protocols (`scripts/ext12_sink_flags.py`). "
             "All comparisons with the BOS run (Phase 3, `results/wino_mixtral_bos_str`, `results/wino_mixtral_bos_str_addback`) are on identical "
             "pairs, so CIs of differences come from a paired pair bootstrap.\n")
    L.append(tab("ext12_4c_w2"))
    L.append(tab("ext12_4c_vs_bos"))
    L.append(tab("ext12_4c_w6"))
    L.append(tab("ext12_4c_fixed_experts"))
    L.append(tab("ext12_4c_sink"))
    L.append("![Mixtral without vs with BOS, final-position curves](../figures/ext12_4c_w2_nobos_vs_bos.png)\n")
    L.append("**Reading.**\n")
    L.append(READ_4C)
    return "\n".join(L)


READ_4A = READ_4B = READ_4C = ""


def main():
    global READ_4A, READ_4B, READ_4C
    SUMMARY, READ_4A, READ_4B, READ_4C, CAVEATS, FILES = reading()
    txt = "\n".join([SUMMARY, "", sec_4a(), sec_4b(), sec_4c(), CAVEATS, FILES])
    os.makedirs(os.path.dirname(SEC), exist_ok=True)
    with open(SEC, "w") as fh:
        fh.write(txt)
    print(f"wrote {SEC} ({len(txt)} chars)")


# ---------------------------------------------------------------------------------------------------------------
# reading paragraphs (numbers from the summary JSON; wording checked against the results)
# ---------------------------------------------------------------------------------------------------------------
def g(dct, *keys, default=None):
    x = dct
    for k in keys:
        if isinstance(x, dict) and k in x:
            x = x[k]
        else:
            return default
    return x


def gpu_minutes():
    """GPU minutes of this agent's jobs (ext12-*) from logs/gpu_queue.log (END lines, seconds in parentheses)."""
    import re
    tot, per = 0.0, {}
    for line in open("/home/ubuntu/MOE/logs/gpu_queue.log"):
        m = re.search(r"END\s+(ext12-\S+) rc=(\d+) \((\d+)s\)", line)
        if m:
            name, sec = m.group(1), int(m.group(3))
            grp = "smoke" if "smoke" in name else ("4a/4c add-back" if "addback" in name else ("4b" if "-4b-" in name else "4c inputs"))
            per[grp] = per.get(grp, 0) + sec / 60
            tot += sec / 60
    return tot, per


def qual_4a_short():
    lw = [(k, b(k).get("layerwise_minus_pop_auc")) for k in REP_KEYS if b(k).get("layerwise_minus_pop_auc")]
    tied = [short(k) for k, x in lw if not num(x[2]) < 0]
    return ("The paper's layer-first order stays below the population ranking" + (f" except in {', '.join(tied)} (tied)" if tied else "") + ".")


def picks():
    out = []
    for key in REP_KEYS + (["wino_mixtral_nobos"] if "wino_mixtral_nobos" in RUNS else []):
        new = dict((a, v) for a, v in g(RUNS[key], "ext8_analyse", "greedy", "in_first5_top", default=[]))
        old = dict((a, v) for a, v in g(E8, C.TASKS[key]["main_key"], "greedy", "in_first5_top", default=[]))
        top = list(new)[:3]
        out.append(f"{short(key)} " + ", ".join(f"{e} {new[e]:.2f} ({old[e]:.2f})" if e in old else f"{e} {new[e]:.2f} (–)" for e in top))
    return "; ".join(out)


def qual_4a():
    """Data-driven check of the qualitative Phase-3 add-back claims on the new runs."""
    out = []
    dl = [(k, num(b(k)["dla_minus_oracle_auc"][0])) for k in REP_KEYS if "dla_minus_oracle_auc" in b(k)]
    if dl:
        worst = min(v for _, v in dl)
        out.append("the patch-free DLA ranking is " + ("at or above the single-patch oracle in every run" if worst >= -0.005 else
                   f"within {abs(worst):.2f} of the single-patch oracle or above it (AUC over log k; lowest: {short(min(dl, key=lambda x: x[1])[0])})"))
    gb = [(k, b(k).get("greedy_minus_best_static_r10")) for k in REP_KEYS if b(k).get("greedy_minus_best_static_r10")]
    if gb:
        pos = [k for k, x in gb if num(x[1]) > 0]
        out.append("adaptive greedy beats the best static ordering at k = 10 in " + (f"all {len(gb)} runs" if len(pos) == len(gb) else
                   f"{len(pos)} of {len(gb)} runs") + " (" + ", ".join(f"{short(k)} {ci(x, sign=True)}" for k, x in gb) + ")")
    ov = [(k, b(k)["oracle"]["overshoot"]) for k in REP_KEYS if C.TASKS[k]["kind"] == "cf_swap"]
    if ov:
        out.append("on CounterFact the oracle subsets again overshoot the all-MoE ceiling (" + ", ".join(f"{short(k)} {ci(x, sign=True)}" for k, x in ov) + ")")
    lw = [(k, b(k).get("layerwise_minus_pop_auc")) for k in REP_KEYS if b(k).get("layerwise_minus_pop_auc")]
    if lw:
        below = [k for k, x in lw if num(x[2]) < 0]
        out.append("the paper's layer-first order stays below the population ranking " + (f"in all {len(lw)} runs" if len(below) == len(lw) else
                   f"in {len(below)} of {len(lw)} runs") + " (AUC difference " + ", ".join(f"{short(k)} {ci(x, sign=True)}" for k, x in lw) + ")")
    return ("On the new cases " + "; ".join(out) + ".") if out else ""


def reading():
    R4b = S.get("role_vs_option", {})
    N = S.get("nobos", {})
    tot, per = gpu_minutes()
    # ---------------- 4a
    lines_a = []
    for key in REP_KEYS:
        n, o, dd = b(key), mb(key), d(key)
        ok = all(contains0(dd.get(k)) for k in ("ceiling",)) and contains0(g(dd, "oracle", "auc_log")) and contains0(g(dd, "dla", "auc_log"))
        lines_a.append((key, n, o, dd, ok))
    rep_ok = all(x[4] for x in lines_a)

    def ab(key, n, o):
        gg = n.get("greedy", {})
        return (f"{short(key)} ceiling {f(n['ceiling'][0])} (Phase 3 {f(o['ceiling'][0])}), k80 oracle / greedy / random "
                f"{kci(n['oracle']['k80'])[:kci(n['oracle']['k80']).index(' ')]} / {kci(gg.get('k80'))[:kci(gg.get('k80')).index(' ')] if gg else 'n/a'} / "
                f"{kci(n['rand']['k80'])[:kci(n['rand']['k80']).index(' ')]}")

    s_a = "; ".join(ab(k, n, o) for k, n, o, _, _ in lines_a)
    # ---------------- 4b
    s_b = []
    for proto, lab in (("qwen3", "Qwen3"), ("mixtral_bos", "Mixtral BOS")):
        x = g(R4b, proto, "all")
        if x:
            s_b.append(f"{lab} {f(x['share_role'][0])} vs {f(x['share_option'][0])} (paired difference {ci(x['share_diff'], sign=True)}, "
                       f"{x['n_twins']} twins; direct-path attention {f(g(x, 'A_direct_role', default=[None])[0])} vs {f(g(x, 'A_direct_option', default=[None])[0])})")
    # ---------------- 4c
    w6m = g(N, "w6", "main", "two_stage") or {}
    ev = w6m.get("eval") or {}
    eq = w6m.get("equal_norm") or {}
    vb = g(N, "vs_bos", "main_validation") or {}
    sk = N.get("sink") or {}
    s_c = ""
    if ev:
        s_c = (f"Without BOS the two-stage selection picks L{w6m['layer']} and **L{w6m['layer']}E{int(w6m['selection']['e_star']):03d}** again "
               f"(validation rescue {ci([ev['rescue']['mean'], ev['rescue']['ci_lo'], ev['rescue']['ci_hi']], sign=True)}, Spec "
               f"{ci([ev['spec']['mean'], ev['spec']['ci_lo'], ev['spec']['ci_hi']], sign=True)}, equal-norm Spec "
               f"{ci([eq['spec_eq']['mean'], eq['spec_eq']['ci_lo'], eq['spec_eq']['ci_hi']], sign=True) if eq.get('spec_eq') else 'n/a'}; "
               f"pattern {w6m.get('pattern')}; re-selected on the replication split)")
    if vb.get("share"):
        s_c += (f", and the attention/MoE balance does not move: attention share {f(vb['share']['nobos'][0])} vs {f(vb['share']['bos'][0])} with BOS "
                f"(paired difference {ci(vb['share']['diff'], sign=True)})")
        if vb.get("M_all_moe"):
            s_c += f", all-MoE M {f(vb['M_all_moe']['nobos'][0])} vs {f(vb['M_all_moe']['bos'][0])} ({ci(vb['M_all_moe']['diff'], sign=True)})"
        if vb.get("A_direct"):
            s_c += f", direct-path attention {f(vb['A_direct']['nobos'][0])} vs {f(vb['A_direct']['bos'][0])} ({ci(vb['A_direct']['diff'], sign=True)})"
    if sk:
        s_c += (f". Only {sk['nobos']['final_is_max_L5']} of {sk['nobos']['n_prompts']} WinoGrande prompts carry the sink state at their final token "
                f"without BOS ({sk['bos']['final_is_max_L5']} with BOS; CounterFact in Direction 3: 63 of 256)")
    if "wino_mixtral_nobos" in RUNS:
        n, o, dd = b("wino_mixtral_nobos"), mb("wino_mixtral_nobos"), d("wino_mixtral_nobos")
        s_c += (f"; add-back ceiling {f(n['ceiling'][0])} vs {f(o['ceiling'][0])} (paired {ci(dd.get('ceiling'), sign=True)}), greedy k80 "
                f"{kci(g(n, 'greedy', 'k80'))} vs {kci(g(o, 'greedy', 'k80'))}")
    s_c += "."
    def kk(x):
        v = num(x[0]) if x else float("nan")
        return "never" if not np.isfinite(v) else str(int(v))
    order = [k for k in ("wino_qwen3_rep", "cf_qwen3_fold", "wino_mixtral_rep", "cf_mixtral_fold") if k in RUNS]
    lab = " / ".join(short(k) for k in order)
    ceil_new = " / ".join(f(b(k)["ceiling"][0]) for k in order)
    ceil_old = " / ".join(f(mb(k)["ceiling"][0]) for k in order)
    dceil = ", ".join(ci(d(k).get("ceiling"), sign=True) for k in order)
    kg_new = " / ".join(kk(g(b(k), "greedy", "k80")) for k in order)
    kg_old = " / ".join(kk(g(mb(k), "greedy", "k80")) for k in order)
    kr = " / ".join(kk(b(k)["rand"]["k80"]) for k in order)
    dla = " / ".join(f(b(k)["dla_minus_oracle_auc"][0], 2, sign=True) for k in order)
    gm = " / ".join(f(b(k)["greedy_minus_best_static_r10"][0], 2, sign=True) for k in order)
    SUMMARY = ("**Summary.** Three completeness checks of the Phase-3 claims (Direction 12; no engine change; ≈ "
               f"{tot:.0f} GPU-min). The expert-level claims replicate; the Phase-3 reading that the WinoGrande role swap is more "
               "attention-driven than the option swap does not survive identical items.\n\n"
               f"- **4a, add-back replication.** On cases Phase 3 did not use (WinoGrande replication split; CounterFact with the paper folds "
               f"swapped) the all-MoE ceilings are {ceil_new} of the drop ({lab}; Phase 3 {ceil_old}; differences {dceil}, all CIs contain 0). "
               f"80 % of the ceiling still takes {kg_new} experts with adaptive greedy (Phase 3 {kg_old}) against {kr} in random order; greedy "
               f"beats the best static ordering at k = 10 by {gm}; the patch-free DLA ranking is within 0.03 of the single-patch oracle or above it "
               f"(AUC difference {dla}); oracle subsets again overshoot the CounterFact ceiling; greedy's first picks are the Phase-3 experts. "
               + qual_4a_short() + "\n"
               "- **4b, role swap vs option swap on identical items.** On the same WinoGrande twins the two corruption sites give the same "
               "final-position decomposition: attention share role vs option " + "; ".join(s_b) + ". The role-swapped prompt of every item is the "
               "option-swapped prompt with the two names exchanged (verified on all items), so for name pairs W7 is the option swap up to a "
               "renaming that both models nearly ignore. The Phase-3 gap (Qwen3 0.32 vs 0.16, Mixtral 0.39 vs 0.34) compared name items with "
               "mostly object items: name items lean towards attention under either corruption.\n"
               f"- **4c, Mixtral without BOS on WinoGrande.** {s_c}\n")
    READ_4A = ("The Phase-3 add-back conclusions hold on independent cases. (1) The all-MoE ceiling is a property of the task, not of the "
               "particular 128 pairs or ~107 facts: replication / fold-swap ceilings differ from Phase 3 by at most 0.03 (every CI of the difference "
               "contains 0), and CounterFact stays far below WinoGrande (≈ 0.4–0.5 vs ≈ 0.75–0.85 of the drop; all MoE outputs restore the answer in "
               "≈ 40 % of facts vs ≈ 90–95 % of WinoGrande cases). (2) A handful of experts still carries the expert repair: k80 of the ceiling with "
               "greedy is 4–7 experts except Qwen3 WinoGrande (12 vs 10 in Phase 3), the per-case static oracle needs 5–8 except Qwen3 WinoGrande (48 "
               "in both), and random orders still need most of the clean-active experts. (3) Adaptivity matters most where Phase 3 found expert "
               "interactions (Qwen3 WinoGrande: greedy − static oracle at k = 10 +0.10 vs +0.12) and still adds +0.04–0.09 elsewhere. (4) The "
               "patch-free DLA ranking stays above the single-patch oracle in Qwen3 (+0.03 to +0.05 AUC), equal to it in Mixtral CounterFact and "
               "0.02 below it in Mixtral WinoGrande, exactly the Phase-3 pattern. (5) With the population ranking taken from the other fold, the "
               "pop and layer-wise curves are out-of-sample in both directions and keep their Phase-3 positions (oracle ≈ DLA > pop ≥ layer-wise > "
               "|δ_e|, weight > random); only in Mixtral CounterFact are pop and layer-wise tied on the fold swap, as they nearly were in Phase 3 "
               "(0.307 vs 0.292). (6) Oracle subsets overshoot the CounterFact ceiling again by +0.06 / +0.09 of the drop. (7) Greedy picks the "
               "same experts: share of rows with the expert among greedy's first five picks, new run (Phase 3): " + picks() + ".\n")
    def ren(proto, k):
        return g(R4b, proto, "renaming", k, default=float("nan"))
    diffs = [g(R4b, p_, "all", "share_diff") for p_ in ("qwen3", "mixtral_bos") if g(R4b, p_, "all", "share_diff")]
    maxabs = max(abs(num(x[0])) for x in diffs) if diffs else float("nan")
    READ_4B = ("The role-swap pairs were built only from name items (two person names, each mentioned once before the blank). For such an item "
               "the role swap (exchange the two first mentions, keep the filled name) and the option swap (fill the blank with the other name) "
               "produce the same corrupted sentence up to exchanging the two names (verified on every item: "
               f"{ren('qwen3', 'renamed_role_corrupt_equals_option_corrupt')}/{ren('qwen3', 'n_matched_cases')} Qwen3, "
               f"{ren('mixtral_bos', 'renamed_role_corrupt_equals_option_corrupt')}/{ren('mixtral_bos', 'n_matched_cases')} Mixtral clean-prompt-matched cases), "
               "and both keep the clean prompt and (r, r′). The models are close to equivariant under that renaming: per case (clean-prompt matched), "
               f"the single-layer rescue curves of the two corruptions correlate at r = {ren('qwen3', 'r_case_layer_layer'):.2f} (MoE) / "
               f"{ren('qwen3', 'r_case_layer_attn_layer'):.2f} (attention) in Qwen3 and {ren('mixtral_bos', 'r_case_layer_layer'):.2f} / "
               f"{ren('mixtral_bos', 'r_case_layer_attn_layer'):.2f} in Mixtral, the drops at r = {ren('qwen3', 'r_drop'):.2f} / {ren('mixtral_bos', 'r_drop'):.2f}, "
               f"and the population curves coincide (largest share difference {maxabs:.3f}). The Phase-3 reading \"the role swap moves the balance "
               "towards attention\" is therefore withdrawn: what moved the balance was the item type. Name items are the attention-leaning stratum "
               "under either corruption (in the 776-pair option-swap set its 23 name pairs already had attention share 0.30 in Qwen3 and 0.43 in "
               "Mixtral, against 0.16 / 0.34 overall), object items are not. A role swap that is not a renaming of the option swap would need object "
               "items with two swappable mentions, which W7 excluded; Zhang & Nanda's corruption-site question (Z7) is therefore not yet answered "
               "for WinoGrande.\n")
    fx = {(x["layer"], x["expert"]): x for x in (g(N, "w6", "main", "cf_experts") or [])}
    e6 = fx.get((19, 6))
    sk = N.get("sink") or {}
    ra = sk.get("routing_agreement_L19") or {}
    st = sk.get("strata") or []
    s_sink = ""
    if sk:
        tok = g(sk, "nobos", "final_tokens_flagged_str", default={}) or {}
        tok = ", ".join(f"'{k}'" for k in tok if k != "error")
        pos = g(sk, "nobos", "max_norm_position_L5_top5", default={}) or {}
        s_sink = (f"Sink flags: without BOS the final position carries the maximal residual norm at layer 5 in only {sk['nobos']['final_is_max_L5']} of "
                  f"{sk['nobos']['n_prompts']} WinoGrande prompts" + (f" (one pair whose sentence ends in {tok})" if tok else "") +
                  f", with BOS in {sk['bos']['final_is_max_L5']}, against 63 of 256 CounterFact prompts in Direction 3. Without BOS the massive-norm "
                  f"state sits on an early token instead (position 0 in {sk['nobos']['pos0_is_max_L5']} prompts; most often positions "
                  f"{', '.join(str(k) for k in list(pos)[:4])}; median relative position {g(sk, 'nobos', 'max_norm_position_L5_rel_median', default=float('nan')):.2f}), "
                  f"and the final position's attention mass on position 0 at layer 1 is {sk['nobos']['pos0_mass_L1']:.2f} (BOS {sk['bos']['pos0_mass_L1']:.2f}). ")
        r19 = g(sk, "routing_L19_flagged", "nobos", default={}) or {}
        if r19:
            s_sink += (f"The flagged prompts are routed to {{{', '.join('E' + str(k).zfill(3) for k in r19)}}} at L19, the `<s>` routing of Direction 3. ")
        if ra:
            s_sink += (f"Final-position routing agrees between the protocols (identical top-2 set) in {ra['all']:.2f} of the prompts at L19 and "
                       f"{g(sk, 'routing_agreement_L20', 'all', default=float('nan')):.2f} at L20 (mean over layers "
                       f"{sk.get('routing_set_agreement_mean', float('nan')):.2f}). ")
        nb_ = [x for x in st if x.get("stratum", "").startswith("neither")]
        if nb_ and nb_[0].get("sel_spec"):
            x = nb_[0]
            s_sink += (f"Excluding the flagged pair changes nothing (L20E000 Spec {ci([x['sel_spec']['mean'], x['sel_spec']['ci_lo'], x['sel_spec']['ci_hi']], sign=True)} "
                       f"on the {x['n_pairs']} remaining main pairs). ")
    w4 = g(N, "vs_bos", "main_validation") or {}
    s_w4 = ""
    if w4.get("M_all_moe"):
        s_w4 += (f"All-MoE patch M {ci(w4['M_all_moe']['nobos'])} vs {ci(w4['M_all_moe']['bos'])} with BOS (paired difference "
                 f"{ci(w4['M_all_moe']['diff'], sign=True)}). ")
    if w4.get("A_direct"):
        s_w4 += (f"Direct-path attention share {ci(w4['A_direct']['nobos'])} vs {ci(w4['A_direct']['bos'])} (paired difference "
                 f"{ci(w4['A_direct']['diff'], sign=True)}). ")
    s_ab = ""
    if "wino_mixtral_nobos" in RUNS:
        n, o, dd = b("wino_mixtral_nobos"), mb("wino_mixtral_nobos"), d("wino_mixtral_nobos")
        s_ab = (f"Add-back without BOS: ceiling {ci(n['ceiling'])} vs {ci(o['ceiling'])} (paired {ci(dd.get('ceiling'), sign=True)}), "
                f"k80 oracle / greedy / random {kci(n['oracle']['k80'])} / {kci(g(n, 'greedy', 'k80'))} / {kci(n['rand']['k80'])} (BOS "
                f"{kci(o['oracle']['k80'])} / {kci(g(o, 'greedy', 'k80'))} / {kci(o['rand']['k80'])}), AUC over log k oracle "
                f"{f(n['oracle']['auc_log'][0])} vs {f(o['oracle']['auc_log'][0])}, DLA − oracle {f(n['dla_minus_oracle_auc'][0], sign=True)} vs "
                f"{f(o['dla_minus_oracle_auc'][0], sign=True)}, greedy − static oracle at k = 10 {f(n['greedy_minus_oracle_r10'][0], sign=True)} vs "
                f"{f(o['greedy_minus_oracle_r10'][0], sign=True)}: the curves are the BOS curves within ≈ 0.01. ")
    READ_4C = ("Without BOS the WinoGrande final-position picture is unchanged: the same layers (MoE L20, attention L13, block L19), the same "
               "selected expert with the same specificity and equal-norm specificity, re-selected on the replication split, the same joint "
               "top-3 (L20E000, L19E006, L21E006), and attention / MoE shares within 0.01. " + s_w4 + s_ab + s_sink + " This is what Direction 3's "
               "mechanism predicts: the BOS effect on CounterFact came from short cloze prompts whose final preposition became the attention sink, "
               "whereas WinoGrande prompts are full sentences (median 19 tokens without BOS) with delimiters and function words long before the final "
               "position. L19E006, the Direction-3 'sink expert', is routed at the WinoGrande final position in "
               + (f"{e6['val_active']} of 256 validation directed cases without BOS" if e6 else "nearly every case")
               + (f" and is positively specific (Spec {ci([e6['spec']['mean'], e6['spec']['ci_lo'], e6['spec']['ci_hi']], sign=True)})" if e6 else "")
               + ", as with BOS: on WinoGrande it is an ordinary content expert (among greedy's first five picks in "
               + f"{dict(g(RUNS, 'wino_mixtral_nobos', 'ext8_analyse', 'greedy', 'in_first5_top', default=[])).get('L19E006', float('nan')):.0%} of rows, "
               "as often as L20E000), "
               "not a sink artefact. The CounterFact "
               "experts (L19E002, L21E001, L18E001) stay idle on WinoGrande without BOS as with BOS. Answer to the brief: L20E000 survives without "
               "BOS, and the attention/MoE balance does not change.\n")
    CAVEATS = ("### Caveats\n\n"
               "- 4a: the replication split and the fold swap are new evaluation cases but the same models, tasks and protocol; beam / exact / "
               "Shapley / deletion were not repeated (brief). Greedy and the oracle ordering use each row's own single-expert patches (in-sample), "
               "as in Phase 3. The fold swap evaluates on the paper's discovery fold, on which the Direction-6 single-expert rows were selected "
               "(layer and expert selection, not the add-back orderings); no add-back quantity was tuned on it.\n"
               "- 4a / 4c passes were shared between tasks of one model (multi-task driver); bf16 batch composition therefore differs from Phase 3 "
               "(run-to-run noise of identical rows ≈ 0.06–0.17 logits median; differences below ≈ 0.01 of the drop are not resolved).\n"
               "- Engine: `moetrace/engine.py` was replaced by the ext9 engine at 06:58Z (E4 additions; regression identical in every result "
               "field). The 4c sweep / expert passes, the Qwen3 4b sweep / joint and the first two Qwen3 add-back jobs ran on the ext8 engine, all "
               "later jobs on the ext9 engine; no kind used here changed.\n"
               "- 4b compares twin-weighted curves (a twin with two role pairs counts once); the Phase-3 pair-weighted shares on the same items are "
               "in the peaks table and agree to ≤ 0.01. Role swaps are name items only; the conclusion is about name items.\n"
               "- 4c: the sink flag is Direction 3's (final position = max residual norm at layer 5); the BOS comparison is on identical pairs, but "
               "the BOS add-back run is Phase 3's (different pass composition).\n")
    FILES = ("### Files\n\n"
             "- Code: `moetrace/ext12_complete.py` (task registry, CounterFact fold-swap loader, ext12 work queue), `scripts/ext12_addback_run.py` "
             "(multi-task add-back driver over the ext8 functions), `scripts/ext12_roleitems_casesets.py` (4b case sets and links), "
             "`scripts/ext12_sink_flags.py` (4c sink flags), `scripts/ext12_analyze.py`, `scripts/ext12_complete_text.py`, chains "
             "`scripts/ext12_chain_inputs.sh`, `scripts/ext12_chain_addback.sh` (logs `logs/ext12_chain_*.log`).\n"
             "- Runs: `results/wino_qwen3_str_addback_rep`, `results/wino_mixtral_bos_str_addback_rep`, `results/qwen3_str_addback_fold_rep`, "
             "`results/mixtral_bos_str_addback_fold_rep`, `results/wino_mixtral_nobos_str_addback` (ext8 file format), `results/wino_mixtral_nobos_str` "
             "(W2 / W6 / W4 / direct split / `sink_flags.parquet`, `sink_diag.json`), `results/wino_roleitems_{qwen3,mixtral_bos}_str` (option swap on "
             "the role items), `results/wino_roleitems_inputs/` (case sets, link tables).\n"
             "- Numbers `results/ext12_complete_summary.json`; tables `results/tables/ext12_*`; figures `results/figures/ext12_*`.\n"
             f"- GPU: ≈ {tot:.0f} min in ext12 jobs (" + ", ".join(f"{k} {v:.0f}" for k, v in sorted(per.items())) + ").\n")
    return SUMMARY, READ_4A, READ_4B, READ_4C, CAVEATS, FILES


if __name__ == "__main__":
    main()
