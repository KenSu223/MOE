"""ext11 section writer: results/sections/ext11_writer.md from results/ext11_writer_summary.json and the ext11 tables.

Usage: python scripts/ext11_writer_text.py
"""
import json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace.models import RESULTS
from moetrace import ext11_writer as W

SEC = os.path.join(RESULTS, "sections", "ext11_writer.md")
TAB = os.path.join(RESULTS, "tables")


def tab(name):
    p = os.path.join(TAB, name + ".md")
    return open(p).read().strip() if os.path.exists(p) else f"(table {name} missing)"


def c(d, digits=2, sign=False):
    if not d or d.get("v") is None or not np.isfinite(d["v"]):
        return "n/a"
    f = f"{{:{'+' if sign else ''}.{digits}f}}"
    return f"{f.format(d['v'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


def ex(r, name):
    for x in r["experts_val"]:
        if x["expert"] == name:
            return x
    return None


def crossover(r, thr=0.02):
    """First layer l (sum T / drop > thr) with share_l >= 0.95 and pooled share >= 1 over the positive layers l .. L-4."""
    L = r["L"]
    lay = [x for x in r["layers_all"] if x["T_frac"]["v"] > thr and x["layer"] <= L - 4]
    for i, x in enumerate(lay):
        tail = lay[i:]
        pooled = sum(y["D_frac"]["v"] for y in tail) / sum(y["T_frac"]["v"] for y in tail)
        if x["share"]["v"] >= 0.95 and pooled >= 1:
            return x["layer"]
    return None


def part_a(S):
    A = S["A"]
    K = ["cf_qwen3", "wino_qwen3", "cf_mixtral", "wino_mixtral"]
    lines = []
    lab = lambda k: f"{A[k]['label'].split('-')[0]} {A[k]['tlabel']}"
    tg = {"wino_qwen3": "L41E117", "cf_qwen3": ["L44E069", "L42E115"], "wino_mixtral": "L20E000",
          "cf_mixtral": ["L19E002", "L21E001", "L18E001"]}
    return A, K, lab, tg


def summary_paragraph(S):
    A, K, lab, tg = part_a(S)
    q_wg = ex(A["wino_qwen3"], "L41E117")
    q69 = ex(A["cf_qwen3"], "L44E069")
    q115 = ex(A["cf_qwen3"], "L42E115")
    m_wg = ex(A["wino_mixtral"], "L20E000")
    m2 = ex(A["cf_mixtral"], "L19E002")
    m21 = ex(A["cf_mixtral"], "L21E001")
    m18 = ex(A["cf_mixtral"], "L18E001")
    s = ("**Summary.** Splitting each single-expert STR patch at the final position into its direct path (the expert's write "
         "δ_e projected on W_U[r] − W_U[r′], final norm frozen, as ext8's DLA) and the rest (total − direct = everything later "
         "layers do with it) shows that the Qwen3 experts are **writers** and the Mixtral experts **partly writers** (half to most of "
         "their effect is direct). Direct share "
         f"(Σ direct / Σ total over the validation cases where the expert is routed): Qwen3 WinoGrande L41E117 **{c(q_wg['share'])}**, "
         f"CounterFact L42E115 **{c(q115['share'])}** and L44E069 **{c(q69['share'])}** (its write is larger than its net effect: "
         f"later layers undo {abs(q69['I']['v']):.2f} of {q69['D']['v']:.2f} logits); Mixtral WinoGrande L20E000 **{c(m_wg['share'])}**, "
         f"CounterFact L19E002 {c(m2['share'])}, L21E001 {c(m21['share'])}, L18E001 {c(m18['share'])}. ")
    # layer pattern
    bands = []
    for k in K:
        b = A[k]["agreement_all"]
        bands.append(f"{lab(k)} {b['middle']['share']['v']:.2f} / {b['late']['share']['v']:.2f}")
    s += ("Over all clean-active experts the direct share grows with depth (middle band / late band: " + "; ".join(bands) +
          "); experts in the first half of the network contribute ≤ 0.03 of the drop (Mixtral WinoGrande 0.11), almost only "
          "indirectly, and above a crossover layer ("
          + ", ".join(f"{lab(k)} L{crossover(A[k])}" for k in K if crossover(A[k]) is not None) +
          ") experts write more than their net effect, so downstream layers partly cancel late writes. ")
    s += ("The patch-free DLA ranks experts well because the largest total effects belong to writers: the top-10 experts by "
          "direct and by total effect overlap in " + ", ".join(f"{A[k]['ranking_val']['top_overlap']}/10 ({lab(k)})" for k in K) +
          ". ")
    if "C" in S:
        s += S.get("_c_summary", "")
    if "B" in S:
        s += S.get("_b_summary", "")
    fz = [A[k]["frozen_norm_aggregate"]["moe"] for k in K]
    s += ("Freezing the final norm is harmless at these scales: for the summed MoE writes the exact-norm and frozen-norm direct "
          "effects differ by ≤ " + f"{max(abs(x['exact_frac'] - x['frozen_frac']) for x in fz):.3f}" + " of the drop on the mean "
          "(median per-row |error| " + f"{min(x['median_abs_err_frac_of_drop'] for x in fz):.3f}–{max(x['median_abs_err_frac_of_drop'] for x in fz):.3f}" +
          "), and at the last layer, where total − direct is the norm error alone, total ≈ direct (slope "
          + ", ".join(f"{A[k]['agreement_all']['last']['corr']['slope_xy']:.2f}" for k in K) + ").")
    return s


def build():
    S = json.load(open(W.SUMMARY))
    A, K, lab, tg = part_a(S)
    out = []
    if "C" in S:
        S["_c_summary"] = c_summary(S)
    if "B" in S:
        S["_b_summary"] = b_summary(S)
    out.append(summary_paragraph(S))
    out.append("")
    out.append("### What was run")
    out.append("")
    out.append(
        "**Part A (CPU, existing rows).** For every STR row of the four ext8 add-back tasks (CounterFact STR: Direction-6 cases and donors; "
        "WinoGrande STR: the 512 directed cases of the main family) and every clean-active (layer, expert) at the final position: "
        "total T = single-expert patch rescue (ext6 / ext7 `str_expert_rows`, parent = corrupted run, c_e := c_e(clean)), direct "
        "D = ext8's DLA of δ_e = c_e(clean) − c_e(corrupt), (δ_e ⊙ γ)·(W_U[r] − W_U[r′]) / rms(h_final, corrupted run) "
        "(`addback_dla.parquet`, pass 0), indirect I = T − D. I collects everything that happens after layer l at the final "
        "position (later attention reads the changed residual through the final token's own query, key and value; later routers and "
        "experts see a different input) plus the final-norm nonlinearity, which D ignores. At the last layer there is no later "
        "computation, so T − D there measures the frozen-norm error plus pass noise. CounterFact values are donor means per case, "
        "bootstrap over cases; WinoGrande directed cases, bootstrap over pairs; 2,000 resamples. Population (band, layer) values use "
        "all cases (no selection involved); expert values use validation cases (the experts and the ext8 population ranking were "
        "chosen on discovery). Share = Σ D / Σ T over the same units; slope = OLS of T on D (T carries the pass noise, D is "
        "nearly noise-free).")
    out.append("")
    if "C" in S:
        out.append(c_what_was_run(S))
        out.append("")
    if "B" in S:
        out.append(b_what_was_run(S))
        out.append("")
    out.append("### A1. Selected experts and each task's population top-10: total, direct, indirect")
    out.append("")
    out.append(tab("ext11_A_experts"))
    out.append("")
    out.append("![Total vs direct per (case, expert)](figures/ext11_A_scatter.png)")
    out.append("")
    out.append("### A2. By depth")
    out.append("")
    out.append(tab("ext11_A_bands"))
    out.append("")
    out.append("![Per-layer sums of total and direct effects](figures/ext11_A_layers.png)")
    out.append("")
    out.append("![Direct share by layer](figures/ext11_A_share_by_layer.png)")
    out.append("")
    out.append("### A3. Sets, frozen-norm error, pass noise")
    out.append("")
    out.append(tab("ext11_A_sets_norm"))
    out.append("")
    rk = []
    for k in K:
        r = A[k]["ranking_val"]
        rk.append(f"| {A[k]['label']} | {A[k]['tlabel']} | {r['n_experts']} | {r['spearman']:.2f} | {r['top_overlap']}/10 | "
                  f"{', '.join(r['top_by_T'][:5])} | {', '.join(r['top_by_D'][:5])} | "
                  f"{', '.join(f'{e} {v}' for e, v in r['rank_under_D_of_T_top3'].items())} |")
    out.append("**Population rankings by total vs by direct effect** (validation, all-case means; experts routed in ≥ 5 % of the cases)")
    out.append("")
    out.append("| model | task | experts | Spearman ρ | top-10 overlap | top-5 by T | top-5 by D | rank under D of the T top-3 |")
    out.append("|---|---|---|---|---|---|---|---|")
    out += rk
    out.append("")
    if "C" in S:
        out += c_sections(S)
    if "B" in S:
        out += b_sections(S)
    out.append("### Reading")
    out.append("")
    out += reading(S)
    out.append("")
    out.append("### Caveats")
    out.append("")
    out += caveats(S)
    out.append("")
    out.append("### Files")
    out.append("")
    out += files(S)
    os.makedirs(os.path.dirname(SEC), exist_ok=True)
    open(SEC, "w").write("\n".join(out) + "\n")
    print("written", SEC)


# ------------------------------------------------------------------------------------------------------------------
# Part C text (filled once the routing runs exist)
# ------------------------------------------------------------------------------------------------------------------
def _cr(C, m, set_, e):
    return C[m]["C1"][set_][e]


def _ctx(C, m, e, name):
    x = C[m]["C5"]["contexts"][e].get(name)
    return x


def c_summary(S):
    C = S["C"]
    out = []
    if "qwen3" in C:
        q = C["qwen3"]
        f, l = _cr(C, "qwen3", "wino_main", "L41E117"), _cr(C, "qwen3", "wino_local", "L41E117")
        cop = _ctx(C, "qwen3", "L41E117", "current = copula (was / is / were / ...)")
        deg = _ctx(C, "qwen3", "L41E117", "current = degree adverb")
        out.append(
            f"Routing is set by the local context and the write by the full context: L41E117 is routed at the final position of "
            f"{100 * f['rate']['v']:.0f} % of the WinoGrande prompts and of {100 * l['rate']['v']:.0f} % of their context-free local "
            f"prompts ('The bag was too'), but its DLA there is {l['dla_active']['v']:+.2f} vs {f['dla_active']['v']:+.2f} logits; in "
            f"wikitext it fires on {100 * C['qwen3']['C5']['contexts']['L41E117']['base']:.0f} % of tokens but on "
            f"{100 * cop['rate']['v']:.0f} % of copulas and {100 * deg['rate']['v']:.0f} % of degree adverbs (a predicate-complement slot "
            f"detector), never at IOI's final token. ")
    if "mixtral" in C:
        f, l = _cr(C, "mixtral", "wino_main", "L20E000"), _cr(C, "mixtral", "wino_local", "L20E000")
        out.append(
            f"Mixtral's L20E000 is routed at {100 * f['rate']['v']:.0f} % of WinoGrande final positions and {100 * l['rate']['v']:.0f} % of "
            f"the local prompts (DLA {f['dla_active']['v']:+.2f} vs {l['dla_active']['v']:+.2f}), and at "
            f"{100 * C['mixtral']['C5']['contexts']['L20E000']['base']:.0f} % of wikitext tokens. ")
    if "qwen3" in C:
        pl = _ctx(C, "qwen3", "L44E069", "next word = CF place name")
        out.append(
            f"The CounterFact experts are broad: L44E069 / L42E115 are routed at {100 * C['qwen3']['C5']['contexts']['L44E069']['base']:.0f} / "
            f"{100 * C['qwen3']['C5']['contexts']['L42E115']['base']:.0f} % of wikitext tokens, rising to {100 * pl['rate']['v']:.0f} / "
            f"{100 * _ctx(C, 'qwen3', 'L42E115', 'next word = CF place name')['rate']['v']:.0f} % before CounterFact place names, and "
            f"L44E069 writes most for place relations (DLA when routed "
            f"{C['qwen3']['C3']['L44E069']['place']['dla_active']['v']:+.2f} logits vs "
            f"{C['qwen3']['C3']['L44E069']['occupation / field']['dla_active']['v']:+.2f} for occupations). ")
    return "".join(out)


def c_what_was_run(S):
    C = S["C"]
    parts = []
    for m, lab in (("qwen3", "Qwen3"), ("mixtral", "Mixtral (BOS)")):
        if m not in C:
            continue
        meta = json.load(open(os.path.join(RESULTS, C[m]["run"], "run_meta.json")))
        sets = ", ".join(f"{k} {v}" for k, v in meta["sets"].items())
        parts.append(f"{lab}: {meta['n_prompts']:,} prompts / {meta['n_tokens']:,} tokens in {meta['n_chunks']} prefill passes "
                     f"({sum(meta.get('pass_times_s', [])) / 60:.1f} GPU min; {sets}); all-token routing at layers "
                     f"{', '.join(map(str, meta['route_all_layers']))}; final-position routing at the WinoGrande main prompts agrees with "
                     f"the ext7 source run for {100 * C[m]['sanity']['wino_main']['agreement_with_source_routing']:.1f} % of (prompt, expert) pairs")
    return ("**Part C (GPU, prefill only, existing DiagSpec).** `scripts/ext11_writer_routing.py`: prompt sets = WinoGrande main "
            "(the 512 clean prompts of the main family's directed cases), their context-free local prompts (`local_prompt()` of "
            "`scripts/ext7_wino_scan.py`, both answers; true = own trigger, foil = twin trigger), the remaining margin-pool pairs and "
            "their local prompts, CounterFact STR (Direction-6 clean prompts and donors), the 1,024 clean prompts of the base CounterFact "
            "filter scan, the 1,600 IOI clean prompts (true = IO, foil = S) and 1,100 consecutive wikitext-103 test windows of 127 tokens "
            "(Mixtral: BOS prepended). Recorded: final-position routing at every layer, the routed experts' own DLA at the final "
            "position (`contrib_dla`, c_e itself, norm frozen at the prompt's own scale) and the routing of every token at the "
            "layers of the targets and both tasks' population top-10 (`route_all_layers`). wikitext contexts use heuristic word "
            "lists (no POS tagger installed): copulas (was, is, were, be, became, seemed, ...), degree adverbs (too, very, so, "
            "quite, more, ...), determiners, prepositions; 'WinoGrande trigger vocabulary' = all sentence-final trigger words of the "
            "model's W1-W6 pairs; 'CF place name' = a target_true / target_new string of a CounterFact place relation beginning at the "
            "next word. " + "; ".join(parts) + ".")


def c_sections(S):
    out = ["### C1. Final-position routing and DLA of the target experts by prompt set", "", tab("ext11_C_final_rates"), "",
           "### C2. WinoGrande: by the word before the trigger, full vs local prompt", "", tab("ext11_C_wino_finalword"), ""]
    C = S["C"]
    rows = ["| model | expert | routed full / local / both (%) | routed in local given routed in full (%) | DLA full / local when both (logits) | r(DLA full, DLA local) |",
            "|---|---|---|---|---|---|"]
    for m, e in (("qwen3", "L41E117"), ("qwen3", "L43E081"), ("qwen3", "L39E071"), ("mixtral", "L20E000"), ("mixtral", "L19E006"),
                 ("mixtral", "L21E006")):
        if m not in C or e not in C[m]["C2"]["paired_main_full_vs_local"]:
            continue
        x = C[m]["C2"]["paired_main_full_vs_local"][e]
        rows.append(f"| {'Qwen3' if m == 'qwen3' else 'Mixtral (BOS)'} | {e} | {100 * x['full']:.1f} / {100 * x['local']:.1f} / {100 * x['both']:.1f} | "
                    f"{100 * x['local_given_full']:.1f} | {x['dla_full_when_both']:+.2f} / {x['dla_local_when_both']:+.2f} | {x['r_dla_full_local']:.2f} |")
    out += ["**Paired main prompts: the same directed case as a full and as a local prompt**", ""] + rows + [""]
    ld = "; ".join(f"{'Qwen3' if m == 'qwen3' else 'Mixtral'} mean Δ full {C[m]['C2']['local_delta']['main_mean_delta']:+.2f}, "
                   f"local {C[m]['C2']['local_delta']['main_local_mean_delta']:+.2f} (local Δ > 0 in {100 * C[m]['C2']['local_delta']['local_correct_frac']:.0f} %)"
                   for m in C)
    out += [f"Local prompts carry little of the answer: {ld}.", ""]
    out += ["### C3. CounterFact: by relation group", "", tab("ext11_C_cf_relations"), "",
            "### C4. All-token routing inside the task prompts", "", tab("ext11_C_positions"), "",
            "### C5. wikitext-103 contexts", "", tab("ext11_C_wiki_contexts"), "", tab("ext11_C_wiki_tokens"), ""]
    return out


def _bx(S, m, e, key):
    return S["B"].get(m, {}).get("experts", {}).get(e, {}).get(key)


def _enr(x, cls):
    """top-50 share of a class / its vocabulary base rate."""
    return x[f"top50_{cls}"] / x[f"base_{cls}"] if x and x.get(f"base_{cls}") else float("nan")


def b_summary(S):
    B = S["B"]
    out = []
    if "qwen3" in B:
        d = _bx(S, "qwen3", "L41E117", "delta:wino")
        c = _bx(S, "qwen3", "L41E117", "clean:wino")
        d69 = _bx(S, "qwen3", "L44E069", "delta:cf")
        c69 = _bx(S, "qwen3", "L44E069", "clean:cf_clean")
        if d and c and d69 and c69:
            out.append(
                f"In vocabulary space the writes are answer-shaped: L41E117's δ_e puts the correct trigger at median rank "
                f"{d['median_rank_true']:.0f} of {B['qwen3']['V']:,} tokens (top-10 in {100 * d['frac_true_top10']:.0f} % of rows) and "
                f"its 50 most promoted tokens are WinoGrande trigger words {_enr(d, 'wg_trigger'):.0f}× more often than the vocabulary "
                f"base rate (antonym axes such as small / smaller / larger); L44E069's δ_e puts the true object at median rank "
                f"{d69['median_rank_true']:.0f} and its clean output c_e promotes CounterFact object tokens {_enr(c69, 'cf_any'):.0f}× "
                f"over base (place-name pieces). ")
    if "mixtral" in B:
        d = _bx(S, "mixtral", "L20E000", "delta:wino")
        c0 = _bx(S, "mixtral", "L20E000", "clean:wino")
        c21 = _bx(S, "mixtral", "L21E001", "clean:cf_clean")
        if d and c0 and c21:
            out.append(f"In Mixtral the clean outputs are even more answer-like: L20E000's c_e ranks the correct trigger at median "
                       f"{c0['median_rank_true']:.0f} of {B['mixtral']['V']:,} (top-10 in {100 * c0['frac_true_top10']:.0f} % of prompts; δ_e "
                       f"{d['median_rank_true']:.0f}), L21E001's c_e the true object at {c21['median_rank_true']:.0f}. ")
    np_ = [x["exact_over_frozen"] for m in B for x in B[m]["norm_pooled"].values()]
    ea = [x["mean_abs_err"] for m in B for x in B[m]["norm_pooled"].values()]
    if np_:
        out.append(f"For single experts the exact-norm direct effect is {min(np_):.3f}–{max(np_):.3f} times the frozen-norm DLA "
                   f"(mean |difference| {min(ea):.3f}–{max(ea):.3f} logits). ")
    return "".join(out)


def b_what_was_run(S):
    B = S["B"]
    parts = []
    for m, lab in (("qwen3", "Qwen3"), ("mixtral", "Mixtral (BOS)")):
        if m not in B:
            continue
        meta = json.load(open(os.path.join(RESULTS, B[m]["run"], "run_meta.json")))
        parts.append(f"{lab}: {meta['n_prompts']:,} prompts in one prefill pass ({(meta.get('pass_s') or 0) / 60:.1f} GPU min, "
                     f"peak {meta.get('peak_GB') or 0:.1f} GB), {meta['n_vectors']:,} vectors; same-pass check frozen projection vs the "
                     f"engine's contrib_dla max |diff| {meta['check_frozen_vs_contrib_dla_max_abs']:.1e} logits")
    return ("**Part B (GPU, one prefill pass per model, ext9 engine E4c `DiagSpec.contrib_final_vectors`).** "
            "`scripts/ext11_writer_vocab.py`: prompts = CounterFact STR clean prompts and every selected donor, the 512 WinoGrande "
            "main prompts (the corrupted prompt of a directed case is its twin's clean prompt), their local prompts and the first 400 "
            "IOI clean prompts; experts = targets + both tasks' population top-10. Vectors at the final position: δ_e = c_e(clean) − "
            "c_e(corrupt) per STR row where e is clean-active (c_e(corrupt) = 0 if not routed), projected as ((δ_e ⊙ γ) W_U^T) / "
            "rms(h_final, corrupted run), i.e. ext8's DLA over the whole vocabulary; c_e(clean) projected at the prompt's own "
            "scale. Per vector: rank of r among all tokens (1 = most promoted) and of r′ from the bottom, top-20 promoted / "
            "suppressed tokens, class means (z = (class mean − vocabulary mean) / SD of the projection) and top-50 class shares for "
            "the WinoGrande trigger vocabulary (every trigger of the model's W1-W6 pairs), the first tokens of CounterFact objects "
            "per relation group (target_true and target_new strings) and the IOI names, rank of r within its own class; exact-norm "
            "direct effect Δ(norm(h_corrupt + δ_e)) − Δ(norm(h_corrupt)) through the real final RMSNorm (fp32, offline from "
            "`resid_final`) vs the frozen DLA. Analysis `scripts/ext11_writer_vocab_analyze.py`. " + "; ".join(parts) + ".")


SHOW_B = {"Qwen3-30B-A3B-Base": ["L41E117", "L43E081", "L44E069", "L42E115"],
          "Mixtral-8x7B (BOS)": ["L20E000", "L19E006", "L19E002", "L21E001", "L18E001"]}


def tab_filtered(name, keep_vectors=None):
    """Section version of a B table: targets + each model's second WinoGrande expert (full table in results/tables)."""
    p = os.path.join(TAB, name + ".csv")
    if not os.path.exists(p):
        return f"(table {name} missing)"
    df = pd.read_csv(p, keep_default_na=False, dtype=str)
    m = [e in SHOW_B.get(mo, []) for mo, e in zip(df["model"], df["expert"])]
    df = df[m]
    if keep_vectors is not None:
        df = df[df["vectors"].isin(keep_vectors)]
    note = open(os.path.join(TAB, name + ".md")).read().strip().split("\n\n")[-1]
    return W.md_table(df) + "\n\n" + note + f" Section: targets and each model's second WinoGrande expert; all experts of interest in `results/tables/{name}.md`."


def b_sections(S):
    own = ["δ_e, CounterFact STR rows", "δ_e, WinoGrande STR rows", "c_e, CounterFact clean prompts", "c_e, WinoGrande prompts",
           "c_e, WinoGrande local prompts", "c_e, IOI prompts"]
    out = ["### B1. Vocabulary projections: where r and r′ land", "", tab_filtered("ext11_B_ranks"), "",
           "### B2. Token classes", "", tab_filtered("ext11_B_classes"), "",
           "### B3. Exact final norm vs frozen norm for single experts", "",
           tab_filtered("ext11_B_norm", keep_vectors=own[:2] + own[2:4]), ""]
    rows = ["| model | task (δ_e rows) | vectors | Σ exact / Σ frozen | r | mean abs error (logits) | 95th pct abs error | Σ |err| / Σ |frozen| |",
            "|---|---|---|---|---|---|---|---|"]
    for m in S["B"]:
        for t, x in S["B"][m]["norm_pooled"].items():
            rows.append(f"| {'Qwen3' if m == 'qwen3' else 'Mixtral (BOS)'} | {'CounterFact' if t == 'cf' else 'WinoGrande'} | {x['n']} | "
                        f"{x['exact_over_frozen']:.3f} | {x['r']:.4f} | {x['mean_abs_err']:.4f} | {x['p95_abs_err']:.4f} | {x['rel_abs_err']:.3f} |")
    out += ["**Pooled over every δ_e vector of the experts of interest**", ""] + rows + [""]
    return out


def b_reading(S):
    B = S["B"]
    out = []
    if "qwen3" not in B:
        return out
    g = lambda m, e, k: _bx(S, m, e, k)
    d117, c117, l117 = g("qwen3", "L41E117", "delta:wino"), g("qwen3", "L41E117", "clean:wino"), g("qwen3", "L41E117", "clean:wino_local")
    d81, c81, l81 = g("qwen3", "L43E081", "delta:wino"), g("qwen3", "L43E081", "clean:wino"), g("qwen3", "L43E081", "clean:wino_local")
    d69, c69 = g("qwen3", "L44E069", "delta:cf"), g("qwen3", "L44E069", "clean:cf_clean")
    d115, c115 = g("qwen3", "L42E115", "delta:cf"), g("qwen3", "L42E115", "clean:cf_clean")
    tp = lambda x, n=5: ", ".join(repr(a) for a, _ in x["top_promoted"][:n])
    V = B["qwen3"]["V"]
    txt = (f"**7. What the writes say (vocabulary projections).** Projected on the whole vocabulary ({V:,} tokens), the δ_e of the Qwen3 "
           f"writers put the answer high: L41E117 median rank {d117['median_rank_true']:.0f} (r in the top 100 in "
           f"{100 * d117['frac_true_top100']:.0f} % of rows, rank {d117['median_rank_true_in_class']:.0f} of {d117['median_n_class']:.0f} "
           f"within the trigger vocabulary; r′ at median rank {d117['median_rank_low_foil']:.0f} from the bottom, the mirror image "
           f"because the two directions of a pair carry δ_e of opposite sign), L42E115 {d115['median_rank_true']:,.0f}, L43E081 "
           f"{d81['median_rank_true']:,.0f}, L44E069 {d69['median_rank_true']:,.0f} (top {100 * d117['median_rank_true'] / V:.2f} % to "
           f"{100 * max(d81['median_rank_true'], d69['median_rank_true'], d115['median_rank_true']) / V:.1f} %; within the CounterFact "
           f"objects of the case's relation group L44E069 ranks r {d69['median_rank_true_in_class']:.0f} of {d69['median_n_class']:.0f}). "
           "The clean outputs c_e alone are already "
           f"answer-like: L41E117's c_e in the WinoGrande prompts ranks the correct trigger at median {c117['median_rank_true']:.0f} "
           f"(most promoted {tp(c117)}), L43E081's at {c81['median_rank_true']:.0f} with trigger words making up "
           f"{100 * c81['top50_wg_trigger']:.0f} % of its top 50 (base {100 * c81['base_wg_trigger']:.1f} %), L42E115's c_e on CounterFact "
           f"at {c115['median_rank_true']:.0f} (languages and nationalities: {tp(c115)}), L44E069's at {c69['median_rank_true']:.0f} "
           f"({tp(c69)}). In the context-free local prompts L41E117's c_e still promotes the same adjective family ({tp(l117, 4)}) but "
           f"ranks the specific trigger only at median {l117['median_rank_true']:.0f}: the expert writes a property axis (size, "
           "weight, temperature, ...) whose direction is set by the context it receives, which is why its DLA toward r vs r′ "
           "collapses without the context (C2). The class means (z) are small because the trigger vocabulary (1,648 words) and the "
           "object classes are broad; the top-50 shares carry the class signal.")
    out.append(txt)
    if "mixtral" in B:
        d0, c0 = g("mixtral", "L20E000", "delta:wino"), g("mixtral", "L20E000", "clean:wino")
        d2, d21 = g("mixtral", "L19E002", "delta:cf"), g("mixtral", "L21E001", "delta:cf")
        c2, c21, c18 = g("mixtral", "L19E002", "clean:cf_clean"), g("mixtral", "L21E001", "clean:cf_clean"), g("mixtral", "L18E001", "clean:cf_clean")
        d18, l0 = g("mixtral", "L18E001", "delta:cf"), g("mixtral", "L20E000", "clean:wino_local")
        if d0 and c0 and d2 and d21 and c2 and c21 and c18 and d18 and l0:
            V2 = B["mixtral"]["V"]
            out.append("")
            out.append(
                f"In Mixtral (vocabulary {V2:,}) the clean outputs of the targets rank the answer near the top: L20E000's c_e in the "
                f"WinoGrande prompts at median {c0['median_rank_true']:.0f} (top-10 in {100 * c0['frac_true_top10']:.0f} %, top-100 in "
                f"{100 * c0['frac_true_top100']:.0f} %; most promoted {tp(c0)}; δ_e {d0['median_rank_true']:.0f}), in the local prompts "
                f"only at {l0['median_rank_true']:.0f} (the same pattern as Qwen3); the CounterFact experts' c_e at "
                f"{c21['median_rank_true']:.0f} (L21E001), {c2['median_rank_true']:.0f} (L19E002) and {c18['median_rank_true']:.0f} "
                f"(L18E001; δ_e {d21['median_rank_true']:.0f}, {d2['median_rank_true']:.0f}, {d18['median_rank_true']:,.0f}), i.e. "
                "L18E001, the most indirect of the three in Part A, also writes the least answer-like vector. Many Mixtral vectors "
                "also promote a handful of answer-unrelated tokens (' /******/', byte-order mark, code fragments), a shared "
                "direction that is irrelevant to the logit difference (which reads only r and r′) and hence to DLA.")
    return out


def reading(S):
    A, K, lab, tg = part_a(S)
    out = []
    q_wg = ex(A["wino_qwen3"], "L41E117")
    q69 = ex(A["cf_qwen3"], "L44E069")
    q115 = ex(A["cf_qwen3"], "L42E115")
    m_wg = ex(A["wino_mixtral"], "L20E000")
    m19 = ex(A["wino_mixtral"], "L19E006")
    m16 = ex(A["wino_mixtral"], "L16E007")
    m26 = ex(A["wino_mixtral"], "L26E002")
    out.append(
        f"**1. Qwen3: the localised experts are writers.** L41E117 writes its whole effect on WinoGrande directly (T {c(q_wg['T'], sign=True)}, "
        f"D {c(q_wg['D'], sign=True)} logits, r(D, T) across cases {q_wg['corr']['r']:.2f}); so does L42E115 on CounterFact "
        f"(share {q115['share']['v']:.2f}). L44E069 writes {q69['D']['v']:.2f} logits but the patch only gains {q69['T']['v']:.2f}: the three "
        "layers after it remove about a quarter of its write (indirect effect negative). The same holds for every Qwen3 expert at or "
        "above the crossover layer (L43E081 1.42, L44E122 1.64, L44E006 1.33 ...), while experts a few layers below (L39E071 0.72, "
        "L34E119 0.52, L31E124 0.51 on WinoGrande) are partly computers whose effect later layers amplify.")
    out.append("")
    out.append(
        f"**2. Mixtral: partly writers.** L20E000 writes {m_wg['D']['v']:.2f} of its {m_wg['T']['v']:.2f} logits (share {c(m_wg['share'])}); "
        f"L19E006, the second WinoGrande expert, {m19['share']['v']:.2f}; L16E007 {m16['share']['v']:.2f} (a computer). The CounterFact "
        "experts are closer to writers (L21E001 0.87, L19E002 0.71, L18E001 0.54). The late WinoGrande experts (L24E002 1.02, L26E002 "
        f"{m26['share']['v']:.2f}, L22E005, L25E002 > 1) write more than they gain. In Mixtral the band L16-L21, where the "
        "WinoGrande patches peak (ext7 W2: L20), does about half of its work through layers 22-31.")
    out.append("")
    out.append(
        "**3. Depth profile.** In every task the direct share rises with depth: experts in the first half of the network have "
        "total effects near zero and almost no direct effect (they act, if at all, through later layers); in the band just "
        "below the peak the share is 0.4-0.8 (later layers amplify); at and above a crossover layer the share is ≥ 1 (later "
        "layers partly cancel the write, the downstream self-repair known from dense models, McGrath et al. 2023); Mixtral "
        "CounterFact has no crossover (share 0.8-0.9 from L19 to L28). The experts of the last two or three layers (Qwen3 "
        "L46-L47, Mixtral L29-L31) push against the answer directly (negative D and T; at the very last layer T = D). "
        "Summed over all experts the share is "
        + ", ".join(f"{lab(k)} {A[k]['agreement_all']['all']['share']['v']:.2f}" for k in K) + ".")
    out.append("")
    out.append(
        "**4. Why the DLA ordering works (ext8).** Ranking by D recovers the top experts by T (top-10 overlap "
        + ", ".join(f"{A[k]['ranking_val']['top_overlap']}/10" for k in K) +
        "), because the strongest experts sit at or above the crossover where D ≥ T. In Mixtral WinoGrande, D promotes the late "
        "over-writers (L26E002 first) and demotes the half-writers of L19-L21 (L19E006 rank "
        f"{A['wino_mixtral']['ranking_val']['rank_under_D_of_T_top3'].get('L19E006', 'n/a')} under D), which is where ext8 found DLA "
        "slightly below the oracle (AUC 0.55 vs 0.58). A good DLA ranking is therefore evidence that the top experts write, not "
        "that every expert does.")
    if "C" in S:
        out.append("")
        out += c_reading(S)
    if "B" in S:
        out.append("")
        out += b_reading(S)
    return out


def c_reading(S):
    C = S["C"]
    out = []
    if "qwen3" not in C or "mixtral" not in C:
        return out
    pct = lambda x: f"{100 * x:.0f} %"
    q, m = C["qwen3"], C["mixtral"]

    def ctx(mm, e, name):
        return mm["C5"]["contexts"][e][name]
    qf = {w: q["C2"]["L41E117"][w] for w in q["C2"]["L41E117"]}
    e117 = "L41E117"
    pf = q["C2"]["paired_main_full_vs_local"][e117]
    pm = m["C2"]["paired_main_full_vs_local"]["L20E000"]
    inp = "current = ' in', next = CF place"
    out.append(
        f"**5. The WinoGrande experts are slot detectors that write context.** Qwen3 L41E117 is routed at the final position of "
        f"{pct(q['C1']['wino_main'][e117]['rate']['v'])} of the main WinoGrande prompts ({pct(qf['too']['full']['v'])} after 'too', "
        f"{pct(qf['very']['full']['v'])} after 'very', {pct(qf['was']['full']['v'])} after 'was'), at the same tokens earlier in the "
        f"same sentences ({pct(q['C4'][e117]['wino_main_finalword_tokens_nonfinal'])}), and at {pct(pf['local'])} of the context-free "
        f"local prompts; in wikitext at {pct(ctx(q, e117, 'current = copula (was / is / were / ...)')['rate']['v'])} of copulas, "
        f"{pct(ctx(q, e117, 'current = degree adverb')['rate']['v'])} of degree adverbs and "
        f"{pct(ctx(q, e117, 'copula or degree, next word in WG trigger vocabulary')['rate']['v'])} of copula / degree tokens followed by a "
        f"WinoGrande trigger word (base {pct(ctx(q, e117, 'all tokens')['rate']['v'])}); its most selective next words in wikitext are "
        "predicative adjectives and participles (important, better, too, highly, unable, able, significant). It never fires at the "
        f"IOI final token ({pct(q['C1']['ioi'][e117]['rate']['v'])}) and rarely at CounterFact's ({pct(q['C1']['cf_scan'][e117]['rate']['v'])}). "
        f"Mixtral's L20E000 is the same kind of expert, less selective: {pct(m['C1']['wino_main']['L20E000']['rate']['v'])} of WinoGrande "
        f"final positions, {pct(pm['local'])} of local prompts, {pct(ctx(m, 'L20E000', 'current = copula (was / is / were / ...)')['rate']['v'])} "
        f"of wikitext copulas (base {pct(ctx(m, 'L20E000', 'all tokens')['rate']['v'])}), and also {pct(m['C1']['cf_scan']['L20E000']['rate']['v'])} "
        "of CounterFact final positions. The routing decision is therefore local (the token and its syntactic slot), but what the "
        f"expert writes is not: its DLA toward the right trigger is {pf['dla_full_when_both']:+.2f} logits in the full prompt and "
        f"{pf['dla_local_when_both']:+.2f} in the local prompt of the same case (Mixtral {pm['dla_full_when_both']:+.2f} / "
        f"{pm['dla_local_when_both']:+.2f}), where the model itself barely knows the answer (local Δ "
        f"{q['C2']['local_delta']['main_local_mean_delta']:+.2f} vs {q['C2']['local_delta']['main_mean_delta']:+.2f}). The expert "
        "reads the option-dependent information from its input residual and turns it into the trigger logit.")
    out.append("")
    e69, e115 = "L44E069", "L42E115"
    out.append(
        f"**6. The CounterFact experts are broad name-slot experts.** Qwen3 L44E069 / L42E115 are routed at "
        f"{pct(ctx(q, e69, 'all tokens')['rate']['v'])} / {pct(ctx(q, e115, 'all tokens')['rate']['v'])} of all wikitext tokens, at "
        f"{pct(ctx(q, e69, 'next word = CF place name')['rate']['v'])} / {pct(ctx(q, e115, 'next word = CF place name')['rate']['v'])} before a "
        f"CounterFact place name and {pct(ctx(q, e69, 'next word capitalised, not a CF object')['rate']['v'])} / "
        f"{pct(ctx(q, e115, 'next word capitalised, not a CF object')['rate']['v'])} before other capitalised words, and at "
        f"{pct(ctx(q, e69, 'current = copula (was / is / were / ...)')['rate']['v'])} / {pct(ctx(q, e115, 'current = copula (was / is / were / ...)')['rate']['v'])} "
        "of copulas: yes, L44E069 fires before place names, but as part of a general 'a name follows' slot, not place-specifically. "
        f"L42E115 is routed at {pct(q['C1']['ioi'][e115]['rate']['v'])} of IOI final tokens (' to') and writes nothing there (DLA "
        f"{q['C1']['ioi'][e115]['dla_active']['v']:+.2f}). What they write is relation-specific: L44E069's DLA when routed is "
        f"{q['C3'][e69]['place']['dla_active']['v']:+.2f} logits on place relations, {q['C3'][e69]['language']['dla_active']['v']:+.2f} on "
        f"languages, {q['C3'][e69]['organisation']['dla_active']['v']:+.2f} on organisations and {q['C3'][e69]['occupation / field']['dla_active']['v']:+.2f} "
        f"on occupations (L42E115 {q['C3'][e115]['place']['dla_active']['v']:+.2f} / {q['C3'][e115]['language']['dla_active']['v']:+.2f} / "
        f"{q['C3'][e115]['organisation']['dla_active']['v']:+.2f} / {q['C3'][e115]['occupation / field']['dla_active']['v']:+.2f}). In Mixtral, "
        f"L18E001 and L21E001 are preposition-slot experts (wikitext ' in' + place {pct(ctx(m, 'L18E001', inp)['rate']['v'])} / "
        f"{pct(ctx(m, 'L21E001', inp)['rate']['v'])}; L18E001's most selective next words are "
        f"months; IOI final ' to' {pct(m['C1']['ioi']['L18E001']['rate']['v'])} / {pct(m['C1']['ioi']['L21E001']['rate']['v'])} with DLA ≈ 0), "
        f"L19E002 is a pre-entity expert that does not fire on IOI ({pct(m['C1']['ioi']['L19E002']['rate']['v'])}).")
    return out


def caveats(S):
    A, K, lab, tg = part_a(S)
    nz = {k: A[k]["inpass_noise"] for k in K}
    out = [
        "- T and D come from different passes (T: the ext6 / ext7 single-expert rows; D: ext8 pass 0). The pass-to-pass SD of a "
        "single-expert rescue is " + ", ".join(f"{lab(k)} {nz[k]['sd_single']:.2f}" for k in K) + " logits (ext8 in-pass singles of "
        "each first-donor row's top-10 vs the source rows; test-retest r " + ", ".join(f"{nz[k]['r_T_T8']:.2f}" for k in K) +
        "). Pairs with |T| below these values are noise; the bands' r over all pairs is dominated by them, the shares are not.",
        "- D freezes the final RMSNorm at the corrupted run's scale. For the summed MoE writes the exact-norm direct effect is "
        "known (ext8 `addback_direct.parquet`) and differs by ≤ 0.006 of the drop on the mean; for single experts the last-layer "
        "check (T − D with no downstream computation) gives "
        + ", ".join(f"{A[k]['agreement_all']['last']['mean_T_minus_D_logits']:+.3f}" for k in K) + " logits per expert on average"
        + (" and the exact computation of Part B (B3) " + "; ".join(
            f"{'Qwen3' if m == 'qwen3' else 'Mixtral'} Σ exact / Σ frozen " + " / ".join(f"{x['exact_over_frozen']:.3f}" for x in S['B'][m]['norm_pooled'].values())
            for m in S['B']) if "B" in S else "") + ".",
        "- 'Indirect' includes interactions with every later layer at the final position only (the K/V of earlier positions are "
        "the corrupted run's); it does not separate later attention from later MoE layers (that needs path patching with "
        "downstream components frozen, not available in the engine).",
        "- Direct share is a ratio of sums; for experts whose total is near zero (early layers, CounterFact experts on WinoGrande "
        "and vice versa, n active < 20) it is undefined or has very wide CIs.",
        "- Final position only, STR only, Qwen3-30B-A3B-Base and Mixtral-8x7B with BOS.",
    ]
    if "B" in S:
        v = json.load(open(os.path.join(RESULTS, "verify_ext9_engine_olmoe.json")))["contrib_final_vectors"]
        out.append(
            "- Part B uses the ext9 engine's `DiagSpec.contrib_final_vectors` (E4c). Its OLMoE verification "
            "(`results/verify_ext9_engine_olmoe.json`): Σ slots vs the fp32 MoE output max |diff| "
            f"{v['sum_vs_moe_out_maxabs']:.1e}, vector norms vs `route_cnorm` {v['norm_vs_route_cnorm_maxabs']:.1e}, per-expert "
            f"vectors vs transformers hooks median relative norm difference {v['vs_hf_contrib_relnorm']['median']:.3f} (max "
            f"{v['vs_hf_contrib_relnorm']['max']:.2f}, n = {v['vs_hf_contrib_relnorm']['n']}; bf16 level). In-run checks: the "
            "frozen projection of c_e equals the engine's `contrib_dla` exactly, and the δ_e DLA reproduces ext8's DLA from a "
            "different pass (r " + "; ".join(f"{'Qwen3' if m == 'qwen3' else 'Mixtral'} {x['cf']['r']:.3f} CounterFact / {x['wino']['r']:.3f} WinoGrande"
                                             for m, x in ((m, S['B'][m].get('xcheck_ext8')) for m in S['B']) if x) + ").")
        out.append("- Token classes are first tokens of words (CounterFact objects, triggers, names); a projection that promotes "
                   "a different piece of the same word, another casing or a translation (' small', 'small', '小') counts only "
                   "for the exact class token. Top-promoted tokens list the share of vectors with the token in their top 10.")
    return out


def files(S):
    return [
        "- Code: `moetrace/ext11_writer.py` (loaders, cluster-bootstrap statistics, token classes), `scripts/ext11_writer_partA.py` "
        "(Part A), `scripts/ext11_writer_routing.py` + `scripts/ext11_chain_routing.sh` (Part C GPU passes), "
        "`scripts/ext11_writer_routing_analyze.py` (Part C analysis), `scripts/ext11_writer_vocab.py` + `scripts/ext11_chain_vocab.sh` "
        "(Part B GPU passes), `scripts/ext11_writer_vocab_analyze.py` (Part B analysis), `scripts/ext11_writer_text.py` (this section).",
        "- Runs: `results/{qwen3,mixtral_bos}_writer_routing` (prompts_input, final_routing, token_events, run_meta), "
        "`results/{qwen3,mixtral_bos}_writer_vocab` (prompts, vec_rows = one row per projected vector, vec_top = top-20 "
        "promoted / suppressed ids, run_meta); raw arrays in `/opt/dlami/nvme/moe_ext11/` (routing chunks, vectors.pt; ephemeral).",
        "- Numbers: `results/ext11_writer_summary.json` (keys A, C, B); tables `results/tables/ext11_*.md|csv`; figures "
        "`results/figures/ext11_*`.",
        "- GPU time (gpu_queue jobs): Part C 8.0 min (Qwen3 176 s, Mixtral 303 s), Part B 4.0 min (Qwen3 96 s, Mixtral 146 s); "
        "Part A is CPU only.",
    ]


if __name__ == "__main__":
    build()
