"""ext6: write results/sections/ext6_str.md from results/ext6_str_summary.json and the ext6_str tables (CPU only).

Usage: python scripts/ext6_str_text.py
"""
import json, os, sys

ROOT = "/home/ubuntu/MOE/results"
S = json.load(open(os.path.join(ROOT, "ext6_str_summary.json")))
V = json.load(open(os.path.join(ROOT, "verify_ext6_str_olmoe.json")))
KEYS = [k for k in ("qwen3", "mixtral_nobos", "mixtral_bos") if k in S]
SHORT = {"qwen3": "Qwen3", "mixtral_nobos": "Mixtral no BOS", "mixtral_bos": "Mixtral BOS"}
PAPER_LAYER = {"qwen3": 44, "mixtral_nobos": 19, "mixtral_bos": 19}


def f(s, d=3):
    return f"{s['mean']:+.{d}f} [{s['ci_lo']:+.{d}f}, {s['ci_hi']:+.{d}f}]"


def r3(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def tab(name):
    return open(os.path.join(ROOT, "tables", f"ext6_str_{name}.md")).read().strip()


def L(k, corr="STR (donor mean)"):
    return S[k]["layers"][corr]


def E(k, layer, corr="STR (donor mean)"):
    return S[k]["experts"][corr].get(str(layer))


def N(k, pair, corr="STR (donor mean)"):
    return S[k]["named"][corr].get(pair)


def J(k, corr="STR (donor mean)"):
    return S[k]["joint"][corr]


def same_sel(k):
    a, b = L(k)["L_star"], L(k, "STR (first donor)")["L_star"]
    ea, eb = E(k, a), E(k, b, "STR (first donor)")
    return a, b, (ea or {}).get("e_star"), (eb or {}).get("e_star")


def red(k):
    return 1 - L(k)["normalised_at_Lstar"]["ratio"][0] / L(k, "GN (same cases)")["normalised_at_Lstar"]["ratio"][0]


MIX = [k for k in KEYS if k.startswith("mixtral")]
RED = f"{min(100 * red(k) for k in MIX):.0f}–{max(100 * red(k) for k in MIX):.0f} %" if MIX else "n/a"
DIFF = [k for k in KEYS if same_sel(k)[0] != same_sel(k)[1]]


def donor_sentence():
    if not DIFF:
        return "The single-donor sensitivity run reproduces every selection."
    parts = []
    for k in DIFF:
        a, b, ea, eb = same_sel(k)
        parts.append(f"{SHORT[k]}: donor mean L{a}E{ea:03d}, first donor L{b}E{eb:03d} (discovery "
                     f"L{a} {dict(map(tuple, L(k, 'STR (first donor)')['disc_top5'])).get(a, float('nan')):+.2f} vs "
                     f"L{b} {dict(map(tuple, L(k, 'STR (first donor)')['disc_top5']))[b]:+.2f} in the first-donor run)")
    ok = [SHORT[k] for k in KEYS if k not in DIFF]
    return ("The single-donor sensitivity run reproduces the selections of " + ", ".join(ok) + "; it differs only where the layer "
            "argmax is a tie inside a flat band: " + "; ".join(parts) + ".")


out = []
w = out.append
q, mn, mb = "qwen3", "mixtral_nobos", "mixtral_bos"

# ---------------------------------------------------------------- summary
fl = {k: S[k]["filter"] for k in KEYS}
w("**Summary.** Zhang & Nanda (2024, arXiv:2309.16042) recommend symmetric token replacement (STR) over Gaussian noising (GN) as "
  "the corruption for activation patching, because GN can put the model off-distribution and inflate localisation. The paper "
  "(and our reproduction) uses GN on the subject embeddings. We re-ran the paper's full two-stage procedure with STR: the "
  "subject is replaced by another CounterFact subject of the same relation whose true object is the case's foil (same template, "
  "identical token positions), so the paper's metric Δ = logit(true) − logit(foil) becomes Zhang & Nanda's logit difference "
  "LD(r, r′) with r′ the corrupted prompt's answer. "
  f"On {', '.join(f'{fl[k]['cases_kept']}' for k in KEYS)} of 256 paper cases (Qwen3, Mixtral no BOS, Mixtral BOS) with up to "
  "five known donor facts per case, **every selection of the paper survives**: ")
w(f"Qwen3 selects L{L(q)['L_star']} and L44E069 (validation Spec {f(E(q, 44)['spec'])} logits; {r3(E(q, 44)['spec_norm'])} of the drop "
  f"vs {r3(S[q]['named']['GN (same cases)']['L44E069']['spec_norm'])} under GN), with L42E115 as the second locus; Mixtral under the "
  f"paper's no-BOS protocol selects L{L(mn)['L_star']} and L19E006, whose Spec is again negative ({f(E(mn, 19)['spec'])}) while the "
  f"clean top-2 coalition recovers the block ({f(E(mn, 19)['coalition_clean'])} vs {f(E(mn, 19)['layer_same_pass'])}). ")
w("STR roughly doubles the drop (the corrupted run now prefers the foil instead of being indifferent), the STR and GN layer curves "
  f"correlate at r = {', '.join(f'{S[k]['curve_corr_str_gn']:.2f}' for k in KEYS)}, and the drop-normalised block rescue at the "
  f"selected layer is {', '.join(f'{L(k)['normalised_at_Lstar']['ratio'][0]:.3f}' for k in KEYS)} under STR vs "
  f"{', '.join(f'{L(k, 'GN (same cases)')['normalised_at_Lstar']['ratio'][0]:.3f}' for k in KEYS)} under GN: no GN inflation in Qwen3, "
  f"a mild one ({RED} lower under STR) in Mixtral. With BOS, Mixtral's flat L19–L21 band makes the layer argmax move to "
  f"L{L(mb)['L_star']} (discovery gap to L19 {L(mb)['disc_top5'][0][1] - dict(map(tuple, L(mb)['disc_top5']))[19]:+.2f}), where the "
  f"selected L{L(mb)['L_star']}E001 is positively specific; L19E002 keeps a positive Spec ({f(E(mb, 19)['spec'])}). " + donor_sentence() + "\n")

# ---------------------------------------------------------------- setup
w("### Why and how\n")
w("Zhang & Nanda compare GN (the ROME/causal-tracing corruption: N(0, (3σ)²) added to the subject embeddings) with STR (the key "
  "tokens are swapped for tokens of the same kind so that the corrupted prompt is an ordinary in-distribution prompt with its own "
  "answer r′), and logit difference (normalised by LD_clean − LD_corrupt) with probability and KL. Their recommendations: STR "
  "whenever possible, logit difference rather than probability, single-layer before sliding-window patching, and trying several "
  "corruption sites. The paper follows the metric and the single-layer patch but uses GN; with GN the foil is not the corrupted "
  "run's answer, so Δ_noised ≈ 0 (\"indifferent\") rather than negative (\"prefers the other fact\").\n")
w("Construction (`moetrace/ext6_str.py`). For a case with template t, subject s, true object o and foil o_f (CounterFact "
  "`target_new`), a donor is any other CounterFact subject s′ with the same relation whose `target_true` is o_f, such that "
  "t.format(s′) has the same length, the subject at the same token positions, identical template tokens and the same "
  "single-token continuation ids for o and o_f. The corrupted run is t.format(s′) as a plain prefill row; every patch is the "
  "paper's (final-position MoE output, expert update δ_e = c_e(clean) − c_e(corrupt), coalitions) with the donor run as parent. "
  "User decisions (2026-09-28): the paper's case IDs and discovery/validation split, restricted to cases with a donor, recurrence "
  "threshold = half of the retained discovery cases; a donor qualifies when the model knows its fact, "
  "logit(o_f) − logit(o) ≥ 1.0 on the donor prompt (mirror of the clean-margin filter); up to five qualifying donors per case, "
  "taken in a fixed random order (`random.Random(2000 + case_id)`), per-case values = donor means, with the first donor alone as "
  "the sensitivity run; protocols Qwen3, Mixtral without BOS (the paper's) and Mixtral with BOS. GN baselines are the existing "
  "runs (`*_bos_alllayers`, `mixtral_nobos_alllayers`: base sweep + all-layer expert pass) restricted to exactly the same cases "
  "and split, and on the full paper set.\n")
w(f"Verification (`scripts/ext6_str_verify.py`, `results/verify_ext6_str_olmoe.json`): on OLMoE, {V['n_pairs']} (case, donor) pairs "
  f"against transformers hooks: prefill Δ max diff {V['delta_clean_maxdiff_vs_hf']:.3f} (one bf16 ulp); layer patch "
  f"max {V['layer_maxdiff']:.2f}, mean |diff| {V['layer_meanabsdiff']:.3f}, {100 * V['layer_frac_within_0.25']:.0f} % within 0.25, "
  f"rescue r = {V['layer_rescue_corr_vs_hf']:.3f}; literal expert patches ({V['expert_n']} rows) max {V['expert_maxdiff']:.2f}, "
  f"r = {V['expert_rescue_corr_vs_hf']:.3f}; identity {V['identity_maxdiff']:.3f}. The GN layer patch on the same model agrees with "
  f"transformers to max {V['calibration_gn_layer']['maxdiff']:.2f}, r = {V['calibration_gn_layer']['rescue_corr']:.3f}, so STR adds "
  "no new numerical error (it has no noise injection to amplify bf16 differences).\n")
w(tab("overview") + "\n")
w("Donor dispersion = median over validation cases with ≥ 2 donors of the SD of the block rescue at the paper's layer across the "
  "case's donors (|mean| = median absolute case mean); sign agreement = mean fraction of a case's donors whose rescue has the "
  "majority sign.\n")

# ---------------------------------------------------------------- layers
w("### Layer level\n")
w(tab("layers") + "\n")
w("![STR vs GN layer curves: raw (top) and normalised by the mean validation drop (bottom)](figures/ext6_str_layers.png)\n")
rows = []
for k in KEYS:
    s, g = L(k), L(k, "GN (same cases)")
    rows.append(f"{SHORT[k]}: STR L*{s['L_star']} (val {f(s['val_at_Lstar'])}; validation argmax L{s['sharpness']['top_layer']}, "
                f"2nd L{s['sharpness']['next_layer']}, gap {s['sharpness']['gap']:+.2f}) vs GN L*{g['L_star']} (val {f(g['val_at_Lstar'])}; "
                f"validation argmax L{g['sharpness']['top_layer']}, gap {g['sharpness']['gap']:+.2f}); per-case r at the paper layer "
                f"{S[k]['percase_corr_paper_layer']:.2f}")
w("- " + "\n- ".join(rows) + "\n")
dcl = [L(k)["mean_delta_clean"] for k in KEYS]
dco = [L(k)["mean_delta_clean"] - L(k)["mean_drop"] for k in KEYS]
dgn = [L(k, "GN (same cases)")["mean_delta_clean"] - L(k, "GN (same cases)")["mean_drop"] for k in KEYS]
w(f"Raw rescues are about twice as large under STR because the drop is (mean Δ_clean {min(dcl):+.1f} to {max(dcl):+.1f}; "
  f"Δ_corrupt {max(dco):+.1f} to {min(dco):+.1f} under STR vs Δ_noised {min(dgn):+.1f} to {max(dgn):+.1f} under GN). Normalised by the drop, the layer effect is the same in Qwen3 and {RED} smaller under STR in Mixtral — "
  "the direction Zhang & Nanda report for GPT-2 XL (GN peaks 2–5× STR's), but far weaker. The Mixtral no-BOS and BOS curves are "
  "flat over L19–L21 under both corruptions, so the layer argmax is not a stable fingerprint there (the paper already noted the "
  "small gap, Table 6).\n")

# ---------------------------------------------------------------- experts
w("### Expert level\n")
w(tab("experts") + "\n")
w(tab("named") + "\n")
w("Spec / drop and rescue / drop are population ratios (mean over validation cases / mean drop, paired bootstrap), the scale on "
  "which STR and GN are comparable.\n")
w(f"- **Qwen3.** L44E069 is selected under STR as under GN (discovery active {E(q, 44)['disc_active']}/{L(q)['n_disc']}), with a "
  f"larger and still specific effect: rescue {f(E(q, 44)['rescue'])}, Spec {f(E(q, 44)['spec'])}; per unit of drop "
  f"{r3(E(q, 44)['spec_norm'])} vs GN {r3(S[q]['named']['GN (same cases)']['L44E069']['spec_norm'])}. L42E115 is the L42 winner "
  f"(Spec {f(E(q, 42)['spec'])}). The clean top-8 coalition carries {E(q, 44)['coalition_clean']['mean'] / E(q, 44)['layer_same_pass']['mean']:.0%} "
  "of the L44 block.")
w(f"- **Mixtral, no BOS (paper protocol).** L19E006 is again the only recurrent L19 candidate "
  f"({E(mn, 19)['disc_active']}/{L(mn)['n_disc']}) and again not specific: rescue {f(E(mn, 19)['rescue'])}, Spec {f(E(mn, 19)['spec'])} "
  f"(GN same cases {f(S[mn]['named']['GN (same cases)']['L19E006']['spec'])}); the clean top-2 coalition "
  f"{f(E(mn, 19)['coalition_clean'])} recovers the block {f(E(mn, 19)['layer_same_pass'])}. L18E001 stays positively specific "
  f"({f(N(mn, 'L18E001')['spec'])}); L19E002 is specific ({f(N(mn, 'L19E002')['spec'])}) but below the recurrence gate "
  f"({N(mn, 'L19E002')['disc_active']}/{L(mn)['n_disc']}), exactly the GN picture of Directions 1 and 3.")
w(f"- **Mixtral, BOS.** At L19 the content expert E002 is selected with a positive Spec {f(E(mb, 19)['spec'])} (GN "
  f"{f(S[mb]['named']['GN (same cases)']['L19E002']['spec'])}); E006 is negatively specific ({f(N(mb, 'L19E006')['spec'])}). Because "
  f"the discovery argmax moves to L{L(mb)['L_star']}, the two-stage rule now reports L{L(mb)['L_star']}E001 "
  f"(Spec {f(E(mb, L(mb)['L_star'])['spec'])}); both are single, positive, specific experts (pattern A of Direction 2).\n")

# ---------------------------------------------------------------- equal norm (Mixtral)
EQ = {k: S[k].get("equal_norm_pair", {}) for k in KEYS if S[k].get("equal_norm_pair")}
if EQ:
    w("**Equal-norm active-pair check (Mixtral L19, paper Table 11 analogue).**\n")
    w(tab("eqnorm") + "\n")

    def eq(k, p, c):
        return f(EQ[k][p][c]["spec_eq"])
    w("Under STR every L19 expert is indistinguishable from its co-active partner once both patch vectors have the same norm: "
      f"E006 {eq(mn, 'L19E006', 'STR (donor mean)')} and E002 {eq(mn, 'L19E002', 'STR (donor mean)')} without BOS, E002 "
      f"{eq(mb, 'L19E002', 'STR (donor mean)')} and E006 {eq(mb, 'L19E006', 'STR (donor mean)')} with BOS. Under GN on the same cases "
      f"E002 kept a small direction advantage ({eq(mn, 'L19E002', 'GN (same cases)')} / {eq(mb, 'L19E002', 'GN (same cases)')}) and "
      f"E006 a small deficit without BOS ({eq(mn, 'L19E006', 'GN (same cases)')}; paper Table 11: −0.062 [−0.130, −0.003]). So in "
      "Mixtral the raw Spec differences at L19 (E006 negative, E002 positive) are, under STR, differences in how much each expert's "
      "update changes between the two facts, not in the direction of the update. The Qwen3 gate-matched / equal-norm control "
      "(Table 9) needs pair rows at L44, which the STR expert pass did not record.\n")

# ---------------------------------------------------------------- joint
w("### Joint layer × expert search\n")
w(tab("joint") + "\n")
jl = {k: S[k]["str_expert_layers"] for k in KEYS}
cov = ", ".join(f"{SHORT[k]} {len(jl[k])}" for k in KEYS)
w(f"Layers covered by the STR expert pass: {cov} (all layers when 48 / 32); the GN column is restricted to the same layers. "
  "Joint top-1 (STR / GN, same cases): "
  + "; ".join(f"{SHORT[k]} {J(k)[0]['pair']} / {J(k, 'GN (same cases)')[0]['pair']}" for k in KEYS) + ". ")
gl = []
for k in KEYS:
    g = S[k].get("joint_top1_full_set_gate", {}).get("STR (donor mean)")
    if g and g["full_disc_active"] < g["full_gate"]:
        gl.append(f"{SHORT[k]} {g['pair']} is active in {g['full_disc_active']}/{g['full_n_disc']} discovery cases of the full paper "
                  f"set (gate {g['full_gate']}), i.e. it enters the ranking only because the retained set is smaller")
w("The recurrence gate is half of the retained discovery cases (53–54 instead of the paper's 64/128). "
  + ("; ".join(gl) + " — under the paper's gate on the full set, the GN joint winner is L18E001 (Direction 1), which STR also "
     "finds positively specific. " if gl else "")
  + "Under STR the paper's two-stage choice is the joint top-1 in Qwen3 (under GN on the same cases L42E115 and L44E069 tie on "
  "discovery, Direction 1's second-locus result). Joint top-1 identical under STR and GN: "
  + ", ".join(f"{SHORT[k]} {'yes' if J(k)[0]['pair'] == J(k, 'GN (same cases)')[0]['pair'] else 'no'}" for k in KEYS)
  + (("; where it differs (discovery all-case rescue of the two leaders under each corruption: "
       + "; ".join(f"{SHORT[k]} STR {J(k)[0]['pair']} {J(k)[0]['disc_allcase']:+.2f} vs {J(k, 'GN (same cases)')[0]['pair']} "
                   f"{next((x['disc_allcase'] for x in J(k) if x['pair'] == J(k, 'GN (same cases)')[0]['pair']), float('nan')):+.2f}; "
                   f"GN {J(k, 'GN (same cases)')[0]['pair']} {J(k, 'GN (same cases)')[0]['disc_allcase']:+.2f} vs {J(k)[0]['pair']} "
                   f"{next((x['disc_allcase'] for x in J(k, 'GN (same cases)') if x['pair'] == J(k)[0]['pair']), float('nan')):+.2f}"
                   for k in KEYS if J(k)[0]['pair'] != J(k, 'GN (same cases)')[0]['pair']) + ")")
     if any(J(k)[0]['pair'] != J(k, 'GN (same cases)')[0]['pair'] for k in KEYS) else "")
  + ".\n")

# ---------------------------------------------------------------- donors
w("### Donor sensitivity\n")
w("First donor only (one replacement per case, the analogue of the paper's single noise draw). " + donor_sentence() + " Across a case's donors the block rescue at the paper's "
  "layer varies with a median SD of " + ", ".join(f"{S[k]['donor_dispersion_paper_layer']['median_sd_across_donors']:.2f}" for k in KEYS)
  + " logits (median |case mean| " + ", ".join(f"{S[k]['donor_dispersion_paper_layer']['median_abs_case_mean']:.2f}" for k in KEYS)
  + "), and " + ", ".join(f"{100 * S[k]['donor_dispersion_paper_layer']['mean_sign_agreement']:.0f} %" for k in KEYS)
  + " of a case's donors agree on the sign: the per-case effect is mostly a property of the clean fact, not of which other fact "
  "replaces it.\n")

# ---------------------------------------------------------------- reading
w("### Reading\n")
w("1. The paper's conclusions do not depend on its corruption. Under the corruption Zhang & Nanda recommend, Qwen3 still "
  "localises to L44 and to a positive specific expert L44E069, and Mixtral (paper protocol) still shows a validated mid-layer "
  "block whose single selected expert L19E006 is not specific while the routed coalition recovers it. The earlier extension "
  "findings also hold: L42E115 as Qwen3's second locus, L18E001 as a positive Mixtral expert, and the BOS dependence of Mixtral's "
  "L19 selection (E006 without BOS, E002 with). The one selection that changes is outside the paper's protocol: with BOS, STR "
  "ranks L21E001 above L19E002 in both the two-stage and the joint search, where GN had L19E002 first by a hair; both are "
  "positive, specific single experts inside the flat L19–L21 band, their validation intervals overlap, and the single-donor run "
  "returns L19E002 — a tie rather than a reversal.")
w("2. GN does not inflate the localisation here the way it does in GPT-2 XL. Normalised block rescue is unchanged in Qwen3 and "
  f"{RED} lower under STR in Mixtral; STR and GN layer curves correlate at r ≥ {min(S[k]['curve_corr_str_gn'] for k in KEYS):.2f}. A plausible reason is the patch site: the "
  "paper patches the final position late in the network, whereas Zhang & Nanda's GN/STR gap is at the last subject token in early "
  "MLPs, where GN's off-distribution embeddings act directly.")
w("3. STR changes the meaning of the corrupted run, and that matters for interpretation more than for selection: the corrupted "
  "run prefers the foil (Δ_corrupt ≈ −6), so a rescue measures how much of the *switch between two facts* a component carries; "
  "components that compute the same thing for both facts (e.g. \"the answer is a country\") cancel by construction. Qwen3's E069 "
  "keeps its share of the drop under this stricter reading, Mixtral's experts lose some of theirs, and at equal norm Mixtral's "
  "L19 experts no longer differ from their co-active partner (the specificity that remains is a magnitude effect).")
w("4. Protocol recommendation for this project: keep GN as the paper's protocol for comparability, report STR alongside it as "
  "the best-practice check, and report drop-normalised rescue for any comparison across corruptions.\n")
w("Caveats. STR covers 83–84 % of the paper cases (cases without a same-relation subject whose true object is the foil, or none "
  "of equal token length, drop out), so all comparisons use the GN runs on exactly the same cases. Donors are CounterFact facts "
  "the model prefers by ≥ 1 logit, not facts it necessarily outputs (the foil is the donor prompt's top-1 in 26–30 % of selected "
  "donors, as the true object is for 29 % of clean prompts). The relaxed-filter and our own strict sets were not re-run under STR.\n")
w("Files: `moetrace/ext6_str.py`; `scripts/ext6_str_{filter,sweep,expert,verify,analyze,text}.py`, `scripts/ext6_str_chain.sh`; runs "
  "`results/{qwen3_str,mixtral_nobos_str,mixtral_bos_str}` (`str_candidates.parquet` with every symmetric candidate and its "
  "Δ_donor, `str_sweep_rows.parquet` / `str_expert_rows.parquet` at donor level, `str_sweep_routing.parquet`, `sweep_cases.parquet`, "
  "`case_sets.json`, `run_meta.json`); tables `results/tables/ext6_str_*`; figure `results/figures/ext6_str_layers.*`; numbers "
  "`results/ext6_str_summary.json`.\n")

os.makedirs(os.path.join(ROOT, "sections"), exist_ok=True)
with open(os.path.join(ROOT, "sections", "ext6_str.md"), "w") as fh:
    fh.write("\n".join(out))
print("wrote results/sections/ext6_str.md")
