"""Section writer for Phase 5 / Direction 11 per condition: results/sections/ext13_writer.md (from
results/ext13wr_summary<suffix>.json and the ext13wr_* tables).

    python scripts/ext13wr_text.py [--suffix _smoke]
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
from moetrace import ext13ko as K
from moetrace.models import RESULTS

SEC = os.path.join(RESULTS, "sections")
TAB = os.path.join(RESULTS, "tables")


def embed(name, sfx):
    p = os.path.join(TAB, f"{name}{sfx}.md")
    if not os.path.exists(p):
        return "_(table not available)_\n"
    lines = open(p).read().strip().splitlines()
    if lines and lines[0].startswith("**"):
        lines = lines[2:]
    return "\n".join(lines) + "\n"


def ci(d, digits=3):
    if not d or d.get("v") is None or not np.isfinite(d.get("v", np.nan)):
        return "n/a"
    return f"{d['v']:+.{digits}f} [{d['lo']:+.{digits}f}, {d['hi']:+.{digits}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    sfx = a.suffix
    S = json.load(open(os.path.join(RESULTS, f"ext13wr_summary{sfx}.json")))
    out = []
    summ = ["**Summary.** Writer vs computer (Phase 4 Direction 11) per Phase-5 condition: the single-expert STR patch rescue T "
            "(p5-data's expert rows) split into the expert's direct write D (DLA of δ_e = c_e(clean) − c_e(corrupted twin), final "
            "norm frozen at the corrupted twin; one prefill pass per condition with `DiagSpec.contrib_dla`) and the indirect rest "
            "I = T − D; vocabulary projections of the targets' writes (`DiagSpec.contrib_final_vectors`)."]
    for proto, cd in S["models"].items():
        lab = K.PROTO[proto]["label"]
        wg = K.le(*K.PROTO[proto]["wg"])
        xs = []
        for c, r in cd.items():
            A = r.get("A")
            if not A:
                continue
            sel = A["targets"].get("selected")
            bits = []
            for t in dict.fromkeys([sel, wg]):
                if not t:
                    continue
                x = next((y for y in A["experts_val"] if y["expert"] == t), None)
                if x and x.get("share"):
                    B = r["B"]["experts"].get(t, {}).get("delta", {})
                    rk = f", δ_e rank of r {B['median_rank_true']:.0f}" if B else ""
                    bits.append(f"{t} D/drop {x['D_frac']['v']:+.3f}, D/T {x['share']['v']:.2f}{rk}")
            s10 = A["set_pop10_val"]
            bits.append(f"top-10 D/T {s10['share']['v']:.2f}" if np.isfinite(s10["share"]["v"]) else "top-10 D/T n/a")
            xs.append(f"{c}: " + ", ".join(bits))
        if xs:
            summ.append(f"{lab} (validation cases) — " + " | ".join(xs) + ".")
    out.append(" ".join(summ) + "\n")
    out.append("### What was run\n")
    out.append("For each condition × model one prefill pass over both twins of every pair of the case sets (discovery, validation, "
               "replication) records, at the final position, every routed expert's DLA numerator (c_e ⊙ γ)·(W_U[r] − W_U[r']) and the "
               "final RMS (`DiagSpec.contrib_dla`), the contribution vectors of the experts of interest (targets + the condition's "
               "population top-10; `contrib_final_vectors`) and the final residual. For directed case (X clean, Y = twin corrupted) "
               "D = (cdla_X + cdla_Y) / rms_Y, since the twin's readout direction is the negative of X's; T comes from p5-data's "
               "single-expert patch rows of the same case (a different pass: (case, expert) pairs whose routing differs between the two "
               "passes are dropped; routing match in the band table). Statistics as `moetrace/ext11_writer.py` (cluster bootstrap over "
               "the condition's units). Vocabulary: projections ((v ⊙ γ) W_Uᵀ) / rms with the norm frozen at the corrupted twin (δ_e) or "
               "at the prompt (c_e); classes = the condition's own attribute words (trigger words; readout class of entity → attribute "
               "conditions), its entities (answers; readout class of attribute → entity conditions), Phase 3's trigger vocabulary, IOI "
               "names, CounterFact objects.\n")
    out.append("\n### Part A. Total vs direct vs indirect, by expert\n")
    out.append(embed("ext13wr_A_experts", sfx))
    out.append("\n### Part A. Summed writes by depth band\n")
    out.append(embed("ext13wr_A_bands", sfx))
    out.append("\n### Part B. Vocabulary projections\n")
    out.append(embed("ext13wr_B_vocab", sfx))
    out.append("\n### Caveats\n")
    out.append("- D freezes the final norm (Phase 4: single-expert exact / frozen 0.996-0.999; the exact-norm check is in "
               "`vec_rows.parquet` per vector).\n- T and D come from different passes (bf16 batch noise in T, Phase 4: per-row SD "
               "≈ 0.1-0.3 logits); shares are ratios of sums over many cases.\n- A direct share above 1 means later layers partly "
               "undo the expert's write; a share near 0 marks a computer (acts through later layers).\n"
               f"- Figures: `results/figures/ext13wr_layers_<proto>_<cond>{sfx}.png`, `results/figures/ext13wr_overview{sfx}.png`.\n")
    os.makedirs(SEC, exist_ok=True)
    p = os.path.join(SEC, f"ext13_writer{sfx}.md")
    open(p, "w").write("\n".join(out))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
