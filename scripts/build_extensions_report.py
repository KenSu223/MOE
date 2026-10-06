"""Assemble results/EXTENSIONS_REPORT.md from the per-direction section files written by the extension agents.

Usage: python scripts/build_extensions_report.py
Sections are included in the fixed order below when present; missing ones are listed as pending. Each section file is
owned by one agent (results/sections/ext<N>_*.md) and is included verbatim, with its heading level normalised so that
the report has one H2 per direction. Image links in the sections are relative to results/, as in REPORT.md.
"""
import os, re, time

ROOT = "/home/ubuntu/MOE"
SECTIONS = [
    ("1", "Layer-then-expert versus joint layer × expert search", "results/sections/ext1_joint_search.md",
     "Does the paper's two-stage selection miss a stronger or more specific single expert in another layer?"),
    ("3", "Why the BOS token moves Mixtral's router", "results/sections/ext3_bos_mechanism.md",
     "Mechanism behind the BOS-dependence of Mixtral's expert-level result (hypotheses H1 sink relocation, H2 BOS semantics, "
     "H3 position shift, H4 default expert); literature review in docs/ext3_literature_review.md."),
    ("2", "Generalisation across models and attention-vs-MoE attribution", "results/sections/ext2_model_zoo.md",
     "Qwen3-30B-A3B-Instruct-2507, Qwen3-Coder-30B-A3B-Instruct, Mixtral-8x7B-Instruct, OLMoE base/Instruct under intended and "
     "paper protocols; attention-output / MoE-output / whole-layer rescue curves."),
    ("2b", "Attention-sublayer patching (engine extension)", "results/sections/ext2_attn_patch.md",
     "New intervention kinds attn_layer and block; verification against transformers hooks."),
    ("4", "Expert-aware tracing on code (CodeFact)", "results/sections/ext4_codefact.md",
     "CounterFact-style code counterfactuals (S1-S3 syntax, R1-R3 recall) on Python; per-category localisation and cross-category "
     "expert overlap."),
    # Phase 2 follow-ups (RESEARCH_PLAN.md "Phase 2"), wave 1: F4, F1, F2, F5.
    ("5-F4", "Patching at the last subject token", "results/sections/ext5_f4_subject.md",
     "Where does the noised subject's information get repaired when the patch is applied at the last subject token instead "
     "of the final position, and is there a shared expert there? (suffix-row executor moetrace/ext5_subject.py)"),
    ("5-F1", "Expert rankings and minimal sufficient sets", "results/sections/ext5_f1_rankings.md",
     "Do rescue, active-only rescue, Spec, block share and per-case percentile rank experts the same way; how many experts of a "
     "layer carry 50/80/90% of the block rescue; exhaustive subsets vs the additive approximation."),
    ("5-F2", "Attention heads at the final position", "results/sections/ext5_f2_heads.md",
     "Per-head decomposition of the attention-output rescue at the peak layers (Qwen3 L40/L43/L44, Mixtral L15/L18/L19/L24): "
     "mover heads, their specificity, additivity and minimal head sets."),
    ("5-F5", "Probability metrics alongside the logit difference", "results/sections/ext5_f5_metrics.md",
     "Do the layer/expert selections and effect sizes change under log p(true), p(true), rank of the true token and KL to the "
     "clean distribution instead of logit(true) - logit(foil)?"),
    # Best-practices check (Zhang & Nanda 2024, arXiv:2309.16042): corruption by symmetric token replacement.
    ("6", "Symmetric token replacement instead of Gaussian noise", "results/sections/ext6_str.md",
     "Do the paper's layer and expert selections survive when the subject is replaced by a same-relation subject whose answer is "
     "the foil (STR, the corruption recommended by Zhang & Nanda) instead of being noised (GN)?"),
    ("6b", "Layer × position grid under symmetric token replacement", "results/sections/ext6_str_grid.md",
     "Zhang & Nanda Section 4.1 / Figure 4 for the paper's MoE-output patch: which token positions and layers carry the "
     "rescue, is the last subject token special, and do logit difference and probability agree? (single layer and 5-layer "
     "sliding window)"),
    # Phase 3 (2026-10-04): add-back curves and WinoGrande / IOI under STR (RESEARCH_PLAN.md "Phase 3").
    ("7-8", "Phase 3 synthesis: expert add-back and attention vs MoE across tasks", "results/sections/ext7_synthesis.md",
     "How many experts restore the answer, are greedy add-back strategies good enough, and is WinoGrande attention-driven like "
     "IOI or MoE-driven like factual recall?"),
    ("7", "WinoGrande under symmetric token replacement", "results/sections/ext7_wino.md",
     "Filling the blank with each twin's answer and predicting the sentence-final trigger: which layers, sublayers, positions, "
     "heads and experts carry the repair?"),
    ("7b", "Role swap, IOI and the three-task comparison", "results/sections/ext7_controls.md",
     "Does a second WinoGrande corruption site (swapping the two candidates' roles) or IOI (the attention-driven reference) "
     "change the attention / MoE balance, measured with the same protocol?"),
    ("8", "Expert add-back curves", "results/sections/ext8_addback.md",
     "Patching experts back jointly at the final position: ceilings, saturation curves, how many experts restore the answer, "
     "and greedy vs better strategies (CounterFact STR and WinoGrande)."),
    # Phase 4 (2026-10-05): necessity, full-repair circuits, writer experts, completeness (RESEARCH_PLAN.md "Phase 4").
    ("9-12", "Phase 4 synthesis: necessity, full repair, writer experts, replication", "results/sections/ext9_synthesis.md",
     "Are the localised experts necessary and task-specific, how few heads and experts restore the answer fully, do the "
     "experts write the answer directly, and do the Phase-3 claims replicate?"),
    ("9", "Expert knockout: necessity and task specificity", "results/sections/ext9_knockout.md",
     "Removing an expert from the router's menu (the next-best expert takes its slot): does it hurt its own task, only its own "
     "task, and more than same-layer experts or generic text?"),
    ("10", "Head + expert add-back to full repair", "results/sections/ext10_circuit.md",
     "With single attention heads as candidates next to experts, how few components restore the answer at the final position, "
     "and how many of them are heads?"),
    ("11", "Writer vs computer experts", "results/sections/ext11_writer.md",
     "Do the localised experts write the answer directly into the logit or act through later layers; where else do they fire "
     "and what do they write in vocabulary space?"),
    ("12", "Completeness checks", "results/sections/ext12_complete.md",
     "Do the add-back results replicate on held-out splits, is the role/option-swap difference an item effect, and does "
     "Mixtral without BOS change the WinoGrande picture?"),
]


def normalise_headings(txt: str) -> str:
    """Shift the section's own headings so that its top heading becomes H3 (the report uses H2 per direction)."""
    levels = [len(m.group(1)) for m in re.finditer(r"^(#{1,6})\s", txt, flags=re.M)]
    if not levels:
        return txt
    shift = 3 - min(levels)
    if shift <= 0:
        return txt
    return re.sub(r"^(#{1,6})(\s)", lambda m: "#" * min(6, len(m.group(1)) + shift) + m.group(2), txt, flags=re.M)


def main():
    out = ["# Extensions of the expert-aware causal-tracing reproduction (arXiv 2606.03780)\n",
           f"Assembled {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from results/sections/ by scripts/build_extensions_report.py. "
           "Plan and decisions: RESEARCH_PLAN.md. Base reproduction: REPORT.md. Timeline: logs/PROGRESS.md.\n",
           "## Status\n", "| Direction | Question | Section | Status |", "|---|---|---|---|"]
    bodies = []
    for num, title, path, question in SECTIONS:
        full = os.path.join(ROOT, path)
        ok = os.path.exists(full) and os.path.getsize(full) > 0
        out.append(f"| {num} | {question} | `{path}` | {'included' if ok else 'pending'} |")
        if ok:
            txt = open(full).read()
            bodies.append(f"\n## Direction {num}: {title}\n\n" + normalise_headings(txt).strip() + "\n")
        else:
            bodies.append(f"\n## Direction {num}: {title}\n\n_Pending: no section file yet._\n")
    out.append("")
    out.extend(bodies)
    dst = os.path.join(ROOT, "results", "EXTENSIONS_REPORT.md")
    open(dst, "w").write("\n".join(out))
    print(dst, sum(len(b) for b in bodies), "chars")


if __name__ == "__main__":
    main()
