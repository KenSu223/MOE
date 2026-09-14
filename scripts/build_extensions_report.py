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
