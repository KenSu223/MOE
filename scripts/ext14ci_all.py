"""ext14 circuit (p6-circuit): re-run every CPU analysis of P6-5 from disk and rebuild the section.

  C1  scripts/ext14ci_c1_analyze.py      (neurons in experts, Mistral-7B dense analogue, upcycling tables)
  C2  scripts/ext14ci_relp_analyze.py    (RelP pilot validation, circuits, positions)
  C3  scripts/ext14ci_c3_existing.py --report (concept circuits from existing rows, AUROC, knockout matrix, add-back, edges)
  text scripts/ext14ci_text.py           (results/sections/ext14_circuit.md)
Each step is skipped when its inputs are missing; a failing step is reported and the others still run.

Usage: python scripts/ext14ci_all.py
"""
import os, subprocess, sys

ROOT = "/home/ubuntu/MOE"
STEPS = [["scripts/ext14ci_c1_analyze.py"], ["scripts/ext14ci_relp_analyze.py"],
         ["scripts/ext14ci_c3_existing.py", "--proto", "qwen3", "--report"],
         ["scripts/ext14ci_c3_existing.py", "--proto", "mixtral_bos", "--report"],
         ["scripts/ext14ci_text.py"]]


def main():
    bad = []
    for st in STEPS:
        if not os.path.exists(os.path.join(ROOT, st[0])):
            print("skip (missing script)", st[0])
            continue
        if st[0].endswith("c3_existing.py") and not os.path.exists(os.path.join(ROOT, "results/p6ci_c3_data", f"circuits_{st[2]}.json")):
            print("skip (no C3 runs yet)", " ".join(st))
            continue
        r = subprocess.run([sys.executable] + st, cwd=ROOT, capture_output=True, text=True)
        print(" ".join(st), "rc", r.returncode)
        if r.returncode:
            bad.append(st[0])
            print(r.stderr[-2000:])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
