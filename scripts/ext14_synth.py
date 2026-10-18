"""Phase 6 Stage R (coordinator): cross-model synthesis figure and the pre-registration check table.

Reads results/ext14_summary.json (p6-analysis), results/ext14ci_summary.json (p6-circuit) and results/tables/ext14_dial_e.csv.
Writes
  results/figures/ext14syn_answer_type.{png,pdf}  attention share at E1 (stated) vs E3 (event) per direction and model
  results/tables/ext14_prereg_check.md             every pre-registered prediction: prediction, observed, verdict
Usage: python scripts/ext14_synth.py
"""
import csv, json, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/ubuntu/MOE/results"
INK, INK2, GRID = "#1f2328", "#52514e", "#e4e4e0"
C_A2E, C_E2A = "#2a78d6", "#eb6834"
MODELS = ["Qwen3-30B-A3B", "Mixtral-8x7B", "Qwen2-57B-A14B", "Mistral-7B", "Qwen2-7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]
MOE = {"Qwen3-30B-A3B", "Mixtral-8x7B", "Qwen2-57B-A14B"}


def parse(s):
    m = re.match(r"\s*([-+0-9.]+)\s*\[([-+0-9.]+),\s*([-+0-9.]+)\]", s or "")
    return tuple(float(x) for x in m.groups()) if m else None


def dial_rows():
    rows = {}
    with open(f"{R}/tables/ext14_dial_e.csv") as f:
        for r in csv.DictReader(f):
            m = r["Model"].replace("-Base", "").replace(" (BOS)", "")
            for k in MODELS:
                if m.startswith(k):
                    m = k
            rows[(m, r["Direction"], r["Level"])] = (parse(r["Share"]), "†" in r["Level"])
    return rows


def pick(rows, m, d, prefixes):
    for p in prefixes:
        for (mm, dd, lev), v in rows.items():
            if mm == m and dd == d and lev.startswith(p):
                return v
    return (None, False)


def all_strata_rows():
    """Mistral-7B E1 cells exist only as all-association-strata variants (ext14_dense_twins.csv, dense own set)."""
    out = {}
    with open(f"{R}/tables/ext14_dense_twins.csv") as f:
        for r in csv.DictReader(f):
            if r.get("Twin (dense → MoE)", "").startswith("Mistral-7B") and "(all strata)" in r.get("Cell", ""):
                d = "attribute → entity" if "a→e" in r["Cell"] else "entity → attribute"
                out[("Mistral-7B", d)] = (parse(r["Dense, own set"]), True)
    return out


def figure():
    rows = dial_rows()
    alls = all_strata_rows()
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    y = 0
    yt, yl = [], []
    for m in MODELS:
        for d, col in (("attribute → entity", C_A2E), ("entity → attribute", C_E2A)):
            e1, p1 = pick(rows, m, d, ["E1 stated"])
            if e1 is None and (m, d) in alls:
                e1, p1 = alls[(m, d)]
            e3, p3 = pick(rows, m, d, ["E3 event, P3", "E3 event (original)"] if d.startswith("attribute") else ["E3 event (original)"])
            if e1 is None and e3 is None:
                continue
            if e1 and e3:
                ax.annotate("", xy=(e1[0], y), xytext=(e3[0], y), arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
            for v, mk, pooled in ((e3, "o", p3), (e1, "D", p1)):
                if v:
                    ax.plot([v[1], v[2]], [y, y], color=col, lw=1.0, alpha=0.6)
                    ax.plot([v[0]], [y], marker=mk, color=col, ms=6, mfc="white" if pooled else col)
            yt.append(y)
            yl.append(f"{m}{' (MoE)' if m in MOE else ''}  ·  {'a→e' if d.startswith('attribute') else 'e→a'}")
            y -= 1
        y -= 0.6
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("attention share of the final-token repair (validation, 95% CI)")
    ax.grid(True, axis="x", color=GRID)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    h = [plt.Line2D([], [], color=INK2, marker="o", ls="", mfc="white"), plt.Line2D([], [], color=INK2, marker="o", ls=""),
         plt.Line2D([], [], color=INK2, marker="D", ls=""), plt.Line2D([], [], color=C_A2E, lw=2), plt.Line2D([], [], color=C_E2A, lw=2)]
    ax.legend(h, ["pooled or all-strata cell (open)", "E3: binding implied by the event", "E1: binding stated verbatim",
                  "attribute → entity (answer = entity)", "entity → attribute (answer = attribute)"],
              loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, fontsize=8, frameon=False)
    ax.set_title("Stating the binding (E3 → E1) vs the answer type, same WinoGrande twins", fontsize=10, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{R}/figures/ext14syn_answer_type.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print("written ext14syn_answer_type")


def short_obs(p):
    pm = p.get("per_model", {})
    parts = []
    for k, v in pm.items():
        if not isinstance(v, dict):
            continue
        val = None
        for key in ("value", "observed", "share", "spearman", "rate", "decision"):
            if key in v:
                val = v[key]
                break
        if isinstance(val, list) and len(val) == 3 and all(isinstance(x, (int, float)) for x in val):
            val = f"{val[0]:.2f} [{val[1]:.2f}, {val[2]:.2f}]"
        elif isinstance(val, float):
            val = f"{val:.2f}"
        parts.append(f"{k}: {val if val is not None else ''} ({v.get('verdict', '')})".replace(":  (", ": ("))
    return "; ".join(parts)


VERDICT_ORDER = {"held": 0, "partly": 1, "failed": 2, "insufficient data": 3, "insufficient (pooled only)": 3, "pending": 4}


def overall(p):
    vs = [v.get("verdict") for v in p.get("per_model", {}).values() if isinstance(v, dict) and v.get("verdict")]
    test = [v for v in vs if v in ("held", "failed")]
    if not vs or all(v == "pending" for v in vs):
        return "pending"
    if not test:
        return "insufficient"
    if all(v == "held" for v in test):
        return "held" if len(test) == len(vs) else "held where testable"
    if all(v == "failed" for v in test):
        return "failed" if len(test) == len(vs) else "failed where testable"
    return f"mixed ({sum(v == 'held' for v in test)} held / {sum(v == 'failed' for v in test)} failed)"


def circuit_overrides(P):
    """N1–N3 from p6-circuit's summary (C1 done; C3 partial)."""
    C = json.load(open(f"{R}/ext14ci_summary.json"))
    out = {}
    out["N1"] = ("Qwen3 experts L44E069 2.1, L42E115 4.3, L41E117 2.7 of 768 (per-case DLA ranking); indirect L37E032 511 by DLA, 33 by "
                 "|activation change| x weight", "held (L37E032 only with the magnitude ranking)")
    out["N2"] = ("Mixtral experts L19E002 6.7, L21E001 3.8, L18E001 9.9, L20E000 17 of 14,336", "held")
    out["N3"] = ("selectivity experts / heads: Qwen3 SIZE/STATE 0.98 / 0.56, PERSON-ROLE 0.95 / 0.82, COUNTRY 0.71 / 0.82; Mixtral SIZE/STATE "
                 "0.93 / 0.69, PERSON-ROLE 0.82 / 0.88, COUNTRY -0.35 / 0.74 (p6-circuit C3 DONE 2026-10-17 18:18Z)",
                 "partly (experts > heads in 3 of 6)")
    return out


def prereg_table():
    S = json.load(open(f"{R}/ext14_summary.json"))
    P = S["predictions"]
    over = circuit_overrides(P)
    L = ["**Phase 6 pre-registration check** (predictions committed in `data/dial/preregistration.md` / `predictions.json` before any Phase-6 "
         "GPU run, commit 4143e09; observed values from `results/ext14_summary.json` and `results/ext14ci_summary.json`; written by "
         "`scripts/ext14_synth.py`). 'Where testable' = models with a non-pooled cell of ≥ 64 validation pairs.", "",
         "| ID | Cell | Prediction | Observed | Verdict (as pre-registered, testable models) | Per model |", "|---|---|---|---|---|---|"]
    counts = {}
    for pid, p in P.items():
        if pid in over:
            obs, verd = over[pid]
            per = verd
        else:
            obs, per = short_obs(p), overall(p)
            vt = p.get("verdict_testable") or p.get("verdict") or "pending"
            verd = {"–": "insufficient", "": "pending"}.get(vt, vt)
            if verd.startswith("insufficient"):
                verd = "insufficient"
        key = verd.split(" ")[0]
        counts[key] = counts.get(key, 0) + 1
        L.append(f"| {pid} | {p.get('cell', '')} | {p.get('prediction', '')} | {obs} | **{verd}** | {per} |")
    L += ["", "Counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))]
    open(f"{R}/tables/ext14_prereg_check.md", "w").write("\n".join(L) + "\n")
    print("written ext14_prereg_check.md", counts)


if __name__ == "__main__":
    figure()
    prereg_table()
