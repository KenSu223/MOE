"""ext14 C3 (p6-circuit): one-line headline of the concept-circuit results for the PROGRESS handshake (reads results/ext14ci_summary.json)."""
import json, sys
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np
s = json.load(open("/home/ubuntu/MOE/results/ext14ci_summary.json")).get("c3", {})
out = []
for proto in ("qwen3", "mixtral_bos"):
    r = s.get(proto)
    if not r or "knockout_specificity" not in r:
        out.append(f"{proto}: pending")
        continue
    S = {(x["concept"], x["part"]): x for x in r["knockout_specificity"]}
    parts = []
    for conc in ("SIZE_STATE", "COUNTRY", "PERSON_ROLE"):
        c = r["circuits"].get(conc, {})
        if c.get("insufficient") or (conc, "all") not in S:
            parts.append(f"{conc} insufficient")
            continue
        a, h, e = S[(conc, "all")], S.get((conc, "heads")), S.get((conc, "experts"))
        sel = (f", selectivity experts {e['selectivity']:.2f} vs heads {h['selectivity']:.2f}" if h and e else "")
        parts.append(f"{conc} {len(c['heads'])}H+{len(c['experts'])}E knockout F own {a['F_own']:.2f} / other {a['F_other_abs']:.2f}{sel}")
    out.append(f"{proto}: " + "; ".join(parts))
print(" | ".join(out))
