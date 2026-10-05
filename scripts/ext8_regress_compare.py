"""Compare a re-run verification JSON with its saved copy on every non-timing field (ext8 engine merge regression).
Usage: python scripts/ext8_regress_compare.py results/verify_olmoe.json results/verify_olmoe_before_ext8.json"""
import json, sys

def is_timing(k):
    return k.endswith("_s") or "time" in k or "elapsed" in k


def walk(a, b, path, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if is_timing(k):
                continue
            if k not in a or k not in b:
                diffs.append((path + "/" + k, "missing"))
                continue
            walk(a[k], b[k], path + "/" + k, diffs)
    elif a != b:
        diffs.append((path, (str(a)[:80], str(b)[:80])))


a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
diffs = []
walk(a, b, "", diffs)
n = 0
def count(x):
    global n
    if isinstance(x, dict):
        for k, v in x.items():
            if not is_timing(k):
                count(v)
    else:
        n += 1
count(b)
print(f"{sys.argv[1]} vs {sys.argv[2]}: {n} non-timing leaf fields, {len(diffs)} differ")
for d in diffs[:20]:
    print("  ", d)
sys.exit(1 if diffs else 0)
