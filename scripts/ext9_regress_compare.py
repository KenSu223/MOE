"""Compare a re-run verification JSON with its saved copy (ext9 engine merge regression).

Like scripts/ext8_regress_compare.py (timing fields skipped), but GPU peak-memory fields ("mem" in the key) are reported as
resource fields and not counted as differences: they are not results and are not reproducible run to run with an
unchanged engine (scripts/ext8_engine_verify.py with byte-identical engine and script gave pass1_peak_mem_GB 2.010271 vs
2.010057 in two runs on 2026-10-04, every result field identical).
Usage: python scripts/ext9_regress_compare.py NEW.json OLD.json"""
import json, sys


def is_timing(k):
    return k.endswith("_s") or "time" in k or "elapsed" in k


def is_resource(k):
    return "mem" in k


def walk(a, b, path, diffs, res):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if is_timing(k):
                continue
            if k not in a or k not in b:
                diffs.append((path + "/" + k, "missing"))
                continue
            if is_resource(k):
                if a[k] != b[k]:
                    res.append((path + "/" + k, a[k], b[k]))
                continue
            walk(a[k], b[k], path + "/" + k, diffs, res)
    elif a != b:
        diffs.append((path, (str(a)[:80], str(b)[:80])))


a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
diffs, res = [], []
walk(a, b, "", diffs, res)
n = 0


def count(x):
    global n
    if isinstance(x, dict):
        for k, v in x.items():
            if not is_timing(k) and not is_resource(k):
                count(v)
    else:
        n += 1


count(b)
print(f"{sys.argv[1]} vs {sys.argv[2]}: {n} non-timing result fields, {len(diffs)} differ; resource fields differing: {res}")
for d in diffs[:20]:
    print("  ", d)
sys.exit(1 if diffs else 0)
