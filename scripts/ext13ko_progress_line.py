"""Print a one-line summary of a finished condition (knockout + writer) for logs/PROGRESS.md (used by the chains).

    python scripts/ext13ko_progress_line.py <proto> <cond> [--suffix S]
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.models import RESULTS
from moetrace import ext13ko as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto")
    ap.add_argument("cond")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    S = json.load(open(os.path.join(RESULTS, f"ext13ko_summary{a.suffix}.json")))
    d = S["models"].get(a.proto, {}).get(a.cond)
    if not d:
        print(f"{a.cond}/{a.proto}: no knockout summary")
        return
    parts = []
    for t in d["targets"]["all"]:
        e = d["effects"].get(f"tgt:{t}", {}).get("full")
        if e:
            role = d["conditions"][f"tgt:{t}"]["meta"].get("role", "")
            parts.append(f"{t}({role}) F {e['F']:+.3f} [{e['F_lo']:+.3f}, {e['F_hi']:+.3f}]")
    s10 = d["sets"].get("k10")
    if s10:
        parts.append(f"top-10 F {s10['pop_full']:+.3f} (sub {s10['pop']:+.3f} vs random {s10['rand_mean']:+.3f}, max {s10['rand_max']:+.3f})")
    nl = d["effects"].get("null", {}).get("full")
    if nl:
        parts.append(f"null {nl['F']:+.3f}")
    b = d["baseline"]["full"]
    head = f"{a.cond}/{a.proto} knockout ({b['n_pairs']} eval pairs, pair acc {b['pair_acc']:.3f}, GPU {d['gpu_s'] / 60:.1f} min): "
    w = ""
    wp = os.path.join(RESULTS, f"ext13wr_summary{a.suffix}.json")
    if os.path.exists(wp):
        W = json.load(open(wp)).get("models", {}).get(a.proto, {}).get(a.cond, {})
        A = W.get("A")
        if A:
            sel = A["targets"].get("selected")
            for x in A["experts_val"]:
                if x["expert"] in (sel, A["targets"]["wg"]) and x.get("share"):
                    w += f"; writer {x['expert']} D/drop {x['D_frac']['v']:+.3f}, D/T {x['share']['v']:.2f}"
        B = W.get("B", {}).get("experts", {})
        for t in (d["targets"]["selected"], d["targets"]["wg"]):
            st = B.get(t, {}).get("delta") if t else None
            if st:
                w += f"; {t} delta median rank of r {st['median_rank_true']:.0f}"
    print(head + "; ".join(parts) + w)


if __name__ == "__main__":
    main()
