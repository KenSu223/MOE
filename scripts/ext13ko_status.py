"""ext13ko / ext13wr completion check (CPU), used by the detached chains.

    python scripts/ext13ko_status.py <proto> <cond> [--phase main|cross] [--suffix S]   -> exit 0 if complete, 1 otherwise
    python scripts/ext13ko_status.py <proto> --ready                                    -> print the ready (BASE READY) conditions
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace import ext13ko as K
from moetrace import ext13wr as WR


def complete(proto, cond, phase="main", suffix=""):
    e = K.entry(cond, proto)
    od = K.out_dir(e, suffix)
    p = os.path.join(od, "run_meta.json")
    if not os.path.exists(p):
        return False
    ok = bool(json.load(open(p)).get("phases", {}).get(phase, {}).get("complete"))
    if phase == "main":
        ok = ok and WR.pass_done(e, suffix)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto")
    ap.add_argument("cond", nargs="?")
    ap.add_argument("--phase", default="main")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--ready", action="store_true")
    a = ap.parse_args()
    if a.ready:
        print(" ".join(K.ready_conditions(a.proto)))
        return 0
    return 0 if complete(a.proto, a.cond, a.phase, a.suffix) else 1


if __name__ == "__main__":
    sys.exit(main())
