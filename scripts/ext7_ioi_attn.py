"""ext7-controls W8 / W5 support (GPU, prefill only): final-position attention of every head on the IOI name positions.

For the items of the IOI case sets (discovery, validation, replication; data/ioi/case_sets_s2io.json) one prefill pass
with DiagSpec(attn_final=True) over three prompts per item: clean, (i) S2 -> IO, (ii) S1 and IO -> other names. Records
the attention probability from the final position to the IO, S1 and S2 token positions (and position 0) for every
layer and head. Used to classify the heads found by attn_head patching (name-mover-like = high clean attention to IO
that moves to S1 in the S2 -> IO corrupted run, where S1 holds the new IO name; Zhang & Nanda Figure 3).

Usage: python scripts/ext7_ioi_attn.py <proto: qwen3|mixtral_bos>
Outputs results/ioi_<proto>/attn_names.npz: mass float16 [3 prompts, n_items, L, n_heads, 4 targets (IO, S1, S2,
pos0)], pair_idx [n_items], prompts, targets; run_meta.json key "attn_names"
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from moetrace import ext7_controls as C
from moetrace.models import MODELS, RESULTS
from moetrace.engine import Engine, PrefillSpec, DiagSpec

PROMPTS = ("clean", "s2io", "s1io")
TARGETS = ("IO", "S1", "S2", "pos0")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto", choices=list(C.PROTOS))
    ap.add_argument("--max-rows", type=int, default=800)
    args = ap.parse_args()
    key, st = C.PROTOS[args.proto]
    od = os.path.join(RESULTS, f"ioi_{args.proto}")
    cs = json.load(open(os.path.join(C.IOI_DIR, "case_sets_s2io.json")))
    idx = sorted({i for k in ("discovery", "validation", "rep_discovery", "rep_validation") for i in cs["pairs"][k]})
    p1 = pd.read_parquet(os.path.join(C.IOI_DIR, f"pairs_s2io_{args.proto}.parquet"))
    p2 = pd.read_parquet(os.path.join(C.IOI_DIR, f"pairs_s1io_{args.proto}.parquet"))
    eng = Engine(MODELS[key]["repo"])
    L, H = eng.spec.n_layers, eng.spec.n_heads
    mass = np.zeros((3, len(idx), L, H, 4), dtype=np.float16)
    rows = []
    for j, i in enumerate(idx):
        r1, r2 = p1.iloc[i], p2.iloc[i]
        for k, ids in enumerate((r1.ids_a, r1.ids_b, r2.ids_b)):
            rows.append((k, j, int(r1.io_pos), int(r1.s1_pos), int(r1.s2_pos), PrefillSpec(json.loads(ids), int(r1.trig_a), int(r1.trig_b))))
    t0 = time.time()
    deltas = np.zeros((3, len(idx)))
    for s0 in range(0, len(rows), args.max_rows):
        ch = rows[s0: s0 + args.max_rows]
        res = eng.run([r[5] for r in ch], [], record_routing=False, diag=DiagSpec(attn_final=True), log=log if s0 == 0 else None)
        A = res.extra["diag"]["attn_final"]  # [L, B, H, T]
        for b, (k, j, pio, ps1, ps2, _) in enumerate(ch):
            mass[k, j] = np.stack([A[:, b, :, pio], A[:, b, :, ps1], A[:, b, :, ps2], A[:, b, :, 0]], axis=-1)
            deltas[k, j] = res.delta[b]
        log(f"rows {s0 + len(ch)}/{len(rows)}")
    np.savez_compressed(os.path.join(od, "attn_names.npz"), mass=mass, pair_idx=np.array(idx), prompts=np.array(PROMPTS),
                        targets=np.array(TARGETS), delta=deltas)
    mp = os.path.join(od, "run_meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    meta["attn_names"] = {"model": key, "special_tokens": st, "n_items": len(idx), "prompts": PROMPTS, "targets": TARGETS,
                          "command": "python " + " ".join(sys.argv), "seconds": time.time() - t0,
                          "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(meta, open(mp, "w"), indent=1)
    m = mass.astype(np.float32).mean(1)  # [3, L, H, 4]
    top = np.dstack(np.unravel_index(np.argsort(-m[0, :, :, 0].ravel())[:10], (L, H)))[0]
    for l, h in top:
        log(f"L{l}H{h}: clean IO {m[0, l, h, 0]:.2f} S1 {m[0, l, h, 1]:.2f} S2 {m[0, l, h, 2]:.2f} | s2io IO {m[1, l, h, 0]:.2f} "
            f"S1 {m[1, l, h, 1]:.2f} | s1io IO {m[2, l, h, 0]:.2f}")


if __name__ == "__main__":
    main()
