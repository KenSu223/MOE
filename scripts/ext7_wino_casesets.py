"""ext7: WinoGrande case sets (user decision (g), 2026-10-04).

Primary pool = pairs that pass the STR margin (Delta_A >= 1, Delta_B <= -1) under all three scanned protocols (Qwen3,
Mixtral BOS, Mixtral no BOS), so that cross-model comparisons use identical items and Mixtral no BOS can be added later on
the same pairs. Seed-0 shuffle (random.Random(0)) of the pool sorted by pair_idx -> discovery 128, validation 128,
replication discovery 128, replication validation 128 (pairs). Per-model sensitivity pools: a seed-1 sample of 256 pairs of
each model's own margin pool (128/128).

pair_idx = index of the HIT id in the sorted list of all train_xl twins (rule W1), stable across tokenizers.
Directed case id = 2 * pair_idx + d: d = 0 -> clean = prompt A, corrupted = prompt B, true = trig_a, foil = trig_b;
d = 1 -> clean = prompt B, corrupted = prompt A, true = trig_b, foil = trig_a.
Writes data/wino_str/case_sets.json"""
import json
import random
import sys

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
from moetrace.ext7_wino import load_twins

twins, _ = load_twins("train_xl")
pidx = {t.pair_id: i for i, t in enumerate(twins)}
R = "/home/ubuntu/MOE/results"
pools = {}
for proto in ("qwen3", "mixtral_bos", "mixtral_nobos"):
    p = pd.read_parquet(f"{R}/wino_{proto}/scan_pairs.parquet")
    pools[proto] = sorted(pidx[x] for x in p[p.margin].pair_id)
shared = sorted(set(pools["qwen3"]) & set(pools["mixtral_bos"]) & set(pools["mixtral_nobos"]))
order = list(shared)
random.Random(0).shuffle(order)
sets = {"discovery": order[:128], "validation": order[128:256],
        "rep_discovery": order[256:384], "rep_validation": order[384:512], "shared_unused": order[512:]}
own = {}
for proto in ("qwen3", "mixtral_bos"):
    o = list(pools[proto])
    random.Random(1).shuffle(o)
    own[proto] = {"discovery": o[:128], "validation": o[128:256]}


def directed(pairs):
    return [2 * i + d for i in pairs for d in (0, 1)]


out = {"note": __doc__, "pair_idx_of": {t.pair_id: i for i, t in enumerate(twins) if i in set(shared) | set(pools["qwen3"]) | set(pools["mixtral_bos"]) | set(pools["mixtral_nobos"])},
       "pools": {k: v for k, v in pools.items()}, "shared": shared,
       "pairs": sets, "own_pool_pairs": own,
       "directed": {k: directed(v) for k, v in sets.items()},
       "own_pool_directed": {m: {k: directed(v) for k, v in s.items()} for m, s in own.items()}}
json.dump(out, open("/home/ubuntu/MOE/data/wino_str/case_sets.json", "w"), indent=1)
print({k: len(v) for k, v in sets.items()}, "shared", len(shared), {k: len(v) for k, v in pools.items()})
