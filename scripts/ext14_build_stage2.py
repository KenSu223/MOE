"""ext14 Phase 6 Stage-2 ladder anchors as cells (CPU, agent p6-data): l_ioi and l_cf for the models without Phase-3 runs.

  --what l_ioi   data/ioi/pairs_s2io_<qwen3|mixtral_bos>.parquet (Phase-3 IOI, S2 -> IO swap; identical texts under both tokenizers)
                 -> data/dial/build/l_ioi_<fam>.parquet in the Phase-6 build format. pair_idx = the Phase-3 item index; split = the
                 Phase-3 split for the Phase-3 discovery / validation items (preferred: rank first), the other items seeded half / half.
  --what l_cf --proto <p>   results/<model>_str (scripts/ext6_str_filter.py run of the model: paper case set, symmetric donors) ->
                 data/dial/build/l_cf_<proto>.parquet: one pair per case = (case prompt, its first selected donor prompt); answers =
                 the case's true object / foil (the donor's true object). STR = the subject tokens (CounterFact subjects can be several
                 tokens: the documented R2 exception of the Direction-6 anchors). split = the paper's discovery / validation.
Usage: python scripts/ext14_build_stage2.py --what l_ioi | --what l_cf --proto mistral7b_bos
"""
import argparse
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, "/home/ubuntu/MOE")
os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf")
from moetrace import ext14_data as X
from moetrace.models import MODELS, RESULTS

SRC = {"qwen": "qwen3", "mistral": "mixtral_bos"}


def build_ioi():
    cs = json.load(open(f"{X.ROOT}/data/ioi/case_sets_s2io.json"))
    main = {int(i): s for s in ("discovery", "validation") for i in cs["pairs"][s]}
    for fam, proto in SRC.items():
        p = pd.read_parquet(f"{X.ROOT}/data/ioi/pairs_s2io_{proto}.parquet")
        p["pair_idx"] = p.pair_id.map(cs["pair_idx_of"]).astype(int)
        rest = sorted(set(p.pair_idx) - set(main))
        random.Random(14).shuffle(rest)
        sp = dict(main)
        sp.update({i: ("discovery" if j < len(rest) // 2 else "validation") for j, i in enumerate(rest)})
        rank = {i: (0 if i in main else 1) * 100000 + j for j, i in enumerate(sorted(main) + rest)}
        p["split"] = p.pair_idx.map(sp)
        p["rank"] = p.pair_idx.map(rank)
        p["src_pair_id"] = p.pair_id
        p["order"] = p.template_type.map({"ABBA": 1, "BABA": 2})
        p["answer_a"], p["answer_b"] = p.word_a, p.word_b
        p["cell"], p["level"], p["direction"], p["source"] = "l_ioi", "l_ioi", "", "ioi_phase3"
        p["names"] = True
        p["phase3_main"] = p.pair_idx.isin(main)
        p.to_parquet(f"{X.BUILD}/l_ioi_{fam}.parquet", index=False)
        print(fam, "l_ioi pairs", len(p), "phase-3 main", int(p.phase3_main.sum()))


def build_cf(proto):
    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    from moetrace.protocol import cases_by_id
    from moetrace.data import load_paper_ids
    key, st, fam = X.PROTOS[proto]
    run = f"{RESULTS}/{key}_str"
    cand = pd.read_parquet(f"{run}/str_candidates.parquet")
    cs = json.load(open(f"{run}/case_sets.json"))
    first = cand[cand.selected & (cand.slot == 0)].set_index("case_id")
    ids = sorted(first.index.astype(int))
    tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[key]["repo"]))
    cases, _rej = cases_by_id(key, ids, tok=tok, special_tokens=st)
    split = {}
    for s in ("discovery", "validation"):
        for c in (cs.get("paper", {}) or {}).get(s, []) if isinstance(cs.get("paper"), dict) else []:
            split[int(c)] = s
    rows = []
    for c in ids:
        case = cases.get(c)
        if case is None:
            continue
        d = first.loc[c]
        ia, ib = list(case.ids), json.loads(d.ids)
        if len(ia) != len(ib):
            continue
        diff = [k for k, (x, y) in enumerate(zip(ia, ib)) if x != y]
        if not diff or diff[-1] == len(ia) - 1:
            continue
        rows.append({"pair_id": f"cf|{c}", "cell": "l_cf", "level": "l_cf", "direction": "", "src_pair_id": f"cf|{c}", "pair_idx": int(c),
                     "order": 1, "ids_a": json.dumps(ia), "ids_b": json.dumps(ib), "str_pos": json.dumps(diff), "trig_a": int(case.true_id),
                     "trig_b": int(case.foil_id), "n_tokens": len(ia), "prompt_a": case.prompt, "prompt_b": d.prompt,
                     "answer_a": case.true_str, "answer_b": case.foil_str, "relation": case.relation, "subject": case.subject,
                     "names": False, "source": "counterfact_str", "split": split.get(int(c), "discovery" if c % 2 == 0 else "validation"),
                     "rank": int(c), "donor_record": int(d.donor_record), "n_str": len(diff)})
    df = pd.DataFrame(rows)
    df.to_parquet(f"{X.BUILD}/l_cf_{proto}.parquet", index=False)
    print(proto, "l_cf pairs", len(df), df.split.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["l_ioi", "l_cf"])
    ap.add_argument("--proto", default=None)
    a = ap.parse_args()
    build_ioi() if a.what == "l_ioi" else build_cf(a.proto)
