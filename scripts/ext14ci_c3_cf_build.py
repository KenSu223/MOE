"""ext14 C3 (p6-circuit): CounterFact STR pair sets per relation for the COUNTRY concept (and its control relations),
in the ext7 pair-file contract, so the Phase-3 generic runners (ext7_wino_sweep / expert / heads) run on them.

Relations: P17 (country of a place / organisation), P36 (capital of a country / region), P27 (citizenship),
P495 (country of origin); controls (non-country answers): P103 (native language), P178 (developer).
Per relation: CounterFact records in a seeded order (numpy default_rng(17)), at most --max-per-rel cases tokenised with the
model's tokenizer (Qwen3: tokenizer defaults; Mixtral / Mistral: with BOS), symmetric donors as Direction 6
(moetrace.ext6_str.candidates: same relation, true object = the case's foil, same template and token positions, same
continuation ids), the first 3 donors of each case's fixed random order scanned. One engine prefill pass gives Delta on the
clean prompts and the donors. Pair = (A = case prompt, B = first donor with Delta_donor <= -1 in the fixed order) for
cases with Delta_clean >= 1, i.e. margin >= 1 in both directions; trig_a = true object id, trig_b = foil id. Split per
relation: seeded shuffle (random.Random(23)), 50 / 50 discovery / validation, capped at --cap pairs per split.

Usage: python scripts/ext14ci_c3_cf_build.py <model key> [--no-special-tokens] [--max-per-rel 500] [--cap 64]
Outputs results/p6ci_c3_data/cf_<rel>_<proto>.parquet + case_sets_cf_<rel>_<proto>.json + build_cf_<proto>.json
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd
from transformers import AutoTokenizer
from moetrace.models import MODELS, RESULTS
from moetrace.arch import snapshot_dir
from moetrace.data import load_records, prepare_case
from moetrace import ext6_str as S

OUT = os.path.join(RESULTS, "p6ci_c3_data")
RELS = {"P17": "country", "P36": "capital", "P27": "citizenship", "P495": "origin", "P103": "language", "P178": "developer"}
N_DONOR_SCAN = 3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--proto", default=None)
    ap.add_argument("--max-per-rel", type=int, default=500)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=4096)
    ap.add_argument("--rels", default=",".join(RELS), help="comma list of relations (default: the C3 set)")
    ap.add_argument("--merge-as", default=None, help="also write one merged pair set cf_<name>_<proto> of all relations (C2 pilot)")
    a = ap.parse_args()
    rels = [r for r in a.rels.split(",") if r]
    st = not a.no_special_tokens
    proto = a.proto or (a.model + ("_bos" if st and a.model.startswith(("mixtral", "mistral")) else ""))
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(snapshot_dir(MODELS[a.model]["repo"]))
    recs = load_records()
    index = S.donor_index(recs)
    order = np.random.default_rng(17).permutation(len(recs))
    cases, cand = {}, []
    stats = {}
    for rel in rels:
        n_tok, n_cand = 0, 0
        for ri in order:
            r = recs[int(ri)]
            if r["requested_rewrite"]["relation_id"] != rel:
                continue
            c, why = prepare_case(r, tok, "space", st)
            if c is None:
                continue
            n_tok += 1
            ds = S.candidates(c, r, tok, index, st)[:N_DONOR_SCAN]
            if ds:
                cases[c.case_id] = (rel, c)
                n_cand += 1
                for d in ds:
                    cand.append(dict(case_id=c.case_id, rel=rel, order=d.order, donor_record=d.donor_record, donor_subject=d.subject,
                                     ids=d.ids))
            if n_cand >= a.max_per_rel:
                break
        stats[rel] = {"tokenised": n_tok, "with_donor": n_cand}
        log(f"{rel}: {n_tok} tokenised, {n_cand} with >= 1 symmetric donor")
    from moetrace.engine import Engine, PrefillSpec
    eng = Engine(MODELS[a.model]["repo"])
    ids = list(cases)
    pre = [PrefillSpec(cases[c][1].ids, cases[c][1].true_id, cases[c][1].foil_id) for c in ids]
    pre += [PrefillSpec(x["ids"], cases[x["case_id"]][1].true_id, cases[x["case_id"]][1].foil_id) for x in cand]
    deltas = []
    for s0 in range(0, len(pre), a.max_rows):
        res = eng.run(pre[s0:s0 + a.max_rows], [], record_routing=False)
        deltas.append(res.delta)
        log(f"pass rows {s0}-{s0 + len(pre[s0:s0 + a.max_rows])}")
    delta = np.concatenate(deltas)
    dclean = dict(zip(ids, delta[:len(ids)]))
    for j, x in enumerate(cand):
        x["delta_donor"] = float(delta[len(ids) + j])
    cd = pd.DataFrame(cand)
    summary = {"model": a.model, "proto": proto, "special_tokens": st, "relations": {}}
    merged = []
    for rel in rels:
        lab = RELS.get(rel, rel)
        rows = []
        for c in [c for c in ids if cases[c][0] == rel]:
            if dclean[c] < 1.0:
                continue
            dd = cd[(cd.case_id == c) & (cd.delta_donor <= -1.0)].sort_values("order")
            if not len(dd):
                continue
            d = dd.iloc[0]
            cs = cases[c][1]
            diff = [i for i, (p, q) in enumerate(zip(cs.ids, d["ids"])) if p != q]
            rows.append(dict(pair_id=f"cf|{rel}|{c}|{int(d.donor_record)}", ids_a=json.dumps([int(t) for t in cs.ids]),
                             ids_b=json.dumps([int(t) for t in d["ids"]]), str_pos=json.dumps(diff), trig_a=int(cs.true_id),
                             trig_b=int(cs.foil_id), relation=rel, concept="COUNTRY" if rel in ("P17", "P36", "P27", "P495") else "OTHER",
                             case_id=int(c), donor_record=int(d.donor_record), prompt_a=cs.prompt, word_a=cs.subject,
                             word_b=d.donor_subject, ans_a=cs.true_str, ans_b=cs.foil_str, delta_clean=float(dclean[c]),
                             delta_donor=float(d.delta_donor), n_tokens=len(cs.ids), names=False))
        pdf = pd.DataFrame(rows)
        merged.append(pdf)
        n = len(pdf)
        summary["relations"][rel] = {**stats[rel], "pairs_margin": n}
        if n == 0:
            continue
        idx = list(range(n))
        random.Random(23).shuffle(idx)
        half = n // 2
        disc, val = idx[:half][:a.cap], idx[half:][:a.cap]
        pdf = pdf.iloc[disc + val].reset_index(drop=True)
        nd = len(disc)
        cs_json = {"note": f"p6-circuit C3 CounterFact {rel} ({lab}) STR pairs for {proto}; scripts/ext14ci_c3_cf_build.py",
                   "pair_idx_of": {p: i for i, p in enumerate(pdf.pair_id)},
                   "pairs": {"discovery": list(range(nd)), "validation": list(range(nd, len(pdf)))},
                   "directed": {"discovery": [2 * i + d for i in range(nd) for d in (0, 1)],
                                "validation": [2 * i + d for i in range(nd, len(pdf)) for d in (0, 1)]}}
        pdf.to_parquet(os.path.join(OUT, f"cf_{rel}_{proto}.parquet"), index=False)
        json.dump(cs_json, open(os.path.join(OUT, f"case_sets_cf_{rel}_{proto}.json"), "w"), indent=1)
        summary["relations"][rel].update(discovery=nd, validation=len(pdf) - nd)
        log(f"{rel}: {n} margin pairs -> {nd} / {len(pdf) - nd}")
    if a.merge_as:
        pdf = pd.concat([m for m in merged if len(m)], ignore_index=True)
        idx = list(range(len(pdf)))
        random.Random(23).shuffle(idx)
        half = len(pdf) // 2
        disc, val = idx[:half][:a.cap], idx[half:][:a.cap]
        pdf = pdf.iloc[disc + val].reset_index(drop=True)
        nd = len(disc)
        cs_json = {"note": f"p6-circuit merged CounterFact STR pairs ({','.join(rels)}) for {proto}",
                   "pair_idx_of": {p: i for i, p in enumerate(pdf.pair_id)},
                   "pairs": {"discovery": list(range(nd)), "validation": list(range(nd, len(pdf)))},
                   "directed": {"discovery": [2 * i + d for i in range(nd) for d in (0, 1)],
                                "validation": [2 * i + d for i in range(nd, len(pdf)) for d in (0, 1)]}}
        pdf.to_parquet(os.path.join(OUT, f"cf_{a.merge_as}_{proto}.parquet"), index=False)
        json.dump(cs_json, open(os.path.join(OUT, f"case_sets_cf_{a.merge_as}_{proto}.json"), "w"), indent=1)
        summary["merged"] = {"name": a.merge_as, "discovery": nd, "validation": len(pdf) - nd}
    summary["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(summary, open(os.path.join(OUT, f"build_cf_{proto}" + (f"_{a.merge_as}" if a.merge_as else "") + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
