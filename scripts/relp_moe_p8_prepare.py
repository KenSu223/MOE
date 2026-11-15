#!/usr/bin/env python3
"""Phase 8 stage 3 (docs/relp_moe_phase8_plan.md section 8d): manifests for tasks other than SVA, in the Phase-7 row schema
(input_ids = clean run A, reference_ids = corrupted run B, target_id / foil_id, special_tokens_mask, changed_positions,
row_id, group_id) plus `roles` = {"subject": last token of the changed word, "final": last token}.
Prompt lengths vary across rows (A and B of a row always have the same length); no padding.

  counterfact --model qwen3|mixtral   STR pairs of Direction 6 (results/<run>_str): case x donor slot 0 / 1. discovery = the
      paper's discovery cases; the paper's validation cases, shuffled with seed 12: 15 cases calibration, the rest halved into
      validation / test. group_id = case. Token ids are re-derived with the pinned tokenizer and must equal the stored ids.
  twohop-candidates                    data/relp_p8/twohop_candidates.json (city, state, capital), fixed list.
  twohop --model ... --scan <parquet>  two-hop state capitals: pairs of cities that pass the competence scan (own capital's first
      token >= 1 logit above every other capital's), different states, same token count; both directions as rows; states
      shuffled with seed 12 into 60 % discovery / 20 % validation / 20 % test; calibration = discovery-state pairs not used
      for discovery; up to 300 / 40 / 40 / 40 rows.
CPU only (.venv-relp for the tokenizers). Output: data/relp_p8/<model>_<task>_v1/{manifest.json, <task>/<split>.jsonl}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/relp_p8"
TEMPLATE = "Fact: the capital of the state containing {} is"
STR_RUN = {"qwen3": "qwen3_str", "mixtral": "mixtral_bos_str"}
CAPITALS = {"Alabama": "Montgomery", "Alaska": "Juneau", "Arizona": "Phoenix", "Arkansas": "Little Rock", "California": "Sacramento",
            "Colorado": "Denver", "Connecticut": "Hartford", "Delaware": "Dover", "Florida": "Tallahassee", "Georgia": "Atlanta",
            "Hawaii": "Honolulu", "Idaho": "Boise", "Illinois": "Springfield", "Indiana": "Indianapolis", "Iowa": "Des Moines",
            "Kansas": "Topeka", "Kentucky": "Frankfort", "Louisiana": "Baton Rouge", "Maine": "Augusta", "Maryland": "Annapolis",
            "Massachusetts": "Boston", "Michigan": "Lansing", "Minnesota": "Saint Paul", "Mississippi": "Jackson",
            "Missouri": "Jefferson City", "Montana": "Helena", "Nebraska": "Lincoln", "Nevada": "Carson City",
            "New Hampshire": "Concord", "New Jersey": "Trenton", "New Mexico": "Santa Fe", "New York": "Albany",
            "North Carolina": "Raleigh", "North Dakota": "Bismarck", "Ohio": "Columbus", "Oklahoma": "Oklahoma City",
            "Oregon": "Salem", "Pennsylvania": "Harrisburg", "Rhode Island": "Providence", "South Carolina": "Columbia",
            "South Dakota": "Pierre", "Tennessee": "Nashville", "Texas": "Austin", "Utah": "Salt Lake City",
            "Vermont": "Montpelier", "Virginia": "Richmond", "Washington": "Olympia", "West Virginia": "Charleston",
            "Wisconsin": "Madison", "Wyoming": "Cheyenne"}
CITIES = {"Alabama": ["Birmingham", "Huntsville", "Mobile", "Tuscaloosa"], "Alaska": ["Anchorage", "Fairbanks"],
          "Arizona": ["Tucson", "Mesa", "Scottsdale", "Flagstaff", "Tempe"], "Arkansas": ["Fayetteville", "Fort Smith"],
          "California": ["Los Angeles", "San Diego", "San Francisco", "San Jose", "Fresno", "Oakland", "Pasadena", "Anaheim"],
          "Colorado": ["Boulder", "Colorado Springs", "Fort Collins"], "Connecticut": ["Bridgeport", "New Haven", "Stamford"],
          "Delaware": ["Wilmington"], "Florida": ["Miami", "Orlando", "Tampa", "Jacksonville"], "Georgia": ["Savannah", "Macon"],
          "Hawaii": ["Hilo"], "Idaho": ["Nampa", "Pocatello"], "Illinois": ["Chicago", "Peoria", "Naperville"],
          "Indiana": ["Fort Wayne", "Evansville", "Gary"], "Iowa": ["Cedar Rapids", "Davenport"], "Kansas": ["Wichita"],
          "Kentucky": ["Louisville", "Lexington"], "Louisiana": ["New Orleans", "Shreveport"], "Maine": ["Bangor"],
          "Maryland": ["Baltimore"], "Massachusetts": ["Worcester", "Cambridge"],
          "Michigan": ["Detroit", "Grand Rapids", "Ann Arbor", "Flint"], "Minnesota": ["Minneapolis", "Duluth"],
          "Mississippi": ["Biloxi", "Gulfport"], "Missouri": ["St. Louis"], "Montana": ["Billings", "Missoula", "Bozeman"],
          "Nebraska": ["Omaha"], "Nevada": ["Las Vegas", "Reno"], "New Hampshire": ["Manchester", "Nashua"],
          "New Jersey": ["Newark", "Jersey City", "Princeton"], "New Mexico": ["Albuquerque", "Roswell"],
          "New York": ["Buffalo", "Syracuse", "Brooklyn"], "North Carolina": ["Charlotte", "Greensboro", "Durham", "Asheville"],
          "North Dakota": ["Fargo"], "Ohio": ["Cleveland", "Cincinnati", "Toledo", "Akron", "Dayton"],
          "Oklahoma": ["Tulsa", "Norman"], "Oregon": ["Portland", "Eugene"],
          "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown", "Scranton"], "Rhode Island": ["Newport", "Warwick"],
          "South Carolina": ["Greenville", "Myrtle Beach"], "South Dakota": ["Sioux Falls", "Rapid City"],
          "Tennessee": ["Memphis", "Knoxville", "Chattanooga"],
          "Texas": ["Dallas", "Houston", "San Antonio", "El Paso", "Fort Worth", "Lubbock", "Amarillo"],
          "Utah": ["Provo", "Ogden"], "Vermont": ["Burlington"], "Virginia": ["Norfolk", "Virginia Beach", "Arlington"],
          "Washington": ["Seattle", "Spokane", "Tacoma"], "West Virginia": ["Morgantown", "Huntington"],
          "Wisconsin": ["Milwaukee", "Green Bay"], "Wyoming": ["Casper", "Laramie"]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT / f"data/relp_sva/tokenizer_{model}")


def write_manifest(model, task, splits, extra):
    """splits: {split: [rows]}. Calibration starts with 4 rows of its modal length (the verify gate batches them)."""
    cal = splits["calibration"]
    modal = Counter(len(r["input_ids"]) for r in cal).most_common(1)[0][0]
    first = [r for r in cal if len(r["input_ids"]) == modal][:4]
    assert len(first) == 4, "need 4 calibration rows of one length"
    splits["calibration"] = first + [r for r in cal if r not in first]
    d = OUT / f"{model}_{task}_v1"
    (d / task).mkdir(parents=True, exist_ok=True)
    files, audit = {}, {}
    for sp, rows in splits.items():
        f = d / task / f"{sp}.jsonl"
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        files[sp] = sha(f)
        audit[sp] = {"rows": len(rows), "groups": len({r["group_id"] for r in rows}),
                     "lengths": dict(sorted(Counter(len(r["input_ids"]) for r in rows).items()))}
    for a, b in (("discovery", "calibration"), ("discovery", "validation"), ("discovery", "test"), ("validation", "test"),
                 ("calibration", "validation"), ("calibration", "test")):
        assert not ({r["group_id"] for r in splits[a]} & {r["group_id"] for r in splits[b]}), (a, b)
    sva = json.loads((ROOT / f"data/relp_sva/{model}_paper_v1/manifest.json").read_text())
    man = {"schema": 1, "profile": "p8", "seed": 12, "model": sva["model"], "tokenizer_sha256": sva["tokenizer_sha256"],
           "source_revision": extra.pop("source_revision"), "tokenization": "raw prefix, add_special_tokens=True, no chat template, no padding",
           "roles": {"subject": "last token of the changed word", "final": "last token"},
           "prepare_script_sha256": sha(__file__), "tasks": {task: {"task": task, "files_sha256": files, "audit": audit, **extra}}}
    (d / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(d), "audit": audit}, indent=1))


def counterfact(model):
    import pandas as pd
    run = ROOT / "results" / STR_RUN[model]
    cases = pd.read_parquet(run / "sweep_cases.parquet")
    cand = pd.read_parquet(run / "str_candidates.parquet")
    cand = cand[cand.selected & (cand.slot < 2)].sort_values(["case_id", "slot"])
    tok = tokenizer(model)
    rows = {"discovery": [], "heldout": []}
    for c in cases.itertuples(index=False):
        ids = json.loads(c.ids)
        enc = tok(c.prompt, add_special_tokens=True, return_special_tokens_mask=True)
        assert enc["input_ids"] == ids, ("clean ids differ from the pinned tokenizer", c.case_id)
        subj = json.loads(c.subject_pos)
        for dnr in cand[cand.case_id == c.case_id].itertuples(index=False):
            ref = json.loads(dnr.ids)
            assert tok(dnr.prompt, add_special_tokens=True)["input_ids"] == ref and len(ref) == len(ids)
            changed = [i for i, (a, b) in enumerate(zip(ids, ref)) if a != b]
            assert changed and set(changed) <= set(subj) and subj[-1] < len(ids) - 1
            rows["discovery" if c.split_paper == "discovery" else "heldout"].append({
                "clean_prefix": c.prompt, "patch_prefix": dnr.prompt, "clean_answer": c.true_str, "patch_answer": c.foil_str,
                "case": c.relation, "row_id": f"cf:{int(c.case_id)}:{int(dnr.slot)}", "group_id": f"case{int(c.case_id)}",
                "input_ids": ids, "reference_ids": ref, "target_id": int(c.true_id), "foil_id": int(c.foil_id),
                "special_tokens_mask": enc["special_tokens_mask"], "changed_positions": changed, "subject_positions": subj,
                "roles": {"subject": subj[-1], "final": len(ids) - 1},
                "margin_clean_phase6": float(c.delta_clean), "margin_donor_phase6": float(dnr.delta_donor)})
    held_cases = sorted({r["group_id"] for r in rows["heldout"]}, key=lambda g: int(g[4:]))
    random.Random(12).shuffle(held_cases)
    cal, rest = set(held_cases[:15]), held_cases[15:]
    val, test = set(rest[: len(rest) // 2]), set(rest[len(rest) // 2:])
    splits = {"discovery": rows["discovery"], "calibration": [r for r in rows["heldout"] if r["group_id"] in cal],
              "validation": [r for r in rows["heldout"] if r["group_id"] in val],
              "test": [r for r in rows["heldout"] if r["group_id"] in test]}
    write_manifest(model, "counterfact", splits, {"source_revision": f"results/{STR_RUN[model]} (Direction 6 STR, donors slot 0-1)",
                   "source_sha256": {"sweep_cases.parquet": sha(run / "sweep_cases.parquet"),
                                     "str_candidates.parquet": sha(run / "str_candidates.parquet")},
                   "selection_uses_model_outputs": True,
                   "note": "cases = the paper's Table-8 cases with a qualifying STR donor (Direction 6: donor margin <= -1)"})


def twohop_items(model):
    """Every candidate city: ids, city token positions, first token of its capital."""
    tok = tokenizer(model)
    pre = tok("Fact: the capital of the state containing", add_special_tokens=True)["input_ids"]
    caps = {}
    for st, cap in CAPITALS.items():
        base = tok(TEMPLATE.format("Dallas"), add_special_tokens=True)["input_ids"]
        full = tok(TEMPLATE.format("Dallas") + " " + cap, add_special_tokens=True)["input_ids"]
        assert full[: len(base)] == base
        caps[st] = full[len(base)]
    items = []
    for st, cs in CITIES.items():
        for city in cs:
            enc = tok(TEMPLATE.format(city), add_special_tokens=True, return_special_tokens_mask=True)
            ids = enc["input_ids"]
            assert ids[: len(pre)] == pre and len(ids) >= len(pre) + 2
            items.append({"city": city, "state": st, "capital": CAPITALS[st], "prompt": TEMPLATE.format(city), "ids": ids,
                          "special_tokens_mask": enc["special_tokens_mask"], "city_positions": list(range(len(pre), len(ids) - 1)),
                          "capital_token": caps[st]})
    return items, caps


def twohop(model, scan, min_margin, relaxed_from=None):
    import pandas as pd
    items, caps = twohop_items(model)
    sc = pd.read_parquet(scan).set_index("city")
    ok = [it for it in items if float(sc.loc[it["city"], "margin"]) >= min_margin]
    states = sorted({it["state"] for it in ok})
    random.Random(12).shuffle(states)
    nd, nv = round(0.6 * len(states)), round(0.2 * len(states))
    part = {s: "discovery" for s in states[:nd]}
    part.update({s: "validation" for s in states[nd:nd + nv]})
    part.update({s: "test" for s in states[nd + nv:]})

    def pairs(p):
        its = [it for it in ok if part[it["state"]] == p]
        out = []
        for i in range(len(its)):
            for j in range(i + 1, len(its)):
                a, b = its[i], its[j]
                if a["state"] != b["state"] and len(a["ids"]) == len(b["ids"]) and a["capital_token"] != b["capital_token"]:
                    out.append((a, b))
        random.Random(12).shuffle(out)
        return out

    def rows_of(pl):
        out = []
        for a, b in pl:
            for x, y in ((a, b), (b, a)):
                ids, ref = x["ids"], y["ids"]
                changed = [i for i, (u, v) in enumerate(zip(ids, ref)) if u != v]
                assert changed and set(changed) <= set(x["city_positions"])
                out.append({"clean_prefix": x["prompt"], "patch_prefix": y["prompt"], "clean_answer": " " + x["capital"],
                            "patch_answer": " " + y["capital"], "case": f"{len(x['city_positions'])}tok",
                            "row_id": f"twohop:{x['city']}>{y['city']}", "group_id": "|".join(sorted([a["city"], b["city"]])),
                            "input_ids": ids, "reference_ids": ref, "target_id": x["capital_token"], "foil_id": y["capital_token"],
                            "special_tokens_mask": x["special_tokens_mask"], "changed_positions": changed,
                            "subject_positions": x["city_positions"],
                            "roles": {"subject": x["city_positions"][-1], "final": len(ids) - 1},
                            "state_clean": x["state"], "state_patch": y["state"],
                            "scan_margin_clean": float(sc.loc[x["city"], "margin"]), "scan_margin_patch": float(sc.loc[y["city"], "margin"])})
        return out

    dp, vp, tp = pairs("discovery"), pairs("validation"), pairs("test")
    ncal = min(20, max(2, len(dp) // 8))
    splits = {"calibration": rows_of(dp[:ncal]), "discovery": rows_of(dp[ncal:ncal + 150]),
              "validation": rows_of(vp[:20]), "test": rows_of(tp[:20])}
    nrows = {k_: len(v_) for k_, v_ in splits.items()}
    # rule fixed in the plan (S3.2): fewer than 60 discovery or 20 held-out rows at margin >= 1 -> relax to margin >= 0 and report;
    # fewer than 40 rows in total -> no run for this model
    if (nrows["discovery"] < 60 or nrows["validation"] + nrows["test"] < 20) and min_margin > 0:
        print(json.dumps({"relax": "too few rows at margin >= %g" % min_margin, "rows": nrows}))
        return twohop(model, scan, 0.0, relaxed_from={"min_margin": min_margin, "rows": nrows})
    if sum(nrows.values()) < 40 or nrows["validation"] == 0 or nrows["test"] == 0 or nrows["calibration"] < 4:
        raise SystemExit("two-hop: too few passing pairs for %s: %s" % (model, nrows))
    funnel = {"candidates": len(items), "pass_scan": len(ok), "min_margin": min_margin, "relaxed_from": relaxed_from, "states_passing": len(states),
              "states": {p: sorted(s for s in states if part[s] == p) for p in ("discovery", "validation", "test")},
              "unordered_pairs_available": {"discovery": len(dp), "validation": len(vp), "test": len(tp)}}
    print(json.dumps(funnel, indent=1))
    write_manifest(model, "twohop", splits, {"source_revision": "fixed list in scripts/relp_moe_p8_prepare.py + competence scan",
                   "scan_sha256": sha(scan), "funnel": funnel, "selection_uses_model_outputs": True, "template": TEMPLATE})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("counterfact", "twohop-candidates", "twohop"))
    p.add_argument("--model", choices=("qwen3", "mixtral"))
    p.add_argument("--scan", type=Path)
    p.add_argument("--min-margin", type=float, default=1.0)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.command == "counterfact":
        counterfact(a.model)
    elif a.command == "twohop-candidates":
        for model in ("qwen3", "mixtral"):
            items, caps = twohop_items(model)
            (OUT / f"twohop_candidates_{model}.json").write_text(json.dumps({"template": TEMPLATE, "capital_tokens": caps, "items": items}, indent=1) + "\n")
            print(model, len(items), "cities;", dict(sorted(Counter(len(i["city_positions"]) for i in items).items())), "city tokens;",
                  "capital first tokens shared:", [k for k, v in Counter(caps.values()).items() if v > 1])
    else:
        twohop(a.model, a.scan, a.min_margin)


if __name__ == "__main__":
    main()
