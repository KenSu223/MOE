"""ext13 Phase-5 base runner job list for one condition and model (agent p5-data). Prints one line per pending GPU job,
"<job-name>\t<command>", in the Phase-3 order sweep -> joint -> dla -> direct -> expert -> heads -> grid (exact invocations
of scripts/ext13_p1_chain.sh on the condition's pair / case-set files). Jobs whose marker results/p5_<cond>_<proto>/
.done_<stage> exists are skipped (the chain writes the marker after rc = 0). The expert stage is split into layer halves
when the run has more than --max-dir-expert directed cases (keeps queued jobs <= ~25 min). Behaviour-only conditions print
nothing. While the registry entry has rebuild_pending = true the planner waits (polls every 60 s), so the chain blocks there.
Usage: python scripts/ext13_p5_jobs.py <cond> <proto>
"""
import argparse, json, os, sys, time

ROOT = "/home/ubuntu/MOE"
L = {"qwen3": 48, "mixtral": 32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond")
    ap.add_argument("proto")
    ap.add_argument("--max-dir-expert", type=int, default=1600)
    a = ap.parse_args()
    e = json.load(open(f"{ROOT}/data/wino_ref/p5_conditions.json"))["conditions"][a.cond][a.proto]
    while e.get("rebuild_pending"):  # case sets being rebuilt (cp_* options prefix): block the chain here until they are registered
        time.sleep(60)
        try:
            e = json.load(open(f"{ROOT}/data/wino_ref/p5_conditions.json"))["conditions"][a.cond][a.proto]
        except Exception:
            pass
    if e["behaviour_only"]:
        return
    model = e["model"]
    run = f"p5_{a.cond}_{a.proto}"
    od = f"{ROOT}/results/{run}"
    P = f"--model {model} --pairs {e['pair_file']} --case-sets {e['case_sets']} --out {run} --agent p5-data"
    fams = ",".join(e["families"])
    splits = "discovery,validation" + (",rep_discovery,rep_validation" if "rep" in e["families"] else "")
    n_dir = 2 * sum(e["split_sizes"].values())
    big = model == "qwen3"
    jobs = [("sweep", f"python scripts/ext7_wino_sweep.py {P} --max-rows {60000 if big else 40000}"),
            ("joint", f"python scripts/ext7_wino_joint.py {P} --families {fams} --max-rows 30000"),
            ("dla", f"python scripts/ext7_wino_dla.py {P} --families {fams} --chunk 512"),
            ("direct", f"python -m moetrace.ext7_controls direct {P.replace(' --agent p5-data', '')} --splits {splits}")]
    ms = 60000 if big else 40000
    if n_dir > a.max_dir_expert:
        h = L[model] // 2
        jobs += [("expert_lo", f"python scripts/ext7_wino_expert.py {P} --families {fams} --layers 0-{h - 1} --pairs-layers auto --max-spawn {ms}"),
                 ("expert_hi", f"python scripts/ext7_wino_expert.py {P} --families {fams} --layers {h}-{L[model] - 1} --pairs-layers auto --max-spawn {ms}")]
    else:
        jobs.append(("expert", f"python scripts/ext7_wino_expert.py {P} --families {fams} --layers all --pairs-layers auto --max-spawn {ms}"))
    jobs += [("heads", f"python scripts/ext7_wino_heads.py {P} --families main --layers auto --max-rows 50000"),
             ("grid", f"python scripts/ext7_wino_grid.py {P} --families main --kinds layer,attn_layer,resid --window 1 --max-rows 60000 --token-budget 200000")]
    for st, cmd in jobs:
        if os.path.exists(os.path.join(od, f".done_{st}")):
            continue
        print(f"ext13-p5-{a.cond}-{a.proto}-{st}\t{st}\t{cmd}")


if __name__ == "__main__":
    main()
