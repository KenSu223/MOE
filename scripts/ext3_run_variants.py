"""ext3: one GPU pass for several position-0 variants of the paper case set (Mixtral or Qwen3).

For every variant: clean + noised prefill rows of the paper's 256 cases, MoE-block (layer) patches at every layer, and
at the expert layer(s) the layer patch, both coalition patches and an expert patch for EVERY expert (post-hoc split
into clean-active / noised-only rows using the routing recorded in the same pass). Diagnostics (final-position
attention per head, per-position residual norms, final-position router logits and residuals, next-token log-probs)
are recorded for all prefill rows.

Outputs per variant (results/<run_dir>/): case_sets.json, sweep_rows/sweep_routing/sweep_cases.parquet,
sweep_summary.json, expert_rows.parquet, expert_prefill_L<l>.parquet, run_meta.json, diag_summary.json;
raw diagnostics in /opt/dlami/nvme/moe_ext3/<run_dir>/diag.npz (+ resid_final.pt).

Usage: python scripts/ext3_run_variants.py <model_key> --variants bos,nobos,... [--expert-layers 19] [--sigma-mult 3.0]
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/ubuntu/MOE")
import numpy as np, pandas as pd, torch
from moetrace.models import MODELS, RESULTS
from moetrace.protocol import cases_by_id, membership_table
from moetrace.engine import Engine, PrefillSpec, SpawnSpec, DiagSpec
from moetrace.noise import noise_draw
from moetrace.ext3_variants import VARIANTS, diag_dir, write_run_meta


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--expert-layers", default=None, help="comma list; default: the paper's layer for the model")
    ap.add_argument("--sigma-mult", type=float, default=3.0)
    ap.add_argument("--base-run", default=None, help="run whose case_sets.json['paper'] defines the cases (default: model key)")
    ap.add_argument("--no-resid-final", action="store_true")
    ap.add_argument("--skip-outputs", default="", help="comma list of variants that are in the pass (e.g. as sink donors) but whose outputs are not written")
    args = ap.parse_args()
    m = MODELS[args.model]
    variants = [VARIANTS[args.model][k] for k in args.variants.split(",")]
    exp_layers = [int(x) for x in (args.expert_layers or str(m["paper_layer"])).split(",")]
    with open(os.path.join(RESULTS, args.base_run or args.model, "case_sets.json")) as f:
        paper = json.load(f)["paper"]
    sets = {"paper": {"discovery": paper["discovery"], "validation": paper["validation"]}}
    all_ids = paper["discovery"] + paper["validation"]
    n = len(all_ids)

    from transformers import AutoTokenizer
    from moetrace.arch import snapshot_dir
    tok = AutoTokenizer.from_pretrained(snapshot_dir(m["repo"]))
    cases_v = {}
    for v in variants:
        cases, rej = cases_by_id(args.model, all_ids, tok=tok, special_tokens=v.special_tokens, prefix_ids=list(v.prefix_ids) or None)
        assert not rej, (v.key, rej)
        cases_v[v.key] = cases
        c0 = cases[all_ids[0]]
        log(f"variant {v.key:10s} -> {v.run_dir}: T range {min(len(c.ids) for c in cases.values())}-{max(len(c.ids) for c in cases.values())}, "
            f"example ids[:4]={c0.ids[:4]} subject_pos[:3]={c0.subject_pos[:3]} offset={v.pos_offset}")

    eng = Engine(m["repo"])
    sigma = args.sigma_mult * eng.embed_std
    Hd, L, E, k = eng.hidden, eng.spec.n_layers, eng.spec.n_experts, eng.spec.top_k
    pre, spawns, tags = [], [], []
    base = {v.key: 2 * n * vi for vi, v in enumerate(variants)}  # first prefill row of each variant (clean rows, then noised rows)
    for vi, v in enumerate(variants):
        cases = cases_v[v.key]
        b = len(pre)
        assert b == base[v.key]
        if v.sink_from:
            assert v.sink_from in base, f"{v.key} needs variant {v.sink_from} in the same pass"
            donors = [base[v.sink_from] + i for i in range(n)]  # the donor variant's CLEAN row of the same case
        else:
            donors = [-1] * n
        pre += [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id, pos_offset=v.pos_offset,
                            sink_donor=donors[i], sink_vscale=v.sink_vscale) for i, c in enumerate(all_ids)]
        pre += [PrefillSpec(cases[c].ids, cases[c].true_id, cases[c].foil_id, cases[c].subject_pos,
                            noise_draw(c, len(cases[c].subject_pos), Hd, sigma), pos_offset=v.pos_offset,
                            sink_donor=donors[i], sink_vscale=v.sink_vscale) for i, c in enumerate(all_ids)]
        for i, c in enumerate(all_ids):
            for l in range(L):
                spawns.append(SpawnSpec(l, b + n + i, b + i, "layer")); tags.append((v.key, c, l, "sweep_layer", -1))
            for l in exp_layers:
                spawns.append(SpawnSpec(l, b + n + i, b + i, "layer")); tags.append((v.key, c, l, "layer", -1))
                spawns.append(SpawnSpec(l, b + n + i, b + i, "coalition_clean")); tags.append((v.key, c, l, "coalition_clean", -1))
                spawns.append(SpawnSpec(l, b + n + i, b + i, "coalition_union")); tags.append((v.key, c, l, "coalition_union", -1))
                for e in range(E):
                    spawns.append(SpawnSpec(l, b + n + i, b + i, "expert", expert=e)); tags.append((v.key, c, l, "expert", e))
    early = tuple(sorted(set([0, 1, 2, 3] + exp_layers)))  # all-token routing at the early (sink-forming) layers and the expert layer
    diag = DiagSpec(attn_final=True, resid_norms=True, router_logits_final=True, resid_final=not args.no_resid_final, token_logprobs=True,
                    route_all_layers=early)
    log(f"pass: {len(variants)} variants, {len(pre)} prefill rows, {len(spawns)} spawn rows, expert layers {exp_layers}")
    t0 = time.time()
    res = eng.run(pre, spawns, record_routing=True, log=log, diag=diag)
    log(f"pass time {time.time() - t0:.1f}s  T={res.extra['T']} row_chunk={res.extra['row_chunk']} offsets={res.extra['use_offsets']} sink={res.extra['use_sink']}")
    dg = res.extra["diag"]
    d = res.delta
    dfull = res.logit_true_full - res.logit_foil_full
    sd = res.sp_delta
    tags_arr = np.array([t[0] for t in tags])
    kind_arr = np.array([t[3] for t in tags])
    skip = set(filter(None, args.skip_outputs.split(",")))
    for v in variants:
        if v.key in skip:
            log(f"{v.key}: outputs skipped (--skip-outputs)")
            continue
        cases = cases_v[v.key]
        b = base[v.key]
        od = os.path.join(RESULTS, v.run_dir)
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "case_sets.json"), "w") as f:
            json.dump(sets, f)
        sel_v = np.nonzero(tags_arr == v.key)[0]
        # ---- sweep rows
        rows = []
        for i, c in enumerate(all_ids):
            for kind, j in (("clean", b + i), ("noised", b + n + i)):
                rows.append(dict(case_id=c, kind=kind, layer=-1, sigma_mult=args.sigma_mult, logit_true=float(res.logit_true[j]),
                                 logit_foil=float(res.logit_foil[j]), delta=float(d[j]), delta_full=float(dfull[j]),
                                 top1=int(res.top1[j]), rescue=np.nan))
        for j in sel_v[kind_arr[sel_v] == "sweep_layer"]:
            _, c, l, _, _ = tags[j]
            i = all_ids.index(c)
            rows.append(dict(case_id=c, kind="layer", layer=l, sigma_mult=args.sigma_mult, logit_true=float(res.sp_logit_true[j]),
                             logit_foil=float(res.sp_logit_foil[j]), delta=float(sd[j]), delta_full=np.nan, top1=-1,
                             rescue=float(sd[j] - d[b + n + i])))
        df = pd.DataFrame(rows)
        df.to_parquet(os.path.join(od, "sweep_rows.parquet"), index=False)
        # ---- routing
        rr = []
        for run, off in (("clean", b), ("noised", b + n)):
            ri = res.route_idx[:, off : off + n]
            rw = res.route_w[:, off : off + n]
            rc = res.route_cnorm[:, off : off + n]
            Ls, Ns, Ks = np.meshgrid(np.arange(L), np.arange(n), np.arange(k), indexing="ij")
            rr.append(pd.DataFrame(dict(case_id=np.array(all_ids)[Ns.ravel()], run=run, layer=Ls.ravel().astype(np.int16),
                                        slot=Ks.ravel().astype(np.int8), expert=ri.ravel().astype(np.int16),
                                        weight=rw.ravel(), cnorm=rc.ravel())))
        routing = pd.concat(rr)
        routing.to_parquet(os.path.join(od, "sweep_routing.parquet"), index=False)
        # ---- case table
        memb = membership_table(sets, all_ids)
        meta = pd.DataFrame([cases[c].to_row() for c in all_ids])
        ct = meta.merge(memb, on="case_id")
        dc = df[df.kind == "clean"].set_index("case_id").delta
        dn = df[df.kind == "noised"].set_index("case_id").delta
        ct["delta_clean"] = ct.case_id.map(dc)
        ct["delta_noised"] = ct.case_id.map(dn)
        ct["drop"] = ct.delta_clean - ct.delta_noised
        ct["strict"] = (ct.delta_clean >= 1.0) & (ct["drop"] >= 0.5)
        ct["relaxed"] = (ct.delta_clean >= 0.5) & (ct["drop"] >= 0.25)
        ct["pos_offset"] = v.pos_offset
        ct.to_parquet(os.path.join(od, "sweep_cases.parquet"), index=False)
        # ---- sweep summary
        R = df[df.kind == "layer"].pivot(index="case_id", columns="layer", values="rescue")
        dsc, val = sets["paper"]["discovery"], sets["paper"]["validation"]
        md_, mv = R.loc[dsc].mean(0), R.loc[val].mean(0)
        lstar = int(md_.idxmax())
        summ = {"n_cases": n, "pass_time_s": res.extra["total_s"], "T": res.extra["T"], "variant": v.as_dict(),
                "paper": {"n_disc": len(dsc), "n_val": len(val), "L_star_discovery": lstar, "disc_mean_at_Lstar": float(md_[lstar]),
                          "val_mean_at_Lstar": float(mv[lstar]), "val_argmax": int(mv.idxmax()), "val_max": float(mv.max()),
                          "val_curve": [round(float(x), 3) for x in mv.values], "disc_curve": [round(float(x), 3) for x in md_.values],
                          "funnel_strict_pass": int(ct.strict.sum()), "funnel_relaxed_pass": int(ct.relaxed.sum())}}
        with open(os.path.join(od, "sweep_summary.json"), "w") as f:
            json.dump(summ, f, indent=1)
        # ---- expert rows (run_expert.py schema; 'expert' = clean-active, 'expert_noised_only' = noised-only, others dropped)
        rt = {}
        for l in exp_layers:
            for i, c in enumerate(all_ids):
                rt[(c, "clean", l)] = dict(zip(res.route_idx[l, b + i].tolist(), res.route_w[l, b + i].tolist()))
                rt[(c, "noised", l)] = dict(zip(res.route_idx[l, b + n + i].tolist(), res.route_w[l, b + n + i].tolist()))
        erows = []
        for j in sel_v[kind_arr[sel_v] != "sweep_layer"]:
            _, c, l, kind, e = tags[j]
            i = all_ids.index(c)
            ce, ne = rt[(c, "clean", l)], rt[(c, "noised", l)]
            if kind == "expert":
                if e in ce:
                    kind_out = "expert"
                elif e in ne:
                    kind_out = "expert_noised_only"
                else:
                    continue
            else:
                kind_out = kind
            erows.append(dict(case_id=c, layer=l, kind=kind_out, expert=e, partner=-1, alpha=float(res.sp_alpha[j]),
                              norm_e=float(res.sp_norm_e[j]), norm_partner=float(res.sp_norm_partner[j]), vnorm=float(res.sp_vnorm[j]),
                              logit_true=float(res.sp_logit_true[j]), logit_foil=float(res.sp_logit_foil[j]), delta=float(sd[j]),
                              delta_clean=float(d[b + i]), delta_noised=float(d[b + n + i]), rescue=float(sd[j] - d[b + n + i]),
                              clean_active=bool(e in ce), noised_active=bool(e in ne),
                              clean_weight=float(ce.get(e, np.nan)), noised_weight=float(ne.get(e, np.nan)),
                              n_clean_active=len(ce), sigma_mult=args.sigma_mult))
        edf = pd.DataFrame(erows)
        edf.to_parquet(os.path.join(od, "expert_rows.parquet"), index=False)
        pd.DataFrame(dict(case_id=all_ids, delta_clean=d[b : b + n], delta_noised=d[b + n : b + 2 * n])).to_parquet(
            os.path.join(od, f"expert_prefill_L{'_'.join(map(str, exp_layers))}.parquet"), index=False)
        # ---- diagnostics: raw to NVMe, summary to results
        rows_v = np.arange(b, b + 2 * n)
        lens = res.lens[rows_v]
        dd = diag_dir(v.run_dir)
        extra_arrays = {"attn_sink": dg["attn_sink"][:, rows_v]} if "attn_sink" in dg else {}
        for l_, (ti_, tv_, rl_) in dg.get("route_all", {}).items():  # all-token routing at the early layers and the expert layer
            extra_arrays[f"route_all_L{l_}_topi"] = ti_[rows_v]
            extra_arrays[f"route_all_L{l_}_logits"] = rl_[rows_v].astype(np.float16)
        np.savez_compressed(os.path.join(dd, "diag.npz"), attn_final=dg["attn_final"][:, rows_v], resid_norms=dg["resid_norms"][:, rows_v],
                            router_logits_final=dg["router_logits_final"][:, rows_v], token_logprobs=dg["token_logprobs"][rows_v],
                            logprob_true=dg["logprob_true"][rows_v], logprob_foil=dg["logprob_foil"][rows_v], top1_all=dg["top1_all"][rows_v],
                            lens=lens, row_kind=np.array(["clean"] * n + ["noised"] * n), case_ids=np.array(all_ids + all_ids),
                            pos_offset=v.pos_offset, prefix_len=len(v.prefix_ids) + (1 if (v.special_tokens and args.model == "mixtral") else 0),
                            **extra_arrays)
        if "resid_final" in dg:
            torch.save(dg["resid_final"][:, rows_v].clone(), os.path.join(dd, "resid_final.pt"))
        clean_rows = np.arange(b, b + n)
        att = dg["attn_final"][:, clean_rows].astype(np.float32)  # [L, n, nH, T]
        rn = dg["resid_norms"][:, clean_rows]  # [L, n, T]
        lp = dg["token_logprobs"][rows_v]
        dsum = {"variant": v.as_dict(), "n_clean_rows": n, "T": int(res.extra["T"]),
                "sink_mass_pos0_per_layer": att[:, :, :, 0].mean(axis=(1, 2)).round(4).tolist(),
                "sink_mass_pos1_per_layer": att[:, :, :, 1].mean(axis=(1, 2)).round(4).tolist(),
                "resid_norm_pos0_per_layer": rn[:, :, 0].mean(axis=1).round(2).tolist(),
                "resid_norm_pos1_per_layer": rn[:, :, 1].mean(axis=1).round(2).tolist(),
                "resid_norm_final_per_layer": np.array([rn[:, i, lens[i] - 1] for i in range(n)]).mean(axis=0).round(2).tolist(),
                "mean_token_logprob_clean": float(np.nanmean(lp[:n])), "mean_token_logprob_noised": float(np.nanmean(lp[n:])),
                "mean_logprob_true_clean": float(dg["logprob_true"][clean_rows].mean())}
        with open(os.path.join(od, "diag_summary.json"), "w") as f:
            json.dump(dsum, f, indent=1)
        write_run_meta(v.run_dir, {
            "model": args.model, "repo": m["repo"], "experiment": "ext3 BOS mechanism (RESEARCH_PLAN Direction 3)", "agent": "ext3-bos-mechanism",
            "variant": v.as_dict(), "special_tokens": v.special_tokens, "prefix_ids": list(v.prefix_ids), "pos_offset": v.pos_offset,
            "token_rule": "space", "sigma_mult": args.sigma_mult, "case_sets": ["paper"], "expert_layers": exp_layers,
            "pass_variants": [x.key for x in variants], "pass_time_s": res.extra["total_s"],
            "command": "python scripts/ext3_run_variants.py " + " ".join(sys.argv[1:]),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "outputs": ["sweep_rows/sweep_routing/sweep_cases.parquet, sweep_summary.json (layer sweep, paper set)",
                        f"expert_rows.parquet (kinds: layer, coalition_clean, coalition_union, expert, expert_noised_only; all experts patched at L{exp_layers}; no expert_scaled)",
                        "diag_summary.json (per-layer means); raw diagnostics at " + dd + " (diag.npz, resid_final.pt)"]})
        log(f"{v.key}: L*={lstar} disc={md_[lstar]:+.3f} val@L*={mv[lstar]:+.3f} val argmax L{int(mv.idxmax())} {mv.max():+.3f} "
            f"strict {int(ct.strict.sum())}/{n}; {len(edf)} expert rows; sink mass pos0 @L1 {dsum['sink_mass_pos0_per_layer'][1]:.3f}")
        for l in exp_layers:
            e_rows = edf[(edf.layer == l) & (edf.kind == "expert")]
            act = e_rows[e_rows.case_id.isin(dsc)].groupby("expert").size()
            actv = e_rows[e_rows.case_id.isin(val)].groupby("expert").size()
            log(f"   L{l} clean-active disc/val: " + ", ".join(f"E{e:03d} {int(act.get(e, 0))}/{int(actv.get(e, 0))}" for e in sorted(set(act.index) | set(actv.index))
                                                            if max(act.get(e, 0), actv.get(e, 0)) >= 8))
    log("done")


if __name__ == "__main__":
    main()
