"""GPU tests of the layer-streamed RelP engine (moetrace/relp_moe_stream.py) on tiny random HF checkpoints of the three
families (Qwen3-MoE, Mixtral, OLMoE), saved by transformers 5.22.2 in their original per-expert format.

1. The streamed functional forward equals the full HF model (eager attention, eager experts) bit for bit in bf16 for the
   same batch, and so does the streamed native harness used by `verify`.
2. RelP paired / unpaired scores and supports equal an explicit all-expert, whole-graph autograd reference written with
   FModel's math (q = w * act for all E experts, o = sum_e W_down,e q_e, relp detach rules) - the guide's 9.2 check that
   a sparse implementation equals the explicit all-expert one node by node.
3. q interventions (zero / mean-routed / source, partial and complete selections) equal the dense reference
   where(S, V, q) -> o = sum_e W_down,e q_e.
4. A complete-position q patch from a source run equals the output patch bit for bit.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moetrace.relp_moe_stream import Group, Override, StreamRelP, forward_pass, relp_backward  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (layer streamer uses pinned memory)")
DEV = "cuda"


def tiny(family, tmp_path, seed=7):
    from transformers import (MixtralConfig, MixtralForCausalLM, OlmoeConfig, OlmoeForCausalLM, Qwen3MoeConfig,
                              Qwen3MoeForCausalLM)
    torch.manual_seed(seed)
    common = dict(vocab_size=48, hidden_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                  num_experts_per_tok=2, max_position_embeddings=64, rms_norm_eps=1e-6)
    if family == "qwen3_moe":
        c = Qwen3MoeConfig(num_experts=6, moe_intermediate_size=12, norm_topk_prob=True, head_dim=16, **common)
        m = Qwen3MoeForCausalLM(c)
    elif family == "mixtral":
        c = MixtralConfig(num_local_experts=6, intermediate_size=12, **common)
        m = MixtralForCausalLM(c)
    else:
        c = OlmoeConfig(num_experts=6, intermediate_size=12, norm_topk_prob=False, **common)
        m = OlmoeForCausalLM(c)
    with torch.no_grad():  # larger random weights so that routing and attention are not degenerate
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.3)
    d = tmp_path / family
    m.save_pretrained(d)
    return d


def hf_model(d, dtype):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(d, dtype=dtype, attn_implementation="eager",
                                             experts_implementation="eager").to(DEV).eval()
    return m


def hf_run(model, ids, target, foil):
    outs, routes = {}, {}
    hs = []
    for l, lay in enumerate(model.model.layers):
        hs.append(lay.mlp.register_forward_hook(lambda m_, a, o, l=l: outs.__setitem__(l, o.detach())))
        hs.append(lay.mlp.experts.register_forward_pre_hook(
            lambda m_, a, l=l: routes.__setitem__(l, (a[1].detach(), a[2].detach()))))
    try:
        with torch.no_grad():
            normed = model.model(ids, use_cache=False).last_hidden_state
    finally:
        for h in hs:
            h.remove()
    W = model.lm_head.weight
    u = W[torch.tensor(target, device=DEV)].float() - W[torch.tensor(foil, device=DEV)].float()
    return (normed[:, -1].float() * u).sum(-1), outs, routes


FAMS = ("qwen3_moe", "mixtral", "olmoe")


def make_ids(B, T, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, 48, (B, T), generator=g)


@pytest.mark.parametrize("family", FAMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_streamed_forward_equals_hf_model(family, dtype, tmp_path):
    d = tiny(family, tmp_path)
    ref = hf_model(d, dtype)
    m = StreamRelP(str(d), DEV, dtype)
    ids = make_ids(5, 6)
    tg, fo = [3] * 5, [9] * 5
    delta_hf, outs_hf, routes_hf = hf_run(ref, ids.to(DEV), tg, fo)
    gf = Group("f", ids, tg, fo, record_out=True, record_route=True)
    gn = Group("n", ids, tg, fo, native=True, record_out=True, record_route=True)
    forward_pass(m, [gf, gn])
    for g in (gf, gn):
        assert torch.equal(g.delta, delta_hf), (g.name, (g.delta - delta_hf).abs().max())
        for l in range(m.L):
            assert torch.equal(g.outs[l], outs_hf[l]), (g.name, l)
            assert torch.equal(g.routes[l][0].view(-1, m.k), routes_hf[l][0].cpu())
            assert torch.equal(g.routes[l][1].view(-1, m.k), routes_hf[l][1].cpu())


# ---- explicit all-expert reference (FModel math, whole graph) ---------------------------------------------------------
class Ref:
    def __init__(self, model, family):
        self.m, self.f = model, family
        c = model.config
        self.L, self.H, self.nH, self.nKV = c.num_hidden_layers, c.hidden_size, c.num_attention_heads, c.num_key_value_heads
        self.D = getattr(c, "head_dim", None) or self.H // self.nH
        self.E = getattr(c, "num_experts", None) or c.num_local_experts
        self.k = c.num_experts_per_tok
        self.eps = c.rms_norm_eps
        self.norm_topk = True if family == "mixtral" else bool(c.norm_topk_prob)

    def rms(self, x, w, relp):
        r = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        if relp:
            r = r.detach()
        return w * (x.float() * r).to(x.dtype)

    def act(self, g, u, relp):
        if relp:
            sg = torch.sigmoid(g).detach()
            s = g * sg
            s = s + (F.silu(g) - s).detach()
            return 0.5 * (s * u.detach() + s.detach() * u)
        return F.silu(g) * u

    def forward(self, ids, target, foil, relp=False, grad=False, interv=None):
        """interv: {layer: (S [B, T, E, I] bool, V [B, T, E, I], routed_only bool)}"""
        mm = self.m.model
        B, T = ids.shape
        h = mm.embed_tokens.weight[ids]
        if grad:
            h = h.detach().requires_grad_(True)
        cos, sin = mm.rotary_emb(h, torch.arange(T, device=DEV)[None])
        cos, sin = cos[:, None], sin[:, None]
        causal = torch.full((T, T), float("-inf"), device=DEV).triu(1)
        qs = []
        for l, lay in enumerate(mm.layers):
            at = lay.self_attn
            x = self.rms(h, lay.input_layernorm.weight, relp)
            q, k, v = F.linear(x, at.q_proj.weight), F.linear(x, at.k_proj.weight), F.linear(x, at.v_proj.weight)
            if self.f == "olmoe":
                q, k = self.rms(q, at.q_norm.weight, relp), self.rms(k, at.k_norm.weight, relp)
            q = q.view(B, T, self.nH, self.D)
            k = k.view(B, T, self.nKV, self.D)
            if self.f == "qwen3_moe":
                q, k = self.rms(q, at.q_norm.weight, relp), self.rms(k, at.k_norm.weight, relp)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            v = v.view(B, T, self.nKV, self.D).transpose(1, 2)
            rot = lambda t: torch.cat((-t[..., self.D // 2:], t[..., : self.D // 2]), -1)
            q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
            rep = self.nH // self.nKV
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
            p = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * self.D ** -0.5 + causal, -1, dtype=torch.float32).to(q.dtype)
            if relp:
                p = p.detach()
            a = F.linear(torch.matmul(p, v).transpose(1, 2).reshape(B, T, -1), at.o_proj.weight)
            h = h + a
            x2 = self.rms(h, lay.post_attention_layernorm.weight, relp).reshape(B * T, self.H)
            pr = torch.softmax(F.linear(x2, lay.mlp.gate.weight).float(), -1)
            tv, ti = torch.topk(pr, self.k, -1)
            if self.norm_topk:
                tv = tv / tv.sum(-1, keepdim=True)
            w = torch.zeros(B * T, self.E, dtype=torch.float32, device=DEV).scatter(1, ti, tv).detach()
            ex = lay.mlp.experts
            gu = torch.einsum("nh,eih->nei", x2, ex.gate_up_proj)
            g, u = gu.chunk(2, -1)
            qn = (w[..., None] * self.act(g, u, relp)).to(h.dtype).view(B, T, self.E, -1)
            if interv and l in interv:
                S, V, routed = interv[l]
                if routed:
                    S = S & (w.view(B, T, self.E, 1) > 0)
                qn = torch.where(S, V.to(qn.dtype), qn)
            qs.append(qn)
            h = h + torch.einsum("btei,ehi->bth", qn, ex.down_proj)
        hN = self.rms(h, mm.norm.weight, relp)[:, -1]
        W = self.m.lm_head.weight
        u = W[torch.tensor(target, device=DEV)].float() - W[torch.tensor(foil, device=DEV)].float()
        return (hN.float() * u).sum(-1), qs


@pytest.mark.parametrize("family", FAMS)
@pytest.mark.parametrize("mode", ["paired", "unpaired"])
def test_relp_scores_equal_all_expert_reference(family, mode, tmp_path):
    d = tiny(family, tmp_path)
    ref_m = hf_model(d, torch.float32)
    R = Ref(ref_m, family)
    m = StreamRelP(str(d), DEV, torch.float32)
    n, T = 4, 6
    A, Bb = make_ids(n, T, 11), make_ids(n, T, 11).clone()
    Bb[:, 2] = make_ids(n, 1, 12)[:, 0]          # one changed position, as in SVA pairs
    tg, fo = [5, 6, 7, 8], [9, 10, 11, 12]
    pool = torch.ones(n, T, dtype=torch.bool)
    pool[:, 0] = False                            # a special-token-like position is excluded from scoring
    ids = torch.cat([A, Bb]) if mode == "paired" else A
    g = Group("s", ids, tg * (2 if mode == "paired" else 1), fo * (2 if mode == "paired" else 1), record_h_in=True)
    forward_pass(m, [g])
    total, support, _ = relp_backward(m, g, n, pool, mode)
    exp_total = torch.zeros_like(total)
    exp_sup = torch.zeros_like(support)
    for i in range(n):
        a_ids = A[i:i + 1].to(DEV)
        with torch.no_grad():
            _, qb = R.forward(Bb[i:i + 1].to(DEV), tg[i:i + 1], fo[i:i + 1])
        with torch.enable_grad():
            da, qa = R.forward(a_ids, tg[i:i + 1], fo[i:i + 1], relp=True, grad=True)
            beta = torch.autograd.grad(da.sum(), qa)
        for l in range(R.L):
            ref = qb[l][0] if mode == "paired" else torch.zeros_like(qa[l][0])
            sc = (qa[l][0].detach().float() - ref.float()) * beta[l][0].float()
            pm = pool[i].to(DEV)[:, None, None]
            exp_total[l] += torch.where(pm, sc, 0).double()
            exp_sup[l] |= ((qa[l][0] != 0) | (ref != 0)) & pm
    assert torch.equal(support, exp_sup)
    torch.testing.assert_close(total, exp_total, rtol=2e-4, atol=1e-5)
    # expert totals (sum over neurons) also agree: the ranking unit of the protocol
    torch.testing.assert_close(total.sum(-1), exp_total.sum(-1), rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("family", FAMS)
def test_overrides_equal_dense_reference(family, tmp_path):
    d = tiny(family, tmp_path)
    ref_m = hf_model(d, torch.float32)
    R = Ref(ref_m, family)
    m = StreamRelP(str(d), DEV, torch.float32)
    T = 6
    A = make_ids(1, T, 21)
    Bb = A.clone()
    Bb[0, 3] = 40
    tg, fo = [5], [9]
    gen = torch.Generator().manual_seed(5)
    S = {l: torch.rand(T, m.E, m.I, generator=gen) < 0.4 for l in range(m.L)}       # partial selections
    for l in range(m.L):
        S[l][-1, : m.E // 2] = True                                                    # some complete pairs
    mean_q = {l: torch.randn(T, m.E, m.I, generator=gen) * 0.1 for l in range(m.L)}
    with torch.no_grad():
        _, qa_ref = R.forward(A.to(DEV), tg, fo)
        _, qb_ref = R.forward(Bb.to(DEV), tg, fo)
    cases = {"zero": (A, None), "mean": (A, None), "noise": (A, "B"), "denoise": (Bb, "A")}
    for kind, (inp, src) in cases.items():
        interv = {}
        for l in range(m.L):
            Sl = S[l][None].to(DEV)
            if kind == "zero":
                interv[l] = (Sl, torch.zeros_like(qa_ref[l]), False)
            elif kind == "mean":
                interv[l] = (Sl, mean_q[l][None].to(DEV), True)
            else:
                interv[l] = (Sl, (qb_ref if src == "B" else qa_ref)[l], False)
        with torch.no_grad():
            d_ref, _ = R.forward(inp.to(DEV), tg, fo, interv=interv)
        base = Group("base", torch.cat([A, Bb]), tg * 2, fo * 2, pairs=True, keep=set())

        def ofn(l, w, st, kind=kind, src=src):
            Sl = S[l].to(DEV)
            if kind in ("zero", "mean"):
                return Override(rows=torch.tensor([True], device=DEV), S=Sl, value=kind,
                                mean_q=mean_q[l].to(DEV, m.dtype) if kind == "mean" else None)
            return Override(rows=torch.tensor([True], device=DEV), S=Sl, value="source", src=st["layer_out"]["base"],
                            src_rows=torch.tensor([1 if src == "B" else 0], device=DEV))
        gi = Group("iv", inp, tg, fo, override_fn=ofn)
        forward_pass(m, [base, gi])
        torch.testing.assert_close(gi.delta, d_ref, rtol=1e-4, atol=1e-4), kind


@pytest.mark.parametrize("family", FAMS)
def test_complete_position_q_patch_is_bitwise_output_patch(family, tmp_path):
    d = tiny(family, tmp_path)
    m = StreamRelP(str(d), DEV, torch.bfloat16)
    T = 6
    A = make_ids(1, T, 31)
    Bb = A.clone()
    Bb[0, 1] = 41
    tg, fo = [5], [9]
    for l_patch in range(m.L):
        Sfull = torch.zeros(T, m.E, m.I, dtype=torch.bool, device=DEV)
        Sfull[-1] = True
        bt = torch.zeros(1, T, dtype=torch.bool, device=DEV)
        bt[0, -1] = True
        ga = Group("A", A, tg, fo, pairs=True, keep=set(), record_out=True)
        gq = Group("q", Bb, tg, fo, override_fn=lambda l, w, st: Override(
            rows=torch.tensor([True], device=DEV), S=Sfull, value="source", src=st["layer_out"]["A"],
            src_rows=torch.tensor([0], device=DEV)) if l == l_patch else None)
        go = Group("o", Bb, tg, fo, out_patch_fn=lambda l, st: (bt, st["groups"]["A"].outs[l]) if l == l_patch else None)
        gn = Group("n", Bb, tg, fo, native=True, out_patch_fn=lambda l, st: (bt, st["groups"]["A"].outs[l]) if l == l_patch else None)
        forward_pass(m, [ga, gq, go, gn])
        assert torch.equal(gq.delta, go.delta), (l_patch, gq.delta, go.delta)
        assert torch.equal(gq.delta, gn.delta), (l_patch, gq.delta, gn.delta)


@pytest.mark.parametrize("family", ["qwen3_moe", "mixtral"])
@pytest.mark.parametrize("baseline", ["zero", "mean_q_routed"])
def test_cli_verify_score_evaluate_and_stale_gate(family, baseline, tmp_path):
    import json
    import subprocess
    from moetrace import relp_sva as S
    d = tiny(family, tmp_path)
    root = tmp_path / "manifest"
    task = root / "simple"
    task.mkdir(parents=True)
    rows = []
    gen = torch.Generator().manual_seed(9)
    for i in range(6):
        a = torch.randint(2, 48, (5,), generator=gen).tolist()
        b = list(a)
        b[1] = (a[1] + 7) % 46 + 2
        rows.append({"row_id": f"tiny:{i}", "group_id": f"g{i % 3}", "input_ids": a, "reference_ids": b,
                     "target_id": 6 + i, "foil_id": 20 + i, "special_tokens_mask": [1, 0, 0, 0, 0]})
    hashes = {}
    for split in ("calibration", "discovery", "validation", "test"):
        path = task / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        hashes[split] = S.sha(path)
    S.write_json(root / "manifest.json", {"model": {"revision": "synthetic"}, "source_revision": "synthetic",
                                          "tokenizer_sha256": {"config.json": S.sha(d / "config.json")},
                                          "tasks": {"simple": {"files_sha256": hashes}}})
    script = ROOT / "scripts/relp_sva_moe_run.py"
    common = ["--model-dir", str(d), "--manifest-dir", str(root), "--task", "simple", "--dtype", "bfloat16"]

    def run(command, extra, ok=True):
        r = subprocess.run([sys.executable, str(script), command] + common + extra, capture_output=True, text=True)
        assert (r.returncode == 0) == ok, r.stdout + r.stderr
        return r
    ver = tmp_path / "verify"
    run("verify", ["--output-dir", str(ver), "--verify-cases", "2"])
    rep = json.loads((ver / "verification.json").read_text())
    assert rep["passed"] and len(rep["checks"]) > 50
    score = tmp_path / "score"
    run("score", ["--output-dir", str(score), "--verification", str(ver / "verification.json"), "--baseline", baseline,
                  "--neuron-budgets", "1", "3", "--expert-budgets", "1", "2", "--random-seeds", "101", "--score-chunk", "4"])
    agg = torch.load(score / "aggregate.pt", weights_only=True)
    assert agg["mean"].shape[0] == 3 and not agg["support"][:, 0].any()   # special position 0 is never scored
    for unit in ("neuron", "expert_position", "expert_all_positions"):
        dest = tmp_path / ("evaluate_" + unit)
        run("evaluate", ["--output-dir", str(dest), "--verification", str(ver / "verification.json"),
                         "--score-dir", str(score), "--unit", unit, "--eval-chunk", "7"])
        summ = json.loads((dest / "summary.json").read_text())
        assert json.loads((dest / "run_meta.json").read_text())["status"] == "complete" and summ
        recs = [json.loads(x) for x in (dest / "interventions.jsonl").read_text().splitlines()]
        assert len(recs) == len(rows) * 2 * 2 * 4   # rows x budgets x (relp + 1 random seed) x interventions
    # one streamed pass for all three units = the three single evaluations (same records; bf16 chunking noise only)
    multi = tmp_path / "multi"
    run("evaluate-multi", ["--output-dir", str(multi), "--verification", str(ver / "verification.json"),
                           "--score-dirs", str(score), "--units", "neuron", "expert_position", "expert_all_positions",
                           "--eval-chunk", "13"])
    for unit in ("neuron", "expert_position", "expert_all_positions"):
        single = [json.loads(x) for x in (tmp_path / ("evaluate_" + unit) / "interventions.jsonl").read_text().splitlines()]
        both = [json.loads(x) for x in (multi / f"validation_{unit}_{baseline}" / "interventions.jsonl").read_text().splitlines()]
        assert [(r["row_id"], r["ranking"], r["seed"], r["budget"], r["intervention"]) for r in single] == \
               [(r["row_id"], r["ranking"], r["seed"], r["budget"], r["intervention"]) for r in both]
        assert max(abs(x["margin"] - y["margin"]) for x, y in zip(single, both)) < 0.05 * max(1.0, max(abs(x["margin"]) for x in single))
    if baseline == "zero":  # the mean-baseline score run may reuse the (baseline-independent) discovery scores
        smean = tmp_path / "score_mean"
        run("score", ["--output-dir", str(smean), "--verification", str(ver / "verification.json"), "--baseline",
                      "mean_q_routed", "--reuse-scores", str(score), "--neuron-budgets", "1", "3", "--expert-budgets", "1",
                      "2", "--random-seeds", "101"])
        a0 = torch.load(score / "aggregate.pt", weights_only=True)
        a1 = torch.load(smean / "aggregate.pt", weights_only=True)
        assert torch.equal(a0["mean"], a1["mean"]) and torch.equal(a0["support"], a1["support"]) and "qmean" in a1
        assert (score / "neuron_ranking.json").read_text() == (smean / "neuron_ranking.json").read_text()
    data = json.loads((ver / "verification.json").read_text())
    data["identity"]["dtype"] = "stale-dtype"
    S.write_json(ver / "bad.json", data)
    run("score", ["--output-dir", str(tmp_path / "rejected"), "--verification", str(ver / "bad.json")], ok=False)
    assert not (tmp_path / "rejected").exists()
