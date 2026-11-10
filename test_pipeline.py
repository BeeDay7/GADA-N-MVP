"""Pipeline-invariant tests (no network / GPU needed).

Run with either (from the gadanplus/ repo root):
    pytest -q
    python tests/test_pipeline.py               # built-in runner, no pytest needed
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gadanplus import metrics, stats, integrity
from gadanplus.env import CPSEnv, check_observation_leakage, renderer_version_hash
from gadanplus.attacker import _APIBackend, MalformedOutputError, run_episode, make_backend
from gadanplus.run import run_condition


# ---- renderer / leakage ----------------------------------------------------
def test_renderer_no_structural_leak():
    for defence in ("none", "naive", "gadanplus"):
        env = CPSEnv(defence=defence, awareness="A2", seed=1)
        obs = env.observe()
        issues = check_observation_leakage(obs, env._cur_decoys, strict=False)
        assert issues == [], f"{defence}: {issues}"


def test_leakage_scan_clean_gadanplus(tmp_path=None):
    out = str(tmp_path) if tmp_path else "results/_test"
    res = run_condition(dict(condition="T", defence="gadanplus", attacker="mock",
                             awareness="A2", seeds=3, episodes=20, max_steps=40), out)
    assert res["leak"]["any_leak"] is False, res["leak"]["leaks"]
    assert res["leak"]["n_decoy_views"] > 0


def test_leakage_scan_catches_injected_oracle():
    # a field 'apparent_vuln' that perfectly separates decoys should be flagged
    trace = []
    for i in range(20):
        trace.append(dict(action={"op": "exploit", "node": 5000 + i},
                          decoys_shown=[5000 + i] if i % 2 else [],
                          neighbors=[dict(node=5000 + i, apparent_vuln=(0.99 if i % 2 else 0.1),
                                          kind="UAV", service_banner="x", firmware="y",
                                          cert_expiry_days=10, latency_ms=10, consistency=0.9,
                                          corroborations=2)]))
    scan = integrity.leakage_scan([dict(trace=trace)])
    assert scan["any_leak"] and "apparent_vuln" in scan["leaks"]


# ---- RCDI labeller ---------------------------------------------------------
def test_rcdi_labeller_cases():
    correct = dict(action={"op": "exploit", "node": 7}, real_next=7, decoys_shown=[])
    dead = dict(action={"op": "exploit", "node": 99}, real_next=7, decoys_shown=[99])
    wrong = dict(action={"op": "exploit", "node": 3}, real_next=7, decoys_shown=[])
    undef = dict(action={"op": "exploit", "node": 3}, real_next=None, decoys_shown=[])
    scan = dict(action={"op": "scan", "node": None}, real_next=7, decoys_shown=[])
    assert metrics.label_step(correct) == metrics.CORRECT
    assert metrics.label_step(dead) == metrics.DEAD_END
    assert metrics.label_step(wrong) == metrics.INCORRECT
    assert metrics.label_step(undef) == metrics.UNDEFINED
    assert metrics.label_step(scan) is None


def test_rcdi_multipath_and_undefined_excluded():
    trace = [
        dict(action={"op": "exploit", "node": 5}, real_next=[5, 9], decoys_shown=[]),  # correct (set)
        dict(action={"op": "exploit", "node": 8}, real_next=7, decoys_shown=[]),        # incorrect
        dict(action={"op": "exploit", "node": 2}, real_next=None, decoys_shown=[]),     # undefined -> excluded
    ]
    # denominator excludes the undefined step: 1 disrupted / 2 scored = 0.5
    assert abs(metrics.rcdi_episode(trace) - 0.5) < 1e-9


def test_cohens_kappa():
    assert metrics.cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert abs(metrics.cohens_kappa(["a", "a", "b", "b"], ["a", "b", "b", "a"])) < 1e-9


# ---- run integrity ---------------------------------------------------------
class _BadBackend(_APIBackend):
    name = "mock"          # reuse a valid cache tag; never hits network

    def _call(self, messages):
        return "this is not json"


def test_malformed_output_excludes_episode():
    env = CPSEnv(defence="gadanplus", awareness="A0", seed=0)
    bad = _BadBackend(awareness="A0", cache_path=None)
    rec = run_episode(env, bad, max_steps=10)
    assert rec["excluded"] is True and rec["exclude_reason"]


def test_provenance_manifest_fields():
    m = integrity.run_manifest(dict(defence="gadanplus", seeds=1))
    assert m["renderer_version"] == renderer_version_hash()
    assert m["config_hash"] and m["lib_versions"]["python"]


# ---- stats -----------------------------------------------------------------
def test_stats_basic():
    assert abs(stats.auroc([3, 4, 5], [1, 2]) - 1.0) < 1e-9
    u, p = stats.mann_whitney_u([5, 6, 7, 8], [1, 2, 3, 4])
    assert p < 0.05
    assert stats.cliffs_delta([1, 2, 3], [4, 5, 6]) > 0.99   # b >> a


# ---- GDE (torch-guarded; skips cleanly if torch is absent) -----------------
def test_gde_smoke_if_torch():
    try:
        from gadanplus import gde
        if not gde._TORCH:
            print("    (skip test_gde_smoke_if_torch: torch not installed)")
            return
    except Exception:
        print("    (skip test_gde_smoke_if_torch: gde import failed)")
        return
    import torch
    spec = gde.FeatureSpec(numeric=[("a", 0, 1), ("b", 0, 1)],
                           categorical=[("k", 3)], n_kinds=3)
    N = 128
    num = 0.15 + 0.7 * torch.rand(N, spec.num_dim)
    oh = torch.zeros(N, 3); oh[torch.arange(N), torch.randint(0, 3, (N,))] = 1.0
    real = torch.cat([num, oh], dim=1)
    kind = torch.zeros(N, 3); kind[torch.arange(N), torch.randint(0, 3, (N,))] = 1.0
    cond = torch.cat([kind, torch.rand(N, 1)], dim=1)
    out = gde.train_cgan(real, cond, spec, epochs=10, batch=64, seed=0)
    g = out["G"].sample(cond[:4])
    assert g.shape == (4, spec.feat_dim)
    assert all(v == v for h in out["history"] for v in h)   # no NaN


# ---- GDE L_ling helpers (offline, no torch/endpoint needed) ----------------
def test_perplexity_and_render():
    import math
    import json as _json
    from gadanplus import gde
    assert abs(gde.perplexity_from_logprobs([-1.0, -1.0]) - math.e) < 1e-6
    assert gde.perplexity_from_logprobs([]) == float("inf")
    spec = gde.FeatureSpec(numeric=[("latency", 0, 100)], categorical=[("kind", 3)], n_kinds=3)
    meta = {"categorical": [("kind", ["A", "B", "C"])], "numeric": [("latency", 0, 100)]}
    d = _json.loads(gde.render_features([0.5, 0.1, 0.8, 0.1], spec, meta))
    assert d["latency"] == 50.0 and d["kind"] == "B"


def test_llama_logprob_parsing():
    from gadanplus import gde
    S = gde.LlamaPerplexityScorer
    together = {"prompt": [{"logprobs": {"token_logprobs": [None, -1.0, -2.0]}}]}
    openai_legacy = {"choices": [{"logprobs": {"token_logprobs": [-0.5, -1.5]}}]}
    assert S._extract_prompt_logprobs(together) == [None, -1.0, -2.0]
    assert S._extract_prompt_logprobs(openai_legacy) == [-0.5, -1.5]
    try:
        S._extract_prompt_logprobs({"nonsense": 1})
        assert False, "should have raised on missing logprobs"
    except RuntimeError:
        pass


def test_data_loader_shapes():
    try:
        import pandas as pd
    except Exception:
        print("    (skip test_data_loader_shapes: pandas missing)")
        return
    from gadanplus import data
    import numpy as _np
    rng = _np.random.default_rng(0); n = 120
    df = pd.DataFrame(dict(x=rng.uniform(0, 1, n), y=rng.uniform(0, 5, n),
                           kind=rng.choice(["A", "B"], n),
                           idcol=[f"i{i}" for i in range(n)],
                           label=rng.integers(0, 2, n)))
    spec, meta = data.fit_feature_spec(df, label_col="label", kind_col="kind")
    feats = data.encode(df, meta); cond = data.make_conditioning(df, meta)
    assert feats.shape == (n, spec.feat_dim) and cond.shape == (n, spec.cond_dim)
    assert "idcol" not in [c for c, _ in meta["categorical"]]     # id-like dropped


def test_localhf_json_extraction():
    """The local-LLM attacker's reply parser survives prose / code-fenced JSON."""
    import json as _json
    from gadanplus.attacker import _extract_json, LocalHFBackend  # noqa: F401 (import ok)
    fenced = "Sure:\n```json\n{\"thought\":\"go\",\"action\":{\"op\":\"exploit\",\"node\":42}}\n```"
    prose = "I pick {\"thought\":\"x\",\"action\":{\"op\":\"scan\",\"node\":1}} now."
    for s in (fenced, prose, '{"thought":"a","action":{"op":"probe","node":7}}'):
        assert "action" in _json.loads(_extract_json(s))


def test_gadan_env_contrast():
    """The real GADA-N+ env (Route 2) gives the intended graded contrast:
    undefended attacker mostly wins; GADA-N+ suppresses it and stays leakage-clean."""
    from gadanplus.gadan_env import make_env, REACHES_HVT
    from gadanplus.attacker import make_backend, run_episode
    from gadanplus import metrics, integrity
    assert 3 not in REACHES_HVT and 6 not in REACHES_HVT      # dead-ends excluded

    def run(defence, seeds=2, eps=20):
        recs = []
        for s in range(seeds):
            b = make_backend("mock", awareness="A0", seed=s)
            for ep in range(eps):
                recs.append(run_episode(make_env(defence=defence, seed=s * 999 + ep,
                                                 max_steps=60), b, max_steps=60))
        return recs, metrics.summarize(recs), integrity.leakage_scan(recs)

    _, none_s, _ = run("none")
    _, gp_s, gp_leak = run("gadanplus")
    assert none_s["HCR"] > 0.5, f"undefended HCR too low: {none_s['HCR']}"
    assert gp_s["HCR"] < 0.3, f"GADA-N+ HCR too high: {gp_s['HCR']}"
    assert gp_s["MADT"] > none_s["MADT"], "GADA-N+ should extend dwell time"
    assert not gp_leak["any_leak"], f"GADA-N+ leaked: {gp_leak['leaks']}"


def test_adapter_harness_env():
    """A real-env adapter (Route 1 seam) drives a full episode through the harness
    and stays leakage-clean — the contract any real GADA-N env must satisfy."""
    from gadanplus.adapter import HarnessEnv, MockDynamics
    from gadanplus.attacker import make_backend, run_episode
    from gadanplus import metrics, integrity
    env = HarnessEnv(MockDynamics(), defence="gadanplus", awareness="A0",
                     max_steps=30, seed=1)
    rec = run_episode(env, make_backend("mock", awareness="A0", seed=1), max_steps=30)
    assert rec.get("trace"), "adapter produced no trace"
    assert "result" in rec and "compromised" in rec
    assert metrics.summarize([rec])["n"] == 1
    leak = integrity.leakage_scan([rec])
    assert not leak["any_leak"], f"adapter observation leaked: {leak['leaks']}"


# ---- built-in runner (no pytest needed) ------------------------------------
def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
