"""Generative Deception Engine (GDE) — conditional GAN for coherent decoy
telemetry, with the L_ling surrogate-coherence signal (paper §4.1).

    min_theta max_phi  E[log D(x,c)] + E[log(1 - D(G(z,c),c))]  +  lambda_ell * L_ling

REFERENCE IMPLEMENTATION. The architecture and training loop are real and
runnable (torch, CPU-testable). What must come from YOUR infra before any number
is reported:
  * a real telemetry corpus fit into a FeatureSpec (see DATA_REQUIREMENTS.md),
  * the real L_ling signal = Llama-3.1-70B perplexity over rendered telemetry
    (the LlamaPerplexityScorer below; GPT-4o stays eval-only),
  * the lambda_ell in {0.01,0.1,1.0} grid on the held-out split.

The bundled `HeuristicScorer` is a DIFFERENTIABLE CPU stand-in for the surrogate
coherence head so the training loop runs end-to-end in tests. Its output is NOT a
perplexity and NOT paper data — it only exercises the mechanics. The real
`LlamaPerplexityScorer` is a black-box reward (non-differentiable); using it
requires the policy-gradient variant (integration hook noted below).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                       # torch optional for import-time
    _TORCH = False


# ---- feature schema (fit this from the real dataset) -----------------------
@dataclass
class FeatureSpec:
    """Describes the decoy telemetry feature vector the generator produces.
    `numeric` = [(name, lo, hi)]; `categorical` = [(name, cardinality)];
    `n_kinds` = number of node kinds used for conditioning."""
    numeric: List[Tuple[str, float, float]]
    categorical: List[Tuple[str, int]]
    n_kinds: int = 4

    @property
    def num_dim(self) -> int:
        return len(self.numeric)

    @property
    def cat_dim(self) -> int:
        return sum(c for _, c in self.categorical)

    @property
    def feat_dim(self) -> int:
        return self.num_dim + self.cat_dim

    @property
    def cond_dim(self) -> int:
        return self.n_kinds + 1          # kind one-hot + normalised kill-chain stage

    def cat_slices(self):
        off = self.num_dim
        for _, c in self.categorical:
            yield off, off + c
            off += c


# ---- surrogate coherence scorer (pluggable) --------------------------------
class SurrogateScorer(Protocol):
    def coherence(self, feats, spec: "FeatureSpec"):
        """Return a per-sample coherence signal (higher = more coherent)."""
        ...


def perplexity_from_logprobs(token_logprobs: List[float]) -> float:
    """PPL = exp(-mean tokenwise logprob). Pure function (offline-testable)."""
    lps = [lp for lp in token_logprobs if lp is not None]
    if not lps:
        return float("inf")
    return math.exp(-sum(lps) / len(lps))


def render_features(feat_row, spec: "FeatureSpec", meta: Optional[dict] = None) -> str:
    """Decode one generator feature vector back to a telemetry JSON string, the
    text over which the surrogate LM's perplexity is measured. Numeric features
    are de-normalised via the FeatureSpec; categoricals argmax'd to their vocab
    (from `meta` when available, else the class index)."""
    row = list(feat_row)
    out, off = {}, 0
    for name, lo, hi in spec.numeric:
        out[name] = round(lo + float(row[off]) * (hi - lo), 3)
        off += 1
    cat_meta = {n: v for n, v in (meta.get("categorical", []) if meta else [])}
    for name, card in spec.categorical:
        seg = row[off:off + card]
        j = max(range(card), key=lambda i: seg[i]) if card else 0
        vocab = cat_meta.get(name)
        out[name] = vocab[j] if vocab and j < len(vocab) else j
        off += card
    return json.dumps(out, sort_keys=True)


class LlamaPerplexityScorer:
    """REAL L_ling: render feats -> telemetry text -> Llama perplexity via an
    OpenAI-compatible completions endpoint (echo + logprobs). Returns -PPL as the
    coherence reward (higher = more coherent).

    Black-box (no gradient through G): integrate via a policy-gradient term
    (REINFORCE with -PPL as reward) rather than the differentiable G-step used by
    the CPU HeuristicScorer. Network call is guarded; the perplexity maths
    (`perplexity_from_logprobs`) and rendering (`render_features`) are unit-tested
    offline.
    """
    def __init__(self, endpoint: Optional[str] = None,
                 model: str = "meta-llama/Llama-3.1-70B-Instruct",
                 api_key_env: str = "LLAMA_API_KEY", meta: Optional[dict] = None,
                 timeout: int = 60):
        self.endpoint = endpoint or os.environ.get("LLAMA_ENDPOINT")
        self.model = os.environ.get("LLAMA_MODEL", model)
        self.api_key = os.environ.get(api_key_env)
        self.meta = meta
        self.timeout = timeout

    @staticmethod
    def _extract_prompt_logprobs(data: dict):
        """Pull per-prompt-token logprobs from a completions response.

        Together returns them under `prompt[0].logprobs.token_logprobs`; the
        OpenAI-legacy / vLLM style puts them under `choices[0].logprobs.
        token_logprobs`. Handle both."""
        p = data.get("prompt")
        if isinstance(p, list) and p and isinstance(p[0], dict):
            lg = p[0].get("logprobs") or {}
            if lg.get("token_logprobs"):
                return lg["token_logprobs"]
        ch = data.get("choices")
        if isinstance(ch, list) and ch:
            lg = ch[0].get("logprobs") or {}
            if lg.get("token_logprobs"):
                return lg["token_logprobs"]
        raise RuntimeError(f"no prompt token_logprobs in response (keys={list(data)}); "
                           "check the provider supports echo+logprobs on /completions")

    def _ppl(self, text: str) -> float:                # pragma: no cover - needs endpoint
        import requests
        if not self.endpoint:
            raise RuntimeError("LLAMA_ENDPOINT not set for LlamaPerplexityScorer")
        # OpenAI-compatible /completions with echo+logprobs returns prompt logprobs.
        url = self.endpoint.replace("/chat/completions", "/completions")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = requests.post(url, headers=headers, timeout=self.timeout, json=dict(
            model=self.model, prompt=text, max_tokens=1, echo=True, logprobs=1))
        r.raise_for_status()
        return perplexity_from_logprobs(self._extract_prompt_logprobs(r.json()))

    def coherence(self, feats, spec):                  # pragma: no cover - needs endpoint
        rewards = []
        for row in feats:
            row = row.tolist() if hasattr(row, "tolist") else list(row)
            rewards.append(-self._ppl(render_features(row, spec, self.meta)))
        try:
            import torch
            return torch.tensor(rewards, dtype=torch.float32)
        except Exception:
            return rewards


if _TORCH:
    class HeuristicScorer(nn.Module):
        """DIFFERENTIABLE CPU stand-in for the surrogate coherence head.

        Rewards numeric features that sit in a plausible mid-range and penalises
        implausible tails. NOT a perplexity; NOT paper data — it only lets the
        training loop run in tests."""
        def coherence(self, feats, spec: FeatureSpec):
            num = feats[:, :spec.num_dim]
            # plausibility: distance outside [0.15, 0.85] (features are in [0,1])
            lo = torch.clamp(0.15 - num, min=0.0)
            hi = torch.clamp(num - 0.85, min=0.0)
            penalty = (lo + hi).mean(dim=1)
            return -penalty                    # higher = more coherent

    # ---- conditional GAN ---------------------------------------------------
    class Generator(nn.Module):
        def __init__(self, spec: FeatureSpec, z_dim: int = 32, hidden: int = 128):
            super().__init__()
            self.spec = spec
            self.z_dim = z_dim
            self.net = nn.Sequential(
                nn.Linear(z_dim + spec.cond_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, spec.feat_dim))

        def forward(self, z, c):
            raw = self.net(torch.cat([z, c], dim=1))
            num = torch.sigmoid(raw[:, :self.spec.num_dim])          # numeric -> [0,1]
            parts = [num]
            for lo, hi in self.spec.cat_slices():
                parts.append(torch.softmax(raw[:, lo:hi], dim=1))    # categorical -> probs
            return torch.cat(parts, dim=1)

        def sample(self, c):
            z = torch.randn(c.shape[0], self.z_dim, device=c.device)
            return self.forward(z, c)

    class Discriminator(nn.Module):
        def __init__(self, spec: FeatureSpec, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(spec.feat_dim + spec.cond_dim, hidden), nn.LeakyReLU(0.2),
                nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
                nn.Linear(hidden, 1))

        def forward(self, x, c):
            return self.net(torch.cat([x, c], dim=1))

    def train_cgan(real_feats, cond, spec: FeatureSpec, *, scorer: SurrogateScorer = None,
                   lambda_ell: float = 0.1, epochs: int = 200, batch: int = 128,
                   lr: float = 2e-4, seed: int = 0, log_every: int = 0):
        """Train the cGAN with the extra lambda_ell * L_ling coherence term.

        `real_feats`: (N, feat_dim) tensor of real telemetry (numeric in [0,1] +
        categorical one-hots). `cond`: (N, cond_dim) conditioning. Returns the
        loss history. With the differentiable HeuristicScorer this runs on CPU;
        with LlamaPerplexityScorer, replace the L_ling term with the PG estimator.
        """
        torch.manual_seed(seed)
        real_feats = torch.as_tensor(real_feats, dtype=torch.float32)   # accept numpy (data.py)
        cond = torch.as_tensor(cond, dtype=torch.float32)
        G, D = Generator(spec), Discriminator(spec)
        scorer = scorer or HeuristicScorer()
        optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        n = real_feats.shape[0]
        hist = []
        for ep in range(epochs):
            idx = torch.randint(0, n, (batch,))
            xr, cr = real_feats[idx], cond[idx]
            # --- D step ---
            xf = G.sample(cr).detach()
            lossD = bce(D(xr, cr), torch.ones(batch, 1)) + \
                bce(D(xf, cr), torch.zeros(batch, 1))
            optD.zero_grad(); lossD.backward(); optD.step()
            # --- G step (adversarial + coherence) ---
            xg = G.sample(cr)
            adv = bce(D(xg, cr), torch.ones(batch, 1))       # non-saturating
            ling = -scorer.coherence(xg, spec).mean()        # maximise coherence
            lossG = adv + lambda_ell * ling
            optG.zero_grad(); lossG.backward(); optG.step()
            hist.append((lossD.detach().item(), adv.detach().item(), ling.detach().item()))
            if log_every and ep % log_every == 0:
                print(f"ep{ep:4d}  D={hist[-1][0]:.3f}  adv={hist[-1][1]:.3f}  ling={hist[-1][2]:.3f}")
        return dict(G=G, D=D, history=hist)


def _smoke():
    """CPU smoke test: build synthetic 'real' telemetry, train a few epochs,
    assert the loop runs and produces finite losses. NOT PAPER DATA."""
    assert _TORCH, "torch required for the GDE smoke test"
    print("=" * 8 + " NOT PAPER DATA (GDE reference smoke test) " + "=" * 8)
    spec = FeatureSpec(numeric=[("latency_ms", 0, 1), ("cert_days", 0, 1),
                                ("pktloss", 0, 1)],
                       categorical=[("kind", 4), ("banner", 6)], n_kinds=4)
    N = 512
    # synthetic 'real' data: mid-range numeric + valid one-hots
    num = 0.15 + 0.7 * torch.rand(N, spec.num_dim)
    cats = []
    for _, c in spec.categorical:
        oh = torch.zeros(N, c); oh[torch.arange(N), torch.randint(0, c, (N,))] = 1.0
        cats.append(oh)
    real = torch.cat([num] + cats, dim=1)
    kind = torch.zeros(N, spec.n_kinds)
    kind[torch.arange(N), torch.randint(0, spec.n_kinds, (N,))] = 1.0
    stage = torch.rand(N, 1)
    cond = torch.cat([kind, stage], dim=1)

    out = train_cgan(real, cond, spec, lambda_ell=0.1, epochs=60, batch=128, seed=1)
    lossD0, lossDn = out["history"][0][0], out["history"][-1][0]
    ling_last = out["history"][-1][2]
    assert all(all(map(lambda x: x == x, h)) for h in out["history"]), "NaN in history"
    # generator produces valid-shaped, in-range samples
    g = out["G"].sample(cond[:8])
    assert g.shape == (8, spec.feat_dim)
    assert float(g[:, :spec.num_dim].min()) >= 0.0 and float(g[:, :spec.num_dim].max()) <= 1.0
    print(f"  ran 60 epochs: D {lossD0:.3f}->{lossDn:.3f}, ling(last)={ling_last:.4f}")
    print(f"  generator output shape {tuple(g.shape)}, numeric in [0,1] OK")
    print("=" * 8 + " GDE smoke OK (NOT PAPER DATA) " + "=" * 8)
    return True


if __name__ == "__main__":
    _smoke()
