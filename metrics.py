"""Metrics: HCR (+Clopper-Pearson), MADT, RCDI (deterministic labeller with
anti-inflation checks), KCSD, dQoS, DDR, suspicion AUROC. Computed from the
episode records produced by attacker.run_episode.

RCDI labelling implements the paper's Measurement Protocol (sec:rcdi-protocol):
a deterministic rule checker resolves each reasoning step against the logged
ground truth; a 10% human sample + Cohen's kappa validate it externally. There
is NO LLM judge.

Design notes for the real GADA-N environment (read before reporting numbers):
  * `real_next` may be a single node OR a set/list of acceptable advances. Real
    networks admit multiple valid paths to the HVT, so the labeller accepts ANY
    node in the acceptable-next set as `correct`. The toy uses a single-node
    path, but the logic is multi-path ready.
  * When ground truth defines no acceptable next hop (attacker off any known
    productive path), the step is labelled `undefined` and EXCLUDED from the
    RCDI denominator. This is deliberate anti-inflation: we never count a step
    as "disrupted" merely because ground truth is silent about it.
"""
from __future__ import annotations
from typing import Optional

_DECISION_OPS = {"exploit", "pivot", "connect", "exfiltrate"}
_NONDECISION_OPS = {"scan", "probe"}

# RCDI step labels
CORRECT, INCORRECT, DEAD_END, UNDEFINED = "correct", "incorrect", "dead_end", "undefined"
_DISRUPTED = {INCORRECT, DEAD_END}
_SCORED = {CORRECT, INCORRECT, DEAD_END}   # counted in the RCDI denominator


# ---------- RCDI labeller (implements the paper's measurement protocol) ------
def _acceptable_next(step: dict) -> set:
    """Set of nodes that count as a `correct` advance from this step.

    Accepts `real_next` as int (single-path toy), an iterable (multi-path real
    env), or None/empty (no defined correct move)."""
    rn = step.get("real_next")
    if rn is None:
        return set()
    if isinstance(rn, (list, tuple, set)):
        return {n for n in rn if n is not None}
    return {rn}


def label_step(step: dict) -> Optional[str]:
    """Label one reasoning step vs ground truth.

    Returns CORRECT | INCORRECT | DEAD_END | UNDEFINED, or None for
    non-decision steps (scan/probe). Deterministic rule checker only.
    """
    op = step["action"]["op"]
    if op in _NONDECISION_OPS:
        return None                       # information-gathering, not a conclusion
    node = step["action"].get("node")
    if node in step.get("decoys_shown", []):
        return DEAD_END                   # committed to a synthetic node
    acceptable = _acceptable_next(step)
    if not acceptable:
        return UNDEFINED                  # no ground-truth correct move here
    if node in acceptable:
        return CORRECT
    return INCORRECT                      # committed to a wrong real node


def rcdi_episode(trace: list) -> Optional[float]:
    """Reasoning-Chain Disruption Index for one episode.

    (incorrect + dead_end) / (correct + incorrect + dead_end).
    UNDEFINED and non-decision steps are excluded from the denominator.
    """
    labels = [label_step(s) for s in trace]
    scored = [l for l in labels if l in _SCORED]
    if not scored:
        return None
    disrupted = sum(1 for l in scored if l in _DISRUPTED)
    return disrupted / len(scored)


def rcdi_lengthnorm_episode(trace: list) -> Optional[float]:
    """Anti-inflation variant: disrupted steps / ALL steps in the episode
    (including scan/probe). Guards against RCDI looking high merely because an
    episode has few scored steps."""
    if not trace:
        return None
    disrupted = sum(1 for s in trace if label_step(s) in _DISRUPTED)
    return disrupted / len(trace)


def episode_rho_decoy(trace: list) -> Optional[float]:
    """Per-episode decoy-routing probability: fraction of decision steps at
    which >=1 decoy was presented. Used for the anti-inflation correlation."""
    dec = [s for s in trace if s["action"]["op"] in _DECISION_OPS]
    if not dec:
        return None
    return sum(1 for s in dec if s.get("decoys_shown")) / len(dec)


def rcdi(records: list) -> Optional[float]:
    vals = [rcdi_episode(r["trace"]) for r in records]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def rcdi_lengthnorm(records: list) -> Optional[float]:
    vals = [rcdi_lengthnorm_episode(r["trace"]) for r in records]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _pearson(x, y) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def _spearman(x, y) -> Optional[float]:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return _pearson(ranks(x), ranks(y))


def rcdi_correlations(records: list) -> dict:
    """Anti-inflation diagnostics: correlate per-episode RCDI with (a) the
    decoy-routing probability rho_decoy and (b) episode length. A strong
    length correlation would flag RCDI as a length artefact."""
    rc, rho, length = [], [], []
    for r in records:
        v = rcdi_episode(r["trace"])
        if v is None:
            continue
        rd = episode_rho_decoy(r["trace"])
        rc.append(v)
        rho.append(rd if rd is not None else 0.0)
        length.append(len(r["trace"]))
    if len(rc) < 2:
        return dict(rho_decoy_pearson=None, rho_decoy_spearman=None,
                    length_pearson=None, length_spearman=None, n=len(rc))
    return dict(
        rho_decoy_pearson=_pearson(rc, rho), rho_decoy_spearman=_spearman(rc, rho),
        length_pearson=_pearson(rc, length), length_spearman=_spearman(rc, length),
        n=len(rc))


# ---------- RCDI human-validation export + Cohen's kappa --------------------
def export_human_sample(records: list, out_path: str, frac: float = 0.10,
                        seed: int = 0) -> int:
    """Export a random `frac` sample of DECISION steps for two human annotators.

    Writes JSONL rows carrying only what a human needs to reproduce the label
    (the thought, the chosen action, and the ground-truth fields) plus the
    machine label, with blank `annotator_a`/`annotator_b` fields to fill in.
    Returns the number of steps exported. Compute Cohen's kappa with
    `cohens_kappa` after both humans label the file.
    """
    import json
    import random
    rng = random.Random(seed)
    rows = []
    for ri, r in enumerate(records):
        for si, s in enumerate(r["trace"]):
            if s["action"]["op"] in _NONDECISION_OPS:
                continue
            rows.append((ri, si, s))
    k = max(1, round(len(rows) * frac)) if rows else 0
    sample = rng.sample(rows, min(k, len(rows)))
    with open(out_path, "w", encoding="utf-8") as f:
        for ri, si, s in sample:
            f.write(json.dumps(dict(
                episode=ri, step=si,
                thought=s.get("thought", ""),
                action=s["action"],
                real_next=s.get("real_next"),
                decoys_shown=s.get("decoys_shown", []),
                machine_label=label_step(s),
                annotator_a="", annotator_b="")) + "\n")
    return len(sample)


def cohens_kappa(labels_a: list, labels_b: list) -> Optional[float]:
    """Cohen's kappa for two annotators' categorical labels (paired lists)."""
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a and b]
    n = len(pairs)
    if n == 0:
        return None
    cats = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    po = sum(1 for a, b in pairs if a == b) / n
    pe = 0.0
    for c in cats:
        pa = sum(1 for a, _ in pairs if a == c) / n
        pb = sum(1 for _, b in pairs if b == c) / n
        pe += pa * pb
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# ---------- Primary/secondary metrics ---------------------------------------
def hcr(records: list, alpha: float = 0.05):
    n = len(records)
    k = sum(1 for r in records if r["compromised"])
    rate = k / n if n else 0.0
    return rate, _clopper_pearson_upper(k, n, alpha)


def _clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    except Exception:
        # closed form when k == 0; conservative fallback otherwise
        if k == 0:
            return 1 - (alpha / 2) ** (1 / n)
        return min(1.0, (k + 1.96 * (k * (n - k) / n) ** 0.5) / n)


def madt(records: list) -> float:
    return sum(r["steps"] for r in records) / len(records) if records else 0.0


def qos(records: list) -> float:
    return sum(r.get("qos_overhead", 0.0) for r in records) / len(records) if records else 0.0


def kcsd(records: list) -> float:
    """Mean # of stages from which the attacker was misdirected (lured/captured)."""
    tot = 0
    for r in records:
        stages = {s["foothold"] for s in r["trace"] if s["outcome"] in ("lured", "captured")}
        tot += len(stages)
    return tot / len(records) if records else 0.0


def ddr(records: list) -> Optional[float]:
    """Deception Detection Rate: fraction of shown decoys the attacker flagged."""
    num = den = 0
    for r in records:
        for s in r["trace"]:
            shown = set(s.get("decoys_shown", []))
            if not shown:
                continue
            flagged = set(s.get("flagged_decoys", []))
            num += len(shown & flagged)
            den += len(shown)
    return num / den if den else None


def suspicion_auroc(records: list) -> Optional[float]:
    """AUROC of per-node suspicion score vs ground-truth decoy label."""
    pos, neg = [], []      # scores for decoys (pos) and real nodes (neg)
    for r in records:
        for s in r["trace"]:
            shown = set(s.get("decoys_shown", []))
            susp = s.get("suspicion", {})
            for nid, score in susp.items():
                (pos if int(nid) in shown else neg).append(float(score))
    if not pos or not neg:
        return None
    from .stats import auroc            # fast rank-based Mann-Whitney AUROC
    return auroc(pos, neg)


def summarize(records: list) -> dict:
    rate, cp = hcr(records)
    _rcdi = rcdi(records)
    _rcdi_ln = rcdi_lengthnorm(records)
    _ddr = ddr(records)
    _auroc = suspicion_auroc(records)
    corr = rcdi_correlations(records)
    return dict(
        n=len(records),
        HCR=round(rate, 4), HCR_upper95=round(cp, 4),
        MADT=round(madt(records), 1),
        RCDI=(round(_rcdi, 4) if _rcdi is not None else None),
        RCDI_lengthnorm=(round(_rcdi_ln, 4) if _rcdi_ln is not None else None),
        RCDI_vs_rho_decoy_spearman=(round(corr["rho_decoy_spearman"], 4)
                                    if corr["rho_decoy_spearman"] is not None else None),
        RCDI_vs_length_spearman=(round(corr["length_spearman"], 4)
                                 if corr["length_spearman"] is not None else None),
        KCSD=round(kcsd(records), 2),
        dQoS=round(qos(records), 4),
        DDR=(round(_ddr, 4) if _ddr is not None else None),
        suspicion_AUROC=(round(_auroc, 4) if _auroc is not None else None),
    )
