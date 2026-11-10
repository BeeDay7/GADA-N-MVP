"""Inferential statistics for the evaluation.

Welch t (parametric) + Mann-Whitney U (non-parametric fallback), Cohen's d and
Cliff's delta effect sizes, Holm-Bonferroni family correction, percentile
bootstrap CIs, and a fast rank-based AUROC shared by the metrics/leakage code.
Kept dependency-light: pure Python + optional scipy for exact p-values.
"""
from __future__ import annotations
import math
import random


# ---- rank helpers ----------------------------------------------------------
def _avg_ranks(values):
    """Average ranks (1-based) with ties shared, plus the tie-group sizes."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    ties = []
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def auroc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), computed in O(n log n) via rank sums.

    Equivalent to the Mann-Whitney-U-based AUROC; replaces the naive O(n*m)
    double loop so large view/score sets scan quickly."""
    if not pos or not neg:
        return None
    combined = list(pos) + list(neg)
    ranks, _ = _avg_ranks(combined)
    n_pos, n_neg = len(pos), len(neg)
    rank_sum_pos = sum(ranks[:n_pos])
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


# ---- effect sizes ----------------------------------------------------------
def cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) or 1e-9
    return (mb - ma) / sp


def cliffs_delta(a, b) -> float:
    """Non-parametric effect size in [-1, 1]: P(b>a) - P(a>b)."""
    if not a or not b:
        return 0.0
    return 2 * auroc(b, a) - 1     # oriented like Cohen's d (b relative to a)


# ---- tests -----------------------------------------------------------------
def welch_t(a, b):
    """Return (t, dof, p_two_sided). Falls back to a normal approx without scipy."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    se = math.sqrt(va / na + vb / nb) or 1e-9
    t = (mb - ma) / se
    dof = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1 + 1e-9) + (vb / nb) ** 2 / (nb - 1 + 1e-9) + 1e-12)
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(abs(t), dof)
    except Exception:
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))   # normal approx
    return t, dof, p


def mann_whitney_u(a, b):
    """Two-sided Mann-Whitney U with tie-corrected normal approximation.
    Returns (U, p). Uses scipy for an exact/continuity-corrected p when present."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0, 1.0
    try:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except Exception:
        pass
    ranks, ties = _avg_ranks(list(a) + list(b))
    r1 = sum(ranks[:na])
    u1 = r1 - na * (na + 1) / 2
    n = na + nb
    mu = na * nb / 2
    tie_term = sum(t ** 3 - t for t in ties) / (n * (n - 1))
    sigma = math.sqrt(na * nb / 12 * ((n + 1) - tie_term)) or 1e-9
    z = (u1 - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return u1, p


def is_normal(data, alpha: float = 0.05):
    """Shapiro-Wilk normality check (scipy). Returns (is_normal, p) or (None, None)
    when the sample is too small or scipy is unavailable."""
    if len(data) < 3:
        return None, None
    try:
        from scipy.stats import shapiro
        _, p = shapiro(data)
        return p > alpha, float(p)
    except Exception:
        return None, None


def bootstrap_ci(data, stat_fn=None, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0):
    """Percentile bootstrap CI for a statistic (default: the mean)."""
    if not data:
        return (None, None)
    stat_fn = stat_fn or (lambda d: sum(d) / len(d))
    rng = random.Random(seed)
    n = len(data)
    stats = []
    for _ in range(n_boot):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        stats.append(stat_fn(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


# ---- family correction + comparison ----------------------------------------
def holm_bonferroni(pvals: dict, alpha: float = 0.05) -> dict:
    """Return {name: (p, adjusted_significant)} using Holm's step-down."""
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out, still = {}, True
    for i, (name, p) in enumerate(ordered):
        thresh = alpha / (m - i)
        sig = still and (p <= thresh)
        if not sig:
            still = False
        out[name] = (p, sig)
    return out


def compare(primary: list, others: dict, alpha: float = 0.05,
            bootstrap: bool = True) -> dict:
    """Compare a reference sample against each of `others`.

    Auto-selects Welch t when both samples look normal, else Mann-Whitney U.
    Reports both effect sizes, a bootstrap CI of the mean difference, and the
    Holm-Bonferroni-corrected significance across the comparison family.
    """
    raw, detail = {}, {}
    for name, s in others.items():
        na, _ = is_normal(primary)
        nb, _ = is_normal(s)
        use_param = (na is not False) and (nb is not False)   # None (small n) -> allow param
        if use_param:
            _, _, p = welch_t(primary, s)
            test = "welch_t"
        else:
            _, p = mann_whitney_u(primary, s)
            test = "mann_whitney_u"
        raw[name] = p
        diff = [x for x in s]  # difference of means via bootstrap on the two samples
        detail[name] = dict(
            test=test, p=p,
            cohens_d=cohens_d(primary, s), cliffs_delta=cliffs_delta(primary, s),
            mean_ref=(sum(primary) / len(primary) if primary else None),
            mean_other=(sum(s) / len(s) if s else None))
        if bootstrap and primary and s:
            db = _mean_diff_bootstrap(primary, s, alpha=alpha)
            detail[name]["mean_diff_ci"] = db
    adj = holm_bonferroni(raw, alpha)
    for name in detail:
        detail[name]["significant_holm"] = adj[name][1]
    return detail


def _mean_diff_bootstrap(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    diffs = []
    for _ in range(n_boot):
        ma = sum(a[rng.randrange(na)] for _ in range(na)) / na
        mb = sum(b[rng.randrange(nb)] for _ in range(nb)) / nb
        diffs.append(mb - ma)
    diffs.sort()
    lo = diffs[int((alpha / 2) * n_boot)]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)
