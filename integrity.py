"""Reproducibility provenance + oracle-leakage auditing.

Two jobs:
  1. Stamp every run with the git commit, config hash, renderer version-hash,
     PREREG hash, and library versions, so a result log is self-describing and a
     cross-condition comparison can be rejected if the renderer/config differed.
  2. `leakage_scan` audits the LOGGED observations for any single exposed node
     field that (near-)perfectly separates decoys from real nodes — the Paper-2
     `attack_progress` oracle-leakage lesson, operationalised. Run this before
     trusting DDR / suspicion-AUROC on any real backend.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from typing import Optional

from .env import renderer_version_hash
from .stats import auroc as _auroc

_NUMERIC_FIELDS = ["apparent_vuln", "consistency", "corroborations",
                   "latency_ms", "cert_expiry_days", "node"]
_CATEGORICAL_FIELDS = ["kind", "service_banner", "firmware"]
_LEAK_HI, _LEAK_LO = 0.99, 0.01     # AUROC bands that flag a near-perfect separator


# ---- provenance ------------------------------------------------------------
def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=repo_dir or os.getcwd(),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def file_hash(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]


def lib_versions() -> dict:
    v = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "yaml"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def run_manifest(config: dict, repo_dir: Optional[str] = None,
                 prereg_path: Optional[str] = None) -> dict:
    """Self-describing provenance record to write alongside every run."""
    return dict(
        git_commit=git_commit(repo_dir),
        config_hash=config_hash(config),
        renderer_version=renderer_version_hash(),
        prereg_hash=file_hash(prereg_path) if prereg_path else None,
        lib_versions=lib_versions(),
        config=config,
    )


# ---- oracle-leakage audit --------------------------------------------------
def leakage_scan(records: list) -> dict:
    """Scan logged node views for fields that separate decoys from real nodes.

    Returns {fields: {name: {...}}, leaks: [names], any_leak: bool}. A numeric
    field with AUROC >= 0.99 or <= 0.01 is a near-perfect separator; a
    categorical field whose decoy value-set is disjoint from its real value-set
    is a categorical oracle. These indicate the attacker can flag decoys WITHOUT
    the intended coherence reasoning — invalidating DDR / suspicion-AUROC.
    """
    num = {f: ([], []) for f in _NUMERIC_FIELDS}      # field -> (decoy_vals, real_vals)
    cat = {f: (set(), set()) for f in _CATEGORICAL_FIELDS}
    n_steps = n_decoy_views = n_real_views = 0

    for r in records:
        for s in r.get("trace", []):
            shown = set(s.get("decoys_shown", []))
            views = s.get("neighbors", [])
            if not views:
                continue
            n_steps += 1
            for v in views:
                is_decoy = v.get("node") in shown
                n_decoy_views += is_decoy
                n_real_views += (not is_decoy)
                for f in _NUMERIC_FIELDS:
                    if f in v and isinstance(v[f], (int, float)):
                        num[f][0 if is_decoy else 1].append(float(v[f]))
                for f in _CATEGORICAL_FIELDS:
                    if f in v:
                        cat[f][0 if is_decoy else 1].add(v[f])

    fields, leaks = {}, []
    for f, (dec, real) in num.items():
        a = _auroc(dec, real)
        leak = a is not None and (a >= _LEAK_HI or a <= _LEAK_LO)
        fields[f] = dict(type="numeric", auroc=(round(a, 4) if a is not None else None),
                         leak=leak)
        if leak:
            leaks.append(f)
    for f, (dec, real) in cat.items():
        disjoint = bool(dec) and bool(real) and dec.isdisjoint(real)
        fields[f] = dict(type="categorical", decoy_values=len(dec),
                         real_values=len(real), disjoint=disjoint, leak=disjoint)
        if disjoint:
            leaks.append(f)

    return dict(fields=fields, leaks=leaks, any_leak=bool(leaks),
                n_steps=n_steps, n_decoy_views=n_decoy_views, n_real_views=n_real_views)
