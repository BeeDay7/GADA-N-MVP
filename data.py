"""Dataset loader / preprocessor: fit a FeatureSpec from a real telemetry table
and encode it for the GDE cGAN. Column-adaptive, so it works on TON_IoT, CICIDS,
UNSW-NB15, SWaT/WADI exports, etc. without hard-coding schemas.

Pipeline: load_table -> fit_feature_spec -> encode (+ make_conditioning) ->
split -> save_feature_spec. Returns numpy float32 arrays (gde.train_cgan accepts
them directly). See DATA_REQUIREMENTS.md for dataset choices + licensing.
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np

from .gde import FeatureSpec


def load_table(path: str):
    import pandas as pd
    df = pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path)
    df = df.dropna(axis=1, how="all")
    return df


def fit_feature_spec(df, label_col: Optional[str] = None, kind_col: Optional[str] = None,
                     drop_cols: Optional[List[str]] = None, max_card: int = 20):
    """Infer numeric + (low-cardinality) categorical features from the table.

    - numeric  -> (name, min, max) for min-max scaling
    - categorical (object / <= max_card uniques) -> (name, sorted vocabulary)
    - id-like / high-cardinality / label / kind columns are excluded from features
    Returns (FeatureSpec, meta) where meta carries the encoders.
    """
    import pandas as pd
    drop = set(drop_cols or [])
    if label_col:
        drop.add(label_col)
    if kind_col:
        drop.add(kind_col)

    numeric, categorical = [], []
    for col in df.columns:
        if col in drop:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            lo, hi = float(s.min()), float(s.max())
            if hi <= lo:                          # constant column: skip
                continue
            numeric.append((col, lo, hi))
        else:
            vocab = sorted(map(str, s.dropna().unique()))
            if 1 < len(vocab) <= max_card:        # skip id-like/high-cardinality text
                categorical.append((col, vocab))

    kind_vocab = (sorted(map(str, df[kind_col].dropna().unique()))
                  if kind_col and kind_col in df.columns else ["node"])
    spec = FeatureSpec(numeric=[(n, lo, hi) for n, lo, hi in numeric],
                       categorical=[(n, len(v)) for n, v in categorical],
                       n_kinds=max(1, len(kind_vocab)))
    meta = dict(numeric=numeric, categorical=categorical,
                kind_col=kind_col, kind_vocab=kind_vocab, label_col=label_col)
    return spec, meta


def encode(df, meta) -> np.ndarray:
    """Encode the table into the (N, feat_dim) feature matrix (numeric min-max to
    [0,1] then per-categorical one-hot). Unknown categories -> all-zeros."""
    cols = []
    for name, lo, hi in meta["numeric"]:
        x = df[name].to_numpy(dtype=float)
        cols.append(np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)[:, None])
    for name, vocab in meta["categorical"]:
        idx = {v: i for i, v in enumerate(vocab)}
        oh = np.zeros((len(df), len(vocab)), dtype=np.float32)
        for r, val in enumerate(df[name].astype(str)):
            j = idx.get(val)
            if j is not None:
                oh[r, j] = 1.0
        cols.append(oh)
    return (np.concatenate(cols, axis=1) if cols else np.zeros((len(df), 0))).astype(np.float32)


def make_conditioning(df, meta, stage_col: Optional[str] = None) -> np.ndarray:
    """Conditioning c = kind one-hot + normalised stage scalar.

    In the real GADA-N integration, conditioning should carry (node type,
    kill-chain stage, defence context); here we build kind + an optional stage
    column (else a constant), matching FeatureSpec.cond_dim."""
    kind_col, vocab = meta.get("kind_col"), meta.get("kind_vocab", ["node"])
    idx = {v: i for i, v in enumerate(vocab)}
    kind = np.zeros((len(df), len(vocab)), dtype=np.float32)
    if kind_col and kind_col in df.columns:
        for r, val in enumerate(df[kind_col].astype(str)):
            j = idx.get(val)
            if j is not None:
                kind[r, j] = 1.0
    else:
        kind[:, 0] = 1.0
    if stage_col and stage_col in df.columns:
        s = df[stage_col].to_numpy(dtype=float)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        s = np.full(len(df), 0.5, dtype=float)
    return np.concatenate([kind, s[:, None]], axis=1).astype(np.float32)


def train_val_test_split(*arrays, val=0.1, test=0.1, seed=0):
    """Deterministic 80/10/10 (default) split across parallel arrays."""
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test, n_val = int(n * test), int(n * val)
    te, va, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    out = []
    for a in arrays:
        a = np.asarray(a)
        out.append((a[tr], a[va], a[te]))
    return out if len(out) > 1 else out[0]


def save_feature_spec(spec: FeatureSpec, meta: dict, path: str):
    import hashlib
    payload = dict(numeric=spec.numeric, categorical=spec.categorical,
                   n_kinds=spec.n_kinds, meta_numeric=meta["numeric"],
                   meta_categorical=meta["categorical"], kind_vocab=meta.get("kind_vocab"),
                   label_col=meta.get("label_col"), kind_col=meta.get("kind_col"))
    blob = json.dumps(payload, sort_keys=True, default=str)
    payload["preprocessing_hash"] = hashlib.sha1(blob.encode()).hexdigest()[:12]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return payload["preprocessing_hash"]


def _smoke():
    """Build a synthetic telemetry table, fit + encode + condition + split.
    NOT PAPER DATA — exercises the loader mechanics only."""
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(dict(
        latency_ms=rng.uniform(5, 45, n),
        cert_days=rng.integers(1, 720, n),
        pktloss=rng.uniform(0, 0.1, n),
        kind=rng.choice(["UAV", "UGV", "EDGE", "SENSOR"], n),
        banner=rng.choice([f"svc/{i}" for i in range(6)], n),
        flow_id=[f"f{i}" for i in range(n)],          # id-like -> should be dropped
        label=rng.integers(0, 2, n)))
    spec, meta = fit_feature_spec(df, label_col="label", kind_col="kind")
    feats = encode(df, meta)
    cond = make_conditioning(df, meta)
    assert feats.shape == (n, spec.feat_dim), (feats.shape, spec.feat_dim)
    assert cond.shape == (n, spec.cond_dim), (cond.shape, spec.cond_dim)
    assert "flow_id" not in [c for c, _ in meta["categorical"]], "id-like col leaked in"
    (ftr, fva, fte), (ctr, cva, cte) = train_val_test_split(feats, cond, seed=0)
    h = save_feature_spec(spec, meta, "results/_feature_spec.json")
    print(f"NOT PAPER DATA — loader smoke OK: feat_dim={spec.feat_dim} "
          f"cond_dim={spec.cond_dim} numeric={spec.num_dim} "
          f"cats={len(spec.categorical)} split={len(ftr)}/{len(fva)}/{len(fte)} hash={h}")
    return True


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    _smoke()
