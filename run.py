"""CLI driver: run ONE condition (N seeds x M episodes), log JSONL, print summary.

Smoke test (no API/GPU):
    python -m gadanplus.run --defence gadanplus --attacker mock --seeds 3 --episodes 20

Real run:
    python -m gadanplus.run --condition C1 --defence gadanplus --attacker gpt4o \
        --seeds 40 --episodes 100 --out results/C1

Every run writes a provenance manifest (git commit, config/renderer/PREREG
hashes, lib versions) and an oracle-leakage audit alongside the metrics, and
excludes malformed-output episodes per the pre-registered rule.

`run_condition()` is the reusable core used by both this CLI and
`run_conditions.py` (the C1..C6c suite driver).
"""
from __future__ import annotations
import argparse
import json
import os

from . import integrity, metrics
from .attacker import make_backend, run_episode
from .env import CPSEnv

BANNER = "=" * 8 + " NOT PAPER DATA (reference harness) " + "=" * 8
_DEFAULT_PREREG = os.path.join(os.path.dirname(__file__), "..", "PREREG.md")


def run_condition(cfg: dict, out_dir: str, prereg_path: str = _DEFAULT_PREREG) -> dict:
    """Run one condition from a config dict. Returns a result dict with the
    summary, leakage audit, manifest, and output paths. Writes JSONL + summary
    + manifest + leakage json under `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    condition = cfg.get("condition", "Cx")
    defence = cfg["defence"]
    attacker = cfg["attacker"]
    awareness = cfg.get("awareness", "A0")
    variant = int(cfg.get("variant", 0))
    seeds = int(cfg.get("seeds", 3))
    episodes = int(cfg.get("episodes", 20))
    max_steps = int(cfg.get("max_steps", 60))
    env_kind = str(cfg.get("env", "gadan"))            # "gadan" (real) | "cps_toy" (reference)

    tag = f"{condition}_{defence}_{attacker}_{awareness}_v{variant}"
    log_path = os.path.join(out_dir, f"{tag}.jsonl")
    config = dict(condition=condition, defence=defence, attacker=attacker,
                  awareness=awareness, variant=variant, seeds=seeds,
                  episodes=episodes, max_steps=max_steps, env=env_kind)
    manifest = integrity.run_manifest(config, repo_dir=os.getcwd(), prereg_path=prereg_path)

    records = []
    with open(log_path, "w", encoding="utf-8") as f:
        for seed in range(seeds):
            cache = os.path.join(out_dir, f"cache_{attacker}_{awareness}_v{variant}.jsonl")
            backend = make_backend(attacker, awareness=awareness, seed=seed,
                                   variant=variant, cache_path=cache)
            for ep in range(episodes):
                if env_kind == "gadan":
                    from .gadan_env import make_env
                    env = make_env(defence=defence, awareness=awareness,
                                   max_steps=max_steps, seed=seed * 10_000 + ep)
                else:
                    env = CPSEnv(defence=defence, awareness=awareness,
                                 max_steps=max_steps, seed=seed * 10_000 + ep)
                rec = run_episode(env, backend, max_steps=max_steps)
                rec.update(seed=seed, episode=ep, condition=condition,
                           attacker=attacker, variant=variant,
                           renderer_version=manifest["renderer_version"])
                f.write(json.dumps(rec) + "\n")
                records.append(rec)

    kept = [r for r in records if not r.get("excluded")]
    summary = metrics.summarize(kept)
    summary["n_excluded"] = len(records) - len(kept)
    summary["n_total"] = len(records)
    leak = integrity.leakage_scan(kept)

    with open(os.path.join(out_dir, f"{tag}_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(out_dir, f"{tag}_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(out_dir, f"{tag}_leakage.json"), "w", encoding="utf-8") as fh:
        json.dump(leak, fh, indent=2)

    return dict(tag=tag, config=config, summary=summary, leak=leak,
                manifest=manifest, log_path=log_path, records=kept)


def print_result(res: dict):
    cfg = res["config"]
    is_mock = cfg["attacker"] == "mock"
    if is_mock:
        print(BANNER)
    print(f"Condition {cfg['condition']} | defence={cfg['defence']} attacker={cfg['attacker']} "
          f"awareness={cfg['awareness']} v{cfg['variant']} "
          f"({cfg['seeds']}x{cfg['episodes']}={res['summary']['n_total']} episodes, "
          f"{res['summary']['n_excluded']} excluded)")
    print(json.dumps(res["summary"], indent=2))
    leak = res["leak"]
    if leak["any_leak"]:
        marker = "WARNING" if is_mock else "ERROR (real backend — do NOT report DDR/AUROC)"
        print(f"\n[LEAKAGE {marker}] single-field decoy oracle(s) detected: {leak['leaks']}")
        for name in leak["leaks"]:
            print(f"    - {name}: {leak['fields'][name]}")
    else:
        print(f"\n[leakage audit] no single-field decoy oracle detected "
              f"({leak['n_decoy_views']} decoy / {leak['n_real_views']} real views).")
    if is_mock:
        print(BANNER)


def _load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML condition config (CLI flags override)")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--defence", default=None, choices=["none", "naive", "gadanplus"])
    ap.add_argument("--attacker", default=None, choices=["mock", "gpt4o", "llama"])
    ap.add_argument("--awareness", default=None, choices=["A0", "A1", "A2", "A3"])
    ap.add_argument("--variant", type=int, default=None,
                    help="prompt-paraphrase index (0=pre-registered default; 0..2)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prereg", default=_DEFAULT_PREREG)
    args = ap.parse_args()

    cfg = _load_yaml(args.config) if args.config else {}
    defaults = dict(condition="Cx", defence="gadanplus", attacker="mock",
                    awareness="A0", variant=0, seeds=3, episodes=20, max_steps=60,
                    out="results/run")
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    # CLI overrides (only when explicitly provided)
    for k in ("condition", "defence", "attacker", "awareness", "variant", "seeds", "episodes"):
        val = getattr(args, k)
        if val is not None:
            cfg[k] = val
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.out is not None:
        cfg["out"] = args.out

    res = run_condition(cfg, cfg["out"], prereg_path=args.prereg)
    print_result(res)


if __name__ == "__main__":
    main()
