"""Figures: the awareness-ladder curve (HCR/MADT/DDR vs A0->A3) and per-stage
RCDI distributions. Data is always written to CSV so plots regenerate from saved
artifacts; PNG/PDF are produced when matplotlib is available.

Titles are stamped NOT PAPER DATA whenever the underlying runs used the mock
backend, so reference figures can never be mistaken for results.
"""
from __future__ import annotations
import csv
import os

from . import metrics

_AWARE_ORDER = {"A0": 0, "A1": 1, "A2": 2, "A3": 3}


def _stamp(title: str, mock: bool) -> str:
    return title + ("  [NOT PAPER DATA]" if mock else "")


# ---- per-stage RCDI --------------------------------------------------------
def per_stage_rcdi(records: list) -> dict:
    """RCDI aggregated by kill-chain stage: {stage: rcdi}."""
    by_stage = {}   # stage -> [scored labels]
    for r in records:
        for s in r.get("trace", []):
            lab = metrics.label_step(s)
            if lab not in metrics._SCORED:
                continue
            by_stage.setdefault(s.get("stage"), []).append(lab)
    out = {}
    for stage, labs in by_stage.items():
        if stage is None or not labs:
            continue
        disrupted = sum(1 for l in labs if l in metrics._DISRUPTED)
        out[stage] = disrupted / len(labs)
    return dict(sorted(out.items()))


# ---- assembly --------------------------------------------------------------
def _ladder_rows(results: list):
    """(attacker, awareness) -> summary, for gadanplus conditions only."""
    rows = []
    for r in results:
        c, s = r["config"], r["summary"]
        if c["defence"] != "gadanplus":
            continue
        rows.append((c["attacker"], c["awareness"], s))
    return sorted(rows, key=lambda t: (t[0], _AWARE_ORDER.get(t[1], 9)))


def render_all(results: list, out_dir: str, mock: bool = False) -> str:
    os.makedirs(out_dir, exist_ok=True)
    made = []

    # 1) awareness-ladder data
    ladder = _ladder_rows(results)
    ladder_csv = os.path.join(out_dir, "fig_awareness_ladder.csv")
    with open(ladder_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["attacker", "awareness", "HCR", "HCR_upper95", "MADT", "DDR", "suspicion_AUROC"])
        for att, aw, s in ladder:
            w.writerow([att, aw, s["HCR"], s["HCR_upper95"], s["MADT"], s["DDR"], s["suspicion_AUROC"]])
    made.append(ladder_csv)

    # 2) per-stage RCDI data (per gadanplus condition)
    stage_csv = os.path.join(out_dir, "fig_per_stage_rcdi.csv")
    with open(stage_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["attacker", "awareness", "stage", "rcdi"])
        for r in results:
            c = r["config"]
            if c["defence"] != "gadanplus":
                continue
            for stage, val in per_stage_rcdi(r["records"]).items():
                w.writerow([c["attacker"], c["awareness"], stage, round(val, 4)])
    made.append(stage_csv)

    # 3) plots (optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return f"CSV only (matplotlib unavailable): {', '.join(os.path.basename(m) for m in made)}"

    # awareness-ladder curve, one line per attacker
    by_att = {}
    for att, aw, s in ladder:
        by_att.setdefault(att, []).append((aw, s))
    if by_att:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
        for metric_i, (key, label) in enumerate([("HCR", "HCR"), ("MADT", "MADT (steps)"),
                                                  ("DDR", "DDR")]):
            for att, seq in by_att.items():
                seq = sorted(seq, key=lambda t: _AWARE_ORDER.get(t[0], 9))
                xs = [aw for aw, _s in seq]
                ys = [(s.get(key) if s.get(key) is not None else float("nan"))
                      for _aw, s in seq]
                ax[metric_i].plot(xs, ys, marker="o", label=att)
            ax[metric_i].set_xlabel("awareness")
            ax[metric_i].set_ylabel(label)
            ax[metric_i].grid(True, alpha=0.3)
        ax[0].legend(fontsize=8)
        fig.suptitle(_stamp("Awareness ladder: GADA-N+ under A0->A3", mock))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = os.path.join(out_dir, f"fig_awareness_ladder.{ext}")
            fig.savefig(p, dpi=150)
            made.append(p)
        plt.close(fig)

    # per-stage RCDI bars (gadanplus/A0 if present, else first gadanplus)
    gp = [r for r in results if r["config"]["defence"] == "gadanplus"]
    target = next((r for r in gp if r["config"]["awareness"] == "A0"), gp[0] if gp else None)
    if target:
        ps = per_stage_rcdi(target["records"])
        if ps:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.bar([str(k) for k in ps], list(ps.values()), color="#b03a2e")
            ax.set_xlabel("kill-chain stage")
            ax.set_ylabel("RCDI")
            ax.set_ylim(0, 1)
            ax.set_title(_stamp(f"Per-stage RCDI ({target['config']['attacker']}/"
                                f"{target['config']['awareness']})", mock))
            fig.tight_layout()
            for ext in ("png", "pdf"):
                p = os.path.join(out_dir, f"fig_per_stage_rcdi.{ext}")
                fig.savefig(p, dpi=150)
                made.append(p)
            plt.close(fig)

    return ", ".join(os.path.basename(m) for m in made)
