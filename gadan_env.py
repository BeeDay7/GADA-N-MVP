"""Real GADA-N+ CPS mission-network deception environment (Route 2).

Adapts the *real* dynamics of Paper 4's CPSDeceptionEnv — a branching CPS graph,
per-step ephemeral deception, detection-to-capture, and exfil-at-HVT — into a
SINGLE-defender GADA-N+ setting driven by an LLM attacker through the harness. It
implements `adapter.DynamicsBackend`, so it plugs into `HarnessEnv` behind the
frozen presentation-id renderer + leakage contract.

Unlike the earlier toy (a single canonical path), this is a real DAG with
MULTIPLE productive paths and DEAD-ENDS, so navigation is non-trivial and the
metrics are graded (not saturated 0/1). Deception is defence-conditioned:
  none      : no deception (undefended baseline; productive nodes look attractive)
  naive     : incoherent decoys, true path NOT disguised (detectable deception)
  gadanplus : coherent decoys (ideally the trained GDE) + the true path disguised
Wire the trained generator via `decoy_view_fn`; the built-in fallback uses
leakage-clean overlapping distributions (real & decoy indistinguishable by any
single field — verified by integrity.leakage_scan).
"""
from __future__ import annotations

import random
from typing import Callable, Optional

# ---- CPS mission-network topology (DAG toward the HVT) ----------------------
# id : (kind, [forward node ids])
NODES = {
    0: ("EDGE",        [1, 2, 3]),     # entry / foothold
    1: ("UAV_C2",      [4, 6]),
    2: ("UGV_C2",      [5]),
    3: ("SENSOR_AGG",  [6]),           # branch that dead-ends
    4: ("UAV_RELAY",   [7]),
    5: ("UGV_RELAY",   [7]),
    6: ("SENSOR_NODE", []),            # DEAD-END (cannot reach HVT)
    7: ("HVT",         []),            # goal
}
ENTRY, HVT = 0, 7
KIND_VOCAB = ["UAV_C2", "UGV_C2", "UAV_RELAY", "UGV_RELAY", "SENSOR_NODE",
              "SENSOR_AGG", "EDGE", "host", "workstation"]

DEFENCE = {
    "none":      dict(deception=False, p_decoy=0.0, p_capture=0.0,  disguise=False, decoy_coherent=True),
    "naive":     dict(deception=True,  p_decoy=0.6, p_capture=0.15, disguise=False, decoy_coherent=False),
    "gadanplus": dict(deception=True,  p_decoy=0.9, p_capture=0.15, disguise=True,  decoy_coherent=True),
}
_DECISION_OPS = {"exploit", "pivot", "connect", "exfiltrate"}


def _reachable_to_hvt() -> set:
    """Nodes from which the HVT is reachable (BFS backward over the DAG)."""
    rev = {n: [] for n in NODES}
    for n, (_, fwd) in NODES.items():
        for m in fwd:
            rev[m].append(n)
    seen, stack = {HVT}, [HVT]
    while stack:
        n = stack.pop()
        for p in rev[n]:
            if p not in seen:
                seen.add(p); stack.append(p)
    return seen


REACHES_HVT = _reachable_to_hvt()          # {0,1,2,4,5,7} — dead-ends 3,6 excluded


def _depth() -> dict:
    """Shortest-hop distance from ENTRY (for the kill-chain stage)."""
    dist, frontier = {ENTRY: 0}, [ENTRY]
    while frontier:
        nxt = []
        for n in frontier:
            for m in NODES[n][1]:
                if m not in dist:
                    dist[m] = dist[n] + 1; nxt.append(m)
        frontier = nxt
    return dist


DEPTH = _depth()


class GADANDynamics:
    """DynamicsBackend for the real GADA-N+ CPS deception environment."""

    def __init__(self, defence: str = "gadanplus",
                 decoy_view_fn: Optional[Callable] = None,
                 detect_to_catch: int = 3):
        assert defence in DEFENCE, defence
        self.defence = defence
        self.cfg = DEFENCE[defence]
        self.decoy_view_fn = decoy_view_fn      # trained GDE generator (optional)
        self.detect_to_catch = detect_to_catch
        self.reset(0)

    # --- DynamicsBackend interface ---
    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self._foothold = ENTRY
        self._stage = 1
        self._detections = 0
        self._qos = 0.0
        self._cur_decoy_ids = set()
        self._forward_real = set()
        self._decoy_seq = 0

    def foothold(self):
        return self._foothold

    def hvt(self):
        return HVT

    def stage(self) -> int:
        return self._stage

    def real_next(self):
        """Forward neighbours of the foothold that can still reach the HVT
        (multiple = multi-path). Dead-end forward nodes are excluded."""
        fwd = NODES[self._foothold][1]
        return {m for m in fwd if m in REACHES_HVT}

    def presented_nodes(self) -> list:
        """Real forward neighbours (productive + dead-end) plus injected decoys."""
        fwd = list(NODES[self._foothold][1])
        self._forward_real = set(fwd)
        self._cur_decoy_ids = set()
        rn = self.real_next()
        out = []
        for n in fwd:
            out.append({"id": n, "is_decoy": False,
                        "view": self._view(kind=NODES[n][0],
                                           is_real_next=(n in rn), is_decoy=False)})
        if self.cfg["deception"] and rn and self.rng.random() < self.cfg["p_decoy"]:
            self._decoy_seq += 1
            did = ("decoy", self._decoy_seq)       # opaque decoy id (adapter hides it)
            self._cur_decoy_ids.add(did)
            out.append({"id": did, "is_decoy": True,
                        "view": self._view(kind=self.rng.choice(KIND_VOCAB),
                                           is_real_next=False, is_decoy=True)})
        return out

    def apply(self, node_id, op: str) -> dict:
        if self.cfg["deception"]:
            self._qos = min(0.05, self._qos + self.rng.uniform(0.0, 0.008))
        outcome, done, result = "wasted", False, "running"
        if op in _DECISION_OPS:
            if node_id in self._cur_decoy_ids:
                self._detections += 1
                if self.rng.random() < self.cfg["p_capture"]:
                    outcome, done, result = "captured", True, "captured"
                else:
                    outcome = "lured"
            elif node_id in self._forward_real:
                self._foothold = node_id
                self._stage = min(5, DEPTH.get(node_id, self._stage) + 1)
                if node_id == HVT:
                    outcome, done, result = "compromise", True, "compromise"
                elif node_id in REACHES_HVT:
                    outcome = "progress"
                else:
                    outcome = "wasted"              # advanced into a dead-end
        elif op == "probe":
            outcome = "probe"
        elif op == "scan":
            outcome = "scan"
        if not done and self._detections >= self.detect_to_catch:
            outcome, done, result = "captured", True, "captured"
        return {"outcome": outcome, "done": done, "result": result}

    def qos_overhead(self) -> float:
        return self._qos

    # --- leakage-clean telemetry views (overlapping distributions) ----------
    def _view(self, kind: str, is_real_next: bool, is_decoy: bool) -> dict:
        if is_decoy and self.decoy_view_fn is not None:
            v = dict(self.decoy_view_fn(self.rng))   # trained GDE generator output
            v.setdefault("kind", kind)
            return v
        r = self.rng
        if is_decoy:
            apparent = r.uniform(0.55, 0.90)                       # decoys look attractive
        elif is_real_next and self.cfg["disguise"]:
            apparent = r.uniform(0.10, 0.45)                       # GADA-N+ disguises true path
        elif is_real_next:
            apparent = r.uniform(0.70, 0.95)
        else:
            apparent = r.uniform(0.45, 0.85)                       # dead-end / sibling
        # coherence: overlaps for coherent decoys; weaker for naive (detectable)
        if not is_decoy:
            coh = r.uniform(0.70, 1.0)
        elif self.cfg["decoy_coherent"]:
            coh = r.uniform(0.60, 0.95)
        else:
            coh = r.uniform(0.20, 0.60)
        p2 = 0.85 if not is_decoy else (0.55 if self.cfg["decoy_coherent"] else 0.2)
        return dict(kind=kind,
                    service_banner=f"svc/{r.randint(1, 9)}.{r.randint(0, 20)}",
                    firmware=f"fw-{r.randint(1, 5)}.{r.randint(0, 30)}",
                    cert_expiry_days=r.randint(1, 720),
                    latency_ms=round(r.uniform(5, 45), 1),
                    apparent_vuln=round(apparent, 3),
                    consistency=round(coh, 3),
                    corroborations=2 if r.random() < p2 else 1)


def make_env(defence: str = "gadanplus", awareness: str = "A0", max_steps: int = 60,
             seed: int = 0, decoy_view_fn=None, strict_leakage: bool = True):
    """Build a harness-ready GADA-N+ environment (GADANDynamics behind HarnessEnv)."""
    from .adapter import HarnessEnv
    return HarnessEnv(GADANDynamics(defence=defence, decoy_view_fn=decoy_view_fn),
                      defence=defence, awareness=awareness, max_steps=max_steps,
                      seed=seed, strict_leakage=strict_leakage)
