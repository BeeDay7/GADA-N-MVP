"""Route-1 integration seam: plug a REAL environment's dynamics into the harness
behind the frozen presentation-id renderer + leakage contract.

Your GADA-N environment implements `DynamicsBackend` (below). `HarnessEnv` wraps
it to expose the exact `reset/observe/step/summary` interface the rest of the
harness already consumes (`attacker.run_episode`, `metrics`, `run.py`,
`run_conditions.py`) — while the adapter, NOT your env, owns:
  * opaque presentation ids (real & decoy nodes share one namespace, so the
    attacker can't separate them by id), and
  * the leakage contract (`env.check_observation_leakage`, strict for real runs).

This keeps the observation renderer identical across every condition (the
methodological linchpin) no matter what your env's internal dynamics are.

You implement DynamicsBackend over your GADA-N env; everything else is done.
See `MockDynamics` for a minimal, working reference implementation.
"""
from __future__ import annotations

import random
from typing import Optional, Protocol, runtime_checkable

from .env import StepInfo, check_observation_leakage, _PRES_LO, _PRES_HI

# Attacker-visible node attributes the renderer exposes. Your env supplies these
# per presented node (real AND decoy) with an IDENTICAL key set — no ground-truth
# fields (is_decoy/synthetic/…) may appear here; that's what the contract enforces.
OBS_ATTRS = ("kind", "service_banner", "firmware", "cert_expiry_days",
             "latency_ms", "apparent_vuln", "consistency", "corroborations")


@runtime_checkable
class DynamicsBackend(Protocol):
    """Implement this over your real GADA-N env. Node ids are YOUR env's own ids
    (the adapter assigns opaque presentation ids). Ground truth (real vs decoy,
    the true next hop) is provided here but NEVER shown to the attacker."""

    def reset(self, seed: int) -> None:
        """Reset to the episode start for `seed`."""

    def foothold(self):
        """Attacker's current real node id."""

    def hvt(self):
        """High-value-target real node id."""

    def stage(self) -> int:
        """Current kill-chain stage (1..5)."""

    def presented_nodes(self) -> list:
        """Nodes visible from the foothold THIS step. Each item:
        {"id": <real id>, "is_decoy": bool, "view": {<OBS_ATTRS>}}.
        `view` is what the attacker sees; give real and decoy nodes the SAME keys
        and overlapping value distributions (the GDE/LCAM's job)."""

    def real_next(self):
        """Real id of the true productive next hop from the foothold (or None)."""

    def apply(self, node_id, op: str) -> dict:
        """Advance the dynamics for the attacker's chosen real `node_id` + `op`.
        Return {"outcome": str, "done": bool, "result": str}. `result` is
        'compromise' on attacker win (used for HCR)."""

    def qos_overhead(self) -> float:
        """Cumulative QoS-violation overhead in [0,1] (for dQoS)."""


class HarnessEnv:
    """Wraps a DynamicsBackend to the harness env interface with pres-id rendering
    + leakage contract. Drop-in for CPSEnv in run.py / run_episode."""

    def __init__(self, backend: DynamicsBackend, defence: str = "gadanplus",
                 awareness: str = "A0", max_steps: int = 60, seed: int = 0,
                 strict_leakage: bool = True):
        self.backend = backend
        self.defence = defence
        self.awareness = awareness
        self.max_steps = max_steps
        self._seed = seed
        self.strict_leakage = strict_leakage
        self.reset()

    def reset(self):
        self.backend.reset(self._seed)
        self.steps = 0
        self.done = False
        self.result = "running"
        self._pres_of_real, self._real_of_pres, self._used_pres = {}, {}, set()
        self._rng = random.Random(self._seed ^ 0x9E3779B1)
        self._cur_decoys = []
        self._cur_real_next_pres = None
        self._cur_real_next_real = None
        return self.observe()

    def _fresh_pres(self) -> int:
        while True:
            pid = self._rng.randint(_PRES_LO, _PRES_HI)
            if pid not in self._used_pres:
                self._used_pres.add(pid)
                return pid

    def _pres(self, real_id):
        if real_id is None:
            return None
        if real_id not in self._pres_of_real:
            pid = self._fresh_pres()
            self._pres_of_real[real_id] = pid
            self._real_of_pres[pid] = real_id
        return self._pres_of_real[real_id]

    def observe(self) -> dict:
        b = self.backend
        rn = b.real_next()                       # single id, a collection, or None
        rn_set = (set(rn) if isinstance(rn, (list, tuple, set))
                  else set() if rn is None else {rn})
        views, self._cur_decoys, real_next_pres = [], [], []
        for nd in b.presented_nodes():
            pid = self._pres(nd["id"])
            v = {"node": pid}
            v.update({a: nd["view"][a] for a in OBS_ATTRS if a in nd["view"]})
            views.append(v)
            if nd.get("is_decoy"):
                self._cur_decoys.append(pid)
            elif nd["id"] in rn_set:
                real_next_pres.append(pid)       # multi-path: any of these is "correct"
        self._rng.shuffle(views)
        self._cur_real_next_pres = real_next_pres          # list of pres ids
        self._cur_real_next_real = rn_set
        check_observation_leakage({"neighbors": views}, self._cur_decoys,
                                  strict=self.strict_leakage)
        return dict(step=self.steps, foothold=self._pres(b.foothold()),
                    stage=b.stage(), hvt_id=self._pres(b.hvt()), neighbors=views,
                    note=("Some telemetry may be synthetic." if self.awareness != "A0" else ""))

    def step(self, action: dict):
        assert not self.done
        self.steps += 1
        op = action.get("op", "scan")
        real_id = self._real_of_pres.get(action.get("node"))
        decoys = list(self._cur_decoys)
        rn_pres = self._cur_real_next_pres           # list of pres ids (may be empty)
        out = self.backend.apply(real_id, op)
        outcome = out.get("outcome", "wasted")
        if out.get("done"):
            self.done, self.result = True, out.get("result", "done")
        if not self.done and self.steps >= self.max_steps:
            self.done, self.result = True, "timeout"
        info = StepInfo(
            stage=self.backend.stage(), foothold=self.backend.foothold(),
            real_next=(rn_pres if rn_pres else None), decoys_shown=decoys,
            synthetic_fields={},
            correct_action=({"op": "exploit", "node": rn_pres[0]} if rn_pres else None),
            outcome=outcome,
            gt=dict(foothold_real=real_id, real_next_real=sorted(self._cur_real_next_real),
                    decoy_pres_ids=decoys))
        return (self.observe() if not self.done else None), self.done, info

    def summary(self) -> dict:
        return dict(result=self.result, steps=self.steps,
                    stage_reached=self.backend.stage(),
                    compromised=(self.result == "compromise"),
                    qos_overhead=round(self.backend.qos_overhead(), 4),
                    defence=self.defence, awareness=self.awareness)


# --------------------------------------------------------------------------- #
class MockDynamics:
    """Minimal reference DynamicsBackend (NOT a model) — shows exactly what your
    GADA-N env must implement, and lets `HarnessEnv` be tested end-to-end."""

    def __init__(self):
        self._path = [0, 1, 2, 3]            # 3 = HVT
        self.reset(0)

    def reset(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._i, self._stage, self._qos = 0, 1, 0.0

    def foothold(self):
        return self._path[self._i]

    def hvt(self):
        return self._path[-1]

    def stage(self) -> int:
        return self._stage

    def real_next(self):
        return self._path[self._i + 1] if self._i + 1 < len(self._path) else None

    def _view(self) -> dict:                 # identical distribution for real & decoy
        r = self._rng
        return dict(kind=r.choice(["UAV", "UGV", "EDGE"]),
                    service_banner=f"svc/{r.randint(1, 9)}", firmware=f"fw-{r.randint(1, 5)}",
                    cert_expiry_days=r.randint(1, 720), latency_ms=round(r.uniform(5, 45), 1),
                    apparent_vuln=round(r.uniform(0.5, 0.9), 3),
                    consistency=round(r.uniform(0.6, 1.0), 3),
                    corroborations=2 if r.random() < 0.8 else 1)

    def presented_nodes(self) -> list:
        rn = self.real_next()
        nodes = []
        if rn is not None:
            nodes.append({"id": rn, "is_decoy": False, "view": self._view()})
        nodes.append({"id": 100 + self._i, "is_decoy": False, "view": self._view()})
        if self._rng.random() < 0.7:
            nodes.append({"id": 900 + self._rng.randint(0, 50), "is_decoy": True, "view": self._view()})
        return nodes

    def apply(self, node_id, op: str) -> dict:
        self._qos = min(0.05, self._qos + 0.005)
        if op in ("exploit", "pivot", "connect", "exfiltrate"):
            if node_id == self.real_next():
                self._i += 1
                self._stage = min(5, self._stage + 1)
                if self.foothold() == self.hvt():
                    return {"outcome": "compromise", "done": True, "result": "compromise"}
                return {"outcome": "progress", "done": False}
            return {"outcome": "wasted", "done": False}
        return {"outcome": op, "done": False}

    def qos_overhead(self) -> float:
        return self._qos
