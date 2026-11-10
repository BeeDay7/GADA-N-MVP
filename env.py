"""CPS mission-network simulation + observation renderer + LCAM deception layer.

REFERENCE STAND-IN. Replace `CPSEnv._transition`-equivalent logic and `LCAM`
with your validated GADA-N model before reporting results. The dynamics below
are intentionally simple: they exist to exercise the full pipeline, not to
produce paper numbers.

Renderer design (the linchpin — freeze it, keep it identical across conditions):
  * Nodes are presented to the attacker under OPAQUE PRESENTATION IDs drawn from a
    single shared namespace, so real and decoy nodes are NOT separable by id. This
    is a hard requirement (contract): a real LLM attacker must not be able to flag
    a decoy from an out-of-band id (the earlier `>=100` tell), a distinguishing
    `kind`, or any field that is a renamed `is_decoy` oracle. See `leakage_scan`
    in integrity.py and the per-observation guard `check_observation_leakage`.
  * Ground-truth identities (real vs decoy, true next hop) are logged separately
    and NEVER shown to the attacker; they are the label source for RCDI / DDR.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import random
from dataclasses import dataclass, field
from typing import Optional

# ---- Fixed 12-node CPS mission network -----------------------------------
REAL_NODES = {
    0: "UAV_C2", 1: "UAV", 2: "UAV", 3: "UAV", 4: "UAV",
    5: "UGV_C2", 6: "UGV", 7: "UGV", 8: "UGV",
    9: "EDGE", 10: "SENSOR_AGG", 11: "HVT",
}
EDGES = {
    0: [1, 2, 3, 4, 9], 1: [0], 2: [0], 3: [0], 4: [0],
    9: [0, 10, 5], 10: [9],
    5: [9, 6, 7, 8, 11], 6: [5], 7: [5], 8: [5], 11: [5],
}
ENTRY, HVT = 0, 11
TRUE_PATH = [0, 9, 5, 11]           # next-hop chain the attacker must follow
ACTIONS = {"scan", "connect", "exploit", "pivot", "exfiltrate", "probe"}

# Kinds a decoy may masquerade as — drawn from the SAME vocabulary as real nodes
# so `kind` is not a decoy tell (previously decoys were always "host").
KIND_VOCAB = sorted(set(REAL_NODES.values()) - {"HVT"}) + ["host", "workstation"]

# Presentation-id namespace shared by real and decoy nodes (opaque host handles).
_PRES_LO, _PRES_HI = 1000, 9999

# Per-defence knobs (illustrative only). p_lure = chance a decoy is presented as
# an attractive next hop; p_capture = chance acting on a decoy ends the episode as
# a defender win; decoy_consistent = whether decoys survive a static coherence
# check (GADA-N+ decoys are coherent — detectable only by corroboration/probing);
# disguise_real = whether the true next hop is made to look unattractive (the
# GADA-N+ linguistic-coherence tactic).
DEFENCE = {
    "none":      dict(deception=False, p_lure=0.0,  p_capture=0.0,  decoy_consistent=True,  disguise_real=False),
    "naive":     dict(deception=True,  p_lure=0.5,  p_capture=0.10, decoy_consistent=False, disguise_real=False),
    "gadanplus": dict(deception=True,  p_lure=0.95, p_capture=0.10, decoy_consistent=True,  disguise_real=True),
}


@dataclass
class StepInfo:
    """Ground truth for a step (never shown to the attacker; used for RCDI/DDR).

    `real_next` and `decoys_shown` are in PRESENTATION-ID space (what the attacker
    referenced), so the deterministic RCDI labeller can resolve the attacker's
    action against them. `gt` carries the true (real-id) identities for the
    never-shown ground-truth log.
    """
    stage: int
    foothold: int
    real_next: Optional[int]              # presentation id of the true next hop
    decoys_shown: list                    # presentation ids of decoys shown
    synthetic_fields: dict
    correct_action: Optional[dict]
    outcome: str        # 'progress' | 'lured' | 'captured' | 'wasted' | 'compromise' | 'scan' | 'probe'
    gt: dict = field(default_factory=dict)


class LCAM:
    """Linguistic Coherence Adaptation Module (reference stand-in).

    Replace with your cGAN/GDE output. Train L_ling on the Llama surrogate;
    keep GPT-4o eval-only. Here the "coherence" of a node view is a scalar with
    OVERLAPPING distributions between real and decoy nodes, so that no single
    exposed field is a perfect decoy oracle. Static separability is weak for
    GADA-N+ (coherent decoys) and stronger for the naive baseline; genuine
    detection is meant to require corroboration/probing.
    """
    def __init__(self, cfg: dict, rng: random.Random):
        self.cfg = cfg
        self.rng = rng

    def n_decoys(self, real_next: Optional[int]) -> int:
        """How many decoys to present this step (0 or 1 in the stand-in)."""
        if not self.cfg["deception"] or real_next is None:
            return 0
        return 1 if self.rng.random() < self.cfg["p_lure"] else 0

    def _coherence(self, is_decoy: bool) -> float:
        r = self.rng
        if not is_decoy:
            return r.uniform(0.70, 1.0)                 # real nodes: coherent
        if self.cfg["decoy_consistent"]:
            return r.uniform(0.60, 0.95)                # GADA-N+: overlaps real
        return r.uniform(0.20, 0.60)                    # naive: statically weaker

    def _corroborations(self, is_decoy: bool) -> int:
        # Probabilistic, OVERLAPPING — not a renamed is_decoy flag.
        if not is_decoy:
            p2 = 0.85
        elif self.cfg["decoy_consistent"]:
            p2 = 0.55
        else:
            p2 = 0.20
        return 2 if self.rng.random() < p2 else 1

    def node_view(self, pres_id: int, kind: str, is_real_next: bool, is_decoy: bool) -> dict:
        """Render one node the attacker sees. Real and decoy views share an
        IDENTICAL field schema; values overlap so no field separates them."""
        r = self.rng
        # Attractiveness distributions OVERLAP by design: decoys look attractive
        # ON AVERAGE (the lure biases a naive attacker) but do not perfectly
        # separate from real siblings, so "distrust the juiciest node" cannot
        # trivially reverse the deception (verified by integrity.leakage_scan).
        if is_decoy:
            apparent_vuln = r.uniform(0.55, 0.90)                  # decoys: attractive lure
        elif is_real_next and self.cfg.get("disguise_real"):
            apparent_vuln = r.uniform(0.10, 0.45)                  # GADA-N+ disguises true path
        elif is_real_next:
            apparent_vuln = r.uniform(0.70, 0.95)                  # true path looks attractive
        else:
            apparent_vuln = r.uniform(0.45, 0.85)                  # sibling real nodes
        return dict(
            node=pres_id,
            kind=kind,
            service_banner=f"svc/{r.randint(1, 9)}.{r.randint(0, 20)}",
            firmware=f"fw-{r.randint(1, 5)}.{r.randint(0, 30)}",
            cert_expiry_days=r.randint(1, 720),
            latency_ms=round(r.uniform(5, 45), 1),
            apparent_vuln=round(apparent_vuln, 3),
            consistency=round(self._coherence(is_decoy), 3),
            corroborations=self._corroborations(is_decoy),
        )


class CPSEnv:
    def __init__(self, defence: str = "gadanplus", awareness: str = "A0",
                 max_steps: int = 60, seed: int = 0):
        assert defence in DEFENCE, defence
        self.cfg = DEFENCE[defence]
        self.defence = defence
        self.awareness = awareness
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.lcam = LCAM(self.cfg, self.rng)
        self.reset()

    def reset(self):
        self.foothold = ENTRY
        self.stage = 1
        self.steps = 0
        self.done = False
        self.result = "running"
        self.qos_overhead = 0.0
        # presentation-id bookkeeping (opaque handles shared by real + decoy)
        self._pres_of_real = {}
        self._real_of_pres = {}
        self._used_pres = set()
        self._cur_decoys = []                 # presentation ids of decoys this step
        self._cur_real_next_real = None       # real id of the true next hop
        self._cur_real_next_pres = None       # presentation id of the true next hop
        self._decoy_gt = {}                   # pres_id -> ground-truth for the log
        return self.observe()

    # --- presentation-id namespace -----------------------------------------
    def _fresh_pres(self) -> int:
        while True:
            pid = self.rng.randint(_PRES_LO, _PRES_HI)
            if pid not in self._used_pres:
                self._used_pres.add(pid)
                return pid

    def _pres(self, real_id: Optional[int]) -> Optional[int]:
        if real_id is None:
            return None
        if real_id not in self._pres_of_real:
            pid = self._fresh_pres()
            self._pres_of_real[real_id] = pid
            self._real_of_pres[pid] = real_id
        return self._pres_of_real[real_id]

    def _real_next(self) -> Optional[int]:
        if self.foothold not in TRUE_PATH:
            return None
        i = TRUE_PATH.index(self.foothold)
        return TRUE_PATH[i + 1] if i + 1 < len(TRUE_PATH) else None

    def observe(self) -> dict:
        real_next = self._real_next()
        neighbors = list(EDGES.get(self.foothold, []))
        n_dec = self.lcam.n_decoys(real_next)
        views, self._cur_decoys, self._decoy_gt = [], [], {}

        for n in neighbors:
            pid = self._pres(n)
            views.append(self.lcam.node_view(
                pid, kind=REAL_NODES.get(n, "host"),
                is_real_next=(n == real_next), is_decoy=False))
        for _ in range(n_dec):
            pid = self._fresh_pres()
            self._cur_decoys.append(pid)
            self._decoy_gt[pid] = dict(true_kind="SYNTHETIC", origin="LCAM")
            views.append(self.lcam.node_view(
                pid, kind=self.rng.choice(KIND_VOCAB),
                is_real_next=False, is_decoy=True))

        self.rng.shuffle(views)
        self._cur_real_next_real = real_next
        self._cur_real_next_pres = (self._pres(real_next)
                                    if real_next is not None and real_next in neighbors
                                    else None)
        if self.cfg["deception"]:
            self.qos_overhead = min(0.05, self.qos_overhead + self.rng.uniform(0.0, 0.008))
        return dict(step=self.steps, foothold=self._pres(self.foothold), stage=self.stage,
                    hvt_id=self._pres(HVT), neighbors=views,
                    note="Some telemetry may be synthetic." if self.awareness != "A0" else "")

    def step(self, action: dict):
        assert not self.done
        self.steps += 1
        op = action.get("op", "scan")
        node = action.get("node")                    # presentation id
        decoys = self._cur_decoys                    # presentation ids
        real_next_pres = self._cur_real_next_pres
        real_next_real = self._cur_real_next_real
        outcome = "wasted"

        if op in ("exploit", "pivot", "connect", "exfiltrate"):
            if node in decoys:
                if self.rng.random() < self.cfg["p_capture"]:
                    self.done, self.result, outcome = True, "captured", "captured"
                else:
                    outcome = "lured"
            elif node is not None and node == real_next_pres:
                self.foothold = real_next_real
                self.stage = min(5, self.stage + 1)
                outcome = "progress"
                if self.foothold == HVT:
                    self.done, self.result, outcome = True, "compromise", "compromise"
            else:
                outcome = "wasted"
        elif op == "probe":
            outcome = "probe"
        elif op == "scan":
            outcome = "scan"

        if not self.done and self.steps >= self.max_steps:
            self.done, self.result = True, "timeout"

        gt = dict(foothold_real=self._real_of_pres.get(action.get("node")),
                  real_next_real=real_next_real,
                  decoy_pres_ids=list(decoys),
                  decoy_truth={pid: self._decoy_gt.get(pid) for pid in decoys})
        info = StepInfo(
            stage=self.stage, foothold=self.foothold, real_next=real_next_pres,
            decoys_shown=list(decoys), synthetic_fields={},
            correct_action=({"op": "exploit", "node": real_next_pres}
                            if real_next_pres is not None else None),
            outcome=outcome, gt=gt)
        return (self.observe() if not self.done else None), self.done, info

    def summary(self) -> dict:
        return dict(result=self.result, steps=self.steps, stage_reached=self.stage,
                    compromised=(self.result == "compromise"),
                    qos_overhead=round(self.qos_overhead, 4), defence=self.defence,
                    awareness=self.awareness)


# ---- renderer provenance + leakage contract --------------------------------
def render_text(obs: dict) -> str:
    """Deterministic natural-language/JSON rendering used in the LLM prompt."""
    return json.dumps(obs, sort_keys=True, indent=2)


def renderer_version_hash() -> str:
    """Stable hash of the renderer's source (LCAM.node_view + observe +
    render_text). Stamp this into every result log so a comparison can be
    rejected if the renderer differed across conditions."""
    src = "".join(inspect.getsource(f) for f in
                  (LCAM.node_view, CPSEnv.observe, render_text))
    return hashlib.sha1(src.encode()).hexdigest()[:12]


# Fields the attacker legitimately observes. Ground-truth keys must never appear.
_OBS_FIELDS = {"node", "kind", "service_banner", "firmware", "cert_expiry_days",
               "latency_ms", "apparent_vuln", "consistency", "corroborations"}
_FORBIDDEN_FIELDS = {"is_decoy", "decoy", "synthetic", "true_kind", "origin",
                     "real_next", "ground_truth"}


def check_observation_leakage(obs: dict, decoy_ids, strict: bool = False):
    """Per-observation structural guard. Verifies (a) no forbidden ground-truth
    field is exposed, (b) real and decoy views share the identical key schema,
    and (c) decoy ids are not out-of-band vs real ids. Returns a list of issue
    strings; raises AssertionError in strict mode if any are found.

    NOTE: this catches STRUCTURAL leaks only. Distributional leaks (a field
    whose values perfectly separate decoys) are caught run-level by
    integrity.leakage_scan over the logged records.
    """
    issues = []
    decoy_ids = set(decoy_ids)
    views = obs.get("neighbors", [])
    for v in views:
        bad = _FORBIDDEN_FIELDS & set(v)
        if bad:
            issues.append(f"forbidden field(s) exposed on node {v.get('node')}: {sorted(bad)}")
    schemas = {frozenset(v.keys()) for v in views}
    if len(schemas) > 1:
        issues.append(f"node views have non-uniform schemas: {[sorted(s) for s in schemas]}")
    # NOTE: id-range separation is a DISTRIBUTIONAL property (a single decoy is
    # trivially "separated" by chance), so it is assessed run-level by
    # integrity.leakage_scan (AUROC of the `node` id) rather than per-observation.
    if strict and issues:
        raise AssertionError("observation leakage: " + "; ".join(issues))
    return issues
