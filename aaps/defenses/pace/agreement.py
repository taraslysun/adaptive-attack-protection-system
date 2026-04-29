"""Quorum voter -- requires q-of-K cluster agreement before a tool fires.

Given the K candidate fillings emitted by the per-cluster Executors,
the voter aggregates ``ProposedToolCall`` objects by their canonical
``(tool, args)`` form (see :func:`defenses.pace.plan.canonicalise_args`)
and only allows a tool call when at least ``q`` clusters proposed an
equivalent call.

This is the PACE enforcement of the quorum invariant from
``docs/design/spq.md`` §2.2. Theoretically and per Nasr et al. (2025,
arXiv:2510.09023, ``nasr2025attacker``) §"adaptive evaluation",
quorum is the minimal change that turns single-injection attacks
("plant one poisoned doc -> one tool call") into a coordinated-injection
problem ("plant ``q`` semantically distinct payloads that map to ``q``
clusters and all justify the same call"), which is strictly harder.

The aggregation is intentionally pure-Python so the voter can run as
part of any harness and the JSON trace is faithfully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from aaps.defenses.pace.plan import ProposedToolCall, canonicalise_args


@dataclass
class AgreementDecision:
    """One aggregate decision over a (tool, args) canonical key.

    A decision is *fire* when the number of supporting clusters is
    ``>= q`` AND the call satisfies the CFI gate (the gate is applied
    by :class:`defenses.pace.pipeline.PACEDefense` before voting; the
    voter only enforces quorum).
    """

    tool: str
    args: Dict[str, Any]
    args_canonical: str
    supporting_clusters: List[int]
    dissenting_clusters: List[int]
    K: int
    q: int

    @property
    def agreement(self) -> int:
        return len(self.supporting_clusters)

    @property
    def margin(self) -> float:
        if self.K <= 0:
            return 0.0
        return self.agreement / self.K

    @property
    def decision(self) -> str:
        return "fire" if self.agreement >= self.q else "abstain"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "args": dict(self.args),
            "args_canonical": self.args_canonical,
            "supporting_clusters": list(self.supporting_clusters),
            "dissenting_clusters": list(self.dissenting_clusters),
            "agreement": self.agreement,
            "K": self.K,
            "q": self.q,
            "margin": self.margin,
            "decision": self.decision,
        }


class AgreementVoter:
    """Aggregates per-cluster ``ProposedToolCall`` lists into decisions."""

    def __init__(self, K: int, q: int, tool_registry: Dict[str, Any] | None = None) -> None:
        if K <= 0:
            raise ValueError("K must be positive")
        if q <= 0 or q > K:
            raise ValueError("q must be in [1, K]")
        self.K = K
        self.q = q
        # tool_registry retained for backward-compat; no longer used post capability-shim removal.
        self._tool_registry: Dict[str, Any] = tool_registry or {}

    def vote(
        self,
        per_cluster_calls: Sequence[Sequence[ProposedToolCall]],
    ) -> List[AgreementDecision]:
        """Return a deduplicated list of decisions, one per canonical call."""
        if len(per_cluster_calls) != self.K:
            raise ValueError(
                f"per_cluster_calls has {len(per_cluster_calls)} entries; expected K={self.K}"
            )
        index: Dict[str, Dict[str, Any]] = {}
        for cid, calls in enumerate(per_cluster_calls):
            seen_in_cluster: set[str] = set()
            for call in calls:
                key = call.args_canonical
                if key in seen_in_cluster:
                    continue
                seen_in_cluster.add(key)
                slot = index.setdefault(
                    key,
                    {
                        "tool": call.tool,
                        "args": dict(call.args),
                        "supporting": [],
                    },
                )
                slot["supporting"].append(cid)
        decisions: List[AgreementDecision] = []
        for key, slot in index.items():
            supporting = sorted(set(slot["supporting"]))
            dissenting = [c for c in range(self.K) if c not in supporting]
            decisions.append(
                AgreementDecision(
                    tool=slot["tool"],
                    args=slot["args"],
                    args_canonical=key,
                    supporting_clusters=supporting,
                    dissenting_clusters=dissenting,
                    K=self.K,
                    q=self.q,
                )
            )
        decisions.sort(key=lambda d: (-d.agreement, d.tool, d.args_canonical))
        return decisions
