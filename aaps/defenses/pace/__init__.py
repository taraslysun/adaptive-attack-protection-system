"""Plan, Agree, Confirm, Execute (PACE) — architectural defense for LLM agents.

PACE is the thesis' main contribution after the AIS layer-stack redesign
(see ``docs/design/pace.md`` for the full design and ``docs/wiki/defenses/pace/``
for the user-facing wiki).

It splits an agent into:

* a **Planner** LLM that sees only trusted control-plane input
  (user request, tool schemas, system prompt) and emits a typed
  PACEPlan -- a directed graph of allowed tool calls;
* a **per-cluster Executor** LLM that fills the PACEPlan from a
  k-means partitioned evidence pool;
* a **CFI gate** that rejects any tool call not present in the
  PACEPlan;
* an **Agreement gate** that fires a tool call only when at least ``q`` of
  ``K`` cluster-Executors agree on it.

The defense is content-agnostic: enforcement is purely structural
(plan match + executor agreement). An optional NLI redundancy filter
on retrieved evidence is the only content-aware component.

Anchor papers (cited in the per-module docstrings):

* CaMeL (Debenedetti et al. 2025, arXiv:2503.18813) -- ceiling reference.
* IsolateGPT (Wu et al. 2025) -- planner / executor split.
* Spotlighting (Hines et al. 2024, arXiv:2403.14647) -- delimiter idea.
* TrustRAG (Zhou et al. 2025) -- k-means evidence partition.
* Self-Consistency (Wang et al. 2022, arXiv:2203.11171) and Multi-Agent
  Debate (Du et al. 2023, arXiv:2305.14325) -- voting under disagreement.
* Tramèr et al. (2020) and Nasr et al. (2025, arXiv:2510.09023) --
  adaptive-attack methodology.
"""

from aaps.defenses.pace.plan import (
    PACEPlan,
    PACEPlanNode,
    ProposedToolCall,
    canonicalise_args,
)
from aaps.defenses.pace.pipeline import PACEDefense
from aaps.defenses.pace.planner import Planner, PACEPlanner
from aaps.defenses.pace.executor import Executor
from aaps.defenses.pace.clusters import EvidenceSpan, EvidencePool, kmeans_cluster
from aaps.defenses.pace.agreement import AgreementVoter, AgreementDecision
from aaps.defenses.pace.trace_logger import PACETraceLogger

__all__ = [
    "PACEDefense",
    "PACEPlan",
    "PACEPlanNode",
    "ProposedToolCall",
    "canonicalise_args",
    "Planner",
    "PACEPlanner",
    "Executor",
    "EvidenceSpan",
    "EvidencePool",
    "kmeans_cluster",
    "AgreementVoter",
    "AgreementDecision",
    "PACETraceLogger",
]
