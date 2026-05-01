"""Agent Integrity Stack (AIS) - layered defense for adaptive attacks.

The package exports six defense layers and an orchestrator pipeline.
Each layer is sourced from a specific paper (bibkeys below resolve in
``Overleaf/bibliography.bib``); per-module docstrings carry the full
arXiv URLs.

Layers
------
L1  ChannelSeparationDefense   - StruQ-inspired structured channels
                                 (``chen2024struq`` primary,
                                 ``hines2024spotlighting`` secondary)
L2  ToolOutputProbeDefense     - DataSentinel-inspired known-answer probe
                                 (Liu et al. 2025 -- pending bibkey,
                                 see bibliography_justification.md
                                 "Forecast" §7)
L3  ActionConsistencyDefense   - MELON-inspired masked re-execution
                                 (``zhu2025melon``)
L4  OutputConsistencyDefense   - TrustAgent + Circuit-Breaker style
                                 intent check (``yu2025trustagent`` +
                                 ``zou2024circuit``)
L5  MemoryWriteGuardDefense    - multi-signal A-MemGuard++
                                 (``wei2025amemguard``)
L6  RetrievalIntegrityDefense  - TrustRAG-inspired clustering + KB
                                 cross-check (``zhou2025trustrag``)

Cross-cutting
-------------
TraceLogger             - structured per-layer JSON decisions
AdaptiveFeedbackLearner - replays blocked attacks into L2/L5 detectors
AgentIntegrityStack     - orchestrator exposing the unified BaseDefense API
"""

from aaps.defenses.integrity.l1_channels import ChannelSeparationDefense
from aaps.defenses.integrity.l2_probe import ToolOutputProbeDefense
from aaps.defenses.integrity.l3_action_consistency import ActionConsistencyDefense
from aaps.defenses.integrity.l4_output_consistency import OutputConsistencyDefense
from aaps.defenses.integrity.l5_memory_guard import MemoryWriteGuardDefense
from aaps.defenses.integrity.l6_retrieval_guard import RetrievalIntegrityDefense
from aaps.defenses.integrity.trace_logger import TraceLogger
from aaps.defenses.integrity.adaptive_feedback import AdaptiveFeedbackLearner
from aaps.defenses.integrity.pipeline import AgentIntegrityStack

__all__ = [
    "ChannelSeparationDefense",
    "ToolOutputProbeDefense",
    "ActionConsistencyDefense",
    "OutputConsistencyDefense",
    "MemoryWriteGuardDefense",
    "RetrievalIntegrityDefense",
    "TraceLogger",
    "AdaptiveFeedbackLearner",
    "AgentIntegrityStack",
]
