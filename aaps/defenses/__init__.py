"""Defense implementations against adaptive attacks.

Public API
----------
* The thesis contribution lives in :mod:`defenses.integrity` and exposes
  the :class:`AgentIntegrityStack` (AIS) plus its six layers.
* The single-paper baselines live in :mod:`defenses.baselines` and are
  re-exported here so the comparison matrix can keep using
  ``from defenses import PromptGuard2Defense, MELONDefense, ...``.
* Earlier multi-layer experiments (``AdaptiveDefensePipeline`` and
  ``AM2IFramework``) are archived in :mod:`defenses._legacy`.

Baseline coverage (one defense per published paper). Bibkeys without a
``%`` prefix already resolve in ``Overleaf/bibliography.bib``; entries
prefixed with ``% TODO[bib]:`` are tracked in
``docs/bibliography_justification.md`` "Forecast: expected
``% TODO[bib]:`` placeholders". Most of these baselines were stress-tested
under the umbrella adaptive-attack methodology of Nasr *et al.*, "The
Attacker Moves Second" (arXiv:2510.09023, bibkey ``nasr2025attacker``).

  1. Prompting defenses:    Spotlighting (``hines2024spotlighting``),
                            PromptSandwiching (folk technique; bibkey
                            ``% TODO[bib]: prompt-sandwiching``),
                            RPO (``zhou2024rpo``).
  2. Training-based:        CircuitBreakers (``zou2024circuit``),
                            StruQ (``chen2024struq``),
                            SecAlign (Chen 2024 -- see bibkey
                            ``% TODO[bib]: chen2024secalign``).
  3. Filtering:             PromptGuard / PromptGuard2 (``inan2023llamaguard``
                            placeholder),
                            WildGuard (``han2024wildguard``),
                            LlamaFirewall (``meta2024llamafirewall``;
                            bibkey ``% TODO[bib]:`` if absent),
                            RAGuard (TrustRAG-aligned;
                            ``zhou2025trustrag``),
                            SmoothLLM (``robey2023smoothllm``).
  4. Secret-knowledge:      DataSentinel (Liu 2025 -- bibkey
                            ``% TODO[bib]: liu2025datasentinel``),
                            MELON (``zhu2025melon``).
  5. Memory:                A-MemGuard (``wei2025amemguard``).
"""

from aaps.defenses.base_defense import BaseDefense, DefenseResult, Severity
from aaps.defenses.baselines import (
    AMemGuard,
    CircuitBreakerDefense,
    DataSentinelDefense,
    LlamaFirewall,
    MELONDefense,
    PromptGuard2Defense,
    PromptGuardFilter,
    PromptSandwiching,
    RAGuard,
    RPODefense,
    SecAlignDefense,
    SmoothLLMDefense,
    Spotlighting,
    StruQDefense,
    WildGuardDefense,
)
from aaps.defenses.integrity import (
    ActionConsistencyDefense,
    AdaptiveFeedbackLearner,
    AgentIntegrityStack,
    ChannelSeparationDefense,
    MemoryWriteGuardDefense,
    OutputConsistencyDefense,
    RetrievalIntegrityDefense,
    ToolOutputProbeDefense,
    TraceLogger,
)

__all__ = [
    "BaseDefense",
    "DefenseResult",
    "Severity",
    "Spotlighting",
    "PromptSandwiching",
    "RPODefense",
    "StruQDefense",
    "CircuitBreakerDefense",
    "PromptGuardFilter",
    "PromptGuard2Defense",
    "WildGuardDefense",
    "RAGuard",
    "LlamaFirewall",
    "AMemGuard",
    "DataSentinelDefense",
    "MELONDefense",
    "SmoothLLMDefense",
    "SecAlignDefense",
    "AgentIntegrityStack",
    "ChannelSeparationDefense",
    "ToolOutputProbeDefense",
    "ActionConsistencyDefense",
    "OutputConsistencyDefense",
    "MemoryWriteGuardDefense",
    "RetrievalIntegrityDefense",
    "TraceLogger",
    "AdaptiveFeedbackLearner",
]
