"""Reproduced single-paper baselines used in the comparison matrix.

Each baseline below is one paper; bibkeys resolve in
``Overleaf/bibliography.bib`` (entries prefixed with ``% TODO[bib]:``
are tracked in ``docs/bibliography_justification.md`` "Forecast"). The
shared umbrella adaptive-attack methodology is Nasr *et al.*, "The
Attacker Moves Second" (arXiv:2510.09023, bibkey ``nasr2025attacker``).

* Prompting:        Spotlighting (``hines2024spotlighting``),
                    PromptSandwiching (``% TODO[bib]: prompt-sandwiching``),
                    RPO (``zhou2024rpo``).
* Training-based:   StruQ (``chen2024struq``),
                    CircuitBreakers (``zou2024circuit``),
                    SecAlign (``% TODO[bib]: chen2024secalign``).
* Filtering:        PromptGuard / PromptGuard2 (``inan2023llamaguard``
                    placeholder), LlamaFirewall
                    (``% TODO[bib]: meta2024llamafirewall``),
                    WildGuard (``han2024wildguard``),
                    RAGuard (``zhou2025trustrag``-aligned),
                    SmoothLLM (``robey2023smoothllm``).
* Secret-knowledge: DataSentinel (``% TODO[bib]: liu2025datasentinel``),
                    MELON (``zhu2025melon``).
* Memory:           A-MemGuard (``wei2025amemguard``).

These exist as orthogonal references to the thesis contribution
(:mod:`defenses.integrity`).  Each module loads its third-party
dependency (transformers, torch, etc.) lazily so importing
``defenses`` works without GPU or HuggingFace tokens.
"""

from aaps.defenses.baselines.prompt_guards import PromptSandwiching, Spotlighting
from aaps.defenses.baselines.rpo import RPODefense
from aaps.defenses.baselines.struq import StruQDefense
from aaps.defenses.baselines.data_sentinel import DataSentinelDefense
from aaps.defenses.baselines.melon import MELONDefense

try:
    from aaps.defenses.baselines.circuit_breaker import CircuitBreakerDefense
except Exception:
    CircuitBreakerDefense = None

try:
    from aaps.defenses.baselines.prompt_guard_filter import PromptGuardFilter
except Exception:
    PromptGuardFilter = None

try:
    from aaps.defenses.baselines.prompt_guard2 import PromptGuard2Defense
except Exception:
    PromptGuard2Defense = None

try:
    from aaps.defenses.baselines.wildguard_defense import WildGuardDefense
except Exception:
    WildGuardDefense = None

try:
    from aaps.defenses.baselines.rag_guard import RAGuard
except Exception:
    RAGuard = None

try:
    from aaps.defenses.baselines.firewall import LlamaFirewall
except Exception:
    LlamaFirewall = None

try:
    from aaps.defenses.baselines.a_memguard import AMemGuard
except Exception:
    AMemGuard = None

try:
    from aaps.defenses.baselines.smoothllm import SmoothLLMDefense
except Exception:
    SmoothLLMDefense = None

try:
    from aaps.defenses.baselines.secalign import SecAlignDefense
except Exception:
    SecAlignDefense = None

__all__ = [
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
]
