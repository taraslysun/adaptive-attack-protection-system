"""Base classes and result types for the defense framework.

This is the unified interface used by every defense in the project.

Backward compatibility
----------------------
The legacy ``DefenseResult(allowed, confidence, reason, metadata)`` call
form still works because the new fields are all keyword-only and have
defaults.  Legacy defenses that only override ``check_memory_write`` and
``check_retrieval`` continue to work since these are now optional concrete
methods (default = ALLOW) instead of ``@abstractmethod``.

The five canonical hooks all accept the standardised inputs documented
below and all return a :class:`DefenseResult`.  Agents may probe a
defense for any subset of these hooks via ``hasattr`` and call only the
ones that are present, but if the defense subclasses :class:`BaseDefense`
all five hooks are guaranteed to exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    """Outcome severity emitted by a defense check.

    ``ALLOW``        - the check passed; the call site should proceed.
    ``SOFT_FILTER``  - the call site should still proceed but the result
                       contains a sanitised replacement (e.g. retrieval
                       with poisoned documents stripped).
    ``HARD_BLOCK``   - the call site MUST stop; downstream layers and the
                       agent should not see the original input/output.
    """

    ALLOW = "allow"
    SOFT_FILTER = "soft_filter"
    HARD_BLOCK = "hard_block"


@dataclass
class DefenseResult:
    """Canonical result emitted by every defense hook.

    The four legacy fields (``allowed``, ``confidence``, ``reason``,
    ``metadata``) keep their original meaning.  The four new fields are
    optional and all default to a backward-compatible value.

    Parameters
    ----------
    allowed:
        ``True`` if the layer permits the call to continue.  Composes
        with ``severity``: ``allowed=True`` + ``SOFT_FILTER`` means
        proceed but with a sanitised replacement.
    confidence:
        Score in ``[0, 1]`` describing how sure the layer is in its
        decision.  Used by the trace logger and adaptive feedback loop.
    reason:
        Short human-readable summary (one line, used in dashboards).
    metadata:
        Arbitrary structured payload (per-layer signals, similarities,
        timing, model names, etc.).  Always a dict; never ``None``.
    severity:
        See :class:`Severity`.  When omitted, derived from ``allowed``.
    sanitised_input:
        If a layer rewrites the user/tool input (e.g. wrapping it in
        delimiter markers), the rewritten text goes here.  ``None`` if
        no rewrite was performed.
    sanitised_output:
        Same as ``sanitised_input`` but for the agent response.
    rationale:
        Multi-line, optionally LLM-authored explanation for an audit
        trail.  Falls back to ``reason`` when omitted.
    layer:
        Optional layer identifier (``"L1"`` ... ``"L6"``).  Used by the
        Agent Integrity Stack trace logger; legacy single-defenses can
        leave it as ``None``.
    latency_ms:
        Wall-clock time the layer spent producing this result, in
        milliseconds.  Populated by the orchestrator when available.
    """

    allowed: bool
    confidence: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    severity: Optional[Severity] = None
    sanitised_input: Optional[str] = None
    sanitised_output: Optional[str] = None
    rationale: Optional[str] = None
    layer: Optional[str] = None
    latency_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.severity is None:
            self.severity = Severity.ALLOW if self.allowed else Severity.HARD_BLOCK
        if self.rationale is None:
            self.rationale = self.reason
        if self.layer is None:
            self.layer = self.metadata.get("layer")

    @classmethod
    def allow(
        cls,
        reason: str = "ok",
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> "DefenseResult":
        """Convenience constructor for an ALLOW result."""
        return cls(
            allowed=True,
            confidence=confidence,
            reason=reason,
            severity=Severity.ALLOW,
            **kwargs,
        )

    @classmethod
    def hard_block(
        cls,
        reason: str,
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> "DefenseResult":
        """Convenience constructor for a HARD_BLOCK result."""
        return cls(
            allowed=False,
            confidence=confidence,
            reason=reason,
            severity=Severity.HARD_BLOCK,
            **kwargs,
        )

    @classmethod
    def soft_filter(
        cls,
        reason: str,
        sanitised_input: Optional[str] = None,
        sanitised_output: Optional[str] = None,
        confidence: float = 0.7,
        **kwargs: Any,
    ) -> "DefenseResult":
        """Convenience constructor for a SOFT_FILTER result."""
        return cls(
            allowed=True,
            confidence=confidence,
            reason=reason,
            severity=Severity.SOFT_FILTER,
            sanitised_input=sanitised_input,
            sanitised_output=sanitised_output,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable view used by the trace logger."""
        return {
            "allowed": self.allowed,
            "confidence": self.confidence,
            "reason": self.reason,
            "severity": self.severity.value if self.severity else None,
            "layer": self.layer,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "sanitised_input": self.sanitised_input,
            "sanitised_output": self.sanitised_output,
            "rationale": self.rationale,
        }


class BaseDefense:
    """Common base class for every defense.

    Every hook is an *optional* concrete method whose default behaviour
    is to ALLOW.  Subclasses override the hooks they care about; agents
    can probe with ``hasattr`` for the legacy duck-typed flow, or call
    the hook unconditionally because the default exists here.

    Concrete subclasses must call ``super().__init__(config)``.
    """

    name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self.stats: Dict[str, int] = {
            "total_checks": 0,
            "blocked": 0,
            "allowed": 0,
            "false_positives": 0,
        }

    # ------------------------------------------------------------------
    # The five canonical hooks.  Subclasses override what they care about.
    # ------------------------------------------------------------------

    def check_input(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        """Inspect the raw user query before the agent sees it."""
        self.stats["total_checks"] += 1
        self.stats["allowed"] += 1
        return DefenseResult.allow(reason=f"{self.name}: check_input default ALLOW")

    def check_output(
        self,
        user_query: str,
        agent_response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        """Inspect the final agent response before it reaches the user."""
        self.stats["total_checks"] += 1
        self.stats["allowed"] += 1
        return DefenseResult.allow(reason=f"{self.name}: check_output default ALLOW")

    def check_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_intent: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        """Inspect a single tool call before execution."""
        self.stats["total_checks"] += 1
        self.stats["allowed"] += 1
        return DefenseResult.allow(reason=f"{self.name}: check_tool_call default ALLOW")

    def check_tool_calls(
        self,
        user_prompt: str,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        """Inspect a batch of tool calls.

        Default behaviour: iterate ``check_tool_call`` and HARD_BLOCK on
        the first failing call; this lets a defense override only the
        singular hook and still get correct batch behaviour.
        """
        if not tool_calls:
            return DefenseResult.allow(reason=f"{self.name}: no tool calls to inspect")
        for tc in tool_calls:
            res = self.check_tool_call(
                tool_name=tc.get("name", ""),
                tool_args=tc.get("args", {}) or tc.get("arguments", {}),
                user_intent=user_prompt,
                context=context,
            )
            if not res.allowed:
                return res
        return DefenseResult.allow(reason=f"{self.name}: tool batch passed")

    def check_memory_write(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        """Inspect a candidate persistent-memory write."""
        self.stats["total_checks"] += 1
        self.stats["allowed"] += 1
        return DefenseResult.allow(reason=f"{self.name}: check_memory_write default ALLOW")

    def check_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        """Inspect documents returned from a retrieval store."""
        self.stats["total_checks"] += 1
        self.stats["allowed"] += 1
        return DefenseResult.allow(reason=f"{self.name}: check_retrieval default ALLOW")

    # ------------------------------------------------------------------
    # Statistics helpers.
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return dict(self.stats)

    def reset_stats(self) -> None:
        self.stats = {
            "total_checks": 0,
            "blocked": 0,
            "allowed": 0,
            "false_positives": 0,
        }
