"""PromptGuard-style filtering defense -- DeBERTa-style classifier filter.

Source
------
Inan, H. *et al.*, "Llama Guard: LLM-based Input-Output Safeguard for
Human-AI Conversations", arXiv:2312.06674 (2023,
https://arxiv.org/abs/2312.06674) -- the umbrella Meta safeguard line
that includes the PromptGuard 2 classifier
(https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M). Bibkey
``inan2023llamaguard`` (placeholder for the published PromptGuard 2
model card; see ``docs/bibliography_justification.md`` "Forecast" §6).

Tested as a baseline alongside the Nasr et al. "The Attacker Moves
Second" suite (arXiv:2510.09023, bibkey ``nasr2025attacker``).

Fine-tuned classifier (DeBERTa-based) that detects prompt injection
attempts in the input text.  If the classifier flags the input as
an injection, the request is blocked.

This implementation uses a lightweight heuristic classifier as a
stand-in; swap with the real ProtectAI/deberta-v3-base-prompt-injection
model for production use.
"""

from typing import Dict, Any, Optional, List
import re

from aaps.defenses.base_defense import BaseDefense, DefenseResult


INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+(instructions?|prompts?|rules?)",
    r"(system|admin)\s*(override|prompt|message|instruction)",
    r"you\s+are\s+now\s+",
    r"(forget|disregard|bypass)\s+(your|all|safety)\s+(rules?|instructions?|guidelines?)",
    r"(new|updated?)\s+(directive|instruction|behavior|policy)",
    r"debug\s+mode",
    r"\[\s*(INST|SYSTEM|INTERNAL)\s*\]",
    r"<<<\s*(END|TRUSTED|SYSTEM)",
    r"---\s*END\s*(DATA|DOCUMENT|INPUT)",
    r"prerequisite\s+(step|workflow|task)",
    r"(step\s*0|mandatory\s+prerequisite)",
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


class PromptGuardFilter(BaseDefense):
    """Rule-based + heuristic prompt injection filter."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        pattern_threshold: int = 1,
        use_ml_model: bool = False,
    ):
        super().__init__(config)
        self.pattern_threshold = pattern_threshold
        self.use_ml_model = use_ml_model
        self._ml_model = None

    def _pattern_score(self, text: str) -> int:
        return sum(1 for p in COMPILED if p.search(text))

    def _ml_score(self, text: str) -> float:
        """Optional ML-based scoring using a HuggingFace classifier."""
        if self._ml_model is None:
            try:
                from transformers import pipeline
                self._ml_model = pipeline(
                    "text-classification",
                    model="ProtectAI/deberta-v3-base-prompt-injection-v2",
                    truncation=True,
                    max_length=512,
                )
            except Exception:
                self._ml_model = lambda x: [{"label": "SAFE", "score": 0.9}]

        result = self._ml_model(text[:512])
        if isinstance(result, list) and result:
            r = result[0]
            if r.get("label", "").upper() in ("INJECTION", "UNSAFE", "1"):
                return r.get("score", 0.5)
        return 0.0

    def check_input(self, user_input: str) -> DefenseResult:
        self.stats["total_checks"] += 1
        hits = self._pattern_score(user_input)

        if self.use_ml_model:
            ml_score = self._ml_score(user_input)
            if ml_score >= 0.7:
                self.stats["blocked"] += 1
                return DefenseResult(
                    allowed=False,
                    confidence=ml_score,
                    reason=f"ML classifier flagged injection (score={ml_score:.2f})",
                    metadata={"pattern_hits": hits, "ml_score": ml_score},
                )

        if hits >= self.pattern_threshold:
            self.stats["blocked"] += 1
            return DefenseResult(
                allowed=False,
                confidence=min(hits / 5.0, 1.0),
                reason=f"Pattern-based filter: {hits} injection pattern(s) detected",
                metadata={"pattern_hits": hits},
            )

        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            confidence=1.0 - hits / 10.0,
            reason="Input passed prompt guard filter",
            metadata={"pattern_hits": hits},
        )

    def check_memory_write(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> DefenseResult:
        return self.check_input(content)

    def check_retrieval(
        self, query: str, retrieved_docs: List[Dict[str, Any]]
    ) -> DefenseResult:
        """Drop retrieved docs that match the injection pattern bank.

        Thesis remediation ``codebase-fix-soft-filter-shape``: when at
        least one document is filtered, the agent expects a
        ``Severity.SOFT_FILTER`` result with a ``kept_docs`` list in the
        metadata so it can replace the input retrieval bundle in place.
        Returning ``allowed=True`` with no severity, no ``kept_docs``,
        and only a count was effectively a no-op for callers using the
        ``DefenseResult`` schema correctly.
        """
        self.stats["total_checks"] += 1
        kept: List[Dict[str, Any]] = []
        flagged: List[Dict[str, Any]] = []
        for doc in retrieved_docs:
            text = doc.get("text", "") or doc.get("content", "")
            if self._pattern_score(text) >= self.pattern_threshold:
                flagged.append(doc)
            else:
                kept.append(doc)

        if flagged:
            self.stats["blocked"] += len(flagged)
            return DefenseResult.soft_filter(
                reason=f"Filtered {len(flagged)} suspicious retrieved docs",
                confidence=0.6,
                metadata={
                    "filtered_count": len(flagged),
                    "kept_docs": kept,
                    "filtered_docs": flagged,
                },
            )

        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason="All retrieved docs passed filter",
            confidence=0.9,
            metadata={"kept_docs": kept},
        )
