"""PromptGuard defense -- DeBERTa-v3 prompt injection classifier.

Default model is ``ProtectAI/deberta-v3-base-prompt-injection-v2``
(~86 M params, open, no gating), a DeBERTa-v3 sequence classifier
that labels text as ``INJECTION`` or ``SAFE``.  Can be swapped for
Meta's ``Llama-Prompt-Guard-2-86M`` via the ``PROMPT_GUARD2_MODEL``
env var (requires HuggingFace access grant).

Reference:
    ProtectAI -- deberta-v3-base-prompt-injection-v2
    https://huggingface.co/ProtectAI/deberta-v3-base-prompt-injection-v2

Integration note:
    512-token context window.  Long inputs are chunked; a single
    INJECTION/MALICIOUS chunk flags the whole input.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from aaps.defenses.base_defense import BaseDefense, DefenseResult, Severity

_MODEL_ID = os.getenv(
    "PROMPT_GUARD2_MODEL",
    "ProtectAI/deberta-v3-base-prompt-injection-v2",
)
_SCORE_THRESHOLD = float(os.getenv("PROMPT_GUARD2_THRESHOLD", "0.5"))
_MAX_CHUNK = 512


class PromptGuard2Defense(BaseDefense):
    """Binary injection classifier based on Llama Prompt Guard 2."""

    name = "PromptGuard2"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_id: str = _MODEL_ID,
        threshold: float = _SCORE_THRESHOLD,
    ):
        super().__init__(config)
        self.model_id = model_id
        self.threshold = threshold
        self._pipe = None
        self._available = True
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                "text-classification",
                model=self.model_id,
                device=-1,
                truncation=True,
                max_length=_MAX_CHUNK,
            )
        except Exception as exc:
            print(f"[PromptGuard2] model load failed ({exc}); defense disabled")
            self._available = False

    def _classify(self, text: str) -> tuple[bool, float, str]:
        """Return (is_malicious, score, label) for *text*."""
        if not self._available or self._pipe is None:
            return False, 0.0, "unavailable"

        chunks = [text[i:i + _MAX_CHUNK * 4]
                  for i in range(0, len(text), _MAX_CHUNK * 4)] or [text]

        max_score = 0.0
        worst_label = "BENIGN"
        for chunk in chunks:
            if not chunk.strip():
                continue
            out = self._pipe(chunk)[0]
            label = out["label"].upper()
            score = out["score"]
            if "MALICIOUS" in label or "INJECTION" in label:
                if score > max_score:
                    max_score = score
                    worst_label = label
        return max_score >= self.threshold, max_score, worst_label

    def check_input(
        self, user_input: str, context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        is_mal, score, label = self._classify(user_input)
        if is_mal:
            self.stats["blocked"] += 1
            return DefenseResult.hard_block(
                reason=f"PromptGuard2: {label} (score={score:.2f})",
                confidence=score,
                metadata={"label": label, "score": score},
            )
        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason=f"PromptGuard2: BENIGN ({score:.2f})",
            confidence=1.0 - score,
        )

    def check_retrieval(
        self, query: str, retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        flagged_indices: List[int] = []
        for idx, doc in enumerate(retrieved_docs):
            text = doc.get("content", doc.get("text", ""))
            is_mal, score, label = self._classify(text)
            if is_mal:
                flagged_indices.append(idx)

        if flagged_indices:
            clean = [d for i, d in enumerate(retrieved_docs)
                     if i not in flagged_indices]
            self.stats["blocked"] += 1
            return DefenseResult.soft_filter(
                reason=(f"PromptGuard2: flagged {len(flagged_indices)} "
                        f"doc(s) as injection"),
                sanitised_input=None,
                confidence=0.85,
                metadata={
                    "flagged_indices": flagged_indices,
                    "clean_docs": clean,
                },
            )
        self.stats["allowed"] += 1
        return DefenseResult.allow(reason="PromptGuard2: retrieval clean")

    def check_memory_write(
        self, content: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        is_mal, score, label = self._classify(content)
        if is_mal:
            self.stats["blocked"] += 1
            return DefenseResult.hard_block(
                reason=f"PromptGuard2: memory write blocked ({label}, {score:.2f})",
                confidence=score,
            )
        self.stats["allowed"] += 1
        return DefenseResult.allow(reason="PromptGuard2: memory write clean")
