"""Constitutional Classifiers defence — academic proxy.

Reference:
    Sharma et al., "Constitutional Classifiers: Defending Against Universal
    Jailbreaks across Thousands of Hours of Red Teaming", arXiv:2501.18837
    (Anthropic, 2025).

NOTE on faithfulness:
    Anthropic's own classifier weights are closed and trained on a
    proprietary constitution-derived synthetic dataset. This class is an
    *academic proxy* implementing the paper's described architecture
    (input + output classifier heads + a constitution rule list). The
    default classifier is ``unitary/toxic-bert`` augmented with a
    keyword/regex constitution. Documented limitations:

    * Toxic-bert is a coarser classifier than Anthropic's; expected to
      under-detect subtle policy violations.
    * Constitution is a small starter list; production deployments should
      replace it with a fully populated rule set.
    * No fine-tuning step is included; users training their own classifier
      should subclass and replace ``self._scorer``.

Cite as inspiration of mechanism, not a reproduction of Anthropic's
weights.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from aaps.defenses.base_defense import BaseDefense, DefenseResult

log = logging.getLogger("aaps.defenses.baselines.constitutional_classifiers")


_DEFAULT_HF_MODEL = os.environ.get(
    "CONSTITUTIONAL_HF_MODEL", "unitary/toxic-bert"
)


# Starter constitution. One regex per rule; a hit adds 1.0 to the score.
DEFAULT_CONSTITUTION: List[tuple[str, str]] = [
    ("no_dangerous_synthesis", r"\b(synthes(is|ize)|how\s+to\s+make).+(weapon|bomb|virus|toxin|chemical)\b"),
    ("no_self_harm_methods", r"\b(suicide|self.?harm)\s+(method|how)"),
    ("no_minor_sexual", r"\b(minor|child).*\b(sexual|porn|nude)"),
    ("no_credential_exfil", r"\b(exfiltr|leak|steal)\b.{0,40}\b(password|api.?key|token|credential)\b"),
    ("no_malware_authoring", r"\b(write|create|develop)\b.{0,30}\b(malware|ransomware|keylogger|rootkit)\b"),
    ("no_election_disinfo", r"\b(election|ballot|voter).{0,40}\b(rigged|fraud|fake)\b"),
]


class ConstitutionalClassifiersDefense(BaseDefense):
    """Constitutional Classifiers (academic proxy) — input + output guard."""

    name = "constitutional_classifiers"

    def __init__(
        self,
        hf_model: Optional[str] = None,
        constitution: Optional[List[tuple[str, str]]] = None,
        toxic_threshold: float = 0.7,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config)
        self.hf_model = hf_model or _DEFAULT_HF_MODEL
        self.constitution = constitution or list(DEFAULT_CONSTITUTION)
        self.toxic_threshold = toxic_threshold
        self._scorer = None
        self._available = self._probe()
        if not self._available:
            log.warning(
                "ConstitutionalClassifiersDefense: classifier model unavailable. "
                "Falling back to constitution-only (regex) checks."
            )

    def _probe(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_scorer(self):
        if self._scorer is None and self._available:
            try:
                from transformers import pipeline
                self._scorer = pipeline(
                    "text-classification",
                    model=self.hf_model,
                    truncation=True,
                    top_k=None,
                )
            except Exception as e:
                log.warning("Could not load classifier %s: %s", self.hf_model, e)
                self._available = False

    def _constitution_violations(self, text: str) -> List[str]:
        hits = []
        for name, pattern in self.constitution:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    hits.append(name)
            except re.error:
                continue
        return hits

    def _classifier_score(self, text: str) -> float:
        self._ensure_scorer()
        if not self._available or self._scorer is None:
            return 0.0
        try:
            out = self._scorer(text)
            # `pipeline(top_k=None)` returns list[list[dict]] for batched calls
            # or list[dict] for single string.
            scores = out[0] if isinstance(out, list) and out and isinstance(out[0], list) else out
            if not isinstance(scores, list):
                return 0.0
            mx = 0.0
            for s in scores:
                if not isinstance(s, dict):
                    continue
                label = s.get("label", "").lower()
                score = float(s.get("score", 0.0))
                if "toxic" in label or "hate" in label or "obscen" in label:
                    mx = max(mx, score)
            return mx
        except Exception as e:
            log.warning("classifier call failed: %s", e)
            return 0.0

    def _evaluate(self, text: str) -> tuple[bool, Dict[str, Any]]:
        hits = self._constitution_violations(text)
        score = self._classifier_score(text)
        unsafe = bool(hits) or score >= self.toxic_threshold
        return unsafe, {
            "constitution_hits": hits,
            "classifier_score": score,
            "threshold": self.toxic_threshold,
        }

    def check_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> DefenseResult:
        unsafe, meta = self._evaluate(user_input)
        return DefenseResult(confidence=1.0,
            allowed=not unsafe,
            severity="high" if unsafe else "low",
            reason=("constitutional: " + (",".join(meta["constitution_hits"])
                                          or f"score={meta['classifier_score']:.2f}")),
            metadata=meta,
        )

    def check_output(self, assistant_output: str, context: Optional[Dict[str, Any]] = None) -> DefenseResult:
        unsafe, meta = self._evaluate(assistant_output)
        return DefenseResult(confidence=1.0,
            allowed=not unsafe,
            severity="high" if unsafe else "low",
            reason=("constitutional: " + (",".join(meta["constitution_hits"])
                                          or f"score={meta['classifier_score']:.2f}")),
            metadata=meta,
        )

    def check_retrieval(self, *args, **kwargs) -> DefenseResult:
        return DefenseResult(confidence=1.0, allowed=True, severity="info", reason="constitutional: pass-through")

    def check_memory_write(self, *args, **kwargs) -> DefenseResult:
        return DefenseResult(confidence=1.0, allowed=True, severity="info", reason="constitutional: pass-through")

    def check_tool_call(self, *args, **kwargs) -> DefenseResult:
        return DefenseResult(confidence=1.0, allowed=True, severity="info", reason="constitutional: pass-through")
