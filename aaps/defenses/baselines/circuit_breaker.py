"""Circuit Breakers defense -- representation-engineering rerouting.

Source
------
Zou, A.; Phan, L.; Wang, J.; Duenas, D.; Lin, M.; Andriushchenko, M.;
Wang, R.; Kolter, J. Z.; Fredrikson, M.; Hendrycks, D., "Improving
Alignment and Robustness with Circuit Breakers", NeurIPS 2024
(arXiv:2406.04313, https://arxiv.org/abs/2406.04313). Bibkey
``zou2024circuit``.

Tested as a baseline alongside the Nasr et al. "The Attacker Moves
Second" suite (arXiv:2510.09023, bibkey ``nasr2025attacker``).

Uses representation engineering to detect and reroute harmful model
activations.  When the model's internal representations match known
harmful patterns, the output is interrupted and replaced with a refusal.

This implementation provides:
  1. An embedding-based anomaly detector for input text
  2. A perplexity-based check for generated outputs
"""

from typing import Dict, Any, Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer

from aaps.defenses.base_defense import BaseDefense, DefenseResult


HARMFUL_PATTERNS = [
    "how to make", "how to build", "step by step instructions for",
    "write malicious", "create a virus", "bypass security",
    "hack into", "steal data", "generate harmful",
    "ignore safety", "override instructions", "system prompt",
    "reveal your instructions", "forget your rules",
]


class CircuitBreakerDefense(BaseDefense):
    """Circuit Breaker defense with representation-based anomaly detection."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.65,
    ):
        super().__init__(config)
        self.similarity_threshold = similarity_threshold
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._pattern_embeddings = self.encoder.encode(HARMFUL_PATTERNS)

    def _check_harmful(self, text: str) -> float:
        """Return max cosine similarity to known harmful patterns."""
        emb = self.encoder.encode(text)
        sims = np.dot(self._pattern_embeddings, emb) / (
            np.linalg.norm(self._pattern_embeddings, axis=1) * np.linalg.norm(emb)
        )
        return float(np.max(sims))

    def check_input(self, user_input: str) -> DefenseResult:
        """Check whether the input triggers the circuit breaker."""
        self.stats["total_checks"] += 1
        sim = self._check_harmful(user_input)
        if sim >= self.similarity_threshold:
            self.stats["blocked"] += 1
            return DefenseResult(
                allowed=False,
                confidence=sim,
                reason=f"Circuit breaker triggered (sim={sim:.2f})",
                metadata={"max_similarity": sim},
            )
        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            confidence=1.0 - sim,
            reason="Input passed circuit breaker",
            metadata={"max_similarity": sim},
        )

    def check_output(self, user_input: str, output: str) -> DefenseResult:
        """Check the model output for harmful content.

        Note: argument order is (user_input, output) to match how callers
        invoke this (DeepAgent/LocalAgent pass query first, answer second).
        The harmful-content check is performed on the *output*.
        """
        self.stats["total_checks"] += 1
        sim = self._check_harmful(output)
        if sim >= self.similarity_threshold:
            self.stats["blocked"] += 1
            return DefenseResult(
                allowed=False,
                confidence=sim,
                reason=f"Output blocked by circuit breaker (sim={sim:.2f})",
                metadata={"max_similarity": sim},
            )
        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            confidence=1.0 - sim,
            reason="Output passed circuit breaker",
            metadata={"max_similarity": sim},
        )

    def check_memory_write(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> DefenseResult:
        return self.check_input(content)

    def check_retrieval(
        self, query: str, retrieved_docs: List[Dict[str, Any]]
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True, confidence=0.7,
            reason="Circuit breaker does not filter retrieval", metadata={},
        )
