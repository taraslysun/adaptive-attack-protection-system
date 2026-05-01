"""RAGuard: Perplexity and similarity-based RAG defense."""

from typing import Dict, Any, Optional, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from aaps.defenses.base_defense import BaseDefense, DefenseResult


class RAGuard(BaseDefense):
    """RAGuard defense using perplexity and similarity filtering."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        perplexity_threshold: float = 50.0,
        similarity_threshold: float = 0.3,
    ):
        """Initialize RAGuard."""
        super().__init__(config)
        self.perplexity_threshold = perplexity_threshold
        self.similarity_threshold = similarity_threshold

        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.perplexity_model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def check_memory_write(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        """Check memory write using perplexity."""
        self.stats["total_checks"] += 1

        perplexity = self._compute_perplexity(content)

        if perplexity > self.perplexity_threshold:
            # High perplexity indicates anomalous content
            self.stats["blocked"] += 1
            return DefenseResult(
                allowed=False,
                confidence=0.8,
                reason=f"High perplexity: {perplexity:.2f}",
                metadata={"perplexity": perplexity},
            )

        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            confidence=1.0 - (perplexity / self.perplexity_threshold),
            reason=f"Acceptable perplexity: {perplexity:.2f}",
            metadata={"perplexity": perplexity},
        )

    def check_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        """Check retrieved documents using perplexity and similarity."""
        self.stats["total_checks"] += 1

        if not retrieved_docs:
            self.stats["allowed"] += 1
            return DefenseResult(
                allowed=True,
                confidence=1.0,
                reason="No documents retrieved",
                metadata={},
            )

        # Thesis remediation ``codebase-fix-soft-filter-shape``: when at
        # least one document is filtered, we now emit a SOFT_FILTER
        # severity with ``kept_docs`` so the agent can replace the
        # retrieval bundle in place. The previous shape (``allowed=True``
        # + opaque ``filtered_count``) was effectively a no-op for
        # callers that respect the ``DefenseResult`` schema.
        kept_docs: List[Dict[str, Any]] = []
        suspicious_docs: List[Dict[str, Any]] = []
        for doc in retrieved_docs:
            doc_text = doc.get("text", "") or doc.get("content", "")

            perplexity = self._compute_perplexity(doc_text)
            if perplexity > self.perplexity_threshold:
                suspicious_docs.append(doc)
                continue

            similarity = self._compute_similarity(query, doc_text)
            if similarity < self.similarity_threshold:
                suspicious_docs.append(doc)
                continue

            kept_docs.append(doc)

        if suspicious_docs:
            self.stats["blocked"] += len(suspicious_docs)
            return DefenseResult.soft_filter(
                reason=f"Filtered {len(suspicious_docs)} suspicious documents",
                confidence=0.6,
                metadata={
                    "filtered_count": len(suspicious_docs),
                    "total_docs": len(retrieved_docs),
                    "kept_docs": kept_docs,
                    "filtered_docs": suspicious_docs,
                },
            )

        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason="All documents passed checks",
            confidence=1.0,
            metadata={"total_docs": len(retrieved_docs), "kept_docs": kept_docs},
        )

    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text."""
        if not text:
            return 0.0

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                outputs = self.perplexity_model(input_ids, labels=input_ids)
                loss = outputs.loss

            perplexity = torch.exp(loss).item()
            return perplexity
        except Exception:
            # Fallback: simple heuristic
            return 30.0  # Default moderate perplexity

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between texts."""
        if not text1 or not text2:
            return 0.0

        emb1 = self.encoder.encode(text1)
        emb2 = self.encoder.encode(text2)

        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )

        return float(similarity)
