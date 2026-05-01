"""A-MemGuard: Consensus-based validation defense."""

from typing import Dict, Any, Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer

from aaps.defenses.base_defense import BaseDefense, DefenseResult
from aaps.agent.memory_manager import MemoryManager


class AMemGuard(BaseDefense):
    """A-MemGuard defense using consensus-based validation."""

    def __init__(
        self,
        memory_manager: Optional["MemoryManager"] = None,
        config: Optional[Dict[str, Any]] = None,
        consensus_threshold: float = 0.7,
    ):
        """Initialize A-MemGuard.

        When ``memory_manager`` is None (e.g. Qdrant unavailable) the defense
        degrades gracefully: every consensus check returns "insufficient
        context" and allows the write. This is the honest degraded mode — the
        defense is loaded in the matrix but cannot exercise its core logic
        without a memory backend.
        """
        super().__init__(config)
        self.memory_manager = memory_manager
        self.consensus_threshold = consensus_threshold
        try:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.encoder = None

        # Dual-memory: lessons storage
        self.lessons_collection = "memguard_lessons"

    def check_memory_write(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        """Check memory write using consensus validation."""
        self.stats["total_checks"] += 1

        if not self.memory_manager:
            # No memory backend — can't do consensus, allow
            self.stats["allowed"] += 1
            return DefenseResult(
                allowed=True,
                confidence=0.3,
                reason="No memory backend available for consensus check",
                metadata={"degraded": True},
            )

        # Retrieve related memories
        related_memories = self.memory_manager.retrieve(
            content, k=5, similarity_threshold=0.6
        )

        if len(related_memories) < 2:
            # Not enough context for consensus, allow but with low confidence
            self.stats["allowed"] += 1
            return DefenseResult(
                allowed=True,
                confidence=0.5,
                reason="Insufficient context for consensus check",
                metadata={"related_memories": len(related_memories)},
            )

        # Compute consensus: check if reasoning paths are consistent
        consensus_score = self._compute_consensus(content, related_memories)

        # Check lessons (dual-memory)
        lesson_match = self._check_lessons(content)

        # Decision
        if lesson_match:
            # Matches a known attack pattern
            self.stats["blocked"] += 1
            return DefenseResult(
                allowed=False,
                confidence=0.9,
                reason=f"Matches known attack pattern from lessons",
                metadata={
                    "consensus_score": consensus_score,
                    "lesson_match": True,
                },
            )

        if consensus_score < self.consensus_threshold:
            # Low consensus indicates potential attack
            self.stats["blocked"] += 1
            return DefenseResult(
                allowed=False,
                confidence=0.8,
                reason=f"Low consensus score: {consensus_score:.2f}",
                metadata={
                    "consensus_score": consensus_score,
                    "related_memories": len(related_memories),
                },
            )

        # High consensus, allow
        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            confidence=consensus_score,
            reason=f"High consensus score: {consensus_score:.2f}",
            metadata={
                "consensus_score": consensus_score,
                "related_memories": len(related_memories),
            },
        )

    def _compute_consensus(
        self, content: str, related_memories: List
    ) -> float:
        """Compute consensus score by comparing reasoning paths."""
        if not related_memories:
            return 0.0

        # Encode content and memories
        content_embedding = self.encoder.encode(content)
        memory_embeddings = [
            self.encoder.encode(mem.content) for mem in related_memories
        ]

        # Compute similarities
        similarities = []
        for mem_emb in memory_embeddings:
            similarity = np.dot(content_embedding, mem_emb) / (
                np.linalg.norm(content_embedding) * np.linalg.norm(mem_emb)
            )
            similarities.append(similarity)

        # Consensus: average similarity (higher = more consensus)
        consensus = np.mean(similarities) if similarities else 0.0

        return float(consensus)

    def _check_lessons(self, content: str) -> bool:
        """Check if content matches known attack patterns in lessons."""
        if not self.memory_manager:
            return False
        # Retrieve lessons
        lessons = self.memory_manager.retrieve(
            content,
            k=3,
            entry_type="lesson",
            similarity_threshold=0.7,
        )

        return len(lessons) > 0

    def store_lesson(self, attack_content: str, description: str):
        """Store detected attack as a lesson."""
        self.memory_manager.store(
            content=f"Attack pattern: {attack_content}. Description: {description}",
            metadata={"type": "attack_lesson", "description": description},
            entry_type="lesson",
        )

    def check_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        """Check retrieved documents (A-MemGuard focuses on memory writes)."""
        # A-MemGuard primarily validates memory writes, not retrieval
        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            confidence=0.5,
            reason="A-MemGuard does not filter retrieval",
            metadata={},
        )
