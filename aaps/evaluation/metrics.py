"""Metrics collection and calculation."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time
from collections import defaultdict


@dataclass
class AttackMetrics:
    """Metrics for a single attack attempt."""

    success: bool
    query_count: int
    latency_ms: float
    trigger_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    arg_breach_detected: bool = False


@dataclass
class DefenseMetrics:
    """Metrics for defense performance."""

    attack_success_rate: float
    detection_rate: float
    false_positive_rate: float
    latency_overhead_ms: float
    task_success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and computes evaluation metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.attack_results: List[AttackMetrics] = []
        self.defense_results: List[Dict[str, Any]] = []
        self.task_results: List[Dict[str, Any]] = []

    def record_attack(
        self,
        success: bool,
        query_count: int = 1,
        latency_ms: float = 0.0,
        trigger_length: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        arg_breach_detected: bool = False,
    ):
        """Record an attack attempt."""
        self.attack_results.append(
            AttackMetrics(
                success=success,
                query_count=query_count,
                latency_ms=latency_ms,
                trigger_length=trigger_length,
                metadata=metadata or {},
                arg_breach_detected=arg_breach_detected,
            )
        )

    def record_defense_check(
        self,
        blocked: bool,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a defense check."""
        self.defense_results.append(
            {
                "blocked": blocked,
                "latency_ms": latency_ms,
                "metadata": metadata or {},
            }
        )

    def record_task(
        self,
        success: bool,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a legitimate task execution."""
        self.task_results.append(
            {
                "success": success,
                "latency_ms": latency_ms,
                "metadata": metadata or {},
            }
        )

    def compute_attack_success_rate(self) -> float:
        """Compute attack success rate (ASR)."""
        if not self.attack_results:
            return 0.0

        successful = sum(1 for r in self.attack_results if r.success)
        return successful / len(self.attack_results)

    def compute_detection_rate(self) -> float:
        """Compute defense detection rate."""
        if not self.defense_results:
            return 0.0

        blocked = sum(1 for r in self.defense_results if r["blocked"])
        return blocked / len(self.defense_results)

    def compute_false_positive_rate(self, legitimate_blocks: int = 0) -> float:
        """Compute false positive rate -- placeholder.

        Thesis remediation ``codebase-fix-fpr-placeholder``: this is a
        *stub*, not a measurement of FPR. ``MetricsCollector`` does not
        currently distinguish "block on a legitimate task" from "block
        on an attack"; the caller must supply ``legitimate_blocks``
        explicitly. Until a benign-task harness feeds blocks into this
        method, ``false_positive_rate`` in :py:meth:`get_summary` is
        always ``0.0`` and MUST NOT be cited as the system's FPR in
        thesis tables. Use the matrix runner's per-defense
        ``benign_block_rate`` (when populated) instead, or report FPR
        as ``n/a`` when no benign baseline has been run.
        """
        if not self.task_results:
            return 0.0

        total_tasks = len(self.task_results)
        false_positives = legitimate_blocks

        return false_positives / total_tasks if total_tasks > 0 else 0.0

    def compute_avg_latency(self) -> float:
        """Compute average latency."""
        if not self.attack_results:
            return 0.0

        total_latency = sum(r.latency_ms for r in self.attack_results)
        return total_latency / len(self.attack_results)

    def compute_avg_query_efficiency(self) -> float:
        """Compute average queries until first success."""
        successful_attacks = [r for r in self.attack_results if r.success]
        if not successful_attacks:
            return float("inf")

        total_queries = sum(r.query_count for r in successful_attacks)
        return total_queries / len(successful_attacks)

    def compute_task_success_rate(self) -> float:
        """Compute legitimate task success rate."""
        if not self.task_results:
            return 0.0

        successful = sum(1 for r in self.task_results if r["success"])
        return successful / len(self.task_results)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "attack_metrics": {
                "success_rate": self.compute_attack_success_rate(),
                "avg_latency_ms": self.compute_avg_latency(),
                "avg_query_efficiency": self.compute_avg_query_efficiency(),
                "total_attempts": len(self.attack_results),
                "arg_breach_rate": (
                    sum(1 for r in self.attack_results if r.arg_breach_detected)
                    / len(self.attack_results)
                    if self.attack_results else 0.0
                ),
            },
            "defense_metrics": {
                "detection_rate": self.compute_detection_rate(),
                "false_positive_rate": self.compute_false_positive_rate(),
                "total_checks": len(self.defense_results),
            },
            "task_metrics": {
                "success_rate": self.compute_task_success_rate(),
                "total_tasks": len(self.task_results),
            },
        }

    def reset(self):
        """Reset all collected metrics."""
        self.attack_results = []
        self.defense_results = []
        self.task_results = []
