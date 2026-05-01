"""Per-call PACE trace logger.

Writes one JSONL record per :py:meth:`PACEDefense.process_query` (and
related per-call hooks) to a configurable path. The schema is the one
documented in ``docs/design/spq.md`` §5 and is consumed by
:mod:`evaluation.defense_benchmark` for the ``quorum_margin`` and
``plan_deviation_rate`` aggregations.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PACETraceRecord:
    session_id: str
    user_request: str
    pace_plan: Dict[str, Any]
    evidence: Dict[str, Any]
    filled_plans: List[Dict[str, Any]]
    gates: Dict[str, Any]
    outcome: str
    latency_ms: Dict[str, Any]
    ts: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "session_id": self.session_id,
            "user_request": self.user_request[:200],
            "pace_plan": self.pace_plan,
            "evidence": self.evidence,
            "filled_plans": self.filled_plans,
            "gates": self.gates,
            "outcome": self.outcome,
            "latency_ms": self.latency_ms,
        }


class PACETraceLogger:
    """Append-only JSONL writer.

    Thread-safe (a single coarse lock); the per-call payload is small
    so contention is not measurable at the rates we emit.
    """

    def __init__(self, path: Optional[str | os.PathLike[str]] = None) -> None:
        self.path: Optional[Path] = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.records: List[PACETraceRecord] = []

    def write(self, record: PACETraceRecord) -> None:
        payload = record.to_dict()
        with self._lock:
            self.records.append(record)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False))
                    f.write("\n")

    def aggregate(self) -> Dict[str, Any]:
        """Roll up trace records into the matrix-friendly summary.

        Returns the structure consumed by
        :func:`evaluation.defense_benchmark.attach_spq_metrics`.
        """
        n = len(self.records)
        if n == 0:
            return {
                "n_calls": 0,
                "cfi_violation_count": 0,
                "cfi_execution_violation_count": 0,
                "cfi_block_count": 0,
                "abstain_rate": 0.0,
                "replan_rate": 0.0,
                "fire_rate": 0.0,
                "quorum_margin_histogram": {},
                "plan_deviation_rate": 0.0,
            }
        # Design invariant: execution-level CFI violations must be zero.
        # Keep blocked CFI attempts in a separate counter so attacks remain auditable.
        cfi_execution_violations = 0
        cfi_block_count = 0
        margins: List[float] = []
        outcomes = {"fired": 0, "abstained": 0, "replanned": 0}
        deviations = 0
        for rec in self.records:
            blocked = len(rec.gates.get("cfi_violations", []) or [])
            cfi_block_count += blocked
            if blocked and rec.outcome == "fired":
                cfi_execution_violations += blocked
            for d in rec.gates.get("quorum_decisions", []) or []:
                margins.append(float(d.get("margin", 0.0)))
            outcome = rec.outcome
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if rec.gates.get("cfi_violations"):
                deviations += 1
        bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for m in margins:
            if m < 0.2:
                bins["0.0-0.2"] += 1
            elif m < 0.4:
                bins["0.2-0.4"] += 1
            elif m < 0.6:
                bins["0.4-0.6"] += 1
            elif m < 0.8:
                bins["0.6-0.8"] += 1
            else:
                bins["0.8-1.0"] += 1
        fired = outcomes.get("fired", 0)
        abstained = outcomes.get("abstained", 0)
        replanned = outcomes.get("replanned", 0)
        total = max(1, n)
        return {
            "n_calls": n,
            # Backward-compatible alias expected by table/rendering code.
            "cfi_violation_count": cfi_execution_violations,
            "cfi_execution_violation_count": cfi_execution_violations,
            "cfi_block_count": cfi_block_count,
            "abstain_rate": abstained / total,
            "replan_rate": replanned / total,
            "fire_rate": fired / total,
            "quorum_margin_histogram": bins,
            "quorum_margin_mean": sum(margins) / len(margins) if margins else 0.0,
            "plan_deviation_rate": deviations / total,
        }
