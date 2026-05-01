"""Structured per-layer trace logger for the Agent Integrity Stack.

One JSON record is emitted per layer evaluation, suitable for the
ablation tables, attribution heatmaps and case-study diagrams in the
thesis.  Records can be flushed to disk (JSONL) and/or kept in memory
for in-process querying.

Schema
------
::

    {
      "ts":         <unix epoch seconds, float>,
      "session":    <str>,
      "step":       <int -- monotonic per session>,
      "hook":       <one of: input | output | tool_call | tool_output |
                              memory_write | retrieval>,
      "layer":      <"L1" ... "L6" or layer.name>,
      "allowed":    <bool>,
      "severity":   <"allow" | "soft_filter" | "hard_block">,
      "confidence": <float in [0,1]>,
      "reason":     <str>,
      "latency_ms": <float | null>,
      "metadata":   <dict>
    }
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from aaps.defenses.base_defense import DefenseResult


class TraceLogger:
    """Append-only structured logger with optional JSONL persistence."""

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        keep_in_memory: bool = True,
        session_id: Optional[str] = None,
    ) -> None:
        self.path = path
        self.keep_in_memory = keep_in_memory
        self.session_id = session_id or f"session_{int(time.time())}"
        self._records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._step_counter = itertools.count(0)
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "session": self.session_id,
                            "event": "session_start",
                        }
                    )
                    + "\n"
                )

    def new_session(self, session_id: Optional[str] = None) -> str:
        with self._lock:
            self.session_id = session_id or f"session_{int(time.time())}"
            self._step_counter = itertools.count(0)
        return self.session_id

    def log(
        self,
        hook: str,
        result: DefenseResult,
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            step = next(self._step_counter)
        rec = {
            "ts": time.time(),
            "session": self.session_id,
            "step": step,
            "hook": hook,
            "layer": result.layer or result.metadata.get("layer"),
            "allowed": result.allowed,
            "severity": result.severity.value if result.severity else None,
            "confidence": result.confidence,
            "reason": result.reason,
            "latency_ms": result.latency_ms,
            "metadata": _scrub(result.metadata),
        }
        if extra:
            rec.update(extra)
        if self.keep_in_memory:
            with self._lock:
                self._records.append(rec)
        if self.path:
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        return rec

    # ------------------------------------------------------------------
    # In-memory aggregation helpers (used by the eval harness).
    # ------------------------------------------------------------------

    def records(self, *, session: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            recs = list(self._records)
        if session is not None:
            recs = [r for r in recs if r.get("session") == session]
        return recs

    def block_attribution(
        self, *, session: Optional[str] = None
    ) -> Dict[str, int]:
        recs = self.records(session=session)
        return dict(
            Counter(r["layer"] for r in recs if r.get("severity") == "hard_block")
        )

    def soft_filter_attribution(
        self, *, session: Optional[str] = None
    ) -> Dict[str, int]:
        recs = self.records(session=session)
        return dict(
            Counter(r["layer"] for r in recs if r.get("severity") == "soft_filter")
        )

    def latency_summary(
        self, *, session: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        recs = self.records(session=session)
        per_layer: Dict[str, List[float]] = {}
        for r in recs:
            lat = r.get("latency_ms")
            if lat is None:
                continue
            per_layer.setdefault(r["layer"], []).append(float(lat))
        out: Dict[str, Dict[str, float]] = {}
        for layer, lats in per_layer.items():
            lats_sorted = sorted(lats)
            out[layer] = {
                "n": float(len(lats)),
                "mean": sum(lats) / len(lats),
                "p50": lats_sorted[len(lats_sorted) // 2],
                "p95": lats_sorted[int(0.95 * (len(lats_sorted) - 1))],
                "max": lats_sorted[-1],
            }
        return out


def _scrub(obj: Any, depth: int = 0) -> Any:
    """Drop heavy fields (full embeddings, ndarrays, large blobs) so the
    log file does not blow up to gigabytes.  Best-effort, never raises."""
    if depth > 4:
        return "<truncated>"
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in ("wrapped_docs", "kept_docs", "dropped_docs"):
                out[k] = f"<{len(v) if hasattr(v, '__len__') else '?'} docs>"
            elif k in ("embedding", "embeddings"):
                out[k] = "<embedding omitted>"
            else:
                out[k] = _scrub(v, depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        if len(obj) > 20:
            return [_scrub(x, depth + 1) for x in obj[:20]] + ["<truncated>"]
        return [_scrub(x, depth + 1) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return str(obj)[:500]
    except Exception:
        return "<unserialisable>"
