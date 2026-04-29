"""Adaptive Feedback Learner (AFL).

Replaces the deliberately empty ``_retrain`` from
``defenses/am2i/adaptive_learner.py``.

Idea
----
Every time a layer HARD_BLOCKs an attack, the AFL captures the
offending text into a ring buffer.  Periodically (every ``batch_size``
captures) it asks the layers that support online updates -- L2's
``ToolOutputProbeDefense`` and L5's ``MemoryWriteGuardDefense`` -- to
ingest the buffered captures via ``learn_signature``.

This is a *minimum viable* online learner: a frequency-bounded n-gram
extractor over recent attacks.  We deliberately avoid a full DistilBERT
fine-tune to (a) keep wall-clock manageable on a laptop and (b) keep
the thesis claim honest -- the AFL's role is to demonstrate the
*architectural* feedback loop, not to set state-of-the-art on detector
accuracy.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Protocol


class _SupportsLearn(Protocol):
    def learn_signature(self, attack_text: str) -> int: ...
    def n_learned(self) -> int: ...


class AdaptiveFeedbackLearner:
    """Buffer + dispatcher for AFL updates."""

    def __init__(
        self,
        learners: Optional[List[_SupportsLearn]] = None,
        *,
        batch_size: int = 8,
        max_buffer: int = 256,
    ) -> None:
        self.learners: List[_SupportsLearn] = list(learners or [])
        self.batch_size = batch_size
        self._buffer: Deque[str] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self.refit_count = 0
        self.captured = 0
        self.last_refit_ts: Optional[float] = None
        self.history: List[Dict[str, Any]] = []

    def register(self, learner: _SupportsLearn) -> None:
        self.learners.append(learner)

    def observe_block(
        self,
        attack_text: str,
        *,
        layer: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not attack_text:
            return None
        with self._lock:
            self._buffer.append(attack_text)
            self.captured += 1
            should_refit = len(self._buffer) >= self.batch_size
            if should_refit:
                batch = list(self._buffer)
                self._buffer.clear()
        if should_refit:
            return self._refit(batch, triggering_layer=layer)
        return None

    def force_refit(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._buffer:
                return None
            batch = list(self._buffer)
            self._buffer.clear()
        return self._refit(batch, triggering_layer="manual")

    def _refit(
        self, batch: List[str], *, triggering_layer: Optional[str]
    ) -> Dict[str, Any]:
        per_learner: Dict[str, int] = {}
        for learner in self.learners:
            added = 0
            for text in batch:
                try:
                    added += learner.learn_signature(text)
                except Exception:
                    continue
            per_learner[learner.__class__.__name__] = added
        self.refit_count += 1
        self.last_refit_ts = time.time()
        record = {
            "ts": self.last_refit_ts,
            "batch_size": len(batch),
            "triggered_by": triggering_layer,
            "per_learner_added": per_learner,
            "totals": {
                learner.__class__.__name__: learner.n_learned()
                for learner in self.learners
            },
        }
        self.history.append(record)
        return record

    def stats(self) -> Dict[str, Any]:
        return {
            "captured": self.captured,
            "refit_count": self.refit_count,
            "last_refit_ts": self.last_refit_ts,
            "buffered": len(self._buffer),
            "n_learners": len(self.learners),
            "totals": {
                learner.__class__.__name__: learner.n_learned()
                for learner in self.learners
            },
        }
