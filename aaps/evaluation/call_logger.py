"""Lightweight per-call logger used by the migrated PACE/baseline modules.

API kept compatible with the legacy repo:

    get_call_logger().log_call(role=..., model=..., prompt=..., response=...,
                               latency_ms=..., tokens_in=..., tokens_out=..., ...)

By default writes nothing (singleton no-op). Set ``AAPS_CALL_LOG`` to a path
to append JSONL records.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_LOCK = threading.Lock()
_INSTANCE: "Optional[CallLogger]" = None


@dataclass
class CallLogger:
    path: Optional[str] = None
    defense_path: Optional[str] = None

    def _append(self, target: Optional[str], rec: dict) -> None:
        if not target:
            return
        try:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def log_call(self, **fields: Any) -> None:
        self._append(self.path, {"ts": time.time(), **fields})

    def log_defense_decision(
        self,
        *,
        hook: str,
        defense_class: str,
        allowed: bool,
        reason: str = "",
        layer: str = "",
        latency_ms: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> None:
        target = self.defense_path or self.path
        rec = {
            "ts": time.time(),
            "kind": "defense_decision",
            "hook": hook,
            "defense_class": defense_class,
            "allowed": bool(allowed),
            "reason": str(reason or ""),
            "layer": str(layer or ""),
            "latency_ms": float(latency_ms or 0.0),
            "metadata": metadata or {},
        }
        self._append(target, rec)


def get_call_logger() -> CallLogger:
    global _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            _INSTANCE = CallLogger(
                path=os.getenv("AAPS_CALL_LOG"),
                defense_path=os.getenv("AAPS_DEFENSE_DECISION_LOG"),
            )
        return _INSTANCE
