"""Request/response call logger for every LLM interaction.

Captures planner, executor, agent, attacker, judge, and defense_probe
calls to JSONL files for post-hoc analysis and thesis reproducibility.

Usage::

    from aaps.evaluation.call_logger import get_call_logger

    logger = get_call_logger()
    logger.log_call(
        role="agent", model="openai/gpt-4o-mini",
        prompt=messages, response=text,
        latency_ms=42.0, tokens_in=100, tokens_out=50,
    )

Configuration (env vars):
    CALL_LOG_LEVEL   full | summary | off   (default: full)
    CALL_LOG_DIR     override output dir     (default: inferred from run --out)
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aaps.attacks._core.logging_config import get_logger

log = get_logger("evaluation.call_logger")


_LEVEL = os.getenv("CALL_LOG_LEVEL", "full").lower()
_ENABLED = _LEVEL in ("full", "summary")
_FULL = _LEVEL == "full"

_lock = threading.Lock()
_instance: Optional["CallLogger"] = None


def get_call_logger() -> "CallLogger":
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = CallLogger()
    return _instance


def reset_call_logger() -> None:
    """Reset the singleton (for tests)."""
    global _instance
    _instance = None


class CallLogger:
    """Thread-safe JSONL logger for LLM calls."""

    def __init__(self, log_dir: Optional[str] = None):
        self._log_dir = log_dir or os.getenv("CALL_LOG_DIR", "")
        self._handles: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._call_counts: Dict[str, int] = {}

    @property
    def log_dir(self) -> str:
        return self._log_dir

    @log_dir.setter
    def log_dir(self, value: str) -> None:
        with self._lock:
            for fh in self._handles.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._handles.clear()
            self._log_dir = value

    def log_call(
        self,
        role: str,
        model: str,
        prompt: Union[str, List[Dict[str, str]]],
        response: str,
        latency_ms: float = 0.0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        attempt: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if not _ENABLED or not self._log_dir:
            return

        record: Dict[str, Any] = {
            "ts": time.time(),
            "role": role,
            "model": model,
            "attempt": attempt,
            "latency_ms": round(latency_ms, 1),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

        if _FULL:
            record["prompt"] = prompt
            record["response"] = response
        else:
            if isinstance(prompt, str):
                record["prompt_len"] = len(prompt)
            else:
                record["prompt_len"] = sum(len(m.get("content", "")) for m in prompt)
            record["response_len"] = len(response) if response else 0

        if error:
            record["error"] = error
        if metadata:
            record["metadata"] = metadata

        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        self._write(role, line)

        # Mirror to unified log (root logger)
        log.debug(f"LLM {role.upper()} CALL: model={model} attempt={attempt}")
        if response:
            resp_preview = response.replace("\n", " ")[:200]
            log.debug(f"LLM {role.upper()} RESP: {resp_preview}...")
        if error:
            log.error(f"LLM {role.upper()} ERROR: {error}")

    def log_defense_decision(
        self,
        hook: str,
        defense_class: str,
        allowed: bool,
        reason: str = "",
        layer: str = "",
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single defense-hook decision (input/output/tool_call/memory/retrieval).

        Lands at ``<log_dir>/calls/defense_decision.jsonl``. One record per
        invocation of ``check_input`` / ``check_output`` / ``check_tool_call`` /
        ``check_memory_write`` / ``check_retrieval`` from any defense
        (PACE, baselines, or future). Enables post-hoc audit of every
        block/allow without scraping ``unified.log``.
        """
        if not _ENABLED or not self._log_dir:
            return
        record: Dict[str, Any] = {
            "ts": time.time(),
            "role": "defense_decision",
            "hook": hook,
            "defense": defense_class,
            "allowed": bool(allowed),
            "reason": str(reason)[:500],
            "layer": str(layer),
            "latency_ms": round(latency_ms, 1),
        }
        if metadata:
            record["metadata"] = metadata
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        self._write("defense_decision", line)

    def _write(self, role: str, line: str) -> None:
        with self._lock:
            fh = self._handles.get(role)
            if fh is None:
                calls_dir = Path(self._log_dir) / "calls"
                calls_dir.mkdir(parents=True, exist_ok=True)
                path = calls_dir / f"{role}.jsonl"
                fh = open(path, "a", encoding="utf-8")
                self._handles[role] = fh
            fh.write(line)
            fh.flush()
            self._call_counts[role] = self._call_counts.get(role, 0) + 1

    def write_manifest(
        self,
        expected_calls_min: int = 1,
        expected_calls_max: int = 999999,
    ) -> None:
        """Write _manifest.json summarising what was logged in this cell."""
        if not self._log_dir:
            return
        calls_dir = Path(self._log_dir) / "calls"
        calls_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "expected_calls_min": expected_calls_min,
            "expected_calls_max": expected_calls_max,
            "roles_present": sorted(self._call_counts.keys()),
            "call_counts": dict(self._call_counts),
        }
        (calls_dir / "_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        with self._lock:
            for fh in self._handles.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._handles.clear()
