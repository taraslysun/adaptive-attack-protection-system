"""Centralised logging configuration for the thesis codebase.

Every module in attacks/, defenses/, evaluation/, and agent/ should
obtain its logger via::

    from aaps.attacks._core.logging_config import get_logger
    log = get_logger(__name__)

Logger hierarchy (mirrors package layout):
    spq.planner
    spq.executor
    spq.clusters
    spq.pipeline
    spq.feature_extractors
    attacks.base
    attacks.pair
    attacks.rl
    attacks.hrt
    attacks.poisoned_rag
    attacks.supply_chain
    agent.local
    evaluation.benchmark

Environment variables
---------------------
LOG_LEVEL   DEBUG | INFO | WARNING | ERROR  (default: INFO)
LOG_FILE    path to append structured JSONL records  (default: off)

Setting LOG_LEVEL=DEBUG will print every LLM request/response, every
defense hook decision, and every fallback activation. For normal runs
INFO is enough to catch all warnings and errors without flooding stdout.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

_CONFIGURED = False
_LOCK = threading.Lock()

_ENV_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FILE = os.getenv("LOG_FILE", "")

# Map string → logging int level, with sane fallback
_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_ROOT_LEVEL = _LEVEL_MAP.get(_ENV_LEVEL, logging.INFO)


def set_unified_log_path(path: Path) -> None:
    """Add a rotating-file handler to the root logger pointing to the experiment dir.

    Long thesis runs (3 seeds × 5 attacks × 8 defenses with `LOG_LEVEL=DEBUG`
    can produce >1 GB of `unified.log`. We use ``RotatingFileHandler`` with a
    100 MB cap × 10 backups (≈ 1 GB ceiling per run) so a 24-hour smoke
    cannot exhaust disk. Override via env vars ``UNIFIED_LOG_MAX_BYTES`` and
    ``UNIFIED_LOG_BACKUPS``.
    """
    root = logging.getLogger()
    unified_path = path / "unified.log"
    path.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    max_bytes = int(os.getenv("UNIFIED_LOG_MAX_BYTES", str(100 * 1024 * 1024)))
    backups = int(os.getenv("UNIFIED_LOG_BACKUPS", "10"))
    fh = logging.handlers.RotatingFileHandler(
        unified_path,
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)  # Unified log is always unfiltered
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Also log a header to mark the start of this log stream
    root.info(
        f"=== Unified Log Initialized at {unified_path} "
        f"(rotating: max={max_bytes // (1024 * 1024)} MiB × {backups} backups) ==="
    )


class _JSONLHandler(logging.Handler):
    """Append structured JSONL records to a file (thread-safe)."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        doc: Dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra", None)
        if extra:
            doc["extra"] = extra
        line = json.dumps(doc, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass
        super().close()


def _configure() -> None:
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED:
            return

        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)  # handlers filter; root passes all

        # Stdout handler — respects LOG_LEVEL
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(_ROOT_LEVEL)
        sh.setFormatter(fmt)
        root.addHandler(sh)

        # JSONL file handler — always DEBUG when file is configured
        if _LOG_FILE:
            jh = _JSONLHandler(_LOG_FILE)
            jh.setLevel(logging.DEBUG)
            root.addHandler(jh)

        # Silence noisy third-party loggers
        for noisy in ("httpx", "httpcore", "openai", "litellm", "urllib3",
                       "requests", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name*, configuring root logging on first call."""
    _configure()
    return logging.getLogger(name)


class LogExtra:
    """Helper to attach structured extra fields to a log record.

    Usage::

        log.warning("fallback activated", extra=LogExtra(
            fallback="hash_embed", reason="sentence_transformers unavailable",
        ))
    """

    def __init__(self, **kwargs: Any) -> None:
        self.extra = kwargs

    def __repr__(self) -> str:  # pragma: no cover
        return f"LogExtra({self.extra!r})"
