"""Backward-compatibility shim. Real module: :mod:`attacks._core.local_agent`."""

from aaps.attacks._core.local_agent import *  # noqa: F401,F403
from aaps.attacks._core.local_agent import (  # noqa: F401
    AVAILABLE_TOOLS,
    DEFAULT_SYSTEM_PROMPT,
    MINIMAL_SYSTEM_PROMPT,
    LocalAgent,
    MemoryEntry,
    OllamaError,
)

__all__ = [
    "AVAILABLE_TOOLS",
    "DEFAULT_SYSTEM_PROMPT",
    "MINIMAL_SYSTEM_PROMPT",
    "LocalAgent",
    "MemoryEntry",
    "OllamaError",
]
