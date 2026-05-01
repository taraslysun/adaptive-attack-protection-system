"""Backward-compatibility shim. Real module: :mod:`attacks._core.config`."""

from aaps.attacks._core.config import *  # noqa: F401,F403
from aaps.attacks._core.config import (  # noqa: F401
    LITELLM_ATTACKER_MODEL,
    LITELLM_JUDGE_MODEL,
    OLLAMA_HELPER_MODEL,
    OLLAMA_JUDGE_MODEL,
    OLLAMA_TARGET_MODEL,
    OLLAMA_URL,
    USE_LITELLM,
)

GCG_DEFAULT_SURROGATE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

__all__ = [
    "GCG_DEFAULT_SURROGATE_MODEL",
    "LITELLM_ATTACKER_MODEL",
    "LITELLM_JUDGE_MODEL",
    "OLLAMA_HELPER_MODEL",
    "OLLAMA_JUDGE_MODEL",
    "OLLAMA_TARGET_MODEL",
    "OLLAMA_URL",
    "USE_LITELLM",
]
