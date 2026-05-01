"""Backward-compatibility shim. Real module: :mod:`attacks._core.model_registry`."""

from aaps.attacks._core.model_registry import *  # noqa: F401,F403
from aaps.attacks._core.model_registry import (  # noqa: F401
    ALL_MODELS,
    DEFAULT_OLLAMA_URL,
    NON_TOOL,
    TOOL_CAPABLE,
    get_available_models,
    get_model_endpoint,
    is_remote,
    is_tool_capable,
    load_models_config,
    non_tool_models,
    register_model,
    tool_capable_models,
)
