"""Backward-compatibility shim. Real module: :mod:`attacks._core.scoring`."""

from aaps.attacks._core.scoring import *  # noqa: F401,F403
from aaps.attacks._core.scoring import (  # noqa: F401
    SUCCESS_THRESHOLD,
    composite_score,
    get_litellm_judge,
    llm_judge_score,
    perplexity_score,
    secret_key_score,
    tool_call_score,
)
from aaps.attacks._core.scoring import _keyword_and_refusal_score  # noqa: F401
