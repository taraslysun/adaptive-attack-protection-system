"""Centralized model configuration for attacks.

All attack modules import model names and URLs from here instead of
hardcoding them.  Override via environment variables or a .env file.

Primary pipeline: OpenRouter (set OPENROUTER_API_KEY).
Legacy pipeline: local Ollama (OLLAMA_* vars, kept for backward compat).
"""

import os
import warnings
from dotenv import load_dotenv

load_dotenv()

# ---- OpenRouter (primary) ------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_VICTIM_MODEL = os.getenv("OPENROUTER_VICTIM_MODEL", "openai/gpt-4o-mini")
OPENROUTER_ATTACKER_MODEL = os.getenv(
    "OPENROUTER_ATTACKER_MODEL", "openai/gpt-4o-mini"
)
OPENROUTER_JUDGE_MODEL = os.getenv(
    "OPENROUTER_JUDGE_MODEL", "google/gemini-2.5-flash"
)
OPENROUTER_ONLY = os.getenv("OPENROUTER_ONLY", "").lower() in ("1", "true", "yes")

# ---- Ollama (legacy — kept for backward compat) --------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TARGET_MODEL = os.getenv("OLLAMA_TARGET_MODEL",
                                os.getenv("OLLAMA_MODEL", "gemma4:e4b"))
OLLAMA_HELPER_MODEL = os.getenv("OLLAMA_HELPER_MODEL", OLLAMA_TARGET_MODEL)
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "gemma4:e4b")

if OPENROUTER_ONLY and not OPENROUTER_API_KEY:
    warnings.warn(
        "OPENROUTER_ONLY=1 but OPENROUTER_API_KEY is not set; "
        "all LLM calls will fail.",
        stacklevel=1,
    )

# ---- LiteLLM (attacker / judge fallback) ---------------------------------
LITELLM_ATTACKER_MODEL = os.getenv("ATTACK_LLM_MODEL", "gemini/gemini-2.5-pro")
LITELLM_JUDGE_MODEL = os.getenv("JUDGE_LLM_MODEL", "gemini/gemini-2.5-flash")
USE_LITELLM = os.getenv("USE_LITELLM", "true").lower() in ("true", "1", "yes")
