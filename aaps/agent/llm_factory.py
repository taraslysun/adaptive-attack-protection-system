"""Factory for chat models (OpenRouter-first, vendor fallback, optional Ollama).

Imports are lazy so missing optional SDKs or API keys do not fail at import time.
"""

from __future__ import annotations

import os
from typing import Any

_DEFAULT_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/taraslysun/ucu-diploma",
    "X-Title": "UCU AIS Thesis",
}


def _openrouter_base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def _normalise_openai_model_id(model_id: str) -> str:
    mid = model_id.strip()
    if mid.startswith("openai/"):
        return mid[len("openai/") :]
    return mid


def _normalise_anthropic_model_id(model_id: str) -> str:
    mid = model_id.strip()
    if mid.startswith("anthropic/"):
        return mid[len("anthropic/") :]
    return mid


def _is_anthropic_style(model_id: str) -> bool:
    m = model_id.strip().lower()
    return m.startswith("anthropic/") or m.startswith("claude")


def _is_openai_style(model_id: str) -> bool:
    m = model_id.strip().lower()
    return m.startswith("openai/") or m.startswith("gpt")


def _normalise_ollama_model_id(model_id: str) -> str:
    mid = model_id.strip()
    if mid.startswith("ollama/"):
        return mid[len("ollama/") :]
    return mid


def _force_ollama_backend() -> bool:
    flag = os.getenv("LLM_FORCE_OLLAMA", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _is_ollama_style(model_id: str) -> bool:
    mid = model_id.strip().lower()
    if mid.startswith("ollama/"):
        return True
    # Optional escape hatch: force local Ollama routing for plain model IDs.
    if _force_ollama_backend() and not (
        _is_openai_style(model_id) or _is_anthropic_style(model_id)
    ):
        return True
    return False


def make_remote_chat_model(
    model_id: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    allow_tools: bool = True,
):
    """Return a LangChain chat model for remote inference.

    Resolution: explicit Ollama (``ollama/...`` or ``LLM_FORCE_OLLAMA=1``) →
    OpenRouter (if ``OPENROUTER_API_KEY``) → OpenAI API → Anthropic API.
    ``allow_tools`` is accepted for API compatibility; tool binding is done by callers.
    """
    _ = allow_tools  # Tool capability is determined by the model + bind_tools() on caller.

    if _is_ollama_style(model_id):
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except Exception:
            from langchain_community.chat_models import ChatOllama

        kwargs: dict[str, Any] = {
            "model": _normalise_ollama_model_id(model_id),
            "base_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["num_predict"] = max_tokens
        return ChatOllama(**kwargs)

    router_key = os.getenv("OPENROUTER_API_KEY")
    if router_key:
        from langchain_openai import ChatOpenAI

        _timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
        kwargs: dict[str, Any] = {
            "model": model_id,
            "base_url": _openrouter_base_url(),
            "api_key": router_key,
            "temperature": temperature,
            "default_headers": dict(_DEFAULT_OPENROUTER_HEADERS),
            "request_timeout": _timeout,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    if _is_openai_style(model_id):
        okey = os.getenv("OPENAI_API_KEY")
        if okey:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": _normalise_openai_model_id(model_id),
                "api_key": okey,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return ChatOpenAI(**kwargs)

    if _is_anthropic_style(model_id):
        akey = os.getenv("ANTHROPIC_API_KEY")
        if akey:
            from langchain_anthropic import ChatAnthropic

            kwargs = {
                "model": _normalise_anthropic_model_id(model_id),
                "api_key": akey,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return ChatAnthropic(**kwargs)

    raise RuntimeError(
        "No compatible LLM backend configured; either set DEFAULT_LLM_MODEL=ollama/<model> "
        "for local inference or configure OPENROUTER_API_KEY / vendor API keys."
    )


def get_remote_chat_completion(
    model_id: str, messages: list[dict], **kwargs: Any
) -> dict[str, Any]:
    """One-shot chat completion via OpenAI-compatible API (OpenRouter preferred).

    Returns ``{"content": str, "tool_calls": list, "finish_reason": str}``.
    """
    from openai import OpenAI

    router_key = os.getenv("OPENROUTER_API_KEY")
    client: OpenAI
    use_model = model_id

    if router_key:
        client = OpenAI(
            base_url=_openrouter_base_url(),
            api_key=router_key,
            default_headers=dict(_DEFAULT_OPENROUTER_HEADERS),
        )
    elif _is_openai_style(model_id) and os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        use_model = _normalise_openai_model_id(model_id)
    else:
        raise RuntimeError(
            "No remote LLM credentials available; set OPENROUTER_API_KEY or a vendor key."
        )

    resp = client.chat.completions.create(
        model=use_model,
        messages=messages,
        **kwargs,
    )
    choice = resp.choices[0]
    msg = choice.message
    raw_tool_calls = getattr(msg, "tool_calls", None) or []
    tool_calls: list[Any] = []
    for tc in raw_tool_calls:
        if hasattr(tc, "model_dump"):
            tool_calls.append(tc.model_dump())
        elif isinstance(tc, dict):
            tool_calls.append(tc)
        else:
            tool_calls.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", None),
                        "arguments": getattr(
                            getattr(tc, "function", None), "arguments", None
                        ),
                    },
                }
            )
    finish = getattr(choice, "finish_reason", None) or ""
    content = (msg.content or "") if msg.content is not None else ""
    return {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": str(finish),
    }


def list_openrouter_models() -> list[dict]:
    """GET public model list from OpenRouter (no auth). Returns ``[]`` on failure."""
    try:
        import requests

        r = requests.get(
            "https://openrouter.ai/api/v1/models",
            timeout=30,
        )
        if r.status_code != 200:
            return []
        payload = r.json()
        data = payload.get("data")
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []
