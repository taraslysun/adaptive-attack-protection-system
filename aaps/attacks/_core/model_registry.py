"""Model registry with tool-calling and endpoint metadata.

Supports both local Ollama models and remote Ollama-compatible
endpoints (self-hosted, hosted clouds with Ollama-compatible APIs,
Ollama Cloud, etc.).

Each model entry can override:
  * ``url``         -- base URL for the Ollama API (defaults to env ``OLLAMA_URL``)
  * ``api_key``     -- bearer token sent as Authorization header
  * ``headers``     -- arbitrary extra HTTP headers
  * ``tool_calling``-- whether the model supports native ``tools`` param

Model definitions can be loaded from YAML/JSON via ``load_models_config``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def _build_headers(api_key: Optional[str], extra: Optional[Dict[str, str]]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers


# ---------------------------------------------------------------------------
# Pinned model identifiers (thesis remediation: expand-model-matrix)
# ---------------------------------------------------------------------------
#
# Every model id below is the *exact* string we test against. When a
# vendor ships a new revision under the same tag (Ollama is rolling,
# OpenAI re-points "gpt-4o" without a date suffix, etc.) the run is
# reproducible only if the matrix runner records the resolved version
# at experiment time. ``scripts/run_thesis_experiments.py`` writes the
# resolved version + sha hash into ``summary.json`` per cell -- this
# table is the *source of truth* for what we *intended* to evaluate.
#
# Categories we cover for the bachelor case study:
#   - small-dense same family (8B): scale-down baseline
#   - mid-dense same family (32B): scale contrast (intra-family)
#   - large-dense different family (70B): cross-family scale contrast
#   - reasoning model (deepseek-reasoner): different inference shape
#   - MoE (mixtral): different architecture
#   - closed frontier (claude / gpt-4o / gemini): commercial-grade victim
#
# Closed frontier entries are routed via OpenRouter (OpenAI-compatible API)
# when ``OPENROUTER_API_KEY`` is set, with per-vendor keys as fallback.

TOOL_CAPABLE: Dict[str, Dict[str, Any]] = {
    "qwen3:8b": {
        "params": "8B",
        "family": "qwen3",
        "tool_calling": True,
        "url": DEFAULT_OLLAMA_URL,
        "category": "small_dense",
        "version_pin": "ollama:qwen3:8b@latest",
        "notes": "Primary local victim; Ollama official tool-calling example",
    },
    "qwen3:32b": {
        "params": "32B",
        "family": "qwen3",
        "tool_calling": True,
        "url": DEFAULT_OLLAMA_URL,
        "category": "scale_contrast_intra_family",
        "version_pin": "ollama:qwen3:32b@latest",
        "notes": "Scale contrast vs qwen3:8b (same family, 4x params)",
    },
    "llama3.1:8b": {
        "params": "8B",
        "family": "llama",
        "tool_calling": True,
        "url": DEFAULT_OLLAMA_URL,
        "category": "cross_family_baseline",
        "version_pin": "ollama:llama3.1:8b@latest",
        "notes": "Cross-family 8B baseline vs qwen3:8b",
    },
    "llama3.3:70b": {
        "params": "70B",
        "family": "llama",
        "tool_calling": True,
        "url": DEFAULT_OLLAMA_URL,
        "category": "scale_contrast_cross_family",
        "version_pin": "ollama:llama3.3:70b@latest",
        "notes": (
            "Large dense; cross-family scale contrast. Requires GPU "
            "host; configure URL via configs/models.example.yaml."
        ),
    },
    "mixtral:8x7b": {
        "params": "8x7B (MoE)",
        "family": "mistral",
        "tool_calling": True,
        "url": DEFAULT_OLLAMA_URL,
        "category": "moe_mistral",
        "version_pin": "ollama:mixtral:8x7b@latest",
        "notes": (
            "Mixture-of-Experts architecture; tests whether sparse "
            "routing changes attack surface vs dense models."
        ),
    },
    "gpt-oss:20b": {
        "params": "20B (MoE)",
        "family": "openai-oss",
        "tool_calling": True,
        "url": DEFAULT_OLLAMA_URL,
        "category": "moe_openai",
        "version_pin": "ollama:gpt-oss:20b@latest",
        "notes": (
            "OpenAI open-weight MoE (harmony format); brings OpenAI "
            "vendor into the local MoE bucket and is a reasoning-friendly "
            "victim for Crescendo / TAP comparisons."
        ),
    },
    "gemma4:26b": {
        "params": "26B",
        "family": "gemma4",
        "tool_calling": True,
        "url": DEFAULT_OLLAMA_URL,
        "category": "moe_google",
        "version_pin": "ollama:gemma4:26b@latest",
        "notes": (
            "Google Gemma 4 large variant; tool-calling enabled. "
            "Pairs with non-tool gemma4:e4b for the FPR/refusal contrast row."
        ),
    },
    "nemotron-3-nano:30b": {
        "params": "30B",
        "family": "nemotron",
        "tool_calling": True,
        "url": DEFAULT_OLLAMA_URL,
        "category": "scale_contrast_cross_family",
        "version_pin": "ollama:nemotron-3-nano:30b@latest",
        "notes": (
            "NVIDIA Nemotron-Nano-3 30B; agentic-tuned, tool-calling + "
            "thinking. Brings NVIDIA into the matrix and is the natural "
            "victim for AgentDojo and InjecAgent runs."
        ),
    },
    "deepseek-r1:7b": {
        "params": "7B (Qwen-distilled)",
        "family": "deepseek-r1-distill",
        "tool_calling": True,
        "url": DEFAULT_OLLAMA_URL,
        "category": "reasoning_local",
        "version_pin": "ollama:deepseek-r1:7b@latest",
        "notes": (
            "Local DeepSeek-R1 distill onto Qwen-7B; explicit-reasoning "
            "victim that pairs with qwen3:8b 'thinking-mode on/off' "
            "ablation and is the offline counterpart of the remote "
            "deepseek/deepseek-reasoner row."
        ),
    },
    # ---- Closed frontier (OpenRouter; no Ollama URL) ---------------------
    # Registered so ``--target-model <id>`` resolves and the matrix runner
    # can record the closed-frontier tier in ``summary.json``.
    "anthropic/claude-sonnet-4-20250514": {
        "params": "frontier (closed)",
        "family": "claude",
        "tool_calling": True,
        "url": "openrouter",
        "category": "closed_frontier",
        "version_pin": "openrouter:anthropic/claude-sonnet-4-20250514",
        "notes": "Closed-API frontier victim via OpenRouter / DeepAgent",
    },
    "openai/gpt-4o-2024-11-20": {
        "params": "frontier (closed)",
        "family": "openai",
        "tool_calling": True,
        "url": "openrouter",
        "category": "closed_frontier",
        "version_pin": "openrouter:openai/gpt-4o-2024-11-20",
        "notes": "Closed-API frontier victim via OpenRouter / DeepAgent",
    },
    "gemini/gemini-2.5-pro": {
        "params": "frontier (closed)",
        "family": "gemini",
        "tool_calling": True,
        "url": "openrouter",
        "category": "closed_frontier",
        "version_pin": "openrouter:google/gemini-2.5-pro",
        "notes": (
            "Closed-API frontier victim via OpenRouter; slug on OpenRouter is "
            "typically google/gemini-2.5-pro (registry id kept as gemini/ for legacy)."
        ),
    },
    # ---- Reasoning model -------------------------------------------------
    "deepseek/deepseek-reasoner": {
        "params": "reasoning",
        "family": "deepseek",
        "tool_calling": True,
        "url": "openrouter",
        "category": "reasoning",
        "version_pin": "openrouter:deepseek/deepseek-reasoner",
        "notes": (
            "Explicit-reasoning victim (Chain-of-Thought leaked into "
            "transcript); enables H-CoT-class attacks (arXiv:2502.12893)."
        ),
    },
    # ---- Thesis slim-3 cross-vendor victims (T4 vendor-comparison table) ---
    "google/gemini-2.0-flash-lite": {
        "params": "frontier (closed)",
        "family": "gemini",
        "tool_calling": True,
        "url": "openrouter",
        "category": "headline_cheap",
        "version_pin": "openrouter:google/gemini-2.0-flash-lite",
        "notes": "Gemini 2.0 Flash Lite via OpenRouter; slim-3 victim (refusal-strong baseline)",
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "params": "8B",
        "family": "llama",
        "tool_calling": True,
        "url": "openrouter",
        "category": "cheap_contrast",
        "version_pin": "openrouter:meta-llama/llama-3.1-8b-instruct",
        "notes": "Llama 3.1 8B Instruct via OpenRouter; slim-3 victim (mid baseline)",
    },
    "qwen/qwen3-8b": {
        "params": "8B",
        "family": "qwen3",
        "tool_calling": True,
        "url": "openrouter",
        "category": "cheap_contrast",
        "version_pin": "openrouter:qwen/qwen3-8b",
        "notes": "Qwen3 8B via OpenRouter; slim-3 victim (most attack-prone). Local alt: qwen3:8b via Ollama.",
    },
    # ---- Thesis slim-5 additional victims ------------------------------------
    "mistralai/mistral-small-2603": {
        "params": "22B", "family": "mistral", "tool_calling": True,
        "url": "openrouter", "category": "cheap_contrast",
        "version_pin": "openrouter:mistralai/mistral-small-2603",
        "notes": "Mistral Small March 2603 — tool-capable, thesis victim.",
    },
    "deepseek/deepseek-v4-flash": {
        "params": "frontier (closed)", "family": "deepseek", "tool_calling": True,
        "url": "openrouter", "category": "cheap_contrast",
        "version_pin": "openrouter:deepseek/deepseek-v4-flash",
        "notes": "DeepSeek V4 Flash — tool-capable, thesis victim.",
    },
    "google/gemini-2.0-flash-lite-001": {
        "params": "frontier (closed)", "family": "gemini", "tool_calling": True,
        "url": "openrouter", "category": "headline_cheap",
        "version_pin": "openrouter:google/gemini-2.0-flash-lite-001",
        "notes": "Gemini 2.0 Flash Lite (pinned 001) via OpenRouter — thesis slim-3 victim.",
    },
    # ---- Provider breadth (OpenRouter; thesis matrix coverage) ------------
    "deepseek/deepseek-chat": {
        "params": "chat",
        "family": "deepseek",
        "tool_calling": True,
        "url": "openrouter",
        "category": "provider_breadth",
        "version_pin": "openrouter:deepseek/deepseek-chat",
        "notes": "DeepSeek non-reasoning chat via OpenRouter",
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "params": "70B",
        "family": "llama",
        "tool_calling": True,
        "url": "openrouter",
        "category": "provider_breadth",
        "version_pin": "openrouter:meta-llama/llama-3.3-70b-instruct",
        "notes": "Llama 3.3 70B via OpenRouter (alternative to local llama3.3:70b)",
    },
    "mistralai/mixtral-8x7b-instruct": {
        "params": "8x7B (MoE)",
        "family": "mistral",
        "tool_calling": True,
        "url": "openrouter",
        "category": "provider_breadth",
        "version_pin": "openrouter:mistralai/mixtral-8x7b-instruct",
        "notes": "Mixtral via OpenRouter (alternative to local mixtral:8x7b)",
    },
    "qwen/qwen-2.5-72b-instruct": {
        "params": "72B",
        "family": "qwen",
        "tool_calling": True,
        "url": "openrouter",
        "category": "provider_breadth",
        "version_pin": "openrouter:qwen/qwen-2.5-72b-instruct",
        "notes": "Qwen2.5 72B instruct via OpenRouter",
    },
    "x-ai/grok-2-1212": {
        "params": "frontier (closed)",
        "family": "grok",
        "tool_calling": True,
        "url": "openrouter",
        "category": "provider_breadth",
        "version_pin": "openrouter:x-ai/grok-2-1212",
        "notes": "xAI Grok via OpenRouter",
    },
    # ---- Free-tier OpenRouter models (PACE matrix speed runs) -------------
    "google/gemma-4-26b-a4b-it:free": {
        "params": "26B (MoE-distill)",
        "family": "gemma4",
        "tool_calling": True,
        "url": "openrouter",
        "category": "free_tier",
        "version_pin": "openrouter:google/gemma-4-26b-a4b-it:free",
        "notes": "Gemma 4 26B free via OpenRouter; cross-family victim for PACE matrix",
    },
    "qwen/qwen3-next-80b-a3b-instruct:free": {
        "params": "80B (MoE-active-3B)",
        "family": "qwen3",
        "tool_calling": True,
        "url": "openrouter",
        "category": "free_tier",
        "version_pin": "openrouter:qwen/qwen3-next-80b-a3b-instruct:free",
        "notes": "Qwen3 80B MoE free via OpenRouter; same family as local qwen3:8b",
    },
    "nvidia/nemotron-nano-9b-v2:free": {
        "params": "9B",
        "family": "nemotron",
        "tool_calling": True,
        "url": "openrouter",
        "category": "free_tier",
        "version_pin": "openrouter:nvidia/nemotron-nano-9b-v2:free",
        "notes": "Nemotron nano 9B free via OpenRouter; third family for PACE matrix",
    },
    # ---- Cheap headline models (thesis small_paid matrix) ----------------
    "google/gemini-2.5-flash": {
        "params": "frontier (closed)",
        "family": "gemini",
        "tool_calling": True,
        "url": "openrouter",
        "category": "headline_cheap",
        "version_pin": "openrouter:google/gemini-2.5-flash",
        "notes": "Gemini 2.5 Flash via OpenRouter; victim + judge model",
    },
    "anthropic/claude-haiku-4-5": {
        "params": "frontier (closed)",
        "family": "claude",
        "tool_calling": True,
        "url": "openrouter",
        "category": "headline_cheap",
        "version_pin": "openrouter:anthropic/claude-haiku-4-5",
        "notes": "Claude Haiku 4.5 via OpenRouter; cheapest Anthropic victim",
    },
    "openai/gpt-4o-mini": {
        "params": "frontier (closed)",
        "family": "openai",
        "tool_calling": True,
        "url": "openrouter",
        "category": "headline_cheap",
        "version_pin": "openrouter:openai/gpt-4o-mini",
        "notes": "GPT-4o-mini via OpenRouter; victim + planner + executor + attacker",
    },
    # ---- Cheap / weaker models for attack-success contrast ---------------
    # These are added for thesis contrast rows: attacks land more easily on
    # smaller / less RLHF-tuned models, which makes PACE's relative benefit
    # more visible. Both are cited heavily in baseline defense papers.
    "mistralai/mistral-7b-instruct": {
        "params": "7B",
        "family": "mistral",
        "tool_calling": True,
        "url": "openrouter",
        "category": "cheap_contrast",
        "version_pin": "openrouter:mistralai/mistral-7b-instruct",
        "notes": (
            "Mistral 7B Instruct via OpenRouter; cheap victim cited in "
            "SecAlign, Circuit Breakers, SmoothLLM papers. Higher ASR than "
            "frontier models — good contrast row for PACE benefit."
        ),
    },
    "microsoft/phi-3.5-mini-128k-instruct": {
        "params": "3.8B",
        "family": "phi",
        "tool_calling": True,
        "url": "openrouter",
        "category": "cheap_contrast",
        "version_pin": "openrouter:microsoft/phi-3.5-mini-128k-instruct",
        "notes": (
            "Phi-3.5-Mini 3.8B via OpenRouter; smallest cheap victim for "
            "upper-bound ASR row (attacks succeed most reliably; PACE "
            "benefit largest). Long-context window tests indirect injection."
        ),
    },
}

NON_TOOL: Dict[str, Dict[str, Any]] = {
    "gemma4:e4b": {
        "params": "4B (embed-distill)",
        "family": "gemma4",
        "tool_calling": False,
        "url": DEFAULT_OLLAMA_URL,
        "category": "small_dense_no_tools",
        "version_pin": "ollama:gemma4:e4b@latest",
        "notes": "Google Gemma 4 e4b distill -- no native tool calling; FPR baseline",
    },
    "qwen2.5:7b": {
        "params": "7.6B",
        "family": "qwen2",
        "tool_calling": False,
        "url": DEFAULT_OLLAMA_URL,
        "category": "scale_baseline_no_tools",
        "version_pin": "ollama:qwen2.5:7b@latest",
        "notes": "Instruction-tuned; no reliable tool_calls output; FPR baseline",
    },
    "qwen2.5:1.5b": {
        "params": "1.5B",
        "family": "qwen2",
        "tool_calling": False,
        "url": DEFAULT_OLLAMA_URL,
        "category": "smoke",
        "version_pin": "ollama:qwen2.5:1.5b@latest",
        "notes": (
            "Smallest viable victim; used by run_quickmatrix_demo.py and "
            "scripts/setup/preflight.py as the pipeline smoke target."
        ),
    },
    "qwen2.5:32b": {
        "params": "32B",
        "family": "qwen2",
        "tool_calling": False,
        "url": DEFAULT_OLLAMA_URL,
        "category": "scale_contrast_no_tools",
        "version_pin": "ollama:qwen2.5:32b@latest",
        "notes": "Mid-size baseline without native tool calling (NOT pulled by default)",
    },
}

ALL_MODELS: Dict[str, Dict[str, Any]] = {**TOOL_CAPABLE, **NON_TOOL}


CASE_STUDY_MATRIX: Dict[str, str] = {
    "small_dense_qwen": "qwen3:8b",
    "small_dense_llama": "llama3.1:8b",
    "scale_contrast_intra_family": "qwen3:32b",
    "scale_contrast_cross_family": "nemotron-3-nano:30b",
    "moe_mistral": "mixtral:8x7b",
    "moe_openai": "gpt-oss:20b",
    "moe_google": "gemma4:26b",
    "reasoning_local": "deepseek-r1:7b",
    "reasoning_remote": "deepseek/deepseek-reasoner",
    "closed_frontier_openai": "openai/gpt-4o-2024-11-20",
    "closed_frontier_anthropic": "anthropic/claude-sonnet-4-20250514",
    "smoke": "qwen2.5:1.5b",
}

PROVIDER_BREADTH_MATRIX: Dict[str, str] = {
    "openai": "openai/gpt-4o-2024-11-20",
    "anthropic": "anthropic/claude-sonnet-4-20250514",
    "google": "gemini/gemini-2.5-pro",
    "deepseek": "deepseek/deepseek-chat",
    "meta-llama": "meta-llama/llama-3.3-70b-instruct",
    "mistralai": "mistralai/mixtral-8x7b-instruct",
    "qwen": "qwen/qwen-2.5-72b-instruct",
    "x-ai": "x-ai/grok-2-1212",
}

# THESIS_MULTIMODEL_MATRIX: the 7-victim matrix for the bachelor thesis.
# All models are OpenRouter (cloud mode, no Ollama required).
# Judge = google/gemini-2.5-flash (differs from every victim).
#
# Rationale for each row:
#  - openai/gpt-4o-mini       : main result row, cheap reliable tool-calling
#  - anthropic/claude-haiku-4-5: Anthropic family, cheapest, good RLHF
#  - google/gemma-4-26b-a4b-it:free : free quota, Google family, no cost
#  - deepseek/deepseek-chat    : DeepSeek non-reasoning, popular in papers
#  - meta-llama/llama-3.3-70b-instruct : large open-weight, cross-family
#  - mistralai/mistral-7b-instruct : cheap/weak — higher ASR, wide paper coverage
#  - microsoft/phi-3.5-mini-128k-instruct : smallest, upper-bound ASR row
THESIS_MULTIMODEL_MATRIX: Dict[str, str] = {
    "openai_mini": "openai/gpt-4o-mini",
    "anthropic_haiku": "anthropic/claude-haiku-4-5",
    "google_gemma_free": "google/gemma-4-26b-a4b-it:free",
    "deepseek_chat": "deepseek/deepseek-chat",
    "meta_llama_70b": "meta-llama/llama-3.3-70b-instruct",
    "mistral_7b": "mistralai/mistral-7b-instruct",
    "phi_mini": "microsoft/phi-3.5-mini-128k-instruct",
}

# THESIS_SLIM3_MATRIX: 3-victim cross-vendor slice used in the headline
# vendor-comparison table (T4). These are the models in Overleaf/Generated/
# tab_vendor_comparison.tex. All routed via OpenRouter except qwen3:8b
# which can also run locally via Ollama (alt key: "qwen/qwen3-8b").
THESIS_SLIM3_MATRIX: Dict[str, str] = {
    "gemini_flash_lite": "google/gemini-2.0-flash-lite",
    "llama_3_1_8b": "meta-llama/llama-3.1-8b-instruct",
    "qwen3_8b": "qwen3:8b",  # local Ollama; OpenRouter alt: "qwen/qwen3-8b"
}

THESIS_JUDGE_MODEL: str = "google/gemini-2.5-flash"

# ``PAPER_ANCHOR_MAP``: each cited adaptive-attack / defense paper used a
# specific victim family. To produce the cross-reference column in §5.4
# table T2 ("our ASR vs paper-reported ASR") we map every paper to its
# canonical victim plus the closest-equivalent model in our registry.
# This is metadata only; no runner reads it. Update when new papers are
# added to bibliography.bib.
PAPER_ANCHOR_MAP: Dict[str, Dict[str, str]] = {
    "PAIR (Chao 2023)":          {"paper_victim": "vicuna-13b / llama-2-7b-chat / gpt-4",
                                  "our_proxy":    "llama3.1:8b + openai/gpt-4o-2024-11-20"},
    "TAP (Mehrotra 2023)":       {"paper_victim": "vicuna / llama-2 / gpt-4",
                                  "our_proxy":    "llama3.1:8b + openai/gpt-4o-2024-11-20"},
    "Crescendo (Russinovich 2024)": {"paper_victim": "gpt-4 / gpt-3.5 / llama-2-70b / gemini-pro",
                                  "our_proxy":    "openai/gpt-4o-2024-11-20 + nemotron-3-nano:30b"},
    "AdvPrompter (Paulus 2024)": {"paper_victim": "vicuna-7b/13b / llama-2-7b-chat",
                                  "our_proxy":    "llama3.1:8b"},
    "PoisonedRAG (Zou 2024)":    {"paper_victim": "llama-2-7b / vicuna-7b / gpt-3.5/4",
                                  "our_proxy":    "llama3.1:8b + openai/gpt-4o-2024-11-20"},
    "SmoothLLM (Robey 2023)":    {"paper_victim": "vicuna-13b / llama-2-7b-chat / gpt-3.5/4 / claude",
                                  "our_proxy":    "llama3.1:8b + anthropic/claude-sonnet-4-20250514"},
    "SecAlign (Chen 2024)":      {"paper_victim": "llama-3-8b-instruct / mistral-7b-instruct",
                                  "our_proxy":    "llama3.1:8b + mixtral:8x7b"},
    "MELON (Zhu 2025)":          {"paper_victim": "gpt-4o / llama-3-8b",
                                  "our_proxy":    "openai/gpt-4o-2024-11-20 + llama3.1:8b"},
    "TrustRAG (Zhou 2025)":      {"paper_victim": "llama-3-8b / gpt-4",
                                  "our_proxy":    "llama3.1:8b + openai/gpt-4o-2024-11-20"},
    "A-MemGuard (Wei 2025)":     {"paper_victim": "llama-3 / gpt-4 / multiple",
                                  "our_proxy":    "llama3.1:8b + openai/gpt-4o-2024-11-20"},
    "Circuit Breakers (Zou 2024)": {"paper_victim": "llama-3-8b-instruct / mistral-7b-instruct",
                                  "our_proxy":    "llama3.1:8b + mixtral:8x7b"},
}

# ``CASE_STUDY_MATRIX``: canonical preset for ``scripts/run_model_matrix.py
# --preset case_study``. Every id is verified to be either pullable on
# Ollama (run ``ollama pull <id>`` first) or available via OpenRouter.
# ``PROVIDER_BREADTH_MATRIX``: one OpenRouter slug per vendor family
# (openai/, anthropic/, google|gemini/, deepseek/, meta-llama/, mistralai/,
# qwen/, x-ai/) for full remote provider spread; drift must be recorded in
# ``summary.json``.


def register_model(
    name: str,
    *,
    tool_calling: bool,
    params: str = "?",
    family: str = "remote",
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    notes: str = "",
) -> None:
    """Add or override a model in the registry."""
    entry = {
        "params": params,
        "family": family,
        "tool_calling": tool_calling,
        "url": url or DEFAULT_OLLAMA_URL,
        "notes": notes,
    }
    if api_key:
        entry["api_key"] = api_key
    if headers:
        entry["headers"] = dict(headers)

    ALL_MODELS[name] = entry
    if tool_calling:
        TOOL_CAPABLE[name] = entry
        NON_TOOL.pop(name, None)
    else:
        NON_TOOL[name] = entry
        TOOL_CAPABLE.pop(name, None)


def load_models_config(path: str | Path) -> List[str]:
    """Load model definitions from YAML or JSON.

    The file should contain a top-level ``models`` list, each entry
    being a dict with ``name`` and the optional fields listed above.
    Environment variables in ``api_key`` values are expanded with
    ``${VAR_NAME}`` syntax.

    Returns the list of model names that were registered.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")

    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError(
                "PyYAML required for YAML configs. Install with: pip install pyyaml"
            ) from e
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise ValueError(
            f"{p}: expected top-level 'models' list (got {type(models).__name__})"
        )

    registered: List[str] = []
    for entry in models:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"{p}: each model entry needs a 'name' field")
        name = entry["name"]
        api_key = entry.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var, "")
        register_model(
            name,
            tool_calling=bool(entry.get("tool_calling", False)),
            params=entry.get("params", "?"),
            family=entry.get("family", "remote"),
            url=entry.get("url"),
            api_key=api_key,
            headers=entry.get("headers"),
            notes=entry.get("notes", ""),
        )
        registered.append(name)
    return registered


def get_model_endpoint(model_name: str) -> Dict[str, Any]:
    """Return URL + headers for a model.

    When the registry entry has ``url == "openrouter"`` the function
    resolves it to the real OpenRouter base URL
    (``https://openrouter.ai/api/v1``) and injects the
    ``OPENROUTER_API_KEY`` as a bearer token.  The returned dict also
    carries ``"backend": "openrouter"`` so callers can switch between
    the Ollama and OpenAI-compatible chat APIs.

    Falls back to ``OLLAMA_URL`` env var for unregistered models.
    """
    entry = ALL_MODELS.get(model_name, {})
    url = entry.get("url") or DEFAULT_OLLAMA_URL
    headers = _build_headers(entry.get("api_key"), entry.get("headers"))
    backend = "ollama"

    # Unregistered model with a provider/ slug and an available API key
    # should route through OpenRouter rather than falling back to Ollama.
    if url == DEFAULT_OLLAMA_URL and not entry and "/" in model_name:
        from aaps.attacks._core.config import OPENROUTER_API_KEY as _or_key
        if _or_key:
            url = "openrouter"

    if url == "openrouter":
        from aaps.attacks._core.config import (
            OPENROUTER_API_KEY,
            OPENROUTER_BASE_URL,
        )
        url = OPENROUTER_BASE_URL
        backend = "openrouter"
        if OPENROUTER_API_KEY:
            headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        headers.setdefault("HTTP-Referer", "https://github.com/taraslysun/diploma")
    return {"url": url, "headers": headers, "backend": backend}


def get_available_models(check_remote: bool = True) -> List[str]:
    """Return registered model names that respond on their endpoint.

    When ``check_remote`` is False, returns all registered names without
    pinging endpoints (useful for offline planning).

    OpenRouter models are included when ``OPENROUTER_API_KEY`` is set
    (a single key-presence check; no per-model HTTP probe).
    """
    if not check_remote:
        return list(ALL_MODELS.keys())

    available: List[str] = []
    by_endpoint: Dict[str, List[str]] = {}
    for name, entry in ALL_MODELS.items():
        url = entry.get("url") or DEFAULT_OLLAMA_URL
        by_endpoint.setdefault(url, []).append(name)

    from aaps.attacks._core.config import OPENROUTER_API_KEY

    for url, names in by_endpoint.items():
        if url == "openrouter":
            if OPENROUTER_API_KEY:
                available.extend(names)
            continue
        try:
            entry0 = ALL_MODELS.get(names[0], {})
            headers = _build_headers(entry0.get("api_key"), entry0.get("headers"))
            resp = requests.get(f"{url}/api/tags", timeout=5, headers=headers)
            resp.raise_for_status()
            local = {m["name"] for m in resp.json().get("models", [])}
            for n in names:
                if n in local:
                    available.append(n)
        except Exception:
            pass
    return available


def is_tool_capable(model_name: str) -> bool:
    entry = ALL_MODELS.get(model_name)
    if entry is not None:
        return entry.get("tool_calling", False)
    return False


def tool_capable_models() -> List[str]:
    return list(TOOL_CAPABLE.keys())


def non_tool_models() -> List[str]:
    return list(NON_TOOL.keys())


def is_openrouter(model_name: str) -> bool:
    """True if the registry entry is routed via OpenRouter (not Ollama)."""
    entry = ALL_MODELS.get(model_name, {})
    url = entry.get("url")
    if url == "openrouter":
        return True
    if not entry and "/" in model_name:
        from aaps.attacks._core.config import OPENROUTER_API_KEY
        return bool(OPENROUTER_API_KEY)
    return False


def is_remote(model_name: str) -> bool:
    """True if the model's endpoint is not a local loopback address."""
    entry = ALL_MODELS.get(model_name, {})
    url = entry.get("url") or DEFAULT_OLLAMA_URL
    if url == "openrouter":
        return True
    return not any(loc in url for loc in ("localhost", "127.0.0.1", "::1"))
