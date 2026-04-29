#!/usr/bin/env python3
"""External-service health dashboard.

Usage:
    python scripts/check_services.py
    python scripts/check_services.py --json

Probes Qdrant, Ollama, sandbox container, OpenRouter, OpenAI, Anthropic,
Gemini, HuggingFace, WandB, LiteLLM proxy. Returns 0 if every reachable
service is healthy; non-zero if a *configured* service fails. Services with
no env-var configuration are reported as "skipped" but do not fail the
overall exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Callable, Optional


@dataclass
class Probe:
    name: str
    status: str       # "ok" | "fail" | "skip"
    detail: str = ""


def _http_ok(url: str, headers: dict | None = None, timeout: int = 3) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return False, str(e)


def probe_qdrant() -> Probe:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    ok, detail = _http_ok(f"{url}/healthz")
    return Probe("qdrant", "ok" if ok else "fail", f"{url} → {detail}")


def probe_ollama() -> Probe:
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    ok, detail = _http_ok(f"{url}/api/tags")
    return Probe("ollama", "ok" if ok else "fail", f"{url} → {detail}")


def probe_sandbox() -> Probe:
    try:
        out = subprocess.run(
            ["docker", "exec", "aaps-sandbox", "echo", "ok"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and "ok" in out.stdout:
            return Probe("sandbox", "ok", "container responds")
        return Probe("sandbox", "fail", out.stderr.strip() or "no output")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return Probe("sandbox", "fail", str(e))


def _probe_keyed(name: str, env_var: str, url: str, header: str = "Authorization") -> Probe:
    key = os.environ.get(env_var)
    if not key:
        return Probe(name, "skip", f"{env_var} not set")
    headers = {header: f"Bearer {key}"}
    ok, detail = _http_ok(url, headers=headers, timeout=5)
    return Probe(name, "ok" if ok else "fail", detail)


def probe_openrouter() -> Probe:
    return _probe_keyed("openrouter", "OPENROUTER_API_KEY",
                        "https://openrouter.ai/api/v1/models")


def probe_openai() -> Probe:
    return _probe_keyed("openai", "OPENAI_API_KEY",
                        "https://api.openai.com/v1/models")


def probe_anthropic() -> Probe:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return Probe("anthropic", "skip", "ANTHROPIC_API_KEY not set")
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    ok, detail = _http_ok("https://api.anthropic.com/v1/models", headers=headers)
    return Probe("anthropic", "ok" if ok else "fail", detail)


def probe_gemini() -> Probe:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return Probe("gemini", "skip", "GEMINI_API_KEY/GOOGLE_API_KEY not set")
    ok, detail = _http_ok(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
    )
    return Probe("gemini", "ok" if ok else "fail", detail)


def probe_huggingface() -> Probe:
    return _probe_keyed("huggingface", "HF_TOKEN",
                        "https://huggingface.co/api/whoami-v2")


def probe_wandb() -> Probe:
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        return Probe("wandb", "skip", "WANDB_API_KEY not set")
    ok, detail = _http_ok("https://api.wandb.ai/", timeout=3)
    return Probe("wandb", "ok" if ok else "fail", detail)


def probe_litellm_proxy() -> Probe:
    url = os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000")
    ok, detail = _http_ok(f"{url}/health/liveliness")
    if ok:
        return Probe("litellm-proxy", "ok", detail)
    # Many people don't run the proxy; treat unreachable as skip not fail.
    return Probe("litellm-proxy", "skip", detail)


PROBES: list[Callable[[], Probe]] = [
    probe_qdrant,
    probe_ollama,
    probe_sandbox,
    probe_openrouter,
    probe_openai,
    probe_anthropic,
    probe_gemini,
    probe_huggingface,
    probe_wandb,
    probe_litellm_proxy,
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="emit JSON")
    args = p.parse_args()

    results = [fn() for fn in PROBES]

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        widths = max((len(r.name) for r in results), default=12)
        print(f"{'service':<{widths}}  status  detail")
        print(f"{'-' * widths}  ------  ------")
        for r in results:
            mark = {"ok": "✓ ok", "fail": "✗ FAIL", "skip": "·skip"}[r.status]
            print(f"{r.name:<{widths}}  {mark:<6}  {r.detail}")

    # Exit non-zero only if a *configured* service is failing.
    failed = [r for r in results if r.status == "fail"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
