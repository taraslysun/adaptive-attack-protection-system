"""Planner stage: emits a typed PACEPlan from trusted input only.

The Planner LLM is the first stage of PACE. By construction it sees
only:

* the user's raw request,
* the static tool schemas for tools the agent is allowed to use,
* the system prompt.

It does NOT see tool outputs, retrieved documents, memory entries, or
any other byte that an attacker could control. This is the "control-
plane isolation" half of the PACE contribution; together with the
Executor stage's "data-plane isolation per cluster" it is the
architectural reason adaptive prompt-injection becomes a coordinated-
injection problem rather than a single-payload problem.

The Planner can call the same Ollama URL the agent uses (default), or
any pluggable callable supplied via the ``llm_call`` constructor
argument (used by the test suite to inject a deterministic stub).

The fall-back JSON parser is intentionally permissive: if the LLM
emits prose around the JSON, we extract the largest balanced JSON
object; if no JSON is present, we degrade to a *trivial plan* that
admits no tool calls (hard-fail-safe). The CFI gate then rejects all
tool calls and the agent can either re-plan or refuse.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

import requests

from aaps.attacks._core.logging_config import get_logger
from aaps.defenses.pace.plan import PACEPlan, PACEPlanNode

log = get_logger("pace.planner")


_PLANNER_SYSTEM = """You are the Planner half of a Plan, Agree, Confirm, Execute agent.

Your only job is to emit a JSON PACEPlan describing which tool calls
the Executor is permitted to make in order to fulfil the user request.

Rules:
1. You will NEVER read tool outputs, retrieved documents, or memory.
2. You will only output a JSON object with exactly the fields below.
3. You will be conservative: include the smallest set of nodes
   sufficient to fulfil the request. Each extra node is an extra
   surface the attacker can target.
4. Each node names exactly one tool from the provided schemas with a
   typed `arg_schema` (use one of: str, int, float, bool, list[str],
   list[int], dict, any). Do NOT inline literal argument values.

Output schema:
{
  "nodes": [
    {
      "tool": "<tool_name>",
      "arg_schema": {"<arg_name>": "<type>"},
      "post_condition": "<one-line free-form note>"
    }
  ],
  "notes": "<one-line free-form note for the audit log>"
}
"""


class Planner:
    """LLM-driven Planner that emits a typed PACEPlan."""

    def __init__(
        self,
        model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 768,
        llm_call: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> None:
        if not model:
            model = os.environ.get("PACE_PLANNER_MODEL", "google/gemini-2.5-flash")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from aaps.attacks._core.model_registry import get_model_endpoint
            ep = get_model_endpoint(model)
            self.ollama_url = ollama_url or ep["url"]
            self._backend = ep.get("backend", "ollama")
            self._headers: Dict[str, str] = dict(ep.get("headers", {}))
            log.debug("planner model=%s backend=%s url=%s", model, self._backend, self.ollama_url)
        except Exception as exc:
            self.ollama_url = ollama_url or os.environ.get(
                "OLLAMA_URL", "http://127.0.0.1:11434"
            )
            self._backend = "ollama"
            self._headers = {}
            log.warning(
                "planner model_registry lookup failed for %r (%s); "
                "defaulting to ollama at %s",
                model, exc, self.ollama_url,
            )

        self.llm_call = llm_call or self._default_llm_call

    def _default_llm_call(self, messages: List[Dict[str, str]]) -> str:
        import time as _time
        if self._backend == "openrouter":
            return self._call_openrouter(messages)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        import time as _time
        t0 = _time.time()
        log.debug("planner Ollama call model=%s url=%s", self.model, self.ollama_url)
        try:
            r = requests.post(
                f"{self.ollama_url}/api/chat", json=payload,
                headers=self._headers, timeout=120,
            )
            r.raise_for_status()
            msg = r.json().get("message", {})
            content = msg.get("content") or msg.get("thinking") or ""
            latency = (_time.time() - t0) * 1000
            log.debug(
                "planner Ollama response model=%s latency_ms=%.0f "
                "response_len=%d",
                self.model, latency, len(content),
            )
            from aaps.evaluation.call_logger import get_call_logger
            get_call_logger().log_call(
                role="planner", model=self.model,
                prompt=messages, response=content,
                latency_ms=latency,
            )
            return content
        except Exception as exc:
            latency = (_time.time() - t0) * 1000
            log.error(
                "planner Ollama FAILED model=%s latency_ms=%.0f error=%s",
                self.model, latency, exc,
            )
            from aaps.evaluation.call_logger import get_call_logger
            get_call_logger().log_call(
                role="planner", model=self.model,
                prompt=messages, response="",
                latency_ms=latency,
                error=str(exc),
            )
            return json.dumps({"nodes": [], "notes": f"planner-llm-error: {exc}"})

    def _call_openrouter(self, messages: List[Dict[str, str]]) -> str:
        import time as _time
        from aaps.evaluation.call_logger import get_call_logger
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        t0 = _time.time()
        for attempt in range(4):
            try:
                r = requests.post(
                    f"{self.ollama_url}/chat/completions",
                    json=payload, headers=self._headers, timeout=120,
                )
                if r.status_code == 429:
                    _time.sleep(2 ** attempt + 1)
                    continue
                r.raise_for_status()
                data = r.json()
                content = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage", {})
                get_call_logger().log_call(
                    role="planner", model=self.model,
                    prompt=messages, response=content,
                    latency_ms=(_time.time() - t0) * 1000,
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=usage.get("completion_tokens", 0),
                    attempt=attempt + 1,
                )
                return content
            except Exception as exc:
                if attempt < 3:
                    _time.sleep(2 ** attempt)
                    continue
                get_call_logger().log_call(
                    role="planner", model=self.model,
                    prompt=messages, response="",
                    latency_ms=(_time.time() - t0) * 1000,
                    error=str(exc),
                )
                return json.dumps({"nodes": [], "notes": f"planner-llm-error: {exc}"})

    def emit(
        self,
        user_request: str,
        tool_schemas: Dict[str, Dict[str, Any]],
        system_prompt: str = "",
    ) -> PACEPlan:
        """Return a PACEPlan for ``user_request``.

        ``tool_schemas`` is the same dict the agent uses, e.g.
        ``{"send_email": {"description": "...", "parameters": [...]}}``.
        We re-format it for the Planner prompt so it is self-contained.
        """
        prompt_payload = {
            "system": system_prompt,
            "user_request": user_request,
            "tool_schemas": tool_schemas,
        }
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ]
        raw = self.llm_call(messages)
        plan = _parse_plan_or_fallback(raw, user_request, tool_schemas)
        return plan


def _extract_json(text: str) -> Optional[str]:
    """Return the first balanced JSON object substring in ``text`` or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _parse_plan_or_fallback(
    raw: str,
    user_request: str,
    tool_schemas: Dict[str, Dict[str, Any]],
) -> PACEPlan:
    payload: Dict[str, Any] = {}
    candidate = _extract_json(raw or "")
    if candidate:
        try:
            payload = json.loads(candidate)
        except Exception:
            payload = {}
    nodes_raw: Any = None
    if isinstance(payload, dict):
        nodes_raw = payload.get("nodes")
        if not isinstance(nodes_raw, list):
            for key in ("plan", "steps", "tools"):
                alt = payload.get(key)
                if isinstance(alt, list):
                    nodes_raw = alt
                    break
    notes = payload.get("notes", "") if isinstance(payload, dict) else ""
    if not isinstance(nodes_raw, list):
        preview = (raw or "")[:120]
        log.warning(
            "planner produced empty/malformed plan (no 'nodes' list); "
            "falling back to empty plan. raw_preview=%r", preview,
        )
        return PACEPlan(
            nodes=[],
            user_request=user_request,
            notes=f"empty-or-malformed-plan; raw_preview={raw[:80]!r}",
        )
    allowed_tools = set(tool_schemas.keys())
    nodes: List[PACEPlanNode] = []
    for n in nodes_raw:
        if not isinstance(n, dict):
            log.debug("planner skipping non-dict node: %r", n)
            continue
        tool = str(n.get("tool", "")).strip()
        if not tool:
            log.debug("planner skipping node with missing tool name")
            continue
        if tool not in allowed_tools:
            log.warning(
                "planner node references unknown tool %r (allowed: %s). "
                "Node skipped — check planner LLM output.",
                tool, sorted(allowed_tools),
            )
            continue
        schema_raw = n.get("arg_schema") or {}
        if not isinstance(schema_raw, dict):
            log.warning(
                "planner node tool=%r has non-dict arg_schema %r; skipping.",
                tool, schema_raw,
            )
            continue
        arg_schema = {str(k): str(v) for k, v in schema_raw.items()}
        if not arg_schema:
            params = tool_schemas[tool].get("parameters", []) or []
            arg_schema = {str(p): "str" for p in params}
        post = str(n.get("post_condition", ""))
        nodes.append(
            PACEPlanNode(tool=tool, arg_schema=arg_schema, post_condition=post)
        )
    plan = PACEPlan(nodes=nodes, user_request=user_request, notes=str(notes))
    skipped = len([n for n in nodes_raw if isinstance(n, dict)]) - len(nodes)
    log.debug(
        "planner parsed plan n_nodes=%d skipped=%d plan_id=%s",
        len(nodes), skipped, plan.plan_id,
    )
    if skipped > 0:
        log.warning(
            "planner skipped %d node(s) — see individual node warnings above",
            skipped,
        )
    return plan


# Back-compat and smoke tests that import the planner under this name
PACEPlanner = Planner
