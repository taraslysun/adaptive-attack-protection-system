"""AgentDojo pipeline elements that wrap PACEDefense.

Three elements plug PACE into AgentDojo's AgentPipeline:

  PACEInitElement     — runs once before ToolsExecutionLoop:
                       calls check_input() → emits PACEPlan,
                       stores session_id in extra_args.

  PACEGateElement     — inside ToolsExecutionLoop, BEFORE ToolsExecutor:
                       calls check_tool_call() for each pending tool call;
                       HARD_BLOCK → removes the call and injects a synthetic
                       blocked result so ToolsExecutor has nothing to execute.

  PACEEvidenceElement — inside ToolsExecutionLoop, AFTER ToolsExecutor:
                       calls process_tool_output() for each new tool result;
                       advances a cursor to avoid double-processing.

Usage::

    from aaps.defenses.pace import PACEDefense
    from aaps.defenses.pace.agentdojo_pipeline import (
        PACEInitElement, PACEGateElement, PACEEvidenceElement,
    )
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    import openai

    spq = PACEDefense(planner_model="google/gemini-2.5-flash", K=5, q=3)
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)
    pipeline = AgentPipeline([
        SystemMessage(system_msg),
        InitQuery(),
        PACEInitElement(spq),
        ToolsExecutionLoop([
            OpenAILLM(client, model_id),
            PACEGateElement(spq),
            ToolsExecutor(),
            PACEEvidenceElement(spq),
        ]),
    ])
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Dict

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime, FunctionCall
from agentdojo.types import (
    ChatMessage,
    ChatAssistantMessage,
    ChatToolResultMessage,
)

try:
    from agentdojo.types import text_content_block_from_string
except ImportError:
    # fallback for older agentdojo versions
    def text_content_block_from_string(s: str):  # type: ignore[misc]
        return {"type": "text", "text": s}

from aaps.defenses.pace.pipeline import PACEDefense


def _extract_tool_schemas(runtime: FunctionsRuntime) -> Dict[str, Dict[str, Any]]:
    """Convert FunctionsRuntime functions to PACE tool_schemas format."""
    schemas: Dict[str, Dict[str, Any]] = {}
    for name, fn in runtime.functions.items():
        try:
            param_schema = fn.parameters.model_json_schema()
        except Exception:
            param_schema = {}
        schemas[name] = {
            "description": fn.description or "",
            "parameters": param_schema,
        }
    return schemas


def _tool_result_content(msg: ChatToolResultMessage) -> str:
    """Extract plain text from a tool result message's content blocks."""
    parts = []
    for block in (msg.get("content") or []):
        if isinstance(block, dict) and block.get("type") == "text":
            # agentdojo uses "content" key; fall back to "text" for compat
            parts.append(block.get("content") or block.get("text", ""))
        elif hasattr(block, "content"):
            parts.append(block.content or "")
        elif hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


class PACEInitElement(BasePipelineElement):
    """Initialises PACE session before the tool execution loop.

    Calls ``PACEDefense.check_input()`` with the user query and the runtime's
    tool schemas to generate the PACEPlan. Stores a fresh ``session_id``
    in ``extra_args["spq_session_id"]`` for downstream elements.
    """

    name = "spq_init"

    def __init__(self, defense: PACEDefense) -> None:
        self.defense = defense

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        session_id = extra_args.get("spq_session_id") or str(uuid.uuid4())

        # Override tool schemas with actual suite tools for this run.
        tool_schemas = _extract_tool_schemas(runtime)
        old_schemas = self.defense.tool_schemas
        self.defense.tool_schemas = tool_schemas

        system_prompt = ""
        if messages and messages[0].get("role") == "system":
            blocks = messages[0].get("content") or []
            system_prompt = " ".join(
                b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                for b in blocks
            )

        self.defense.check_input(
            query,
            context={
                "session_id": session_id,
                "tool_schemas": tool_schemas,
                "system_prompt": system_prompt,
            },
        )

        # Restore original schemas so other sessions are unaffected.
        self.defense.tool_schemas = old_schemas

        new_extra = {
            **extra_args,
            "spq_session_id": session_id,
            "spq_evidence_cursor": 0,
        }
        return query, runtime, env, messages, new_extra


class PACEGateElement(BasePipelineElement):
    """Intercepts LLM tool calls and applies CFI + quorum gate.

    Placed BETWEEN the LLM element and ToolsExecutor inside
    ``ToolsExecutionLoop``. For each tool_call in the latest assistant message:

    - ALLOW  → call passes through unchanged.
    - HARD_BLOCK → tool_call is removed from the assistant message and a
      synthetic ``ChatToolResultMessage`` is injected so ``ToolsExecutor``
      sees nothing to execute for that call and the LLM receives a clear
      "blocked" signal.
    """

    name = "spq_gate"

    def __init__(self, defense: PACEDefense) -> None:
        self.defense = defense

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if not messages:
            return query, runtime, env, messages, extra_args

        last = messages[-1]
        if last.get("role") != "assistant":
            return query, runtime, env, messages, extra_args

        tool_calls: list[FunctionCall] | None = last.get("tool_calls")
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        session_id = extra_args.get("spq_session_id", "default")
        context = {"session_id": session_id}

        allowed_calls: list[FunctionCall] = []
        synthetic_results: list[ChatToolResultMessage] = []

        for tc in tool_calls:
            args = dict(tc.args) if tc.args else {}
            result = self.defense.check_tool_call(
                tool_name=tc.function,
                tool_args=args,
                user_intent=query,
                context=context,
            )
            if result.allowed:
                allowed_calls.append(tc)
            else:
                # Inject blocked result so the LLM sees the refusal.
                blocked_msg: ChatToolResultMessage = {
                    "role": "tool",
                    "tool_call": tc,
                    "tool_call_id": tc.id,
                    "content": [
                        text_content_block_from_string(
                            f"[BLOCKED by PACE defense: {result.reason}]"
                        )
                    ],
                    "error": None,
                }
                synthetic_results.append(blocked_msg)

        if not synthetic_results:
            # Nothing was blocked — pass through unchanged.
            return query, runtime, env, messages, extra_args

        if not allowed_calls:
            # ALL calls blocked.  Return a terminal assistant message with no
            # tool_calls so ToolsExecutionLoop exits cleanly after this iteration.
            # OpenAI rejects assistant messages with content=null and no tool_calls,
            # so always provide a non-null content string.
            terminal: ChatAssistantMessage = {
                **last,  # type: ignore[misc]
                "content": [
                    text_content_block_from_string(
                        "[PACE defense blocked all proposed tool calls. "
                        "Cannot proceed with the current request.]"
                    )
                ],
                "tool_calls": None,
            }
            return query, runtime, env, [*messages[:-1], terminal], extra_args

        # PARTIAL block: only allowed_calls remain.  Replace the assistant
        # message's tool_calls with the allowed subset; blocked calls are
        # silently dropped so OpenAI never sees mismatched tool_call IDs.
        modified_assistant: ChatAssistantMessage = {
            **last,  # type: ignore[misc]
            "tool_calls": allowed_calls,
        }
        return query, runtime, env, [*messages[:-1], modified_assistant], extra_args


class PACEEvidenceElement(BasePipelineElement):
    """Feeds tool results into PACE's evidence pool after ToolsExecutor.

    Placed AFTER ``ToolsExecutor`` inside ``ToolsExecutionLoop``. Uses
    ``extra_args["spq_evidence_cursor"]`` to process only newly-added tool
    result messages each iteration.
    """

    name = "spq_evidence"

    def __init__(self, defense: PACEDefense) -> None:
        self.defense = defense

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        session_id = extra_args.get("spq_session_id", "default")
        cursor: int = extra_args.get("spq_evidence_cursor", 0)
        context = {"session_id": session_id}

        new_cursor = cursor
        for i, msg in enumerate(messages):
            if i < cursor:
                continue
            if msg.get("role") == "tool":
                tool_name = None
                tc = msg.get("tool_call")
                if tc is not None:
                    tool_name = getattr(tc, "function", None) or tc.get("function")
                content = _tool_result_content(msg)  # type: ignore[arg-type]
                self.defense.process_tool_output(
                    text=content,
                    tool_name=tool_name,
                    context=context,
                )
            new_cursor = i + 1

        new_extra = {**extra_args, "spq_evidence_cursor": new_cursor}
        return query, runtime, env, messages, new_extra
