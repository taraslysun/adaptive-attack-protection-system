"""Agent Integrity Stack (AIS) pipeline orchestrator.

Composes the six layers in the order shown in the architecture diagram
of the thesis plan:

* ``check_input``     -> L1 (delimiter-escape on user input)
* ``check_output``    -> L4 (intent consistency)
* ``check_tool_call`` -> L3 (action consistency, MELON dual-run)
* ``process_tool_output`` (custom) -> L2 (probe) then L1 (wrap)
* ``check_memory_write`` -> L1 (escape) then L5 (ensemble)
* ``check_retrieval`` -> L6 (cluster + KB cross-check) then L1 (wrap)

Each layer hook is wrapped so the trace logger sees a record per call
and the AFL captures every HARD_BLOCK.

Layers can be turned on/off via the constructor flags
``enabled_layers={"L1", "L2", ...}``; the orchestrator simply skips
disabled layers.  This is the mechanism used by the per-layer ablation
sweep in the evaluation harness.

Trace attribution convention
----------------------------
Every per-layer hook emits a :class:`DefenseResult` with ``layer`` set
to its own ID (``"L1"`` ... ``"L6"``); these are the rows the trace
logger records and the rows that drive the per-layer block /
soft-filter attribution tables. The orchestrator's *return* value, on
the other hand, is labelled ``layer="AIS"`` -- it summarises the joint
verdict of all enabled layers. ALLOW results that escape the pipeline
therefore look like ``layer="AIS"`` in caller-side trace records;
that is intentional. When you need to know which individual layer
allowed a call, read the per-hook records inside ``self.trace``, not
the orchestrator's return value (thesis remediation
``codebase-fix-trace-allow-attribution``).

Known evaluation limitations (Threats to Validity)
---------------------------------------------------
* **L3 (MELON dual-run)**: When constructed standalone the default
  ``AgentIntegrityStack()`` has no ``agent_fn``, so the heuristic
  token-overlap branch fires.  ``LocalAgent`` calls
  :py:meth:`AgentIntegrityStack.set_agent_fn` from its constructor
  (thesis remediation ``implement-real-melon-l3``) so the masked
  re-execution branch from the MELON paper runs in the matrix runs.
  When you instantiate AIS in a non-LocalAgent harness, install
  ``set_agent_fn`` (or pass ``agent_fn=...`` to the constructor) to
  get the paper-faithful behaviour.
* **L6 (Retrieval Guard)**: Real cluster + cross-check executes when
  ``answer_fn`` is wired (also done by ``LocalAgent`` automatically).
  ``check_retrieval`` is invoked from ``DeepAgent`` (full RAG path) and
  from ``LocalAgent._format_memory_context`` on the memory-as-retrieval
  surface (thesis remediation ``codebase-fix-l6-not-wired-locally``).
  When neither agent has any RAG / memory traffic the hook is simply
  never invoked, so the layer reports zero traffic; that is honest.
* **Tool calls are regex-parsed when the model has no native tool
  calling**: ``LocalAgent._detect_tool_calls`` scans free-text output
  for tool-name substrings.  When ``native_tools=True`` we instead
  read ``message.tool_calls`` from Ollama's API.  L3 decisions are
  exact under native tools and approximate under regex.
* **DeepAgent has no tool execution loop**: ``_extract_tool_calls``
  parses proposed calls but never executes them or feeds results back
  into the reasoning chain.  Agentic attack evaluation against
  DeepAgent tests intent-detection only.
* **Category ASR is non-independent**: when ``run_all_attacks.py``
  pools 5 attack types per goal into a single category ASR, entries
  are correlated.  Report per-attack-type ASR alongside category.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from aaps.defenses.base_defense import BaseDefense, DefenseResult, Severity
from aaps.defenses.integrity.l1_channels import ChannelSeparationDefense
from aaps.defenses.integrity.l2_probe import ToolOutputProbeDefense
from aaps.defenses.integrity.l3_action_consistency import ActionConsistencyDefense
from aaps.defenses.integrity.l4_output_consistency import OutputConsistencyDefense
from aaps.defenses.integrity.l5_memory_guard import MemoryWriteGuardDefense
from aaps.defenses.integrity.l6_retrieval_guard import RetrievalIntegrityDefense
from aaps.defenses.integrity.trace_logger import TraceLogger
from aaps.defenses.integrity.adaptive_feedback import AdaptiveFeedbackLearner


_ALL_LAYERS: Set[str] = {"L1", "L2", "L3", "L4", "L5", "L6"}


class AgentIntegrityStack(BaseDefense):
    """Six-layer defense orchestrator with trace logging and AFL."""

    name = "agent_integrity_stack"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        enabled_layers: Optional[Iterable[str]] = None,
        l1: Optional[ChannelSeparationDefense] = None,
        l2: Optional[ToolOutputProbeDefense] = None,
        l3: Optional[ActionConsistencyDefense] = None,
        l4: Optional[OutputConsistencyDefense] = None,
        l5: Optional[MemoryWriteGuardDefense] = None,
        l6: Optional[RetrievalIntegrityDefense] = None,
        trace_logger: Optional[TraceLogger] = None,
        adaptive_feedback: Optional[AdaptiveFeedbackLearner] = None,
        agent_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        answer_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__(config)
        self.enabled_layers: Set[str] = set(enabled_layers or _ALL_LAYERS)

        self.l1 = l1 or ChannelSeparationDefense()
        self.l2 = l2 or ToolOutputProbeDefense()
        self.l3 = l3 or ActionConsistencyDefense(agent_fn=agent_fn)
        self.l4 = l4 or OutputConsistencyDefense()
        self.l5 = l5 or MemoryWriteGuardDefense()
        self.l6 = l6 or RetrievalIntegrityDefense(answer_fn=answer_fn)

        if agent_fn is not None and self.l3.agent_fn is None:
            self.l3.agent_fn = agent_fn
        if answer_fn is not None and self.l6.answer_fn is None:
            self.l6.answer_fn = answer_fn

        self.trace = trace_logger or TraceLogger()
        self.afl = adaptive_feedback or AdaptiveFeedbackLearner(
            learners=[self.l2, self.l5]
        )

    # ------------------------------------------------------------------
    # Plumbing helpers.
    # ------------------------------------------------------------------

    def set_agent_fn(self, agent_fn: Callable[[str], Dict[str, Any]]) -> None:
        self.l3.agent_fn = agent_fn

    def set_answer_fn(self, answer_fn: Callable[[str], str]) -> None:
        self.l6.answer_fn = answer_fn

    def enable(self, layer: str) -> None:
        self.enabled_layers.add(layer)

    def disable(self, layer: str) -> None:
        self.enabled_layers.discard(layer)

    def is_enabled(self, layer: str) -> bool:
        return layer in self.enabled_layers

    def stack_summary(self) -> Dict[str, Any]:
        return {
            "enabled_layers": sorted(self.enabled_layers),
            "afl": self.afl.stats(),
            "trace": {
                "block_attribution": self.trace.block_attribution(),
                "soft_filter_attribution": self.trace.soft_filter_attribution(),
                "latency": self.trace.latency_summary(),
            },
        }

    def _record(
        self,
        hook: str,
        result: DefenseResult,
        attack_text: Optional[str] = None,
    ) -> DefenseResult:
        self.trace.log(hook, result)
        if (
            result.severity == Severity.HARD_BLOCK
            and attack_text
            and self.afl is not None
        ):
            self.afl.observe_block(attack_text, layer=result.layer)
        return result

    # ------------------------------------------------------------------
    # BaseDefense hooks.
    # ------------------------------------------------------------------

    def check_input(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        if "L1" in self.enabled_layers:
            res = self.l1.check_input(user_input, context)
            self._record("input", res, attack_text=user_input)
            if not res.allowed:
                return res
        return DefenseResult.allow(
            reason="AIS: input passed", layer="AIS"
        )

    def check_output(
        self,
        user_query: str,
        agent_response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        if "L4" in self.enabled_layers:
            res = self.l4.check_output(user_query, agent_response, context)
            self._record("output", res, attack_text=agent_response)
            if not res.allowed:
                return res
        return DefenseResult.allow(reason="AIS: output passed", layer="AIS")

    def check_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_intent: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        if "L3" in self.enabled_layers:
            res = self.l3.check_tool_call(tool_name, tool_args, user_intent, context)
            self._record(
                "tool_call",
                res,
                attack_text=f"{tool_name}({tool_args})",
            )
            if not res.allowed:
                return res
        return DefenseResult.allow(
            reason=f"AIS: tool call '{tool_name}' passed", layer="AIS"
        )

    def check_tool_calls(
        self,
        user_prompt: str,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        if "L3" not in self.enabled_layers or not tool_calls:
            return DefenseResult.allow(reason="AIS: no L3 / no tools", layer="AIS")
        res = self.l3.check_tool_calls(user_prompt, tool_calls, context)
        self._record("tool_call", res)
        if not res.allowed:
            return res
        return DefenseResult.allow(reason="AIS: tool batch passed", layer="AIS")

    def check_memory_write(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        if "L1" in self.enabled_layers:
            res = self.l1.check_memory_write(content, metadata)
            self._record("memory_write", res, attack_text=content)
            if not res.allowed:
                return res
        if "L5" in self.enabled_layers:
            res = self.l5.check_memory_write(content, metadata)
            self._record("memory_write", res, attack_text=content)
            if not res.allowed:
                return res
        return DefenseResult.allow(
            reason="AIS: memory write passed", layer="AIS"
        )

    def check_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        docs = list(retrieved_docs)
        soft_filter = False
        last_meta: Dict[str, Any] = {}

        if "L6" in self.enabled_layers:
            r6 = self.l6.check_retrieval(query, docs)
            self._record("retrieval", r6)
            if r6.severity == Severity.HARD_BLOCK:
                return r6
            if r6.severity == Severity.SOFT_FILTER:
                docs = r6.metadata.get("kept_docs", docs)
                soft_filter = True
                last_meta["L6"] = r6.metadata

        if "L1" in self.enabled_layers:
            r1 = self.l1.check_retrieval(query, docs)
            self._record("retrieval", r1)
            if r1.severity == Severity.HARD_BLOCK:
                return r1
            if r1.severity == Severity.SOFT_FILTER:
                docs = r1.metadata.get("wrapped_docs", docs)
                soft_filter = True
                last_meta["L1"] = r1.metadata

        if "L2" in self.enabled_layers:
            r2 = self.l2.check_retrieval(query, docs)
            self._record("retrieval", r2)
            if r2.severity == Severity.HARD_BLOCK:
                return r2
            if r2.severity == Severity.SOFT_FILTER:
                docs = r2.metadata.get("kept_docs", docs)
                soft_filter = True
                last_meta["L2"] = r2.metadata

        last_meta["final_doc_count"] = len(docs)
        if soft_filter:
            return DefenseResult.soft_filter(
                reason=(
                    f"AIS retrieval pipeline filtered docs to "
                    f"{len(docs)} kept"
                ),
                metadata={
                    "layer": "AIS",
                    "kept_docs": docs,
                    "per_layer": last_meta,
                },
                layer="AIS",
            )
        return DefenseResult.allow(
            reason=f"AIS: retrieval passed ({len(docs)} docs)",
            metadata={"layer": "AIS", "kept_docs": docs, "per_layer": last_meta},
            layer="AIS",
        )

    # ------------------------------------------------------------------
    # Custom hook used by agent wiring.
    # ------------------------------------------------------------------

    def process_tool_output(
        self,
        tool_output: str,
        tool_name: Optional[str] = None,
    ) -> DefenseResult:
        """L2-then-L1 pipeline for raw tool outputs.

        Returns a SOFT_FILTER with the wrapped output ready to insert
        into the planner context, or a HARD_BLOCK with the offending
        diagnostics.
        """
        if "L2" in self.enabled_layers:
            r2 = self.l2.check_tool_output(tool_output, tool_name=tool_name)
            self._record("tool_output", r2, attack_text=tool_output)
            if not r2.allowed:
                return r2

        if "L1" in self.enabled_layers:
            wrapped = self.l1.wrap_untrusted(tool_output)
            res = DefenseResult.soft_filter(
                reason=f"AIS: tool output wrapped (tool={tool_name})",
                sanitised_input=wrapped,
                metadata={
                    "layer": "AIS",
                    "tool_name": tool_name,
                    "nonce": self.l1.nonce,
                },
                layer="AIS",
            )
            self.trace.log("tool_output", res)
            return res

        return DefenseResult.allow(
            reason="AIS: tool output passed (L1/L2 disabled)",
            metadata={"layer": "AIS", "tool_name": tool_name},
            layer="AIS",
        )
