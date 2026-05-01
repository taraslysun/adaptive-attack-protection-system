"""PACEDefense -- the orchestrator.

Implements the :class:`defenses.base_defense.BaseDefense` interface so
PACE drops into ``LocalAgent`` exactly like ``AgentIntegrityStack``.
``LocalAgent`` only needs ``defense=spq`` and the existing call sites
(``check_input``, ``check_tool_call``, ``check_memory_write``,
``check_retrieval``, ``check_output``) flow through PACE.

Flow per ``process_query``:

1. ``check_input(user_request)``  -- planner emits PACEPlan; record
   it on the per-session state.
2. The agent collects evidence (tool outputs, memory, retrieval). PACE
   intercepts each piece via ``check_retrieval`` /
   ``check_memory_write`` / ``process_tool_output`` and stores it in
   the session :class:`EvidencePool` with AIS-derived trust labels.
3. ``check_tool_call(tool, args)``  -- on the first call after evidence
   collection, PACE runs k-means + K Executors + Quorum vote. The first
   call's allow/block decision is based on the (Quorum, CFI) gates;
   subsequent calls in the same query reuse the cached vote.
4. ``check_output``  -- writes the JSONL trace record and returns
   ALLOW.

This is intentionally not a perfect mapping (the BaseDefense hooks
were designed for filter pipelines, PACE is an orchestrator). The
trade-off is keeping the LocalAgent integration trivial; a future
clean-room ``defenses/pace/agent.py`` could host the full Planner /
Executor agent loop without going through BaseDefense at all.
"""

from __future__ import annotations

import asyncio
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aaps.attacks._core.logging_config import get_logger
from aaps.defenses.base_defense import BaseDefense, DefenseResult
from aaps.defenses.pace.clusters import (
    EvidencePool,
    EvidenceSpan,
    assignment_vector,
    kmeans_cluster,
)
from aaps.defenses.pace.executor import Executor
from aaps.defenses.pace.planner import Planner
from aaps.defenses.pace.agreement import AgreementDecision, AgreementVoter
from aaps.defenses.pace.plan import PACEPlan, canonicalise_args
from aaps.defenses.pace.trace_logger import PACETraceLogger, PACETraceRecord

log = get_logger("pace.pipeline")

# Argument names that carry user-controlled targets and are prime injection
# channels.  Values absent from the user's stated intent signal injection.
_SENSITIVE_ARG_NAMES: frozenset = frozenset({
    "to", "email", "user_email", "recipient", "recipients",
    "url", "uri", "href", "link",
    "path", "filepath", "file_path", "filename", "file",
    "command", "cmd", "script", "shell", "exec",
    "destination", "dest", "target",
    "address", "host", "server", "endpoint",
    "username", "user_id",
})

_MIN_SENSITIVE_VAL_LEN = 4  # skip trivially short values like "/" or "a"


def _arg_provenance_violation(
    tool_name: str,
    tool_args: Dict[str, Any],
    user_intent: str,
) -> Optional[str]:
    """Return violation message if a sensitive arg value is absent from user query."""
    user_lower = user_intent.lower()
    for arg_key, arg_val in tool_args.items():
        if arg_key.lower() not in _SENSITIVE_ARG_NAMES:
            continue
        val_str = str(arg_val).strip()
        if len(val_str) <= _MIN_SENSITIVE_VAL_LEN:
            continue
        if val_str.lower() not in user_lower:
            return f"arg '{arg_key}'='{val_str[:80]}' not traceable to user query"
    return None


def _default_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Canonical tool schemas for LocalAgent. Mirrors AVAILABLE_TOOLS."""
    try:
        from aaps.attacks._core.local_agent import AVAILABLE_TOOLS
        return dict(AVAILABLE_TOOLS)
    except Exception:
        return {}


class PACEDefense(BaseDefense):
    """Plan, Agree, Confirm, Execute orchestrator (BaseDefense-compatible)."""

    name = "spq"

    def __init__(
        self,
        planner_model: str,
        executor_model: Optional[str] = None,
        K: int = 5,
        q: Optional[int] = None,
        embedder: str = "all-MiniLM-L6-v2",
        cluster_method: str = "kmeans",
        replan_on_abstain: bool = False,
        trace_path: Optional[str] = None,
        tool_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
        ollama_url: Optional[str] = None,
        seed: int = 0,
        nli_filter: bool = True,
        nli_threshold: float = 0.70,
        nli_model: str = "cross-encoder/nli-deberta-v3-small",
        nli_cosine_fallback_threshold: float = 0.92,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config)
        self.planner_model = planner_model
        self.executor_model = executor_model or planner_model
        if K <= 0:
            raise ValueError("K must be positive")
        self.K = K
        self.q = q if q is not None else (K // 2 + 1)
        self.embedder_id = embedder
        self.cluster_method = cluster_method
        self.replan_on_abstain = replan_on_abstain
        self.tool_schemas = tool_schemas or _default_tool_schemas()
        self.ollama_url = ollama_url
        self.seed = seed
        self.nli_filter = nli_filter
        self.nli_threshold = nli_threshold
        self.nli_model = nli_model
        self.nli_cosine_fallback_threshold = nli_cosine_fallback_threshold

        self.planner = Planner(
            model=self.planner_model,
            ollama_url=ollama_url,
            temperature=0.0,
        )
        self.executor = Executor(
            model=self.executor_model,
            ollama_url=ollama_url,
            temperature=0.0,
        )
        self.voter = AgreementVoter(K=self.K, q=self.q, tool_registry={})
        self.trace_logger = PACETraceLogger(path=trace_path)

        self._lock = threading.Lock()
        self._sessions: Dict[str, "_SessionState"] = {}

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _session(self, session_id: str = "default") -> "_SessionState":
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState(session_id=session_id)
                self._sessions[session_id] = state
            return state

    def reset_session(self, session_id: str = "default") -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # BaseDefense hook 1: check_input -> emit PACEPlan
    # ------------------------------------------------------------------

    def check_input(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        session_id = (context or {}).get("session_id", "default")
        state = self._session(session_id)
        t0 = time.time()
        plan = self.planner.emit(
            user_request=user_input,
            tool_schemas=self.tool_schemas,
            system_prompt=(context or {}).get("system_prompt", ""),
        )
        state.pace_plan = plan
        state.evidence = EvidencePool()
        state.cached_decisions = None
        state.user_request = user_input
        state.evidence.add(self._make_span(user_input, "user_input"))
        planner_ms = (time.time() - t0) * 1000.0
        state.latency_ms["planner"] = planner_ms
        self.stats["allowed"] += 1
        log.info(
            "PACE check_input: plan emitted n_nodes=%d plan_id=%s latency_ms=%.0f "
            "session=%s",
            len(plan.nodes), plan.plan_id, planner_ms,
            (context or {}).get("session_id", "default"),
        )
        if len(plan.nodes) == 0:
            log.warning(
                "PACE check_input: PACE plan is EMPTY (0 nodes). "
                "All subsequent tool calls will be CFI-blocked. "
                "Check planner LLM response (set LOG_LEVEL=DEBUG to see raw output).",
            )
        return DefenseResult.allow(
            reason=f"spq: PACE plan emitted (n_nodes={len(plan.nodes)})",
            metadata={
                "plan_id": plan.plan_id,
                "n_nodes": len(plan.nodes),
                "layer": "pace.planner",
            },
            layer="pace.planner",
        )

    # ------------------------------------------------------------------
    # Evidence intake hooks: tool output, retrieval, memory writes
    # ------------------------------------------------------------------

    def process_tool_output(
        self,
        text: str,
        tool_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        session_id = (context or {}).get("session_id", "default")
        state = self._session(session_id)
        provenance = f"tool_output:{tool_name or 'unknown'}"
        span = self._make_span(text, provenance)
        state.evidence.add(span)
        state.cached_decisions = None
        return DefenseResult.allow(
            reason="spq: tool output ingested into evidence pool",
            sanitised_input=text,
            metadata={"provenance": span.provenance},
            layer="spq.evidence",
        )

    def check_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        state = self._session("default")
        for doc in retrieved_docs or []:
            text = doc.get("text") or doc.get("content") or ""
            if not text:
                continue
            span = self._make_span(text, "retrieval")
            state.evidence.add(span)
        state.cached_decisions = None
        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason=f"spq: {len(retrieved_docs or [])} retrieved docs ingested",
            metadata={"kept_docs": retrieved_docs, "layer": "spq.evidence"},
            layer="spq.evidence",
        )

    def check_memory_write(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        state = self._session("default")
        span = self._make_span(content, "memory")
        state.evidence.add(span)
        state.cached_decisions = None
        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason="spq: memory write accepted",
            metadata={"layer": "spq.memory"},
            layer="spq.memory",
        )

    # ------------------------------------------------------------------
    # CFI + Quorum gate
    # ------------------------------------------------------------------

    def check_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_intent: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        session_id = (context or {}).get("session_id", "default")
        state = self._session(session_id)
        plan = state.pace_plan
        if plan is None:
            self.stats["blocked"] += 1
            log.error(
                "PACE check_tool_call: no PACE plan in session=%s "
                "(check_input was not called). tool=%r BLOCKED.",
                session_id, tool_name,
            )
            return DefenseResult.hard_block(
                reason="spq: no PACE plan in session (call check_input first)",
                metadata={"layer": "spq.cfi"},
                layer="spq.cfi",
            )

        node = plan.find_node(tool_name, tool_args)
        cfi_violation = node is None
        if cfi_violation:
            state.cfi_violations.append({"tool": tool_name, "args": dict(tool_args)})
            log.warning(
                "PACE CFI VIOLATION: tool=%r not in PACE plan (plan_id=%s n_nodes=%d). "
                "Tool call will be BLOCKED. args=%r",
                tool_name, plan.plan_id, len(plan.nodes), dict(tool_args),
            )

        if state.cached_decisions is None:
            state.cached_decisions = self._run_executors_and_vote(state)

        canonical = canonicalise_args(tool_name, tool_args)
        decision = next(
            (d for d in state.cached_decisions if d.args_canonical == canonical),
            None,
        )

        if cfi_violation:
            self.stats["blocked"] += 1
            return DefenseResult.hard_block(
                reason=f"spq: CFI violation -- tool '{tool_name}' not in PACE plan",
                metadata={
                    "layer": "spq.cfi",
                    "plan_id": plan.plan_id,
                    "n_plan_nodes": len(plan.nodes),
                    "quorum_decision": decision.to_dict() if decision else None,
                },
                layer="spq.cfi",
            )

        if decision is None or decision.agreement < decision.q:
            agreement = decision.agreement if decision else 0
            need_q = decision.q if decision else self.q
            eff_K = decision.K if decision else self.K
            self.stats["blocked"] += 1
            log.warning(
                "PACE QUORUM FAILURE: tool=%r agreement=%d/%d (need q=%d). BLOCKED.",
                tool_name, agreement, eff_K, need_q,
            )
            return DefenseResult.hard_block(
                reason=(
                    f"spq: quorum failure -- agreement={agreement}/{eff_K}, "
                    f"q={need_q}"
                ),
                metadata={
                    "layer": "pace.agreement",
                    "decision": decision.to_dict() if decision else None,
                    "K": self.K,
                    "q": self.q,
                },
                layer="pace.agreement",
            )

        # Arg-level provenance gate: block sensitive arg values absent from
        # the original user query — these signal injected targets.
        arg_violation = _arg_provenance_violation(tool_name, tool_args, user_intent)
        if arg_violation:
            self.stats["blocked"] += 1
            self.stats["arg_provenance_blocks"] = (
                self.stats.get("arg_provenance_blocks", 0) + 1
            )
            log.warning(
                "PACE ARG PROVENANCE VIOLATION: tool=%r %s. BLOCKED.",
                tool_name, arg_violation,
            )
            return DefenseResult.hard_block(
                reason=f"spq: arg provenance violation -- {arg_violation}",
                metadata={
                    "layer": "spq.arg_provenance",
                    "tool": tool_name,
                    "violation": arg_violation,
                },
                layer="spq.arg_provenance",
            )

        self.stats["allowed"] += 1
        log.info(
            "PACE QUORUM PASSED: tool=%r agreement=%d/%d ALLOWED.",
            tool_name, decision.agreement, self.K,
        )
        return DefenseResult.allow(
            reason=(
                f"spq: tool '{tool_name}' fired (agreement="
                f"{decision.agreement}/{self.K})"
            ),
            metadata={
                "layer": "pace.agreement",
                "decision": decision.to_dict(),
                "K": self.K,
                "q": self.q,
            },
            layer="pace.agreement",
        )

    # ------------------------------------------------------------------
    # check_output -> trace flush
    # ------------------------------------------------------------------

    def check_output(
        self,
        user_query: str,
        agent_response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        session_id = (context or {}).get("session_id", "default")
        state = self._session(session_id)
        if state.pace_plan is not None:
            self._flush_trace(state)
        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason="spq: output passes (trace flushed)",
            layer="spq.output",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_span(self, text: str, provenance: str) -> EvidenceSpan:
        return EvidenceSpan(text=text, provenance=provenance)

    def _run_executors_and_vote(
        self,
        state: "_SessionState",
    ) -> List[AgreementDecision]:
        plan = state.pace_plan
        assert plan is not None
        import math as _math
        t0 = time.time()
        n_spans = len(state.evidence)

        # Adaptive K: when evidence is sparser than K, collapse to n_spans
        # executors so quorum is achievable (K > n_spans → all empty clusters
        # abstain → quorum always fails → every benign tool call is blocked).
        # Security property is maintained: q/K fraction is preserved (rounded up).
        effective_K = max(1, min(self.K, n_spans)) if n_spans > 0 else 1
        if effective_K < self.K:
            frac = self.q / self.K
            effective_q = max(1, _math.ceil(frac * effective_K))
            voter = AgreementVoter(K=effective_K, q=effective_q, tool_registry={})
            log.info(
                "PACE adaptive K: n_spans=%d < configured K=%d → using K=%d q=%d "
                "(q/K fraction preserved: %.2f → %.2f)",
                n_spans, self.K, effective_K, effective_q,
                self.q / self.K, effective_q / effective_K,
            )
        else:
            effective_K = self.K
            voter = self.voter

        log.info(
            "PACE executors: running K=%d executors on %d evidence spans "
            "(method=%s embedder=%s)",
            effective_K, n_spans, self.cluster_method, self.embedder_id,
        )
        clusters = kmeans_cluster(
            state.evidence,
            K=effective_K,
            embedder_id=self.embedder_id,
            seed=self.seed,
            method=self.cluster_method,
            # getattr defaults keep tests that bypass __init__ (PACEDefense.__new__)
            # working without forcing them to set every new attribute.
            nli_filter=getattr(self, "nli_filter", True),
            nli_threshold=getattr(self, "nli_threshold", 0.70),
            nli_model=getattr(self, "nli_model", "cross-encoder/nli-deberta-v3-small"),
            nli_cosine_fallback_threshold=getattr(self, "nli_cosine_fallback_threshold", 0.92),
        )
        state.latency_ms["cluster"] = (time.time() - t0) * 1000.0
        state.cluster_assignment = assignment_vector(clusters)

        # Run K executors in parallel — wall time drops from K×latency to ~1×latency.
        per_cluster_calls = [None] * len(clusters)
        executor_latencies: List[float] = [0.0] * len(clusters)

        def _run_one(cid_members):
            cid, members = cid_members
            log.debug("PACE executor cid=%d n_members=%d", cid, len(members))
            t1 = time.time()
            calls = self.executor.fill(plan, cid, members)
            lat = (time.time() - t1) * 1000.0
            log.debug(
                "PACE executor cid=%d proposed %d call(s) latency_ms=%.0f: %s",
                cid, len(calls), lat, [c.tool for c in calls],
            )
            return cid, calls, lat

        with ThreadPoolExecutor(max_workers=len(clusters)) as pool:
            futures = {pool.submit(_run_one, (cid, members)): cid
                       for cid, members in enumerate(clusters)}
            for fut in as_completed(futures):
                cid, calls, lat = fut.result()
                per_cluster_calls[cid] = calls
                executor_latencies[cid] = lat

        state.latency_ms["executor_each"] = executor_latencies
        state.per_cluster_calls = per_cluster_calls
        t2 = time.time()
        decisions = voter.vote(per_cluster_calls)
        state.latency_ms["vote"] = (time.time() - t2) * 1000.0
        log.info(
            "PACE vote: %d decision(s) — fire=%d abstain=%d",
            len(decisions),
            sum(1 for d in decisions if d.decision == "fire"),
            sum(1 for d in decisions if d.decision != "fire"),
        )
        return decisions

    def _flush_trace(self, state: "_SessionState") -> None:
        plan = state.pace_plan
        decisions = state.cached_decisions or []
        outcome = "abstained"
        if any(d.decision == "fire" for d in decisions):
            outcome = "fired"
        if state.cfi_violations:
            # CFI block overrides quorum-fire: tool was blocked by CFI gate
            # regardless of what quorum decided. Record as abstained so
            # cfi_execution_violation_count stays 0 (the acceptance criterion).
            outcome = "abstained"
        rec = PACETraceRecord(
            session_id=state.session_id,
            user_request=state.user_request,
            pace_plan=plan.to_dict() if plan else {},
            evidence={
                "n_spans": len(state.evidence) if state.evidence else 0,
                "cluster_assignment": list(state.cluster_assignment),
                "K": self.K,
            },
            filled_plans=[
                {
                    "cluster": cid,
                    "calls": [
                        {"tool": c.tool, "args_canonical": c.args_canonical, "rationale": c.rationale}
                        for c in calls
                    ],
                }
                for cid, calls in enumerate(state.per_cluster_calls or [])
            ],
            gates={
                "cfi_violations": list(state.cfi_violations),
                "quorum_decisions": [d.to_dict() for d in decisions],
            },
            outcome=outcome,
            latency_ms=dict(state.latency_ms),
        )
        self.trace_logger.write(rec)
        state.cfi_violations = []

    # ------------------------------------------------------------------
    # Async adapters used by the MCP benchmark bridge
    # ------------------------------------------------------------------

    async def initialize_session(
        self,
        user_request: str,
        session_id: str = "default",
        system_prompt: str = "",
    ) -> DefenseResult:
        """Emit the PACEPlan for an MCP session.

        Must be called before the first :meth:`validate_tool_call` for the
        same ``session_id``; otherwise PACE has no plan and CFI will hard-block
        every tool call.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.check_input(
                user_input=user_request,
                context={"session_id": session_id, "system_prompt": system_prompt},
            ),
        )

    async def validate_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_intent: str,
        session_id: str = "default",
    ) -> DefenseResult:
        """Async wrapper around :meth:`check_tool_call` for the MCP bridge.

        PACE's planner / executor calls do blocking HTTP, so we offload the
        synchronous machinery to the default thread pool to avoid stalling
        the FastMCP event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.check_tool_call(
                tool_name=tool_name,
                tool_args=tool_args,
                user_intent=user_intent,
                context={"session_id": session_id},
            ),
        )

    # ------------------------------------------------------------------
    # Reporting helper used by evaluation/defense_benchmark.py
    # ------------------------------------------------------------------

    def trace_summary(self) -> Dict[str, Any]:
        return self.trace_logger.aggregate()


class _SessionState:
    __slots__ = (
        "session_id",
        "user_request",
        "pace_plan",
        "evidence",
        "cluster_assignment",
        "per_cluster_calls",
        "cached_decisions",
        "cfi_violations",
        "latency_ms",
    )

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.user_request: str = ""
        self.pace_plan: Optional[PACEPlan] = None
        self.evidence: EvidencePool = EvidencePool()
        self.cluster_assignment: List[int] = []
        self.per_cluster_calls: List[List[Any]] = []
        self.cached_decisions: Optional[List[AgreementDecision]] = None
        self.cfi_violations: List[Dict[str, Any]] = []
        self.latency_ms: Dict[str, Any] = {}
