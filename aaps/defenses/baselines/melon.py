"""MELON defense -- masked re-execution / dual-run consistency.

Source
------
Zhu, K. *et al.*, "MELON: Indirect Prompt Injection Defense via Masked
Re-execution and Tool Dependency Analysis", ICML 2025
(arXiv:2502.05174, https://arxiv.org/abs/2502.05174). Bibkey
``zhu2025melon``.

Tested as a baseline alongside the Nasr et al. "The Attacker Moves
Second" suite (arXiv:2510.09023, bibkey ``nasr2025attacker``).

This baseline reproduces a simpler tool-name set-overlap variant of
MELON. A separate AIS layer ``defenses/integrity/l3_action_consistency.py``
implements an embedding-cosine-on-tool-arguments variant; the two
share a paper but are DIFFERENT algorithms and must be reported under
distinct columns in the thesis tables.

Detects indirect prompt injection by running the model twice:
  1. Normal run: record all tool calls and outputs.
  2. Verification run: replace the user prompt with a benign dummy
     ("summarize the tool outputs") and check whether the same
     tool calls are issued.

If the verification run still issues tool calls from the first run,
then the tool output itself likely contains an injection.
"""

from typing import Dict, Any, Optional, List, Callable, Set
import re

from aaps.defenses.base_defense import BaseDefense, DefenseResult


SENSITIVE_TOOL_NAMES = {
    "send_email", "delete_file", "transfer_money", "invite_user",
    "execute_code", "write_file", "create_user", "modify_permissions",
    "forward_email", "download_file", "run_command", "update_settings",
}


class MELONDefense(BaseDefense):
    """MELON dual-run consistency defense for tool-calling agents."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        agent_fn: Optional[Callable] = None,
        dummy_prompt: str = "Please summarize the information provided above.",
    ):
        super().__init__(config)
        self.agent_fn = agent_fn
        self.dummy_prompt = dummy_prompt

    def check_tool_calls(
        self,
        user_prompt: str,
        normal_tool_calls: List[Dict[str, Any]],
    ) -> DefenseResult:
        """Compare normal-run tool calls with a verification run."""
        self.stats["total_checks"] += 1

        if not normal_tool_calls:
            self.stats["allowed"] += 1
            return DefenseResult(
                allowed=True, confidence=1.0,
                reason="No tool calls to verify", metadata={},
            )

        if self.agent_fn is not None:
            return self._dual_run_check(user_prompt, normal_tool_calls)

        return self._heuristic_check_tool_calls(normal_tool_calls)

    def _dual_run_check(
        self,
        user_prompt: str,
        normal_tool_calls: List[Dict[str, Any]],
    ) -> DefenseResult:
        """Actual dual-run check: re-run with dummy prompt and compare."""
        try:
            verify_response = self.agent_fn(self.dummy_prompt)
            verify_tool_calls = verify_response.get("tool_calls", [])
        except Exception:
            verify_tool_calls = []

        normal_names: Set[str] = {tc.get("name", "") for tc in normal_tool_calls}
        verify_names: Set[str] = {tc.get("name", "") for tc in verify_tool_calls}

        overlap = normal_names & verify_names
        if overlap:
            self.stats["blocked"] += 1
            return DefenseResult(
                allowed=False,
                confidence=0.85,
                reason=f"MELON: verification run repeated tool calls: {overlap}",
                metadata={"overlapping_tools": list(overlap)},
            )

        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            confidence=0.9,
            reason="MELON: tool calls differ between runs",
            metadata={"normal_tools": list(normal_names)},
        )

    def check_input(self, user_input: str) -> DefenseResult:
        """Check input text for embedded tool-call instructions."""
        self.stats["total_checks"] += 1

        detected_tools = self._extract_tool_references(user_input)
        sensitive_refs = detected_tools & SENSITIVE_TOOL_NAMES
        if sensitive_refs:
            self.stats["blocked"] += 1
            return DefenseResult(
                allowed=False,
                confidence=0.7,
                reason=f"MELON: input references sensitive tools: {sensitive_refs}",
                metadata={"detected_tools": list(sensitive_refs)},
            )

        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True, confidence=0.8,
            reason="MELON: no sensitive tool references in input",
            metadata={},
        )

    def check_output(self, user_input: str, output: str) -> DefenseResult:
        """Check if the output indicates tool calls were made."""
        self.stats["total_checks"] += 1

        detected_tools = self._extract_tool_references(output)
        sensitive_refs = detected_tools & SENSITIVE_TOOL_NAMES
        if sensitive_refs:
            self.stats["blocked"] += 1
            return DefenseResult(
                allowed=False,
                confidence=0.6,
                reason=f"MELON: output references sensitive tools: {sensitive_refs}",
                metadata={"detected_tools": list(sensitive_refs)},
            )

        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True, confidence=0.8,
            reason="MELON: output passed check", metadata={},
        )

    @staticmethod
    def _extract_tool_references(text: str) -> Set[str]:
        """Extract tool name references from text."""
        lower = text.lower()
        found = set()
        for tool in SENSITIVE_TOOL_NAMES:
            if tool in lower or tool.replace("_", " ") in lower:
                found.add(tool)
        tool_call_pattern = re.findall(r"['\"](\w+)['\"]", text)
        for name in tool_call_pattern:
            if name.lower() in SENSITIVE_TOOL_NAMES:
                found.add(name.lower())
        return found

    @staticmethod
    def _heuristic_check_tool_calls(
        tool_calls: List[Dict[str, Any]],
    ) -> DefenseResult:
        """Heuristic fallback: flag sensitive tool call patterns."""
        flagged = [
            tc for tc in tool_calls
            if tc.get("name", "").lower() in SENSITIVE_TOOL_NAMES
        ]
        if flagged:
            return DefenseResult(
                allowed=False,
                confidence=0.7,
                reason=f"MELON heuristic: {len(flagged)} suspicious tool call(s)",
                metadata={"flagged_tools": [t["name"] for t in flagged]},
            )
        return DefenseResult(
            allowed=True, confidence=0.8,
            reason="Tool calls passed heuristic check", metadata={},
        )

    def check_memory_write(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True, confidence=0.5,
            reason="MELON focuses on tool call verification", metadata={},
        )

    def check_retrieval(
        self, query: str, retrieved_docs: List[Dict[str, Any]]
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        flagged = 0
        for doc in retrieved_docs:
            text = doc.get("text", "") or doc.get("content", "")
            refs = self._extract_tool_references(text)
            sensitive = refs & SENSITIVE_TOOL_NAMES
            if sensitive:
                flagged += 1

        if flagged:
            self.stats["blocked"] += flagged
            return DefenseResult(
                allowed=False,
                confidence=0.7,
                reason=f"MELON: {flagged} docs contain tool call instructions",
                metadata={"flagged_count": flagged},
            )

        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True, confidence=0.8,
            reason="MELON: retrieval passed check", metadata={},
        )
