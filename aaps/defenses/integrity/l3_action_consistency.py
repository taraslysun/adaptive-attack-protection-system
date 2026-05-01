"""L3 Action Consistency -- MELON-inspired masked re-execution.

Source
------
Zhu et al., "MELON: Indirect Prompt Injection Defense via Masked
Re-execution and Tool Dependency Analysis", ICML 2025
(arXiv:2502.05174, https://arxiv.org/abs/2502.05174). Bibkey
``zhu2025melon``.

A separate baseline ``defenses/baselines/melon.py`` reproduces a
simpler tool-name set-overlap variant; AIS L3 (this file) implements
the embedding-cosine-on-tool-arguments variant. The two implementations
share a paper but are DIFFERENT algorithms; in thesis prose they must
be reported under distinct columns.

Idea
----
A successful indirect prompt injection produces tool calls that
*depend on the injected data, not on the user's task*.  We detect this
by re-running the planner with a **masked** user prompt (a neutral
filler like ``"summarise the information provided"``) and comparing
the resulting tool calls with the originals:

* If the masked re-execution produces the *same* dangerous tool call
  (same name, similar args), the call cannot come from the legitimate
  user task -- the injection has hijacked the planner.

The original MELON paper reports a similarity threshold of about 0.8
on tool-call embeddings.  We expose this as a parameter.

What this layer needs from the agent
------------------------------------
* ``agent_fn(prompt: str) -> dict`` -- the planner replay function.
  It must return ``{"tool_calls": [{"name": str, "args": dict}, ...]}``.
  ``LocalAgent`` wires this automatically (thesis remediation
  ``implement-real-melon-l3``), so the matrix runs always exercise the
  paper-faithful masked-re-execution branch.  When you call this
  defense from a harness that does not wire it (e.g. unit tests), we
  fall back to the *heuristic* MELON-lite branch implemented in
  ``defenses/baselines/melon.py``: any sensitive tool whose args
  overlap with recently observed untrusted text is flagged.  The
  decision metadata records ``"branch": "melon" | "heuristic"`` so
  downstream analyses can split the two cleanly.

Adaptive-attack note
--------------------
MELON itself is bypassed by attacks that make the injection text appear
to *also* satisfy the user task (so the masked re-run no longer matches).
Combined with L1 + L2, this raises the joint cost noticeably; on its
own it is not robust. See Nasr et al. "The Attacker Moves Second"
(arXiv:2510.09023, bibkey ``nasr2025attacker``) for the umbrella
adaptive-attack methodology.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Set

from aaps.defenses.base_defense import BaseDefense, DefenseResult


_SENSITIVE_TOOLS: Set[str] = {
    "send_email",
    "delete_file",
    "transfer_money",
    "execute_code",
    "write_file",
    "invite_user",
    "add_user_to_channel",
    "create_user",
    "modify_permissions",
    "forward_email",
    "download_file",
    "run_command",
    "update_settings",
    "publish",
    "post_message",
    "share_file",
    "reset_password",
}


def _cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class ActionConsistencyDefense(BaseDefense):
    """L3: masked re-execution + embedding similarity over tool calls."""

    name = "L3_action_consistency"
    LAYER_ID = "L3"
    DEFAULT_MASK_PROMPT = (
        "Please summarise the information that was provided to you above."
    )

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        agent_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        mask_prompt: Optional[str] = None,
        similarity_threshold: float = 0.8,
        sensitive_tools: Optional[Set[str]] = None,
        encoder_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        super().__init__(config)
        self.agent_fn = agent_fn
        self.mask_prompt = mask_prompt or self.DEFAULT_MASK_PROMPT
        self.similarity_threshold = similarity_threshold
        self.sensitive_tools = set(sensitive_tools or _SENSITIVE_TOOLS)
        self.encoder_name = encoder_name
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.encoder_name)
            except Exception:
                self._encoder = False
        return self._encoder if self._encoder else None

    @staticmethod
    def _tool_repr(tool_name: str, tool_args: Dict[str, Any]) -> str:
        kvs = ", ".join(f"{k}={v}" for k, v in sorted((tool_args or {}).items()))
        return f"{tool_name}({kvs})"

    def _embed_tool(self, tool_name: str, tool_args: Dict[str, Any]):
        enc = self._get_encoder()
        if enc is None:
            return None
        return enc.encode(self._tool_repr(tool_name, tool_args))

    def _is_sensitive(self, tool_name: str) -> bool:
        return tool_name.lower() in self.sensitive_tools

    # ------------------------------------------------------------------
    # MELON dual-run.
    # ------------------------------------------------------------------

    def _run_masked(self) -> List[Dict[str, Any]]:
        if self.agent_fn is None:
            return []
        try:
            response = self.agent_fn(self.mask_prompt)
        except Exception:
            return []
        if isinstance(response, dict):
            return response.get("tool_calls", []) or []
        return []

    def _compare_calls(
        self,
        original: Dict[str, Any],
        masked_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return the best-matching masked call for ``original``."""
        orig_name = original.get("name", "")
        orig_args = original.get("args", {}) or original.get("arguments", {})
        orig_emb = self._embed_tool(orig_name, orig_args)

        best_sim = 0.0
        best_match: Optional[Dict[str, Any]] = None
        for tc in masked_calls:
            name = tc.get("name", "")
            args = tc.get("args", {}) or tc.get("arguments", {})
            if name != orig_name:
                continue
            if orig_emb is None:
                sim = 1.0
            else:
                emb = self._embed_tool(name, args)
                sim = _cosine(orig_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_match = {"name": name, "args": args, "similarity": sim}

        return {
            "best_similarity": best_sim,
            "best_match": best_match,
            "n_masked_calls": len(masked_calls),
        }

    # ------------------------------------------------------------------
    # Heuristic fallback.
    # ------------------------------------------------------------------

    def _heuristic_check(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_intent: str,
    ) -> Dict[str, Any]:
        """When no replay agent is available, use the MELON-lite heuristic.

        Flag any sensitive tool whose serialised args do not share at
        least one substantive token with the user intent.
        """
        if not self._is_sensitive(tool_name):
            return {"should_block": False, "reason": "tool not in sensitive list"}

        intent_tokens = {
            t for t in user_intent.lower().split() if len(t) > 3
        }
        arg_text = self._tool_repr(tool_name, tool_args).lower()
        arg_tokens = {t for t in arg_text.split() if len(t) > 3}
        overlap = intent_tokens & arg_tokens
        if not overlap:
            return {
                "should_block": True,
                "reason": (
                    "sensitive tool with no token overlap with user intent"
                ),
                "intent_token_count": len(intent_tokens),
            }
        return {
            "should_block": False,
            "reason": "sensitive tool whose args overlap with intent",
            "overlap": sorted(overlap),
        }

    # ------------------------------------------------------------------
    # Hooks.
    # ------------------------------------------------------------------

    def check_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_intent: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        t0 = time.perf_counter()
        self.stats["total_checks"] += 1

        if not self._is_sensitive(tool_name):
            latency = (time.perf_counter() - t0) * 1000.0
            self.stats["allowed"] += 1
            return DefenseResult.allow(
                reason=f"L3: '{tool_name}' is not in sensitive list",
                metadata={"layer": self.LAYER_ID, "tool_name": tool_name},
                layer=self.LAYER_ID,
                latency_ms=latency,
            )

        if self.agent_fn is None:
            heur = self._heuristic_check(tool_name, tool_args, user_intent)
            latency = (time.perf_counter() - t0) * 1000.0
            if heur["should_block"]:
                self.stats["blocked"] += 1
                return DefenseResult.hard_block(
                    reason=f"L3 (heuristic): {heur['reason']}",
                    metadata={
                        "layer": self.LAYER_ID,
                        "branch": "heuristic",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "heuristic": heur,
                    },
                    layer=self.LAYER_ID,
                    latency_ms=latency,
                )
            self.stats["allowed"] += 1
            return DefenseResult.allow(
                reason=f"L3 (heuristic): {heur['reason']}",
                metadata={
                    "layer": self.LAYER_ID,
                    "branch": "heuristic",
                    "tool_name": tool_name,
                    "heuristic": heur,
                },
                layer=self.LAYER_ID,
                latency_ms=latency,
            )

        masked = self._run_masked()
        comparison = self._compare_calls(
            {"name": tool_name, "args": tool_args}, masked
        )
        latency = (time.perf_counter() - t0) * 1000.0

        sim = comparison["best_similarity"]
        if comparison["best_match"] is not None and sim >= self.similarity_threshold:
            self.stats["blocked"] += 1
            return DefenseResult.hard_block(
                reason=(
                    f"L3 (MELON): masked re-run repeats '{tool_name}' "
                    f"(sim={sim:.2f} >= {self.similarity_threshold:.2f}); "
                    "tool call likely hijacked by injection"
                ),
                metadata={
                    "layer": self.LAYER_ID,
                    "branch": "melon",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    **comparison,
                },
                layer=self.LAYER_ID,
                latency_ms=latency,
            )

        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason=(
                f"L3 (MELON): masked re-run differs (best sim={sim:.2f} < "
                f"{self.similarity_threshold:.2f})"
            ),
            metadata={
                "layer": self.LAYER_ID,
                "branch": "melon",
                "tool_name": tool_name,
                **comparison,
            },
            layer=self.LAYER_ID,
            latency_ms=latency,
        )

    def check_tool_calls(
        self,
        user_prompt: str,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        if not tool_calls:
            return DefenseResult.allow(
                reason="L3: no tool calls to verify",
                metadata={"layer": self.LAYER_ID},
                layer=self.LAYER_ID,
            )
        t0 = time.perf_counter()
        masked = self._run_masked() if self.agent_fn else []
        worst: Optional[DefenseResult] = None
        per_call: List[Dict[str, Any]] = []
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {}) or tc.get("arguments", {})
            if self.agent_fn is not None:
                cmp = self._compare_calls({"name": name, "args": args}, masked)
                sim = cmp["best_similarity"]
                blocked = (
                    cmp["best_match"] is not None
                    and sim >= self.similarity_threshold
                )
                per_call.append(
                    {"tool": name, "sim": sim, "blocked": blocked, "cmp": cmp}
                )
                if blocked and (worst is None or sim > worst.metadata.get("sim", 0)):
                    worst = DefenseResult.hard_block(
                        reason=(
                            f"L3 (MELON): masked re-run repeats '{name}' "
                            f"(sim={sim:.2f})"
                        ),
                        metadata={
                            "layer": self.LAYER_ID,
                            "tool_name": name,
                            "sim": sim,
                            **cmp,
                        },
                        layer=self.LAYER_ID,
                    )
            else:
                heur = self._heuristic_check(name, args, user_prompt)
                per_call.append({"tool": name, "heuristic": heur})
                if heur["should_block"] and worst is None:
                    worst = DefenseResult.hard_block(
                        reason=f"L3 (heuristic): {heur['reason']}",
                        metadata={
                            "layer": self.LAYER_ID,
                            "tool_name": name,
                            "heuristic": heur,
                        },
                        layer=self.LAYER_ID,
                    )
        latency = (time.perf_counter() - t0) * 1000.0
        if worst is not None:
            self.stats["blocked"] += 1
            worst.latency_ms = latency
            worst.metadata["per_call"] = per_call
            return worst
        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason="L3: tool batch passed",
            metadata={"layer": self.LAYER_ID, "per_call": per_call},
            layer=self.LAYER_ID,
            latency_ms=latency,
        )
