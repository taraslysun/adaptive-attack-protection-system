"""Prompt-based defenses (Spotlighting, Prompt Sandwiching).

Thesis remediation ``implement-missing-baselines``. The Spotlighting
class previously only stored a delimiter and did not actually wrap
untrusted spans on any hook. This is a faithful reproduction of the
``datamarking`` variant from Hines et al., "Defending Against Indirect
Prompt Injection Attacks With Spotlighting", arXiv:2403.14720 (2024):
the defense interleaves a high-entropy marker token between every
character of untrusted text so the planner can identify (and the
attacker cannot strip) the data span.

The two flavours from §3 of the paper:

* ``delimiting`` -- ASCII delimiter band around the span (already what
  ``wrap_trusted_text`` did before). Cheap and used as a sanity
  baseline.
* ``datamarking`` -- single-character marker interleaved between every
  character of the untrusted text. Requires a marker token unlikely to
  appear in user content. We use ``"\u200B"`` (zero-width space)
  by default so the marked span renders identically to the original
  for the attacker but is trivially recognisable to the planner.

Both modes route through ``check_input`` and ``check_retrieval`` so the
matrix runner sees a real ``soft_filter`` decision (rather than a
no-op ``allow``).
"""

from typing import Dict, Any, Optional, List

from aaps.defenses.base_defense import BaseDefense, DefenseResult, Severity


class Spotlighting(BaseDefense):
    """Hines 2024 Spotlighting (delimiting + datamarking variants)."""

    name = "spotlighting"
    SUPPORTED_MODES = ("delimiting", "datamarking")

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        mode: str = "datamarking",
        delimiter: str = "<<<TRUSTED>>>",
        marker: str = "\u200B",
    ) -> None:
        super().__init__(config)
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Spotlighting mode must be one of {self.SUPPORTED_MODES}; "
                f"got {mode!r}."
            )
        self.mode = mode
        self.delimiter = delimiter
        self.marker = marker

    def _mark(self, text: str) -> str:
        if self.mode == "delimiting":
            return f"{self.delimiter}{text}{self.delimiter}"
        return self.marker.join(text)

    def wrap_trusted_text(self, text: str) -> str:
        """Public helper: wrap a span in the chosen marker."""
        return self._mark(text)

    def wrap_untrusted(self, text: str) -> str:
        """Spotlighting *itself* doesn't distinguish trusted/untrusted;
        the caller supplies the untrusted span and we mark it.
        """
        return self._mark(text)

    def check_input(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        marked = self._mark(user_input)
        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            severity=Severity.SOFT_FILTER,
            confidence=0.5,
            reason=f"Spotlighting ({self.mode}) marked input span",
            sanitised_input=marked,
            metadata={"mode": self.mode, "marker_len": len(self.marker)},
        )

    def check_memory_write(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        marked = self._mark(content)
        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            severity=Severity.SOFT_FILTER,
            confidence=0.5,
            reason=(
                f"Spotlighting ({self.mode}) applied to memory write content"
            ),
            sanitised_input=marked,
            metadata={"mode": self.mode},
        )

    def check_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        kept: List[Dict[str, Any]] = []
        for d in retrieved_docs:
            d2 = dict(d)
            txt = d2.get("text") or d2.get("content") or ""
            if "text" in d2:
                d2["text"] = self._mark(txt)
            elif "content" in d2:
                d2["content"] = self._mark(txt)
            else:
                d2["text"] = self._mark(txt)
            kept.append(d2)
        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            severity=Severity.SOFT_FILTER,
            confidence=0.5,
            reason=(
                f"Spotlighting ({self.mode}) marked {len(kept)} retrieved docs"
            ),
            metadata={"mode": self.mode, "kept_docs": kept},
        )


class PromptSandwiching(BaseDefense):
    """Prompt Sandwiching defense by repeating the user prompt.

    Source
    ------
    Folk technique with no canonical paper; tracked under the placeholder
    bibkey ``% TODO[bib]: prompt-sandwiching`` (see
    ``docs/bibliography_justification.md`` "Forecast"). The trick is also
    discussed as a "sandwich reminder" inside the StruQ paper
    (``chen2024struq``).

    Hook coverage (thesis remediation
    ``codebase-fix-prompt-sandwiching-noops``)
    ------------------------------------------
    Sandwiching is *purely* a prompt-construction technique applied at
    inference time: it inserts the legitimate user instruction both
    before and after the untrusted span. As such:

    * ``sandwich_prompt(user_prompt, untrusted_input)`` is the only
      meaningful entry point.
    * ``check_memory_write`` and ``check_retrieval`` are deliberately
      *no-ops* (return ALLOW) because the defense never inspects or
      filters those surfaces. The stub implementations exist so the
      common :class:`BaseDefense` hook contract is satisfied and the
      matrix runner does not raise ``AttributeError`` when iterating
      defenses; do *not* read them as "Sandwiching examined the
      retrieved docs and approved them". When the thesis matrix tables
      report Sandwiching results for the memory or retrieval surface
      they should be marked ``n/a`` (defense out of scope) rather than
      ``allow``. The AIS L1 layer in ``defenses/integrity/l1_channels.py``
      absorbs the sandwich-reminder helper for the prompt surface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def check_memory_write(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        """No-op (out of scope). See class docstring."""
        self.stats["total_checks"] += 1
        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason="PromptSandwiching: out of scope for memory writes (no-op)",
            confidence=0.0,
            metadata={"hook_status": "out_of_scope"},
        )

    def sandwich_prompt(self, user_prompt: str, untrusted_input: str) -> str:
        """Sandwich untrusted input between two copies of the user prompt."""
        return f"{user_prompt}\n\n{untrusted_input}\n\n{user_prompt}"

    def check_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        """No-op (out of scope). See class docstring."""
        self.stats["total_checks"] += 1
        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason="PromptSandwiching: out of scope for retrieval (no-op)",
            confidence=0.0,
            metadata={"hook_status": "out_of_scope", "kept_docs": list(retrieved_docs)},
        )
