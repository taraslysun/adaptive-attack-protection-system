"""SmoothLLM defense (Robey et al., 2023).

Reference: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas,
"SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks",
arXiv:2310.03684 (2023).

Idea (paper abstract): adversarial suffixes are *brittle* under
character-level perturbation. SmoothLLM samples ``n`` perturbed copies
of the input (random insertion / swap / patch), queries the model on
each, and returns the majority vote. The vote is over a coarse
"refusal" classifier; if a majority of perturbed copies refuse, the
original is treated as adversarial and HARD_BLOCKed.

This module reproduces the *defense decision* path of SmoothLLM:

1. ``perturbations`` -- sample ``n`` perturbed copies of the input
   (default ``swap``/``insert``/``patch`` modes from §4 of the paper).
2. ``classifier`` -- a refusal classifier function ``cls(text) -> bool``
   that the caller supplies. We default to a simple keyword-based
   ``_keyword_refusal`` matcher so the defense is usable without the
   victim model in the loop. Pass ``classifier=lambda text: ...`` to
   plug in a real model-side oracle.

The perturbation-vote classifier is what runs in ``check_input``.
``check_output`` is a no-op (paper acts before generation), and
``check_retrieval`` simply applies the same perturbation vote to each
retrieved document text.

Honest reproduction status (thesis remediation
``implement-missing-baselines``): we reproduce the input-side
perturbation-vote contract but do *not* call the victim LLM ``n`` times
per query; that is left to the caller via ``classifier``. The default
keyword classifier is a cheap proxy that lets the matrix runner exercise
the decision branch end-to-end.
"""

from __future__ import annotations

import random
import string
from typing import Any, Callable, Dict, List, Optional

from aaps.defenses.base_defense import BaseDefense, DefenseResult, Severity


_PRINTABLE = string.printable.strip()


_DEFAULT_REFUSAL_KEYWORDS = (
    "i cannot", "i can't", "i won't", "i refuse", "sorry, but",
    "as an ai", "i'm not able", "cannot provide", "not appropriate",
    "ignore previous", "harmful", "unsafe",
)


def _keyword_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in _DEFAULT_REFUSAL_KEYWORDS)


def _swap(text: str, q: float, rng: random.Random) -> str:
    out = list(text)
    for i in range(len(out)):
        if rng.random() < q:
            out[i] = rng.choice(_PRINTABLE)
    return "".join(out)


def _insert(text: str, q: float, rng: random.Random) -> str:
    out = []
    for ch in text:
        out.append(ch)
        if rng.random() < q:
            out.append(rng.choice(_PRINTABLE))
    return "".join(out)


def _patch(text: str, q: float, rng: random.Random) -> str:
    if not text:
        return text
    n = max(1, int(round(q * len(text))))
    start = rng.randint(0, max(0, len(text) - n))
    patch = "".join(rng.choice(_PRINTABLE) for _ in range(n))
    return text[:start] + patch + text[start + n:]


_PERTURB_MODES = {"swap": _swap, "insert": _insert, "patch": _patch}


class SmoothLLMDefense(BaseDefense):
    """Robey 2023 SmoothLLM perturbation-vote defense."""

    name = "smoothllm"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        n_perturbations: int = 6,
        perturbation_rate: float = 0.10,
        modes: Optional[List[str]] = None,
        majority_threshold: float = 0.5,
        classifier: Optional[Callable[[str], bool]] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(config)
        if n_perturbations <= 0:
            raise ValueError("n_perturbations must be > 0")
        if not (0.0 < perturbation_rate < 1.0):
            raise ValueError("perturbation_rate must be in (0, 1)")
        for m in modes or []:
            if m not in _PERTURB_MODES:
                raise ValueError(
                    f"unknown perturbation mode {m!r}; allowed: "
                    f"{sorted(_PERTURB_MODES)}"
                )
        self.n_perturbations = n_perturbations
        self.perturbation_rate = perturbation_rate
        self.modes = list(modes or list(_PERTURB_MODES))
        self.majority_threshold = majority_threshold
        self.classifier = classifier or _keyword_refusal
        self._rng = random.Random(seed)

    def perturbations(self, text: str) -> List[str]:
        out: List[str] = []
        for _ in range(self.n_perturbations):
            mode = self._rng.choice(self.modes)
            out.append(_PERTURB_MODES[mode](text, self.perturbation_rate, self._rng))
        return out

    def _vote(self, text: str) -> Dict[str, Any]:
        copies = self.perturbations(text)
        votes = [bool(self.classifier(c)) for c in copies]
        n = len(votes)
        n_refuse = sum(1 for v in votes if v)
        frac = n_refuse / n if n else 0.0
        decided_adversarial = frac >= self.majority_threshold
        return {
            "n": n,
            "n_refuse": n_refuse,
            "fraction_refuse": frac,
            "threshold": self.majority_threshold,
            "decided_adversarial": decided_adversarial,
        }

    def check_input(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        vote = self._vote(user_input)
        if vote["decided_adversarial"]:
            self.stats["blocked"] += 1
            return DefenseResult.hard_block(
                reason=(
                    f"SmoothLLM: {vote['n_refuse']}/{vote['n']} perturbed "
                    f"copies classified as refusal "
                    f"(>= {self.majority_threshold:.2f})"
                ),
                metadata={"vote": vote, "mode": self.modes},
            )
        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason=(
                f"SmoothLLM: {vote['n_refuse']}/{vote['n']} perturbed "
                f"copies refused (< {self.majority_threshold:.2f})"
            ),
            metadata={"vote": vote, "mode": self.modes},
        )

    def check_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        if not retrieved_docs:
            self.stats["allowed"] += 1
            return DefenseResult.allow(
                reason="SmoothLLM: no docs to inspect",
                metadata={},
            )
        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        per_doc: List[Dict[str, Any]] = []
        for d in retrieved_docs:
            txt = d.get("text") or d.get("content") or ""
            v = self._vote(txt)
            per_doc.append({"doc": d, "vote": v})
            if v["decided_adversarial"]:
                dropped.append(d)
            else:
                kept.append(d)
        if dropped:
            self.stats["blocked"] += len(dropped)
            return DefenseResult(
                allowed=True,
                severity=Severity.SOFT_FILTER,
                confidence=0.7,
                reason=(
                    f"SmoothLLM: dropped {len(dropped)}/{len(retrieved_docs)} "
                    "perturbation-vote-flagged docs"
                ),
                metadata={
                    "kept_docs": kept,
                    "dropped_docs": dropped,
                    "per_doc": per_doc,
                },
            )
        self.stats["allowed"] += 1
        return DefenseResult.allow(
            reason=f"SmoothLLM: all {len(retrieved_docs)} docs passed",
            metadata={"kept_docs": kept, "per_doc": per_doc},
        )
