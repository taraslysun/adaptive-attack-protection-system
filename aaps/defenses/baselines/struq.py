"""StruQ (Structured Queries) defense -- bracket-style separation.

Source
------
Chen, S.; Zharmagambetov, A.; Mahloujifar, S.; Chaudhuri, K.; Guo, C.,
"StruQ: Defending Against Prompt Injection with Structured Queries",
arXiv:2402.06363 (2024), USENIX Security 2025
(https://arxiv.org/abs/2402.06363). Bibkey ``chen2024struq``.

Tested as a baseline alongside the Nasr et al. "The Attacker Moves
Second" suite (arXiv:2510.09023, bibkey ``nasr2025attacker``).

This baseline implements the simpler ``[INST]``/``[DATA]`` bracket
variant with a no-op ``check_input``. A separate AIS layer
``defenses/integrity/l1_channels.py`` implements a nonce-anchored
``<<UNTRUSTED-...>>`` variant with delimiter-escape detection -- the
two share a paper but are DIFFERENT algorithms and must be reported
under distinct columns in the thesis tables.

Separates the instruction and data portions of a prompt using special
delimiter tokens. The model is trained (or prompted) to only follow
instructions from the instruction portion and treat the data portion
as pure data.

This implementation provides the prompt-level separation logic.
"""

from typing import Dict, Any, Optional, List

from aaps.defenses.base_defense import BaseDefense, DefenseResult


INSTRUCTION_START = "[INST]"
INSTRUCTION_END = "[/INST]"
DATA_START = "[DATA]"
DATA_END = "[/DATA]"


class StruQDefense(BaseDefense):
    """Structured Queries defense with instruction/data separation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def structure_prompt(
        self,
        instruction: str,
        data: str,
    ) -> str:
        """Build a structured prompt with explicit instruction/data boundaries."""
        return (
            f"{INSTRUCTION_START}\n{instruction}\n{INSTRUCTION_END}\n"
            f"{DATA_START}\n{data}\n{DATA_END}"
        )

    def check_memory_write(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DefenseResult:
        self.stats["total_checks"] += 1

        injection_indicators = [
            "[INST]", "[/INST]", "SYSTEM:", "OVERRIDE",
            "ignore previous", "new instruction",
        ]
        content_lower = content.lower()
        found = [ind for ind in injection_indicators if ind.lower() in content_lower]

        if found:
            self.stats["blocked"] += 1
            return DefenseResult(
                allowed=False,
                confidence=0.8,
                reason=f"Data contains instruction-like tokens: {found}",
                metadata={"indicators": found},
            )

        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            confidence=0.9,
            reason="No instruction tokens in data",
            metadata={},
        )

    def check_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
    ) -> DefenseResult:
        self.stats["total_checks"] += 1
        suspicious = []
        for doc in retrieved_docs:
            text = doc.get("text", "") or doc.get("content", "")
            if any(tok in text for tok in [INSTRUCTION_START, INSTRUCTION_END]):
                suspicious.append(doc)

        if suspicious:
            self.stats["blocked"] += len(suspicious)
            return DefenseResult(
                allowed=False,
                confidence=0.6,
                reason=f"Blocked {len(suspicious)} docs with instruction tokens",
                metadata={"filtered_count": len(suspicious), "filtered_docs": suspicious},
            )

        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            confidence=0.9,
            reason="All docs passed StruQ check",
            metadata={},
        )

    def check_input(self, user_input: str) -> DefenseResult:
        self.stats["total_checks"] += 1
        self.stats["allowed"] += 1
        return DefenseResult(
            allowed=True,
            confidence=0.7,
            reason="StruQ structure applied at prompt level",
            metadata={},
        )
