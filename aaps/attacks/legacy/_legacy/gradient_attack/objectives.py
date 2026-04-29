"""Objective functions for gradient-based attacks.

Supports multi-objective loss combining target-token NLL, tool-call
probability, and defense-evasion terms.

NOTE: This module is a standalone experimental utility. It is not imported
by the main GCG attack pipeline (``gcg.py``), which uses its own inline
cross-entropy loss and nanoGCG's mellowmax loss. These classes are provided
as reference implementations for alternative objective formulations.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


class ToolCallObjective:
    """Objective for optimizing adversarial triggers toward specific tool calls."""

    def __init__(self, model_name: str = "gpt2", target_tool: str = "send_email"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.target_tool = target_tool
        self.target_tokens = self.tokenizer.encode(target_tool, add_special_tokens=False)

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_output: Optional[str] = None,
        target_tool_params: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        losses = []

        if self.target_tokens:
            tool_logits = logits[0, -1, self.target_tokens]
            losses.append(-F.log_softmax(tool_logits, dim=-1).mean())

        if target_output:
            tgt_ids = self.tokenizer.encode(target_output, add_special_tokens=False)
            if tgt_ids:
                tgt_logits = logits[0, -len(tgt_ids):, tgt_ids]
                losses.append(-F.log_softmax(tgt_logits, dim=-1).mean())

        return sum(losses) / len(losses) if losses else torch.tensor(0.0)


class DefenseAwareObjective:
    """Combined objective that penalises filter-detectable triggers.

    Adds a perplexity regularisation term so the optimised suffix
    looks more natural and evades perplexity-based detectors.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        perplexity_weight: float = 0.1,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.perplexity_weight = perplexity_weight

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
    ) -> torch.Tensor:
        full = torch.cat([input_ids, target_ids], dim=1)
        outputs = self.model(input_ids=full)
        logits = outputs.logits
        start = input_ids.shape[1] - 1
        end = start + target_ids.shape[1]
        target_logits = logits[:, start:end, :]
        nll = F.cross_entropy(
            target_logits.reshape(-1, target_logits.size(-1)),
            target_ids.reshape(-1),
        )

        ppl_out = self.model(input_ids=suffix_ids, labels=suffix_ids)
        ppl_loss = ppl_out.loss * self.perplexity_weight

        return nll + ppl_loss
