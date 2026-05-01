"""RL policy for generating attack triggers.

Supports both in-context learning (via Ollama) and gradient-based
weight updates (via local GPT-2/Llama policy network).

NOTE: This module is a standalone experimental utility. It is not imported
by the main RL attack pipeline (``attack.py``), which creates its own
``AutoModelForCausalLM`` in ``_init_policy``. This class provides a clean
``nn.Module`` wrapper for the policy with ``generate`` and ``get_log_probs``
methods suitable for use with ``GRPOTrainer``.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class RLPolicy(nn.Module):
    """Policy network for generating attack triggers."""

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        num_samples: int = 1,
    ) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        results = []
        for _ in range(num_samples):
            out = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            results.append(
                self.tokenizer.decode(out[0], skip_special_tokens=True)
            )
        return results

    def get_log_probs(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        logits = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        log_probs = nn.functional.log_softmax(shift_logits, dim=-1)
        return torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
