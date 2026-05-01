"""RL trainer using GRPO (Group Relative Preference Optimisation).

Implements weight-update GRPO as described in Appendix A.2 of Nasr *et
al.*, "The Attacker Moves Second" (arXiv:2510.09023, bibkey
``nasr2025attacker``); GRPO itself is DeepSeek-AI (arXiv:2501.12948,
2025; bibkey ``deepseek2025r1``):
within each session the attacker proposes candidates and observes scores;
across sessions the policy is updated using group-relative preference pairs.

NOTE: This module is a standalone experimental utility. It is not imported
by the main RL attack pipeline (``attack.py``), which implements the GRPO
weight update logic inline in ``_grpo_weight_update``. This class provides
a cleaner, more general-purpose trainer that uses ``RLPolicy`` and correctly
conditions log-probabilities on the prompt context.
"""

from typing import List, Dict, Any, Callable
import torch
from torch.optim import Adam

from aaps.attacks.legacy._legacy.rl_attack.policy import RLPolicy
from aaps.attacks.slim5.rl.reward import RewardFunction
from aaps.attacks._core.base_attack import AttackResult


class GRPOTrainer:
    """Trainer for RL attack policy using GRPO weight updates."""

    def __init__(
        self,
        policy: RLPolicy,
        reward_fn: RewardFunction,
        learning_rate: float = 1e-5,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.reward_fn = reward_fn
        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)
        self.device = device
        self.training_history: List[Dict[str, Any]] = []

    def train_step(
        self,
        prompts: List[str],
        target_goal: str,
        attack_executor: Callable[[str, str], AttackResult],
        num_samples_per_prompt: int = 4,
    ) -> Dict[str, Any]:
        all_triggers: List[tuple] = []
        all_results: List[AttackResult] = []
        all_rewards: List[float] = []

        for prompt in prompts:
            triggers = self.policy.generate(
                prompt=prompt,
                max_length=100,
                temperature=1.0,
                num_samples=num_samples_per_prompt,
            )
            for trigger in triggers:
                result = attack_executor(trigger, target_goal)
                reward = self.reward_fn.compute_reward(
                    result, trigger, [t for t, _ in all_triggers]
                )
                all_triggers.append((trigger, prompt))
                all_results.append(result)
                all_rewards.append(reward)

        groups: Dict[str, List[tuple]] = {}
        for i, (trigger, prompt) in enumerate(all_triggers):
            groups.setdefault(prompt, []).append((i, trigger, all_rewards[i]))

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for prompt, group_data in groups.items():
            if len(group_data) < 2:
                continue
            group_data.sort(key=lambda x: x[2], reverse=True)
            for i in range(len(group_data) - 1):
                _, hi_trigger, hi_reward = group_data[i]
                _, lo_trigger, lo_reward = group_data[i + 1]
                strength = hi_reward - lo_reward
                if strength > 0:
                    hi_lp = self._log_probs(hi_trigger, prompt)
                    lo_lp = self._log_probs(lo_trigger, prompt)
                    loss = loss - strength * (hi_lp.mean() - lo_lp.mean())

        if loss.requires_grad:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        avg_r = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        sr = (
            sum(1 for r in all_results if r.success) / len(all_results)
            if all_results
            else 0.0
        )
        metrics = {
            "loss": loss.item(),
            "avg_reward": avg_r,
            "success_rate": sr,
            "num_attempts": len(all_results),
        }
        self.training_history.append(metrics)
        return metrics

    def _log_probs(self, trigger: str, prompt: str) -> torch.Tensor:
        full = f"{prompt} {trigger}"
        enc = self.policy.tokenizer(
            full, return_tensors="pt", padding=True, truncation=True
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        return self.policy.get_log_probs(ids, mask)
