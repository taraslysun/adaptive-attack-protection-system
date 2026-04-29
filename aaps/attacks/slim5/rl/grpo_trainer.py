"""Real GRPO training for the slim-5 RL attack.

Wraps TRL's :class:`trl.GRPOTrainer` (Shao et al. 2024 *DeepSeekMath*,
arXiv:2402.03300; also DeepSeek-R1, arXiv:2501.12948 §3) so the attacker
policy learns from group-relative advantage with a KL term against a
frozen reference model and PPO-clipped policy ratios — i.e. the actual
GRPO update, not the hand-rolled pairwise loss the legacy code shipped.

Usage::

    from aaps.attacks.slim5.rl.grpo_trainer import GRPOAttackerTrainer
    trainer = GRPOAttackerTrainer(
        policy_model="HuggingFaceTB/SmolLM-135M-Instruct",
        prompts=["jailbreak prompt 1", ...],
        reward_fn=my_reward_callable,
        out_dir="logs/thesis/<HHMM-DDMMYYYY>/grpo/",
        max_steps=50,
    )
    trainer.train()

Notebook smoke and CI use ``HuggingFaceTB/SmolLM-135M-Instruct`` so a
single optimiser step finishes within ~30 s on CPU. Thesis-grade runs
swap in ``Qwen/Qwen2.5-0.5B-Instruct`` or larger and require a real GPU.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

log = logging.getLogger("aaps.attacks.rl.grpo")


def _have_grpo_deps() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
        import datasets  # noqa: F401
    except ImportError:
        return False
    return True


class GRPOAttackerTrainer:
    """Train an attacker policy via real GRPO over a list of prompts.

    Parameters
    ----------
    policy_model:
        HuggingFace repo id or local path. Loaded as a causal LM.
    prompts:
        Training prompts. The trainer samples ``num_generations``
        completions per prompt; group-relative advantage is computed
        within each group.
    reward_fn:
        Callable ``reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]``.
        Must return finite floats; one per (prompt, completion).
    out_dir:
        Directory for checkpoints, logs, and TRL artefacts.
    max_steps:
        Optimiser-step cap. Smoke runs use 5–50; thesis runs 200+.
    num_generations:
        Group size G in GRPO. Smaller is cheaper; defaults to 4.
    learning_rate:
        Adam LR. Defaults to 1e-6 (TRL recommendation for small models).
    beta:
        KL coefficient (β in Shao 2024 Algorithm 1). 0.0 disables the KL
        term; default 0.04 matches TRL's GRPO default.
    epsilon:
        PPO clip ratio. Default 0.2.
    max_completion_length:
        Tokens generated per rollout. Smaller is cheaper.
    seed:
        Determinism seed for trainer + sampler.
    device:
        ``"auto" | "cpu" | "mps" | "cuda"``. Defaults to auto-pick.
    """

    def __init__(
        self,
        policy_model: str,
        prompts: Sequence[str],
        reward_fn: Callable[..., Sequence[float]],
        out_dir: str | Path,
        max_steps: int = 50,
        num_generations: int = 4,
        learning_rate: float = 1e-6,
        beta: float = 0.04,
        epsilon: float = 0.2,
        max_completion_length: int = 64,
        per_device_train_batch_size: int = 4,
        seed: int = 0,
        device: str = "auto",
    ) -> None:
        if not _have_grpo_deps():
            raise ImportError(
                "GRPO requires the [grpo] extra: "
                "pip install -e '.[grpo]'  # transformers + trl + accelerate"
            )

        self.policy_model_id = policy_model
        self.prompts = list(prompts)
        self.reward_fn = reward_fn
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.num_generations = num_generations
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.max_completion_length = max_completion_length
        self.per_device_train_batch_size = per_device_train_batch_size
        self.seed = seed
        self.device = device

        self._trainer = None  # constructed in build()

    # ------------------------------------------------------------------
    def build(self):
        """Materialise the TRL GRPOTrainer. Idempotent."""
        if self._trainer is not None:
            return self._trainer

        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        log.info(
            "GRPO build: policy=%s G=%d max_steps=%d β=%g ε=%g",
            self.policy_model_id, self.num_generations, self.max_steps,
            self.beta, self.epsilon,
        )

        # Resolve device.
        device = self.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        log.info("GRPO device=%s", device)

        # Tokenizer + policy model.
        tok = AutoTokenizer.from_pretrained(self.policy_model_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        # bf16 only if supported on this device; fp16 otherwise on cuda; fp32 on cpu.
        if device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

        policy = AutoModelForCausalLM.from_pretrained(
            self.policy_model_id,
            torch_dtype=dtype,
        )
        # Reference model is implicit in TRL's GRPOTrainer (kept frozen
        # internally for the KL term).

        ds = Dataset.from_dict({"prompt": self.prompts})

        cfg = GRPOConfig(
            output_dir=str(self.out_dir),
            overwrite_output_dir=True,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self.learning_rate,
            num_train_epochs=1,
            max_steps=self.max_steps,
            num_generations=self.num_generations,
            max_completion_length=self.max_completion_length,
            beta=self.beta,
            epsilon=self.epsilon,
            loss_type="grpo",  # honest GRPO loss, not DAPO variant
            scale_rewards="group",  # Shao 2024 group-relative scaling
            temperature=1.0,
            top_p=1.0,
            seed=self.seed,
            disable_dropout=True,
            save_strategy="no",  # we save manually after train()
            logging_steps=1,
            report_to=None,
            use_cpu=(device == "cpu"),
            use_mps_device=(device == "mps"),
        )

        # Wrap user reward_fn in TRL's expected signature.
        rf = self.reward_fn

        def _reward_wrapper(prompts, completions, **kwargs):
            scores = rf(prompts, completions, **kwargs)
            # Coerce to floats; replace NaN/inf with 0.
            out: list[float] = []
            for s in scores:
                try:
                    f = float(s)
                    if not (f == f and f != float("inf") and f != float("-inf")):
                        f = 0.0
                except (TypeError, ValueError):
                    f = 0.0
                out.append(f)
            return out

        _reward_wrapper.__name__ = "user_reward"

        self._trainer = GRPOTrainer(
            model=policy,
            reward_funcs=_reward_wrapper,
            args=cfg,
            train_dataset=ds,
            processing_class=tok,
        )
        return self._trainer

    # ------------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        """Run training. Returns summary dict with final metrics."""
        trainer = self.build()
        log.info("GRPO train: starting (max_steps=%d)", self.max_steps)
        train_result = trainer.train()
        # Save final policy for downstream use.
        trainer.save_model(str(self.out_dir / "policy_final"))
        log.info("GRPO train: done; saved → %s", self.out_dir / "policy_final")
        metrics = dict(getattr(train_result, "metrics", {}))
        metrics.setdefault("max_steps", self.max_steps)
        metrics.setdefault("num_generations", self.num_generations)
        return metrics

    # ------------------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Sample one completion from the trained policy."""
        trainer = self.build()
        tok = trainer.processing_class
        model = trainer.model
        import torch
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=1.0,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
            )
        completion = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return completion
