"""GRPO training smoke test. Marked pytest.mark.grpo, opt-in.

Verifies the trainer actually runs a real GRPO step on a tiny model with
an easy reward signal. Requires the [grpo] extra installed.

Run:
    pytest -m grpo tests/test_grpo.py -q
"""
from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.grpo


def _have_grpo() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
        import datasets  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def tiny_trainer(tmp_path_factory):
    if not _have_grpo():
        pytest.skip("[grpo] extras not installed")
    from aaps.attacks.slim5.rl.grpo_trainer import GRPOAttackerTrainer

    out = tmp_path_factory.mktemp("grpo_smoke")
    # 5 short prompts; reward = length of completion (max 64 tokens)
    # so a longer completion = higher reward — easy signal.
    prompts = [
        "Write a long sentence about cats:",
        "Explain in detail what is sunlight:",
        "Describe in many words the colour blue:",
        "List ten things that are heavy:",
        "Tell a long story about a robot:",
    ]

    def reward_fn(prompts, completions, **_):
        return [float(len(c)) for c in completions]

    trainer = GRPOAttackerTrainer(
        policy_model="HuggingFaceTB/SmolLM-135M-Instruct",
        prompts=prompts,
        reward_fn=reward_fn,
        out_dir=out,
        max_steps=2,
        num_generations=2,
        per_device_train_batch_size=2,
        max_completion_length=24,
        beta=0.04,
        epsilon=0.2,
        seed=0,
        device="cpu",
    )
    return trainer


def test_grpo_builds(tiny_trainer):
    """Trainer constructs without error."""
    t = tiny_trainer.build()
    assert t is not None


def test_grpo_runs_one_step(tiny_trainer):
    """Two optimiser steps complete; loss is finite."""
    metrics = tiny_trainer.train()
    assert isinstance(metrics, dict)
    if "train_loss" in metrics:
        loss = metrics["train_loss"]
        assert math.isfinite(loss), f"loss not finite: {loss}"


def test_grpo_can_generate(tiny_trainer):
    """Trained policy still generates."""
    out = tiny_trainer.generate("hello", max_new_tokens=8)
    assert isinstance(out, str)


def test_grpo_imports_no_extras_skip():
    """When extras missing, the attacker class still imports."""
    # Just verify the wrapper module import does not crash even if the
    # heavy deps are absent at import time (deps are only required at
    # instantiation).
    import importlib
    mod = importlib.import_module("aaps.attacks.slim5.rl.grpo_trainer")
    assert hasattr(mod, "GRPOAttackerTrainer")
    assert hasattr(mod, "_have_grpo_deps")
