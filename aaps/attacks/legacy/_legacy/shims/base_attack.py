"""Backward-compatibility shim. Real module: :mod:`attacks._core.base_attack`."""

from aaps.attacks._core.base_attack import *  # noqa: F401,F403
from aaps.attacks._core.base_attack import AttackConfig, AttackResult, BaseAttack  # noqa: F401

__all__ = ["AttackConfig", "AttackResult", "BaseAttack"]
