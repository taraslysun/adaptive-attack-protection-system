"""Backward-compatibility shim. Real module: :mod:`attacks.static.static_attacks`."""

from aaps.attacks.legacy.static.static_attacks import *  # noqa: F401,F403
from aaps.attacks.legacy.static.static_attacks import StaticAttackSuite  # noqa: F401

__all__ = ["StaticAttackSuite"]
