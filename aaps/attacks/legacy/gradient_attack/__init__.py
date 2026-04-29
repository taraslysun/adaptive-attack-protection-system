"""Compat shim — GCG moved to attacks._legacy.gradient_attack (2026-04-25)."""
import warnings as _w
_w.warn(
    "attacks.adaptive.gradient_attack is legacy; "
    "import from attacks._legacy.gradient_attack instead.",
    DeprecationWarning, stacklevel=2,
)
from aaps.attacks.legacy._legacy.gradient_attack import *  # noqa: F401,F403
