"""Compat shim — GCGAttack moved to attacks._legacy.gradient_attack.gcg."""
import warnings as _w
_w.warn(
    "attacks.adaptive.gradient_attack.gcg is legacy; "
    "import from attacks._legacy.gradient_attack.gcg instead.",
    DeprecationWarning, stacklevel=2,
)
from aaps.attacks.legacy._legacy.gradient_attack.gcg import *  # noqa: F401,F403
