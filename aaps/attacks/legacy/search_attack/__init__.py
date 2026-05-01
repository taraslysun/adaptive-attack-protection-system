"""Compat shim — SearchAttack moved to attacks._legacy.search_attack."""
import warnings as _w
_w.warn(
    "attacks.adaptive.search_attack is legacy; "
    "import from attacks._legacy.search_attack instead.",
    DeprecationWarning, stacklevel=2,
)
from aaps.attacks.legacy._legacy.search_attack import *  # noqa: F401,F403
