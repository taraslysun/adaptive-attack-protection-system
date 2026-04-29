"""Compat shim — GCG runner moved to attacks._legacy.run_gcg_attack."""
import warnings as _w
_w.warn(
    "attacks.runners.run_gcg_attack is legacy; "
    "GCG is no longer in the headline attack set.",
    DeprecationWarning, stacklevel=2,
)
from aaps.attacks.legacy._legacy.run_gcg_attack import *  # noqa: F401,F403
