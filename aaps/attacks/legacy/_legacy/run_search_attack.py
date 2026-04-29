"""Compat shim — search runner moved to attacks._legacy.run_search_attack."""
import warnings as _w
_w.warn(
    "attacks.runners.run_search_attack is legacy; "
    "Search/MAP-Elites is no longer in the headline attack set.",
    DeprecationWarning, stacklevel=2,
)
from aaps.attacks.legacy._legacy.run_search_attack import *  # noqa: F401,F403
