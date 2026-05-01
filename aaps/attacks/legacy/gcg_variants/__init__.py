"""Compat shim — GCG variants moved to attacks._legacy.gcg_variants."""
import warnings as _w
_w.warn(
    "attacks.adaptive.gcg_variants is legacy; "
    "import from attacks._legacy.gcg_variants instead.",
    DeprecationWarning, stacklevel=2,
)
from aaps.attacks.legacy._legacy.gcg_variants import *  # noqa: F401,F403
