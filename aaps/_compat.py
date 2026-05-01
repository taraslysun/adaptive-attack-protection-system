"""Back-compat aliases so legacy `from defenses.pace...` style imports
still resolve when this package is installed.

To be deleted in 0.3.x. Loaded from aaps/__init__.py.
"""
from __future__ import annotations

import sys
from importlib import import_module

_aliases = {
    "defenses": "aaps.defenses",
    "attacks": "aaps.attacks",
    "agent": "aaps.agent",
    "evaluation": "aaps.evaluation",
    # legacy SPQ → PACE
    "defenses.spq": "aaps.defenses.pace",
}


def install() -> None:
    for old, new in _aliases.items():
        try:
            mod = import_module(new)
        except Exception:
            continue
        sys.modules.setdefault(old, mod)
