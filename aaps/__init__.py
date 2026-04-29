"""adaptive-attack-protection-system — PACE control-plane defence + adaptive attack harness.

See docs/ for architecture overview.
"""

__version__ = "0.1.0"

# Install legacy import aliases (`defenses.pace.*` → `aaps.defenses.pace.*` etc.)
# Drop in 0.3.x.
from aaps import _compat as _aaps_compat
_aaps_compat.install()
del _aaps_compat
