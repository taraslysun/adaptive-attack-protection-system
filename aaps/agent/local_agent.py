"""Re-export of LocalAgent from aaps.attacks._core.local_agent.

The agent shim historically imported from aaps.agent.local_agent.
Authoritative implementation lives under aaps.attacks._core.
"""
from aaps.attacks._core.local_agent import *  # noqa: F401,F403
from aaps.attacks._core.local_agent import (  # noqa: F401
    LocalAgent,
    MemoryEntry,
    AVAILABLE_TOOLS,
)
