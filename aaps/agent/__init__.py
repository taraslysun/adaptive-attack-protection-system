"""Agents.

Two agent classes ship:

* :class:`LocalAgent` — minimal ReAct-style agent (5 hooks, in-memory store, tool
  schema only). Default for the headline matrix and the notebooks; no Qdrant /
  CLIP / LangChain dependency.
* :class:`DeepAgent` — full DeepAgent surface from the parent repo: persistent
  Qdrant long-term memory, multimodal RAG (text + CLIP images), real
  ``ToolSuite`` (web_search, read/write/exec, analyse_image, send_email,
  delete_file, create_user) bound through LangChain. Optional dependency:
  ``pip install 'adaptive-attack-protection-system[deepagent]'``.

Both accept ``defense=...`` (PACE or any baseline). Both implement the same
five hook events H1–H5 from the paper.
"""
from aaps.agent.local_agent import LocalAgent, MemoryEntry, AVAILABLE_TOOLS

# DeepAgent + its support modules pull heavy optional deps; import lazily so
# users who never install the extras still get a working ``aaps.agent``.
try:
    from aaps.agent.deep_agent import DeepAgent  # noqa: F401
    _DEEPAGENT_AVAILABLE = True
except Exception:
    DeepAgent = None  # type: ignore
    _DEEPAGENT_AVAILABLE = False

try:
    from aaps.agent.config import AgentConfig  # noqa: F401
except Exception:
    AgentConfig = None  # type: ignore

try:
    from aaps.agent.memory_manager import MemoryManager, MEMORY_AVAILABLE  # noqa: F401
except Exception:
    MemoryManager = None  # type: ignore
    MEMORY_AVAILABLE = False

try:
    from aaps.agent.multimodal_retrieval import (
        MultimodalRetrieval, MULTIMODAL_AVAILABLE,
    )  # noqa: F401
except Exception:
    MultimodalRetrieval = None  # type: ignore
    MULTIMODAL_AVAILABLE = False

try:
    from aaps.agent.tools import ToolSuite  # noqa: F401
except Exception:
    ToolSuite = None  # type: ignore

__all__ = [
    "LocalAgent", "MemoryEntry", "AVAILABLE_TOOLS",
    "DeepAgent", "AgentConfig",
    "MemoryManager", "MEMORY_AVAILABLE",
    "MultimodalRetrieval", "MULTIMODAL_AVAILABLE",
    "ToolSuite",
]
