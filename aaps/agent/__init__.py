"""Agent module for deep reasoning agents with multimodal RAG."""

# Lazy imports: optional heavy dependencies (qdrant_client, etc.) are only
# pulled in when actually used so that the baseline evaluation harness
# can import AMemGuard without crashing in environments without qdrant.
def __getattr__(name: str):
    if name == "DeepAgent":
        from aaps.agent.deep_agent import DeepAgent
        return DeepAgent
    if name == "MemoryManager":
        from aaps.agent.memory_manager import MemoryManager
        return MemoryManager
    if name == "MultimodalRetrieval":
        from aaps.agent.multimodal_retrieval import MultimodalRetrieval
        return MultimodalRetrieval
    if name == "ToolSuite":
        from aaps.agent.tools import ToolSuite
        return ToolSuite
    raise AttributeError(f"module 'agent' has no attribute {name!r}")

__all__ = ["DeepAgent", "MemoryManager", "MultimodalRetrieval", "ToolSuite"]
