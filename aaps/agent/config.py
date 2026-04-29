"""Configuration for the deep agent system."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

# AGENT_ENV switches default endpoints between host-local development
# (`local`) and the docker-compose stack (`docker`). Explicit env vars
# (QDRANT_URL, WORKSPACE_DIR, ...) always take precedence.
_AGENT_ENV = os.getenv("AGENT_ENV", "local").lower()
_DOCKER = _AGENT_ENV == "docker"


class AgentConfig:
    """Configuration class for agent settings."""

    # LLM Configuration
    DEFAULT_LLM_MODEL: str = os.getenv(
        "DEFAULT_LLM_MODEL",
        "openai/gpt-4o-mini" if _OPENROUTER_KEY else "gemini/gemini-2.5-pro",
    )
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # Vector Database Configuration
    QDRANT_URL: str = os.getenv(
        "QDRANT_URL",
        "http://qdrant:6333" if _DOCKER else "http://localhost:6333",
    )
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = "agent_memory"

    # Embedding Configuration
    TEXT_EMBEDDING_MODEL: str = "text-embedding-3-large"
    IMAGE_EMBEDDING_MODEL: str = "openai/clip-vit-base-patch32"

    # Memory Configuration
    MAX_MEMORY_ENTRIES: int = 10000
    MEMORY_RETRIEVAL_K: int = 5
    MEMORY_SIMILARITY_THRESHOLD: float = 0.7

    # RAG Configuration
    RAG_TOP_K: int = 5
    RAG_CHUNK_SIZE: int = 512
    RAG_CHUNK_OVERLAP: int = 50

    # Agent Configuration
    MAX_ITERATIONS: int = 50
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 4096

    # File System Configuration
    WORKSPACE_DIR: str = os.getenv(
        "WORKSPACE_DIR",
        "/home/agent/workspace" if _DOCKER else "workspace",
    )
    CONTEXT_DIR: str = os.getenv("CONTEXT_DIR", "context")

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not (
            cls.OPENROUTER_API_KEY
            or cls.OPENAI_API_KEY
            or cls.ANTHROPIC_API_KEY
        ):
            raise ValueError(
                "At least one remote LLM credential must be set: "
                "OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY"
            )
        return True

    @classmethod
    def preferred_remote_provider(cls) -> str:
        """Which remote credential wins for routing (OpenRouter > OpenAI > Anthropic)."""
        if cls.OPENROUTER_API_KEY:
            return "openrouter"
        if cls.OPENAI_API_KEY:
            return "openai"
        if cls.ANTHROPIC_API_KEY:
            return "anthropic"
        return "none"
