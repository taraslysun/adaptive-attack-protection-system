"""Memory manager for long-term memory storage and retrieval."""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    from sentence_transformers import SentenceTransformer
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    # Stubs so module-level references never NameError; class __init__ raises
    # a clear ImportError when MemoryManager is actually instantiated.
    QdrantClient = None  # type: ignore[assignment,misc]
    Distance = None  # type: ignore[assignment]
    VectorParams = None  # type: ignore[assignment]
    PointStruct = None  # type: ignore[assignment]
    Filter = None  # type: ignore[assignment]
    FieldCondition = None  # type: ignore[assignment]
    MatchValue = None  # type: ignore[assignment]
    SentenceTransformer = None  # type: ignore[assignment]

# Public flag for downstream callers (agent factories, deep_agent.py, etc.).
MEMORY_AVAILABLE = _QDRANT_AVAILABLE

from aaps.agent.config import AgentConfig


@dataclass
class MemoryEntry:
    """Represents a single memory entry."""

    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str
    session_id: Optional[str] = None
    entry_type: str = "general"  # general, preference, fact, todo, reasoning_trace

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MemoryManager:
    """Manages long-term memory for the agent using Qdrant vector database."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize memory manager."""
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant_client and sentence_transformers are required for MemoryManager. "
                "Install with: pip install qdrant-client sentence-transformers"
            )
        self.config = config or AgentConfig()
        self.config.validate()

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.config.QDRANT_URL,
            api_key=self.config.QDRANT_API_KEY,
        )

        # Initialize text embedding model
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the memory collection exists in Qdrant."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.config.QDRANT_COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.text_encoder.get_embedding_dimension(),
                    distance=Distance.COSINE,
                ),
            )

    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        entry_type: str = "general",
    ) -> str:
        """
        Store a memory entry.

        Args:
            content: The content to store
            metadata: Additional metadata
            session_id: Session identifier
            entry_type: Type of memory entry

        Returns:
            Memory entry ID
        """
        entry_id = str(uuid.uuid4())
        metadata = metadata or {}

        # Create memory entry
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            entry_type=entry_type,
        )

        # Generate embedding
        embedding = self.text_encoder.encode(content).tolist()

        # Prepare payload
        payload = {
            "content": entry.content,
            "metadata": json.dumps(entry.metadata),
            "timestamp": entry.timestamp,
            "session_id": entry.session_id or "",
            "entry_type": entry.entry_type,
        }

        # Store in Qdrant (entry_id is already a UUID string from uuid.uuid4())
        self.client.upsert(
            collection_name=self.config.QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=entry_id,  # UUID string is valid for Qdrant
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        return entry_id

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        entry_type: Optional[str] = None,
        session_id: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[MemoryEntry]:
        """
        Retrieve similar memory entries.

        Args:
            query: Search query
            k: Number of results to return
            entry_type: Filter by entry type
            session_id: Filter by session ID
            similarity_threshold: Minimum similarity score

        Returns:
            List of memory entries
        """
        k = k or self.config.MEMORY_RETRIEVAL_K
        similarity_threshold = (
            similarity_threshold or self.config.MEMORY_SIMILARITY_THRESHOLD
        )

        # Generate query embedding
        query_embedding = self.text_encoder.encode(query).tolist()

        # Build filter
        filters = []
        if entry_type:
            filters.append(
                FieldCondition(
                    key="entry_type",
                    match=MatchValue(value=entry_type),
                )
            )
        if session_id:
            filters.append(
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=session_id),
                )
            )

        filter_condition = Filter(must=filters) if filters else None

        # Search in Qdrant (qdrant-client 1.17+ uses query_points, not search)
        resp = self.client.query_points(
            collection_name=self.config.QDRANT_COLLECTION_NAME,
            query=query_embedding,
            limit=k,
            query_filter=filter_condition,
            score_threshold=similarity_threshold,
        )
        results = resp.points

        # Convert to MemoryEntry objects
        entries = []
        for result in results:
            payload = result.payload
            entry = MemoryEntry(
                id=str(result.id),
                content=payload.get("content", ""),
                metadata=json.loads(payload.get("metadata", "{}")),
                timestamp=payload.get("timestamp", ""),
                session_id=payload.get("session_id"),
                entry_type=payload.get("entry_type", "general"),
            )
            entries.append(entry)

        return entries

    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID."""
        try:
            self.client.delete(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                points_selector=[entry_id],
            )
            return True
        except Exception:
            return False

    def get_all(self, entry_type: Optional[str] = None) -> List[MemoryEntry]:
        """Get all memory entries, optionally filtered by type."""
        # This is a simplified implementation
        # In production, you'd use scroll API for better performance
        query = "memory"  # Generic query to retrieve all
        return self.retrieve(query, k=1000, entry_type=entry_type)
