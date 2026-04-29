"""Multi-modal RAG retrieval system for text and image content.

Optional dependency — install ``adaptive-attack-protection-system[multimodal]``
(qdrant-client, sentence-transformers, transformers, torch, pillow) to use it.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from PIL import Image
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        ScoredPoint,
    )
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPProcessor, CLIPModel
    import torch
    _MULTIMODAL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    QdrantClient = None  # type: ignore
    Distance = VectorParams = PointStruct = ScoredPoint = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    CLIPProcessor = CLIPModel = None  # type: ignore
    torch = None  # type: ignore
    _MULTIMODAL_AVAILABLE = False

MULTIMODAL_AVAILABLE = _MULTIMODAL_AVAILABLE

from aaps.agent.config import AgentConfig


class MultimodalRetrieval:
    """Multi-modal RAG retrieval system supporting text and images."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize multi-modal retrieval system."""
        if not _MULTIMODAL_AVAILABLE:
            raise ImportError(
                "Multimodal RAG requires extras: pip install "
                "'adaptive-attack-protection-system[multimodal]'"
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
        self.text_dim = self.text_encoder.get_embedding_dimension()

        # Initialize CLIP for image embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_dim = 512  # CLIP ViT-B/32 embedding dimension

        # Collection names
        self.text_collection = f"{self.config.QDRANT_COLLECTION_NAME}_text"
        self.image_collection = f"{self.config.QDRANT_COLLECTION_NAME}_image"

        # Ensure collections exist
        self._ensure_collections()

    def _ensure_collections(self):
        """Ensure text and image collections exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        # Create text collection
        if self.text_collection not in collection_names:
            self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config=VectorParams(
                    size=self.text_dim,
                    distance=Distance.COSINE,
                ),
            )

        # Create image collection
        if self.image_collection not in collection_names:
            self.client.create_collection(
                collection_name=self.image_collection,
                vectors_config=VectorParams(
                    size=self.clip_dim,
                    distance=Distance.COSINE,
                ),
            )

    def add_text_document(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a text document to the knowledge base."""
        import uuid
        metadata = metadata or {}

        # Generate embedding
        embedding = self.text_encoder.encode(text).tolist()

        # Convert doc_id to UUID if it's a string
        if isinstance(doc_id, str):
            try:
                # Try to parse as UUID first
                point_id = uuid.UUID(doc_id)
            except ValueError:
                # If not a valid UUID, generate one from the string
                point_id = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
        else:
            point_id = doc_id

        # Store in Qdrant
        self.client.upsert(
            collection_name=self.text_collection,
            points=[
                PointStruct(
                    id=str(point_id),
                    vector=embedding,
                    payload={
                        "text": text,
                        "type": "text",
                        "doc_id": doc_id,  # Keep original ID in payload
                        **metadata,
                    },
                )
            ],
        )

    def add_image_document(
        self,
        image_path: str,
        doc_id: str,
        caption: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add an image document to the knowledge base."""
        import uuid
        metadata = metadata or {}

        # Check if file exists and is a valid image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check if it's actually an image file (not a text placeholder)
        valid_image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        file_ext = os.path.splitext(image_path)[1].lower()
        
        if file_ext not in valid_image_extensions:
            # Skip placeholder files - just store metadata
            print(f"  Warning: Skipping {image_path} (not a valid image file)")
            return

        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt")

            # Generate embedding
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                embedding = image_features[0].cpu().numpy().tolist()
        except Exception as e:
            print(f"  Warning: Could not process image {image_path}: {e}")
            # Use text caption as fallback embedding
            if caption:
                embedding = self.text_encoder.encode(caption).tolist()
            else:
                embedding = self.text_encoder.encode("image").tolist()

        # Convert doc_id to UUID if it's a string
        if isinstance(doc_id, str):
            try:
                point_id = uuid.UUID(doc_id)
            except ValueError:
                point_id = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
        else:
            point_id = doc_id

        # Store in Qdrant
        self.client.upsert(
            collection_name=self.image_collection,
            points=[
                PointStruct(
                    id=str(point_id),
                    vector=embedding,
                    payload={
                        "image_path": image_path,
                        "caption": caption or "",
                        "type": "image",
                        "doc_id": doc_id,  # Keep original ID in payload
                        **metadata,
                    },
                )
            ],
        )

    def add_multimodal_document(
        self,
        text: str,
        image_path: Optional[str] = None,
        doc_id: str = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a multi-modal document (text + image) to the knowledge base."""
        import uuid

        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}

        # Add text component
        self.add_text_document(text, f"{doc_id}_text", metadata)

        # Add image component if provided
        if image_path and os.path.exists(image_path):
            self.add_image_document(image_path, f"{doc_id}_image", text, metadata)

    def retrieve_text(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve text documents similar to query."""
        # Generate query embedding
        query_embedding = self.text_encoder.encode(query).tolist()

        resp = self.client.query_points(
            collection_name=self.text_collection,
            query=query_embedding,
            limit=k,
            score_threshold=score_threshold,
        )
        results = resp.points

        # Format results
        documents = []
        for result in results:
            documents.append(
                {
                    "id": str(result.id),
                    "text": result.payload.get("text", ""),
                    "score": result.score,
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                }
            )

        return documents

    def retrieve_images(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve images similar to text query."""
        # Generate text query embedding using CLIP
        inputs = self.clip_processor(text=query, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            query_embedding = text_features[0].cpu().numpy().tolist()

        resp = self.client.query_points(
            collection_name=self.image_collection,
            query=query_embedding,
            limit=k,
            score_threshold=score_threshold,
        )
        results = resp.points

        # Format results
        images = []
        for result in results:
            images.append(
                {
                    "id": str(result.id),
                    "image_path": result.payload.get("image_path", ""),
                    "caption": result.payload.get("caption", ""),
                    "score": result.score,
                    "metadata": {
                        k: v
                        for k, v in result.payload.items()
                        if k not in ["image_path", "caption"]
                    },
                }
            )

        return images

    def retrieve_multimodal(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Retrieve both text and image documents."""
        text_results = self.retrieve_text(query, k, score_threshold)
        image_results = self.retrieve_images(query, k, score_threshold)

        return text_results, image_results
