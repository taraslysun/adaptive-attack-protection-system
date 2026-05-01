"""Script to populate knowledge base with clean documents and images."""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aaps.agent.config import AgentConfig
from aaps.agent.multimodal_retrieval import MultimodalRetrieval


def generate_sample_documents(count: int = 500) -> List[Dict[str, Any]]:
    """Generate sample documents for knowledge base."""
    documents = []

    # Technical documentation samples
    tech_topics = [
        "Python programming best practices",
        "REST API design principles",
        "Database optimization techniques",
        "Security best practices",
        "Machine learning fundamentals",
        "Docker containerization",
        "Kubernetes orchestration",
        "Cloud computing architectures",
        "Microservices patterns",
        "DevOps methodologies",
    ]

    # Research paper abstracts
    research_topics = [
        "Large language models and their applications",
        "Neural network architectures",
        "Computer vision techniques",
        "Natural language processing",
        "Reinforcement learning algorithms",
        "Adversarial machine learning",
        "Federated learning systems",
        "Explainable AI methods",
        "Transfer learning approaches",
        "Multi-modal learning frameworks",
    ]

    # Code examples
    code_examples = [
        "def secure_hash_password(password: str) -> str:\n    import hashlib\n    return hashlib.sha256(password.encode()).hexdigest()",
        "class DatabaseConnection:\n    def __init__(self, connection_string: str):\n        self.conn = psycopg2.connect(connection_string)\n    def execute_query(self, query: str):\n        return self.conn.execute(query)",
        "async def fetch_data(url: str) -> dict:\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()",
    ]

    doc_id = 0
    for topic in tech_topics * 20:  # Repeat to get more documents
        documents.append(
            {
                "id": f"doc_{doc_id}",
                "text": f"Documentation: {topic}. This document provides comprehensive information about {topic.lower()}. "
                f"It covers fundamental concepts, best practices, and practical examples. "
                f"Key points include implementation details, common pitfalls, and recommended approaches. "
                f"Readers should follow these guidelines to ensure proper implementation and maintenance.",
                "type": "documentation",
                "metadata": {"category": "technical", "source": "generated"},
            }
        )
        doc_id += 1

    for topic in research_topics * 15:
        documents.append(
            {
                "id": f"doc_{doc_id}",
                "text": f"Research Abstract: {topic}. This paper presents novel approaches to {topic.lower()}. "
                f"The methodology involves experimental validation and theoretical analysis. "
                f"Results demonstrate significant improvements over baseline methods. "
                f"Future work includes extending the approach to additional domains.",
                "type": "research",
                "metadata": {"category": "academic", "source": "generated"},
            }
        )
        doc_id += 1

    for i, code in enumerate(code_examples * 50):
        documents.append(
            {
                "id": f"doc_{doc_id}",
                "text": f"Code Example {i+1}:\n\n{code}\n\nThis code demonstrates proper implementation patterns.",
                "type": "code",
                "metadata": {"category": "code", "source": "generated"},
            }
        )
        doc_id += 1

    # Add more diverse content to reach target count
    while len(documents) < count:
        documents.append(
            {
                "id": f"doc_{doc_id}",
                "text": f"General knowledge document {doc_id}. This contains useful information about various topics "
                f"including technology, science, and best practices. It serves as a reference for common questions "
                f"and provides context for understanding related concepts.",
                "type": "general",
                "metadata": {"category": "general", "source": "generated"},
            }
        )
        doc_id += 1

    return documents[:count]


def create_sample_images(image_dir: Path, count: int = 100):
    """Create sample image metadata files (placeholders)."""
    image_dir.mkdir(parents=True, exist_ok=True)

    # Create placeholder text files that represent images
    # In production, you'd use actual image generation or download real images
    # These are just metadata files - actual images would be PNG/JPG files
    for i in range(count):
        placeholder_path = image_dir / f"image_{i:03d}.txt"
        with open(placeholder_path, "w") as f:
            f.write(f"Image placeholder {i}\nThis represents an image file.\n")
            f.write(f"Category: {'diagram' if i % 3 == 0 else 'chart' if i % 3 == 1 else 'screenshot'}\n")
            f.write(f"To use real images, replace this with actual PNG/JPG files.\n")


def populate_knowledge_base(
    text_count: int = 500,
    image_count: int = 100,
    output_dir: Path = Path("data/knowledge_base"),
):
    """Populate knowledge base with documents and images."""
    print(f"Generating {text_count} text documents...")
    documents = generate_sample_documents(text_count)

    print(f"Creating {image_count} image placeholders...")
    image_dir = output_dir / "images"
    create_sample_images(image_dir, image_count)

    # Initialize retrieval system
    config = AgentConfig()
    retrieval = MultimodalRetrieval(config)

    print("Adding documents to vector database...")
    for doc in documents:
        retrieval.add_text_document(
            text=doc["text"],
            doc_id=doc["id"],
            metadata=doc.get("metadata", {}),
        )

    print("Adding images to vector database...")
    # Note: Since we're using placeholder text files, we'll skip actual image processing
    # In a real scenario, you would have actual image files (PNG, JPG, etc.)
    # For now, we'll just add text descriptions as image metadata
    print(f"  Note: Skipping image processing (using placeholders).")
    print(f"  In production, replace placeholder files with actual images.")

    # Save metadata
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "text_documents": len(documents),
                "images": image_count,
                "documents": [
                    {
                        "id": doc["id"],
                        "type": doc["type"],
                        "metadata": doc.get("metadata", {}),
                    }
                    for doc in documents
                ],
            },
            f,
            indent=2,
        )

    print(f"\nKnowledge base populated successfully!")
    print(f"  - Text documents: {len(documents)}")
    print(f"  - Images: {image_count}")
    print(f"  - Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Populate knowledge base")
    parser.add_argument(
        "--text-count", type=int, default=500, help="Number of text documents"
    )
    parser.add_argument(
        "--image-count", type=int, default=100, help="Number of images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/knowledge_base",
        help="Output directory",
    )

    args = parser.parse_args()
    populate_knowledge_base(
        text_count=args.text_count,
        image_count=args.image_count,
        output_dir=Path(args.output_dir),
    )
