"""Lightweight in-memory vector store for semantic search.

Uses cosine similarity for nearest-neighbor retrieval. Stores document
chunks with their embeddings and metadata for RAG retrieval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """A chunk of text with its embedding and metadata.

    Attributes:
        content: The text content of this chunk.
        embedding: Numerical embedding vector.
        source: Source file path or identifier.
        chunk_id: Unique identifier within the source.
        metadata: Additional context (e.g., heading, line range).
    """

    content: str
    embedding: list[float]
    source: str
    chunk_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """In-memory vector store with cosine similarity search.

    Stores document chunks indexed by their embedding vectors.
    Supports adding documents and querying by semantic similarity.
    """

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._id_counter = 0

    def add(
        self,
        content: str,
        embedding: list[float],
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentChunk:
        """Add a document chunk to the store.

        Args:
            content: Text content of the chunk.
            embedding: Embedding vector for the chunk.
            source: Source file path or identifier.
            metadata: Optional additional metadata.

        Returns:
            The created DocumentChunk.
        """
        chunk = DocumentChunk(
            content=content,
            embedding=embedding,
            source=source,
            chunk_id=self._id_counter,
            metadata=metadata or {},
        )
        self._chunks.append(chunk)
        self._id_counter += 1
        return chunk

    def add_many(
        self,
        chunks: list[tuple[str, list[float], str, dict[str, Any] | None]],
    ) -> list[DocumentChunk]:
        """Add multiple chunks at once.

        Args:
            chunks: List of (content, embedding, source, metadata) tuples.

        Returns:
            List of created DocumentChunks.
        """
        return [
            self.add(content, emb, src, meta)
            for content, emb, src, meta in chunks
        ]

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
        source_filter: str | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Query the store for semantically similar chunks.

        Args:
            embedding: Query embedding vector.
            top_k: Maximum number of results to return.
            min_score: Minimum cosine similarity threshold.
            source_filter: Optional source path to filter results.

        Returns:
            List of (DocumentChunk, score) tuples, sorted by score descending.
        """
        if not self._chunks:
            return []

        candidates: list[tuple[DocumentChunk, float]] = []

        for chunk in self._chunks:
            if source_filter and chunk.source != source_filter:
                continue

            score = _cosine_similarity(embedding, chunk.embedding)

            if score >= min_score:
                candidates.append((chunk, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_k]

    def clear(self) -> None:
        """Remove all chunks from the store."""
        self._chunks.clear()
        self._id_counter = 0

    def count(self) -> int:
        """Return the number of chunks in the store."""
        return len(self._chunks)

    def sources(self) -> set[str]:
        """Return the set of all source paths in the store."""
        return {chunk.source for chunk in self._chunks}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in range [-1, 1]. Returns 0 if either vector is zero.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector dimension mismatch: {len(a)} vs {len(b)}"
        )

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)
