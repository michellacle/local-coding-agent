"""Retrieval-Augmented Generation (RAG) over local documents.

Provides:
- Document indexer (split text into chunks)
- Embedding generation (via Ollama or local models)
- Vector similarity search (in-memory or persistent)
- RAG query pipeline

Supports multiple embedding backends:
- Ollama (local embedding models)
- SentenceTransformers (local HuggingFace models)
- Fallback to TF-IDF if no ML libs available
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

@dataclass
class DocumentChunk:
    """A chunk of text from a document.

    Attributes:
        doc_id: Unique identifier for the source document.
        chunk_id: Index within the document (0-based).
        text: The chunk text content.
        embedding: Vector embedding (list of floats).
        metadata: Optional metadata dict.
    """

    doc_id: str
    chunk_id: int
    text: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A result from a vector similarity search.

    Attributes:
        chunk: The matching DocumentChunk.
        score: Similarity score (cosine similarity, 0-1).
    """

    chunk: DocumentChunk
    score: float


class EmbeddingBackend:
    """Embedding generation backend.

    Supports Ollama, SentenceTransformers, and TF-IDF fallback.
    """

    def __init__(self, backend: str = "ollama", model: str = "nomic-embed-text", url: str = "http://localhost:11434"):
        """Initialize the embedding backend.

        Args:
            backend: Backend type ("ollama", "sentence-transformers", "tfidf").
            model: Model name to use.
            url: Ollama URL (only for "ollama" backend).
        """
        self._backend = backend
        self._model = model
        self._url = url
        self._embedding_dim = 0
        self._tfidf_index: list[dict[str, Any]] = []

        if backend == "ollama":
            self._embedder = self._ollama_embed
        elif backend == "sentence-transformers":
            self._embedder = self._sentence_transformers_embed
        elif backend == "tfidf":
            self._embedder = self._tfidf_embed
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return self._embedder(texts)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim

    def _ollama_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Ollama."""
        all_embeddings: list[list[float]] = []

        for text in texts:
            try:
                import urllib.request
                import urllib.error

                payload = json.dumps({"model": self._model, "prompt": text}).encode()
                req = urllib.request.Request(
                    f"{self._url}/api/embeddings",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                    emb = data.get("embedding", [])
                    if emb:
                        self._embedding_dim = len(emb)
                    all_embeddings.append(emb)
            except (urllib.error.URLError, json.JSONDecodeError, Exception) as e:
                # Fallback to zero vector if Ollama fails
                if self._embedding_dim == 0:
                    self._embedding_dim = 768  # nomic-embed-text default
                all_embeddings.append([0.0] * self._embedding_dim)

        return all_embeddings

    def _sentence_transformers_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using SentenceTransformers."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self._model)
            embeddings = model.encode(texts)
            self._embedding_dim = embeddings.shape[1]
            return embeddings.tolist()
        except ImportError:
            raise RuntimeError(
                "SentenceTransformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def _tfidf_embed(self, texts: list[str]) -> list[list[float]]:
        """Generate TF-IDF embeddings as fallback (no ML needed).

        This is a simple bag-of-words TF-IDF implementation.
        """
        import re
        from collections import Counter

        # Tokenize all documents
        all_tokens: list[list[str]] = []
        for text in texts:
            tokens = re.findall(r"\w+", text.lower())
            all_tokens.append(tokens)

        # Build vocabulary
        vocab: dict[str, int] = {}
        for tokens in all_tokens:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        if not vocab:
            self._embedding_dim = 1
            return [[0.0] for _ in texts]

        self._embedding_dim = len(vocab)

        # Calculate IDF
        n_docs = len(texts)
        idf: dict[str, float] = {}
        for token in vocab:
            df = sum(1 for tokens in all_tokens if token in set(tokens))
            idf[token] = math.log((1 + n_docs) / (1 + df)) + 1

        # Calculate TF-IDF vectors
        embeddings: list[list[float]] = []
        for tokens in all_tokens:
            tf: Counter = Counter(tokens)
            total = len(tokens) if tokens else 1
            vec = [0.0] * self._embedding_dim

            for token, count in tf.items():
                if token in vocab:
                    idx = vocab[token]
                    vec[idx] = (count / total) * idf.get(token, 1.0)

            # Normalize to unit vector
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]

            embeddings.append(vec)

        return embeddings


class VectorStore:
    """In-memory or SQLite-backed vector similarity store.

    Stores document chunks and their embeddings for similarity search.
    """

    def __init__(self, db_path: str | None = None, embedding_dim: int = 768):
        """Initialize the vector store.

        Args:
            db_path: Optional path to SQLite database. If None, uses in-memory.
            embedding_dim: Dimension of embedding vectors.
        """
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._chunks: list[DocumentChunk] = []
        self._conn: sqlite3.Connection | None = None

        if db_path:
            self._conn = sqlite3.connect(db_path)
            self._init_db()
            self._load_from_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        if not self._conn:
            return

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                doc_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                PRIMARY KEY (doc_id, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        """)
        self._conn.commit()

    def _load_from_db(self) -> None:
        """Load existing chunks from SQLite database."""
        if not self._conn:
            return

        cursor = self._conn.execute("SELECT doc_id, chunk_id, text, embedding, metadata FROM chunks")
        for row in cursor.fetchall():
            chunk = DocumentChunk(
                doc_id=row[0],
                chunk_id=row[1],
                text=row[2],
                embedding=json.loads(row[3]),
                metadata=json.loads(row[4]),
            )
            self._chunks.append(chunk)

    def add(self, chunk: DocumentChunk) -> None:
        """Add a document chunk to the store.

        Args:
            chunk: DocumentChunk to add.
        """
        self._chunks.append(chunk)

        if self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO chunks (doc_id, chunk_id, text, embedding, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    chunk.doc_id,
                    chunk.chunk_id,
                    chunk.text,
                    json.dumps(chunk.embedding),
                    json.dumps(chunk.metadata),
                ),
            )
            self._conn.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for similar chunks using cosine similarity.

        Args:
            query_embedding: Query embedding vector.
            top_k: Maximum number of results.
            min_score: Minimum similarity score threshold.

        Returns:
            List of SearchResult objects, sorted by relevance.
        """
        results: list[SearchResult] = []

        for chunk in self._chunks:
            if not chunk.embedding:
                continue

            score = self._cosine_similarity(query_embedding, chunk.embedding)

            if score >= min_score:
                results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity (0 to 1).
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    def delete(self, doc_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            doc_id: Document ID to delete.

        Returns:
            Number of chunks deleted.
        """
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
        deleted = before - len(self._chunks)

        if self._conn:
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._conn.commit()

        return deleted

    def count(self) -> int:
        """Return the number of chunks in the store."""
        return len(self._chunks)

    def get_docs(self) -> set[str]:
        """Return a set of all document IDs in the store."""
        return {c.doc_id for c in self._chunks}

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class DocumentIndexer:
    """Document indexer — splits documents into chunks.

    Supports various file types: .md, .txt, .py, .js, .html, etc.
    """

    SUPPORTED_EXTENSIONS = {
        ".md", ".txt", ".py", ".js", ".ts", ".json",
        ".html", ".css", ".yaml", ".yml", ".toml",
        ".cfg", ".ini", ".csv", ".xml", ".rst",
        ".sh", ".bash", ".zsh", ".rs", ".go", ".java",
        ".c", ".cpp", ".h", ".hpp", ".rb", ".php",
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the document indexer.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def index_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        exclude_patterns: list[str] | None = None,
    ) -> list[DocumentChunk]:
        """Index all supported files in a directory.

        Args:
            dir_path: Directory path to index.
            recursive: Whether to recurse into subdirectories.
            exclude_patterns: List of glob patterns to exclude.

        Returns:
            List of DocumentChunk objects.
        """
        chunks: list[DocumentChunk] = []
        path = Path(dir_path)

        if recursive:
            glob_pattern = "**/*"
        else:
            glob_pattern = "*"

        for file_path in sorted(path.glob(glob_pattern)):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            if exclude_patterns:
                skip = False
                for pattern in exclude_patterns:
                    if file_path.match(pattern):
                        skip = True
                        break
                if skip:
                    continue

            doc_id = str(file_path)
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except (PermissionError, OSError):
                continue

            file_chunks = self.chunk_text(doc_id, content, metadata={
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix,
            })
            chunks.extend(file_chunks)

        return chunks

    def chunk_text(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Split text into overlapping chunks.

        Args:
            doc_id: Document identifier.
            text: Text content to chunk.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of DocumentChunk objects.
        """
        if not text.strip():
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + self._chunk_size

            if end < len(text):
                # Try to break at a sentence or line boundary
                break_point = self._find_break_point(text, end, start)
                if break_point > start:
                    end = break_point

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_idx,
                    text=chunk_text,
                    metadata=metadata or {},
                )
                chunks.append(chunk)
                chunk_idx += 1

            start = end - self._chunk_overlap

        return chunks

    def _find_break_point(
        self,
        text: str,
        target: int,
        start: int,
    ) -> int:
        """Find a good break point near the target position.

        Args:
            text: Full text.
            target: Target position to break at.
            start: Start of current chunk.

        Returns:
            Break position (falls back to target if no good break found).
        """
        # Try to break at sentence boundary
        for delimiter in ["\n\n", ". ", "!\n", "?\n", "\n"]:
            pos = text.rfind(delimiter, start, target)
            if pos >= start:
                return pos + len(delimiter)

        return target


class RAGPipeline:
    """RAG pipeline — combine embedding, vector store, and retrieval.

    Example:
        pipeline = RAGPipeline(embedding_backend="ollama")
        pipeline.index_directory("/path/to/docs")
        results = pipeline.query("How do I configure the LLM provider?")
    """

    def __init__(
        self,
        embedding_backend: str = "ollama",
        embedding_model: str = "nomic-embed-text",
        embedding_url: str = "http://localhost:11434",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        db_path: str | None = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            embedding_backend: Embedding backend ("ollama", "sentence-transformers", "tfidf").
            embedding_model: Embedding model name.
            embedding_url: Ollama URL (for "ollama" backend).
            chunk_size: Chunk size for document indexing.
            chunk_overlap: Overlap between chunks.
            db_path: Optional SQLite database path for persistence.
        """
        self._embedding = EmbeddingBackend(
            backend=embedding_backend,
            model=embedding_model,
            url=embedding_url,
        )
        self._indexer = DocumentIndexer(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._store = VectorStore(
            db_path=db_path,
            embedding_dim=self._embedding.dimension,
        )

    def index_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        exclude_patterns: list[str] | None = None,
    ) -> int:
        """Index all documents in a directory.

        Args:
            dir_path: Directory to index.
            recursive: Whether to recurse into subdirectories.
            exclude_patterns: Glob patterns to exclude.

        Returns:
            Number of chunks indexed.
        """
        chunks = self._indexer.index_directory(
            dir_path, recursive=recursive, exclude_patterns=exclude_patterns
        )

        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self._embedding.embed(texts)

        # Store chunks with embeddings
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            self._store.add(chunk)

        return len(chunks)

    def index_text(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Index a single text document.

        Args:
            doc_id: Document identifier.
            text: Document text.
            metadata: Optional metadata.

        Returns:
            Number of chunks indexed.
        """
        chunks = self._indexer.chunk_text(doc_id, text, metadata=metadata)

        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._embedding.embed(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            self._store.add(chunk)

        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Query the knowledge base.

        Args:
            query_text: Query string.
            top_k: Maximum number of results.
            min_score: Minimum similarity score.

        Returns:
            List of SearchResult objects.
        """
        query_embedding = self._embedding.embed([query_text])[0]
        return self._store.search(query_embedding, top_k=top_k, min_score=min_score)

    def delete_document(self, doc_id: str) -> int:
        """Delete a document from the knowledge base.

        Args:
            doc_id: Document ID to delete.

        Returns:
            Number of chunks deleted.
        """
        return self._store.delete(doc_id)

    def count_chunks(self) -> int:
        """Return the number of indexed chunks."""
        return self._store.count()

    def get_document_ids(self) -> set[str]:
        """Return all document IDs in the store."""
        return self._store.get_docs()

    def close(self) -> None:
        """Close the vector store."""
        self._store.close()
