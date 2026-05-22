"""Document indexer — chunk and embed local docs for RAG retrieval.

Reads markdown, text, Python, and PDF files, splits them into chunks,
and stores embeddings in a VectorStore for semantic search.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from local_agent.vector_store import VectorStore, DocumentChunk


# File extensions we know how to index
INDEXABLE_EXTENSIONS = {
    ".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".toml", ".rst", ".csv", ".html", ".xml", ".cfg", ".ini",
    ".sh", ".bash", ".zsh", ".rs", ".go", ".java", ".c", ".cpp",
    ".h", ".hpp", ".rb", ".php", ".sql", ".graphql", ".proto",
}

# Patterns to skip (build artifacts, dependencies, etc.)
SKIP_PATTERNS = [
    r"node_modules",
    r"\.git/",
    r"__pycache__",
    r"\.egg-info",
    r"\.pyc$",
    r"\.so$",
    r"\.whl$",
    r"dist/",
    r"build/",
    r"\.venv/",
    r"venv/",
]


class DocumentIndexer:
    """Index local documents for semantic search.

    Reads files from a directory, chunks them, generates embeddings,
    and stores them in a VectorStore.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize the indexer.

        Args:
            vector_store: The VectorStore to store embeddings in.
            embedder: Embedding generator. If None, uses DummyEmbedder.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlapping characters between chunks.
        """
        self._store = vector_store
        self._embedder = embedder or DummyEmbedder()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._indexed_files: list[str] = []

    def index_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> list[str]:
        """Index all eligible files in a directory.

        Args:
            directory: Root directory to index.
            recursive: Whether to scan subdirectories.
            extensions: File extensions to include (default: all indexable).

        Returns:
            List of indexed file paths.
        """
        exts = extensions or INDEXABLE_EXTENSIONS
        dir_path = Path(directory)
        files: list[Path] = []

        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in dir_path.glob(pattern):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in exts:
                continue
            if self._should_skip(str(file_path)):
                continue
            files.append(file_path)

        indexed: list[str] = []
        for file_path in sorted(files):
            try:
                self._index_file(str(file_path))
                indexed.append(str(file_path))
            except Exception:
                # Skip files that can't be read
                pass

        self._indexed_files = indexed
        return indexed

    def index_file(self, file_path: str) -> int:
        """Index a single file.

        Args:
            file_path: Path to the file.

        Returns:
            Number of chunks created.
        """
        count_before = self._store.count()
        self._index_file(file_path)
        if file_path not in self._indexed_files:
            self._indexed_files.append(file_path)
        return self._store.count() - count_before

    def _index_file(self, file_path: str) -> None:
        """Read, chunk, and embed a single file."""
        content = _read_file_content(file_path)
        if not content.strip():
            return

        chunks = self._chunk_text(content)

        for i, chunk_text in enumerate(chunks):
            embedding = self._embedder.embed(chunk_text)
            self._store.add(
                content=chunk_text,
                embedding=embedding,
                source=file_path,
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_size": os.path.getsize(file_path),
                },
            )

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Tries to respect paragraph/code-block boundaries when possible.
        """
        if len(text) <= self._chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self._chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to find a good split point (blank line or sentence boundary)
            split_point = self._find_split_point(text, end)

            chunks.append(text[start:split_point].strip())
            start = split_point - self._chunk_overlap

            if start < 0:
                start = 0

        return chunks

    def _find_split_point(self, text: str, target: int) -> int:
        """Find a natural split point near the target position.

        Prefers blank lines, then newlines, then spaces.
        """
        # Look for blank lines first
        for i in range(target, max(target - 100, 0), -1):
            if text[i : i + 2] == "\n\n":
                return i + 2

        # Then single newlines
        for i in range(target, max(target - 50, 0), -1):
            if text[i] == "\n":
                return i + 1

        # Then spaces
        for i in range(target, max(target - 50, 0), -1):
            if text[i] == " ":
                return i + 1

        # Fallback: split at target
        return target

    def _should_skip(self, file_path: str) -> bool:
        """Check if a file should be skipped based on path patterns."""
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, file_path):
                return True
        return False

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        source_filter: str | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Maximum number of results.
            min_score: Minimum similarity threshold.
            source_filter: Filter to specific source file.

        Returns:
            List of (DocumentChunk, score) tuples.
        """
        query_embedding = self._embedder.embed(query)
        return self._store.query(
            embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            source_filter=source_filter,
        )

    @property
    def indexed_files(self) -> list[str]:
        """Return the list of indexed file paths."""
        return list(self._indexed_files)

    @property
    def chunk_count(self) -> int:
        """Return the total number of indexed chunks."""
        return self._store.count()


def _read_file_content(file_path: str) -> str:
    """Read file content as text, handling encoding issues."""
    encodings = ["utf-8", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Last resort: read with error replacement
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


class Embedder:
    """Base class for embedding generators."""

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: Input text.

        Returns:
            Embedding vector (list of floats).
        """
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    """Generate embeddings using Ollama's embedding API.

    Uses the nomic-embed-text model by default.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Generate embedding via Ollama API."""
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self._model,
            "prompt": text,
            "options": {"temperature": 0},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data["embedding"]
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, Exception) as e:
            raise RuntimeError(
                f"Failed to generate embedding via Ollama: {e}"
            ) from e


class DummyEmbedder(Embedder):
    """Simple hash-based embedder for testing.

    Produces deterministic but semantically meaningless embeddings.
    Uses a fixed dimension of 128.
    """

    DIMENSIONS = 128

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic dummy embedding.

        Uses character-level hashing spread across dimensions.
        """
        import hashlib

        # Use SHA-256 to get a fixed-size hash
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()

        # Expand to our dimension size using multiple hash rounds
        values: list[float] = []
        for i in range(self.DIMENSIONS):
            round_hash = hashlib.sha256(
                hash_bytes + i.to_bytes(4, "big")
            ).digest()
            # Convert first 4 bytes to a normalized float [-1, 1]
            val = int.from_bytes(round_hash[:4], "big")
            normalized = ((val % 10000) / 5000.0) - 1.0
            values.append(round(normalized, 6))

        return values
