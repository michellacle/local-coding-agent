"""Tests for document_indexer — DocumentIndexer, chunking, and embedders."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_agent.document_indexer import (
    DocumentIndexer,
    DummyEmbedder,
    OllamaEmbedder,
    _read_file_content,
    INDEXABLE_EXTENSIONS,
)
from local_agent.vector_store import VectorStore


class TestDummyEmbedder:
    """Test DummyEmbedder."""

    def test_returns_list_of_floats(self):
        embedder = DummyEmbedder()
        vec = embedder.embed("hello")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_fixed_dimensions(self):
        embedder = DummyEmbedder()
        vec = embedder.embed("hello")
        assert len(vec) == DummyEmbedder.DIMENSIONS

    def test_deterministic(self):
        embedder = DummyEmbedder()
        v1 = embedder.embed("same text")
        v2 = embedder.embed("same text")
        assert v1 == v2

    def test_different_text_different_embedding(self):
        embedder = DummyEmbedder()
        v1 = embedder.embed("text A")
        v2 = embedder.embed("text B")
        assert v1 != v2

    def test_values_in_range(self):
        embedder = DummyEmbedder()
        vec = embedder.embed("test")
        assert all(-1.0 <= v <= 1.0 for v in vec)


class TestReadFileContent:
    """Test _read_file_content helper."""

    def test_read_utf8_file(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        content = _read_file_content(str(f))
        assert content == "hello world"

    def test_read_latin1_file(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"caf\xe9")  # Latin-1 encoded
        content = _read_file_content(str(f))
        assert "caf" in content

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        content = _read_file_content(str(f))
        assert content == ""

    def test_read_python_file(self, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    pass\n")
        content = _read_file_content(str(f))
        assert "def hello" in content


class TestDocumentIndexer:
    """Test DocumentIndexer."""

    def _make_indexer(self, store: VectorStore | None = None) -> DocumentIndexer:
        return DocumentIndexer(
            vector_store=store or VectorStore(),
            embedder=DummyEmbedder(),
            chunk_size=100,
            chunk_overlap=20,
        )

    def test_index_single_file(self, tmp_path: Path):
        indexer = self._make_indexer()
        f = tmp_path / "test.md"
        f.write_text("# Hello\n\nThis is a test document with some content.")
        count = indexer.index_file(str(f))
        assert count >= 1
        assert str(f) in indexer.indexed_files

    def test_index_directory(self, tmp_path: Path):
        indexer = self._make_indexer()
        (tmp_path / "a.md").write_text("Document A")
        (tmp_path / "b.md").write_text("Document B")
        (tmp_path / "skip.dat").write_text("Skip this")

        files = indexer.index_directory(str(tmp_path))
        assert len(files) == 2
        assert store_count(indexer) >= 2

    def test_index_recursive(self, tmp_path: Path):
        indexer = self._make_indexer()
        (tmp_path / "root.md").write_text("Root doc")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "child.md").write_text("Child doc")

        files = indexer.index_directory(str(tmp_path), recursive=True)
        assert len(files) == 2

    def test_index_non_recursive(self, tmp_path: Path):
        indexer = self._make_indexer()
        (tmp_path / "root.md").write_text("Root doc")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "child.md").write_text("Child doc")

        files = indexer.index_directory(str(tmp_path), recursive=False)
        assert len(files) == 1
        assert "root.md" in files[0]

    def test_index_skips_pycache(self, tmp_path: Path):
        indexer = self._make_indexer()
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_text("skip")
        (tmp_path / "real.py").write_text("keep")

        files = indexer.index_directory(str(tmp_path))
        assert len(files) == 1

    def test_index_skips_node_modules(self, tmp_path: Path):
        indexer = self._make_indexer()
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "pkg.js").write_text("skip")

        files = indexer.index_directory(str(tmp_path))
        assert len(files) == 0

    def test_index_skips_git(self, tmp_path: Path):
        indexer = self._make_indexer()
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("skip")

        files = indexer.index_directory(str(tmp_path))
        assert len(files) == 0

    def test_index_skips_empty_files(self, tmp_path: Path):
        indexer = self._make_indexer()
        (tmp_path / "empty.md").write_text("")
        (tmp_path / "whitespace.md").write_text("   \n\n  ")

        files = indexer.index_directory(str(tmp_path))
        assert len(files) == 2  # Files indexed but no chunks created
        assert store_count(indexer) == 0

    def test_index_file_with_chunks(self, tmp_path: Path):
        indexer = self._make_indexer()
        f = tmp_path / "long.md"
        f.write_text("A" * 500)  # Long enough to be chunked
        count = indexer.index_file(str(f))
        assert count >= 2  # Should produce multiple chunks

    def test_retrieve_returns_results(self, tmp_path: Path):
        indexer = self._make_indexer()
        (tmp_path / "config.md").write_text("Database connection settings for PostgreSQL")
        (tmp_path / "auth.md").write_text("OAuth2 authentication flow and token management")
        indexer.index_directory(str(tmp_path))

        results = indexer.retrieve("database settings")
        assert len(results) >= 1

    def test_retrieve_source_filter(self, tmp_path: Path):
        indexer = self._make_indexer()
        (tmp_path / "a.md").write_text("Content A about databases")
        (tmp_path / "b.md").write_text("Content B about auth")
        indexer.index_directory(str(tmp_path))

        results = indexer.retrieve("databases", source_filter=str(tmp_path / "a.md"))
        for chunk, _ in results:
            assert "a.md" in chunk.source

    def test_chunk_count_property(self, tmp_path: Path):
        indexer = self._make_indexer()
        assert indexer.chunk_count == 0

        (tmp_path / "test.md").write_text("Some content here")
        indexer.index_file(str(tmp_path / "test.md"))
        assert indexer.chunk_count >= 1

    def test_extensions_filter(self, tmp_path: Path):
        indexer = self._make_indexer()
        (tmp_path / "a.md").write_text("A")
        (tmp_path / "b.py").write_text("B")

        files = indexer.index_directory(str(tmp_path), extensions={".md"})
        assert len(files) == 1

    def test_indexed_files_property(self, tmp_path: Path):
        indexer = self._make_indexer()
        (tmp_path / "x.md").write_text("X")
        indexer.index_directory(str(tmp_path))
        assert len(indexer.indexed_files) == 1

    def test_unknown_extension_skipped(self, tmp_path: Path):
        indexer = self._make_indexer()
        (tmp_path / "unknown.xyz").write_text("Skip")
        files = indexer.index_directory(str(tmp_path))
        assert len(files) == 0


class TestOllamaEmbedder:
    """Test OllamaEmbedder (with mocked HTTP)."""

    def test_embed_calls_api(self):
        embedder = OllamaEmbedder(base_url="http://localhost:7778")

        mock_response = MagicMock()
        mock_response.read.return_value = json_bytes({"embedding": [0.1, 0.2, 0.3]})
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = embedder.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_custom_model(self):
        embedder = OllamaEmbedder(
            base_url="http://localhost:7778",
            model="custom-embed",
        )
        mock_response = MagicMock()
        mock_response.read.return_value = json_bytes({"embedding": [1.0]})
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            embedder.embed("test")
            # Verify the request payload contains the model name
            call_args = mock_urlopen.call_args
            payload = call_args[0][0].data.decode("utf-8")
            assert "custom-embed" in payload

    def test_embed_connection_error(self):
        embedder = OllamaEmbedder()
        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            with pytest.raises(RuntimeError, match="Failed to generate embedding"):
                embedder.embed("test")

    def test_embed_trailing_slash_removed(self):
        embedder = OllamaEmbedder(base_url="http://localhost:7778/")
        assert embedder._base_url == "http://localhost:7778"


class TestChunking:
    """Test text chunking behavior."""

    def test_short_text_single_chunk(self, tmp_path: Path):
        indexer = DocumentIndexer(
            VectorStore(), DummyEmbedder(), chunk_size=500, chunk_overlap=50
        )
        chunks = indexer._chunk_text("Short text")
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self, tmp_path: Path):
        indexer = DocumentIndexer(
            VectorStore(), DummyEmbedder(), chunk_size=50, chunk_overlap=10
        )
        text = "A" * 200
        chunks = indexer._chunk_text(text)
        assert len(chunks) >= 3

    def test_chunks_have_overlap(self, tmp_path: Path):
        indexer = DocumentIndexer(
            VectorStore(), DummyEmbedder(), chunk_size=50, chunk_overlap=10
        )
        text = "ABCDEFGHIJ" * 20  # 200 chars
        chunks = indexer._chunk_text(text)
        if len(chunks) >= 2:
            # Last chars of first chunk should overlap with first chars of second
            overlap_chars = chunks[0][-10:]
            assert overlap_chars in chunks[1]


def json_bytes(data: dict) -> bytes:
    import json
    return json.dumps(data).encode("utf-8")


def store_count(indexer: DocumentIndexer) -> int:
    return indexer._store.count()
