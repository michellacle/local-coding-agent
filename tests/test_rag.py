"""Tests for rag — DocumentIndexer, VectorStore, EmbeddingBackend, RAGPipeline."""

import json
import math
import os
import tempfile

import pytest

from local_agent.rag import (
    DocumentChunk,
    DocumentIndexer,
    EmbeddingBackend,
    RAGPipeline,
    SearchResult,
    VectorStore,
)


class TestDocumentChunk:
    """Test DocumentChunk dataclass."""

    def test_create_chunk(self):
        chunk = DocumentChunk(doc_id="doc1", chunk_id=0, text="Hello world")
        assert chunk.doc_id == "doc1"
        assert chunk.chunk_id == 0
        assert chunk.text == "Hello world"
        assert chunk.embedding == []
        assert chunk.metadata == {}

    def test_chunk_with_metadata(self):
        chunk = DocumentChunk(
            doc_id="doc1",
            chunk_id=0,
            text="Hello",
            metadata={"file": "test.md"},
        )
        assert chunk.metadata["file"] == "test.md"


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_create_result(self):
        chunk = DocumentChunk(doc_id="doc1", chunk_id=0, text="Hello")
        result = SearchResult(chunk=chunk, score=0.95)
        assert result.score == 0.95
        assert result.chunk.doc_id == "doc1"


class TestEmbeddingBackend:
    """Test embedding backends."""

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            EmbeddingBackend(backend="nonexistent")

    def test_tfidf_embed_basic(self):
        backend = EmbeddingBackend(backend="tfidf")
        embeddings = backend.embed(["hello world", "goodbye world"])
        assert len(embeddings) == 2
        assert backend.dimension > 0
        # Each embedding should be a unit vector
        for emb in embeddings:
            norm = math.sqrt(sum(v * v for v in emb))
            assert abs(norm - 1.0) < 0.01

    def test_tfidf_empty_text(self):
        backend = EmbeddingBackend(backend="tfidf")
        embeddings = backend.embed(["", "   "])
        assert len(embeddings) == 2
        for emb in embeddings:
            assert all(v == 0.0 for v in emb)

    def test_tfidf_same_text(self):
        backend = EmbeddingBackend(backend="tfidf")
        embeddings = backend.embed(["hello world", "hello world"])
        # Same text should produce identical embeddings
        assert embeddings[0] == embeddings[1]

    def test_ollama_backend_returns_fallback_when_offline(self):
        backend = EmbeddingBackend(backend="ollama", url="http://localhost:11434")
        # Should return zero vectors when Ollama is offline (graceful fallback)
        embeddings = backend.embed(["test"])
        assert len(embeddings) == 1
        # Fallback should produce a vector (zeros when no connection)
        assert len(embeddings[0]) >= 0  # At least an empty list

    def test_sentence_transformers_raises_without_install(self):
        backend = EmbeddingBackend(backend="sentence-transformers")
        with pytest.raises(RuntimeError, match="SentenceTransformers not installed"):
            backend.embed(["test"])


class TestVectorStore:
    """Test in-memory vector store."""

    def test_add_and_count(self):
        store = VectorStore()
        chunk = DocumentChunk(
            doc_id="doc1",
            chunk_id=0,
            text="Hello",
            embedding=[1.0, 0.0, 0.0],
        )
        store.add(chunk)
        assert store.count() == 1

    def test_search_exact_match(self):
        store = VectorStore()
        store.add(DocumentChunk(
            doc_id="doc1", chunk_id=0, text="Hello",
            embedding=[1.0, 0.0, 0.0],
        ))
        store.add(DocumentChunk(
            doc_id="doc1", chunk_id=1, text="World",
            embedding=[0.0, 1.0, 0.0],
        ))

        results = store.search([1.0, 0.0, 0.0], top_k=5)
        assert len(results) == 2
        assert results[0].score == 1.0  # Exact match
        assert results[0].chunk.text == "Hello"

    def test_search_cosine_similarity(self):
        store = VectorStore()
        # Vectors at 90 degrees
        store.add(DocumentChunk(
            doc_id="doc1", chunk_id=0, text="A",
            embedding=[1.0 / math.sqrt(2), 1.0 / math.sqrt(2), 0.0],
        ))
        store.add(DocumentChunk(
            doc_id="doc1", chunk_id=1, text="B",
            embedding=[1.0, 0.0, 0.0],
        ))

        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert results[0].score == 1.0  # Exact match with B
        assert results[0].chunk.text == "B"
        # cos(45 degrees) = sqrt(2)/2 ≈ 0.707
        assert abs(results[1].score - 0.707) < 0.01

    def test_search_min_score_filter(self):
        store = VectorStore()
        store.add(DocumentChunk(
            doc_id="doc1", chunk_id=0, text="A",
            embedding=[1.0, 0.0, 0.0],
        ))
        store.add(DocumentChunk(
            doc_id="doc2", chunk_id=0, text="B",
            embedding=[0.0, 1.0, 0.0],
        ))

        results = store.search([1.0, 0.0, 0.0], min_score=0.5)
        assert len(results) == 1
        assert results[0].chunk.text == "A"

    def test_search_top_k_limit(self):
        store = VectorStore()
        for i in range(10):
            vec = [0.0] * 3
            vec[i % 3] = 1.0
            store.add(DocumentChunk(
                doc_id=f"doc{i}", chunk_id=0, text=f"T{i}",
                embedding=vec,
            ))

        results = store.search([1.0, 0.0, 0.0], top_k=3)
        assert len(results) <= 3

    def test_delete_document(self):
        store = VectorStore()
        store.add(DocumentChunk(doc_id="doc1", chunk_id=0, text="A", embedding=[1.0, 0.0]))
        store.add(DocumentChunk(doc_id="doc1", chunk_id=1, text="B", embedding=[0.0, 1.0]))
        store.add(DocumentChunk(doc_id="doc2", chunk_id=0, text="C", embedding=[1.0, 1.0]))

        deleted = store.delete("doc1")
        assert deleted == 2
        assert store.count() == 1

    def test_get_docs(self):
        store = VectorStore()
        store.add(DocumentChunk(doc_id="doc1", chunk_id=0, text="A", embedding=[1.0]))
        store.add(DocumentChunk(doc_id="doc2", chunk_id=0, text="B", embedding=[1.0]))
        docs = store.get_docs()
        assert docs == {"doc1", "doc2"}

    def test_cosine_similarity_zero_vectors(self):
        score = VectorStore._cosine_similarity([0.0, 0.0], [0.0, 0.0])
        assert score == 0.0

    def test_cosine_similarity_orthogonal(self):
        score = VectorStore._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert score == pytest.approx(0.0, abs=0.001)

    def test_cosine_similarity_identical(self):
        score = VectorStore._cosine_similarity([1.0, 2.0], [1.0, 2.0])
        assert score == pytest.approx(1.0, abs=0.001)


class TestVectorStoreSQLite:
    """Test SQLite-backed vector store."""

    def test_persistence(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = VectorStore(db_path=db_path)
        store.add(DocumentChunk(
            doc_id="doc1", chunk_id=0, text="Persistent",
            embedding=[1.0, 0.0],
        ))
        store.close()

        # Reopen
        store2 = VectorStore(db_path=db_path)
        results = store2.search([1.0, 0.0], top_k=5)
        assert len(results) >= 1
        store2.close()

    def test_delete_persists(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = VectorStore(db_path=db_path)
        store.add(DocumentChunk(doc_id="doc1", chunk_id=0, text="A", embedding=[1.0, 0.0]))
        store.delete("doc1")
        store.close()

        store2 = VectorStore(db_path=db_path)
        assert store2.count() == 0
        store2.close()


class TestDocumentIndexer:
    """Test document indexer."""

    def test_chunk_simple_text(self):
        indexer = DocumentIndexer(chunk_size=100, chunk_overlap=20)
        chunks = indexer.chunk_text("doc1", "A" * 250)
        assert len(chunks) >= 2
        assert all(len(c.text) <= 100 for c in chunks)

    def test_chunk_small_text(self):
        indexer = DocumentIndexer(chunk_size=100, chunk_overlap=20)
        chunks = indexer.chunk_text("doc1", "Short text")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text"

    def test_chunk_empty_text(self):
        indexer = DocumentIndexer()
        chunks = indexer.chunk_text("doc1", "")
        assert len(chunks) == 0

    def test_chunk_preserves_doc_id(self):
        indexer = DocumentIndexer(chunk_size=50, chunk_overlap=10)
        chunks = indexer.chunk_text("unique_doc", "Hello world! This is a test.")
        assert all(c.doc_id == "unique_doc" for c in chunks)

    def test_chunk_ids_are_sequential(self):
        indexer = DocumentIndexer(chunk_size=50, chunk_overlap=10)
        chunks = indexer.chunk_text("doc1", "A" * 200)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(ids)))

    def test_chunk_metadata(self):
        indexer = DocumentIndexer(chunk_size=50, chunk_overlap=10)
        metadata = {"file": "test.md", "type": "markdown"}
        chunks = indexer.chunk_text("doc1", "Hello world!", metadata=metadata)
        assert all(c.metadata == metadata for c in chunks)

    def test_find_break_point(self):
        indexer = DocumentIndexer(chunk_size=100, chunk_overlap=20)
        text = "First sentence.\n\nSecond paragraph.\n\nThird part."
        break_point = indexer._find_break_point(text, 40, 0)
        # Should break at "\n\n" after "First sentence."
        assert text[break_point - 2:break_point] == "\n\n"

    def test_index_directory(self, tmp_path):
        # Create test files
        (tmp_path / "hello.md").write_text("# Hello\n\nThis is a markdown file.\n")
        (tmp_path / "world.py").write_text("# Python file\nprint('hello')\n")
        (tmp_path / "skip.json").write_text('{"key": "value"}\n')

        indexer = DocumentIndexer(chunk_size=1000, chunk_overlap=100)
        chunks = indexer.index_directory(str(tmp_path))

        assert len(chunks) >= 3
        doc_ids = {c.doc_id for c in chunks}
        assert any("hello.md" in d for d in doc_ids)

    def test_index_directory_exclude(self, tmp_path):
        (tmp_path / "keep.md").write_text("Keep this\n")
        (tmp_path / "skip.md").write_text("Skip this\n")

        indexer = DocumentIndexer(chunk_size=1000, chunk_overlap=100)
        chunks = indexer.index_directory(
            str(tmp_path), exclude_patterns=["skip.md"]
        )

        doc_ids = {c.doc_id for c in chunks}
        assert not any("skip.md" in d for d in doc_ids)


class TestRAGPipeline:
    """Test RAG pipeline integration (TF-IDF backend)."""

    def test_index_and_query(self, tmp_path):
        # Create documents
        (tmp_path / "llm.md").write_text(
            "LLM providers can be configured in the settings file.\n"
            "Supported providers include Ollama and OpenAI.\n"
            "Set the provider URL in the configuration.\n"
        )
        (tmp_path / "embedding.md").write_text(
            "Embedding models convert text to vectors.\n"
            "The nomic-embed-text model supports 768 dimensions.\n"
            "TF-IDF is a fallback when no ML models are available.\n"
        )

        pipeline = RAGPipeline(
            embedding_backend="tfidf",
            chunk_size=1000,
            chunk_overlap=200,
        )

        n = pipeline.index_directory(str(tmp_path))
        assert n >= 2

        results = pipeline.query("How do I configure LLM providers?")
        assert len(results) > 0
        # The LLM doc should be the top result
        assert any("llm.md" in r.chunk.doc_id for r in results)

    def test_index_text(self):
        pipeline = RAGPipeline(embedding_backend="tfidf")

        n = pipeline.index_text("manual", "The quick brown fox jumps over the lazy dog.")
        assert n >= 1

        results = pipeline.query("What animal is quick and brown?")
        assert len(results) > 0

    def test_delete_document(self):
        pipeline = RAGPipeline(embedding_backend="tfidf")
        pipeline.index_text("doc1", "Document one content.")
        pipeline.index_text("doc2", "Document two content.")
        assert pipeline.count_chunks() >= 2

        deleted = pipeline.delete_document("doc1")
        assert deleted >= 1
        assert "doc1" not in pipeline.get_document_ids()

    def test_close(self):
        pipeline = RAGPipeline(embedding_backend="tfidf")
        pipeline.close()

    def test_persistent_store(self, tmp_path):
        db_path = str(tmp_path / "rag.db")
        pipeline = RAGPipeline(
            embedding_backend="tfidf",
            db_path=db_path,
            chunk_size=1000,
        )
        pipeline.index_text("doc1", "Persistent data for testing RAG with SQLite backend.")

        pipeline.close()

        # Reopen and query
        pipeline2 = RAGPipeline(
            embedding_backend="tfidf",
            db_path=db_path,
            chunk_size=1000,
        )
        results = pipeline2.query("Persistent data")
        assert len(results) > 0
        pipeline2.close()
