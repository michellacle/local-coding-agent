"""Tests for vector_store — VectorStore and cosine similarity."""

import pytest

from local_agent.vector_store import VectorStore, DocumentChunk, _cosine_similarity


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_identical_vectors(self):
        assert _cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == pytest.approx(0.0)

    def test_both_zero_vectors(self):
        assert _cosine_similarity([0, 0], [0, 0]) == pytest.approx(0.0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            _cosine_similarity([1, 2], [1, 2, 3])

    def test_single_dimension(self):
        assert _cosine_similarity([5], [3]) == pytest.approx(1.0)

    def test_normalized_vectors(self):
        a = [0.6, 0.8]
        b = [0.8, 0.6]
        score = _cosine_similarity(a, b)
        assert 0 < score < 1


class TestVectorStore:
    """Test VectorStore operations."""

    def _make_vec(self, values: list[int]) -> list[float]:
        """Helper: convert ints to normalized float vector."""
        return [float(v) for v in values]

    def test_add_and_count(self):
        store = VectorStore()
        assert store.count() == 0

        store.add("hello", [1.0, 0.0, 0.0], "test.txt")
        assert store.count() == 1

    def test_add_returns_chunk(self):
        store = VectorStore()
        chunk = store.add("hello", [1.0, 0.0], "test.txt")
        assert isinstance(chunk, DocumentChunk)
        assert chunk.content == "hello"
        assert chunk.source == "test.txt"
        assert chunk.chunk_id == 0

    def test_add_metadata(self):
        store = VectorStore()
        chunk = store.add(
            "hello",
            [1.0, 0.0],
            "test.txt",
            metadata={"line": 1},
        )
        assert chunk.metadata["line"] == 1

    def test_add_many(self):
        store = VectorStore()
        chunks = store.add_many([
            ("a", [1.0, 0.0], "f1.txt", None),
            ("b", [0.0, 1.0], "f2.txt", None),
        ])
        assert len(chunks) == 2
        assert store.count() == 2

    def test_query_returns_most_similar(self):
        store = VectorStore()
        store.add("doc A", [1.0, 0.0, 0.0], "a.txt")
        store.add("doc B", [0.0, 1.0, 0.0], "b.txt")
        store.add("doc C", [1.0, 0.1, 0.0], "c.txt")

        # Query close to doc A and C
        results = store.query([1.0, 0.05, 0.0], top_k=2)
        assert len(results) == 2
        # First result should have highest score
        assert results[0][1] >= results[1][1]

    def test_query_empty_store(self):
        store = VectorStore()
        results = store.query([1.0, 0.0])
        assert results == []

    def test_query_top_k_limit(self):
        store = VectorStore()
        for i in range(10):
            store.add(f"doc {i}", [float(i), 0.0], f"doc_{i}.txt")

        results = store.query([9.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_query_min_score_filter(self):
        store = VectorStore()
        store.add("close", [1.0, 0.0], "a.txt")
        store.add("far", [0.0, 1.0], "b.txt")

        results = store.query([1.0, 0.0], min_score=0.5)
        assert len(results) == 1
        assert results[0][0].content == "close"

    def test_query_source_filter(self):
        store = VectorStore()
        store.add("a content", [1.0, 0.0], "a.txt")
        store.add("b content", [1.0, 0.0], "b.txt")

        results = store.query([1.0, 0.0], source_filter="a.txt")
        assert len(results) == 1
        assert results[0][0].source == "a.txt"

    def test_query_returns_tuples(self):
        store = VectorStore()
        store.add("test", [1.0, 0.0], "t.txt")

        results = store.query([1.0, 0.0])
        assert len(results) == 1
        chunk, score = results[0]
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(score, float)

    def test_query_sorted_by_score(self):
        store = VectorStore()
        store.add("very close", [0.99, 0.1], "a.txt")
        store.add("medium", [0.7, 0.7], "b.txt")
        store.add("far", [0.1, 0.99], "c.txt")

        results = store.query([1.0, 0.0], top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_clear(self):
        store = VectorStore()
        store.add("a", [1.0], "a.txt")
        store.add("b", [1.0], "b.txt")
        store.clear()
        assert store.count() == 0

    def test_clear_resets_counter(self):
        store = VectorStore()
        store.add("a", [1.0], "a.txt")
        store.clear()
        chunk = store.add("b", [1.0], "b.txt")
        assert chunk.chunk_id == 0

    def test_sources(self):
        store = VectorStore()
        store.add("a", [1.0], "a.txt")
        store.add("b", [1.0], "b.txt")
        store.add("c", [1.0], "a.txt")

        sources = store.sources()
        assert sources == {"a.txt", "b.txt"}

    def test_chunk_id_increments(self):
        store = VectorStore()
        c1 = store.add("a", [1.0], "a.txt")
        c2 = store.add("b", [1.0], "b.txt")
        c3 = store.add("c", [1.0], "c.txt")
        assert c1.chunk_id == 0
        assert c2.chunk_id == 1
        assert c3.chunk_id == 2

    def test_document_chunk_default_metadata(self):
        chunk = DocumentChunk(
            content="test",
            embedding=[1.0],
            source="t.txt",
            chunk_id=0,
        )
        assert chunk.metadata == {}
