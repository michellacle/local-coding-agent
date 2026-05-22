"""Tests for memory_store — persistent memory with SQLite + FTS5."""

import tempfile

import pytest

from local_agent.tools.memory_store import (
    MemoryStore,
    MemoryEntry,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a MemoryStore backed by a temp file."""
    db_file = str(tmp_path / "test_memory.db")
    store = MemoryStore(db_path=db_file)
    yield store
    store.close()


class TestMemoryEntry:
    """Test MemoryEntry dataclass."""

    def test_creation(self):
        entry = MemoryEntry(id=1, target="user", content="hello", created_at="now", updated_at="now")
        assert entry.id == 1
        assert entry.target == "user"
        assert entry.content == "hello"


class TestMemoryStoreAdd:
    """Test adding memory entries."""

    def test_add_user_entry(self, tmp_db):
        entry = tmp_db.add(target="user", content="User prefers Python")
        assert entry.id == 1
        assert entry.target == "user"
        assert "Python" in entry.content

    def test_add_memory_entry(self, tmp_db):
        entry = tmp_db.add(target="memory", content="Project uses pytest")
        assert entry.id == 1
        assert entry.target == "memory"

    def test_add_multiple(self, tmp_db):
        tmp_db.add(target="user", content="Entry 1")
        tmp_db.add(target="memory", content="Entry 2")
        entries = tmp_db.get_all()
        assert len(entries) == 2

    def test_add_invalid_target_raises(self, tmp_db):
        with pytest.raises(ValueError, match="target must be"):
            tmp_db.add(target="bad", content="x")

    def test_add_too_long_raises(self, tmp_db):
        with pytest.raises(ValueError, match="exceeds max length"):
            tmp_db.add(target="user", content="x" * 3000)


class TestMemoryStoreReplace:
    """Test replacing memory entries."""

    def test_replace_existing(self, tmp_db):
        tmp_db.add(target="user", content="Old value here")
        entry = tmp_db.replace(target="user", old_text="Old value", content="New value here")
        assert entry.content == "New value here"
        assert tmp_db.get_all(target="user")[0].content == "New value here"

    def test_replace_not_found_raises(self, tmp_db):
        with pytest.raises(ValueError, match="No entry found"):
            tmp_db.replace(target="user", old_text="nonexistent", content="x")

    def test_replace_wrong_target_raises(self, tmp_db):
        tmp_db.add(target="memory", content="Some fact")
        with pytest.raises(ValueError, match="No entry found"):
            tmp_db.replace(target="user", old_text="Some fact", content="x")

    def test_replace_case_insensitive(self, tmp_db):
        tmp_db.add(target="user", content="Hello World")
        entry = tmp_db.replace(target="user", old_text="hello world", content="Goodbye")
        assert entry.content == "Goodbye"


class TestMemoryStoreRemove:
    """Test removing memory entries."""

    def test_remove_existing(self, tmp_db):
        tmp_db.add(target="user", content="To be deleted")
        tmp_db.remove(target="user", old_text="To be deleted")
        assert tmp_db.get_all(target="user") == []

    def test_remove_not_found_raises(self, tmp_db):
        with pytest.raises(ValueError, match="No entry found"):
            tmp_db.remove(target="user", old_text="nonexistent")


class TestMemoryStoreGetAll:
    """Test listing memory entries."""

    def test_get_all_returns_all(self, tmp_db):
        tmp_db.add(target="user", content="U1")
        tmp_db.add(target="memory", content="M1")
        entries = tmp_db.get_all()
        assert len(entries) == 2

    def test_get_all_filtered(self, tmp_db):
        tmp_db.add(target="user", content="U1")
        tmp_db.add(target="memory", content="M1")
        entries = tmp_db.get_all(target="user")
        assert len(entries) == 1
        assert entries[0].target == "user"

    def test_get_all_empty(self, tmp_db):
        assert tmp_db.get_all() == []


class TestMemoryStoreSearch:
    """Test FTS5 full-text search."""

    def test_search_finds_match(self, tmp_db):
        tmp_db.add(target="user", content="I love Python")
        tmp_db.add(target="memory", content="Docker is used for builds")
        results = tmp_db.search(query="Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_search_multiple_matches(self, tmp_db):
        tmp_db.add(target="user", content="I use Linux")
        tmp_db.add(target="memory", content="Linux kernel version 6.x")
        results = tmp_db.search(query="Linux")
        assert len(results) == 2

    def test_search_no_match(self, tmp_db):
        tmp_db.add(target="user", content="Something")
        results = tmp_db.search(query="xyz_nonexistent")
        assert results == []

    def test_search_filtered_by_target(self, tmp_db):
        tmp_db.add(target="user", content="I like Rust")
        tmp_db.add(target="memory", content="Rust is a language")
        results = tmp_db.search(query="Rust", target="user")
        assert len(results) == 1
        assert results[0].target == "user"

    def test_search_respects_limit(self, tmp_db):
        for i in range(5):
            tmp_db.add(target="memory", content=f"item number {i}")
        results = tmp_db.search(query="item", limit=2)
        assert len(results) == 2


class TestMemoryStoreFormatForContext:
    """Test formatting memory for LLM context injection."""

    def test_format_with_entries(self, tmp_db):
        tmp_db.add(target="user", content="Prefers concise")
        tmp_db.add(target="memory", content="Uses pytest")
        text = tmp_db.format_for_context()
        assert "[USER]" in text
        assert "[NOTE]" in text
        assert "concise" in text

    def test_format_empty(self, tmp_db):
        text = tmp_db.format_for_context()
        assert text == ""

    def test_format_filtered(self, tmp_db):
        tmp_db.add(target="user", content="U1")
        tmp_db.add(target="memory", content="M1")
        text = tmp_db.format_for_context(target="user")
        assert "[USER]" in text
        assert "[NOTE]" not in text


class TestMemoryStoreTotalChars:
    """Test character counting."""

    def test_total_chars(self, tmp_db):
        tmp_db.add(target="user", content="abc")
        assert tmp_db.total_chars(target="user") == 3

    def test_total_chars_empty(self, tmp_db):
        assert tmp_db.total_chars() == 0

    def test_total_chars_all(self, tmp_db):
        tmp_db.add(target="user", content="abc")
        tmp_db.add(target="memory", content="xyz")
        assert tmp_db.total_chars() == 6


class TestMemoryStoreToolAPI:
    """Test the memory store via its public API (simulating tool function behavior)."""

    def test_add_via_api(self, tmp_db):
        entry = tmp_db.add(target="user", content="Test pref")
        assert entry.id >= 1
        assert tmp_db.get_all(target="user")[0].content == "Test pref"

    def test_replace_via_api(self, tmp_db):
        tmp_db.add(target="user", content="Old stuff")
        entry = tmp_db.replace(target="user", old_text="Old stuff", content="New stuff")
        assert entry.content == "New stuff"

    def test_remove_via_api(self, tmp_db):
        tmp_db.add(target="user", content="Delete me")
        tmp_db.remove(target="user", old_text="Delete me")
        assert tmp_db.get_all(target="user") == []

    def test_search_via_api(self, tmp_db):
        tmp_db.add(target="user", content="Hello world")
        results = tmp_db.search(query="hello")
        assert len(results) == 1
        assert "Hello world" in results[0].content

    def test_search_no_results_via_api(self, tmp_db):
        results = tmp_db.search(query="nonexistent")
        assert results == []

    def test_list_via_api(self, tmp_db):
        tmp_db.add(target="user", content="List me")
        entries = tmp_db.get_all()
        assert len(entries) == 1
        assert "List me" in entries[0].content

    def test_list_empty_via_api(self, tmp_db):
        entries = tmp_db.get_all()
        assert entries == []


class TestPersistence:
    """Test that data persists across store instances."""

    def test_persists_across_instantiation(self, tmp_path):
        db_file = str(tmp_path / "persist.db")
        store1 = MemoryStore(db_path=db_file)
        store1.add(target="user", content="Persistent data")
        store1.close()

        store2 = MemoryStore(db_path=db_file)
        entries = store2.get_all(target="user")
        assert len(entries) == 1
        assert entries[0].content == "Persistent data"
        store2.close()
