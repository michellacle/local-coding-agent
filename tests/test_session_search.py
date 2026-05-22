"""Tests for session search — FTS over conversation history."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from local_agent.tools.session_search import (
    SessionStore,
    session_search,
    session_create,
    session_add_message,
    session_delete,
)


@pytest.fixture
def tmp_store():
    """Create a SessionStore with a temp database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = SessionStore(db_path=path)
    yield store
    store.close()
    os.unlink(path)


@pytest.fixture(autouse=True)
def reset_store(tmp_store):
    """Reset the module-level store singleton for each test."""
    from local_agent.tools import session_search as ss
    with patch.object(ss, "_get_store", return_value=tmp_store):
        yield tmp_store


class TestSessionCreate:
    """Test session creation."""

    def test_create_session(self, tmp_store):
        sid = tmp_store.create_session("Test session")
        assert sid.startswith("sess_")
        info = tmp_store.get_session_info(sid)
        assert info is not None
        assert info.title == "Test session"
        assert info.message_count == 0

    def test_create_multiple_sessions(self, tmp_store):
        sid1 = tmp_store.create_session("Session 1")
        sid2 = tmp_store.create_session("Session 2")
        assert sid1 != sid2

    def test_create_session_via_tool(self, tmp_store):
        result = session_create("My session")
        assert result.startswith("sess_")


class TestAddMessage:
    """Test adding messages to sessions."""

    def test_add_user_message(self, tmp_store):
        sid = tmp_store.create_session("Test")
        msg_id = tmp_store.add_message(sid, "user", "Hello world")
        assert msg_id > 0

        info = tmp_store.get_session_info(sid)
        assert info.message_count == 1

    def test_add_assistant_message(self, tmp_store):
        sid = tmp_store.create_session("Test")
        msg_id = tmp_store.add_message(sid, "assistant", "Hi there!")
        assert msg_id > 0

    def test_add_multiple_messages(self, tmp_store):
        sid = tmp_store.create_session("Test")
        tmp_store.add_message(sid, "user", "First")
        tmp_store.add_message(sid, "assistant", "Response 1")
        tmp_store.add_message(sid, "user", "Second")

        info = tmp_store.get_session_info(sid)
        assert info.message_count == 3

    def test_add_message_via_tool(self, tmp_store):
        sid = tmp_store.create_session("Test")
        result = session_add_message(sid, "user", "Hello")
        assert "id=" in result
        assert sid in result


class TestBrowse:
    """Test browsing recent sessions."""

    def test_browse_empty(self, tmp_store):
        sessions = tmp_store.browse()
        assert len(sessions) == 0

    def test_browse_with_sessions(self, tmp_store):
        tmp_store.create_session("First")
        tmp_store.create_session("Second")
        tmp_store.create_session("Third")

        sessions = tmp_store.browse()
        assert len(sessions) == 3
        # Newest first
        assert sessions[0].title == "Third"

    def test_browse_limit(self, tmp_store):
        for i in range(5):
            tmp_store.create_session(f"Session {i}")

        sessions = tmp_store.browse(limit=3)
        assert len(sessions) == 3

    def test_browse_via_tool(self, tmp_store):
        tmp_store.create_session("Test session")
        result = session_search(limit=5)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) >= 1


class TestSearch:
    """Test FTS5 discovery search."""

    def test_search_finds_match(self, tmp_store):
        sid = tmp_store.create_session("Auth refactor")
        tmp_store.add_message(sid, "user", "Let's refactor the authentication system")
        tmp_store.add_message(sid, "assistant", "I'll help with the auth refactor.")

        results = tmp_store.search("authentication")
        assert len(results) == 1
        assert results[0].session_id == sid

    def test_search_multiple_sessions(self, tmp_store):
        sid1 = tmp_store.create_session("Docker setup")
        tmp_store.add_message(sid1, "user", "Need to set up Docker networking")

        sid2 = tmp_store.create_session("Auth fix")
        tmp_store.add_message(sid2, "user", "Fix the authentication timeout")

        results = tmp_store.search("Docker")
        assert len(results) == 1
        assert results[0].session_id == sid1

    def test_search_no_match(self, tmp_store):
        sid = tmp_store.create_session("Test")
        tmp_store.add_message(sid, "user", "Hello world")

        results = tmp_store.search("nonexistent_xyz")
        assert len(results) == 0

    def test_search_with_bookends(self, tmp_store):
        sid = tmp_store.create_session("Test")
        tmp_store.add_message(sid, "user", "Start message")
        tmp_store.add_message(sid, "assistant", "Response to start")
        tmp_store.add_message(sid, "user", "This is the matching message for search")
        tmp_store.add_message(sid, "assistant", "Final message")

        results = tmp_store.search("matching")
        assert len(results) == 1
        assert len(results[0].bookend_start) > 0
        assert len(results[0].bookend_end) > 0

    def test_search_via_tool(self, tmp_store):
        sid = tmp_store.create_session("Test search")
        tmp_store.add_message(sid, "user", "Looking for database optimization tips")

        result = session_search(query="database")
        assert "No matching sessions found" not in result
        data = json.loads(result)
        assert len(data) >= 1


class TestScroll:
    """Test scrolling inside a session."""

    def test_scroll_centered_on_message(self, tmp_store):
        sid = tmp_store.create_session("Test")
        id1 = tmp_store.add_message(sid, "user", "Message 1")
        id2 = tmp_store.add_message(sid, "assistant", "Message 2")
        id3 = tmp_store.add_message(sid, "user", "Message 3")
        id4 = tmp_store.add_message(sid, "assistant", "Message 4")
        id5 = tmp_store.add_message(sid, "user", "Message 5")

        result = tmp_store.scroll(sid, id3, window=2)
        assert result["anchor"]["id"] == id3
        assert result["anchor"]["content"] == "Message 3"
        assert len(result["messages_before"]) <= 2
        assert len(result["messages_after"]) <= 2

    def test_scroll_at_start(self, tmp_store):
        sid = tmp_store.create_session("Test")
        id1 = tmp_store.add_message(sid, "user", "First")
        id2 = tmp_store.add_message(sid, "assistant", "Second")

        result = tmp_store.scroll(sid, id1, window=5)
        assert result["anchor"]["id"] == id1
        assert len(result["messages_before"]) == 0

    def test_scroll_at_end(self, tmp_store):
        sid = tmp_store.create_session("Test")
        id1 = tmp_store.add_message(sid, "user", "First")
        id2 = tmp_store.add_message(sid, "assistant", "Last")

        result = tmp_store.scroll(sid, id2, window=5)
        assert result["anchor"]["id"] == id2
        assert len(result["messages_after"]) == 0

    def test_scroll_window_clamped(self, tmp_store):
        sid = tmp_store.create_session("Test")
        id1 = tmp_store.add_message(sid, "user", "Msg")

        # Window > 20 should be clamped to 20
        result = tmp_store.scroll(sid, id1, window=100)
        assert result["anchor"] is not None

    def test_scroll_invalid_session(self, tmp_store):
        result = tmp_store.scroll("nonexistent", 999, window=5)
        assert result["anchor"] is None


class TestDelete:
    """Test session deletion."""

    def test_delete_session(self, tmp_store):
        sid = tmp_store.create_session("Delete me")
        tmp_store.add_message(sid, "user", "Some content")

        assert tmp_store.delete_session(sid) is True
        assert tmp_store.get_session_info(sid) is None

    def test_delete_nonexistent(self, tmp_store):
        assert tmp_store.delete_session("nonexistent") is False

    def test_delete_via_tool(self, tmp_store):
        sid = tmp_store.create_session("Delete me")
        result = session_delete(sid)
        assert "Deleted" in result

    def test_delete_nonexistent_via_tool(self, tmp_store):
        result = session_delete("nonexistent")
        assert "not found" in result


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, tmp_store):
        """Create session, add messages, search, scroll, delete."""
        sid = tmp_store.create_session("RAG system design")
        tmp_store.add_message(sid, "user", "I want to build a RAG system")
        tmp_store.add_message(sid, "assistant", "Great, let's plan the RAG architecture")
        tmp_store.add_message(sid, "user", "Use ChromaDB for the vector store")

        # Search should find it
        results = tmp_store.search("RAG")
        assert len(results) == 1
        assert results[0].session_id == sid

        # Browse should list it
        sessions = tmp_store.browse()
        assert any(s.session_id == sid for s in sessions)

        # Delete should work
        assert tmp_store.delete_session(sid) is True
        assert tmp_store.get_session_info(sid) is None

    def test_search_dedupe_across_sessions(self, tmp_store):
        """Messages in multiple sessions matching the same query."""
        sid1 = tmp_store.create_session("Session A")
        tmp_store.add_message(sid1, "user", "Fix the login bug")

        sid2 = tmp_store.create_session("Session B")
        tmp_store.add_message(sid2, "user", "Fix the login timeout")

        results = tmp_store.search("login")
        # Should return both sessions, deduped
        assert len(results) == 2