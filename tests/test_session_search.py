"""Tests for session_search — FTS-backed conversation history."""

import os
import sqlite3
import tempfile
import pytest

# Import the module under test
from local_agent.tools.session_search import (
    SessionStore,
    session_search,
    session_create,
    session_add_message,
    session_delete,
    _get_store,
    _store,
)


@pytest.fixture(autouse=True)
def _reset_global_store():
    """Reset the global singleton between tests."""
    import local_agent.tools.session_search as mod
    mod._store = None
    yield
    mod._store = None


@pytest.fixture
def store():
    """Create a SessionStore backed by a temp database."""
    tmp = tempfile.mktemp(suffix=".db")
    s = SessionStore(db_path=tmp)
    yield s
    s.close()
    os.unlink(tmp)


class TestSessionStore:
    def test_create_session(self, store):
        sid = store.create_session("test session")
        assert sid.startswith("sess_")
        info = store.get_session_info(sid)
        assert info is not None
        assert info.title == "test session"
        assert info.message_count == 0

    def test_add_message(self, store):
        sid = store.create_session("msg test")
        mid = store.add_message(sid, "user", "hello world")
        assert mid == 1
        info = store.get_session_info(sid)
        assert info.message_count == 1

    def test_browse(self, store):
        store.create_session("first")
        store.create_session("second")
        sessions = store.browse(limit=10)
        assert len(sessions) == 2
        assert sessions[0].title == "second"  # newest first

    def test_browse_empty(self, store):
        sessions = store.browse()
        assert sessions == []

    def test_search_discovery(self, store):
        sid = store.create_session("docker networking")
        store.add_message(sid, "user", "How do I set up docker networking between containers?")
        store.add_message(sid, "assistant", "You can use docker network create ...")
        results = store.search("docker networking")
        assert len(results) == 1
        assert results[0].session_id == sid
        assert results[0].match_message_id > 0

    def test_search_no_results(self, store):
        sid = store.create_session("something else")
        store.add_message(sid, "user", "talking about rain")
        results = store.search("quantum entanglement")
        assert results == []

    def test_search_role_filter(self, store):
        sid = store.create_session("filtered")
        store.add_message(sid, "user", "user message about api")
        store.add_message(sid, "assistant", "assistant message about api")
        results = store.search("api", role_filter="user")
        assert len(results) == 1
        assert results[0].messages[0]["role"] == "user"

    def test_scroll(self, store):
        sid = store.create_session("scroll test")
        m1 = store.add_message(sid, "user", "first message")
        m2 = store.add_message(sid, "assistant", "second message")
        m3 = store.add_message(sid, "user", "third message")
        result = store.scroll(sid, m2, window=5)
        assert result["session_id"] == sid
        assert result["anchor"]["id"] == m2
        assert result["total_messages"] == 3

    def test_scroll_missing_anchor(self, store):
        sid = store.create_session("scroll test")
        store.add_message(sid, "user", "only message")
        result = store.scroll(sid, 9999, window=5)
        assert result["anchor"] is None

    def test_delete_session(self, store):
        sid = store.create_session("delete me")
        store.add_message(sid, "user", "bye")
        assert store.delete_session(sid) is True
        assert store.get_session_info(sid) is None
        assert store.delete_session(sid) is False  # already gone

    def test_prune_old_sessions(self, store):
        # Create many sessions to exceed MAX_SESSIONS
        for i in range(105):
            store.create_session(f"session {i}")
        sessions = store.browse(limit=200)
        assert len(sessions) <= 100


class TestPublicFunctions:
    def test_session_create_public(self):
        sid = session_create("public test")
        assert sid.startswith("sess_")
        # Clean up
        session_delete(sid)

    def test_session_add_message_public(self):
        sid = session_create("public msg test")
        result = session_add_message(sid, "user", "test message")
        assert "Added message" in result
        session_delete(sid)

    def test_session_search_discovery(self):
        sid = session_create("searchable session")
        session_add_message(sid, "user", "unique keyword for search")
        result = session_search(query="unique keyword")
        assert "unique keyword" in result
        session_delete(sid)

    def test_session_search_browse(self):
        result = session_search()
        # Should return valid JSON array
        import json
        data = json.loads(result)
        assert isinstance(data, list)

    def test_session_search_scroll(self):
        sid = session_create("scrollable")
        mid = session_add_message(sid, "user", "anchor message")
        # Parse message ID from response
        import re
        mid_int = int(re.search(r"id=(\d+)", mid).group(1))  # type: ignore[union-attr]
        result = session_search(session_id=sid, around_message_id=mid_int)
        import json
        data = json.loads(result)
        assert data["anchor"]["id"] == mid_int
        session_delete(sid)


class TestBookends:
    def test_bookend_start(self, store):
        sid = store.create_session("bookend test")
        store.add_message(sid, "user", "first")
        store.add_message(sid, "assistant", "second")
        store.add_message(sid, "user", "third")
        store.add_message(sid, "assistant", "fourth")
        store.add_message(sid, "user", "fifth")
        results = store.search("third")
        assert len(results) == 1
        assert len(results[0].bookend_start) > 0

    def test_bookend_end(self, store):
        sid = store.create_session("bookend end")
        store.add_message(sid, "user", "beginning")
        store.add_message(sid, "assistant", "middle")
        store.add_message(sid, "user", "end message")
        results = store.search("middle")
        assert len(results[0].bookend_end) > 0
