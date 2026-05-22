"""Tests for Session Persistence — save/restore conversation state."""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from local_agent.session_persist import (
    SESSION_DIR,
    SessionManager,
    SessionMeta,
    SessionState,
)


# ------------------------------------------------------------------ #
# SessionMeta
# ------------------------------------------------------------------ #


class TestSessionMeta:
    def test_defaults(self):
        meta = SessionMeta()
        assert len(meta.session_id) == 12
        assert meta.turn_count == 0
        assert meta.model == ""
        assert meta.restored is False

    def test_custom_values(self):
        meta = SessionMeta(
            session_id="abc123",
            model="gpt-4",
            work_dir="/tmp",
            turn_count=5,
            restored=True,
        )
        assert meta.session_id == "abc123"
        assert meta.model == "gpt-4"
        assert meta.turn_count == 5


# ------------------------------------------------------------------ #
# SessionState
# ------------------------------------------------------------------ #


class TestSessionState:
    def test_save_and_load(self):
        state = SessionState(
            meta=SessionMeta(session_id="test1", model="test-model"),
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
        )
        with TemporaryDirectory() as tmp:
            path = state.save(f"{tmp}/session.json")
            assert Path(path).exists()

            loaded = SessionState.load(path)
            assert loaded.meta.session_id == "test1"
            assert loaded.meta.model == "test-model"
            assert loaded.meta.restored is True
            assert len(loaded.messages) == 2
            assert loaded.messages[0]["content"] == "hello"

    def test_save_auto_path(self):
        state = SessionState(
            meta=SessionMeta(session_id="auto1"),
            messages=[{"role": "user", "content": "x"}],
        )
        with TemporaryDirectory() as tmp:
            # Override SESSION_DIR temporarily
            import local_agent.session_persist as sp
            orig = sp.SESSION_DIR
            try:
                sp.SESSION_DIR = Path(tmp) / "sessions"
                path = state.save()
                assert Path(path).exists()
                assert Path(path).suffix == ".json"
            finally:
                sp.SESSION_DIR = orig

    def test_roundtrip_preserves_all_fields(self):
        state = SessionState(
            meta=SessionMeta(
                session_id="round",
                created_at=100.0,
                turn_count=10,
                model="m",
                work_dir="/w",
                restored=True,
            ),
            messages=[
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "tool result: ok"},
            ],
        )
        with TemporaryDirectory() as tmp:
            path = state.save(f"{tmp}/r.json")
            loaded = SessionState.load(path)

            assert loaded.meta.session_id == "round"
            assert loaded.meta.turn_count == 10
            assert loaded.meta.model == "m"
            assert loaded.meta.work_dir == "/w"
            assert loaded.meta.restored is True
            assert len(loaded.messages) == 3

    def test_latest_no_sessions(self):
        # SESSION_DIR probably doesn't exist in tests
        result = SessionState.latest()
        # Could be None or a valid state if the dir exists
        assert result is None or isinstance(result, SessionState)

    def test_list_sessions_empty(self):
        sessions = SessionState.list_sessions()
        assert isinstance(sessions, list)

    def test_list_sessions_with_data(self):
        import local_agent.session_persist as sp
        orig = sp.SESSION_DIR
        with TemporaryDirectory() as tmp:
            try:
                sp.SESSION_DIR = Path(tmp) / "sessions"
                sp.SESSION_DIR.mkdir(parents=True, exist_ok=True)

                # Save two sessions
                s1 = SessionState(
                    meta=SessionMeta(session_id="s1", turn_count=3, model="m1"),
                    messages=[{"role": "user", "content": "a"}],
                )
                s1.save(str(sp.SESSION_DIR / "s1.json"))
                time.sleep(0.01)

                s2 = SessionState(
                    meta=SessionMeta(session_id="s2", turn_count=7, model="m2"),
                    messages=[{"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}],
                )
                s2.save(str(sp.SESSION_DIR / "s2.json"))

                sessions = SessionState.list_sessions()
                assert len(sessions) == 2
                # Most recent first
                assert sessions[0]["session_id"] == "s2"
                assert sessions[1]["session_id"] == "s1"
                assert sessions[0]["message_count"] == 2
                assert sessions[1]["message_count"] == 1
            finally:
                sp.SESSION_DIR = orig


# ------------------------------------------------------------------ #
# SessionManager
# ------------------------------------------------------------------ #


class TestSessionManager:
    def test_create_new(self):
        mgr = SessionManager()
        state = mgr.create_new(model="test", work_dir="/tmp")
        assert mgr.state is state
        assert mgr.state.meta.model == "test"
        assert mgr.state.meta.work_dir == "/tmp"

    def test_save(self):
        mgr = SessionManager()
        mgr.create_new(model="t")
        mgr.state.messages.append({"role": "user", "content": "hi"})

        with TemporaryDirectory() as tmp:
            path = mgr.save()
            assert Path(path).exists()

    def test_save_without_session_raises(self):
        mgr = SessionManager()
        with pytest.raises(RuntimeError, match="No active session"):
            mgr.save()

    def test_try_restore_no_sessions(self):
        mgr = SessionManager()
        result = mgr.try_restore()
        # Likely False if no sessions exist
        if not result:
            assert mgr.state is None
        else:
            assert mgr.state is not None

    def test_try_restore_with_sessions(self):
        import local_agent.session_persist as sp
        orig = sp.SESSION_DIR
        with TemporaryDirectory() as tmp:
            try:
                sp.SESSION_DIR = Path(tmp) / "sessions"
                sp.SESSION_DIR.mkdir(parents=True, exist_ok=True)

                # Save a session
                state = SessionState(
                    meta=SessionMeta(session_id="restore-test", model="r-model"),
                    messages=[
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ],
                )
                state.save()

                mgr = SessionManager()
                result = mgr.try_restore()
                assert result is True
                assert mgr.state is not None
                assert len(mgr.state.messages) == 2
                assert mgr.state.meta.session_id == "restore-test"
            finally:
                sp.SESSION_DIR = orig

    def test_list_sessions(self):
        mgr = SessionManager()
        sessions = mgr.list_sessions()
        assert isinstance(sessions, list)
