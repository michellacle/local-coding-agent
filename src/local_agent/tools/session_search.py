"""Session search — FTS5-backed retrieval over conversation history.

Stores agent conversation turns in a local SQLite database with FTS5
full-text search, enabling cross-session recall of past discussions,
decisions, and artifacts.

Supports three modes:
  1. DISCOVERY — search by query keyword, return matching sessions
  2. SCROLL — navigate inside a specific session by message ID
  3. BROWSE — list recent sessions chronologically

Each result includes:
  - session metadata (title, timestamp, message count)
  - snippet: matching text excerpt
  - bookends: first/last messages of the session
  - messages: context window around the match
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DB_PATH = "~/.local/share/local-coding-agent/sessions.db"
MAX_SESSIONS = 100  # Prune oldest when exceeded


@dataclass
class Message:
    """A single message in a conversation."""

    id: int
    session_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str  # ISO 8601
    source: str = ""  # Optional source tag


@dataclass
class SessionInfo:
    """Metadata about a session."""

    session_id: str
    title: str
    created_at: str
    message_count: int
    preview: str


@dataclass
class SearchResult:
    """Result from a discovery search."""

    session_id: str
    title: str
    when: str
    source: str
    snippet: str
    bookend_start: list[dict[str, str]]
    bookend_end: list[dict[str, str]]
    messages: list[dict[str, Any]]
    match_message_id: int
    messages_before: int
    messages_after: int


class SessionStore:
    """SQLite-backed session history with FTS5 search.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = DB_PATH
        self._db_path = str(Path(db_path).expanduser())
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and triggers."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                USING fts5(session_id, role, content, content='messages', content_rowid='id');

            CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, session_id, role, content)
                VALUES (new.id, new.session_id, new.role, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, session_id, role, content)
                VALUES ('delete', old.id, old.session_id, old.role, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, session_id, role, content)
                VALUES ('delete', old.id, old.session_id, old.role, old.content);
                INSERT INTO messages_fts(rowid, session_id, role, content)
                VALUES (new.id, new.session_id, new.role, new.content);
            END;
        """)
        self._conn.commit()

    def create_session(self, title: str) -> str:
        """Create a new session.

        Args:
            title: Descriptive title for the session.

        Returns:
            The generated session_id.
        """
        now = _iso_now()
        session_id = f"sess_{now.replace(':', '').replace('-', '')}_{_random_suffix()}"
        self._conn.execute(
            "INSERT INTO sessions (session_id, title, created_at, updated_at, message_count) "
            "VALUES (?, ?, ?, ?, 0)",
            (session_id, title, now, now),
        )
        self._conn.commit()
        self._prune_old_sessions()
        return session_id

    def add_message(self, session_id: str, role: str, content: str, source: str = "") -> int:
        """Add a message to a session.

        Args:
            session_id: The session to add to.
            role: "user" or "assistant".
            content: Message text.
            source: Optional source tag.

        Returns:
            The message ID.
        """
        now = _iso_now()
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, now, source),
        )
        self._conn.execute(
            "UPDATE sessions SET updated_at = ?, message_count = message_count + 1 "
            "WHERE session_id = ?",
            (now, session_id),
        )
        self._conn.commit()
        return self._conn.execute(
            "SELECT last_insert_rowid()"
        ).fetchone()[0]

    def browse(
        self, limit: int = 10
    ) -> list[SessionInfo]:
        """List recent sessions chronologically.

        Args:
            limit: Max sessions to return.

        Returns:
            List of SessionInfo ordered newest first.
        """
        rows = self._conn.execute(
            "SELECT session_id, title, created_at, message_count "
            "FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        sessions: list[SessionInfo] = []
        for row in rows:
            sid, title, created_at, count = row
            preview = self._get_preview(sid)
            sessions.append(SessionInfo(
                session_id=sid,
                title=title,
                created_at=created_at,
                message_count=count,
                preview=preview,
            ))
        return sessions

    def search(
        self,
        query: str,
        limit: int = 3,
        role_filter: str | None = None,
    ) -> list[SearchResult]:
        """Discovery search over conversation history.

        Runs FTS5 search, dedupes by session, returns top N sessions
        with matching excerpts and context.

        Args:
            query: Search keywords/phrases.
            limit: Max sessions to return.
            role_filter: Comma-separated roles (e.g., "user,assistant").

        Returns:
            List of SearchResult with match context.
        """
        # FTS5 search
        role_clause = ""
        if role_filter:
            roles = [r.strip() for r in role_filter.split(",") if r.strip()]
            if roles:
                role_clause = " AND role IN (" + ", ".join("?" * len(roles)) + ")"

        params: list[str] = [query]
        if role_filter:
            roles = [r.strip() for r in role_filter.split(",") if r.strip()]
            params.extend(roles)

        rows = self._conn.execute(
            f"SELECT m.id, m.session_id, m.role, m.content, m.timestamp "
            f"FROM messages m "
            f"INNER JOIN messages_fts f ON m.id = f.rowid "
            f"WHERE messages_fts MATCH ? {role_clause} "
            f"ORDER BY rank LIMIT ?",
            params + [limit * 5],  # Get more for dedup
        ).fetchall()

        # Dedupe by session_id (keep first match per session)
        seen: dict[str, tuple] = {}
        for row in rows:
            sid = row[1]
            if sid not in seen:
                seen[sid] = row

        results: list[SearchResult] = []
        for sid, match_row in list(seen.items())[:limit]:
            msg_id, _, role, content, timestamp = match_row

            # Get session info
            sess = self._conn.execute(
                "SELECT session_id, title, created_at FROM sessions WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if not sess:
                continue
            _, title, created_at = sess

            # Build bookends
            bookend_start = self._get_bookend(sid, "start", 3)
            bookend_end = self._get_bookend(sid, "end", 3)

            # Get messages around the match
            window_size = 5
            before_msgs = self._conn.execute(
                "SELECT id, role, content, timestamp FROM messages "
                "WHERE session_id = ? AND id < ? ORDER BY id DESC LIMIT ?",
                (sid, msg_id, window_size),
            ).fetchall()
            after_msgs = self._conn.execute(
                "SELECT id, role, content, timestamp FROM messages "
                "WHERE session_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
                (sid, msg_id, window_size),
            ).fetchall()

            before_list = [
                {"id": r[0], "role": r[1], "content": r[2], "timestamp": r[3], "hit": False}
                for r in reversed(before_msgs)
            ]
            match_msg = {
                "id": msg_id, "role": role, "content": content,
                "timestamp": timestamp, "hit": True,
            }
            after_list = [
                {"id": r[0], "role": r[1], "content": r[2], "timestamp": r[3], "hit": False}
                for r in after_msgs
            ]

            messages = before_list + [match_msg] + after_list

            # Truncate long content for snippets
            snippet = content[:200] + ("..." if len(content) > 200 else "")

            results.append(SearchResult(
                session_id=sid,
                title=title,
                when=created_at,
                source="session_db",
                snippet=snippet,
                bookend_start=bookend_start,
                bookend_end=bookend_end,
                messages=messages,
                match_message_id=msg_id,
                messages_before=len(before_msgs),
                messages_after=len(after_msgs),
            ))

        return results

    def scroll(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
    ) -> dict[str, Any]:
        """Scroll inside a session around a specific message.

        Args:
            session_id: The session to scroll in.
            around_message_id: The message to center on.
            window: Messages to return on each side (1-20).

        Returns:
            Dict with messages window and session info.
        """
        window = max(1, min(window, 20))

        before_msgs = self._conn.execute(
            "SELECT id, role, content, timestamp FROM messages "
            "WHERE session_id = ? AND id < ? ORDER BY id DESC LIMIT ?",
            (session_id, around_message_id, window),
        ).fetchall()

        anchor = self._conn.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE id = ?",
            (around_message_id,),
        ).fetchone()

        after_msgs = self._conn.execute(
            "SELECT id, role, content, timestamp FROM messages "
            "WHERE session_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
            (session_id, around_message_id, window),
        ).fetchall()

        before_list = [
            {"id": r[0], "role": r[1], "content": r[2], "timestamp": r[3]}
            for r in reversed(before_msgs)
        ]
        anchor_dict = (
            {"id": anchor[0], "role": anchor[1], "content": anchor[2], "timestamp": anchor[3]}
            if anchor else None
        )
        after_list = [
            {"id": r[0], "role": r[1], "content": r[2], "timestamp": r[3]}
            for r in after_msgs
        ]

        # Check boundaries
        all_msg_ids = [r[0] for r in before_msgs] + [r[0] for r in after_msgs]
        total_in_session = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]

        return {
            "session_id": session_id,
            "messages_before": before_list,
            "anchor": anchor_dict,
            "messages_after": after_list,
            "total_messages": total_in_session,
        }

    def get_session_info(self, session_id: str) -> SessionInfo | None:
        """Get metadata for a specific session.

        Args:
            session_id: The session ID.

        Returns:
            SessionInfo or None.
        """
        row = self._conn.execute(
            "SELECT session_id, title, created_at, message_count FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        return SessionInfo(
            session_id=row[0],
            title=row[1],
            created_at=row[2],
            message_count=row[3],
            preview=self._get_preview(session_id),
        )

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.

        Args:
            session_id: The session to delete.

        Returns:
            True if deleted, False if not found.
        """
        cursor = self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session_id,),
        )
        msg_count = cursor.rowcount
        cursor = self._conn.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()

    def _get_preview(self, session_id: str) -> str:
        """Get the first user message as preview."""
        row = self._conn.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'user' LIMIT 1",
            (session_id,),
        ).fetchone()
        if row:
            text = row[0]
            return text[:100] + ("..." if len(text) > 100 else "")
        return ""

    def _get_bookend(self, session_id: str, direction: str, count: int) -> list[dict[str, str]]:
        """Get bookend messages (start or end) of a session."""
        if direction == "start":
            order = "ASC"
        else:
            order = "DESC"

        rows = self._conn.execute(
            f"SELECT role, content FROM messages WHERE session_id = ? ORDER BY id {order} LIMIT ?",
            (session_id, count * 2),  # Get more to pair user+assistant
        ).fetchall()

        if direction == "end":
            rows = list(reversed(rows))

        result: list[dict[str, str]] = []
        seen_roles: set[str] = set()
        for role, content in rows:
            if len(result) >= count:
                break
            key = f"{direction}_{len(result)}"
            if role not in seen_roles or len(result) < count:
                snippet = content[:150] + ("..." if len(content) > 150 else "")
                result.append({"role": role, "content": snippet})
                seen_roles.add(role)

        return result

    def _prune_old_sessions(self) -> None:
        """Remove oldest sessions when exceeding MAX_SESSIONS."""
        excess = self._conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0] - MAX_SESSIONS
        if excess > 0:
            old_ids = self._conn.execute(
                "SELECT session_id FROM sessions ORDER BY created_at ASC LIMIT ?",
                (excess,),
            ).fetchall()
            for (sid,) in old_ids:
                self._conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            self._conn.commit()


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

_store: SessionStore | None = None


def _get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store


def session_search(
    query: str | None = None,
    session_id: str | None = None,
    around_message_id: int | None = None,
    window: int = 5,
    limit: int = 3,
    role_filter: str | None = None,
) -> str:
    """Search or browse past conversation sessions.

    Three modes:
    1. DISCOVERY — provide query to search across sessions
    2. SCROLL — provide session_id + around_message_id to navigate
    3. BROWSE — omit all args to see recent sessions

    Args:
        query: Search keywords (discovery mode).
        session_id: Session to scroll inside (scroll mode).
        around_message_id: Message to center on (scroll mode).
        window: Context window size for scroll (1-20, default 5).
        limit: Max sessions for discovery (default 3, max 10).
        role_filter: Filter by role(s) e.g., "user,assistant".

    Returns:
        JSON-formatted search results.
    """
    store = _get_store()

    # SCROLL mode
    if session_id and around_message_id is not None:
        result = store.scroll(session_id, around_message_id, window)
        return json.dumps(result, indent=2)

    # BROWSE mode
    if query is None:
        sessions = store.browse(limit)
        output: list[dict[str, Any]] = []
        for s in sessions:
            output.append({
                "session_id": s.session_id,
                "title": s.title,
                "when": s.created_at,
                "messages": s.message_count,
                "preview": s.preview,
            })
        return json.dumps(output, indent=2)

    # DISCOVERY mode
    results = store.search(query, limit=limit, role_filter=role_filter)
    output = []
    for r in results:
        output.append({
            "session_id": r.session_id,
            "title": r.title,
            "when": r.when,
            "snippet": r.snippet,
            "match_message_id": r.match_message_id,
            "messages_before": r.messages_before,
            "messages_after": r.messages_after,
            "messages": r.messages,
            "bookend_start": r.bookend_start,
            "bookend_end": r.bookend_end,
        })
    if not output:
        return "No matching sessions found."
    return json.dumps(output, indent=2)


def session_create(title: str) -> str:
    """Create a new session.

    Args:
        title: Descriptive title for the session.

    Returns:
        The generated session_id.
    """
    store = _get_store()
    return store.create_session(title)


def session_add_message(session_id: str, role: str, content: str, source: str = "") -> str:
    """Add a message to a session.

    Args:
        session_id: The session ID.
        role: "user" or "assistant".
        content: Message text.
        source: Optional source tag.

    Returns:
        Confirmation with message ID.
    """
    store = _get_store()
    msg_id = store.add_message(session_id, role, content, source)
    return f"Added message id={msg_id} to session {session_id}."


def session_delete(session_id: str) -> str:
    """Delete a session.

    Args:
        session_id: The session to delete.

    Returns:
        Confirmation message.
    """
    store = _get_store()
    if store.delete_session(session_id):
        return f"Deleted session {session_id}."
    return f"Session {session_id} not found."


def _iso_now() -> str:
    """Return current time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _random_suffix() -> str:
    """Generate a short random suffix."""
    import random
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
