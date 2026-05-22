"""Persistent memory — user profile and environment notes.

Stores durable facts that survive across sessions:
- User profile: who the user is, preferences, communication style
- Memory notes: environment facts, project conventions, tool quirks, lessons

Backed by SQLite for persistence with FTS5 for full-text search.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MemoryEntry:
    """A single memory entry."""

    id: int
    target: str  # "user" or "memory"
    content: str
    created_at: str  # ISO timestamp
    updated_at: str  # ISO timestamp


class MemoryStore:
    """SQLite-backed persistent memory store.

    Supports two stores:
    - "user": who the user is, preferences, habits
    - "memory": environment facts, project notes, lessons learned

    Each entry has a character budget (~2,200 total per store) to keep
    context window overhead low.
    """

    DB_PATH = "~/.local/share/local-coding-agent/memory.db"
    MAX_TOTAL_CHARS = 4400  # Soft limit per store
    MAX_ENTRY_CHARS = 2000  # Hard limit per entry

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = self.DB_PATH
        self._db_path = str(Path(db_path).expanduser())
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL CHECK(target IN ('user', 'memory')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(target, content, content='memory', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
                INSERT INTO memory_fts(rowid, target, content)
                VALUES (new.id, new.target, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, target, content)
                VALUES ('delete', old.id, old.target, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, target, content)
                VALUES ('delete', old.id, old.target, old.content);
                INSERT INTO memory_fts(rowid, target, content)
                VALUES (new.id, new.target, new.content);
            END;
        """)
        self._conn.commit()

    def add(self, target: str, content: str) -> MemoryEntry:
        """Add a new memory entry.

        Args:
            target: "user" or "memory".
            content: The entry content.

        Returns:
            The created MemoryEntry.

        Raises:
            ValueError: If content exceeds max length or target is invalid.
        """
        if target not in ("user", "memory"):
            raise ValueError(f"target must be 'user' or 'memory', got '{target}'")
        if len(content) > self.MAX_ENTRY_CHARS:
            raise ValueError(
                f"Content exceeds max length ({self.MAX_ENTRY_CHARS} chars)"
            )

        now = _iso_now()
        cur = self._conn.execute(
            "INSERT INTO memory (target, content, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (target, content, now, now),
        )
        self._conn.commit()
        entry_id = cur.lastrowid
        assert entry_id is not None
        return MemoryEntry(
            id=entry_id, target=target, content=content,
            created_at=now, updated_at=now,
        )

    def replace(self, target: str, old_text: str, content: str) -> MemoryEntry:
        """Replace an existing memory entry.

        Matches the first entry in the given target whose content contains
        old_text (case-insensitive substring match).

        Args:
            target: "user" or "memory".
            old_text: Substring identifying the entry to replace.
            content: New content.

        Returns:
            The updated MemoryEntry.

        Raises:
            ValueError: If no matching entry is found or content too long.
        """
        if len(content) > self.MAX_ENTRY_CHARS:
            raise ValueError(
                f"Content exceeds max length ({self.MAX_ENTRY_CHARS} chars)"
            )

        row = self._find_entry(target, old_text)
        if row is None:
            raise ValueError(
                f"No entry found in '{target}' containing '{old_text}'"
            )

        entry_id, _, created_at = row
        now = _iso_now()
        self._conn.execute(
            "UPDATE memory SET content = ?, updated_at = ? WHERE id = ?",
            (content, now, entry_id),
        )
        self._conn.commit()
        return MemoryEntry(
            id=entry_id, target=target, content=content,
            created_at=created_at, updated_at=now,
        )

    def remove(self, target: str, old_text: str) -> None:
        """Remove a memory entry.

        Args:
            target: "user" or "memory".
            old_text: Substring identifying the entry to remove.

        Raises:
            ValueError: If no matching entry is found.
        """
        row = self._find_entry(target, old_text)
        if row is None:
            raise ValueError(
                f"No entry found in '{target}' containing '{old_text}'"
            )

        self._conn.execute("DELETE FROM memory WHERE id = ?", (row[0],))
        self._conn.commit()

    def get_all(self, target: str | None = None) -> list[MemoryEntry]:
        """Get all memory entries, optionally filtered by target.

        Args:
            target: If set, only return entries for this target.

        Returns:
            List of MemoryEntry objects ordered by id.
        """
        if target:
            rows = self._conn.execute(
                "SELECT id, target, content, created_at, updated_at "
                "FROM memory WHERE target = ? ORDER BY id",
                (target,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, target, content, created_at, updated_at "
                "FROM memory ORDER BY id"
            ).fetchall()

        return [
            MemoryEntry(id=r[0], target=r[1], content=r[2],
                        created_at=r[3], updated_at=r[4])
            for r in rows
        ]

    def search(self, query: str, target: str | None = None, limit: int = 10) -> list[MemoryEntry]:
        """Full-text search over memory entries using FTS5.

        Args:
            query: Search query (FTS5 syntax).
            target: If set, filter to this target.
            limit: Max results.

        Returns:
            List of matching MemoryEntry objects.
        """
        if target:
            rows = self._conn.execute(
                "SELECT m.id, m.target, m.content, m.created_at, m.updated_at "
                "FROM memory m "
                "INNER JOIN memory_fts f ON m.id = f.rowid "
                "WHERE m.target = ? AND memory_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (target, query, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT m.id, m.target, m.content, m.created_at, m.updated_at "
                "FROM memory m "
                "INNER JOIN memory_fts f ON m.id = f.rowid "
                "WHERE memory_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()

        return [
            MemoryEntry(id=r[0], target=r[1], content=r[2],
                        created_at=r[3], updated_at=r[4])
            for r in rows
        ]

    def format_for_context(self, target: str | None = None) -> str:
        """Format memory entries for injection into LLM context.

        Returns a compact text block suitable for system prompt injection.

        Args:
            target: If set, only format entries for this target.

        Returns:
            Formatted string with memory content.
        """
        entries = self.get_all(target)
        if not entries:
            return ""

        lines: list[str] = []
        for entry in entries:
            prefix = "USER" if entry.target == "user" else "NOTE"
            lines.append(f"[{prefix}] {entry.content}")
        return "\n".join(lines)

    def total_chars(self, target: str | None = None) -> int:
        """Count total characters in all entries, optionally filtered by target.

        Args:
            target: If set, only count entries for this target.

        Returns:
            Total character count.
        """
        if target:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM memory WHERE target = ?",
                (target,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM memory"
            ).fetchone()
        return row[0]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()

    def _find_entry(self, target: str, old_text: str):
        """Find first entry matching old_text (case-insensitive substring).

        Returns (id, content, created_at) or None.
        """
        return self._conn.execute(
            "SELECT id, content, created_at FROM memory "
            "WHERE target = ? AND LOWER(content) LIKE LOWER(?) "
            "LIMIT 1",
            (target, f"%{old_text}%"),
        ).fetchone()


# ---------------------------------------------------------------------------
# Public tool functions registered with the tool registry
# ---------------------------------------------------------------------------


# Module-level singleton — reused across calls
_store: MemoryStore | None = None


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def memory(action: str, target: str, content: str | None = None, old_text: str | None = None) -> str:
    """Save, update, or remove persistent memory entries.

    Memory survives across sessions and is injected into future turns.

    Args:
        action: One of "add", "replace", "remove".
        target: "user" for user profile, "memory" for agent notes.
        content: Entry content. Required for "add" and "replace".
        old_text: Substring identifying an existing entry.
                  Required for "replace" and "remove".

    Returns:
        Confirmation message.
    """
    store = _get_store()

    if action == "add":
        if content is None:
            raise ValueError("content is required for action='add'")
        entry = store.add(target=target, content=content)
        return f"Added to {target} memory (id={entry.id})."

    elif action == "replace":
        if content is None:
            raise ValueError("content is required for action='replace'")
        if old_text is None:
            raise ValueError("old_text is required for action='replace'")
        entry = store.replace(target=target, old_text=old_text, content=content)
        return f"Replaced entry in {target} memory (id={entry.id})."

    elif action == "remove":
        if old_text is None:
            raise ValueError("old_text is required for action='remove'")
        store.remove(target=target, old_text=old_text)
        return f"Removed entry from {target} memory."

    else:
        raise ValueError(f"Unknown action '{action}'. Use 'add', 'replace', or 'remove'.")


def memory_search(query: str, target: str | None = None, limit: int = 10) -> str:
    """Search persistent memory entries using full-text search.

    Args:
        query: Search query (keywords, phrases).
        target: If set, only search in this target ("user" or "memory").
        limit: Max results to return.

    Returns:
        Formatted search results.
    """
    store = _get_store()
    results = store.search(query=query, target=target, limit=limit)
    if not results:
        return "No matching memory entries found."

    lines: list[str] = [f"Found {len(results)} matching memory entries:"]
    for entry in results:
        prefix = "USER" if entry.target == "user" else "NOTE"
        lines.append(f"[{prefix} id={entry.id}] {entry.content}")
    return "\n".join(lines)


def memory_list(target: str | None = None) -> str:
    """List all memory entries, optionally filtered by target.

    Args:
        target: If set, only list entries for this target.

    Returns:
        Formatted list of all memory entries.
    """
    store = _get_store()
    entries = store.get_all(target)
    if not entries:
        return f"No {target} memory entries." if target else "No memory entries."

    lines: list[str] = []
    for entry in entries:
        prefix = "USER" if entry.target == "user" else "NOTE"
        lines.append(f"[{prefix} id={entry.id}] {entry.content}")
    return "\n".join(lines)


def _iso_now() -> str:
    """Return current time as ISO 8601 string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
