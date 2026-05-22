"""Session Persistence — save and restore conversation history across crashes.

Automatically checkpoints the agent's message history to disk so that
if the process is killed or crashes, the next launch can restore the
conversation state.

Supports:
- Auto-save after each turn
- Restore on startup
- Session metadata (start time, turn count, model used)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SESSION_DIR = Path.home() / ".local-coding-agent" / "sessions"


@dataclass
class SessionMeta:
    """Metadata for a persisted session."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turn_count: int = 0
    model: str = ""
    work_dir: str = ""
    restored: bool = False


@dataclass
class SessionState:
    """Complete session state for persistence."""

    meta: SessionMeta = field(default_factory=SessionMeta)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def save(self, path: str | None = None) -> str:
        """Save session to disk.

        Args:
            path: Optional path. Defaults to auto-generated session file.

        Returns:
            Path where session was saved.
        """
        self.meta.updated_at = time.time()

        if path is None:
            SESSION_DIR.mkdir(parents=True, exist_ok=True)
            path = str(SESSION_DIR / f"{self.meta.session_id}.json")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "meta": {
                "session_id": self.meta.session_id,
                "created_at": self.meta.created_at,
                "updated_at": self.meta.updated_at,
                "turn_count": self.meta.turn_count,
                "model": self.meta.model,
                "work_dir": self.meta.work_dir,
                "restored": self.meta.restored,
            },
            "messages": self.messages,
        }
        p.write_text(json.dumps(data, indent=2))
        return str(p)

    @classmethod
    def load(cls, path: str) -> SessionState:
        """Load session from disk.

        Args:
            path: Path to session JSON file.

        Returns:
            Restored SessionState.
        """
        data = json.loads(Path(path).read_text())
        meta_data = data.get("meta", {})
        meta = SessionMeta(
            session_id=meta_data.get("session_id", uuid.uuid4().hex[:12]),
            created_at=meta_data.get("created_at", time.time()),
            updated_at=meta_data.get("updated_at", time.time()),
            turn_count=meta_data.get("turn_count", 0),
            model=meta_data.get("model", ""),
            work_dir=meta_data.get("work_dir", ""),
            restored=True,
        )
        return cls(
            meta=meta,
            messages=data.get("messages", []),
        )

    @classmethod
    def latest(cls) -> SessionState | None:
        """Load the most recent session.

        Returns:
            Most recent SessionState or None if no sessions exist.
        """
        if not SESSION_DIR.exists():
            return None

        sessions = sorted(
            SESSION_DIR.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not sessions:
            return None

        try:
            return cls.load(str(sessions[0]))
        except (json.JSONDecodeError, OSError):
            return None

    @classmethod
    def list_sessions(cls) -> list[dict[str, Any]]:
        """List all persisted sessions.

        Returns:
            List of session summaries.
        """
        if not SESSION_DIR.exists():
            return []

        sessions = []
        for f in sorted(
            SESSION_DIR.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        ):
            try:
                data = json.loads(f.read_text())
                meta = data.get("meta", {})
                msg_count = len(data.get("messages", []))
                sessions.append({
                    "path": str(f),
                    "session_id": meta.get("session_id", ""),
                    "created_at": meta.get("created_at", 0),
                    "updated_at": meta.get("updated_at", 0),
                    "turn_count": meta.get("turn_count", 0),
                    "model": meta.get("model", ""),
                    "message_count": msg_count,
                })
            except (json.JSONDecodeError, OSError):
                continue

        return sessions


class SessionManager:
    """Manages session persistence for an agent.

    Usage:
        manager = SessionManager()
        # Try to restore previous session
        if manager.try_restore():
            print(f"Restored session with {len(manager.state.messages)} messages")

        # Use the agent...
        agent.run_turn("hello")

        # Save after each turn
        manager.state.messages = agent._history
        manager.state.meta.turn_count += 1
        manager.save()
    """

    def __init__(self) -> None:
        self.state: SessionState | None = None

    def try_restore(self) -> bool:
        """Try to restore the latest session.

        Returns:
            True if a session was restored.
        """
        state = SessionState.latest()
        if state and state.messages:
            self.state = state
            return True
        return False

    def create_new(
        self,
        model: str = "",
        work_dir: str = "",
    ) -> SessionState:
        """Create a new session state.

        Args:
            model: Model name being used.
            work_dir: Working directory.

        Returns:
            New SessionState.
        """
        self.state = SessionState(
            meta=SessionMeta(model=model, work_dir=work_dir)
        )
        return self.state

    def save(self) -> str:
        """Save the current session state.

        Returns:
            Path where session was saved.
        """
        if self.state is None:
            raise RuntimeError("No active session. Call create_new() or try_restore() first.")
        return self.state.save()

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all persisted sessions."""
        return SessionState.list_sessions()
