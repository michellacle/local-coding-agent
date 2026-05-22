"""Explainability & Audit Trail — track agent actions with explanations and timestamps.

Provides:
- Action explanations before tool execution
- Session audit log (all actions, timestamps, results)
- Confidence scores for recommendations
- Session summary at end
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditEntry:
    """A single entry in the audit log.

    Attributes:
        timestamp: Unix timestamp when the action occurred.
        action: Type of action (tool_call, llm_call, user_input, error, etc.).
        description: Human-readable explanation of what happened.
        details: Structured details about the action.
        duration_ms: How long the action took (milliseconds).
        status: Outcome status (success, error, partial).
        confidence: Confidence score [0.0, 1.0] if applicable.
    """

    timestamp: float
    action: str
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    status: str = "success"
    confidence: float | None = None


class AuditLog:
    """Session audit log that records all agent actions.

    Entries are stored in-memory during the session and can be
    exported as JSON or a human-readable summary.
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._session_start = time.time()

    def log(
        self,
        action: str,
        description: str,
        details: dict[str, Any] | None = None,
        status: str = "success",
        confidence: float | None = None,
    ) -> AuditEntry:
        """Add an entry to the audit log.

        Args:
            action: Type of action (tool_call, llm_call, etc.).
            description: Human-readable explanation.
            details: Optional structured details.
            status: Outcome status.
            confidence: Optional confidence score.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            timestamp=time.time(),
            action=action,
            description=description,
            details=details or {},
            status=status,
            confidence=confidence,
        )
        self._entries.append(entry)
        return entry

    def log_action(
        self,
        action: str,
        description: str,
        details: dict[str, Any] | None = None,
    ) -> ActionTimer:
        """Start timing an action. Use as a context manager.

        Args:
            action: Type of action.
            description: Human-readable explanation.
            details: Optional structured details.

        Returns:
            An ActionTimer context manager.
        """
        return ActionTimer(self, action, description, details or {})

    def entries(self) -> list[AuditEntry]:
        """Return all audit entries."""
        return list(self._entries)

    def recent(self, count: int = 10) -> list[AuditEntry]:
        """Return the most recent N entries."""
        return list(self._entries[-count:])

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()

    def export_json(self) -> str:
        """Export the audit log as formatted JSON."""
        data = []
        for entry in self._entries:
            data.append({
                "timestamp": entry.timestamp,
                "action": entry.action,
                "description": entry.description,
                "details": entry.details,
                "duration_ms": entry.duration_ms,
                "status": entry.status,
                "confidence": entry.confidence,
            })
        return json.dumps(data, indent=2)

    def summary(self) -> str:
        """Generate a human-readable session summary.

        Returns:
            Formatted text summary of the session.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("SESSION AUDIT SUMMARY")
        lines.append("=" * 60)

        session_duration = time.time() - self._session_start
        lines.append(f"Session duration: {session_duration:.1f}s")
        lines.append(f"Total actions: {len(self._entries)}")
        lines.append("")

        # Count by action type
        action_counts: dict[str, int] = {}
        error_count = 0
        for entry in self._entries:
            action_counts[entry.action] = action_counts.get(entry.action, 0) + 1
            if entry.status == "error":
                error_count += 1

        lines.append("Actions by type:")
        for action, count in sorted(action_counts.items()):
            lines.append(f"  {action}: {count}")

        if error_count > 0:
            lines.append(f"\nErrors: {error_count}")

        # Average confidence
        confidences = [e.confidence for e in self._entries if e.confidence is not None]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            lines.append(f"Average confidence: {avg_conf:.2f}")

        lines.append("")

        # Last N entries
        lines.append("Recent actions:")
        for entry in self._entries[-10:]:
            time_str = time.strftime(
                "%H:%M:%S", time.localtime(entry.timestamp)
            )
            status_icon = "✓" if entry.status == "success" else "✗"
            dur = f" ({entry.duration_ms:.0f}ms)" if entry.duration_ms > 0 else ""
            lines.append(
                f"  [{time_str}] {status_icon} {entry.action}: {entry.description}{dur}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class ActionTimer:
    """Context manager that logs an action with timing.

    Usage:
        with audit.log_action("tool_call", "read_file: config.yaml") as timer:
            # do work
            pass
        # timer.entry has duration_ms set
    """

    def __init__(
        self,
        audit_log: AuditLog,
        action: str,
        description: str,
        details: dict[str, Any],
    ) -> None:
        self._audit = audit_log
        self._action = action
        self._description = description
        self._details = details
        self._start = time.time()
        self.entry: AuditEntry | None = None

    def __enter__(self) -> ActionTimer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration_ms = (time.time() - self._start) * 1000
        status = "error" if exc_type is not None else "success"
        self._details["exception"] = str(exc_val) if exc_val else None

        self.entry = self._audit.log(
            action=self._action,
            description=self._description,
            details=self._details,
            status=status,
        )
        self.entry.duration_ms = duration_ms


class ExplanationGenerator:
    """Generate human-readable explanations for agent actions."""

    @staticmethod
    def explain_tool_call(tool_name: str, args: dict[str, Any]) -> str:
        """Generate an explanation for a tool call.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments being passed.

        Returns:
            Human-readable explanation string.
        """
        explanations: dict[str, str] = {
            "read_file": f"Reading file '{args.get('path', 'unknown')}'",
            "write_file": f"Writing file '{args.get('path', 'unknown')}'",
            "patch": f"Patching file '{args.get('path', 'unknown')}'",
            "execute_command": f"Running command: {args.get('command', 'unknown')}",
            "process_manage": f"Managing process: {args.get('action', 'unknown')}",
            "search_files": f"Searching for '{args.get('pattern', 'unknown')}'",
            "git_init": f"Initializing git repo at '{args.get('path', 'unknown')}'",
            "git_add": f"Staging files in '{args.get('path', 'unknown')}'",
            "git_commit": f"Committing with message: {args.get('message', 'unknown')}",
            "git_branch": f"{'Deleting' if args.get('delete') else 'Creating'} branch '{args.get('branch_name', 'unknown')}'",
            "git_merge": f"Merging branch '{args.get('branch', 'unknown')}'",
            "git_push": f"Pushing to remote '{args.get('remote', 'origin')}'",
            "git_log": "Viewing commit history",
            "clarify": "Asking user for clarification",
            "confirm": "Requesting user confirmation",
            "retrieve_context": f"Searching docs for: {args.get('query', 'unknown')}",
            "delegate_task": f"Delegating task: {args.get('goal', 'unknown')[:50]}",
        }

        base = explanations.get(tool_name, f"Calling tool '{tool_name}'")

        # Add confidence note for potentially risky operations
        risky_ops = {"execute_command", "git_push", "git_merge", "patch", "write_file"}
        if tool_name in risky_ops:
            bg = args.get("background")
            if tool_name == "execute_command" and bg:
                base += " (background)"

        return base

    @staticmethod
    def explain_error(tool_name: str, error: str) -> str:
        """Generate an explanation for a tool error.

        Args:
            tool_name: Name of the tool that failed.
            error: Error message.

        Returns:
            Human-readable error explanation.
        """
        return f"Tool '{tool_name}' failed: {error[:200]}"

    @staticmethod
    def explain_llm_response(
        response: str, has_tool_call: bool, tokens_used: int | None = None
    ) -> str:
        """Generate an explanation for an LLM response.

        Args:
            response: The LLM's response text.
            has_tool_call: Whether the response contained a tool call.
            tokens_used: Number of tokens used (if known).

        Returns:
            Human-readable explanation.
        """
        if has_tool_call:
            desc = "LLM requested a tool call"
        else:
            preview = response[:80].replace("\n", " ")
            desc = f"LLM responded: '{preview}...'" if len(response) > 80 else f"LLM responded: '{preview}'"

        if tokens_used:
            desc += f" ({tokens_used} tokens)"

        return desc
