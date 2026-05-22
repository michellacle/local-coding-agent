"""Diff Review — preview and manage file changes before applying them.

Provides a staged change management system: the agent proposes changes as
"patches", the user can review them, accept/reject individual changes,
and only approved changes are written to disk.

This mirrors Claude Code's "apply" workflow where changes are reviewed
before being committed to files.
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ChangeStatus(Enum):
    """Status of a proposed change."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ProposedChange:
    """A proposed file change awaiting review."""

    file_path: str
    old_content: str
    new_content: str
    status: ChangeStatus = ChangeStatus.PENDING
    reason: str = ""
    proposed_at: float = field(default_factory=time.time)

    @property
    def unified_diff(self) -> str:
        """Generate a unified diff between old and new content."""
        old_lines = self.old_content.splitlines(keepends=True)
        new_lines = self.new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{self.file_path}",
            tofile=f"b/{self.file_path}",
            n=3,
        )
        return "".join(diff)

    @property
    def num_added(self) -> int:
        """Number of added lines."""
        return sum(1 for line in self.unified_diff.splitlines() if line.startswith("+") and not line.startswith("+++"))

    @property
    def num_removed(self) -> int:
        """Number of removed lines."""
        return sum(1 for line in self.unified_diff.splitlines() if line.startswith("-") and not line.startswith("---"))

    @property
    def has_changes(self) -> bool:
        """Whether there are actual differences."""
        return self.old_content != self.new_content

    def summary(self) -> str:
        """Generate a human-readable summary."""
        status_map = {
            ChangeStatus.PENDING: "[dim]PENDING[/dim]",
            ChangeStatus.APPROVED: "[bold green]APPROVED[/bold green]",
            ChangeStatus.REJECTED: "[bold red]REJECTED[/bold red]",
        }
        status_str = status_map.get(self.status, str(self.status))
        lines = [
            f"  [bold]{self.file_path}[/bold] {status_str}",
            f"    +{self.num_added} -{self.num_removed} lines",
        ]
        if self.reason:
            lines.append(f"    [dim]{self.reason}[/dim]")
        return "\n".join(lines)

    def apply(self) -> None:
        """Apply this change by writing new content to the file."""
        if self.status != ChangeStatus.APPROVED:
            raise RuntimeError(f"Cannot apply {self.status.value} change: {self.file_path}")
        p = Path(self.file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.new_content, encoding="utf-8")

    def revert(self) -> str:
        """Revert this change (restore old content). Returns the restored content."""
        Path(self.file_path).write_text(self.old_content, encoding="utf-8")
        self.status = ChangeStatus.REJECTED
        return self.old_content


@dataclass
class DiffReview:
    """Manages a collection of proposed changes for review.

    Usage:
        review = DiffReview()
        review.propose("/path/to/file.py", old_content, new_content, "Fix bug #42")
        review.list_changes()
        review.approve("/path/to/file.py")
        review.apply_all()
    """

    changes: list[ProposedChange] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def propose(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        reason: str = "",
    ) -> ProposedChange:
        """Propose a change to a file.

        If a change for the same file already exists, it is replaced.

        Args:
            file_path: Path to the file.
            old_content: Current file content.
            new_content: Proposed new content.
            reason: Reason for the change.

        Returns:
            The ProposedChange object.
        """
        # Replace existing change for same file
        self.changes = [c for c in self.changes if c.file_path != file_path]

        change = ProposedChange(
            file_path=file_path,
            old_content=old_content,
            new_content=new_content,
            reason=reason,
        )
        self.changes.append(change)
        return change

    def list_changes(self) -> list[ProposedChange]:
        """List all pending changes."""
        return [c for c in self.changes if c.status == ChangeStatus.PENDING]

    def list_all(self) -> list[ProposedChange]:
        """List all changes regardless of status."""
        return self.changes

    def approve(self, file_path: str) -> ProposedChange | None:
        """Approve a change by file path.

        Args:
            file_path: Path to the file.

        Returns:
            The approved change, or None if not found.
        """
        for change in self.changes:
            if change.file_path == file_path:
                change.status = ChangeStatus.APPROVED
                return change
        return None

    def reject(self, file_path: str) -> ProposedChange | None:
        """Reject a change by file path.

        Args:
            file_path: Path to the file.

        Returns:
            The rejected change, or None if not found.
        """
        for change in self.changes:
            if change.file_path == file_path:
                change.status = ChangeStatus.REJECTED
                return change
        return None

    def approve_all(self) -> int:
        """Approve all pending changes. Returns the number approved."""
        count = 0
        for change in self.changes:
            if change.status == ChangeStatus.PENDING:
                change.status = ChangeStatus.APPROVED
                count += 1
        return count

    def reject_all(self) -> int:
        """Reject all pending changes. Returns the number rejected."""
        count = 0
        for change in self.changes:
            if change.status == ChangeStatus.PENDING:
                change.status = ChangeStatus.REJECTED
                count += 1
        return count

    def apply_all(self) -> list[str]:
        """Apply all approved changes. Returns list of applied file paths."""
        applied: list[str] = []
        for change in self.changes:
            if change.status == ChangeStatus.APPROVED and change.has_changes:
                change.apply()
                applied.append(change.file_path)
        return applied

    def pending_count(self) -> int:
        """Number of pending changes."""
        return sum(1 for c in self.changes if c.status == ChangeStatus.PENDING)

    def approved_count(self) -> int:
        """Number of approved changes."""
        return sum(1 for c in self.changes if c.status == ChangeStatus.APPROVED)

    def format_review(self) -> str:
        """Format all changes for review display."""
        pending = self.list_changes()
        if not pending:
            return "[dim]No pending changes.[/dim]"

        lines = [
            f"[bold cyan]Change Review ({len(pending)} pending)[/bold cyan]",
            "",
        ]

        for i, change in enumerate(pending, 1):
            lines.append(f"[bold][{i}][/bold] {change.summary()}")
            lines.append("")
            # Show diff (truncated)
            diff = change.unified_diff
            diff_lines = diff.splitlines()
            if len(diff_lines) > 30:
                lines.append("\n".join(diff_lines[:30]))
                lines.append(f"  [dim]... ({len(diff_lines) - 30} more lines)[/dim]")
            else:
                lines.append(diff)
            lines.append("")

        lines.append("[dim]Use /apply N to approve, /reject N to reject, /apply all to approve all[/dim]")

        return "\n".join(lines)

    def clear_completed(self) -> int:
        """Remove approved/rejected changes. Returns count removed."""
        before = len(self.changes)
        self.changes = [c for c in self.changes if c.status == ChangeStatus.PENDING]
        return before - len(self.changes)

    def find_change(self, file_path: str) -> ProposedChange | None:
        """Find a change by file path."""
        for change in self.changes:
            if change.file_path == file_path:
                return change
        return None
