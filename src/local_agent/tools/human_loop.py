"""Human-in-the-loop tools — clarification prompts and approval workflows.

Allows the agent to pause execution and ask the user for input:
- Multiple choice prompts (up to 4 options + 'Other')
- Open-ended questions
- Approval confirmations for destructive operations
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClarificationRequest:
    """A clarification request from the agent to the user."""

    question: str
    choices: list[str] = field(default_factory=list)
    # If choices is empty, it's an open-ended question.
    # If choices has items, user picks one or types their own ('Other' option).

    @property
    def is_multiple_choice(self) -> bool:
        return len(self.choices) > 0

    def to_json(self) -> str:
        """Serialize to JSON for the agent core to detect."""
        return json.dumps({
            "type": "clarification",
            "question": self.question,
            "choices": self.choices,
        })


@dataclass
class ClarificationResponse:
    """The user's answer to a clarification request."""

    question: str
    answer: str
    was_other: bool = False  # True if user typed their own answer instead of picking a choice


@dataclass
class ApprovalRequest:
    """An approval request for a destructive or sensitive operation."""

    action: str  # Description of what will happen
    details: str = ""  # Additional context
    default_deny: bool = True  # Require explicit 'y' to proceed


class HumanLoopManager:
    """Manages human-in-the-loop interactions.

    Handles:
    - Presenting clarification questions to the user
    - Collecting answers (multiple choice or free text)
    - Presenting approval prompts for destructive actions
    - Logging all interactions for audit trail
    """

    def __init__(self) -> None:
        self._history: list[dict[str, Any]] = []

    def ask_clarification(
        self,
        question: str,
        choices: list[str] | None = None,
    ) -> ClarificationResponse:
        """Present a clarification question to the user and collect their answer.

        Args:
            question: The question to ask.
            choices: Up to 4 choices. If None/empty, open-ended question.

        Returns:
            The user's response.
        """
        if choices is None:
            choices = []

        req = ClarificationRequest(question=question, choices=choices)
        self._history.append({"event": "ask", "request": req.to_json()})

        # In non-interactive mode, this would raise or return a placeholder.
        # In interactive mode, present the prompt to the user.
        # The agent core intercepts this and handles I/O.

        raise BlockingInteraction(req)

    def request_approval(self, action: str, details: str = "") -> bool:
        """Ask the user to approve a destructive or sensitive action.

        Args:
            action: Short description of the action.
            details: Extra context about what will happen.

        Returns:
            True if approved, False if denied.
        """
        req = ApprovalRequest(action=action, details=details)
        self._history.append({"event": "approve", "action": action, "details": details})

        raise BlockingInteraction(req)

    def get_audit_trail(self) -> list[dict[str, Any]]:
        """Return the full history of human interactions."""
        return list(self._history)


class BlockingInteraction(Exception):
    """Raised when a tool needs to block and wait for user input.

    The agent core catches this and routes it to the appropriate
    I/O handler (interactive prompt or non-interactive error).
    """

    def __init__(self, request: Any) -> None:
        self.request = request
        super().__init__(f"Blocking interaction: {type(request).__name__}")


# ---------------------------------------------------------------------------
# Public tool functions registered with the tool registry
# ---------------------------------------------------------------------------


def clarify(
    question: str,
    choices: list[str] | None = None,
) -> str:
    """Ask the user a clarification question and wait for their answer.

    Use this when the task is ambiguous and you need the user to choose
    an approach, provide missing information, or give feedback.

    Supports two modes:
    1. Multiple choice — provide up to 4 choices. User picks one or types
       their own answer via a 5th 'Other' option.
    2. Open-ended — omit choices entirely. User types a free-form response.

    Args:
        question: The question to present to the user.
        choices: Up to 4 answer choices. Omit for open-ended questions.

    Returns:
        The user's answer as a string.
    """
    if choices is None:
        choices = []

    if len(choices) > 4:
        raise ValueError("clarify() accepts at most 4 choices")

    req = ClarificationRequest(question=question, choices=choices)
    raise BlockingInteraction(req)


def confirm(action: str, details: str = "") -> str:
    """Request user approval before proceeding with a sensitive action.

    Use this before destructive operations (deleting files, overwriting
    config, dropping databases, etc.) or actions that are hard to undo.

    Args:
        action: Short description of the action to approve.
        details: Extra context about what will happen.

    Returns:
        "approved" if user says yes, "denied" if user says no.
    """
    req = ApprovalRequest(action=action, details=details)
    raise BlockingInteraction(req)
