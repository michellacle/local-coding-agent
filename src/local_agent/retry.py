"""Adaptive retry — context-aware failure recovery with strategy escalation.

Classifies failures into categories (transient, rate_limit, tool_error, llm_error)
and applies escalating strategies:
  1. Immediate retry (transient failures)
  2. Exponential backoff (rate limits, connection errors)
  3. Context-aware retry (feed error back to LLM so it can adjust approach)
  4. Escalation to human (after max retries exhausted)

Dependencies injected for testability:
    - backoff_fn: callable(delay) -> None (default: time.sleep)
    - escalation_fn: callable(error, attempt) -> str | None (default: None)
"""

from __future__ import annotations

import enum
import time
from typing import Any, Callable

BackoffFn = Callable[[float], None]
EscalationFn = Callable[[str, int], str | None]


class FailureCategory(str, enum.Enum):
    """Categories of failures for strategy selection."""

    TRANSIENT = "transient"        # Connection drops, timeouts — retry immediately
    RATE_LIMIT = "rate_limit"      # 429s, throttling — exponential backoff
    TOOL_ERROR = "tool_error"      # Tool returned error — feed back to LLM
    LLM_ERROR = "llm_error"        # Malformed JSON, bad tool call — feed back to LLM
    PERMANENT = "permanent"        # Not found, auth failure — do not retry


class RetryStrategy:
    """Adaptive retry controller with strategy escalation.

    Args:
        max_retries: Maximum number of retry attempts (default: 3).
        max_backoff: Cap on exponential backoff in seconds (default: 30).
        backoff_fn: Function to sleep for a delay. For tests, inject a no-op.
        escalation_fn: Called when retries exhausted. Return a response string
            to short-circuit, or None to continue failing.
    """

    def __init__(
        self,
        max_retries: int = 3,
        max_backoff: float = 30.0,
        backoff_fn: BackoffFn | None = None,
        escalation_fn: EscalationFn | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self._backoff_fn = backoff_fn if backoff_fn is not None else time.sleep
        self._escalation_fn = escalation_fn
        self.attempt: int = 0
        self.consecutive_failures: int = 0
        self.failure_log: list[dict[str, Any]] = []

    def classify(self, error: Exception | str) -> FailureCategory:
        """Classify an error into a FailureCategory.

        Heuristics:
            - "429", "rate limit", "throttl" -> RATE_LIMIT
            - "Connection", "timeout", "reset", "broken pipe" -> TRANSIENT
            - "tool.*not found", "not found", "no such file" -> PERMANENT
            - "JSONDecodeError", "invalid.*json", "malformed" -> LLM_ERROR
            - Everything else -> TOOL_ERROR

        Args:
            error: Exception or error string to classify.

        Returns:
            A FailureCategory enum value.
        """
        msg: str = str(error).lower()

        if any(kw in msg for kw in ("429", "rate limit", "throttl")):
            return FailureCategory.RATE_LIMIT

        if any(kw in msg for kw in ("connection", "timeout", "reset", "broken pipe")):
            return FailureCategory.TRANSIENT

        if any(kw in msg for kw in ("not found", "no such file", "permission denied")):
            return FailureCategory.PERMANENT

        if any(kw in msg for kw in ("jsondecodeerror", "invalid.*json", "malformed")):
            return FailureCategory.LLM_ERROR

        return FailureCategory.TOOL_ERROR

    def should_retry(self, error: Exception | str) -> bool:
        """Decide whether to retry based on error classification and attempt count.

        Args:
            error: The error that occurred.

        Returns:
            True if another attempt should be made.
        """
        category = self.classify(error)

        if category == FailureCategory.PERMANENT:
            return False

        if self.attempt >= self.max_retries:
            return False

        return True

    def backoff_delay(self) -> float:
        """Calculate exponential backoff delay for the current attempt.

        Returns:
            Delay in seconds, capped at max_backoff.
        """
        raw: float = min(2 ** self.attempt, self.max_backoff)
        return raw

    def wait(self) -> None:
        """Sleep for the calculated backoff delay."""
        delay = self.backoff_delay()
        self._backoff_fn(delay)

    def record_failure(self, error: Exception | str) -> None:
        """Record a failure for tracking and context-aware retry.

        Args:
            error: The error that occurred.
        """
        self.consecutive_failures += 1
        self.failure_log.append({
            "attempt": self.attempt,
            "error": str(error),
            "category": self.classify(error).value,
        })

    def record_success(self) -> None:
        """Record a successful attempt, resetting failure counters."""
        self.consecutive_failures = 0
        self.failure_log.clear()

    def escalate(self, error: Exception | str) -> str | None:
        """Escalate to the escalation handler after max retries.

        Args:
            error: The final error that triggered escalation.

        Returns:
            Response string from escalation handler, or None if no handler.
        """
        if self._escalation_fn is not None:
            return self._escalation_fn(str(error), self.attempt)
        return None

    def context_summary(self) -> str:
        """Build a context summary of failures for feeding back to the LLM.

        Returns:
            A formatted string summarizing recent failures.
        """
        if not self.failure_log:
            return ""

        lines: list[str] = [
            f"Previous attempt(s) failed ({len(self.failure_log)} attempt(s)):",
        ]
        for entry in self.failure_log[-3:]:
            lines.append(
                f"  - Attempt {entry['attempt']} [{entry['category']}]: {entry['error']}"
            )
        lines.append(
            "Please adjust your approach based on these errors and try again."
        )
        return "\n".join(lines)

    def next_attempt(self) -> None:
        """Increment the attempt counter."""
        self.attempt += 1

    def reset(self) -> None:
        """Reset all counters and logs for a new turn."""
        self.attempt = 0
        self.consecutive_failures = 0
        self.failure_log.clear()
