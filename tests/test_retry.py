"""Tests for adaptive retry — context-aware failure recovery."""

from unittest.mock import MagicMock

import pytest

from local_agent.retry import (
    FailureCategory,
    RetryStrategy,
)


class TestFailureClassification:
    """Test error classification heuristics."""

    def setup_method(self) -> None:
        self.strategy = RetryStrategy()

    def test_rate_limit_429(self):
        assert self.strategy.classify("HTTP 429 Too Many Requests") == FailureCategory.RATE_LIMIT

    def test_rate_limit_text(self):
        assert self.strategy.classify("rate limit exceeded") == FailureCategory.RATE_LIMIT

    def test_rate_limit_throttle(self):
        assert self.strategy.classify("throttled by upstream") == FailureCategory.RATE_LIMIT

    def test_transient_connection(self):
        assert self.strategy.classify("ConnectionError: failed to connect") == FailureCategory.TRANSIENT

    def test_transient_timeout(self):
        assert self.strategy.classify("Timeout: request timed out") == FailureCategory.TRANSIENT

    def test_transient_reset(self):
        assert self.strategy.classify("Connection reset by peer") == FailureCategory.TRANSIENT

    def test_transient_broken_pipe(self):
        assert self.strategy.classify("Broken pipe") == FailureCategory.TRANSIENT

    def test_permanent_not_found(self):
        assert self.strategy.classify("404 Not Found") == FailureCategory.PERMANENT

    def test_permanent_no_such_file(self):
        assert self.strategy.classify("No such file or directory") == FailureCategory.PERMANENT

    def test_permanent_permission_denied(self):
        assert self.strategy.classify("Permission denied") == FailureCategory.PERMANENT

    def test_llm_json_error(self):
        assert self.strategy.classify("json.JSONDecodeError: invalid json") == FailureCategory.LLM_ERROR

    def test_llm_malformed(self):
        assert self.strategy.classify("Malformed tool call response") == FailureCategory.LLM_ERROR

    def test_tool_error_default(self):
        assert self.strategy.classify("File write failed: disk full") == FailureCategory.TOOL_ERROR

    def test_tool_error_generic(self):
        assert self.strategy.classify("Unknown error occurred") == FailureCategory.TOOL_ERROR

    def test_classify_from_exception(self):
        assert self.strategy.classify(ValueError("rate limit exceeded")) == FailureCategory.RATE_LIMIT


class TestShouldRetry:
    """Test retry decision logic."""

    def test_retries_transient(self):
        strategy = RetryStrategy(max_retries=3)
        assert strategy.should_retry("ConnectionError: timeout") is True

    def test_retries_rate_limit(self):
        strategy = RetryStrategy(max_retries=3)
        assert strategy.should_retry("HTTP 429 rate limited") is True

    def test_retries_tool_error(self):
        strategy = RetryStrategy(max_retries=3)
        assert strategy.should_retry("Tool failed: disk full") is True

    def test_retries_llm_error(self):
        strategy = RetryStrategy(max_retries=3)
        assert strategy.should_retry("json.JSONDecodeError") is True

    def test_no_retry_permanent(self):
        strategy = RetryStrategy(max_retries=3)
        assert strategy.should_retry("404 Not Found") is False

    def test_no_retry_max_exceeded(self):
        strategy = RetryStrategy(max_retries=2)
        strategy.attempt = 2
        assert strategy.should_retry("ConnectionError") is False

    def test_no_retry_at_boundary(self):
        strategy = RetryStrategy(max_retries=1)
        strategy.attempt = 1
        assert strategy.should_retry("ConnectionError") is False


class TestBackoff:
    """Test exponential backoff calculation."""

    def test_backoff_attempt_0(self):
        strategy = RetryStrategy()
        assert strategy.backoff_delay() == 1.0  # 2^0

    def test_backoff_attempt_1(self):
        strategy = RetryStrategy()
        strategy.attempt = 1
        assert strategy.backoff_delay() == 2.0  # 2^1

    def test_backoff_attempt_2(self):
        strategy = RetryStrategy()
        strategy.attempt = 2
        assert strategy.backoff_delay() == 4.0  # 2^2

    def test_backoff_attempt_5(self):
        strategy = RetryStrategy()
        strategy.attempt = 5
        assert strategy.backoff_delay() == 30.0  # min(2^5=32, 30) -> capped

    def test_backoff_capped(self):
        strategy = RetryStrategy(max_backoff=10.0)
        strategy.attempt = 5
        assert strategy.backoff_delay() == 10.0

    def test_wait_calls_backoff_fn(self):
        mock_sleep = MagicMock()
        strategy = RetryStrategy(backoff_fn=mock_sleep)
        strategy.attempt = 0
        strategy.wait()
        mock_sleep.assert_called_once_with(1.0)

    def test_wait_uses_custom_backoff(self):
        mock_sleep = MagicMock()
        strategy = RetryStrategy(max_backoff=5.0, backoff_fn=mock_sleep)
        strategy.attempt = 10
        strategy.wait()
        mock_sleep.assert_called_once_with(5.0)


class TestFailureTracking:
    """Test failure recording and context summary."""

    def setup_method(self) -> None:
        self.strategy = RetryStrategy(max_retries=5)

    def test_record_failure_increments_counter(self):
        assert self.strategy.consecutive_failures == 0
        self.strategy.record_failure("error 1")
        assert self.strategy.consecutive_failures == 1
        self.strategy.record_failure("error 2")
        assert self.strategy.consecutive_failures == 2

    def test_record_success_resets(self):
        self.strategy.record_failure("error 1")
        self.strategy.record_failure("error 2")
        self.strategy.record_success()
        assert self.strategy.consecutive_failures == 0
        assert len(self.strategy.failure_log) == 0

    def test_failure_log_entries(self):
        self.strategy.record_failure("ConnectionError")
        self.strategy.record_failure("timeout")
        assert len(self.strategy.failure_log) == 2
        assert self.strategy.failure_log[0]["error"] == "ConnectionError"
        assert self.strategy.failure_log[0]["category"] == "transient"

    def test_context_summary_empty(self):
        assert self.strategy.context_summary() == ""

    def test_context_summary_with_failures(self):
        self.strategy.record_failure("ConnectionError")
        self.strategy.record_failure("429 rate limit")
        summary = self.strategy.context_summary()
        assert "Previous attempt(s) failed" in summary
        assert "2 attempt(s)" in summary
        assert "ConnectionError" in summary
        assert "rate limit" in summary
        assert "adjust your approach" in summary

    def test_context_summary_truncates_old(self):
        for i in range(5):
            self.strategy.record_failure(f"error {i}")
        summary = self.strategy.context_summary()
        # Only last 3 should appear
        assert "error 0" not in summary
        assert "error 1" not in summary
        assert "error 3" in summary
        assert "error 4" in summary


class TestEscalation:
    """Test escalation to human handler."""

    def test_escalate_no_handler(self):
        strategy = RetryStrategy()
        assert strategy.escalate("final error") is None

    def test_escalate_with_handler(self):
        handler = MagicMock(return_value="User said: try a different approach")
        strategy = RetryStrategy(escalation_fn=handler)
        result = strategy.escalate("final error")
        assert result == "User said: try a different approach"
        handler.assert_called_once_with("final error", 0)

    def test_escalate_handler_gets_attempt_count(self):
        handler = MagicMock(return_value="denied")
        strategy = RetryStrategy(escalation_fn=handler, max_retries=3)
        strategy.attempt = 4
        strategy.escalate("error")
        handler.assert_called_once()
        assert handler.call_args[0][1] == 4


class TestAttemptManagement:
    """Test attempt counter and reset."""

    def test_next_attempt_increments(self):
        strategy = RetryStrategy()
        assert strategy.attempt == 0
        strategy.next_attempt()
        assert strategy.attempt == 1
        strategy.next_attempt()
        assert strategy.attempt == 2

    def test_reset_clears_everything(self):
        strategy = RetryStrategy()
        strategy.attempt = 5
        strategy.record_failure("error")
        strategy.reset()
        assert strategy.attempt == 0
        assert strategy.consecutive_failures == 0
        assert len(strategy.failure_log) == 0


class TestIntegration:
    """Test the full retry flow end-to-end."""

    def test_successful_retry_flow(self):
        """Simulate: fail twice, succeed on third attempt."""
        mock_sleep = MagicMock()
        strategy = RetryStrategy(max_retries=5, backoff_fn=mock_sleep)

        # First attempt fails
        strategy.next_attempt()
        strategy.record_failure("ConnectionError: timeout")
        assert strategy.should_retry("ConnectionError") is True
        strategy.wait()

        # Second attempt fails
        strategy.next_attempt()
        strategy.record_failure("ConnectionError: timeout")
        assert strategy.should_retry("ConnectionError") is True
        strategy.wait()

        # Third attempt succeeds
        strategy.next_attempt()
        strategy.record_success()
        assert strategy.consecutive_failures == 0

    def test_exhaust_retries_then_escalate(self):
        """Simulate exhausting retries and escalating."""
        handler = MagicMock(return_value="human response")
        mock_sleep = MagicMock()
        strategy = RetryStrategy(max_retries=3, backoff_fn=mock_sleep, escalation_fn=handler)

        # Attempt 0-2 fail
        for i in range(3):
            strategy.next_attempt()
            strategy.record_failure("ConnectionError")

        # Attempt 3 should not retry
        assert strategy.should_retry("ConnectionError") is False

        # Escalate
        result = strategy.escalate("ConnectionError")
        assert result == "human response"

    def test_permanent_no_retry(self):
        strategy = RetryStrategy(max_retries=5)
        assert strategy.should_retry("Permission denied") is False
