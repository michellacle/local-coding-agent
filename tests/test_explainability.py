"""Tests for explainability — AuditLog, ActionTimer, ExplanationGenerator."""

import time

import pytest

from local_agent.explainability import (
    AuditLog,
    AuditEntry,
    ActionTimer,
    ExplanationGenerator,
)


class TestAuditLog:
    """Test AuditLog operations."""

    def test_log_creates_entry(self):
        log = AuditLog()
        entry = log.log("tool_call", "read_file: config.yaml")
        assert entry.action == "tool_call"
        assert entry.description == "read_file: config.yaml"
        assert entry.status == "success"
        assert entry.timestamp > 0

    def test_log_with_details(self):
        log = AuditLog()
        entry = log.log(
            "tool_call",
            "read_file: config.yaml",
            details={"path": "config.yaml", "lines": 50},
        )
        assert entry.details["path"] == "config.yaml"

    def test_log_with_confidence(self):
        log = AuditLog()
        entry = log.log(
            "recommendation",
            "Use pytest for testing",
            confidence=0.85,
        )
        assert entry.confidence == pytest.approx(0.85)

    def test_log_error_status(self):
        log = AuditLog()
        entry = log.log("tool_call", "run tests", status="error")
        assert entry.status == "error"

    def test_entries_returns_all(self):
        log = AuditLog()
        log.log("a", "first")
        log.log("b", "second")
        log.log("c", "third")
        assert len(log.entries()) == 3

    def test_recent_returns_last_n(self):
        log = AuditLog()
        for i in range(20):
            log.log("action", f"entry {i}")
        recent = log.recent(count=5)
        assert len(recent) == 5
        assert "entry 19" in recent[-1].description

    def test_clear(self):
        log = AuditLog()
        log.log("a", "first")
        log.clear()
        assert len(log.entries()) == 0

    def test_export_json(self):
        log = AuditLog()
        log.log("tool_call", "test", details={"key": "value"}, confidence=0.9)

        json_str = log.export_json()
        assert "tool_call" in json_str
        assert "test" in json_str
        assert "key" in json_str

        # Verify it's valid JSON
        import json
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_summary_contains_counts(self):
        log = AuditLog()
        log.log("tool_call", "read")
        log.log("tool_call", "write")
        log.log("llm_call", "response")

        summary = log.summary()
        assert "tool_call: 2" in summary
        assert "llm_call: 1" in summary
        assert "Total actions: 3" in summary

    def test_summary_shows_session_duration(self):
        log = AuditLog()
        log.log("tool_call", "read")
        summary = log.summary()
        assert "Session duration:" in summary

    def test_summary_shows_errors(self):
        log = AuditLog()
        log.log("tool_call", "read", status="success")
        log.log("tool_call", "write", status="error")
        summary = log.summary()
        assert "Errors: 1" in summary

    def test_summary_shows_confidence(self):
        log = AuditLog()
        log.log("recommendation", "use pytest", confidence=0.8)
        log.log("recommendation", "use mypy", confidence=0.9)
        summary = log.summary()
        assert "Average confidence:" in summary

    def test_summary_recent_actions(self):
        log = AuditLog()
        log.log("tool_call", "read_file")
        summary = log.summary()
        assert "read_file" in summary


class TestActionTimer:
    """Test ActionTimer context manager."""

    def test_logs_action_with_timing(self):
        log = AuditLog()
        with log.log_action("tool_call", "read config.yaml", {"path": "config.yaml"}) as timer:
            time.sleep(0.05)  # 50ms

        entries = log.entries()
        assert len(entries) == 1
        assert entries[0].action == "tool_call"
        assert entries[0].duration_ms >= 40  # at least ~40ms

    def test_timer_success_status(self):
        log = AuditLog()
        with log.log_action("tool_call", "read"):
            pass

        assert log.entries()[0].status == "success"

    def test_timer_error_status(self):
        log = AuditLog()
        try:
            with log.log_action("tool_call", "read"):
                raise ValueError("file not found")
        except ValueError:
            pass

        entry = log.entries()[0]
        assert entry.status == "error"
        assert "file not found" in str(entry.details.get("exception"))

    def test_timer_reference(self):
        log = AuditLog()
        with log.log_action("tool_call", "read") as timer:
            pass

        assert timer.entry is not None
        assert timer.entry.action == "tool_call"


class TestExplanationGenerator:
    """Test ExplanationGenerator."""

    gen = ExplanationGenerator()

    def test_explain_read_file(self):
        expl = self.gen.explain_tool_call("read_file", {"path": "config.yaml"})
        assert "config.yaml" in expl

    def test_explain_write_file(self):
        expl = self.gen.explain_tool_call("write_file", {"path": "new.py"})
        assert "new.py" in expl

    def test_explain_execute_command(self):
        expl = self.gen.explain_tool_call(
            "execute_command", {"command": "npm install"}
        )
        assert "npm install" in expl

    def test_explain_background_command(self):
        expl = self.gen.explain_tool_call(
            "execute_command", {"command": "npm test", "background": True}
        )
        assert "background" in expl

    def test_explain_git_commit(self):
        expl = self.gen.explain_tool_call(
            "git_commit", {"message": "fix: typo"}
        )
        assert "fix: typo" in expl

    def test_explain_unknown_tool(self):
        expl = self.gen.explain_tool_call("custom_tool", {"a": 1})
        assert "custom_tool" in expl

    def test_explain_error(self):
        expl = self.gen.explain_error("read_file", "No such file")
        assert "read_file" in expl
        assert "No such file" in expl

    def test_explain_error_truncation(self):
        long_error = "x" * 500
        expl = self.gen.explain_error("tool", long_error)
        # Should be truncated to 200 chars
        assert "x" * 201 not in expl

    def test_explain_llm_with_tool_call(self):
        expl = self.gen.explain_llm_response(
            '{"tool": "read_file", "args": {}}',
            has_tool_call=True,
        )
        assert "tool call" in expl

    def test_explain_llm_plain_text(self):
        expl = self.gen.explain_llm_response(
            "The answer is 42.",
            has_tool_call=False,
        )
        assert "42" in expl

    def test_explain_llm_with_tokens(self):
        expl = self.gen.explain_llm_response(
            "Hello",
            has_tool_call=False,
            tokens_used=150,
        )
        assert "150" in expl

    def test_explain_llm_truncates_long_response(self):
        expl = self.gen.explain_llm_response(
            "x" * 500,
            has_tool_call=False,
        )
        assert "..." in expl

    def test_explain_search_files(self):
        expl = self.gen.explain_tool_call(
            "search_files", {"pattern": "def main"}
        )
        assert "def main" in expl

    def test_explain_git_merge(self):
        expl = self.gen.explain_tool_call(
            "git_merge", {"branch": "feature-1"}
        )
        assert "feature-1" in expl

    def test_explain_delegate_task(self):
        expl = self.gen.explain_tool_call(
            "delegate_task", {"goal": "Write tests for the auth module"}
        )
        assert "Write tests" in expl

    def test_explain_git_branch_delete(self):
        expl = self.gen.explain_tool_call(
            "git_branch", {"branch_name": "old", "delete": True}
        )
        assert "Deleting" in expl

    def test_explain_git_branch_create(self):
        expl = self.gen.explain_tool_call(
            "git_branch", {"branch_name": "new", "delete": False}
        )
        assert "Creating" in expl
