"""Tests for explainability — AuditLogger, DecisionRecord, ChainOfThought, SelfAssessment."""

import json
import time
from pathlib import Path

import pytest

from local_agent.explainability import (
    AuditEntry,
    AuditLogger,
    ChainOfThought,
    ConfidenceLevel,
    DecisionRecord,
    DecisionType,
    SelfAssessment,
)


class TestDecisionRecord:
    """Test DecisionRecord dataclass."""

    def test_create_decision(self):
        record = DecisionRecord(
            decision_type=DecisionType.TOOL_CALL,
            action="terminal: ls -la",
            reasoning="Need to list files in directory",
            confidence=ConfidenceLevel.HIGH,
        )
        assert record.decision_type == DecisionType.TOOL_CALL
        assert record.action == "terminal: ls -la"
        assert record.confidence == ConfidenceLevel.HIGH

    def test_decision_to_dict(self):
        record = DecisionRecord(
            decision_type=DecisionType.CODE_GENERATION,
            action="create function",
            reasoning="User requested a new function",
        )
        d = record.to_dict()
        assert d["decision_type"] == "code_generation"
        assert d["confidence"] == "medium"
        assert d["action"] == "create function"

    def test_decision_to_json(self):
        record = DecisionRecord(action="test")
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["action"] == "test"

    def test_decision_auto_id(self):
        r1 = DecisionRecord()
        r2 = DecisionRecord()
        assert r1.decision_id != r2.decision_id


class TestChainOfThought:
    """Test ChainOfThought dataclass."""

    def test_create_chain(self):
        chain = ChainOfThought(steps=["Step 1", "Step 2"], conclusion="Done")
        assert len(chain.steps) == 2
        assert chain.conclusion == "Done"

    def test_add_step(self):
        chain = ChainOfThought()
        chain.add_step("First")
        chain.add_step("Second")
        assert len(chain.steps) == 2
        assert chain.steps[0] == "First"

    def test_chain_to_dict(self):
        chain = ChainOfThought(steps=["A", "B"], conclusion="C")
        d = chain.to_dict()
        assert d["steps"] == ["A", "B"]
        assert d["conclusion"] == "C"


class TestAuditEntry:
    """Test AuditEntry dataclass."""

    def test_create_entry(self):
        entry = AuditEntry(event_type="tool_call", input="ls", output="file.txt")
        assert entry.event_type == "tool_call"
        assert entry.input == "ls"

    def test_entry_to_dict_with_decision(self):
        decision = DecisionRecord(action="test")
        entry = AuditEntry(decision=decision)
        d = entry.to_dict()
        assert d["decision"]["action"] == "test"

    def test_entry_to_dict_with_chain(self):
        chain = ChainOfThought(steps=["step1"], conclusion="done")
        entry = AuditEntry(chain_of_thought=chain)
        d = entry.to_dict()
        assert d["chain_of_thought"]["steps"] == ["step1"]


class TestAuditLogger:
    """Test AuditLogger functionality."""

    def test_log_decision(self):
        logger = AuditLogger()
        record = logger.log_decision(
            decision_type=DecisionType.TOOL_CALL,
            action="ls -la",
            reasoning="List files",
            confidence=ConfidenceLevel.HIGH,
        )
        assert record.action == "ls -la"
        assert len(logger.get_decisions()) == 1

    def test_log_chain_of_thought(self):
        logger = AuditLogger()
        chain = logger.log_chain_of_thought(
            steps=["Analyze input", "Choose approach", "Execute"],
            conclusion="Approach chosen",
        )
        assert len(chain.steps) == 3
        assert len(logger.get_chains()) == 1

    def test_log_tool_call(self):
        logger = AuditLogger()
        entry = logger.log_tool_call(
            tool_name="terminal",
            arguments={"command": "ls"},
            result="file1.txt",
        )
        assert entry.event_type == "tool_call"
        assert entry.metadata["tool_name"] == "terminal"

    def test_log_error(self):
        logger = AuditLogger()
        entry = logger.log_error(
            error_type="timeout",
            error_message="Connection timed out",
        )
        assert entry.event_type == "error.timeout"

    def test_log_state_snapshot(self):
        logger = AuditLogger()
        entry = logger.log_state_snapshot(
            {"tokens_used": 1000, "model": "qwen3.5:4b"}
        )
        assert entry.event_type == "state_snapshot"
        assert entry.agent_state["model"] == "qwen3.5:4b"

    def test_get_entries_filter(self):
        logger = AuditLogger()
        logger.log_decision(DecisionType.TOOL_CALL, "ls", "list files")
        logger.log_tool_call("terminal", {"command": "ls"}, "output")
        logger.log_error("timeout", "timeout")

        entries = logger.get_entries(event_type="tool_call")
        assert len(entries) == 1
        assert entries[0].event_type == "tool_call"

    def test_get_summary(self):
        logger = AuditLogger()
        logger.log_decision(DecisionType.TOOL_CALL, "ls", "list")
        logger.log_decision(DecisionType.CODE_GENERATION, "write", "create")
        logger.log_tool_call("terminal", {"command": "ls"}, "out")

        summary = logger.get_summary()
        assert summary["total_entries"] == 3
        assert summary["total_decisions"] == 2

    def test_clear(self):
        logger = AuditLogger()
        logger.log_decision(DecisionType.TOOL_CALL, "ls", "list")
        logger.clear()
        assert len(logger.get_entries()) == 0

    def test_persistent_log(self, tmp_path):
        log_file = tmp_path / "audit.log"
        logger = AuditLogger(log_path=str(log_file))
        logger.log_decision(DecisionType.TOOL_CALL, "ls", "list files")
        del logger

        # Reopen
        logger2 = AuditLogger(log_path=str(log_file))
        entries = logger2.get_entries()
        assert len(entries) >= 1

    def test_export_json(self, tmp_path):
        logger = AuditLogger()
        logger.log_decision(DecisionType.TOOL_CALL, "ls", "list")
        export_path = str(tmp_path / "export.json")
        logger.export_json(export_path)

        data = json.loads(Path(export_path).read_text())
        assert "entries" in data
        assert "summary" in data
        assert len(data["entries"]) == 1

    def test_log_decision_with_context(self):
        logger = AuditLogger()
        record = logger.log_decision(
            decision_type=DecisionType.FILE_EDIT,
            action="patch file.py",
            reasoning="Fix bug",
            context={"file": "file.py", "line": 42},
        )
        assert record.context["file"] == "file.py"

    def test_log_decision_with_alternatives(self):
        logger = AuditLogger()
        record = logger.log_decision(
            decision_type=DecisionType.TOOL_CALL,
            action="grep pattern",
            reasoning="Best approach",
            alternatives=["awk", "sed"],
        )
        assert record.alternatives == ["awk", "sed"]

    def test_get_entries_limit(self):
        logger = AuditLogger()
        for i in range(10):
            logger.log_decision(DecisionType.TOOL_CALL, f"cmd{i}", f"reason{i}")
        entries = logger.get_entries(limit=5)
        assert len(entries) == 5

    def test_export_json_clear(self, tmp_path):
        log_file = tmp_path / "audit.log"
        logger = AuditLogger(log_path=str(log_file))
        logger.log_decision(DecisionType.TOOL_CALL, "ls", "list")
        logger.clear()
        assert len(logger.get_entries()) == 0


class TestSelfAssessment:
    """Test SelfAssessment functionality."""

    def test_assess_code_has_comments(self):
        code = """# This is a comment\ndef hello():\n    print("hi")\n"""
        result = SelfAssessment.assess_code(code)
        assert result["has_comments"] is True
        assert result["language"] == "python"

    def test_assess_code_has_error_handling(self):
        code = """try:\n    do_thing()\nexcept Exception:\n    pass\n"""
        result = SelfAssessment.assess_code(code)
        assert result["has_error_handling"] is True

    def test_assess_code_has_logging(self):
        code = "print('debug')\n"
        result = SelfAssessment.assess_code(code)
        assert result["has_logging"] is True

    def test_assess_code_line_count(self):
        code = "a = 1\nb = 2\nc = 3\n"
        result = SelfAssessment.assess_code(code)
        assert result["line_count"] == 3

    def test_assess_code_meets_criteria(self):
        code = "def my_function(): pass\n"
        result = SelfAssessment.assess_code(
            code, criteria={"has_def": "def", "has_pass": "pass"}
        )
        assert result["meets_criteria"]["has_def"] is True
        assert result["meets_criteria"]["has_pass"] is True

    def test_assess_code_no_criteria(self):
        code = "x = 1\n"
        result = SelfAssessment.assess_code(code)
        assert result["meets_criteria"] == {}

    def test_assess_response_word_count(self):
        result = SelfAssessment.assess_response("hello world foo bar")
        assert result["word_count"] == 4

    def test_assess_response_has_headings(self):
        result = SelfAssessment.assess_response("# Title\nSome text\n")
        assert result["has_headings"] is True

    def test_assess_response_has_lists(self):
        result = SelfAssessment.assess_response("- Item 1\n- Item 2\n")
        assert result["has_lists"] is True

    def test_assess_response_has_code_blocks(self):
        result = SelfAssessment.assess_response("Here is code:\n```python\nx = 1\n```\n")
        assert result["has_code_blocks"] is True

    def test_assess_response_has_references(self):
        result = SelfAssessment.assess_response("See README.md for more info.")
        assert result["has_references"] is True

    def test_assess_response_completeness(self):
        result = SelfAssessment.assess_response(" " * 10)
        assert result["completeness_score"] <= 1.0

    def test_assess_empty_code(self):
        result = SelfAssessment.assess_code("")
        assert result["indentation_level"] == 0
