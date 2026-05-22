"""Tests for explainability module — audit trail, decisions, self-assessment."""

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

    def test_default_values(self):
        record = DecisionRecord()
        assert record.decision_id is not None
        assert record.decision_type == DecisionType.REASONING
        assert record.action == ""
        assert record.reasoning == ""
        assert record.alternatives == []
        assert record.confidence == ConfidenceLevel.MEDIUM
        assert record.context == {}
        assert record.outcome == ""

    def test_custom_values(self):
        record = DecisionRecord(
            decision_type=DecisionType.TOOL_CALL,
            action="read_file",
            reasoning="Need to check file contents",
            confidence=ConfidenceLevel.HIGH,
            alternatives=["search_files"],
            context={"path": "test.py"},
        )
        assert record.decision_type == DecisionType.TOOL_CALL
        assert record.action == "read_file"
        assert record.confidence == ConfidenceLevel.HIGH
        assert "test.py" in record.context["path"]

    def test_to_dict(self):
        record = DecisionRecord(
            decision_type=DecisionType.CODE_GENERATION,
            action="write_file",
            confidence=ConfidenceLevel.HIGH,
        )
        d = record.to_dict()
        assert d["decision_type"] == "code_generation"
        assert d["confidence"] == "high"
        assert d["action"] == "write_file"

    def test_to_json(self):
        record = DecisionRecord(
            decision_type=DecisionType.REASONING,
            reasoning="Test reasoning",
        )
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["decision_type"] == "reasoning"
        assert parsed["reasoning"] == "Test reasoning"

    def test_uuid_uniqueness(self):
        r1 = DecisionRecord()
        r2 = DecisionRecord()
        assert r1.decision_id != r2.decision_id


class TestChainOfThought:
    """Test ChainOfThought dataclass."""

    def test_default_values(self):
        chain = ChainOfThought()
        assert chain.chain_id is not None
        assert chain.steps == []
        assert chain.conclusion == ""

    def test_add_step(self):
        chain = ChainOfThought()
        chain.add_step("First step")
        chain.add_step("Second step")
        assert len(chain.steps) == 2
        assert chain.steps[0] == "First step"
        assert chain.steps[1] == "Second step"

    def test_to_dict(self):
        chain = ChainOfThought(
            steps=["step1", "step2"],
            conclusion="Final answer",
        )
        d = chain.to_dict()
        assert d["steps"] == ["step1", "step2"]
        assert d["conclusion"] == "Final answer"

    def test_uuid_uniqueness(self):
        c1 = ChainOfThought()
        c2 = ChainOfThought()
        assert c1.chain_id != c2.chain_id


class TestAuditEntry:
    """Test AuditEntry dataclass."""

    def test_default_values(self):
        entry = AuditEntry()
        assert entry.entry_id is not None
        assert entry.event_type == ""
        assert entry.agent_state == {}
        assert entry.decision is None
        assert entry.chain_of_thought is None

    def test_to_dict_with_decision(self):
        decision = DecisionRecord(
            decision_type=DecisionType.TOOL_CALL,
            action="test",
        )
        entry = AuditEntry(
            event_type="tool_call",
            decision=decision,
        )
        d = entry.to_dict()
        assert d["decision"]["decision_type"] == "tool_call"

    def test_to_dict_with_chain(self):
        chain = ChainOfThought(
            steps=["step"],
            conclusion="conclusion",
        )
        entry = AuditEntry(
            event_type="reasoning",
            chain_of_thought=chain,
        )
        d = entry.to_dict()
        assert d["chain_of_thought"]["steps"] == ["step"]

    def test_uuid_uniqueness(self):
        e1 = AuditEntry()
        e2 = AuditEntry()
        assert e1.entry_id != e2.entry_id


class TestAuditLogger:
    """Test AuditLogger class."""

    def test_init_no_path(self):
        logger = AuditLogger()
        assert logger._log_path is None
        assert logger._entries == []

    def test_init_with_path(self, tmp_path):
        log_file = tmp_path / "audit.jsonl"
        log_file.write_text("")
        logger = AuditLogger(str(log_file))
        assert logger._log_path == str(log_file)

    def test_log_decision(self):
        logger = AuditLogger()
        record = logger.log_decision(
            decision_type=DecisionType.TOOL_CALL,
            action="read_file",
            reasoning="Need to check file",
            confidence=ConfidenceLevel.HIGH,
        )
        assert record.action == "read_file"
        assert record.confidence == ConfidenceLevel.HIGH
        assert len(logger._entries) == 1
        assert len(logger._decisions) == 1

    def test_log_chain_of_thought(self):
        logger = AuditLogger()
        chain = logger.log_chain_of_thought(
            steps=["Step 1", "Step 2"],
            conclusion="Final answer",
        )
        assert len(chain.steps) == 2
        assert chain.conclusion == "Final answer"
        assert len(logger._chains) == 1

    def test_log_tool_call(self):
        logger = AuditLogger()
        entry = logger.log_tool_call(
            tool_name="read_file",
            arguments={"path": "test.py"},
            result="content",
        )
        assert entry.event_type == "tool_call"
        assert entry.metadata["tool_name"] == "read_file"

    def test_log_error(self):
        logger = AuditLogger()
        entry = logger.log_error(
            error_type="file_not_found",
            error_message="test.py not found",
            context={"path": "test.py"},
        )
        assert entry.event_type == "error.file_not_found"

    def test_log_state_snapshot(self):
        logger = AuditLogger()
        state = {"turn": 5, "tools_used": 3}
        entry = logger.log_state_snapshot(state)
        assert entry.event_type == "state_snapshot"
        assert entry.agent_state == state

    def test_get_entries(self):
        logger = AuditLogger()
        logger.log_decision(DecisionType.TOOL_CALL, "a", "r")
        logger.log_decision(DecisionType.REASONING, "b", "r")
        logger.log_tool_call("tool1", {})
        entries = logger.get_entries()
        assert len(entries) == 3

    def test_get_entries_filtered(self):
        logger = AuditLogger()
        logger.log_decision(DecisionType.TOOL_CALL, "a", "r")
        logger.log_decision(DecisionType.REASONING, "b", "r")
        logger.log_tool_call("tool1", {})
        entries = logger.get_entries("decision")
        assert len(entries) == 2

    def test_get_entries_limited(self):
        logger = AuditLogger()
        for i in range(10):
            logger.log_decision(DecisionType.REASONING, str(i), "r")
        entries = logger.get_entries(limit=3)
        assert len(entries) == 3

    def test_get_decisions(self):
        logger = AuditLogger()
        logger.log_decision(DecisionType.TOOL_CALL, "a", "r")
        logger.log_decision(DecisionType.REASONING, "b", "r")
        decisions = logger.get_decisions()
        assert len(decisions) == 2

    def test_get_chains(self):
        logger = AuditLogger()
        logger.log_chain_of_thought(["s1"], "c1")
        logger.log_chain_of_thought(["s2"], "c2")
        chains = logger.get_chains()
        assert len(chains) == 2

    def test_get_summary(self):
        logger = AuditLogger()
        logger.log_decision(DecisionType.TOOL_CALL, "a", "r", ConfidenceLevel.HIGH)
        logger.log_decision(DecisionType.REASONING, "b", "r", ConfidenceLevel.LOW)
        logger.log_tool_call("tool1", {})
        summary = logger.get_summary()
        assert summary["total_entries"] == 3
        assert summary["total_decisions"] == 2
        assert summary["total_chains"] == 0
        assert summary["event_counts"]["decision"] == 2
        assert summary["event_counts"]["tool_call"] == 1
        assert summary["confidence_distribution"]["high"] == 1
        assert summary["confidence_distribution"]["low"] == 1

    def test_export_json(self, tmp_path):
        logger = AuditLogger()
        logger.log_decision(DecisionType.TOOL_CALL, "a", "r")
        export_path = str(tmp_path / "export.json")
        logger.export_json(export_path)
        data = json.loads(Path(export_path).read_text())
        assert "entries" in data
        assert "summary" in data
        assert len(data["entries"]) == 1

    def test_clear(self):
        logger = AuditLogger()
        logger.log_decision(DecisionType.TOOL_CALL, "a", "r")
        logger.clear()
        assert logger._entries == []
        assert logger._decisions == []
        assert logger._chains == []

    def test_persist_to_file(self, tmp_path):
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(str(log_file))
        logger.log_decision(DecisionType.TOOL_CALL, "a", "r")
        content = log_file.read_text()
        assert len(content.strip().split("\n")) == 1
        parsed = json.loads(content.strip())
        assert parsed["event_type"] == "decision.tool_call"

    def test_load_existing_entries(self, tmp_path):
        log_file = tmp_path / "audit.jsonl"
        entry_data = {
            "entry_id": "test123",
            "timestamp": time.time(),
            "event_type": "decision.reasoning",
            "agent_state": {},
            "input": "",
            "output": "loaded",
            "decision": None,
            "chain_of_thought": None,
            "metadata": {},
        }
        log_file.write_text(json.dumps(entry_data) + "\n")
        logger = AuditLogger(str(log_file))
        assert len(logger._entries) == 1
        assert logger._entries[0].entry_id == "test123"

    def test_clear_clears_file(self, tmp_path):
        log_file = tmp_path / "audit.jsonl"
        logger = AuditLogger(str(log_file))
        logger.log_decision(DecisionType.TOOL_CALL, "a", "r")
        logger.clear()
        assert log_file.read_text() == ""


class TestSelfAssessment:
    """Test SelfAssessment class."""

    def test_assess_code_basic(self):
        code = "print('hello')\n"
        result = SelfAssessment.assess_code(code)
        assert result["type"] == "code_assessment"
        assert result["language"] == "python"
        assert result["line_count"] == 1
        assert result["has_comments"] is False
        assert result["has_error_handling"] is False

    def test_assess_code_with_comments(self):
        code = "# This is a comment\nprint('hello')\n"
        result = SelfAssessment.assess_code(code)
        assert result["has_comments"] is True

    def test_assess_code_with_error_handling(self):
        code = "try:\n    x = 1\nexcept:\n    pass\n"
        result = SelfAssessment.assess_code(code)
        assert result["has_error_handling"] is True

    def test_assess_code_with_logging(self):
        code = "import logging\nlogging.info('test')\n"
        result = SelfAssessment.assess_code(code)
        assert result["has_logging"] is True

    def test_assess_code_criteria(self):
        code = "def my_func():\n    return 42\n"
        result = SelfAssessment.assess_code(code, criteria={"my_func": "my_func"})
        assert result["meets_criteria"]["my_func"] is True

    def test_assess_code_indentation(self):
        code = "def f():\n    return 1\n"
        result = SelfAssessment.assess_code(code)
        assert result["indentation_level"] >= 0

    def test_assess_code_empty(self):
        result = SelfAssessment.assess_code("")
        assert result["line_count"] == 1
        assert result["indentation_level"] == 0

    def test_assess_response_basic(self):
        response = "This is a test response with some content."
        result = SelfAssessment.assess_response(response)
        assert result["type"] == "response_assessment"
        assert result["word_count"] > 0
        assert result["line_count"] == 1
        assert result["has_headings"] is False
        assert result["has_lists"] is False

    def test_assess_response_with_headings(self):
        response = "# Title\nSome content."
        result = SelfAssessment.assess_response(response)
        assert result["has_headings"] is True

    def test_assess_response_with_lists(self):
        response = "- Item 1\n- Item 2"
        result = SelfAssessment.assess_response(response)
        assert result["has_lists"] is True

    def test_assess_response_with_code(self):
        response = "Here is code:\n```python\nprint('hello')\n```"
        result = SelfAssessment.assess_response(response)
        assert result["has_code_blocks"] is True

    def test_assess_response_completeness(self):
        # Short response
        short = "Hi"
        result = SelfAssessment.assess_response(short)
        assert result["completeness_score"] < 1.0

        # Long response
        long_text = " ".join(["word"] * 150)
        result = SelfAssessment.assess_response(long_text)
        assert result["completeness_score"] == 1.0

    def test_estimate_indentation(self):
        code = "def f():\n    x = 1\n    if x:\n        print(x)\n"
        level = SelfAssessment._estimate_indentation(code)
        assert level > 0

    def test_estimate_indentation_empty(self):
        level = SelfAssessment._estimate_indentation("")
        assert level == 0

    def test_assess_code_javascript(self):
        code = "// comment\ntry { console.log('hi'); } catch(e) {}\n"
        result = SelfAssessment.assess_code(code, language="javascript")
        assert result["language"] == "javascript"
        assert result["has_comments"] is True
        assert result["has_error_handling"] is True
