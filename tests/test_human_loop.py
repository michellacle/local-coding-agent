"""Tests for human_loop — clarification prompts and approval workflows."""

from pathlib import Path

import pytest

from local_agent.tools.human_loop import (
    clarify,
    confirm,
    BlockingInteraction,
    ClarificationRequest,
    ClarificationResponse,
    ApprovalRequest,
    HumanLoopManager,
)


class TestClarificationRequest:
    """Test ClarificationRequest dataclass."""

    def test_open_ended(self):
        req = ClarificationRequest(question="What framework?")
        assert not req.is_multiple_choice
        assert req.choices == []

    def test_multiple_choice(self):
        req = ClarificationRequest(
            question="Choose:", choices=["React", "Vue", "Svelte"]
        )
        assert req.is_multiple_choice
        assert len(req.choices) == 3

    def test_to_json(self):
        req = ClarificationRequest(
            question="Pick one", choices=["A", "B"]
        )
        data = req.to_json()
        assert "clarification" in data
        assert "Pick one" in data


class TestClarifyTool:
    """Test the clarify tool raises BlockingInteraction."""

    def test_open_ended_raises_blocking(self):
        with pytest.raises(BlockingInteraction) as exc:
            clarify("What is your name?")
        assert isinstance(exc.value.request, ClarificationRequest)
        assert exc.value.request.question == "What is your name?"
        assert not exc.value.request.is_multiple_choice

    def test_multiple_choice_raises_blocking(self):
        with pytest.raises(BlockingInteraction) as exc:
            clarify("Pick a framework", choices=["React", "Vue"])
        assert isinstance(exc.value.request, ClarificationRequest)
        assert exc.value.request.choices == ["React", "Vue"]

    def test_too_many_choices(self):
        with pytest.raises(ValueError, match="at most 4"):
            clarify("Pick", choices=["A", "B", "C", "D", "E"])

    def test_no_choices_raises_blocking(self):
        with pytest.raises(BlockingInteraction) as exc:
            clarify("Tell me something", choices=[])
        assert isinstance(exc.value.request, ClarificationRequest)


class TestConfirmTool:
    """Test the confirm tool raises BlockingInteraction."""

    def test_confirm_raises_blocking(self):
        with pytest.raises(BlockingInteraction) as exc:
            confirm("Delete all files", details="This will remove 5 files")
        assert isinstance(exc.value.request, ApprovalRequest)
        assert exc.value.request.action == "Delete all files"
        assert exc.value.request.details == "This will remove 5 files"
        assert exc.value.request.default_deny is True

    def test_confirm_no_details(self):
        with pytest.raises(BlockingInteraction) as exc:
            confirm("Run migration")
        assert isinstance(exc.value.request, ApprovalRequest)
        assert exc.value.request.details == ""


class TestApprovalRequest:
    """Test ApprovalRequest dataclass."""

    def test_defaults(self):
        req = ApprovalRequest(action="test")
        assert req.action == "test"
        assert req.details == ""
        assert req.default_deny is True


class TestBlockingInteraction:
    """Test BlockingInteraction exception."""

    def test_stores_request(self):
        req = ClarificationRequest(question="hi")
        exc = BlockingInteraction(req)
        assert exc.request is req

    def test_error_message(self):
        req = ApprovalRequest(action="rm -rf")
        exc = BlockingInteraction(req)
        assert "Blocking interaction" in str(exc)


class TestHumanLoopManager:
    """Test HumanLoopManager."""

    def test_ask_clarification_raises(self):
        mgr = HumanLoopManager()
        with pytest.raises(BlockingInteraction):
            mgr.ask_clarification("What framework?", choices=["React", "Vue"])

    def test_request_approval_raises(self):
        mgr = HumanLoopManager()
        with pytest.raises(BlockingInteraction):
            mgr.request_approval("Drop database")

    def test_audit_trail_records(self):
        mgr = HumanLoopManager()
        with pytest.raises(BlockingInteraction):
            mgr.ask_clarification("What?")
        trail = mgr.get_audit_trail()
        assert len(trail) == 1
        assert trail[0]["event"] == "ask"

    def test_audit_trail_empty(self):
        mgr = HumanLoopManager()
        assert mgr.get_audit_trail() == []

    def test_audit_trail_multiple(self):
        mgr = HumanLoopManager()
        with pytest.raises(BlockingInteraction):
            mgr.ask_clarification("What?")
        with pytest.raises(BlockingInteraction):
            mgr.request_approval("Delete")
        trail = mgr.get_audit_trail()
        assert len(trail) == 2
        assert trail[0]["event"] == "ask"
        assert trail[1]["event"] == "approve"

    def test_clarification_response(self):
        resp = ClarificationResponse(
            question="What?", answer="React"
        )
        assert resp.question == "What?"
        assert resp.answer == "React"
        assert resp.was_other is False
