"""Tests for AgentCore human-in-the-loop integration."""

import json

import pytest

from local_agent.agent_core import AgentCore
from local_agent.tool_registry import ToolRegistry, ToolSchema
from local_agent.tools.human_loop import (
    BlockingInteraction,
    ClarificationRequest,
)
from local_agent.model_router import ModelRouter, LLMConfig


def _make_agent(human_io=None):
    """Build an AgentCore with a mock router and registry."""
    config = LLMConfig(model="test")
    router = ModelRouter(config)
    registry = ToolRegistry()

    # Register a tool that raises BlockingInteraction
    def _ask(question: str, choices: list[str] | None = None):
        req = ClarificationRequest(question=question, choices=choices or [])
        raise BlockingInteraction(req)

    registry.register(
        schema=ToolSchema(
            name="ask_human",
            description="Ask human a question",
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "choices": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question"],
            },
        ),
        fn=_ask,
    )

    return AgentCore(
        router=router,
        registry=registry,
        max_turns=5,
        human_io=human_io,
    )


class TestHumanIoCallback:
    """Test that BlockingInteraction is routed to the human_io callback."""

    def test_calls_human_io_callback(self):
        received = []

        def _handler(exc: BlockingInteraction) -> str:
            received.append(exc.request)
            return "Python"

        agent = _make_agent(human_io=_handler)

        call_count = [0]

        def mock_send(msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: LLM returns a tool call
                return json.dumps({"tool": "ask_human", "args": {"question": "Pick a lang?"}})
            # After tool result fed back, return final answer
            return "Okay, I'll use Python."

        agent._router.send_message = mock_send

        result = agent.run_turn("help me pick")
        assert len(received) == 1
        assert received[0].question == "Pick a lang?"
        assert call_count[0] == 2
        assert "Python" in result

    def test_no_callback_returns_error(self):
        """Without human_io, get an error string fed back to LLM."""
        agent = _make_agent(human_io=None)

        call_count = [0]

        def mock_send(msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"tool": "ask_human", "args": {"question": "hi?"}})
            return "done"

        agent._router.send_message = mock_send

        result = agent.run_turn("test")
        assert call_count[0] == 2  # tool call + final answer
        assert result == "done"

    def test_tool_result_contains_user_response(self):
        """Verify user answer is included in tool result fed to LLM."""
        def _handler(exc: BlockingInteraction) -> str:
            return "React"

        agent = _make_agent(human_io=_handler)

        call_count = [0]

        def mock_send(msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"tool": "ask_human", "args": {"question": "Framework?"}})
            # Second call: the tool result is in the messages
            # Check it contains the user response
            last_user = [m for m in msgs if m["role"] == "user"][-1]
            assert "React" in last_user["content"]
            return "Using React then."

        agent._router.send_message = mock_send

        result = agent.run_turn("choose")
        assert result == "Using React then."


class TestBlockingInteractionNoCallback:
    """Verify agent degrades gracefully when no human_io is set."""

    def test_blocking_error_feeds_back(self):
        """The error about no callback becomes a tool result fed back to LLM."""
        agent = _make_agent(human_io=None)

        call_count = [0]

        def mock_send(msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"tool": "ask_human", "args": {"question": "x?"}})
            return "final"

        agent._router.send_message = mock_send

        result = agent.run_turn("go")
        assert call_count[0] == 2
        assert result == "final"
