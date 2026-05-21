"""Tests for AgentCore — main agent loop."""

from unittest.mock import MagicMock, patch

import pytest

from local_agent.agent_core import AgentCore
from local_agent.tool_registry import ToolRegistry, ToolSchema


class TestAgentCore:
    """Test AgentCore main loop."""

    def _make_agent(self, streaming=False) -> AgentCore:
        """Create an AgentCore with mocked dependencies."""
        router = MagicMock()
        registry = ToolRegistry()
        return AgentCore(router, registry, streaming=streaming)

    def test_run_turn_calls_llm_with_system_prompt(self):
        agent = self._make_agent()
        agent._router.send_message = MagicMock(return_value="Response text")

        result = agent.run_turn("hello")
        call_args = agent._router.send_message.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert "tool" in messages[0]["content"].lower() or "assistant" in messages[0]["content"].lower()
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hello"

    def test_run_turn_returns_llm_response(self):
        agent = self._make_agent()
        agent._router.send_message = MagicMock(return_value="The answer is 42")

        result = agent.run_turn("what is the answer?")
        assert result == "The answer is 42"

    def test_run_turn_executes_tool_call(self):
        agent = self._make_agent()
        # Register a simple tool
        agent._registry.register(
            ToolSchema(
                name="echo",
                description="Echo a message",
                parameters={
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                },
            ),
            lambda msg: f"echoed: {msg}",
        )

        # LLM returns a tool call first, then a follow-up response
        agent._router.send_message = MagicMock(
            side_effect=[
                '{"tool": "echo", "args": {"msg": "hello"}}',
                "The echo result was: echoed: hello",
            ]
        )

        result = agent.run_turn("echo hello")
        assert "echoed: hello" in result

    def test_run_turn_handles_multiple_tool_calls(self):
        agent = self._make_agent()
        call_log = []
        agent._registry.register(
            ToolSchema(name="calc", description="Calc", parameters={
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            }),
            lambda expr: call_log.append(expr) or str(eval(expr)),
        )

        # LLM returns two sequential tool calls (simulated by two messages)
        def side_effect(messages):
            if len(messages) == 2:
                return '{"tool": "calc", "args": {"expr": "2+2"}}'
            else:
                return "The result is 4"

        agent._router.send_message = MagicMock(side_effect=side_effect)
        result = agent.run_turn("calculate 2+2")
        assert "4" in result
        assert "2+2" in call_log

    def test_run_turn_handles_no_tool_call(self):
        agent = self._make_agent()
        agent._router.send_message = MagicMock(return_value="Just a plain response, no tools needed.")

        result = agent.run_turn("tell me a fact")
        assert result == "Just a plain response, no tools needed."

    def test_run_turn_uses_streaming_when_available(self):
        agent = self._make_agent(streaming=True)
        agent._router.stream_message = MagicMock(return_value=iter(["chunk1 ", "chunk2"]))

        result = agent.run_turn("stream me")
        assert result == "chunk1 chunk2"

    def test_system_prompt_includes_tool_definitions(self):
        agent = self._make_agent()
        agent._registry.register(
            ToolSchema(name="greet", description="Say hello", parameters={}),
            lambda: "hi",
        )
        agent._router.send_message = MagicMock(return_value="done")

        agent.run_turn("test")
        messages = agent._router.send_message.call_args[0][0]
        system_content = messages[0]["content"]
        assert "greet" in system_content
        assert "Say hello" in system_content

    def test_run_turn_invalid_tool_call_is_handled(self):
        agent = self._make_agent()
        agent._router.send_message = MagicMock(
            return_value='{"tool": "nonexistent", "args": {}}'
        )

        result = agent.run_turn("try bad tool")
        assert "error" in result.lower() or "not found" in result.lower()

    def test_run_turn_handles_plain_text_with_tool_syntax(self):
        """Test that non-JSON responses are passed through as-is."""
        agent = self._make_agent()
        agent._router.send_message = MagicMock(
            return_value="Here's some text with {brackets} but not a tool call."
        )

        result = agent.run_turn("test")
        assert result == "Here's some text with {brackets} but not a tool call."

    def test_run_turn_executes_tool_call_with_prose_prefix(self):
        """Test that JSON tool calls embedded in prose are extracted and executed."""
        agent = self._make_agent()
        agent._registry.register(
            ToolSchema(
                name="echo",
                description="Echo a message",
                parameters={
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                },
            ),
            lambda msg: f"echoed: {msg}",
        )

        # LLM returns prose before the JSON tool call
        agent._router.send_message = MagicMock(
            side_effect=[
                "I'll help you with that.\n\n{\"tool\": \"echo\", \"args\": {\"msg\": \"hello\"}}",
                "The echo result was: echoed: hello",
            ]
        )

        result = agent.run_turn("echo hello")
        assert "echoed: hello" in result
