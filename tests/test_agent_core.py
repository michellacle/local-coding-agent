"""Tests for AgentCore — main agent loop with multi-turn tool chaining."""

from unittest.mock import MagicMock, call

import pytest

from local_agent.agent_core import AgentCore
from local_agent.tool_registry import ToolRegistry, ToolSchema


class TestAgentCore:
    """Test AgentCore main loop."""

    def _make_agent(self, streaming=False, max_turns=10) -> AgentCore:
        """Create an AgentCore with mocked dependencies."""
        router = MagicMock()
        registry = ToolRegistry()
        return AgentCore(router, registry, streaming=streaming, max_turns=max_turns)

    def test_run_turn_calls_llm_with_system_prompt(self):
        agent = self._make_agent()
        agent._router.send_message = MagicMock(return_value="Response text")

        agent.run_turn("hello")
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

    def test_run_turn_single_tool_call(self):
        """Agent executes one tool call, then returns final answer."""
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

        # Turn 1: LLM returns a tool call
        # Turn 2: After tool result, LLM returns final plain text
        agent._router.send_message = MagicMock(
            side_effect=[
                '{"tool": "echo", "args": {"msg": "hello"}}',
                "I echoed the message for you.",
            ]
        )

        result = agent.run_turn("echo hello")
        assert result == "I echoed the message for you."
        assert agent._router.send_message.call_count == 2

    def test_run_turn_chains_multiple_tool_calls(self):
        """Agent chains read_file -> write_file -> final answer (3 LLM calls)."""
        agent = self._make_agent()
        read_content = "original content line 1\noriginal content line 2"

        def fake_read(path, offset=1, limit=500):
            return read_content

        def fake_write(path, content):
            return {"path": path, "content": content}

        agent._registry.register(
            ToolSchema(
                name="read_file",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            ),
            fake_read,
        )
        agent._registry.register(
            ToolSchema(
                name="write_file",
                description="Write a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            ),
            fake_write,
        )

        # 3 LLM turns: read_file tool -> write_file tool -> final answer
        agent._router.send_message = MagicMock(
            side_effect=[
                '{"tool": "read_file", "args": {"path": "foo.txt"}}',
                '{"tool": "write_file", "args": {"path": "foo.txt", "content": "new content"}}',
                "Done — read the file and wrote new content.",
            ]
        )

        result = agent.run_turn("read and rewrite foo.txt")
        assert result == "Done — read the file and wrote new content."
        assert agent._router.send_message.call_count == 3

    def test_run_turn_handles_no_tool_call(self):
        """Plain text response returned immediately without looping."""
        agent = self._make_agent()
        agent._router.send_message = MagicMock(return_value="Just a plain response, no tools needed.")

        result = agent.run_turn("tell me a fact")
        assert result == "Just a plain response, no tools needed."
        assert agent._router.send_message.call_count == 1

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

    def test_run_turn_invalid_tool_call_handled(self):
        """Unknown tool name returns error, which is fed back to LLM for recovery."""
        agent = self._make_agent(max_turns=5)
        # Turn 1: LLM tries unknown tool
        # Turn 2: LLM recovers with plain text after seeing error
        agent._router.send_message = MagicMock(
            side_effect=[
                '{"tool": "nonexistent", "args": {}}',
                "Apologies, that tool isn't available. Here's what I can do instead.",
            ]
        )

        result = agent.run_turn("try bad tool")
        assert "instead" in result
        assert agent._router.send_message.call_count == 2

    def test_run_turn_plain_text_with_brackets_passed_through(self):
        """Non-JSON text with curly braces is NOT treated as a tool call."""
        agent = self._make_agent()
        agent._router.send_message = MagicMock(
            return_value="Here's some text with {brackets} but not a tool call."
        )

        result = agent.run_turn("test")
        assert result == "Here's some text with {brackets} but not a tool call."
        assert agent._router.send_message.call_count == 1

    def test_run_turn_prose_wrapped_tool_call(self):
        """JSON tool call embedded in prose is extracted and executed."""
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

        # Turn 1: Prose + JSON tool call
        # Turn 2: Final answer after tool result
        agent._router.send_message = MagicMock(
            side_effect=[
                "I'll help you with that.\n\n{\"tool\": \"echo\", \"args\": {\"msg\": \"hello\"}}",
                "Done echoing.",
            ]
        )

        result = agent.run_turn("echo hello")
        assert result == "Done echoing."
        assert agent._router.send_message.call_count == 2

    def test_run_turn_respects_max_turns(self):
        """When max_turns is hit, agent returns the last tool result."""
        agent = self._make_agent(max_turns=3)
        agent._registry.register(
            ToolSchema(
                name="loop_tool",
                description="Loops forever",
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            lambda: "loop result",
        )

        # LLM keeps returning tool calls forever
        agent._router.send_message = MagicMock(
            return_value='{"tool": "loop_tool", "args": {}}'
        )

        result = agent.run_turn("loop")
        # Should stop at max_turns, not infinite loop
        assert agent._router.send_message.call_count == 3
        assert "loop result" in result

    def test_history_persists_across_tool_calls(self):
        """History accumulates: user msg -> assistant tool call -> tool result -> assistant final."""
        agent = self._make_agent()
        agent._registry.register(
            ToolSchema(
                name="echo",
                description="Echo",
                parameters={
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                },
            ),
            lambda msg: f"echoed: {msg}",
        )

        agent._router.send_message = MagicMock(
            side_effect=[
                '{"tool": "echo", "args": {"msg": "hello"}}',
                "Done.",
            ]
        )

        agent.run_turn("say hello")

        # History should contain: user, assistant(tool), user(result), assistant(final)
        roles = [m["role"] for m in agent._history]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_reset_history_clears_context(self):
        """reset_history clears conversation for next independent turn."""
        agent = self._make_agent()
        agent._router.send_message = MagicMock(
            side_effect=[
                '{"tool": "echo", "args": {"msg": "hi"}}',
                "Done.",
            ]
        )
        agent._registry.register(
            ToolSchema(
                name="echo",
                description="Echo",
                parameters={
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                },
            ),
            lambda msg: f"echoed: {msg}",
        )

        agent.run_turn("say hi")
        assert len(agent._history) == 4

        agent.reset_history()
        assert len(agent._history) == 0

    def test_tool_result_dict_serialized_to_json(self):
        """Dict results from tools are JSON-serialized for the LLM feedback."""
        agent = self._make_agent()
        agent._registry.register(
            ToolSchema(
                name="calc",
                description="Calculate",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            ),
            lambda x: {"result": x * 2, "status": "ok"},
        )

        # Track what feedback the LLM receives on turn 2
        received_messages: list[list[dict]] = []
        call_count = [0]
        def side_effect(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"tool": "calc", "args": {"x": 5}}'
            received_messages.append(messages)
            return "Calculation done."

        agent._router.send_message = MagicMock(side_effect=side_effect)
        agent.run_turn("double 5")

        # The second call's messages should include the JSON tool result
        # History after tool call: [system, user("double 5"), assistant(tool), user(result)]
        tool_result_msg = received_messages[0][3]
        assert tool_result_msg["role"] == "user"
        assert '"result": 10' in tool_result_msg["content"]
        assert '"status": "ok"' in tool_result_msg["content"]
