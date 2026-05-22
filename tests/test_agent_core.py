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


class TestExtractJsonBraces:
    """Test brace-counting JSON extraction."""

    def setup_method(self) -> None:
        self.agent = AgentCore.__new__(AgentCore)  # no __init__ needed

    def test_simple_json(self):
        result = self.agent._extract_json_braces('{"tool": "echo", "args": {"msg": "hi"}}')
        assert result == '{"tool": "echo", "args": {"msg": "hi"}}'

    def test_nested_braces(self):
        text = '{"a": {"b": {"c": 1}}}'
        result = self.agent._extract_json_braces(text)
        assert result == text

    def test_braces_in_string_value(self):
        # Content has { } inside a string — brace counting still works
        text = '{"content": "print({len(numbers)})"}'
        result = self.agent._extract_json_braces(text)
        assert result == text

    def test_prose_before_json(self):
        text = 'I will do that.\n\n{"tool": "echo", "args": {"msg": "hi"}}'
        result = self.agent._extract_json_braces(text)
        assert result == '{"tool": "echo", "args": {"msg": "hi"}}'

    def test_prose_after_json(self):
        text = '{"tool": "echo", "args": {"msg": "hi"}}\nDone.'
        result = self.agent._extract_json_braces(text)
        assert result == '{"tool": "echo", "args": {"msg": "hi"}}'

    def test_no_json(self):
        result = self.agent._extract_json_braces("Just plain text, nothing here.")
        assert result is None

    def test_multiple_objects(self):
        """Returns the first complete JSON object."""
        text = '{"a": 1} {"b": 2}'
        result = self.agent._extract_json_braces(text)
        assert result == '{"a": 1}'

    def test_deeply_nested(self):
        text = '{"tool": "write_file", "args": {"content": "f\\\"{x}\\\""}}'
        result = self.agent._extract_json_braces(text)
        assert result == text


class TestRepairJson:
    """Test JSON repair for literal control characters in strings."""

    def setup_method(self) -> None:
        self.agent = AgentCore.__new__(AgentCore)

    def test_repair_literal_newlines(self):
        text = '{"tool": "write_file", "args": {"content": "line1\nline2\nline3"}}'
        result = self.agent._repair_json(text)
        assert result == '{"tool": "write_file", "args": {"content": "line1\\nline2\\nline3"}}'

    def test_repair_literal_tabs(self):
        text = '{"msg": "col1\tcol2"}'
        result = self.agent._repair_json(text)
        assert result == '{"msg": "col1\\tcol2"}'

    def test_repair_literal_carriage_returns(self):
        text = '{"msg": "hello\rworld"}'
        result = self.agent._repair_json(text)
        assert result == '{"msg": "hello\\rworld"}'

    def test_already_valid_json(self):
        text = '{"tool": "echo", "args": {"msg": "hello"}}'
        result = self.agent._repair_json(text)
        assert result == text

    def test_unrepairable_returns_none(self):
        text = '{"tool": "echo" "args": {}}'  # missing comma
        result = self.agent._repair_json(text)
        assert result is None

    def test_preserves_backslash_escapes(self):
        text = '{"msg": "path\\\\to\\\\file"}'
        result = self.agent._repair_json(text)
        assert result == text

    def test_repair_fstring_braces_in_content(self):
        # The exact failure case: f-string braces in content + literal newlines
        # Use escaped quotes properly so the JSON structure itself is valid
        # after newline repair — only the newlines need fixing.
        text = (
            '{"tool": "write_file", "args": {"path": "new.py", "content": '
            '"import random\n\nnumbers = [random.randint(1,100) for _ in range(10)]\n'
            'total = sum(numbers)\nprint(f\\"Sum: {total}\\")"}}'
        )
        repaired = self.agent._repair_json(text)
        assert repaired is not None
        parsed = __import__("json").loads(repaired)
        assert parsed["tool"] == "write_file"
        assert "import random" in parsed["args"]["content"]


class TestExecuteToolWithMalformedJson:
    """Test that tool calls with literal newlines in content are executed."""

    def _make_agent(self, max_turns=10) -> AgentCore:
        router = MagicMock()
        registry = ToolRegistry()
        return AgentCore(router, registry, max_turns=max_turns)

    def test_literal_newlines_in_write_file_content(self):
        """Reproduce the real bug: LLM sends literal newlines in file content."""
        agent = self._make_agent()

        written_files: list[dict] = []

        def fake_write(path, content):
            written_files.append({"path": path, "content": content})
            return {"status": "ok", "path": path}

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

        # Simulate LLM response with literal newlines in content (the real bug)
        llm_response = (
            '{"tool": "write_file", "args": {"path": "new.py", "content": '
            '"import random\\n\\nnumbers = [random.randint(1,100) for _ in range(10)]\\n'
            'total = sum(numbers)\\nprint(f\\"Sum: {total}\\")"}}'
        )

        # Turn 1: tool call with newlines, Turn 2: final answer
        agent._router.send_message = MagicMock(
            side_effect=[llm_response, "File created successfully."]
        )

        result = agent.run_turn("create new.py with random sum")

        # The tool should have been called — file was written
        assert len(written_files) == 1
        assert written_files[0]["path"] == "new.py"
        assert "import random" in written_files[0]["content"]
        assert result == "File created successfully."

    def test_fstring_braces_in_content(self):
        """f-string syntax like {total} in content should not break parsing."""
        agent = self._make_agent()

        called = [False]

        def fake_write(path, content):
            called[0] = True
            return {"status": "ok"}

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

        # JSON with {total} braces inside content string
        agent._router.send_message = MagicMock(
            side_effect=[
                '{"tool": "write_file", "args": {"path": "x.py", "content": "print(f\\"Sum: {total}\\")"}}',
                "Done.",
            ]
        )

        result = agent.run_turn("write file with fstring")
        assert called[0] is True
        assert result == "Done."

    def test_parentheses_in_content(self):
        """Parentheses in content like random.randint(1,100) should work."""
        agent = self._make_agent()

        written: list[str] = []

        def fake_write(path, content):
            written.append(content)
            return {"status": "ok"}

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

        agent._router.send_message = MagicMock(
            side_effect=[
                '{"tool": "write_file", "args": {"path": "x.py", "content": "import random\\nnums = [random.randint(1, 100) for _ in range(10)]"}}',
                "Done.",
            ]
        )

        result = agent.run_turn("write file")
        assert len(written) == 1
        assert "randint" in written[0]
        assert result == "Done."
