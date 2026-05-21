"""Tests for Terminal UI — Rich-based CLI with streaming output."""

from unittest.mock import MagicMock, patch

import pytest

from local_agent.terminal_ui import TerminalUI


class TestTerminalUI:
    """Test TerminalUI components."""

    def _make_ui(self):
        """Create a TerminalUI with mocked dependencies."""
        agent = MagicMock()
        agent.run_turn = MagicMock(return_value="Agent response")
        config = MagicMock()
        config.work_dir = "/tmp/test"
        ui = TerminalUI(agent, config)
        return ui, agent

    def test_init_sets_prompt_prefix(self):
        ui, _ = self._make_ui()
        assert ui.prompt_prefix == "You"

    def test_init_sets_stop_phrase(self):
        ui, _ = self._make_ui()
        assert ui.stop_phrase == "quit"

    def test_render_welcome_message(self):
        ui, _ = self._make_ui()
        welcome = ui.render_welcome()
        assert "Local Coding Agent" in welcome
        assert "quit" in welcome

    def test_render_input_prompt(self):
        ui, _ = self._make_ui()
        prompt = ui.render_input_prompt()
        assert "You" in prompt
        assert ">>>" in prompt

    def test_render_agent_response(self):
        ui, _ = self._make_ui()
        output = ui.render_agent_response("Some response text")
        assert "Assistant" in output
        assert "Some response text" in output

    def test_render_streaming_chunks(self):
        ui, _ = self._make_ui()
        chunks = ["Hello", " ", "world"]
        result = list(ui.render_streaming_chunks(iter(chunks)))
        # Each yield accumulates the buffer progressively
        assert result == ["Hello", "Hello ", "Hello world", "Hello world\n"]

    def test_process_input_trims_whitespace(self):
        ui, _ = self._make_ui()
        assert ui.process_input("  hello  ") == "hello"

    def test_process_input_empty_returns_none(self):
        ui, _ = self._make_ui()
        assert ui.process_input("   ") is None

    def test_should_stop_with_quit(self):
        ui, _ = self._make_ui()
        assert ui.should_stop("quit") is True

    def test_should_stop_with_exit(self):
        ui, _ = self._make_ui()
        assert ui.should_stop("exit") is True

    def test_should_stop_with_regular_input(self):
        ui, _ = self._make_ui()
        assert ui.should_stop("hello") is False

    def test_run_turn_delegates_to_agent(self):
        ui, agent = self._make_ui()
        ui.run_turn("hello")
        agent.run_turn.assert_called_once_with("hello")

    def test_format_tool_call_display(self):
        ui, _ = self._make_ui()
        output = ui.format_tool_call("read_file", {"path": "test.txt"})
        assert "read_file" in output
        assert "test.txt" in output

    def test_format_error_display(self):
        ui, _ = self._make_ui()
        output = ui.format_error("Something went wrong")
        assert "Something went wrong" in output
        assert "Error" in output or "error" in output
