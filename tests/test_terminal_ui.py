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
        config.llm = MagicMock()
        config.llm.model = "qwen3.5:4b"
        config.toolsets = ["file", "terminal", "git"]
        ui = TerminalUI(agent, config)
        return ui, agent

    def test_init_sets_prompt_prefix(self):
        ui, _ = self._make_ui()
        assert ui.prompt_prefix == "localcode"

    def test_init_sets_stop_phrase(self):
        ui, _ = self._make_ui()
        assert ui.stop_phrase == "quit"

    def test_render_welcome_message(self):
        ui, _ = self._make_ui()
        welcome = ui.render_welcome()
        assert "Local Coding Agent" in welcome
        assert "quit" in welcome
        assert "/help" in welcome

    def test_render_input_prompt(self):
        ui, _ = self._make_ui()
        prompt = ui.render_input_prompt()
        assert "localcode" in prompt
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

    def test_should_stop_with_q(self):
        ui, _ = self._make_ui()
        assert ui.should_stop("q") is True

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

    # Slash command tests

    def test_is_special_command_slash_prefix(self):
        ui, _ = self._make_ui()
        assert ui.is_special_command("/help") is True
        assert ui.is_special_command("/status") is True
        assert ui.is_special_command("/quit") is True

    def test_is_special_command_no_slash(self):
        ui, _ = self._make_ui()
        assert ui.is_special_command("hello") is False
        assert ui.is_special_command("help") is False

    def test_handle_help_command(self):
        ui, _ = self._make_ui()
        result = ui.handle_special_command("/help")
        assert result is not None
        assert "Slash Commands" in result
        assert "/help" in result
        assert "/status" in result
        assert "/quit" in result
        assert "Read, write, and edit files" in result
        assert "Search codebases" in result
        assert "Run shell commands" in result
        assert "Manage git repos" in result

    def test_handle_status_command(self):
        ui, _ = self._make_ui()
        result = ui.handle_special_command("/status")
        assert result is not None
        assert "Session Status" in result
        assert "/tmp/test" in result
        assert "qwen3.5:4b" in result
        assert "file" in result

    def test_handle_quit_command(self):
        ui, _ = self._make_ui()
        assert ui.handle_special_command("/quit") is None
        assert ui.handle_special_command("/exit") is None

    def test_handle_unknown_command(self):
        ui, _ = self._make_ui()
        result = ui.handle_special_command("/foobar")
        assert result is not None
        assert "Unknown command" in result
        assert "/foobar" in result
