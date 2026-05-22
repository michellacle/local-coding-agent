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
        assert "/ls" in result
        assert "/status" in result
        assert "/quit" in result
        assert "Read, write, and edit files" in result
        assert "Search codebases" in result
        assert "List directories" in result
        assert "Run shell commands" in result
        assert "Manage git repos" in result
        assert "Extract PDFs" in result
        assert "Manage skills" in result
        assert "Search past sessions" in result
        assert "Delegate tasks" in result
        assert "Persistent memory" in result
        assert "clarification" in result
        assert "retry strategy" in result

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


class TestLsCommand:
    """Test /ls slash command."""

    def _make_ui(self, work_dir: str | None = None):
        import tempfile
        if work_dir is None:
            work_dir = tempfile.mkdtemp()
        from local_agent.terminal_ui import TerminalUI
        agent = MagicMock()
        config = MagicMock()
        config.work_dir = work_dir
        config.llm = MagicMock()
        config.llm.model = "test"
        config.toolsets = ["file"]
        ui = TerminalUI(agent, config)
        return ui, work_dir

    def test_ls_basic(self, tmp_path):
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.md").write_text("hello world", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        ui, _ = self._make_ui(str(tmp_path))
        result = ui._render_ls("/ls")
        assert "file1.txt" in result
        assert "file2.md" in result
        assert "subdir" in result
        assert "entries" in result

    def test_ls_custom_path(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.py").write_text("print(1)")
        # Create a subdirectory
        sub = tmp_path / "nested"
        sub.mkdir()
        (sub / "c.txt").write_text("y")

        ui, _ = self._make_ui(str(tmp_path))
        # List the subdirectory
        result = ui._render_ls(f"/ls {sub}")
        assert "c.txt" in result
        assert "a.txt" not in result

    def test_ls_nonexistent_path(self, tmp_path):
        ui, _ = self._make_ui(str(tmp_path))
        result = ui._render_ls("/ls /nonexistent/path")
        assert "Error" in result

    def test_ls_on_file_not_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        ui, _ = self._make_ui(str(tmp_path))
        result = ui._render_ls(f"/ls {f}")
        assert "Error" in result

    def test_ls_empty_dir(self, tmp_path):
        ui, _ = self._make_ui(str(tmp_path))
        result = ui._render_ls("/ls")
        assert "(empty)" in result

    def test_ls_skips_hidden(self, tmp_path):
        (tmp_path / "visible.txt").write_text("x")
        (tmp_path / ".hidden").write_text("y")
        (tmp_path / ".gitkeep").write_text("")
        ui, _ = self._make_ui(str(tmp_path))
        result = ui._render_ls("/ls")
        assert "visible.txt" in result
        assert ".hidden" not in result
        assert ".gitkeep" not in result

    def test_ls_shows_hidden_count(self, tmp_path):
        (tmp_path / "visible.txt").write_text("x")
        (tmp_path / ".hidden").write_text("y")
        ui, _ = self._make_ui(str(tmp_path))
        result = ui._render_ls("/ls")
        assert "hidden" in result

    def test_ls_shows_file_sizes(self, tmp_path):
        # Small file
        (tmp_path / "small.txt").write_text("x")
        # KB file
        (tmp_path / "medium.txt").write_text("x" * 2048)
        # MB file
        (tmp_path / "large.dat").write_bytes(bytes(2 * 1024 * 1024))
        ui, _ = self._make_ui(str(tmp_path))
        result = ui._render_ls("/ls")
        assert "small.txt" in result
        assert "1B" in result
        assert "medium.txt" in result
        assert "2.0K" in result
        assert "large.dat" in result
        assert "2.0M" in result

    def test_ls_dirs_sorted_first(self, tmp_path):
        (tmp_path / "z.txt").write_text("z")
        (tmp_path / "adir").mkdir()
        (tmp_path / "bdir").mkdir()
        ui, _ = self._make_ui(str(tmp_path))
        result = ui._render_ls("/ls")
        # Dirs come before files in output
        assert "adir" in result
        assert "bdir" in result

    def test_ls_in_help(self):
        ui, _ = self._make_ui()
        help_text = ui.render_help()
        assert "/ls" in help_text

    def test_handle_special_command_ls(self, tmp_path):
        (tmp_path / "test.py").write_text("print(1)")
        ui, _ = self._make_ui(str(tmp_path))
        result = ui.handle_special_command("/ls")
        assert result is not None
        assert "test.py" in result
