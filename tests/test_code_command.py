"""Tests for /code slash command — force file-producing coding tasks."""

import pytest
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

from local_agent.terminal_ui import TerminalUI


@pytest.fixture
def ui(tmp_path):
    """Create a TerminalUI with a mock agent and temp work dir."""
    agent = MagicMock()
    agent.run_turn.return_value = "Created files: main.py, utils.py"

    from local_agent.config import AppConfig

    config = AppConfig(work_dir=str(tmp_path))

    ui = TerminalUI(agent, config)
    return ui


class TestCmdCodeUsage:
    """Test /code without arguments shows usage."""

    def test_no_args_shows_usage(self, ui):
        result = ui._cmd_code("")
        assert "Code Mode" in result or "code" in result.lower()
        assert "write" in result.lower() or "file" in result.lower()

    def test_no_args_shows_examples(self, ui):
        result = ui._cmd_code("")
        # Should show example commands
        assert "/" in result  # contains example usage


class TestCmdCodeExecution:
    """Test /code with arguments triggers the agent with file-writing instructions."""

    def test_triggers_agent_turn(self, ui):
        ui._cmd_code("A Python script to calculate fibonacci")
        ui.agent.run_turn.assert_called_once()

    def test_includes_user_request(self, ui):
        ui._cmd_code("A bash script to backup files")
        call_args = ui.agent.run_turn.call_args[0][0]
        assert "bash script" in call_args.lower() or "backup files" in call_args.lower()

    def test_includes_strict_file_rules(self, ui):
        ui._cmd_code("todo app")
        call_args = ui.agent.run_turn.call_args[0][0]
        assert "MUST" in call_args or "must" in call_args
        assert "file" in call_args.lower()

    def test_includes_working_directory(self, ui):
        ui._cmd_code("simple calculator")
        call_args = ui.agent.run_turn.call_args[0][0]
        assert str(ui.work_dir) in call_args

    def test_includes_write_file_instruction(self, ui):
        ui._cmd_code("hello world")
        call_args = ui.agent.run_turn.call_args[0][0]
        assert "write_file" in call_args or "write" in call_args.lower()

    def test_returns_agent_response(self, ui):
        result = ui._cmd_code("make a script")
        assert result == "Created files: main.py, utils.py"

    def test_code_instruction_has_strict_rules(self, ui):
        ui._cmd_code("data processor")
        call_args = ui.agent.run_turn.call_args[0][0]
        # Check for numbered rules
        assert "1." in call_args or "STRICT" in call_args.upper()

    def test_code_instruction_says_not_stdout(self, ui):
        ui._cmd_code("api server")
        call_args = ui.agent.run_turn.call_args[0][0]
        assert "stdout" in call_args.lower() or "print" in call_args.lower()


class TestCmdCodeInDispatch:
    """Test that /code is wired into the command dispatcher."""

    def test_code_in_dispatch(self, ui):
        result = ui.handle_special_command("/code test description")
        assert ui.agent.run_turn.called

    def test_code_no_args_in_dispatch(self, ui):
        result = ui.handle_special_command("/code")
        # Should show usage, not call agent
        assert not ui.agent.run_turn.called
        assert "Code Mode" in result or "code" in result.lower()
