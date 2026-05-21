"""Tests for terminal_tools — execute_command."""

from unittest.mock import MagicMock, patch

import pytest

from local_agent.tools.terminal_tools import execute_command


class TestExecuteCommand:
    """Test execute_command function."""

    def test_returns_output_and_exit_code(self):
        mock_result = MagicMock()
        mock_result.stdout = "hello\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = execute_command("echo hello")
            assert result["output"] == "hello\n"
            assert result["exit_code"] == 0

    def test_captures_stderr_on_error(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "error: something failed\n"
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            result = execute_command("false_command")
            assert result["exit_code"] == 1
            assert "error: something failed" in result["output"]

    def test_passes_timeout(self):
        mock_result = MagicMock()
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            execute_command("echo ok", timeout=30)
            mock_run.assert_called_once()
            assert mock_run.call_args[1]["timeout"] == 30

    def test_passes_workdir(self):
        mock_result = MagicMock()
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            execute_command("pwd", workdir="/tmp/test")
            mock_run.assert_called_once()
            assert mock_run.call_args[1]["cwd"] == "/tmp/test"

    def test_timeout_error_raises(self):
        with patch("subprocess.run", side_effect=TimeoutError("timed out")):
            with pytest.raises(TimeoutError):
                execute_command("sleep 999", timeout=1)

    def test_default_timeout_is_180(self):
        mock_result = MagicMock()
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            execute_command("echo ok")
            assert mock_run.call_args[1]["timeout"] == 180

    def test_runs_with_shell_true(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            execute_command("ls -la")
            assert mock_run.call_args[1]["shell"] is True
