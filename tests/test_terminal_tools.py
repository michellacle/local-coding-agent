"""Tests for terminal_tools — execute_command and background process management."""

import subprocess as sp
from unittest.mock import MagicMock, patch

import pytest

from local_agent.tools.terminal_tools import (
    execute_command,
    process_manage,
    clear_background_processes,
    list_background_processes,
    get_process_output,
)


@pytest.fixture(autouse=True)
def _clean_processes():
    """Clear background process registry after each test."""
    yield
    clear_background_processes()


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


class TestBackgroundExecution:
    """Test background process execution."""

    def test_start_background_returns_session_id(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            result = execute_command("sleep 100", background=True)

            assert "session_id" in result
            assert "message" in result
            assert len(result["session_id"]) == 8

    def test_background_passes_workdir(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            execute_command("sleep 100", workdir="/tmp", background=True)

            mock_popen.assert_called_once()
            assert mock_popen.call_args[1]["cwd"] == "/tmp"

    def test_list_empty_when_none(self):
        result = list_background_processes()
        assert result["processes"] == []
        assert "No background processes" in result["message"]

    def test_list_shows_running_processes(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            execute_command("sleep 100", background=True)

            result = list_background_processes()
            assert len(result["processes"]) == 1
            assert result["processes"][0]["status"] == "running"

    def test_list_shows_exited_processes(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0
            mock_proc.stdout.read.return_value = ""
            mock_proc.stderr.read.return_value = ""
            mock_popen.return_value = mock_proc

            execute_command("echo done", background=True)

            result = list_background_processes()
            assert len(result["processes"]) == 1
            assert "exited" in result["processes"][0]["status"]

    def test_poll_running_process(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.stdout.readline.return_value = ""
            mock_popen.return_value = mock_proc

            start = execute_command("sleep 100", background=True)
            sid = start["session_id"]

            result = process_manage("poll", session_id=sid)
            assert result["session_id"] == sid
            assert result["status"] == "running"

    def test_poll_exited_process(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 42
            mock_proc.returncode = 42
            mock_proc.stdout.read.return_value = "output\n"
            mock_proc.stderr.read.return_value = ""
            mock_popen.return_value = mock_proc

            start = execute_command("exit 42", background=True)
            sid = start["session_id"]

            result = process_manage("poll", session_id=sid)
            assert result["exit_code"] == 42
            assert "exited" in result["status"]

    def test_wait_exited_process(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0
            mock_proc.returncode = 0
            mock_proc.wait.return_value = None
            mock_proc.stdout.read.return_value = "done\n"
            mock_proc.stderr.read.return_value = ""
            mock_popen.return_value = mock_proc

            start = execute_command("echo done", background=True)
            sid = start["session_id"]

            result = process_manage("wait", session_id=sid)
            assert result["status"] == "exited"
            assert result["exit_code"] == 0

    def test_wait_timeout(self):
        import subprocess as sp

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.wait.side_effect = sp.TimeoutExpired("cmd", 5)
            mock_popen.return_value = mock_proc

            start = execute_command("sleep 100", background=True)
            sid = start["session_id"]

            result = process_manage("wait", session_id=sid, timeout=5)
            assert result["status"] == "timeout"

    def test_kill_process(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.wait.return_value = None
            mock_proc.stdout.read.return_value = ""
            mock_proc.stderr.read.return_value = ""
            mock_popen.return_value = mock_proc

            start = execute_command("sleep 100", background=True)
            sid = start["session_id"]

            result = process_manage("kill", session_id=sid)
            mock_proc.kill.assert_called_once()
            assert result["exit_code"] == -9

    def test_log_process(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0
            mock_proc.returncode = 0
            mock_proc.stdout.read.return_value = "line1\nline2\nline3\n"
            mock_proc.stderr.read.return_value = ""
            mock_popen.return_value = mock_proc

            start = execute_command("echo output", background=True)
            sid = start["session_id"]

            # Trigger read of output
            process_manage("poll", session_id=sid)

            result = process_manage("log", session_id=sid)
            assert result["total_lines"] >= 3
            assert "line1" in result["output"]

    def test_log_pagination(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0
            mock_proc.returncode = 0
            mock_proc.stdout.read.return_value = "\n".join(str(i) for i in range(100))
            mock_proc.stderr.read.return_value = ""
            mock_popen.return_value = mock_proc

            start = execute_command("seq 100", background=True)
            sid = start["session_id"]

            process_manage("poll", session_id=sid)

            result = process_manage("log", session_id=sid, offset=0, limit=10)
            assert result["offset"] == 0
            assert result["limit"] == 10

    def test_write_to_process(self):
        import subprocess as sp

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.stdin.write.return_value = 5
            mock_proc.stdin.flush.return_value = None
            mock_popen.return_value = mock_proc

            start = execute_command("cat", background=True)
            sid = start["session_id"]

            result = process_manage("write", session_id=sid, data="hello")
            assert result["message"] is not None
            mock_proc.stdin.write.assert_called_with("hello")

    def test_submit_to_process(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.stdin.write.return_value = 6
            mock_proc.stdin.flush.return_value = None
            mock_popen.return_value = mock_proc

            start = execute_command("cat", background=True)
            sid = start["session_id"]

            result = process_manage("submit", session_id=sid, data="hello")
            assert result["message"] is not None
            mock_proc.stdin.write.assert_called_with("hello\n")

    def test_close_process(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.stdin.close.return_value = None
            mock_popen.return_value = mock_proc

            start = execute_command("cat", background=True)
            sid = start["session_id"]

            result = process_manage("close", session_id=sid)
            assert result["message"] is not None
            mock_proc.stdin.close.assert_called_once()

    def test_process_manage_missing_session(self):
        with pytest.raises(KeyError, match="not found"):
            process_manage("poll", session_id="nonexistent")

    def test_process_manage_missing_session_id(self):
        with pytest.raises(ValueError, match="requires session_id"):
            process_manage("poll")

    def test_process_manage_invalid_action(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            start = execute_command("sleep 1", background=True)
            sid = start["session_id"]

            with pytest.raises(ValueError, match="Unknown action"):
                process_manage("invalid_action", session_id=sid)

    def test_get_process_output(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0
            mock_proc.returncode = 0
            mock_proc.stdout.read.return_value = "done\n"
            mock_proc.stderr.read.return_value = ""
            mock_popen.return_value = mock_proc

            start = execute_command("echo done", background=True)
            sid = start["session_id"]

            result = get_process_output(sid)
            assert result["session_id"] == sid

    def test_get_process_output_missing(self):
        with pytest.raises(KeyError, match="not found"):
            get_process_output("nonexistent")

    def test_write_to_exited_process(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0
            mock_proc.returncode = 0
            mock_proc.stdout.read.return_value = ""
            mock_proc.stderr.read.return_value = ""
            mock_popen.return_value = mock_proc

            start = execute_command("echo done", background=True)
            sid = start["session_id"]

            # Poll first to mark as exited
            process_manage("poll", session_id=sid)

            result = process_manage("write", session_id=sid, data="hello")
            assert "error" in result
            assert "exited" in result["error"]

    def test_write_no_data(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            start = execute_command("cat", background=True)
            sid = start["session_id"]

            result = process_manage("write", session_id=sid)
            assert "error" in result

    def test_clear_background_processes(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            execute_command("sleep 100", background=True)
            clear_background_processes()

            result = list_background_processes()
            assert result["processes"] == []

    def test_foreground_mode_default(self):
        mock_result = MagicMock()
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch("subprocess.Popen") as mock_popen:
                execute_command("echo ok")
                mock_run.assert_called_once()
                mock_popen.assert_not_called()

    def test_background_mode_uses_popen(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            with patch("subprocess.run") as mock_run:
                execute_command("sleep 100", background=True)
                mock_popen.assert_called_once()
                mock_run.assert_not_called()
