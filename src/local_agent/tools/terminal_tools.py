"""Terminal tools — execute shell commands with foreground and background modes.

Supports:
- Foreground execution (blocking, returns output)
- Background execution (non-blocking, returns session_id)
- Process management: list, poll, wait, kill, log, write, submit, close
"""

from __future__ import annotations

import select
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any

# Global registry for background processes (singleton across the agent session)
_background_processes: dict[str, BackgroundProcess] = {}


@dataclass
class BackgroundProcess:
    """Track a background subprocess."""

    session_id: str
    command: str
    workdir: str | None
    process: subprocess.Popen
    stdout_data: list[str] = field(default_factory=list)
    stderr_data: list[str] = field(default_factory=list)
    exited: bool = False
    exit_code: int | None = None


def execute_command(
    command: str,
    timeout: int = 180,
    workdir: str | None = None,
    background: bool = False,
) -> dict[str, Any]:
    """Execute a shell command and return output and exit code.

    Args:
        command: Shell command string to execute.
        timeout: Maximum seconds to wait (default 180).
        workdir: Working directory for the command (default: None = inherit).
        background: If True, run in background and return session_id immediately.

    Returns:
        Dict with 'output' (str) and 'exit_code' (int) for foreground,
        or 'session_id' (str) and 'message' for background.

    Raises:
        TimeoutError: If the command exceeds the timeout (foreground only).
    """
    if background:
        return _start_background(command, workdir)
    return _run_foreground(command, timeout, workdir)


def _run_foreground(
    command: str,
    timeout: int,
    workdir: str | None,
) -> dict[str, Any]:
    """Run a command in the foreground (blocking)."""
    kwargs: dict[str, Any] = {
        "shell": True,
        "capture_output": True,
        "text": True,
        "timeout": float(timeout),
    }
    if workdir is not None:
        kwargs["cwd"] = workdir

    result = subprocess.run(command, **kwargs)

    output: str = result.stdout
    if result.returncode != 0 and result.stderr:
        output = output.rstrip() + "\nSTDERR:\n" + result.stderr

    return {
        "output": output,
        "exit_code": result.returncode,
    }


def _start_background(
    command: str,
    workdir: str | None,
) -> dict[str, Any]:
    """Start a command in the background and track it."""
    session_id = str(uuid.uuid4())[:8]

    kwargs: dict[str, Any] = {
        "shell": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.PIPE,
        "text": True,
    }
    if workdir is not None:
        kwargs["cwd"] = workdir

    proc = subprocess.Popen(command, **kwargs)  # type: ignore[arg-type]

    bg = BackgroundProcess(
        session_id=session_id,
        command=command,
        workdir=workdir,
        process=proc,
    )
    _background_processes[session_id] = bg

    return {
        "session_id": session_id,
        "message": f"Process started in background: {command}",
    }


def process_manage(
    action: str,
    session_id: str | None = None,
    data: str | None = None,
    timeout: int | None = None,
    offset: int = 0,
    limit: int = 200,
) -> dict[str, Any]:
    """Manage background processes.

    Actions:
        list:    Show all background processes
        poll:    Check status and new output for a process
        wait:    Block until process finishes or timeout
        kill:    Terminate a process
        log:     Get full output with pagination
        write:   Send raw stdin data (no newline)
        submit:  Send data + Enter (for answering prompts)
        close:   Close stdin / send EOF

    Args:
        action: One of the actions above.
        session_id: Process session ID (required for all except 'list').
        data: Text to send to process stdin (for 'write' and 'submit').
        timeout: Max seconds to block for 'wait' action.
        offset: Line offset for 'log' action.
        limit: Max lines to return for 'log' action.

    Returns:
        Dict with action-specific results.

    Raises:
        ValueError: If action is invalid or session_id is missing.
        KeyError: If session_id not found.
    """
    if action == "list":
        return _list_processes()

    if session_id is None:
        raise ValueError(f"Action '{action}' requires session_id")

    if session_id not in _background_processes:
        raise KeyError(f"Process '{session_id}' not found")

    bg = _background_processes[session_id]

    if action == "poll":
        return _poll_process(bg)
    elif action == "wait":
        return _wait_process(bg, timeout if timeout is not None else 300)
    elif action == "kill":
        return _kill_process(bg)
    elif action == "log":
        return _log_process(bg, offset, limit)
    elif action == "write":
        return _write_to_process(bg, data, add_newline=False)
    elif action == "submit":
        return _write_to_process(bg, data, add_newline=True)
    elif action == "close":
        return _close_process(bg)
    else:
        raise ValueError(f"Unknown action: {action}. Valid: list, poll, wait, kill, log, write, submit, close")


def _list_processes() -> dict[str, Any]:
    """List all background processes with their status."""
    if not _background_processes:
        return {"processes": [], "message": "No background processes."}

    processes = []
    for sid, bg in _background_processes.items():
        status = _get_process_status(bg)
        processes.append({
            "session_id": sid,
            "command": bg.command,
            "status": status,
            "exit_code": bg.exit_code,
            "output_lines": len(bg.stdout_data),
        })

    return {"processes": processes}


def _get_process_status(bg: BackgroundProcess) -> str:
    """Get human-readable status string for a process."""
    if bg.exited:
        return f"exited ({bg.exit_code})"
    poll_code = bg.process.poll()
    if poll_code is not None:
        bg.exited = True
        bg.exit_code = poll_code
        # Read remaining output
        _read_remaining_output(bg)
        return f"exited ({poll_code})"
    return "running"


def _read_remaining_output(bg: BackgroundProcess) -> None:
    """Read any remaining output from the process pipes."""
    try:
        proc = bg.process
        if proc.stdout:
            remaining = proc.stdout.read()
            if remaining:
                bg.stdout_data.append(remaining)
        if proc.stderr:
            remaining = proc.stderr.read()
            if remaining:
                bg.stderr_data.append(remaining)
    except Exception:
        pass


def _poll_process(bg: BackgroundProcess) -> dict[str, Any]:
    """Check process status and return new output."""
    status = _get_process_status(bg)

    # Read new output (non-blocking)
    new_output = _read_available_output(bg)

    result: dict[str, Any] = {
        "session_id": bg.session_id,
        "status": status,
        "output": new_output,
    }

    if bg.exit_code is not None:
        result["exit_code"] = bg.exit_code

    return result


def _read_available_output(bg: BackgroundProcess) -> str:
    """Read available output from process without blocking."""
    output_parts: list[str] = []
    proc = bg.process

    try:
        if proc.stdout:
            try:
                readable, _, _ = select.select([proc.stdout], [], [], 0)
                if readable:
                    line = proc.stdout.readline()
                    if line:
                        output_parts.append(line)
                        bg.stdout_data.append(line)
            except (OSError, TypeError):
                pass

        if proc.stderr:
            try:
                readable, _, _ = select.select([proc.stderr], [], [], 0)
                if readable:
                    line = proc.stderr.readline()
                    if line:
                        output_parts.append(line)
                        bg.stderr_data.append(line)
            except (OSError, TypeError):
                pass
    except Exception:
        pass

    return "".join(output_parts)


def _wait_process(bg: BackgroundProcess, timeout: int) -> dict[str, Any]:
    """Wait for process to finish or timeout."""
    if bg.exited:
        return {
            "session_id": bg.session_id,
            "status": "exited",
            "exit_code": bg.exit_code,
            "output": "".join(bg.stdout_data),
        }

    try:
        proc = bg.process
        proc.wait(timeout=timeout)
        bg.exited = True
        bg.exit_code = proc.returncode
        _read_remaining_output(bg)

        return {
            "session_id": bg.session_id,
            "status": "exited",
            "exit_code": bg.exit_code,
            "output": "".join(bg.stdout_data),
        }
    except subprocess.TimeoutExpired:
        return {
            "session_id": bg.session_id,
            "status": "timeout",
            "message": f"Process did not exit within {timeout}s",
            "output": "".join(bg.stdout_data),
        }


def _kill_process(bg: BackgroundProcess) -> dict[str, Any]:
    """Terminate a background process."""
    try:
        bg.process.kill()
        bg.process.wait(timeout=5)
        bg.exited = True
        bg.exit_code = -9
        _read_remaining_output(bg)
        return {
            "session_id": bg.session_id,
            "message": f"Process '{bg.session_id}' terminated",
            "exit_code": -9,
        }
    except Exception as e:
        return {
            "session_id": bg.session_id,
            "error": f"Failed to kill process: {e}",
        }


def _log_process(bg: BackgroundProcess, offset: int, limit: int) -> dict[str, Any]:
    """Get process output log with pagination."""
    all_output = "".join(bg.stdout_data)
    lines = all_output.split("\n")

    # Apply pagination
    total = len(lines)
    start = offset
    end = min(start + limit, total)
    page_lines = lines[start:end]

    return {
        "session_id": bg.session_id,
        "total_lines": total,
        "offset": start,
        "limit": limit,
        "lines": page_lines,
        "output": "\n".join(page_lines),
    }


def _write_to_process(
    bg: BackgroundProcess,
    data: str | None,
    add_newline: bool,
) -> dict[str, Any]:
    """Write data to process stdin."""
    if bg.exited:
        return {
            "session_id": bg.session_id,
            "error": "Process has already exited",
        }

    if data is None:
        return {
            "session_id": bg.session_id,
            "error": "No data provided",
        }

    try:
        proc = bg.process
        if proc.stdin:
            text = data + ("\n" if add_newline else "")
            proc.stdin.write(text)
            proc.stdin.flush()
            return {
                "session_id": bg.session_id,
                "message": f"Wrote {len(text)} bytes to stdin",
            }
        else:
            return {
                "session_id": bg.session_id,
                "error": "Process stdin not available",
            }
    except Exception as e:
        return {
            "session_id": bg.session_id,
            "error": f"Failed to write to stdin: {e}",
        }


def _close_process(bg: BackgroundProcess) -> dict[str, Any]:
    """Close process stdin (send EOF)."""
    try:
        proc = bg.process
        if proc.stdin:
            proc.stdin.close()
            return {
                "session_id": bg.session_id,
                "message": "Stdin closed (EOF sent)",
            }
        else:
            return {
                "session_id": bg.session_id,
                "message": "Stdin already closed or not available",
            }
    except Exception as e:
        return {
            "session_id": bg.session_id,
            "error": f"Failed to close stdin: {e}",
        }


def list_background_processes() -> dict[str, Any]:
    """List all background processes (convenience wrapper)."""
    return _list_processes()


def get_process_output(session_id: str) -> dict[str, Any]:
    """Get output for a specific background process."""
    if session_id not in _background_processes:
        raise KeyError(f"Process '{session_id}' not found")
    return _poll_process(_background_processes[session_id])


def clear_background_processes() -> None:
    """Clear the background process registry (for testing)."""
    for bg in _background_processes.values():
        try:
            if not bg.exited:
                bg.process.kill()
                bg.process.wait(timeout=2)
        except Exception:
            pass
    _background_processes.clear()
