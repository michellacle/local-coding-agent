"""Terminal tools — execute shell commands."""

import subprocess


def execute_command(command: str, timeout: int = 180, workdir: str | None = None) -> dict:
    """Execute a shell command and return output and exit code.

    Args:
        command: Shell command string to execute.
        timeout: Maximum seconds to wait (default 180).
        workdir: Working directory for the command (default: None = inherit).

    Returns:
        Dict with 'output' (str) and 'exit_code' (int).
        Stderr is appended to output if exit_code is non-zero.

    Raises:
        TimeoutError: If the command exceeds the timeout.
    """
    kwargs = {
        "shell": True,
        "capture_output": True,
        "text": True,
        "timeout": timeout,
    }
    if workdir is not None:
        kwargs["cwd"] = workdir

    result = subprocess.run(command, **kwargs)

    output = result.stdout
    if result.returncode != 0 and result.stderr:
        output = output.rstrip() + "\nSTDERR:\n" + result.stderr

    return {
        "output": output,
        "exit_code": result.returncode,
    }
