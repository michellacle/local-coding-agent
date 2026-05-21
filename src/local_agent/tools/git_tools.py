"""Git tools — git init, add, commit, status, diff."""

import subprocess


def _run_git(path: str, args: list[str]) -> str:
    """Run a git command in the given path and return stdout."""
    result = subprocess.run(
        ["git", "-C", path] + args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def git_init(path: str) -> str:
    """Initialize a git repository at the given path.

    Args:
        path: Directory to initialize.

    Returns:
        Git command output.
    """
    return _run_git(path, ["init"])


def git_add(path: str, files: list[str]) -> str:
    """Stage files for commit.

    Args:
        path: Repository directory.
        files: File paths or "." to add all.

    Returns:
        Git command output.
    """
    return _run_git(path, ["add"] + files)


def git_commit(path: str, message: str) -> str:
    """Create a commit with the given message.

    Args:
        path: Repository directory.
        message: Commit message.

    Returns:
        Git command output.
    """
    return _run_git(path, ["commit", "-m", message])


def git_status(path: str) -> str:
    """Get the git status of the repository.

    Args:
        path: Repository directory.

    Returns:
        Git status output.
    """
    return _run_git(path, ["status"])


def git_diff(path: str) -> str:
    """Get the diff of unstaged changes.

    Args:
        path: Repository directory.

    Returns:
        Git diff output.
    """
    return _run_git(path, ["diff"])
