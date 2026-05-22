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

    Auto-configures a local author identity if none is set, so commits
    work in isolated environments (CI, temp dirs, etc.) without requiring
    a global git config.

    Args:
        path: Repository directory.
        message: Commit message.

    Returns:
        Git command output.
    """
    # Ensure a local author identity exists
    _ensure_git_identity(path)
    return _run_git(path, ["commit", "-m", message])


def _ensure_git_identity(path: str) -> None:
    """Ensure the repo has a local user.name and user.email configured.

    Only sets them locally (repo-level) if they are not already defined
    globally or in the repo.
    """
    def _get_var(var: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", path, "config", var],
            capture_output=True, text=True,
        )
        return result.stdout.strip() or None

    if not _get_var("user.name"):
        subprocess.run(
            ["git", "-C", path, "config", "user.name", "Local Agent"],
            capture_output=True, text=True,
        )

    if not _get_var("user.email"):
        subprocess.run(
            ["git", "-C", path, "config", "user.email", "agent@local"],
            capture_output=True, text=True,
        )


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
