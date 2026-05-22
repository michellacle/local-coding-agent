"""Git tools — git init, add, commit, status, diff, branch, push, log, merge."""

from __future__ import annotations

import subprocess
from typing import Any


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


def git_branch(path: str, branch_name: str, delete: bool = False) -> str:
    """Create or checkout a branch.

    Args:
        path: Repository directory.
        branch_name: Name of the branch to create or checkout.
        delete: If True, delete the branch instead of creating/checking out.

    Returns:
        Git command output.
    """
    if delete:
        return _run_git(path, ["branch", "-D", branch_name])
    return _run_git(path, ["checkout", "-b", branch_name])


def git_checkout(path: str, ref: str) -> str:
    """Checkout an existing branch, tag, or commit.

    Args:
        path: Repository directory.
        ref: Branch name, tag, or commit SHA to checkout.

    Returns:
        Git command output.
    """
    return _run_git(path, ["checkout", ref])


def git_push(path: str, remote: str = "origin", branch: str | None = None) -> str:
    """Push commits to a remote repository.

    Args:
        path: Repository directory.
        remote: Remote name (default: origin).
        branch: Branch to push (default: current branch).

    Returns:
        Git command output.
    """
    args = ["push"]
    if branch:
        args.extend([remote, branch])
    else:
        args.append(remote)
    return _run_git(path, args)


def git_log(
    path: str,
    count: int = 10,
    oneline: bool = True,
    all_branches: bool = False,
) -> str:
    """Get commit history.

    Args:
        path: Repository directory.
        count: Number of commits to return (default: 10).
        oneline: Use --oneline format (default: True).
        all_branches: Show commits from all branches (default: False).

    Returns:
        Git log output.
    """
    args = ["log", f"-{count}"]
    if oneline:
        args.append("--oneline")
    if all_branches:
        args.append("--all")
    return _run_git(path, args)


def git_merge(path: str, branch: str, no_fast_forward: bool = False) -> str | dict[str, Any]:
    """Merge a branch into the current branch.

    Args:
        path: Repository directory.
        branch: Branch to merge.
        no_fast_forward: Force a merge commit even if fast-forward is possible.

    Returns:
        Git command output, or a dict with conflict info if conflicts detected.
    """
    args = ["merge"]
    if no_fast_forward:
        args.append("--no-ff")
    args.append(branch)

    result = subprocess.run(
        ["git", "-C", path] + args,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        combined = result.stdout + result.stderr
        # Check for merge conflicts
        if "CONFLICT" in combined or "Automatic merge failed" in combined:
            # Get list of conflicted files
            status = _run_git(path, ["status", "--short"])
            conflicted = [
                line.split()[-1] for line in status.split("\n")
                if line.startswith("UU")
            ]
            return {
                "status": "conflict",
                "message": "Merge conflicts detected",
                "conflicted_files": conflicted,
                "stderr": result.stderr.strip(),
            }
        raise RuntimeError(f"git merge failed: {combined.strip()}")

    return result.stdout


def git_remote(path: str) -> str:
    """List remote repositories.

    Args:
        path: Repository directory.

    Returns:
        Git remote output.
    """
    return _run_git(path, ["remote", "-v"])


def git_fetch(path: str, remote: str = "origin") -> str:
    """Fetch updates from a remote repository.

    Args:
        path: Repository directory.
        remote: Remote name (default: origin).

    Returns:
        Git fetch output.
    """
    return _run_git(path, ["fetch", remote])


def git_stash(path: str, message: str | None = None) -> str:
    """Stash changes in a temporary storage area.

    Args:
        path: Repository directory.
        message: Optional stash message.

    Returns:
        Git stash output.
    """
    args = ["stash"]
    if message:
        args.extend(["save", "-m", message])
    else:
        args.append("save")
    return _run_git(path, args)


def git_stash_pop(path: str) -> str:
    """Apply and remove the most recent stash.

    Args:
        path: Repository directory.

    Returns:
        Git stash pop output.
    """
    return _run_git(path, ["stash", "pop"])


def git_tag(path: str, tag_name: str, message: str | None = None) -> str:
    """Create a tag.

    Args:
        path: Repository directory.
        tag_name: Tag name.
        message: Optional annotated tag message.

    Returns:
        Git tag output.
    """
    if message:
        return _run_git(path, ["tag", "-a", tag_name, "-m", message])
    return _run_git(path, ["tag", tag_name])
