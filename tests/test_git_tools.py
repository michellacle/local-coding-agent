"""Tests for git_tools — git_init, git_add, git_commit, git_status, git_diff."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_agent.tools.git_tools import (
    git_init,
    git_add,
    git_commit,
    git_status,
    git_diff,
)


class TestGitInit:
    """Test git_init function."""

    def test_creates_git_directory(self, tmp_path: Path):
        git_init(str(tmp_path))
        assert (tmp_path / ".git").is_dir()

    def test_already_initialized_is_idempotent(self, tmp_path: Path):
        git_init(str(tmp_path))
        git_init(str(tmp_path))
        assert (tmp_path / ".git").is_dir()


class TestGitAdd:
    """Test git_add function."""

    def test_adds_specific_files(self, tmp_path: Path):
        git_init(str(tmp_path))
        f = tmp_path / "new_file.txt"
        f.write_text("hello")
        git_add(str(tmp_path), ["new_file.txt"])

        # Verify with git status that it's staged
        result = git_status(str(tmp_path))
        assert "new_file.txt" in result

    def test_adds_all_files(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        git_add(str(tmp_path), ["."])

        result = git_status(str(tmp_path))
        assert "a.txt" in result
        assert "b.txt" in result


class TestGitCommit:
    """Test git_commit function."""

    def test_creates_commit(self, tmp_path: Path):
        git_init(str(tmp_path))
        f = tmp_path / "file.txt"
        f.write_text("content")
        git_add(str(tmp_path), ["file.txt"])
        git_commit(str(tmp_path), "initial commit")

        # Config git user for bare repo
        _run_git(tmp_path, ["config", "user.email", "test@test.com"])
        _run_git(tmp_path, ["config", "user.name", "Test"])

    def test_commit_with_message(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        _run_git(tmp_path, ["config", "user.email", "test@test.com"])
        _run_git(tmp_path, ["config", "user.name", "Test"])
        git_commit(str(tmp_path), "my message")

        result = _run_git(tmp_path, ["log", "--oneline"])
        assert "my message" in result


class TestGitStatus:
    """Test git_status function."""

    def test_returns_status_output(self, tmp_path: Path):
        git_init(str(tmp_path))
        result = git_status(str(tmp_path))
        assert "No commits yet" in result or "nothing to commit" in result

    def test_shows_untracked_files(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "untracked.txt").write_text("data")
        result = git_status(str(tmp_path))
        assert "untracked.txt" in result


class TestGitDiff:
    """Test git_diff function."""

    def test_returns_diff_output(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("v1")
        git_add(str(tmp_path), ["f.txt"])
        _run_git(tmp_path, ["config", "user.email", "test@test.com"])
        _run_git(tmp_path, ["config", "user.name", "Test"])
        git_commit(str(tmp_path), "v1")

        (tmp_path / "f.txt").write_text("v2")
        result = git_diff(str(tmp_path))
        assert "v2" in result


def _run_git(path: Path, args: list[str]) -> str:
    """Helper to run git commands for test verification."""
    import subprocess
    r = subprocess.run(
        ["git"] + args,
        cwd=str(path),
        capture_output=True,
        text=True,
    )
    return r.stdout
