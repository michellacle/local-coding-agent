"""Tests for git_tools — git_init, git_add, git_commit, git_status, git_diff, branch, push, log, merge."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_agent.tools.git_tools import (
    git_init,
    git_add,
    git_commit,
    git_status,
    git_diff,
    git_branch,
    git_checkout,
    git_push,
    git_log,
    git_merge,
    git_remote,
    git_fetch,
    git_stash,
    git_stash_pop,
    git_tag,
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

    def test_commit_with_message(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "my message")

        result = git_log(str(tmp_path))
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
        git_commit(str(tmp_path), "v1")

        (tmp_path / "f.txt").write_text("v2")
        result = git_diff(str(tmp_path))
        assert "v2" in result


class TestGitBranch:
    """Test git_branch function."""

    def test_create_branch(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "init")

        result = git_branch(str(tmp_path), "feature-1")
        # checkout -b outputs to stderr, so check we get no error

    def test_delete_branch(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "init")

        git_branch(str(tmp_path), "to-delete")
        # Must checkout master before deleting "to-delete" if on it
        # Since we just created it, we're on it
        # Use git checkout directly via subprocess
        import subprocess as sp
        sp.run(["git", "-C", str(tmp_path), "checkout", "master"],
               capture_output=True, text=True)
        git_branch(str(tmp_path), "to-delete", delete=True)


class TestGitCheckout:
    """Test git_checkout function."""

    def test_checkout_existing_branch(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "init")

        git_branch(str(tmp_path), "other")
        git_checkout(str(tmp_path), "master")

    def test_checkout_nonexistent_raises(self, tmp_path: Path):
        git_init(str(tmp_path))
        with pytest.raises(RuntimeError):
            git_checkout(str(tmp_path), "does-not-exist")


class TestGitPush:
    """Test git_push function."""

    def test_push_raises_no_remote(self, tmp_path: Path):
        git_init(str(tmp_path))
        with pytest.raises(RuntimeError):
            git_push(str(tmp_path))

    def test_push_with_remote_and_branch(self, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "To fake-remote\n * [new branch] main -> main\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = git_push(str(tmp_path), remote="fake-remote", branch="main")
            assert isinstance(result, str)


class TestGitLog:
    """Test git_log function."""

    def test_log_returns_history(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "first commit")

        result = git_log(str(tmp_path))
        assert "first commit" in result

    def test_log_oneline(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "oneline test")

        result = git_log(str(tmp_path), oneline=True)
        # oneline format has hash prefix
        assert "oneline test" in result

    def test_log_not_oneline(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "full log test")

        result = git_log(str(tmp_path), oneline=False)
        assert "full log test" in result

    def test_log_count(self, tmp_path: Path):
        git_init(str(tmp_path))
        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text(str(i))
            git_add(str(tmp_path), ["."])
            git_commit(str(tmp_path), f"commit {i}")

        result = git_log(str(tmp_path), count=3)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        assert len(lines) == 3

    def test_log_all_branches(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "root")

        git_branch(str(tmp_path), "other")
        (tmp_path / "g.txt").write_text("y")
        git_add(str(tmp_path), ["g.txt"])
        git_commit(str(tmp_path), "on other")

        result = git_log(str(tmp_path), all_branches=True)
        assert "root" in result


class TestGitMerge:
    """Test git_merge function."""

    def test_merge_fast_forward(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "init")

        git_branch(str(tmp_path), "feature")
        (tmp_path / "g.txt").write_text("y")
        git_add(str(tmp_path), ["g.txt"])
        git_commit(str(tmp_path), "feature work")

        git_checkout(str(tmp_path), "master")
        result = git_merge(str(tmp_path), "feature")
        assert isinstance(result, str)

    def test_merge_no_ff(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "a.txt").write_text("a")
        git_add(str(tmp_path), ["a.txt"])
        git_commit(str(tmp_path), "master commit")

        git_branch(str(tmp_path), "feature")
        (tmp_path / "b.txt").write_text("b")
        git_add(str(tmp_path), ["b.txt"])
        git_commit(str(tmp_path), "feature commit")

        git_checkout(str(tmp_path), "master")
        (tmp_path / "c.txt").write_text("c")
        git_add(str(tmp_path), ["c.txt"])
        git_commit(str(tmp_path), "another master commit")

        result = git_merge(str(tmp_path), "feature", no_fast_forward=True)
        assert isinstance(result, str)

    def test_merge_conflict(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "shared.txt").write_text("base\n")
        git_add(str(tmp_path), ["shared.txt"])
        git_commit(str(tmp_path), "base")

        # Create diverging changes
        git_branch(str(tmp_path), "feature")
        (tmp_path / "shared.txt").write_text("feature change\n")
        git_add(str(tmp_path), ["shared.txt"])
        git_commit(str(tmp_path), "feature edit")

        git_checkout(str(tmp_path), "master")
        (tmp_path / "shared.txt").write_text("master change\n")
        git_add(str(tmp_path), ["shared.txt"])
        git_commit(str(tmp_path), "master edit")

        result = git_merge(str(tmp_path), "feature")
        assert isinstance(result, dict)
        assert result["status"] == "conflict"
        assert "conflicted_files" in result
        assert "shared.txt" in result["conflicted_files"]

    def test_merge_error_not_conflict_raises(self, tmp_path: Path):
        """Merge error that isn't a conflict should raise."""
        git_init(str(tmp_path))
        with pytest.raises(RuntimeError, match="git merge failed"):
            git_merge(str(tmp_path), "nonexistent-branch")


class TestGitRemote:
    """Test git_remote function."""

    def test_remote_no_remotes(self, tmp_path: Path):
        git_init(str(tmp_path))
        result = git_remote(str(tmp_path))
        assert result == ""


class TestGitFetch:
    """Test git_fetch function."""

    def test_fetch_no_remote_raises(self, tmp_path: Path):
        git_init(str(tmp_path))
        with pytest.raises(RuntimeError):
            git_fetch(str(tmp_path))


class TestGitStash:
    """Test git_stash function."""

    def test_stash_uncommitted_changes(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "init")

        (tmp_path / "f.txt").write_text("modified")
        result = git_stash(str(tmp_path))
        assert "Saved working directory" in result or "No local changes" in result or result == ""

    def test_stash_with_message(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "init")

        (tmp_path / "f.txt").write_text("modified")
        result = git_stash(str(tmp_path), message="my stash")
        # stash save outputs to stderr; stdout may be empty
        assert isinstance(result, str)

    def test_stash_pop(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "init")

        (tmp_path / "f.txt").write_text("modified")
        git_stash(str(tmp_path))
        result = git_stash_pop(str(tmp_path))
        assert isinstance(result, str)


class TestGitTag:
    """Test git_tag function."""

    def test_create_lightweight_tag(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "init")

        result = git_tag(str(tmp_path), "v1.0.0")
        assert isinstance(result, str)

    def test_create_annotated_tag(self, tmp_path: Path):
        git_init(str(tmp_path))
        (tmp_path / "f.txt").write_text("x")
        git_add(str(tmp_path), ["f.txt"])
        git_commit(str(tmp_path), "init")

        result = git_tag(str(tmp_path), "v2.0.0", message="Release v2")
        assert isinstance(result, str)
