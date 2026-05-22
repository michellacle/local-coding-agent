"""Tests for DiffReview — preview and manage file changes before applying."""

import pytest
from pathlib import Path
from unittest.mock import patch

from local_agent.diff_review import DiffReview, ProposedChange, ChangeStatus


class TestProposedChange:
    """Test ProposedChange dataclass."""

    def test_unified_diff_no_changes(self):
        content = "hello\nworld\n"
        change = ProposedChange(
            file_path="/tmp/test.py",
            old_content=content,
            new_content=content,
        )
        diff = change.unified_diff
        assert diff == ""
        assert change.has_changes is False

    def test_unified_diff_with_changes(self):
        old = "hello\nworld\nfoo\n"
        new = "hello\nearth\nfoo\nbar\n"
        change = ProposedChange(
            file_path="/tmp/test.py",
            old_content=old,
            new_content=new,
        )
        diff = change.unified_diff
        assert "-world" in diff
        assert "+earth" in diff
        assert "+bar" in diff
        assert change.has_changes is True

    def test_num_added(self):
        old = "a\nb\n"
        new = "a\nb\nc\nd\n"
        change = ProposedChange(file_path="/tmp/t.py", old_content=old, new_content=new)
        assert change.num_added == 2

    def test_num_removed(self):
        old = "a\nb\nc\nd\n"
        new = "a\nb\n"
        change = ProposedChange(file_path="/tmp/t.py", old_content=old, new_content=new)
        assert change.num_removed == 2

    def test_summary(self):
        change = ProposedChange(
            file_path="/tmp/test.py",
            old_content="old\n",
            new_content="new\nextra\n",
            reason="Fix bug",
            status=ChangeStatus.PENDING,
        )
        s = change.summary()
        assert "test.py" in s
        assert "Fix bug" in s
        assert "+2" in s

    def test_apply_approved(self, tmp_path):
        file_path = str(tmp_path / "test.txt")
        Path(file_path).write_text("original")
        change = ProposedChange(
            file_path=file_path,
            old_content="original",
            new_content="updated",
            status=ChangeStatus.APPROVED,
        )
        change.apply()
        assert Path(file_path).read_text() == "updated"

    def test_apply_rejected_raises(self):
        change = ProposedChange(
            file_path="/tmp/test.py",
            old_content="old",
            new_content="new",
            status=ChangeStatus.REJECTED,
        )
        with pytest.raises(RuntimeError, match="rejected"):
            change.apply()

    def test_revert(self, tmp_path):
        file_path = str(tmp_path / "test.txt")
        Path(file_path).write_text("current")
        change = ProposedChange(
            file_path=file_path,
            old_content="original",
            new_content="current",
            status=ChangeStatus.APPROVED,
        )
        change.revert()
        assert Path(file_path).read_text() == "original"
        assert change.status == ChangeStatus.REJECTED


class TestDiffReview:
    """Test DiffReview change manager."""

    def test_propose_change(self):
        review = DiffReview()
        review.propose("/tmp/test.py", "old", "new", "Fix it")
        assert len(review.changes) == 1
        assert review.changes[0].reason == "Fix it"

    def test_propose_replaces_existing(self):
        review = DiffReview()
        review.propose("/tmp/test.py", "v1", "v2", "First")
        review.propose("/tmp/test.py", "v2", "v3", "Second")
        assert len(review.changes) == 1
        assert review.changes[0].reason == "Second"

    def test_list_changes(self):
        review = DiffReview()
        review.propose("/tmp/a.py", "old", "new")
        review.propose("/tmp/b.py", "old", "new")
        assert len(review.list_changes()) == 2

    def test_approve(self):
        review = DiffReview()
        review.propose("/tmp/a.py", "old", "new")
        result = review.approve("/tmp/a.py")
        assert result is not None
        assert result.status == ChangeStatus.APPROVED

    def test_approve_nonexistent(self):
        review = DiffReview()
        result = review.approve("/tmp/nonexistent.py")
        assert result is None

    def test_reject(self):
        review = DiffReview()
        review.propose("/tmp/a.py", "old", "new")
        result = review.reject("/tmp/a.py")
        assert result.status == ChangeStatus.REJECTED

    def test_approve_all(self):
        review = DiffReview()
        review.propose("/tmp/a.py", "old", "new")
        review.propose("/tmp/b.py", "old", "new")
        review.propose("/tmp/c.py", "old", "new")
        count = review.approve_all()
        assert count == 3
        assert review.pending_count() == 0

    def test_reject_all(self):
        review = DiffReview()
        review.propose("/tmp/a.py", "old", "new")
        review.propose("/tmp/b.py", "old", "new")
        count = review.reject_all()
        assert count == 2

    def test_apply_all(self, tmp_path):
        review = DiffReview()
        f1 = str(tmp_path / "a.txt")
        f2 = str(tmp_path / "b.txt")
        Path(f1).write_text("original_a")
        Path(f2).write_text("original_b")
        review.propose(f1, "original_a", "updated_a")
        review.propose(f2, "original_b", "updated_b")
        review.approve_all()
        applied = review.apply_all()
        assert len(applied) == 2
        assert Path(f1).read_text() == "updated_a"
        assert Path(f2).read_text() == "updated_b"

    def test_apply_all_skips_rejected(self, tmp_path):
        review = DiffReview()
        f1 = str(tmp_path / "a.txt")
        Path(f1).write_text("original")
        review.propose(f1, "original", "updated")
        review.reject(f1)
        applied = review.apply_all()
        assert len(applied) == 0
        assert Path(f1).read_text() == "original"

    def test_format_review(self):
        review = DiffReview()
        review.propose("/tmp/a.py", "old", "new", "Fix bug")
        formatted = review.format_review()
        assert "pending" in formatted.lower()
        assert "a.py" in formatted

    def test_format_review_empty(self):
        review = DiffReview()
        formatted = review.format_review()
        assert "No pending" in formatted

    def test_clear_completed(self):
        review = DiffReview()
        review.propose("/tmp/a.py", "old", "new")
        review.propose("/tmp/b.py", "old", "new")
        review.approve("/tmp/a.py")
        review.reject("/tmp/b.py")
        cleared = review.clear_completed()
        assert cleared == 2
        assert len(review.changes) == 0

    def test_find_change(self):
        review = DiffReview()
        review.propose("/tmp/a.py", "old", "new")
        found = review.find_change("/tmp/a.py")
        assert found is not None
        assert review.find_change("/tmp/missing.py") is None

    def test_pending_count(self):
        review = DiffReview()
        review.propose("/tmp/a.py", "old", "new")
        review.propose("/tmp/b.py", "old", "new")
        review.approve("/tmp/a.py")
        assert review.pending_count() == 1
        assert review.approved_count() == 1
