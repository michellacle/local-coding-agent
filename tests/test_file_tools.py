"""Tests for file_tools — read_file, write_file, patch_file."""

import os
import tempfile
from pathlib import Path

import pytest

from local_agent.tools.file_tools import read_file, write_file, patch_file


class TestReadFile:
    """Test read_file function."""

    def test_returns_content(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3")
        result = read_file(str(f))
        assert result == "line1\nline2\nline3"

    def test_offset_limits_lines(self, tmp_path: Path):
        f = tmp_path / "nums.txt"
        f.write_text("1\n2\n3\n4\n5")
        # offset=2 (skip first line), limit=2
        result = read_file(str(f), offset=2, limit=2)
        assert "2" in result
        assert "3" in result
        assert "1" not in result
        assert "4" not in result

    def test_offset_defaults_to_1(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("first\nsecond")
        result = read_file(str(f))
        assert result.startswith("first")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_file("/nonexistent/path/file.txt")

    def test_limit_respects_max(self, tmp_path: Path):
        f = tmp_path / "big.txt"
        f.write_text("\n".join(str(i) for i in range(100)))
        result = read_file(str(f), limit=5)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) <= 5


class TestWriteFile:
    """Test write_file function."""

    def test_creates_file(self, tmp_path: Path):
        f = tmp_path / "new.txt"
        write_file(str(f), "hello")
        assert f.read_text() == "hello"

    def test_overwrites_existing(self, tmp_path: Path):
        f = tmp_path / "exist.txt"
        f.write_text("old")
        write_file(str(f), "new")
        assert f.read_text() == "new"

    def test_creates_parent_dirs(self, tmp_path: Path):
        f = tmp_path / "a" / "b" / "c" / "deep.txt"
        write_file(str(f), "deep content")
        assert f.read_text() == "deep content"

    def test_empty_content(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        write_file(str(f), "")
        assert f.read_text() == ""


class TestPatchFile:
    """Test patch_file function."""

    def test_replaces_text(self, tmp_path: Path):
        f = tmp_path / "code.py"
        f.write_text("def old():\n    pass\n")
        result = patch_file(str(f), "old", "new")
        assert result is True
        assert f.read_text() == "def new():\n    pass\n"

    def test_returns_false_when_not_found(self, tmp_path: Path):
        f = tmp_path / "code.py"
        f.write_text("hello world")
        result = patch_file(str(f), "nonexistent", "replacement")
        assert result is False
        assert f.read_text() == "hello world"

    def test_multi_line_replacement(self, tmp_path: Path):
        f = tmp_path / "code.py"
        f.write_text("foo\nbar\nbaz\n")
        result = patch_file(str(f), "foo\nbar", "qux\nquux")
        assert result is True
        assert f.read_text() == "qux\nquux\nbaz\n"

    def test_missing_file_returns_false(self):
        result = patch_file("/nonexistent/file.txt", "old", "new")
        assert result is False

    def test_empty_old_string(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("abc")
        result = patch_file(str(f), "", "insert")
        # Empty old_string: should not replace
        assert result is False
