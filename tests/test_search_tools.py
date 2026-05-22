"""Tests for search tools — file search and directory listing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from local_agent.tools.search_tools import search_files, list_directory


class TestSearchFiles:
    """Test search_files tool."""

    def test_search_content_by_pattern(self, tmp_path: Path):
        """Search finds files containing a pattern."""
        (tmp_path / "a.py").write_text("def hello():\n    pass\n")
        (tmp_path / "b.py").write_text("def goodbye():\n    pass\n")
        (tmp_path / "c.txt").write_text("nothing here\n")

        results = search_files("hello", str(tmp_path))
        assert len(results) == 1
        assert "a.py" in results[0]

    def test_search_with_file_glob(self, tmp_path: Path):
        """file_glob filters to only matching files."""
        (tmp_path / "a.py").write_text("target\n")
        (tmp_path / "b.py").write_text("target\n")
        (tmp_path / "c.txt").write_text("target\n")

        results = search_files("target", str(tmp_path), file_glob="*.py")
        # Results are "path:line:content" — check the file part ends with .py
        assert len(results) >= 2
        for r in results:
            file_part = r.split(":")[0]
            assert file_part.endswith(".py"), f"Expected .py file, got: {r}"

    def test_search_regex_pattern(self, tmp_path: Path):
        """Regex patterns work for content search."""
        (tmp_path / "a.py").write_text("var_foo = 1\nvar_bar = 2\n")
        (tmp_path / "b.py").write_text("normal text\n")

        results = search_files("var_\\w+", str(tmp_path))
        assert len(results) >= 1
        assert "a.py" in results[0]

    def test_search_returns_line_numbers(self, tmp_path: Path):
        """Results include line numbers."""
        (tmp_path / "test.py").write_text("line one\nline two\nline three\n")

        results = search_files("two", str(tmp_path))
        assert len(results) >= 1
        # Format should be path:line_number:content
        assert ":" in results[0]
        assert "2:" in results[0]

    def test_search_respects_limit(self, tmp_path: Path):
        """limit caps the number of results."""
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text("match\n")

        results = search_files("match", str(tmp_path), limit=3)
        assert len(results) == 3

    def test_search_nonexistent_pattern(self, tmp_path: Path):
        """No results for pattern that doesn't match."""
        (tmp_path / "a.py").write_text("hello world\n")

        results = search_files("xyz123", str(tmp_path))
        assert len(results) == 0

    def test_search_skip_hidden_and_cache_dirs(self, tmp_path: Path):
        """Hidden dirs, __pycache__, .venv are skipped."""
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("target\n")

        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.pyc").write_text("target\n")

        (tmp_path / "visible.py").write_text("target\n")

        results = search_files("target", str(tmp_path))
        assert len(results) >= 1
        assert not any(".hidden" in r for r in results)
        assert not any("__pycache__" in r for r in results)

    def test_search_content_false_matches_filenames(self, tmp_path: Path):
        """When search_content is False, match filenames."""
        (tmp_path / "config.yaml").write_text("foo: bar\n")
        (tmp_path / "settings.yaml").write_text("baz: qux\n")
        (tmp_path / "data.py").write_text("x = 1\n")

        results = search_files("config", str(tmp_path), search_content=False)
        assert len(results) >= 1
        assert any("config" in r for r in results)


class TestListDirectory:
    """Test list_directory tool."""

    def test_list_flat_directory(self, tmp_path: Path):
        """Lists files in a flat directory."""
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "c.txt").write_text("c")

        results = list_directory(str(tmp_path))
        assert "a.py" in results
        assert "b.py" in results
        assert "c.txt" in results

    def test_list_nested_directory(self, tmp_path: Path):
        """Lists files in nested directories."""
        (tmp_path / "root.py").write_text("root")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "child.py").write_text("child")
        (subdir / "child.txt").write_text("child")

        results = list_directory(str(tmp_path))
        assert "root.py" in results
        assert "sub/" in results
        assert any("child.py" in r for r in results)

    def test_list_respects_max_depth(self, tmp_path: Path):
        """max_depth limits recursion."""
        (tmp_path / "root.py").write_text("root")
        l1 = tmp_path / "l1"
        l1.mkdir()
        (l1 / "f1.py").write_text("f1")
        l2 = l1 / "l2"
        l2.mkdir()
        (l2 / "f2.py").write_text("f2")
        l3 = l2 / "l3"
        l3.mkdir()
        (l3 / "f3.py").write_text("f3")

        results = list_directory(str(tmp_path), max_depth=1)
        assert "root.py" in results
        assert "l1/" in results
        # l1/f1.py should NOT be at depth 1
        assert not any("f1.py" in r for r in results)

    def test_list_skips_hidden_files(self, tmp_path: Path):
        """Hidden files and dirs are skipped by default."""
        (tmp_path / "visible.py").write_text("v")
        (tmp_path / ".hidden.py").write_text("h")
        hidden_dir = tmp_path / ".hiddendir"
        hidden_dir.mkdir()
        (hidden_dir / "secret.py").write_text("s")

        results = list_directory(str(tmp_path))
        assert "visible.py" in results
        assert ".hidden.py" not in results
        assert not any(".hiddendir" in r for r in results)

    def test_list_show_hidden(self, tmp_path: Path):
        """show_hidden=True includes hidden files."""
        (tmp_path / "visible.py").write_text("v")
        (tmp_path / ".hidden.py").write_text("h")

        results = list_directory(str(tmp_path), show_hidden=True)
        assert "visible.py" in results
        assert ".hidden.py" in results

    def test_list_file_glob_filter(self, tmp_path: Path):
        """file_glob filters to matching files only."""
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "c.txt").write_text("c")
        (tmp_path / "d.md").write_text("d")

        results = list_directory(str(tmp_path), file_glob="*.py")
        files = [r for r in results if not r.endswith("/")]
        assert all(r.endswith(".py") for r in files)
        assert len(files) == 2

    def test_list_nonexistent_path_raises(self):
        """Nonexistent path raises ValueError."""
        with pytest.raises(ValueError, match="Not a directory"):
            list_directory("/nonexistent/path/12345")

    def test_list_results_are_sorted(self, tmp_path: Path):
        """Results are sorted alphabetically."""
        (tmp_path / "z.py").write_text("z")
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "m.py").write_text("m")

        results = list_directory(str(tmp_path))
        files = [r for r in results if not r.endswith("/")]
        assert files == sorted(files)

    def test_list_nested_with_subdirs(self, tmp_path: Path):
        """Subdirectories are marked with trailing slash."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("main")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_main.py").write_text("test")

        results = list_directory(str(tmp_path))
        assert "src/" in results
        assert "tests/" in results
        assert any("main.py" in r for r in results)
        assert any("test_main.py" in r for r in results)
