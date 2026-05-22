"""Search tools — ripgrep-backed file search and directory listing.

Provides codebase-wide content search with regex and directory tree
listing with depth limits. Wraps ripgrep (rg) for fast, .gitignore-aware
searches. Falls back to Python os.walk if ripgrep is not installed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

# Directories to always skip during traversal
SKIP_DIRS: set[str] = {
    "node_modules", "__pycache__", ".git", ".venv", "venv",
    ".pytest_cache", ".mypy_cache", ".eggs", "dist", "build",
}


def search_files(
    pattern: str,
    path: str = ".",
    file_glob: str | None = None,
    search_content: bool = True,
    limit: int = 50,
) -> list[str]:
    """Search for files or content using ripgrep.

    Args:
        pattern: Regex pattern to search for.
        path: Directory or file to search in (default: current directory).
        file_glob: Glob pattern to filter files (e.g., "*.py").
        search_content: If True, search file contents. If False, match filenames.
        limit: Maximum number of results to return.

    Returns:
        List of matched "filename:line_number:content" strings for content search,
        or file paths for filename search.

    Raises:
        ValueError: If ripgrep is not found and fallback fails.
    """
    rg = shutil.which("rg")
    if rg and search_content:
        return _search_with_ripgrep(pattern, path, file_glob, limit)
    if rg and not search_content:
        return _search_filenames_ripgrep(pattern, path, file_glob, limit)

    # Fallback to Python-based search
    if search_content:
        return _search_content_python(pattern, path, file_glob, limit)
    return _search_filenames_python(pattern, path, file_glob, limit)


def list_directory(
    path: str = ".",
    max_depth: int = 3,
    show_hidden: bool = False,
    file_glob: str | None = None,
) -> list[str]:
    """List directory contents as a tree.

    Args:
        path: Directory to list.
        max_depth: Maximum recursion depth (default: 3).
        show_hidden: Include dotfiles and hidden directories.
        file_glob: If set, only list files matching this glob pattern.

    Returns:
        List of file/directory paths relative to the given path.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {path}")

    results: list[str] = []
    _walk_directory(root, root, max_depth, show_hidden, file_glob, results)
    results.sort()
    return results


def _search_with_ripgrep(
    pattern: str, path: str, file_glob: str | None, limit: int
) -> list[str]:
    """Search file contents using ripgrep, returning path:line:content."""
    cmd: list[str] = [
        "rg", "--no-heading", "-n", "--color", "never",
        "--glob", "!.git",
    ]

    if file_glob:
        cmd.extend(["-g", file_glob])

    cmd.extend([pattern, path])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        # Filter out results from hidden/skip dirs
        filtered: list[str] = []
        for line in lines:
            file_part = line.split(":")[0] if ":" in line else line
            if not _path_has_skip_component(file_part):
                filtered.append(line)
        return filtered[:limit]
    except subprocess.TimeoutExpired:
        return []
    except FileNotFoundError:
        return _search_content_python(pattern, path, file_glob, limit)


def _search_filenames_ripgrep(
    pattern: str, path: str, file_glob: str | None, limit: int
) -> list[str]:
    """Search filenames using ripgrep."""
    cmd: list[str] = ["rg", "--files", "--color", "never"]

    if file_glob:
        cmd.extend(["-g", file_glob])

    cmd.append(path)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        # Filter by regex pattern
        try:
            regex = re.compile(pattern)
            files = [f for f in files if regex.search(os.path.basename(f))]
        except re.error:
            pass
        # Filter out skip dirs
        files = [f for f in files if not _path_has_skip_component(f)]
        return files[:limit]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return _search_filenames_python(pattern, path, file_glob, limit)


def _path_has_skip_component(filepath: str) -> bool:
    """Check if a file path contains any skip directory component."""
    parts = Path(filepath).parts
    return any(p in SKIP_DIRS or (p.startswith(".") and p not in {".", ".."}) for p in parts)


def _search_content_python(
    pattern: str, path: str, file_glob: str | None, limit: int
) -> list[str]:
    """Fallback: search file contents using Python os.walk."""
    import fnmatch

    root = Path(path).resolve()
    if not root.is_dir():
        return []

    try:
        regex = re.compile(pattern)
    except re.error:
        return []

    results: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs and common non-code dirs
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d not in SKIP_DIRS
        ]

        for filename in filenames:
            if file_glob and not fnmatch.fnmatch(filename, file_glob):
                continue

            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(f"{filepath}:{line_num}:{line.strip()}")
                            if len(results) >= limit:
                                return results
            except (OSError, UnicodeDecodeError):
                continue

    return results


def _search_filenames_python(
    pattern: str, path: str, file_glob: str | None, limit: int
) -> list[str]:
    """Fallback: search filenames using Python os.walk."""
    import fnmatch

    root = Path(path).resolve()
    if not root.is_dir():
        return []

    try:
        regex = re.compile(pattern)
    except re.error:
        regex = None

    results: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d not in SKIP_DIRS
        ]

        for filename in filenames:
            if file_glob and not fnmatch.fnmatch(filename, file_glob):
                continue

            filepath = os.path.join(dirpath, filename)
            if regex and regex.search(filename):
                results.append(filepath)
            elif not regex:
                results.append(filepath)

            if len(results) >= limit:
                return results

    return results


def _walk_directory(
    root: Path,
    current: Path,
    max_depth: int,
    show_hidden: bool,
    file_glob: str | None,
    results: list[str],
) -> None:
    """Recursively walk a directory and collect file paths."""
    import fnmatch

    # Depth is the number of components from root to current
    current_depth = len(current.relative_to(root).parts)

    # List entries at current depth
    try:
        entries = sorted(current.iterdir())
    except PermissionError:
        return

    for entry in entries:
        name = entry.name

        # Skip hidden unless explicitly requested
        if not show_hidden and name.startswith("."):
            continue

        rel_path = str(entry.relative_to(root))

        if entry.is_file():
            if file_glob is None or fnmatch.fnmatch(name, file_glob):
                results.append(rel_path)
        elif entry.is_dir():
            # Add directory entry
            results.append(rel_path + "/")
            # Only recurse if we haven't exceeded max_depth
            # child_depth = current_depth + 1 (entry is one level deeper)
            if current_depth + 1 < max_depth:
                _walk_directory(root, entry, max_depth, show_hidden, file_glob, results)
