"""File tools — read, write, and patch files on disk."""

from pathlib import Path


def read_file(path: str, offset: int = 1, limit: int = 500) -> str:
    """Read a text file with optional line offset and limit.

    Args:
        path: Path to the file.
        offset: 1-based starting line number.
        limit: Maximum number of lines to read.

    Returns:
        File content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = p.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    start = max(0, offset - 1)
    end = start + limit
    selected = lines[start:end]
    return "".join(selected)


def write_file(path: str, content: str) -> None:
    """Write content to a file, creating parent directories if needed.

    Args:
        path: Destination file path.
        content: Text content to write.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def patch_file(path: str, old_string: str, new_string: str) -> bool:
    """Find-and-replace in a file.

    Args:
        path: File to patch.
        old_string: Text to find and replace.
        new_string: Replacement text.

    Returns:
        True if the replacement was made, False if old_string was not found.
    """
    if not old_string:
        return False

    p = Path(path)
    if not p.exists():
        return False

    content = p.read_text(encoding="utf-8")
    if old_string not in content:
        return False

    new_content = content.replace(old_string, new_string, 1)
    p.write_text(new_content, encoding="utf-8")
    return True
