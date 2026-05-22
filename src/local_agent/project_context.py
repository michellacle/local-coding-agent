"""Project Context — auto-load project-specific context files for the agent.

Detects and loads files like AGENTS.md, CLAUDE.md, .cursorrules, SKILL.md,
pyproject.toml, package.json, etc. into the system prompt so the agent
understands the project conventions without being told.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Well-known context files and their priority/order
CONTEXT_FILES: list[tuple[str, int]] = [
    ("AGENTS.md", 1),
    ("CLAUDE.md", 2),
    (".cursorrules", 3),
    (".cursor/rules", 4),
    ("SKILL.md", 5),
    ("README.md", 10),
    ("pyproject.toml", 15),
    ("setup.cfg", 16),
    ("package.json", 17),
    ("Cargo.toml", 18),
    ("go.mod", 19),
    ("requirements.txt", 20),
    ("Pipfile", 21),
    ("Gemfile", 22),
]


@dataclass
class ContextFile:
    """A loaded project context file."""

    name: str
    path: Path
    content: str
    size: int
    loaded_at: float = field(default_factory=time.time)
    priority: int = 0

    def truncated_content(self, max_chars: int = 4000) -> str:
        """Get content, truncated if too large."""
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + f"\n\n... [truncated: {len(self.content)} total chars]"


class ProjectContext:
    """Load and manage project-specific context files.

    Scans a directory for well-known context files and loads them.
    Files are sorted by priority and can be injected into system prompts.

    Usage:
        ctx = ProjectContext("/path/to/project")
        ctx.scan()
        for cf in ctx.files:
            print(cf.name, cf.size)
        prompt_block = ctx.system_prompt_block()
    """

    def __init__(
        self,
        project_dir: str | Path,
        max_file_size: int = 64 * 1024,  # 64KB per file
        max_prompt_chars: int = 16_000,  # Total chars in prompt
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.max_file_size = max_file_size
        self.max_prompt_chars = max_prompt_chars
        self.files: list[ContextFile] = []
        self._last_scan: float = 0

    def scan(self) -> list[ContextFile]:
        """Scan the project directory for context files.

        Returns:
            List of loaded ContextFile objects.
        """
        self.files = []
        seen: set[str] = set()

        for name, priority in CONTEXT_FILES:
            if name in seen:
                continue

            # Check direct child
            path = self.project_dir / name
            if path.is_file():
                cf = self._load_file(path, name, priority)
                if cf:
                    self.files.append(cf)
                    seen.add(name)

            # Also check .agent/ subdirectory
            agent_dir = self.project_dir / ".agent"
            agent_path = agent_dir / name
            if agent_path.is_file():
                cf = self._load_file(agent_path, f".agent/{name}", priority)
                if cf:
                    self.files.append(cf)
                    seen.add(f".agent/{name}")

        # Look for in-repo SKILL.md files
        skills_dir = self.project_dir / ".skills"
        if skills_dir.is_dir():
            for skill_file in sorted(skills_dir.glob("*.md")):
                name = f".skills/{skill_file.name}"
                cf = self._load_file(skill_file, name, 50)
                if cf:
                    self.files.append(cf)
                    seen.add(name)

        self._last_scan = time.time()

        # Sort by priority
        self.files.sort(key=lambda x: x.priority)

        return self.files

    def system_prompt_block(self, max_chars: int | None = None) -> str:
        """Generate a prompt block with all context files.

        Args:
            max_chars: Maximum total characters. Defaults to self.max_prompt_chars.

        Returns:
            Formatted string suitable for inclusion in a system prompt.
        """
        limit = max_chars if max_chars is not None else self.max_prompt_chars
        if not self.files:
            return ""

        parts: list[str] = ["\n## Project Context\n"]
        remaining = limit - len(parts[0])

        for cf in self.files:
            header = f"\n### {cf.name}\n\n"
            content = cf.truncated_content(max_chars=max(500, remaining - len(header)))

            if len(content) + len(header) > remaining:
                # Include at least a header with note
                parts.append(f"\n### {cf.name}\n[dim][truncated to fit context window[/dim]\n")
                break

            parts.append(header + content)
            remaining -= len(content) + len(header)

        return "".join(parts)

    def get_context(self, name: str) -> ContextFile | None:
        """Get a specific context file by name.

        Args:
            name: Name of the context file.

        Returns:
            ContextFile if found, None otherwise.
        """
        for cf in self.files:
            if cf.name == name or cf.name.endswith("/" + name):
                return cf
        return None

    def needs_rescan(self, interval: float = 60.0) -> bool:
        """Check if a rescan is needed based on time interval.

        Args:
            interval: Seconds between rescans.

        Returns:
            True if a rescan is needed.
        """
        return (time.time() - self._last_scan) > interval

    def total_size(self) -> int:
        """Get total size of all loaded context files."""
        return sum(cf.size for cf in self.files)

    def _load_file(self, path: Path, name: str, priority: int) -> ContextFile | None:
        """Load a single context file.

        Args:
            path: Path to the file.
            name: Display name.
            priority: Sort priority.

        Returns:
            ContextFile or None if loading failed.
        """
        try:
            if path.stat().st_size > self.max_file_size:
                logger.debug("Skipping large context file: %s", path)
                return None

            content = path.read_text(encoding="utf-8", errors="replace")
            return ContextFile(
                name=name,
                path=path,
                content=content,
                size=len(content),
                priority=priority,
            )
        except Exception as e:
            logger.debug("Failed to load context file %s: %s", path, e)
            return None
