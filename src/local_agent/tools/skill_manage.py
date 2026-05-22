"""Skill management — procedural memory for reusable workflows.

Skills are stored as directories under ~/.local/share/local-coding-agent/skills/
with a SKILL.md file and optional subdirectories (references/, templates/, scripts/).

Supports:
- Create/update/delete skills
- Skill discovery by name or category
- Loading full skill content or linked files
- In-repo SKILL.md for project-specific workflows
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SKILLS_DIR = Path.home() / ".local" / "share" / "local-coding-agent" / "skills"
VALID_FILE_DIRS = {"references", "templates", "scripts", "assets"}


@dataclass
class SkillInfo:
    """Metadata about a skill."""

    name: str
    description: str = ""
    category: str = ""
    has_files: bool = False
    file_count: int = 0


class SkillManager:
    """Manage skills stored on disk."""

    def __init__(self, skills_dir: Path | None = None) -> None:
        if skills_dir is None:
            skills_dir = SKILLS_DIR
        self._skills_dir = skills_dir

    def create(self, name: str, content: str, category: str = "") -> SkillInfo:
        """Create a new skill.

        Args:
            name: Lowercase, hyphens/underscores, max 64 chars.
            content: Full SKILL.md content (YAML frontmatter + markdown).
            category: Optional category for organizing skills.

        Returns:
            SkillInfo for the created skill.
        """
        self._validate_name(name)
        skill_dir = self._skill_dir(name)
        if skill_dir.exists():
            raise ValueError(f"Skill '{name}' already exists")

        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        return self._build_info(name, content)

    def update(self, name: str, content: str) -> SkillInfo:
        """Update an existing skill's SKILL.md.

        Args:
            name: Skill name.
            content: New full SKILL.md content.

        Returns:
            SkillInfo for the updated skill.
        """
        skill_dir = self._skill_dir(name)
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            raise ValueError(f"Skill '{name}' not found")

        skill_file.write_text(content, encoding="utf-8")
        return self._build_info(name, content)

    def delete(self, name: str) -> None:
        """Delete a skill and all its files.

        Args:
            name: Skill name.
        """
        skill_dir = self._skill_dir(name)
        if not skill_dir.exists():
            raise ValueError(f"Skill '{name}' not found")

        import shutil
        shutil.rmtree(skill_dir)

    def view(self, name: str, file_path: str | None = None) -> dict[str, Any]:
        """View a skill's content or a linked file.

        Args:
            name: Skill name.
            file_path: Optional path to a linked file (e.g., 'references/api.md').

        Returns:
            Dict with 'content' and optionally 'linked_files'.
        """
        skill_dir = self._skill_dir(name)
        if not skill_dir.exists():
            raise ValueError(f"Skill '{name}' not found")

        if file_path:
            fpath = skill_dir / file_path
            if not fpath.exists():
                raise ValueError(f"File '{file_path}' not found in skill '{name}'")
            return {"content": fpath.read_text(encoding="utf-8")}

        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            raise ValueError(f"SKILL.md not found in skill '{name}'")

        content = skill_file.read_text(encoding="utf-8")

        # List linked files
        linked_files: dict[str, str] = {}
        for d in VALID_FILE_DIRS:
            dir_path = skill_dir / d
            if dir_path.exists():
                for f in sorted(dir_path.rglob("*")):
                    if f.is_file():
                        rel = str(f.relative_to(skill_dir))
                        linked_files[rel] = f"({f.stat().st_size} bytes)"

        return {"content": content, "linked_files": linked_files}

    def list_skills(self, category: str | None = None) -> list[SkillInfo]:
        """List all skills, optionally filtered by category.

        Args:
            category: If set, only return skills in this category.

        Returns:
            List of SkillInfo objects.
        """
        if not self._skills_dir.exists():
            return []

        skills: list[SkillInfo] = []
        for skill_dir in sorted(self._skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            content = skill_file.read_text(encoding="utf-8")
            info = self._build_info(skill_dir.name, content)
            if category and info.category != category:
                continue
            skills.append(info)
        return skills

    def write_file(self, name: str, file_path: str, content: str) -> str:
        """Write a supporting file within a skill directory.

        Args:
            name: Skill name.
            file_path: Path within the skill (e.g., 'references/api.md').
            content: File content.

        Returns:
            Absolute path of the written file.
        """
        fpath = Path(file_path)
        top_dir = fpath.parts[0] if fpath.parts else ""
        if top_dir not in VALID_FILE_DIRS:
            raise ValueError(
                f"File must be under one of: {', '.join(sorted(VALID_FILE_DIRS))}"
            )

        skill_dir = self._skill_dir(name)
        skill_dir.mkdir(parents=True, exist_ok=True)
        full_path = skill_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return str(full_path)

    def remove_file(self, name: str, file_path: str) -> None:
        """Remove a supporting file from a skill directory.

        Args:
            name: Skill name.
            file_path: Path within the skill.
        """
        skill_dir = self._skill_dir(name)
        full_path = skill_dir / file_path
        if not full_path.exists():
            raise ValueError(f"File '{file_path}' not found in skill '{name}'")
        full_path.unlink()

    def _validate_name(self, name: str) -> None:
        if not name:
            raise ValueError("Skill name cannot be empty")
        if len(name) > 64:
            raise ValueError("Skill name must be 64 characters or less")
        import re
        if not re.match(r'^[a-z0-9_\-]+$', name):
            raise ValueError(
                "Skill name must be lowercase with hyphens/underscores only"
            )

    def _skill_dir(self, name: str) -> Path:
        return self._skills_dir / name

    def _build_info(self, name: str, content: str) -> SkillInfo:
        # Parse YAML frontmatter for description and category
        desc = ""
        category = ""
        has_files = False
        file_count = 0

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                yaml_str = parts[1].strip()
                for line in yaml_str.split("\n"):
                    if ":" in line:
                        key, _, val = line.partition(":")
                        key = key.strip().lower()
                        val = val.strip().strip('"').strip("'")
                        if key == "description":
                            desc = val
                        elif key == "category":
                            category = val

        skill_dir = self._skill_dir(name)
        if skill_dir.exists():
            for d in VALID_FILE_DIRS:
                dir_path = skill_dir / d
                if dir_path.exists():
                    files = list(dir_path.rglob("*"))
                    file_count += sum(1 for f in files if f.is_file())

        has_files = file_count > 0
        return SkillInfo(
            name=name, description=desc, category=category,
            has_files=has_files, file_count=file_count,
        )


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

_manager: SkillManager | None = None


def _get_manager() -> SkillManager:
    global _manager
    if _manager is None:
        _manager = SkillManager()
    return _manager


def skill_manage(action: str, name: str, content: str | None = None,
                 old_string: str | None = None, new_string: str | None = None,
                 replace_all: bool = False, category: str = "",
                 file_path: str | None = None, file_content: str | None = None,
                 absorbed_into: str | None = None) -> str:
    """Manage skills (create, update, delete, patch).

    Args:
        action: One of 'create', 'patch', 'edit', 'delete', 'write_file', 'remove_file'.
        name: Skill name (lowercase, hyphens/underscores).
        content: Full SKILL.md content. Required for 'create' and 'edit'.
        old_string: Text to find (required for 'patch').
        new_string: Replacement text (required for 'patch').
        replace_all: Replace all occurrences (default: false).
        category: Category for organizing skills (create only).
        file_path: Path to a supporting file (write_file/remove_file).
        file_content: Content for write_file.
        absorbed_into: For delete — umbrella skill name or empty string.

    Returns:
        Confirmation message.
    """
    mgr = _get_manager()

    if action == "create":
        if content is None:
            raise ValueError("content is required for action='create'")
        info = mgr.create(name=name, content=content, category=category)
        return f"Created skill '{name}' (category={category or 'none'})."

    elif action == "edit":
        if content is None:
            raise ValueError("content is required for action='edit'")
        info = mgr.update(name=name, content=content)
        return f"Updated skill '{name}'."

    elif action == "patch":
        if old_string is None:
            raise ValueError("old_string is required for action='patch'")
        if new_string is None:
            new_string = ""
        skill_dir = mgr._skill_dir(name)
        target = skill_dir / file_path if file_path else skill_dir / "SKILL.md"
        if not target.exists():
            raise ValueError(f"Target file not found: {target}")
        text = target.read_text(encoding="utf-8")
        if replace_all:
            text = text.replace(old_string, new_string)
        else:
            text = text.replace(old_string, new_string, 1)
        target.write_text(text, encoding="utf-8")
        return f"Patched '{target.name}' in skill '{name}'."

    elif action == "delete":
        mgr.delete(name=name)
        note = f" (absorbed into '{absorbed_into}')" if absorbed_into else ""
        return f"Deleted skill '{name}'{note}."

    elif action == "write_file":
        if file_path is None:
            raise ValueError("file_path is required for action='write_file'")
        if file_content is None:
            raise ValueError("file_content is required for action='write_file'")
        path = mgr.write_file(name=name, file_path=file_path, content=file_content)
        return f"Wrote file to {path}."

    elif action == "remove_file":
        if file_path is None:
            raise ValueError("file_path is required for action='remove_file'")
        mgr.remove_file(name=name, file_path=file_path)
        return f"Removed file '{file_path}' from skill '{name}'."

    else:
        raise ValueError(
            f"Unknown action '{action}'. Use 'create', 'patch', 'edit', "
            f"'delete', 'write_file', or 'remove_file'."
        )


def skill_view(name: str, file_path: str | None = None) -> str:
    """View a skill's content or a linked file.

    Args:
        name: Skill name.
        file_path: Optional path to a linked file.

    Returns:
        Skill content or file content.
    """
    mgr = _get_manager()
    result = mgr.view(name=name, file_path=file_path)
    if "linked_files" in result:
        lines = [result["content"]]
        if result["linked_files"]:
            lines.append("")
            lines.append("Linked files:")
            for fp, sz in result["linked_files"].items():
                lines.append(f"  - {fp} {sz}")
        return "\n".join(lines)
    return result["content"]


def skills_list(category: str | None = None) -> str:
    """List available skills.

    Args:
        category: Optional category filter.

    Returns:
        Formatted list of skills with descriptions.
    """
    mgr = _get_manager()
    skills = mgr.list_skills(category=category)
    if not skills:
        return "No skills found." if not category else f"No skills in category '{category}'."

    lines: list[str] = []
    for s in skills:
        desc = f" — {s.description}" if s.description else ""
        files = f" ({s.file_count} supporting files)" if s.has_files else ""
        cat = f" [{s.category}]" if s.category else ""
        lines.append(f"- {s.name}{cat}: {s.description}{desc}{files}")
    return "\n".join(lines)
