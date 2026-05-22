"""Tests for skill_manage — skill system for procedural memory."""

import tempfile
from pathlib import Path

import pytest

from local_agent.tools.skill_manage import (
    SkillManager,
    SkillInfo,
)


@pytest.fixture
def tmp_skills_dir(tmp_path):
    return tmp_path / "skills"


@pytest.fixture
def mgr(tmp_skills_dir):
    return SkillManager(skills_dir=tmp_skills_dir)


class TestSkillNameValidation:
    """Test skill name validation."""

    def test_empty_name_raises(self, mgr):
        with pytest.raises(ValueError, match="cannot be empty"):
            mgr.create("", "content")

    def test_uppercase_raises(self, mgr):
        with pytest.raises(ValueError, match="lowercase"):
            mgr.create("MySkill", "content")

    def test_spaces_raises(self, mgr):
        with pytest.raises(ValueError, match="lowercase"):
            mgr.create("my skill", "content")

    def test_too_long_raises(self, mgr):
        with pytest.raises(ValueError, match="64 characters"):
            mgr.create("a" * 65, "content")

    def test_valid_names(self, mgr):
        mgr.create("my-skill", "content")
        mgr.create("my_skill", "content")
        mgr.create("skill123", "content")


class TestSkillCreate:
    """Test skill creation."""

    def test_create_skill(self, mgr):
        content = "---\ndescription: My skill\n---\n\nSteps:\n1. Do thing\n"
        info = mgr.create("my-skill", content)
        assert info.name == "my-skill"
        assert info.description == "My skill"

    def test_create_creates_directory(self, mgr, tmp_skills_dir):
        mgr.create("my-skill", "content")
        assert (tmp_skills_dir / "my-skill").is_dir()
        assert (tmp_skills_dir / "my-skill" / "SKILL.md").exists()

    def test_create_duplicate_raises(self, mgr):
        mgr.create("my-skill", "content")
        with pytest.raises(ValueError, match="already exists"):
            mgr.create("my-skill", "content2")

    def test_create_with_category(self, mgr):
        content = "---\ncategory: devops\ndescription: Docker skill\n---\n\nDo stuff\n"
        info = mgr.create("docker-setup", content, category="devops")
        assert info.category == "devops"


class TestSkillUpdate:
    """Test skill updates."""

    def test_update_existing(self, mgr):
        mgr.create("my-skill", "old content")
        info = mgr.update("my-skill", "new content")
        assert info.name == "my-skill"

        skill_file = mgr._skill_dir("my-skill") / "SKILL.md"
        assert skill_file.read_text() == "new content"

    def test_update_nonexistent_raises(self, mgr):
        with pytest.raises(ValueError, match="not found"):
            mgr.update("nonexistent", "content")


class TestSkillDelete:
    """Test skill deletion."""

    def test_delete_existing(self, mgr):
        mgr.create("my-skill", "content")
        mgr.delete("my-skill")
        assert not mgr._skill_dir("my-skill").exists()

    def test_delete_nonexistent_raises(self, mgr):
        with pytest.raises(ValueError, match="not found"):
            mgr.delete("nonexistent")


class TestSkillView:
    """Test viewing skills."""

    def test_view_returns_content(self, mgr):
        mgr.create("my-skill", "hello world")
        result = mgr.view("my-skill")
        assert result["content"] == "hello world"
        assert "linked_files" in result

    def test_view_nonexistent_raises(self, mgr):
        with pytest.raises(ValueError, match="not found"):
            mgr.view("nonexistent")

    def test_view_with_linked_file(self, mgr):
        mgr.create("my-skill", "main content")
        mgr.write_file("my-skill", "references/api.md", "api docs")
        result = mgr.view("my-skill")
        assert "linked_files" in result
        assert "references/api.md" in result["linked_files"]

    def test_view_linked_file(self, mgr):
        mgr.create("my-skill", "main")
        mgr.write_file("my-skill", "templates/config.yaml", "key: value")
        result = mgr.view("my-skill", file_path="templates/config.yaml")
        assert result["content"] == "key: value"

    def test_view_linked_file_not_found(self, mgr):
        mgr.create("my-skill", "main")
        with pytest.raises(ValueError, match="not found"):
            mgr.view("my-skill", file_path="nonexistent.md")


class TestSkillList:
    """Test listing skills."""

    def test_list_empty(self, mgr):
        assert mgr.list_skills() == []

    def test_list_returns_skills(self, mgr):
        mgr.create("skill-a", "---\ndescription: Skill A\n---\ncontent")
        mgr.create("skill-b", "---\ndescription: Skill B\n---\ncontent")
        skills = mgr.list_skills()
        assert len(skills) == 2

    def test_list_filtered_by_category(self, mgr):
        mgr.create("skill-a", "---\ncategory: web\ndescription: A\n---\ncontent")
        mgr.create("skill-b", "---\ncategory: devops\ndescription: B\n---\ncontent")
        skills = mgr.list_skills(category="web")
        assert len(skills) == 1
        assert skills[0].name == "skill-a"


class TestWriteRemoveFile:
    """Test writing and removing supporting files."""

    def test_write_file(self, mgr):
        mgr.create("my-skill", "content")
        path = mgr.write_file("my-skill", "references/api.md", "api stuff")
        assert "api stuff" in Path(path).read_text()

    def test_write_file_invalid_dir_raises(self, mgr):
        mgr.create("my-skill", "content")
        with pytest.raises(ValueError, match="must be under"):
            mgr.write_file("my-skill", "bad/file.txt", "content")

    def test_remove_file(self, mgr):
        mgr.create("my-skill", "content")
        mgr.write_file("my-skill", "references/api.md", "api stuff")
        mgr.remove_file("my-skill", "references/api.md")
        assert not (mgr._skill_dir("my-skill") / "references" / "api.md").exists()

    def test_remove_nonexistent_file_raises(self, mgr):
        mgr.create("my-skill", "content")
        with pytest.raises(ValueError, match="not found"):
            mgr.remove_file("my-skill", "nonexistent.md")


class TestSkillInfo:
    """Test SkillInfo parsing from frontmatter."""

    def test_parse_description(self, mgr):
        mgr.create("my-skill", "---\ndescription: A test skill\n---\ncontent")
        info = mgr.list_skills()[0]
        assert info.description == "A test skill"

    def test_parse_category(self, mgr):
        mgr.create("my-skill", "---\ncategory: tools\ndescription: A skill\n---\ncontent")
        info = mgr.list_skills()[0]
        assert info.category == "tools"

    def test_no_frontmatter(self, mgr):
        mgr.create("my-skill", "Just content, no frontmatter")
        info = mgr.list_skills()[0]
        assert info.description == ""
        assert info.category == ""
