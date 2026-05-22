"""Tests for ProjectContext — auto-load project context files."""

import pytest
from pathlib import Path
import time

from local_agent.project_context import ProjectContext, ContextFile


@pytest.fixture
def project_dir(tmp_path):
    """Create a temp project directory with context files."""
    # Create standard context files
    (tmp_path / "AGENTS.md").write_text("# Agent instructions\nBe helpful.\n")
    (tmp_path / "CLAUDE.md").write_text("# Claude rules\nNo cloud APIs.\n")
    (tmp_path / "README.md").write_text("# My Project\nA test project.\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
    return tmp_path


@pytest.fixture
def rich_project(tmp_path):
    """Create a project with many context files."""
    (tmp_path / "AGENTS.md").write_text("Agent rules.\n")
    (tmp_path / "CLAUDE.md").write_text("Claude rules.\n")
    (tmp_path / "README.md").write_text("# README\nDetails here.\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
    (tmp_path / "package.json").write_text('{"name": "test"}\n')
    (tmp_path / "requirements.txt").write_text("flask\nrequests\n")
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    (agent_dir / "AGENTS.md").write_text("Agent subdir rules.\n")
    skills_dir = tmp_path / ".skills"
    skills_dir.mkdir()
    (skills_dir / "deploy.md").write_text("# Deploy skill\nSteps...\n")
    return tmp_path


class TestContextFile:
    """Test ContextFile dataclass."""

    def test_init(self):
        cf = ContextFile(name="test.md", path=Path("/tmp/t.md"), content="hello", size=5, priority=1)
        assert cf.name == "test.md"
        assert cf.size == 5

    def test_truncated_content_short(self):
        cf = ContextFile(name="t.md", path=Path("/tmp/t.md"), content="short", size=5)
        assert cf.truncated_content(max_chars=100) == "short"

    def test_truncated_content_long(self):
        long_content = "x" * 5000
        cf = ContextFile(name="t.md", path=Path("/tmp/t.md"), content=long_content, size=5000)
        truncated = cf.truncated_content(max_chars=1000)
        assert len(truncated) <= 1000 + 50  # allow for truncation notice
        assert "truncated" in truncated


class TestProjectContextScan:
    """Test scanning for context files."""

    def test_scan_finds_files(self, project_dir):
        ctx = ProjectContext(project_dir)
        files = ctx.scan()
        assert len(files) >= 3  # AGENTS.md, CLAUDE.md, README.md

    def test_scan_finds_pyproject(self, project_dir):
        ctx = ProjectContext(project_dir)
        files = ctx.scan()
        names = [f.name for f in files]
        assert "pyproject.toml" in names

    def test_scan_empty_dir(self, tmp_path):
        ctx = ProjectContext(tmp_path)
        files = ctx.scan()
        assert files == []

    def test_scan_priority_order(self, project_dir):
        ctx = ProjectContext(project_dir)
        files = ctx.scan()
        # AGENTS.md should come before README.md
        agent_idx = next((i for i, f in enumerate(files) if f.name == "AGENTS.md"), None)
        readme_idx = next((i for i, f in enumerate(files) if f.name == "README.md"), None)
        assert agent_idx is not None
        assert readme_idx is not None
        assert agent_idx < readme_idx

    def test_scan_rich_project(self, rich_project):
        ctx = ProjectContext(rich_project)
        files = ctx.scan()
        names = [f.name for f in files]
        assert "AGENTS.md" in names
        assert "CLAUDE.md" in names
        assert "package.json" in names
        assert "requirements.txt" in names
        assert ".agent/AGENTS.md" in names
        assert any("deploy.md" in n for n in names)

    def test_context_file_has_content(self, project_dir):
        ctx = ProjectContext(project_dir)
        files = ctx.scan()
        agents = next(f for f in files if f.name == "AGENTS.md")
        assert "Agent instructions" in agents.content


class TestSystemPromptBlock:
    """Test system prompt block generation."""

    def test_empty_context(self, tmp_path):
        ctx = ProjectContext(tmp_path)
        block = ctx.system_prompt_block()
        assert block == ""

    def test_block_includes_files(self, project_dir):
        ctx = ProjectContext(project_dir)
        ctx.scan()
        block = ctx.system_prompt_block()
        assert "Project Context" in block
        assert "AGENTS.md" in block

    def test_block_respects_max_chars(self, rich_project):
        ctx = ProjectContext(rich_project)
        ctx.scan()
        block = ctx.system_prompt_block(max_chars=500)
        assert len(block) < 1000  # generous margin

    def test_block_truncation_notice(self, rich_project):
        ctx = ProjectContext(rich_project)
        ctx.scan()
        block = ctx.system_prompt_block(max_chars=200)
        assert "truncated" in block.lower()


class TestGetContext:
    """Test getting specific context files."""

    def test_get_existing(self, project_dir):
        ctx = ProjectContext(project_dir)
        ctx.scan()
        cf = ctx.get_context("AGENTS.md")
        assert cf is not None
        assert "Agent instructions" in cf.content

    def test_get_nonexistent(self, project_dir):
        ctx = ProjectContext(project_dir)
        ctx.scan()
        cf = ctx.get_context("NONEXISTENT.md")
        assert cf is None


class TestNeedsRescan:
    """Test rescan detection."""

    def test_needs_rescan_after_interval(self, project_dir):
        ctx = ProjectContext(project_dir)
        ctx._last_scan = time.time() - 120
        assert ctx.needs_rescan(interval=60.0) is True

    def test_no_rescan_within_interval(self, project_dir):
        ctx = ProjectContext(project_dir)
        ctx._last_scan = time.time() - 10
        assert ctx.needs_rescan(interval=60.0) is False


class TestTotalSize:
    """Test total size calculation."""

    def test_total_size(self, project_dir):
        ctx = ProjectContext(project_dir)
        ctx.scan()
        size = ctx.total_size()
        assert size > 0

    def test_total_size_empty(self, tmp_path):
        ctx = ProjectContext(tmp_path)
        ctx.scan()
        assert ctx.total_size() == 0
