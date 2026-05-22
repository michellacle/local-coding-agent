"""Tests for InContextLearner — learn from user corrections."""

import pytest
import json
import time
from pathlib import Path

from local_agent.icl import InContextLearner, Correction


class TestCorrection:
    """Test Correction dataclass."""

    def test_prompt_example(self):
        c = Correction(
            agent_output="Used snake_case",
            user_correction="Use camelCase",
            category="naming",
        )
        example = c.prompt_example()
        assert "snake_case" in example
        assert "camelCase" in example
        assert "naming" in example


class TestInContextLearner:
    """Test InContextLearner core functionality."""

    @pytest.fixture
    def icl(self):
        return InContextLearner()

    def test_record_correction(self, icl):
        icl.record_correction("bad output", "good output", "style")
        assert len(icl.corrections) == 1
        assert icl.corrections[0].category == "style"

    def test_record_multiple(self, icl):
        icl.record_correction("a", "b", "style")
        icl.record_correction("c", "d", "naming")
        assert len(icl.corrections) == 2

    def test_max_corrections_limit(self):
        icl = InContextLearner(max_corrections=3)
        for i in range(5):
            icl.record_correction(f"output {i}", f"fix {i}", "other")
        assert len(icl.corrections) == 3
        # Should keep the most recent 3
        assert icl.corrections[0].agent_output == "output 2"

    def test_system_prompt_addendum_empty(self, icl):
        addendum = icl.system_prompt_addendum()
        assert addendum == ""

    def test_system_prompt_addendum_with_corrections(self, icl):
        icl.record_correction("snake_case", "camelCase", "naming")
        icl.record_correction("use print", "use logging", "style")
        addendum = icl.system_prompt_addendum()
        assert "User Corrections" in addendum
        assert "naming" in addendum
        assert "camelCase" in addendum

    def test_system_prompt_respects_max_examples(self, icl):
        for i in range(10):
            icl.record_correction(f"output {i}", f"fix {i}", "other")
        addendum = icl.system_prompt_addendum(max_examples=3)
        # Should not include all 10
        count = addendum.count("Correction [")
        assert count <= 3

    def test_has_corrections(self, icl):
        assert icl.has_corrections() is False
        icl.record_correction("a", "b", "style")
        assert icl.has_corrections() is True

    def test_correction_summary(self, icl):
        icl.record_correction("a", "b", "style")
        icl.record_correction("c", "d", "naming")
        summary = icl.correction_summary()
        assert "style" in summary
        assert "naming" in summary

    def test_correction_summary_empty(self, icl):
        summary = icl.correction_summary()
        assert "No corrections" in summary

    def test_get_by_category(self, icl):
        icl.record_correction("a", "b", "style")
        icl.record_correction("c", "d", "naming")
        icl.record_correction("e", "f", "style")
        style_corrections = icl.get_corrections_by_category("style")
        assert len(style_corrections) == 2

    def test_clear(self, icl):
        icl.record_correction("a", "b", "style")
        icl.record_correction("c", "d", "naming")
        cleared = icl.clear()
        assert cleared == 2
        assert icl.has_corrections() is False


class TestDetectCorrectionCategory:
    """Test correction category detection."""

    def test_naming(self):
        icl = InContextLearner()
        assert icl.detect_correction_category("Use camelCase for variable names") == "naming"

    def test_style(self):
        icl = InContextLearner()
        assert icl.detect_correction_category("Use double quotes instead of single") == "style"

    def test_logic(self):
        icl = InContextLearner()
        assert icl.detect_correction_category("This logic is wrong, it should be a loop") == "logic"

    def test_security(self):
        icl = InContextLearner()
        assert icl.detect_correction_category("This is a security vulnerability") == "security"

    def test_other(self):
        icl = InContextLearner()
        assert icl.detect_correction_category("Make it faster") == "other"


class TestPersistence:
    """Test saving and loading corrections."""

    def test_save_and_load(self, tmp_path):
        icl = InContextLearner()
        icl.record_correction("a", "b", "style")
        icl.record_correction("c", "d", "naming")

        path = str(tmp_path / "corrections.json")
        icl.save_to_file(path)

        new_icl = InContextLearner()
        loaded = new_icl.load_from_file(path)
        assert loaded == 2
        assert len(new_icl.corrections) == 2

    def test_load_nonexistent(self):
        icl = InContextLearner()
        loaded = icl.load_from_file("/tmp/definitely_not_here.json")
        assert loaded == 0
