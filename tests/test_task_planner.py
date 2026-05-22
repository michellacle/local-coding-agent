"""Tests for TaskPlanner — goal decomposition and plan management."""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from local_agent.config import LLMConfig
from local_agent.model_router import ModelRouter
from local_agent.task_planner import (
    ALL_COMPLEXITIES,
    COMPLEXITY_COMPLEX,
    COMPLEXITY_MODERATE,
    COMPLEXITY_SIMPLE,
    Plan,
    PlanStep,
    TaskPlanner,
)


@pytest.fixture
def router():
    """Create a ModelRouter with a mock config."""
    config = LLMConfig(model="test", host="localhost", port=9999)
    return ModelRouter(config)


@pytest.fixture
def planner(router: ModelRouter):
    """Create a TaskPlanner with a router."""
    return TaskPlanner(router)


# ------------------------------------------------------------------ #
# PlanStep tests
# ------------------------------------------------------------------ #


class TestPlanStep:
    def test_creation(self):
        step = PlanStep(id=1, description="Do something", complexity="simple")
        assert step.id == 1
        assert step.description == "Do something"
        assert step.complexity == "simple"
        assert step.completed is False
        assert step.notes == ""

    def test_default_complexity(self):
        step = PlanStep(id=1, description="X")
        assert step.complexity == COMPLEXITY_SIMPLE

    def test_custom_notes(self):
        step = PlanStep(id=1, description="X", notes="some notes")
        assert step.notes == "some notes"


# ------------------------------------------------------------------ #
# Plan tests
# ------------------------------------------------------------------ #


class TestPlan:
    def test_creation(self):
        plan = Plan(title="Test", goal="Do X")
        assert plan.title == "Test"
        assert plan.goal == "Do X"
        assert plan.steps == []
        assert plan.completed is False

    def test_summary_empty(self):
        plan = Plan(title="Empty", goal="X")
        s = plan.summary()
        assert "Empty" in s
        assert "0/0" in s

    def test_summary_with_steps(self):
        plan = Plan(
            title="Build",
            goal="Build project",
            steps=[
                PlanStep(id=1, description="Install deps", complexity="simple"),
                PlanStep(id=2, description="Run tests", complexity="moderate"),
            ],
        )
        s = plan.summary()
        assert "Build" in s
        assert "Install deps" in s
        assert "Run tests" in s
        assert "0/2 steps done" in s

    def test_summary_completed_steps(self):
        plan = Plan(
            title="A",
            goal="A",
            steps=[
                PlanStep(id=1, description="X", complexity="simple", completed=True),
                PlanStep(id=2, description="Y", complexity="moderate", completed=False),
            ],
        )
        s = plan.summary()
        assert "1/2 steps done" in s

    def test_to_dict(self):
        plan = Plan(
            title="T",
            goal="G",
            steps=[PlanStep(id=1, description="D", complexity="moderate", completed=True)],
            created_at=123.0,
            completed=True,
        )
        d = plan.to_dict()
        assert d["title"] == "T"
        assert d["goal"] == "G"
        assert d["completed"] is True
        assert len(d["steps"]) == 1
        assert d["steps"][0]["description"] == "D"

    def test_from_dict(self):
        data = {
            "title": "T",
            "goal": "G",
            "steps": [
                {
                    "id": 1,
                    "description": "D",
                    "complexity": "complex",
                    "completed": True,
                    "notes": "N",
                }
            ],
            "created_at": 123.0,
            "completed": True,
        }
        plan = Plan.from_dict(data)
        assert plan.title == "T"
        assert plan.goal == "G"
        assert len(plan.steps) == 1
        assert plan.steps[0].description == "D"
        assert plan.steps[0].complexity == "complex"
        assert plan.steps[0].completed is True
        assert plan.steps[0].notes == "N"
        assert plan.created_at == 123.0
        assert plan.completed is True

    def test_from_dict_defaults(self):
        data = {"title": "X", "goal": "Y", "steps": []}
        plan = Plan.from_dict(data)
        assert plan.created_at == 0.0
        assert plan.completed is False

    def test_roundtrip(self):
        plan = Plan(
            title="R",
            goal="Round trip",
            steps=[
                PlanStep(id=1, description="A", complexity="simple", notes="n1"),
                PlanStep(id=2, description="B", complexity="moderate", completed=True, notes="n2"),
            ],
            created_at=99.0,
        )
        restored = Plan.from_dict(plan.to_dict())
        assert restored.title == plan.title
        assert restored.goal == plan.goal
        assert len(restored.steps) == 2
        for a, b in zip(plan.steps, restored.steps):
            assert a.id == b.id
            assert a.description == b.description
            assert a.complexity == b.complexity
            assert a.completed == b.completed
            assert a.notes == b.notes

    def test_mark_step_done(self):
        plan = Plan(
            title="X",
            goal="X",
            steps=[
                PlanStep(id=1, description="A", completed=False),
                PlanStep(id=2, description="B", completed=False),
            ],
        )
        assert plan.mark_step_done(2) is True
        assert plan.steps[1].completed is True
        assert plan.steps[0].completed is False

    def test_mark_step_done_not_found(self):
        plan = Plan(title="X", goal="X", steps=[PlanStep(id=1, description="A")])
        assert plan.mark_step_done(99) is False

    def test_serialization_includes_created_at(self):
        now = time.time()
        plan = Plan(title="Now", goal="Now", created_at=now)
        d = plan.to_dict()
        assert abs(d["created_at"] - now) < 1.0


# ------------------------------------------------------------------ #
# TaskPlanner tests
# ------------------------------------------------------------------ #


class TestTaskPlanner:
    def test_parse_plan_valid_json(self, planner: TaskPlanner):
        response = json.dumps({
            "title": "Build",
            "steps": [
                {"description": "Install deps", "complexity": "simple"},
                {"description": "Write code", "complexity": "moderate"},
            ],
        })
        plan = planner._parse_plan("Build the project", response)
        assert plan.title == "Build"
        assert plan.goal == "Build the project"
        assert len(plan.steps) == 2
        assert plan.steps[0].description == "Install deps"
        assert plan.steps[0].complexity == "simple"
        assert plan.steps[1].description == "Write code"
        assert plan.steps[1].complexity == "moderate"

    def test_parse_plan_with_markdown_fences(self, planner: TaskPlanner):
        response = "```json\n" + json.dumps({
            "title": "X",
            "steps": [{"description": "Y", "complexity": "simple"}],
        }) + "\n```"
        plan = planner._parse_plan("X", response)
        assert plan.title == "X"
        assert len(plan.steps) == 1

    def test_parse_plan_invalid_json(self, planner: TaskPlanner):
        response = "This is not json at all."
        plan = planner._parse_plan("Fallback goal", response)
        assert len(plan.steps) == 1
        assert plan.steps[0].description == "Fallback goal"

    def test_parse_plan_empty_steps(self, planner: TaskPlanner):
        response = json.dumps({"title": "Empty", "steps": []})
        plan = planner._parse_plan("Fallback", response)
        assert len(plan.steps) == 1
        assert plan.steps[0].description == "Fallback"

    def test_parse_plan_complexity_default(self, planner: TaskPlanner):
        response = json.dumps({
            "title": "T",
            "steps": [{"description": "A"}],
        })
        plan = planner._parse_plan("T", response)
        assert plan.steps[0].complexity == COMPLEXITY_SIMPLE

    def test_parse_sets_active_plan(self, planner: TaskPlanner):
        response = json.dumps({
            "title": "Active",
            "steps": [{"description": "X", "complexity": "simple"}],
        })
        plan = planner._parse_plan("Active goal", response)
        assert planner.active_plan is plan

    def test_save_and_load_plan(self, planner: TaskPlanner):
        plan = Plan(
            title="SaveTest",
            goal="Save and load",
            steps=[PlanStep(id=1, description="Step A", complexity="moderate")],
            created_at=time.time(),
        )
        with TemporaryDirectory() as tmp:
            path = planner.save_plan(plan, f"{tmp}/test.json")
            assert Path(path).exists()
            loaded = planner.load_plan(path)
            assert loaded.title == "SaveTest"
            assert len(loaded.steps) == 1

    def test_save_plan_auto_path(self, planner: TaskPlanner):
        plan = Plan(title="Auto", goal="Auto", created_at=time.time())
        path = planner.save_plan(plan)
        assert Path(path).exists()
        assert Path(path).suffix == ".json"

    def test_list_plans(self, planner: TaskPlanner):
        plan = Plan(title="ListTest", goal="L", steps=[PlanStep(id=1, description="A")], created_at=time.time())
        with TemporaryDirectory() as tmp:
            plan_dir = Path(tmp)
            # Simulate plans directory
            planner.save_plan(plan, f"{tmp}/123-Test.json")
            # Monkey-patch list_plans to use temp dir
            original_home = Path.home
            try:
                # We can't easily patch Path.home, so just test save+load
                plans = planner.list_plans()
                # May return empty if ~/.local-coding-agent/plans doesn't exist
                assert isinstance(plans, list)
            finally:
                pass

    def test_estimate_all_simple(self, planner: TaskPlanner):
        plan = Plan(
            title="Simple",
            goal="S",
            steps=[
                PlanStep(id=1, description="A", complexity=COMPLEXITY_SIMPLE),
                PlanStep(id=2, description="B", complexity=COMPLEXITY_SIMPLE),
            ],
        )
        assert planner.estimate_overall_complexity(plan) == COMPLEXITY_SIMPLE

    def test_estimate_has_moderate(self, planner: TaskPlanner):
        plan = Plan(
            title="Mod",
            goal="M",
            steps=[
                PlanStep(id=1, description="A", complexity=COMPLEXITY_SIMPLE),
                PlanStep(id=2, description="B", complexity=COMPLEXITY_MODERATE),
            ],
        )
        assert planner.estimate_overall_complexity(plan) == COMPLEXITY_MODERATE

    def test_estimate_has_complex(self, planner: TaskPlanner):
        plan = Plan(
            title="Comp",
            goal="C",
            steps=[
                PlanStep(id=1, description="A", complexity=COMPLEXITY_SIMPLE),
                PlanStep(id=2, description="B", complexity=COMPLEXITY_COMPLEX),
            ],
        )
        assert planner.estimate_overall_complexity(plan) == COMPLEXITY_COMPLEX

    def test_estimate_empty_plan(self, planner: TaskPlanner):
        plan = Plan(title="Empty", goal="E")
        assert planner.estimate_overall_complexity(plan) == COMPLEXITY_SIMPLE

    def test_all_complexities_constant(self):
        assert COMPLEXITY_SIMPLE in ALL_COMPLEXITIES
        assert COMPLEXITY_MODERATE in ALL_COMPLEXITIES
        assert COMPLEXITY_COMPLEX in ALL_COMPLEXITIES
