"""Task Planner — decompose complex goals into actionable step plans.

Takes a user goal, sends it to the LLM in deterministic mode to produce a
structured step list, and returns a Plan object that the agent can follow.

Supports:
- Goal decomposition into ordered steps
- Per-step complexity estimation (simple/moderate/complex)
- Plan persistence (save/load from disk)
- Interactive plan approval via human-in-the-loop
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_agent.model_router import ModelRouter

# Complexity tiers
COMPLEXITY_SIMPLE = "simple"
COMPLEXITY_MODERATE = "moderate"
COMPLEXITY_COMPLEX = "complex"
ALL_COMPLEXITIES = (COMPLEXITY_SIMPLE, COMPLEXITY_MODERATE, COMPLEXITY_COMPLEX)

PLAN_SYSTEM_PROMPT = """\
You are a task planner. Your job is to decompose a user goal into a clear,\nordered list of actionable steps.

Rules:
- Each step should be a single, focused action.
- Steps must be ordered logically (prerequisites first).
- Estimate complexity for each step: simple, moderate, or complex.
  - simple: one tool call or quick lookup (< 2 min)
  - moderate: requires reasoning or multiple operations (2-10 min)
  - complex: involves debugging, multi-file changes, or research (> 10 min)
- Keep steps concise — one line each.
- If the goal is already simple (1-2 steps), just return it as-is.

Output ONLY valid JSON with this structure:
{
  "title": "Short title for the plan",
  "steps": [
    {"id": 1, "description": "Do X", "complexity": "simple"},
    {"id": 2, "description": "Then do Y", "complexity": "moderate"}
  ]
}

Do NOT include any text before or after the JSON. Do NOT use markdown code fences."""


@dataclass
class PlanStep:
    """A single step in a task plan."""

    id: int
    description: str
    complexity: str = COMPLEXITY_SIMPLE
    completed: bool = False
    notes: str = ""


@dataclass
class Plan:
    """A decomposed task plan."""

    title: str
    goal: str  # original user goal
    steps: list[PlanStep] = field(default_factory=list)
    created_at: float = 0.0
    completed: bool = False

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [f"Plan: {self.title}"]
        for step in self.steps:
            status = "✓" if step.completed else "○"
            comp = step.complexity.upper()
            lines.append(f"  [{status}] Step {step.id} [{comp}]: {step.description}")
        done = sum(1 for s in self.steps if s.completed)
        lines.append(f"Progress: {done}/{len(self.steps)} steps done")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "title": self.title,
            "goal": self.goal,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "complexity": s.complexity,
                    "completed": s.completed,
                    "notes": s.notes,
                }
                for s in self.steps
            ],
            "created_at": self.created_at,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        """Deserialize from dict."""
        steps = [
            PlanStep(
                id=s["id"],
                description=s["description"],
                complexity=s.get("complexity", COMPLEXITY_SIMPLE),
                completed=s.get("completed", False),
                notes=s.get("notes", ""),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            title=data["title"],
            goal=data["goal"],
            steps=steps,
            created_at=data.get("created_at", 0.0),
            completed=data.get("completed", False),
        )

    def mark_step_done(self, step_id: int) -> bool:
        """Mark a step as completed. Returns True if found."""
        for step in self.steps:
            if step.id == step_id:
                step.completed = True
                return True
        return False


class TaskPlanner:
    """Decompose complex goals into step plans using the LLM."""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router
        self._active_plan: Plan | None = None

    @property
    def active_plan(self) -> Plan | None:
        """Current active plan."""
        return self._active_plan

    def plan(self, goal: str) -> Plan:
        """Decompose a goal into a structured plan.

        Args:
            goal: The user's goal or task description.

        Returns:
            A Plan object with ordered steps.
        """
        import time

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": f"Goal: {goal}\n\nProduce the step plan."},
        ]

        # Force deterministic mode for structured output
        original_deterministic = self._router.config.deterministic
        self._router.config.deterministic = True

        try:
            response = self._router.send_message(messages)
        finally:
            self._router.config.deterministic = original_deterministic

        return self._parse_plan(goal, response)

    def _parse_plan(self, goal: str, response: str) -> Plan:
        """Parse LLM response into a Plan object."""
        import time

        # Strip markdown fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: treat entire response as a single step
            return Plan(
                title=goal[:60],
                goal=goal,
                steps=[PlanStep(id=1, description=goal, complexity=COMPLEXITY_SIMPLE)],
                created_at=time.time(),
            )

        steps: list[PlanStep] = []
        for i, step_data in enumerate(data.get("steps", []), 1):
            steps.append(
                PlanStep(
                    id=i,
                    description=step_data.get("description", f"Step {i}"),
                    complexity=step_data.get("complexity", COMPLEXITY_SIMPLE),
                )
            )

        if not steps:
            steps.append(
                PlanStep(id=1, description=goal, complexity=COMPLEXITY_SIMPLE)
            )

        title = data.get("title", goal[:60])
        plan = Plan(
            title=title,
            goal=goal,
            steps=steps,
            created_at=time.time(),
        )

        self._active_plan = plan
        return plan

    def save_plan(self, plan: Plan, path: str | None = None) -> str:
        """Save a plan to disk as JSON.

        Args:
            plan: The plan to save.
            path: Optional output path. Defaults to ~/.local-coding-agent/plans/

        Returns:
            Path where the plan was saved.
        """
        if path is None:
            plan_dir = Path.home() / ".local-coding-agent" / "plans"
            plan_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(plan.created_at)
            safe_title = "".join(c if c.isalnum() or c in "-_" else "-" for c in plan.title)
            path = str(plan_dir / f"{timestamp}-{safe_title}.json")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(plan.to_dict(), indent=2))
        return str(p)

    def load_plan(self, path: str) -> Plan:
        """Load a plan from disk.

        Args:
            path: Path to saved plan JSON.

        Returns:
            Deserialized Plan.
        """
        data = json.loads(Path(path).read_text())
        plan = Plan.from_dict(data)
        self._active_plan = plan
        return plan

    def list_plans(self) -> list[dict[str, Any]]:
        """List all saved plans.

        Returns:
            List of plan summaries (title, path, steps, created_at, completed).
        """
        plan_dir = Path.home() / ".local-coding-agent" / "plans"
        if not plan_dir.exists():
            return []

        plans = []
        for f in sorted(plan_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text())
                plans.append({
                    "path": str(f),
                    "title": data.get("title", ""),
                    "steps": len(data.get("steps", [])),
                    "created_at": data.get("created_at", 0),
                    "completed": data.get("completed", False),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return plans

    def estimate_overall_complexity(self, plan: Plan) -> str:
        """Estimate overall plan complexity from step complexities.

        Returns:
            simple if all steps are simple, moderate if any moderate, complex if any complex.
        """
        complexities = {s.complexity for s in plan.steps}
        if COMPLEXITY_COMPLEX in complexities:
            return COMPLEXITY_COMPLEX
        if COMPLEXITY_MODERATE in complexities:
            return COMPLEXITY_MODERATE
        return COMPLEXITY_SIMPLE
