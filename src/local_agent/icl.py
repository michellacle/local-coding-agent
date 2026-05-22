"""In-Context Learning — learn from user corrections during a session.

Tracks when the user corrects the agent's output and uses those corrections
to improve future responses within the same session. Corrections are stored
as few-shot examples that are injected into the system prompt.

This implements the "In-Context Learning (ICL)" pattern from Arsanjani & Bustos:
the agent improves within a session without fine-tuning by adapting its
prompt based on user feedback.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Correction:
    """A single user correction of agent output."""

    agent_output: str       # What the agent produced
    user_correction: str    # What the user said to fix it
    category: str           # "style", "format", "logic", "security", "naming", "other"
    timestamp: float = field(default_factory=time.time)
    applied: bool = True

    def prompt_example(self) -> str:
        """Format as a few-shot example for the system prompt."""
        return (
            f"  Correction [{self.category}]:\n"
            f"    Agent: {self.agent_output[:200]}\n"
            f"    User fix: {self.user_correction[:200]}"
        )


class InContextLearner:
    """Track user corrections and adapt the system prompt.

    When a user corrects the agent, the correction is stored and used to
    refine future responses. Corrections are grouped by category and
    injected as few-shot examples.

    Usage:
        icl = InContextLearner()
        icl.record_correction(
            agent_output="Used snake_case for variable names",
            user_correction="Use camelCase instead",
            category="naming"
        )
        prompt_addendum = icl.system_prompt_addendum()
    """

    def __init__(self, max_corrections: int = 20) -> None:
        self.corrections: list[Correction] = []
        self.max_corrections = max_corrections
        self._correction_counts: dict[str, int] = {}

    def record_correction(
        self,
        agent_output: str,
        user_correction: str,
        category: str = "other",
    ) -> Correction:
        """Record a user correction.

        Args:
            agent_output: What the agent produced.
            user_correction: What the user said to fix it.
            category: Category of correction.

        Returns:
            The recorded Correction object.
        """
        correction = Correction(
            agent_output=agent_output,
            user_correction=user_correction,
            category=category,
        )
        self.corrections.append(correction)
        self._correction_counts[category] = self._correction_counts.get(category, 0) + 1

        # Trim if over limit
        if len(self.corrections) > self.max_corrections:
            self.corrections = self.corrections[-self.max_corrections:]

        return correction

    def system_prompt_addendum(self, max_examples: int = 5) -> str:
        """Generate a system prompt addendum based on corrections.

        Args:
            max_examples: Maximum number of examples to include.

        Returns:
            Formatted string to append to the system prompt.
        """
        if not self.corrections:
            return ""

        # Group by category and pick the most relevant ones
        recent = self.corrections[-max_examples:]

        parts = ["\n## User Corrections (apply these preferences)\n"]

        # Show category summary
        parts.append("Correction history:")
        for cat, count in sorted(self._correction_counts.items(), key=lambda x: -x[1]):
            parts.append(f"  {cat}: {count} time{'s' if count > 1 else ''}")

        # Show recent examples
        parts.append("\nRecent corrections:")
        for c in recent:
            parts.append(c.prompt_example())

        parts.append("\nIMPORTANT: Apply these corrections to all future responses.")

        return "\n".join(parts)

    def get_corrections_by_category(self, category: str) -> list[Correction]:
        """Get corrections for a specific category."""
        return [c for c in self.corrections if c.category == category]

    def has_corrections(self) -> bool:
        """Check if any corrections have been recorded."""
        return len(self.corrections) > 0

    def correction_summary(self) -> str:
        """Get a summary of all corrections."""
        if not self.corrections:
            return "No corrections recorded."

        lines = [f"Total corrections: {len(self.corrections)}"]
        for cat, count in sorted(self._correction_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: {count}")

        return "\n".join(lines)

    def detect_correction_category(self, user_input: str) -> str:
        """Heuristically detect the category of a user correction.

        Args:
            user_input: The user's correction text.

        Returns:
            Detected category.
        """
        lower = user_input.lower()

        naming_keywords = ["name", "variable", "function name", "camelcase", "snake_case", "naming"]
        style_keywords = ["style", "format", "indent", "spacing", "quotes", "convention"]
        logic_keywords = ["wrong", "incorrect", "bug", "error", "logic", "doesn't work", "should be"]
        security_keywords = ["security", "injection", "unsafe", "vulnerable", "leak", "sanitize"]

        for kw in naming_keywords:
            if kw in lower:
                return "naming"
        for kw in style_keywords:
            if kw in lower:
                return "style"
        for kw in security_keywords:
            if kw in lower:
                return "security"
        for kw in logic_keywords:
            if kw in lower:
                return "logic"
        for kw in style_keywords:
            if kw in lower:
                return "format"

        return "other"

    def save_to_file(self, path: str) -> None:
        """Save corrections to a JSON file.

        Args:
            path: Path to save the corrections.
        """
        data = {
            "corrections": [
                {
                    "agent_output": c.agent_output,
                    "user_correction": c.user_correction,
                    "category": c.category,
                    "timestamp": c.timestamp,
                }
                for c in self.corrections
            ],
            "counts": self._correction_counts,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_from_file(self, path: str) -> int:
        """Load corrections from a JSON file.

        Args:
            path: Path to load corrections from.

        Returns:
            Number of corrections loaded.
        """
        p = Path(path)
        if not p.exists():
            return 0

        data = json.loads(p.read_text(encoding="utf-8"))
        count = 0
        for item in data.get("corrections", []):
            self.corrections.append(Correction(
                agent_output=item["agent_output"],
                user_correction=item["user_correction"],
                category=item.get("category", "other"),
                timestamp=item.get("timestamp", time.time()),
            ))
            count += 1

        self._correction_counts = data.get("counts", {})
        return count

    def clear(self) -> int:
        """Clear all corrections. Returns the number cleared."""
        count = len(self.corrections)
        self.corrections.clear()
        self._correction_counts.clear()
        return count
