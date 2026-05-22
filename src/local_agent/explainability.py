"""Explainability and audit trail for agent actions.

Provides:
- Decision logging (why the agent chose a specific action)
- Chain-of-thought recording
- Action confidence scoring
- Audit trail with timestamps and context
- Self-assessment of outputs
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class DecisionType(str, Enum):
    """Types of decisions an agent can make."""

    TOOL_CALL = "tool_call"
    CODE_GENERATION = "code_generation"
    FILE_EDIT = "file_edit"
    TERMINAL_COMMAND = "terminal_command"
    REASONING = "reasoning"
    FALLBACK = "fallback"
    ERROR_RECOVERY = "error_recovery"


class ConfidenceLevel(str, Enum):
    """Confidence levels for agent decisions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class DecisionRecord:
    """A single decision made by the agent.

    Attributes:
        decision_id: Unique identifier for this decision.
        timestamp: When the decision was made.
        decision_type: Type of decision.
        action: What action was taken.
        reasoning: Why this action was chosen.
        alternatives: Other options considered.
        confidence: Confidence level in this decision.
        context: Additional context (inputs, state, etc.).
        outcome: Result of the action (filled in after execution).
    """

    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    decision_type: DecisionType = DecisionType.REASONING
    action: str = ""
    reasoning: str = ""
    alternatives: list[str] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    context: dict[str, Any] = field(default_factory=dict)
    outcome: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        d = asdict(self)
        d["decision_type"] = self.decision_type.value
        d["confidence"] = self.confidence.value
        return d

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class ChainOfThought:
    """A chain of reasoning steps.

    Attributes:
        chain_id: Unique identifier for this chain.
        steps: Ordered list of reasoning steps.
        conclusion: Final conclusion drawn from the chain.
        timestamp: When the chain was created.
    """

    chain_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    steps: list[str] = field(default_factory=list)
    conclusion: str = ""
    timestamp: float = field(default_factory=time.time)

    def add_step(self, step: str) -> None:
        """Add a reasoning step."""
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return asdict(self)


@dataclass
class AuditEntry:
    """An entry in the audit trail.

    Attributes:
        entry_id: Unique identifier.
        timestamp: When the event occurred.
        event_type: Type of event.
        agent_state: Snapshot of agent state at the time.
        input: Input that triggered the event.
        output: Output produced by the event.
        decision: Associated decision record (if any).
        chain_of_thought: Associated reasoning chain (if any).
        metadata: Additional metadata.
    """

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    agent_state: dict[str, Any] = field(default_factory=dict)
    input: str = ""
    output: str = ""
    decision: DecisionRecord | None = None
    chain_of_thought: ChainOfThought | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        d = asdict(self)
        if self.decision:
            d["decision"] = self.decision.to_dict()
        if self.chain_of_thought:
            d["chain_of_thought"] = self.chain_of_thought.to_dict()
        return d


class AuditLogger:
    """Logger that records agent actions to an audit trail.

    Writes entries to a JSON Lines file for persistence.
    """

    def __init__(self, log_path: str | None = None):
        """Initialize the audit logger.

        Args:
            log_path: Optional path to JSON Lines log file.
        """
        self._log_path = log_path
        self._entries: list[AuditEntry] = []
        self._decisions: list[DecisionRecord] = []
        self._chains: list[ChainOfThought] = []

        if log_path:
            log_file = Path(log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            # Load existing entries if file exists
            if log_file.exists():
                self._load_entries(log_file)

    def _load_entries(self, path: Path) -> None:
        """Load existing audit entries from file."""
        for line in path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    data = json.loads(line)
                    entry = AuditEntry(**{k: v for k, v in data.items() if k != "decision" and k != "chain_of_thought"})
                    self._entries.append(entry)
                except (json.JSONDecodeError, TypeError):
                    pass

    def log_decision(
        self,
        decision_type: DecisionType,
        action: str,
        reasoning: str,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        alternatives: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> DecisionRecord:
        """Log a decision made by the agent.

        Args:
            decision_type: Type of decision.
            action: Action taken.
            reasoning: Why this action was chosen.
            confidence: Confidence level.
            alternatives: Other options considered.
            context: Additional context.

        Returns:
            The created DecisionRecord.
        """
        record = DecisionRecord(
            decision_type=decision_type,
            action=action,
            reasoning=reasoning,
            confidence=confidence,
            alternatives=alternatives or [],
            context=context or {},
        )
        self._decisions.append(record)

        entry = AuditEntry(
            event_type=f"decision.{decision_type.value}",
            input=json.dumps(context or {}),
            output=action,
            decision=record,
        )
        self._append_entry(entry)

        return record

    def log_chain_of_thought(
        self,
        steps: list[str],
        conclusion: str,
    ) -> ChainOfThought:
        """Log a chain of reasoning.

        Args:
            steps: Ordered reasoning steps.
            conclusion: Final conclusion.

        Returns:
            The created ChainOfThought.
        """
        chain = ChainOfThought(steps=steps, conclusion=conclusion)
        self._chains.append(chain)

        entry = AuditEntry(
            event_type="reasoning.chain",
            output=conclusion,
            chain_of_thought=chain,
        )
        self._append_entry(entry)

        return chain

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: str = "",
        decision: DecisionRecord | None = None,
    ) -> AuditEntry:
        """Log a tool call.

        Args:
            tool_name: Name of the tool called.
            arguments: Arguments passed to the tool.
            result: Result of the tool call.
            decision: Associated decision record.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            event_type="tool_call",
            input=json.dumps(arguments, default=str),
            output=result,
            decision=decision,
            metadata={"tool_name": tool_name},
        )
        self._append_entry(entry)
        return entry

    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Log an error event.

        Args:
            error_type: Type of error.
            error_message: Error message.
            context: Additional context.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            event_type=f"error.{error_type}",
            input=error_message,
            metadata=context or {},
        )
        self._append_entry(entry)
        return entry

    def log_state_snapshot(self, state: dict[str, Any]) -> AuditEntry:
        """Log an agent state snapshot.

        Args:
            state: Current agent state.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            event_type="state_snapshot",
            agent_state=state,
        )
        self._append_entry(entry)
        return entry

    def _append_entry(self, entry: AuditEntry) -> None:
        """Append an entry to the log."""
        self._entries.append(entry)

        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    def get_entries(
        self,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Get audit entries, optionally filtered by event type.

        Args:
            event_type: Filter by event type (supports prefix matching).
            limit: Maximum number of entries.

        Returns:
            List of matching AuditEntry objects.
        """
        entries = self._entries

        if event_type:
            entries = [e for e in entries if e.event_type.startswith(event_type)]

        return entries[-limit:]

    def get_decisions(self) -> list[DecisionRecord]:
        """Get all logged decisions."""
        return self._decisions[:]

    def get_chains(self) -> list[ChainOfThought]:
        """Get all logged chains of thought."""
        return self._chains[:]

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the audit trail.

        Returns:
            Dict with counts and stats.
        """
        event_counts: dict[str, int] = {}
        for entry in self._entries:
            base = entry.event_type.split(".")[0]
            event_counts[base] = event_counts.get(base, 0) + 1

        confidence_counts: dict[str, int] = {}
        for decision in self._decisions:
            confidence_counts[decision.confidence.value] = (
                confidence_counts.get(decision.confidence.value, 0) + 1
            )

        return {
            "total_entries": len(self._entries),
            "total_decisions": len(self._decisions),
            "total_chains": len(self._chains),
            "event_counts": event_counts,
            "confidence_distribution": confidence_counts,
        }

    def export_json(self, path: str) -> None:
        """Export the full audit trail as a JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "entries": [e.to_dict() for e in self._entries],
            "summary": self.get_summary(),
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._decisions.clear()
        self._chains.clear()

        if self._log_path:
            Path(self._log_path).write_text("")


class SelfAssessment:
    """Self-assessment of agent outputs.

    Evaluates quality, completeness, and correctness of generated content.
    """

    @staticmethod
    def assess_code(
        code: str,
        language: str = "python",
        criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assess the quality of generated code.

        Args:
            code: Generated code string.
            language: Programming language.
            criteria: Custom assessment criteria.

        Returns:
            Dict with assessment results.
        """
        assessment: dict[str, Any] = {
            "type": "code_assessment",
            "language": language,
            "line_count": len(code.strip().split("\n")),
            "has_comments": "#" in code or "//" in code or "/*" in code,
            "has_error_handling": any(
                kw in code for kw in ["try:", "except", "catch", "if err", "raise"]
            ),
            "has_logging": any(
                kw in code for kw in ["print(", "logging.", "log.", "console.log"]
            ),
            "indentation_level": SelfAssessment._estimate_indentation(code),
            "meets_criteria": {},
        }

        if criteria:
            for key, expected in criteria.items():
                assessment["meets_criteria"][key] = expected in code

        return assessment

    @staticmethod
    def assess_response(
        response: str,
        criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assess the quality of a text response.

        Args:
            response: Generated response text.
            criteria: Custom assessment criteria.

        Returns:
            Dict with assessment results.
        """
        lines = response.strip().split("\n")
        words = response.split()

        assessment: dict[str, Any] = {
            "type": "response_assessment",
            "word_count": len(words),
            "line_count": len(lines),
            "has_headings": any(
                line.startswith("#") or line.startswith("=") for line in lines
            ),
            "has_lists": any(
                line.strip().startswith(("-", "*", "•", "1.", "2.")) for line in lines
            ),
            "has_code_blocks": "```" in response,
            "has_references": any(
                word.lower().endswith((".md", ".py", ".js", "http")) for word in words
            ),
            "completeness_score": min(1.0, len(words) / 100),
        }

        return assessment

    @staticmethod
    def _estimate_indentation(code: str) -> int:
        """Estimate the average indentation level of code.

        Args:
            code: Code string.

        Returns:
            Average indentation level.
        """
        lines = code.split("\n")
        indents = []
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                indents.append(indent)

        if not indents:
            return 0

        total = sum(indents)
        avg = total / len(indents)

        # Convert to indentation level (assuming 4 spaces per level)
        return int(avg / 4)
