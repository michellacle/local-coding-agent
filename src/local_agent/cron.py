"""Cron jobs and scheduled tasks for the local coding agent.

Provides:
- Schedule management (add, remove, list jobs)
- Cron expression parsing (simplified)
- Job execution with timeout and error handling
- Job state persistence
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable


@dataclass
class CronJob:
    """A scheduled cron job.

    Attributes:
        job_id: Unique identifier.
        name: Human-readable name.
        schedule: Cron expression (simplified: "*/5 * * * *").
        prompt: Task prompt to execute.
        enabled: Whether the job is active.
        last_run: Timestamp of last execution.
        next_run: Timestamp of next scheduled execution.
        last_result: Result of last execution.
        last_error: Error from last execution (if any).
        created_at: When the job was created.
        max_retries: Maximum retry attempts on failure.
        retry_count: Current retry count.
        timeout: Execution timeout in seconds.
        metadata: Additional metadata.
    """

    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    schedule: str = ""
    prompt: str = ""
    enabled: bool = True
    last_run: float | None = None
    next_run: float | None = None
    last_result: str = ""
    last_error: str = ""
    created_at: float = field(default_factory=time.time)
    max_retries: int = 3
    retry_count: int = 0
    timeout: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class CronParser:
    """Parse simplified cron expressions.

    Supports standard 5-field cron format:
    minute hour day-of-month month day-of-week

    Also supports shortcuts:
    - "*/N" for every N units
    - "N" for specific value
    - "*" for any value
    - "H" for hashed/random within range
    """

    # Standard cron field ranges
    FIELD_RANGES = {
        "minute": (0, 59),
        "hour": (0, 23),
        "day": (1, 31),
        "month": (1, 12),
        "dow": (0, 6),  # 0=Sunday
    }

    MONTH_NAMES = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    DAY_NAMES = {
        "sun": 0, "mon": 1, "tue": 2, "wed": 3,
        "thu": 4, "fri": 5, "sat": 6,
    }

    @classmethod
    def parse(cls, expression: str) -> list[str]:
        """Parse a cron expression into 5 fields.

        Args:
            expression: Cron expression string.

        Returns:
            List of 5 field strings.

        Raises:
            ValueError: If expression is invalid.
        """
        fields = expression.strip().split()
        if len(fields) != 5:
            raise ValueError(
                f"Invalid cron expression: '{expression}'. "
                f"Expected 5 fields, got {len(fields)}. "
                f"Format: minute hour day-of-month month day-of-week"
            )
        return fields

    @classmethod
    def next_run_time(cls, expression: str, from_time: datetime | None = None) -> datetime:
        """Calculate the next run time for a cron expression.

        Args:
            expression: Cron expression string.
            from_time: Starting point (defaults to now).

        Returns:
            Next datetime when the job should run.
        """
        if from_time is None:
            from_time = datetime.now()

        fields = cls.parse(expression)
        minute_field, hour_field, day_field, month_field, dow_field = fields

        # Start from the next minute
        candidate = from_time.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Try up to a year ahead
        max_iterations = 525_600  # Minutes in a year
        for _ in range(max_iterations):
            if (
                cls._matches_field(minute_field, candidate.minute, 0, 59)
                and cls._matches_field(hour_field, candidate.hour, 0, 23)
                and cls._matches_field(month_field, candidate.month, 1, 12)
                and (
                    cls._matches_field(day_field, candidate.day, 1, 31)
                    or cls._matches_field(dow_field, candidate.weekday() % 7, 0, 6)
                )
            ):
                return candidate

            candidate += timedelta(minutes=1)

        raise ValueError(f"Could not find next run time for '{expression}'")

    @classmethod
    def _matches_field(cls, field: str, value: int, min_val: int, max_val: int) -> bool:
        """Check if a value matches a cron field.

        Args:
            field: Cron field string.
            value: Value to check.
            min_val: Minimum valid value.
            max_val: Maximum valid value.

        Returns:
            True if the value matches.
        """
        # Handle name substitutions
        field = cls._substitute_names(field)

        if field == "*":
            return True

        if field.startswith("*/"):
            step = int(field[2:])
            return value % step == 0

        if "," in field:
            return any(
                cls._matches_field(part, value, min_val, max_val)
                for part in field.split(",")
            )

        if "-" in field:
            parts = field.split("-")
            start, end = int(parts[0]), int(parts[1])
            return start <= value <= end

        try:
            return value == int(field)
        except ValueError:
            return False

    @classmethod
    def _substitute_names(cls, field: str) -> str:
        """Substitute month/day names with numbers.

        Args:
            field: Cron field string.

        Returns:
            Field with names substituted.
        """
        lower = field.lower()
        for name, num in cls.MONTH_NAMES.items():
            lower = lower.replace(name, str(num))
        for name, num in cls.DAY_NAMES.items():
            lower = lower.replace(name, str(num))
        return lower

    @classmethod
    def validate(cls, expression: str) -> bool:
        """Validate a cron expression.

        Args:
            expression: Cron expression string.

        Returns:
            True if valid.
        """
        try:
            cls.parse(expression)
            return True
        except ValueError:
            return False


class JobScheduler:
    """Scheduler that manages cron jobs.

    Handles job registration, scheduling, execution, and persistence.
    """

    def __init__(self, state_path: str | None = None):
        """Initialize the scheduler.

        Args:
            state_path: Optional path for job state persistence.
        """
        self._jobs: dict[str, CronJob] = {}
        self._state_path = state_path
        self._handlers: dict[str, Callable] = {}
        self._running = False
        self._last_check = 0.0

        if state_path:
            self._load_state()

    def _load_state(self) -> None:
        """Load jobs from state file."""
        if not self._state_path:
            return

        path = Path(self._state_path)
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text())
            for job_data in data.get("jobs", []):
                job = CronJob(**job_data)
                self._jobs[job.job_id] = job
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    def _save_state(self) -> None:
        """Save jobs to state file."""
        if not self._state_path:
            return

        data = {
            "jobs": [job.to_dict() for job in self._jobs.values()],
            "saved_at": time.time(),
        }
        path = Path(self._state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))

    def add_job(
        self,
        name: str,
        schedule: str,
        prompt: str,
        timeout: int = 300,
        metadata: dict[str, Any] | None = None,
    ) -> CronJob:
        """Add a new cron job.

        Args:
            name: Human-readable name.
            schedule: Cron expression.
            prompt: Task prompt.
            timeout: Execution timeout in seconds.
            metadata: Optional metadata.

        Returns:
            The created CronJob.

        Raises:
            ValueError: If the schedule is invalid.
        """
        if not CronParser.validate(schedule):
            raise ValueError(f"Invalid cron schedule: {schedule}")

        # Calculate next run time
        next_run = CronParser.next_run_time(schedule)

        job = CronJob(
            name=name,
            schedule=schedule,
            prompt=prompt,
            timeout=timeout,
            next_run=next_run.timestamp(),
            metadata=metadata or {},
        )

        self._jobs[job.job_id] = job
        self._save_state()

        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a cron job.

        Args:
            job_id: Job ID to remove.

        Returns:
            True if job was found and removed.
        """
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save_state()
            return True
        return False

    def enable_job(self, job_id: str) -> bool:
        """Enable a cron job.

        Args:
            job_id: Job ID to enable.

        Returns:
            True if job was found and enabled.
        """
        if job_id in self._jobs:
            self._jobs[job_id].enabled = True
            self._save_state()
            return True
        return False

    def disable_job(self, job_id: str) -> bool:
        """Disable a cron job.

        Args:
            job_id: Job ID to disable.

        Returns:
            True if job was found and disabled.
        """
        if job_id in self._jobs:
            self._jobs[job_id].enabled = False
            self._save_state()
            return True
        return False

    def get_job(self, job_id: str) -> CronJob | None:
        """Get a job by ID.

        Args:
            job_id: Job ID to retrieve.

        Returns:
            CronJob or None if not found.
        """
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[CronJob]:
        """List all jobs.

        Returns:
            List of all CronJob objects.
        """
        return list(self._jobs.values())

    def get_due_jobs(self) -> list[CronJob]:
        """Get jobs that are due for execution.

        Returns:
            List of enabled jobs that are due.
        """
        now = time.time()
        due = []
        for job in self._jobs.values():
            if job.enabled and job.next_run and job.next_run <= now:
                due.append(job)
        return due

    def execute_job(
        self,
        job_id: str,
        executor: Callable[[CronJob], str] | None = None,
    ) -> str:
        """Execute a job immediately.

        Args:
            job_id: Job ID to execute.
            executor: Optional custom executor function.

        Returns:
            Execution result string.

        Raises:
            ValueError: If job not found.
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job.last_run = time.time()

        try:
            if executor:
                result = executor(job)
            else:
                result = f"Job '{job.name}' executed with prompt: {job.prompt[:100]}"

            job.last_result = result
            job.retry_count = 0
            job.last_error = ""

            # Schedule next run
            next_run = CronParser.next_run_time(job.schedule)
            job.next_run = next_run.timestamp()

            self._save_state()
            return result

        except Exception as e:
            error_msg = str(e)
            job.last_error = error_msg
            job.retry_count += 1

            if job.retry_count >= job.max_retries:
                job.enabled = False

            self._save_state()
            raise

    def run_due_jobs(
        self,
        executor: Callable[[CronJob], str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run all due jobs.

        Args:
            executor: Optional custom executor function.

        Returns:
            List of execution results.
        """
        results = []
        due_jobs = self.get_due_jobs()

        for job in due_jobs:
            try:
                result = self.execute_job(job.job_id, executor)
                results.append({
                    "job_id": job.job_id,
                    "name": job.name,
                    "status": "success",
                    "result": result,
                })
            except Exception as e:
                results.append({
                    "job_id": job.job_id,
                    "name": job.name,
                    "status": "error",
                    "error": str(e),
                })

        return results

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all jobs.

        Returns:
            Dict with job counts and stats.
        """
        enabled = sum(1 for j in self._jobs.values() if j.enabled)
        disabled = sum(1 for j in self._jobs.values() if not j.enabled)

        return {
            "total_jobs": len(self._jobs),
            "enabled_jobs": enabled,
            "disabled_jobs": disabled,
            "due_jobs": len(self.get_due_jobs()),
        }

    def clear(self) -> None:
        """Clear all jobs."""
        self._jobs.clear()
        if self._state_path:
            Path(self._state_path).write_text("{}")


# Convenience functions for common schedules

def every_n_minutes(n: int) -> str:
    """Create a cron expression for every N minutes.

    Args:
        n: Number of minutes.

    Returns:
        Cron expression string.
    """
    return f"*/{n} * * * *"


def every_n_hours(n: int) -> str:
    """Create a cron expression for every N hours.

    Args:
        n: Number of hours.

    Returns:
        Cron expression string.
    """
    return f"0 */{n} * * *"


def daily(hour: int = 0, minute: int = 0) -> str:
    """Create a cron expression for daily at a specific time.

    Args:
        hour: Hour (0-23).
        minute: Minute (0-59).

    Returns:
        Cron expression string.
    """
    return f"{minute} {hour} * * *"


def weekly(day: int = 0, hour: int = 0, minute: int = 0) -> str:
    """Create a cron expression for weekly on a specific day.

    Args:
        day: Day of week (0=Sunday, 1=Monday, ..., 6=Saturday).
        hour: Hour (0-23).
        minute: Minute (0-59).

    Returns:
        Cron expression string.
    """
    return f"{minute} {hour} * * {day}"


def monthly(day: int = 1, hour: int = 0, minute: int = 0) -> str:
    """Create a cron expression for monthly on a specific day.

    Args:
        day: Day of month (1-31).
        hour: Hour (0-23).
        minute: Minute (0-59).

    Returns:
        Cron expression string.
    """
    return f"{minute} {hour} {day} * *"
