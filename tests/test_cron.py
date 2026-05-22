"""Tests for cron module — scheduled jobs, cron parser, scheduler."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from local_agent.cron import (
    CronJob,
    CronJob,
    CronParser,
    JobScheduler,
    daily,
    every_n_hours,
    every_n_minutes,
    monthly,
    weekly,
)


class TestCronJob:
    """Test CronJob dataclass."""

    def test_default_creation(self):
        job = CronJob(name="test", schedule="* * * * *", prompt="do stuff")
        assert job.name == "test"
        assert job.schedule == "* * * * *"
        assert job.prompt == "do stuff"
        assert job.enabled is True
        assert job.last_run is None
        assert job.next_run is None
        assert job.max_retries == 3
        assert job.retry_count == 0
        assert job.timeout == 300
        assert len(job.job_id) == 8

    def test_custom_fields(self):
        job = CronJob(
            name="custom",
            schedule="0 0 * * *",
            prompt="nightly",
            enabled=False,
            max_retries=5,
            timeout=600,
            metadata={"key": "value"},
        )
        assert job.enabled is False
        assert job.max_retries == 5
        assert job.timeout == 600
        assert job.metadata == {"key": "value"}

    def test_to_dict(self):
        job = CronJob(name="test", schedule="* * * * *", prompt="p")
        d = job.to_dict()
        assert d["name"] == "test"
        assert d["schedule"] == "* * * * *"
        assert d["job_id"] == job.job_id
        assert d["enabled"] is True

    def test_to_json(self):
        job = CronJob(name="test", schedule="* * * * *", prompt="p")
        j = job.to_json()
        parsed = json.loads(j)
        assert parsed["name"] == "test"
        assert parsed["job_id"] == job.job_id

    def test_round_trip(self):
        job = CronJob(name="rt", schedule="0 * * * *", prompt="p", timeout=120)
        d = job.to_dict()
        job2 = CronJob(**d)
        assert job2.name == job.name
        assert job2.schedule == job.schedule
        assert job2.job_id == job.job_id
        assert job2.timeout == job.timeout

    def test_unique_ids(self):
        jobs = [CronJob(name=f"j{i}", schedule="* * * * *", prompt="p") for i in range(10)]
        ids = [j.job_id for j in jobs]
        assert len(set(ids)) == 10  # all unique


class TestCronParser:
    """Test cron expression parsing."""

    def test_parse_valid(self):
        fields = CronParser.parse("*/5 * * * *")
        assert fields == ["*/5", "*", "*", "*", "*"]

    def test_parse_full(self):
        fields = CronParser.parse("0 0 1 1 0")
        assert fields == ["0", "0", "1", "1", "0"]

    def test_parse_invalid_too_few(self):
        with pytest.raises(ValueError, match="Expected 5 fields"):
            CronParser.parse("* * *")

    def test_parse_invalid_too_many(self):
        with pytest.raises(ValueError, match="Expected 5 fields"):
            CronParser.parse("* * * * * *")

    def test_parse_whitespace(self):
        fields = CronParser.parse("  */15  *  *  *  *  ")
        assert fields == ["*/15", "*", "*", "*", "*"]

    def test_validate_valid(self):
        assert CronParser.validate("*/5 * * * *") is True

    def test_validate_invalid(self):
        assert CronParser.validate("* * *") is False

    def test_matches_star(self):
        assert CronParser._matches_field("*", 30, 0, 59) is True
        assert CronParser._matches_field("*", 0, 0, 59) is True
        assert CronParser._matches_field("*", 59, 0, 59) is True

    def test_matches_step(self):
        assert CronParser._matches_field("*/15", 0, 0, 59) is True
        assert CronParser._matches_field("*/15", 15, 0, 59) is True
        assert CronParser._matches_field("*/15", 30, 0, 59) is True
        assert CronParser._matches_field("*/15", 45, 0, 59) is True
        assert CronParser._matches_field("*/15", 10, 0, 59) is False

    def test_matches_specific(self):
        assert CronParser._matches_field("30", 30, 0, 59) is True
        assert CronParser._matches_field("30", 45, 0, 59) is False

    def test_matches_range(self):
        assert CronParser._matches_field("9-17", 12, 0, 23) is True
        assert CronParser._matches_field("9-17", 8, 0, 23) is False
        assert CronParser._matches_field("9-17", 18, 0, 23) is False

    def test_matches_list(self):
        assert CronParser._matches_field("9,12,17", 9, 0, 23) is True
        assert CronParser._matches_field("9,12,17", 12, 0, 23) is True
        assert CronParser._matches_field("9,12,17", 10, 0, 23) is False

    def test_matches_month_name(self):
        assert CronParser._matches_field("jan", 1, 1, 12) is True
        assert CronParser._matches_field("dec", 12, 1, 12) is True

    def test_matches_day_name(self):
        assert CronParser._matches_field("mon", 1, 0, 6) is True
        assert CronParser._matches_field("sun", 0, 0, 6) is True

    def test_next_run_every_minute(self):
        base = datetime(2026, 1, 1, 12, 0, 0)
        next_time = CronParser.next_run_time("* * * * *", from_time=base)
        assert next_time == datetime(2026, 1, 1, 12, 1, 0)

    def test_next_run_every_hour(self):
        base = datetime(2026, 1, 1, 12, 30, 0)
        next_time = CronParser.next_run_time("0 * * * *", from_time=base)
        assert next_time == datetime(2026, 1, 1, 13, 0, 0)

    def test_next_run_daily(self):
        base = datetime(2026, 1, 1, 23, 30, 0)
        next_time = CronParser.next_run_time("0 0 * * *", from_time=base)
        assert next_time == datetime(2026, 1, 2, 0, 0, 0)

    def test_next_run_every_15_min(self):
        base = datetime(2026, 1, 1, 12, 10, 0)
        next_time = CronParser.next_run_time("*/15 * * * *", from_time=base)
        assert next_time == datetime(2026, 1, 1, 12, 15, 0)

    def test_next_run_specific_time(self):
        base = datetime(2026, 1, 1, 10, 0, 0)
        next_time = CronParser.next_run_time("30 14 * * *", from_time=base)
        assert next_time == datetime(2026, 1, 1, 14, 30, 0)

    def test_next_run_specific_day(self):
        # Mon Jan 5 2026 is a Thursday (weekday=3)
        base = datetime(2026, 1, 1, 0, 0, 0)
        next_time = CronParser.next_run_time("0 9 * * thu", from_time=base)
        assert next_time.hour == 9
        assert next_time.minute == 0


class TestJobScheduler:
    """Test job scheduler operations."""

    def test_add_job(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("test", "*/5 * * * *", "do stuff")
        assert job.name == "test"
        assert job.schedule == "*/5 * * * *"
        assert job.enabled is True
        assert job.next_run is not None

    def test_add_invalid_schedule(self):
        scheduler = JobScheduler()
        with pytest.raises(ValueError, match="Invalid cron schedule"):
            scheduler.add_job("bad", "* * *", "do stuff")

    def test_remove_job(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("test", "*/5 * * * *", "do stuff")
        assert scheduler.remove_job(job.job_id) is True
        assert scheduler.remove_job(job.job_id) is False

    def test_list_jobs(self):
        scheduler = JobScheduler()
        scheduler.add_job("a", "*/5 * * * *", "a")
        scheduler.add_job("b", "0 * * * *", "b")
        jobs = scheduler.list_jobs()
        assert len(jobs) == 2

    def test_get_job(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("test", "*/5 * * * *", "do")
        found = scheduler.get_job(job.job_id)
        assert found is not None
        assert found.name == "test"
        assert scheduler.get_job("nonexistent") is None

    def test_enable_disable(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("test", "*/5 * * * *", "do")
        assert scheduler.disable_job(job.job_id) is True
        assert job.enabled is False
        assert scheduler.enable_job(job.job_id) is True
        assert job.enabled is True
        assert scheduler.disable_job("fake") is False

    def test_get_due_jobs(self):
        scheduler = JobScheduler()
        # Job due immediately (next_run in past)
        job = scheduler.add_job("due", "*/5 * * * *", "do")
        job.next_run = time.time() - 100
        due = scheduler.get_due_jobs()
        assert len(due) == 1

    def test_get_due_jobs_skips_disabled(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("due", "*/5 * * * *", "do")
        job.next_run = time.time() - 100
        job.enabled = False
        due = scheduler.get_due_jobs()
        assert len(due) == 0

    def test_execute_job(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("test", "*/5 * * * *", "do stuff")
        result = scheduler.execute_job(job.job_id)
        assert "test" in result
        assert job.last_run is not None

    def test_execute_with_custom_executor(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("test", "*/5 * * * *", "do")
        result = scheduler.execute_job(job.job_id, executor=lambda j: f"exec:{j.name}")
        assert result == "exec:test"

    def test_execute_not_found(self):
        scheduler = JobScheduler()
        with pytest.raises(ValueError, match="Job not found"):
            scheduler.execute_job("nonexistent")

    def test_execute_error_retry(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("fail", "*/5 * * * *", "fail")
        with pytest.raises(Exception):
            scheduler.execute_job(
                job.job_id,
                executor=lambda j: (_ for _ in ()).throw(Exception("boom")),
            )
        assert job.retry_count == 1
        assert job.last_error == "boom"

    def test_execute_auto_disable_on_max_retries(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("fail", "*/5 * * * *", "fail")
        job.max_retries = 2
        fail_exec = lambda j: (_ for _ in ()).throw(Exception("err"))
        for _ in range(2):
            try:
                scheduler.execute_job(job.job_id, executor=fail_exec)
            except Exception:
                pass
        assert job.enabled is False

    def test_run_due_jobs(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("due", "*/5 * * * *", "do")
        job.next_run = time.time() - 100
        results = scheduler.run_due_jobs(executor=lambda j: "ok")
        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert results[0]["result"] == "ok"

    def test_get_summary(self):
        scheduler = JobScheduler()
        scheduler.add_job("a", "*/5 * * * *", "a")
        job_b = scheduler.add_job("b", "0 * * * *", "b")
        job_b.enabled = False
        summary = scheduler.get_summary()
        assert summary["total_jobs"] == 2
        assert summary["enabled_jobs"] == 1
        assert summary["disabled_jobs"] == 1

    def test_clear(self):
        scheduler = JobScheduler()
        scheduler.add_job("a", "*/5 * * * *", "a")
        scheduler.clear()
        assert len(scheduler.list_jobs()) == 0

    def test_persistence(self, tmp_path):
        state_file = str(tmp_path / "cron_state.json")
        scheduler = JobScheduler(state_path=state_file)
        job = scheduler.add_job("persist", "*/5 * * * *", "do")
        # Load from same path
        scheduler2 = JobScheduler(state_path=state_file)
        loaded = scheduler2.get_job(job.job_id)
        assert loaded is not None
        assert loaded.name == "persist"

    def test_persistence_invalid_file(self, tmp_path):
        state_file = str(tmp_path / "bad.json")
        Path(state_file).write_text("not json")
        scheduler = JobScheduler(state_path=state_file)
        assert len(scheduler.list_jobs()) == 0


class TestScheduleHelpers:
    """Test convenience schedule functions."""

    def test_every_n_minutes(self):
        assert every_n_minutes(5) == "*/5 * * * *"
        assert every_n_minutes(30) == "*/30 * * * *"

    def test_every_n_hours(self):
        assert every_n_hours(2) == "0 */2 * * *"
        assert every_n_hours(6) == "0 */6 * * *"

    def test_daily(self):
        assert daily() == "0 0 * * *"
        assert daily(9, 30) == "30 9 * * *"

    def test_weekly(self):
        assert weekly() == "0 0 * * 0"
        assert weekly(1, 8, 0) == "0 8 * * 1"

    def test_monthly(self):
        assert monthly() == "0 0 1 * *"
        assert monthly(15, 12, 30) == "30 12 15 * *"
