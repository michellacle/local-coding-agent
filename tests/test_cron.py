"""Tests for cron — CronParser, CronJob, JobScheduler."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from local_agent.cron import (
    CronJob,
    CronParser,
    JobScheduler,
    every_n_minutes,
    every_n_hours,
    daily,
    weekly,
    monthly,
)


class TestCronParser:
    """Test cron expression parsing."""

    def test_parse_valid_expression(self):
        fields = CronParser.parse("*/5 * * * *")
        assert len(fields) == 5
        assert fields == ["*/5", "*", "*", "*", "*"]

    def test_parse_invalid_expression(self):
        with pytest.raises(ValueError, match="Invalid cron expression"):
            CronParser.parse("* * *")

    def test_validate_valid(self):
        assert CronParser.validate("*/5 * * * *") is True
        assert CronParser.validate("0 9 * * 1") is True

    def test_validate_invalid(self):
        assert CronParser.validate("* *") is False

    def test_matches_star(self):
        assert CronParser._matches_field("*", 5, 0, 59) is True
        assert CronParser._matches_field("*", 0, 0, 59) is True

    def test_matches_specific_value(self):
        assert CronParser._matches_field("5", 5, 0, 59) is True
        assert CronParser._matches_field("5", 6, 0, 59) is False

    def test_matches_step(self):
        assert CronParser._matches_field("*/5", 0, 0, 59) is True
        assert CronParser._matches_field("*/5", 5, 0, 59) is True
        assert CronParser._matches_field("*/5", 10, 0, 59) is True
        assert CronParser._matches_field("*/5", 7, 0, 59) is False

    def test_matches_range(self):
        assert CronParser._matches_field("1-5", 3, 0, 59) is True
        assert CronParser._matches_field("1-5", 7, 0, 59) is False

    def test_matches_list(self):
        assert CronParser._matches_field("1,5,10", 5, 0, 59) is True
        assert CronParser._matches_field("1,5,10", 3, 0, 59) is False

    def test_next_run_time_every_minute(self):
        now = datetime(2026, 5, 22, 10, 30, 0)
        next_run = CronParser.next_run_time("* * * * *", from_time=now)
        assert next_run == datetime(2026, 5, 22, 10, 31, 0)

    def test_next_run_time_every_5_minutes(self):
        now = datetime(2026, 5, 22, 10, 30, 0)
        next_run = CronParser.next_run_time("*/5 * * * *", from_time=now)
        # 10:31 is not divisible by 5, so should be 10:35
        assert next_run == datetime(2026, 5, 22, 10, 35, 0)

    def test_next_run_time_specific_hour(self):
        now = datetime(2026, 5, 22, 8, 0, 0)
        next_run = CronParser.next_run_time("0 9 * * *", from_time=now)
        assert next_run == datetime(2026, 5, 22, 9, 0, 0)

    def test_next_run_time_monday(self):
        # May 22, 2026 is a Friday (weekday=4)
        now = datetime(2026, 5, 22, 8, 0, 0)
        # Monday = weekday 0 = Monday
        next_run = CronParser.next_run_time("0 9 * * 1", from_time=now)
        assert next_run.hour == 9
        assert next_run.minute == 0

    def test_month_name_substitution(self):
        # Names are substituted at matching time, not parse time
        assert CronParser._matches_field("jan", 1, 1, 12) is True
        assert CronParser._matches_field("jan", 2, 1, 12) is False

    def test_day_name_substitution(self):
        # Names are substituted at matching time, not parse time
        assert CronParser._matches_field("mon", 1, 0, 6) is True
        assert CronParser._matches_field("mon", 0, 0, 6) is False


class TestCronJob:
    """Test CronJob dataclass."""

    def test_create_job(self):
        job = CronJob(name="test", schedule="*/5 * * * *", prompt="do something")
        assert job.name == "test"
        assert job.enabled is True
        assert job.retry_count == 0

    def test_job_to_dict(self):
        job = CronJob(name="test")
        d = job.to_dict()
        assert d["name"] == "test"
        assert d["enabled"] is True

    def test_job_to_json(self):
        job = CronJob(name="test")
        j = job.to_json()
        parsed = json.loads(j)
        assert parsed["name"] == "test"

    def test_job_auto_id(self):
        j1 = CronJob()
        j2 = CronJob()
        assert j1.job_id != j2.job_id


class TestJobScheduler:
    """Test JobScheduler."""

    def test_add_job(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("Test Job", "*/5 * * * *", "Do something")
        assert job.name == "Test Job"
        assert job.next_run is not None

    def test_add_job_invalid_schedule(self):
        scheduler = JobScheduler()
        with pytest.raises(ValueError, match="Invalid cron schedule"):
            scheduler.add_job("Bad Job", "* *", "Do something")

    def test_remove_job(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("Test Job", "*/5 * * * *", "Do something")
        assert scheduler.remove_job(job.job_id) is True
        assert scheduler.get_job(job.job_id) is None

    def test_remove_nonexistent_job(self):
        scheduler = JobScheduler()
        assert scheduler.remove_job("nonexistent") is False

    def test_enable_disable_job(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("Test Job", "*/5 * * * *", "Do something")
        assert scheduler.disable_job(job.job_id) is True
        assert scheduler.get_job(job.job_id).enabled is False
        assert scheduler.enable_job(job.job_id) is True
        assert scheduler.get_job(job.job_id).enabled is True

    def test_list_jobs(self):
        scheduler = JobScheduler()
        scheduler.add_job("Job 1", "*/5 * * * *", "Prompt 1")
        scheduler.add_job("Job 2", "*/10 * * * *", "Prompt 2")
        jobs = scheduler.list_jobs()
        assert len(jobs) == 2

    def test_execute_job(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("Test Job", "*/5 * * * *", "Do something")
        result = scheduler.execute_job(job.job_id)
        assert "Test Job" in result
        assert job.last_run is not None

    def test_execute_job_custom_executor(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("Test Job", "*/5 * * * *", "Do something")

        def custom_executor(j):
            return f"Executed: {j.name}"

        result = scheduler.execute_job(job.job_id, executor=custom_executor)
        assert result == "Executed: Test Job"

    def test_execute_nonexistent_job(self):
        scheduler = JobScheduler()
        with pytest.raises(ValueError, match="Job not found"):
            scheduler.execute_job("nonexistent")

    def test_execute_job_with_error(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("Test Job", "*/5 * * * *", "Do something")

        def failing_executor(j):
            raise RuntimeError("Something went wrong")

        with pytest.raises(RuntimeError, match="Something went wrong"):
            scheduler.execute_job(job.job_id, executor=failing_executor)

        updated = scheduler.get_job(job.job_id)
        assert updated.last_error == "Something went wrong"
        assert updated.retry_count == 1

    def test_job_disabled_after_max_retries(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("Test Job", "*/5 * * * *", "Do something", timeout=10)
        job.max_retries = 2

        def failing_executor(j):
            raise RuntimeError("fail")

        # First retry
        with pytest.raises(RuntimeError):
            scheduler.execute_job(job.job_id, executor=failing_executor)
        # Second retry (should disable)
        with pytest.raises(RuntimeError):
            scheduler.execute_job(job.job_id, executor=failing_executor)

        updated = scheduler.get_job(job.job_id)
        assert updated.enabled is False

    def test_get_due_jobs(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("Test Job", "*/5 * * * *", "Do something")
        # Set next_run to the past
        job.next_run = time.time() - 100
        scheduler._jobs[job.job_id] = job

        due = scheduler.get_due_jobs()
        assert len(due) == 1

    def test_get_summary(self):
        scheduler = JobScheduler()
        scheduler.add_job("Job 1", "*/5 * * * *", "P1")
        scheduler.add_job("Job 2", "*/10 * * * *", "P2")
        summary = scheduler.get_summary()
        assert summary["total_jobs"] == 2
        assert summary["enabled_jobs"] == 2

    def test_clear(self):
        scheduler = JobScheduler()
        scheduler.add_job("Test Job", "*/5 * * * *", "Do something")
        scheduler.clear()
        assert len(scheduler.list_jobs()) == 0

    def test_persistent_state(self, tmp_path):
        state_file = tmp_path / "scheduler.json"
        scheduler = JobScheduler(state_path=str(state_file))
        scheduler.add_job("Test Job", "*/5 * * * *", "Do something")
        del scheduler

        # Reopen
        scheduler2 = JobScheduler(state_path=str(state_file))
        jobs = scheduler2.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].name == "Test Job"

    def test_run_due_jobs(self):
        scheduler = JobScheduler()
        job = scheduler.add_job("Test Job", "*/5 * * * *", "Do something")
        job.next_run = time.time() - 100
        scheduler._jobs[job.job_id] = job

        results = scheduler.run_due_jobs()
        assert len(results) == 1
        assert results[0]["status"] == "success"


class TestConvenienceFunctions:
    """Test convenience schedule functions."""

    def test_every_n_minutes(self):
        assert every_n_minutes(5) == "*/5 * * * *"
        assert every_n_minutes(30) == "*/30 * * * *"

    def test_every_n_hours(self):
        assert every_n_hours(2) == "0 */2 * * *"
        assert every_n_hours(6) == "0 */6 * * *"

    def test_daily(self):
        assert daily(hour=9, minute=30) == "30 9 * * *"
        assert daily() == "0 0 * * *"

    def test_weekly(self):
        assert weekly(day=1, hour=10) == "0 10 * * 1"
        assert weekly() == "0 0 * * 0"

    def test_monthly(self):
        assert monthly(day=15, hour=8) == "0 8 15 * *"
        assert monthly() == "0 0 1 * *"
