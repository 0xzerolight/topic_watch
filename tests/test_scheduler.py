"""Tests for the APScheduler integration."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.config import LLMSettings, Settings
from app.scheduler import _scheduled_check, _vacuum_db, start_scheduler, stop_scheduler


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "check_interval_hours": 4,
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestStartStopScheduler:
    """Tests for scheduler lifecycle."""

    async def test_start_creates_four_jobs(self) -> None:
        settings = _make_settings()
        scheduler = start_scheduler(settings)
        try:
            jobs = scheduler.get_jobs()
            job_ids = {j.id for j in jobs}
            assert "check_all_topics" in job_ids
            assert "recover_stuck_researching" in job_ids
            assert "vacuum_db" in job_ids
            assert "cleanup_old_articles" in job_ids
            assert len(jobs) == 4
        finally:
            stop_scheduler()

    async def test_check_job_ticks_every_minute(self) -> None:
        settings = _make_settings(check_interval_hours=12)
        scheduler = start_scheduler(settings)
        try:
            job = scheduler.get_job("check_all_topics")
            assert job is not None
            # Scheduler now ticks every minute; per-topic intervals are
            # handled by get_topics_due_for_check inside the callback.
            assert job.trigger.interval.total_seconds() == 60
        finally:
            stop_scheduler()

    async def test_check_job_has_default_jitter(self) -> None:
        settings = _make_settings()
        assert settings.scheduler_jitter_seconds == 30
        scheduler = start_scheduler(settings)
        try:
            job = scheduler.get_job("check_all_topics")
            assert job is not None
            assert job.trigger.jitter == 30
        finally:
            stop_scheduler()

    async def test_check_job_respects_custom_jitter(self) -> None:
        settings = _make_settings(scheduler_jitter_seconds=15)
        scheduler = start_scheduler(settings)
        try:
            job = scheduler.get_job("check_all_topics")
            assert job is not None
            assert job.trigger.jitter == 15
        finally:
            stop_scheduler()

    async def test_check_job_zero_jitter_is_valid(self) -> None:
        settings = _make_settings(scheduler_jitter_seconds=0)
        scheduler = start_scheduler(settings)
        try:
            job = scheduler.get_job("check_all_topics")
            assert job is not None
            assert job.trigger.jitter == 0
        finally:
            stop_scheduler()

    async def test_stop_scheduler_clears_global(self) -> None:
        import app.scheduler as sched_module

        settings = _make_settings()
        start_scheduler(settings)
        assert sched_module._scheduler is not None

        stop_scheduler()
        assert sched_module._scheduler is None

    def test_stop_scheduler_when_not_started(self) -> None:
        """stop_scheduler should not error when no scheduler exists."""
        stop_scheduler()  # Should not raise


class TestScheduledCheck:
    """Tests for the _scheduled_check callback."""

    async def test_calls_check_all_topics(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        from app.database import init_db

        init_db(db_path)
        settings = _make_settings()

        with patch(
            "app.scheduler.check_all_topics",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_check:
            await _scheduled_check(settings, db_path)

        mock_check.assert_called_once()

    async def test_does_not_raise_on_error(self, tmp_path: Path) -> None:
        """Scheduled check should catch exceptions, not crash the scheduler."""
        db_path = tmp_path / "test.db"
        from app.database import init_db

        init_db(db_path)
        settings = _make_settings()

        with patch(
            "app.scheduler.check_all_topics",
            new_callable=AsyncMock,
            side_effect=Exception("DB error"),
        ):
            # Should not raise
            await _scheduled_check(settings, db_path)


class TestVacuumDb:
    """Tests for the _vacuum_db callback."""

    async def test_executes_vacuum(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        from app.database import init_db

        init_db(db_path)

        # Should not raise
        await _vacuum_db(db_path)

    async def test_does_not_raise_on_error(self) -> None:
        """VACUUM failure should be caught, not crash the scheduler."""
        # Non-existent path will cause an error
        await _vacuum_db(Path("/nonexistent/path.db"))
