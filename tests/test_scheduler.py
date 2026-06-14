"""Tests for the APScheduler integration."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.config import LLMSettings, Settings
from app.scheduler import _scheduled_check, _vacuum_db, start_scheduler, stop_scheduler


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "check_interval": "4h",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_ready_topic(conn, name: str = "Topic"):
    from app.crud import create_topic
    from app.models import Topic, TopicStatus

    topic = create_topic(conn, Topic(name=name, description="d", status=TopicStatus.READY))
    conn.commit()
    return topic


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
        settings = _make_settings(check_interval="12h")
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

    async def test_runs_check_cycle(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        from app.database import init_db

        init_db(db_path)
        settings = _make_settings()

        with patch(
            "app.scheduler._run_check_cycle",
            new_callable=AsyncMock,
        ) as mock_cycle:
            await _scheduled_check(settings, db_path)

        mock_cycle.assert_awaited_once()

    async def test_does_not_raise_on_error(self, tmp_path: Path) -> None:
        """Scheduled check should catch exceptions, not crash the scheduler."""
        db_path = tmp_path / "test.db"
        from app.database import init_db

        init_db(db_path)
        settings = _make_settings()

        with patch(
            "app.scheduler._run_check_cycle",
            new_callable=AsyncMock,
            side_effect=Exception("DB error"),
        ):
            # Should not raise
            await _scheduled_check(settings, db_path)

    async def test_uses_fresh_connection_per_topic(self, tmp_path: Path) -> None:
        """The check cycle must open a new short-lived connection per topic check
        rather than holding one connection across the whole cycle."""
        db_path = tmp_path / "test.db"
        from app.database import get_connection, init_db

        init_db(db_path)
        settings = _make_settings()

        # Two due topics, each with their own check.
        conn = get_connection(db_path)
        topics = [_make_ready_topic(conn, name=f"T{i}") for i in range(2)]
        conn.close()

        seen_conn_ids: list[int] = []

        async def fake_check_topic(topic, c, s):
            seen_conn_ids.append(id(c))
            from app.models import CheckResult

            return CheckResult(topic_id=topic.id)

        from app.scheduler import _run_check_cycle

        with (
            patch("app.checker.check_topic", side_effect=fake_check_topic),
            patch("app.checker.retry_pending_notifications", new_callable=AsyncMock),
            patch("app.checker.retry_pending_webhooks", new_callable=AsyncMock),
            patch(
                "app.checker.get_topics_due_for_check",
                return_value=topics,
            ),
        ):
            await _run_check_cycle(settings, db_path)

        # Each topic check received a distinct connection object.
        assert len(seen_conn_ids) == 2
        assert len(set(seen_conn_ids)) == 2


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

    async def test_runs_in_thread(self, tmp_path: Path) -> None:
        """VACUUM must run off the event loop via asyncio.to_thread."""
        db_path = tmp_path / "test.db"
        from app.database import init_db

        init_db(db_path)

        with patch("app.scheduler.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            await _vacuum_db(db_path)

        mock_to_thread.assert_awaited_once()
        # The blocking VACUUM helper is what's offloaded to the thread.
        from app.scheduler import _vacuum_db_sync

        assert mock_to_thread.await_args.args[0] is _vacuum_db_sync
