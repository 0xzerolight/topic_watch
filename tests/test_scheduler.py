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

    async def test_check_job_serializes_overlapping_ticks(self) -> None:
        """OVH-171: max_instances=1 and coalesce guard against overlapping check cycles."""
        settings = _make_settings()
        scheduler = start_scheduler(settings)
        try:
            job = scheduler.get_job("check_all_topics")
            assert job is not None
            assert job.max_instances == 1
            assert job.coalesce is True
        finally:
            stop_scheduler()

    async def test_maintenance_jobs_have_generous_misfire_grace(self) -> None:
        """OVH-029: cron maintenance jobs survive a slept/woken host (large misfire grace)."""
        settings = _make_settings()
        scheduler = start_scheduler(settings)
        try:
            for job_id in ("vacuum_db", "cleanup_old_articles"):
                job = scheduler.get_job(job_id)
                assert job is not None
                # At least an hour so a delayed/woken host still runs missed maintenance.
                assert job.misfire_grace_time is not None
                assert job.misfire_grace_time >= 3600
        finally:
            stop_scheduler()

    async def test_start_is_idempotent_no_leak(self) -> None:
        """OVH-067/125: a second start_scheduler shuts the first down, no orphan."""
        import asyncio

        import app.scheduler as sched_module

        settings = _make_settings()
        first = start_scheduler(settings)
        assert first.running
        second = start_scheduler(settings)
        try:
            # The single ownership token now points at the new scheduler.
            assert second.running
            assert sched_module._scheduler is second
            # The previously-running scheduler is shut down (AsyncIOScheduler defers the
            # state flip to the loop), so it leaves no orphaned live ticks.
            await asyncio.sleep(0)
            assert not first.running
        finally:
            stop_scheduler()

    async def test_check_job_reads_live_settings_from_app(self) -> None:
        """OVH-015/036: when wired to an app, the tick reads settings from app.state."""
        from types import SimpleNamespace

        captured: list[Settings] = []

        async def fake_run_check_cycle(settings, db_path=None):
            captured.append(settings)

        initial = _make_settings(check_interval="4h")
        app = SimpleNamespace(state=SimpleNamespace(settings=initial, db_path=None))

        scheduler = start_scheduler(initial, app=app)
        try:
            # Simulate an in-place settings edit after the scheduler is running.
            edited = _make_settings(check_interval="12h")
            app.state.settings = edited

            with (
                patch("app.scheduler._run_check_cycle", side_effect=fake_run_check_cycle),
                # Without this, the tick's _init_new_topics(settings, None) runs
                # against the real data/ DB (db_path=None). This test only asserts
                # settings propagation to the check cycle, so stub it out.
                patch("app.scheduler._init_new_topics", new_callable=AsyncMock),
            ):
                job = scheduler.get_job("check_all_topics")
                assert job is not None
                await job.func(*job.args, **job.kwargs)

            assert captured, "the check cycle should have run"
            # The tick used the edited settings, not the ones bound at start.
            assert captured[-1].check_interval == "12h"
        finally:
            stop_scheduler()


class TestTickWrappers:
    """OVH-015/036: tick wrappers resolve live settings/db_path from app.state."""

    async def test_tick_recover_uses_live_db_path(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from app.scheduler import _tick_recover

        live_db = tmp_path / "live.db"
        app = SimpleNamespace(state=SimpleNamespace(settings=_make_settings(), db_path=live_db))
        with patch("app.scheduler._recover_stuck", new_callable=AsyncMock) as mock_recover:
            await _tick_recover(timeout_minutes=15, db_path=None, app=app)
        mock_recover.assert_awaited_once()
        assert mock_recover.await_args.kwargs["db_path"] == live_db

    async def test_tick_vacuum_uses_live_db_path(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from app.scheduler import _tick_vacuum

        live_db = tmp_path / "live.db"
        app = SimpleNamespace(state=SimpleNamespace(settings=_make_settings(), db_path=live_db))
        with patch("app.scheduler._vacuum_db", new_callable=AsyncMock) as mock_vacuum:
            await _tick_vacuum(db_path=None, app=app)
        mock_vacuum.assert_awaited_once_with(live_db)

    async def test_tick_cleanup_uses_live_settings(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from app.scheduler import _tick_cleanup

        edited = _make_settings(article_retention_days=7)
        app = SimpleNamespace(state=SimpleNamespace(settings=edited, db_path=tmp_path / "live.db"))
        with patch("app.scheduler._cleanup_old_articles", new_callable=AsyncMock) as mock_cleanup:
            await _tick_cleanup(settings=_make_settings(article_retention_days=90), db_path=None, app=app)
        mock_cleanup.assert_awaited_once()
        passed_settings = mock_cleanup.await_args.args[0]
        assert passed_settings.article_retention_days == 7

    async def test_tick_falls_back_to_bound_settings_without_app(self, tmp_path: Path) -> None:
        from app.scheduler import _tick_cleanup

        bound = _make_settings(article_retention_days=42)
        with patch("app.scheduler._cleanup_old_articles", new_callable=AsyncMock) as mock_cleanup:
            await _tick_cleanup(settings=bound, db_path=tmp_path / "x.db", app=None)
        passed_settings = mock_cleanup.await_args.args[0]
        assert passed_settings.article_retention_days == 42


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

    async def test_passes_a_path_not_a_connection_per_topic(self, tmp_path: Path) -> None:
        """AUG-136: the cycle hands each check a path, never an open connection.

        It used to wrap every per-topic check in ``with get_db(...)``, so the
        handle — and, through the feed-health callback, the single WAL writer —
        stayed open across that topic's fetch, LLM and notification awaits.
        """
        db_path = tmp_path / "test.db"
        from app.database import get_connection, init_db

        init_db(db_path)
        settings = _make_settings()

        # Two due topics, each with their own check.
        conn = get_connection(db_path)
        topics = [_make_ready_topic(conn, name=f"T{i}") for i in range(2)]
        conn.close()

        calls: list[tuple] = []

        async def fake_check_topic(topic, s, **kwargs):
            calls.append((topic.id, args_seen := kwargs.get("db_path")))
            assert args_seen == db_path
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

        assert len(calls) == 2
        assert {t for t, _ in calls} == {t.id for t in topics}


class TestVacuumDb:
    """Tests for the _vacuum_db callback."""

    async def test_executes_vacuum(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        from app.database import init_db

        init_db(db_path)

        # Should not raise
        await _vacuum_db(db_path)

    async def test_does_not_raise_on_error(self, caplog) -> None:
        """VACUUM failure should be caught, not crash the scheduler.

        AUG-056: ``get_connection`` creates missing parent directories, so
        ``/nonexistent/path.db`` only fails under an unprivileged runner —
        under root it succeeds and writes under the filesystem root, and either
        way the old test asserted neither the warning nor which branch ran.
        Mock the failure directly so the error path is exercised deterministically.
        """
        import logging

        with (
            patch("app.scheduler._vacuum_db_sync", side_effect=OSError("disk full")),
            caplog.at_level(logging.WARNING, logger="app.scheduler"),
        ):
            await _vacuum_db(Path("unused.db"))  # must not raise

        assert any("VACUUM failed" in r.message for r in caplog.records)

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


class TestInitNewTopics:
    """Tests for the scheduler's gradual NEW-topic init guard (OVH-032)."""

    async def test_init_claims_new_topic_then_runs(self, tmp_path: Path) -> None:
        """A NEW topic is atomically claimed (NEW -> RESEARCHING) before init runs."""
        from unittest.mock import AsyncMock

        from app.crud import create_topic, get_topic
        from app.database import get_db, init_db
        from app.models import Topic, TopicStatus
        from app.scheduler import _init_new_topics
        from app.web.state import _checking_state

        db_path = tmp_path / "init.db"
        init_db(db_path)
        with get_db(db_path) as conn:
            topic = create_topic(conn, Topic(name="Pending", description="d", status=TopicStatus.NEW))
            conn.commit()

        _checking_state._topics.clear()
        captured: list[int] = []

        async def _fake_init(t, settings, **kwargs):
            # The claim has already flipped the row to RESEARCHING.
            captured.append(t.id)
            assert t.status == TopicStatus.RESEARCHING

        try:
            with patch("app.scheduler.initialize_new_topic", new=AsyncMock(side_effect=_fake_init)):
                await _init_new_topics(_make_settings(), db_path)
        finally:
            _checking_state._topics.clear()

        assert captured == [topic.id]
        with get_db(db_path) as conn:
            assert get_topic(conn, topic.id).status == TopicStatus.RESEARCHING

    async def test_init_skips_when_topic_no_longer_new(self, tmp_path: Path) -> None:
        """If the topic was claimed elsewhere (no longer NEW), init does not run (OVH-032)."""
        from unittest.mock import AsyncMock

        from app.crud import create_topic
        from app.database import get_db, init_db
        from app.models import Topic, TopicStatus
        from app.scheduler import _init_new_topics
        from app.web.state import _checking_state

        db_path = tmp_path / "init2.db"
        init_db(db_path)
        with get_db(db_path) as conn:
            create_topic(conn, Topic(name="Pending", description="d", status=TopicStatus.NEW))
            conn.commit()

        _checking_state._topics.clear()
        init_mock = AsyncMock()

        # Simulate a concurrent winner: flip the row out of NEW just before the claim,
        # by patching claim_new_topic_for_init to report a lost race.
        with (
            patch("app.scheduler.claim_new_topic_for_init", return_value=False),
            patch("app.scheduler.initialize_new_topic", new=init_mock),
        ):
            await _init_new_topics(_make_settings(), db_path)

        init_mock.assert_not_awaited()
        _checking_state._topics.clear()

    async def test_init_skips_when_in_process_guard_held(self, tmp_path: Path) -> None:
        """If the in-process guard is already held (web init in flight), the tick skips."""
        from unittest.mock import AsyncMock

        from app.crud import create_topic
        from app.database import get_db, init_db
        from app.models import Topic, TopicStatus
        from app.scheduler import _init_new_topics
        from app.web.state import _checking_state

        db_path = tmp_path / "init3.db"
        init_db(db_path)
        with get_db(db_path) as conn:
            topic = create_topic(conn, Topic(name="Pending", description="d", status=TopicStatus.NEW))
            conn.commit()

        _checking_state._topics.clear()
        init_mock = AsyncMock()
        try:
            await _checking_state.start_check(topic.id)
            with patch("app.scheduler.initialize_new_topic", new=init_mock):
                await _init_new_topics(_make_settings(), db_path)
            init_mock.assert_not_awaited()
        finally:
            _checking_state._topics.clear()


class TestGradualInitRespectsPausedTopics:
    """AUG-140: automatic initialization never spends on a paused topic."""

    async def test_inactive_new_topic_is_never_initialized(self, tmp_path: Path) -> None:
        from app.crud import create_topic, get_topic
        from app.database import get_db, init_db
        from app.models import Topic, TopicStatus
        from app.scheduler import _init_new_topics

        db_path = tmp_path / "paused.db"
        init_db(db_path)
        with get_db(db_path) as conn:
            topic = create_topic(
                conn,
                Topic(name="Paused", description="d", status=TopicStatus.NEW, is_active=False),
            )
            conn.commit()

        init_mock = AsyncMock()
        with patch("app.scheduler.initialize_new_topic", new=init_mock):
            await _init_new_topics(_make_settings(), db_path)

        init_mock.assert_not_awaited()
        with get_db(db_path) as conn:
            assert get_topic(conn, topic.id).status == TopicStatus.NEW

    async def test_pausing_between_selection_and_claim_stops_the_init(self, tmp_path: Path) -> None:
        """The claim repeats the is_active filter, so the toggle race is closed too."""
        from app.crud import claim_new_topic_for_init, create_topic, get_topic
        from app.database import get_db, init_db
        from app.models import Topic, TopicStatus
        from app.scheduler import _init_new_topics

        db_path = tmp_path / "paused_race.db"
        init_db(db_path)
        with get_db(db_path) as conn:
            topic = create_topic(conn, Topic(name="Racy", description="d", status=TopicStatus.NEW))
            conn.commit()

        def _pause_then_claim(conn, topic_id):
            # The user hits Disable in the window between selection and claim.
            conn.execute("UPDATE topics SET is_active = 0 WHERE id = ?", (topic_id,))
            conn.commit()
            return claim_new_topic_for_init(conn, topic_id)

        init_mock = AsyncMock()
        with (
            patch("app.scheduler.claim_new_topic_for_init", new=_pause_then_claim),
            patch("app.scheduler.initialize_new_topic", new=init_mock),
        ):
            await _init_new_topics(_make_settings(), db_path)

        init_mock.assert_not_awaited()
        with get_db(db_path) as conn:
            assert get_topic(conn, topic.id).status == TopicStatus.NEW


class TestGradualInitIsBounded:
    """AUG-139: a scheduler init cannot outlive the stuck-recovery window."""

    async def test_hanging_init_times_out_and_lands_error(self, tmp_path: Path) -> None:
        from app.crud import create_topic, get_topic
        from app.database import get_db, init_db
        from app.models import Topic, TopicStatus
        from app.scheduler import _init_new_topics

        db_path = tmp_path / "bounded.db"
        init_db(db_path)
        with get_db(db_path) as conn:
            topic = create_topic(conn, Topic(name="Hangs", description="d", status=TopicStatus.NEW))
            conn.commit()

        async def _hang(*args, **kwargs):
            import asyncio

            await asyncio.sleep(9999)

        with (
            patch("app.scheduler._INIT_TIMEOUT_SECONDS", 0.05),
            patch("app.checker.fetch_new_articles_for_topic", side_effect=_hang),
        ):
            await _init_new_topics(_make_settings(), db_path)

        with get_db(db_path) as conn:
            refreshed = get_topic(conn, topic.id)
        assert refreshed.status == TopicStatus.ERROR
        assert refreshed.error_message == "Research timed out. Click Retry."


class TestLifespanShutdown:
    """AUG-265: the scheduler stops however the lifespan context ends."""

    async def _run_lifespan(self, monkeypatch, *, boom: bool) -> list[str]:
        import pytest

        from app.main import app, lifespan

        calls: list[str] = []
        monkeypatch.setattr("app.main.load_settings", _make_settings)
        monkeypatch.setattr("app.main.start_scheduler", lambda *a, **k: calls.append("start"))
        monkeypatch.setattr("app.main.stop_scheduler", lambda: calls.append("stop"))

        if not boom:
            async with lifespan(app):
                pass
            return calls

        with pytest.raises(RuntimeError, match="serving blew up"):
            async with lifespan(app):
                raise RuntimeError("serving blew up")
        return calls

    async def test_clean_exit_stops_the_scheduler(self, monkeypatch) -> None:
        assert await self._run_lifespan(monkeypatch, boom=False) == ["start", "stop"]

    async def test_exceptional_exit_still_stops_the_scheduler(self, monkeypatch) -> None:
        """The exception is thrown in at the yield; cleanup after it never ran."""
        assert await self._run_lifespan(monkeypatch, boom=True) == ["start", "stop"]

    async def test_cancelled_exit_still_stops_the_scheduler(self, monkeypatch) -> None:
        import pytest

        from app.main import app, lifespan

        calls: list[str] = []
        monkeypatch.setattr("app.main.load_settings", _make_settings)
        monkeypatch.setattr("app.main.start_scheduler", lambda *a, **k: calls.append("start"))
        monkeypatch.setattr("app.main.stop_scheduler", lambda: calls.append("stop"))

        import asyncio

        with pytest.raises(asyncio.CancelledError):
            async with lifespan(app):
                raise asyncio.CancelledError()

        assert calls == ["start", "stop"]


class TestDeliveryLedgerRetention:
    """AUG-153: the delivery ledger is pruned by the existing maintenance tick."""

    async def test_cleanup_tick_prunes_finished_intents(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from app.scheduler import _tick_cleanup

        app = SimpleNamespace(state=SimpleNamespace(settings=_make_settings(), db_path=tmp_path / "live.db"))
        with (
            patch("app.scheduler._cleanup_old_articles", new_callable=AsyncMock),
            patch("app.scheduler.delete_old_delivery_intents", return_value=3) as mock_prune,
        ):
            await _tick_cleanup(settings=_make_settings(), db_path=None, app=app)

        mock_prune.assert_called_once()
        assert mock_prune.call_args.args[1] == 30

    def test_only_finished_intents_older_than_the_window_are_removed(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime, timedelta

        from app.crud import (
            DELIVERY_INTENT_RETENTION_DAYS,
            create_pending_notification,
            create_topic,
            create_webhook_intents,
            delete_old_delivery_intents,
        )
        from app.database import get_connection, init_db
        from app.models import PendingNotification, PendingWebhook, Topic, TopicStatus

        db_path = tmp_path / "ledger.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            topic = create_topic(conn, Topic(name="T", description="d", status=TopicStatus.READY))
            old = datetime.now(UTC) - timedelta(days=DELIVERY_INTENT_RETENTION_DAYS + 1)
            rows = (
                ("old-sent", "sent", old),
                ("old-abandoned", "abandoned", old),
                ("old-pending", "pending", old),
                ("fresh-sent", "sent", datetime.now(UTC)),
            )
            for title, status, created in rows:
                intent = create_pending_notification(
                    conn,
                    PendingNotification(topic_id=topic.id, title=title, body="B", url="json://x", created_at=created),
                )
                conn.execute("UPDATE pending_notifications SET status = ? WHERE id = ?", (status, intent.id))
                # Both halves of the ledger are pruned by the same call.
                (hook_id,) = create_webhook_intents(
                    conn,
                    [
                        PendingWebhook(
                            topic_id=topic.id,
                            url=f"https://hooks.example.com/{title}",
                            payload={"t": title},
                            created_at=created,
                        )
                    ],
                )
                conn.execute("UPDATE pending_webhooks SET status = ? WHERE id = ?", (status, hook_id))
            conn.commit()

            removed = delete_old_delivery_intents(conn, DELIVERY_INTENT_RETENTION_DAYS)
            conn.commit()

            assert removed == 4
            titles = {r["title"] for r in conn.execute("SELECT title FROM pending_notifications")}
            # An undelivered intent still owes a delivery, however old it is.
            assert titles == {"old-pending", "fresh-sent"}
            hooks = {r["url"].rsplit("/", 1)[-1] for r in conn.execute("SELECT url FROM pending_webhooks")}
            assert hooks == {"old-pending", "fresh-sent"}
        finally:
            conn.close()


class TestStartupHeartbeatReset:
    """AUG-260: startup reconciles heartbeat state when the feature is off."""

    async def _latched_topic(self, db_path: Path):
        from datetime import UTC, datetime

        from app.crud import claim_heartbeat_alert, create_notification_intents
        from app.database import get_db, init_db
        from app.models import NotificationKind, PendingNotification, Topic, TopicStatus

        init_db(db_path)
        with get_db(db_path) as conn:
            from app.crud import create_topic

            topic = create_topic(conn, Topic(name="Outage", description="d", status=TopicStatus.READY))
            claim_heartbeat_alert(conn, topic.id, datetime.now(UTC))
            create_notification_intents(
                conn,
                [
                    PendingNotification(
                        topic_id=topic.id,
                        title="Topic Watch: Outage (sources failing)",
                        body="body",
                        url="json://localhost",
                        kind=NotificationKind.HEARTBEAT_ALERT,
                        latch_value="2026-08-20T10:00:00+00:00",
                    )
                ],
            )
            conn.commit()
        return topic

    async def _boot(self, monkeypatch, tmp_path: Path, threshold: int):
        from app.main import app, lifespan

        db_path = tmp_path / "lifespan.db"
        topic = await self._latched_topic(db_path)
        monkeypatch.setattr(
            "app.main.load_settings", lambda *a, **k: _make_settings(silence_heartbeat_checks=threshold)
        )
        monkeypatch.setattr("app.main.start_scheduler", lambda *a, **k: None)
        monkeypatch.setattr("app.main.stop_scheduler", lambda: None)

        async with lifespan(app):
            pass

        from app.database import get_db

        with get_db(db_path) as conn:
            latch = conn.execute("SELECT heartbeat_alerted_at FROM topics WHERE id = ?", (topic.id,)).fetchone()[0]
            statuses = [
                row[0]
                for row in conn.execute(
                    "SELECT status FROM pending_notifications WHERE topic_id = ?", (topic.id,)
                ).fetchall()
            ]
        return latch, statuses

    async def test_disabled_at_startup_clears_parked_state(self, monkeypatch, tmp_path: Path) -> None:
        latch, statuses = await self._boot(monkeypatch, tmp_path, threshold=0)
        assert latch is None
        assert statuses == ["revoked"]

    async def test_enabled_at_startup_preserves_state(self, monkeypatch, tmp_path: Path) -> None:
        latch, statuses = await self._boot(monkeypatch, tmp_path, threshold=3)
        assert latch is not None
        assert statuses == ["pending"]


class TestRetryDrainDoesNotStarveDueTopics:
    """AUG-027: a queued retry backlog cannot hold every due topic behind it."""

    async def test_due_topics_run_alongside_the_retry_drain(self, db_conn, db_path: Path) -> None:
        import asyncio

        from app.checker import check_all_topics

        _make_ready_topic(db_conn)
        drain_started = asyncio.Event()
        topic_checked = asyncio.Event()

        async def _slow_drain(*args, **kwargs) -> None:
            drain_started.set()
            # The backlog only finishes once a due topic has been checked: with the
            # drain in front of the cycle this never happens and the wait times out.
            await asyncio.wait_for(topic_checked.wait(), timeout=5)

        async def _fake_check(topic, settings, *, db_path=None, guard=True):
            await drain_started.wait()
            topic_checked.set()
            return None

        with (
            patch("app.checker.retry_pending_notifications", _slow_drain),
            patch("app.checker.retry_pending_webhooks", new=AsyncMock()),
            patch("app.checker.check_topic", _fake_check),
        ):
            await asyncio.wait_for(check_all_topics(_make_settings(), db_path), timeout=5)

        assert topic_checked.is_set()

    async def test_the_drain_still_runs_with_no_due_topics(self, db_conn, db_path: Path) -> None:
        from app.checker import check_all_topics

        drain = AsyncMock()
        with (
            patch("app.checker.retry_pending_notifications", drain),
            patch("app.checker.retry_pending_webhooks", new=AsyncMock()),
        ):
            assert await check_all_topics(_make_settings(), db_path) == []
        drain.assert_awaited_once()
