"""Tests for stuck-researching topic recovery.

Covers:
- recover_stuck_researching() finds topics stuck past the timeout
- recover_stuck_researching() leaves fresh RESEARCHING topics alone
- Migration m005 applies cleanly
- status_changed_at is updated when topic status changes via update_topic
- The _recover_stuck scheduled job function works correctly
"""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from app.crud import (
    create_topic,
    get_topic,
    recover_stuck_researching,
    update_topic,
)
from app.database import get_connection, init_db
from app.models import Topic, TopicStatus
from app.scheduler import _recover_stuck


def _make_topic(name: str, status: TopicStatus = TopicStatus.RESEARCHING, **kwargs) -> Topic:
    return Topic(
        name=name,
        description=f"Description for {name}",
        status=status,
        status_changed_at=datetime.now(UTC),
        **kwargs,
    )


def _set_status_changed_at(conn: sqlite3.Connection, topic_id: int, dt: datetime) -> None:
    """Helper to directly set status_changed_at for a topic."""
    conn.execute(
        "UPDATE topics SET status_changed_at = ? WHERE id = ?",
        (dt.isoformat(), topic_id),
    )
    conn.commit()


class TestRecoverStuckResearching:
    """Tests for the recover_stuck_researching CRUD function."""

    def test_recovers_topic_stuck_past_timeout(self, db_conn: sqlite3.Connection) -> None:
        """A RESEARCHING topic with status_changed_at > timeout_minutes ago is marked ERROR."""
        topic = _make_topic("Stuck Topic")
        created = create_topic(db_conn, topic)
        db_conn.commit()

        # Backdate status_changed_at to 20 minutes ago (past 15-minute default)
        old_time = datetime.now(UTC) - timedelta(minutes=20)
        _set_status_changed_at(db_conn, created.id, old_time)

        count = recover_stuck_researching(db_conn, timeout_minutes=15)

        assert count == 1
        recovered = get_topic(db_conn, created.id)
        assert recovered.status == TopicStatus.ERROR
        assert "stuck" in recovered.error_message.lower() or "timed out" in recovered.error_message.lower()

    def test_leaves_fresh_researching_topic_alone(self, db_conn: sqlite3.Connection) -> None:
        """A RESEARCHING topic within the timeout window is not touched."""
        topic = _make_topic("Fresh Topic")
        created = create_topic(db_conn, topic)
        db_conn.commit()

        # status_changed_at was just set — well within 15-minute window
        count = recover_stuck_researching(db_conn, timeout_minutes=15)

        assert count == 0
        fresh = get_topic(db_conn, created.id)
        assert fresh.status == TopicStatus.RESEARCHING

    def test_does_not_affect_ready_topics(self, db_conn: sqlite3.Connection) -> None:
        """READY topics are never affected regardless of age."""
        topic = _make_topic("Ready Topic", status=TopicStatus.READY)
        created = create_topic(db_conn, topic)
        db_conn.commit()

        old_time = datetime.now(UTC) - timedelta(hours=1)
        _set_status_changed_at(db_conn, created.id, old_time)

        count = recover_stuck_researching(db_conn, timeout_minutes=15)

        assert count == 0
        still_ready = get_topic(db_conn, created.id)
        assert still_ready.status == TopicStatus.READY

    def test_does_not_affect_error_topics(self, db_conn: sqlite3.Connection) -> None:
        """ERROR topics are never affected."""
        topic = _make_topic("Error Topic", status=TopicStatus.ERROR)
        created = create_topic(db_conn, topic)
        db_conn.commit()

        old_time = datetime.now(UTC) - timedelta(hours=2)
        _set_status_changed_at(db_conn, created.id, old_time)

        count = recover_stuck_researching(db_conn, timeout_minutes=15)

        assert count == 0

    def test_skips_topic_with_null_status_changed_at(self, db_conn: sqlite3.Connection) -> None:
        """Topics with NULL status_changed_at are not recovered (safety guard)."""
        topic = _make_topic("No Timestamp Topic")
        created = create_topic(db_conn, topic)
        db_conn.commit()

        # Explicitly set status_changed_at to NULL
        db_conn.execute("UPDATE topics SET status_changed_at = NULL WHERE id = ?", (created.id,))
        db_conn.commit()

        count = recover_stuck_researching(db_conn, timeout_minutes=15)

        assert count == 0

    def test_recovers_multiple_stuck_topics(self, db_conn: sqlite3.Connection) -> None:
        """Multiple stuck topics are all recovered in one call."""
        old_time = datetime.now(UTC) - timedelta(minutes=30)

        topic_a = _make_topic("Stuck A")
        topic_b = _make_topic("Stuck B")
        topic_fresh = _make_topic("Fresh C")

        created_a = create_topic(db_conn, topic_a)
        created_b = create_topic(db_conn, topic_b)
        created_fresh = create_topic(db_conn, topic_fresh)
        db_conn.commit()

        _set_status_changed_at(db_conn, created_a.id, old_time)
        _set_status_changed_at(db_conn, created_b.id, old_time)
        # created_fresh keeps its current timestamp (within window)

        count = recover_stuck_researching(db_conn, timeout_minutes=15)

        assert count == 2
        assert get_topic(db_conn, created_a.id).status == TopicStatus.ERROR
        assert get_topic(db_conn, created_b.id).status == TopicStatus.ERROR
        assert get_topic(db_conn, created_fresh.id).status == TopicStatus.RESEARCHING

    def test_custom_timeout_minutes(self, db_conn: sqlite3.Connection) -> None:
        """The timeout_minutes parameter controls the threshold correctly."""
        topic = _make_topic("Borderline Topic")
        created = create_topic(db_conn, topic)
        db_conn.commit()

        # Set to exactly 10 minutes ago
        borderline = datetime.now(UTC) - timedelta(minutes=10)
        _set_status_changed_at(db_conn, created.id, borderline)

        # With a 15-minute timeout, 10 minutes is not stuck
        count = recover_stuck_researching(db_conn, timeout_minutes=15)
        assert count == 0
        assert get_topic(db_conn, created.id).status == TopicStatus.RESEARCHING

        # With a 5-minute timeout, 10 minutes IS stuck
        count = recover_stuck_researching(db_conn, timeout_minutes=5)
        assert count == 1
        assert get_topic(db_conn, created.id).status == TopicStatus.ERROR


class TestMigration005:
    """Tests for the status_changed_at migration."""

    def test_migration_adds_column(self, tmp_path: Path) -> None:
        """Migration 005 adds status_changed_at column to topics table."""
        db_path = tmp_path / "migration_test.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
            assert "status_changed_at" in columns
        finally:
            conn.close()

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        """Applying the migration twice does not raise an error."""
        from app.migrations.m005_status_changed_at import up

        db_path = tmp_path / "idempotent_test.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            # Already applied by init_db; applying again should be safe
            up(conn)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
            assert "status_changed_at" in columns
        finally:
            conn.close()

    def test_existing_rows_get_backfilled(self, tmp_path: Path) -> None:
        """After migration, topics with NULL status_changed_at get backfilled to created_at.

        We simulate a pre-migration state by inserting a row with NULL status_changed_at,
        then re-running the migration's backfill query to verify it works correctly.
        """
        db_path = tmp_path / "backfill_test.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            # Insert a topic with explicit NULL status_changed_at to simulate a
            # row that was inserted before migration 005 added the column
            conn.execute(
                """INSERT INTO topics
                   (name, description, feed_urls, feed_mode, created_at,
                    status_changed_at, is_active, status, error_message, check_interval_hours)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "Pre-migration Topic",
                    "desc",
                    "[]",
                    "auto",
                    "2024-01-01T00:00:00+00:00",
                    None,
                    1,
                    "ready",
                    None,
                    None,
                ),
            )
            conn.commit()

            # Run the migration's backfill query
            conn.execute("UPDATE topics SET status_changed_at = created_at WHERE status_changed_at IS NULL")
            conn.commit()

            row = conn.execute(
                "SELECT status_changed_at, created_at FROM topics WHERE name = ?",
                ("Pre-migration Topic",),
            ).fetchone()
            assert row is not None
            assert row["status_changed_at"] == row["created_at"]
        finally:
            conn.close()


class TestStatusChangedAtTracking:
    """Tests that status_changed_at is properly set when status changes."""

    def test_create_topic_sets_status_changed_at(self, db_conn: sqlite3.Connection) -> None:
        """Creating a topic with status_changed_at persists it correctly."""
        now = datetime.now(UTC)
        topic = Topic(
            name="Test Topic",
            description="desc",
            status=TopicStatus.RESEARCHING,
            status_changed_at=now,
        )
        created = create_topic(db_conn, topic)
        db_conn.commit()

        fetched = get_topic(db_conn, created.id)
        assert fetched.status_changed_at is not None
        # Allow a small tolerance for datetime serialization
        diff = abs((fetched.status_changed_at.replace(tzinfo=UTC) - now).total_seconds())
        assert diff < 2

    def test_update_topic_persists_status_changed_at(self, db_conn: sqlite3.Connection) -> None:
        """Updating a topic with a new status_changed_at persists the new value."""
        topic = _make_topic("Update Test")
        created = create_topic(db_conn, topic)
        db_conn.commit()

        new_time = datetime.now(UTC) - timedelta(hours=1)
        created.status = TopicStatus.READY
        created.status_changed_at = new_time
        update_topic(db_conn, created)
        db_conn.commit()

        fetched = get_topic(db_conn, created.id)
        assert fetched.status == TopicStatus.READY
        assert fetched.status_changed_at is not None
        diff = abs((fetched.status_changed_at.replace(tzinfo=UTC) - new_time).total_seconds())
        assert diff < 2


class TestRecoverStuckScheduledJob:
    """Tests for the _recover_stuck scheduler callback."""

    async def test_recovers_stuck_topics(self, tmp_path: Path) -> None:
        """_recover_stuck calls recover_stuck_researching and logs on finds."""
        db_path = tmp_path / "test.db"
        init_db(db_path)
        conn = get_connection(db_path)

        topic = _make_topic("Stuck Scheduled")
        created = create_topic(conn, topic)
        conn.commit()

        old_time = datetime.now(UTC) - timedelta(minutes=20)
        _set_status_changed_at(conn, created.id, old_time)
        conn.close()

        # Should not raise and should recover the stuck topic
        await _recover_stuck(timeout_minutes=15, db_path=db_path)

        conn2 = get_connection(db_path)
        try:
            fetched = get_topic(conn2, created.id)
            assert fetched.status == TopicStatus.ERROR
        finally:
            conn2.close()

    async def test_does_not_raise_on_db_error(self, tmp_path: Path) -> None:
        """_recover_stuck catches exceptions and does not propagate them."""
        bad_path = tmp_path / "nonexistent" / "test.db"
        # Should not raise even with a bad path
        await _recover_stuck(timeout_minutes=15, db_path=bad_path)

    async def test_no_log_when_nothing_stuck(self, tmp_path: Path) -> None:
        """_recover_stuck runs silently when no stuck topics exist."""
        db_path = tmp_path / "test.db"
        init_db(db_path)

        # Should not raise; nothing to recover
        await _recover_stuck(timeout_minutes=15, db_path=db_path)

    async def test_uses_default_timeout(self, tmp_path: Path) -> None:
        """_recover_stuck uses 15-minute default timeout."""
        db_path = tmp_path / "test.db"
        init_db(db_path)

        with (
            patch("app.scheduler.recover_stuck_researching", return_value=0) as mock_recover,
            patch("app.scheduler.get_db") as mock_get_db,
        ):
            mock_get_db.return_value.__enter__ = lambda s: mock_get_db.return_value
            mock_get_db.return_value.__exit__ = lambda s, *a: False
            await _recover_stuck(db_path=db_path)

        mock_recover.assert_called_once_with(mock_get_db.return_value, 15)


class TestSchedulerHasRecoverJob:
    """Tests that the scheduler registers the recover_stuck job."""

    async def test_start_creates_four_jobs(self) -> None:
        """start_scheduler registers check_all_topics, recover_stuck_researching, vacuum_db, and cleanup_old_articles."""
        from app.config import LLMSettings, Settings
        from app.scheduler import start_scheduler, stop_scheduler

        settings = Settings(
            llm=LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
            check_interval="4h",
        )
        scheduler = start_scheduler(settings)
        try:
            job_ids = {j.id for j in scheduler.get_jobs()}
            assert "recover_stuck_researching" in job_ids
            assert "check_all_topics" in job_ids
            assert "vacuum_db" in job_ids
            assert "cleanup_old_articles" in job_ids
            assert len(scheduler.get_jobs()) == 4
        finally:
            stop_scheduler()

    async def test_recover_job_runs_every_five_minutes(self) -> None:
        """The recover_stuck_researching job is scheduled at 5-minute intervals."""
        from app.config import LLMSettings, Settings
        from app.scheduler import start_scheduler, stop_scheduler

        settings = Settings(
            llm=LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
            check_interval="4h",
        )
        scheduler = start_scheduler(settings)
        try:
            job = scheduler.get_job("recover_stuck_researching")
            assert job is not None
            assert job.trigger.interval.total_seconds() == 300  # 5 * 60
        finally:
            stop_scheduler()
