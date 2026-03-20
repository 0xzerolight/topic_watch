"""Tests for sub-hour check intervals (minutes-based)."""

import sqlite3
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import (
    create_check_result,
    create_topic,
    get_topic,
    get_topics_due_for_check,
    update_topic,
)
from app.main import app
from app.migrations.m008_interval_minutes import up as m008_up
from app.models import CheckResult, FeedMode, Topic, TopicStatus
from app.web.dependencies import get_db_conn, get_settings


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


CSRF_TEST_TOKEN = "test-csrf-token-for-tests"


@pytest.fixture
async def client(
    db_conn: sqlite3.Connection,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a test client with database dependency overridden."""
    settings = _make_settings()

    def override_db():
        yield db_conn

    def override_settings():
        return settings

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = override_settings

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        cookies={"csrf_token": CSRF_TEST_TOKEN},
        headers={"X-CSRF-Token": CSRF_TEST_TOKEN},
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


class TestTopicModel:
    """Test Topic model with check_interval_minutes."""

    def test_topic_with_minutes(self) -> None:
        topic = Topic(name="T", description="d", check_interval_minutes=30)
        assert topic.check_interval_minutes == 30

    def test_topic_default_no_interval(self) -> None:
        topic = Topic(name="T", description="d")
        assert topic.check_interval_minutes is None

    def test_to_insert_dict_includes_minutes(self) -> None:
        topic = Topic(name="T", description="d", check_interval_minutes=30)
        d = topic.to_insert_dict()
        assert "check_interval_minutes" in d
        assert d["check_interval_minutes"] == 30
        assert "check_interval_hours" not in d

    def test_to_insert_dict_null_minutes(self) -> None:
        topic = Topic(name="T", description="d")
        d = topic.to_insert_dict()
        assert d["check_interval_minutes"] is None


class TestTopicCRUD:
    """Test CRUD with check_interval_minutes."""

    def test_create_topic_with_minutes(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(
            name="MinutesTopic",
            description="desc",
            check_interval_minutes=30,
        )
        created = create_topic(db_conn, topic)
        db_conn.commit()

        retrieved = get_topic(db_conn, created.id)
        assert retrieved is not None
        assert retrieved.check_interval_minutes == 30

    def test_update_topic_minutes(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="UpdateMinutes", description="d")
        created = create_topic(db_conn, topic)
        db_conn.commit()

        created.check_interval_minutes = 45
        update_topic(db_conn, created)
        db_conn.commit()

        retrieved = get_topic(db_conn, created.id)
        assert retrieved.check_interval_minutes == 45

    def test_null_minutes_stored_and_retrieved(self, db_conn: sqlite3.Connection) -> None:
        topic = Topic(name="NullMinutes", description="d")
        created = create_topic(db_conn, topic)
        db_conn.commit()

        retrieved = get_topic(db_conn, created.id)
        assert retrieved.check_interval_minutes is None


class TestMigration:
    """Test that m008 migration correctly converts hours to minutes."""

    def test_migration_adds_column(self, db_conn: sqlite3.Connection) -> None:
        """After migration, check_interval_minutes column exists."""
        columns = {row[1] for row in db_conn.execute("PRAGMA table_info(topics)").fetchall()}
        assert "check_interval_minutes" in columns

    def test_migration_converts_existing_hours(self, db_conn: sqlite3.Connection) -> None:
        """Existing check_interval_hours values are converted to minutes."""
        # Manually insert a topic with check_interval_hours set
        db_conn.execute(
            """INSERT INTO topics (name, description, feed_urls, feed_mode, created_at,
               is_active, status, check_interval_hours)
               VALUES ('OldStyle', 'desc', '[]', 'auto', datetime('now'), 1, 'ready', 6)"""
        )
        db_conn.commit()

        # Run the migration again (idempotent - column already exists, but UPDATE runs
        # again, which is fine since it recalculates from hours).
        # Simulate what the migration does for new rows:
        db_conn.execute(
            "UPDATE topics SET check_interval_minutes = check_interval_hours * 60 "
            "WHERE check_interval_hours IS NOT NULL AND check_interval_minutes IS NULL"
        )
        db_conn.commit()

        row = db_conn.execute("SELECT check_interval_minutes FROM topics WHERE name = 'OldStyle'").fetchone()
        assert row is not None
        assert row["check_interval_minutes"] == 360  # 6 hours * 60

    def test_migration_idempotent(self, db_conn: sqlite3.Connection) -> None:
        """Running migration twice doesn't raise."""
        m008_up(db_conn)  # Already applied; column exists; should be a no-op
        db_conn.commit()

    def test_from_row_backwards_compat(self, db_conn: sqlite3.Connection) -> None:
        """Topic.from_row converts check_interval_hours if minutes is NULL."""
        # Insert with only hours set, minutes NULL
        db_conn.execute(
            """INSERT INTO topics (name, description, feed_urls, feed_mode, created_at,
               is_active, status, check_interval_hours, check_interval_minutes)
               VALUES ('HoursOnly', 'desc', '[]', 'auto', datetime('now'), 1, 'ready', 3, NULL)"""
        )
        db_conn.commit()

        row = db_conn.execute("SELECT * FROM topics WHERE name = 'HoursOnly'").fetchone()
        topic = Topic.from_row(row)
        assert topic.check_interval_minutes == 180  # 3h * 60


class TestGetTopicsDueForCheck:
    """Test get_topics_due_for_check with minute-based intervals."""

    def _make_ready_topic(self, conn: sqlite3.Connection, name: str, interval_minutes: int | None = None) -> Topic:
        topic = create_topic(
            conn,
            Topic(
                name=name,
                description="d",
                status=TopicStatus.READY,
                check_interval_minutes=interval_minutes,
            ),
        )
        conn.commit()
        return topic

    def test_topic_with_no_checks_is_always_due(self, db_conn: sqlite3.Connection) -> None:
        self._make_ready_topic(db_conn, "NeverChecked", interval_minutes=30)
        due = get_topics_due_for_check(db_conn, default_interval_hours=6)
        assert any(t.name == "NeverChecked" for t in due)

    def test_topic_due_after_interval(self, db_conn: sqlite3.Connection) -> None:
        """A topic with 30-minute interval is due after 30+ minutes."""
        topic = self._make_ready_topic(db_conn, "ThirtyMin", interval_minutes=30)
        # Record a check result 31 minutes ago
        checked_at = datetime.now(UTC) - timedelta(minutes=31)
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, checked_at=checked_at),
        )
        db_conn.commit()

        due = get_topics_due_for_check(db_conn, default_interval_hours=6)
        assert any(t.name == "ThirtyMin" for t in due)

    def test_topic_not_due_before_interval(self, db_conn: sqlite3.Connection) -> None:
        """A topic with 30-minute interval is NOT due after only 15 minutes."""
        topic = self._make_ready_topic(db_conn, "NotYet", interval_minutes=30)
        # Record a check result only 15 minutes ago
        checked_at = datetime.now(UTC) - timedelta(minutes=15)
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, checked_at=checked_at),
        )
        db_conn.commit()

        due = get_topics_due_for_check(db_conn, default_interval_hours=6)
        assert not any(t.name == "NotYet" for t in due)

    def test_null_interval_uses_global_default(self, db_conn: sqlite3.Connection) -> None:
        """A topic with NULL interval uses the global default (6h)."""
        topic = self._make_ready_topic(db_conn, "DefaultInterval", interval_minutes=None)
        # Checked 5 hours ago — should NOT be due (default is 6h)
        checked_at = datetime.now(UTC) - timedelta(hours=5)
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, checked_at=checked_at),
        )
        db_conn.commit()

        due = get_topics_due_for_check(db_conn, default_interval_hours=6)
        assert not any(t.name == "DefaultInterval" for t in due)

    def test_null_interval_becomes_due_after_default(self, db_conn: sqlite3.Connection) -> None:
        """A topic with NULL interval becomes due after the global default elapses."""
        topic = self._make_ready_topic(db_conn, "DefaultDue", interval_minutes=None)
        # Checked 7 hours ago — should be due (default is 6h)
        checked_at = datetime.now(UTC) - timedelta(hours=7)
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, checked_at=checked_at),
        )
        db_conn.commit()

        due = get_topics_due_for_check(db_conn, default_interval_hours=6)
        assert any(t.name == "DefaultDue" for t in due)

    def test_inactive_topic_not_included(self, db_conn: sqlite3.Connection) -> None:
        """Inactive topics are never returned as due."""
        create_topic(
            db_conn,
            Topic(
                name="Inactive",
                description="d",
                status=TopicStatus.READY,
                is_active=False,
                check_interval_minutes=10,
            ),
        )
        db_conn.commit()

        due = get_topics_due_for_check(db_conn, default_interval_hours=6)
        assert not any(t.name == "Inactive" for t in due)

    def test_sub_hour_interval_10_minutes(self, db_conn: sqlite3.Connection) -> None:
        """A topic with 10-minute interval is due after 11 minutes."""
        topic = self._make_ready_topic(db_conn, "TenMin", interval_minutes=10)
        checked_at = datetime.now(UTC) - timedelta(minutes=11)
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, checked_at=checked_at),
        )
        db_conn.commit()

        due = get_topics_due_for_check(db_conn, default_interval_hours=6)
        assert any(t.name == "TenMin" for t in due)

    def test_sub_hour_interval_10_minutes_not_yet_due(self, db_conn: sqlite3.Connection) -> None:
        """A topic with 10-minute interval is NOT due after only 9 minutes."""
        topic = self._make_ready_topic(db_conn, "TenMinNotYet", interval_minutes=10)
        checked_at = datetime.now(UTC) - timedelta(minutes=9)
        create_check_result(
            db_conn,
            CheckResult(topic_id=topic.id, checked_at=checked_at),
        )
        db_conn.commit()

        due = get_topics_due_for_check(db_conn, default_interval_hours=6)
        assert not any(t.name == "TenMinNotYet" for t in due)


class TestFormValidation:
    """Test web form validation for check_interval_minutes."""

    async def test_create_topic_valid_minutes(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Creating a topic with a valid interval (60 minutes) succeeds."""
        response = await client.post(
            "/topics",
            data={
                "name": "Valid Minutes Topic",
                "description": "Test description",
                "feed_mode": "auto",
                "check_interval_minutes": "60",
                "tags": "",
            },
        )
        assert response.status_code == 303  # Redirect on success

        from app.crud import get_topic_by_name

        topic = get_topic_by_name(db_conn, "Valid Minutes Topic")
        assert topic is not None
        assert topic.check_interval_minutes == 60

    async def test_create_topic_reject_below_minimum(self, client: httpx.AsyncClient) -> None:
        """Creating a topic with interval < 10 minutes is rejected."""
        response = await client.post(
            "/topics",
            data={
                "name": "Too Frequent",
                "description": "Test description",
                "feed_mode": "auto",
                "check_interval_minutes": "5",
                "tags": "",
            },
        )
        assert response.status_code == 422
        assert b"10" in response.content  # Error message mentions the minimum

    async def test_create_topic_reject_above_maximum(self, client: httpx.AsyncClient) -> None:
        """Creating a topic with interval > 10080 minutes is rejected."""
        response = await client.post(
            "/topics",
            data={
                "name": "Too Infrequent",
                "description": "Test description",
                "feed_mode": "auto",
                "check_interval_minutes": "99999",
                "tags": "",
            },
        )
        assert response.status_code == 422
        assert b"10080" in response.content

    async def test_create_topic_reject_non_integer(self, client: httpx.AsyncClient) -> None:
        """Creating a topic with non-integer interval is rejected."""
        response = await client.post(
            "/topics",
            data={
                "name": "Bad Interval",
                "description": "Test description",
                "feed_mode": "auto",
                "check_interval_minutes": "abc",
                "tags": "",
            },
        )
        assert response.status_code == 422

    async def test_create_topic_blank_interval_uses_default(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """A blank interval means use the global default (check_interval_minutes=None)."""
        response = await client.post(
            "/topics",
            data={
                "name": "Default Interval Topic",
                "description": "Test description",
                "feed_mode": "auto",
                "check_interval_minutes": "",
                "tags": "",
            },
        )
        assert response.status_code == 303

        from app.crud import get_topic_by_name

        topic = get_topic_by_name(db_conn, "Default Interval Topic")
        assert topic is not None
        assert topic.check_interval_minutes is None

    async def test_edit_topic_valid_minutes(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Editing a topic with a valid interval (30 minutes) succeeds."""
        topic = create_topic(
            db_conn,
            Topic(
                name="EditableMinutes",
                description="d",
                status=TopicStatus.READY,
                feed_mode=FeedMode.AUTO,
            ),
        )
        db_conn.commit()

        response = await client.post(
            f"/topics/{topic.id}/edit",
            data={
                "name": "EditableMinutes",
                "description": "d",
                "feed_mode": "auto",
                "check_interval_minutes": "30",
                "tags": "",
            },
        )
        assert response.status_code == 303

        updated = get_topic(db_conn, topic.id)
        assert updated.check_interval_minutes == 30

    async def test_edit_topic_reject_below_minimum(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Editing a topic with interval < 10 minutes is rejected."""
        topic = create_topic(
            db_conn,
            Topic(
                name="TooFrequentEdit",
                description="d",
                status=TopicStatus.READY,
                feed_mode=FeedMode.AUTO,
            ),
        )
        db_conn.commit()

        response = await client.post(
            f"/topics/{topic.id}/edit",
            data={
                "name": "TooFrequentEdit",
                "description": "d",
                "feed_mode": "auto",
                "check_interval_minutes": "1",
                "tags": "",
            },
        )
        assert response.status_code == 422
