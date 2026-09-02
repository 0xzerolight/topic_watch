"""Tests for bulk delete and bulk check routes."""

import asyncio
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_check_intents, create_topic, get_topic
from app.database import get_connection
from app.main import app
from app.models import CheckIntent, CheckResult, FeedMode, Topic, TopicStatus
from app.web.dependencies import get_db_conn, get_settings
from app.web.state import _checking_state

CSRF_TEST_TOKEN = "test-csrf-token-for-bulk-tests"


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_topic(conn: sqlite3.Connection, **overrides) -> Topic:
    defaults = {
        "name": "Test Topic",
        "description": "A test topic",
        "feed_urls": ["https://example.com/feed.xml"],
        "feed_mode": FeedMode.MANUAL,
        "status": TopicStatus.READY,
    }
    defaults.update(overrides)
    topic = create_topic(conn, Topic(**defaults))
    conn.commit()
    return topic


@pytest.fixture
async def client(
    db_conn: sqlite3.Connection,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a test client with CSRF credentials set."""
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


@pytest.fixture
async def client_no_csrf(
    db_conn: sqlite3.Connection,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a test client without any CSRF credentials."""
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
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# --- Bulk Delete ---


class TestBulkDelete:
    """Tests for POST /topics/bulk-delete."""

    async def test_bulk_delete_redirects_to_dashboard(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Bulk delete redirects to dashboard after deletion."""
        topic = _make_topic(db_conn, name="To Delete")
        response = await client.post(
            "/topics/bulk-delete",
            data={"topic_ids": str(topic.id)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/"

    async def test_bulk_delete_removes_topics(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Bulk delete removes all specified topics from the database."""
        topic1 = _make_topic(db_conn, name="Delete Me 1")
        topic2 = _make_topic(db_conn, name="Delete Me 2")
        topic3 = _make_topic(db_conn, name="Keep Me")

        body = f"topic_ids={topic1.id}&topic_ids={topic2.id}"
        response = await client.post(
            "/topics/bulk-delete",
            content=body.encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            follow_redirects=False,
        )
        assert response.status_code == 303

        assert get_topic(db_conn, topic1.id) is None
        assert get_topic(db_conn, topic2.id) is None
        assert get_topic(db_conn, topic3.id) is not None

    async def test_bulk_delete_empty_list_does_not_crash(self, client: httpx.AsyncClient) -> None:
        """Bulk delete with no topic_ids does not crash and redirects."""
        response = await client.post(
            "/topics/bulk-delete",
            data={},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/"

    async def test_bulk_delete_invalid_id_skipped(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Non-numeric or nonexistent topic IDs are skipped gracefully."""
        topic = _make_topic(db_conn, name="Survivor")
        response = await client.post(
            "/topics/bulk-delete",
            content=b"topic_ids=not-a-number&topic_ids=99999",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        # The valid topic should still exist
        assert get_topic(db_conn, topic.id) is not None

    async def test_bulk_delete_requires_csrf(
        self, client_no_csrf: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Bulk delete without CSRF token returns 403."""
        topic = _make_topic(db_conn, name="CSRF Test Topic")
        response = await client_no_csrf.post(
            "/topics/bulk-delete",
            data={"topic_ids": str(topic.id)},
            follow_redirects=False,
        )
        assert response.status_code == 403
        # Topic should still exist
        assert get_topic(db_conn, topic.id) is not None


# --- Bulk Check ---


class TestBulkCheck:
    """Tests for POST /topics/bulk-check."""

    async def test_bulk_check_redirects_to_dashboard(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Bulk check redirects to dashboard."""
        topic = _make_topic(db_conn, name="Check Me")
        with patch("app.web.routers.background._run_single_check", new_callable=AsyncMock):
            response = await client.post(
                "/topics/bulk-check",
                data={"topic_ids": str(topic.id)},
                follow_redirects=False,
            )
        assert response.status_code == 303
        assert response.headers["location"] == "/"

    async def test_bulk_check_queues_ready_topics(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Bulk check queues background tasks only for READY topics."""
        ready_topic = _make_topic(db_conn, name="Ready Topic", status=TopicStatus.READY)
        researching_topic = _make_topic(db_conn, name="Busy Topic", status=TopicStatus.RESEARCHING)

        body = f"topic_ids={ready_topic.id}&topic_ids={researching_topic.id}"
        with patch("app.web.routers.background._run_single_check", new_callable=AsyncMock) as mock_check:
            await client.post(
                "/topics/bulk-check",
                content=body.encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                follow_redirects=False,
            )

        # Only the READY topic should be queued
        assert mock_check.call_count == 1
        called_intent = mock_check.call_args[0][0]
        assert called_intent.topic_id == ready_topic.id

    async def test_bulk_check_empty_list_does_not_crash(self, client: httpx.AsyncClient) -> None:
        """Bulk check with no topic_ids does not crash and redirects."""
        response = await client.post(
            "/topics/bulk-check",
            data={},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/"

    async def test_bulk_check_invalid_id_skipped(self, client: httpx.AsyncClient) -> None:
        """Non-numeric topic IDs are skipped gracefully."""
        response = await client.post(
            "/topics/bulk-check",
            content=b"topic_ids=not-a-number&topic_ids=99999",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            follow_redirects=False,
        )
        assert response.status_code == 303

    async def test_bulk_check_dedups_duplicate_topic_ids(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """OVH-166: a duplicated topic_id queues exactly one background check.

        A crafted form (or a double-submit) can repeat the same checkbox id; the
        route must not launch a redundant second re-check of the same topic.
        """
        topic = _make_topic(db_conn, name="Dup Topic", status=TopicStatus.READY)

        body = f"topic_ids={topic.id}&topic_ids={topic.id}&topic_ids={topic.id}"
        with patch("app.web.routers.background._run_single_check", new_callable=AsyncMock) as mock_check:
            response = await client.post(
                "/topics/bulk-check",
                content=body.encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                follow_redirects=False,
            )

        assert response.status_code == 303
        # Three identical ids → exactly one queued check.
        assert mock_check.call_count == 1
        assert mock_check.call_args[0][0].topic_id == topic.id

    async def test_bulk_check_admits_one_intent_per_ready_topic_in_one_commit(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection, db_path: Path
    ) -> None:
        """AUG-286: a crash mid-batch must leave a knowable record of what was accepted."""
        first = _make_topic(db_conn, name="Bulk One", status=TopicStatus.READY)
        second = _make_topic(db_conn, name="Bulk Two", status=TopicStatus.READY)
        skipped = _make_topic(db_conn, name="Bulk Busy", status=TopicStatus.RESEARCHING)

        body = f"topic_ids={first.id}&topic_ids={second.id}&topic_ids={skipped.id}"
        with (
            patch("app.web.routers.background._run_single_check", new_callable=AsyncMock) as mock_check,
            patch("app.web.routers.topics.create_check_intents", side_effect=create_check_intents) as spy_create,
        ):
            response = await client.post(
                "/topics/bulk-check",
                content=body.encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                follow_redirects=False,
            )

        assert response.status_code == 303
        # One insert call for the whole batch, so the rows land in one transaction.
        spy_create.assert_called_once()
        assert len(spy_create.call_args[0][1]) == 2

        verify = get_connection(db_path)
        try:
            rows = verify.execute("SELECT request_id, topic_id, status FROM check_intents ORDER BY id").fetchall()
        finally:
            verify.close()
        assert [r["topic_id"] for r in rows] == [first.id, second.id]
        assert {r["status"] for r in rows} == {"pending"}
        assert {r["request_id"] for r in rows} == {response.headers["X-Request-ID"]}
        assert mock_check.call_count == 2

    async def test_bulk_check_skips_a_topic_already_being_checked(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """A topic mid-check already owes this answer, so bulk admits nothing for it."""
        busy = _make_topic(db_conn, name="Bulk Busy Guard", status=TopicStatus.READY)
        free = _make_topic(db_conn, name="Bulk Free", status=TopicStatus.READY)

        owner = await _checking_state.start_check(busy.id)
        try:
            body = f"topic_ids={busy.id}&topic_ids={free.id}"
            with patch("app.web.routers.background._run_single_check", new_callable=AsyncMock) as mock_check:
                response = await client.post(
                    "/topics/bulk-check",
                    content=body.encode(),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    follow_redirects=False,
                )
        finally:
            await _checking_state.finish_check(busy.id, owner)

        assert response.status_code == 303
        rows = db_conn.execute("SELECT topic_id FROM check_intents").fetchall()
        assert [r["topic_id"] for r in rows] == [free.id]
        assert mock_check.call_count == 1

    async def test_bulk_check_requires_csrf(
        self, client_no_csrf: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Bulk check without CSRF token returns 403."""
        topic = _make_topic(db_conn, name="CSRF Check Topic")
        response = await client_no_csrf.post(
            "/topics/bulk-check",
            data={"topic_ids": str(topic.id)},
            follow_redirects=False,
        )
        assert response.status_code == 403


# --- _run_single_check per-topic guard (bulk-check + manual share it, OVH-033/096) ---


class TestSingleCheckGuard:
    """The guard now lives inside ``check_topic``, below which the intent runner sits."""

    def _seed(self, db_path: Path, name: str) -> tuple[Topic, CheckIntent]:
        from app.database import get_db, init_db

        init_db(db_path)
        with get_db(db_path) as seed:
            topic = _make_topic(seed, name=name)
            intent = CheckIntent(request_id="req-1", topic_id=topic.id)
            create_check_intents(seed, [intent])
        return topic, intent

    async def test_run_single_check_skips_when_already_checking(self, tmp_path) -> None:
        """A check whose topic is already in flight backs off instead of running (OVH-033)."""
        from app.web.routers import background

        db_path = tmp_path / "bulk.db"
        topic, intent = self._seed(db_path, "Busy")
        settings = _make_settings()

        # Slot already taken (e.g. the manual /check is mid-flight). The guard is
        # inside check_topic now, so patch the layer below it.
        assert await _checking_state.start_check(topic.id) is not None
        try:
            with patch(
                "app.checker._check_topic_guarded",
                new_callable=AsyncMock,
                return_value=CheckResult(topic_id=topic.id),
            ) as mock_check:
                await background._run_single_check(intent, settings, db_path)
            mock_check.assert_not_awaited()
        finally:
            _checking_state._topics.clear()
            _checking_state._start_times.clear()

        conn = get_connection(db_path)
        try:
            row = conn.execute("SELECT * FROM check_intents WHERE id = ?", (intent.id,)).fetchone()
        finally:
            conn.close()
        assert row["status"] == "pending"
        assert row["attempts"] == 1
        assert row["last_error"] == "skipped: already in flight"

    async def test_run_single_check_acquires_and_releases(self, tmp_path) -> None:
        """The runner delegates the guard to check_topic and leaves no slot held."""
        from app.web.routers import background

        db_path = tmp_path / "bulk2.db"
        topic, intent = self._seed(db_path, "Free")
        settings = _make_settings()

        with patch(
            "app.checker.check_topic", new_callable=AsyncMock, return_value=CheckResult(topic_id=topic.id)
        ) as mock_check:
            await background._run_single_check(intent, settings, db_path)

        assert mock_check.await_count == 1
        assert mock_check.await_args.kwargs.get("guard") is True
        assert await _checking_state.is_checking(topic.id) is False

    async def test_concurrent_run_single_check_only_one_runs(self, tmp_path) -> None:
        """Two intents for one topic: the guard lets exactly one through and the other waits."""
        from app.database import get_db
        from app.web.routers import background

        db_path = tmp_path / "bulk3.db"
        topic, first = self._seed(db_path, "Racer")
        second = CheckIntent(request_id="req-2", topic_id=topic.id)
        with get_db(db_path) as seed:
            create_check_intents(seed, [second])
        settings = _make_settings()

        async def _slow_check(t, settings, db_path):
            await asyncio.sleep(0.05)
            return CheckResult(topic_id=topic.id)

        with patch("app.checker._check_topic_guarded", side_effect=_slow_check) as mock_check:
            await asyncio.gather(
                background._run_single_check(first, settings, db_path),
                background._run_single_check(second, settings, db_path),
            )
        assert mock_check.await_count == 1

        conn = get_connection(db_path)
        try:
            rows = {r["id"]: r for r in conn.execute("SELECT * FROM check_intents").fetchall()}
        finally:
            conn.close()
        loser = [r for r in rows.values() if r["status"] == "pending"]
        assert len(loser) == 1
        assert loser[0]["attempts"] == 1
        assert loser[0]["last_error"] == "skipped: already in flight"
