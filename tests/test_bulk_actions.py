"""Tests for bulk delete and bulk check routes."""

import asyncio
import sqlite3
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_topic, get_topic
from app.main import app
from app.models import FeedMode, Topic, TopicStatus
from app.web.dependencies import get_db_conn, get_settings

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
        called_topic_id = mock_check.call_args[0][0]
        assert called_topic_id == ready_topic.id

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
        assert mock_check.call_args[0][0] == topic.id

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
    """The bulk-check / manual background task is the authoritative guard owner."""

    async def test_run_single_check_skips_when_already_checking(self, tmp_path) -> None:
        """A second _run_single_check on an in-flight topic skips the pipeline (OVH-033)."""
        from app.database import get_db, init_db
        from app.web.routers import background
        from app.web.state import _checking_state

        db_path = tmp_path / "bulk.db"
        init_db(db_path)
        with get_db(db_path) as seed:
            topic = _make_topic(seed, name="Busy")
        settings = _make_settings()

        _checking_state._topics.clear()
        _checking_state._start_times.clear()
        try:
            # Slot already taken (e.g. the manual /check is mid-flight).
            assert await _checking_state.start_check(topic.id) is True
            with patch("app.web.routers.background.check_topic", new_callable=AsyncMock) as mock_check:
                await background._run_single_check(topic.id, settings, db_path)
            mock_check.assert_not_awaited()
        finally:
            _checking_state._topics.clear()
            _checking_state._start_times.clear()

    async def test_run_single_check_acquires_and_releases(self, tmp_path) -> None:
        """_run_single_check claims the guard, runs with guard=False, then releases (OVH-033)."""
        from app.database import get_db, init_db
        from app.web.routers import background
        from app.web.state import _checking_state

        db_path = tmp_path / "bulk2.db"
        init_db(db_path)
        with get_db(db_path) as seed:
            topic = _make_topic(seed, name="Free")
        settings = _make_settings()

        _checking_state._topics.clear()
        _checking_state._start_times.clear()
        try:
            with patch("app.web.routers.background.check_topic", new_callable=AsyncMock) as mock_check:
                await background._run_single_check(topic.id, settings, db_path)
            # check_topic invoked with guard=False (task owns the guard).
            assert mock_check.await_count == 1
            assert mock_check.await_args.kwargs.get("guard") is False
            # Guard released after completion.
            assert await _checking_state.is_checking(topic.id) is False
        finally:
            _checking_state._topics.clear()
            _checking_state._start_times.clear()

    async def test_concurrent_run_single_check_only_one_runs(self, tmp_path) -> None:
        """Two concurrent _run_single_check of the same topic: only one runs the pipeline."""
        from app.database import get_db, init_db
        from app.web.routers import background
        from app.web.state import _checking_state

        db_path = tmp_path / "bulk3.db"
        init_db(db_path)
        with get_db(db_path) as seed:
            topic = _make_topic(seed, name="Racer")
        settings = _make_settings()

        runs = 0

        async def _slow_check(t, conn, settings, *, guard=True):
            nonlocal runs
            runs += 1
            await asyncio.sleep(0.05)

        _checking_state._topics.clear()
        _checking_state._start_times.clear()
        try:
            with patch("app.web.routers.background.check_topic", side_effect=_slow_check):
                await asyncio.gather(
                    background._run_single_check(topic.id, settings, db_path),
                    background._run_single_check(topic.id, settings, db_path),
                )
            assert runs == 1
        finally:
            _checking_state._topics.clear()
            _checking_state._start_times.clear()
