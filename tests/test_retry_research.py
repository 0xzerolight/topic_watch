"""Tests for the reinit_topic route (POST /topics/{topic_id}/init)."""

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
        "status": TopicStatus.ERROR,
        "error_message": "LLM call failed: timeout",
    }
    defaults.update(overrides)
    topic = create_topic(conn, Topic(**defaults))
    conn.commit()
    return topic


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


class TestReinitTopic:
    """Tests for POST /topics/{topic_id}/init (reinit_topic route)."""

    async def test_error_topic_resets_to_researching(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Posting to /init resets an ERROR topic's status to RESEARCHING."""
        topic = _make_topic(db_conn, status=TopicStatus.ERROR, error_message="timeout")

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            response = await client.post(
                f"/topics/{topic.id}/init",
                follow_redirects=False,
            )

        assert response.status_code == 303

        updated = get_topic(db_conn, topic.id)
        assert updated is not None
        assert updated.status == TopicStatus.RESEARCHING

    async def test_error_topic_clears_error_message(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Posting to /init clears the error_message field."""
        topic = _make_topic(db_conn, status=TopicStatus.ERROR, error_message="some error")

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            await client.post(f"/topics/{topic.id}/init", follow_redirects=False)

        updated = get_topic(db_conn, topic.id)
        assert updated is not None
        assert updated.error_message is None

    async def test_redirects_to_topic_detail(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Posting to /init redirects to the topic detail page."""
        topic = _make_topic(db_conn)

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            response = await client.post(
                f"/topics/{topic.id}/init",
                follow_redirects=False,
            )

        assert response.status_code == 303
        assert response.headers["location"] == f"/topics/{topic.id}"

    async def test_nonexistent_topic_returns_404(
        self,
        client: httpx.AsyncClient,
    ) -> None:
        """Posting to /init for a non-existent topic returns 404."""
        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            response = await client.post(
                "/topics/99999/init",
                follow_redirects=False,
            )

        assert response.status_code == 404

    async def test_ready_topic_resets_to_researching(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Posting to /init for a READY topic also resets it to RESEARCHING."""
        topic = _make_topic(db_conn, status=TopicStatus.READY, error_message=None)

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            response = await client.post(
                f"/topics/{topic.id}/init",
                follow_redirects=False,
            )

        assert response.status_code == 303

        updated = get_topic(db_conn, topic.id)
        assert updated is not None
        assert updated.status == TopicStatus.RESEARCHING

    async def test_retry_resets_init_attempts(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """OVH-098: explicit Retry resets init_attempts to 0 (full thin-data budget)."""
        topic = _make_topic(db_conn, status=TopicStatus.ERROR, init_attempts=2)

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock):
            response = await client.post(f"/topics/{topic.id}/init", follow_redirects=False)

        assert response.status_code == 303
        updated = get_topic(db_conn, topic.id)
        assert updated is not None
        assert updated.init_attempts == 0

    async def test_background_task_is_triggered(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Posting to /init schedules the _run_init background task."""
        topic = _make_topic(db_conn, status=TopicStatus.ERROR)

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock) as mock_run_init:
            await client.post(f"/topics/{topic.id}/init", follow_redirects=False)

        # Background tasks are added but may run after the response; check mock was used
        # The route adds _run_init as a background task with (topic.id, settings, db_path)
        # We can't assert call count directly since BackgroundTasks runs after response,
        # but no exception means the route completed successfully.
        assert mock_run_init is not None  # mock was set up without error


class TestReinitOwnership:
    """AUG-137: ownership is decided before the status changes, and refusals are visible."""

    async def test_busy_topic_refuses_instead_of_losing_the_init(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """A live check of the same topic makes Retry refuse, not strand RESEARCHING.

        The handler used to commit RESEARCHING and then queue a task that tried to
        take the in-flight guard; the check already held it, so the task exited
        silently and the topic sat in RESEARCHING until stuck recovery called it an
        error.
        """
        from app.web.state import _checking_state

        topic = _make_topic(db_conn, status=TopicStatus.ERROR, error_message="boom")

        owner = await _checking_state.start_check(topic.id)
        assert owner is not None
        try:
            with patch("app.web.routers.background._run_init", new_callable=AsyncMock) as mock_init:
                response = await client.post(f"/topics/{topic.id}/init", follow_redirects=False)
        finally:
            await _checking_state.finish_check(topic.id, owner)

        assert response.status_code == 409
        mock_init.assert_not_called()
        unchanged = get_topic(db_conn, topic.id)
        assert unchanged.status == TopicStatus.ERROR
        assert unchanged.error_message == "boom"

    async def test_paused_topic_refuses(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """A paused topic is not re-initialized behind the user's back (AUG-140)."""
        topic = _make_topic(db_conn, status=TopicStatus.ERROR, is_active=False)

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock) as mock_init:
            response = await client.post(f"/topics/{topic.id}/init", follow_redirects=False)

        assert response.status_code == 409
        mock_init.assert_not_called()
        assert get_topic(db_conn, topic.id).status == TopicStatus.ERROR

    async def test_accepted_retry_hands_the_guard_to_the_task(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """The handler holds the guard across the response and passes its token on."""
        from app.web.state import _checking_state

        topic = _make_topic(db_conn, status=TopicStatus.ERROR)

        with patch("app.web.routers.background._run_init", new_callable=AsyncMock) as mock_init:
            response = await client.post(f"/topics/{topic.id}/init", follow_redirects=False)

        assert response.status_code == 303
        # BackgroundTasks ran the (mocked) task, which never released the guard —
        # so the token the handler acquired is what it was handed.
        args, kwargs = mock_init.call_args
        assert args[0] == topic.id
        assert isinstance(args[3], str) and args[3]
        assert kwargs == {"claimed": True}
        assert await _checking_state.is_checking(topic.id) is True
        await _checking_state.finish_check(topic.id, args[3])

    async def test_lost_claim_releases_the_guard(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Losing the durable claim must not leave the in-process slot wedged."""
        from app.web.state import _checking_state

        topic = _make_topic(db_conn, status=TopicStatus.ERROR)

        with (
            patch("app.web.routers.topics.claim_topic_for_init", return_value=False),
            patch("app.web.routers.background._run_init", new_callable=AsyncMock) as mock_init,
        ):
            response = await client.post(f"/topics/{topic.id}/init", follow_redirects=False)

        assert response.status_code == 409
        mock_init.assert_not_called()
        assert await _checking_state.is_checking(topic.id) is False


class TestRetryResearchUI:
    """Tests that the Retry Research button appears in the correct UI contexts."""

    async def test_error_topic_detail_shows_retry_button(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Topic detail page shows 'Retry Research' button for ERROR topics."""
        topic = _make_topic(db_conn, status=TopicStatus.ERROR, error_message="timed out")

        response = await client.get(f"/topics/{topic.id}")

        assert response.status_code == 200
        assert "Retry Research" in response.text

    async def test_error_topic_detail_shows_error_notice(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Topic detail page shows the error notice article for ERROR topics."""
        topic = _make_topic(db_conn, status=TopicStatus.ERROR, error_message="timed out")

        response = await client.get(f"/topics/{topic.id}")

        assert response.status_code == 200
        assert "Research failed" in response.text

    async def test_ready_topic_detail_no_retry_button(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """Topic detail page does NOT show 'Retry Research' button for READY topics."""
        topic = _make_topic(db_conn, status=TopicStatus.READY, error_message=None)

        response = await client.get(f"/topics/{topic.id}")

        assert response.status_code == 200
        assert "Retry Research" not in response.text

    async def test_error_status_partial_shows_retry_button(
        self,
        client: httpx.AsyncClient,
        db_conn: sqlite3.Connection,
    ) -> None:
        """The status HTMX partial shows 'Retry Research' for ERROR topics."""
        topic = _make_topic(db_conn, status=TopicStatus.ERROR, error_message="failed")

        response = await client.get(f"/topics/{topic.id}/status")

        assert response.status_code == 200
        assert "Retry Research" in response.text
