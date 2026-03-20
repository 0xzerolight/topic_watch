"""Tests for the toggle-active route."""

import sqlite3
from collections.abc import AsyncGenerator

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_topic, get_topic
from app.main import app
from app.models import FeedMode, Topic, TopicStatus
from app.web.dependencies import get_db_conn, get_settings

CSRF_TEST_TOKEN = "test-csrf-token-for-toggle-tests"


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
        "is_active": True,
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


class TestToggleActive:
    """Tests for POST /topics/{topic_id}/toggle-active."""

    async def test_toggle_active_to_inactive(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Toggling an active topic makes it inactive and redirects."""
        topic = _make_topic(db_conn, name="Active Topic", is_active=True)
        assert topic.is_active is True

        response = await client.post(
            f"/topics/{topic.id}/toggle-active",
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == f"/topics/{topic.id}"

        updated = get_topic(db_conn, topic.id)
        assert updated is not None
        assert updated.is_active is False

    async def test_toggle_inactive_to_active(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Toggling an inactive topic makes it active and redirects."""
        topic = _make_topic(db_conn, name="Inactive Topic", is_active=False)
        assert topic.is_active is False

        response = await client.post(
            f"/topics/{topic.id}/toggle-active",
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == f"/topics/{topic.id}"

        updated = get_topic(db_conn, topic.id)
        assert updated is not None
        assert updated.is_active is True

    async def test_toggle_nonexistent_topic_returns_404(self, client: httpx.AsyncClient) -> None:
        """Toggling a non-existent topic returns 404."""
        response = await client.post(
            "/topics/99999/toggle-active",
            follow_redirects=False,
        )
        assert response.status_code == 404

    async def test_toggle_persists_in_database(self, client: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Toggle state persists across multiple requests."""
        topic = _make_topic(db_conn, name="Persist Topic", is_active=True)

        # Disable
        await client.post(f"/topics/{topic.id}/toggle-active", follow_redirects=False)
        after_disable = get_topic(db_conn, topic.id)
        assert after_disable is not None
        assert after_disable.is_active is False

        # Re-enable
        await client.post(f"/topics/{topic.id}/toggle-active", follow_redirects=False)
        after_enable = get_topic(db_conn, topic.id)
        assert after_enable is not None
        assert after_enable.is_active is True

    async def test_toggle_htmx_request_returns_partial(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """HTMX request returns a table row partial instead of redirecting."""
        topic = _make_topic(db_conn, name="HTMX Topic", is_active=True)

        response = await client.post(
            f"/topics/{topic.id}/toggle-active",
            headers={"HX-Request": "true"},
            follow_redirects=False,
        )
        assert response.status_code == 200
        assert f'id="topic-{topic.id}"' in response.text

    async def test_toggle_htmx_response_reflects_new_state(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """HTMX response shows the updated (inactive) state after toggling."""
        topic = _make_topic(db_conn, name="State Topic", is_active=True)

        response = await client.post(
            f"/topics/{topic.id}/toggle-active",
            headers={"HX-Request": "true"},
            follow_redirects=False,
        )
        assert response.status_code == 200
        # After toggling from active, should show "Enable" button
        assert "Enable" in response.text

    async def test_toggle_requires_csrf(self, client_no_csrf: httpx.AsyncClient, db_conn: sqlite3.Connection) -> None:
        """Toggle without CSRF token returns 403."""
        topic = _make_topic(db_conn, name="CSRF Topic", is_active=True)

        response = await client_no_csrf.post(
            f"/topics/{topic.id}/toggle-active",
            follow_redirects=False,
        )
        assert response.status_code == 403
        # State should not have changed
        unchanged = get_topic(db_conn, topic.id)
        assert unchanged is not None
        assert unchanged.is_active is True
