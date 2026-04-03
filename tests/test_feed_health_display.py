"""Tests for feed health display in the UI."""

import sqlite3
from collections.abc import AsyncGenerator

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.crud import create_topic, upsert_feed_health_failure, upsert_feed_health_success
from app.main import app
from app.models import FeedMode, Topic, TopicStatus
from app.web.dependencies import get_db_conn, get_settings

CSRF_TEST_TOKEN = "test-csrf-token-for-tests"


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


# --- Feed Health page tests ---


class TestFeedHealthPage:
    """Tests for GET /feeds."""

    async def test_feed_health_page_returns_200(self, client: httpx.AsyncClient) -> None:
        """GET /feeds returns 200 with empty feed health data."""
        response = await client.get("/feeds")
        assert response.status_code == 200

    async def test_feed_health_page_empty_message(self, client: httpx.AsyncClient) -> None:
        """Page shows an empty-state message when no feed health records exist."""
        response = await client.get("/feeds")
        assert response.status_code == 200
        assert "No feed health data yet" in response.text

    async def test_feed_health_page_shows_healthy_feed(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Page shows healthy status for feeds with no failures."""
        upsert_feed_health_success(db_conn, "https://healthy.example.com/feed.xml")
        db_conn.commit()

        response = await client.get("/feeds")
        assert response.status_code == 200
        assert "Healthy" in response.text
        assert "healthy.example.com" in response.text

    async def test_feed_health_page_shows_degraded_feed(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Page shows degraded status for feeds with 1-2 consecutive failures."""
        url = "https://flaky.example.com/feed.xml"
        upsert_feed_health_success(db_conn, url)
        upsert_feed_health_failure(db_conn, url, "timeout")
        db_conn.commit()

        response = await client.get("/feeds")
        assert response.status_code == 200
        assert "Degraded" in response.text

    async def test_feed_health_page_shows_unhealthy_feed(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Page shows unhealthy status for feeds with 3+ consecutive failures."""
        url = "https://broken.example.com/feed.xml"
        upsert_feed_health_failure(db_conn, url, "connection refused")
        upsert_feed_health_failure(db_conn, url, "connection refused")
        upsert_feed_health_failure(db_conn, url, "connection refused")
        db_conn.commit()

        response = await client.get("/feeds")
        assert response.status_code == 200
        assert "Unhealthy" in response.text
        assert "broken.example.com" in response.text

    async def test_feed_health_page_shows_failure_rate(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Page shows failure rate percentage for feeds with fetches."""
        url = "https://example.com/feed.xml"
        upsert_feed_health_success(db_conn, url)
        upsert_feed_health_success(db_conn, url)
        upsert_feed_health_failure(db_conn, url, "error")
        db_conn.commit()

        response = await client.get("/feeds")
        assert response.status_code == 200
        # 1/3 failures = 33%
        assert "33%" in response.text

    async def test_feed_health_page_has_nav_link(self, client: httpx.AsyncClient) -> None:
        """Navigation includes a Feed Health link."""
        response = await client.get("/feeds")
        assert response.status_code == 200
        assert 'href="/feeds"' in response.text
        assert "Feed Health" in response.text


# --- Topic detail page feed health indicators tests ---


class TestTopicDetailFeedHealthIndicators:
    """Tests for feed health indicators on the topic detail page."""

    async def test_topic_detail_shows_gray_dot_when_no_health_data(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Feed URLs with no health data show a gray indicator."""
        topic = _make_topic(db_conn, feed_urls=["https://example.com/feed.xml"])

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "No health data" in response.text

    async def test_topic_detail_shows_green_dot_for_healthy_feed(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Feed URLs that are healthy show a green indicator."""
        url = "https://example.com/feed.xml"
        topic = _make_topic(db_conn, feed_urls=[url])
        upsert_feed_health_success(db_conn, url)
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "color: green" in response.text
        assert "Healthy" in response.text

    async def test_topic_detail_shows_orange_dot_for_degraded_feed(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Feed URLs with 1-2 consecutive failures show an orange indicator."""
        url = "https://flaky.example.com/feed.xml"
        topic = _make_topic(db_conn, feed_urls=[url])
        upsert_feed_health_success(db_conn, url)
        upsert_feed_health_failure(db_conn, url, "timeout")
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "color: orange" in response.text
        assert "Degraded" in response.text

    async def test_topic_detail_shows_red_dot_for_unhealthy_feed(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Feed URLs with 3+ consecutive failures show a red indicator."""
        url = "https://broken.example.com/feed.xml"
        topic = _make_topic(db_conn, feed_urls=[url])
        upsert_feed_health_failure(db_conn, url, "err1")
        upsert_feed_health_failure(db_conn, url, "err2")
        upsert_feed_health_failure(db_conn, url, "err3")
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "color: red" in response.text
        assert "Unhealthy" in response.text

    async def test_topic_detail_auto_mode_shows_health_indicator(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Auto-mode topic detail shows health indicator next to provider URL."""
        from app.scraping.routing import router as provider_router

        topic = _make_topic(db_conn, name="Auto Topic", feed_mode=FeedMode.AUTO, feed_urls=[])
        auto_url = provider_router.get_provider().build_feed_url(topic)
        upsert_feed_health_success(db_conn, auto_url)
        db_conn.commit()

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "color: green" in response.text
        assert "Healthy" in response.text

    async def test_topic_detail_auto_mode_gray_when_no_health(
        self, client: httpx.AsyncClient, db_conn: sqlite3.Connection
    ) -> None:
        """Auto-mode topic detail shows gray dot when no health data exists."""
        topic = _make_topic(db_conn, name="Auto No Health", feed_mode=FeedMode.AUTO, feed_urls=[])

        response = await client.get(f"/topics/{topic.id}")
        assert response.status_code == 200
        assert "No health data" in response.text


# --- Navigation tests ---


class TestNavigation:
    """Tests for Feed Health navigation link."""

    async def test_dashboard_has_feed_health_nav_link(self, client: httpx.AsyncClient) -> None:
        """Dashboard page includes the Feed Health nav link."""
        response = await client.get("/")
        assert response.status_code == 200
        assert 'href="/feeds"' in response.text
        assert "Feed Health" in response.text

    async def test_settings_page_has_feed_health_nav_link(self, client: httpx.AsyncClient) -> None:
        """Settings page includes the Feed Health nav link."""
        response = await client.get("/settings")
        assert response.status_code == 200
        assert 'href="/feeds"' in response.text
