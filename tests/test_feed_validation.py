"""Tests for feed URL validation endpoint and rate limiter."""

import time
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.main import app
from app.scraping.rss import FeedEntry
from app.web.dependencies import get_db_conn, get_settings
from app.web.state import _check_rate_limit, _rate_limit_store


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
    db_conn,
) -> AsyncGenerator[httpx.AsyncClient, None]:
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


# --- Rate limiter unit tests ---


def test_rate_limit_allows_up_to_max():
    """First 10 calls for a unique IP are allowed."""
    test_ip = "10.0.0.1"
    _rate_limit_store.pop(test_ip, None)

    for i in range(10):
        assert _check_rate_limit(test_ip) is True, f"Call {i + 1} should be allowed"


def test_rate_limit_blocks_on_eleventh_call():
    """11th call within the window is rejected."""
    test_ip = "10.0.0.2"
    _rate_limit_store.pop(test_ip, None)

    for _ in range(10):
        _check_rate_limit(test_ip)

    assert _check_rate_limit(test_ip) is False


def test_rate_limit_resets_after_window():
    """Calls succeed again after old timestamps expire."""
    test_ip = "10.0.0.3"
    _rate_limit_store.pop(test_ip, None)

    # Fill the window with old timestamps (older than 60s)
    old_time = time.time() - 61
    _rate_limit_store[test_ip] = [old_time] * 10

    # Should be allowed now because all timestamps are stale
    assert _check_rate_limit(test_ip) is True


# --- Route integration tests ---


async def test_validate_empty_input(client: httpx.AsyncClient):
    """Empty textarea returns a 'No URLs provided' message."""
    response = await client.post("/feeds/validate", data={"feed_urls": ""})
    assert response.status_code == 200
    assert "No URLs provided" in response.text


async def test_validate_valid_url(client: httpx.AsyncClient):
    """A fetchable feed URL returns success with entry count."""
    fake_entries = [
        FeedEntry(title="Entry 1", url="https://example.com/1", source_feed="https://example.com/feed.xml"),
        FeedEntry(title="Entry 2", url="https://example.com/2", source_feed="https://example.com/feed.xml"),
    ]

    from app.scraping.source import FeedFetchResult, FetchStatus

    ok = AsyncMock(return_value=FeedFetchResult(entries=fake_entries, status=FetchStatus.OK))
    with patch("app.scraping.rss.fetch_feed_outcome", new=ok):
        response = await client.post(
            "/feeds/validate",
            data={"feed_urls": "https://example.com/feed.xml"},
        )

    assert response.status_code == 200
    assert "Valid RSS feed with 2 entries" in response.text
    assert "&#10004;" in response.text  # checkmark


async def test_validate_invalid_url(client: httpx.AsyncClient):
    """A network failure is reported with its reason, not as an empty valid feed.

    The fetch layer is fail-safe and swallows the exception this test used to
    mock, so the failure has to arrive as a failed outcome (AUG-175).
    """
    from app.scraping.source import FeedFetchResult, FeedHealthOutcome, FetchStatus

    async def refused(url, client=None, **kwargs):
        callback = kwargs.get("health_callback")
        if callback:
            callback(FeedHealthOutcome(url, FetchStatus.FAILED, "Network error: ConnectError: Connection refused"))
        return FeedFetchResult(status=FetchStatus.FAILED)

    with patch("app.scraping.rss.fetch_feed_outcome", new=refused):
        response = await client.post(
            "/feeds/validate",
            data={"feed_urls": "https://bad.example.com/feed.xml"},
        )

    assert response.status_code == 200
    assert "Connection refused" in response.text
    assert "&#10008;" in response.text  # cross mark


async def test_validate_private_url(client: httpx.AsyncClient):
    """Private/local URLs are rejected without fetching."""
    response = await client.post(
        "/feeds/validate",
        data={"feed_urls": "http://localhost/feed.xml"},
    )

    assert response.status_code == 200
    assert "Private/local URLs are not allowed" in response.text
    assert "&#10008;" in response.text


async def test_validate_rate_limit_exceeded(client: httpx.AsyncClient):
    """After 10 requests the endpoint returns 429."""
    # httpx.ASGITransport reports the client IP as "127.0.0.1"
    test_ip = "127.0.0.1"
    _rate_limit_store.pop(test_ip, None)

    # Saturate the rate limit manually
    _rate_limit_store[test_ip] = [time.time()] * 10

    response = await client.post(
        "/feeds/validate",
        data={"feed_urls": "https://example.com/feed.xml"},
    )

    assert response.status_code == 429
    assert "Rate limit exceeded" in response.text


async def test_validate_reports_a_failed_fetch_as_invalid(client: httpx.AsyncClient):
    """AUG-175: a feed that failed to fetch must not validate as an empty-but-valid one."""
    from app.scraping.source import FeedFetchResult, FeedHealthOutcome, FetchStatus

    _rate_limit_store.pop("127.0.0.1", None)

    async def failed_fetch(url, client=None, **kwargs):
        callback = kwargs.get("health_callback")
        if callback:
            callback(FeedHealthOutcome(url, FetchStatus.FAILED, "HTTP 404"))
        return FeedFetchResult(status=FetchStatus.FAILED)

    with patch("app.scraping.rss.fetch_feed_outcome", new=failed_fetch):
        response = await client.post(
            "/feeds/validate",
            data={"feed_urls": "https://gone.example.com/feed.xml"},
        )

    assert response.status_code == 200
    assert "Valid RSS feed" not in response.text
    assert "HTTP 404" in response.text
    assert "&#10008;" in response.text  # cross mark


async def test_validate_accepts_a_genuinely_empty_feed(client: httpx.AsyncClient):
    """A feed that fetched fine with no items is still a usable feed."""
    from app.scraping.source import FeedFetchResult, FetchStatus

    _rate_limit_store.pop("127.0.0.1", None)

    async def empty_fetch(url, client=None, **kwargs):
        return FeedFetchResult(status=FetchStatus.EMPTY)

    with patch("app.scraping.rss.fetch_feed_outcome", new=empty_fetch):
        response = await client.post(
            "/feeds/validate",
            data={"feed_urls": "https://quiet.example.com/feed.xml"},
        )

    assert "Valid RSS feed with 0 entries" in response.text
