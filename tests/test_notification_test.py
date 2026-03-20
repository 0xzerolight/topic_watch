"""Tests for POST /notifications/test route."""

import sqlite3
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.config import LLMSettings, NotificationSettings, Settings
from app.main import app
from app.web.dependencies import get_db_conn, get_settings

CSRF_TEST_TOKEN = "test-csrf-token-for-notification-tests"


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key-12345678"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
async def client(
    db_conn: sqlite3.Connection,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create a test client with database and settings dependencies overridden."""
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
async def client_no_urls(
    db_conn: sqlite3.Connection,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Test client with no notification URLs configured."""
    settings = _make_settings(notifications=NotificationSettings(urls=[]))

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


# --- Success case ---


async def test_test_notification_success(client: httpx.AsyncClient) -> None:
    """Successful send_notification returns a green success message."""
    with patch("app.web.routes.send_notification", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        response = await client.post("/notifications/test")

    assert response.status_code == 200
    assert "Notification sent successfully" in response.text
    assert "pico-ins-color" in response.text


async def test_test_notification_calls_send_with_correct_args(client: httpx.AsyncClient) -> None:
    """Route calls send_notification with the expected title and body."""
    with patch("app.web.routes.send_notification", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = True
        await client.post("/notifications/test")

    mock_send.assert_called_once()
    call_args = mock_send.call_args
    title, body = call_args.args[0], call_args.args[1]
    assert title == "Topic Watch Test"
    assert "test notification" in body.lower()


# --- Failure case ---


async def test_test_notification_delivery_failure(client: httpx.AsyncClient) -> None:
    """When send_notification returns False, a red failure message is shown."""
    with patch("app.web.routes.send_notification", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = False
        response = await client.post("/notifications/test")

    assert response.status_code == 200
    assert "Notification delivery failed" in response.text
    assert "border-left" in response.text


# --- Exception case ---


async def test_test_notification_exception(client: httpx.AsyncClient) -> None:
    """When send_notification raises, a red error message is shown."""
    with patch("app.web.routes.send_notification", new_callable=AsyncMock) as mock_send:
        mock_send.side_effect = RuntimeError("connection refused")
        response = await client.post("/notifications/test")

    assert response.status_code == 200
    assert "Notification error" in response.text
    assert "pico-del-color" in response.text


# --- No URLs configured ---


async def test_test_notification_no_urls(client_no_urls: httpx.AsyncClient) -> None:
    """When no notification URLs are configured, returns an informative message."""
    with patch("app.web.routes.send_notification", new_callable=AsyncMock) as mock_send:
        response = await client_no_urls.post("/notifications/test")

    assert response.status_code == 200
    assert "No notification URLs configured" in response.text
    assert "border-left" in response.text
    mock_send.assert_not_called()


# --- CSRF protection ---


async def test_test_notification_requires_csrf(db_conn: sqlite3.Connection) -> None:
    """POST /notifications/test returns 403 without a valid CSRF token."""
    settings = _make_settings()

    def override_db():
        yield db_conn

    def override_settings():
        return settings

    app.dependency_overrides[get_db_conn] = override_db
    app.dependency_overrides[get_settings] = override_settings

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            response = await ac.post("/notifications/test")

        assert response.status_code == 403
    finally:
        app.dependency_overrides.clear()
